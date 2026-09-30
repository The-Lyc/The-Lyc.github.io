---
layout: "post"
title: "Decoder-only 模型的推理与 KV Cache"
date: "2026-09-30"
description: "从 Prefill 与 Decode 的矩阵计算理解 KV Cache 的复用、复杂度收益与多头注意力。"
categories: ["Models"]
tags: ["transformer", "kv-cache", "inference", "attention"]
permalink: "/blog/decoder-only-inference-kv-cache/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/vLLM/decode-only模型的理解.md"
---

目前通用的decode-only的大模型有一些特点，需要理解了这些特点才能理解vLLM中的一些策略。

---

decode-only的模型的推理过程：

没有encoder，因此也没有cross attention

推理过程包括prefill和decode两个阶段

要理解为什么需要kv cache以及计算细节，需要知道整个推理过程：

先看prefill阶段（先不考虑multi-head），输入的是包含所有prompt的token的embedding矩阵，分别经过线性变换后得到Q,K,V,后续的attention和FFN过程如下：

- 全量的一次推理，时间复杂度是 $$O(t^2*h)$$

<img src="{{ '/assets/blog/decoder-only-inference-kv-cache/image-20251202152005209.png' | relative_url }}" alt="image-20251202152005209" style="max-width: 100%; height: auto;" loading="lazy" />

这里就涉及到为什么KV是可以cache的，即为什么生成下一个token的时候可以用之前计算出来的kv

<img src="{{ '/assets/blog/decoder-only-inference-kv-cache/image-20251202153610773.png' | relative_url }}" alt="image-20251202153610773" style="max-width: 100%; height: auto;" loading="lazy" />

可以发现，在没有kv cache的时候，新增一个token（此时总token数为$$t+1$$），如果还是做全量的推理，那么原来的K和V（空白部分）是完全可以复用的：

- 以 $$Q*K^T$$ 为例，结果矩阵中有 $$t*t$$ 的部分都是之前k v计算出来的，完全不受 $$token_{i+1}$$ 的影响
- 最重要的是softmax操作时，由于加了mask，所以每一行被新的token影响的logits都被掩码掩掉了，也就是说：除了 $$token_{i+1}$$ 自身在的最后那一行，最后softmax算出来的概率分布完全不受$$token_{i+1}$$的qk的影响
- $$S*V$$ 计算中，由于新token的信息都在V矩阵的最后一行，S矩阵的上三角全为0，所以只有S的最后一行会真正与V最后一行运算，结果中除了最后一行别的行也是与上一轮迭代保持一致的

其实主要就是要证明每一层transformer块的输入是增量的而非完全变化的，增量就意味着之前的kv是可以复用的，而要证明这一点就需要理清整个transformer的计算过程。

证明了kv是可复用的之后，就发现隐状态向量总是结果矩阵的最后一行，因此所有的计算都围绕着最后一行这个隐状态向量的来源来裁剪，大量无关的部分根本不需要计算了。

> **Important**
>
> 那kv cache带来的收益是什么？如果没有kv cache，那么不仅每一层要重新计算所有的kv，还要每一层都要计算全量的attention，不然之前层只会输出最后一行隐状态向量，没法生成下一层的qkv，这样既不能像原生transformer那样重新计算kv（即使这些值是完全重复的），又不能用kv cache。

因此prefill就是做一次全量的forward，prompt的所有token的kv cache就生成了。第一轮decode的迭代就用prompt最后一个token作为输入embedding然后线性变换为qkv正常decode，因为attention不仅需要之前token的kv cache，还需要当前最后一个token的logits。

因此，decode阶段，只需要对当前token做计算，qk计算只需要用当前q与所有k点乘，复杂度变为$$O(t*h)$$，softmax也只计算当前token对所有token的注意力得分（即只计算最后一行的softmax，复杂度变为$$O(t)$$, $$S*V$$也是只计算当前token对所有token的这一个概率分布与所有v的点积，复杂度变为$$O(t*d)$$。所以整个attention计算的复杂度就变为了$$O(t*d)$$。

prefill的时候输入的就是整个prompt矩阵，因此transformer层之间传递的也是相同维度的矩阵。decode阶段输入就是当前token，因此transformer层之间传递的也是这个token经过attention和FFN承载的信息（隐状态向量），维度就是一个token的维度。在最后一个transformer层计算结束之后，与字典（包含所有合法token的embedding向量）做点积，得到$$logits_t=W_{lm}⋅h_t$$,维度是：

$$

(|V| \times d) \cdot (d) = |V|


$$

也就是说，**对词表中的每一个词，算一个匹配分数（logit）**。然后再用softmax变成概率分布以供选择。

---

multi-head attention

MHA的计算是将多个头分别跑完整的attention，而不是attention的每次计算都要让每个头同步结果

<img src="{{ '/assets/blog/decoder-only-inference-kv-cache/image-20251202215016355.png' | relative_url }}" alt="image-20251202215016355" style="max-width: 100%; height: auto;" loading="lazy" />

MHA的计算体现出的含义本身就是与单头的transformer有略微差异的，比如 $$Q*K^T$$ 这一步，MHA得到的结果实际上也是部分和，但是MHA的目的就是要有多个channel来捕捉不同的特征，因此从每个head的角度来看这个结果就是这个head的$$Q*K^T$$全量结果，softmax的结果也是每个token对之前所有token的注意力得分，直接用这个score与V做点积，得到attention结果的一部分最终结果，当然可以此时做一次all-gather通信再接着做FFN，但是更优的方法是把FFN的$$W^o$$矩阵按行拆分成多头，这样每个head与之点积之后就得到升维结果的部分和，此时再做all-reduce就能得到全量的升维结果，后续再降维就完成了整个FFN。
