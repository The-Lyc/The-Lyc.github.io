---
layout: "post"
title: "Transformer、注意力与自回归模型笔记"
date: "2026-09-30"
description: "整理 Encoder–Decoder、自回归、LayerNorm、多头注意力、KV Cache 与大语言模型架构。"
categories: ["Models"]
tags: ["transformer", "attention", "kv-cache", "models"]
permalink: "/blog/transformer-autoregressive-models/"
lang: "zh-CN"
notes_import: true
source_path: "LLM/Transformer.md"
toc:
  beginning: true
---

### 1.encoder-decoder

在transformer出现之前，大部分神经序列转换模型也都是采用encoder-decoder的形式。

**Encoder 的任务**是把这个离散序列变换为一个**连续的向量序列**（通常是高维 embedding）。Decoder的任务就是从这个中间序列出发，生成输出序列。

transformer的结构也是先多个encoder层，再多个decoder层。

### 2.自回归

自回归模型是指当前的输出依赖于它之前生成的所有输出。transformer的每一步解码都是自回归的。

在训练时（GPT等大模型都是采用预测下一个词的训练方式），会把整句话都丢给tf块，为了达到自回归的效果，需要加mask遮挡预测位置后面的词。并且输入的序列是从BOS开始的，预测是从第一个真正的token开始的。因此，训练时，当预测第i个token时，模型能看到输入序列前i个token的embedding向量，是不包括当前token的（因为输入有一个BOS token导致错位）。

推理时就不需要加mask，因为后面的词本来就还没生成。

### 3.LayerNorm

给定输入向量 $$\mathbf{x} \in \mathbb{R}^d$$，LayerNorm 会对每一个 token 的 embedding 向量做如下归一化：

$$

\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta


$$

其中：

- $$\mu$$：当前 token 的均值（在维度维上求的均值）
- $$\sigma$$：标准差
- $$\gamma, \beta$$：可学习的缩放（scale）和平移（shift）参数，维度是 $$d$$

LayerNorm是针对每次输入的词序列的每个token的，也就是每一个token自己做归一化。假设一个 token 的表示是向量：

$$

\mathbf{x} = [x_1, x_2, ..., x_d]


$$

LayerNorm 对这个向量计算：

- 均值 $$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$
- 方差 $$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

然后归一化：

$$

\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}


$$

---

不同于 BatchNorm：

- BatchNorm 是对**整个 batch**的同一维度做统计，归一化
- LayerNorm 是对**单个样本的单个 token 向量**做统计，归一化

这也是为什么 LayerNorm 更适合 NLP/Transformer 这种变长、batch 大小可变的场景。

---

Transformer 中每个子层都像这样使用 LayerNorm（有两种常见结构）：

Post-Norm（原论文结构）：

```
x → SubLayer(x) → Residual Add → LayerNorm
```

Pre-Norm（后来更稳定的做法）：

```
x → LayerNorm → SubLayer(x) → Residual Add
```

现在主流实现（如 HuggingFace、OpenAI）都用 **Pre-Norm**，训练更稳定。

### 4.并行

Transformer相较于RNN很重要的一点改进是，虽然都可以处理时序信息，但是RNN必须串行地依次计算每一个token的隐藏状态h（h(t)由h(t-1)和当前token决定）。但是Transformer只依赖注意力机制，不用时序循环计算，训练时对每一个token的预测都是可以并行的。

### 5.多头注意力

采用多头是因为Transformer虽然可以将整个序列作为输入（而非像卷积一样每次只能看到卷积核大小的输入，如果元素距离较远则需要多次卷积才能同时将这两个元素纳入一次运算中），但是卷积的优点在于可以输出多通道的特征（即，可以提取多个维度的特征）。因此，Transformer就提出了多头的注意力机制，多头就代表多个维度。如果模型维度（指的是transformer中每个token的embedding向量的维度）是d，头数是n，那么每个头计算的token的向量表示是d/n维 。因此，总的计算量并没有增加。

### 6.Transformer中的三种attention

encoder侧只有一种attention，就是标准的**self attention**，QKV就是input分别malmul $$W_Q$$ ,$$W_K$$ ,$$W_V$$得到的。

decoder侧有两种attention。

decoder接受了当前已经生成的output作为输入之后，先做一个**mask attention**（当然，只有训练的时候会加mask；推理的时候本来也还没生成之后的token）。输入给mask attention的QKV是由当前已生成的序列对应的向量线性变换来的。也就是说，这里的QKV和encoder的QKV意义和维度都是不一样的。其实，mask attention也是self attention，只不过训练的时候加了掩码。

mask attention输出的向量作为**cross attention**的输入中的Q，KV来自于最后一层encoder的输出分别经过$$W_K$$ ,$$W_V$$线性变换得到。

### 7.KV cache

要理解KV cache，以及为什么没有Q cache，需要对decoder的计算过程同时有微观和宏观的理解。

先看结构：

<img src="{{ '/assets/blog/transformer-autoregressive-models/image-20250704111447907.png' | relative_url }}" alt="image-20250704111447907" style="max-width: 100%; height: auto;" loading="lazy" />

1. decoder的输入是当前生成的output。output embedding是可以学习的。
2. mask attention接受的QKV输入都是由当前的output线性变换来的。
3. mask attention计算完之后，输出的向量作为cross attention的Q输入，并接受来自最后一个encoder的输出作为KV（也是线性变换过后的）。
4. cross attention的输出作为FFN的输入，作一个非线性升维再恢复到原来的维度。
5. 如果是最后一个decoder，就需要把FFN的输出中最后一个位置的向量跟字典矩阵malmul生成字典中每一个token的权重向量。
6. 将字典权重向量作softmax就得到了最后的预测概率，选概率最高的作为当前位置预测的token。

可以发现，最后生成字典权重向量的时候，只有最后一个位置的向量是有用的。

倒退回cross attention，假设输入的Q是一个形状为 $$t \times d$$ 的向量序列：

```
H_masked ∈ ℝ^{t × d}
```

显然，不论从encoder来的KV有多少个向量（假设是e\*d，e是input的token数，跟t不一样，而且t还是随着推理过程变化的值），做完cross attention之后仍然是一个t\*d的矩阵。那么cross attention的输出中最后一个向量跟哪些有关系呢？

可以倒退回mask attention的计算。此时QKV维度一致，把KQV写成分块的形式，像这样：

<img src="{{ '/assets/blog/transformer-autoregressive-models/image-20250704113945577.png' | relative_url }}" alt="image-20250704113945577" style="max-width: 100%; height: auto;" loading="lazy" />

然后Q和K转置的矩阵乘就变成了这样：

<img src="{{ '/assets/blog/transformer-autoregressive-models/image-20250704114010632.png' | relative_url }}" alt="image-20250704114010632" style="max-width: 100%; height: auto;" loading="lazy" />

忽略 的系数，第i行的softmax简写成 $$S_i$$，attention操作的结果变成了这样：

<img src="{{ '/assets/blog/transformer-autoregressive-models/image-20250704114123094.png' | relative_url }}" alt="image-20250704114123094" style="max-width: 100%; height: auto;" loading="lazy" />

加上掩码之后就变成了：

<img src="{{ '/assets/blog/transformer-autoregressive-models/image-20250704114200210.png' | relative_url }}" alt="image-20250704114200210" style="max-width: 100%; height: auto;" loading="lazy" />

可以发现，最后一个向量是由KV的全部向量和Q的最后一个向量得到的。

cross attention的计算过程和mask attention是一样的，因此，cross attention输出中最后一个向量也是这样的形式，跟KV的所有向量都有关，但是只跟Q的最后一个向量有关。

所以，追根溯源，每次推理，decoder最后生成概率，会用到output的所有KV，encoder的所有KV，output的Q的最后一个向量。

而encoder的所有KV是固定的，output的KV是增量的，生成一个新的token，下一轮推理时output的KV就会多一个向量，但是历史向量是一致的。因此，KV cache存的就是output的KV。另外，既然output的Q每次都只有最后一个向量有用，并且每次的最后一个向量都是新生成的，所以就没有理由做一个Q cache了。

有一个比较迷惑的点在于，为什么每次生成新token只需要Q的最后一个向量？从宏观上来说，就只用到了上一个token的信息？

实际上，只用到Q的最后一个向量不等于只用了上一个token的信息。从计算过程可以看到，只用最后一个向量是因为去和字典做malmul的是最后一个向量，而这个向量包含了已生成的所有token的上下文信息，之前的向量不会包含其之后当前之前的这些token。而mask attention中KV都是整个output序列生成的，因此，就算Q中只用了最后一个token的向量，经过mask attention也包含了其他token的信息。

### 8. Teacher Forcing

一个很自然的疑惑：Transformer在训练的时候，decoder部分是并行训练的，也就是利用掩码机制，同时对待预测序列的每个位置token都做预测，用的上文就是正确的待预测序列的上文。例如，此次需要预测的完整序列是"cats are the best."，那么这四个token在推理时是一个一个自回归预测的，但是训练的时候就是同时预测的，比如预测"the"时，就直接用"carts are"做上文。这样就有一个问题，训练预测下一个词时，总是建立在前文都是正确的基础上的，实际推理时应对前文就出错的情况的能力自然没怎么被训练过。

这种训练方式就叫做"Teacher Forcing"，所谓Teacher Forcing模式，就是在训练时不再以模型预测作为输入，取而代之的是以**目标真值**（ground truth）作为输入，其名称中的“Teacher”自然指的就是真值。

- 优点：提升序列模型训练稳定性、加速模型收敛，非常好并行，对于大模型非常重要。
- 缺点：训练环节和预测环节存在差异（这种差异叫"exposure bias"，曝光误差），模型对“错误上下文”的鲁棒性较差。

有一些折中方案：

1. Scheduled Sampling（计划采样）：训练时，以一定概率用模型生成的 token 替换真实 token 作为下一步输入。刚开始替换概率小，随着训练增加。能让模型逐渐学会在不完美前缀下预测，但收敛变慢。
2. Sequence-level training：训练输入是模型生成的前缀，利用强化学习的方法将Transformer视为一个策略，用一个奖励函数来评估模型输出的整个序列的好坏，再用策略梯度更新参数。
3. Data augmentation：人为构造带“错前缀”的样本进行训练，让模型学会在错误条件下尽量生成合理的结果。

### 9. LLM都是decoder-only

#### 9.1 LLM分类

##### 9.1.1 Encoder-only

代表是BERT：

- **结构**
  - 只有 Transformer 的 **encoder 堆叠**，每一层都是 **双向自注意力（bidirectional self-attention）**。

- **输入**
  - 整个序列一次性输入，模型可以在同一层同时关注左边和右边的上下文。

- **输出**
  - 对每个 token 都得到一个 contextual embedding，可以用于分类（取 [CLS] 向量）、标注（NER）、句子对判断等。

- **训练目标**
  - **MLM（Masked Language Modeling**：随机 mask 一部分 token，让模型预测它们。

  - **NSP/SOP**（BERT 原版用 NSP，RoBERTa 去掉了）。

- **适合任务**
  - 自然语言理解（NLU）：分类、匹配、标注等。

- **注意**
  - 因为是双向的，所以不能直接用于自回归生成（不能一步步生成文本）。

##### 9.1.2 Encoder-Decoder

代表是T5，其实就是经典的Transformer架构：encoder负责对输入序列双向建模，decoder是单项注意力并且用cross attention读encoder的结果。

适合理解+生成任务，解码器自回归生成。

##### 9.1.3 Decoder-only

代表就是GPT，只用了单向自注意力，训练和推理都是自回归，目标就是language modeling。

没有显式的encoder-decoder分离，但是也可以把前面的decoder看作具有编码性质的。

##### 9.1.4 PrefixLM

物理上是一个统一的 Transformer，但通过**不同的 attention mask 模式**模拟 encoder-only / decoder-only / encoder–decoder 三种情况。

| 架构            | 注意力可见性           | 典型模型      | 输入/输出模式   | 主要任务    |
| --------------- | ---------------------- | ------------- | --------------- | ----------- |
| Encoder-only    | 双向                   | BERT, RoBERTa | 输入 → 向量表示 | 理解 (NLU)  |
| Encoder–Decoder | 编码器双向，解码器单向 | T5, BART      | 输入 → 输出     | 理解 + 生成 |
| Decoder-only    | 单向                   | GPT           | 前缀 → 继续生成 | 生成 (NLG)  |
| PrefixLM        | 前缀双向，生成部分单向 | UniLM         | 前缀 + 生成     | 混合任务    |

所谓“单向注意力”和“双向注意力”其实就是看当前预测的token前后的子序列在预测时刻是否可见。

双向注意力在 self-attention 计算中，每个位置的 Query 可以和 **序列中所有位置的 Key** 交互，包括自己左边的 token 和右边的 token。

单向注意力在 self-attention 计算中，位置 _t_ 只能看到 **自己以及左边的 token**，不能看到右边的未来信息。

#### 9.2 比较

**_BERT_**：BERT从通吃NLP变成了特定领域（NLU：文本分类、匹配、序列标注、阅读理解）的热门工具了。但是BERT确实不如GPT更适合做生成任务，特别是下游任务的zero shot和few shot泛化性能远不如GPT这种decoder only且用 next token prediction 预训练的模型好。

> **Note**
>
> zero shot的意思是：**模型没有见过该任务的标注数据，也不经过针对任务的微调，就直接做任务**。
>
> - **BERT 不擅长 zero-shot**，因为它的预训练目标是 MLM，不是跨任务泛化，它不能直接通过自然语言指令做任务，通常需要针对任务微调。
> - **GPT、T5 等大语言模型**在 zero-shot 上更强，因为它们的预训练目标更贴近生成式任务，并且往往见过各种各样的任务模式。
>
> few shot类似，模型在**只给少量该任务的示例**的情况下，就能学会做任务的能力。

重点在于，为什么兼顾了encoder-decoder的双向attention的LLM也没有被广泛采用？

##### 9.2.1 decoder only有更好的Zero-Shot性能、更适合于大语料自监督学习

Google Brain 和 HuggingFace联合发表的 [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2204.05832) 曾经在5B的参数量级下对比了两者性能。论文最主要的一个结论是decoder-only模型**在没有任何tuning数据的情况下、zero-shot表现最好**，而encoder-decoder则需要在一定量的标注数据上做multitask finetuning才能激发最佳性能（finetune之后会比decoder only的更好）。

> **Note**
>
> **Multi-task fine-tuning** ：在**同一个模型**上，同时针对多个不同的下游任务进行微调，而不是一个任务微调完再换到另一个任务。好处在于可以提升模型的泛化能力（天然提升zero shot能力，更容易适应新任务），减少训练成本。
>
> - 模型的 encoder/decoder（共享部分）参数是公共的。
> - 每个任务可能有**单独的输出层**（task head），比如分类任务用 softmax 层，NER 用 CRF 层，问答用 span prediction 层。
> - 训练时交替喂不同任务的数据，让模型同时学习多个任务。
> - Loss 是多个任务 loss 的加权和
>
> 通常，multi task finetuning是需要人工标注的

而目前的Large LM的训练范式还是在大规模语料上做自监督学习，很显然，Zero-Shot性能更好的decoder-only架构才能更好地利用这些无标注数据。

##### 9.2.2 涌现能力替代了multi task finetuning

前面说到，5B参数量+170B token数据量时，在做multitask finetuning后encoder-decoder相比decoder-only反而在新任务上会有一定的优势。

但是LLM现在是大数据训练+大参数，表现出涌现能力。

> **Note**
>
> 涌现能力：
>
> 简言之，在模型参数量足够大时，模型的能力提升不再遵守以往的[log-linear的提升法则](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2001.08361)，而是突然急速增强性能。涌现能力的一个表现是，参数量达到一定量级后，模型具有了"复杂的推理能力"——譬如从非结构化的文本中自动地提取结构化的知识。
>
> ![img]({{ '/assets/blog/transformer-autoregressive-models/v2-690fde72f8fadeb15333a53ae826725f_1440w.webp' | relative_url }}){: .img-fluid loading="lazy" }

那么，LLM也可以自动地从大数据里面做self multitask finetuning。具体来讲，大数据里面本身天然蕴含了许多任务：比如 双语网页数据=>机器翻译、论文(摘要+正文)=>数据-文本摘要、维基百科数据=>命名实体识别等等。

因此，对于常见的NLP任务、LLM可以视为已经self finetuning过了；对于复杂问题，LLM的推理能力可以把这些问题转换成几个基本任务的组合。

这一点的论点是涌现能力完全可以覆盖对encoder-decoder做multi task finetuning的差距。

##### 9.2.3 [In-context learning](https://zhida.zhihu.com/search?content_id=563197831&content_type=Answer&match_order=1&q=In-context+learning&zhida_source=entity)对LLM有[few-shot finetune](https://zhida.zhihu.com/search?content_id=563197831&content_type=Answer&match_order=1&q=few-shot+finetune&zhida_source=entity)的作用

在实际使用LLM时，我们经常会加入[Chain-of-Thought](https://zhida.zhihu.com/search?content_id=563197831&content_type=Answer&match_order=1&q=Chain-of-Thought&zhida_source=entity)或者In-Context信息来作为prompt进一步激发模型潜力——例如加入一些例句让GPT类模型来模仿、生成更好的结果。

论文《Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers》指出In-Context信息可以视为一种task finetuning，论文的数学推导是定性的，大体上是将prompt信息归为对Transformer Attention层参数的微调。

按照这篇论文的思路，decoder-only的架构相比encoder-decoder在In-Context的学习上会更有优势，因为前者的prompt可以更加直接地作用于decoder每一层的参数，微调信号更强。也因此，更适合ChatGPT这类开放域的对话模型作为基础模型。同理，在总数据量少的情况下，In-Context + Decoder-only也更具有few-shot的优势——Google近期的论文在机器翻译上也观察到了类似现象，即中等量的单语数据+大模型+In context就能学习出很好的翻译效果。

##### 9.2.4 注意力秩

双向attention的注意力矩阵容易退化为低秩状态，而causal attention的注意力矩阵是下三角矩阵，必然是满秩的，建模能力更强。

这里可能会有个疑惑，为什么全矩阵反而容易信息冗余？

主要还是attention的模式带来的：双向attention每次都能看到全局信息，不容易带来新的线性独立性，但是单向attention每次做attention都是不同的token组合（是当前位置和之前位置的组合，因此每次都会引入新的token），一定会引入新的线性独立性，信息更丰富。

##### 9.2.5 causal attention隐式的位置编码功能

decoder only的causal attention相较于双向attention具有隐式的位置编码功能，打破了transformer的位置不变性，而带有双向attention的模型，如果不带位置编码，双向attention的部分token可以对换也不改变表示，对语序的区分能力天生较弱。

##### 9.2.6 decoder only支持一直复用KV cache

decoder only的模型，每个token的表示只和它之前的输入有关，而encoder-decoder和PrefixLM就难以做到。

##### 9.2.7 **轨迹依赖**

以decoder-only架构为基础摸索出了一套行之有效的训练方法和Scaling Law，后来者鉴于时间和计算成本，自然不愿意做太多结构上的大改动，继续沿用decoder-only架构。在工程生态上，decoder-only架构也形成了先发优势，Megatron和flash attention等重要工具对causal attention的支持更好。
