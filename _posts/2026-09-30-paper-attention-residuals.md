---
layout: "post"
title: "Attention Residuals：深度方向的注意力聚合"
date: "2026-09-30"
description: "解析 Block AttnRes、伪查询与两阶段推理如何降低历史表示读取开销。"
categories: ["Papers"]
tags: ["models", "transformer", "attention", "inference"]
permalink: "/blog/paper-attention-residuals/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/AttentionResidual.md"
published: false
toc:
  beginning: true
---

## Background

Transformer 中的 standard residual connection 通常写作：

$$

h_l = h_{l-1} + f_{l-1}(h_{l-1})


$$

把它展开后，当前层输入实际上等于 token embedding 与所有 earlier-layer outputs 的统一加和。也就是说，residual connection 除了提供 gradient highway 之外，还隐含定义了 **深度方向的信息聚合规则**。在这个规则里，每一层的贡献系数固定为 1，缺乏 selective aggregation 机制。

---

## Motivation

sequence mixing 和 expert routing 已经普遍采用 learnable、input-dependent 的加权方式，但 depth-wise aggregation 仍然停留在固定加和。

---

## Challenges

### 算法层面

如果直接把 Full AttnRes 实现为“当前层对所有 earlier-layer outputs 做 softmax attention”，那么当前层输入需要显式访问所有前层输出，意味着 source set 随网络深度线性增长。

### 推理阶段

如果每一层都单独对前面所有 blocks 做一次 attention residual，那么每层都要完整扫描历史 blocks，造成 $$O(L \cdot N)$$ 的 memory accesses。换句话说，问题不只是 arithmetic cost，而是 **HBM 上对历史表示的重复读取**。

### 算法与实际实现的冲突

更强的 input-dependent query 形式理论上会更有表达力，但如果 query 依赖当前 hidden state，就无法提前或批量计算 block 内多层的 inter-block attention，系统上难以 amortize memory access。因此论文必须做权衡。

---

## Innovations

> 图片待补充：image-20260327143435545.png

### 1 Attention Residuals：把固定 residual accumulation 改成 depth-wise softmax attention

论文提出 AttnRes，用 softmax attention 替代标准 residual 中的固定加和。当前层输入不再是 earlier-layer outputs 的统一和，而是：

- 先由一个 learned pseudo-query $$w_l$$ 对 earlier-layer representations 打分；
- 再通过 softmax 得到注意力权重；
- 最后对这些 earlier representations 做加权求和，形成当前层输入。

这一点的本质创新在于：
**把 residual 从固定深度累积，改成了深度方向的 selective retrieval。**

---

### 2 Block AttnRes：把 layer-level source set 压缩成 block-level source set

为了让 AttnRes 在大规模训练中可扩展，作者提出 Block AttnRes。它将 $$L$$ 层划分为 $$N$$ 个 blocks，在 block 内仍然保留标准 residual accumulation，而只在 block 间做 attention。每个 block 的表示定义为该 block 内所有 layer outputs 的和：

$$

b_n = \sum_{j \in B_n} f_j(h_j)


$$

这样就把需要显式保留和通信的对象，从逐层 outputs 压缩成 block-level summaries，将 memory 与 communication overhead 从 $$O(Ld)$$ 降到 $$O(Nd)$$。

---

### 3 pseudo-query：为两阶段执行服务的算法设计

论文没有采用 input-dependent query，而是为每层使用一个 learned pseudo-query $$w_l$$。这个设计的重要意义不只是“轻量”，而是 **与 forward hidden state 解耦**。因此，同一个 block 内所有层的 queries 在 block 开始前就已知，可以一次 batched 地对历史 block representations 做 inter-block attention。

换句话说，pseudo-query 的价值不是消除当前 block 内的 sequential dependency，而是：

- 把 inter-block attention 从 sequential path 中剥离出来；
- 使这部分计算可以提前、批量执行；
- 从而 amortize 对历史 blocks 的 memory reads。

---

### 4 two-phase inference：推理侧的 memory I/O 优化

作者提出 two-phase computation strategy：

- **Phase 1**：对一个 block 内所有层，同时执行对历史 blocks 的 inter-block attention；
- **Phase 2**：顺序处理当前 block 内依赖 evolving partial sum 的 intra-block attention，并通过 online softmax 与 Phase 1 结果合并。

这一设计的关键价值在于：

- 历史 block representations 只需读取一次，而不是每层重复读取；
- 额外 memory access 被 block 内多层均摊；
- inference latency overhead 最终控制在 2% 以内。

---

## Implementation

### 1 Block representation 与 partial sum

Block AttnRes 中维护两类状态：

- `blocks`：已经完成的 block representations，即 $$[b_0, \dots, b_{n-1}]$$
- `partial_block`：当前 block 到当前子层之前为止的 partial sum，即 $$b_n^i$$

当前子层做 attention residual 时，候选 source 由这两部分组成，而不是逐层访问所有 earlier-layer outputs。

### 2 `block_attn_res` 的逻辑

伪代码中：

- `V = torch.stack(blocks + [partial_block])`：把历史 block reps 与当前 partial sum 拼成 values；
- `K = norm(V)`：对候选 source 做 RMSNorm，作为 keys；
- `proj.weight.squeeze()`：作为当前子层的 pseudo-query；
- `logits.softmax(0)`：沿 depth/block 维度做 softmax；
- 最终对 `V` 做加权和，得到当前子层输入 `h`。

也就是说，AttnRes 的输出不是一个额外加项，而是 直接形成当前 attention 或 MLP 子层的输入 hidden state。

### 3 在 Transformer block 内部的执行位置

在 Block AttnRes 的伪代码里，这个操作不是每个 transformer layer 只做一次，而是：

- 在 attention 子层前 做一次；
- 在 MLP 子层前 再做一次。

因此，AttnRes 替代的是“当前子层输入 hidden state 如何由深度历史构造”的机制，而不是替代 attention / MLP 本身。Transformer 内部的 attention、FFN 仍然负责生成新的 layer outputs，供后续层继续使用。
