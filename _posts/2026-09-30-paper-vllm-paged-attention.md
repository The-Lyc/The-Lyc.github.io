---
layout: "post"
title: "vLLM：PagedAttention 与 KV 显存管理"
date: "2026-09-30"
description: "梳理 KV 碎片、共享、换出恢复、分布式映射与融合内核。"
categories: ["Papers"]
tags: ["mlsys", "vllm", "paged-attention", "kv-cache"]
permalink: "/blog/paper-vllm-paged-attention/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/vLLM.md"
toc:
  beginning: true
---

《Efficient Memory Management for Large Language Model Serving with PagedAttention》这是vLLM的原版论文。

---

## Motivation：

### 1. Memory Fraction

decode阶段，单request的服务是memory bound，因为需要加载之前的kv cache，但只生成一个token，没法像prefill阶段那样batch，因此需要想办法batch多个request的服务。但多request服务会带来显存利用率低的问题：

<img src="{{ '/assets/blog/paper-vllm-paged-attention/image-20251128115108308.png' | relative_url }}" alt="image-20251128115108308" style="max-width: 100%; height: auto;" loading="lazy" />

可以看到，vllm之前的框架orca在多request服务的场景下显存碎片占比非常高。

显存碎片包括三种：

![image-20251128115249945]({{ '/assets/blog/paper-vllm-paged-attention/image-20251128115249945.png' | relative_url }}){: .img-fluid loading="lazy" }

1. reserved fraction

这个其实就是internal fraction的一种，只不过强调的是，reserved fraction部分在整个decode结束之后是会被利用上的，但是过程中还没decode的部分就成了reserved fraction。因为这部分即使还没被decode用上也要预留，就成了reserved fraction。

1. internal fraction

internal fraction强调的是由于按照最大token数分配的空间中，就算decode完也没用上的那部分空间，这部分是完全的浪费。

1. external fraction

这是由于每个request的最大token可能都不一样，因此会产生许多大小不一的内存空洞，下一个request要用的时候又无法填满。

论文中也谈到了LLM Serving中Memory Challenge，以13B的OPT model为例，一个token的KV Cache就要占***800KB***（calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)）。而OPT默认的配置是一条request最多推理2048个tokens，也就是说一条request就要占***1.6GB***的KV cache。

### 2. KV cache Sharing

LLM服务常用的一些decode方法，比如parallel sampling和beam search等场景下，不同request可能会共享它们的KV cache。如果没有Paged Attention，每个request的KV cache就在不同的连续空间里，是没法share的。

---

## Paged Attention

总结起来，思路其实很简单，跟操作系统上的分页非常类似。

对于同一个模型，每个token的KV cache的大小应该是一样的，即使是多头注意力，每个head上每个token的kv cache大小也是一样的，因此作为内存分配的基本单位的block的大小就是整数倍的每个head上每个token的kv cache大小（这个整数倍的取值可以用实验来得到较好的值）。然后就按需分配block，把每条request的fraction的大下控制在一个block的大小之内（每条request的kv cache都是在已分配的block中顺序存储的，因此最多有一个block没填满）。

对于***kv cache sharing***的问题，采用COW的方法，按需去分配新的block存储有变化的部分，没有变化的部分share同一个block。

对于***显存耗尽***的问题，需要考虑哪些blocks需要被驱逐以及当需要这些blocks的时候如何恢复。vLLM采用的是要么驱逐整个request的kv cache要么完全不驱逐某个request的kv cache。有sharing的requests需要被考虑为一个整体。

然后就有了怎么恢复的问题：两种手段，swap或者recomputation。swap从后来的request开始驱逐，并且在当前留在显存中的requests全部完成之前是不会接收新的request。当一条request完成之后，其kv cache被释放，把之前swap出去的request换回显存，继续处理。recomputation就是直接放弃这个request的kv cache，下一次有空间服务这个request的时候，重新开始处理，减小的是换入换出的延迟。这两种方法可以有tradeoff。

对于 **_分布式_** 的问题，观察到多头注意力天然是TP，且每个worker上的shard只在attention做完之后再all-reduce做concat，也就是说每个worker上对应的kv也是完全可以切片且各自单独计算。vLLM只有一个KV cache manager用于管理分布式的kv cache，每个request的每个token在每个worker上都是一致的与物理block的映射关系。

既然用了page attention来管理kv cache，那么就要重写相关的kernel，vLLM在kernel层级也做了对应的优化：

1. Fused Reshape and Block Write (融合的变形与写入算子)

**场景**：当模型计算出新的 Key 和 Value 向量时，需要把它们存入 KV Cache。

- **痛点**：
  - 传统的 KV Cache 是连续的大张量，直接写进去就行。
  - 在 vLLM 中，数据必须被切分成小块（Split into blocks）。
  - 为了后续读取更快，数据还需要重排（Reshape），因为 vLLM 采用了一种特殊的内存布局（例如将 `x` 维度向量化以利用 tensor cores 或向量指令）。
  - 最后还要查 Block Table 找到物理地址写入。
  - 如果分三步做（Split -> Reshape -> Write），需要启动三次 Kernel，**启动开销（Launch Overhead）** 太大。

- **优化**：
  - **算子融合（Fusion）**：vLLM 写了一个 Kernel，一次性完成“切分、重排、查表、写入”所有动作。这极大地减少了 CPU 与 GPU 之间的交互次数。

2. Fusing Block Read and Attention (融合的读取与注意力算子)

**场景**：这就是核心的 **PagedAttention Kernel**，用于在计算 Self-Attention 时读取非连续的 KV Cache。

- **基础**：他们魔改了 NVIDIA **FasterTransformer** 的 Attention Kernel。
- **挑战**：传统的 Attention 假设 KV 是连续的。现在 KV 散落在显存的各个角落，需要查 Block Table 才能找到。
- **关键优化技术**：
  - **Warp-Level读取（Assign a GPU warp to read each block）**：这是一个非常底层的显存优化。
    - GPU 的最小执行单元是 **Warp**（通常包含 32 个线程）。
    - vLLM 让一个 Warp 里的 32 个线程协同工作，去读取同一个 Block 的数据。
    - **目的**：实现 **Coalesced Memory Access（合并访存）**。简单说，就是让 32 个线程像一支整齐的军队一样，一次性把一段连续的显存数据“搬”上来，而不是每个人乱跑去读零散的数据，从而占满显存带宽。
  - **支持变长序列**：原生支持 Continuous Batching，即一个 Batch 里有的请求长、有的请求短，Kernel 能自适应处理，不会因为 padding 浪费计算。

3. Fused Block Copy (融合的块复制算子)

**场景**：主要用于 **Copy-on-Write（写时复制）** 机制。这常见于 **Beam Search**（集束搜索）或 **Prefix Caching** 场景。

- **问题**：
  - 当一个序列分叉（Fork）时（比如 Beam Search 中生成了 top-k 个候选），我们需要复制一些 Block。
  - 这些需要复制的 Block 在物理显存里通常是**不连续**的。
  - **笨办法**：如果我有 100 个不连续的块要复制，调用 100 次 `cudaMemcpyAsync` API。
  - **后果**：CPU 发送这 100 次指令的时间，可能比 GPU 实际拷贝数据的时间还长！这叫“CPU 瓶颈”或“Launch Overhead 过高”。
- **优化**：
  - **批处理（Batching）**：vLLM 写了一个 Kernel，允许 CPU 把所有需要复制的 `(源地址, 目的地址)` 对打包成一个列表传给 GPU。
  - **一次启动**：GPU 收到列表后，在这个单一的 Kernel 里并行地把这 100 个块全部搬完。
