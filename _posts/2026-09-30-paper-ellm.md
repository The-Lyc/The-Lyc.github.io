---
layout: "post"
title: "eLLM：自适应 KV 缓存提升推理吞吐"
date: "2026-09-30"
description: "解析 token/layer 粒度缓存、交换重算重叠、内核融合及双层优化。"
categories: ["Papers"]
tags: ["mlsys", "kv-cache", "recomputation", "kernel-fusion"]
permalink: "/blog/paper-ellm/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/eLLM.md"
toc:
  beginning: true
---

**_eLLM: High Throughput and Low Latency LLM Serving via Adaptive KV Caching_**

---

## Motivation

这篇论文的出发点是：**LLM decoding 的 GPU memory 已经被 KV cache 占满，但 GPU compute 往往没有被充分利用。**

论文在 A100 上运行 Llama2-13B 时观察到，KV cache 长时间占据 50%–100% 的 GPU memory，而 compute utilization 主要只有 0%–10%。即使换成使用 MLA 的 DeepSeek-V2-Lite，memory occupancy 仍达到 97.4%，median GPU utilization 只有 35%。

现有 KV offloading/recomputation 方法存在两个问题：

1. **采用 all-or-nothing 的 request-level policy。**

   一个 request 的 KV 要么全部留在 GPU，要么整体 swap 到 host，要么全部丢弃后重新计算。这些方法可以恢复被 preempt 的 request，但 active requests 在 decoding 过程中仍然需要保存完整 KV，因此 GPU memory 仍然限制 concurrent request 数量。

2. **单独使用 swap 或 recomputation 都存在瓶颈。**

   swap 受 PCIe bandwidth 限制，完整 recomputation 又会消耗大量 GPU compute。与此同时，decoding 本身偏 memory-bound，GPU 上仍然存在未被利用的 compute resource。

也就是说，现有系统把 KV cache 看成“保存或不保存”的离散选择，而没有利用 decoding 阶段的 memory–compute imbalance，把部分空闲 compute 转化为额外的 KV-cache capacity。

---

## Key Insight

eLLM 的核心想法是：**不要求 active request 的全部历史 KV 常驻 GPU，而是只缓存部分 token 的 KV，对其余 token 在 decoding 过程中逐层重新计算。**

假设一个 request 有 $$n$$ 个历史 token，eLLM只缓存后面的 $$n-s$$ 个 token，而将前面的 $$s$$ 个 token 设为 uncached。执行到某个 layer 时：

1. 重新计算这 $$s$$ 个 token 在当前 layer 的 KV；
2. 与 GPU 中已经缓存的其余 KV 拼接；
3. 完成当前 token 的 attention 和 decoding；
4. 立即释放临时重算得到的 KV。

这样可以形成连续的 memory–compute trade-off：

$$

\text{降低 cached-token ratio}
\rightarrow
\text{减少单个 request 的 GPU memory}
\rightarrow
\text{扩大 batch concurrency}
\rightarrow
\text{提高 throughput、降低排队 TTFT}


$$

代价是额外 recomputation 可能增加 TPOT，因此需要在 TPOT SLO 约束下动态寻找合适的 batch size 和 cache ratio。

论文还观察到：historical-token recomputation 偏 compute-intensive，而 current-token decoding 偏 memory-intensive。两类 operation 可以通过 kernel fusion 并行执行，从而利用互补的 GPU resource demand。

---

## Design

### 1. Adaptive Token-wise and Layer-wise KV Caching

第一个设计是把 KV management 从 request granularity 细化到 token 和 layer granularity。

对于每个 active request，eLLM动态维护 cached-token ratio：

- cached tokens 的全部 layer KV 常驻 GPU；
- uncached tokens 的一部分 layer KV 保存在 host，使用时 swap 回 GPU；
- 其余 layer KV 不保存，使用时直接在 GPU 上 recompute。

原始 PagedAttention block 通常包含一组 token 在全部 $$L$$ 层的 KV，不适合只保存特定 token、特定 layer，否则会造成严重的 internal fragmentation。

eLLM因此将一个 memory block 改为只覆盖连续 $$F$$ 层，默认 $$F=4$$，并维护包含 `seq_id`、`token_id`、`layer_id`、logical/physical block ID 等信息的 mapping table。这样，同一 request 的不同 token 和 layer 可以分别进行 cache、swap 或 recompute。

### 2. Request-level Optimization

第二个设计负责决定：**一轮 decoding 应当同时执行多少 request，以及每个 request 有多少比例的 token 不缓存。**

其主要控制变量是：

- batch size $$b$$；
- uncached-token ratio $$r$$。

eLLM根据当前 request 的 sequence length、waiting time、模型参数、GPU memory 和 FLOPS，建立 recomputation/decoding latency model，并求解：

$$

\max_{b,r}\frac{b}{T(b,r)}


$$

约束包括：

- batch processing time 加等待时间不能超过 TPOT SLO；
- cached KV、model weights 和临时 KV 必须放入 GPU memory；
- batch size 不能超过 waiting queue 中的 request 数。

每个 decoding iteration 都会使用当前 request metadata，通过 SLSQP 重新求解 $$(b,r)$$。因此，request arrival 或 sequence length 发生变化时，eLLM可以重新调整 concurrency 和 cache ratio。

### 3. Layer-wise Swap–Recompute Overlapping

确定 $$(b,r)$$ 后，layer-level optimizer 决定 uncached tokens 的 KV 在接下来若干层中如何恢复。

它不是 LRU，也不是根据 layer 的访问热度选择。Transformer decoding 会顺序访问所有 layer，因此各 layer 不存在传统 cache 中的冷热差异。它的目标只是：

> 让 GPU recomputation 和 CPU→GPU KV transfer 的执行时间尽量接近，从而减少两条 CUDA stream 之间的 time bubble。

eLLM限制每次最多同时：

- swap 3 个 layer；
- recompute 2 个 layer。

因此只枚举六种 $$(l_{swap},l_{recompute})$$ 组合。swap latency 由 token 数、layer 数和预先 profile 的 PCIe 参数估算；recomputation latency 根据 batch size、sequence length、uncached ratio 和 GPU FLOPS 估算。最终选择：

$$

(l_{swap}^{*},l_{recompute}^{*})
=
\arg\min
\left|
T_{swap}-T_{recompute}
\right|


$$

按照论文给出的 pipeline，当前附近的 layer 通过 recompute 恢复，而更后面的 future layers 通过 H2D swap 提前 prefetch。例如：

- recompute $$L_i,L_{i+1}$$，同时 swap $$L_{i+2}$$；
- recompute $$L_i$$，同时 swap $$L_{i+1},L_{i+2}$$。

swap 后面的 layer 不是因为这些 layer 更重要，而是因为当前 layer 的 KV 在 decode 前必须已经 ready；能够与当前计算 overlap 的 transfer 只能为 future layers 做 prefetch。

### 4. Layer-wise Kernel Fusion

为了进一步隐藏 recomputation overhead，eLLM将两类 operation 融合：

$$

\underbrace{\text{Decode current token at }L_i}_{K_2}
\parallel
\underbrace{\text{Recompute historical-token KV at }L_{i+1}}_{K_1}


$$

其中 $$K_1$$ 更 compute-intensive，$$K_2$$ 更 memory-intensive。eLLM使用 horizontal fusion 将二者放进同一个 kernel，并根据两类 operation 的估算 FLOPs 比例划分 threads；thread 数按照 warp granularity 对齐到 32 的整数倍。

此外，相邻 layer 的 operation 还会进行 vertical fusion，以减少 kernel launch 和 intermediate state 的 global-memory write/read。实现中预编译了 thread count 从 32 到 1024 的 31 种 kernel configuration，runtime 直接选择合适的版本。

### 5. Dual-level Closed-loop Optimization

request-level optimizer 最初不知道 overlap 和 kernel fusion 会使用多少 temporary memory，因此先使用最坏情况：

$$

M_o^{initial}=40hbsr


$$

layer-level optimizer 得到实际的 $$(l_{swap},l_{recompute})$$ 后，将 temporary memory 更新为：

$$

M_o=8hbsr(l_{swap}+l_{recompute})


$$

再把 $$M_o$$ 反馈给 request-level optimizer，重新调整 $$(b,r)$$：

$$

(b,r)
\rightarrow
(l_{swap},l_{recompute},\delta)
\rightarrow
M_o
\rightarrow
(b,r)


$$

这里的 closed loop 主要是 request-level 与 layer-level configuration 之间的迭代，并不是根据实际 compute/IO completion time 进行 feedback control。

---

## Evaluation

eLLM基于 vLLM 实现，包含约 3,500 行 Python 和 1,700 行 CUDA code。在 4×A100、Llama2-13B/70B、ShareGPT 与长上下文 L-Eval 上，相比 vLLM-Recompute、vLLM-Swap 和 HCache：

- 最高提升 3.03× token throughput；
- 最高降低 2.63× TTFT；
- SM utilization 最高提高约 15%；
- long-context 场景平均 cached ratio 为 0.53，节省超过 47% KV memory；
- TPOT SLO attainment 达到约 96.6%–98.6%。

性能提升主要不是来自降低单个 request 的计算量，而是 partial caching 使 GPU 能够同时容纳更多 active requests，从而提高 decoding throughput 并减少 request 在 prefilling 前的排队时间。

---

## Takeaway

eLLM的核心贡献可以总结为：

> **将 active-request KV caching 从 all-or-nothing policy 变成 token-wise、layer-wise 的连续 memory–compute trade-off，并联合优化 batch concurrency、KV residency、swap/recompute overlap 和 fused-kernel execution。**

需要注意，eLLM的 layer-level optimizer 本质上是 small-space enumeration 加 analytical/profile-based latency model。它以完整 layer 为 swap/recompute 单位，只能选择时间差最小的离散组合，不能保证恰好实现 compute–IO balance；当 effective PCIe bandwidth 或 available GPU compute 动态变化时，已有配置也不会根据真实 time bubble 自动修正。

这与 RunKV 的边界在于：eLLM主要决定为了扩大 concurrency，应当让多少 KV 不再常驻 GPU；RunKV则面向已经 non-resident 的 KV，通过 layer 内更细粒度的 replay budget 和实际 execution imbalance，在线决定 transfer 与 replay 的分工。
