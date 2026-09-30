---
layout: "post"
title: "KTransformers：CPU 与 GPU 混合 MoE 推理"
date: "2026-09-30"
description: "总结算术强度感知 CPU 内核、异步调度、专家延迟与 NUMA 并行。"
categories: ["Papers"]
tags: ["mlsys", "moe", "cpu-gpu", "inference"]
permalink: "/blog/paper-ktransformers/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/EdgeInfer/KTransformers.md"
toc:
  beginning: true
---

**_KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models_**

---

## 1. Motivation

在本地部署超大规模 MoE 模型（如 DeepSeek-V3 671B）时，单张消费级显存（VRAM）容量严重不足。虽然 Fiddler 等工作提出了 CPU/GPU 混合推理，但依然存在三个核心瓶颈：

1. CPU 计算能力被严重低估（7% 的利用率） 这是本文最核心的观察。尽管现代 CPU 有 AMX/AVX-512 等指令集，但在现有框架（如 PyTorch + oneDNN）中，由于内存布局不匹配和线程同步开销，实际性能仅能达到理论峰值的 7% 。对于处理 Prefill 阶段的高算术强度任务，CPU 成了系统的拖油瓶。

2. 混合推理中的“内核启动”开销爆炸 在混合执行模式下，CPU 频繁调度 GPU 任务。Fiddler 在 Decode 阶段每个 token 会触发超过 7,000 次 CUDA kernel 启动，启动开销占 GPU 总执行时间的 73% 。由于 CPU 和 GPU 之间频繁的同步，现有的 CUDA Graph 优化难以生效，导致 GPU 经常处于“饥饿”状态。

3. 串行执行导致硬件闲置 传统的 Transformer 结构规定了 Attention（GPU）和 MoE（CPU）必须交替执行。这种严格的依赖关系导致 CPU 在 GPU 计算 Attention 时只能空转，反之亦然，整体硬件利用率极低（GPU 往往低于 30%） 。

---

## 2. Design

### 2.1 算术强度（ARI）感知的 CPU 算子优化

针对 CPU 性能释放，论文设计了专门的指令集切换逻辑：

- **Prefill 阶段（高 ARI）**：使用 **AMX-specialized kernels**。通过 Tile-aware 的内存布局预处理，让权重矩阵在加载时就对齐到 64 字节缓存行，消除推理时的转置开销 。
- **Decode 阶段（低 ARI）**：由于 token 数量少，AMX 的瓦片处理反而开销大，系统会自动降级到更轻量的 **AVX-512 kernels**，从而获得 1.20x 的 Decode 加速 。

### 2.2 极致异步的调度：单 CUDA Graph 封装

为了解决同步开销，KTransformers 引入了异步任务队列：

- **cudaLaunchHostFunc**：利用此接口将 CPU 的控制流（如路由专家推送）封装进 CUDA 流中 。
- **全流程封装**：将一整层的 Decode 计算逻辑封装进**单个 CUDA Graph 实例**。通过 CUDA-based spinning 减少主机中断，将 GPU 启动开销降至接近 0 。

### 2.3 核心杀手锏：Expert Deferral（专家延迟执行）

> **Important**
>
> **Expert Deferral 的实现逻辑：**
>
> 论文利用了 Transformer **残差连接（Residual Connections）** 的鲁棒性 。
>
> 1. **立即专家（Immediate Experts）**：计算结果立即喂给下一层 。 2. **延迟专家（Deferred Experts）**：计算结果被推迟到 $$k+2$$ 层才合并 。
>
> **逻辑流程图：**
>
> ```
> Layer k Attention (GPU)
>       │
>      ├─ GPU 计算 Shared Experts
>       │         │
>      │         └─ 并行开启 CPU 计算 Layer k-1 的 Deferred Experts
>       │
>      └─ CPU 计算 Layer k 的 Immediate Experts
> ```
>
> 这种重叠策略将 CPU 利用率从 75% 提升至 **100%**，带来了额外 1.45x 的吞吐提升 。

### 2.4 NUMA 感知的张量并行（TP）

针对双路服务器，论文发现传统的专家并行（EP）会导致 Socket 负载严重不均。

- **策略**：将每个专家的权重矩阵沿 row/column 维度切分到不同 Socket 内存中 。

- **收益**：避免了昂贵的跨 NUMA 节点内存访问，Decode 吞吐提升 1.63x 。

---

## 3. Evaluation

**端到端性能：**

- **Prefill**：相比 Fiddler 获得 **4.62-19.74x** 的加速 。
- **Decode**：相比 Llama.cpp 获得 **1.66-4.90x** 的加速 。

**精度损失：**

- 在 HumanEval, GSM8K 等 Benchmark 上，Expert Deferral 带来的准确率波动控制在 **0.5%** 以内 。

**硬件利用率：**

- 在 DeepSeek-V3 案例中，GPU 利用率从 28% 提升至 **37%**，CPU 利用率彻底打满 。
