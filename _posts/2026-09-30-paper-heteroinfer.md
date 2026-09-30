---
layout: "post"
title: "HeteroInfer：移动 SoC 上的 GPU 与 NPU 协同推理"
date: "2026-09-30"
description: "梳理移动 SoC 的张量敏感性能、混合分区、快速同步与零拷贝设计。"
categories: ["Papers"]
tags: ["mlsys", "edge-inference", "gpu", "npu"]
permalink: "/blog/paper-heteroinfer/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/EdgeInfer/HeteroInfer.md"
published: false
toc:
  beginning: true
---

**_Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference_**

---

## 1. Motivation

移动SoC上已经有GPU和NPU两个AI加速器，但现有推理引擎（MLC、MNN、llm.npu、PowerInfer2）都只用其中一个，资源浪费明显。论文的切入点是：为什么没人同时用GPU+NPU？

作者给出了三个根本障碍：

**1. NPU性能高度不稳定（tensor-sensitive）** 这是motivation里最有价值的观察。NPU的FLOAT性能不是一个固定数字，而是随tensor的shape和order剧烈变化。这意味着简单地按算力比例切分workload，NPU可能跑出远低于预期的性能，GPU+NPU并行的收益根本无法保证。这也是之前工作不敢这么做的核心原因之一。

> **Important**
>
> NPU sensitive的三个方面：
>
> ### NPU-1：Stage Performance
>
> 原因是NPU内部的systolic array是固定尺寸的，Snapdragon 8 Gen 3上是多个32×32的阵列。
>
> 当tensor的维度不是32的整数倍时，编译器会自动padding到对齐值，但执行时间按padding后的大小计算。更关键的是，执行时间不是随tensor size线性增长的，而是**阶梯状**的。

> 图片待补充：image-20260410002513991.png

> 也就是说K=1和K=31跑的时间完全一样，因为都被padding到32，占用同样多的systolic array资源。这对decode阶段（K很小）伤害极大，明明计算量很少，却要付出处理32个元素的时间。
>
> ### NPU-2：Order-sensitive Performance
>
> 原因是NPU采用**weight stall**计算范式：weight预加载到每个PE里保持不动，activation流过去做计算，目的是减少weight的反复load/store。
>
> 这个范式的前提是**weight tensor能完整放进片上SRAM**。一旦weight太大放不下，就需要反复从外部DRAM把weight分块load进来，内存访问开销急剧上升。
>
> 所以input和weight应该选维度小的那个作为驻留PE的。
>
> ### NPU-3：Shape-sensitive Performance
>
> 这个是在weight已经确定是维度更小的前提下，进一步看activation（input tensor）自身的row/column比例对性能的影响。
>
> 原因同样是weight stall：input tensor的column size是和weight tensor共享的（矩阵乘法的内积维度），column越大意味着weight越大，越容易超出片上SRAM容量，weight stall的优势就越弱。
>
> 所以即使input比weight大，**当input的row size > column size时，NPU性能更好**；反之column size大时性能下降。

**2. GPU-NPU同步开销过大** `clFinish`在Snapdragon 8 Gen 3上有约400μs的固定开销，而decode阶段一个Matmul kernel本身只需几百μs。同步开销和计算时间同量级，并行反而可能变慢。

**3. Decode阶段是memory-bound的** Decode阶段的瓶颈不是算力不够，而是内存带宽打不满。单个处理器（GPU或NPU）在decode时只能用到40-45 GB/s，而SoC理论峰值是68 GB/s，存在明显的带宽浪费。

---

## 2. Design

design就是针对motivation提出的三大问题一一展开的。

### 2.1 适合GPU-NPU协作的并行方式

论文这部分的design思路是递进的，从最简单的静态prompt length开始，进一步考虑动态变化的length，再进一步考虑NPU的硬件特点。

LLM的核心计算是矩阵乘法，形式是：

$$

\text{Output} = \text{Activation} \times \text{Weight}


$$

- **Activation**（输入）：shape是 $$[M, N] $$，其中 $$M $$ 是sequence length，$$N $$ 是hidden size
- **Weight**（权重）：shape是 $$[N, K] $$，$$K $$ 是输出维度
- **Output**：shape是 $$[M, K] $$

NPU的根本问题是：它只能接受编译时确定的静态shape，而且即使shape确定了，不同shape的性能也差异巨大。

#### 2.1.1 静态shape的weight-centric partition(TP)

##### 适用场景

sequence length已知且固定（比如离线推理，或者prefill的标准长度档位）。

##### 核心思路

沿**weight tensor的row维度**切分，GPU和NPU各拿一块weight，同时对**同一份Activation**做矩阵乘法，最后把输出拼起来。

> 图片待补充：image-20260409234232369.png

按row做TP的好处是，NPU和GPU各拿到完整的activation可以独立做sub-matmul，最后直接concat结果就可以了。TP的切分比例就可以根据NPU的计算图可接受的维度和NPU、GPU各自的算力来决定。当然仅作TP，是很有可能选不到一个合适的ratio的。

#### 2.1.2 动态shape的activation-centric partition(DP)

##### 问题背景

现实中用户输入的sequence length是任意的，比如300、537、1000……

NPU需要提前编译静态计算图，如果sequence length不在预编译的档位里（比如只预编译了128、256、512、1024），就需要做额外的适配操作。

传统做法是**padding**：sequence length=300，就pad到512，NPU跑完再把多余的输出扔掉。但这样浪费了 $$\frac{512-300}{512} \approx 41\% $$ 的计算。

##### 核心思路

不对Activation整体padding，而是**沿sequence length维度切分**（其实就是按row做DP，下图里activation是做了transpose，所以seq length在column维度）：

- 能对齐标准shape的部分 → 送NPU（用预编译的静态图）
- 对不齐的余量部分 → 送GPU（GPU可以处理任意shape）

> 图片待补充：image-20260409234926717.png

这样相当于就是NPU和GPU各拿一部分sub-seq做sub-matmul，仍然是只需要各自计算完做一次concat就可以了。

这种DP方式就是针对dynamic shape里面引起dynamic的维度（seq length）做的切分，也就解决了dynamic shape带来的NPU计算图适配的问题。

#### 2.1.3 扩大切分方式搜索空间的Hybrid Partition（TP+DP）

##### 问题背景

Activation-centric有一个缺陷：切完之后，GPU拿到的那块可能：

- 太小 → GPU利用率低（GPU-1: linear performance，小tensor是memory-bound）
- NPU拿到的shape比例不对 → 触发NPU-3: shape-sensitive performance问题

具体来说，**FFN-down layer**的weight shape是 $$[2048, 8192] $$（column > row），对NPU非常不友好（NPU的shape sensitive），单靠Activation切分无法解决这个问题。

##### 核心思路

**同时在两个维度上切**：

1. **Activation维度**（处理动态shape）：把余量扔给GPU
2. **Weight维度**（处理shape-sensitive问题）：在NPU内部或GPU-NPU之间进一步按weight row切分

这个过程可以描述为：

```
Activation [M=300, N]
      │
      │  先按seq维度切（activation-centric）
      │
      ├── Activation_NPU [256, N]
      │         │
      │         │  再按weight row维度切（weight-centric）
      │         │
      │         ├── Weight_NPU_part1 [npu_size1, N]^T → NPU sub-graph 1
      │         └── Weight_NPU_part2 [npu_size2, N]^T → NPU sub-graph 2
      │
      └── Activation_GPU [44, N]
                │
                └── Weight_GPU [gpu_size, N]^T → GPU

最终merge所有输出 → Output [300, K]
```

### 2.2 针对边缘设备SoC的Fast Synchronization

这个design是基于：GPU-NPU同步开销过大， `clFinish`在Snapdragon 8 Gen 3上有约400μs的固定开销，而decode阶段一个Matmul kernel本身只需几百μs。同步开销和计算时间同量级，并行反而可能变慢。

现有的三种synchronization的比较：

- `clFinish`：精确但太贵（400μs overhead）

- busy-wait polling：精确但一直占用CPU core，功耗高

- `usleep`：便宜但不精确（粒度80-100μs，且有抖动）

这里的精确指的是这个操作返回的时候同步操作一定已经完成。

基于另一个observation：layer之间参与各个kernel计算的data shape是固定的，执行时间也比较固定。因此可以先离线profile kernel执行时长，然后用usleep非阻塞地等一个 预计时长-余量 的时长（usleep返回的时刻就应该是kernel结束之前一小段时间），然后忙等直至精确的同步时刻。

这个过程其实就是：

```
第一阶段：usleep(预测时间 - 一点余量)
    └── 便宜，让出CPU，消耗掉大部分等待时间

第二阶段：小核polling flag bit
    └── 精确，只持续几微秒
```

### 2.3 Zero-copy的内存池管理

利用UMA的特性，预分配一批buffer slot同时映射到CPU/GPU/NPU地址空间，这样既消除了数据拷贝，也避免了每次推理重新建立映射的overhead。

design部分总结：

> 图片待补充：image-20260410001546466.png

这里的kernel scheduler的逻辑是：

```
来了一个kernel请求
        │
        ├─ 当前是prefill还是decode？  ← runtime状态
        │
        ├─ 当前seq_len是多少？        ← runtime输入
        │
        └─ 查离线Solver的策略表
                │
                ├─ GPU-only
                ├─ NPU-only
                └─ GPU+NPU并行（选哪种partition，比例是多少）
```

---

## Evaluation

**端到端**：相比SOTA有1.34×到6.02×的加速，其中prefill提升更大（最高24.9×），decode相对保守（1.50×到2.53×）。

**Prefill**：

- Hetero-layer已经比NPU-only的PowerInfer2快3.29×，主要靠operator-level的GPU/NPU affinity分配和tensor order变换
- Hetero-tensor在此基础上再提升30.2%，靠weight-centric partition
- 动态shape场景下比padding方案快2.21×，比NPU-pipe快1.35×

**Decode**：

- GPU+NPU并发时内存带宽从43.3 GB/s提升到59.5 GB/s，达到最大可用带宽的96%
- 这个数字是decode加速的核心来源，逻辑上非常干净：decode瓶颈是带宽，带宽打满了，性能自然接近上界

**Fast Sync的单独贡献**：

- Prefill阶段提升15-24%
- Decode阶段提升2-4×（decode kernel太短，同步overhead占比更大，所以效果更显著）

没有和serving场景下的concurrent request做测试（作者声称移动端batch=1是现实）
