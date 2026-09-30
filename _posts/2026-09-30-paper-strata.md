---
layout: "post"
title: "Strata：长上下文的分层缓存与 I/O 调度"
date: "2026-09-30"
description: "分析碎片 KV 加载、delay hit、GPU 辅助 I/O 和缓存感知批量调度。"
categories: ["Papers"]
tags: ["mlsys", "kv-cache", "prefix-cache", "scheduling"]
permalink: "/blog/paper-strata/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/Strata.md"
toc:
  beginning: true
---

**_Strata: Hierarchical Context Caching for Long Context Language Model Serving_**

---

### 1.Motivation

作者指出：

**hierarchical cache 已经解决了容量问题，却没有解决性能问题。**

真正影响系统吞吐的不是KV 是否存在而是**KV 怎样重新加载回来。**

实际 profiling 发现，大部分时间 GPU 并没有在计算，而是在等待 cache load。

于是系统变成Loading-bound而不是Compute-bound。

### 2.Challenges

#### 2.1 Fragmented KV layout 导致 I/O 极差

PagedAttention 会把 KV 划分成大量 page。Host memory 中也是离散布局。于是一次 cache hit 实际需要多次零散的传输，导致带宽利用率低。

#### 2.2 Scheduler 无法感知 cache load latency

现在的推理框架的scheduler考虑一个batch时，只考虑compute和HBM资源，没有考虑多级存储带来的load latency对batch延迟的影响。比如，按照FIFO的调度策略，下一个batch中大部分都是要等cache load的requests，那么这个batch运行时GPU就会因为等待load latency闲置很久。

#### 2.3 Delay Hit

delay hit 会在两种不同的调度场景下导致冗余 prefill：

##### 2.3.1 同一batch内的delay hit

多个共享相同 context 或 prefix 的请求，被同时放进一个 prefill batch：

```
Request A: [shared context] + query A
Request B: [shared context] + query B
```

在 batch 开始时，shared context 的 KV cache 还不存在，因此 A 和 B 都被判定为 cache miss：

```
同一 prefill batch
├── A：重新计算 shared context
└── B：也重新计算 shared context
```

虽然 A 计算完成后会产生可供 B 复用的 KV cache，但由于二者已经被放进同一个并行执行的 batch，B 无法等待并使用 A 的结果，最终造成重复计算。

##### 2.3.2 异步准备下一batch时的delay hit

现代 serving engine 通常会在当前 batch 执行期间，异步准备下一个 batch。

当 scheduler 构造 Batch N+1 时，Request A 的 prefill 尚未完成，因此 shared context 的 KV cache 还没有正式进入 cache index。Request B 此时仍然被识别为 miss，并按照重新计算该 context 的方式完成资源分配和 batch preparation。

等 Batch N 真正完成时，cache 虽然已经可用，但 Batch N+1 的执行计划已经准备好了，于是 Request B 仍可能执行冗余 prefill。

### 3. Design

Strata 的设计基本对应三个 challenge。

#### 3.1 GPU-assisted IO

利用GPU的多线程来拷贝零散数据。strata也观察到了GPU多线程拷贝会占据大量的warp，导致计算kernel无法并行，因此提出copy kernel不能占满GPU，而是通过launch的blocks数来限制copy kernel可以使用的warp数，从而不影响copy-compute的并行。这里的blocks数strata只给出了经验值。

#### 3.2 Cache aware scheduling

strata从两方面来考虑cache aware：

##### 3.2.1 利用推迟来解决delay hit

在本来的radix tree里引入transient nodes跟踪已经在准备中只是还没有准备完成的kv。如果准备batch时，某个request需要的kv在radix tree被标记为了transient nodes，就说明这部分kv正在被准备，那么这个request就被推迟到下一个batch再执行。

##### 3.2.2 平衡batch的组成

strata要做layer-wise的overlap，即计算i层的同时load i+1层的kv。为了让这个overlap更充分，strata在scheduler做batch决策的时候就把batch的组成调整为计算和IO接近的模式。具体做法是按需要计算的tokens和需要IO的tokens的比例来决策。strata这里也只给了一个经验值。

### 4. Evaluation

主要关注做实验的方式以及测试数据。
