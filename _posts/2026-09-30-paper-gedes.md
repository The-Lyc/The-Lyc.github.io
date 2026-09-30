---
layout: "post"
title: "GeDES：GPU 离散事件仿真与 NSX 对比"
date: "2026-09-30"
description: "分析 GeDES 的事件重排、负载编排、内存优化与 NSX 的差异。"
categories: ["Papers"]
tags: ["gpu", "des", "cuda-graph", "simulation"]
permalink: "/blog/paper-gedes/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/DES/GeDES.md"
published: false
toc:
  beginning: true
---

## 1.Backgroud

GeDES是针对network DES的GPU优化。network DES中常用的FatTree中，一个FatTree64就有65536的server，最高就有65536的并行度。一块顶级的CPU如AMD EPYC 9005 CPU也只有192 cores，而一块普通的消费级显卡如RTX 3080就有8960 cuda cores，显然理论上的并行度大大增加，适合把DES迁移到GPU上。

## 2.Challenges

### 2.1 DES sequential model 与 GPU SIMT 不兼容

CPU 是 MIMD 架构，不同 core 可以同时执行不同类型的 event handler。
但 GPU 是 SIMT 架构，一个 warp 中的 threads 更适合执行相同类型的指令。

传统 DES 中，一个 node/LP 内部的 event sequence 往往混合多种 event type：

```
receive P1 → forward P2 → send P3 → receive P4 → forward P5
```

如果直接映射到 GPU，同一个 warp 内可能有的 thread 执行 receive，有的执行 forward，有的执行 send，导致 branch divergence 和 GPU idle。

### 2.2 workload dynamics 导致 warp 内负载不均衡

即使 event type 已经对齐，同一个 system 里的不同 node workload 也可能差异很大。

例如在 Forward Sys 中：

```
lane 0: 处理 1 个 packet
lane 1: 处理 3 个 packets
lane 2: 处理 80 个 packets
lane 3: 处理 100 个 packets
```

warp 内 threads 必须一起执行、一起结束，因此轻负载 lane 会等待重负载 lane，造成 tail effect 和 GPU wastage。

### 2.3 GPU memory 限制 simulation scale

DES 需要维护大量 packet、routing table、queue state、protocol state。
大规模 FatTree 网络的 memory demand 可能远超 GPU memory capacity。

因此，GeDES 不能简单地把 CPU DES 迁移到 GPU，而是需要重新设计 execution model、workload mapping 和 memory management。

## 3.Design

### 3.1 放松严格的时序关系，捕捉内在的依赖关系

Event orchestration这一点算是论文最核心的创新点了。传统DES是严格按照时许来执行event，但是DES本质上遵守的并不是时序关系，而且逻辑依赖关系。

传统 DES 强调按 timestamp 严格执行 events。
但是 GPU 更适合同类任务批处理，因此 GeDES 希望把 mixed event sequence 改造成 homogeneous event batches。

传统执行方式：

```
send → receive → forward → send → receive
```

GPU-friendly 执行方式：

```
Batch 1: all send events
Batch 2: all forward events
Batch 3: all receive events
```

GeDES 将 network DES 中的 dependency 抽象成两类：

#### 3.1.1 FCD: Functional Causal Dependency

FCD 表示同一个 packet 的生命周期顺序不能被打乱。

例如：

```
send → forward → receive
```

同一个 packet 必须按照网络协议栈和物理传输逻辑依次处理。
如果 receive 没发生，forward event 就没有语义。

因此，FCD 保护的是：

```
同一个 packet 自身的功能处理顺序
```

#### 3.1.2 TSD: Temporal Sequential Dependency

TSD 表示同一个 node 内，同一类 event 必须遵守 queue order。

例如某个 output queue 中有：

```
P1, P2, P3
```

那么 send / forward 顺序必须保持：

```
send P1 → send P2 → send P3
```

不能变成：

```
send P3 → send P1 → send P2
```

否则会破坏 FIFO queue semantics，进而影响 queueing delay、RTT、FCT 等仿真结果。

因此，TSD 保护的是：

```
同一个 node 内，同类 event 的 queueing order
```

#### 3.1.3 FCD + TSD 形成 DAG

在 GeDES 中，一个 slot 内的 events 可以看成一个 DAG。

DAG 中的边包括：

```
FCD edge: 同一个 packet 的 functional causality
TSD edge: 同一个 node 内同类 event 的 queue order
```

只要重排后的执行顺序仍然是这个 DAG 的合法 topological order，就不会破坏 simulation correctness。

因此，GeDES 的逻辑是：

```
不保留所有 chronological order；
只保留必要 causal dependency；
在 dependency 允许的范围内聚合同类 events。
```

### 3.2 分时隙重排events

Event orchestration 还需要解决一个问题：有些 events 是执行前驱事件后才生成的。

如果提前重排 events，可能会漏掉尚未生成的 event。

GeDES 的解决方式是将 simulation time 切成 slots，并将 slot length 设为：

```
slot length = minimum packet propagation time $T_{min}$
```

也就是一个单跳的时间，一个node的新动作最短也要一个单跳才能影响到别的节点。

这样可以保证：

```
当前 slot 内，一个 node 的处理结果不会立刻影响另一个 node 的当前 slot 处理；
当前 slot 内的 events 是确定的；
所有 nodes 可以在 slot 内独立处理，slot 结束后再进入下一轮。
```

这与 PDES 中的 lookahead 思想类似。
$$T_{min}$$ 可以理解为一个全局 conservative lookahead。

因此，GeDES 更接近 conservative synchronization，而不是 optimistic rollback。
它不会先执行错误事件再 rollback，而是只在安全时间窗口内做 event orchestration。

> **Note**
>
> slot-based orchestration本质上是把传统的FEL切成了按slot分的桶。传统的FEL之所以要基于堆，就是因为一般是按严格时序来的，必须用堆来保证取到时间戳最小的event。但是GeDES本身就不再基于严格时序，而是在一个slot内按FCD和TSD组DAG，因为新生成的任务不再需要像堆插入那样做复杂的操作，而是可以直接丢到slot桶里面，等后边的event orchestration组DAG就行了。而且network DES中本来TSD就是FIFO，天然符合这个优化。

### 3.3 GPU友好的event分发

GeDES 并不是直接维护一个复杂 event list 后手动重排，而是把网络处理流程拆成多个 simulation systems，并在warp层面也做了Workload Orchestration

#### 3.3.1 Simulation systems

典型 systems 包括：

```
Send Sys
Recv Sys
L3 Routing Sys
L2 Routing Sys
Forward Sys
L2 Decap / L2 Encap
L3 Decap / L3 Encap
```

每个 simulation system 处理一类 homogeneous operation。
一个 slot 内，GeDES 按照协议栈处理顺序依次运行这些 systems。

例如：

```
for each slot:
    run Send Sys
    run L3 Routing Sys
    run L2 Routing Sys
    run Forward Sys
    run Recv Sys
```

这样就间接实现了 homogeneous event batching。

#### 3.3.2 In-warp Workload Orchestration

Event orchestration 已经保证一个 simulation system 内主要处理同一种任务。
但是同类型任务不等于同工作量。因此，该设计解决的是：同类型任务内部的 workload imbalance。

GeDES 不直接将 node workload 固定绑定到某个 LP 或 warp lane。
它引入一个共享 workload pool。

传统映射：

```
node workload → LP → GPU thread / warp
```

GeDES 映射：

```
node workload → workload pool
LP / warp lane → index → fetch workload from pool
```

也就是说，LP 不固定拥有某个 node，而是根据 index 从 workload pool 中取任务。

当需要 workload realignment 时，GeDES 不直接重组 GPU warp，而是更新 indices：

```
before:
warp 1 = {node 1, node 7, node 20, node 31}

after index update:
warp 1 = {node 2, node 3, node 4, node 5}
```

本质是：

```
不移动 GPU threads；
只改变 GPU threads 看到的 workload。
```

### 3.4 Cache Mechanism / Memory Optimization

GPU memory 容量有限，而 DES memory demand 很高。
大规模 network simulation 需要存储大量：

```
packet objects
routing tables
routing paths
queue states
protocol states
device states
```

因此，GeDES 需要降低 GPU memory pressure。

本质上就是根据数据真正需要的时间gap来决定是否offload到CPU侧，以及利用network的特性用少量在线计算换取大量 routing memory reduction。

## 4.Inspiration

结合GeDES的源码，其中体现event orchestration的DAG构建，其实是将每个slot内的所有操作捕捉成了一张大的cuda graph，每个slot就重放一次这个graph，graph的内部结构就能保证FCD，TSD，以及同类event的聚合：

> 图片待补充：gedes_dag_full_diagram.svg

第二个字图里的T0,Tn都是代表warp里的线程，每个线程负责处理一个node（比如一个server）的事件队列。这个事件队列就是已经分好的了，一定是同类型的事件（大的graph就按照FCD拆分好了），只存在负载可能不均衡的问题，这个问题就是用In-warp的调整来解决的了。

总的看下来，这篇paper的主要优点还是把network DES的问题模型简化得很清晰。

类比到军事DES里，graph的每条链就可以对应每个种类的实体，本质上GeDES也是把不同种类的实体（本质上就是操作逻辑不同质）分开。

GeDES里可以用单跳事件作为$$T_{min}$$来确定slot长度，类比到军事DES中，lookahead是比较短暂的，且目前没看到一个原子的时间单位，如果能发掘这一点的话，确实能解决FEL迁移到GPU上的问题，完全不需要一个大的FEL了。之前军事DES里的本地更新，其实也对应了FIFO的思想。

本质上，GeDES仍没有解决FEL的排序问题。因为只是把一个全局的FEL变成了每个实体每种操作的小FEL了，要实现TSD的约束，仍需要先按时间戳排序。实际上GeDES的源码里就是对每个小FEL做单线程n2的排序。

## 5. 与NSX的对比

NSX（NSX: Large-Scale Network Simulation on an AI Server）是Nvidia自己的network DES引擎，并且宣称是Nvidia内部大量使用的仿真工具，比GeDES早一年发表。

事实上，NSX已经做了GeDES最重要的contribution了：把DES sequential model迁移到GPU的SIMT上：

1. GeDES声称捕捉内在的依赖关系，NSX的section 3.1已经做到了这件事，也是把一个iteration的操作捕捉到一张cuda graph中，这就满足了FCD；然后NSX的event queue是基于FIFO的，这就利用了TSD。
2. GeDES中使用了slot based events orchestration，NSX中同样考虑了这一点，只不过NSX考虑了node内lookahead通常和node间的lookahead差了一个数量级，因此提出decentralized synchronization algorithm，每个simulator有自己的lookahead。

GeDES多的部分就是考虑了能减少发散的events分发方式，以及数据offload以解决显存不足的问题。

另一个需要重点对比的point在于GeDES与NSX在GPU上处理FEL的方式。

上文已经说过，GeDES并未完全解决FEL带来的排序问题，虽然把FEL细分了，每个sub FEL的排序没有那么耗时，但是底层仍然是朴素的排序算法。但是NSX设计的Flex queues机制就很好地解决了这个问题。NSX的design里，每个module（类似GeDES里的一个system的实体）为每个会发送packet到自己的module维护一个单独的FIFO缓冲，这样在一轮迭代中，每个ingress module发送到当前module的packet在per-module的FIFO queue里面天然满足了局部的TSD。下一轮这个module要按时间戳处理所有ingress的packets时，本质上是一个基于多指针的多路归并问题了，这个就是一个GPU亲和的问题了。
