---
layout: "post"
title: "GPU 统一内存与资源共享：TGS、MPS 和 AntMan"
date: "2026-09-30"
description: "梳理统一内存支持的显存共享、机会型任务及 GPU 计算资源调控。"
categories: ["Papers"]
tags: ["gpu", "uvm", "mps", "scheduling"]
permalink: "/blog/paper-gpu-unified-memory-sharing/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/GPU Unified memory.md"
published: false
---

GPU Unified memory

这是pascal及以后的架构才有的机制。

CPU和GPU都可以处理缺页，因此不用在跑一个kernel的时候把所有的memory全都迁移到当前gpu上来，而是用到的时候缺页再迁移。

TGS做memory sharing的时候，主要有三个方面的工作：

1. 对production job和opportunistic job做性能隔离，即优先把production job安排在gpu memory中。
2. 透明地允许超额订阅gpu memory。gpu memory没用完的时候，所有的内存分配请求都安排在gpu memory中；当gpu memory满了之后，就考虑把opportunistic job驱逐到主存中去，为production job腾出gpu memory。
3. 解决了DL框架过分请求gpu memory的问题。当DL框架申请所有的gpu可用memory时，TGS不是分配这么多的gpu memory，而是分配这么多的cuda unified memory，然后kernel跑的时候有gpu缺页再把页调进gpu memory。这样就能在不修改DL框架的前提下，做到把kernel使用的memory放在gpu中，而那些没用到的就留在主存中了。

而作者提出的Sharing GPU Memory Resources的几个问题分别是：

1. The major limitation of this solution is that it has large overhead for production jobs.当production job没有使用所有的gpu memory时，opportunistic job就可以用剩下的gpu memory，但是这样的话如果后续production job需要新的gpu memory，就没有充足的gpu memory了。
2. Another limitation of this solution is that it does not consider the characteristics of DL frameworks.一些DL框架如tensorflow在开始一个job的时候会申请所有的可用gpu memory，本意是模仿CRT的机制，先批发再零售减少gpu memory allocate的开销。但是一个DL框架的内存池不会归还使用完的内存，而是继续保留在内存池中。这样就阻止了别的DL框架share同一个gpu。

TGS用驱逐策略解决第一个问题，保证优先让production job留在gpu memory中。

TGS用伪gpu地址的方法解决第二个问题，tensorflow申请所有的可用gpu memory时，不给这么多gpu memory，而是给等量的cuda unified memory，此时实质上空出了很多gpu memory，然后在kernel运行的时候，按需（gpu 缺页）调入gpu memory，就完成了按需分配gpu memory的任务。

TGS做计算资源的share的时候

一个事实：一个小的DNN training job用不完一个gpu的资源，如果每个container以exclusive的方式使用gpu会造成资源浪费。TGS可以让多个containers share一个gpu。

strawman solution：production job高优先级，opportunistic job低优先级，截获container发出的kernel，入队。只有当production job queue为空（表明现在还有剩余资源）时才可能调度opportunistic job。

这种solution的缺点：

1. production job queue为空不代表没有production job在使用gpu，此时加入opportunistic job就会和production job产生竞争。
2. 跟踪kernel的运行情况不现实，因为GPU state并不完全可见。

TGS选kernel的到达频率作为feedback。不选GPU使用率的理由：

1. GPU使用率是一个模糊的概念，并且跟硬件有关。Today’s GPUs contain different types of compute units on a single chip, e.g., Tensor cores and CUDA cores for different data types on NVIDIA GPUs.GPU上有不同种类的单元，一个使用率无法确切反映GPU的使用情况。
2. GPU使用率和性能并不挂钩。比如，production job和opportunistic job可能在竞争某一种类型的计算单元，但是另一种计算单元在空置，导致虽然使用率不足100%，但是其实gpu已经满负载了。

选kernel arrival rate的理由：

计算图是有依赖的。前面的kernel的吞吐量下降，就会导致后面的kernel 发送速率下降。这种方式忽略了GPU内部的制约关系，相当于处理一个黑盒来控制计算资源的使用。

然后用tcpip的拥塞控制算法来处理。

---

nVidia MPS（multi-process service）

The Multi-Process Service (MPS) is an alternative, binary-compatible implementation of the CUDA Application Programming Interface (API). The MPS runtime architecture is designed to transparently enable co-operative multi-process CUDA applications, typically MPI jobs, to utilize Hyper-Q capabilities on the latest NVIDIA (Kepler-based) Tesla and Quadro GPUs .

MPI jobs：Message Passing Interface jobs

Hyper-Q：Hyper-Q 是 NVIDIA 在 **Kepler 架构**中引入的一项关键技术，用于优化 GPU 的多任务并行性能。全称为 **Hyper-Queue**，它的主要目的是**增加 GPU 的硬件任务队列数量，从而更高效地利用 GPU 的计算资源**。应用程序开发者无需特别优化任务分配逻辑，Hyper-Q 自动处理底层的任务调度和资源分配。

MPS的架构图：

> 图片待补充：image-20250109161420212.png

AntMan

提出的DL cluster中的三大问题

1. Low utilization of in-use GPUs. only 20% of the GPUs are running applications that consume more than half of the GPU memory.
2. Idle waiting for gang-schedule. Multi-GPU training jobs require gang-scheduling, which means a job will not start training unless all required GPUs are simultaneously available
3. Dynamic resource demand. + the memory caching design in existing DL frameworks

AntMan用到的几大技术

**Dynamic Scaling**

1. memory management

![image-20250114134358911]({{ '/assets/blog/paper-gpu-unified-memory-sharing/image-20250114134358911.png' | relative_url }}){: .img-fluid loading="lazy" }

growth是由AntMan检测到的，并且只有这个检测的batch有因为部分tensor放在主存中有额外的开销，后续的batch所有的tensor都会被优先放在gpu memory中了。

2. Computation Management

加了一个GpuOpManager来管理不同job的kernel的执行频率，就是一个中间层缓冲待执行的kernels。然后有空闲的时间段时就launch GpuOpManager中缓冲的kernels。

**Scheduling Policy**

1. global scheduler：为了利用gang-schedule策略下idle的gpu资源，让opportunistic job可以直接使用idle的gpu资源（Only GPUs with a utilization of less than M (set as 80% for now) in the past 10 seconds can be selected as candidates.）
2. local coordinator，本来的DL 框架为了保证resource-guarantee job的运行，一个gpu只分配给一个这样job，但AntMan可以让这个gpu上并行跑别的opportunistic job，只是需要限制opportunistic job的memory和 SM的使用率。**这样就和前面的gang-schedule策略下idle的gpu资源可以先让opportunistic job利用起来闭环了**。
