---
layout: "post"
title: "CSE234 课程笔记：并行计算与分布式训练"
date: "2026-09-30"
description: "整理并行类型、通信模型、数据与模型并行、流水线、ZeRO 以及自动并行策略。"
categories: ["Courses"]
tags: ["cse234", "parallelism", "distributed-training", "communication"]
permalink: "/blog/cse234-parallelism-distributed-training/"
lang: "zh-CN"
notes_import: true
source_path: "lecture-learning/cse234/Parallelism.md"
series: "cse234"
series_order: 2
toc:
  beginning: true
---

## 1. 并行的Motivation

简而言之，“涌现能力”让我们需要更大的模型，而硬件计算能力和存储容量的提升速度赶不上模型参数量增加的速度。

## 2. 并行类型概览

1. 类型：data parallelism和model parallelism

data parallelism就是把每个batch的data分成多份，分给多个device计算。data parallelism是数据并行，每个device上都有完整的模型。pytorch的DDP就属于data parallelism。

model parallelism就是把模型切分成子模块，分布到多个device上，因此每个deivce上只有模型的一部分。只用model parallelism的情况下，所有数据都要流经每个device。

2. 并行带来的通信

NVLink，networks，etc

对于data和model的切换是可以影响通信开销的。

3. Parallelism workflow

对于data parallelism来说，每个device负责一部分数据的计算，最后反向传播的时候，每部分数据对应的梯度需要同步然后再更新参数。

对于model parallelism来说，切分的是参数和梯度的计算。

4. 从op的角度分：intra和inter

inter-op就是把op分布到不同device上，而非把同一个op计算的数据分布到不同device上。inter的通信是p2p（point to point）的，这种通信开销较小，但是会产生较多的device idle，因为同一时刻只有一个device在计算（因此一般要用流水线来优化）。inter是解决一个device的存储放不下整个model的。

intra-op就是把op拷贝到不同device上，每个device上的相同op计算一个batch数据的一部分。intra需要较多、较复杂的collective通信（all-reduce, all-gather, broadcast等）。

一个直观的对比图：

![image-20250808215648985]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808215648985.png' | relative_url }}){: .img-fluid loading="lazy" }

inter和intra tradeoff 的点在于：

inter较容易（如果model的op本来就较为独立），通信开销小，但是容易产生大量idle，单纯的inter也并不能提升速度；

而intra是能提升速度的并行，代价是较复杂而且通信开销大。

## 3. 并行的衡量指标 MFU

并行效率用Model Flops Utilization (MFU)来衡量：

![image-20250808220747111]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808220747111.png' | relative_url }}){: .img-fluid loading="lazy" }

其中

#FLOPs : Total floating − point operations per formed by the ML program

t : Time taken to complete the program

peak FLOPS : The maximum theoretical FLOP s the hardware can perform

与 MFU 相对的还有一个 HFU（Hardware Flops Utilization）指标，不过MFU里面的Flops是模型理论上应该进行的计算次数，而HFU里的Flops是模型实际的计算次数。当使用了checkpointing这些内存优化技巧时，会产生额外的计算次数，这些是MFU不考虑但是HFU会考虑的，因此只用MFU衡量并行效率。

> **Important**
>
> 当不使用checkpointing的时候，只需要1 forward & 1 backward。
>
> 当使用checkpointing的时候，需要2 forwards & 1 backward。
>
> 一般来说，1 backward ≈ 2 forwards，理由是forward时需要只计算activation，但是backward时既需要计算activation的梯度也需要计算parameters的梯度。
>
> 因此，MFU = 3 /4 HFU

MFU高的计算：Matmul这类AI高的，一次数据移动可以做多次运算。

MFU低的计算：ReLU这类逐元素的计算，一次数据移动只做一次运算。

## 4. 并行通信方式

### 4.1 collective communications

1. broadcast

![image-20250808222215142]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222215142.png' | relative_url }}){: .img-fluid loading="lazy" }

2. Reduce

![image-20250808222232789]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222232789.png' | relative_url }}){: .img-fluid loading="lazy" }

3. scatter

![image-20250808222254605]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222254605.png' | relative_url }}){: .img-fluid loading="lazy" }

4. gather

![image-20250808222313965]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222313965.png' | relative_url }}){: .img-fluid loading="lazy" }

5. All gather

![image-20250808222330871]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222330871.png' | relative_url }}){: .img-fluid loading="lazy" }

6. Reduce-Scatter

![image-20250808222407680]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222407680.png' | relative_url }}){: .img-fluid loading="lazy" }

7. All Reduce

![image-20250808222424368]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222424368.png' | relative_url }}){: .img-fluid loading="lazy" }

8. all to all

![img]({{ '/assets/blog/cse234-parallelism-distributed-training/v2-3ae0d7692c7ca53cf2a4f57bbb7a651d_1440w.jpg' | relative_url }}){: .img-fluid loading="lazy" }

### 4.2 通信模型

通信延迟的计算公式：

![image-20250808222548691]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250808222548691.png' | relative_url }}){: .img-fluid loading="lazy" }

α是建立通信的开销（如确认连接）, β 是带宽的倒数, n 是消息大小.

• Small Message Size (α >> nβ): Latency (α) dominates; the focus is on reducing communication latency.

• Large Message Size (α << nβ): Bandwidth utilization (nβ) dominates; the focus is on maximizing bandwidth.

在ML sys中，消息一般都是大数据（data，权重，优化器状态等），因此带宽是主要因素。

### 4.3 通信优化方式

#### 4.3.1 最小生成树（MST）优化

目的是在一个带权图（带宽倒数为权）找到最小传输开销的分封传输方案。但是这种方法并没有完全利用带宽，因为是分层的，必须等上一层传输完，下一层才开始传输。

#### 4.3.2 环优化

这是高效利用带宽的优化方式。

1. All gather

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809153326434.png' | relative_url }}" alt="image-20250809153326434" style="max-width: 100%; height: auto;" loading="lazy" />

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809153340509.png' | relative_url }}" alt="image-20250809153340509" style="max-width: 100%; height: auto;" loading="lazy" />

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809153357909.png' | relative_url }}" alt="image-20250809153357909" style="max-width: 100%; height: auto;" loading="lazy" />

这样就利用了每个device的带宽，每个周期既发送一份数据也接受一份数据。

2. Reduce Scatter

![image-20250809153532387]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809153532387.png' | relative_url }}){: .img-fluid loading="lazy" }

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809153539265.png' | relative_url }}" alt="image-20250809153539265" style="max-width: 100%; height: auto;" loading="lazy" />

每个device按序将一部分数据发送给下一个device，并接受上一个device发送来的数据，将这部分数据与本device对应的数据做Reduce，这个Reduce结果就是本device下个周期发送的数据，经历device数个周期后，每个device最后收到并Reduce的那份数据就是Scatter到这个device的最终数据。

3. Broadcast

Broadcast = Scatter + All gather

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809154044409.png' | relative_url }}" alt="image-20250809154044409" style="max-width: 100%; height: auto;" loading="lazy" />

4. Reduce(To One)

Reduce = Reduce Scatter + gather

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809154339845.png' | relative_url }}" alt="image-20250809154339845" style="max-width: 100%; height: auto;" loading="lazy" />

5. All Reduce

All Reduce = Reduce Scatter + All gather

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809154431839.png' | relative_url }}" alt="image-20250809154431839" style="max-width: 100%; height: auto;" loading="lazy" />

## 5. Data Parallelism

![image-20250809155049341]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809155049341.png' | relative_url }}){: .img-fluid loading="lazy" }

data parallelism的关键前提是每个device的存储能放下整个model。

实现data parallelism的两种方式：parameter server，all reduce。

### 5.1 Parameter server

使用parameter server的假设是每次迭代参数的时候通信开销很大，比如Compute : communication = 1:10 in the era of 2012。

方式如下图：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809155502363.png' | relative_url }}" alt="image-20250809155502363" style="max-width: 100%; height: auto;" loading="lazy" />

这样，每个device只需要对server做P2P通信即可。

PS模式的两个关键考虑：

1. server的通信瓶颈：所有的worker都从一个PS拉取、发送参数；
2. server容错困难：整个model的参数都在一个PS上，PS出现故障时需要恢复整个模型的参数。

因此有了Sharded parameter server: sharded KV stores。

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809155854222.png' | relative_url }}" alt="image-20250809155854222" style="max-width: 100%; height: auto;" loading="lazy" />

sharded PS的优势有：

1. 避免通信瓶颈：将整个模型参数分布到多个PS上，每个PS的通信量大幅减少。
2. server容错更便捷：整个model的参数分布在多个PS上，因此某个PS出现故障时，只需要根据备份恢复这个PS上的参数即可。

#### 5.1.1 Straggler

多台device做data parallelism，很大概率会有straggler存在。如果追求强一致性，那么其他先完成这轮计算的device就会idle等待straggler从而产生bubbles，同步点叫做Global Synchronization Barrier。

#### 5.1.2 宽松一致性

ml算法能高度容忍错误，因此可以在不做强制同步的情况下仍允许一些偏差。

但是完全异步通常是不对的，这违背了并行的原则，完全没有做聚合。

因此，Bounded Consistency (SSP)被提出，这是一个处于强一致性和无一致性中间的方法，叫stale consistency：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809162349454.png' | relative_url }}" alt="image-20250809162349454" style="max-width: 100%; height: auto;" loading="lazy" />

引入了一个staleness超参数，体现允许的最大偏差范围，如果在这个偏差范围内就允许不做同步；超出这个范围就必须做强制同步了。

对比不同staleness对loss收敛的影响的实验结果：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809163145763.png' | relative_url }}" alt="image-20250809163145763" style="max-width: 100%; height: auto;" loading="lazy" />

可以看到，SSP的staleness超过3收敛速度就明显减缓甚至难以收敛了。

有一个计算SSP模型参数距离最优解的**期望距离界**：

![image-20250809163643840]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809163643840.png' | relative_url }}){: .img-fluid loading="lazy" }

其中，L是常数，s是staleness，P是worker数量，T是迭代次数，可以直观的感受一下这些因素对最终模型表现的影响：s越宽松，worker数量越多，不一致性带来的效果下降越严重；但是迭代次数的增多能减小与最优解之间的距离。

### 5.2 All Reduce

All Reduce就是分布式地做同步，分成两个阶段：Reduce， Broadcast。

最出名的All Reduce接口就是Pytorch的DDP。

第一个All Reduce是uber的Horovod（**Horovod** 是一个 **分布式深度学习训练框架**，最早由 Uber 在 2017 年开源，主要目标是**简化和加速多 GPU / 多节点训练**，尤其是使用 TensorFlow、PyTorch、MXNet 等主流框架时。使用Horovod时仍用Tensorflow这类框架来定义计算图，完成前向、后向计算，但是当Tensorflow完成梯度计算之后，Horovod就开始做All Reduce。相当于，Horovod就是在Tensorflow这类框架上封了一层）。

后面nVidia将优化版的All Reduce加进了NCCL（NVIDIA Collective Communications Library）。

再后来，Pytorch用NCCL完成了DDP。

All reduce没有解决计算Straggler的问题，但是现在高带宽的通信能缓解通信Straggler的影响。并且，All Reduce很容易实现，虽然没有容错，但是这是ml算法可以接受的。

## 6. Model Parallelism

### 6.1 Inter op

直观的示意图：

![image-20250809165437117]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809165437117.png' | relative_url }}){: .img-fluid loading="lazy" }

计算资源的使用情况：

![image-20250809165500135]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809165500135.png' | relative_url }}){: .img-fluid loading="lazy" }

存在大量的bubbles。

#### 6.1.1 Device Placement

这是Inter op中很基本的前提，即需要将model划分成计算量相近的阶段，也就是流水线的每一阶段的时间需要几乎相等，这样能减少不同阶段等待引起的bubbles。

训练阶段，inter op需要考虑不同输入计算出的梯度怎么更新，分为同步和异步两种。

#### 6.1.2 同步的Bubbles Reduce

##### 6.1.2.1 GPipe

![image-20250809170152517]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809170152517.png' | relative_url }}){: .img-fluid loading="lazy" }

将一个batch划分为多个micro batch，这样就能把流水线并行起来了。下图可以发现，micro batch数越大，device间的idle越少，吞吐量越高：

![image-20250809170945041]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809170945041.png' | relative_url }}){: .img-fluid loading="lazy" }

有一个缺点是，由于梯度计算是在整个batch前向结束之后做的，所以需要保存所有前向的activations，这就导致内存使用有一个较大的峰值，且micro batch越多，这个峰值越大。

##### 6.1.2.2 1F1B

1F1B是GPipe的内存优化版本：

![image-20250809171420398]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809171420398.png' | relative_url }}){: .img-fluid loading="lazy" }

意思就是做完一次Forward，马上做这次Forward对应的Backward，计算完梯度就可以把这次Forward的activations丢弃了，这样内存峰值就跟micro batch数量无关了，而跟devices数量有关，因为需要经过devices数量个周期第一次forward才能结束，才能计算完梯度丢弃相应的activations。

1F1B也是GPT使用的方法。

##### 6.1.2.3 Interleaved 1F1B

本质上就是把每个device上的阶段继续划分为子阶段。

![image-20250809195521833]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809195521833.png' | relative_url }}){: .img-fluid loading="lazy" }

##### 6.1.2.4 Chimera

如图：

![image-20250809200309525]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250809200309525.png' | relative_url }}){: .img-fluid loading="lazy" }

#### 6.1.3 异步的Bubbles Reduce

### 6.2 Intra op

#### 6.2.1 Megatron-LM

Megatron对MLP的intra op并行策略，不对x做shard，而是对weight做分块：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250811175036931.png' | relative_url }}" alt="image-20250811175036931" style="max-width: 100%; height: auto;" loading="lazy" />

column-partitioned的matmul计算结果就是分块的，因此继续按块做intra op的并行逐元素计算（gelu或其他），只需要在backward的时候做all reduce合并梯度即可。

但是column-partitioned和row-partitioned做matmul时，每个device上都只有对应位置的部分和，因此在forward的时候就需要all reduce得到正确的结果。

#### 6.2.2 GShard MoE

#### 6.2.3 ZeRO Optimizer

ZeRO是为了内存优化：

![image-20250812095812960]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812095812960.png' | relative_url }}){: .img-fluid loading="lazy" }

stage2和stage1总体都只做一次all reduce，但是stage2的内存占用比stage1小，因此一般都是从stage2开始考虑使用。

stage2：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812100631382.png' | relative_url }}" alt="image-20250812100631382" style="max-width: 100%; height: auto;" loading="lazy" />

每个device上都有一套梯度，但对于整个batch来说都是部分梯度。因此先reduce scatter让每个device上都有一部分完整的梯度，然后用这个划分做优化器的并行，计算完再all gather将完整梯度还原到每个device上。

stage3：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812104156069.png' | relative_url }}" alt="image-20250812104156069" style="max-width: 100%; height: auto;" loading="lazy" />

就是因为对weights也做了划分，所以前向和后向的时候都要all gather，确保每个设备上有需要的完整参数。

这里的weights划分是对每一层的参数进行划分，也就是说，forward到某一层时，这一层的参数是分布到各个设备上的，因此要完成这一层的forward，就需要一次针对这层参数的all gather而非broadcast，backward同理。因此每个设备上参数的内存使用峰值就是 总参数量/设备数 + 某一层参数量。

#### 6.2.4 Mesh Tensorflow

主要思想就是不再仅仅把设备群看作一维的，而是多维的，因此可以按多个维度进行划分并行。

## 7. Auto Parallelism

要解决的问题就是如此多的inter/intra op策略、model和cluster的组合，怎么找到最优的并行策略？

![image-20250812112023622]({{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812112023622.png' | relative_url }}){: .img-fluid loading="lazy" }

方法和优化计算图的方法类似，都是在一个剪枝后的搜索空间里面搜，然后用一个evaluator来评估每种可能的性能：

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812113129991.png' | relative_url }}" alt="image-20250812113129991" style="max-width: 100%; height: auto;" loading="lazy" />

### 7.1 基于学习的方法：ColocRL

把计算图表示为DAG，节点就是op(matmul, activation,etc)，边表示数据依赖关系。

主体是一个LSTM采用强化学习，实时的评估作为reward，评估是将当前的放置策略在真实硬件上运行得到的。

搜索的时候先用一个encoder处理原计算图，再用decoder生成每个op的放置策略，生成过程是逐op的。

效果是很好的，能发现一些人类专家无法构造的更优策略，甚至是难以解释的更优策略。

### 7.2 基于优化的方法：Alpa

Alpa是用类似DP的方法搜索。首先把整个model切分成流水线，然后考虑将每一个stage放置到Submesh的组合策略，这个过程就是用类似DP的搜随策略实现的。

<img src="{{ '/assets/blog/cse234-parallelism-distributed-training/image-20250812140557772.png' | relative_url }}" alt="image-20250812140557772" style="max-width: 100%; height: auto;" loading="lazy" />
