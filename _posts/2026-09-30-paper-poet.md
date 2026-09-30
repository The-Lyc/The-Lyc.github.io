---
layout: "post"
title: "POET：微型设备训练中的重计算与分页"
date: "2026-09-30"
description: "梳理在内存与期限约束下联合优化重计算、分页和训练能耗的 MILP。"
categories: ["Papers"]
tags: ["mlsys", "edge-training", "rematerialization", "offloading"]
permalink: "/blog/paper-poet/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/EdgeInfer/POET.md"
toc:
  beginning: true
---

POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging

---

## Motivation

1. **重计算 (Rematerialization)** 虽然能节省内存，但在低内存预算下会导致计算量急剧增加（$$O(n^2)$$），从而大幅增加能耗。

2. **分页 (Paging/Offloading)** 将数据暂存到闪存/SD卡，但数据传输非常耗能，甚至可能超过重计算的能耗。

3. **单一维度的不足**：现有的方法通常要么只优化内存，要么只优化速度，未能将“重计算”和“分页”结合起来进行**能耗最优**的全局考量，或者依赖启发式算法导致次优解。

---

## Contribution

将边缘训练问题形式化为一个**混合整数线性规划 (MILP)** 问题。给定内存预算和运行时约束，它能找到能耗最低的最佳执行调度 ：

### 1. 在构建方程之前，POET 首先需要在目标设备上进行 Profiling（分析），获取以下系数作为已知常量：

- **$$\Phi_{compute}$$**：执行某个算子（Operator）消耗的**能量**。
- **$$\Psi_{compute}$$**：执行某个算子所需的**时间**。
- **$$\Phi_{pagein} / \Phi_{pageout}$$**：将张量从 Flash 读入 RAM 或从 RAM 写出到 Flash 的**能量**。
- **内存大小**：每个张量占用的字节数。

### 2. POET 将训练过程离散化为一系列逻辑时间步（Timesteps, $$t$$），针对图中的每个节点（Node, $$i$$），定义了以下二进制变量（0 或 1）：

- **$$R_{t,i}$$ (Rematerialize/Compute)**: 在时间步 $$t$$ 是否（重新）计算节点 $$i$$。
- **$$S_{t,i}^{RAM}$$ (Storage RAM)**: 在时间步 $$t$$，节点 $$i$$ 的数据是否驻留在 **RAM** 中。
- **$$S_{t,i}^{AUX}$$ (Storage Auxiliary)**: 在时间步 $$t$$，节点 $$i$$ 的数据是否驻留在**辅助存储（Flash/SD卡）**中。
- **$$M_{t,i}^{in}$$ (Page In)**: 在时间步 $$t$$，是否将节点 $$i$$ 从 Flash **搬运进** RAM。
- **$$M_{t,i}^{out}$$ (Page Out)**: 在时间步 $$t$$，是否将节点 $$i$$ 从 RAM **搬运出** 到 Flash。

### 3. 为了保证数学正确性和物理可行性，POET 施加了极其严格的线性约束：

#### A. 依赖性约束 (Dependency)

如果要在时间步 $$t$$ 计算节点 $$j$$，那么它的所有输入依赖节点 $$i$$ 必须已经在 RAM 里了（要么刚算出来，要么一直存着）。

- 公式逻辑：$$R_{t,i} + S_{t,i}^{RAM} \ge R_{t,j}$$

#### B. 内存流转逻辑 (Memory Flow)

这一步是 POET 的核心，它定义了 RAM 和 Flash 之间的数据流动规则：

- **RAM 驻留条件**：某数据在 $$t$$ 时刻在 RAM 里，通过以下三种方式之一实现：
  1. 刚被计算出来 ($$R_{t,i}$$)
  2. 上一时刻就在 RAM 里 ($$S_{t-1,i}^{RAM}$$)
  3. 刚从 Flash 读进来 ($$M_{t-1,i}^{in}$$)
- **Flash 驻留条件**：某数据在 $$t$$ 时刻在 Flash 里，必须满足：
  1. 上一时刻就在 Flash 里 ($$S_{t-1,i}^{AUX}$$)
  2. 或者刚被写出去 ($$M_{t-1,i}^{out}$$)
- **分页动作限制**：
  - 要 Page-in，数据必须先在 Flash 里。
  - 要 Page-out，数据必须先在 RAM 里。

#### C. 硬性资源约束 (Hard Resource Constraints)

这是 MILP 能够保证“可行性”的关键：

- **内存墙 (Memory Limit)**：在任何时间步 $$t$$，所有驻留在 RAM 中的张量总大小不得超过设备的物理内存限制 $$\mu_{RAM}$$。
  - 公式逻辑：$$U_{t}^{RAM} \le \mu_{RAM}$$

- **运行时死线 (Deadline Constraint)**：这是 POET 独有的。所有被选择执行的计算操作（包括前向、反向、重计算）的总耗时，不得超过用户设定的时间预算 $$\mu_{deadline}$$。这保证了训练不会无限拖延。
  - 公式：$$\sum_{T} [R \cdot \Psi_{compute}]_T \le \mu_{deadline}$$

### 4. 目标函数

包含了所有操作的能耗代价：

$$\text{Minimize} \sum_{t, i} \left( R_{t,i} \cdot \Phi_{compute} + M_{t,i}^{in} \cdot \Phi_{pagein} + M_{t,i}^{out} \cdot \Phi_{pageout} \right)$$

- **$$R \cdot \Phi_{compute}$$**: 计算（或重计算）产生的能耗。
- **$$M \cdot \Phi_{page}$$**: 读写 SD 卡/Flash 产生的 I/O 能耗（这部分能耗在边缘设备上通常很高，甚至高于计算）。
