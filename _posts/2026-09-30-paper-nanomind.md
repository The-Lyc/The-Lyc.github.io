---
layout: "post"
title: "NanoMind：小型电池设备上的多模态推理"
date: "2026-09-30"
description: "介绍跨加速器调度、零拷贝环形缓冲与电量感知执行模式。"
categories: ["Papers"]
tags: ["mlsys", "edge-inference", "multimodal", "energy"]
permalink: "/blog/paper-nanomind/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/EdgeInfer/NanoMind.md"
toc:
  beginning: true
---

TINY BUT MIGHTY: A SOFTWARE-HARDWARE CO-DESIGN APPROACH FOR EFFICIENT MULTI-MODAL INFERENCE ON BATTERY-POWERED SMALL DEVICES

---

## Motivation

- 大型多模态模型很容易解耦成模块被分配到适合的计算单元上
- 边缘设备通常是UMA，需要利用这一点做到比device GPU高效的零拷贝

---

## Contribution

### 1. Cross-accelerator scheduling for modular VLMs.

- NPU适合低比特张量计算（INT4/INT8），且需要固定尺寸的输入 ====> 把量化后精度损失较小的vision encoder量化为INT8部署到NPU上，且输入都padding为固定尺寸
- GPU适合做大规模并行浮点计算 ====> 把LLM量化为W4A16部署到GPU上
- CPU适合做调度和运行数据监测 ====> 监测电量等数据来决定分级的调度策略

### 2. Dynamic workload Offloading.

- 构造生产者-消费者模型：NPU上vision encoder的输出直接放在DRAM上的ring buffer，GPU直接从ring buffer里去，bypass CPU的内存读写，实现高效的zero copy
- 数据流bypass CPU，但是控制流依旧在CPU上：NPU输出的slot位置是CPU启动kernel的时候指定，GPU的kernel启动也是绑定对应的slot位置。NPU与GPU的通信也是通过CPU传递

### 3. Battery-aware execution modes.

三级执行模式：

- 第一级：无限制性能状态 (Unconstrained Performance State)
  - **触发条件：** 电池电量充足 ($$B > T_{high}$$)。

  - **行为：** 系统全速运行，**激进地并行卸载**任务到加速器。

- 第二级：比例节流状态 (Proportional Throttling State)
  - **触发条件：** 电池电量中等 ($$T_{low} < B \le T_{high}$$)。

  - **行为：** 系统进入“降级”状态。

  - **核心机制：** 它不是直接切换到串行模式，而是使用一个线性插值因子 $$\alpha=(B-T_{low})/(T_{high}-T_{low})$$ 来动态调整性能。

  - **具体动作：** 根据 $$\alpha$$ 因子，**线性降低摄像头的帧率（Frame Rate）和内存的读写速率**。这意味着随着电量下降，设备反应会变慢，视频会变卡，但依然保持并行的流水线工作方式，直到电量跌破下限。

- 第三级：紧急节能状态 (Critical Conservation State)
  - **触发条件：** 电池电量告急 ($$B \le T_{low}$$)。

  - **行为：** 激活 **“按需级联推理 (On-Demand Cascade Inference)”** 模式。

  - **机制：** 暂停所有并行执行，切换到**串行工作流**。每个模块遵循 **"load $$\rightarrow$$ execute $$\rightarrow$$ release"** 的生命周期。release是指从内存中释放

<img src="{{ '/assets/blog/paper-nanomind/image-20251204105004404.png' | relative_url }}" alt="image-20251204105004404" style="max-width: 100%; height: auto;" loading="lazy" />
