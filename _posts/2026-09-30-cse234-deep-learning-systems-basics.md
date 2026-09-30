---
layout: "post"
title: "CSE234 课程笔记：深度学习系统基础"
date: "2026-09-30"
description: "从系统角度梳理 CNN、RNN、Transformer 与 MoE 需要的算子及网络表示。"
categories: ["Courses"]
tags: ["cse234", "mlsys", "models", "operators"]
permalink: "/blog/cse234-deep-learning-systems-basics/"
lang: "zh-CN"
notes_import: true
source_path: "lecture-learning/cse234/2-basics.md"
series: "cse234"
series_order: 1
---

这一节总结起来就是做DL system需要关注的网络以及我们如何去表达这些网络。

## 1. 需要关注的网络

CNN, RNN, Attention&Transformer, MoE.

1. CNN

CNN主要关注的就是卷积操作。网络结构上主要关注AlexNet，ResNet，U-Net。

从系统的角度说，完成CNN，我们需要完成这几个算子：conv，Matmul，softmax，一些逐元素的运算如非线性激活（ReLU）、池化等。

2. RNN

RNN是一种思想，而并不对应某种特定的模块。理论上，任何网络都可以做成RNN，理由如下图：

![image-20250808203619048]({{ '/assets/blog/cse234-deep-learning-systems-basics/image-20250808203619048.png' | relative_url }}){: .img-fluid loading="lazy" }

可以发现，任何网络都可以替换图中的RNN，因为RNN的特点在于隐藏状态和时序。

RNN中需要关注的计算有Matmul和逐元素运算。

RNN的致命劣势在于输入为长序列时会随着自回归地进行忘记信息，并且这种自回归的训练方式难以并行。

3. Transformer

Transformer的出现正是为了解决RNN的缺陷。Attention操作是对每一个位置单独计算，因此不存在依赖问题，就能够并行。

Transformer需要关注三类模型：BERT，GPT/LLMs，DiT(Diffusion,DiT is the reason why U-Net is no longer used in diffusion models.)

4. MoE

MoE是以上网络的集合体，不过通常的MoE指的是同构的专家系统，即所有的Expert是同构网络，只是参数不同，因此能力不同。如果用异构的网络，就成了异构专家系统，这并不常见。

MoE中唯一的新模块是Router，用于决定对于某一次输入，激活哪个Expert参数。激活的数量通常是一个超参数（训练与推理相同），体现的是准确性与稀疏激活优势的tradeoff。

MoE中主要关心的计算：Matmul和Softmax。

总结下来：

![image-20250808205405987]({{ '/assets/blog/cse234-deep-learning-systems-basics/image-20250808205405987.png' | relative_url }}){: .img-fluid loading="lazy" }

因此， MLSys ≈ Matmul Sys。

## 2. 网络的表示形式

从系统的角度，我们用计算图来表示所有的网络，包含了数据是如何在不同的计算之间流动的信息。
