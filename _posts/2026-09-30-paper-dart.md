---
layout: "post"
title: "DART：实时 DNN 任务的流水线调度分析"
date: "2026-09-30"
description: "整理 DART 的响应时间分析、流水阶段划分与 CPU/GPU 优先级实现。"
categories: ["Papers"]
tags: ["gpu", "rtos", "scheduling", "dnn"]
permalink: "/blog/paper-dart/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/DART.md"
---

This paper is more about a static analysis method than a pipeline implementation.

It split DNN tasks into several stages(one stage on one node). There are two node types:

- CPU node: only CPU cores. OpenBLAS-rt runs on CPU nodes and creates two sets of BLAS threads (RT and BE). Each blas thread is strictly mapped to one core

- GPU node: one CPU core + one GPU device. DART(one context in one node, because one cuda context will occupy the whole GPU device but different streams in one context can share the GPU device at the same time) make use of stream priority and thead-level(block) preemption.

Note that not one layer of DNN corresponds to one node. It's possible that one node has several layers.

---

The most important portion of this paper is schedulability analysis, task pipeline stages design and node configuration.

1. **schedulability analysis**

It bounds the worst-case response time Ri of a task τi by the following iterative equation:

$$

\begin{equation*} R_{i}^{(0)}=\mathbb{C}_{e}^{\ast}(i);\quad R_{i}^{(k)}=\mathbb{C}_{e}^{\ast}(i)+\sum_{\tau_{h}\in hp(\tau_{i})}\lceil\frac{R_{i}^{(k-1)}}{T_{h}}\rceil \mathbb{C}_{h}^{\ast} \tag{5} \end{equation*}


$$

where C∗h is the maximum stage execution time of a task τh among all of its stages (denoted as Ch,max),hp(i) is the set of higher-priority RT tasks than τi, and C∗e(i) is given by:

$$

\begin{align*} & \mathbb{C}_{e}^{\ast}(i)=\sum_{\tau_{w}\in hep(i)}(\mathbb{C}_{w,max}+\mathbb{C}_{w,max}\cdot SM_{w,i}) +\sum_{p_{k}path_{i}\wedge\atop k \leq N_{P-1}}\max_{\tau_{u}\in\Gamma^{\mathrm{RT}}} \mathbb{C}_{u,k}+\sum_{p_{k}\in path_{i}}\max_{\tau_{l}\in l_{p}(i)}\mathbb{C}_{l,max} \tag{6} \end{align*}


$$

This is called RTA(Response Time Analysis) iteration, firstly considering only the chain itself and using the initial respond time to calculate other tasks which will arrive during the initial response time and taking these tasks into account and continuing iteration like this until the response time doesn't change any more.

2. Designing Task Pipeline Stages

It use DP to minimize the maximum utilization of any node：

$$

\begin{equation*} M[n,k]=\min_{x=0}^{n}\max(M[x,[k-1],\ w[k]+\sum_{y=x+1}^{n}U_{i,y}(p_{k})) \tag{7} \end{equation*}


$$

3. Node Configuration

Traverse all the possibilities. For every configuration, use the DP method for `Designing Task Pipeline Stages` to get a staging method and use `schedulability analysis` to check its schedulability. If one method pass the test, it computes the sum of the _weighted worst-case response time_ of all RT tasks. Finally, the algorithm chooses the one with the minimum sum of weight response time and returns it. If it cannot find any configuration that satisfies the schedulability of RT tasks, an empty set is returned as failure.

---

Implementation

On CPU, RT and BE threads hold different priority. On GPU, RT tasks use high-priority CUDA stream while BE tasks use low-priority stream(only two priorities).

Note that if there is a new task at runtime, the node configuration stays still and:

- if it is a RT task, the task is accepted only when all RT tasks including the new one are schedulable.
- if it is a BE task, it is accepted as long as its longest GPU memory copy time is smaller than or equal to the existing value, because the other parts of the BE task will not delay existing RT tasks due to the scheduling class design of DART.
