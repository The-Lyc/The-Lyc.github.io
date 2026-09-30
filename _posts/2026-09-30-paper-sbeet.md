---
layout: "post"
title: "sBEET：兼顾期限与能耗的 GPU 调度"
date: "2026-09-30"
description: "整理双工作线程、动态 SM 资源划分与可行调度生成的设计思路。"
categories: ["Papers"]
tags: ["gpu", "rtos", "energy", "scheduling"]
permalink: "/blog/paper-sbeet/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/sBEET.md"
---

this paper makes the following contributions:

- derive a power and energy consumption analysis for GPU kernels scheduled with and without spatial multitasking on the GPU, and find that the use of spatial multitasking could result in higher energy consumption.

- develop a runtime scheduling algorithm that reduces deadline misses of non-preemptive GPU kernels by dynamically adjusting the degree of resource partitioning and improves energy efficiency over the existing spatial multitasking approach.

- demonstrate the practical effectiveness of sBEET in real-time performance and energy consumption through a diverse set of experimental scenarios on the latest commercially available embedded GPU platform.

---

Mainly concern about the runtime scheduling algorithm and its corresponding implementation.

sBEET only launch two worker threads on CPU(namely two streams to commit tasks). The reasons given by the author are:

- the use of more workers can lead to more SM going idle at different times, which increases energy consumption
- more workers mean more combinations of SM allocation available for each kernel launched by each worker, and the overhead from increased computational complexity may become unacceptable for the runtime framework running on embedded platforms
- more workers may increase contention on shared resources as reported in [7]
- based on our observation, creating more workers does not necessarily contribute to reducing deadline misses on embedded GPUs.

And the most important content in this paper are the SM allocation algorithm and the schedule generation algorithm.

1. SM allocation algorithm

- Situation 1: There's no task running on GPU at the schedule point, so the whole GPU is idling. Iterating through m from M to 1, calculate the end time of the task needs to be executed and then, list all the expected arriving tasks between now and the end time and generate a schedule within which tasks won't miss their deadline.
- Situation 2: The GPU is partially occupied, the scheduler decides whether the next job **_J_** should be dispatched to the worker thread right away.(because the remaining SMs may be not enough for **_J_** to meet its deadline and **_J_** should wait for the running job ending). Following the same approach as when the GPU is idling, the algorithm calls SchedGen and returns when the generated schedule is feasible.

2. Schedule Generation algorithm

This approach ensures that one or two kernels are running whenever possible, dynamically allocating idle SMs to new, waiting tasks as they arrive and resources become available, all within the given time constraint.

And there is an offline schedule generation, the only difference is offline version can only consider WCET while runtime version can schedule based on practical finished time.
