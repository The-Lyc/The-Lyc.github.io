---
layout: "post"
title: "TX2 的隐藏调度细节：队列、资源与优先级"
date: "2026-09-30"
description: "解析 TX2 上 kernel、block、copy 与 stream 的排队及优先级约束。"
categories: ["Papers"]
tags: ["gpu", "tx2", "cuda", "scheduling"]
permalink: "/blog/paper-tx2-scheduling/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/TX2_hidden_details.md"
toc:
  beginning: true
---

This paper talks about the scheduling rules of NVIDIA TX2.

---

## 1. basic definitions:

This paper focus on the scheduling rules of blocks, not warps.

A whole block should be scheduled on one SM, and then can be split to warps. Note that a kernel may consist of several blocks, these blocks in the same kernel can be assigned to different SMs. So blocks are the basic scheduling unit when talking about scheduling rules with respect to kernels.

This paper hypothesizes that there are some queues to control scheduling:

- EE queue: per address space, order the kernels waiting to be assigned.
- CE queue: order the copy operations.
- Stream queue: per stream, order the operations(kernel or copy) issued within this stream.

And the definitions of `assigned` and `dispatched`:

- `assigned` is with respect to blocks,means a block has been scheduled for execution on an SM
- `dispatched` is with respect to kernels, means at least one block of this kernel has been `assigned` to an SM. If all blocks of this kernel have been `assigned` to SMs, this kernel is `fully dispatched`.

---

## 2. basic scheduling rules and constraints

First, it gives four general scheduling rules:

1. G1: A copy operation or kernel is enqueued on the stream queue for its stream when the associated CUDA API function (memory transfer or kernel launch) is invoked.
2. G2: A kernel is enqueued on the EE queue when it reaches the head of its stream queue.
3. G3: A kernel at the head of the EE queue is dequeued from that queue once it becomes `fully dispatched`.
4. G4: A kernel is dequeued from its stream queue once all of its blocks complete execution.

> **Caution**
>
> Note that G3 and G4 have different invoking time. Once a kernel is `fully dispatched`, this kernel will be dequeued from its EE queue immediately. However, at this time, this kernel will not be dequeued from its stream queue meaning that this kernel will block subsequent kernels or copy operations in that stream queue. This phenomenon can be illustrated by:
>
> <img src="{{ '/assets/blog/paper-tx2-scheduling/image-20250907175509284.png' | relative_url }}" alt="image-20250907175509284" style="max-width: 100%; height: auto;" loading="lazy" />

Second, it talks about some constraints with respect to resources:

1. `Thread` Resources. On the TX2, there are 2 SMs, each SM can hold 2048 threads, and each block can use 1024 threads at most.

> **Note**
>
> R1 A block of the kernel at the head of the EE queue is eligible to be assigned only if its resource constraints are met.
>
> R2 A block of the kernel at the head of the EE queue is eligible to be assigned only if there are sufficient thread resources available on some SM.

2. `Shared Memory` Resources. On the TX2, shared memory usage is limited to 64KB per SM and 48KB per block.

> **Note**
>
> R3 A block of the kernel at the head of the EE queue is eligible to be assigned only if there are sufficient shared-memory resources available on some SM.

3. `Register` Resources. On TX2, each SM hold 65536 regs, and each thread can use up to 255 regs, each block can use up to 32768 regs. But this constraint is weak, because PTX use virtual regs and will be optimized by compilers.

Thirdly, it gives rules with respect to copy operations:(very similar to general rules)

1. C1: A copy operation is enqueued on the CE queue when it reaches the head of its stream queue.
2. C2: A copy operation at the head of the CE queue is eligible to be assigned to the CE.
3. C3: A copy operation at the head of the CE queue is dequeued from the CE queue once the copy is assigned to the CE on the GPU.
4. C4: A copy operation is dequeued from its stream queue once the CE has completed the copy.

---

## 3. Additional rules

First, it discuss about a special stream called NULL stream which is actually the default stream in CUDA but different from user streams and gives two rules:

1. N1: A kernel Kk at the head of the NULL stream queue is enqueued on the EE queue when, for each other stream queue, either that queue is empty or the kernel at its head was launched after Kk.
2. N2: A kernel Kk at the head of a non-NULL stream queue cannot be enqueued on the EE queue unless the NULL stream queue is either empty or the kernel at its head was launched after Kk.

Actually, NULL stream can block user streams unnecessarily because kernels in user stream launched after head of NULL stream need to wait even if there are enough resources.

Secondly, it elucidate the influence brought by stream priorities. In TX2, there are indeed two priority levels: −1 (priority-high) and 0 (priority-low). Note that streams with no priority specified is priority-low.

When considering priorities, we can hypothesize one additional EE queue for priority-high streams. And the additional rules are:

1. A1 A kernel can only be enqueued on the EE queue matching the priority of its stream.
2. A2 A block of a kernel at the head of any EE queue is eligible to be assigned only if all higher-priority EE queues (priority-high over priority-low) are empty.

> **Caution**
>
> This paper also stresses that it is impossible for kernels in priority-high streams experiencing resource blocking to be indefinitely delayed by kernels in priority-low streams that “cut ahead” and consume available resources. When resources are not enough for high-priority kernels to be assigned, low-priority kernels will also be blocked even if resources are enough for them to be assigned.
