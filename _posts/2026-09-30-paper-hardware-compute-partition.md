---
layout: "post"
title: "GPU 硬件计算分区与 libsmctrl"
date: "2026-09-30"
description: "介绍 TPC 分区、任务描述符、调度限制及 libsmctrl 掩码回调实现。"
categories: ["Papers"]
tags: ["gpu", "sm-partitioning", "libsmctrl", "cuda"]
permalink: "/blog/paper-hardware-compute-partition/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/Hardware_Compute_Partition.md"
toc:
  beginning: true
---

This paper talks about how to partition TPC in nvidia's GPU and how we can use this partitioning method.

---

## Basics

### 1. The structure of NVIDIA's GPU：

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250910215709553.png' | relative_url }}" alt="image-20250910215709553" style="max-width: 100%; height: auto;" loading="lazy" />

some basic definitions:

1. GPC: General Processing Cluster. A GPU may have a few GPCs.
2. TPC: Thread Processing Cluster, the component of GPC. One TPC may have a few SMs.

### 2. TMD

TMD(Task MetaData) is the core data structure corresponding to a kernel(indeed the kernel descriptor). In TMD, there are two fields(`SM_DISABLE_MASK_LOWER` and `SM_DISABLE_MASK_UPPER`) can be used as SM mask.

### 3. HOW NVIDIA GPUS SCHEDULE COMPUTE

The whole structure:

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250910222124531.png' | relative_url }}" alt="image-20250910222124531" style="max-width: 100%; height: auto;" loading="lazy" />

1. Kernel instantiation: construct the TMD, describing the thread blocks, threads per block, and shared memory resources needed for the kernel, and includes the address of the kernel’s entry point.
2. Host Interface: use PBDMA(Pushbuffer Direct Memory Access unit) to load commands from the indirect buffer in Main Memory and then parse the commands and pass them to the appropriate engine(if the command is a kernel launch command, it will be passed to Compute Front End).
3. Compute Front End: in charge of CUDA Context switches and pass TMD pointers to Task Management Unit.
4. Task Management Unit(TMU): queues TMDs by priority and arrival order until the Work Distribution Unit is ready to receive them.
5. Work Distribution Unit(WDU): When a task slot becomes available, WDU signals TMU for a new TMD and inserts that TMD into a slot. Once all of a TMD's blocks are completed, it is removed from task table to free a task slot.

The structure of TMU and WDU:

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250911094526045.png' | relative_url }}" alt="image-20250911094526045" style="max-width: 100%; height: auto;" loading="lazy" />

Note that When TPC partitioning is in-use, if a TPC has available capacity, but the TMD at the head of the sorted table is prohibited from executing on that TPC, the WDU will skip forward in the table until it finds a TMD allowed to execute on the available TPC (or reaches the end of the table).

### 4. Memory Parallelism

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250911095523692.png' | relative_url }}" alt="image-20250911095523692" style="max-width: 100%; height: auto;" loading="lazy" />

The L2 cache and MPU are also sliced through crossbar bus, so each GPC can theoretically be configured to operate with an exclusive subset of the GPU’s cache, bus, and DRAM resources (if Memory Partition Units are partitioned, as in [1]).

---

## Characteristics

### 1. Per-kernel partitions

The partition is encoded into the kernel’s TMD---not as a global GPU property—so it automatically takes effect when the kernel begins.

### 2. Preservation of stream-ordering

CUDA stream semantics require subsequent kernels in the same stream to not launch until all prior ones have been completed—we find that to be preserved, even when TPC partitioning is in use.

### 3. Complex and overlapping partitions

TPC partitions may contain holes or overlap, and have no size restrictions. (By holes, we mean non-contiguous sets of TPCs.)

---

## Limitations

### 1. The consequences of greedy assignment

WDU's greedy assignment approach can result in particularly non-optimal block assignments when overlapping partitions are in use. For example:(K1's priority is higher than K2's)

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250911100910845.png' | relative_url }}" alt="image-20250911100910845" style="max-width: 100%; height: auto;" loading="lazy" />

### 2. The consequences of task slot exhaustion

Task slots fill in priority order, with lower-priority tasks evicted as higher-priority ones become available in the TMU. This can cause unrelated compute partitions to block one another when many streams and stream priorities are in use. For example:(Stream 33 is partitioned to SM 4~8, other streams are partitioned to SM 0~3, K33's priority is lower than other kernels and there are 32 task slots)

<img src="{{ '/assets/blog/paper-hardware-compute-partition/image-20250911101027535.png' | relative_url }}" alt="image-20250911101027535" style="max-width: 100%; height: auto;" loading="lazy" />

---

## Applications

### 1. Protection from competitors

TPC partitioning can be used to contain fault and restrain unknown tasks to a relatively constant amount of compute.

### 2. TPC partitioning allows for smooth adjustment of task execution times

The larger an allocation, the sooner it will meet its deadline. This allows for TPC allocations to be used as a proxy for priority.

### 3. Providing additional TPCs to a task may provide a negligible performance improvement

---

## Code

The core code is in `libsmctrl`.

The core API provided by `libsmctrl` is

```c
void libsmctrl_set_next_mask(uint64_t mask) {
	setup_sm_control_callback();
	g_next_sm_mask = mask;
}
```

And `libsmctrl_set_next_mask` will call `setup_sm_control_callback`. The procedures of `setup_sm_control_callback` are as follows:

- use a specified CUuuid to get a specified export table(base address is `tbl_base`) containing two critical functions: `subscribe` and `enable`.(Note that cuda driver contains one or more export tables to contain function pointers)
- use `subscribe`(address is `tbl_base`+3) to register the hook function `control_callback_v2` to cuda driver
- use `enable`(address is `tbl_base`+6) to activate the registered hook function
- `control_callback_v2` use the thread local `g_next_sm_mask` to set any next kernel's mask(contained in that kernel's TMD) launched from this thread

Some technical points:

1. The mask is thread local(there is also a global mask), so every `libsmctrl_set_next_mask` call can only effect on next kernel launch from current thread.
2. To avoid race conditions among different threads, `control_callback_v2` use thread local `g_next_sm_mask`. Note that although `control_callback_v2` is registered to cuda driver, it is called in the caller thread's call stack(working in that thread's context) so that `control_callback_v2` can only use the thread local variable to avoid race conditions.
