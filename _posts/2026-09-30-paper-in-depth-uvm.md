---
layout: "post"
title: "深入 GPU UVM：透明分页与迁移开销"
date: "2026-09-30"
description: "整理重复缺页、主机 OS 操作、超额订阅和预取带来的 UVM 开销。"
categories: ["Papers"]
tags: ["gpu", "uvm", "page-faults", "offloading"]
permalink: "/blog/paper-in-depth-uvm/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/In-depth-UVM.md"
---

This paper talks about the overhead within UVM's transparent paging and migration.

---

UVM fault behaviors

<img src="{{ '/assets/blog/paper-in-depth-uvm/image-20251015164654835.png' | relative_url }}" alt="image-20251015164654835" style="max-width: 100%; height: auto;" loading="lazy" />

In UVM, the fault batch is the fundamental unit of work. The batch size is decided by several factors:

1. $$\mu$$TLB fault limit(one $$\mu$$TLB can be connected with several SMs)
2. additional fault rate throttling mechanism prevents a single SM from creating too many faults(SMs are served relatively fairly)

Several critical factors:

1. Data movement.
2. duplicate faults. the UVM driver workload is **application-driven** and **non-uniform** due to variations in batch size and the number of **duplicate faults**. Experimental results show that **larger batch sizes** generally improve performance despite containing more duplicate faults. This is because the **per-batch processing overhead** dominates the cost of handling a moderate number of duplicates.
3. Fault distribution. **performance variance at the batch level arises partly from the uneven distribution of VABlocks**(every VABlock has different faults) and the GPU’s global batching behavior, where each batch aggregates faults from nearly all SMs but only a few pages per SM.
4. Host OS Interaction. （1）**Unmapping host pages occurs on the GPU fault path**, adding substantial overhead.（2）**Host-side parallelism can worsen this cost**, depending on data access patterns and thread affinity.

Several main behaviors:

1. Oversubscription incur significantly **higher latency** because they must:(1)Fail an allocation attempt,(2)Evict a resident VABlock and migrate it to host memory(not remapped to the CPU unless the CPU accesses it),(3)Restart migration and **repopulate pages** on the GPU (zero-filling before data copy).
2. Prefetch greatly reduces the frequency of GPU–CPU fault handling but comes at the cost of **larger, more expensive batches**, since prefetching can trigger initialization of entire VABlocks:(1)**Creating DMA mappings** for all pages in the VABlock, enabling data transfer between CPU and GPU.(2)**Building reverse DMA address mappings** stored in a **radix tree** in the Linux kernel.
3. Prefetch & Eviction. most runtime cost still stems from **host-side OS operations** such as DMA mapping and page unmapping. Prefetching reduces the number of batches but does not eliminate these overheads.

Finally, the authors discuss all the factors and their possible solutions:

1. Duplicate faults. The primary objective is to accept as many unique faults as possible to reduce the total number of batches.
2. Host OS operations.
