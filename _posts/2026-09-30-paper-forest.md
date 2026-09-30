---
layout: "post"
title: "Forest：访问感知的 GPU UVM 预取"
date: "2026-09-30"
description: "分析默认树形预取器的局限，以及访问跟踪、模式识别和异构树设计。"
categories: ["Papers"]
tags: ["gpu", "uvm", "prefetching", "memory-management"]
permalink: "/blog/paper-forest/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/Forest.md"
---

This is about GPU unified virtual memory(UVM) and focus on the prefetcher used by GPUs.

---

UVM far-faults process:

<img src="{{ '/assets/blog/paper-forest/image-20251015151311250.png' | relative_url }}" alt="image-20251015151311250" style="max-width: 100%; height: auto;" loading="lazy" />

---

TBNp(Tree-based Neighboring prefetcher)

TBNp uses a full binary tree structure. UVM uses VABlocks as basic management unit and each VABlock is 2MB managed with a five-level full binary tree having 32 64KB leaf nodes.

Leaf nodes contain real data and non-leaf nodes holds two metadata(each indicating the node’s total and migrated child node size). TBNp manages migrations in the unit of a leaf node. The basic rules used within TBNp:

1. once a page encounters a far-fault, the entire 64KB in the associated leaf node is migrated.
2. proactive prefetching: In each 2MB tree, if more than 50% of the leaf nodes are migrated to the GPU memory, the TBNp proactively prefetches the remaining leaf nodes to the GPU by assuming a high locality.

These rules can be illustrated in the following figure:

![isca25-61-fig3]({{ '/assets/blog/paper-forest/isca25-61-fig3-1760499193536-3.jpg' | relative_url }}){: .img-fluid loading="lazy" }

observe two limitations of TBNp:

1. one tree configuration is used for all applications and all data objects regardless of the memory access patterns
2. the UVM driver decides page eviction based on the page fault events, not by the actual page accesses.

Both limitations stem from the driver-side UVM management. Forest is an access-aware GPU UVM management using heterogeneous tree structures.

---

Three Motivations:

1. One Configuration Doesn’t Fit All
2. while most of the migrated blocks are accessed at least once, many pages within each block are never accessed.
3. suboptimal prefetcher leading to unnecessary migrations and driver-driven LRU decision evicting data soon to be migrated again.

---

Forest Design

Base on above discuss, there are four challenges to managing the UVM with access aware heterogeneous tree structures:

1. the access pattern of individual data objects should be identified. The page access counters used by some GPUs only reflect page hotness, which can not be used for access pattern detection.
2. the prefetcher should be configured at runtime. Because the access pattern is identified during runtime, so Forest should config the original TBNp to have the ability of configuring tree structure at runtime.
3. optimal tree configuration should be identified. Based on the access pattern analysis, Forest should construct optimal tree configure.
4. the access pattern detection should not cause performance overhead. This is achieved by speculative version Forest.

The holistic view:

<img src="{{ '/assets/blog/paper-forest/image-20251015152840622.png' | relative_url }}" alt="image-20251015152840622" style="max-width: 100%; height: auto;" loading="lazy" />

1. Access Time Tracker(ATT)

The critical point is leveraging existing hardware page `access counter` registers. `Access timer` maintains the total page access count of the data object. Upon a page access, the `access timer` value of the associated data is incremented by one and the updated value is written to the page’s `access counter` register. This way, the hardware page `access counter` registers can reflect the access order of each page within the data object.

2. Access Pattern Detector(APD)

Leverage an existing UVM driver function (e.g., fetch_access_counter_buffer_entries(.)) which copies all `access counters`(ability of monitoring access count). APD focuses on the VPN and access counter of each page, denoted as PSet (= a set of (pi, ti ) pairs, where pi and ti are the page’s VPN and hardware access counter register value, respecitvely). APD examines PSet of each data object and performs a pattern check based on the pre-defined access patterns. The four patterns are:

- Linear/Streaming (LS): the pages of the data object are accessed in either forward or inverse order. calculate the coefficient of determination (R2) to decide. Obviously, LS needs aggressive prefetch and eviction so that Forest uses the maximum tree size(4MB tree with 256kB leaf nodes).
- Non-Linear High-Coverage High-Intensity (HCHI):a large number of pages in wide address ranges are accessed for a short time(while only few data within the pages are actually accessed). So configure HCHI to use small trees so that prefetches can happen within the small tree boundary. The leaf node is configured to be default size because small leaf nodes will lead to frequent page faults while too large leaf nodes will make the small trees quickly fill up.
- *Non-Linear High-Coverage Low-Intensity (HCLI):*a wide address range within the target data object is accessed with a small number of pages. So configure both trees and leaf nodes small.
- _Non-Linear Low Coverage (LC):_ objects with small footprints and those not settled to a clear access pattern. use the default tree configuration (2MB tree with 64KB leaf nodes) because LC data neither shows a strong access trend nor has clear performance impacts with different tree configurations.

3. Prefetch Engine(PE)

Split original TBNp to smaller node and use two flags(isolation and motion bits) to control the merge and isolation to construct various tree structures.

Eviction: the object with the smallest recency order (which is 1) is the LRU object. By leveraging that, we evict the leaf node that has the LRU page of the LRU object.

The whole process:

<img src="{{ '/assets/blog/paper-forest/image-20251015163600353.png' | relative_url }}" alt="image-20251015163600353" style="max-width: 100%; height: auto;" loading="lazy" />

---

Speculative Forest

1. Access Pattern Recording
2. Static Access Pattern Detection
3. Access Pattern Similarity Detection
