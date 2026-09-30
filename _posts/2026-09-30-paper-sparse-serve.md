---
layout: "post"
title: "SparseServe：长上下文动态稀疏注意力服务"
date: "2026-09-30"
description: "介绍碎片感知传输、工作集批量控制与逐层分段预填充。"
categories: ["Papers"]
tags: ["mlsys", "sparse-attention", "kv-cache", "cpu-gpu"]
permalink: "/blog/paper-sparse-serve/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/SparseServe.md"
toc:
  beginning: true
---

**_SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving_**

---

## Background

动态稀疏注意力（`Dynamic sparse attention`）的显存瓶颈是显存容量，因为要动态选取attend的kv，如果要保证选完之后快速访问的话，即使选中的是小部分kv也需要让全部kv驻留显存，这样的话显存很快就会出现瓶颈。

---

## Challenges & Innovations

1. Fragmented KV cache transfers reduce effective DRAM access bandwidth ---> fragmentation-aware KV cache transfer

> **Note**
>
> GPU上的decode过程完成之后，一个batch新生成的token的kv是连续的，但是不管是GPU还是CPU存储kv都是按token来的，也就是decode过程要求的数据布局和实际存储是不相符的。如果碎片化地用memcpy传输非常低效。

2. HBM cache thrashing ---> working-set-aware batch size control based on real-time working set estimation

> **Note**
>
> 如果始终维持一个恒定的大batch size，会加剧HBM的争用，导致频繁的kv换入换出

3. High HBM requirement of chunked prefill ---> layer-segmented prefill that bounds HBM use during prefill to a single layer

> **Note**
>
> prefill需要的kv空间非常大，即使用了chunked prefill，这个chunk也需要所有layer的所有历史kv，既然很难从全量的历史kv下手，那么就把粒度从一个chunk的所有layer变成一个chunk的一个layer，这样就能大幅减小一个step内chunk prefill需要的存储预算，减少prefill requests被阻塞的情况

系统的总架构：

<img src="{{ '/assets/blog/paper-sparse-serve/image-20260108102217091.png' | relative_url }}" alt="image-20260108102217091" style="max-width: 100%; height: auto;" loading="lazy" />

---

## Implementation

### 1. FlashH2D: GPU-Direct Loading.

如果直接用memcpy：

<img src="{{ '/assets/blog/paper-sparse-serve/image-20260108105025618.png' | relative_url }}" alt="image-20260108105025618" style="max-width: 100%; height: auto;" loading="lazy" />

改进为CPU侧先把需要传输的数据的指针pack到一起，然后cuda kernel里面利用UVA机制多线程并行传输这些指针对应的数据。

### 2. FlashD2H: CPU-Assisted Saving.

生成的新kv的布局和实际存储布局不相同，但是生成的kv的存储是连续的，因此先把这些连续的kv传输到CPU侧，再由CPU多线程进行解析并存储到对应的碎片化的位置：

<img src="{{ '/assets/blog/paper-sparse-serve/image-20260108105433162.png' | relative_url }}" alt="image-20260108105433162" style="max-width: 100%; height: auto;" loading="lazy" />

### 3. Working-Set-Aware Batch Size Control

prefill阶段的working set就是一个iteration中需要用来存要用的kv的HBM容量。这个容量大小虽然可以精确计算，但是随着chunked prefill的推进而逐渐增大，因为需要历史的所有kv。

decode阶段的working set 就很难计算了，因为DSA是动态选取需要的kv。本文利用working set的时间局部性来估算working set。

最终，Working-Set-Aware Batch Size Control的scheduler有三个约束：

(1) $$R_{max}$$ , which bounds the maximum number of requests in a batch;

(2) $$T_{max}$$ , which limits the total number of tokens in a batch to control the computational workload, especially for prefill requests;

(3) $$M_{avl}$$ , which specifies the available GPU HBM cache capacity.（实际上working set就是体现在这个约束中）

实际运行时，就先用vLLM的scheduler先生成一个候选的batch（这个应当满足$$R_{max}$$和$$T_{max}$$），然后去估算working set大小，让batch满足$$M_{avl}$$。

### 4. Layer-Segmented Prefill

设定一个`maxInjectToken`参数来约束一个batch中最多加入的prefill的token数，这个参数需要离线实验去调参。
