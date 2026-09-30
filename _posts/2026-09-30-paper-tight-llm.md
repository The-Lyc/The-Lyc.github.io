---
layout: "post"
title: "TightLLM：自适应卸载提升 LLM 吞吐"
date: "2026-09-30"
description: "梳理上下文增长带来的 I/O 失衡，以及 KV 重算和权重加载重叠设计。"
categories: ["Papers"]
tags: ["mlsys", "offloading", "kv-cache", "recomputation"]
permalink: "/blog/paper-tight-llm/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/TightLLM.md"
---

**_TightLLM: Maximizing Throughput for LLM Inference via Adaptive Offloading Policy_**

---

## Motivation

静态的offload策略会失效：

<img src="{{ '/assets/blog/paper-tight-llm/image-20251210154012145.png' | relative_url }}" alt="image-20251210154012145" style="max-width: 100%; height: auto;" loading="lazy" />

图的上半部分是prefill阶段，可以看到每个block的权重加载和每个batch的kv cache offload用时都是超过一个batch在一个block上的prefill计算时间的，因此：

- 切换block的时候要等权重的load
- 算完一个batch的prefill后要等上一个batch的kv cache被换出去才能腾出显存开始下一个batch的prefill

图的下半部分是decode阶段：

- 一个block的权重load时间大于这个block上最后一个batch的decode时间 ====> 每个block上的第一个batch要等权重load
- 上下文越来越长之后，每个batch的kv cache load时间越来越长直到超过上一个batch的decode时间 ====> 这个batch就会等kv cache load

论文解决的两个关键问题：

- How to adapt to the increasing KV cache loading time?
- How to reduce the overhead of large weight loading time?

---

## System Design

kv cache distributor

decode阶段，上下文变长之后，kv cache transfer的时间要大于decode时间，gap比较大，所以拿一部分kv cache放到gpu上直接重算：

<img src="{{ '/assets/blog/paper-tight-llm/image-20251210160113467.png' | relative_url }}" alt="image-20251210160113467" style="max-width: 100%; height: auto;" loading="lazy" />

prefill阶段，prompt通常是多个token，虽然不需要load kv，但是算完之后需要offload kv，offload时间超过prefill的时间，解决方法是把确定之后需要重算的kv直接丢弃，只offload之后需要load的kv：

<img src="{{ '/assets/blog/paper-tight-llm/image-20251210161408469.png' | relative_url }}" alt="image-20251210161408469" style="max-width: 100%; height: auto;" loading="lazy" />

weight loader

基本思路就是让gpu上的计算和权重的IO并行起来，并且把并行IO的任务slice给每个batch，变成这样：

<img src="{{ '/assets/blog/paper-tight-llm/image-20251210162013535.png' | relative_url }}" alt="image-20251210162013535" style="max-width: 100%; height: auto;" loading="lazy" />

但是细分可以分成short sequence和long sequence两种情况：

- 短序列只要用并行IO，基本上就能实现计算和IO的overlap，因为此时需要load的kv cache并不多
- 长序列由于需要load的kv cache逐渐变多，导致即使把并行IO的任务slice到每个batch，权重+kv的IO时间仍然会超过每个batch的计算时间，此时就需要把这个因素也考虑进ILP中，利用重算更多的kv来实现更均衡的overlap
