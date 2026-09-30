---
layout: "post"
title: "IMPRESS：重要性感知的多层前缀 KV 存储"
date: "2026-09-30"
description: "分析跨头重要性相似度、KV chunk 重排与重要性驱逐策略。"
categories: ["Papers"]
tags: ["mlsys", "kv-cache", "prefix-cache", "offloading"]
permalink: "/blog/paper-impress/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/mlsys/IMPRESS.md"
---

**_IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference_**

---

## Motivation

这篇论文的前提是Not All KVs Are Equally Important

1. Existing methods must load all prefix keys into GPU memory to compute attention weights and then identify important KVs

也就是说，即使可以根据weight的重要性来降低计算量，仍然避免不了要先将prefix的所有k先全部加载进GPU算weight，IO开销大

2. The existing prefix KV storage and caching systems are suboptimal considering token’s importance.

这是在说明即使分辨出了哪些kv是重要的，但是由于kv存储没有根据重要性来管理（一方面是chunk内部重要和不重要的随机放，一方面是chunk之间哪些放HBM/DRAM/SSD没有根据重要性来放）

---

## Design

IMPRESS的设计基于实验观察到的一个现象：

1. There is a high similarity in the set of important token indices across different heads within the same layer of an LLM.

即多头注意力的不同head之间重要的token分布很近似

于是就有了第一个设计：Similarity-Guided Important Token Identification

<img src="{{ '/assets/blog/paper-impress/image-20251211144734168.png' | relative_url }}" alt="image-20251211144734168" style="max-width: 100%; height: auto;" loading="lazy" />

思路是既然head之间重要性分布相似，那么就不需要把prefix的所有k都IO加载进GPU了，先算少量几个head的qk，根据weights选中各自head的重要token，然后验证这几个head上重要token的分布的相似性是不是超过一个阈值，如果超过就说明上面的观察是成立的，那么后续的head的k以及所有head的v都可以根据重要性只加载部分到GPU上了，大幅减少IO。

效果就变成了这样：

<img src="{{ '/assets/blog/paper-impress/image-20251211145156667.png' | relative_url }}" alt="image-20251211145156667" style="max-width: 100%; height: auto;" loading="lazy" />

第二个设计是关于kv chunk根据重要性来决定存储位置的。

现在的框架有两点问题：

1. kv chunk都是按照token的顺序来存储，那么就会导致重要的token和不重要的混放在一起，即使判断出哪些kv是重要的，IO的时候也要把同chunk内部的不重要kv一起IO

这个问题通过改变token所在的chunk即可解决，把重要的token尽量放在一个chunk里。由于prefix的存储是基于radix tree的，所以这个索引改变仅限于一个node内部，跨node没有必要（跨node可能根本不是一个prefix了）：

<img src="{{ '/assets/blog/paper-impress/image-20251211170148268.png' | relative_url }}" alt="image-20251211170148268" style="max-width: 100%; height: auto;" loading="lazy" />

2. chunk的存储位置也是根据简单的访问频率等参数来决定的，但是还需要考虑chunk内重要token的数量

这一点就是要给每个token维护score，这个score是由往次计算中的weights平均来的，然后决定eviction的依据是score\*frequency
