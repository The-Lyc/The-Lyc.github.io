---
layout: "post"
title: "vLLM 源码阅读（2）：KV 管理"
date: "2026-09-30"
description: "梳理 KVCacheManager、BlockPool、KV Transfer 与 KV 张量逻辑和物理布局。"
categories: ["MLSys"]
tags: ["vllm", "kv-cache", "source-reading", "memory"]
permalink: "/blog/vllm-kv-cache-management/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/vLLM/源码阅读（2）：KV管理.md"
series: "vllm-source-reading"
series_order: 2
toc:
  beginning: true
---

vLLM中KV的管理是非常核心的功能，总体架构图如下：

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20251223205456680.png' | relative_url }}" alt="image-20251223205456680" style="max-width: 100%; height: auto;" loading="lazy" />

也就是说，如上图，整个KV系统包括两大部分：kv cache management和kv transfer system。

---

## 1. KV Cache Management

> **Important**
>
> kv cache management部分主要负责Block分配的工作

整个kv block的分配流程如下：

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20251223211958120.png' | relative_url }}" alt="image-20251223211958120" style="max-width: 100%; height: auto;" loading="lazy" />

scheduler决定了哪些requests会进入下一个step进行decode/prefill，对每个选中的request计算`num_new_tokens`，然后向kv_cache_manager请求新的slots：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
```

在这个`kv_cache_manager.allocate_slots`方法中，做了以下事情：

1. 计算需要分配的block数量
2. 检查可用blocks是否足够（如果不够，返回None）
3. 处理prefix cache命中
4. 分配新的blocks

其中，分配新的blocks这一步就是调用的kv coordinator的`allocate_new_blocks`方法来完成的：

```python
new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )
```

kv coordinator的分配逻辑就是遍历各个single type manager，每个single type manager都调用自己的分配方法：

```python
return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
            )
            for manager in self.single_type_managers
        )
```

`manager.allocate_new_blocks`方法的逻辑就是计算需要新分配的block数，然后从block pool尝试分配block：

```python
req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks
```

block pool的分配逻辑就是从free queue链表中取指定数量的block出来：

```python
if num_blocks > self.get_num_free_blocks():
    raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

# In order to only iterate the list once, we duplicated code a bit
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1
else:
    for block in ret:
        assert block.ref_cnt == 0
        block.ref_cnt += 1
return ret
```

至此，整个block分配部分的逻辑就描述完整了，从 scheduler生成执行计划 开始，到 block pool分配block。

---

## 2. KV Transfer

> **Important**
>
> KV Transfer负责的是kv cache在节点之间以及异构计算单元之间的传输

总体的流程：

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20251223225833031.png' | relative_url }}" alt="image-20251223225833031" style="max-width: 100%; height: auto;" loading="lazy" />

kv connector并不是vLLM运行必需的组件，可以配置为None。但是如果想要做分布式或者offload策略，kv connector就能发挥作用了。

kv connector提供抽象接口，offloading connector这种可以继承并实现抽象接口，并且可以实现多种策略比如LRU。

具体触发时机总结：

| 场景        | 触发条件         | LRU操作                    | 代码位置              |
| ----------- | ---------------- | -------------------------- | --------------------- |
| **CPU存储** | CPU内存不足      | 淘汰最久未使用blocks       | `prepare_store()`     |
| **CPU访问** | lookup/touch操作 | 更新访问顺序               | `touch()`, `lookup()` |
| **GPU分配** | free queue为空   | 淘汰LRU cached blocks      | BlockPool分配逻辑     |
| **GPU释放** | Request完成      | 按反向顺序添加到free queue | Request释放逻辑       |

---

## 3. KV的逻辑存储和物理存储

`_allocate_kv_cache_tensors`函数中给kv cache分配具体的torch tensor：

```python
kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(
        kv_cache_tensor.size, dtype=torch.int8, device=self.device
    )
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

可以看到，初始分配的`kv_cache_raw_tensors`是一个dict：key是`layer_name`（即层），value是该层对应的kv tensor。也就是说，每个layer都有自己的kv tensor，但是这是初始化，仅仅是分配了确定大小的tensor，具体的shape是在后续确定的。

`kv_cache_config`中比较关键的两个成员：

1. `kv_cache_groups`：如果是layer间无差异的模型，那就只有一个group，这个group里记录了:(1)`kv_cache_spec`, including `block_size`(token nums in a block), `page_size_bytes`(bytes in a page, a page is a block); and (2)`layer_names`, including names of all layers in this group, such as

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20260110150555438.png' | relative_url }}" alt="image-20260110150555438" style="max-width: 100%; height: auto;" loading="lazy" />

2. `kv_cache_tensors`：定义是list[KVCacheTensor]，确定了每一层的kv cache tensor大小，以及shared_by参数，比如：

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20260110151147672.png' | relative_url }}" alt="image-20260110151147672" style="max-width: 100%; height: auto;" loading="lazy" />

`_reshape_kv_cache_tensors`函数就是专门为具体的后端reshape还没有具体形状的kv tensor的：

```python
# 取出该层的raw tensor
raw_tensor = kv_cache_raw_tensors[layer_name]

# 先计算 物理block数
num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

# 再通过 逻辑块大小（kv_cache_spec.block_size） 和 内核块大小（kernel_block_size） 计算 内核视角的块个数（kernel_num_blocks）
num_blocks_per_kv_block = (
    kv_cache_spec.block_size // kernel_block_size
)
kernel_num_blocks = num_blocks * num_blocks_per_kv_block

# 构造kv cache tensor的shape
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    kernel_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)

# 根据backend选择最优的stride布局
kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()

# 根据stride策略生成最终的kv cache shape
kv_cache_shape = tuple(
    kv_cache_shape[i] for i in kv_cache_stride_order
)

# 最终reshape成内核接受的kv cache tensor
kv_caches[layer_name] = (
    kv_cache_raw_tensors[layer_name]
    .view(dtype)
    .view(kv_cache_shape)
    .permute(*inv_order)
)
```

举一个实际的例子：

<img src="{{ '/assets/blog/vllm-kv-cache-management/image-20260110155141215.png' | relative_url }}" alt="image-20260110155141215" style="max-width: 100%; height: auto;" loading="lazy" />

在这个例子中，`num_blocks`计算出来是11916个物理块，然后获取`kv_cache_spec`配置中逻辑块大小为16（即一个物理块逻辑上应该装16个token的kv）以及内核块大小也为16（即从内核视角看，一个块应该装16个token），那么这个情况下，逻辑块大小和内核块大小一致，也就是不再把逻辑块细分为多个内核块（`num_blocks_per_kv_block=1`），因此内核块的数量和逻辑块、物理块都保持一致（仍然为11916），然后从计算出的内核视角的block配置得出`kv_cache_shape`。

如果上述过程中，内核块大小变成8，那么还需要将一个逻辑块分成两个内核块，即`num_blocks_per_kv_block=2`，则`kernel_num_blocks=11916*2`，后续的`kv_cache_shape`也将改变。

然后生成stride策略：

![image-20260110160606508]({{ '/assets/blog/vllm-kv-cache-management/image-20260110160606508.png' | relative_url }}){: .img-fluid loading="lazy" }

这里就是NHD布局，N就是token数（16），H就是2（调试用的Qwen2.5-1.5B-Instruct模型，采用了GQA，导致kv head只有2，但是attention head有12，也就是6个q head共享一个kv head），D就是128。这种策略就是把一个block里一个token的head连续放置。

> **Note**
>
> MHA（Multi-Head Attention）：`Hkv = Hq`
> 每个 Q head 都有自己对应的一套 K/V head。
>
> GQA（Grouped-Query Attention）：`1 < Hkv < Hq`
> 多个 Q heads 分组共享 K/V heads。
>
> MQA（Multi-Query Attention）：`Hkv = 1`
> 所有 Q heads 共享同一套 K/V。

有些时候，可能stride策略会变成(0,1,3,2,4)，这个时候就相当于把一个block里所有token的某一个head上的数据连续存。

`BlockPool`的初始化最关键的就是创建所有的`KVCacheBlock`对象，然后创建 `free_block_queue`

```python
# 先创建所有的KVCacheBlock对象
# All kv-cache blocks.
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]

# FreeKVCacheBlockQueue就是初始时按block id把所有block组织成一个双向链表
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```
