---
layout: "post"
title: "vLLM 源码阅读（3）：Scheduler"
date: "2026-09-30"
description: "记录 Scheduler 的运行与等待请求调度、KV slot 分配，以及传递给 ModelRunner 的执行计划。"
categories: ["MLSys"]
tags: ["vllm", "scheduling", "kv-cache", "source-reading"]
permalink: "/blog/vllm-scheduler/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/vLLM/源码阅读（3）：scheduler.md"
series: "vllm-source-reading"
series_order: 3
---

<img src="{{ '/assets/blog/vllm-scheduler/Mermaid%20Chart%20-%20Create%20complex%2C%20visual%20diagrams%20with%20text.-2026-01-10-135956.png' | relative_url }}" alt="Mermaid Chart - Create complex, visual diagrams with text.-2026-01-10-135956" style="max-width: 100%; height: auto;" loading="lazy" />

schedule函数是step中根据当前系统状态以及requests状态来决定哪些requests可以调度、怎么分配kv blocks的函数。

骨干流程如下：

```python
# 先尝试调度RUNNING状态的requests
while req_index < len(self.running) and token_budget > 0:
    # 根据之前得出的这个request需要新计算的token，尝试分配slots（可能会分配新blocks）
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens,
    )
    # 如果不能调度，则根据策略preempt一些requests

# 尝试调度WAITING状态的requests（无论WAITING是从来还没执行过，还是之前的RUNNING requests被preempt成WAITING状态，此时显存中都没有其独占的kv cache了）
# 与RUNNING状态的requests一样，也是向kv_cache_manager请求allocate_slots
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
)

# 一系列处理之后，构造出合法的scheduler output
scheduler_output = SchedulerOutput(
    ...
)

# 构造kv_connector
# 1. Plan the KV cache store
# 2. Wrap up all the KV cache load / save ops into an opaque object
# 3. Clear the internal states of the connector
if self.connector is not None:
    meta: KVConnectorMetadata = self.connector.build_connector_meta(
        scheduler_output
    )
    scheduler_output.kv_connector_metadata = meta
```
