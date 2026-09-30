---
layout: "post"
title: "vLLM 源码阅读（4）：GPUModelRunner"
date: "2026-09-30"
description: "分析 GPUModelRunner 的状态更新、输入准备、BlockTable、Slot Mapping 与 Attention Metadata。"
categories: ["MLSys"]
tags: ["vllm", "source-reading", "inference", "kv-cache"]
permalink: "/blog/vllm-gpu-model-runner/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/vLLM/源码阅读（4）：GPUModelRunner.md"
series: "vllm-source-reading"
series_order: 4
toc:
  beginning: true
---

## 1. 主要流程

vLLM原版的kv cache管理本身就是两阶段的：

- 基于BlockPool管理逻辑索引，scheduler也是基于这个来做决策
- 算子层是基于提交到GPU的BlockTable确定reqs与kv物理存储地址的映射

那么要做offload，则可以借用这个体系的大部分代码：

- 可以复用scheduler做决策依据的逻辑BlockPool以及其上的一整套分配逻辑
- 还可以复用提交给算子BlockTable和slotmapping的逻辑，只不过这里的BlockTable需要由新写的mapper重新映射，但是不影响算子内部的寻址逻辑，也就不需要改算子了

那么关联起scheduler层（代表逻辑索引）和算子层（代表物理索引）的组件就是`GPUModelRunner`，最主要的处理函数就是`execute_model`，大致的流程如下：

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:

    # ========== Phase 1: Pre-Process ==========
    with self.synchronize_input_prep():
        # 1.1 更新持久化状态
        self._update_states(scheduler_output)

        # 1.2 准备输入张量（input_ids, positions, slot_mapping）
        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )

        # 1.3 计算cascade attention前缀长度
        cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(...)

        # 1.4 决定CUDAGraph模式和padding
        cudagraph_mode, batch_desc, should_ubatch, ... = (
            self._determine_batch_execution_and_padding(...)
        )

        # 1.5 构建注意力元数据
        attn_metadata, spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                ...
            )
        )

        # 1.6 预处理（多模态编码、embedding）
        input_ids, inputs_embeds, positions, ... = self._preprocess(
            scheduler_output, num_tokens_padded, intermediate_tensors
        )

    # ========== Phase 2: 模型前向 ==========
    with set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        cudagraph_runtime_mode=cudagraph_mode,
        ...
    ):
        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # ========== Phase 3: 后处理 ==========
    # 计算logits，准备采样
    hidden_states = model_output
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)

    # 保存状态供sample_tokens()使用
    self.execute_model_state = ExecuteModelState(
        scheduler_output, logits, spec_decode_metadata, ...
    )
    return None  # 等待sample_tokens()调用
```

## 2. 关键细节

这个主过程中的一些关键步骤：

### 1. \_update_states

这个步骤就是拿`scheduler_output`去更新GPUModelRunner中的持久化数据以及重要的持久化对象`input_batch`。

骨干流程：

```python
# (1) Remove finished requests from the cached states.
for req_id in scheduler_output.finished_req_ids:
    self.requests.pop(req_id, None)
    self.num_prompt_logprobs.pop(req_id, None)

# (2) Remove unscheduled reqs
unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
for req_id in unscheduled_req_ids:
    self.input_batch.remove_request(req_id)

# (3) Add new reqs' cached states
# 从这里就可以看出来，在GPUModelRunner中，所谓的request就是由CachedRequestState来表示的，有一些确定了就不会再变的基本信息，比如req_id；也有一些每一轮都要根据增量信息改变的，比如block_ids和num_computed_tokens
for new_req_data in scheduler_output.scheduled_new_reqs:
    req_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=new_req_data.prompt_token_ids,
        prompt_embeds=new_req_data.prompt_embeds,
        mm_features=new_req_data.mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        generator=generator,
        block_ids=new_req_data.block_ids,
        num_computed_tokens=new_req_data.num_computed_tokens,
        output_token_ids=[],
        lora_request=new_req_data.lora_request,
    )
    self.requests[req_id] = req_state
    reqs_to_add.append(req_state)

# (4) Update the states of the running/resumed requests.
req_data = scheduler_output.scheduled_cached_reqs
for i, req_id in enumerate(req_data.req_ids):
    req_state = self.requests[req_id]
    req_state.num_computed_tokens = num_computed_tokens
    # 更新block_ids（追加新分配的block）
    if new_block_ids is not None:
        for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
            block_ids.extend(new_ids)

# (5) Add new cached reqs to the persistent batch(InputBatch).
for request in reqs_to_add:
    self.input_batch.add_request(request)

# (6) condense
# self.input_batch.condense() 会把活跃的 requests 往低索引移动，填补中间的空洞，使得 有效 request 行号尽量从 0 开始连续排列
self.input_batch.condense()
```

### 2. \_prepare_inputs

这个函数就是拿着调度决策的结果来准备算子层需要的所有索引数据。

这个函数的输入是：

```python
def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
)
# 这里有一个小细节，num_scheduled_tokens是一个numpy的array，这是在调用这个函数之前专门由记录reqs->num_scheduled_tokens的dict转成的numpy array,就是为了方便这个函数里用这个array 和 req_indices / positions 等向量化操作构建 packed tokens
```

骨干流程如下：

```python
# (1) commit block table(only scheduled reqs' part)
self.input_batch.block_table.commit_block_table(num_reqs)

# (2) Get request indices.
# E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

# (3-1) cumulate req indices: [2, 5, 3] -> [2, 7, 10]
# (3-2) arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

# (4) Get Postions
# 这里也有一个比较细节的写法，self.positions是一个_make_buffer(self.max_num_tokens)的buffer，容量是一个step能schedule的token数量上限，然后这里先取出self.positions的numpy视图（也就是下一步add结果本来就要写入的缓冲区域），用 ufunc 的 'out=' 来直接把结果写进预分配的 buffer，几乎不产生额外临时数组。
positions_np = self.positions.np[:total_num_scheduled_tokens]
np.add(
    self.input_batch.num_computed_tokens_cpu[req_indices],
    arange,
    out=positions_np,
)

# (5) Get token indices.
# E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
# -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
# where M is the max_model_len.
# 这里是因为block table已经经过了condense操作了，所以相邻reqs的起始地址一定是相差M，也就是表的行宽
token_indices = (
    positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
)
token_indices_tensor = torch.from_numpy(token_indices)
torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    token_indices_tensor,
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)

# (6) Compute slot_mapping(physical kv cache slots)
# 在这一步，就完成了每一个token的：block id -> 物理slot的换算(slot_mapping = physical_block_ids * block_size + offset_in_block)
# 需要注意，req_indices和positions_np的size都是scheduled tokens的数量，所以这个过程是对每个token都计算了的
self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

# (7) Prepare the attention metadata.
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
self.seq_lens.np[:num_reqs] = (
    self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
)

# (8) copy to GPU
self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
self.positions.copy_to_gpu(total_num_scheduled_tokens)
```

### 3. \_build_attention_metadata

```python
def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    ...
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:

    # 1. 构建CommonAttentionMetadata（所有层共享的基础信息）
    cm_base = CommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[:num_reqs_padded + 1],
        seq_lens=self.seq_lens.gpu[:num_reqs_padded],
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens_padded,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,  # 物理block映射
        slot_mapping=slot_mapping_gid_0,       # 物理slot映射
        causal=True,
    )

    # 2. 为每个KV Cache Group的每个Attention Group构建metadata
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
        cm = copy(cm_base)

        # 获取该group的block_table和slot_mapping
        cm.block_table_tensor, cm.slot_mapping = (
            _get_block_table_and_slot_mapping(kv_cache_gid)
        )

        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            # 调用具体的Attention Backend构建metadata
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=cm,
            )

            # 将metadata绑定到对应的层
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

    return attn_metadata, spec_decode_common_attn_metadata
```

这里build的metadata包括：

- `query_start_loc` / `query_start_loc_cpu`
  每个 request 在 packed query 中的起始位置，shape 为 `(batch_size + 1,)`。
- `seq_lens`
  每个 request 的序列长度**（**已计算 token 数），shape 为 `(batch_size,)`。
- `num_reqs`
  batch 中 request 数。
- `num_actual_tokens`
  batch 内总 token 数（可能包含 padding）。
- `max_query_len`
  本 batch 中最大 query 长度。
- `max_seq_len`
  本 batch 中最大上下文长度（可能是上界）。
- `block_table_tensor` / `slot_mapping`
  KV cache 的 block 映射与 slot 映射，用于将 token 映射到 KV cache 实际存储位置。
- `causal`
  是否 causal attention。

### 4. set_forward_context

```python
with set_forward_context(
    attn_metadata,           # 每层的注意力元数据
    self.vllm_config,
    num_tokens=num_tokens_padded,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
):
    model_output = self.model(
        input_ids=input_ids,
        positions=positions,
        ...
    )
```

一个forward_context包括以下成员：

- **`no_compile_layers`**：从 `vllm_config.compilation_config.static_forward_context` 拷贝的静态配置。
- **`attn_metadata`**：注意力元数据。
  类型为 `Dict[str, AttentionMetadata]`（v1）或 `List[Dict[str, AttentionMetadata]]`（DBO microbatch）。
- **`virtual_engine`**：当前 forward pass 的虚拟引擎编号。
- **`dp_metadata`**：数据并行相关元数据（可能为 `None`）。
- **`cudagraph_runtime_mode`**：当前运行时 CUDA Graph 模式（NONE/PIECEWISE/FULL）。
- **`batch_descriptor`**：CUDA Graph 的 batch 描述符（可选）。
- **`ubatch_slices`**：microbatch 切分信息（可选）。
- **`additional_kwargs`**：平台/后端额外传递的参数字典。

一个forward_context对象的生命周期就是一次forward pass的过程，`with` 块进入之后，调用self.model()函数进行推理，结束之后就会在`with` 块退出时会恢复/清空全局 `_forward_context`
