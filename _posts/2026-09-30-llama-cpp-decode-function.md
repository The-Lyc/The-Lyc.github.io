---
layout: "post"
title: "llama.cpp 源码阅读（2）：decode 函数"
date: "2026-09-30"
description: "解析 llama_decode 的 batch 划分、KV 分配计划、计算图构建、输入设置与实际计算。"
categories: ["MLSys"]
tags: ["llama-cpp", "source-reading", "kv-cache", "ggml"]
permalink: "/blog/llama-cpp-decode-function/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/llama.cpp/源码阅读（2）：decode函数.md"
series: "llama-cpp-source-reading"
series_order: 2
---

`llama_decode`函数是推理的主体部分，定义如下：

```c++
int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}
```

进而调用的是`llama_context`的`decode`函数，其基本逻辑如下：

1. 尝试将一整个batch分为多个ubatch，且输出一种可行的内存分配方案：

```c++
mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
```

- 先切分batch：

```c++
auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

// 然后将分割得到的ubatches都放到一个vector中
ubatches.push_back(std::move(ubatch));
```

- 尝试生成kv存储分配计划：

```c++
auto sinfos = prepare(ubatches);

// prepare函数的逻辑如下

// （1）对每个ubatch都进行find_slot操作，这个操作的逻辑为：从存储所有cells的ring buffer中找到合适的可用位置，从head指针开始找，找到一个符合要求的位置就记录一次，直到所有的位置都找到了合适的cell，生成一次成功的计划并返回
const auto sinfo_new = find_slot(ubatch, false);

// 需要记录对应位置和旧内容，后面需要回滚
res.push_back(sinfo_new);
{
    state_t state = { sinfo_new, v_heads, {} };

    for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
        auto & cells = v_cells[sinfo_new.strm[s]];

        state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
    }

    states.push_back(std::move(state));
}
// （2）将每个ubatch的计划应用到真实的元数据，作用是检验每一轮为某个ubatch生成的分配计划是否可行
apply_ubatch(sinfo_new, ubatch);

// （3）如果遍历完，生成了一个合法可行的分配计划，就回滚apply的修改，仅返回分配计划
for (auto it = states.rbegin(); it != states.rend(); ++it) {
    const auto & sinfo = it->sinfo;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & cells = v_cells[sinfo.strm[s]];
        auto & head  = v_heads[sinfo.strm[s]];

        cells.set(sinfo.idxs[s], it->v_cells[s]);
        head = it->v_heads_old[s];
    }
}
```

- 遍历ubatch，带着分配计划进入process_ubatch，进行真正的推理：

```c++
const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);
```

- 拿到推理结果之后，进行一系列的后处理

---

process_ubatch

抛开计算图重用的逻辑，process_ubatch中最重要的逻辑就是如何利用分配计划来写新的kv，以及在build_graph的时候构建按layer去读kv的图，最终在graph_compute过程中去执行实际的读写和计算操作。

1. `mctx apply`

这个函数就是实际应用prepare函数生成的分配计划（和prepare函数中调用的apply_ubatch是一致的，只不过这次调用完就是真的要占用了，因此不会再回滚）：

```c++
kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
```

这里修改的只是CPU侧的元数据，并且确定了后续实际要写入的物理位置

2. `build graph`

这个过程就是录制compute graph里的节点和边（结果和操作），构造kv idxs张量，但并不填充具体的索引数据。举一个构造函数的例子：

```c++
// 构造所有的输入向量
auto * inp_attn = build_attn_inp_kv();

// 实际实现在build_attn_inp_kv_impl
inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

// build_input_k_idxs就是一个构造函数，只是构造一个1d的向量用来容纳所有的k_idx，但是不填充有效的索引数据
ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
```

再举一个录制函数的例子（只记录操作类型和操作数，不会做真正的操作）：

```c++
// 可以看出，这个录制矩阵乘法的函数，仅仅记录操作类型（result->op）以及操作数（result->src）
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

也是在build graph过程中，完成了kv的读写逻辑的录制：

```c++
// build_attn函数
// store to KV cache
{
    const auto & k_idxs = inp->get_k_idxs();
    const auto & v_idxs = inp->get_v_idxs();

    // 这里的关键函数cpy_kv就是图中实现把新kv保存到指定位置的操作，cpy也是录制函数，最终调用ggml_set_rows
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
}

const auto & kq_mask = inp->get_kq_mask();

ggml_tensor * q = q_cur;
// 这里的get_kv函数就是在为做attn操作准备所有需要的kv，即把历史kv都读出来
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
```

3. `set inputs`

这个步骤就需要填充各种input数据了，input数据有多种类型，比如pos，kv等，主要关注的是kv：

```c++
// 调用llama_kv_cache::set_input_k_idxs来填充索引数据
// 这里的dst就是上面已经构造的self_k_idxs
int64_t * data = (int64_t *) dst->data;

// 根据分配计划sinfo填充
for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
    const int64_t offs = sinfo.strm[s]*get_size();

    for (uint32_t i = 0; i < sinfo.size(); ++i) {
        data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
    }
}
```

4. `graph compute`

按照构造好的compute graph以及准备好的input，进行计算。
