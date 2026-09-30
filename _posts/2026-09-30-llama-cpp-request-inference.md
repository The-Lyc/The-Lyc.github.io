---
layout: "post"
title: "llama.cpp 源码阅读（1）：请求推理过程"
date: "2026-09-30"
description: "从后端初始化、模型加载、分词与 batch 构造到 llama_decode，梳理一轮请求的推理流程。"
categories: ["MLSys"]
tags: ["llama-cpp", "source-reading", "inference"]
permalink: "/blog/llama-cpp-request-inference/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/llama.cpp/源码阅读（1）：request推理过程.md"
series: "llama-cpp-source-reading"
series_order: 1
---

一条request完成一轮推理的过程：

1. 初始化backend

```c++
ggml_backend_load_all();
```

2. 加载模型

```c++
llama_model* model = llama_model_load_from_file(model_path, model_params);
```

3. 创建context

```c++
// 先设置参数
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 256;         // 总共 256 个 cells
ctx_params.n_batch = 64;        // 每次最多处理 64 tokens
ctx_params.n_ubatch = 32;       // 每个 ubatch 32 tokens
ctx_params.n_seq_max = 1;       // 设为 1，简化调试 (unified 模式)
ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // 禁用 Flash Attention
ctx_params.offload_kqv = false; // 禁用 KQV offload 到 GPU
ctx_params.op_offload = false;  // 禁用 op offload 到 GPU

// 创建context，kv cache分配也是在这里
llama_context* ctx = llama_init_from_model(model, ctx_params);
```

4. 分词

```c++
int n_tokens1 = llama_tokenize(
        llama_model_get_vocab(model),
        prompt1, strlen(prompt1),
        tokens1.data(), max_tokens,
        true,   // add_special (BOS)
        false   // parse_special
    );
```

5. 创建batch

```c++
llama_batch batch1 = llama_batch_init(ctx_params.n_batch, 0, 1);
```

6. 填充batch

```c++
for (int i = 0; i < n_tokens1; i++) {
        common_batch_add(batch1, tokens1[i], i, { 0 }, false);
    }
```

7. 调用decode函数进行推理的主体部分

```c++
int ret = llama_decode(ctx, batch1);
```

8. 清理推理完成的batch

```c++
llama_batch_free(batch1);
```
