---
layout: "post"
title: "llama.cpp 源码阅读（3）：GGML 计算流程"
date: "2026-09-30"
description: "以矩阵乘法为例，记录 GGML 张量元数据、后端缓冲区、图分配、调度与结果获取流程。"
categories: ["MLSys"]
tags: ["llama-cpp", "ggml", "source-reading", "cuda"]
permalink: "/blog/llama-cpp-ggml-compute-flow/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/llama.cpp/源码阅读（3）：ggml计算流程.md"
series: "llama-cpp-source-reading"
series_order: 3
---

---

## 1. GGML张量计算过程

通过一个矩阵乘法的例子profile一下ggml用计算后端进行张量分配、张量计算以及结果返回主存的整个流程。

0. ### 准备工作：首先在主存中分配好要参与计算的操作张量

```c++
const int rows_A = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};
const int rows_B = 3, cols_B = 2;
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};
```

1. ### 初始化后端

```c++
	ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif
    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors
    // no need to allocate anything else!
```

2. ### 创建`ggml_context`

   后续的张量分配就是由这个`ggml_context`和`backend`共同决定的

```c++
struct ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);
```

3. ### 创建张量的元数据

```c++
/* 这里的tensor metadata（ggml结构体）就包括了类型信息：
	- 数据类型（如 GGML_TYPE_F32）
	- 维度信息：ne[0]（宽度）和 ne[1]（高度）
	- 步长信息：nb[0] 和 nb[1]（每维的字节跨度）
	- 数据指针：指向实际数据的 data 指针
	- 操作信息：初始为 GGML_OP_NONE */
struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
```

4. ### 实际分配后端`ggml_backend_buffer`来存储这个`ctx`中所有需要存储的张量

这里分配的后端buffer实际存储是在后端上，管理数据是在CPU侧。

```c++
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
```

> **Important**
>
> `ggml_backend_alloc_ctx_tensors`内部依次调用：
>
> ```
> ·ggml_backend_alloc_ctx_tensors_from_buft_impl
> ·alloc_tensor_range
> ·ggml_backend_buft_alloc_buffer
> ·buft->iface.alloc_buffer
> ```
>
> 去分配指定后端上的存储 。
> 在CUDA后端中，`alloc_buffer`函数指针指向`ggml_backend_cuda_buffer_type_alloc_buffer`：
>
> ```c++
> // ggml-cuda.cu
> static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
>     /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
>     /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
>     /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
>     /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
>     /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
>     /* .is_host          = */ NULL,
> };
> ```
>
> `ggml_backend_cuda_buffer_type_alloc_buffer`中的实际后端存储分配逻辑：
>
> ```c++
> ggml_cuda_set_device(buft_ctx->device);
>
> void * dev_ptr;
> cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
>
> // 利用dev_ptr去构造一个有效的buffer上下文
> ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
>
> // 构造完整的 后端缓冲区对象 ，并返回
> /* 一个缓冲区对象包括：
>     - buft: 缓冲区类型，定义了如何分配内存的接口
>     - iface: 缓冲区接口，定义了对已分配内存的操作（如 free、get_base、init_tensor 等）
>     - ctx: 后端特定的上下文（就是上面创建的这个ctx），存储实际内存指针等信息
>     - size: 缓冲区大小*/
> return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
> ```

5. ### 将操作数拷贝至后端缓冲区

```c++
ggml_backend_tensor_set(tensor_a, matrix_A, 0, ggml_nbytes(tensor_a));
ggml_backend_tensor_set(tensor_b, matrix_B, 0, ggml_nbytes(tensor_b));
```

> **Important**
>
> `ggml_backend_tensor_set`就是`ggml_backend_buffer_i`结构体中的一种操作函数。在CUDA后端中，函数指针指向`ggml_backend_cuda_buffer_set_tensor`，`ggml_backend_cuda_buffer_set_tensor`的内部逻辑如下：
>
> ```c++
> // ggml_backend_cuda_buffer_set_tensor与ggml_backend_tensor_set的参数，除了多了ggml_backend_buffer_t buffer其他都是一致的
> static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
>     ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
>
>     ggml_cuda_set_device(ctx->device);
>     CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
>     CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
> }
> ```

6. ### 建图

利用`ggml`提供的操作录制的接口，录制一遍计算图中的逻辑，保存为`ggml_cgraph`:

```c++
struct ggml_cgraph * gf = NULL;
struct ggml_context * ctx_cgraph = NULL;
{
    // create a temporally context to build the graph
    struct ggml_init_params params0 = {
        /*.mem_size =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    ctx_cgraph = ggml_init(params0);
    gf = ggml_new_graph(ctx_cgraph);

    // result = a*b^T
    // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
    // the result is transposed
    struct ggml_tensor * result0 = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);

    // Add "result" tensor and all of its dependencies to the cgraph
    ggml_build_forward_expand(gf, result0);
}

```

7. ### 图分配

分为两步：

```c++
// 第一步先仅根据buffer_type确定分配器，这一步仅确定分配规则等，不负责具体的分配
ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
// 利用承载了分配规则的allocr对录制好的计算图gf进行实际的分配，包括图优化
ggml_gallocr_alloc_graph(allocr, gf);
```

8. ### 图调度

后端调度器负责：

- 图分割：将计算图分割成多个子图，每个子图在最适合的后端上执行
- 内存管理：自动处理跨后端的内存分配和数据传输
- 后端选择：根据操作支持和张量位置智能选择执行后端
- 并行执行：支持流水线并行，提高多设备利用率

```cpp
ggml_backend_sched
```

9. ### 图计算

这一步就是利用前面已经分配好的所有操作数、所有计算逻辑进行最终的图计算

```c++
int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
if (ggml_backend_is_cpu(backend)) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
}
ggml_backend_graph_compute(backend, gf);
```

10. ### 获取计算结果

图计算完，最后结果还在后端的buffer里，需要取回CPU侧：

```c++
// in this example, output tensor is always the last tensor in the graph
struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
float * result_data = malloc(ggml_nbytes(result));
// because the tensor data is stored in device buffer, we need to copy it back to RAM
ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
for (int j = 0; j < result->ne[1]/* rows */; j++) {
    if (j > 0) {
        printf("\n");
    }

    for (int i = 0; i < result->ne[0]/* cols */; i++) {
        printf(" %.2f", result_data[j * result->ne[0] + i]);
    }
}
```

---

## 2. GGML图调试
