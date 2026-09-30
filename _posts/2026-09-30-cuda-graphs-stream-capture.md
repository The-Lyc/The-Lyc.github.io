---
layout: "post"
title: "CUDA Graph、Stream Capture 与默认流语义"
date: "2026-09-30"
description: "记录 CUDA Graph 的构建与启动、Stream Capture 约束，以及 legacy 和 per-thread 默认流的行为。"
categories: ["MLSys"]
tags: ["gpu", "cuda", "cuda-graph", "streams"]
permalink: "/blog/cuda-graphs-stream-capture/"
lang: "zh-CN"
notes_import: true
source_path: "MLSys/GPU/CUDA-GRAPH.md"
---

1. Tech Blog. "Enabling Dynamic Control Flow in CUDA Graphs with Device Graph Launch"

https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/

Graph launches can also be nested and recursive

cuda graph的两种launch方式：

- fire-and-forget launch，就是最常用的方式，包含隐式的graph upload和立即的dispatch，cpu提交完不再能控制后续的执行和同步。

  ```c
  cudaGraphLaunch(deflateGraph, cudaStreamGraphFireAndForget);
  ```

- tail launch，可以在device侧launch graph的方式，并且可以在device侧做同步

  ```c
  cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
  ```

2. https://zhuanlan.zhihu.com/p/700224642

这篇博客对cuda graph进行了比较详细的分析。

cuda graph首先确实是有向图，边表示节点之间的依赖关系，当一个节点的前序节点都执行完了的时候，后一个节点就可以开始执行。

cuda graph中常见的node类型包括：

- GPU kernel node，表示一个cuda kernel
- Memcpy node，可以进行device与host之间的内存拷贝
- Memset node，对device memory进行初始化
- Host (executable) node, 可以执行一个CPU上的函数
- subgraph node, 一个cudagraph子图。
- Empty (no-op) node, 一个空节点。
- External event wait node，表示等待某个event的节点
- External event record node，表示记录某个event的节点
- Memory allocation node，表示一个内存分配的节点
- Memory free node，表示一个内存释放的节点

使用cuda graph的关键步骤是在cuda代码中获取graph，最常用的步骤是用stream capture来获取，一个实例：

```c
// Start stream capture on stream1
cudaCheck(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

const int64_t repeat = 1000;

// Execute kernels with stream capture
kernel1<<<gridDim, blockDim, 0, stream1>>>(d_data1, repeat);
cudaCheck(cudaEventRecord(event1, stream1)); // Record event1 after kernel1 execution

// Use events to synchronize
cudaCheck(cudaStreamWaitEvent(stream2, event1, 0));
kernel2<<<gridDim, blockDim, 0, stream2>>>(d_data1, repeat);
cudaCheck(cudaEventRecord(event2, stream2)); // Record event after kernel2
kernel3<<<gridDim, blockDim, 0, stream1>>>(d_data2, repeat);
cudaCheck(cudaStreamWaitEvent(stream1, event2, 0)); // Wait on event2 in stream1

// End stream capture
cudaGraph_t graph;
cudaCheck(cudaStreamEndCapture(stream1, &graph));
```

上面这个示例里，不仅展示了capture函数的使用（`cudaStreamBeginCapture`,`cudaStreamEndCapture`），还展示了capture是以一个主stream为起点，可以利用cuda event去获取逻辑上包含多个stream的graph。具体来说，`cudaStreamBeginCapture`的参数就是主stream，这里是stream1，但是在kernel1 launch之后，用`cudaEventRecord` 函数record了event1，然后stream2用`cudaStreamWaitEvent`等待wait到event1之后再launch kernel2，这样stream2就对stream1有依赖了，就能被capture到graph中了。

上面这段代码capture到的graph如下图所示：

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/v2-a7398fd8237e416f09089580cd5851f3_1440w.jpg' | relative_url }}" alt="img" style="max-width: 100%; height: auto;" loading="lazy" />

capture完成之后，`cudaStreamEndCapture(stream1, &graph)`函数是把graph保存到了cudaGraph_t类型的变量graph中了，相当于镜像的概念，要launch一个graph到gpu之前需要先实例化为一个cudaGraphExec_t类型的变量：`cudaGraphInstantiate(&instance, graph, NULL, NULL, 0)`。最后使用`cudaGraphLaunch(instance, stream1)`隐式upload到gpu并fire-and-shot式地启动这个graph。

需要特别注意的是，capture期间是有限制的，主要原因是capture begin和end包裹的过程中，是没有在gpu实际执行任务的，只是在record这个graph的组成，因此是不允许使用同步操作的（同步等待的任务永远不会结束，因为根本就没有开始），具体而言，包括：

- 对stream进行同步（cudaStreamSynchronize）
- 隐含stream同步的操作（对context进行同步、对device进行同步，都会强制内部包含的全部stream进行同步，如果其中有stream处在graph capture状态，就会报错）
- 隐含stream同步的操作,如当前graph capture的stream是blocking stream，则涉及null stream的操作都不可用，例如cudaMalloc
- 对stream上面record的event进行的状态查询、同步操作

这里需要阐明几个概念，`null stream`,`default stream`,`blocking stream`,`nonblocking stream`：

- null stream/default stream是同一个stream，都是0号stream，较新的cuda toolkit document里边写的是可以根据环境变量配置为两种不同的行为：一种是legacy行为，这是老版cuda的null stream的行为（可以看paper-reading/gpu/TX2这篇论文对null stream的解析），null stream头部的任务在EE队列里排队，但是需要等待其他blocking stream上所有比null stream当前头部的任务更早提交的任务全部dispatch之后，null stream头部的任务才可以被调度执行；另一种是per-thread default stream行为，不同于legacy的所有CPU线程共享同一个null stream，per-thread的行为是每个CPU线程都有一个独立的null stream，并且null stream的行为等同于nonblocking的stream行为
- blocking stream，这是直接使用cudaStreamCreate()创建的stream，其行为也是根据null stream的模式来的。如果是legacy模式，那么blocking stream就是会和null stream互相阻塞；如果是per-thread的模式，blocking stream的行为实质上和nonblocking stream是一致的
- nonblocking stream，无论是legacy还是per-thread模式，都是与null stream完全并行的。

上述行为完全可以通过实验验证，但需要注意，一定要把cudaEventRecord(eventEnd)分别放在kernel launch的语句之后，不要一起放在最后；还有cudaDeviceSynchronize要放在event的record语句之后，不然测得没有意义。示例代码：

```c
/** estimate total time of two kernels  */
    cudaCheck(cudaEventRecord(eventStart0));
    cudaCheck(cudaEventRecord(eventStart1, stream));

    long_kernel<<<gridDim, blockDim, 0, 0>>>(d_out_buffer, LOOP_COUNT);
    cudaCheck(cudaEventRecord(eventEnd0));

    long_kernel<<<gridDim, blockDim, 0, stream>>>(d_out_buffer, LOOP_COUNT);

//    long_kernel<<<gridDim, blockDim, 0, 0>>>(d_out_buffer, LOOP_COUNT);

//    long_kernel<<<gridDim, blockDim, 0, stream>>>(d_out_buffer, LOOP_COUNT);

    cudaCheck(cudaEventRecord(eventEnd1, stream));

    cudaCheck(cudaDeviceSynchronize());

    float totalTime0 = 0;
    float totalTime1 = 0;
    cudaCheck(cudaEventElapsedTime(&totalTime0, eventStart0, eventEnd0));
    cudaCheck(cudaEventElapsedTime(&totalTime1, eventStart0, eventEnd1));
```

启用legacy/per-thread模式：编译时启用或者在include语句前加宏定义（需要注意，实验显示加编译选项的优先级高于编译模块中宏定义）

```bash
 # 启用per-thread
 nvcc --default-stream per-thread blkstream.cu -o test
 # 启用legacy
 nvcc --default-stream legacy blkstream.cu -o test
```

```c
// Per-thread
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>

// Legacy
#define CUDA_API_LEGACY_DEFAULT_STREAM
#include <cuda_runtime.h>
```

实验是用一个计算sin和cos的kernel，分别在null stream和created stream上launch，利用event来统计时间。

legacy模式下，如果created stream是blocking stream：

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028172556933.png' | relative_url }}" alt="image-20251028172556933" style="max-width: 100%; height: auto;" loading="lazy" />

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028172644867.png' | relative_url }}" alt="image-20251028172644867" style="max-width: 100%; height: auto;" loading="lazy" />

legacy模式，如果created stream是nonblocking stream：

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028172734699.png' | relative_url }}" alt="image-20251028172734699" style="max-width: 100%; height: auto;" loading="lazy" />

per-thread模式，如果created stream是blocking stream：

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028173324345.png' | relative_url }}" alt="image-20251028173324345" style="max-width: 100%; height: auto;" loading="lazy" />

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028172949106.png' | relative_url }}" alt="image-20251028172949106" style="max-width: 100%; height: auto;" loading="lazy" />

所以回到刚刚的问题，涉及null stream的同步操作，在legacy模式下会影响blocking stream，因此不能在capture操作过程中使用。理论上来讲，per-thread模式下blocking stream是不受null stream同步影响的，其实是可以用同步的cudaMalloc，但是实测不可以：

<img src="{{ '/assets/blog/cuda-graphs-stream-capture/image-20251028175824452.png' | relative_url }}" alt="image-20251028175824452" style="max-width: 100%; height: auto;" loading="lazy" />

---

考查vllm中cuda graph的使用

https://docs.vllm.ai/en/latest/design/cuda_graphs.html#overview
