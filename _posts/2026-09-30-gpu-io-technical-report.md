---
layout: "post"
title: "CPU–GPU I/O：PCIe 路径与碎片化传输优化"
date: "2026-09-30"
description: "从命令提交、PCIe 请求与完成协议解释碎片化 H2D 的瓶颈和优化顺序。"
categories: ["Papers"]
tags: ["gpu", "pcie", "copy-engine", "benchmark", "cpu-gpu"]
permalink: "/blog/gpu-io-technical-report/"
lang: "zh-CN"
notes_import: true
source_path: "paper-reading/GPU/cpu_gpu_io_report/CPU_GPU_IO技术报告.md"
toc:
  beginning: true
---

# CPU–GPU I/O：数据怎样跨过 PCIe，以及碎片化传输怎样优化

本文讨论 **PCIe 连接的离散 NVIDIA GPU 上，CPU 主存到 GPU 显存（H2D）的搬运**。核心问题是：一批逻辑上很小的 copy，怎样让 CPU 持续供给工作、让设备保持足够多的请求在途，并在数据可用时正确通知消费者？沿这条路径理解各环节，才能判断应该合并数据、减少提交、增加流水，还是改用另一种搬运方式。

文中本机实验来自 Ryzen 9 9950X、GeForce RTX 4090（AD102）、驱动 595.91.07、CUDA Toolkit 12.8；raw CE 原型使用 tinygrad 提交 `794d314db5f67328b86886898e30f3c6a6d21a39`。带宽按十进制 GB/s 计算，KiB/MiB 按二进制计算。实验是已有记录，本次编辑未重新运行。**API、CE 命令、队列发布和 PCIe 事务是不同层次；下文没有把推断出的请求并发量当作硬件实测。**

## 1. 从应用请求到数据可用：I/O stack 的完整路径

![CPU、PCIe 与 GPU 组件之间按编号标注的控制流和数据流]({{ '/assets/blog/gpu-io-technical-report/fig01_io_stack.png' | relative_url }}){: .img-fluid loading="lazy" }

_图 1：经典 pinned H2D 的 CE 路径。蓝色为命令／完成控制，橙色为 GPU 向主机发出的读请求，绿色为返回正文及显存写入。图中的 pushbuffer 按本机原型放在主存；组件关系与步骤顺序不代表逐包时序，也不意味着每种 CUDA copy 都选择 CE。[SVG 矢量图]({{ '/assets/blog/gpu-io-technical-report/fig01_io_stack.svg' | relative_url }})、[PDF]({{ '/assets/blog/gpu-io-technical-report/fig01_io_stack.pdf' | relative_url }})。_

图中的编号对应一次 copy 从准备到可用的逻辑顺序。**①—⑧ 是控制路径，⑨ 是向主机发出的读请求，⑩ 是正文返回与显存写入，⑪ 建立完成与消费者依赖。** 设备可预取和流水，所以编号并不表示相邻步骤之间必然等待前一步全部结束。

| 编号 | 组件之间发生什么                                                                                                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ①    | 应用／框架向 CUDA runtime 或用户态驱动提交 `src、dst、size` 和 stream 依赖。一个 tensor 可以展开为多个 copy，也可以与邻项合法合并。                                                                 |
| ②    | 驱动与 RM/UVM 协作检查权限、注册／锁定 host 页，建立 GPU VA 到主机 DMA 地址和目的显存的映射；常驻 buffer 可复用这些冷路径准备。pinned 不等于物理连续，UVA 指针数值相同也不保证任意 channel 可访问。 |
| ③    | 用户态驱动把地址、长度、布局与顺序属性编码到 pushbuffer。普通 pageable 源可能先经 CPU staging 到 pinned 内存；本图之后的正文路径以 pinned 源为例。                                                  |
| ④    | CPU 确保命令可见后，发布指向 pushbuffer 的 GPFIFO entry，更新 PUT。队列条目只描述命令的位置／长度，并未复制正文。                                                                                   |
| ⑤    | CPU 写 GPU doorbell MMIO，通知相应 channel 有新工作；写入或 fence 返回不代表 copy 完成。                                                                                                            |
| ⑥    | GPU 命令前端接收通知，定位已发布的队列工作。                                                                                                                                                        |
| ⑦    | PBDMA 从命令内存获取 pushbuffer 并解析命令；本机原型的命令缓冲区在主存，获取过程也跨 PCIe。取命令的 PBDMA 与搬正文的 CE 不是同一组件。                                                              |
| ⑧    | GPU 前端把搬运命令交给 CE；CE 使用源、目的、长度和流水／顺序属性推进 copy。                                                                                                                         |
| ⑨    | CE 经 GPU MMU／endpoint 发出 PCIe MRd；请求跨 PCIe、Root Complex 和可能的 IOMMU 到达 pinned host 页。CPU 核无需逐包执行 memcpy。                                                                    |
| ⑩    | 主机经相反方向以 CplD 返回数据；GPU 根据在途请求状态关联目的地址，将正文写入 VRAM。一个逻辑 copy 可涉及多笔 MRd／CplD。                                                                             |
| ⑪    | 批末 release／flush、event 或 semaphore 建立完成与可见性；消费者依赖满足后才能安全读数据，并回收源、目的、命令和队列空间。API 返回、队列取走、显存写完和消费者可用是不同事件。                      |

### 1.1 准备和提交：CPU 写描述，GPU 才搬正文

对经典显式 DMA，host 源页应在搬运期间保持有效且可被设备访问，通常复用已注册的 pinned pool。普通 pageable buffer 可能增加一次 CPU staging copy，并改变 API 的同步行为；名字带 `Async` 不保证任何指针和资源状态下都立即返回。[CUDA 12.8 同步语义](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)、[Linux DMA 地址模型](https://docs.kernel.org/core-api/dma-api-howto.html)

提交时，应用提供的 `src/dst/size` 变成命令内存中的地址、长度、布局和顺序属性。CPU 写完命令、保证其对设备可见后，再发布 GPFIFO 条目、更新 PUT、写 doorbell；doorbell 通常是发往设备 MMIO 的 PCIe posted write，**不代表数据已搬完**。发布成本也不只是最后一次寄存器写，还包含队列更新和必要屏障。[NVIDIA UVM 提交实现](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.91.07/kernel-open/nvidia-uvm/uvm_channel.c#L984)、[Linux MMIO 顺序](https://docs.kernel.org/driver-api/device-io.html#accessing-the-device)

本机 raw CE 的 4096 × 4 KiB 实验清楚显示这些计数不能互换：**4096 个逻辑 copy、4096 条正文 `LAUNCH_DMA`、一次 GPFIFO 发布／doorbell**，另有三条只用于时间戳和完成的命令；PCIe 请求及 completion 数量没有测量。每条正文命令在这个原型中编码为 36 B，整批命令内存为 147,528 B；这不是跨 GPU 的通用描述符长度。[原型源码]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_probe.cpp' | relative_url }})、[命令解码]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_ce_launch_sweep_results.txt' | relative_url }})

### 1.2 执行和 PCIe：字节方向不等于请求方向

GPU 前端取得命令后，CE 可执行独立 copy，SM kernel 则通过线程的 load/store 搬运。对典型 CE H2D，**GPU 发出 PCIe Memory Read（MRd）请求，主机以 Completion with Data（CplD）返回正文，GPU 再写入目的显存**。因此“数据从 CPU 到 GPU”不意味着 CPU 主动逐块推送。相反，典型 CE D2H 读取本地显存，再向主机发 posted Memory Write（MWr）；两方向受不同事务与缓冲条件约束，不能假设带宽对称。

每条 copy 可跨多个请求，返回数据又可能分成多个 completion。MRRS 限制单个读请求的最大长度，MPS 限制数据包 payload，常规请求不能跨 4 KiB 边界；它们都不等于应用 copy 的大小。tag 关联未完成读和返回，credit 控制接收缓冲资源，GPU 内部还必须跟踪返回数据的目标位置。PCIe 事务层处理请求与流控，链路层处理可靠传输，物理层在协商的 lanes 上传输比特；“多请求并发”指生命周期重叠，而非多个包同时独占同一方向的链路。主机 NUMA 放置、Root Complex 和主存带宽还会影响取数。**请求并发、请求粒度、返回缓冲和链路带宽共同限制吞吐**；本机没有抓取 TLP 或直接测量 tag 深度。[AMD H2C 拆分示例](https://docs.amd.com/r/4.0-English/pg302-qdma/H2C-MM-Engine)、[读请求跟踪和返回缓冲](https://docs.amd.com/r/en-US/pg054-7series-pcie/Tracking-Non-Posted-Requests-and-Inbound-Completions)

CE 的跨 copy `PIPELINED` 属性允许后续独立工作在前一项结束前推进；`NON_PIPELINED` 施加更强的 copy 间顺序。**non-pipelined 不表示单个大 copy 只能有一笔请求在途**。原型的流水模式是首条 non-pipelined、后续独立 copy pipelined、批末统一建立完成；有写后读或目的重叠时必须保留依赖。[NVIDIA UVM CE 编码](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.91.07/kernel-open/nvidia-uvm/uvm_turing_ce.c#L248)

### 1.3 完成：计时和资源回收的终点

GPU 取到队列条目，不代表显存写入已完成。本机 raw CE 用批末带 SYS flush 的时间戳／完成命令，CPU 观察完成序号后才回收命令缓冲区。CUDA 路径则依赖 stream 顺序、跨 stream event 和必要同步；自定义 CE channel 或 CPU BAR 写入还要证明与 CUDA 消费者之间的内存顺序。源 buffer、目的 allocation 和命令内存必须活到相应完成点。[原型完成逻辑]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_probe.cpp' | relative_url }})、[GPUDirect 内存顺序](https://docs.nvidia.com/cuda/gpudirect-rdma/#synchronization-and-memory-ordering)

## 2. 几种 I/O 方式改变了哪一段路径

下表以“动态小块 H2D 是否合适”为比较目标。它们多数仍受同一 PCIe 链路限制，区别在于**谁准备地址与命令、谁搬正文、怎样表达碎片和依赖、占用什么资源**。

| 方式                         | 改变的环节与优势                                                  | 主要代价／适用边界                                                                     |
| ---------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Pinned + `cudaMemcpyAsync`   | 标准显式搬运；可用 stream 与计算重叠，连续大块易接近链路带宽      | 每个动态小 copy 的 API／驱动处理突出；仍须管理 pinned 源生命周期                       |
| Pageable + CUDA copy         | 使用简单，不要求应用先注册源页                                    | 可能增加 pinned staging、CPU 内存流量和同步；需按实际指针检查行为                      |
| `cudaMemcpy2DAsync`／3D      | 一次表达固定 pitch 的多行／多维复制，压缩规则布局的控制工作       | 不能表达任意源、目的地址数组                                                           |
| `cudaMemcpyBatchAsync`       | 一次 API 接受动态 copy 列表，减少应用层调用                       | 内部仍要处理 O(N) 项；不保证一次硬件发布或某种 CE 流水；批内无固定顺序保证             |
| CUDA Graph memcpy nodes      | 重复拓扑可预先构图和实例化                                        | 动态列表需要更新节点／重建；节点并未消失，本机 fan-out 图没有获益                      |
| SM gather kernel             | 线程可灵活读取任意 host mapping、搬到显存，并可融合格式变换／计算 | 需上传或构造地址表；占用 SM、寄存器及访存资源                                          |
| Mapped-host 直接消费         | 下游 kernel 直接读所需 host 字节，省去完整显存副本                | 多次远程读取可能比预搬运更贵；“零显式 copy”不等于零 PCIe 流量                          |
| CPU BAR 写入／GDRCopy 类路径 | CPU 把正文写进长期映射的 GPU buffer；不占 SM，源地址可动态变化    | 占 CPU 核和 BAR/WC 资源；需管理 pin/map、fence 与 GPU 消费者同步；BAR 读不具备同样优势 |
| 自定义 raw CE                | 可批量编码、分组发布、控制跨 copy 流水；正文不占 SM               | CUDA VA／stream 互操作、完成协议、错误恢复和私有 ABI 维护成本高                        |
| Unified Memory／HMM          | 系统管理远程访问、迁移与页表，可简化所有权管理                    | 缺页、迁移粒度和 CPU/GPU 往返可能放大稀疏访问；不保证更快                              |

GPUDirect RDMA／Storage 面向 NIC 或存储设备与 GPU 的数据路径；Hopper TMA 和 `cp.async` 面向 kernel 内部搬运，都不是把任意 CPU buffer 的 H2D 碎片自动合为一条 CE 命令。本机 4090 也不能直接套用 Hopper TMA。[CUDA 12.8 内存 API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html)、[GDRCopy](https://github.com/NVIDIA/gdrcopy)、[Unified Memory](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#unified-memory-programming)

### 本机 4 KiB 对照：API 层的选择为何重要

对 **4096 个独立的 4 KiB host/device allocation，共 16 MiB**，同一 CUDA benchmark 得到：

| 路径                    | CUDA event 区间 | CPU enqueue 墙钟 | 按 event 计算的正文带宽 |
| ----------------------- | --------------: | ---------------: | ----------------------: |
| 逐项 `cudaMemcpyAsync`  |        5.375 ms |         4.572 ms |               3.12 GB/s |
| CUDA Graph memcpy nodes |        5.520 ms |         5.513 ms |               3.04 GB/s |
| `cudaMemcpyBatchAsync`  |        1.543 ms |         0.907 ms |              10.88 GB/s |
| 单个 SM gather kernel   |        0.629 ms |         0.001 ms |              26.67 GB/s |

来源：[scenario 3 原始结果]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_scenario3_reference_results.txt' | relative_url }})。分配／注册、Graph 构建、SM 地址表上传在计时外；event 起点先提交，因此区间可包含 GPU 等待 CPU 供给的空档。这个表说明**同样 16 MiB，表达与提交方式可改变数倍吞吐**，却不能把 0.629 ms 直接当作 SM 端到端成本，也不能从 batch 的 10.88 GB/s 猜定其内部 doorbell 或 CE mode。

同机连续大块 H2D 的 CE 和 SM 均约 **26.7—26.8 GB/s**。raw CE 对同一类独立 4 KiB allocation 做到 **0.631 ms／26.57 GB/s 的设备区间**（见下节），说明 CE 并非天然不能处理 4 KiB copy。CPU BAR 的优化写路径约 **22 GB/s**，但计时止于 CPU `sfence`，不同于 CE 的 GPU 完成端点；标准 GDRCopy 在本机记录中未成功运行，不能把 BAR 原型称作其性能。[大块及重叠实验]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_bench_results.txt' | relative_url }})、[raw CE 结果]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_4k_results.txt' | relative_url }})、[BAR 对照]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_cpu_bar_compare_4k_results.txt' | relative_url }})

## 3. 性能差异的根因：工作供给、在途并发和有效粒度

单个冷请求需要考虑注册／staging、列表构造、提交、启动、正文和完成；流水批次中这些阶段会重叠，不能把各段平均时间机械相加。对稳态吞吐，更有用的判断是：

> **有效带宽受最慢的供给环节限制：CPU 发布 copy 的速率、GPU 消化命令的速率、在途有效字节／请求往返，以及主机、PCIe、显存能提供的带宽。**

### 3.1 CPU 提交跟不上：合并 API 与批量发布有用，但作用不同

本机约 26.6 GB/s 的平台下，一个 4 KiB copy 平均只有 **约 154 ns 的稳态服务预算**。raw CE 逐项发布 4098 次共花 1672 μs，均摊每次完整发布序列约 **408 ns**；若每次只供给一项，供给上界约 `4096 B / 408 ns = 10.0 GB/s`。这与该模式约 10 GB/s 的观察吻合。这里 408 ns 包含 ring、PUT、fence、doorbell 等，**不是门铃寄存器或 PCIe 往返的单独延迟**。[发布原始结果]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_4k_results.txt' | relative_url }})

若每 copy 编码时间为 `e`、一次发布成本为 `p`、每次发布 `k` 项，CPU 供给成本可近似写为 `e + p/k`。提高 `k` 只解决 CPU 供给，不能保证设备端会并行推进；过大批次还会让早到请求等待凑批。CUDA batch 降低应用调用数，但动态 N 项仍需被读取和解释，且公开 API 不保证其内部发布策略。

### 3.2 设备按 copy 等待：减少 doorbell 仍不够

raw CE 在**相同 4096 × 4 KiB 独立 allocation、每轮动态编码**下，只改变后续 copy 的流水属性和发布粒度：

| 执行模式                | 发布次数 | CPU publish |  设备时间戳区间 | 说明                  |
| ----------------------- | -------: | ----------: | --------------: | --------------------- |
| 后续 copy pipelined     |        1 |    0.220 μs |  **631.456 μs** | 约 26.57 GB/s         |
| 后续 copy pipelined     |       66 |   19.977 μs |  **631.808 μs** | 64 项一组，已足够供给 |
| 后续 copy pipelined     |     4098 | 1672.405 μs | **1672.768 μs** | CPU 逐项发布限制进度  |
| 全部 copy non-pipelined |        1 |    0.130 μs | **3179.264 μs** | 一次发布仍约慢 5 倍   |

来源：[raw CE 二因素实验]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_4k_results.txt' | relative_url }})。原型先构建全部命令，再按组发布；表中没有隔离“逐项编码”的影响。多次发布时设备时间戳区间可包含等待 CPU 的空档。结论是：**批量发布解决 CPU 供给，跨 copy 流水解决设备推进；两者都需要，但发布次数降到足够低以后不必强求一次。** 流水机制说明需要独立工作在途，并未测出 4090 的 tag、CE 队列深度或 PCIe RTT。

SM gather 达到平台的机制类似：连续线程访问可合并，多个 warp/block 在远程读取等待期间持续提供独立访存。对同一 4 KiB 场景，grid 从 1、2、4 增至 8 blocks，带宽从 **5.38、10.48、18.02** 增至 **26.54 GB/s**；继续加至 128 blocks 仅到 **26.67 GB/s**。这支持“需要足够并发”，不能由 block 数倒推出真实在途请求数。[SM 并发扫描]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_sm_concurrency_probe_results.txt' | relative_url }})

### 3.3 请求太小：即使提交与流水都解决了，正文效率仍下降

在**同一对连续映射、总量固定 4 MiB、一次发布、后续 copy pipelined**的命令粒度扫描中：

| 每条数据命令 | 数据命令数 | 设备时间戳区间 | 有效正文带宽 |
| -----------: | ---------: | -------------: | -----------: |
|        4 MiB |          1 |     157.632 μs |   26.61 GB/s |
|        1 KiB |       4096 |     162.656 μs |   25.79 GB/s |
|        256 B |      16384 |     216.640 μs |   19.36 GB/s |
|        128 B |      32768 |     367.616 μs |   11.41 GB/s |

来源：[小粒度原始结果]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_ce_launch_sweep_small_results.txt' | relative_url }})。4 KiB 的 4096 条命令相对单条 16 MiB 命令，设备区间仅从 626.304 增至 631.424 μs，表明足够流水时命令数并不线性增加关键路径。[16 MiB 命令数扫描]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_ce_launch_sweep_results.txt' | relative_url }})

到 128 B 时，每条正文在该原型里仍对应 36 B 命令，命令字节／正文约 **28.1%**；这个比例不能直接当作实际 PCIe 协议开销。请求 payload 变小也提高包头比例，并在给定 request/s 或在途窗口下减少有效在途字节。命令获取、CE 内部粒度和 PCIe 请求效率都可能参与失速，**这组数据没有区分各自份额，也没有测 TLP 数**。相反，PCIe Gen4 x16 扣除编码损耗的单向线速约 **31.51 GB/s**，本机大块及 4 KiB 流水平台约 26.6 GB/s；128 B 的 11.41 GB/s 不能仅用链路峰值解释。[PCI-SIG 速率资料](https://pcisig.com/sites/default/files/files/HOTI_PCIe6.0.pdf)

### 3.4 搬运资源和计时端点也会改变“最好”的答案

SM 的可编程性适合不规则 gather 和与解压、变换、计算融合，但它会占用计算资源。一个特定大网格 overlap 实验中，计算单独耗时 8.591 ms、64 MiB copy 单独约 2.50 ms；与 CE 同时运行总时长 8.615 ms，与 SM copy 同时运行 11.066 ms。它说明该配置下 CE 更能与计算共存，**不证明小网格 SM copy 永远无法重叠**。[overlap 原始记录]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_bench_results.txt' | relative_url }})

CPU BAR 写入把搬运负担移到 CPU 和 GPU BAR。该原型用优化 store 约 22 GB/s，期间基本占用一个 CPU 核；CE 的 CPU issue 很短，但原型后续 busy-poll 完成，因而该记录的 CPU active time 也高。轮询是等待策略，不是 CE 搬正文的固有 CPU 成本。比较两路时应统一到“消费者能安全使用数据”的终点，再同时报告 CPU/SM 占用。[BAR 计时审计]({{ '/assets/blog/gpu-io-technical-report/evidence/report_evidence_audit.md' | relative_url }})

## 4. 从根因导出碎片化 I/O 的优化顺序

“碎片”可能指调用次数多、源/目的不连续、物理页分散、每项字节太少，或请求到达时间零散。**先确定碎在哪一层，再决定改变哪一层。**

| 观察到的形态／瓶颈       | 首选动作                                                                  | 需要防止的额外成本                                         |
| ------------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 相邻 src 与 dst 都连续   | 合并为更大线性 copy；去重并复用已驻留数据                                 | 数值相邻但跨 allocation、映射或依赖边界时不可强合并        |
| 固定 pitch、规则多行布局 | 用 2D／3D 表达                                                            | 任意地址数组不能伪装成规则矩形                             |
| 动态任意地址、4 KiB 左右 | 复用 pinned／GPU pool 和地址映射；批量编码与分组发布；让独立 CE copy 流水 | batch 等待、过多队列深度、错误的依赖和完成协议             |
| 128—256 B 极小请求       | 尝试邻近请求合并、适度读放大、CPU pack 后大块传输，或 SM 消费／变换融合   | 打包、scatter、无用字节和额外同步可能超过收益              |
| 需要与计算重叠           | 优先评估 CE；若 SM 能融合工作或小 grid 已够带宽，则测整体任务时间         | CE 仍可竞争显存／主机带宽；SM copy 可能抢计算资源          |
| 请求零散到达或有依赖链   | 按字节、项数、等待时间设置组批阈值；按依赖分组，按完成 handle 回收        | 无限等批、head-of-line blocking、提前覆盖仍在使用的 buffer |

**第一步：减少不必要的数据与描述。** 复用已注册的 host pool、GPU destination pool、地址表和命令内存，避免把 pin/map、分配与初始化放进每个小请求的热路径。只有 `src[i+1] = src[i]+size[i]` 且 `dst[i+1] = dst[i]+size[i]`，并且映射、依赖、生命周期兼容，才可直接合并为线性 copy。规则 stride 更适合 2D：本机 `4 KiB/64 KiB × 4096` 从逐项 API 的 **2.59 GB/s** 提升到 2D 的 **22.04 GB/s**；这不能推广到任意 scatter/gather。[stride 对照]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_bench_results.txt' | relative_url }})

**第二步：把发布和设备流水分别调到“足够”。** 对动态独立 4 KiB 列表，先降低重复 API 与发布开销，再保留正确的跨 copy 流水。原型 64 项一组已与整批一次发布等速，因此继续扩大 batch 主要节省 CPU 工作，未必提高设备吞吐。生产环境应测首批完成和尾延迟，并设置字节、项数、时间阈值；`k=64` 只是这台机器、这个负载的观察值。CUDA batch 批内没有固定顺序保证，有依赖的 copy 应分批或另建依赖。[CUDA batch 契约](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html)

**第三步：在亚 KiB 粒度权衡“多搬字节，少处理请求”。** 若 CPU 先 gather 到连续 pinned staging，再大块 H2D，必要时 GPU scatter，那么打包路线至少要支付 `T_gather + T_bulk-H2D + T_scatter + T_sync`。仅当它小于原路径的总完成时间，打包才有利。另一选择是扩大读取跨度，用少量无用字节换更少命令或请求；应同时报告 `D_useful`、`D_transferred` 和读放大 `D_transferred / D_useful`。对已经约 0.65 ms 完成 16 MiB 的 raw CE，额外 pack 的收益空间很小；对约 1.54 ms 的 CUDA batch，才更值得做端到端实验，不能直接把二者差值当作可节省预算。

**第四步：把 I/O 藏到真实消费者后面。** 预取和双缓冲可让传输与计算重叠，但每个 slot 都要分别记录“本轮输入已就绪”和“上一轮消费者已完成”。CPU 源与命令内存要活到 DMA 完成，GPU 目的不能在消费者结束前覆写。优化目标应是应用的完成时间及 CPU/SM 资源，而非只看孤立搬运 GB/s。

## 5. 组件实现和下一步验证

面向动态碎片，可把接口收敛为：预先建立 host/GPU 映射池与队列；每批验证有效范围和依赖、合法合并、编码或选择 CE／SM／BAR 路径；发布后返回 **completion handle**；消费者满足可见性条件后再回收源、目的、命令和 ring slot。自定义 CE 要额外解决 CUDA allocation 的 VA 权限、stream 前后依赖、多批在途回绕、反压、错误与驱动 ABI 兼容。**raw CE 微基准证明机制可行，不等于现成 CUDA 互操作库。**

验证时统一报告四个端点：① 冷启动与注册；② 每批列表准备、CPU issue wall 和 CPU active；③ 明确起止位置的设备区间；④ 从动态请求产生到消费者可用的端到端时间。CUDA event 区间可含 CPU 供给空档，raw CE 时间戳可含等待发布，BAR 的 CPU fence 不是 GPU-ready 完成；不同端点的数值不能直接相减或宣称严格 speedup。[计时与证据审计]({{ '/assets/blog/gpu-io-technical-report/evidence/report_evidence_audit.md' | relative_url }})

最有价值的后续实验是：在相同布局和 GPU-ready 终点下，测动态地址表、多批在途、小 grid SM、raw CE、CUDA batch、BAR 与真实计算的共存；对 128/256 B 扫合并跨度和读放大；捕获本机 batch 的 GPFIFO/pushbuffer 以确认其实际发布和流水方式。没有命令流与 PCIe 事务观测前，**batch 低于 raw CE 的具体内部原因仍是未知项**。

**结论：** 对本机 4 KiB 碎片，批量发布和跨 copy 流水分别消除了 CPU 供给与设备推进瓶颈，CE 因而接近大块和 SM 的约 26.6 GB/s 平台。到 128 B，命令密度、请求效率与有效在途字节再次成为问题；应根据访问局部性、额外流量和消费者形态决定合并、打包或融合，而不是继续只减少 API 次数。

### 本地证据入口

- [CUDA 场景与规则 stride]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_bench_results.txt' | relative_url }})、[独立 4 KiB 参考结果]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_scenario3_reference_results.txt' | relative_url }})
- [raw CE 二因素实验]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_raw_ce_4k_results.txt' | relative_url }})、[命令数扫描]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_ce_launch_sweep_results.txt' | relative_url }})、[亚 KiB 扫描]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_ce_launch_sweep_small_results.txt' | relative_url }})
- [SM 并发]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_sm_concurrency_probe_results.txt' | relative_url }})、[CPU BAR 对照]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_cpu_bar_compare_4k_results.txt' | relative_url }})、[GDRCopy 探测]({{ '/assets/blog/gpu-io-technical-report/evidence/h2d_gdrcopy_probe_results.txt' | relative_url }})
- [证据与计时口径审计]({{ '/assets/blog/gpu-io-technical-report/evidence/report_evidence_audit.md' | relative_url }})、[来源核查]({{ '/assets/blog/gpu-io-technical-report/evidence/report_source_notes.md' | relative_url }})
