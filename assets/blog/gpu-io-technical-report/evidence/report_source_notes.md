# CPU–GPU I/O 报告：协议与内存机制的补充核查

核查日期：2026-09-23。此文件是主报告的来源备忘，只记录可由官方文档支持的结论、推导及表述边界。AMD/Altera PCIe IP 文档用于解释协议和展示一种实现，不用于推定 NVIDIA Ada 的队列深度或微架构。

## 1. 建议优先补入主报告的四处边界

1. **Endpoint 的 completion credit 可宣告为 infinite，物理返回缓冲区仍有限。** 不宜将“credit 用完”统一画成每一种包都在链路上等待远端恢复信用。Requester 还必须在发读请求之前约束在途请求，确保全部 completion 有地方接收。
2. **Pinned 不是当今所有 GPU 访问主机内存的唯一入口。** CUDA 12.8 已描述 HMM/full Unified Memory 对系统分配内存的访问；“GPU 不可访问 malloc/pageable 内存”必须限定传统显式 DMA 及本实验所采用的映射路径。
3. **DMA 是执行方式，BAR 是 CPU 地址窗口，二者不是同一层级的严格反义词。** 适合对比的是“CE 发起主机读请求的 H2D”与“CPU 经 GPU BAR 发起 posted writes 的 H2D”。D2H DMA 本身也可以使用 posted writes。
4. **cp.async/TMA 与 host-side cudaMemcpyAsync 的 CE 路径不是同一种提交机制。** 但也不要写成“cp.async 必然只读 GPU 显存”：global 是地址空间名称，物理后备位置还要看受支持的映射。

## 2. PCIe 吞吐与最小序列化时间

PCI-SIG 官方演讲第 4 页给出 Gen4 的 16 GT/s、128b/130b 编码和 x16 每方向约 252 Gb/s。因此：

\[
B_{\rm Gen4,x16}=16\times10^9\times\frac{128}{130}\times16\div8
=31.507692\times10^9\ \mathrm{B/s}.
\]

这是只扣编码、尚未扣 TLP/DLLP/包头及其他链路占用的**单向上限**，不能把全双工和当成 H2D 可用带宽。来源：[PCI-SIG HOTI PCIe 6.0，历史速率表](https://pcisig.com/sites/default/files/files/HOTI_PCIe6.0.pdf)，PDF 第 4 页（0-based page 3）。

由上式计算的纯 payload 序列化下界：64 B ≈ 2.03 ns，256 B ≈ 8.13 ns，4 KiB ≈ 130 ns，1 MiB ≈ 33.28 μs。这些是**算术下界，不是一次 I/O 延迟或本机测量**；不能用它们推算 doorbell 到执行、PCIe read RTT 或 CPU fence 的延迟。GT/s 是信号传输速率，此处 Gen4 NRZ 每次传输对应一位；不要将这个推导未经修改推广到 PAM4/FLIT 的新代际。

## 3. MRRS、MPS、4 KiB 与 RCB 各管什么

| 名称 | 可使用的准确解释 | 不应声称 |
|---|---|---|
| MRRS | requester 的一条 Memory Read Request 能请求的最大范围 | 每个请求实际一定达到 MRRS |
| MPS | 数据 TLP 的 payload 上限；会约束写包和返回 completion 的数据量 | 等于 MRRS；决定一次逻辑 copy 大小 |
| 4 KiB boundary | 常规 Memory Request 的地址/长度不得跨 4 KiB 边界；较大描述符需拆请求 | 使用 huge page 后就能发跨 4 KiB 的普通读包 |
| RCB | split completion 的切分位置边界；首尾允许对应原始范围的非整块边缘 | completion 永远只有 64 B 或 128 B |

AMD QDMA 的 H2C 章节明确说明，描述符按 MRRS 和 PCIe 读请求不得跨 4 KB 的要求拆分，completion 收到后再生成卡内写。这是直接支持“一个 DMA 描述符可以变成多条 PCIe 读请求”的官方例子：[H2C MM Engine](https://docs.amd.com/r/4.0-English/pg302-qdma/H2C-MM-Engine)。

RCB 可为 64 B 或 128 B；中间 completion 是 RCB 的整数倍，首尾需覆盖原请求首尾。故 RCB=128 B 不排除 256 B completion：[Read Completion Boundary](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Read-Completion-Boundary)。MPS 的配置值受链路设备能力限制：[Maximum Payload Size](https://docs.amd.com/r/en-US/pg054-7series-pcie/Maximum-Payload-Size?contentId=bk7Ii9HejzQsRtgoAUe6IQ)。

**来源修复：** 工作区旧报告引用的 Intel 托管 PCIe Base Specification Rev 2.1 PDF，本次打开已重定向到 404。主报告不宜将该链接作为唯一支撑。上面的 AMD 官方页面目前可读取。

## 4. Tag、credit、completion 缓冲区是三个约束

Tag 用于将返回 completion 和在途请求关联。一个读请求可以由多个 completion 满足，不能收第一个包就认定整个请求结束。AMD 的 PG213 实现公开了 Split Completion Table、tag 分配、资源满时回压，以及最终 completion 被接受后复用 tag。这能证明这种功能机制存在，但文档的 **256 项/8-bit tag 是该 IP 实现，不能套给 RTX 4090**。[Tag Management for Non-Posted Transactions](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Tag-Management-for-Non-Posted-Transactions?contentId=slPuC0tfGu5Tk4J911YhiQ)

传统 PCIe 信用分六池：Posted / Non-Posted / Completion 各自分 Header 和 Data。未缩放的一个 data credit 对应 16 B，header credit 对应一个 TLP header。它描述接收能力，不是 CUDA stream 数量，也不是同一种“在途事务编号”。[Flow Control Credit Information](https://docs.amd.com/r/en-US/pg054-7series-pcie/Flow-Control-Credit-Information?contentId=CcCfrLfJKx06_5rTkuDOPg)

**关键细节：** 官方 FPGA 文档明确说 Endpoint receiver 宣告无限 completion credits，同时 requester 必须自行限制发读量，预留接收 completion 的空间。[Tracking Non-Posted Requests and Inbound Completions](https://docs.amd.com/r/en-US/pg054-7series-pcie/Tracking-Non-Posted-Requests-and-Inbound-Completions)。新 Altera F-Tile 文档同样指出 infinite credit 不意味着缓冲无限，应用须依据返回缓冲容量管理 outstanding requests：[Completion Buffer Size](https://docs.altera.com/r/docs/683140/25.3/f-tile-avalon-streaming-ip-for-pci-express-user-guide/completion-buffer-size?contentId=Yj0TLWLMMIx~OMfnI_Yurg)。

适合在主报告使用的性能模型（**推导，不是 NVIDIA 官方参数**）：

\[
B_{\rm payload}\lesssim\min\left(B_{\rm link}\eta,\ \frac{W_{\rm outstanding}}{L_{\rm roundtrip}},\ B_{\rm host},\ B_{\rm dst}\right).
\]

只有在请求近似等长且 tag 是主要限制时，才可进一步写成 \(W_{\rm outstanding}\approx N_{\rm active}\bar R\)。保持链路满载所需的窗口约为带宽延迟积；小请求耗用同样一个 tag，却带回更少 payload。信用、tag、返回缓存和上游发请求能力可能分别约束窗口。不要未经测量就把某个吞吐台阶归因到一个具体隐藏 FIFO。

## 5. 读响应、posted write 和软件完成不是一回事

Memory Read 是 non-posted，Memory Write 是 posted；事务层 completion 是前者的响应。官方类型表可引用：[Altera F-Tile RX Flow Control](https://docs.altera.com/r/docs/683140/25.3/f-tile-avalon-streaming-ip-for-pci-express-user-guide/rx-flow-control?contentId=aI9sz623FThjbTA0I1sFEA)。

据此，主报告宜分出三个时刻：CPU 已提交；设备数据搬运及规定可见性已完成；应用所需的后继执行条件已满足。CPU 的 store/fence 返回、一次 read completion、GPU 写回完成标记各有范围，不能拿同一个“完成”词直接等价。BAR write 没有一条专属事务层 completion，不能凭 CPU 写入循环计时宣称 GPU 消费者已看到完整数据。Linux MMIO 的安全寄存器 readback 可以说明 posted-write 排空，但不能单靠通用 readback 说明 NVIDIA L2 与运行中 kernel 的全部一致性语义；后者应按 GDRCopy/GPUDirect 和具体同步协议说明。

## 6. Pinned、Unified Memory、NUMA：建议采用的限定表述

传统显式 CE DMA 路径会提前建立可稳定访问的页及设备映射；注册和锁页应摊销。NUMA 位置是另一维：锁页不会自动证明页面靠近 GPU 的 Root Port。Linux 默认偏向触发分配的 CPU 所在节点，也可由任务/VMA memory policy 控制；因此分配与 first touch、CPU affinity、GPU PCIe 拓扑应一起看。[Linux NUMA Memory Policy](https://www.kernel.org/doc/html/v6.15/admin-guide/mm/numa_memory_policy.html)、[What is NUMA](https://docs.kernel.org/mm/numa.html)。

CUDA 的官方建议包括复用 pinned buffers、合并小传输，并关注 AutoNUMA 对 GPU 应用的影响。[CUDA Best Practices，Pinned Memory / NUMA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。这支持方向，不提供本机注册一次要多少 μs 的常数。

CUDA 12.8 的 Unified Memory 章节区分 managed-only 与 full UM，后者包括符合条件的 Linux HMM。必须查询 `pageableMemoryAccess` 等属性。页面可迁移，也可能远程映射；`cudaMemPrefetchAsync` 将部分开销提前，并不消灭搬运。稀疏访问可能放大页面迁移，反复 CPU/GPU ownership 切换可能导致额外 fault/页表维护。不要将 4 KiB 固定写成所有系统的迁移单位。来源：[CUDA 12.8 Programming Guide，Unified Memory](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#unified-memory-programming)。

## 7. cp.async、TMA 与 H2D CE 的边界

`cp.async` 是 GPU 指令级 global→shared 异步拷贝；它的完成由相应 device-side 等待机制处理。TMA 在 Hopper 扩展为 1D–5D tensor 的 global↔shared 及 cluster 内共享内存传输，并避免用通用寄存器搬每个元素。它们用于 kernel 内流水，并不是把 host-side `cudaMemcpyAsync` 换个名字。[CUDA 12.8 Programming Guide，Asynchronous Data Copies](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#asynchronous-data-copies)、[Hopper Tuning Guide，Tensor Memory Accelerator](https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html#tensor-memory-accelerator)

应将“global↔shared 地址空间”与“HBM/GDDR↔主机 DRAM 物理位置”分开。主报告可说：本实验中 SM gather 是 kernel 经可访问的 host mapping 拉取数据；使用 cp.async 是否获益需额外测量，它不会把碎片自动变为一条 CE DMA，也不会绕过 PCIe。TMA 是否支持某类 host mapping、tensor descriptor 及平台组合应按该 API/硬件能力确认，不能凭名字作普遍承诺。

## 8. 数量级表的标注建议

- 有原始数据：写“本机、模式、尺寸、计时范围、是否含同步”。
- 链路算术：写“理论下界”，同时说明被省略的协议和往返开销。
- 回归斜率：写“该工作负载的摊销增量”，不要写成“硬件固定启动延迟”。
- 无本机测量的 CPU cache / DRAM / IOMMU / doorbell / PCIe RTT / 缺页迁移：只写机制和依赖因素，或明确为假设示例；不加入无来源的 ns/μs 常数。

已有工作区报告对“PBDMA≠CE”“UVA≠已经建立映射”“一个 doorbell≠一个 copy”区分较好。最应补足的是 infinite completion credit 的例外、HMM 的可访问性边界，以及 BAR/CE 计时完成语义不可直接对等。
