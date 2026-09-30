# CPU–GPU I/O 工作区证据审计

日期：2026-09-23。范围：只核验现有源码和原始结果，不重新运行 GPU。以下路径均相对 `/home/lyc/inference/cuda_val`；数字是现有运行记录，不能视为跨机器常数。

## 1. 最值得保留的实测结论

| 结论 | 原始证据 | 支持范围 |
|---|---|---|
| 大块连续 pinned H2D 的 CE 与 mapped-host SM copy 都能达到约 26.8 GB/s | `h2d_bench_results.txt:12–17`：16–256 MiB，CE 26.71–26.81、SM 26.65–26.83 GB/s | RTX 4090；带宽为十进制 GB/s；这里的 SM copy 会把 host 数据真正写进 device allocation，并非应用直接零拷贝消费 |
| 4096 × 4 KiB 独立 buffer 的常规 API 热路径明显慢于单个 SM gather kernel | `h2d_bench_results.txt:34–37`：CE 5.345 ms、Graph 6.321 ms、Batch 1.541 ms、SM 0.629 ms | CUDA event 区间；均排除注册分配、Graph 构建和 SM 指针表上传；不是全应用端到端 |
| 规则 stride 适合 2D API | `h2d_bench_results.txt:21–26`：4 KiB/64 KiB × 4096，逐次 CE 2.59、CE2D 22.04、SM 26.68 GB/s；64 KiB/1 MiB × 256，CE2D 26.72 | 2D 布局有明确 affine/stride 规则，不能推广到任意独立源/目的地址 |
| CE 性能问题并非不可突破的碎片硬件上限 | `h2d_raw_ce_4k_results.txt:3`：4096 独立分配，1 门铃、动态编码、GPU 631.456 µs / 26.569 GB/s，CPU total 648.870 µs / 25.856 GB/s | tinygrad NVKIface 帮助创建对象、映射和队列；热路径 C++ 动态生成 CE 命令；不是 CUDA 公共接口的性能保证 |
| 仅合并 doorbell 不足以解决 non-pipelined 执行代价 | `h2d_raw_ce_4k_results.txt:3,6`：同一批、同 1 门铃，pipelined 631.456 µs，all non-pipelined 3179.264 µs | 只改变传输类型位；表明设备推进语义影响显著；未测出具体内部 queue depth/tag 数 |
| 发布过碎也会降低 throughput | `h2d_raw_ce_4k_results.txt:3–5`：1 / 4098 / 66 门铃的 GPU 时间 631.456 / 1672.768 / 631.808 µs；CPU publish 0.220 / 1672.405 / 19.977 µs | group=1 包含起始和结尾 2 次附加发布；每次发布还包含 ring、GPPut 和 fence，不能称为 doorbell 寄存器单独延迟 |
| 增加 launch 数在 4 KiB 处不必显著掉速，极小块仍会失速 | `h2d_ce_launch_sweep_results.txt:3–7`：同 16 MiB/同 mappings，1、256、4096、16384 data launch 得到 26.788、26.790、26.570、25.902 GB/s；`h2d_ce_launch_sweep_small_results.txt:3–6`：同 4 MiB，256 B 19.361、128 B 11.409 GB/s | 该实验是连续区域人为切块，不是把任意地址合成一条 copy；不能把 launch 数当 TLP 数 |
| SM 并发足够时能接近链路平台值 | `h2d_sm_concurrency_probe_results.txt:5,10,7,12,9`：1/2/4/8/16 blocks 为 5.381/10.476/18.024/26.541/26.642 GB/s | 256 threads/block，4 KiB chunks，地址表提前上传；8 blocks 已足够，但未观测 block 实际放到哪些 SM，更未测 PCIe 在途请求数 |
| CPU BAR SIMD/NT 写能达到约 22 GB/s | `h2d_cpu_bar_compare_4k_results.txt:8–15`：libc 11.110，AVX2/AVX512 21.862，MOVDIR64B 21.995，raw CE 25.911 GB/s | same fragmented allocations；这是 tinygrad 的 CPU VRAM mapping 数据通路，不是标准 GDRCopy 测试；BAR 与 CE 的完成口径不同，见下文 |
| 示例 full-grid SM copy 与计算抢资源，而 CE 可与该计算重叠 | `h2d_bench_results.txt:51–56`：compute 8.591 ms；CE 2.503，SM 2.502；合并 CE 8.615，合并 SM 11.066 | 特定 spin kernel + 大网格 copy；不证明 SM copy 一定无法重叠，也不证明 CE 永远不干扰 DRAM/缓存/功耗 |

## 2. 计时边界必须在报告中明确

### CUDA benchmark

- `h2d_bench.cu:128–140`：记录 GPU event 起点，然后 CPU 调用 fn，再记录终点，最后 stream sync。event elapsed 包含 GPU 等 CPU 继续提交的空档，不能称为「CE 净执行时间」。`enq` 是 fn 的 CPU 墙钟，不是 CPU process/thread 活跃时间；API 阻塞或排队反压都可能在其中。
- `h2d_bench.cu:466–491`：所有 host/device allocations、初始化，以及 SM 的两个指针表上传在计时外。
- `h2d_bench.cu:514–526`：Graph 有 N 个互相无依赖的 memcpy 节点，实例化在计时外；这里是 memcpy fan-out 图。结果并不支持「CUDA Graph 一定大幅降低这个场景的 CPU 提交成本」，因为实测 Graph enq 为 6.313 ms，超过逐次 CE 的 4.519 ms（`h2d_bench_results.txt:34–35`）。源码 446–447 的概述是期待而不是测量结论。
- `h2d_bench.cu:77–79`：偶数样本「中位数」取排序后的上中位元素，不是两中间值平均。初始结果 10 次，后续诊断多为 15 次。
- `h2d_bench.cu:528–546`：只在计时前验证一次；各方法依次写同一份期望数据，未为每条方法重置目标，弱于后续每轮 poison/rotate 的诊断验证。不要宣称原始 benchmark 每次 timed iteration 都完整校验。

### raw CE

- `h2d_raw_ce_probe.cpp:70–121`：CPU total 从 C 函数内计时点开始，包含参数检查、动态命令编码、publish、busy poll 完成；不含 Python/FFI 外层、alloc/map、输入地址数组构建、数据准备、验证。
- 同文件 `:77–96`：每个 payload copy 9 个 32-bit words = 36 B；三条 semaphore-only launch 共 72 B。所以命令字节 `36N+72`，4096 时为 147528 B；数据 launch=N，全部 `LAUNCH_DMA` 次数=N+3。
- 同文件 `:78,93–95,117–121`：GPU 时间是首尾 CE timestamp semaphore 的差；末尾另一个 release 用于明确 CPU 可读到终点 timestamp。不是 host total 减 CPU submit 的简单等价物。
- 同文件 `:89–91`：pipelined 模式实际上是第一条 NON_PIPELINED，后续 PIPELINED；每条数据 copy 不附加 semaphore/flush，末尾统一 release/flush。不能把这个模式简单描述成所有命令均 pipelined。
- `h2d_raw_ce_probe.py:63–69` 确認 source/destination 是每个 chunk 单独 allocation；`:93–105` 每轮 rotate source pairing、全量验证在计时外，所有列分别取中位数。因此表中 `median(build)+median(publish)` 不必等于 `median(submit)`。
- `h2d_raw_ce_probe.cpp:52–61` publish 包含 ring entry store → sfence → GPPut → sfence → doorbell；`:96` 还有命令可见性 mfence。4098 次 publish 约 1.67 ms，可称「本实现重复发布的均摊约 0.4 µs」，不能报告成纯 MMIO/PCIe 往返固定值。

### CPU BAR/GDRCopy

- `h2d_raw_ce_probe.cpp:149–181`：BAR wall 时间到最后 `_mm_sfence()` 和 CPU clock 采样；不是 GPU-side kernel/CE 确認已消费数据的 timestamp。
- `h2d_cpu_bar_compare.py:130–136`：每轮 BAR 写之后立即 CE D2H 到独立 pinned host memory 并逐字节校验；这证明后续 GPU readback 能读对，但验证不在 BAR timer 内。请用「CPU BAR 热路径带宽」而不是与 raw CE 完成时间完全同义的「GPU-ready 带宽」。
- `h2d_cpu_bar_compare.py:103–128`：CE `cpu_issue` 用 C++ submit wall；CE `cpu_active` 用 Python thread CPU time 跨越 whole C call 的 busy-poll wait。所以 CE ~1 核是轮询策略，并非 DMA 搬数据需要 1 核。BAR `cpu_issue` 与 `cpu_active` 均采用 thread/process CPU time，各路径列定义略有差别。
- `h2d_raw_ce_probe.cpp:185–221`：BAR 多线程含每批创建/销毁线程；CPU time 为全 process 汇总。未采用常驻线程池、NUMA binding 等。2/4/8 线程未加速只支持当前实现/主机观察，不是所有 CPU BAR 路径的硬件定理。
- `h2d_cpu_bar_compare_bulk_results.txt:2`：1×16 MiB case 的多线程只有一 worker 搬运，因为按逻辑 copy 分配任务；不能用该多线程 bulk 表证明大块并行拆分无效。
- `h2d_gdrcopy_probe_results.txt:2,8`：标准 libgdrapi 路径完全跳过；0.097 ms 的 `pin_map_ms` 是加载失败尝试时长，不是 GDRCopy 注册成本。
- `h2d_gdrcopy_compare.cu:176–182`：该替代 CUDA/SM control 使用单个 destination pool、stride=8192，不同于 CPU BAR compare 的 4096 独立目标 allocation。
- `h2d_gdrcopy_compare.cu:241–277`：SM control 计时包含动态 source-list 构建、32 KiB pointer-table H2D 和 blocking event completion；`h2d_gdrcopy_probe_results.txt:11` 的 680.140 µs/24.667 GB/s 是较完整 CPU wall-to-ready 口径，不能硬说比初始 26.66 GB/s 的 SM 性能下降。

## 3. launch / 并发 / 干扰的推断边界

- `h2d_ce_launch_sweep.py:88–94`：所有 launch-count cases 共享相同连续 allocation，只改变切分。因此此实验有效隔离命令粒度，但不能展示任意 scatter/gather 的「一条 launch 合并」。
- 同文件 `:46–67` 的解析器数的是命令方法，不是 PCIe Memory Read/TLP、不含在途请求计数。`launch_count`、API_count、doorbell_count、TLP_count 四者必须分开。
- `h2d_ce_launch_sweep_results.txt:3,6`：从 1 个 data launch 到 4096 增量 GPU 时间约 5.12 µs，CPU build 从 0.241 到 8.145 µs。可描述为高度流水后增量成本很小；不能用 5.12µs/4095 得到「每次 CE launch 物理延迟 1.25ns」，因为这只是整条饱和流水的边际成本。
- `h2d_ce_launch_sweep_small_results.txt:6`：128 B copy 时 1179720 B command / 4194304 B payload ≈28.1%；主命令 fetch 也消耗带宽/前端资源。这是 plausible mechanism，不足以断定全部 slowdown 来自 command byte traffic。
- `h2d_sm_concurrency_probe.cu:9–28` 用 grid-stride 分配 chunk；`:61–81` 扫 block 数。这支持增加可并行发起请求的工作能改善效率；未采集 register occupancy、SM 分配、链路 tag、read-completion RTT，不能倒推出 exact outstanding reads。
- `h2d_bench.cu:592–595`：compute `SMcount×8` blocks，SM copy 最多 `SMcount×64` blocks；`646–660` 中先提 compute 再提 copy，无共同 start-event wait。这是有少量起点不对齐的 illustrative overlap 实验；先后次序和占用配置会影响结果。8 blocks 满带宽结果提示可再用小网格重测 overlap，但当前没有该实验。

## 4. 不应混为本机实测的材料

- `research/cuda_publish/findings.md:7–16` 明确是外部 libmc 研究的 H100 PCIe/610.43.02 驱动记录，不能与本机 RTX 4090/595.91.07 混合。
- 同文件 `:34–58` 的 12 次 cudaMemcpy 无 per-copy RM ioctl/GSP RPC、205 resolved pushbuffer submissions，是该外部 trace 的观察；trace 插桩时间不适合量性能（`:50–52`）。可支持 steady-state userspace command submission 路径存在，但不能写「所有 CUDA API 永远不 syscall」。
- `h2d_gdrcopy_probe_results.txt:1–4` 的本机能力及 GDRCopy DMA-BUF 13.3+ 条件需由主报告按权威文档核验；本审计能確認源码在缺 library 时跳过，不能由 skipped test 断言所有 RTX 4090/Linux 都绝对不可能 CPU BAR mapping。

## 5. 报告建议使用的开销层次

1. 复用前初始化：context、alloc/pin/map、page table、channel setup；当前 microbench 普遍排除，不能从这些文件赋予一个通用微秒数。
2. 每批 host 数据结构/命令准备：动态 4096 源地址 list 约 4.4 µs（`h2d_cuda_submission_reference_results.txt:3–4`）；raw CE 4096 命令约 8 µs（`h2d_raw_ce_4k_results.txt:3`）；CUDA batch 该 case 会有约 0.9–3.2 ms 的实现/布局/测试间差异，不能只挑一条当常数。
3. 发布：raw 单批约 0.1–0.4 µs 的短计时值；大量重复发布约 0.4 µs/次均摊，需说明 fence/GPPut/doorbell 混合。
4. GPU/control path：现有 event 的单块 4 KiB 约 2–4 µs（`h2d_bench_results.txt:30–33`），包含 event/提交空档；不是纯 PCIe RTT。CE non-pipelined 的额外等待与 pipelined 差别在 device timeline 可见。
5. payload：16 MiB 在26.8 GB/s 理想平台约 626 µs，4 KiB 的纯带宽项约0.153 µs；算式可解释小块容易被控制开销淹没，但不是单块延迟预测。
6. 完成通知：raw host total 比 GPU timestamp 区间大约十几 µs，含 host encode、发布起止偏移、poll观察等；不可把差值统称「interrupt latency」。

附注：另一次 raw run 在 `h2d_raw_ce_probe_results.txt:3–8` 得到 pipelined GPU 631.744 µs、all non-pipelined 3263.136 µs，说明非流水 case 有运行间变化。主报告宜选择同一结果文件内的 matched comparisons。
