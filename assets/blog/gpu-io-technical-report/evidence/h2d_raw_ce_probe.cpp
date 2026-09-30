// Dynamic CE submission diagnostic. Queue allocation/mapping comes from the
// pinned tinygrad NVK backend; the timed path below rebuilds every DMA command.
// Command encodings follow NVC6B5 and tinygrad NVCopyQueue/NVQueue.
#include <cstdint>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <thread>
#include <vector>

extern "C" {
struct Queue {
    uint64_t ring_cpu, gpput_cpu, doorbell_cpu, put_cpu;
    uint64_t cmd_cpu, cmd_gpu, signal_cpu, signal_gpu;
    uint32_t entries, token, cmd_capacity;
};
struct Result {
    double build_us, publish_us, submit_us, total_us, gpu_us;
    uint32_t doorbells, command_bytes;
};
struct BarResult {
    double wall_us, cpu_thread_us;
    uint32_t streamed_copies;
};

static uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
static uint64_t thread_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
static uint64_t process_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
static uint32_t method(uint32_t reg, uint32_t count) {
    return (2u << 28) | (count << 16) | (4u << 13) | (reg >> 2);
}
static void semaphore(uint32_t*& p, uint64_t addr, uint32_t value, bool timestamp) {
    *p++ = method(0x240, 3); // SET_SEMAPHORE_A, B, PAYLOAD
    *p++ = uint32_t(addr >> 32);
    *p++ = uint32_t(addr);
    *p++ = value;
    *p++ = method(0x300, 1); // LAUNCH_DMA, no data transfer, SYS flush
    *p++ = 4u | ((timestamp ? 2u : 1u) << 3);
}
static void publish(Queue* q, uint64_t address, uint32_t words) {
    auto ring = reinterpret_cast<volatile uint64_t*>(q->ring_cpu);
    auto put = reinterpret_cast<uint64_t*>(q->put_cpu);
    uint64_t p = *put;
    ring[p % q->entries] = address | (uint64_t(words) << 42) | (1ull << 41);
    *put = p + 1;
    _mm_sfence(); // Publish BAR1 ring entry before BAR1 GPPut.
    *reinterpret_cast<volatile uint32_t*>(q->gpput_cpu) = uint32_t((p + 1) % q->entries);
    _mm_sfence();
    *reinterpret_cast<volatile uint32_t*>(q->doorbell_cpu) = q->token;
}

// Owned, registered buffers only. Caller waits after every batch; no ring reuse
// or command-buffer reuse occurs until the final completion signal is observed.
// group==0 batches the entire command buffer into one GPFIFO entry/doorbell.
int raw_ce_run(Queue* q, const uint64_t* srcs, const uint64_t* dsts,
               const uint32_t* sizes, uint32_t n, uint32_t rotate,
               uint32_t pipelined, uint32_t group, uint32_t sequence, Result* r) {
    const uint64_t start = now_ns();
    if (!n || !sequence || pipelined > 1 || n > 65500) return -1;
    const uint32_t words = 9 * n + 18;
    const uint32_t entries = group ? (n + group - 1) / group + 2 : 1;
    if (words * 4 > q->cmd_capacity || words >= (1u << 21) || entries >= q->entries ||
        q->cmd_gpu + words * 4 >= (1ull << 40) || q->cmd_gpu % 4) return -2;
    for (uint32_t i = 0; i < n; ++i) if (!sizes[i]) return -3;
    uint32_t* p = reinterpret_cast<uint32_t*>(q->cmd_cpu);
    semaphore(p, q->signal_gpu, sequence, true);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t src = srcs[(i + rotate) % n], dst = dsts[i];
        *p++ = method(0x400, 4); // OFFSET_IN_UPPER through OFFSET_OUT_LOWER
        *p++ = uint32_t(src >> 32);
        *p++ = uint32_t(src);
        *p++ = uint32_t(dst >> 32);
        *p++ = uint32_t(dst);
        *p++ = method(0x418, 1); // LINE_LENGTH_IN
        *p++ = sizes[i];
        *p++ = method(0x300, 1);
        // Like the UVM migration path: first non-pipelined, following copies
        // optionally pipelined. Pitch layout; no per-copy flush or semaphore.
        *p++ = (1u << 7) | (1u << 8) | ((pipelined && i) ? 1u : 2u);
    }
    semaphore(p, q->signal_gpu + 16, sequence, true);
    // Separate final signal makes observing the end timestamp unambiguous.
    semaphore(p, q->signal_gpu + 32, sequence, false);
    _mm_mfence(); // Make cached host command writes visible before publishing.
    const uint64_t built = now_ns();
    if (!group) publish(q, q->cmd_gpu, words);
    else {
        publish(q, q->cmd_gpu, 6);
        for (uint32_t i = 0; i < n; i += group) {
            const uint32_t count = n - i < group ? n - i : group;
            publish(q, q->cmd_gpu + (6ull + 9ull * i) * 4, 9 * count);
        }
        publish(q, q->cmd_gpu + (6ull + 9ull * n) * 4, 12);
    }
    _mm_sfence();
    const uint64_t submitted = now_ns();
    auto done = reinterpret_cast<volatile uint32_t*>(q->signal_cpu + 32);
    uint32_t spins = 0;
    while (*done != sequence) {
        _mm_pause();
        if ((++spins & 1023u) == 0 && now_ns() - submitted > 2000000000ull) return -4;
    }
    _mm_mfence();
    const uint64_t end = now_ns();
    const uint64_t gpu_start = *reinterpret_cast<volatile uint64_t*>(q->signal_cpu + 8);
    const uint64_t gpu_end = *reinterpret_cast<volatile uint64_t*>(q->signal_cpu + 24);
    if (!gpu_start || gpu_end <= gpu_start) return -5;
    *r = {(built-start)/1e3, (submitted-built)/1e3, (submitted-start)/1e3,
          (end-start)/1e3, (gpu_end-gpu_start)/1e3, entries, words*4};
    return 0;
}

__attribute__((target("avx2"))) static void copy_avx2_nt(uint8_t* dst, const uint8_t* src,
                                                            uint32_t bytes) {
    for (uint32_t j = 0; j < bytes; j += 32) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + j));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + j), value);
    }
}
__attribute__((target("avx512f"))) static void copy_avx512_nt(uint8_t* dst, const uint8_t* src,
                                                                 uint32_t bytes) {
    for (uint32_t j = 0; j < bytes; j += 64) {
        const __m512i value = _mm512_loadu_si512(reinterpret_cast<const void*>(src + j));
        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + j), value);
    }
}
__attribute__((target("movdir64b"))) static void copy_movdir64b(uint8_t* dst, const uint8_t* src,
                                                                   uint32_t bytes) {
    for (uint32_t j = 0; j < bytes; j += 64) _movdir64b(dst + j, src + j);
}

// CPU-driven H2D through CPU mappings of GPU VRAM. This is the data path used
// by GDRCopy, but it does not call libgdrapi and is reported separately.
// mode 0 uses libc memcpy; 1 uses AVX2 NT; 2 uses AVX-512 NT; 3 uses
// MOVDIR64B when available. The final sfence drains
// CPU write-combining stores before the caller submits a CE readback check.
int cpu_bar_run(const uint64_t* src_cpu, const uint64_t* dst_cpu,
                const uint32_t* sizes, uint32_t n, uint32_t rotate,
                uint32_t mode, BarResult* r) {
    if (!n || !sizes || !r || mode > 3) return -1;
    const uint64_t start = now_ns(), cpu_start = thread_now_ns();
    uint32_t streamed = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const auto* src = reinterpret_cast<const uint8_t*>(src_cpu[(i + rotate) % n]);
        auto* dst = reinterpret_cast<uint8_t*>(dst_cpu[i]);
        const uint32_t bytes = sizes[i];
        if (!src || !dst || !bytes) return -2;
        if (mode == 3 && __builtin_cpu_supports("movdir64b") &&
            (reinterpret_cast<uintptr_t>(dst) & 63u) == 0 && (bytes & 63u) == 0) {
            copy_movdir64b(dst, src, bytes);
            ++streamed;
        }
        else if (mode == 2 && __builtin_cpu_supports("avx512f") &&
                 (reinterpret_cast<uintptr_t>(dst) & 63u) == 0 && (bytes & 63u) == 0) {
            copy_avx512_nt(dst, src, bytes);
            ++streamed;
        }
        else if (mode == 1 && __builtin_cpu_supports("avx2") &&
            (reinterpret_cast<uintptr_t>(dst) & 31u) == 0 && (bytes & 31u) == 0) {
            copy_avx2_nt(dst, src, bytes);
            ++streamed;
        }
        else {
            std::memcpy(dst, src, bytes);
        }
    }
    _mm_sfence();
    const uint64_t cpu_end = thread_now_ns(), end = now_ns();
    *r = {(end - start) / 1e3, (cpu_end - cpu_start) / 1e3, streamed};
    return 0;
}

// Conservative parallel baseline: thread creation and joins are included in
// wall time. CPU time sums all workers. Every worker fences its own WC stores.
int cpu_bar_parallel_run(const uint64_t* src_cpu, const uint64_t* dst_cpu,
                         const uint32_t* sizes, uint32_t n, uint32_t rotate,
                         uint32_t workers, BarResult* r) {
    if (!n || !src_cpu || !dst_cpu || !sizes || !r || workers < 2 || workers > 32) return -1;
    for (uint32_t i = 0; i < n; ++i)
        if (!src_cpu[(i + rotate) % n] || !dst_cpu[i] || !sizes[i]) return -2;
    const uint64_t start = now_ns(), cpu_start = process_now_ns();
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    std::vector<uint32_t> streamed(workers, 0);
    auto run_worker = [&](uint32_t worker) {
        const uint32_t begin = uint64_t(n) * worker / workers;
        const uint32_t end = uint64_t(n) * (worker + 1) / workers;
        for (uint32_t i = begin; i < end; ++i) {
            const auto* src = reinterpret_cast<const uint8_t*>(src_cpu[(i + rotate) % n]);
            auto* dst = reinterpret_cast<uint8_t*>(dst_cpu[i]);
            const uint32_t bytes = sizes[i];
            if (__builtin_cpu_supports("avx2") &&
                (reinterpret_cast<uintptr_t>(dst) & 31u) == 0 && (bytes & 31u) == 0) {
                copy_avx2_nt(dst, src, bytes);
                ++streamed[worker];
            }
            else {
                std::memcpy(dst, src, bytes);
            }
        }
        _mm_sfence();
    };
    for (uint32_t worker = 1; worker < workers; ++worker) threads.emplace_back(run_worker, worker);
    run_worker(0);
    for (auto& thread : threads) thread.join();
    const uint64_t cpu_end = process_now_ns(), end = now_ns();
    uint32_t streamed_total = 0;
    for (uint32_t value : streamed) streamed_total += value;
    *r = {(end - start) / 1e3, (cpu_end - cpu_start) / 1e3, streamed_total};
    return 0;
}

void prepare_owned_buffers(const uint64_t* src_cpu, const uint64_t* dst_cpu,
                            uint32_t n, uint32_t capacity, uint32_t seed) {
    for (uint32_t i = 0; i < n; ++i) {
        auto s = reinterpret_cast<uint32_t*>(src_cpu[i]);
        auto d = reinterpret_cast<volatile uint32_t*>(dst_cpu[i]);
        for (uint32_t j = 0; j < capacity / 4; ++j) {
            s[j] = ((i + 1) * 0x9e3779b1u) ^ (j * 0x85ebca6bu) ^ seed;
            d[j] = 0xa5a5a5a5u;
        }
    }
    _mm_mfence();
}
int verify_owned_buffers(const uint64_t* src_cpu, const uint64_t* dst_cpu,
                          const uint32_t* sizes, uint32_t n, uint32_t rotate) {
    int bad = 0;
    for (uint32_t i = 0; i < n; ++i)
        bad += std::memcmp(reinterpret_cast<const void*>(src_cpu[(i+rotate)%n]),
                           reinterpret_cast<const void*>(dst_cpu[i]), sizes[i]) != 0;
    return bad;
}
}
