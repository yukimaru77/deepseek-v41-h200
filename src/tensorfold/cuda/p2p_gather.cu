// One-shot all-gather of small fp32 rank partials between the GPUs of one host over CUDA peer-to-peer writes
// (tensorfold/cuda/p2p.py; experiment 12, H200 TP2 decode).
//
// Every rank owns a receive ring in its own device memory, [slots][world][max_bytes], and a flag array
// [slots][world][blocks] of sequence numbers. The ranks exchange CUDA IPC handles of both once, so each rank also holds
// mapped pointers into every peer's ring and flags. A gather is ONE kernel of a fixed grid: block b takes chunk b of the
// partial, writes it straight into every peer's ring at (slot, this rank) with plain stores (PCIe peer writes), copies
// it to its own output slice, fences at system scope and releases its flag in every peer's flag array; it then waits
// (acquire, system scope) for every peer's flag of the same (slot, block) at this sequence and copies the peers' chunks
// from its own ring into the output in rank order. The bytes are NCCL's all-gather's (a copy); no host thread, proxy or
// host memory is on the path, so a CUDA graph captures the kernel like any other.
//
// Sequencing: a device word holds the gathers done so far; every block reads it at entry and the last block to finish
// (a counter) advances it, so the next launch (stream order) sees the next sequence. Every rank calls in the same order
// with the same sizes (TensorFold's rule), so the sequences and the chunking agree. Slot reuse is safe at slots >= 2:
// a rank writes slot s of gather q only after its own gather q - 1 saw every peer's flag of q - 1, which each peer
// released only after its kernel of q - 2 (the last reader of slot s) had finished (stream order).
//
// A peer that dies must not leave the others spinning: the waits also read an abort word in mapped pinned host memory
// (every 1024 spins); ``abort`` sets the word (the lane's watchdog: a lost rank, an NCCL error or a step past
// TF_DS_STEP_TIMEOUT), after which every gather on this rank returns at once with garbage, the lane is broken and every
// later step fails (multi.py), and ``failure`` names why. There is no silent timer, as in rdma_gather.cu's collect: a
// wait that gave up on its own would let the step go on with stale partials. Experiment 13: the test-only timer
// (TF_P2P_TIMEOUT_S) is gone too. A gather released by the abort word leaves its output undefined; the lane's
// MultiDecoder checks the watchdog's broken flag after each step's device-to-host read and before it emits any token
// (multi.py, _commit_ok), and the watchdog sets that flag before it raises the abort word, so no such output is emitted.

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int MAX_WORLD = 8;
constexpr int THREADS = 512;

__device__ __forceinline__ uint64_t load_acquire_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_release_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t load_volatile_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

struct Peers {
    char* ring[MAX_WORLD];          // every rank's receive ring as this rank addresses it (own: local pointer)
    uint64_t* flags[MAX_WORLD];     // every rank's flag array as this rank addresses it
};

__global__ void __launch_bounds__(THREADS) p2p_gather_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                                                             int n4, const __grid_constant__ Peers peers, int rank,
                                                             int world, int slots, uint64_t max_bytes, uint64_t* seq,
                                                             unsigned* done, const uint64_t* abort_word,
                                                             unsigned* failed) {
    __shared__ uint64_t s_q;
    __shared__ int s_ok;
    const int G = gridDim.x, b = blockIdx.x;
    if (threadIdx.x == 0) {
        s_q = *seq + 1;
        s_ok = 1;
    }
    __syncthreads();
    const uint64_t q = s_q;
    const int slot = (int)(q % (uint64_t)slots);
    const int per = (n4 + G - 1) / G;
    const int lo = min(n4, b * per), hi = min(n4, lo + per);
    // this block's chunk: into every peer's ring at (slot, rank), and into the output's own slice
    for (int p = 0; p < world; ++p) {
        float4* out = p == rank ? dst + (size_t)rank * n4
                                : reinterpret_cast<float4*>(peers.ring[p] + ((size_t)slot * world + rank) * max_bytes);
        for (int i = lo + threadIdx.x; i < hi; i += THREADS) out[i] = src[i];
    }
    __syncthreads();                                  // every thread's stores are before thread 0's fence
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < world; ++p)
            if (p != rank) store_release_sys(peers.flags[p] + ((size_t)slot * world + rank) * G + b, q);
        // the peers' chunks of the same (slot, block) at this sequence
        for (int p = 0; p < world && s_ok; ++p) {
            if (p == rank) continue;
            const uint64_t* f = peers.flags[rank] + ((size_t)slot * world + p) * G + b;
            for (uint32_t i = 1; load_acquire_sys(f) < q; ++i) {
                if ((i & 1023) == 0 && load_volatile_sys(abort_word)) {
                    s_ok = 0;
                    atomicExch(failed, 1u);
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (s_ok) {
        for (int p = 0; p < world; ++p) {
            if (p == rank) continue;
            const float4* in = reinterpret_cast<const float4*>(peers.ring[rank] + ((size_t)slot * world + p) * max_bytes);
            float4* out = dst + (size_t)p * n4;
            for (int i = lo + threadIdx.x; i < hi; i += THREADS) out[i] = __ldcg(in + i);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(done, 1u) == (unsigned)G - 1) {   // every block read the sequence: advance it
            *done = 0;
            *seq = q;
        }
    }
}

struct Gather {
    int rank = 0, world = 1, slots = 4, blocks = 16;
    uint64_t max_bytes = 0;
    char* ring = nullptr;           // own [slots][world][max_bytes]
    uint64_t* flags = nullptr;      // own [slots][world][blocks]
    uint64_t* seq = nullptr;
    unsigned* done = nullptr;
    unsigned* failed = nullptr;
    uint64_t* abort_host = nullptr; // pinned, mapped
    uint64_t* abort_dev = nullptr;
    Peers peers{};
    std::vector<void*> opened;
    std::string why;
    bool connected = false;
};

std::vector<std::unique_ptr<Gather>> g_all;

Gather& get(int64_t h) {
    TORCH_CHECK(h >= 0 && h < (int64_t)g_all.size() && g_all[h], "bad p2p gather handle");
    return *g_all[h];
}

int env_int(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : dflt;
}

int64_t create(int64_t rank, int64_t world, int64_t max_bytes, int64_t slots, int64_t blocks) {
    TORCH_CHECK(world >= 2 && world <= MAX_WORLD, "p2p gather: world must be 2..", MAX_WORLD);
    TORCH_CHECK(slots >= 2, "p2p gather: at least two slots");
    TORCH_CHECK(max_bytes > 0 && max_bytes % 16 == 0, "p2p gather: max_bytes must be a positive multiple of 16");
    TORCH_CHECK(blocks >= 1 && blocks <= 128, "p2p gather: blocks must be 1..128");
    auto g = std::make_unique<Gather>();
    g->rank = (int)rank;
    g->world = (int)world;
    g->slots = (int)slots;
    g->blocks = (int)blocks;
    g->max_bytes = (uint64_t)max_bytes;
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->ring), (size_t)slots * world * max_bytes));
    const size_t nflags = (size_t)slots * world * blocks;
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->flags), nflags * 8));
    C10_CUDA_CHECK(cudaMemset(g->flags, 0, nflags * 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->seq), 8));
    C10_CUDA_CHECK(cudaMemset(g->seq, 0, 8));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g->done), 8));
    C10_CUDA_CHECK(cudaMemset(g->done, 0, 8));
    g->failed = g->done + 1;
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&g->abort_host), 8, cudaHostAllocMapped));
    *g->abort_host = 0;
    C10_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&g->abort_dev), g->abort_host, 0));
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    g_all.push_back(std::move(g));
    return (int64_t)g_all.size() - 1;
}

// This rank's IPC handles (ring, flags) as bytes, for the store.
py::bytes local_info(int64_t h) {
    Gather& g = get(h);
    cudaIpcMemHandle_t a, b;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&a, g.ring));
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&b, g.flags));
    std::string out(sizeof(a) + sizeof(b), '\0');
    std::memcpy(&out[0], &a, sizeof(a));
    std::memcpy(&out[sizeof(a)], &b, sizeof(b));
    return py::bytes(out);
}

void connect(int64_t h, const std::vector<py::bytes>& infos) {
    Gather& g = get(h);
    TORCH_CHECK((int)infos.size() == g.world, "p2p gather: one info a rank");
    for (int r = 0; r < g.world; ++r) {
        if (r == g.rank) {
            g.peers.ring[r] = g.ring;
            g.peers.flags[r] = g.flags;
            continue;
        }
        const std::string s = infos[r];
        cudaIpcMemHandle_t a, b;
        TORCH_CHECK(s.size() == sizeof(a) + sizeof(b), "p2p gather: bad info from rank ", r);
        std::memcpy(&a, s.data(), sizeof(a));
        std::memcpy(&b, s.data() + sizeof(a), sizeof(b));
        void* pa = nullptr;
        void* pb = nullptr;
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pa, a, cudaIpcMemLazyEnablePeerAccess));
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(&pb, b, cudaIpcMemLazyEnablePeerAccess));
        g.opened.push_back(pa);
        g.opened.push_back(pb);
        g.peers.ring[r] = static_cast<char*>(pa);
        g.peers.flags[r] = static_cast<uint64_t*>(pb);
    }
    g.connected = true;
}

void gather(int64_t h, const at::Tensor& send, const at::Tensor& recv) {
    Gather& g = get(h);
    TORCH_CHECK(g.connected, "p2p gather: not connected");
    TORCH_CHECK(send.is_cuda() && recv.is_cuda() && send.is_contiguous() && recv.is_contiguous(),
                "p2p gather: contiguous CUDA tensors");
    const int64_t bytes = send.numel() * send.element_size();
    TORCH_CHECK(bytes % 16 == 0 && (uint64_t)bytes <= g.max_bytes, "p2p gather: ", bytes, " bytes do not fit");
    TORCH_CHECK(recv.numel() * recv.element_size() == bytes * g.world, "p2p gather: recv must hold world x send");
    const int n4 = (int)(bytes / 16);
    auto stream = at::cuda::getCurrentCUDAStream();
    p2p_gather_kernel<<<g.blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(send.data_ptr()), reinterpret_cast<float4*>(recv.data_ptr()), n4, g.peers,
        g.rank, g.world, g.slots, g.max_bytes, g.seq, g.done, g.abort_dev, g.failed);
    C10_CUDA_CHECK(cudaGetLastError());
}

void abort_all(int64_t h, const std::string& why) {
    Gather& g = get(h);
    if (g.why.empty()) g.why = why.empty() ? std::string("aborted") : why;
    __atomic_store_n(g.abort_host, (uint64_t)1, __ATOMIC_RELEASE);
}

std::string failure(int64_t h) {
    Gather& g = get(h);
    unsigned f = 0;
    C10_CUDA_CHECK(cudaMemcpy(&f, g.failed, sizeof(f), cudaMemcpyDeviceToHost));
    if (!f) return g.why;
    return g.why.empty() ? std::string("a p2p gather wait timed out") : g.why;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create);
    m.def("local_info", &local_info);
    m.def("connect", &connect);
    m.def("gather", &gather);
    m.def("abort", &abort_all);
    m.def("failure", &failure);
}
