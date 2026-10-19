// Grouped EXL3 linears, 1-128 rows: up to GMAX independent layers in two launches, whatever their shapes and plans (the
// projections of one input, the slices of one attention output, or a single layer). rot_many_kernel rotates every
// layer's input rows (each its own suh; the input any of fp16 / bf16 / fp32 with any row stride) into fp16 rows, as
// rot_in does; glinear_kernel then runs every layer's blocks in one grid, each block exactly linear_kernel's block for
// its layer (its plan's K splits and warps, K ranges, mma order, warp-order and split-order sums), writing the layer's
// output rows with any row stride. So each output has the bits of that layer's own rot_in + linear launches, whatever
// rows and layers share the launch (mma keeps rows apart; nothing is summed across rows or layers).
//
// Both kernels are programmatic dependent launches where the GPU has them (sm_90+): each may start while the kernel
// before it finishes, and reads nothing but weights (glinear: its first trellis step) until that kernel is done.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "linear_common.cuh"

using namespace tf_exl3;

namespace {

constexpr int GMAX = 8;                 // layers a launch

struct GLayer {
    const half* xh;                     // rotated input rows [M, K] fp16, row stride ldx elements
    const uint32_t* T;                  // trellis words
    const half* svh;                    // fp16 [N]
    const half* bias;                   // fp16 [N] or null
    void* y;                            // output rows [M, N], row stride ldy elements
    float* Z;                           // fp32 [SK, M, N] when SK > 1
    int* counters;                      // int32 [8 * N / 128], left zero
    long long ldx, ldy, stride_k, stride_nb;
    int y_dtype, K, N, SK, first;       // first: the layer's first block in the grid
    // the next layer's input rotation folded in (or rsuh null): its rotated rows rxh[row * ldr + roff + col] =
    // rot128_in(y as stored, rsuh[roff + col]), the bits rot_many makes of y
    const half* rsuh;
    half* rxh;
    long long ldr;
    int roff;
    // RoPE folded in (or rcos null): the outputs' last rrd columns of every rhd-column head rotated in adjacent pairs
    // by the row's position rpos[row], as rope_heads does to the stored bf16 values (its compiled fma order)
    const float* rcos;
    const float* rsin;
    const long long* rpos;
    int rhd, rrd;
};

struct GArgs {
    GLayer l[GMAX];
    int n, M;
    int apf;                            // experiment 13: the cp.async walk's A fragments loaded a k step ahead
    int discard;                        // the split-K partials leave L2 unwritten once summed (discard.global.L2)
};

struct RLayer {
    const void* x;                      // input rows [M, K], row stride ldx elements
    const half* suh;                    // fp16 [K]
    half* xh;                           // rotated rows [M, K] fp16, row stride K
    long long ldx;
    int x_dtype, K, first;              // first: the layer's first warp
};

struct RArgs {
    RLayer l[GMAX];
    int n, M;
};

// Programmatic dependent launch (sm_90+): wait() returns once the kernel before is done and its writes are visible;
// dependents() lets the next such launch start. Both are no-ops in a launch made without it.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void gl_cp_async16(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void gl_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void gl_cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

__device__ __forceinline__ void pdl_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// rot_in for every layer of a group: a warp a (layer, row, 128-wide block), rot_in's arithmetic (rot128_in).
__global__ void __launch_bounds__(128) rot_many_kernel(const __grid_constant__ RArgs args) {
    pdl_dependents();                                         // the group's linears may start: they read weights first
    pdl_wait();
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int li = 0;
#pragma unroll
    for (int q = 1; q < GMAX; ++q)
        if (q < args.n && w >= args.l[q].first) li = q;
    const RLayer& L = args.l[li];
    const int nblk = L.K >> 7, local = w - L.first;
    if (local >= args.M * nblk) return;                       // past the last layer's rows (whole warps)
    const int row = local / nblk, k = (local % nblk) * 128 + 4 * lane;
    float v[4], s[4];
    load4(L.x, L.x_dtype, (size_t)row * L.ldx + k, v);
    load4(L.suh, F16, k, s);
    rot128_in(v, s, lane);
    store4(L.xh, F16, (size_t)row * L.K + k, v);
}

// The finished outputs of one row's 128 columns (4 a lane), as stored in y, rotated into the next layer's input rows:
// what rot_many computes from y (y's rounding, then rot128_in with the next layer's suh), from registers.
__device__ __forceinline__ void rot_out(const GLayer& L, const float (&v)[4], int y_dtype, int lane, size_t row,
                                        int col0) {
    float u[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        u[j] = y_dtype == BF16 ? __bfloat162float(__float2bfloat16_rn(v[j]))
             : y_dtype == F16 ? __half2float(__float2half_rn(v[j])) : v[j];
    const int c = L.roff + col0 + 4 * lane;
    load4(L.rsuh, F16, c, s);
    rot128_in(u, s, lane);
    store4(L.rxh, F16, row * L.ldr + c, u);
}

// rope_heads (forward) of one row's 128 stored columns (4 a lane) when they hold a head's rotary part: v rounded as
// stored (bf16), then re = fma(xe, cs, -(xo sn)), im = fma(xe, sn, xo cs) on each adjacent pair (rope_heads' order)
__device__ __forceinline__ void rope_out(const GLayer& L, float (&v)[4], int y_dtype, int lane, size_t row, int col0) {
    const int d0 = (col0 % L.rhd) + 4 * lane, lo = L.rhd - L.rrd;
    if (d0 < lo) return;
    const long long p = L.rpos[row];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int j = (d0 + 2 * q - lo) >> 1;
        const float cs = L.rcos[p * (L.rrd >> 1) + j], sn = L.rsin[p * (L.rrd >> 1) + j];
        const float xe = __bfloat162float(__float2bfloat16_rn(v[2 * q]));
        const float xo = __bfloat162float(__float2bfloat16_rn(v[2 * q + 1]));
        v[2 * q] = __fmaf_rn(xe, cs, -__fmul_rn(xo, sn));
        v[2 * q + 1] = __fmaf_rn(xe, sn, __fmul_rn(xo, cs));
    }
}

// linear_kernel for a block of any layer of the group (blocks of layer i: first_i .. first_i + N_i / 128 * SK_i - 1,
// block b of them linear_kernel's (b % (N_i / 128), b / (N_i / 128))).
// (4 warps: at most 170 registers, three programs an SM as linear_kernel gets, which nvcc otherwise gives up for the
// table's registers)
//
// Experiment 12 (H200): NS > 1 splits each k warp's 8 column tiles over NS warps (warp = kw * NS + ns takes tiles
// ns * 8 / NS .. (ns + 1) * 8 / NS - 1 of k warp kw's K range). Every output column's chain is still one warp's walk over
// the same k tiles in the same order with the same decode and mma, and the k warps' partials are still added in k-warp
// order (then the splits in split order), so every output keeps its bits; the block has NS times the warps, each with
// a 1 / NS share of the accumulators and step words (the TP2 H200 decode linears ran 1 block an SM at 8 warps).
template <int WK, int NS>
struct GBounds {
    static constexpr int threads = WK * NS * 32;
    static constexpr int min_blocks = NS == 1 ? (WK == 4 ? 3 : 1) : (NS == 2 && WK == 4 ? 2 : 1);
};

// Experiment 13 TN (TF_EXL3_GLINEAR_TN, the cp.async walk, at most 8 rows): the decoded weights as the mma's A (m = a
// tile's 16 columns) and the rows as B (n = 8 rows), one m16n8k16 a tile instead of two (the expert kernels' TN: every
// (row, column) output the same 16 products in the same k order; acc[j][0] holds (column g / g + 8, rows 2t, 2t + 1)),
// the rows' B fragments a k step ahead.
template <int K2, int CB, int WK, int NS, int ST, bool TN = false>
__global__ void __launch_bounds__(GBounds<WK, NS>::threads, GBounds<WK, NS>::min_blocks)
glinear_kernel(const __grid_constant__ GArgs args) {
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    constexpr int NT = 8 / NS;                                // column tiles a warp
    static_assert(8 % NS == 0, "NS divides the block's 8 column tiles");
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int M = args.M;
    const int RH = min(M, 8);                                 // rows of red a k warp

    int li = 0;                                               // this block's layer
#pragma unroll
    for (int q = 1; q < GMAX; ++q)
        if (q < args.n && (int)blockIdx.x >= args.l[q].first) li = q;
    const GLayer& L = args.l[li];
    const half* xh = L.xh;
    const half* svh = L.svh;
    const half* bias = L.bias;
    void* y = L.y;
    float* Z = L.Z;
    int* counters = L.counters;
    const long long ldx = L.ldx, ldy = L.ldy, stride_k = L.stride_k;
    const int y_dtype = L.y_dtype, K = L.K, N = L.N, SK = L.SK;
    const int NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kw = warp / NS, ns = warp % NS;                 // this warp's k chain and its share of the tiles
    const int j0 = ns * NT;                                   // its first column tile
    constexpr int NW = WK * NS;                               // warps a block
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    // the epilogue's tables (weights, read last): svh, the bias and the folded rotation's suh lines into L2 now (no
    // data moved, nothing computed)
    if (threadIdx.x < 2 && (SK == 1 || split == SK - 1)) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(svh + nb * 128 + threadIdx.x * 64));
        if (bias) asm volatile("prefetch.global.L2 [%0];" ::"l"(bias + nb * 128 + threadIdx.x * 64));
        if (L.rsuh) asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rsuh + L.roff + nb * 128 + threadIdx.x * 64));
    }
    const int kt0 = split * (per_warp * WK) + kw * per_warp;
    const uint32_t* tiles = L.T + nb * L.stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[NT][2][4];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * ldx;
        const half* x1 = xh + (size_t)r1 * ldx;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        if constexpr (ST == 0) {
            // up to 6 bits the next k step's words (this warp's tiles of it) are loaded while this one is decoded; 7 and
            // 8 bits load per tile
            constexpr bool PF = K2 <= 12;
            constexpr int SR = NS == 1 ? step_regs<K2>()
                                       : (step_shuffled<K2>() ? (NT * TW + 31) / 32 : NT * LW);
            uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
            auto load_part = [&](const uint32_t* step, auto& raw) {
                if constexpr (NS == 1) {
                    load_step<K2>(step, lane, raw);
                } else if constexpr (step_shuffled<K2>()) {
                    static_assert(NS == 1 || (NT * TW) % 32 == 0, "a warp's tiles fill whole 32-word chunks");
    #pragma unroll
                    for (int c = 0; c < SR; ++c) raw[c] = __ldg(step + j0 * TW + c * 32 + lane);
                } else {
                    int word, offset;
                    lane_start<K2>(lane, word, offset);
    #pragma unroll
                    for (int j = 0; j < NT; ++j)
    #pragma unroll
                        for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + (j0 + j) * TW + (word + q) % TW);
                }
            };
            if constexpr (PF) load_part(tile, cur);
            if (m0 == 0) {
                pdl_wait();                              // the first weights are on their way; the rows are ready now
                pdl_dependents();
                if (L.rcos && threadIdx.x < M && threadIdx.x < 16) {   // the folded RoPE's rows of the tables, into L2
                    const long long p = L.rpos[threadIdx.x];
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rcos + p * (L.rrd >> 1)));
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rsin + p * (L.rrd >> 1)));
                }
            }
    #pragma unroll 1
            for (int i = 0; i < per_warp; ++i) {
                const int kt = kt0 + i;
                uint32_t a[4];
                a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
                a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
                a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
                a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
                if constexpr (PF) {
                    if (i + 1 < per_warp) load_part(tile + (size_t)(i + 1) * stride_k, nxt);
                }
    #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    uint32_t w[LW];
                    if constexpr (PF) {
                        if constexpr (NS == 1) step_lane_words<K2>(cur, j, lane, prev, w);
                        else if constexpr (step_shuffled<K2>()) {
                            // step_lane_words' shuffle on this warp's chunks (its tiles start a whole chunk in)
                            constexpr int LPW = 8 / K2;
                            const uint32_t r = cur[j * TW / 32];
                            const int base = (j * TW) % 32;
                            const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
                            const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
                            w[0] = prev ? before : own;
                            w[1] = prev ? own : 0u;
                        } else {
    #pragma unroll
                            for (int q = 0; q < LW; ++q) w[q] = cur[j * LW + q];
                        }
                    } else {
                        ldg_lane_words<K2>(tile + (size_t)i * stride_k + (j0 + j) * TW, lane, w);
                    }
                    uint32_t b0[2], b1[2];
                    decode_lane<K2, CB>(w, lane, b0, b1);
                    mma16816(acc[j][0], a, b0);
                    mma16816(acc[j][1], a, b1);
                }
                if constexpr (PF) {
    #pragma unroll
                    for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
                }
            }
        } else {
            // experiment 12 (TF_EXL3_GLINEAR_STAGES): this warp's tiles of each k step copied 16 bytes a lane (cp.async)
            // ST - 1 steps ahead into its own ring of ST shared-memory stages (the start of red, which the reduction
            // reuses after the walk), each lane then reading back the words load_step / ldg_lane_words would hand it:
            // the same words, decode, mma and order, so the same bits as the register-prefetch walk
            constexpr int SWW = NT * TW;                      // words a warp's stage
            constexpr int NV = SWW / 4;                       // its 16-byte chunks
            constexpr int VPL = (NV + 31) / 32;
            uint32_t* ring = reinterpret_cast<uint32_t*>(red) + (size_t)warp * ST * SWW;
            auto issue = [&](int it) {
                uint32_t* dst = ring + (it % ST) * SWW;
                const uint32_t* src = tile + (size_t)it * stride_k + j0 * TW;
#pragma unroll
                for (int v = 0; v < VPL; ++v) {
                    const int c = v * 32 + lane;
                    if ((NV % 32) == 0 || c < NV) gl_cp_async16(dst + 4 * c, src + 4 * c);
                }
            };
#pragma unroll
            for (int d = 0; d < ST - 1; ++d) {
                if (d < per_warp) issue(d);
                gl_cp_async_commit();
            }
            if (m0 == 0) {
                pdl_wait();                          // the first weights are on their way; the rows are ready now
                pdl_dependents();
                if (L.rcos && threadIdx.x < M && threadIdx.x < 16) {
                    const long long p = L.rpos[threadIdx.x];
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rcos + p * (L.rrd >> 1)));
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(L.rsin + p * (L.rrd >> 1)));
                }
            }
            int word = 0, offset = 0;
            lane_start<K2>(lane, word, offset);
            if constexpr (TN) {
                uint32_t bn[2] = {__ldg(reinterpret_cast<const uint32_t*>(x0 + kt0 * 16 + 2 * t)),
                                  __ldg(reinterpret_cast<const uint32_t*>(x0 + kt0 * 16 + 2 * t + 8))};
#pragma unroll 1
                for (int i = 0; i < per_warp; ++i) {
                    const int kt = kt0 + i;
                    const uint32_t bx[2] = {bn[0], bn[1]};
                    if (i + 1 < per_warp) {
                        bn[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + (kt + 1) * 16 + 2 * t));
                        bn[1] = __ldg(reinterpret_cast<const uint32_t*>(x0 + (kt + 1) * 16 + 2 * t + 8));
                    }
                    if (i + ST - 1 < per_warp) issue(i + ST - 1);
                    gl_cp_async_commit();
                    gl_cp_async_wait<ST - 1>();
                    __syncwarp();
                    const uint32_t* st = ring + (i % ST) * SWW;
                    uint32_t wl[NT][LW];
                    if constexpr (step_shuffled<K2>()) {
                        constexpr int CH = (NT * TW + 31) / 32;
                        constexpr int LPW = 8 / K2;
                        uint32_t raw[CH];
#pragma unroll
                        for (int c = 0; c < CH; ++c) raw[c] = st[c * 32 + lane];
#pragma unroll
                        for (int j = 0; j < NT; ++j) {
                            const uint32_t r = raw[j * TW / 32];
                            const int base = (j * TW) % 32;
                            const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
                            const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
                            wl[j][0] = prev ? before : own;
                            wl[j][1] = prev ? own : 0u;
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < NT; ++j)
#pragma unroll
                            for (int q = 0; q < LW; ++q) wl[j][q] = st[j * TW + (word + q) % TW];
                    }
                    __syncwarp();
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        uint32_t b0[2], b1[2];
                        decode_lane<K2, CB>(wl[j], lane, b0, b1);
                        const uint32_t aw[4] = {b0[0], b1[0], b0[1], b1[1]};
                        mma16816(acc[j][0], aw, bx);
                    }
                }
            } else {
            // (args.apf) the rows' A fragments a k step ahead, as grouped_cp_kernel takes them: the same values
            uint32_t an[4] = {0u, 0u, 0u, 0u};
            if (args.apf) {
                an[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt0 * 16 + 2 * t));
                an[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt0 * 16 + 2 * t));
                an[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt0 * 16 + 2 * t + 8));
                an[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt0 * 16 + 2 * t + 8));
            }
#pragma unroll 1
            for (int i = 0; i < per_warp; ++i) {
                const int kt = kt0 + i;
                uint32_t a[4];
                if (args.apf) {
                    a[0] = an[0]; a[1] = an[1]; a[2] = an[2]; a[3] = an[3];
                    if (i + 1 < per_warp) {
                        an[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + (kt + 1) * 16 + 2 * t));
                        an[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + (kt + 1) * 16 + 2 * t));
                        an[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + (kt + 1) * 16 + 2 * t + 8));
                        an[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + (kt + 1) * 16 + 2 * t + 8));
                    }
                } else {
                    a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
                    a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
                    a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
                    a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
                }
                if (i + ST - 1 < per_warp) issue(i + ST - 1);
                gl_cp_async_commit();
                gl_cp_async_wait<ST - 1>();                   // this step's copies (this lane's) are done
                __syncwarp();                                 // ... and every lane's are visible
                const uint32_t* st = ring + (i % ST) * SWW;
                uint32_t wl[NT][LW];
                if constexpr (step_shuffled<K2>()) {
                    // load_step's chunks (word c * 32 + lane of the warp's tiles), then step_lane_words' shuffle
                    constexpr int CH = (NT * TW + 31) / 32;
                    constexpr int LPW = 8 / K2;
                    uint32_t raw[CH];
#pragma unroll
                    for (int c = 0; c < CH; ++c) raw[c] = st[c * 32 + lane];
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const uint32_t r = raw[j * TW / 32];
                        const int base = (j * TW) % 32;
                        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
                        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
                        wl[j][0] = prev ? before : own;
                        wl[j][1] = prev ? own : 0u;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int q = 0; q < LW; ++q) wl[j][q] = st[j * TW + (word + q) % TW];
                }
                __syncwarp();                                 // the stage is read before a later step refills it
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    uint32_t b0[2], b1[2];
                    decode_lane<K2, CB>(wl[j], lane, b0, b1);
                    mma16816(acc[j][0], a, b0);
                    mma16816(acc[j][1], a, b1);
                }
            }
            }
            gl_cp_async_wait<0>();
        }

        // the k warps' sums, added in k-warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if constexpr (TN) {
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const int col = (j0 + i) * 16 + g;
                    if (2 * t < RH) {
                        red[(kw * RH + 2 * t) * 128 + col] = acc[i][0][0];
                        red[(kw * RH + 2 * t) * 128 + col + 8] = acc[i][0][2];
                    }
                    if (2 * t + 1 < RH) {
                        red[(kw * RH + 2 * t + 1) * 128 + col] = acc[i][0][1];
                        red[(kw * RH + 2 * t + 1) * 128 + col + 8] = acc[i][0][3];
                    }
                }
            } else if (g < RH) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = (j0 + i) * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (kw * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += NW) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    if (L.rcos) rope_out(L, v, y_dtype, lane, (size_t)(m0 + rlo + r), col0);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * ldy + col0 + 4 * lane, v);
                    if (L.rsuh) rot_out(L, v, y_dtype, lane, (size_t)(m0 + rlo + r), col0);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += NW * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += NW) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    if (L.rcos) rope_out(L, v, y_dtype, lane, (size_t)m0 + r, col0);
                    store4(y, y_dtype, ((size_t)m0 + r) * ldy + col0 + 4 * lane, v);
                    if (L.rsuh) rot_out(L, v, y_dtype, lane, (size_t)m0 + r, col0);
                }
                if (args.discard) {
                    // the partials of this block's 128 columns (SK splits x R rows x 512 bytes, 128-byte aligned: N
                    // and the layers' Z offsets are multiples of 128 floats) are dead once summed, and only this
                    // program reads them: their L2 lines are dropped instead of written back to DRAM
                    __syncthreads();                         // every warp has read its rows' partials
                    for (int i = threadIdx.x; i < SK * R * 4; i += NW * 32) {
                        const int q = i / (R * 4), r = (i >> 2) % R, c = (i & 3) * 32;
                        const float* zl = Z + ((size_t)q * M + m0 + r) * N + col0 + c;
                        asm volatile("discard.global.L2 [%0], 128;" ::"l"(zl) : "memory");
                    }
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

int gdtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

// A launch, as a programmatic dependent launch when asked and the GPU has it (sm_90+).
template <typename Kernel, typename Args>
void launch(Kernel kernel, unsigned blocks, unsigned threads, int smem, cudaStream_t stream, bool pdl, const Args& a) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = (size_t)smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl) {
        int dev = 0, major = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
        if (major >= 9) {
            cfg.attrs = attr;
            cfg.numAttrs = 1;
        }
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a));
}

int glinear_ns_env(const char* name) {
    const char* e = std::getenv(name);
    if (e && std::strcmp(e, "auto") == 0) return 0;
    if (e && std::strcmp(e, "auto4") == 0) return -1;      // experiment 13: auto, and 4 where the blocks fit one an SM
    const int v = e && *e ? std::atoi(e) : 1;
    TORCH_CHECK(v == 1 || v == 2, name, " must be 1, 2, auto or auto4");
    return v;
}

int glinear_stages_env() {
    const char* e = std::getenv("TF_EXL3_GLINEAR_STAGES");
    const int v = e && *e ? std::atoi(e) : 0;
    TORCH_CHECK(v == 0 || v == 4, "TF_EXL3_GLINEAR_STAGES must be 0 or 4");
    return v;
}

// launch() with the dynamic shared memory opt-in a kernel needs past 48 KiB (once a kernel)
template <typename Kernel, typename Args>
void launch_big(Kernel kernel, unsigned blocks, unsigned threads, int smem, cudaStream_t stream, bool pdl,
                const Args& a) {
    if (smem > 40 * 1024) {                          // the static smem counts toward the 48 KiB default too
        static std::vector<const void*> done;
        const void* k = reinterpret_cast<const void*>(kernel);
        if (std::find(done.begin(), done.end(), k) == done.end()) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            done.push_back(k);
        }
    }
    launch(kernel, blocks, threads, smem, stream, pdl, a);
}

}  // namespace

int exl3_glinear_max() { return GMAX; }

void exl3_rot_many_cuda(const std::vector<at::Tensor>& xs, const std::vector<at::Tensor>& suhs,
                        const std::vector<at::Tensor>& xhs, bool pdl) {
    RArgs a;
    std::memset(&a, 0, sizeof(a));
    a.n = (int)xs.size();
    a.M = (int)xs[0].size(0);
    int warps = 0;
    for (int i = 0; i < a.n; ++i) {
        RLayer& l = a.l[i];
        l.x = xs[i].data_ptr();
        l.x_dtype = gdtype_of(xs[i]);
        l.ldx = xs[i].stride(0);
        l.suh = reinterpret_cast<const half*>(suhs[i].data_ptr());
        l.xh = reinterpret_cast<half*>(xhs[i].data_ptr());
        l.K = (int)xs[i].size(1);
        l.first = warps;
        warps += a.M * (l.K / 128);
    }
    launch(rot_many_kernel, (unsigned)((warps + 3) / 4), 128, 0, at::cuda::getCurrentCUDAStream(), pdl, a);
}

void exl3_glinear_cuda(const std::vector<at::Tensor>& xhs, const std::vector<at::Tensor>& Ts,
                       const std::vector<int64_t>& stride_k, const std::vector<int64_t>& stride_nb,
                       const std::vector<at::Tensor>& svhs, const std::vector<c10::optional<at::Tensor>>& biases,
                       const std::vector<at::Tensor>& ys, const c10::optional<at::Tensor>& Z,
                       const std::vector<at::Tensor>& counters, const std::vector<int64_t>& SK, int64_t K2,
                       int64_t cb, int64_t WK, bool pdl, const std::vector<c10::optional<at::Tensor>>& rsuhs,
                       const std::vector<c10::optional<at::Tensor>>& rxhs, const std::vector<int64_t>& roffs,
                       const std::vector<int64_t>& ropes, const c10::optional<at::Tensor>& rcos,
                       const c10::optional<at::Tensor>& rsin, const c10::optional<at::Tensor>& rpos, int64_t rhd,
                       int64_t rrd, int64_t discard) {
    const int n = (int)xhs.size();
    GArgs a;
    std::memset(&a, 0, sizeof(a));
    static const int apf = [] { const char* e = std::getenv("TF_EXL3_GLINEAR_APF"); return e && *e ? std::atoi(e) : 0; }();
    a.apf = apf;
    a.n = n;
    a.M = (int)xhs[0].size(0);
    a.discard = (int)discard;
    float* zp = Z ? Z->data_ptr<float>() : nullptr;
    size_t zoff = 0;
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        GLayer& l = a.l[i];
        l.xh = reinterpret_cast<const half*>(xhs[i].data_ptr());
        l.ldx = xhs[i].stride(0);
        l.T = reinterpret_cast<const uint32_t*>(Ts[i].data_ptr());
        l.stride_k = stride_k[i];
        l.stride_nb = stride_nb[i];
        l.svh = reinterpret_cast<const half*>(svhs[i].data_ptr());
        l.bias = biases[i] ? reinterpret_cast<const half*>(biases[i]->data_ptr()) : nullptr;
        l.y = ys[i].data_ptr();
        l.y_dtype = gdtype_of(ys[i]);
        l.ldy = ys[i].stride(0);
        l.K = (int)xhs[i].size(1);
        l.N = (int)ys[i].size(1);
        l.SK = (int)SK[i];
        l.counters = counters[i].data_ptr<int>();
        l.Z = nullptr;
        if (SK[i] > 1) {
            l.Z = zp + zoff;
            zoff += (size_t)SK[i] * a.M * l.N;
        }
        l.first = blocks;
        blocks += (l.N / 128) * l.SK;
        if (i < (int)rsuhs.size() && rsuhs[i] && rxhs[i]) {
            l.rsuh = reinterpret_cast<const half*>(rsuhs[i]->data_ptr());
            l.rxh = reinterpret_cast<half*>(rxhs[i]->data_ptr());
            l.ldr = rxhs[i]->stride(0);
            l.roff = (int)roffs[i];
        }
        if (i < (int)ropes.size() && ropes[i] && rcos && rsin && rpos) {
            l.rcos = rcos->data_ptr<float>();
            l.rsin = rsin->data_ptr<float>();
            l.rpos = reinterpret_cast<const long long*>(rpos->data_ptr<int64_t>());
            l.rhd = (int)rhd;
            l.rrd = (int)rrd;
        }
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const int smem = (int)(WK * std::min(a.M, 8) * 128 * sizeof(float));    // at most 32 KiB: no opt-in
    // experiment 12: TF_EXL3_GLINEAR_NS4 / TF_EXL3_GLINEAR_NS8 (1, 2 or "auto"; default 1, today's kernel): the column
    // tiles of a k warp split over that many warps for the WK = 4 / WK = 8 plans; auto: 2 when the launch's blocks fit
    // two an SM, else 1 (wo_b's 320 blocks ran a second wave at two). TF_EXL3_GLINEAR_STAGES (0 or 4; default 0): the
    // trellis steps through a cp.async ring of that many stages instead of the one-step register prefetch. Neither
    // changes a bit (see glinear_kernel); both are a function of the launch's shape alone.
    static const int ns4 = glinear_ns_env("TF_EXL3_GLINEAR_NS4"), ns8 = glinear_ns_env("TF_EXL3_GLINEAR_NS8");
    static const int stages = glinear_stages_env();
    static const int sms = [] {
        int dev = 0, n = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
        return n;
    }();
    int NSV = WK == 4 ? ns4 : ns8;
    // (auto4: 4 for a WK 4 launch of at most one block an SM and more than one row: R = 6 attn_in 19.4 -> 16.4 us,
    // R = 1 14.7 -> 15.1 us on H200; a column split, so the same bits at any setting)
    if (NSV == -1) NSV = WK == 4 && blocks <= sms && a.M > 1 && cb == 2 && (K2 == 10 || K2 == 12) ? 4 : 0;
    if (NSV == 0) NSV = (WK == 4 ? blocks <= 2 * sms : true) ? 2 : 1;
    // experiment 13: TF_EXL3_GLINEAR_DEEP=1 (with STAGES 4): a launch whose k chains are short (every layer's
    // K / 16 / SK / WK <= 16 steps a warp: the attention projections, not the head's 40) takes a ring of 8 / 12 / 16
    // stages (12 / 16 spilled at NS 2), so seven of a warp's trellis steps are requested at once (before the PDL wait)
    // instead of three ahead: the same words in the same order through the same decode and mma, so the same bits.
    static const int deep = [] { const char* e = std::getenv("TF_EXL3_GLINEAR_DEEP"); return e && *e ? std::atoi(e) : 0; }();
    int maxpw = 0;
    for (int i = 0; i < n; ++i) maxpw = std::max(maxpw, (a.l[i].K / 16) / a.l[i].SK / (int)WK);
    int st_eff = stages;
    if (deep && stages && cb == 2 && (K2 == 10 || K2 == 12) && maxpw <= 16) {
        st_eff = 8;
        if ((long long)WK * st_eff * 128 * K2 > 227 * 1024) st_eff = stages;   // the ring must fit an SM's smem
    }
    const int ring = st_eff ? (int)(WK * st_eff * 32 * (4 * K2)) : 0;     // WK * ST * 8 tiles * TW words * 4 bytes
    const int dsmem = std::max(smem, ring);
#define TF_LAUNCH_ONE(K2_, CB_, WK_, NS_, ST_)                                                                    \
    launch_big(glinear_kernel<K2_, CB_, WK_, NS_, ST_>, (unsigned)blocks, (unsigned)(WK_ * NS_ * 32), dsmem, stream, \
               pdl, a)
#define TF_LAUNCH_DEEP(K2_, ST_)                                                                                   \
    if (WK == 4) { if (NSV == 2) TF_LAUNCH_ONE(K2_, 2, 4, 2, ST_); else TF_LAUNCH_ONE(K2_, 2, 4, 1, ST_); }       \
    else { if (NSV == 2) TF_LAUNCH_ONE(K2_, 2, 8, 2, ST_); else TF_LAUNCH_ONE(K2_, 2, 8, 1, ST_); }
    if (st_eff > 4 && cb == 2 && (K2 == 10 || K2 == 12)) {
        if (K2 == 10) { TF_LAUNCH_DEEP(10, 8) }
        else { TF_LAUNCH_DEEP(12, 8) }
        return;
    }
#undef TF_LAUNCH_DEEP
    static const int tn = [] { const char* e = std::getenv("TF_EXL3_GLINEAR_TN"); return e && *e ? std::atoi(e) : 0; }();
    if (tn && st_eff == 4 && cb == 2 && a.M <= 8 && (K2 == 10 || K2 == 12)) {
#define TF_LAUNCH_TN(K2_)                                                                                          \
    if (WK == 4) {                                                                                                \
        if (NSV == 4) launch_big(glinear_kernel<K2_, 2, 4, 4, 4, true>, (unsigned)blocks, 512u, dsmem, stream, pdl, a); \
        else if (NSV == 2) launch_big(glinear_kernel<K2_, 2, 4, 2, 4, true>, (unsigned)blocks, 256u, dsmem, stream, pdl, a); \
        else launch_big(glinear_kernel<K2_, 2, 4, 1, 4, true>, (unsigned)blocks, 128u, dsmem, stream, pdl, a);     \
    } else {                                                                                                      \
        if (NSV == 2) launch_big(glinear_kernel<K2_, 2, 8, 2, 4, true>, (unsigned)blocks, 512u, dsmem, stream, pdl, a); \
        else launch_big(glinear_kernel<K2_, 2, 8, 1, 4, true>, (unsigned)blocks, 256u, dsmem, stream, pdl, a);     \
    }
        if (K2 == 10) { TF_LAUNCH_TN(10) }
        else { TF_LAUNCH_TN(12) }
#undef TF_LAUNCH_TN
        return;
    }
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        if (stages) {                                                                                           \
            if (WK == 4) { if (NSV == 2) TF_LAUNCH_ONE(K2_, CB_, 4, 2, 4); else TF_LAUNCH_ONE(K2_, CB_, 4, 1, 4); } \
            else { if (NSV == 2) TF_LAUNCH_ONE(K2_, CB_, 8, 2, 4); else TF_LAUNCH_ONE(K2_, CB_, 8, 1, 4); }      \
        } else {                                                                                                \
            if (WK == 4) { if (NSV == 2) TF_LAUNCH_ONE(K2_, CB_, 4, 2, 0); else TF_LAUNCH_ONE(K2_, CB_, 4, 1, 0); } \
            else { if (NSV == 2) TF_LAUNCH_ONE(K2_, CB_, 8, 2, 0); else TF_LAUNCH_ONE(K2_, CB_, 8, 1, 0); }      \
        }                                                                                                       \
        return;                                                                                                 \
    }
    if (NSV == 4) {                                   // (auto4: WK 4, cb 2, K2 10 / 12 only)
        TORCH_CHECK(stages == 4, "TF_EXL3_GLINEAR_NS4=auto4 takes TF_EXL3_GLINEAR_STAGES=4");
        if (K2 == 10) TF_LAUNCH_ONE(10, 2, 4, 4, 4);
        else TF_LAUNCH_ONE(12, 2, 4, 4, 4);
        return;
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
#undef TF_LAUNCH_ONE
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}
