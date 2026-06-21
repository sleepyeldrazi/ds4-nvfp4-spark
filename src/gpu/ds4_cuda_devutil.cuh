#include <cuda_pipeline_primitives.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/* Block-wide reduction (sum / max) over a per-thread value using
 * cooperative_groups. Returns the reduced scalar in every thread.
 * blockDim.x must be a multiple of 32 and <= 256 (8 warps).
 * Validated bit-exact against a naive tree reduction in a standalone test. */
__device__ __forceinline__ static float block_sum_f32(float v) {
    __shared__ float scratch;
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<32>(block);
    v = cg::reduce(tile, v, cg::plus<float>());
    const uint32_t lane = tile.thread_rank();
    const uint32_t warp = block.thread_rank() / 32u;
    const uint32_t nwarps = blockDim.x / 32u;
    __shared__ float warp_sums[8];
    if (lane == 0u) warp_sums[warp] = v;
    block.sync();
    if (warp == 0u) {
        float w = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        w = cg::reduce(tile, w, cg::plus<float>());
        if (lane == 0u) scratch = w;
    }
    block.sync();
    return scratch;
}

__device__ __forceinline__ static float block_max_f32(float v) {
    __shared__ float scratch;
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<32>(block);
    v = cg::reduce(tile, v, cg::greater<float>());
    const uint32_t lane = tile.thread_rank();
    const uint32_t warp = block.thread_rank() / 32u;
    const uint32_t nwarps = blockDim.x / 32u;
    __shared__ float warp_sums[8];
    if (lane == 0u) warp_sums[warp] = v;
    block.sync();
    if (warp == 0u) {
        float w = (lane < nwarps) ? warp_sums[lane] : -INFINITY;
        w = cg::reduce(tile, w, cg::greater<float>());
        if (lane == 0u) scratch = w;
    }
    block.sync();
    return scratch;
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}
