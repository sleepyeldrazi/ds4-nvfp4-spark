/*
 * ds4_turbo4.cu — FP8-packed compressed KV cache (packv4 format).
 *
 * Implements the stubbed turbo4 interface (ds4_turbo4_stubs.c) with real
 * FP8-density KV storage matching the DeepSeek-V4 paper:
 *   - non-RoPE dims (n_nope = head_dim - n_rot): e4m3fn, 1 byte each
 *   - per-64-element block scale: e8m0 (1 byte, scale = 2^(k-127))
 *   - RoPE dims (n_rot): BF16, 2 bytes each (kept at full precision per the paper)
 *
 * Packed row layout (packv4):
 *   [ n_nope e4m3 bytes | (n_nope/64) e8m0 scale bytes | 1 pad byte | n_rot BF16 values ]
 *   = 448 + 7 + 1 + 128 = 584 bytes per row (vs 2048 FP32) = 3.5x compression.
 *
 * The 1-byte padding ensures the BF16 section starts at an even byte offset.
 * Without it, rot offset = 448+7 = 455 (odd), causing misaligned __nv_bfloat16
 * loads that crash on GB10 (sm_121a). With padding, rot offset = 456 (even).
 * The 584-byte row stride is already 8-byte aligned (584 = 73×8).
 *
 * Uses CUDA 13's native cuda_fp8.h types (__nv_fp8_e4m3, __nv_fp8_e8m0) for
 * guaranteed-correct, hardware-accelerated FP8 conversion — no hand-rolled
 * bit layout that could drift between CUDA versions.
 *
 * No Hadamard rotation, no PolarQuant, no QJL — plain FP8 KV storage per the
 * DS-V4 paper spec. "turbo4"/"TurboQuant" is ds4's name for the packed-FP8-KV
 * storage format, NOT the Google TurboQuant paper.
 */
#include "ds4_cuda_common.h"

/* ---- format constants ---- */
#define TURBO4_BLOCK 64  /* e4m3 block size for per-block scaling */

/* Packed row layout:
 *   [ n_nope e4m3 bytes | (n_nope/64) e8m0 scale bytes | 1 padding byte | n_rot BF16 values ]
 *   = 448 + 7 + 1 + 128 = 584 bytes per row (vs 2048 FP32) = 3.5x compression.
 *
 * The 1-byte padding ensures the BF16 rot section starts at an even offset within
 * each row. Since row stride (584) is 8-byte aligned, and 448+7+1=456 is even,
 * every __nv_bfloat16 load in the rot section is 2-byte aligned — required on GB10
 * (sm_121a) where misaligned 2-byte loads trigger hardware faults. */
static __host__ __device__ uint64_t turbo4_row_bytes(uint32_t head_dim, uint32_t n_rot) {
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    uint64_t nope_bytes = n_nope;                /* 1 byte per e4m3 */
    uint64_t scale_bytes = n_blocks;             /* 1 e8m0 scale per 64-elem block */
    uint64_t rot_bytes = (uint64_t)n_rot * 2;    /* BF16, 2 bytes each */
    uint64_t total = nope_bytes + scale_bytes + 1 + rot_bytes;  /* +1 padding byte */
    /* align to 8 bytes for clean GPU addressing (584 already divisible by 8) */
    return (total + 7ull) & ~7ull;
}

/* ---- pack kernel: FP32 compressed KV rows -> packed FP8 (e4m3 + e8m0 scale + BF16 rot) ---- */
/* Uses CUDA 13 native __nv_fp8_e4m3 / __nv_fp8_e8m0 for conversion. */
__global__ static void turbo4_pack_kernel(
        uint8_t *dst,            /* packed output, turbo4_row_bytes per row */
        const float *src,        /* FP32 source, head_dim floats per row */
        uint32_t n_rows,
        uint32_t head_dim,
        uint32_t n_rot)
{
    uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    const float *sr = src + (uint64_t)row * head_dim;
    uint8_t *dr = dst + (uint64_t)row * turbo4_row_bytes(head_dim, n_rot);
    uint8_t *nope_out = dr;
    uint8_t *scale_out = dr + n_nope;
    uint8_t *rot_out = scale_out + n_blocks + 1;  /* +1: skip padding byte for BF16 alignment */

    /* Pack non-RoPE dims: per-64-block e4m3 with e8m0 scale */
    __shared__ float s_amax[32];
    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        uint32_t base = blk * TURBO4_BLOCK;
        /* compute block amax */
        float amax = 0.0f;
        for (uint32_t i = tid; i < TURBO4_BLOCK && base + i < n_nope; i += blockDim.x) {
            float v = fabsf(sr[base + i]);
            if (v > amax) amax = v;
        }
        if (tid < 32) s_amax[tid] = amax;
        __syncthreads();
        for (uint32_t s = 16; s > 0; s >>= 1) {
            if (tid < s) { float o = s_amax[tid + s]; if (o > s_amax[tid]) s_amax[tid] = o; }
            __syncthreads();
        }
        float block_amax = s_amax[0];
        /* e8m0 scale: round(amax) to nearest power of 2 (round-to-pos-inf).
         * Use __nv_fp8_e8m0 constructor from float (cudaRoundPosInf, SATFINITE). */
        __nv_fp8_e8m0 scale_e8m0(block_amax);
        __nv_fp8_storage_t scale_byte = *reinterpret_cast<__nv_fp8_storage_t *>(&scale_e8m0);
        if (tid == 0) scale_out[blk] = scale_byte;
        /* decode the scale back to float for quantizing the elements */
        float scale_val = (float)scale_e8m0;  /* __nv_fp8_e8m0 -> float (power of 2) */
        if (scale_val == 0.0f) scale_val = 1.0f;  /* avoid div-by-zero for all-zero blocks */
        /* quantize each element: v/scale -> e4m3 (SATFINITE -> clamps to 448) */
        for (uint32_t i = tid; i < TURBO4_BLOCK && base + i < n_nope; i += blockDim.x) {
            float v = sr[base + i] / scale_val;
            __nv_fp8_e4m3 q(v);  /* SATFINITE, round-to-nearest-even */
            nope_out[base + i] = *reinterpret_cast<__nv_fp8_storage_t *>(&q);
        }
        __syncthreads();
    }

    /* Pack RoPE dims as BF16 (thread per element). Store the BF16 bits via a
     * uint16_t pointer — the rot section is 2-byte aligned (see
     * turbo4_row_bytes), so this is a clean aligned store. */
    for (uint32_t i = tid; i < n_rot; i += blockDim.x) {
        __nv_bfloat16 bf = __float2bfloat16(sr[n_nope + i]);
        uint16_t *p = (uint16_t *)(rot_out + (uint64_t)i * 2);
        p[0] = __bfloat16_as_ushort(bf);
    }
}

/* ---- unpack kernel: packed FP8 -> FP32 (for non-packed attention paths / snapshots) ---- */
__global__ static void turbo4_unpack_kernel(
        float *dst,              /* FP32 output, head_dim floats per row */
        const uint8_t *src,      /* packed source */
        uint32_t n_rows,
        uint32_t head_dim,
        uint32_t n_rot)
{
    uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    float *dr = dst + (uint64_t)row * head_dim;
    const uint8_t *sr = src + (uint64_t)row * turbo4_row_bytes(head_dim, n_rot);
    const uint8_t *nope_in = sr;
    const uint8_t *scale_in = sr + n_nope;
    const uint8_t *rot_in = scale_in + n_blocks + 1;  /* +1: skip padding byte for BF16 alignment */

    /* No shared memory used here — the per-block __syncthreads() is unnecessary
     * (all reads/writes are to global memory with no inter-thread dependency). */
    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        uint32_t base = blk * TURBO4_BLOCK;
        __nv_fp8_storage_t scale_byte = scale_in[blk];
        __nv_fp8_e8m0 scale_e8m0 = *reinterpret_cast<__nv_fp8_e8m0 *>(&scale_byte);
        float scale_val = (float)scale_e8m0;
        for (uint32_t i = tid; i < TURBO4_BLOCK && base + i < n_nope; i += blockDim.x) {
            __nv_fp8_storage_t q_byte = nope_in[base + i];
            __nv_fp8_e4m3 q = *reinterpret_cast<__nv_fp8_e4m3 *>(&q_byte);
            dr[base + i] = (float)q * scale_val;
        }
    }
    for (uint32_t i = tid; i < n_rot; i += blockDim.x) {
        uint16_t raw = ((const uint16_t *)rot_in)[(uint64_t)i];
        dr[n_nope + i] = __bfloat162float(__ushort_as_bfloat16(raw));
    }
}

/* ---- device function: on-the-fly unpack a single element from a packed row ---- */
/* Used by attention kernels that read packed KV directly (no scratch needed). */
__device__ static float turbo4_packed_elem(
        const uint8_t *row_ptr, uint32_t dim, uint32_t head_dim, uint32_t n_rot)
{
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    if (dim < n_nope) {
        const uint8_t *nope = row_ptr;
        const uint8_t *scale = row_ptr + n_nope;
        uint32_t blk = dim / TURBO4_BLOCK;
        __nv_fp8_storage_t scale_byte = scale[blk];
        __nv_fp8_e8m0 scale_e8m0 = *reinterpret_cast<__nv_fp8_e8m0 *>(&scale_byte);
        float sv = (float)scale_e8m0;
        __nv_fp8_storage_t q_byte = nope[dim];
        __nv_fp8_e4m3 q = *reinterpret_cast<__nv_fp8_e4m3 *>(&q_byte);
        return (float)q * sv;
    } else {
        const uint8_t *rot = row_ptr + n_nope + n_blocks + 1;  /* +1: skip padding byte */
        uint16_t raw = ((const uint16_t *)rot)[(uint64_t)(dim - n_nope)];
        return __bfloat162float(__ushort_as_bfloat16(raw));
    }
}

/* ---- public API (replaces ds4_turbo4_stubs.c) ---- */

extern "C" bool ds4_gpu_dsv4_turbo4_packv4_enabled(void) {
    /* Enabled when DS4_KV_TURBO is set. The FP32 path remains default until
     * the attention kernels are fully converted to read packed KV. */
    return getenv("DS4_KV_TURBO") != NULL;
}

extern "C" uint64_t ds4_gpu_dsv4_turbo4_packed_kv_row_bytes(uint32_t hd, uint32_t nr) {
    return turbo4_row_bytes(hd, nr);
}

extern "C" uint64_t ds4_gpu_dsv4_turbo4_packed_kv_bytes(uint32_t nr, uint32_t hd, uint32_t nrot) {
    return (uint64_t)nr * turbo4_row_bytes(hd, nrot);
}

extern "C" int ds4_gpu_dsv4_turbo4_pack_compressed_kv_tensor(
        ds4_gpu_tensor *packed, const ds4_gpu_tensor *src,
        uint32_t dst_row0, uint32_t src_row0, uint32_t n_rows,
        uint32_t head_dim, uint32_t n_rot)
{
    if (!packed || !src || n_rows == 0) return 0;
    uint64_t rb = turbo4_row_bytes(head_dim, n_rot);
    uint8_t *dst = (uint8_t *)packed->ptr + (uint64_t)dst_row0 * rb;
    const float *s = (const float *)src->ptr + (uint64_t)src_row0 * head_dim;
    turbo4_pack_kernel<<<n_rows, 64>>>(dst, s, n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "turbo4_pack_compressed_kv");
}

extern "C" int ds4_gpu_dsv4_turbo4_unpack_compressed_kv_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *packed,
        uint32_t dst_row0, uint32_t src_row0, uint32_t n_rows,
        uint32_t head_dim, uint32_t n_rot)
{
    if (!dst || !packed || n_rows == 0) return 0;
    uint64_t rb = turbo4_row_bytes(head_dim, n_rot);
    float *d = (float *)dst->ptr + (uint64_t)dst_row0 * head_dim;
    const uint8_t *s = (const uint8_t *)packed->ptr + (uint64_t)src_row0 * rb;
    turbo4_unpack_kernel<<<n_rows, 64>>>(d, s, n_rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "turbo4_unpack_compressed_kv");
}

extern "C" int ds4_gpu_dsv4_compressed_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot)
{
    /* In-place FP8 value quantization (the existing fp8_kv_quantize_kernel path).
     * When turbo4 is enabled, the actual storage packing is done by
     * ds4_gpu_dsv4_turbo4_pack_compressed_kv_tensor instead. This function
     * remains a no-op pass-through for the packed path. */
    (void)x; (void)n_tok; (void)head_dim; (void)n_rot;
    return 1;
}
