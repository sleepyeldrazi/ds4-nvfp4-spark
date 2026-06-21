/* Correctness test for the turbo4 FP8 KV pack/unpack.
 *
 * Self-contained: includes the device functions directly.
 * Verifies pack→unpack round-trip against original FP32 values.
 *
 * Layout: nope(448 e4m3) + scales(7 e8m0) + pad(1) + rot(64 BF16) = 584 bytes
 * The 1-byte padding ensures BF16 section starts at even offset (456) for
 * correct alignment on GB10 (sm_121a).
 *
 * Build: nvcc -O2 -arch=native -o tests/test_turbo4 tests/test_turbo4.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

#define TURBO4_BLOCK 64
#define HEAD_DIM 512
#define N_ROT 64
#define N_NOPE (HEAD_DIM - N_ROT)

static __host__ __device__ uint64_t turbo4_row_bytes(uint32_t hd, uint32_t nr) {
    uint32_t n_nope = hd - nr;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    uint64_t total = (uint64_t)n_nope + n_blocks + 1 + (uint64_t)nr * 2;  /* +1 padding */
    return (total + 7ull) & ~7ull;
}

__global__ void turbo4_pack_kernel(uint8_t *dst, const float *src, uint32_t n_rows,
                                    uint32_t head_dim, uint32_t n_rot)
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
    uint8_t *rot_out = scale_out + n_blocks + 1;  /* +1: skip padding byte */

    __shared__ float s_amax[32];
    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        uint32_t base = blk * TURBO4_BLOCK;
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
        __nv_fp8_e8m0 scale_e8m0(block_amax);
        __nv_fp8_storage_t scale_byte = *reinterpret_cast<__nv_fp8_storage_t*>(&scale_e8m0);
        if (tid == 0) scale_out[blk] = scale_byte;
        float scale_val = (float)scale_e8m0;
        if (scale_val == 0.0f) scale_val = 1.0f;
        for (uint32_t i = tid; i < TURBO4_BLOCK && base + i < n_nope; i += blockDim.x) {
            float v = sr[base + i] / scale_val;
            __nv_fp8_e4m3 q(v);
            nope_out[base + i] = *reinterpret_cast<__nv_fp8_storage_t*>(&q);
        }
        __syncthreads();
    }
    /* Pack BF16 via uint16_t pointer — correct byte encoding on sm_121a */
    for (uint32_t i = tid; i < n_rot; i += blockDim.x) {
        __nv_bfloat16 bf = __float2bfloat16(sr[n_nope + i]);
        uint16_t *p = (uint16_t *)(rot_out + (uint64_t)i * 2);
        p[0] = __bfloat16_as_ushort(bf);
    }
}

__global__ void turbo4_unpack_kernel(float *dst, const uint8_t *src, uint32_t n_rows,
                                      uint32_t head_dim, uint32_t n_rot)
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
    const uint8_t *rot_in = scale_in + n_blocks + 1;  /* +1: skip padding byte */

    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        uint32_t base = blk * TURBO4_BLOCK;
        __nv_fp8_storage_t scale_byte = scale_in[blk];
        __nv_fp8_e8m0 scale_e8m0 = *reinterpret_cast<__nv_fp8_e8m0*>(&scale_byte);
        float scale_val = (float)scale_e8m0;
        for (uint32_t i = tid; i < TURBO4_BLOCK && base + i < n_nope; i += blockDim.x) {
            __nv_fp8_storage_t q_byte = nope_in[base + i];
            __nv_fp8_e4m3 q = *reinterpret_cast<__nv_fp8_e4m3*>(&q_byte);
            dr[base + i] = (float)q * scale_val;
        }
        __syncthreads();
    }
    /* Read BF16 via uint16_t pointer — correct on sm_121a */
    for (uint32_t i = tid; i < n_rot; i += blockDim.x) {
        uint16_t raw = ((const uint16_t *)rot_in)[(uint64_t)i];
        dr[n_nope + i] = __bfloat162float(__ushort_as_bfloat16(raw));
    }
}

__device__ float turbo4_packed_elem(const uint8_t *row_ptr, uint32_t dim,
                                     uint32_t head_dim, uint32_t n_rot) {
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    if (dim < n_nope) {
        const uint8_t *nope = row_ptr;
        const uint8_t *scale = row_ptr + n_nope;
        uint32_t blk = dim / TURBO4_BLOCK;
        __nv_fp8_storage_t scale_byte = scale[blk];
        __nv_fp8_e8m0 sv_e8m0 = *reinterpret_cast<__nv_fp8_e8m0*>(&scale_byte);
        float sv = (float)sv_e8m0;
        __nv_fp8_storage_t q_byte = nope[dim];
        __nv_fp8_e4m3 q = *reinterpret_cast<__nv_fp8_e4m3*>(&q_byte);
        return (float)q * sv;
    } else {
        const uint8_t *rot = row_ptr + n_nope + n_blocks + 1;  /* +1: skip padding byte */
        uint16_t raw = ((const uint16_t *)rot)[(uint64_t)(dim - n_nope)];
        return __bfloat162float(__ushort_as_bfloat16(raw));
    }
}

__global__ void verify_elem_kernel(float *out, const uint8_t *packed,
                                    uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t dim = threadIdx.x;
    const uint8_t *row_ptr = packed + (uint64_t)row * turbo4_row_bytes(head_dim, n_rot);
    out[(uint64_t)row * head_dim + dim] = turbo4_packed_elem(row_ptr, dim, head_dim, n_rot);
}

#define CK(call) do { \
    cudaError_t _e = call; \
    if (_e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return 1; } \
} while(0)

int main() {
    const uint32_t n_rows = 64;
    const uint64_t fp32_rb = (uint64_t)HEAD_DIM * sizeof(float);
    const uint64_t pk_rb = turbo4_row_bytes(HEAD_DIM, N_ROT);

    printf("=== Turbo4 FP8 KV Pack/Unpack Test ===\n");
    printf("FP32 row:  %lu B | Packed: %lu B | Ratio: %.2fx\n\n",
           (unsigned long)fp32_rb, (unsigned long)pk_rb, (double)fp32_rb / pk_rb);

    /* Generate test data with realistic KV values (small magnitude) */
    float *h_src = (float *)malloc((size_t)n_rows * fp32_rb);
    unsigned seed = 42;
    for (uint64_t i = 0; i < (uint64_t)n_rows * HEAD_DIM; i++) {
        h_src[i] = ((float)((seed = seed * 1103515245u + 12345u) & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    }

    float *d_src, *d_unpk, *d_elem; uint8_t *d_pk;
    CK(cudaMalloc(&d_src, (size_t)n_rows * fp32_rb));
    CK(cudaMalloc(&d_pk, (size_t)n_rows * pk_rb));
    CK(cudaMalloc(&d_unpk, (size_t)n_rows * fp32_rb));
    CK(cudaMalloc(&d_elem, (size_t)n_rows * fp32_rb));
    CK(cudaMemcpy(d_src, h_src, (size_t)n_rows * fp32_rb, cudaMemcpyHostToDevice));

    bool ok = true;

    /* Test 1: round-trip pack→unpack */
    printf("[1] Round-trip...\n");
    turbo4_pack_kernel<<<n_rows, 64>>>(d_pk, d_src, n_rows, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    turbo4_unpack_kernel<<<n_rows, 64>>>(d_unpk, d_pk, n_rows, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    float *h_unpk = (float *)malloc((size_t)n_rows * fp32_rb);
    CK(cudaMemcpy(h_unpk, d_unpk, (size_t)n_rows * fp32_rb, cudaMemcpyDeviceToHost));

    double max_rn = 0, max_rr = 0; int fn = 0, fr = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            float o = h_src[r * HEAD_DIM + d], g = h_unpk[r * HEAD_DIM + d];
            if (d < N_NOPE) {
                if (fabsf(o) > 1e-6f) {
                    double rel = fabs((double)(g - o) / o);
                    if (rel > max_rn) max_rn = rel;
                    if (rel > 0.15) fn++;
                }
            } else {
                double ae = fabs((double)(g - o));
                double rel = fabsf(o) > 1e-6f ? fabs((double)(g - o) / o) : ae;
                if (rel > max_rr) max_rr = rel;
                if (ae > 0.01 && fabsf(o) < 10.0f) fr++;
            }
        }
    }
    printf("  nope max_rel=%.4f (%d>15%%) | rot max_rel=%.4f (%d>0.01)\n", max_rn, fn, max_rr, fr);
    bool rt_ok = (fn == 0 && fr == 0);
    printf("  %s\n", rt_ok ? "PASS" : "FAIL");
    ok &= rt_ok;

    /* Test 2: element-level match (element access == full unpack) */
    printf("\n[2] Element-level unpack...\n");
    verify_elem_kernel<<<n_rows, HEAD_DIM>>>(d_elem, d_pk, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float *h_elem = (float *)malloc((size_t)n_rows * fp32_rb);
    CK(cudaMemcpy(h_elem, d_elem, (size_t)n_rows * fp32_rb, cudaMemcpyDeviceToHost));
    int mm = 0;
    for (uint64_t i = 0; i < (uint64_t)n_rows * HEAD_DIM; i++) {
        if (h_unpk[i] != h_elem[i] && !(isnan(h_unpk[i]) && isnan(h_elem[i]))) mm++;
    }
    printf("  mismatches: %d %s\n", mm, mm == 0 ? "PASS" : "FAIL");
    ok &= (mm == 0);

    /* Test 3: all-zero input */
    printf("\n[3] All-zero...\n");
    float *d_zs, *d_zu; uint8_t *d_zp;
    CK(cudaMalloc(&d_zs, fp32_rb)); CK(cudaMalloc(&d_zp, pk_rb)); CK(cudaMalloc(&d_zu, fp32_rb));
    CK(cudaMemset(d_zs, 0, fp32_rb));
    turbo4_pack_kernel<<<1, 64>>>(d_zp, d_zs, 1, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    turbo4_unpack_kernel<<<1, 64>>>(d_zu, d_zp, 1, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float hz[HEAD_DIM]; CK(cudaMemcpy(hz, d_zu, fp32_rb, cudaMemcpyDeviceToHost));
    int zn = 0; for (int i = 0; i < HEAD_DIM; i++) if (hz[i] != 0.0f) zn++;
    printf("  non-zero: %d %s\n", zn, zn == 0 ? "PASS" : "FAIL");
    ok &= (zn == 0);
    CK(cudaFree(d_zs)); CK(cudaFree(d_zp)); CK(cudaFree(d_zu));

    /* Test 4: large values */
    printf("\n[4] Large values...\n");
    float *d_bs, *d_bu; uint8_t *d_bp;
    CK(cudaMalloc(&d_bs, fp32_rb)); CK(cudaMalloc(&d_bp, pk_rb)); CK(cudaMalloc(&d_bu, fp32_rb));
    float hb[HEAD_DIM];
    for (uint32_t i = 0; i < HEAD_DIM; i++) hb[i] = (i < N_NOPE) ? 100.0f : 50.0f;
    CK(cudaMemcpy(d_bs, hb, fp32_rb, cudaMemcpyHostToDevice));
    turbo4_pack_kernel<<<1, 64>>>(d_bp, d_bs, 1, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    turbo4_unpack_kernel<<<1, 64>>>(d_bu, d_bp, 1, HEAD_DIM, N_ROT);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float hu[HEAD_DIM]; CK(cudaMemcpy(hu, d_bu, fp32_rb, cudaMemcpyDeviceToHost));
    double bm = 0;
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        double r = fabs((double)(hu[i] - hb[i]) / hb[i]);
        if (r > bm) bm = r;
    }
    printf("  max_rel=%.4f %s\n", bm, bm < 0.15 ? "PASS" : "FAIL");
    ok &= (bm < 0.15);
    CK(cudaFree(d_bs)); CK(cudaFree(d_bp)); CK(cudaFree(d_bu));

    printf("\n=== %s ===\n", ok ? "ALL PASSED" : "SOME FAILED");

    free(h_src); free(h_unpk); free(h_elem);
    CK(cudaFree(d_src)); CK(cudaFree(d_pk)); CK(cudaFree(d_unpk)); CK(cudaFree(d_elem));
    return ok ? 0 : 1;
}
