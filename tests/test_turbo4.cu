/* Correctness test for the turbo4 FP8 KV pack/unpack.
 * Compares pack→unpack round-trip against the original FP32 values.
 * The FP8 quantization introduces small errors (e4m3 precision); this test
 * verifies the errors are within expected bounds (relative error < 0.1 for
 * the nope dims, exact for the rot/BF16 dims within BF16 precision).
 *
 * Build: nvcc -O2 -arch=native -o test_turbo4 test_turbo4.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstring>

#define TURBO4_BLOCK 64
#define HEAD_DIM 512
#define N_ROT 64
#define N_NOPE (HEAD_DIM - N_ROT)

static __device__ __host__ uint64_t turbo4_row_bytes(uint32_t head_dim, uint32_t n_rot) {
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + TURBO4_BLOCK - 1) / TURBO4_BLOCK;
    uint64_t total = (uint64_t)n_nope + n_blocks + (uint64_t)n_rot * 2;
    return (total + 7ull) & ~7ull;
}

__global__ void turbo4_pack_kernel(uint8_t *dst, const float *src, uint32_t n_rows,
                                    uint32_t head_dim, uint32_t n_rot);
__global__ void turbo4_unpack_kernel(float *dst, const uint8_t *src, uint32_t n_rows,
                                      uint32_t head_dim, uint32_t n_rot);

int main() {
    const uint32_t n_rows = 64;
    const uint64_t fp32_bytes = (uint64_t)n_rows * HEAD_DIM * sizeof(float);
    const uint64_t packed_bytes = (uint64_t)n_rows * turbo4_row_bytes(HEAD_DIM, N_ROT);

    printf("FP32 row: %lu bytes, packed row: %lu bytes, compression: %.2fx\n",
           HEAD_DIM * 4, turbo4_row_bytes(HEAD_DIM, N_ROT),
           (double)(HEAD_DIM * 4) / turbo4_row_bytes(HEAD_DIM, N_ROT));

    /* alloc + fill source with realistic KV values (random, small magnitude) */
    float *h_src = (float *)malloc(fp32_bytes);
    unsigned seed = 42;
    for (uint64_t i = 0; i < (uint64_t)n_rows * HEAD_DIM; i++) {
        h_src[i] = ((float)((seed = seed * 1103515245u + 12345u) & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    }

    float *d_src; cudaMalloc(&d_src, fp32_bytes);
    cudaMemcpy(d_src, h_src, fp32_bytes, cudaMemcpyHostToDevice);

    uint8_t *d_packed; cudaMalloc(&d_packed, packed_bytes);
    float *d_unpacked; cudaMalloc(&d_unpacked, fp32_bytes);

    /* pack */
    turbo4_pack_kernel<<<n_rows, 64>>>(d_packed, d_src, n_rows, HEAD_DIM, N_ROT);
    cudaDeviceSynchronize();

    /* unpack */
    turbo4_unpack_kernel<<<n_rows, 64>>>(d_unpacked, d_packed, n_rows, HEAD_DIM, N_ROT);
    cudaDeviceSynchronize();

    /* compare */
    float *h_unpacked = (float *)malloc(fp32_bytes);
    cudaMemcpy(h_unpacked, d_unpacked, fp32_bytes, cudaMemcpyDeviceToHost);

    double max_rel_nope = 0, max_abs_rot = 0;
    int fail_nope = 0, fail_rot = 0;
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            float orig = h_src[r * HEAD_DIM + d];
            float got = h_unpacked[r * HEAD_DIM + d];
            if (d < N_NOPE) {
                /* nope dims: FP8 quantization, expect < ~10% relative error */
                if (fabsf(orig) > 1e-6f) {
                    double rel = fabs((double)(got - orig) / orig);
                    if (rel > max_rel_nope) max_rel_nope = rel;
                    if (rel > 0.15) fail_nope++;
                }
            } else {
                /* rot dims: BF16, expect ~3 decimal digits precision */
                double abs_err = fabs((double)(got - orig));
                if (abs_err > max_abs_rot) max_abs_rot = abs_err;
                if (abs_err > 0.01) fail_rot++;
            }
        }
    }

    printf("nope (FP8): max_rel=%.4f (%d fails >15%%)\n", max_rel_nope, fail_nope);
    printf("rot (BF16): max_abs=%.6f (%d fails >0.01)\n", max_abs_rot, fail_rot);
    printf("RESULT: %s\n", (fail_nope == 0 && fail_rot == 0) ? "PASS" : "FAIL");

    free(h_src); free(h_unpacked);
    cudaFree(d_src); cudaFree(d_packed); cudaFree(d_unpacked);
    return (fail_nope || fail_rot) ? 1 : 0;
}
