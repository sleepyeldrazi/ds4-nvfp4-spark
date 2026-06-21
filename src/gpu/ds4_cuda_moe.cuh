__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* NVFP4 n_tokens==1 decode gate/up kernel. One thread block per (row_chunk, pair).
 * Each thread processes one row, accumulates dot products via dev_dot_nvfp4_q8_K_block,
 * then reduces via quarter_warp_sum_f32. Uses per-expert scale_2 arrays. */
__global__ static void moe_gate_up_mid_decode_lut_nvfp4_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *gate_scale_2,
        const float *up_scale_2) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_nvfp4 *gr = (const cuda_block_nvfp4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_nvfp4 *ur = (const cuda_block_nvfp4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_nvfp4_q8_K_block(gr + b, xqb + b, gate_scale_2[expert]);
            up += dev_dot_nvfp4_q8_K_block(ur + b, xqb + b, up_scale_2[expert]);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[256] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[256] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m) {
    uint32_t e = threadIdx.x;
    if (e >= 256u) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_nvfp4_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *gate_scale_2,
        const float *up_scale_2) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_nvfp4 *gr = (const cuda_block_nvfp4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_nvfp4 *ur = (const cuda_block_nvfp4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_nvfp4_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate_scale_2[expert], gate);
            dev_dot_nvfp4_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up_scale_2[expert], up);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

/* NVFP4 down direct-sum6 decode kernel (n_tokens==1). One thread block per row_chunk,
 * each thread processes one output row, sums 6 expert contributions. Uses per-expert
 * scale_2 arrays. Mirrors moe_down_sum6_qwarp32_kernel but with NVFP4 weights. */
__global__ static void moe_down_sum6_nvfp4_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        const float *down_scale_2) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_nvfp4 *wr = (const cuda_block_nvfp4 *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_nvfp4_q8_K_block(wr + b, xq + b, down_scale_2[expert_i]);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_nvfp4_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out,
        const float *down_scale_2) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_nvfp4 *wr = (const cuda_block_nvfp4 *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_nvfp4_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, down_scale_2[expert], acc);
            if (np > 8u) {
                dev_dot_nvfp4_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                          xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                          xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                          xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, down_scale_2[expert], acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}


template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_block16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block16(wr + b,
                                      xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL,
                                      xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                      xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                      xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                      xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL,
                                      np, acc);
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}
