__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

/* ======================================================================
 * Fused conditional emit kernel (decode-opt #5b / Forgejo #3)
 * ======================================================================
 *
 * Replaces the CPU-gated if(emit) { update_pool + RMS norm + RoPE + pack +
 * shift } path with a single always-launched kernel that no-ops at non-emit
 * positions.  This allows the CUDA Graph to capture a single tape valid for
 * ALL tokens (emit and non-emit), eliminating the manual emit dispatch.
 *
 * At emit boundaries (((pos+1) % ratio) == 0):
 *   1. update_pool: weighted sum from rolling state -> comp_cache[comp_row]
 *   2. RMS norm: normalize comp_cache[comp_row] by norm_weight
 *   3. RoPE tail: rotate comp_cache[comp_row] at position pos+1-ratio
 *   4. turbo4 pack: pack comp_cache[comp_row] -> tq_cache[comp_row] (if tq_base)
 *   5. shift ratio4 state buffer (if ratio==4)
 *
 * comp_row is computed from pos: comp_row = (pos+1)/ratio - 1.
 * Works for both attention (head_dim=512) and indexer (head_dim=128).
 */
__global__ static void compressor_emit_conditional_kernel(
        float *comp_cache_base,       /* 0: FP32 comp cache (or unpack scratch) */
        uint8_t *tq_cache_base,       /* 1: turbo4 packed cache (NULL if no turbo4) */
        float *state_kv,            /* 2: rolling state KV (mutated by shift) */
        float *state_score,         /* 3: rolling state scores (mutated by shift) */
        const float *norm_weight,     /* 4: RMS norm weights (head_dim floats, f32) */
        uint32_t comp_cap,            /* 5: capacity */
        uint32_t head_dim,            /* 6: 512 or 128 */
        uint32_t n_rot,               /* 7: 64 */
        uint32_t ratio,               /* 8: 4 or 128 */
        uint32_t pos,                 /* 9: DYNAMIC -- updated per replay */
        float rms_eps,                /* 10 */
        uint32_t n_ctx_orig,          /* 11 */
        float freq_base,              /* 12 */
        float freq_scale,             /* 13 */
        float ext_factor,             /* 14 */
        float attn_factor,            /* 15 */
        float beta_fast,              /* 16 */
        float beta_slow)              /* 17 */
{
    /* Emit guard -- no-op at non-emit positions */
    if (((pos + 1u) % ratio) != 0u) return;
    uint32_t comp_row = (pos + 1u) / ratio - 1u;
    if (comp_row >= comp_cap) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t swidth = coff * head_dim;
    float *row = comp_cache_base + (uint64_t)comp_row * head_dim;

    /* Step 1: update_pool -- weighted sum from rolling state -> row[] */
    if (tid < head_dim) {
        float vals[128];
        float scores[128];
        float max_s = -INFINITY;
        uint32_t n_cand = 0;
        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * swidth + tid];
                scores[n_cand] = state_score[(uint64_t)r * swidth + tid];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n_cand] = state_kv[(uint64_t)(ratio + r) * swidth + head_dim + tid];
                scores[n_cand] = state_score[(uint64_t)(ratio + r) * swidth + head_dim + tid];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * swidth + tid];
                scores[n_cand] = state_score[(uint64_t)r * swidth + tid];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        float den = 0.0f, acc = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            float w = expf(scores[i] - max_s);
            den += w;
            acc += vals[i] * w;
        }
        row[tid] = den != 0.0f ? acc / den : 0.0f;
    }
    __syncthreads();

    /* Step 2: RMS norm -- normalize row[] by norm_weight */
    {
        float sum = 0.0f;
        for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
            float v = row[d];
            sum += v * v;
        }
        const float scale = rsqrtf(block_sum_f32(sum) / (float)head_dim + rms_eps);
        for (uint32_t d = tid; d < head_dim; d += blockDim.x)
            row[d] = row[d] * scale * norm_weight[d];
    }
    __syncthreads();

    /* Step 3: RoPE tail at position pos+1-ratio */
    {
        const uint32_t rope_pos = pos + 1u - ratio;
        const uint32_t n_nope = head_dim - n_rot;
        float corr0 = 0.0f, corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            const float denom = 2.0f * logf(freq_base);
            corr0 = fmaxf(0.0f, floorf((float)n_rot *
                    logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
            corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot *
                    logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
        }
        const uint32_t n_pairs = n_rot / 2u;
        for (uint32_t pair = tid; pair < n_pairs; pair += blockDim.x) {
            uint32_t i = pair * 2u;
            float theta_extrap = (float)rope_pos * powf(freq_base, -((float)i) / (float)n_rot);
            float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            float c = cosf(theta) * mscale;
            float s = sinf(theta) * mscale;
            float x0 = row[n_nope + i];
            float x1 = row[n_nope + i + 1u];
            row[n_nope + i] = x0 * c - x1 * s;
            row[n_nope + i + 1u] = x0 * s + x1 * c;
        }
    }
    __syncthreads();

    /* Step 4: turbo4 pack -> tq_cache[comp_row] (if tq_cache_base != NULL)
     * NOTE: the pack is handled by a separate conditional kernel launch
     * (turbo4_pack_conditional_kernel in ds4_turbo4.cu) to keep the fused
     * emit kernel in ds4_cuda_compressor.cuh free of turbo4 pack specifics.
     * This step is skipped here. */
    __syncthreads();

    /* Step 5: shift ratio4 state buffer */
    if (ratio == 4u) {
        uint64_t half = 4ull * swidth;
        for (uint32_t i = tid; i < half; i += blockDim.x) {
            float v = state_kv[half + i];
            float s = state_score[half + i];
            state_kv[i] = v;
            state_score[i] = s;
            state_kv[half + i] = v;
            state_score[half + i] = s;
        }
    }
}
