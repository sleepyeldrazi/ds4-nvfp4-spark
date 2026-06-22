/* ======================================================================
 * turbo4 packed-attention: indexed mixed, online softmax, 8-heads/block
 * ======================================================================
 *
 * Reads raw_kv as FP32 and comp_kv as turbo4-packed format:
 *   [ n_nope e4m3 bytes | n_nope/64 e8m0 scales | 1 pad byte | n_rot BF16 ]
 * For DS4: 384 e4m3 + 6 e8m0 + 1 pad + 128 BF16 = 584 bytes/row (73×8).
 *
 * Each warp handles one head. Thread `lane` contributes 16 dims:
 *   q0 → dims lane*4 .. lane*4+3    (float4[lane+0])
 *   q1 → dims lane*4+128 .. +131    (float4[lane+32])
 *   q2 → dims lane*4+256 .. +259    (float4[lane+64])
 *   q3 → dims lane*4+384 .. +387    (float4[lane+96])
 * Warp sum (32 threads × 16 dims) = 512 dims.
 *
 * Single-read design: load KV elements once per row, reuse for score + output.
 * Shared memory: ~3 KB (raw_rows[256] + comp_rows[512] + counters).
 */

/* ---- device helper: turbo4 packed-row stride ---- */
static __device__ inline uint64_t _t4_row_stride(uint32_t head_dim, uint32_t n_rot) {
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + 63u) / 64u;
    uint64_t total = (uint64_t)n_nope + n_blocks + 1u + (uint64_t)n_rot * 2u;
    return (total + 7ull) & ~7ull;
}

/* ---- device helper: read one FP32 element from a turbo4 row ---- */
static __device__ inline float _t4_elem(const uint8_t *row, uint32_t dim,
                                         uint32_t head_dim, uint32_t n_rot) {
    uint32_t n_nope = head_dim - n_rot;
    uint32_t n_blocks = (n_nope + 63u) / 64u;
    if (dim < n_nope) {
        uint32_t blk = dim / 64u;
        __nv_fp8_storage_t sb = row[n_nope + blk];
        __nv_fp8_e8m0 sc = *reinterpret_cast<__nv_fp8_e8m0 *>(&sb);
        float sv = (float)sc;
        __nv_fp8_storage_t qb = row[dim];
        __nv_fp8_e4m3 q  = *reinterpret_cast<__nv_fp8_e4m3 *>(&qb);
        return (float)q * sv;
    } else {
        const uint16_t *rot = (const uint16_t *)(row + n_nope + n_blocks + 1u);
        return __bfloat162float(__ushort_as_bfloat16(rot[dim - n_nope]));
    }
}

__global__ static void attention_indexed_mixed_heads8_online_turbo4_kernel(
        float        *heads,
        const float  *sinks,
        const float  *q,
        const float  *raw_kv,
        const uint8_t *comp_kv,   /* turbo4-packed */
        const int32_t *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
        uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim)
{
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    /* raw window (thread 0) */
    if (threadIdx.x == 0) {
        raw_count = 0; raw_first_idx = 0; comp_count = 0;
        if (n_raw != 0) {
            uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x)
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    /* top-k comp indices (thread 0) */
    if (threadIdx.x == 0)
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp)
                comp_rows[comp_count++] = (uint32_t)c;
        }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t n_rot = 64u;  /* DS4_N_ROT */
    const uint64_t rstride = _t4_row_stride(head_dim, n_rot);

    /* load Q */
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u]; q1 = q4[lane + 32u];
        q2 = q4[lane + 64u]; q3 = q4[lane + 96u];
    }

    /* online softmax state */
    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    /* ---- process each row once ---- */
    for (uint32_t sr = 0; sr < n_score; sr++) {
        if (!valid_head) continue;

        /* --- load 16 KV elements into registers --- */
        float kv0, kv0y, kv0z, kv0w;   /* dims lane*4+0..3   */
        float kv1, kv1y, kv1z, kv1w;   /* dims lane*4+128..131 */
        float kv2, kv2y, kv2z, kv2w;   /* dims lane*4+256..259 */
        float kv3, kv3y, kv3z, kv3w;   /* dims lane*4+384..387 */
        float dot;

        if (sr < raw_count) {
            const float4 *r = (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim);
            float4 rk0 = r[lane +  0u];
            float4 rk1 = r[lane + 32u];
            float4 rk2 = r[lane + 64u];
            float4 rk3 = r[lane + 96u];
            kv0  = rk0.x;  kv0y = rk0.y;  kv0z = rk0.z;  kv0w = rk0.w;
            kv1  = rk1.x;  kv1y = rk1.y;  kv1z = rk1.z;  kv1w = rk1.w;
            kv2  = rk2.x;  kv2y = rk2.y;  kv2z = rk2.z;  kv2w = rk2.w;
            kv3  = rk3.x;  kv3y = rk3.y;  kv3z = rk3.z;  kv3w = rk3.w;
            dot  = q0.x*kv0  + q0.y*kv0y + q0.z*kv0z + q0.w*kv0w
                 + q1.x*kv1  + q1.y*kv1y + q1.z*kv1z + q1.w*kv1w
                 + q2.x*kv2  + q2.y*kv2y + q2.z*kv2z + q2.w*kv2w
                 + q3.x*kv3  + q3.y*kv3y + q3.z*kv3z + q3.w*kv3w;
        } else {
            uint32_t ci = comp_rows[sr - raw_count];
            const uint8_t *row = comp_kv + ci * rstride;
            uint32_t d0 = lane * 4u;
            kv0  = _t4_elem(row, d0,       head_dim, n_rot);
            kv0y = _t4_elem(row, d0 + 1u,  head_dim, n_rot);
            kv0z = _t4_elem(row, d0 + 2u,  head_dim, n_rot);
            kv0w = _t4_elem(row, d0 + 3u,  head_dim, n_rot);
            uint32_t d1 = d0 + 128u;
            kv1  = _t4_elem(row, d1,       head_dim, n_rot);
            kv1y = _t4_elem(row, d1 + 1u,  head_dim, n_rot);
            kv1z = _t4_elem(row, d1 + 2u,  head_dim, n_rot);
            kv1w = _t4_elem(row, d1 + 3u,  head_dim, n_rot);
            uint32_t d2 = d0 + 256u;
            kv2  = _t4_elem(row, d2,       head_dim, n_rot);
            kv2y = _t4_elem(row, d2 + 1u,  head_dim, n_rot);
            kv2z = _t4_elem(row, d2 + 2u,  head_dim, n_rot);
            kv2w = _t4_elem(row, d2 + 3u,  head_dim, n_rot);
            uint32_t d3 = d0 + 384u;
            kv3  = _t4_elem(row, d3,       head_dim, n_rot);
            kv3y = _t4_elem(row, d3 + 1u,  head_dim, n_rot);
            kv3z = _t4_elem(row, d3 + 2u,  head_dim, n_rot);
            kv3w = _t4_elem(row, d3 + 3u,  head_dim, n_rot);
            dot  = q0.x*kv0  + q0.y*kv0y + q0.z*kv0z + q0.w*kv0w
                 + q1.x*kv1  + q1.y*kv1y + q1.z*kv1z + q1.w*kv1w
                 + q2.x*kv2  + q2.y*kv2y + q2.z*kv2z + q2.w*kv2w
                 + q3.x*kv3  + q3.y*kv3y + q3.z*kv3z + q3.w*kv3w;
        }

        /* --- warp-sum dot product, scale, online softmax --- */
        dot = warp_sum_f32(dot);
        float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        float new_m = fmaxf(max_s, score);
        float old_scale = expf(max_s - new_m);
        float row_scale = expf(score - new_m);
        sum_s = sum_s * old_scale + row_scale;

        /* --- output accumulation --- */
        o0.x = o0.x * old_scale + kv0  * row_scale;
        o0.y = o0.y * old_scale + kv0y * row_scale;
        o0.z = o0.z * old_scale + kv0z * row_scale;
        o0.w = o0.w * old_scale + kv0w * row_scale;
        o1.x = o1.x * old_scale + kv1  * row_scale;
        o1.y = o1.y * old_scale + kv1y * row_scale;
        o1.z = o1.z * old_scale + kv1z * row_scale;
        o1.w = o1.w * old_scale + kv1w * row_scale;
        o2.x = o2.x * old_scale + kv2  * row_scale;
        o2.y = o2.y * old_scale + kv2y * row_scale;
        o2.z = o2.z * old_scale + kv2z * row_scale;
        o2.w = o2.w * old_scale + kv2w * row_scale;
        o3.x = o3.x * old_scale + kv3  * row_scale;
        o3.y = o3.y * old_scale + kv3y * row_scale;
        o3.z = o3.z * old_scale + kv3z * row_scale;
        o3.w = o3.w * old_scale + kv3w * row_scale;
        max_s = new_m;
    }

    /* ---- final sink + write output ---- */
    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}
