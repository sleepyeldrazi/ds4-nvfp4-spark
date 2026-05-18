constant float dsv4_e4m3fn_exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

struct ds4_metal_args_dsv4_fp8_kv_quantize {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    ulong nb0;
    ulong nb1;
    ulong nb2;
    ulong nb3;
    int n_rot;
};

struct ds4_metal_args_dsv4_kv_fp8_store {
    int32_t head_dim;
    int32_t n_rot;
    int32_t raw_row;
};

struct ds4_metal_args_dsv4_turbo4_packed_kv {
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t n_rows;
    uint32_t src_row0;
    uint32_t dst_row0;
    uint32_t mode;
    uint32_t reserved;
    ulong src_row_bytes;
    ulong dst_row_bytes;
    ulong packed_row_bytes;
};

struct ds4_metal_args_dsv4_ratio4_shift {
    uint32_t width;
};

struct ds4_metal_args_dsv4_compressor_store_one {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
};

static inline float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? float(mant) * 0.001953125f
        : (1.0f + float(mant) * 0.125f) * dsv4_e4m3fn_exp_scale[exp];
}

static inline float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = min(abs(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best = best + 1;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

// Quantizes the non-RoPE part of a KV row through E4M3FN and writes the
// dequantized value back as float. DS4 uses this to match the FP8 KV-cache
// semantics while keeping the Metal graph's cache buffers float-addressable.

kernel void kernel_dsv4_fp8_kv_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        threadgroup  float * scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device       char * dst_base = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *((device const float *) (src_base + (off + tid)*args.nb00));
            scratch[tid] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float scale = exp2(ceil(log2(amax / 448.0f)));
        if (tid < 64) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
            *((device float *) (dst_base + (off + tid)*args.nb0)) = q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((device float *) (dst_base + i*args.nb0)) = *((device const float *) (src_base + i*args.nb00));
    }
}

constant float dsv4_turbo4_fake_centroids[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

constant float dsv4_turbo3_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

constant float dsv4_turbo_rht_s1[128] = {
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f
};

constant float dsv4_turbo_rht_s2[128] = {
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f
};

static inline uint dsv4_turbo4_fake_centroid_index(float x) {
    uint best = 0;
    float best_d = abs(x - dsv4_turbo4_fake_centroids[0]);
    for (uint i = 1; i < 16; i++) {
        float d = abs(x - dsv4_turbo4_fake_centroids[i]);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

static inline float dsv4_turbo4_fake_centroid(float x) {
    return dsv4_turbo4_fake_centroids[dsv4_turbo4_fake_centroid_index(x)];
}

static inline uint dsv4_turbo3_centroid_index(float x) {
    if (x < -0.154259f) return 0u;
    if (x < -0.091775f) return 1u;
    if (x < -0.043589f) return 2u;
    if (x <  0.000000f) return 3u;
    if (x <  0.043589f) return 4u;
    if (x <  0.091775f) return 5u;
    if (x <  0.154259f) return 6u;
    return 7u;
}

static inline void dsv4_turbo_fwht_128(thread float x[128]) {
    for (uint len = 1; len < 128; len <<= 1) {
        for (uint i = 0; i < 128; i += (len << 1)) {
            for (uint j = 0; j < len; j++) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }

    const float scale = 0.08838834764831845f; // 1 / sqrt(128)
    for (uint i = 0; i < 128; i++) {
        x[i] *= scale;
    }
}

static inline void dsv4_turbo_rht_128_forward(thread float x[128]) {
    for (uint i = 0; i < 128; i++) {
        x[i] *= dsv4_turbo_rht_s1[i];
    }
    dsv4_turbo_fwht_128(x);
    for (uint i = 0; i < 128; i++) {
        x[i] *= dsv4_turbo_rht_s2[i];
    }
}

static inline void dsv4_turbo_rht_128_inverse(thread float x[128]) {
    for (uint i = 0; i < 128; i++) {
        x[i] *= dsv4_turbo_rht_s2[i];
    }
    dsv4_turbo_fwht_128(x);
    for (uint i = 0; i < 128; i++) {
        x[i] *= dsv4_turbo_rht_s1[i];
    }
}

kernel void kernel_dsv4_turbo4_pack_f32(
        constant ds4_metal_args_dsv4_turbo4_packed_kv & args,
        device const uchar * src,
        device       uchar * dst,
        uint gid [[thread_position_in_grid]]) {
    uint row = gid;
    uint turbo3_block = 0u;
    uint turbo3_begin = 0u;
    uint turbo3_end = 0u;
    if (args.mode == 3u) {
        if ((args.head_dim & 127u) != 0u) {
            return;
        }
        const uint blocks = args.head_dim / 128u;
        if (blocks == 0u) {
            return;
        }
        if (args.reserved != 0u) {
            row = gid / blocks;
            turbo3_block = gid - row * blocks;
            turbo3_begin = turbo3_block;
            turbo3_end = turbo3_block + 1u;
        } else {
            row = gid;
            turbo3_begin = 0u;
            turbo3_end = blocks;
        }
    }
    if (row >= args.n_rows || args.head_dim == 0 || args.n_rot > args.head_dim) {
        return;
    }

    const uint n_nope = args.head_dim - args.n_rot;
    const uint full_end = (n_nope / 128u) * 128u;
    const uint n_blocks = full_end / 128u;

    device const uchar * src_row = src + ((ulong)args.src_row0 + row) * args.src_row_bytes;
    device       uchar * dst_row = dst + ((ulong)args.dst_row0 + row) * args.packed_row_bytes;

    if (args.mode == 5u) {
        for (uint block = 0; block < n_blocks; block++) {
            const uint off = block * 128u;
            float norm_sq = 0.0f;
            float v[128];

            for (uint i = 0; i < 128u; i++) {
                const float x = *((device const float *)(src_row + (off + i) * sizeof(float)));
                v[i] = x;
                norm_sq += x * x;
            }

            const float norm = sqrt(norm_sq);
            const float inv_norm = norm > 1.0e-10f ? 1.0f / norm : 0.0f;
            for (uint i = 0; i < 128u; i++) {
                v[i] *= inv_norm;
            }

            dsv4_turbo_rht_128_forward(v);

            uint qidx[128];
            float recon_sq = 0.0f;
            for (uint i = 0; i < 128u; i++) {
                const uint idx = dsv4_turbo3_centroid_index(v[i]);
                const float q = dsv4_turbo3_centroids[idx];
                qidx[i] = idx;
                recon_sq += q * q;
            }

            const float recon_norm = sqrt(recon_sq);
            const float corrected = recon_norm > 1.0e-10f ? norm / recon_norm : norm;
            device uchar * out_block = dst_row + (ulong)block * 50ul;
            *((device half *)out_block) = half(corrected);
            device uchar * qs = out_block + sizeof(half);
            device uchar * signs = qs + 32ul;

            for (uint i = 0; i < 32u; i++) {
                qs[i] = (uchar)((qidx[i * 4u + 0u] & 0x03u) |
                                ((qidx[i * 4u + 1u] & 0x03u) << 2u) |
                                ((qidx[i * 4u + 2u] & 0x03u) << 4u) |
                                ((qidx[i * 4u + 3u] & 0x03u) << 6u));
            }
            for (uint i = 0; i < 16u; i++) {
                uchar bits = 0;
                for (uint j = 0; j < 8u; j++) {
                    bits |= (uchar)(((qidx[i * 8u + j] >> 2u) & 0x01u) << j);
                }
                signs[i] = bits;
            }
        }

        device uchar * tail = dst_row + (ulong)n_blocks * 50ul;
        for (uint i = full_end; i < args.head_dim; i++) {
            *((device float *)(tail + (ulong)(i - full_end) * sizeof(float))) =
                *((device const float *)(src_row + (ulong)i * sizeof(float)));
        }
        return;
    }

    if (args.mode == 3u) {
        for (uint block = turbo3_begin; block < turbo3_end; block++) {
        const uint off = block * 128u;
        float norm_sq = 0.0f;
        float v[128];

        for (uint i = 0; i < 128u; i++) {
            const float x = *((device const float *)(src_row + (off + i) * sizeof(float)));
            v[i] = x;
            norm_sq += x * x;
        }

        const float norm = sqrt(norm_sq);
        const float inv_norm = norm > 1.0e-10f ? 1.0f / norm : 0.0f;
        for (uint i = 0; i < 128u; i++) {
            v[i] *= inv_norm;
        }

        dsv4_turbo_rht_128_forward(v);

        uint qidx[128];
        float recon_sq = 0.0f;
        for (uint i = 0; i < 128u; i++) {
            const uint idx = dsv4_turbo3_centroid_index(v[i]);
            const float q = dsv4_turbo3_centroids[idx];
            qidx[i] = idx;
            recon_sq += q * q;
        }

        const float recon_norm = sqrt(recon_sq);
        const float corrected = recon_norm > 1.0e-10f ? norm / recon_norm : norm;
        device uchar * out_block = dst_row + (ulong)block * 50ul;
        *((device half *)out_block) = half(corrected);
        device uchar * qs = out_block + sizeof(half);
        device uchar * signs = qs + 32ul;

        for (uint i = 0; i < 32u; i++) {
            qs[i] = (uchar)((qidx[i * 4u + 0u] & 0x03u) |
                            ((qidx[i * 4u + 1u] & 0x03u) << 2u) |
                            ((qidx[i * 4u + 2u] & 0x03u) << 4u) |
                            ((qidx[i * 4u + 3u] & 0x03u) << 6u));
        }
        for (uint i = 0; i < 16u; i++) {
            uchar bits = 0;
            for (uint j = 0; j < 8u; j++) {
                bits |= (uchar)(((qidx[i * 8u + j] >> 2u) & 0x01u) << j);
            }
            signs[i] = bits;
        }
        }
        return;
    }

    for (uint block = 0; block < n_blocks; block++) {
        const uint off = block * 128u;
        float norm_sq = 0.0f;
        float v[128];

        for (uint i = 0; i < 128u; i++) {
            const float x = *((device const float *)(src_row + (off + i) * sizeof(float)));
            v[i] = x;
            norm_sq += x * x;
        }

        const float norm = sqrt(norm_sq);
        const float inv_norm = norm > 1.0e-10f ? 1.0f / norm : 0.0f;
        for (uint i = 0; i < 128u; i++) {
            v[i] *= inv_norm;
        }

        dsv4_turbo_fwht_128(v);

        uint qidx[128];
        float recon_sq = 0.0f;
        for (uint i = 0; i < 128u; i++) {
            const uint idx = dsv4_turbo4_fake_centroid_index(v[i]);
            const float q = dsv4_turbo4_fake_centroids[idx];
            qidx[i] = idx;
            recon_sq += q * q;
        }

        const float recon_norm = sqrt(recon_sq);
        const float corrected = recon_norm > 1.0e-10f ? norm / recon_norm : norm;
        device uchar * out_block = dst_row + (ulong)block * 68ul;
        *((device float *)out_block) = corrected;
        device uchar * qs = out_block + sizeof(float);
        for (uint i = 0; i < 64u; i++) {
            qs[i] = (uchar)((qidx[i * 2u] & 0x0fu) | ((qidx[i * 2u + 1u] & 0x0fu) << 4));
        }
    }

    device uchar * tail = dst_row + (ulong)n_blocks * 68ul;
    for (uint i = full_end; i < args.head_dim; i++) {
        *((device float *)(tail + (ulong)(i - full_end) * sizeof(float))) =
            *((device const float *)(src_row + (ulong)i * sizeof(float)));
    }
}

kernel void kernel_dsv4_turbo4_unpack_f32(
        constant ds4_metal_args_dsv4_turbo4_packed_kv & args,
        device const uchar * src,
        device       uchar * dst,
        uint gid [[thread_position_in_grid]]) {
    uint row = gid;
    uint turbo3_block = 0u;
    uint turbo3_begin = 0u;
    uint turbo3_end = 0u;
    if (args.mode == 3u) {
        if ((args.head_dim & 127u) != 0u) {
            return;
        }
        const uint blocks = args.head_dim / 128u;
        if (blocks == 0u) {
            return;
        }
        if (args.reserved != 0u) {
            row = gid / blocks;
            turbo3_block = gid - row * blocks;
            turbo3_begin = turbo3_block;
            turbo3_end = turbo3_block + 1u;
        } else {
            row = gid;
            turbo3_begin = 0u;
            turbo3_end = blocks;
        }
    }
    if (row >= args.n_rows || args.head_dim == 0 || args.n_rot > args.head_dim) {
        return;
    }

    const uint n_nope = args.head_dim - args.n_rot;
    const uint full_end = (n_nope / 128u) * 128u;
    const uint n_blocks = full_end / 128u;

    device const uchar * src_row = src + ((ulong)args.src_row0 + row) * args.packed_row_bytes;
    device       uchar * dst_row = dst + ((ulong)args.dst_row0 + row) * args.dst_row_bytes;

    if (args.mode == 5u) {
        for (uint block = 0; block < n_blocks; block++) {
            device const uchar * in_block = src_row + (ulong)block * 50ul;
            const float corrected = float(*((device const half *)in_block));
            device const uchar * qs = in_block + sizeof(half);
            device const uchar * signs = qs + 32ul;
            float v[128];

            for (uint i = 0; i < 32u; i++) {
                const uchar packed = qs[i];
                const uchar sb = signs[i >> 1u];
                const uint sshift = (i & 1u) * 4u;
                v[i * 4u + 0u] = dsv4_turbo3_centroids[((uint)packed & 0x03u) |
                    ((((uint)sb >> (sshift + 0u)) & 0x01u) << 2u)] * corrected;
                v[i * 4u + 1u] = dsv4_turbo3_centroids[(((uint)packed >> 2u) & 0x03u) |
                    ((((uint)sb >> (sshift + 1u)) & 0x01u) << 2u)] * corrected;
                v[i * 4u + 2u] = dsv4_turbo3_centroids[(((uint)packed >> 4u) & 0x03u) |
                    ((((uint)sb >> (sshift + 2u)) & 0x01u) << 2u)] * corrected;
                v[i * 4u + 3u] = dsv4_turbo3_centroids[(((uint)packed >> 6u) & 0x03u) |
                    ((((uint)sb >> (sshift + 3u)) & 0x01u) << 2u)] * corrected;
            }

            dsv4_turbo_rht_128_inverse(v);

            const uint off = block * 128u;
            for (uint i = 0; i < 128u; i++) {
                *((device float *)(dst_row + (off + i) * sizeof(float))) = v[i];
            }
        }

        device const uchar * tail = src_row + (ulong)n_blocks * 50ul;
        for (uint i = full_end; i < args.head_dim; i++) {
            *((device float *)(dst_row + (ulong)i * sizeof(float))) =
                *((device const float *)(tail + (ulong)(i - full_end) * sizeof(float)));
        }
        return;
    }

    if (args.mode == 3u) {
        for (uint block = turbo3_begin; block < turbo3_end; block++) {
        device const uchar * in_block = src_row + (ulong)block * 50ul;
        const float corrected = float(*((device const half *)in_block));
        device const uchar * qs = in_block + sizeof(half);
        device const uchar * signs = qs + 32ul;
        float v[128];

        for (uint i = 0; i < 32u; i++) {
            const uchar packed = qs[i];
            const uchar sb = signs[i >> 1u];
            const uint sshift = (i & 1u) * 4u;
            v[i * 4u + 0u] = dsv4_turbo3_centroids[((uint)packed & 0x03u) |
                ((((uint)sb >> (sshift + 0u)) & 0x01u) << 2u)] * corrected;
            v[i * 4u + 1u] = dsv4_turbo3_centroids[(((uint)packed >> 2u) & 0x03u) |
                ((((uint)sb >> (sshift + 1u)) & 0x01u) << 2u)] * corrected;
            v[i * 4u + 2u] = dsv4_turbo3_centroids[(((uint)packed >> 4u) & 0x03u) |
                ((((uint)sb >> (sshift + 2u)) & 0x01u) << 2u)] * corrected;
            v[i * 4u + 3u] = dsv4_turbo3_centroids[(((uint)packed >> 6u) & 0x03u) |
                ((((uint)sb >> (sshift + 3u)) & 0x01u) << 2u)] * corrected;
        }

        dsv4_turbo_rht_128_inverse(v);

        const uint off = block * 128u;
        for (uint i = 0; i < 128u; i++) {
            *((device float *)(dst_row + (off + i) * sizeof(float))) = v[i];
        }
        }
        return;
    }

    for (uint block = 0; block < n_blocks; block++) {
        device const uchar * in_block = src_row + (ulong)block * 68ul;
        const float corrected = *((device const float *)in_block);
        device const uchar * qs = in_block + sizeof(float);
        float v[128];

        for (uint i = 0; i < 64u; i++) {
            const uchar packed = qs[i];
            v[i * 2u] = dsv4_turbo4_fake_centroids[packed & 0x0f] * corrected;
            v[i * 2u + 1u] = dsv4_turbo4_fake_centroids[(packed >> 4) & 0x0f] * corrected;
        }

        // The normalized Hadamard transform is its own inverse, so applying it
        // here returns the packed vector to the domain expected by existing
        // attention kernels.
        dsv4_turbo_fwht_128(v);

        const uint off = block * 128u;
        for (uint i = 0; i < 128u; i++) {
            *((device float *)(dst_row + (off + i) * sizeof(float))) = v[i];
        }
    }

    device const uchar * tail = src_row + (ulong)n_blocks * 68ul;
    for (uint i = full_end; i < args.head_dim; i++) {
        *((device float *)(dst_row + (ulong)i * sizeof(float))) =
            *((device const float *)(tail + (ulong)(i - full_end) * sizeof(float)));
    }
}

kernel void kernel_dsv4_turbo4_fake_quantize_f32(
        constant ds4_metal_args_dsv4_fp8_kv_quantize & args,
        device  const char * src0,
        device        char * dst,
        uint row [[thread_position_in_grid]]) {
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if ((int64_t)row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const char * src_base = src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03;
    device       char * dst_base = dst  + i1 * args.nb1  + i2 * args.nb2  + i3 * args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;
    const int64_t full_end = (n_nope / 128) * 128;

    for (int64_t off = 0; off < full_end; off += 128) {
        float norm_sq = 0.0f;
        float v[128];

        for (uint i = 0; i < 128; i++) {
            float x = *((device const float *)(src_base + (off + i) * args.nb00));
            v[i] = x;
            norm_sq += x * x;
        }

        float norm = sqrt(norm_sq);
        float inv_norm = norm > 1.0e-10f ? 1.0f / norm : 0.0f;

        for (uint i = 0; i < 128; i++) {
            v[i] *= inv_norm;
        }

        dsv4_turbo_fwht_128(v);

        float recon_sq = 0.0f;
        for (uint i = 0; i < 128; i++) {
            v[i] = dsv4_turbo4_fake_centroid(v[i]);
            recon_sq += v[i] * v[i];
        }

        float recon_norm = sqrt(recon_sq);
        float corrected = recon_norm > 1.0e-10f ? norm / recon_norm : norm;

        for (uint i = 0; i < 128; i++) {
            *((device float *)(dst_base + (off + i) * args.nb0)) = v[i] * corrected;
        }
    }

    for (int64_t i = full_end; i < args.ne00; i++) {
        *((device float *)(dst_base + i * args.nb0)) =
            *((device const float *)(src_base + i * args.nb00));
    }
}


// Decode-side KV finalizer after RoPE. The normal RoPE kernel intentionally
// remains separate because tiny trigonometric codegen changes can flip later
// sampled tokens. This kernel only fuses the FP8 round-trip for the non-RoPE
// prefix with the F16-rounded raw-cache row used by FlashAttention.
kernel void kernel_dsv4_kv_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            raw[off + tid] = (float)((half)q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
        raw[i] = (float)((half)kv[i]);
    }
}

// Ratio-4 compression keeps two 4-row halves of recurrent state. After an
// emitted compressed row, the second half becomes the next window's previous
// half. The old encoder expressed this as four generic copies; this DS4-specific
// kernel performs the KV and score copies together.
kernel void kernel_dsv4_ratio4_shift_f32(
        constant ds4_metal_args_dsv4_ratio4_shift & args,
        device float * state_kv,
        device float * state_score,
        uint gid [[thread_position_in_grid]]) {
    const uint n = 4u * args.width;
    if (gid >= n) return;

    state_kv[gid] = state_kv[n + gid];
    state_score[gid] = state_score[n + gid];
}

// One-token compressor frontier update. Decode appends exactly one projected KV
// row and one score row into a small recurrent state. The generic batch helper
// expresses this as APE copy, score add, and two set_rows operations; this
// kernel writes both state tensors directly while preserving the same
// score + APE arithmetic.
kernel void kernel_dsv4_compressor_store_one(
        constant ds4_metal_args_dsv4_compressor_store_one & args,
        device const float * kv,
        device const float * score,
        device const char  * ape,
        device       float * state_kv,
        device       float * state_score,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width || args.width == 0 || args.ratio == 0) {
        return;
    }

    const uint pos_mod = args.pos % args.ratio;
    const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod : pos_mod;
    const uint dst = dst_row * args.width + gid;
    const uint ape_i = pos_mod * args.width + gid;

    float ape_v;
    if (args.ape_type == 1u) {
        ape_v = (float)(((device const half *)ape)[ape_i]);
    } else {
        ape_v = ((device const float *)ape)[ape_i];
    }

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}
