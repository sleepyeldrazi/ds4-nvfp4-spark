__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block16(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        const cuda_block_q8_K *y8,
        const cuda_block_q8_K *y9,
        const cuda_block_q8_K *y10,
        const cuda_block_q8_K *y11,
        const cuda_block_q8_K *y12,
        const cuda_block_q8_K *y13,
        const cuda_block_q8_K *y14,
        const cuda_block_q8_K *y15,
        uint32_t n,
        float acc[16]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[16] = {
        y0, y1, y2, y3, y4, y5, y6, y7,
        y8, y9, y10, y11, y12, y13, y14, y15,
    };
    int isum[16] = {0};
    int summs[16] = {0};
    for (uint32_t p = 0; p < n; p++) {
        #pragma unroll
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }

    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

// =========================================================================
// NVFP4 expert weight dequant + dot.
//
// NVFP4 = 2-level-scaled FP4 (e2m1) weights. One weight element = an e2m1
// nibble (4 bits; 16 values {0,0.5,1,1.5,2,3,4,6} + negatives), two nibbles
// packed per byte (low nibble = even element, high nibble = odd). One e4m3fn
// block scale per 16 elements along K; one fp32 per-tensor scale_2 = amax/(6*448).
// Dequant (matches modelopt NVFP4QTensor.dequantize exactly):
//   dequant[i] = e2m1_values[nibble[i]] * e4m3_to_float(scale[i/16]) * scale_2
//
// cuda_block_nvfp4 covers 256 elements (mirrors the Q8_K 256-block granularity
// the expert kernels pair 1:1 with): 128 packed nibble bytes + 16 e4m3 block
// scales. scale_2 is per-tensor and passed as a kernel arg (not in the block).
// =========================================================================
#define CUDA_NVFP4_QK  256
#define CUDA_NVFP4_SUB 16

typedef struct {
    uint8_t qs[CUDA_NVFP4_QK / 2];                  // 128 bytes: 256 e2m1 nibbles, 2/byte
    uint8_t scales[CUDA_NVFP4_QK / CUDA_NVFP4_SUB]; // 16 e4m3fn block scales
} cuda_block_nvfp4;

// 2*e2m1 as int8, indexed by nibble (bit3 = sign): {0,1,2,3,4,6,8,12} + neg.
// Computed branchlessly from the e2m1 bit layout (e=idx>>1, m=idx&1) -- avoids
// any LUT memory access (a __constant__/static-const array indexed by a runtime
// nibble serializes 16-way, the bottleneck that caps IQ2_XXS at 58 GB/s).
__device__ __forceinline__ int8_t nvfp4_e2m1_x2_val(uint8_t nib){
    uint32_t idx  = nib & 0x7u;
    uint32_t sign = (nib >> 3u) & 0x1u;
    uint32_t e    = idx >> 1u;
    uint32_t m    = idx & 0x1u;
    uint32_t two_e = 1u << e;
    uint32_t mag   = (two_e + m * (two_e >> 1u)) * (idx != 0u);
    return sign ? -(int8_t)mag : (int8_t)mag;
}

// SIMD unpack of 8 e2m1 nibbles (4 packed bytes = one uint32) -> 2 int32 lanes
// of 4 signed int8 (the 2*e2m1 values), via __byte_perm + __vsub4. The 4 nibbles
// of packed bytes [b0,b1] ARE the LUT indices [b0.lo,b0.hi,b1.lo,b1.hi] already
// in nibble positions, so the magnitude-LUT selector for lane0 is just
// `packed & 0x7777` (clears each nibble's sign bit -> bit[3]=0, index = nib&7).
// One __byte_perm looks up 4 magnitudes from {0,1,2,3,4,6,8,12}; __vsub4 applies
// per-byte sign (ds4's IQ2 sign trick). ~1.75 ops/elem.
__device__ __forceinline__ void nvfp4_unpack8(uint32_t packed, int32_t &lane0, int32_t &lane1){
    const uint32_t MAG_LO = 0x03020100u;
    const uint32_t MAG_HI = 0x0C080604u;
    const uint32_t SEL04  = 0x00005140u;   // [los0, his0, los1, his1]
    const uint32_t SEL15  = 0x00007362u;   // [los2, his2, los3, his3]
    uint32_t sel0 = packed & 0x00007777u;
    uint32_t sel1 = (packed >> 16) & 0x00007777u;
    uint32_t mag0 = __byte_perm(MAG_LO, MAG_HI, sel0);
    uint32_t mag1 = __byte_perm(MAG_LO, MAG_HI, sel1);
    uint32_t los = (packed >> 3) & 0x01010101u;
    uint32_t his = (packed >> 7) & 0x01010101u;
    uint32_t sm0 = (uint32_t)__vsub4(0, (int32_t)__byte_perm(los, his, SEL04));
    uint32_t sm1 = (uint32_t)__vsub4(0, (int32_t)__byte_perm(los, his, SEL15));
    lane0 = __vsub4((int32_t)(mag0 ^ sm0), (int32_t)sm0);
    lane1 = __vsub4((int32_t)(mag1 ^ sm1), (int32_t)sm1);
}

// e4m3fn -> float32. 1 sign, 4 exp (bias 7), 3 mant. Max normal = 448.
// 0x7F (exp=15,mant=7) = NaN. Subnormals = mant*2^-9. Verified bit-exact vs
// torch.float8_e4m3fn for all 256 byte values. Branchless (predicated SELs).
__device__ __forceinline__ float nvfp4_e4m3_to_float(uint8_t x){
    uint32_t sign = (x >> 7u) & 1u;
    uint32_t exp  = (x >> 3u) & 0xFu;
    uint32_t mant = x & 0x7u;
    uint32_t normal = (sign << 31u) | ((exp + 120u) << 23u) | (mant << 20u);
    float sub = (float)mant * 0.001953125f;
    sub = sign ? -sub : sub;
    float v = (exp == 0u) ? sub : __int_as_float(normal);
    uint32_t is_nan = ((exp == 15u) & (mant == 7u));
    return is_nan ? __int_as_float((sign << 31u) | 0x7fc00000u) : v;
}

// Dot of one 256-element NVFP4 weight block with one Q8_K activation block.
// Mirrors dev_dot_iq2_xxs_q8_K_block / dev_dot_q2_K_q8_K_block structure.
__device__ static float dev_dot_nvfp4_q8_K_block(const cuda_block_nvfp4 *x,
                                                 const cuda_block_q8_K *y,
                                                 float scale_2){
    const int8_t *q8 = y->qs;
    const uint8_t *qs = x->qs;
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < CUDA_NVFP4_QK / CUDA_NVFP4_SUB; b++) {
        const uint32_t *qp = (const uint32_t *)(qs + b * (CUDA_NVFP4_SUB / 2));
        int32_t l0a, l0b, l1a, l1b;
        nvfp4_unpack8(qp[0], l0a, l0b);
        nvfp4_unpack8(qp[1], l1a, l1b);
        int32_t sumi = 0;
        sumi = __dp4a(l0a, *(const int32_t *)(q8 + b * 16 + 0),  sumi);
        sumi = __dp4a(l0b, *(const int32_t *)(q8 + b * 16 + 4),  sumi);
        sumi = __dp4a(l1a, *(const int32_t *)(q8 + b * 16 + 8),  sumi);
        sumi = __dp4a(l1b, *(const int32_t *)(q8 + b * 16 + 12), sumi);
        acc = __fmaf_rn(nvfp4_e4m3_to_float(x->scales[b]), (float)sumi, acc);
    }
    return y->d * scale_2 * 0.5f * acc;
}

// 8-way batched dot: one NVFP4 weight block vs up to 8 Q8_K activation blocks,
// accumulating into acc[8]. Mirrors dev_dot_q2_K_q8_K_block8 / dev_dot_iq2_*_block8.
__device__ static void dev_dot_nvfp4_q8_K_block8(
        const cuda_block_nvfp4 *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float scale_2,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    const uint8_t *qs = x->qs;
    // Per-sub-block e4m3 scale (shared across the 8 activations).
    float bs[16];
    #pragma unroll
    for (int b = 0; b < CUDA_NVFP4_QK / CUDA_NVFP4_SUB; b++) bs[b] = nvfp4_e4m3_to_float(x->scales[b]);
    for (uint32_t p = 0; p < n; p++) {
        const int8_t *q8 = ys[p]->qs;
        float s = 0.0f;
        #pragma unroll
        for (int b = 0; b < CUDA_NVFP4_QK / CUDA_NVFP4_SUB; b++) {
            const uint32_t *qp = (const uint32_t *)(qs + b * (CUDA_NVFP4_SUB / 2));
            int32_t l0a, l0b, l1a, l1b;
            nvfp4_unpack8(qp[0], l0a, l0b);
            nvfp4_unpack8(qp[1], l1a, l1b);
            int32_t sumi = 0;
            sumi = __dp4a(l0a, *(const int32_t *)(q8 + b * 16 + 0),  sumi);
            sumi = __dp4a(l0b, *(const int32_t *)(q8 + b * 16 + 4),  sumi);
            sumi = __dp4a(l1a, *(const int32_t *)(q8 + b * 16 + 8),  sumi);
            sumi = __dp4a(l1b, *(const int32_t *)(q8 + b * 16 + 12), sumi);
            s = __fmaf_rn(bs[b], (float)sumi, s);
        }
        acc[p] += ys[p]->d * scale_2 * 0.5f * s;
    }
}

// 4-way batched dot (mirrors dev_dot_q2_K_q8_K_block4 / dev_dot_iq2_*_block4).
__device__ static void dev_dot_nvfp4_q8_K_block4(
        const cuda_block_nvfp4 *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float scale_2,
        float acc[4]) {
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    const uint8_t *qs = x->qs;
    float bs[16];
    #pragma unroll
    for (int b = 0; b < CUDA_NVFP4_QK / CUDA_NVFP4_SUB; b++) bs[b] = nvfp4_e4m3_to_float(x->scales[b]);
    for (uint32_t p = 0; p < n; p++) {
        const int8_t *q8 = ys[p]->qs;
        float s = 0.0f;
        #pragma unroll
        for (int b = 0; b < CUDA_NVFP4_QK / CUDA_NVFP4_SUB; b++) {
            const uint32_t *qp = (const uint32_t *)(qs + b * (CUDA_NVFP4_SUB / 2));
            int32_t l0a, l0b, l1a, l1b;
            nvfp4_unpack8(qp[0], l0a, l0b);
            nvfp4_unpack8(qp[1], l1a, l1b);
            int32_t sumi = 0;
            sumi = __dp4a(l0a, *(const int32_t *)(q8 + b * 16 + 0),  sumi);
            sumi = __dp4a(l0b, *(const int32_t *)(q8 + b * 16 + 4),  sumi);
            sumi = __dp4a(l1a, *(const int32_t *)(q8 + b * 16 + 8),  sumi);
            sumi = __dp4a(l1b, *(const int32_t *)(q8 + b * 16 + 12), sumi);
            s = __fmaf_rn(bs[b], (float)sumi, s);
        }
        acc[p] += ys[p]->d * scale_2 * 0.5f * s;
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}
