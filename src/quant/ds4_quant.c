/* =========================================================================
 * ds4_quant.c - CPU reference quant block formats + dequant/dot kernels.
 * =========================================================================
 *
 * Owns the GGUF quantized-tensor block layouts (Q2_K, Q4_K, Q8_K, IQ2_XXS),
 * the IQ2 lookup tables (built lazily via ds4_quant_init), and the scalar
 * conversions + CPU dequant/dot-product kernels used by the CPU reference
 * backend and GPU-graph numeric diagnostics.
 *
 * NEON DOTPROD paths are used on __ARM_FEATURE_DOTPROD; otherwise portable
 * scalar/int fallbacks. Depends on ds4_util (ds4_die) only.
 */

#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ds4_internal.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ---- block layouts (validated against the GGUF on-disk sizes) ----------- */

DS4_STATIC_ASSERT(ds4_block_q2_k_size, sizeof(block_q2_K) == 84);
DS4_STATIC_ASSERT(ds4_block_q4_k_size, sizeof(block_q4_K) == 144);
DS4_STATIC_ASSERT(ds4_block_q8_k_size, sizeof(block_q8_K) == 292);
DS4_STATIC_ASSERT(ds4_block_iq2_xxs_size, sizeof(block_iq2_xxs) == 66);

/* ---- IQ2 lookup tables -------------------------------------------------- */

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99,100,101,102,103,104,105,106,107,108,109,110,111,
   112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
   128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
   144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,
   160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
   176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
   192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
   208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
   224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,
   240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
};

static int8_t iq2xxs_signed_grid[256][128][8];
static int8_t iq2xxs_signs[128][8];
static pthread_once_t iq2xxs_signed_grid_once = PTHREAD_ONCE_INIT;

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t s = 0; s < 128; s++) {
        const uint8_t signs = ksigns_iq2xs[s];
        for (uint32_t j = 0; j < 8; j++) {
            iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
        }
    }

    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

/* Build the IQ2 signed lookup grids exactly once.  Exposed through
 * ds4_internal.h; ds4_threads_init() calls this before spawning workers so the
 * first CPU dequant never pays the build cost in a hot path. */
void ds4_quant_init(void) {
    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);
}

static inline DS4_MAYBE_UNUSED int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), gv, vld1q_s8(q8));
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int8x16_t qv = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(gv), vget_low_s8(qv));
    const int16x8_t p1 = vmull_s8(vget_high_s8(gv), vget_high_s8(qv));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
#endif
}

static inline DS4_MAYBE_UNUSED int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t packed = vld1q_u8(q2);
    uint8x16_t shifted;
    switch (shift) {
    case 0: shifted = packed; break;
    case 2: shifted = vshrq_n_u8(packed, 2); break;
    case 4: shifted = vshrq_n_u8(packed, 4); break;
    default: shifted = vshrq_n_u8(packed, 6); break;
    }
    const uint8x16_t vals_u = vandq_u8(shifted, vdupq_n_u8(3));
    const int8x16_t vals = vreinterpretq_s8_u8(vals_u);
    const int8x16_t q8v = vld1q_s8(q8);
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q8v, vals);
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    uint8_t vals_tmp[16];
    for (uint32_t i = 0; i < 16; i++) vals_tmp[i] = (q2[i] >> shift) & 3;
    const int8x16_t vals = vreinterpretq_s8_u8(vld1q_u8(vals_tmp));
    const int8x16_t q8v = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(q8v), vget_low_s8(vals));
    const int16x8_t p1 = vmull_s8(vget_high_s8(q8v), vget_high_s8(vals));
    const int32x4_t s0 = vpaddlq_s16(p0);
    const int32x4_t s1 = vpaddlq_s16(p1);
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
#endif
}

/* ---- scalar conversions ------------------------------------------------- */

float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

uint16_t f32_to_f16(float f) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
#endif
}

void f16_round_inplace_cpu(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = f16_to_f32(f32_to_f16(x[i]));
}

float dsv4_e4m3fn_value_cpu(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };

    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

float dsv4_e4m3fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * dsv4_e4m3fn_value_cpu(best);
}

/* DeepSeek V4 stores the non-RoPE part of compressed KV through an E4M3-style
 * round trip.  Keeping this in the CPU reference makes cache values comparable
 * to the GPU graph's compressed-cache behavior. */
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
        }
    }
}

/* ---- Q8_K activation quantization (shared across expert rows) ----------- */

/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
 * kernels can reuse the same activation for many expert rows. */
void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    if (k % QK_K != 0) ds4_die("Q8_K quantization length is not QK_K aligned");
    const int64_t nb = k / QK_K;

    for (int64_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }

        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

/* ---- quant dot-product kernels ------------------------------------------ */

void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const float dmin = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        uint8_t scale_lanes[16];
        vst1q_u8(scale_lanes, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8x2_t mins16 = {{
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
        }};
        const int32x4_t s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
            vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
            vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * (float)vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;
        for (int j = 0; j < QK_K / 128; j++) {
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
            q2 += 32;

#define DS4_Q2_DOT_NOSHIFT(scale_index) do {                                           \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));\
                const int8x16_t q2hi = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));\
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

#define DS4_Q2_DOT_SHIFT(shift, scale_index) do {                                      \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));                 \
                const int8x16_t q2hi = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));                 \
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

            DS4_Q2_DOT_NOSHIFT(0);
            DS4_Q2_DOT_SHIFT(2, 2);
            DS4_Q2_DOT_SHIFT(4, 4);
            DS4_Q2_DOT_SHIFT(6, 6);
            is += 8;

#undef DS4_Q2_DOT_NOSHIFT
#undef DS4_Q2_DOT_SHIFT
        }

        sum += d * (float)isum;
    }

    *s = sum;
#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    *s = sumf;
#endif
}

static DS4_MAYBE_UNUSED void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        float sumf1 = 0.0f;
        float sumf2 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux32[4];
            memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            int8x16_t q2u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[0])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[1])));
            int8x16_t q2u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[2])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[3])));
            int8x16_t q2u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[8])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[9])));
            int8x16_t q2u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[10])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[11])));

            const int8x16_t q2s0 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >>  7) & 127]));
            const int8x16_t q2s1 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >> 21) & 127]));
            const int8x16_t q2s2 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[2] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[2] >>  7) & 127]));
            const int8x16_t q2s3 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[2] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[2] >> 21) & 127]));

            q2u0 = vmulq_s8(q2u0, q2s0);
            q2u1 = vmulq_s8(q2u1, q2s1);
            q2u2 = vmulq_s8(q2u2, q2s2);
            q2u3 = vmulq_s8(q2u3, q2s3);

            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u0, q8b.val[0]), q2u1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u2, q8b.val[2]), q2u3, q8b.val[3]);

            sumf1 += (float)vaddvq_s32(p1) * (0.5f + (float)(aux32[1] >> 28));
            sumf2 += (float)vaddvq_s32(p2) * (0.5f + (float)(aux32[3] >> 28));
        }

        sumf += d * (sumf1 + sumf2);
    }

    *s = 0.25f * sumf;
#else
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
#endif
}

static inline void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

void ds4_vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d  = y[i].d * f16_to_f32(x[i].d);
        const float dm = -y[i].d * f16_to_f32(x[i].dmin);
        const uint8_t *qs = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int8_t  *q8 = y[i].qs;
        int summs = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);
            int32_t gsum = (int32_t)y[i].bsums[j * 2] + (int32_t)y[i].bsums[j * 2 + 1];
            summs += m_val * gsum;
        }
        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);
            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            for (int l = 0; l < 32; l++) {
                int q4_val = (shift == 0) ? (qs[byte_off + l] & 0xF) : (qs[byte_off + l] >> 4);
                isum += sc_val * q4_val * q8[j * 32 + l];
            }
        }
        sumf += d * (float)isum + dm * (float)summs;
    }
    *s = sumf;
}

void ds4_vec_dot_iq2_xxs_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const block_iq2_xxs *x0,
        const block_iq2_xxs *x1,
        const block_q8_K *y) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int nb = n / QK_K;
    float total0 = 0.0f;
    float total1 = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = f16_to_f32(x0[i].d) * y[i].d;
        const float d1 = f16_to_f32(x1[i].d) * y[i].d;
        const uint16_t *q20 = x0[i].qs;
        const uint16_t *q21 = x1[i].qs;
        const int8_t *q8 = y[i].qs;
        float sum01 = 0.0f;
        float sum02 = 0.0f;
        float sum11 = 0.0f;
        float sum12 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux0[4];
            uint32_t aux1[4];
            memcpy(aux0, q20, sizeof(aux0));
            memcpy(aux1, q21, sizeof(aux1));
            q20 += 8;
            q21 += 8;
            const uint8_t *a0 = (const uint8_t *)aux0;
            const uint8_t *a1 = (const uint8_t *)aux1;

#define DS4_IQ2_PAIR_DOT(aux, aux8, accum_a, accum_b) do {                                              \
                int8x16_t u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[0])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[1])));          \
                int8x16_t u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[2])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[3])));          \
                int8x16_t u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[8])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[9])));          \
                int8x16_t u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[10])),         \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[11])));         \
                const int8x16_t sgn0 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >>  7) & 127]));      \
                const int8x16_t sgn1 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >> 21) & 127]));      \
                const int8x16_t sgn2 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >>  7) & 127]));      \
                const int8x16_t sgn3 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >> 21) & 127]));      \
                u0 = vmulq_s8(u0, sgn0);                                                               \
                u1 = vmulq_s8(u1, sgn1);                                                               \
                u2 = vmulq_s8(u2, sgn2);                                                               \
                u3 = vmulq_s8(u3, sgn3);                                                               \
                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u0, q8b.val[0]), u1, q8b.val[1]); \
                const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u2, q8b.val[2]), u3, q8b.val[3]); \
                (accum_a) += (float)vaddvq_s32(p1) * (0.5f + (float)((aux)[1] >> 28));                  \
                (accum_b) += (float)vaddvq_s32(p2) * (0.5f + (float)((aux)[3] >> 28));                  \
            } while (0)

            DS4_IQ2_PAIR_DOT(aux0, a0, sum01, sum02);
            DS4_IQ2_PAIR_DOT(aux1, a1, sum11, sum12);

#undef DS4_IQ2_PAIR_DOT
        }

        total0 += d0 * (sum01 + sum02);
        total1 += d1 * (sum11 + sum12);
    }

    *s0 = 0.25f * total0;
    *s1 = 0.25f * total1;
#else
    ds4_vec_dot_iq2_xxs_q8_K(n, s0, x0, y);
    ds4_vec_dot_iq2_xxs_q8_K(n, s1, x1, y);
#endif
}
