/* =========================================================================
 * ds4_weights.c - GGUF -> DS4 weight binding + layout validation.
 * =========================================================================
 *
 * Converts the GGUF tensor directory into the DS4-specific layer weight pointer
 * tables (ds4_weights / ds4_mtp_weights), validates every tensor against the
 * fixed DeepSeek V4 Flash layout, and reads REAP pruning metadata.  Handles the
 * NVFP4 multi-tensor expert convention (.nvfp4_weight + .nvfp4_scale_2).  After
 * binding, inference addresses tensors by semantic field (layer->attn_q_a, ...)
 * instead of string lookup.  Depends on ds4_util + ds4_gguf.
 */

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_internal.h"

/* =========================================================================
 * Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static uint64_t required_u64(const ds4_model *m, const char *key) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        uint64_t v = 0;
        if (!cursor_u64(&c, &v)) ds4_die(c.error);
        return v;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) ds4_die(c.error);
        return v;
    }

    fprintf(stderr, "ds4: metadata key has a non-integer type: %s\n", key);
    exit(1);
}

static float required_f32(const ds4_model *m, const char *key) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        float v = 0.0f;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return v;
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return (float)v;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) ds4_die(c.error);
        return (float)v;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return (float)v;
    }

    fprintf(stderr, "ds4: metadata key has a non-float type %u: %s\n", kv->type, key);
    exit(1);
}

static bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

/* Load a routed-expert tensor, handling both standard (.weight) and
 * NVFP4 multi-tensor (.nvfp4_weight + .nvfp4_scale_2) naming.
 * If fmt contains %u, it is formatted with layer. Otherwise used as-is. */
static ds4_tensor *ds4_load_expert_tensor(const ds4_model *m, const char *fmt, uint32_t layer) {
    char base[128], name[160], scale_name[160];
    int n;
    if (strchr(fmt, '%'))
        snprintf(base, sizeof(base), fmt, layer);
    else
        snprintf(base, sizeof(base), "%s", fmt);

    /* Try standard .weight first. */
    n = snprintf(name, sizeof(name), "%s.weight", base);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    ds4_tensor *t = model_find_tensor(m, name);
    if (t) return t;

    /* Try NVFP4 .nvfp4_weight. */
    n = snprintf(name, sizeof(name), "%s.nvfp4_weight", base);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required expert tensor missing: tried '%s.weight' and '%s'\n",
                base, name);
        exit(1);
    }

    /* NVFP4: find the .nvfp4_scale_2 sibling and attach its data as the
     * per-expert scale_2 array. */
    n = snprintf(scale_name, sizeof(scale_name), "%s.nvfp4_scale_2", base);
    if (n < 0 || (size_t)n >= sizeof(scale_name)) ds4_die("tensor name is too long");
    ds4_tensor *st = model_find_tensor(m, scale_name);
    if (!st) {
        fprintf(stderr, "ds4: NVFP4 tensor %s has no .nvfp4_scale_2 sibling\n", name);
        exit(1);
    }
    /* st->abs_offset points into the mmap'd model data. Cast to float*. */
    t->nvfp4_scale_2 = (const float *)(m->map + st->abs_offset);
    /* The GGUF writer stores the NVFP4 weight tensor with the ds4q NVFP4 type id
     * (40), which is not in ds4's gguf_types table and does not match the CUDA
     * dispatch id (DS4_TENSOR_NVFP4 = 31). Force the dispatch type here: detection
     * is by name (.nvfp4_weight), and the layout is fixed (cuda_block_nvfp4). */
    t->type = DS4_TENSOR_NVFP4;
    return t;
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static void tensor_expect_plain_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != DS4_TENSOR_F16 && t->type != DS4_TENSOR_F32) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected F16 or F32\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_NVFP4;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    case DS4_TENSOR_NVFP4:   return 144;  /* cuda_block_nvfp4: 128 qs + 16 scales */
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) ds4_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

uint32_t layer_stored_expert_count(const ds4_layer_weights *l) {
    if (!l || !l->ffn_gate_inp) return DS4_N_EXPERT;
    if (l->ffn_gate_inp->ndim < 2) return DS4_N_EXPERT;
    if (l->ffn_gate_inp->dim[1] > UINT32_MAX) ds4_die("layer expert count exceeds u32");
    return (uint32_t)l->ffn_gate_inp->dim[1];
}

static uint32_t weights_expected_layer_experts(const ds4_weights *w, const ds4_layer_weights *l) {
    if (!w || !w->reap_compact_layout) return DS4_N_EXPERT;
    if (!l || l->reap_moe_disabled) return 0;
    if (l->reap_keep_count > 0) return l->reap_keep_count;
    if (l->reap_expert_count > 0) return l->reap_expert_count;
    return DS4_N_EXPERT;
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * After this succeeds, inference code can rely on fixed DS4 constants. */
static void weights_validate_layout(const ds4_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    tensor_expect_layout(w->token_embd,      DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_layout(w->output_hc_fn,    DS4_TENSOR_F16,  2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output,          DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);

        tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            tensor_expect_layout(l->attn_compressor_ape,  DS4_TENSOR_F16, 2, comp_width, ratio, 0);
            tensor_expect_layout(l->attn_compressor_kv,   DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_gate, DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_layout(l->indexer_attn_q_b,          DS4_TENSOR_F16, 2, DS4_N_LORA_Q, index_q_dim, 0);
            tensor_expect_layout(l->indexer_proj,              DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
            tensor_expect_layout(l->indexer_compressor_ape,    DS4_TENSOR_F16, 2, index_width, ratio, 0);
            tensor_expect_layout(l->indexer_compressor_kv,     DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_gate,   DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        const uint32_t n_layer_expert = weights_expected_layer_experts(w, l);
        tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_F16,  2, DS4_N_EMBD, n_layer_expert, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, n_layer_expert, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, n_layer_expert);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, n_layer_expert);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        const uint64_t shared_out = (w->reap_compact_layout && l->reap_moe_disabled) ? 0 : DS4_N_FF_EXP;
        const uint64_t shared_down_out = (w->reap_compact_layout && l->reap_moe_disabled) ? 0 : DS4_N_EMBD;
        tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, shared_out, 0);
        tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, shared_out, 0);
        tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_FF_EXP, shared_down_out, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

static void mtp_weights_validate_layout(const ds4_mtp_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const ds4_layer_weights *l = &w->block;

    tensor_expect_layout(w->hc_head_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_plain_layout(w->hc_head_fn, 2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->hc_head_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->e_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->h_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->enorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->norm,          DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);

    tensor_expect_plain_layout(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
    tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
    tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
    tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
    tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
    tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

    tensor_expect_plain_layout(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_plain_layout(l->ffn_gate_inp, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
    tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        ds4_die("MTP routed gate/up experts use different quant types");
    }
    tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u: got %u, expected %u\n",
                    il, got, expected);
            exit(1);
        }
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for DeepSeek4 Flash, got %u\n",
            name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for DeepSeek4 Flash, got %.9g\n",
            name, (double)expected, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for DeepSeek4 Flash, got %s\n",
            name, expected ? "true" : "false", got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
void config_validate_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    const uint64_t rope_orig_ctx = required_u64(m, "deepseek4.rope.scaling.original_context_length");
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for DeepSeek4 Flash, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    const float rope_scale_factor = required_f32(m, "deepseek4.rope.scaling.factor");
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    const float rope_yarn_beta_fast = required_f32(m, "deepseek4.rope.scaling.yarn_beta_fast");
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    const float rope_yarn_beta_slow = required_f32(m, "deepseek4.rope.scaling.yarn_beta_slow");
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

static void weights_apply_reap_metadata(ds4_weights *w, const ds4_model *m) {
    bool reap_enabled = false;
    if (!model_get_bool(m, "reap.enabled", &reap_enabled) || !reap_enabled) return;

    ds4_str layout = {0};
    if (!model_get_string(m, "reap.layout", &layout)) {
        ds4_die("reap.enabled is true but reap.layout is missing");
    }
    if (ds4_streq(layout, "ds4-compact-v1")) {
        w->reap_compact_layout = true;
    } else if (!ds4_streq(layout, "ds4-padded-v1")) {
        ds4_die("reap.enabled is true but reap.layout is neither ds4-compact-v1 nor ds4-padded-v1");
    }

    uint32_t policy[DS4_N_LAYER] = {0};
    uint32_t expert_count[DS4_N_LAYER] = {0};
    uint32_t keep_count[DS4_N_LAYER] = {0};
    if (!model_get_u32_array_exact(m, "reap.layer.policy", policy, DS4_N_LAYER)) {
        ds4_die("reap.enabled is true but reap.layer.policy metadata is missing");
    }
    (void)model_get_u32_array_exact(m, "reap.layer.expert_count", expert_count, DS4_N_LAYER);
    (void)model_get_u32_array_exact(m, "reap.layer.keep_count", keep_count, DS4_N_LAYER);

    uint32_t disabled = 0;
    uint32_t router_masked = 0;
    uint32_t hash_preserved = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *layer = &w->layer[il];
        layer->reap_policy = policy[il];
        layer->reap_expert_count = expert_count[il];
        layer->reap_keep_count = keep_count[il];
        layer->reap_moe_disabled = policy[il] == DS4_REAP_POLICY_MOE_DISABLED;

        if (layer->reap_moe_disabled && il < DS4_N_HASH_LAYER) {
            fprintf(stderr, "ds4: REAP metadata disables hash-routed layer %u; hash layers must be preserved\n", il);
            exit(1);
        }
        switch (policy[il]) {
            case DS4_REAP_POLICY_NONE:
                break;
            case DS4_REAP_POLICY_HASH_PRESERVED:
                hash_preserved++;
                break;
            case DS4_REAP_POLICY_ROUTER_MASK_PRUNED:
                router_masked++;
                break;
            case DS4_REAP_POLICY_MOE_DISABLED:
                disabled++;
                break;
            default:
                fprintf(stderr, "ds4: unsupported REAP layer policy %u at layer %u\n", policy[il], il);
                exit(1);
        }
    }

    fprintf(stderr,
            "ds4: REAP runtime metadata enabled: hash_preserved=%u router_masked=%u moe_disabled=%u layout=%.*s\n",
            hash_preserved, router_masked, disabled, (int)layout.len, layout.ptr);
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
void weights_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd       = required_tensor(m, "token_embd.weight");
    w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
    w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
    w->output_norm      = required_tensor(m, "output_norm.weight");
    w->output           = required_tensor(m, "output.weight");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

        l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
        l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
        l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
        l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
        l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
        l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
        l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
        if (compress_ratio != 0) {
            l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
            l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
            l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
            l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
        }
        if (compress_ratio == 4) {
            l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
            l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
            l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
            l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
            l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
            l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
        }
        l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
        l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
        l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
        l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps   = ds4_load_expert_tensor(m, "blk.%u.ffn_gate_exps", il);
        l->ffn_up_exps     = ds4_load_expert_tensor(m, "blk.%u.ffn_up_exps", il);
        l->ffn_down_exps   = ds4_load_expert_tensor(m, "blk.%u.ffn_down_exps", il);
        l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

        if (il < DS4_N_HASH_LAYER) {
            l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
        }
    }

    weights_apply_reap_metadata(w, m);
    weights_validate_layout(w);
}

void mtp_weights_bind(ds4_mtp_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));

    w->hc_head_base  = required_tensor(m, "mtp.0.hc_head_base.weight");
    w->hc_head_fn    = required_tensor(m, "mtp.0.hc_head_fn.weight");
    w->hc_head_scale = required_tensor(m, "mtp.0.hc_head_scale.weight");
    w->e_proj        = required_tensor(m, "mtp.0.e_proj.weight");
    w->h_proj        = required_tensor(m, "mtp.0.h_proj.weight");
    w->enorm         = required_tensor(m, "mtp.0.enorm.weight");
    w->hnorm         = required_tensor(m, "mtp.0.hnorm.weight");
    w->norm          = required_tensor(m, "mtp.0.norm.weight");

    ds4_layer_weights *l = &w->block;
    l->hc_attn_fn      = required_tensor(m, "mtp.0.hc_attn_fn.weight");
    l->hc_attn_scale   = required_tensor(m, "mtp.0.hc_attn_scale.weight");
    l->hc_attn_base    = required_tensor(m, "mtp.0.hc_attn_base.weight");
    l->attn_norm       = required_tensor(m, "mtp.0.attn_norm.weight");
    l->attn_q_a        = required_tensor(m, "mtp.0.attn_q_a.weight");
    l->attn_q_a_norm   = required_tensor(m, "mtp.0.attn_q_a_norm.weight");
    l->attn_q_b        = required_tensor(m, "mtp.0.attn_q_b.weight");
    l->attn_kv         = required_tensor(m, "mtp.0.attn_kv.weight");
    l->attn_kv_a_norm  = required_tensor(m, "mtp.0.attn_kv_a_norm.weight");
    l->attn_sinks      = required_tensor(m, "mtp.0.attn_sinks.weight");
    l->attn_output_a   = required_tensor(m, "mtp.0.attn_output_a.weight");
    l->attn_output_b   = required_tensor(m, "mtp.0.attn_output_b.weight");
    l->hc_ffn_fn       = required_tensor(m, "mtp.0.hc_ffn_fn.weight");
    l->hc_ffn_scale    = required_tensor(m, "mtp.0.hc_ffn_scale.weight");
    l->hc_ffn_base     = required_tensor(m, "mtp.0.hc_ffn_base.weight");
    l->ffn_norm        = required_tensor(m, "mtp.0.ffn_norm.weight");
    l->ffn_gate_inp    = required_tensor(m, "mtp.0.ffn_gate_inp.weight");
    l->ffn_exp_probs_b = required_tensor(m, "mtp.0.exp_probs_b.bias");
    l->ffn_gate_exps   = ds4_load_expert_tensor(m, "mtp.0.ffn_gate_exps", 0);
    l->ffn_up_exps     = ds4_load_expert_tensor(m, "mtp.0.ffn_up_exps", 0);
    l->ffn_down_exps   = ds4_load_expert_tensor(m, "mtp.0.ffn_down_exps", 0);
    l->ffn_gate_shexp  = required_tensor(m, "mtp.0.ffn_gate_shexp.weight");
    l->ffn_up_shexp    = required_tensor(m, "mtp.0.ffn_up_shexp.weight");
    l->ffn_down_shexp  = required_tensor(m, "mtp.0.ffn_down_shexp.weight");

    mtp_weights_validate_layout(w);
}

void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}
