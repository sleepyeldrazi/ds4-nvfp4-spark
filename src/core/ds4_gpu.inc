#ifndef DS4_NO_GPU
static int sample_argmax(const float *logits, uint32_t n_vocab);

/* =========================================================================
 * GPU Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the GPU executor.
 */

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}

static uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

#endif

static void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}

#ifndef DS4_NO_GPU
/* =========================================================================
 * GPU Release Graph State.
 * =========================================================================
 *
 * The release GPU executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

typedef struct {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_gpu_tensor *cur_hc;
    ds4_gpu_tensor *flat_hc;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *hc_pre;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *attn_cur;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  In DS4_KV_TURBO=packv4, attention-compressed
     * rows are stored in layer_attn_comp_tq_cache and materialized into the shared
     * unpack scratch only when an fp32 view is needed.  The small state tensors are
     * compressor frontiers for the next compressed row, so they must be snapshotted
     * with the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_comp_tq_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_comp_tq_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_N_LAYER];

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  The prefix1 buffers are the cheap partial-accept state for the
     * common N=2 case. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_N_LAYER];
    uint32_t layer_n_index_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_index_comp[DS4_N_LAYER];
    bool spec_capture_prefix1;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_N_LAYER];

    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *comp_mask;
    ds4_gpu_tensor *comp_selected;
    ds4_gpu_tensor *attn_comp_unpack_cache;
    ds4_gpu_tensor *index_comp_unpack_cache;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_low;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn_hc;
    ds4_gpu_tensor *ffn_cur;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *zero_embd;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *after_ffn_hc;
    ds4_gpu_tensor *output_pre;
    ds4_gpu_tensor *output_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    ds4_gpu_tensor *mtp_embed;
    ds4_gpu_tensor *mtp_enorm;
    ds4_gpu_tensor *mtp_eproj;
    ds4_gpu_tensor *mtp_eproj_hc;
    ds4_gpu_tensor *mtp_hnorm_hc;
    ds4_gpu_tensor *mtp_hproj_hc;
    ds4_gpu_tensor *mtp_input_hc;
    ds4_gpu_tensor *mtp_state_hc;
    ds4_gpu_tensor *mtp_next_hc;
    ds4_gpu_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur_hc;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor *batch_flat_hc;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_attn_cur;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv;
    ds4_gpu_tensor *batch_comp_kv;
    ds4_gpu_tensor *batch_comp_sc;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_low;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_group_tmp;
    ds4_gpu_tensor *batch_low_tmp;
    ds4_gpu_tensor *batch_after_attn_hc;
    ds4_gpu_tensor *batch_ffn_cur;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_shared_gate;
    ds4_gpu_tensor *batch_shared_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_shared_out;
    ds4_gpu_tensor *batch_zero_embd;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_mid;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_routed_out;
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out;
    bool materialize_ffn_out;
    ds4_gpu_tensor *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    bool mtp_enabled;
    bool turbo_packv4;
} ds4_gpu_graph;

/* Release every GPU tensor owned by the whole-model graph runtime. */
static void gpu_graph_free(ds4_gpu_graph *g) {
    ds4_gpu_tensor_free(g->directional_steering_dirs);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_routed_out);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_mid);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_zero_embd);
    ds4_gpu_tensor_free(g->batch_shared_out);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_shared_up);
    ds4_gpu_tensor_free(g->batch_shared_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_ffn_cur);
    ds4_gpu_tensor_free(g->batch_after_attn_hc);
    ds4_gpu_tensor_free(g->batch_low_tmp);
    ds4_gpu_tensor_free(g->batch_group_tmp);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_attn_low);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_comp_sc);
    ds4_gpu_tensor_free(g->batch_comp_kv);
    ds4_gpu_tensor_free(g->batch_kv);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_qr_norm);
    ds4_gpu_tensor_free(g->batch_qr);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_attn_cur);
    ds4_gpu_tensor_free(g->batch_hc_split);
    ds4_gpu_tensor_free(g->batch_hc_mix);
    ds4_gpu_tensor_free(g->batch_flat_hc);
    ds4_gpu_tensor_free(g->batch_next_hc);
    ds4_gpu_tensor_free(g->batch_cur_hc);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->mtp_raw_cache);
    ds4_gpu_tensor_free(g->mtp_next_hc);
    ds4_gpu_tensor_free(g->mtp_state_hc);
    ds4_gpu_tensor_free(g->mtp_input_hc);
    ds4_gpu_tensor_free(g->mtp_hproj_hc);
    ds4_gpu_tensor_free(g->mtp_hnorm_hc);
    ds4_gpu_tensor_free(g->mtp_eproj_hc);
    ds4_gpu_tensor_free(g->mtp_eproj);
    ds4_gpu_tensor_free(g->mtp_enorm);
    ds4_gpu_tensor_free(g->mtp_embed);
    ds4_gpu_tensor_free(g->spec_logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->zero_embd);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->comp_mask);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->attn_comp_unpack_cache);
    ds4_gpu_tensor_free(g->index_comp_unpack_cache);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_tq_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
        ds4_gpu_tensor_free(g->layer_index_comp_tq_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

static bool gpu_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    float *p = ds4_gpu_tensor_contents(t);
    if (!p || ds4_gpu_tensor_bytes(t) < n * sizeof(float)) return false;
    for (uint64_t i = 0; i < n; i++) p[i] = v;
    return true;
}

static ds4_gpu_tensor *gpu_graph_attn_comp_cache_for_write(
        ds4_gpu_graph *g,
        uint32_t       il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->turbo_packv4
        ? g->attn_comp_unpack_cache
        : g->layer_attn_comp_cache[il];
}

static bool gpu_graph_pack_attn_comp_rows(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row0,
        uint32_t       n_rows) {
    if (!g || !g->turbo_packv4 || n_rows == 0) return true;
    ds4_gpu_tensor *src = gpu_graph_attn_comp_cache_for_write(g, il);
    if (il >= DS4_N_LAYER || !src || !g->layer_attn_comp_tq_cache[il] ||
        (uint64_t)row0 + n_rows > g->layer_comp_cap[il]) {
        return false;
    }
    return ds4_gpu_dsv4_turbo4_pack_compressed_kv_tensor(
            g->layer_attn_comp_tq_cache[il],
            src,
            row0,
            row0,
            n_rows,
            DS4_N_HEAD_DIM,
            DS4_N_ROT) != 0;
}

static ds4_gpu_tensor *gpu_graph_attn_comp_cache_for_read(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_comp) {
    if (!g || il >= DS4_N_LAYER || n_comp == 0) return NULL;
    if (!g->turbo_packv4) return g->layer_attn_comp_cache[il];
    if (!g->layer_attn_comp_tq_cache[il] || !g->attn_comp_unpack_cache ||
        n_comp > g->layer_comp_cap[il] || n_comp > g->comp_cap) {
        return NULL;
    }
    if (ds4_gpu_dsv4_turbo4_unpack_compressed_kv_tensor(
            g->attn_comp_unpack_cache,
            g->layer_attn_comp_tq_cache[il],
            0,
            0,
            n_comp,
            DS4_N_HEAD_DIM,
            DS4_N_ROT) == 0) {
        return NULL;
    }
    return g->attn_comp_unpack_cache;
}

/* ---- Indexer-compressed KV: same packed-FP8 scheme (head_dim=128, n_rot=64) ----
 * The pack/unpack kernels are generic (take head_dim/n_rot params), so the
 * indexer reuses the exact same turbo4 kernels as the attention cache — only
 * the per-row byte count differs (200 B vs 584 B).  A separate small unpack
 * scratch (index_comp_unpack_cache) keeps the indexer and attention paths
 * independent so they cannot clobber each other within a layer. */
static ds4_gpu_tensor *gpu_graph_index_comp_cache_for_write(
        ds4_gpu_graph *g,
        uint32_t       il) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    return g->turbo_packv4
        ? g->index_comp_unpack_cache
        : g->layer_index_comp_cache[il];
}

static bool gpu_graph_pack_index_comp_rows(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row0,
        uint32_t       n_rows) {
    if (!g || !g->turbo_packv4 || n_rows == 0) return true;
    ds4_gpu_tensor *src = gpu_graph_index_comp_cache_for_write(g, il);
    if (il >= DS4_N_LAYER || !src || !g->layer_index_comp_tq_cache[il] ||
        (uint64_t)row0 + n_rows > g->layer_comp_cap[il]) {
        return false;
    }
    return ds4_gpu_dsv4_turbo4_pack_compressed_kv_tensor(
            g->layer_index_comp_tq_cache[il],
            src,
            row0,
            row0,
            n_rows,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_ROT) != 0;
}

static ds4_gpu_tensor *gpu_graph_index_comp_cache_for_read(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_comp) {
    if (!g || il >= DS4_N_LAYER || n_comp == 0) return NULL;
    if (!g->turbo_packv4) return g->layer_index_comp_cache[il];
    if (!g->layer_index_comp_tq_cache[il] || !g->index_comp_unpack_cache ||
        n_comp > g->layer_comp_cap[il] || n_comp > g->comp_cap) {
        return NULL;
    }
    if (ds4_gpu_dsv4_turbo4_unpack_compressed_kv_tensor(
            g->index_comp_unpack_cache,
            g->layer_index_comp_tq_cache[il],
            0,
            0,
            n_comp,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_ROT) == 0) {
        return NULL;
    }
    return g->index_comp_unpack_cache;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the GPU graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

static bool gpu_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        g->directional_steering_dirs = ds4_gpu_tensor_alloc(n * sizeof(dirs[0]));
        ok = g->directional_steering_dirs != NULL &&
             ds4_gpu_tensor_write(g->directional_steering_dirs, 0, dirs, n * sizeof(dirs[0])) != 0;
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

static bool gpu_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}

static bool gpu_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}

static bool gpu_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}

static bool gpu_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

static bool gpu_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

/* =========================================================================
 * GPU Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_GPU_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

static bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const char *prefix = getenv("DS4_GPU_GRAPH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return false;

    const char *name_env = getenv("DS4_GPU_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_GPU_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_GPU_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

static void gpu_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_GPU_GRAPH_DUMP_PREFIX");
    if (!t || n_f32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_GPU_GRAPH_DUMP_PREFIX");
    if (!t || n_i32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool gpu_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return gpu_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           gpu_graph_debug_wants("ffn_out", il, pos);
}

static bool gpu_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}

static bool gpu_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}

/* =========================================================================
 * GPU Release Graph Allocation.
 * ========================================================================= */

/* Allocate the GPU graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool gpu_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp) {
    memset(g, 0, sizeof(*g));
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    g->turbo_packv4 = ds4_gpu_dsv4_turbo4_packv4_enabled();
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = ds4_gpu_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            if (g->turbo_packv4) {
                g->layer_attn_comp_tq_cache[il] = ds4_gpu_tensor_alloc(
                        ds4_gpu_dsv4_turbo4_packed_kv_bytes(g->layer_comp_cap[il],
                                                             DS4_N_HEAD_DIM,
                                                             DS4_N_ROT));
            } else {
                g->layer_attn_comp_cache[il] = ds4_gpu_tensor_alloc(
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM * sizeof(float));
            }
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_mtp) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                if (g->turbo_packv4) {
                    g->layer_index_comp_tq_cache[il] = ds4_gpu_tensor_alloc(
                            ds4_gpu_dsv4_turbo4_packed_kv_bytes(g->layer_comp_cap[il],
                                                                 DS4_N_INDEXER_HEAD_DIM,
                                                                 DS4_N_ROT));
                } else {
                    g->layer_index_comp_cache[il] = ds4_gpu_tensor_alloc((uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                }
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_mtp) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    if (g->turbo_packv4) {
        g->attn_comp_unpack_cache = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->index_comp_unpack_cache = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    }
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->zero_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = ds4_gpu_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
        g->mtp_n_raw = 0;
    }

    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_zero_embd = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool zero_init_ok = true;
    if (g->zero_embd) {
        zero_init_ok = zero_init_ok &&
                       gpu_tensor_fill_f32(g->zero_embd, 0.0f, DS4_N_EMBD);
    }
    if (g->batch_zero_embd) {
        zero_init_ok = zero_init_ok &&
                       gpu_tensor_fill_f32(g->batch_zero_embd, 0.0f, pc * DS4_N_EMBD);
    }

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = (g->turbo_packv4
                                 ? g->layer_attn_comp_tq_cache[il] != NULL
                                 : g->layer_attn_comp_cache[il] != NULL) &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL &&
                               g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = (g->turbo_packv4
                                 ? g->layer_index_comp_tq_cache[il] != NULL
                                 : g->layer_index_comp_cache[il] != NULL) &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL &&
                               g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && zero_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    (!g->turbo_packv4 || (g->attn_comp_unpack_cache && g->index_comp_unpack_cache)) &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out && g->zero_embd &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache && g->spec_logits)) &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out && g->batch_zero_embd &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) gpu_graph_free(g);
    return ok;
}

static bool gpu_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return gpu_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false);
}

static uint32_t gpu_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t gpu_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool gpu_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool gpu_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

static uint32_t gpu_graph_decode_indexer_top_k(const ds4_gpu_graph *g) {
    (void)g;
    static int cached = 0;
    static uint32_t value = DS4_N_INDEXER_TOP_K;
    if (!cached) {
        const char *env = getenv("DS4_DECODE_INDEXER_TOP_K");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long parsed = strtoul(env, &end, 10);
            if (end != env && *end == '\0' &&
                parsed >= 1ul &&
                parsed <= DS4_N_INDEXER_TOP_K &&
                (parsed & (parsed - 1ul)) == 0ul) {
                value = (uint32_t)parsed;
            } else {
                fprintf(stderr,
                        "ds4: ignoring DS4_DECODE_INDEXER_TOP_K=%s (must be a power of two in [1,%u])\n",
                        env,
                        DS4_N_INDEXER_TOP_K);
            }
        }
        cached = 1;
    }
    return value;
}

/* =========================================================================
 * GPU Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_GPU_DISABLE_*_FUSION environment switches.
 */

static bool gpu_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

static bool gpu_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_HC_FUSION", &cache);
}

static bool gpu_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_KV_FUSION", &cache);
}

static bool gpu_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool gpu_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool gpu_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_HC_NORM_FUSION", &cache);
}

static bool gpu_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool gpu_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_GPU_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool gpu_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (gpu_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

static bool gpu_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (gpu_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

/* Encode one DS4 decode layer on GPU.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static bool gpu_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = getenv("DS4_GPU_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_GPU_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        !gpu_graph_use_reference_hc_decode() &&
        !gpu_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_GPU_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_GPU_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->qr, model->map, model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD, q_rank,
                                              g->attn_norm, 1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qnorm", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_GPU_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = gpu_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    DS4_GPU_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    bool comp_cache_turbo4_direct = false;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("DS4_GPU_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: GPU graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        ds4_gpu_tensor *attn_comp_write_cache = NULL;
        if (ok) {
            attn_comp_write_cache = gpu_graph_attn_comp_cache_for_write(g, il);
            if (!attn_comp_write_cache) ok = false;
        }
        if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
            ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                  g->comp_sc_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_compressor_kv->abs_offset,
                                                  layer->attn_compressor_gate->abs_offset,
                                                  DS4_N_EMBD,
                                                  comp_width,
                                                  g->attn_norm,
                                                  1) != 0;
        } else {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        attn_comp_write_cache,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        comp_row,
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS) != 0;
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                    attn_comp_write_cache,
                    (uint64_t)comp_row * DS4_N_HEAD_DIM * sizeof(float),
                    (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
            if (!comp_row_view) {
                ok = false;
            } else {
                /* skip fp8 quantize for now - write float values directly */
                (void)comp_row_view;
                if (ok) {
                    ok = gpu_graph_pack_attn_comp_rows(g, il, comp_row, 1);
                }
                if (ok) {
                    gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
                ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: GPU graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->indexer_compressor_kv->abs_offset,
                                                      layer->indexer_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      index_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            ds4_gpu_tensor *index_comp_write_cache = gpu_graph_index_comp_cache_for_write(g, il);
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            index_comp_write_cache,
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
            if (ok && emit && g->turbo_packv4) {
                ok = gpu_graph_pack_index_comp_rows(g, il, index_row, 1);
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_top_k = gpu_graph_decode_indexer_top_k(g);
            if (ok && g->layer_n_comp[il] > decode_top_k) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: GPU graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: GPU graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_q, model->map, model->size,
                                                         layer->indexer_attn_q_b->abs_offset,
                                                         q_rank, indexer_q_dim,
                                                         g->qr_norm, 1) != 0;
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_weights, model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         g->attn_norm, 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                gpu_graph_index_comp_cache_for_read(g, il, g->layer_n_index_comp[il]),
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           decode_top_k) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    n_selected = decode_top_k < g->layer_n_index_comp[il]
                        ? decode_top_k
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache_turbo4_direct = g->turbo_packv4 &&
            n_comp != 0 &&
            comp_selected != NULL &&
            n_selected != 0 &&
            g->layer_attn_comp_tq_cache[il] != NULL;
        comp_cache = n_comp != 0
            ? (comp_cache_turbo4_direct
                ? g->layer_attn_comp_tq_cache[il]
                : gpu_graph_attn_comp_cache_for_read(g, il, n_comp))
            : NULL;
        if (n_comp != 0 && !comp_cache) ok = false;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            if (comp_cache_turbo4_direct) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_turbo4_tensor(
                        g->heads,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        g->q,
                        raw_cache,
                        comp_cache,
                        comp_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        DS4_N_HEAD,
                        DS4_N_HEAD_DIM) != 0;
            } else {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        g->heads,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        g->q,
                        raw_cache,
                        comp_cache,
                        comp_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        DS4_N_HEAD,
                        DS4_N_HEAD_DIM) != 0;
            }
            if (ok && decode_index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    DS4_GPU_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc =
        !gpu_graph_directional_steering_attn_enabled(g) &&
        !gpu_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_q8_tensor(g->attn_low,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      g->heads) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(g->attn_out,
                                                        g->attn_low,
                                                        g->batch_group_tmp,
                                                        g->batch_low_tmp,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        g->heads, 1) != 0;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_GPU_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && layer->reap_moe_disabled) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->zero_embd,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
        DS4_GPU_PROFILE_DECODE_STAGE("ffn_disabled_hc_post");
        if (ok) {
            gpu_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
        }
        return ok;
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_GPU_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint32_t router_experts = layer_stored_expert_count(layer);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, router_experts, g->ffn_norm, 1);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                router_experts,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    DS4_GPU_PROFILE_DECODE_STAGE("router");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, router_experts, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, router_experts, il, pos);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 model->map, model->size,
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 router_experts,
                                                 g->router_selected, g->router_weights,
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP,
                                                 layer->ffn_gate_exps ? layer->ffn_gate_exps->nvfp4_scale_2 : NULL,
                                                 layer->ffn_up_exps ? layer->ffn_up_exps->nvfp4_scale_2 : NULL,
                                                 layer->ffn_down_exps ? layer->ffn_down_exps->nvfp4_scale_2 : NULL,
                                                 g->ffn_norm) != 0;
    DS4_GPU_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    const bool fuse_shared_gate_up =
        !g->quality &&
        getenv("DS4_GPU_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm) != 0;
    } else {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up, shared_dim, 0.0f, 1.0f) != 0;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("shared_gate_up");
    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_down_hc =
        !keep_ffn_out && !gpu_graph_use_reference_shared_down_hc();
    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_GPU_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_GPU_PROFILE_DECODE_STAGE
    if (!ok) {
        return false;
    }
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on GPU. */
static bool gpu_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
static bool gpu_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: GPU plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

static bool gpu_graph_encode_output_head_mtp(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->output_pre, mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, g->flat_hc, 1);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    return ok;
}

/* =========================================================================
 * GPU Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * GPU tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void gpu_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(g->attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_gate, 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_up, 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_mid, 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(g->router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(g->routed_mid, 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: GPU stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: GPU shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: GPU routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: GPU stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

static int gpu_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: GPU graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc(&g, weights, layer);
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = gpu_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_gpu_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(g.after_ffn_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: GPU graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: GPU graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("GPU graph q", gpu_q, q_dim);
        print_vec_stats("GPU graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("GPU graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: GPU graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after graph test failure also failed\n");
        }
    }

    gpu_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}

static int gpu_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full GPU graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc(&g, weights, &weights->layer[0]);
    const bool trace_layers = getenv("DS4_GPU_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_GPU_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_GPU_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(g.cur_hc, 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: GPU full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    gpu_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = gpu_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
        }

        if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: GPU full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: GPU full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after full graph failure also failed\n");
        }
    }

    gpu_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}

/* =========================================================================
 * GPU Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing GPU backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

/* Encode a full single-token decode step on GPU.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool gpu_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: GPU graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("DS4_GPU_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

static ds4_gpu_tensor *gpu_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool gpu_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool gpu_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0 || n_tokens < 4) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                         model->map,
                                         model->size,
                                         kv_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                         model->map,
                                         model->size,
                                         score_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool gpu_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the GPU get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
static bool gpu_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    uint32_t gpu_min = 512;
    const char *gpu_min_env = getenv("DS4_GPU_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }

    if (tokens && n_tokens >= gpu_min) {
        return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }

    return gpu_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}

static bool gpu_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed || getenv("DS4_GPU_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay GPU's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_hc_mix,
                                         model->map,
                                         model->size,
                                         weights->layer[0].hc_attn_fn->abs_offset,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: GPU prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: GPU indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current GPU
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: GPU layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool gpu_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: GPU Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool gpu_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const bool index_stage_profile = getenv("DS4_GPU_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = getenv("DS4_GPU_LAYER_STAGE_PROFILE") != NULL;
    const bool q_stage_profile = getenv("DS4_GPU_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_GPU_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_GPU_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = gpu_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_attn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_cur_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                       g->batch_attn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("norm");
    DS4_GPU_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_qr,
                                              model->map,
                                              model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD,
                                              q_rank,
                                              g->batch_attn_norm,
                                              n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_GPU_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HEAD_DIM,
                                                  g->batch_attn_norm,
                                                  n_tokens) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_GPU_PROFILE_Q_STAGE("q_a_norm");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q,
                                              model->map,
                                              model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank,
                                              q_dim,
                                              g->batch_qr_norm,
                                              n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qraw", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_GPU_PROFILE_Q_STAGE("q_b");
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qnorm", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_GPU_PROFILE_Q_STAGE("head_norm");
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qcur", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_GPU_PROFILE_Q_STAGE("rope");
    DS4_GPU_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HEAD_DIM,
                                                  g->batch_attn_norm,
                                                  n_tokens) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;
    ds4_gpu_tensor *attn_comp_cache = NULL;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: GPU layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        ds4_gpu_tensor *attn_comp_write_cache = NULL;
        if (ok) {
            attn_comp_write_cache = gpu_graph_attn_comp_cache_for_write(g, il);
            if (!attn_comp_write_cache) ok = false;
        }
        if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                 model->map,
                                                 model->size,
                                                 layer->attn_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 comp_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                 model->map,
                                                 model->size,
                                                 layer->attn_compressor_gate->abs_offset,
                                                 DS4_N_EMBD,
                                                 comp_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: GPU layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_compressor_prefill_tensor(attn_comp_write_cache,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                ok = gpu_graph_pack_attn_comp_rows(g, il, 0, n_comp);
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_write_cache,
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                gpu_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                gpu_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
        } else {
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *comp_view = NULL;
                if (ok) {
                    comp_view = ds4_gpu_tensor_view(attn_comp_write_cache,
                                                      (uint64_t)comp_before * DS4_N_HEAD_DIM * sizeof(float),
                                                      (uint64_t)comp_chunk * DS4_N_HEAD_DIM * sizeof(float));
                    ok = comp_view != NULL;
                }
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            comp_view,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            comp_view,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    ok = gpu_graph_pack_attn_comp_rows(g, il, comp_before, comp_chunk);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  comp_view,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                ds4_gpu_tensor_free(comp_view);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: GPU graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    ds4_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            attn_comp_write_cache,
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            comp_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                                attn_comp_write_cache,
                                (uint64_t)comp_row * DS4_N_HEAD_DIM * sizeof(float),
                                (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
                        ok = comp_row_view;
                        (void)comp_row_view; /* skip fp8 quantize */
                        if (ok) {
                            ok = gpu_graph_pack_attn_comp_rows(g, il, comp_row, 1);
                        }
                        if (ok) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t == 0) ok = gpu_graph_capture_prefix1_attn_state(g, il);
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        DS4_GPU_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: GPU layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_compressor_kv->abs_offset,
                                                     DS4_N_EMBD,
                                                     index_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_compressor_gate->abs_offset,
                                                     DS4_N_EMBD,
                                                     index_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: GPU layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ds4_gpu_tensor *idx_write = gpu_graph_index_comp_cache_for_write(g, il);
                    ok = ds4_gpu_compressor_prefill_tensor(idx_write,
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                    if (ok && g->turbo_packv4) {
                        ok = gpu_graph_pack_index_comp_rows(g, il, 0, n_comp);
                    }
                }
                if (ok) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                gpu_graph_index_comp_cache_for_write(g, il),
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                        if (ok && g->turbo_packv4) {
                            ok = gpu_graph_pack_index_comp_rows(g, il, index_before, index_chunk);
                        }
                    }
                    if (ok) {
                        ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: GPU graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        ds4_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ds4_gpu_tensor *idx_write = gpu_graph_index_comp_cache_for_write(g, il);
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                idx_write,
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS) != 0;
                        if (ok && emit && g->turbo_packv4) {
                            ok = gpu_graph_pack_index_comp_rows(g, il, index_row, 1);
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = gpu_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_GPU_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && n_comp != 0) {
            attn_comp_cache = gpu_graph_attn_comp_cache_for_read(g, il, n_comp);
            if (!attn_comp_cache) ok = false;
        }

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                  g->batch_indexer_q,
                                                                  g->batch_indexer_weights,
                                                                  gpu_graph_index_comp_cache_for_read(g, il, n_comp),
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    gpu_graph_debug_dump_tensor("indexer_scores",
                                                  g->indexer_scores,
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                          g->comp_selected,
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                            model->map,
                                                                            model->size,
                                                                            layer->attn_sinks->abs_offset,
                                                                            g->batch_q,
                                                                            g->layer_raw_cache[il],
                                                                            attn_comp_cache,
                                                                            g->comp_selected,
                                                                            n_tokens,
                                                                            pos0,
                                                                            n_raw,
                                                                            g->raw_cap,
                                                                            raw_start,
                                                                            n_comp,
                                                                            DS4_N_INDEXER_TOP_K,
                                                                            g->raw_window,
                                                                            ratio,
                                                                            DS4_N_HEAD,
                                                                            DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                           model->map,
                                                                           model->size,
                                                                           layer->attn_sinks->abs_offset,
                                                                           g->batch_q,
                                                                           g->layer_raw_cache[il],
                                                                           attn_comp_cache,
                                                                           use_comp_mask ? g->comp_mask : NULL,
                                                                           use_comp_mask,
                                                                           n_tokens,
                                                                           pos0,
                                                                           n_raw,
                                                                           g->raw_cap,
                                                                           raw_start,
                                                                           n_comp,
                                                                           g->raw_window,
                                                                           ratio,
                                                                           DS4_N_HEAD,
                                                                           DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4_gpu_indexer_scores_prefill_tensor(g->indexer_scores,
                                                         g->batch_indexer_q,
                                                         g->batch_indexer_weights,
                                                         gpu_graph_index_comp_cache_for_read(g, il, n_comp),
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok && index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
                gpu_graph_debug_dump_tensor("indexer_scores",
                                              g->indexer_scores,
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                      g->comp_selected,
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                        model->map,
                                                                        model->size,
                                                                        layer->attn_sinks->abs_offset,
                                                                        g->batch_q,
                                                                        g->layer_raw_cache[il],
                                                                        attn_comp_cache,
                                                                        g->comp_selected,
                                                                        n_tokens,
                                                                        pos0,
                                                                        n_tokens,
                                                                        g->raw_cap,
                                                                        0,
                                                                        n_comp,
                                                                        DS4_N_INDEXER_TOP_K,
                                                                        g->raw_window,
                                                                        ratio,
                                                                        DS4_N_HEAD,
                                                                        DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                     model->map,
                                                                     model->size,
                                                                     layer->attn_sinks->abs_offset,
                                                                     g->batch_q,
                                                                     g->batch_kv,
                                                                     attn_comp_cache,
                                                                     n_tokens,
                                                                     n_comp,
                                                                     g->raw_window,
                                                                     ratio,
                                                                     DS4_N_HEAD,
                                                                     DS4_N_HEAD_DIM) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            gpu_graph_index_comp_cache_for_read(g, il, cur_index),
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = gpu_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = gpu_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = gpu_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                            model->map,
                                                                            model->size,
                                                                            layer->attn_sinks->abs_offset,
                                                                            q_view,
                                                                            g->layer_raw_cache[il],
                                                                            attn_comp_cache,
                                                                            g->comp_selected,
                                                                            1,
                                                                            pos,
                                                                            n_raw,
                                                                            g->raw_cap,
                                                                            raw_start,
                                                                            cur_comp,
                                                                            n_selected,
                                                                            g->raw_window,
                                                                            ratio,
                                                                            DS4_N_HEAD,
                                                                            DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                               model->map,
                                                               model->size,
                                                               layer->attn_sinks->abs_offset,
                                                               q_view,
                                                               g->layer_raw_cache[il],
                                                               n_raw,
                                                               g->raw_cap,
                                                               raw_start,
                                                               cur_comp ? attn_comp_cache : NULL,
                                                               cur_comp,
                                                               comp_mask,
                                                               n_selected,
                                                               DS4_N_HEAD,
                                                               DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    DS4_GPU_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("inv_rope");
    if (ok) ok = ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out,
                                                            g->batch_attn_low,
                                                            g->batch_group_tmp,
                                                            g->batch_low_tmp,
                                                            model->map,
                                                            model->size,
                                                            layer->attn_output_a->abs_offset,
                                                            layer->attn_output_b->abs_offset,
                                                            group_dim,
                                                            rank,
                                                            n_groups,
                                                            DS4_N_EMBD,
                                                            g->batch_heads,
                                                            n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("output_proj");
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_GPU_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef DS4_GPU_PROFILE_ATTN_STAGE
#undef DS4_GPU_PROFILE_Q_STAGE
    return ok;
}

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
static bool gpu_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint32_t router_experts = layer_stored_expert_count(layer);
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = getenv("DS4_GPU_LAYER_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_GPU_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_FFN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                       g->batch_ffn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_FFN_STAGE("norm");
    if (ok && layer->reap_moe_disabled) {
        ds4_gpu_tensor *zero_view = ds4_gpu_tensor_view(
                g->batch_zero_embd, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
        ok = zero_view != NULL &&
             ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                            zero_view,
                                            g->batch_after_attn_hc,
                                            hc_split_view,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        ds4_gpu_tensor_free(zero_view);
        if (ok) {
            gpu_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                          (uint64_t)n_tokens * hc_dim, il, pos0);
        }
        DS4_GPU_PROFILE_FFN_STAGE("ffn_disabled_hc_post");
        ds4_gpu_tensor_free(next_hc_view);
        ds4_gpu_tensor_free(ffn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_mix_view);
        return ok;
    }
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_router_logits,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_inp->abs_offset,
                                             DS4_N_EMBD,
                                             router_experts,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      router_experts,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * router_experts, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * router_experts, il, pos0);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_GPU_PROFILE_FFN_STAGE("router");

    if (ok) ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                                   g->batch_routed_gate,
                                                   g->batch_routed_up,
                                                   g->batch_routed_mid,
                                                   g->batch_routed_down,
                                                   model->map,
                                                   model->size,
                                                   layer->ffn_gate_exps->abs_offset,
                                                   layer->ffn_up_exps->abs_offset,
                                                   layer->ffn_down_exps->abs_offset,
                                                   layer->ffn_gate_exps->type,
                                                   layer->ffn_down_exps->type,
                                                   gate_expert_bytes,
                                                   gate_row_bytes,
                                                   down_expert_bytes,
                                                   down_row_bytes,
                                                   (uint32_t)expert_in_dim,
                                                   (uint32_t)down_in_dim,
                                                   (uint32_t)routed_out_dim,
                                                   router_experts,
                                                   g->batch_router_selected,
                                                   g->batch_router_weights,
                                                   DS4_N_EXPERT_USED,
                                                   DS4_SWIGLU_CLAMP_EXP,
                                                   layer->ffn_gate_exps ? layer->ffn_gate_exps->nvfp4_scale_2 : NULL,
                                                   layer->ffn_up_exps ? layer->ffn_up_exps->nvfp4_scale_2 : NULL,
                                                   layer->ffn_down_exps ? layer->ffn_down_exps->nvfp4_scale_2 : NULL,
                                                   g->batch_ffn_norm,
                                                   n_tokens,
                                                   &g->batch_routed_mid_is_f16) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_GPU_PROFILE_FFN_STAGE("routed_moe");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_gate,
                                              model->map,
                                              model->size,
                                              layer->ffn_gate_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              shared_dim,
                                              g->batch_ffn_norm,
                                              n_tokens) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_up,
                                              model->map,
                                              model->size,
                                              layer->ffn_up_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              shared_dim,
                                              g->batch_ffn_norm,
                                              n_tokens) != 0;
    DS4_GPU_PROFILE_FFN_STAGE("shared_gate_up");
    if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                         g->batch_shared_gate,
                                         g->batch_shared_up,
                                         (uint32_t)((uint64_t)n_tokens * shared_dim),
                                         0.0f,
                                         1.0f) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_out,
                                              model->map,
                                              model->size,
                                              layer->ffn_down_shexp->abs_offset,
                                              shared_dim,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens) != 0;
    DS4_GPU_PROFILE_FFN_STAGE("shared_down");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }

    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos0);
    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_GPU_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_GPU_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static bool gpu_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = gpu_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) {
        ok = gpu_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    }
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    return ok;
}

/* Execute one GPU decode token and read back logits. */
static bool gpu_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile = getenv("DS4_GPU_GRAPH_TOKEN_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? now_sec() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: GPU graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
static bool gpu_graph_eval_token_raw_swa_top(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits) {
    if (!top_id) return false;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights,
                                                  token, pos, true, true);
    if (ok) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                           g->logits,
                                           DS4_N_VOCAB,
                                           1,
                                           1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after top-only graph eval failure also failed\n");
        }
    }
    return ok;
}

static bool gpu_graph_eval_mtp_draft_from_hc(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        ds4_gpu_tensor      *prev_hc,
        ds4_gpu_tensor      *out_hc,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    if (!mtp || !mtp->block.attn_q_a || !g->mtp_raw_cache || !prev_hc || !out_hc) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t raw_row = pos % g->raw_cap;
    uint32_t n_raw = g->mtp_n_raw + 1u;
    if (n_raw > g->raw_window) n_raw = g->raw_window;
    if (n_raw > g->raw_cap) n_raw = g->raw_cap;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g->mtp_embed,
                                                  base_model->map,
                                                  base_model->size,
                                                  base_weights->token_embd->abs_offset,
                                                  (uint32_t)base_weights->token_embd->dim[1],
                                                  (uint32_t)token,
                                                  DS4_N_EMBD,
                                                  1) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->mtp_enorm,
                                                  g->mtp_embed,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->enorm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_eproj,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->e_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_enorm,
                                              1) != 0;
    if (ok) ok = ds4_gpu_repeat_hc_tensor(g->mtp_eproj_hc,
                                            g->mtp_eproj,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mtp_hnorm_hc,
                                                       prev_hc,
                                                       mtp_model->map,
                                                       mtp_model->size,
                                                       mtp->hnorm->abs_offset,
                                                       DS4_N_EMBD,
                                                       DS4_N_HC,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_hproj_hc,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->h_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_hnorm_hc,
                                              DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_add_tensor(g->mtp_input_hc,
                                      g->mtp_eproj_hc,
                                      g->mtp_hproj_hc,
                                      (uint32_t)hc_dim) != 0;
    if (ok) {
        g->cur_hc = g->mtp_input_hc;
        g->after_ffn_hc = out_hc;
        ok = gpu_graph_encode_decode_layer(g,
                                             mtp_model,
                                             &mtp->block,
                                             1,
                                             pos,
                                             g->mtp_raw_cache,
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
    }
    if (ok) g->cur_hc = out_hc;
    if (ok) ok = gpu_graph_encode_output_head_mtp(g,
                                                    base_model,
                                                    base_weights,
                                                    mtp_model,
                                                    mtp,
                                                    base_weights->output->dim[1]);
    if (ok && top_id) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                           g->logits,
                                           DS4_N_VOCAB,
                                           1,
                                           1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && top_id) {
        ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    }
    if (ok && g->mtp_n_raw < g->raw_window) g->mtp_n_raw++;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        g->after_ffn_hc = saved_after;
    }
    return ok;
}

static bool gpu_graph_eval_mtp_draft(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    return gpu_graph_eval_mtp_draft_from_hc(g,
                                              base_model,
                                              base_weights,
                                              mtp_model,
                                              mtp,
                                              g->cur_hc,
                                              g->mtp_state_hc,
                                              token,
                                              pos,
                                              logits,
                                              top_id);
}

/* =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During GPU prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */
typedef struct {
    float *gate_up_sum2;   /* [layer][expert][4096] */
    float *down_sum2;      /* [layer][expert][2048] */
    uint32_t gate_up_count[DS4_N_LAYER][DS4_N_EXPERT];
    uint32_t down_count[DS4_N_LAYER][DS4_N_EXPERT];
    float *ffn_norm_buf;
    float *routed_mid_buf;
    uint16_t *routed_mid_f16_buf;
    int   *selected_buf;
    float *sq_tmp;
    uint32_t cap_tokens;
    uint64_t observed_tokens;
    uint64_t observed_routes;
    uint32_t chunks;
    const char *dataset_path;
} ds4_imatrix_collector;

static bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_EMBD;
    const size_t down_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_FF_EXP;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}

static void imatrix_collector_free(ds4_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}

static float *imatrix_gate_up_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
}

static float *imatrix_down_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
}

static bool imatrix_collect_layer_batch(
        ds4_imatrix_collector *c,
        ds4_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_routed_mid, 0, mid_dst, mid_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_router_selected, 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * DS4_N_EMBD;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * DS4_N_EXPERT_USED + slot];
            if (expert < 0 || expert >= DS4_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * DS4_N_EXPERT_USED + slot) * DS4_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}

static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) ds4_die("failed to write imatrix");
}

static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) ds4_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) ds4_die("failed to write imatrix values");
    }
    free(tmp);
}

static bool imatrix_collector_save(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(DS4_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_FF_EXP,
                            c->down_count[il],
                            DS4_N_EXPERT,
                            DS4_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        ds4_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static void reap_json_write_escaped(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
            case '\\': fputs("\\\\", fp); break;
            case '"':  fputs("\\\"", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) {
                    fprintf(fp, "\\u%04x", (unsigned)*p);
                } else {
                    fputc((int)*p, fp);
                }
                break;
        }
    }
    fputc('"', fp);
}

static double reap_sum_float_energy(const float *v, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)v[i];
    return sum;
}

static void reap_json_write_u32_array(FILE *fp, const uint32_t *v, uint32_t n) {
    fputc('[', fp);
    for (uint32_t i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "%u", v[i]);
    }
    fputc(']', fp);
}

static void reap_json_write_layer_energy_array(
        FILE                      *fp,
        const ds4_imatrix_collector *c,
        uint32_t                    il,
        uint32_t                    n_expert,
        bool                        include_gate_up,
        bool                        include_down) {
    fputc('[', fp);
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        if (expert) fputc(',', fp);
        double score = 0.0;
        if (include_gate_up) {
            const float *gate = c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
            score += reap_sum_float_energy(gate, DS4_N_EMBD);
        }
        if (include_down) {
            const float *down = c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
            score += reap_sum_float_energy(down, DS4_N_FF_EXP);
        }
        if (!isfinite(score)) score = 0.0;
        fprintf(fp, "%.9g", score);
    }
    fputc(']', fp);
}

static bool reap_observation_collector_save_json(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path,
        int                          prompts_done,
        int                          tokens_done) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open REAP observation output %s: %s\n", path, strerror(errno));
        return false;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"format\": \"ds4_reap_observation\",\n");
    fprintf(fp, "  \"format_version\": 1,\n");
    fprintf(fp, "  \"source\": \"ds4_reap\",\n");
    fprintf(fp, "  \"dataset_path\": ");
    reap_json_write_escaped(fp, c->dataset_path ? c->dataset_path : "");
    fprintf(fp, ",\n");
    fprintf(fp, "  \"prompts\": %d,\n", prompts_done);
    fprintf(fp, "  \"prompt_tokens\": %d,\n", tokens_done);
    fprintf(fp, "  \"observed_layer_tokens\": %llu,\n", (unsigned long long)c->observed_tokens);
    fprintf(fp, "  \"observed_routes\": %llu,\n", (unsigned long long)c->observed_routes);
    fprintf(fp, "  \"n_layers\": %u,\n", DS4_N_LAYER);
    fprintf(fp, "  \"default_expert_count\": %u,\n", DS4_N_EXPERT);
    fprintf(fp, "  \"num_experts_per_tok\": %u,\n", DS4_N_EXPERT_USED);
    fprintf(fp, "  \"score_metric\": \"activation_energy_sum2\",\n");
    fprintf(fp, "  \"layers\": {\n");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        uint32_t n_expert = layer_stored_expert_count(layer);
        if (n_expert > DS4_N_EXPERT) n_expert = DS4_N_EXPERT;
        const uint32_t *freq = c->gate_up_count[il];
        uint64_t total_routes = 0;
        for (uint32_t expert = 0; expert < n_expert; expert++) total_routes += freq[expert];
        const uint64_t total_tokens = DS4_N_EXPERT_USED ? total_routes / DS4_N_EXPERT_USED : 0;

        fprintf(fp, "    \"%u\": {\n", il);
        fprintf(fp, "      \"layer\": %u,\n", il);
        fprintf(fp, "      \"expert_count\": %u,\n", n_expert);
        fprintf(fp, "      \"total_tokens\": %llu,\n", (unsigned long long)total_tokens);
        fprintf(fp, "      \"total_routes\": %llu,\n", (unsigned long long)total_routes);
        fprintf(fp, "      \"expert_frequency\": ");
        reap_json_write_u32_array(fp, freq, n_expert);
        fprintf(fp, ",\n");
        fprintf(fp, "      \"gate_up_energy\": ");
        reap_json_write_layer_energy_array(fp, c, il, n_expert, true, false);
        fprintf(fp, ",\n");
        fprintf(fp, "      \"down_energy\": ");
        reap_json_write_layer_energy_array(fp, c, il, n_expert, false, true);
        fprintf(fp, ",\n");
        fprintf(fp, "      \"reap\": ");
        reap_json_write_layer_energy_array(fp, c, il, n_expert, true, true);
        fprintf(fp, "\n");
        fprintf(fp, "    }%s\n", il + 1 == DS4_N_LAYER ? "" : ",");
    }

    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close REAP observation output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static bool gpu_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute GPU prefill in layer-major order so intermediate activations stay
 * on the GPU and cache state is built exactly once. */
static bool gpu_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix) {
    if (n_tokens <= 0 || n_tokens > prompt->len || (uint32_t)n_tokens > g->prefill_cap) return false;
    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, 0, (uint32_t)n_tokens);
    if (!ok) return false;

    if (!gpu_graph_warmup_prefill_kernels(g, model, weights, (uint32_t)n_tokens)) return false;

    const bool split_profile = getenv("DS4_GPU_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy long enough for macOS
     * to watchdog WindowServer. Keep short prompts in one command buffer for
     * low overhead, but submit long prompts layer by layer so the display
     * server gets regular scheduling points.
     */
    const bool split_commands = split_profile || n_tokens > 2048 || imatrix != NULL;
    const bool profile = getenv("DS4_GPU_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    if (!split_commands) {
        ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     0,
                                                     (uint32_t)n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = gpu_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                0,
                                                (uint32_t)n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("DS4_GPU_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        ds4_gpu_tensor *last_hc = NULL;
        ds4_gpu_tensor *saved_cur = g->cur_hc;
        if (ok) {
            last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok) {
            g->cur_hc = last_hc;
            ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: GPU synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%d encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    double t_layer0 = profile ? now_sec() : 0.0;
    ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 0,
                                                 (uint32_t)n_tokens);
    const double t_embed_encoded = profile ? now_sec() : 0.0;
    const double t_embed_done = profile ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: GPU layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  0,
                                                                  (uint32_t)n_tokens);
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            0,
                                                            (uint32_t)n_tokens);
            if (ok) {
                ds4_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: GPU layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = profile ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        0,
                                                        (uint32_t)n_tokens);
            const double t_encoded = profile ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = profile ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: GPU synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("DS4_GPU_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc,
                                                            output_row,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;

    const double t_head0 = profile ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%d encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool gpu_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    return gpu_graph_prefill_layer_major(g, model, weights, prompt, n_tokens, logits, show_progress, NULL);
}

static bool gpu_graph_prefill_batch_row_logits(
        ds4_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           batch_row,
        float             *logits) {
    if (!logits) return true;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc,
                                                            batch_row,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;
    return ds4_gpu_tensor_read(g->logits, 0, logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool gpu_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_imatrix_collector *imatrix) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    uint32_t first_chunk = n_tokens < chunk_cap ? n_tokens : chunk_cap;
    if (start != 0 && g->prefill_cap != 0) {
        const uint32_t mod = start % g->prefill_cap;
        if (mod != 0) {
            const uint32_t to_boundary = g->prefill_cap - mod;
            if (to_boundary < first_chunk) first_chunk = to_boundary;
        }
    }
    if (!gpu_graph_warmup_prefill_kernels(g, model, weights, first_chunk)) return false;

    const bool profile = getenv("DS4_GPU_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;
    uint32_t last_chunk_tokens = 0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        last_chunk_tokens = chunk;

        bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, pos0, chunk);
        if (ok) ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                             g->prefill_tokens,
                                                             model,
                                                             weights,
                                                             prompt,
                                                             pos0,
                                                             chunk);
        if (!ok) return false;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const double t_layer0 = profile ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        pos0,
                                                        chunk);
            const double t_encoded = profile ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = profile ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, chunk);
            if (profile) {
                encode_s += t_encoded - t_layer0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu chunked prefill pos=%u tokens=%u layer %u encode=%.3f ms execute=%.3f ms\n",
                        pos0,
                        chunk,
                        il,
                        (t_encoded - t_layer0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
            if (show_progress) {
                fprintf(stderr,
                        "ds4: gpu prefill token %u/%u layer %u/%u\r",
                        pos0 + chunk,
                        (uint32_t)prompt->len,
                        il + 1,
                        (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: GPU synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress && !gpu_graph_prefill_batch_row_logits(g, model, weights,
                                                              chunk - 1u,
                                                              logits))
        {
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)(pos0 + chunk), prompt->len);
        }
        pos0 += chunk;
    }
    if (show_progress) fputc('\n', stderr);
    if (last_chunk_tokens == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc,
                                                            last_chunk_tokens - 1u,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;

    const double t_head0 = profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool gpu_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud) {
    if (n_tokens <= 0) return false;
    return gpu_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             NULL);
}

/* Layer-major speculative target verifier for tiny MTP suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
static bool gpu_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        int                   *row_tops,
        float                 *row_logits) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (ok) ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                         g->prefill_tokens,
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    if (!ok) return false;

    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = capture_prefix1 && n_tokens == 2;

    ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = gpu_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    if (!ok) return false;

    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head_batch(g,
                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok) {
        if (top_rows) {
            ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                               g->spec_logits,
                                               DS4_N_VOCAB,
                                               1,
                                               top_rows) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && top_rows) {
        ok = ds4_gpu_tensor_read(g->comp_selected,
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
    }
    if (ok && row_logits) {
        ok = ds4_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    return ok;
}

static bool gpu_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}

/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
static bool gpu_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        float                 *logits0,
        float                 *logits1) {
    if (!g || !top0 || !logits1 || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *cur0 = gpu_graph_tensor_row_view(g->batch_cur_hc, 0, hc_dim);
    ds4_gpu_tensor *cur1 = gpu_graph_tensor_row_view(g->batch_cur_hc, 1, hc_dim);
    ds4_gpu_tensor *next0 = gpu_graph_tensor_row_view(g->batch_next_hc, 0, hc_dim);
    ds4_gpu_tensor *next1 = gpu_graph_tensor_row_view(g->batch_next_hc, 1, hc_dim);
    bool ok = cur0 && cur1 && next0 && next1;

    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token0,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token1,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;

        g->cur_hc = cur0;
        g->after_ffn_hc = next0;
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             gpu_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        ok = gpu_graph_capture_prefix1_attn_state(g, il) &&
             gpu_graph_capture_prefix1_index_state(g, il);
        if (!ok) break;

        g->cur_hc = cur1;
        g->after_ffn_hc = next1;
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             gpu_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0; cur0 = next0; next0 = tmp;
        tmp = cur1; cur1 = next1; next1 = tmp;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok) {
        g->cur_hc = cur0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->logits,
                                                   DS4_N_VOCAB,
                                                   1,
                                                   1) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0)) != 0;
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
    }

    if (ok) {
        g->cur_hc = cur1;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits1,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
        }
    }
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;
    g->spec_capture_prefix1 = saved_capture;

    ds4_gpu_tensor_free(next1);
    ds4_gpu_tensor_free(next0);
    ds4_gpu_tensor_free(cur1);
    ds4_gpu_tensor_free(cur0);
    return ok;
}

/* Pick a raw SWA cache size for GPU.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t gpu_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("DS4_GPU_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}

/* Choose the prefill ubatch size.  Whole-batch is fastest for normal prompts;
 * long prompts default to 2048-token chunks. */
static uint32_t gpu_graph_prefill_cap_for_prompt(int prompt_len) {
    return ds4_default_prefill_cap_for_prompt(prompt_len);
}

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  The env
 * knob is useful while tuning the crossover point for different Macs. */
static uint32_t gpu_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("DS4_GPU_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 32u;
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        const bool turbo_packv4 = ds4_gpu_dsv4_turbo4_packv4_enabled();
        m.prefill_cap = gpu_graph_prefill_cap_for_prompt((int)ctx);
        m.raw_cap = gpu_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            if (turbo_packv4) {
                m.compressed_bytes += ds4_gpu_dsv4_turbo4_packed_kv_bytes(layer_comp_cap,
                                                                          DS4_N_HEAD_DIM,
                                                                          DS4_N_ROT);
            } else {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_HEAD_DIM *
                                      sizeof(float);
            }
            if (ratio == 4) {
                if (turbo_packv4) {
                    m.compressed_bytes += ds4_gpu_dsv4_turbo4_packed_kv_bytes(layer_comp_cap,
                                                                              DS4_N_INDEXER_HEAD_DIM,
                                                                              DS4_N_ROT);
                } else {
                    m.compressed_bytes += (uint64_t)layer_comp_cap *
                                          DS4_N_INDEXER_HEAD_DIM *
                                          sizeof(float);
                }
            }
        }
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          m.prefill_cap *
                          sizeof(float);
        if (turbo_packv4) {
            m.scratch_bytes += (uint64_t)m.comp_cap * DS4_N_HEAD_DIM * sizeof(float);
            m.scratch_bytes += (uint64_t)m.comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    } else {
        m.raw_cap = ds4_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

static int gpu_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_GPU_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: GPU graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, (uint32_t)n_test, false);
    if (!ok) {
        gpu_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize GPU graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_GPU_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = gpu_graph_prefill_raw_swa(&g, model, weights, prompt, n_test, gpu_logits, true);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_GPU_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote GPU graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_GPU_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_GPU_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys, raw_phys_n * sizeof(float)) != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                ds4_gpu_tensor *gpu_comp_cache = gpu_graph_attn_comp_cache_for_read(&g, il, n_comp);
                if (gpu_comp_cache &&
                    ds4_gpu_tensor_read(gpu_comp_cache, 0, gpu_comp, n * sizeof(float)) != 0) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_index_comp_cache[il], 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: GPU prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g GPU_max=%g GPU_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: GPU prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: GPU synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    gpu_graph_free(&g);
    return ok ? 0 : 1;
}

#endif
