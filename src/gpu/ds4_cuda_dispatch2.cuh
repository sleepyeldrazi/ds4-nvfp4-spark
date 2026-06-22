extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!DS4_ENV_BOOL("DS4_CUDA_DISABLE_QKV_RMS_FUSED")) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_kv->ptr, state_n, 0.0f);
    if (!cuda_ok(cudaGetLastError(), "compressor state kv fill launch")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0 && !ds4_gpu_rope_tail_tensor(comp_cache, n_comp, 1, head_dim,
                                                      n_rot, pos0, n_ctx_orig, false,
                                                      freq_base, freq_scale, ext_factor,
                                                      attn_factor, beta_fast, beta_slow)) return 0;
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0 && !ds4_gpu_rope_tail_tensor(comp_cache, n_comp, 1, head_dim,
                                                  n_rot, pos0, n_ctx_orig, false,
                                                  freq_base, freq_scale, ext_factor,
                                                  attn_factor, beta_fast, beta_slow)) return 0;
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_kv->ptr, state_n, 0.0f);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state kv fill launch")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_kv->ptr, state_n, 0.0f);
    if (!cuda_ok(cudaGetLastError(), "compressor state kv fill launch")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            !DS4_ENV_BOOL("DS4_CUDA_NO_WINDOW_ATTENTION")) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_WINDOW_ATTENTION") &&
        (DS4_ENV_BOOL("DS4_CUDA_WINDOW_ATTENTION") || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_CUBLAS_ATTENTION")) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            !DS4_ENV_BOOL("DS4_CUDA_NO_WINDOW_ATTENTION")) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_WINDOW_ATTENTION") &&
        (DS4_ENV_BOOL("DS4_CUDA_WINDOW_ATTENTION") || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_INDEXED_HEADS8")) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        if (!DS4_ENV_BOOL("DS4_CUDA_INDEXED_TWOPASS")) {
            attention_indexed_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                        sinks,
                                                                        (const float *)q->ptr,
                                                                        (const float *)raw_kv->ptr,
                                                                        (const float *)comp_kv->ptr,
                                                                        (const int32_t *)topk->ptr,
                                                                        n_tokens,
                                                                        pos0,
                                                                        n_raw,
                                                                        raw_cap,
                                                                        raw_start,
                                                                        n_comp,
                                                                        top_k,
                                                                        window,
                                                                        ratio,
                                                                        n_head,
                                                                        head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 (const int32_t *)topk->ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  (const int32_t *)topk->ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_WINDOW_ATTENTION") &&
        (DS4_ENV_BOOL("DS4_CUDA_WINDOW_ATTENTION") || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_CUBLAS_ATTENTION")) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        !DS4_ENV_BOOL("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A")) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    (void)out_b;
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}
extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                            xscale,
                                            (const float *)heads->ptr,
                                            group_dim,
                                            blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q8 prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, 1, 1);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      1,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(), "attention_output_low_q8 launch");
}
