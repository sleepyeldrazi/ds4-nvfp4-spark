#ifndef DS4_INTERNAL_H
#define DS4_INTERNAL_H

/* =========================================================================
 * ds4_internal.h - shared internal API across the ds4 engine modules.
 * =========================================================================
 *
 * The public engine boundary lives in ds4.h (engine + session + token API).
 * This header is the INTERNAL interface used by the engine's own translation
 * units (ds4.c, ds4_util.c, ds4_gguf.c, ds4_quant.c, ds4_tokenizer.c, and the
 * upcoming feature modules). Frontends (ds4_cli / ds4_server / ds4_bench) must
 * NOT include this header -- they go through ds4.h only.
 *
 * Type ordering: value types (ds4_str) -> GGUF types (ds4_tensor/ds4_model) ->
 * weights/vocab -> engine struct. Each layer depends only on the ones above.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4.h"

/* ---- DeepSeek V4 Flash fixed model geometry ----------------------------------
 *
 * The engine implements exactly one model layout. These numbers are the
 * DeepSeek V4 Flash dimensions and are validated against every loaded GGUF;
 * code throughout uses them as compile-time fixed-size path markers. */
enum {
    DS4_N_LAYER            = 43,
    DS4_N_EMBD             = 4096,
    DS4_N_VOCAB            = 129280,
    DS4_N_HEAD             = 64,
    DS4_N_HEAD_KV          = 1,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_VALUE_DIM        = 512,
    DS4_N_ROT              = 64,
    DS4_N_OUT_GROUP        = 8,
    DS4_N_LORA_Q           = 1024,
    DS4_N_LORA_O           = 1024,
    DS4_N_EXPERT           = 256,
    DS4_N_EXPERT_USED      = 6,
    DS4_N_EXPERT_SHARED    = 1,
    DS4_N_FF_EXP           = 2048,
    DS4_N_HASH_LAYER       = 3,
    DS4_N_SWA              = 128,
    DS4_N_INDEXER_HEAD     = 64,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_N_INDEXER_TOP_K    = 512,
    DS4_N_HC               = 4,
    DS4_N_HC_SINKHORN_ITER = 20,
};

enum {
    DS4_REAP_POLICY_NONE = 0,
    DS4_REAP_POLICY_HASH_PRESERVED = 1,
    DS4_REAP_POLICY_ROUTER_MASK_PRUNED = 2,
    DS4_REAP_POLICY_MOE_DISABLED = 3,
};

#if defined(__GNUC__) || defined(__clang__)
#define DS4_MAYBE_UNUSED __attribute__((unused))
#else
#define DS4_MAYBE_UNUSED
#endif

#define DS4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]

/* ---- shared numeric constants -------------------------------------------- */

#define DS4_NEG_INF (-1.0e30f)
#define DS4_POS_INF ( 1.0e30f)
#define DS4_RMS_EPS ( 1.0e-6f)
#define DS4_HC_EPS  ( 1.0e-6f)
#define DS4_EXPERT_WEIGHT_SCALE (1.5f)
#define DS4_SWIGLU_CLAMP_EXP    (10.0f)
#define DS4_ROPE_FREQ_BASE      (10000.0f)
#define DS4_ROPE_SCALE_FACTOR   (16.0f)
#define DS4_ROPE_YARN_BETA_FAST (32.0f)
#define DS4_ROPE_YARN_BETA_SLOW (1.0f)
#define DS4_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define DS4_ROPE_ORIG_CTX       UINT64_C(65536)

/* ---- value types --------------------------------------------------------- */

/* Non-NUL-terminated string slice into the GGUF mapping or a caller buffer. */
typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

/* Sequential read cursor over a mapped region.  Tracks position and records
 * the first error so parsers can bail without propagating return codes. */
typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

/* ---- memory + death ------------------------------------------------------ */

void ds4_die(const char *msg);
void ds4_die_errno(const char *what, const char *path);

void *xcalloc(size_t n, size_t size);
void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t size);
void *xmalloc_zeroed(size_t n, size_t size);
char *ds4_strdup(const char *s);

/* Allocation guard.  CPU decode is expected to run entirely out of preallocated
 * scratch; any malloc/calloc/realloc that slips into the token loop is a bug.
 * Guarded phases call begin/end around the region where allocation is banned. */
void ds4_alloc_guard_begin(const char *phase);
void ds4_alloc_guard_end(void);

/* ---- logging + timing ---------------------------------------------------- */

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
bool ds4_log_is_tty(FILE *fp);

double now_sec(void);

/* ---- strings + hashing --------------------------------------------------- */

bool ds4_streq(ds4_str s, const char *z);
bool ds4_str_eq(ds4_str a, ds4_str b);
uint64_t hash_bytes(const void *ptr, uint64_t len);

/* ---- small file I/O helpers --------------------------------------------- */

bool write_f32_binary_file(const char *path, const float *data, uint64_t n);
bool read_f32_binary_file(const char *path, float *data, uint64_t n);

/* ---- CPU worker thread pool --------------------------------------------- */

typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

void ds4_threads_init(void);
void ds4_threads_shutdown(void);

/* Run a row-parallel CPU kernel.  Small jobs and nested calls run inline to
 * avoid scheduling overhead; large jobs split across the worker pool. */
void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx);
void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows);

/* Override the auto-detected worker count (called from engine open). */
void ds4_threads_set_requested(uint32_t n);

/* ---- model geometry ------------------------------------------------------ */

/* Attention compression alternates after layer 1: dense early layers, then
 * ratio-4 layers with an indexer and ratio-128 layers without one. */
uint32_t ds4_layer_compress_ratio(uint32_t il);

/* ---- quant table lazy init (implemented by the quant module) ------------- */

/* Build the IQ2 signed lookup grids exactly once.  Called automatically by
 * ds4_threads_init(); also safe to call directly before first CPU dequant. */
void ds4_quant_init(void);

/* ---- quant block formats + CPU dequant (ds4_quant.c) -------------------- */

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

/* CPU reference quant + dot-product kernels (NEON where available). */
void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);
void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y);
void ds4_vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y);
void ds4_vec_dot_iq2_xxs_pair_q8_K(int n, float *s0, float *s1,
                                   const block_iq2_xxs *x0, const block_iq2_xxs *x1,
                                   const block_q8_K *y);

/* Scalar conversions used by the CPU path + GPU diagnostics. */
float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
void f16_round_inplace_cpu(float *x, uint32_t n);
float dsv4_e4m3fn_value_cpu(int i);
float dsv4_e4m3fn_dequant_cpu(float x);
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot);

/* ---- GGUF loading + model accessors (ds4_gguf.c) ------------------------ */

#define DS4_MAX_DIMS   8
#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

/* DS4 tensor dtype ids (subset of the GGUF type ids this engine reads). */
enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_NVFP4    = 31,
};

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
    /* Per-expert scale_2 array for NVFP4 expert weights (NULL otherwise). */
    const float *nvfp4_scale_2;
} ds4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;
    int managed;  /* 1 if map is a cudaMallocManaged buffer (single-residency path) */
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

/* Open + map the GGUF once. graph_mapping keeps a shared file mapping (the GPU
 * backend maps slices as host buffers); otherwise the mapping is private.
 * prefetch_cpu touches pages for the CPU path. */
void model_open(ds4_model *m, const char *path, bool graph_mapping, bool prefetch_cpu);
void model_close(ds4_model *m);
void model_prefetch_cpu_mapping(const ds4_model *m);
void model_warm_weights(const ds4_model *m);
void model_summary(const ds4_model *m);

ds4_kv *model_find_kv(const ds4_model *m, const char *key);
ds4_tensor *model_find_tensor(const ds4_model *m, const char *name);
bool model_get_string(const ds4_model *m, const char *key, ds4_str *out);
bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out);
bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out);
bool model_get_bool(const ds4_model *m, const char *key, bool *out);
bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out);
bool model_get_u32_array_exact(const ds4_model *m, const char *key, uint32_t *out, uint64_t expected_len);

const void *tensor_data(const ds4_model *m, const ds4_tensor *t);
const gguf_type_info *tensor_type(uint32_t type);
const char *tensor_type_name(uint32_t type);
bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
uint64_t align_up(uint64_t value, uint64_t alignment);

/* Low-level cursor over the mapping.  Exposed for callers that read raw GGUF
 * metadata fields not covered by the model_get_* accessors (tokenizer
 * token/merge tables, REAP arrays, steering vectors). */
ds4_cursor cursor_at(const ds4_model *m, uint64_t pos);
bool cursor_read(ds4_cursor *c, void *dst, uint64_t n);
bool cursor_u32(ds4_cursor *c, uint32_t *v);
bool cursor_u64(ds4_cursor *c, uint64_t *v);
bool cursor_string(ds4_cursor *c, ds4_str *s);

/* ---- model weight bindings (shared core types) -------------------------- */

/* Bound tensor pointers into the mmaped GGUF for one transformer layer. */
typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
    uint32_t reap_policy;
    uint32_t reap_expert_count;
    uint32_t reap_keep_count;
    bool reap_moe_disabled;
} ds4_layer_weights;

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_N_LAYER];
    bool reap_compact_layout;
} ds4_weights;

typedef struct {
    ds4_tensor *e_proj;
    ds4_tensor *h_proj;
    ds4_tensor *enorm;
    ds4_tensor *hnorm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_layer_weights block;
} ds4_mtp_weights;

/* ---- model weight binding + validation (ds4_weights.c) ----------------- */

/* Bind the GGUF tensor directory into the DS4 layer weight pointer tables,
 * validate the fixed DS-V4-Flash layout, and read REAP metadata.  Called by
 * engine_open; weights_free releases the REAP keep-count scratch (tensor
 * pointers themselves are mmap-backed and not owned). */
void weights_bind(ds4_weights *w, const ds4_model *m);
void weights_free(ds4_weights *w);
void mtp_weights_bind(ds4_mtp_weights *w, const ds4_model *m);
void config_validate_model(const ds4_model *m);

/* Helpers used by the GPU MoE dispatch (not just the weights module). */
uint32_t layer_stored_expert_count(const ds4_layer_weights *l);
uint64_t routed_expert_row_bytes(const ds4_tensor *t);

/* ---- tokenizer vocab (ds4_tokenizer.c) ---------------------------------- */

/* Reasoning-effort max prefix (the --think-max prompt prelude). */
extern const char DS4_REASONING_EFFORT_MAX_PREFIX[];

/* Open-addressed string -> int hash table (tokenizer token_to_id, merge_rank). */
typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

typedef struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
} ds4_vocab;

/* GPT-2 byte-level BPE + DS4 chat prompt encoding.  Operate on ds4_vocab* so
 * the tokenizer is independent of the engine struct.  The public ds4.h
 * wrappers (ds4_tokenize_text etc.) live in ds4.c and forward e->vocab. */

/* token_vec is an alias for the public ds4_tokens used by the engine's
 * internal token-building helpers. */
typedef ds4_tokens token_vec;
void token_vec_push(token_vec *tv, int token);
void token_vec_free(token_vec *tv);

void vocab_load(ds4_vocab *vocab, const ds4_model *model);
void vocab_free(ds4_vocab *vocab);
void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, ds4_tokens *out);
void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text, ds4_tokens *out);
void encode_chat_prompt(const ds4_vocab *vocab, const char *system,
                        const char *prompt, ds4_think_mode think_mode, ds4_tokens *out);
char *vocab_token_text(const ds4_vocab *vocab, int token, size_t *len);
int vocab_token_eos(const ds4_vocab *vocab);
void dump_tokens(const ds4_vocab *vocab, const ds4_tokens *tokens);
void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const ds4_tokens *tokens);

/* ---- engine (ds4.c) ----------------------------------------------------- */

struct ds4_engine {
    ds4_model model;
    ds4_model mtp_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_backend backend;
    int mtp_draft_tokens;
    float mtp_margin;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    bool gpu_ready;
    bool mtp_ready;
};

/* ---- shared inference-core types (ds4.c uses these across 3 features) --- */

/* Forward-declare the opaque GPU tensor type from ds4_gpu.h.  The GPU graph
 * struct holds pointers to these but never dereferences the body from plain C. */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

/* CPU decode scratch arena.  Allocated once per session and reused every token
 * so the CPU reference path never allocates in the hot loop. */
typedef struct {
    uint32_t ctx_size;
    uint32_t comp_cap;
    uint32_t attn_score_cap;
    uint32_t q8_cap;
    float *plain;
    float *cur;
    float *next;
    float *attn_cur;
    float *attn_norm;
    float *attn_residual;
    float *q;
    float *qr;
    float *qr_norm;
    float *kv_raw;
    float *kv;
    float *heads;
    float *attn_low;
    float *attn_out;
    float *after_attn_hc;
    float *attn_score;
    float *comp;
    float *index_comp;
    float *comp_kv_cur;
    float *comp_sc_cur;
    float *comp_pooled;
    bool *index_allowed;
    float *index_q;
    float *index_weights;
    float *index_scores;
    float *ffn_cur;
    float *ffn_norm;
    float *ffn_moe;
    float *ffn_shared;
    float *ffn_out;
    float *shared_gate;
    float *shared_up;
    float *shared_mid;
    float *routed_mid_all;
    block_q8_K *routed_xq;
    block_q8_K *routed_midq;
    int8_t *q8_xq;
    float *q8_xscale;
    float *hc_flat;
    float *output_flat;
    float *output_pre;
    float *output_weights;
    float *output_embd;
    float *output_norm;
} ds4_cpu_decode_scratch;

/* Per-layer CPU KV cache (raw SWA window + compressed KV + compressor state). */
typedef struct {
    float *raw_kv;
    uint32_t n_raw;
    uint32_t cap_raw;
    uint32_t compress_ratio;
    uint32_t comp_cap;
    uint32_t n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;
    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;

typedef struct {
    ds4_layer_cache layer[DS4_N_LAYER];
    uint32_t head_dim;
} ds4_kv_cache;

/* Whole-model GPU graph: tensor residence for single-token decode and
 * batched prefill.  Owned by ds4_session.  All GPU tensors are pointers into
 * the ds4_gpu.h backend; the struct is plain C so it compiles everywhere. */
typedef struct {
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
    ds4_gpu_tensor *layer_raw_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_comp_tq_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_N_LAYER];
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
    uint32_t comp_cap;
    uint32_t layer_comp_cap[DS4_N_LAYER];
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *comp_mask;
    ds4_gpu_tensor *comp_selected;
    ds4_gpu_tensor *attn_comp_unpack_cache;
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

/* A live inference session owns the mutable KV cache, GPU graph state, CPU
 * scratch, and logits.  Its definition is shared across session/CPU/GPU code. */
struct ds4_session {
    ds4_engine *engine;
#ifndef DS4_NO_GPU
    ds4_gpu_graph graph;
#endif
    ds4_kv_cache cpu_cache;
    ds4_cpu_decode_scratch cpu_scratch;
    ds4_tokens checkpoint;
    float *logits;
    float *mtp_logits;
    int mtp_draft_token;
    uint64_t mtp_probe_total;
    uint64_t mtp_probe_hit;
    ds4_session_progress_fn progress;
    void *progress_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool mtp_draft_valid;
};

#endif /* DS4_INTERNAL_H */
