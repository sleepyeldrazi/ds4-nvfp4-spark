#ifndef DS4_INTERNAL_H
#define DS4_INTERNAL_H

/* =========================================================================
 * ds4_internal.h - shared internal API across the ds4 engine modules.
 * =========================================================================
 *
 * The public engine boundary lives in ds4.h (engine + session + token API).
 * This header is the INTERNAL interface used by the engine's own translation
 * units (ds4.c, ds4_util.c, and the upcoming feature modules). Frontends
 * (ds4_cli / ds4_server / ds4_bench) must NOT include this header -- they go
 * through ds4.h only.
 *
 * Contents:
 *   - shared constants (DS4_NEG_INF, RMS eps, rope defaults, ...)
 *   - ds4_str / ds4_cursor value types used by the GGUF loader
 *   - memory + death helpers (xmalloc family, ds4_die)
 *   - logging (ds4_log)
 *   - timing (now_sec)
 *   - CPU worker thread pool (ds4_parallel_for)
 *   - model-geometry helper (ds4_layer_compress_ratio)
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

#endif /* DS4_INTERNAL_H */
