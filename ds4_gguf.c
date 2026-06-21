/* =========================================================================
 * ds4_gguf.c - GGUF model loading and in-place tensor accessors.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code (weights binding, tokenizer,
 * GPU graph) accesses weights by adding tensor offsets to the mapping instead
 * of copying the GGUF into private structures.
 *
 * Two residency paths:
 *   - mmap (default): file-backed shared (GPU) or private (CPU) read-only map.
 *   - managed (DS4_CUDA_MANAGED_MODEL=1): read the GGUF into a
 *     cudaMallocManaged buffer with read-mostly + preferred-location hints so
 *     the GPU reads weights directly at ~97 GB/s (single residency, no span
 *     cache duplication).  Fits K180 on the 128 GB Spark.
 *
 * Depends on ds4_util (memory, logging) and, for the managed path, ds4_gpu.h.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4_internal.h"

#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

/* ---- sequential read cursor over the mapping ---------------------------- */

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

/* ---- GGUF dtype table --------------------------------------------------- */

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
};

const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

/* ---- metadata + tensor parsing ------------------------------------------ */

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
    }
}

/* ---- model open / close / lifecycle ------------------------------------- */

/* Open and map the GGUF once.  graph_mapping keeps a shared file mapping (the
 * GPU backend maps slices as host buffers); otherwise the mapping is private.
 * prefetch_cpu touches pages for the CPU path (tokenizer-only callers pass
 * false so inspecting tokens never walks the huge tensor payload). */
void model_open(ds4_model *m, const char *path, bool graph_mapping,
                bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /* The GPU path keeps the file-backed shared mapping (slices become host
     * buffers); the CPU path uses a private read-only mapping.  The private
     * mapping is defensive: a shared mmap of the very large GGUF has triggered
     * OS-level VM accounting panics on Darwin when the CPU backend streams it. */
    const int mmap_flags = graph_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = NULL;
    int managed_model = 0;
#ifndef DS4_NO_GPU
    /* Managed-memory path (CUDA): read the GGUF into a cudaMallocManaged buffer
     * and hint it (ReadMostly + PreferredLocation=device) so the GPU reads weights
     * directly at ~97 GB/s -- single residency, no cudaMemcpy span cache, no mmap
     * page-cache duplication. This fits K180 on the 128 GB Spark at full speed
     * (the device-cache path duplicates the model and OOMs above ~K155). Gated on
     * the env (not graph_mapping: that flag is the graph-backend flag, true for CUDA). */
    if (getenv("DS4_CUDA_MANAGED_MODEL") != NULL) {
        map = ds4_gpu_managed_alloc((uint64_t)st.st_size);
        if (map) {
            managed_model = 1;
            /* Read the whole file into the managed buffer, dropping each chunk's
             * file page-cache entry immediately (POSIX_FADV_DONTNEED) so we never
             * hold the model twice (managed buffer + page cache). Without this the
             * pread double-counts and large models OOM during load. */
            size_t total = 0;
            const size_t chunk = 256 * 1024 * 1024; /* 256 MiB pread chunks */
            while (total < (size_t)st.st_size) {
                size_t want = (size_t)st.st_size - total;
                if (want > chunk) want = chunk;
                ssize_t got = pread(fd, (char *)map + total, want, (off_t)total);
                if (got < 0) {
                    if (errno == EINTR) continue;
                    ds4_die_errno("cannot read model into managed buffer", path);
                }
                if (got == 0) break;
                (void)posix_fadvise(fd, (off_t)total, (off_t)got, POSIX_FADV_DONTNEED);
                total += (size_t)got;
            }
            if (total != (size_t)st.st_size) ds4_die("short read loading managed model");
            (void)posix_fadvise(fd, 0, (off_t)st.st_size, POSIX_FADV_DONTNEED);
            fprintf(stderr, "ds4: CUDA loaded %.2f GiB model into managed memory\n",
                    (double)st.st_size / 1073741824.0);
        }
    }
#endif
    if (!map) {
        map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
        if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);
    }

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;
    m->managed = managed_model;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

    if (!graph_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}

void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /* CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it. */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

/* Optional startup pass that touches tensor pages before timing generation. */
void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

/* ---- accessors ---------------------------------------------------------- */

ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT64) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u64(&c, out);
}

bool model_get_bool(const ds4_model *m, const char *key, bool *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

bool model_get_u32_array_exact(const ds4_model *m, const char *key, uint32_t *out, uint64_t expected_len) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr)) return false;
    if (arr.len != expected_len) {
        fprintf(stderr, "ds4: metadata array %s has length %" PRIu64 ", expected %" PRIu64 "\n",
                key, arr.len, expected_len);
        exit(1);
    }
    if (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32) {
        fprintf(stderr, "ds4: metadata array %s has non-u32/i32 element type %u\n", key, arr.type);
        exit(1);
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &out[i])) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) {
                fprintf(stderr, "ds4: metadata array %s contains negative value at index %" PRIu64 "\n", key, i);
                exit(1);
            }
            out[i] = (uint32_t)v;
        }
    }
    return true;
}

ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return m->map + t->abs_offset;
}

/* ---- summary (human-readable model header dump) ------------------------- */

static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}

void model_summary(const ds4_model *m) {
    ds4_str name = {0};
    ds4_str arch = {0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u32(m, "deepseek4.block_count", &layers);
    model_get_u64(m, "deepseek4.context_length", &ctx_train);
    model_get_u32(m, "deepseek4.attention.head_count", &n_head);
    model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv);
    model_get_u32(m, "deepseek4.attention.key_length", &head_dim);
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads);
    model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim);
    model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k);
    model_get_u32(m, "deepseek4.expert_count", &n_expert);
    model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used);
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    bool reap_enabled = false;
    if (model_get_bool(m, "reap.enabled", &reap_enabled) && reap_enabled) {
        ds4_str layout = {0};
        uint32_t policy[DS4_N_LAYER] = {0};
        uint32_t keep_count[DS4_N_LAYER] = {0};
        uint32_t disabled = 0;
        uint32_t router_masked = 0;
        uint32_t hash_preserved = 0;
        uint32_t kept_total = 0;
        if (model_get_u32_array_exact(m, "reap.layer.policy", policy, DS4_N_LAYER)) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                if (policy[il] == DS4_REAP_POLICY_HASH_PRESERVED) hash_preserved++;
                if (policy[il] == DS4_REAP_POLICY_ROUTER_MASK_PRUNED) router_masked++;
                if (policy[il] == DS4_REAP_POLICY_MOE_DISABLED) disabled++;
            }
        }
        if (model_get_u32_array_exact(m, "reap.layer.keep_count", keep_count, DS4_N_LAYER)) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) kept_total += keep_count[il];
        }
        if (!model_get_string(m, "reap.layout", &layout)) layout = (ds4_str){ "unknown", 7 };
        printf("reap: enabled layout=%.*s hash_preserved=%u router_masked=%u moe_disabled=%u kept_slots=%u\n",
               (int)layout.len, layout.ptr, hash_preserved, router_masked, disabled, kept_total);
    }
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}
