/* =========================================================================
 * ds4_tokenizer.c - GPT-2 byte-level BPE + DS4 chat prompt encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF
 * ("tokenizer.ggml.pre" = "joyai-llm").  This module loads token strings and
 * merge ranks into two open-addressed hash tables, applies byte-level BPE with
 * the JoyAI/DeepSeek pre-tokenizer to user text, inserts chat special tokens
 * directly by id, and renders chat prompts.
 *
 * All entry points operate on ds4_vocab* so the tokenizer is independent of the
 * engine struct.  The public ds4.h wrappers (ds4_tokenize_text etc. in ds4.c)
 * forward e->vocab to these.  Depends on ds4_util + ds4_gguf.
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_internal.h"

/* Reasoning-effort-max prelude prepended to the prompt under --think-max. */
const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * ========================================================================= */

/* ---- open-addressed string -> int hash table ---------------------------- */

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

/* ---- token vec helpers (public ds4_tokens API) -------------------------- */

typedef ds4_tokens token_vec;

void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

/* ---- utf8 + GPT-2 byte encoder ----------------------------------------- */

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return 256 + n;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* ---- BPE merge --------------------------------------------------------- */

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, ds4_tokens *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

/* ---- JoyAI/DeepSeek pre-tokenization ----------------------------------- */

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, ds4_tokens *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                bpe_emit_piece(vocab, (ds4_str){text + start, last_newline_end - start}, out);
                if (last_newline_end < p) {
                    bpe_emit_piece(vocab, (ds4_str){text + last_newline_end, p - last_newline_end}, out);
                }
            } else {
                bpe_emit_piece(vocab, (ds4_str){text + start, p - start}, out);
            }
            pos = p;
            continue;
        } else {
            pos++;
        }

        bpe_emit_piece(vocab, (ds4_str){text + start, pos - start}, out);
    }
}

/* ---- vocab load/free + chat encoding ----------------------------------- */

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
}

void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        ds4_tokens       *out) {
    token_vec_push(out, vocab->bos_id);
    if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
    if (system && system[0]) {
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, ds4_tokens *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                  ds4_tokens *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

/* ---- token text + diagnostics ----------------------------------------- */

void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const ds4_tokens *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

void dump_tokens(const ds4_vocab *vocab, const ds4_tokens *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *vocab_token_text(const ds4_vocab *vocab, int token, size_t *len) {
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int vocab_token_eos(const ds4_vocab *vocab) {
    return vocab->eos_id;
}
