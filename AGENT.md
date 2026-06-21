# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase. The
GPU path is CUDA-only (Metal support was removed when the project moved to the
CUDA/Spark path); CUDA kernels implementing `ds4_gpu.h` live in `ds4_cuda.cu`.

## Goals

- Keep the production path as whole-model GPU graph inference (CUDA backend).
- Keep model loading mmap-backed (or managed-memory backed); do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short GPU smoke tests for build verification.

## Layout

The engine is split across reusable translation units and included feature
modules. Frontends (CLI/server/bench) go through `ds4.h` only.

### Reusable translation units
- `ds4.h`: public engine + session + tokenizer API.
- `ds4_internal.h`: shared internal API — model geometry, GGUF types,
  inference-core types (cpu_decode_scratch, kv_cache, gpu_graph, session),
  and declarations for all module entry points.
- `ds4_util.c`: leaf utilities (memory, logging, thread pool).
- `ds4_gguf.c`: GGUF loader + in-place tensor accessors.
- `ds4_quant.c`: quant block formats + CPU dequant/dot kernels.
- `ds4_tokenizer.c`: GPT-2 BPE + DS4 chat encoding.
- `ds4_weights.c`: GGUF → DS4 weight binding + layout validation.

### Feature modules (included via #include from ds4.c)
These preserve the original preprocessor guard structure. Each is a focused
area that one developer can own independently.
- `ds4.c` (1999 lines): header + include dispatcher + the pre-CPU setup
  boilerplate (accelerator cache, process lock, etc.).
- `ds4_cpu.inc` (3884 lines): CPU reference forward — Hyper-Connection
  transforms, attention projections (RoPE + compressed-attention indexer),
  mixture-of-experts FFN, KV cache management.
- `ds4_gpu.inc` (6409 lines): GPU graph runtime — graph state, decode/prefill
  dispatch, diagnostic comparisons, imatrix/REAP collection.
- `ds4_session.inc` (3619 lines): Engine API + process lock + session
  snapshot payloads (disk KV persistence).

### Frontends + backend
- `ds4_cli.c` / `ds4_server.c` / `ds4_bench.c`: command-line, HTTP, benchmark.
- `ds4_cuda.cu`: CUDA backend (ds4_gpu.h implementation).
- `ds4_gpu.h`: backend-agnostic GPU API.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model is available. Use live server tests only when intentionally testing the
API surface.
