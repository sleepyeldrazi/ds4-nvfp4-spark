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

The engine is split across several translation units that share an internal
header. Frontends (CLI/server/bench) go through `ds4.h` only; `ds4_internal.h`
is the engine's own cross-module interface.

- `ds4.h`: public engine + session + tokenizer API (the boundary for frontends).
- `ds4_internal.h`: shared internal API -- model geometry, GGUF types, value
  types, memory/logging/threading/quant/loader declarations.
- `ds4.c`: the inference core — CPU reference forward (embed, norms, matvecs,
  and token-at-a-time decode), the GPU graph runtime (drives the backend
  through `ds4_gpu.h`), sessions, sampler, KV/snapshot payload, engine
  lifecycle, imatrix/REAP.  The CPU forward code is intentionally kept here
  with the session model — the scratch-based helpers and the row-parallel
  dispatch share enough internal types that extracting them would create an
  artificial border.
- `ds4_util.c`: leaf utilities (memory, death, logging, timing, strings,
  file I/O, the CPU worker thread pool, model-geometry helper).
- `ds4_gguf.c`: GGUF loader and in-place tensor accessors (mmap + managed path).
- `ds4_quant.c`: quant block formats (Q2_K/Q4_K/Q8_K/IQ2_XXS) + CPU dequant/dot.
- `ds4_tokenizer.c`: GPT-2 byte-level BPE + DS4 chat prompt encoding.
- `ds4_weights.c`: GGUF → DS4 weight binding, tensor-layout validation,
  REAP metadata reader.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_cuda.cu`: CUDA backend implementing the `ds4_gpu.h` tensor/kernel API.
- `ds4_gpu.h`: backend-agnostic GPU API (tensors + dispatch); the engine drives
  the backend only through this header.
- `tests/`: unit and live integration tests.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model is available. Use live server tests only when intentionally testing the
API surface.
