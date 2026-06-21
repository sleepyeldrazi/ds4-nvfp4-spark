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

- `ds4.c`: model loading, tokenizer, CPU reference code, GPU graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_cuda.cu`: CUDA backend implementing the `ds4_gpu.h` tensor/kernel API.
- `ds4_gpu.h`: backend-agnostic GPU API (tensors + dispatch); the engine drives
  the backend only through this header.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model is available. Use live server tests only when intentionally testing the
API surface.
