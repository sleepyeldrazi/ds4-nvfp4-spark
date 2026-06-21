# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase. The
GPU path is CUDA-only (Metal support was removed when the project moved to the
CUDA/Spark path); CUDA kernels implementing `ds4_gpu.h` live in
`src/gpu/ds4_cuda.cu`.

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

## Source Layout (`src/`)

```
src/
├── core/                        # Engine core: public API, internal types, CPU/GPU graph, sessions
│   ├── ds4.h                    # Public engine + session + tokenizer API
│   ├── ds4_internal.h           # Shared internal API: model geometry, GGUF types,
│   │                            #   inference-core types, all module entry point declarations
│   ├── ds4.c                    # Header + include dispatcher + pre-CPU setup boilerplate
│   │                            #   (accelerator cache, process lock, etc.)
│   ├── ds4_cpu.inc              # CPU reference forward: HC transforms, attention (RoPE +
│   │                            #   compressed-attention indexer), MoE FFN, KV cache management
│   ├── ds4_gpu.inc              # GPU graph runtime: graph state, decode/prefill dispatch,
│   │                            #   diagnostic comparisons, imatrix/REAP collection
│   └── ds4_session.inc          # Engine API + process lock + session snapshot payloads
│                                #   (disk KV persistence)
│
├── model/                       # GGUF loading + weight binding
│   ├── ds4_gguf.c               # GGUF loader + in-place tensor accessors
│   └── ds4_weights.c            # GGUF → DS4 weight binding + layout validation + REAP metadata
│
├── tokenizer/
│   └── ds4_tokenizer.c          # GPT-2 BPE + DS4 chat encoding
│
├── quant/
│   └── ds4_quant.c              # Quant block formats + CPU dequant/dot kernels (NEON where available)
│
├── util/
│   └── ds4_util.c               # Leaf utilities: memory, logging, thread pool
│
├── gpu/                         # CUDA backend: device kernels + host-side model management
│   ├── ds4_gpu.h                # Backend-agnostic GPU API (tensor alloc, matmul, attention, MoE, HC…)
│   ├── ds4_cuda.cu              # CUDA backend implementing ds4_gpu.h: host-side tensor + range mgmt,
│   │                            #   model caching, temporary allocator, including all .cuh modules
│   ├── ds4_cuda_common.h        # Shared CUDA types, inline device helpers (dot products, dequant),
│   │                            #   NVFP4 helpers, IQ2 lookup table declarations, global state externs
│   ├── ds4_cuda_attention.cuh   # Attention kernels: decode mixed/raw/indexed/turbo4, prefill raw/mixed
│   ├── ds4_cuda_compressor.cuh  # KV compressor: update, store batch, prefill, ratio-4 replay/state
│   ├── ds4_cuda_devutil.cuh     # Device-side utilities: memcpy, fill, copy, sum, max, gather, scatter
│   ├── ds4_cuda_dispatch1.cuh   # Graph dispatch: prefills (dense, compress-4, compress-128)
│   ├── ds4_cuda_dispatch2.cuh   # Graph dispatch: cached prefill reply, one-shot decode, batch decode
│   ├── ds4_cuda_dispatch3.cuh   # Graph dispatch: speculative decode, decode batched sparse attention
│   ├── ds4_cuda_dispatch4.cuh   # Graph dispatch: REAP hash routing prefill, compress-phase decode
│   ├── ds4_cuda_embed.cuh       # Token embedding + HC initialization kernels
│   ├── ds4_cuda_hc.cuh          # Hyper-Connection kernels: split/sinkhorn, weighted sum, expand
│   ├── ds4_cuda_indexer.cuh     # Compressed-attention indexer: score, topk, mask
│   ├── ds4_cuda_matmul.cuh      # Matrix multiply: Q8_0, F16 pair/single, F32
│   ├── ds4_cuda_moe.cuh         # Routed MoE kernels: one-token, batch, IQ2_XXS/Q2_K/Q4_K/NVFP4 experts
│   ├── ds4_cuda_moe_dispatch.cuh# MoE dispatch helpers: expert scatter/gather, scaling
│   ├── ds4_cuda_norm.cuh        # RMS norm kernels: plain, weighted, head, QKV, DS4-specific
│   ├── ds4_cuda_q8.cuh          # Q8_0 quantize + dequant to F16/F32
│   ├── ds4_cuda_quant.cuh       # Quant format conversion: Q4_K→Q8_K, Q2_K→Q8_K, NVFP4→Q8_K, etc.
│   ├── ds4_cuda_rope.cuh        # RoPE kernels: tail-only, yarn scaling, fused KV-FP8-store
│   ├── ds4_cuda_router.cuh      # MoE router: select (token/batch), bias, hash routing
│   ├── ds4_iq2_tables_cuda.inc  # IQ2_XXS lookup tables (__device__ __constant__ arrays)
│   ├── ds4_turbo4.cu            # FP8-packed compressed KV cache (packv4 format): e4m3 + e8m0 scales + BF16 RoPE
│   └── ds4_turbo4_stubs.c       # CPU fallback stubs for turbo4-packed KV interface
│
├── cli/
│   └── ds4_cli.c                # Command-line interface (REPL, --prompt, interactive chat)
│
├── server/
│   └── ds4_server.c             # HTTP server (OpenAI-compatible /v1/chat/completions, KV disk cache)
│
├── bench/
│   └── ds4_bench.c              # Benchmark runner (throughput, latency, prefill/generation phases)
│
└── vendor/                      # Third-party libraries
    ├── linenoise/
    │   ├── linenoise.c          # Minimal line-editing library
    │   └── linenoise.h
    └── rax/
        ├── rax.c                # Radix tree (used by server for KV-cache session lookups)
        ├── rax.h
        └── rax_malloc.h
```

## Testing

```
tests/
├── ds4_test.c                   # Engine + server integration tests (#include's ../src/server/ds4_server.c)
├── cuda_long_context_smoke.c    # CUDA long-context smoke test
├── test_turbo4.cu               # Turbo4 pack/unpack round-trip correctness test
├── test-vectors/                # Test vector data
└── long_context_security_prompt.txt  # Long-context test prompt
```

Use `make` for build validation. Use `make test` for unit/regression tests when a
model is available. Use live server tests only when intentionally testing the
API surface.

## Other Top-Level Directories

```
dir-steering/                    # Directional steering vector data
gguf-tools/                      # GGUF inspection + manipulation scripts
speed-bench/                     # Speed benchmark harness
```
