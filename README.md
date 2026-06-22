# ds4 — NVFP4 hybrid serving on DGX Spark (GB10)

> **⚠️ This GitHub repository is for archival / mirror purposes only.**
> Active development happens at
> **[git.kokoham.com/sleepy/ds4-nvfp4-spark](https://git.kokoham.com/sleepy/ds4-nvfp4-spark)**.
> Please file issues, PRs, and check for updates there.

Forked from [antirez/ds4](https://github.com/antirez/ds4), a standalone DeepSeek-V4
Flash (~162B) inference engine (C + CUDA, no GGML dependency). This fork adds
**NVFP4 expert quantization**, a **managed-memory serving path** for GB10 unified
memory, the **lossless MXFP4→NVFP4 GGUF emission** pipeline, an **FP8-packed KV
cache**, and **REAP expert pruning** to fit the model on a single 128 GB Spark.

Inspired by [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped) (a
fork of ds4) and the
[0xSero/DeepSeek-V4-Flash-162B-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF)
release, which motivated running a REAP'd DS-V4 Flash on the Spark. See
[Acknowledgements](#acknowledgements) for the REAP paper and related work.

## What this fork adds

### REAP expert pruning (K128 / K150 / K180)
- **REAP plan** (`make_reap_plan.py`): selects top-K routed experts per layer by
  FP4/FP8 REAP score (router gate-value × activation norm) from a calibration
  observation file. Hash-routed layers (0–2) keep all 256 experts; routed layers
  (3–42) keep K. Three plans shipped: K128, K150, K180.
- **Hybrid GGUF emission** (`gguf-tools/deepseek4-quantize.c`): one pass that
  prunes experts per the REAP plan, quantizes gate/up experts to NVFP4, down
  experts to Q2_K, and copies attention/shared/head tensors from the template.
- **Runtime REAP metadata**: the GGUF carries `hash_preserved` / `router_masked`
  masks so the engine routes around pruned experts without code changes.

### NVFP4 expert kernels + serving (verified)
- **Lossless MXFP4→NVFP4 repack**: the HF source experts are MXFP4 (e2m1 + e8m0
  per-32). NVFP4 is e2m1 + e4m3 per-16 + per-expert `scale_2`. The e2m1 nibbles
  are identical — the conversion is a lossless scale-only transform (no requant,
  no amax needed for weights). Verified bit-exact via a round-trip test.
- **ds4_cuda.cu**: `cuda_block_nvfp4` struct, `dev_dot_nvfp4_q8_K_block`
  (`__dp4a` MMQ, ~140 GB/s on GB10), batched tile8/tile16 rowspan kernels + two
  n_tokens==1 decode kernels, per-expert `scale_2` array threaded through dispatch.
- **ds4.c**: name-based NVFP4 GGUF loader (`.nvfp4_weight` + `.nvfp4_scale_2`),
  forces `DS4_TENSOR_NVFP4` (id 31) for dispatch.
- The GGUF writer stores NVFP4 tensors with the ds4q NVFP4 type id **40**, which
  is intentionally not in ds4's `gguf_types` table. The loader prints
  `warning: tensor ... has unsupported GGUF type 40` for each NVFP4 tensor —
  **this is expected and harmless**: `ds4_weights.c` rebinds those tensors by
  name (`.nvfp4_weight`) and forces the dispatch type to 31.

### FP8-packed KV cache (`DS4_KV_TURBO=1`, verified)
DS-V4 attention stores compressed KV at FP8 density. ds4's original code did
in-place FP8 *value* quantization but wrote the result back into **FP32 buffers**
(4× memory waste, zero compression). This fork implements real packed storage
matching the paper spec, for **both** the attention-compressed stream and the
CSA lightning-indexer stream:

- **Attention-compressed packed row (584 B vs 2048 B FP32 = 3.51× compression)**:
  `[ 448 e4m3 (nope) | 7 e8m0 scales (per-64-block) | 1 pad | 128 B BF16 (rot) ]`
- **Indexer-compressed packed row (200 B vs 512 B FP32 = 2.56× compression)**,
  head_dim=128, n_rot=64 — reuses the same generic turbo4 pack/unpack kernels.
- Uses **CUDA 13 native `cuda_fp8.h`** types (`__nv_fp8_e4m3` SATFINITE,
  `__nv_fp8_e8m0` cudaRoundPosInf) — no hand-rolled bit layout.
- **GB10 (sm_121a) fixes**: +1 pad byte so the BF16 section lands at an even
  offset (misaligned 2-byte loads crash on GB10); BF16 store/load via
  `uint16_t*` (the `__bfloat16_as_ushort()` byte-extraction path hits a
  sm_121a compiler bug).
- **Turbo4 direct (packed-read) attention**: the indexed-mixed attention kernel
  reads the turbo4-packed comp rows directly (`turbo4_attention.cuh`) instead of
  unpacking to an FP32 scratch first. Falls back to the unpack path when turbo4
  is off or prerequisites are missing. Gated by `DS4_NO_TURBO4_DIRECT` to disable.
- **Correctness**: greedy decode matches the FP32 path 29/30 tokens on short
  prompts (diverges only at the final token — healthy FP8 quantization, not a
  bug). Indexer-KV packing is bit-identical to the FP32 path (verified at 128K).
- **Savings**: linear at 7.23 KiB/token for the attention stream; ~7.2 GiB at
  1M ctx. Packing the indexer stream reclaims ~1.5 GiB more. **K180 now serves
  at full 1M context** (peaked ~102.8 GiB, fits under the 128 GB Spark).

### Managed-memory serving path (`DS4_CUDA_MANAGED_MODEL=1`)
GB10 has hardware ATS (`PageableMemAccessUsesHostPageTables=1`) — the GPU reads
CPU-allocated memory directly, coherent, no copy. This fork loads the GGUF into
a `cudaMallocManaged` buffer (via chunked `pread` + `posix_fadvise DONTNEED`)
with `cudaMemAdvise(SetReadMostly + SetPreferredLocation=device)` +
`cudaMemPrefetchAsync`, and gates off the redundant on-demand cudaMemcpy span
cache. Result: single residency at ~97 GB/s (measured) → **K180 hybrid (98.6
GiB) serves at ~12 t/s**.

> **K180 requires `DS4_CUDA_MANAGED_MODEL=1`.** At 98.6 GiB the model does not
> fit in any single non-managed allocation on the 128 GB Spark (UMA page-table
> overhead + context buffers push it over). The managed path is the only way to
> load K180. K128 (75 GiB) and K150 (85 GiB) can load without it, but managed
> mode is still recommended for the single-residency bandwidth win.

### Encode-path bugs fixed (8 total)
See `../cuda_debug/ENCODE_VERIFICATION.md` for the full list. The load-bearing
ones: e4m3 bit layout (`(E<<3)|M` not `(E<<4)|M`), per-expert (not per-row)
k_max, interleaved `cuda_block_nvfp4` layout, the 2^8=256 max-scale edge, GGUF
header `n_tensors` count, loader type-id mismatch.

## Hybrid GGUF build pipeline

```bash
# 1. REAP plan (top-K experts/layer by FP4/FP8 REAP score)
python3 make_reap_plan.py --obs calibration_fp4fp8.obs.json --k 180 --out plan.json

# 2. Build the hybrid GGUF (one pass: prune + quantize + GGUF write)
cd gguf-tools && make
./deepseek4-quantize \
  --hf <HF MXFP4 checkpoint dir> \
  --template <existing DS4 GGUF (for metadata/tensor order)> \
  --reap-plan plan.json \
  --routed-w1 nvfp4 --routed-w3 nvfp4 --routed-w2 q2_k \
  --out hybrid-K180.gguf

# 3. Serve on the Spark (K180 at full 1M context)
DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1 ./ds4 -m hybrid-K180.gguf -p "..." --ctx 1048576
# Without DS4_KV_TURBO, the FP32 KV overflows at 1M ctx — use it for large contexts.
# Without DS4_CUDA_MANAGED_MODEL, K180 will not load (see note above).
```

The quant recipe: gate (w1) + up (w3) experts → **NVFP4**, down (w2) experts →
**Q2_K**, attention/shared/head → **copy from template** (preserves required
f16/f32). Non-expert tensors use the copy policy (don't force `--attention q8_0`
— it over-applies to f16-required HC tensors).

## Memory budget on the 128 GB Spark

K180 at **full 1M context** (verified, `DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1`,
both attention and indexer KV packed):

| component | size |
|---|---|
| K180 model (cudaMallocManaged) | 105.2 GB (98.6 GiB) |
| UVM page-tracking overhead (fixed) | ~7 GB |
| Graph/activation tensors | 2.2 GB |
| Packed KV (attention + indexer) @ 1M | ~9.1 GiB (was ~17.5 GiB FP32 attention-only) |
| OS / cuBLAS / other | ~4 GB |
| **peak measured @ 1M ctx** | **~102.8 GiB (fits under 128 GB)** |

Without `DS4_KV_TURBO`, both compressed-KV streams stay FP32 and overflow at 1M.
Packing the attention stream reclaims ~7.2 GiB; packing the indexer stream
reclaims ~1.5 GiB more. KV savings scale linearly with context (attention:
7.23 KiB/token; e.g. ~1.9 GiB at 256K, ~3.7 GiB at 512K).

## Performance

Decode throughput on GB10 (sm_121), K180 hybrid, managed + turbo4, single Spark:

| prompt | prefill t/s | decode t/s |
|---|---|---|
| short (~18 tok) | ~19–20 | **~12** |
| medium (~3500 tok) | ~315 | **~11** |
| long (148 KB / 36 688 tok) | ~281 | **~9–11** |

Decode throughput drops with prompt length because longer context → more
compressed KV rows → more attention work per token (attention is still <1% of
decode). The dominant cost (MoE weight loading) is prompt-length-independent.
Always benchmark at the same prompt length before/after a change.

Planned decode optimizations (MoE down expert-parallelism, software prefetch,
F16 `__half2` loads, getenv caching, CUDA Graph capture, more kernel fusion)
are tracked in [DECODE_OPTIMIZATION_PLAN.md](DECODE_OPTIMIZATION_PLAN.md).
Realistic combined target without MTP: ~13–15 t/s at medium prompt.

## GB10 bandwidth hierarchy (measured)

| path | GB/s |
|---|---|
| cudaMemcpy device cache (duplicates model) | ~118 |
| cudaMallocManaged + hints (single residency) | **97** |
| cudaMallocManaged un-hinted | 68 |
| malloc + ATS (4 KiB pages) | 36 |

## Verifying

```bash
make                                   # build (CUDA backend)
DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1 \
  ./ds4 -m ../DeepSeek-V4-Flash-REAP-K180-hybrid.gguf \
  -p "What is 2+2? Answer in one word." -n 8 --nothink --ctx 1048576
```

`verify.sh` runs math + coherence + factual checks; `bench.sh` is the
reproducible throughput harness (drops OS caches, 3 runs + median). Both target
K180 with the managed + turbo flags.

## Models

Pre-built hybrid GGUF models (NVFP4 + Q2_K, REAP-pruned) on HuggingFace:

| REAP plan | HuggingFace |
|---|---|
| K128 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K128-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K128-NVFP4) |
| K150 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K150-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K150-NVFP4) |
| K180 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K180-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K180-NVFP4) |

## Acknowledgements

- Forked from [antirez/ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo.
- Inspired by [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped)
  (a fork of ds4) and the
  [0xSero/DeepSeek-V4-Flash-162B-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF)
  release, which motivated running a REAP'd DS-V4 Flash on the Spark.
- Expert pruning follows **REAP — Router-weighted Expert Activation Pruning**
  from the paper ["REAP the Experts: Why Pruning Prevails for One-Shot MoE
  compression"](https://arxiv.org/abs/2510.13999) (Cerebras, ICLR 2026).
- NVFP4 format follows NVIDIA Model-Optimizer's `NVFP4QTensor`.
- DS-V4 attention architecture (CSA/HCA) per the DeepSeek-V4 paper.

## License

Same as upstream ds4 (see LICENSE).
