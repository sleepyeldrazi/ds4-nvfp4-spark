# ds4 - Mixed NVFP4 serving of DeepSeek V4 Flash on the NVIDIA Spark family (GB10)

> **⚠️ This GitHub repository is for archival / mirror purposes only.**
> Active development happens at
> **[git.kokoham.com/sleepy/ds4-nvfp4-spark](https://git.kokoham.com/sleepy/ds4-nvfp4-spark)**.
> Please file issues, PRs, and check for updates there.

Forked from [antirez/ds4](https://github.com/antirez/ds4), a standalone DeepSeek-V4
Flash (~162B) inference engine (C + CUDA, no GGML dependency). This fork adds
**NVFP4 expert quantization**, a **managed-memory serving path** for GB10 unified
memory, the **lossless MXFP4→NVFP4 GGUF emission** pipeline, an **FP8-packed KV
cache**, and **REAP expert pruning** to fit the model on a single 128 GB device.

Targeted at the DGX Spark desktop and designed to be **day-one ready** for the
upcoming [RTX Spark](https://www.nvidia.com/en-us/products/rtx-spark/) lineup
(laptops, mini PCs, workstations from ASUS, Dell, HP, Lenovo, Microsoft, MSI,
and others, shipping fall 2026). All devices share the same GB10 Grace Blackwell
superchip so this engine should work across the whole family out of the box.

Inspired by [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped) and
the [0xSero/DeepSeek-V4-Flash-162B-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF)
release.

## What this fork adds

- **REAP expert pruning** (K128 / K150 / K180): selects top-K routed experts per
  layer by FP4/FP8 REAP score, shrinking the model to fit on a 128 GB device while
  preserving quality.
- **NVFP4 expert quantization**: lossless MXFP4→NVFP4 repack with custom `__dp4a`
  decode kernels (~140 GB/s on GB10). NVFP4 is slightly slower than the IQ2_XXS
  path in practice (it reads more bytes per token), but delivers **far better
  precision** (4.5 bpw vs 2.06 bpw).
- **FP8-packed KV cache** (`DS4_KV_TURBO=1`): real packed FP8 storage matching the
  DS-V4 paper spec (3.5× compression over FP32), reclaiming ~7.2 GiB at 1M context.
  Both attention and indexer KV streams are packed.
- **Managed-memory serving** (`DS4_CUDA_MANAGED_MODEL=1`): single-residency loading
  via `cudaMallocManaged` + hints. This avoids duplicating the model in RAM,
  making K180 at 1M context fit, but costs **1-3 t/s** of decode throughput
  compared to the default non-managed path. Use it when you need the RAM savings
  (K180 requires it); K128 and K150 can skip it if top speed matters more.

## Models

Pre-built hybrid GGUF models (NVFP4 gate/up + Q2_K down, REAP-pruned) on HuggingFace:

| REAP plan | HuggingFace |
|---|---|
| K128 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K128-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K128-NVFP4) |
| K150 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K150-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K150-NVFP4) |
| K180 | [sleepyeldrazi/DeepSeek-V4-Flash-REAP-K180-NVFP4](https://huggingface.co/sleepyeldrazi/DeepSeek-V4-Flash-REAP-K180-NVFP4) |

K150 and K180 models are uploaded following K128 (one at a time). The K128 repo is
the first available.

## Memory budget (128 GB device)

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
7.23 KiB/token).

## Performance

Decode throughput on GB10 (sm_121a), K180 hybrid, managed + turbo4, single device:

| prompt | prefill t/s | decode t/s |
|---|---|---|
| short (~18 tok) | ~19–20 | **~12** |
| medium (~3500 tok) | ~315 | **~11** |
| long (148 KB / 36 688 tok) | ~281 | **~9–11** |

Decode throughput drops with prompt length because longer context → more
compressed KV rows → more attention work per token (attention is still <1% of
decode). The dominant cost (MoE weight loading) is prompt-length-independent.
Always benchmark at the same prompt length before/after a change.

Decode is **bandwidth-bound, not compute-bound** - the GPU draws ~33 W at 96%
utilization. Planned optimizations (MoE down expert-parallelism, software
prefetch, CUDA Graph capture, kernel fusion) target ~13–15 t/s. MTP is planned
as future work but requires retraining the draft heads to match the REAP-pruned
expert set (the current un-pruned MTP heads produce garbage at K128/K150/K180).

## GB10 bandwidth & quant analysis

Measured on the DGX Spark (GB10, sm_121a, CUDA 13.2, 128 GB unified LPDDR5X).

### Hardware ceilings

| operation | measured | note |
|---|---|---|
| F32 pure read | 247 GB/s (90% of spec) | physical max |
| read + write (copy) | 207 GB/s (76% of spec) | real ceiling for any output kernel |
| F16/BF16 read | 200 GB/s | sub-word loads are *slower* than F32 on this controller |

The 273 GB/s spec is physically unreachable for real workloads. The honest ceilings:
~247 GB/s pure read, ~207 GB/s read+write. Every real kernel tops out near 207.

### Per-precision decode bandwidth (N=32768, M=1 weight-streaming GEMV)

| format | bpw | kernel | GB/s | bound |
|---|---|---|---|---|
| Q8_0 | 8.5 | naive dequant | **228** | bandwidth (best) |
| NVFP4 packed (no scales) | 4.0 | naive LUT | 186 | bandwidth-ish |
| **Q2_K** | 2.625 | **`__dp4a`** | **160** | near-bandwidth |
| **NVFP4** (prod, w/ e4m3 scales) | **4.5** | **`__dp4a`** | **~140** | mixed |
| **IQ2_XXS** | 2.06 | **`__dp4a`** | **58.6** | **dequant-compute ✗** |

**Key insight**: IQ2_XXS is dequant-compute-bound at 58 GB/s. NVFP4 reads *more*
bytes (4.5 vs 2.06 bpw) and its `__dp4a` kernel runs at 140 GB/s — fast for its
bit-width, but the extra bytes make it **slightly slower end-to-end** than an
IQ2_XXS-based build. The tradeoff is worth it: NVFP4 delivers **far better
precision** (4.5 vs 2.06 bpw) at roughly similar speed. The hybrid recipe (NVFP4
gate/up + Q2_K down) keeps down experts at the fast Q2_K quant (160 GB/s, near
ceiling) while using NVFP4 only where the precision matters most. Q8_0 at 228
GB/s saturates the memory controller and is kept for attention/shared/head
tensors.

## Usage

```bash
# Build (requires CUDA toolkit, targets GB10 sm_121a)
make
```

Builds three binaries:
- `ds4` - interactive CLI
- `ds4-server` - HTTP API server
- `ds4-bench` - throughput benchmark

### Quick serve

```bash
# K180 at full 1M context (recommended flags)
DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1 \
  ./ds4 -m hybrid-K180.gguf -p "Your prompt" --ctx 1048576

# K128 / K150 can load without managed mode. Managed mode saves RAM but costs
# 1-3 t/s - skip it on K128/K150 if you want maximum throughput:
./ds4 -m hybrid-K128.gguf -p "Your prompt"
```

> **K180 requires `DS4_CUDA_MANAGED_MODEL=1`.** At 98.6 GiB the model does not
> fit in any single non-managed allocation on 128 GB. The managed path
> is the only way to load K180, but it costs **1-3 t/s** vs non-managed loading
> (which is only available on smaller models that fit without it).

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

# 3. Serve on the Spark
DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1 ./ds4 -m hybrid-K180.gguf -p "..." --ctx 1048576
```

The quant recipe: gate (w1) + up (w3) experts → **NVFP4**, down (w2) experts →
**Q2_K**, attention/shared/head → **copy from template** (preserves required
f16/f32). Non-expert tensors use the copy policy (don't force `--attention q8_0`
- it over-applies to f16-required HC tensors).

## Verifying

```bash
make

# Quick correctness check
DS4_CUDA_MANAGED_MODEL=1 DS4_KV_TURBO=1 \
  ./ds4 -m ../DeepSeek-V4-Flash-REAP-K180-hybrid.gguf \
  -p "What is 2+2? Answer in one word." -n 8 --ctx 1048576
```

`verify.sh` runs math + coherence + factual checks; `bench.sh` is the
reproducible throughput harness (drops OS caches, 3 runs + median). Both target
K180 with the managed + turbo flags.

## Technical details

### REAP expert pruning

`make_reap_plan.py` selects top-K routed experts per layer by FP4/FP8 REAP score
(router gate-value × activation norm) from a calibration observation file.
Hash-routed layers (0–2) keep all 256 experts; routed layers (3–42) keep K.
Three plans are shipped: K128, K150, K180.

`gguf-tools/deepseek4-quantize.c` performs one-pass pruning + quantization +
GGUF emission. The GGUF carries `hash_preserved` / `router_masked` masks so the
engine routes around pruned experts without code changes.

### NVFP4 expert kernels

The HF source experts are MXFP4 (e2m1 + e8m0 per-32). NVFP4 is e2m1 + e4m3
per-16 + per-expert `scale_2`. The **e2m1 nibbles are identical** - MXFP4→NVFP4
is a lossless scale-only transform (no requant, no amax needed for weights).
Verified bit-exact via round-trip test.

The GGUF writer stores NVFP4 tensors with the ds4q NVFP4 type id **40**, which
is intentionally not in ds4's `gguf_types` table. The loader prints
`warning: tensor ... has unsupported GGUF type 40` for each NVFP4 tensor -
**this is expected and harmless**: `ds4_weights.c` rebinds those tensors by
name (`.nvfp4_weight`) and forces the dispatch type to 31.

### FP8-packed KV cache

DS-V4 attention stores compressed KV at FP8 density. ds4's original code did
in-place FP8 value quantization but wrote the result back into **FP32 buffers**
(4× memory waste, zero compression). This fork implements real packed storage:

- **Attention-compressed packed row (584 B vs 2048 B FP32 = 3.51×)**:
  `[ 448 e4m3 | 7 e8m0 scales | 1 pad | 128 B BF16 (rot) ]`
- **Indexer-compressed packed row (200 B vs 512 B FP32 = 2.56×)**,
  head_dim=128, n_rot=64 - reuses the same turbo4 pack/unpack kernels.
- Uses CUDA 13 native `cuda_fp8.h` types (`__nv_fp8_e4m3`, `__nv_fp8_e8m0`).
- GB10-specific: +1 pad byte for even BF16 alignment (misaligned 2-byte loads
  crash on sm_121a); BF16 via `uint16_t*` (sm_121a compiler bug workaround).
- Correctness: greedy decode matches FP32 path 29/30 tokens (diverges at final
  token - healthy FP8 quantization).

### Managed-memory serving

`DS4_CUDA_MANAGED_MODEL=1` avoids duplicating the model in RAM by loading into a
`cudaMallocManaged` buffer with hints (`SetReadMostly + SetPreferredLocation=device`
+ `cudaMemPrefetchAsync`) instead of the default `mmap` path. This cuts RAM usage
significantly (the model lives once instead of twice), but **costs 1-3 t/s**
depending on context length: managed reads run at ~97 GB/s vs ~118 GB/s for the
default path. It's required for K180 (the model doesn't fit otherwise) and
optional for K128/K150.

### Bandwidth hierarchy (measured)

| path | GB/s |
|---|---|
| cudaMemcpy device cache (duplicates model) | ~118 |
| cudaMallocManaged + hints (single residency) | **97** |
| cudaMallocManaged un-hinted | 68 |
| malloc + ATS (4 KiB pages) | 36 |

## Acknowledgements

- Forked from [antirez/ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo.
- Inspired by [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped)
  and the [0xSero/DeepSeek-V4-Flash-162B-GGUF](https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF)
  release, which motivated running a REAP'd DS-V4 Flash on the Spark.
- Expert pruning follows **REAP - Router-weighted Expert Activation Pruning**
  from ["REAP the Experts: Why Pruning Prevails for One-Shot MoE
  compression"](https://arxiv.org/abs/2510.13999) (Cerebras, ICLR 2026).
- NVFP4 format follows NVIDIA Model-Optimizer's `NVFP4QTensor`.
- DS-V4 attention architecture (CSA/HCA) per the DeepSeek-V4 paper.

## License

Same as upstream ds4 (see LICENSE).
