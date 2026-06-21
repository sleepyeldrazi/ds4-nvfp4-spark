# ds4 — NVFP4 hybrid serving on DGX Spark (GB10)

Forked from [antirez/ds4](https://github.com/antirez/ds4), a standalone DeepSeek-V4
Flash inference engine (C + CUDA/Metal, no GGML dependency). This fork adds **NVFP4
expert quantization**, a **managed-memory serving path** for GB10 unified memory,
and the **lossless MXFP4→NVFP4 GGUF emission** pipeline.

## What this fork adds

### NVFP4 expert kernels + serving (verified)
- **Lossless MXFP4→NVFP4 repack**: the HF source experts are MXFP4 (e2m1 + e8m0
  per-32). NVFP4 is e2m1 + e4m3 per-16 + per-expert scale_2. The e2m1 nibbles are
  identical — the conversion is a lossless scale-only transform (no requant, no
  amax needed for weights). Verified bit-exact via a round-trip test.
- **ds4_cuda.cu**: `cuda_block_nvfp4` struct, `dev_dot_nvfp4_q8_K_block` (`__dp4a`
  MMQ, ~140 GB/s on GB10), batched tile8/tile16 rowspan kernels + two n_tokens==1
  decode kernels, per-expert `scale_2` array threaded through dispatch.
- **ds4.c**: name-based NVFP4 GGUF loader (`.nvfp4_weight` + `.nvfp4_scale_2`),
  forces `DS4_TENSOR_NVFP4` (id 31) for dispatch.
- **gguf-tools/deepseek4-quantize.c**: NVFP4 emission in the HF→GGUF quantizer
  (one pass: REAP prune + NVFP4 gate/up + Q2_K down + copy-policy for attn/head).

### Managed-memory serving path (`DS4_CUDA_MANAGED_MODEL=1`)
GB10 has hardware ATS (`PageableMemAccessUsesHostPageTables=1`) — the GPU reads
CPU-allocated memory directly, coherent, no copy. This fork loads the GGUF into a
`cudaMallocManaged` buffer (via chunked `pread` + `posix_fadvise DONTNEED`) with
`cudaMemAdvise(SetReadMostly + SetPreferredLocation=device)` + `cudaMemPrefetchAsync`,
and gates off the redundant on-demand cudaMemcpy span cache. Result: single
residency at ~97 GB/s (measured) → **K180 hybrid (98.6 GiB) serves at ~12 t/s,
correct, ~118 GiB peak** on the 128 GB Spark.

### Encode-path bugs fixed (8 total)
See `cuda_debug/ENCODE_VERIFICATION.md` for the full list. The load-bearing ones:
e4m3 bit layout (`(E<<3)|M` not `(E<<4)|M`), per-expert (not per-row) k_max,
interleaved `cuda_block_nvfp4` layout, the 2^8=256 max-scale edge, GGUF header
`n_tensors` count, loader type-id mismatch.

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
  --routed-w1 nvfp4 --routed-w3 nvfp4 --routed-w2 q2_K \
  --out hybrid-K180.gguf

# 3. Serve on the Spark
DS4_CUDA_MANAGED_MODEL=1 ./ds4 -m hybrid-K180.gguf -p "..." --ctx 8192
```

The quant recipe: gate (w1) + up (w3) experts → **NVFP4**, down (w2) experts →
**Q2_K**, attention/shared/head → **copy from template** (preserves required f16/f32).
Non-expert tensors use the copy policy (don't force `--attention q8_0` — it
over-applies to f16-required HC tensors).

## Memory budget on the 128 GB Spark

| component | size |
|---|---|
| K180 model (cudaMallocManaged) | 105.2 GB (98.6 GiB) |
| UVM page-tracking overhead (fixed) | ~7 GB |
| Graph/activation tensors | 2.2 GB |
| KV cache (varies with ctx) | 0.15 GiB @ 8K → 17.5 GiB @ 1M (FP32) |
| OS / cuBLAS / other | ~4 GB |

At short context: ~118 GB (fits). At 1M context: the KV grows to ~17.5 GiB (FP32
storage) → ~129 GB (over). **The KV is stored at FP32, not the DS-V4 paper's FP8** —
the `turbo_packv4` FP8-packed-KV path is stubbed. Implementing FP8 KV storage (the
paper spec: e4m3 for non-RoPE dims, BF16 for RoPE) reclaims ~13 GiB → K180 under
120 GB at full 1M context. This is the next step.

## GB10 bandwidth hierarchy (measured)

| path | GB/s |
|---|---|
| cudaMemcpy device cache (duplicates model) | ~118 |
| cudaMallocManaged + hints (single residency) | **97** |
| cudaMallocManaged un-hinted | 68 |
| malloc + ATS (4 KiB pages) | 36 |

## Acknowledgements

- Forked from [antirez/ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo.
- NVFP4 format follows NVIDIA Model-Optimizer's `NVFP4QTensor`.
- DS-V4 attention architecture (CSA/HCA) per the DeepSeek-V4 paper.

## License

Same as upstream ds4 (see LICENSE).
