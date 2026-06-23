# ds4 — Spark serving & development guide

Forked from [antirez/ds4](https://github.com/antirez/ds4).  This document covers serving
configurations (flags, model sizes, memory budgets) and development flags for the
decode-optimizations work.

---

## Model matrix

| model | size | experts | gate/up quant | down quant | requires managed? | notes |
|---|---|---|---|---|---|---|
| **IQ2XXS** (full 256) | 81 GiB | 256 | IQ2_XXS (2.06 bpw) | Q2_K (2.625 bpw) | no | reference baseline, ~14 t/s |
| **K128** NVFP4 hybrid | 75 GiB | 128 | NVFP4 (4.5 bpw) | Q2_K (2.625 bpw) | no | fits without managed, ~12-13 t/s |
| **K150** NVFP4 hybrid | 86 GiB | 150 | NVFP4 (4.5 bpw) | Q2_K (2.625 bpw) | **yes** | OOMs without managed |
| **K180** NVFP4 hybrid | 99 GiB | 180 | NVFP4 (4.5 bpw) | Q2_K (2.625 bpw) | **yes** | OOMs without managed, ~11 t/s |

Memory budget on the 128 GiB Spark: model + UVM overhead (~7 GiB) + graph tensors (~2.2 GiB) + KV cache + OS (~4 GiB).  The non-managed path caches the working set in device memory via the on-demand cudaMemcpy span cache (~118 GB/s effective).  The managed path streams from `cudaMallocManaged` at ~97 GB/s.

---

## Serving flags

### Production flags for K128 @ 1M context

```bash
# With CUDA Graph (3-4x decode speedup, default off)
DS4_KV_TURBO=1 DS4_CUDA_DECODE_GRAPH=1 ./ds4-server \
  -m ../DeepSeek-V4-Flash-REAP-K128-hybrid.gguf \
  --ctx 1048576 \
  --host 0.0.0.0 \
  --port 17777 \
  --kv-disk-dir /tmp/ds4-kv \
  --kv-disk-space-mb 20480 \
  --kv-cache-cold-max-tokens 30000 \
  --kv-cache-continued-interval-tokens 10000

# Without CUDA Graph (stable/default)
DS4_KV_TURBO=1 ./ds4-server \
  -m ../DeepSeek-V4-Flash-REAP-K128-hybrid.gguf \
  --ctx 1048576 \
  --host 0.0.0.0 \
  --port 17777
```

### KV cache flags

| flag | default | description |
|---|---|---|
| `--kv-disk-dir DIR` | (none) | Enable persistent disk KV checkpoints. **Create the dir first.** |
| `--kv-disk-space-mb N` | 4096 | Disk budget in MiB |
| `--kv-cache-min-tokens N` | 512 | Don't save checkpoints shorter than N tokens |
| `--kv-cache-cold-max-tokens N` | 30000 | Save cold first prompts in [min, N]; 0 disables |
| `--kv-cache-continued-interval-tokens N` | 10000 | Save aligned frontiers during long conversations; 0 disables |
| `--kv-cache-boundary-trim-tokens N` | 32 | Trim tail tokens before cold saves (BPE boundary) |
| `--kv-cache-boundary-align-tokens N` | 2048 | Align cold saves to this token multiple; 0 = no align |
| `--kv-cache-reject-different-quant` | false | Refuse checkpoints from models with different expert quant |
| `--trace FILE` | (none) | Detailed per-request trace log (cache hits, token counts, timing) |

**Best practice for agentic workloads:** always enable `--kv-disk-dir` with at least
`--kv-disk-space-mb 20480`.  Memory-token reuse handles back-to-back tool calls,
but the disk cache is the fallback for evicted sessions.  Without it, every eviction
triggers a full re-prefill (issue #7).

### Turbo KV (`DS4_KV_TURBO=1`)

Packs compressed KV rows from FP32 (2048 B/row) to FP8 turbo4 format (584 B/row,
3.51× compression).  **Required for 1M context** on all model sizes — the FP32 KV
overflows at 1M.  Also packs the indexer KV (200 B/row, 2.56× compression).  Context
buffers at 1M: ~9.1 GiB packed vs ~17.5 GiB FP32.

### CUDA Graph (`DS4_CUDA_DECODE_GRAPH=1`)

Captures the 43-layer decode tape once (first token, ~5 ms CPU) and replays it for
all subsequent tokens (one `cudaGraphLaunch`, ~0.3 ms CPU).  Eliminates the ~98 ms
per-token CPU dispatch, leaving ~74 ms GPU work.  **Gated off by default** — the
manual dispatch path is the proven stable path.

Dynamic-arg node updates keep `pos`, `token`, `raw_row`, `n_raw`, and `raw_start`
correct across replays.  The compressor emit is handled by a device-conditional
fused kernel.  8 kernel types are classified for arg updates.

### Managed memory (`DS4_CUDA_MANAGED_MODEL=1`)

Required for K150 and K180 (don't fit without it).  Loads the GGUF into
`cudaMallocManaged` (single residency at ~97 GB/s) instead of the default mmap +
on-demand span cache (~118 GB/s).  Slightly slower (~8-10%) but enables serving
models that exceed the free RAM after context buffers.

**Do not use for K128 or IQ2XXS** — they fit without managed and get better t/s
from the default span-cache path.

---

## Development / profiling flags

| flag | description |
|---|---|
| `DS4_CUDA_DECODE_GRAPH=1` | Enable CUDA Graph capture/replay (decode tape) |
| `DS4_CUDA_MANAGED_MODEL=1` | Load model into managed memory instead of mmap |
| `DS4_KV_TURBO=1` | Enable FP8-packed KV cache (turbo4) |
| `DS4_GPU_GRAPH_TOKEN_PROFILE=1` | Per-token encode/execute/read timing |
| `DS4_GPU_DECODE_STAGE_PROFILE=1` | Per-layer per-stage decode timing |
| `DS4_CUDA_MOE_PROFILE=1` | MoE gate/up/down per-layer timing |
| `DS4_CUDA_NO_Q8_F16_CACHE=1` | Disable Q8→F16 dequant cache (**broken** — issue #1) |
| `DS4_CUDA_SERIAL_F16_MATMUL=1` | Use serial F16 matmul (bypass __half2 kernel) |
| `DS4_NO_TURBO4_DIRECT=1` | Disable turbo4 packed-read attention kernel |

### Profiling examples

```bash
# Token-level timing (encode=CPU dispatch, execute=GPU sync, read=logits D2H)
DS4_KV_TURBO=1 DS4_GPU_GRAPH_TOKEN_PROFILE=1 ./ds4 -m K128.gguf -p "..." -n 20

# Per-layer stage breakdown
DS4_KV_TURBO=1 DS4_GPU_DECODE_STAGE_PROFILE=1 ./ds4 -m K128.gguf -p "..." -n 8

# MoE expert timing
DS4_KV_TURBO=1 DS4_CUDA_MOE_PROFILE=1 ./ds4 -m K128.gguf -p "..." -n 12
```

### Power/util logging (`wd.sh`)

```bash
# Start logger at 50 ms interval (20 Hz), capture decode steady-state
./wd.sh /tmp/wd.csv 50 &
./ds4 -m K128.gguf -p "..." -n 120 ...
kill %1

# Analyze
python3 -c "
import csv, statistics
rows = [dict(r) for r in csv.DictReader(open('/tmp/wd.csv'))]
dec = [r for r in rows if 3000 <= int(r['t_ms']) <= int(rows[-1]['t_ms'])-2000]
g = [int(r['gpu_util_pct']) for r in dec]
print(f'gpu util: {statistics.median(g)}% median, {sum(g)/len(g):.0f}% avg')
p = [float(r['power_w']) for r in dec]
print(f'power: {statistics.median(p):.1f}W median, {sum(p)/len(p):.1f}W avg')
"
```

---

## Building & testing

```bash
# Build (CUDA backend, native arch)
make -j8

# Run the reference tests (needs a GGUF model)
DS4_KV_TURBO=1 ./ds4 -m ../DeepSeek-V4-Flash-REAP-K128-hybrid.gguf \
  -p "What is 2+2? Answer in one word." -n 8 --nothink --ctx 1048576

# verify.sh: math + coherence + factual
DS4_KV_TURBO=1 bash verify.sh

# Bench: reproducible throughput, 3 runs + median
DS4_KV_TURBO=1 bash bench.sh

# NVFP4 verify (decodes "Four" + "Paris" + coherent story)
DS4_KV_TURBO=1 ./ds4 -m ../DeepSeek-V4-Flash-REAP-K128-hybrid.gguf \
  -p "What is the capital of France? One word." -n 16 --nothink --ctx 1048576
```

---

## Branch conventions

- `main` — clean, tested code (all features gated behind env flags, default paths unchanged)
- `decode-optimizations` / `decode-optimizations-dev` — decode-opt work, may have WIP commits
- `issue-batch` — working branch for Forgejo issues
- Feature branches: create from `main`, merge to `main` after testing

---

## Quick reference: optimum configs

| goal | config |
|---|---|
| Fastest single-request decode | K128, no managed, turbo KV, CUDA Graph |
| Best precision (NVFP4, near-lossless) | K128, no managed, turbo KV |
| Largest model (180 experts) | K180, managed, turbo KV |
| Agentic workloads | K128, no managed, turbo KV, CUDA Graph, disk KV cache on |
| Development/debug | manual dispatch (default), DS4_GPU_DECODE_STAGE_PROFILE=1 |
