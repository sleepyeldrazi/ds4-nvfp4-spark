# DS4 Decode Optimization Plan

> Performance spec for non-MTP decode optimizations on DGX Spark (GB10, sm_121).
>
> **Baseline (varies by prompt length):**
>
> | prompt | tokens | prefill t/s | decode t/s | decode ms/token |
> |--------|--------|-------------|------------|-----------------|
> | short ("Hello, what is 2+2?") | ~18 | 19–20 | **12.4–12.5** | ~80 ms |
> | medium (~3500 tok, /tmp/medium_prompt.txt) | ~3500 | 315–323 | **9.3–9.5** | ~105 ms |
> | long (148 KB security prompt) | 36,688 | 281–283 | **8.8** | ~114 ms |
>
> Decode throughput **drops with prompt length** because longer context → more
> compressed KV rows → more comp rows to attend over → more attention work per
> token (though attention is still <1% of decode). The dominant cost (MoE
> weight loading) is prompt-length-independent, so the shift is small (~30%
> range) but real. **Always benchmark at the same prompt length before/after a
> change** — a 12 t/s result on a short prompt is NOT a win if the medium-prompt
> rate didn't improve.
>
> Weight bandwidth floor = 28 ms (27% of decode at medium prompt). The gap is
> L2/TLB miss overhead from random expert access on 75 GB managed memory, UMA
> contention, and quantized compute.
>
> **Hard constraint:** ~0 RAM headroom. Model peaks at ~123/128 GB at 1M context.
> No room for a second copy of weights. All proposals must be RAM-neutral or
> RAM-negative.

## What the codebase already does well

- **Split-flush pipelining:** `gpu_graph_encode_token_raw_swa` (ds4_gpu.inc:2905)
  dispatches layers 0–3 then `ds4_gpu_flush_commands()` (cudaDeviceSynchronize),
  then dispatches layers 4–42 without per-layer sync. Only **2 syncs per token**,
  not 43. CPU dispatch (~3.2 ms) overlaps with GPU layers 0–3.
- **`__dp4a` for all INT8 dots:** Q8_0 matmul, NVFP4 MoE, IQ2/Q4_K paths.
- **Q8→F16 dequant cache** for attention projections (`cuda_q8_f16_ptr`,
  ds4_cuda.cu:527). Caches q_a, q_b, output_a, output_b in `cudaMalloc`'d device
  memory — avoids managed-memory TLB overhead on the 136 MB/layer attention
  weight traffic.
- **Online softmax** in attention (single pass, no materialized score buffer for
  the heads8 path).
- **Fused `moe_down_sum6_nvfp4`:** 6 routed experts in one kernel, no
  intermediate write per expert.
- **`cp.async` double-buffer** in `attention_decode_mixed_heads8_online_kernel`.
- **Expert tile sorting** for batched prefill MoE.
- **Existing fusions:** `head_rms_norm_rope_tail_kernel` (norm+RoPE),
  `hc_split_weighted_sum_norm_fused_kernel` (HC split+norm),
  `dsv4_qkv_rms_norm_rows_tensor` (Q+KV RMS norm in one kernel),
  `ds4_gpu_shared_down_hc_expand_q8_0_tensor` (down+HC expand fused).

---

## Opportunity 1 — MoE down: parallelize 6 experts across warps

**Impact:** 10–20% (~10–20 ms/token). Highest-impact kernel-level change.

### Problem

`moe_down_sum6_nvfp4_qwarp32_kernel` (ds4_cuda_moe.cuh:1261) processes 6
experts **sequentially per thread** via `#pragma unroll`:

```c
for (uint32_t slot = 0; slot < 6u; slot++) {      // ← serial
    int32_t expert_i = selected[slot];
    const cuda_block_nvfp4 *wr = down_base + expert_i * down_expert_bytes + row * down_row_bytes;
    for (uint32_t b = lane; b < midq_blocks; b += 8u)
        acc += dev_dot_nvfp4_q8_K_block(wr + b, xq + b, down_scale_2[expert_i]);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) total += acc;
}
```

Each of the 6 experts' down weights (~4.7 MB each, 4096×2048 NVFP4) are loaded
serially. The L2 (24 MB) cannot hold all 6 experts simultaneously → 6× L2 miss
bursts per row group. The warp loads expert 0, computes, loads expert 1,
computes, … — no overlap between expert weight fetches.

### Fix

Assign each of the 6 experts to a **different warp** within the 256-thread block
(8 warps available, 6 used for experts, 2 idle or used for the shared expert).
Each warp loads its expert's weights concurrently with the other 5 warps loading
theirs → 6× overlap of L2 fills.

**New kernel:** `moe_down_sum6_nvfp4_warpexperts_kernel` in ds4_cuda_moe.cuh.

Sketch:
```c
__global__ static void moe_down_sum6_nvfp4_warpexperts_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t midq_blocks, uint32_t out_dim,
        const float *down_scale_2) {
    const uint32_t warp = threadIdx.x >> 5;       // 0..7
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 32u + (warp < 6u ? lane : 0);  // only 6 warps do work
    if (warp >= 6u || row >= out_dim) {
        // idle warps participate in the reduction barrier only
    }
    __shared__ float warp_totals[8][33];  // per-warp partial, 33 for padding

    float acc = 0.0f;
    if (warp < 6u && row < out_dim) {
        int32_t expert_i = selected[warp];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_nvfp4 *wr = down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes
                                                 + (uint64_t)row * down_row_bytes;
        const cuda_block_q8_K *xq = midq + (uint64_t)warp * midq_blocks;
        for (uint32_t b = lane; b < midq_blocks; b += 32u)
            acc += dev_dot_nvfp4_q8_K_block(wr + b, xq + b, down_scale_2[expert_i]);
        acc = warp_sum_f32(acc);  // full warp reduce (32 threads now, not 8)
        if (lane == 0) warp_totals[warp][row % 32] = acc;
    }
    __syncthreads();
    // Reduce 6 warps' contributions for this block's 32 rows
    // Each warp handles a subset of rows for the final sum
    ...
}
```

**Trade-off:** Each expert gets 32 threads (full warp) instead of 8
(quarter-warp). This is **4× more parallelism per expert** — each thread does
`midq_blocks/32` blocks instead of `midq_blocks/8`. With midq_blocks=8
(2048/256), that's 0.25 blocks/thread — too few. Need to adjust: either use
smaller blocks or group multiple rows per warp. The right granularity needs
empirical tuning, but the principle holds: **concurrent expert fetch > serial
expert fetch**.

**Alternative (simpler):** Keep the 8-thread-per-expert granularity but issue all
6 experts' first weight block loads before any compute, using `__ldg()` to
prefetch into L2, then proceed with the serial compute loop. This is a hybrid of
opportunities 1 and 2.

### Files to change
- `src/gpu/ds4_cuda_moe.cuh` — new kernel `moe_down_sum6_nvfp4_warpexperts_kernel`
- `src/gpu/ds4_cuda_moe_dispatch.cuh:485` — dispatch the new kernel when
  `use_direct_down_sum6 && down_nvfp4` (replace
  `moe_down_sum6_nvfp4_qwarp32_kernel` launch)
- Same pattern for `moe_down_q4K_sum6_qwarp32_kernel` if Q4_K experts are used

### Verification
- A/B test: `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1` (old path) vs new kernel
- Output must be bit-identical (only execution order changes, not math)
- Profile with `DS4_CUDA_MOE_PROFILE=1` to compare down-projection time

---

## Opportunity 2 — Software prefetch in MoE gate/up kernel

**Impact:** 5–15% (~5–15 ms/token). Easiest to implement.

### Problem

`moe_gate_up_mid_decode_lut_nvfp4_kernel` (ds4_cuda_moe.cuh:324) starts computing
dot products immediately. The first access to each expert's weight row is a cold
L2 miss + TLB miss on the 75 GB managed address space. The hardware L2
prefetcher needs a few cache lines of head start to recognize the streaming
pattern.

### Fix

Add a **prefetch phase** at the start of the kernel, before the compute loop.
Each block issues `prefetch.global.L2` PTX for its expert's weight rows, then
proceeds to compute. By the time the first `dev_dot_nvfp4_q8_K_block` reads the
weights, the L2 prefetcher has been primed.

In `moe_gate_up_mid_decode_lut_nvfp4_kernel`, after computing `gr`/`ur` row
pointers and before the `for (uint32_t rr = 0; rr < 4u; rr++)` loop:

```c
/* Prefetch the first few weight rows into L2 before computing. */
for (uint32_t rr = 0; rr < 4u && blockIdx.x * 128u + rr * 32u < expert_mid_dim; rr++) {
    uint32_t row = blockIdx.x * 128u + rr * 32u + (threadIdx.x >> 3u);
    if (row < expert_mid_dim) {
        const void *gr_row = gate_base + (uint64_t)expert * gate_expert_bytes
                           + (uint64_t)row * gate_row_bytes;
        const void *ur_row = up_base + (uint64_t)expert * gate_expert_bytes
                           + (uint64_t)row * gate_row_bytes;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(gr_row));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ur_row));
    }
}
```

Also add the same to `moe_down_sum6_nvfp4_qwarp32_kernel` and the expert-tile
kernels.

### PTX verification

`prefetch.global.L2` is supported on all sm_70+ (confirmed in PTX ISA).
On sm_120/121 it's a no-op hint if the line is already in L2, and a fetch if
not. Safe to issue unconditionally.

### Files to change
- `src/gpu/ds4_cuda_moe.cuh` — add prefetch preamble to:
  - `moe_gate_up_mid_decode_lut_nvfp4_kernel` (line 324)
  - `moe_gate_up_mid_expert_tile8_rowspan_nvfp4_kernel` (line 926)
  - `moe_down_sum6_nvfp4_qwarp32_kernel` (line 1261)
  - `moe_gate_up_mid_decode_lut_q4K_qwarp32_kernel` (line 1177) if Q4_K path
- No dispatch changes needed — the kernels are called from the same places.

### Verification
- Output bit-identical (prefetch is a hint, doesn't change results)
- A/B via env flag `DS4_CUDA_NO_MOE_PREFETCH` (add to gate the prefetch block)
- Measure with `DS4_CUDA_MOE_PROFILE=1`

---

## Opportunity 3 — F16 matmul: `__half2` paired loads

**Impact:** 5–10% (~5–10 ms/token). Medium effort.

### Problem

`matmul_f16_ordered_chunks_kernel` (ds4_cuda_matmul.cuh:42) and
`matmul_f16_pair_ordered_chunks_kernel` (line 74) load one `__half` at a time
and convert to `float` for FMA:

```c
for (uint64_t i = k0; i < k1; i++) {
    sum += __half2float(wr[i]) * xr[i];   // 1 F16 load + 1 F32 load + 1 FMA
}
```

Each iteration issues 2 separate loads (1 for `wr[i]`, 1 for `xr[i]`). The F16
load fetches 2 bytes but the load instruction handles 16-bit — wasting half the
load unit bandwidth.

### Fix

Load 2 `__half` as `__half2` (4 bytes, one 32-bit load), convert to `float2`,
do 2 FMAs per iteration:

```c
const __half2 *wr2 = (const __half2 *)wr;
const float2 *xr2 = (const float2 *)xr;
for (uint64_t i = k0 / 2; i < k1 / 2; i++) {
    float2 w2f = __half22float2(wr2[i]);
    float2 xv2 = xr2[i];
    sum += w2f.x * xv2.x;
    sum += w2f.y * xv2.y;
}
```

This halves the number of load instructions for the F16 weight. The activation
(`xr`) is already `float`, so `float2` loads are natural.

**Edge case:** `in_dim` may be odd. Handle the last element separately.

### Which matmuls benefit

The F16 matmul is used via `gpu_graph_matmul_plain_tensor` (ds4_gpu.inc:2229)
for:
- **HC attn projection** (`hc_attn_fn`): in_dim=16384 (4×4096), out_dim=24
- **HC ffn projection** (`hc_ffn_fn`): same shape
- **Router** (`ffn_gate_inp`): in_dim=4096, out_dim=256
- **Indexer projections** (`indexer_proj`, `indexer_attn_q_b`): various
- **Output head** (`output_hc_fn`, head matmuls)

The attention Q/K/V/O projections use the Q8_0 path (with F16 cache), not this
F16 kernel — so they're unaffected.

### Files to change
- `src/gpu/ds4_cuda_matmul.cuh` — rewrite `matmul_f16_ordered_chunks_kernel`
  and `matmul_f16_pair_ordered_chunks_kernel` with `__half2` loads
- No dispatch changes — same launch config, same grid

### Verification
- Output must match within FP rounding (reassociation of FMAs may change LSBs)
- Compare with `DS4_CUDA_SERIAL_F16_MATMUL=1` reference path
- Tolerance: < 1e-4 relative error (F16 precision floor)

---

## Opportunity 4 — Cache `getenv()` + constant lookups

**Impact:** 3–5% (~3–5 ms/token). Easy, low risk.

### Problem

The decode path calls `getenv()` **157 times per token** (37 in ds4_gpu.inc, 120
in the CUDA dispatch headers). Each `getenv()` is a libc call that scans the
environment block — ~100–200 ns each. 157 × 150 ns = ~24 μs/token. Not huge, but
pure waste.

More significantly, many of these gate **kernel selection** that is constant for
the entire session:
- `DS4_DECODE_INDEXER_TOP_K` (ds4_gpu.inc:1093) — parsed via `strtoul` every call
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6` (moe_dispatch.cuh:148) — checked every MoE launch
- `DS4_CUDA_NO_Q8_DP4A` (ds4_cuda.cu:504) — checked every Q8 matmul
- `DS4_CUDA_NO_Q8_F16_CACHE` (ds4_cuda.cu:469) — checked every cache lookup
- `DS4_GPU_INDEXER_STAGE_PROFILE` (ds4_gpu.inc:1441) — checked every layer
- `DS4_NO_TURBO4_DIRECT` (ds4_gpu.inc:1704) — checked every decode layer
- `gpu_graph_decode_indexer_top_k` uses a `static int cached` but still calls
  `getenv` on first call and if `DS4_DECODE_INDEXER_TOP_K` changes (it won't)

### Fix

**Phase 1 (trivial):** Convert `gpu_graph_decode_indexer_top_k`'s `static cached`
pattern to a one-time init at `gpu_graph_alloc` / first decode. Cache the result
in `g->decode_indexer_top_k`.

**Phase 2 (broader):** Add a `ds4_gpu_env_cache` struct populated once at engine
init (`ds4_gpu_begin_commands` or a new `ds4_gpu_init_env` call). Store all
env-gated booleans as `int` fields. Replace per-token `getenv()` calls with
struct reads.

```c
struct ds4_gpu_env_cache {
    int no_q8_dp4a;
    int no_q8_f16_cache;
    int q8_f16_all;
    int no_direct_down_sum6;
    int no_moe_expert_tiles;
    int moe_tile4;
    int indexer_stage_profile;
    int no_turbo4_direct;
    int decode_indexer_top_k;
    // ...
};
static ds4_gpu_env_cache g_env;
static void ds4_gpu_env_cache_init(void) {
    g_env.no_q8_dp4a = getenv("DS4_CUDA_NO_Q8_DP4A") != NULL;
    g_env.no_direct_down_sum6 = getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") != NULL;
    // ...
}
```

Call `ds4_gpu_env_cache_init()` once from `ds4_gpu_managed_prefetch` or the first
`ds4_gpu_begin_commands`.

### Files to change
- `src/gpu/ds4_cuda.cu` — add `g_env` struct + init function
- `src/gpu/ds4_cuda_moe_dispatch.cuh` — replace `getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6")` etc.
- `src/gpu/ds4_cuda_q8.cuh` / dispatch1.cuh — replace Q8 env checks
- `src/core/ds4_gpu.inc` — replace `getenv("DS4_NO_TURBO4_DIRECT")`,
  `getenv("DS4_GPU_INDEXER_STAGE_PROFILE")`, etc.
- `src/core/ds4_gpu.inc:1093` — `gpu_graph_decode_indexer_top_k` → cache in graph struct

### Verification
- Behavior identical (same env flags → same code paths)
- Test with each env flag set/unset to confirm the cache is populated correctly
- No output change

---

## Opportunity 5 — CUDA Graph capture for the decode tape

**Impact:** 2–5% (~2–5 ms/token). High effort, clean architecture win.

### Problem

The decode path dispatches ~430 kernel launches per token. The split-flush
design (layers 0–3 flushed, then 4–42 back-to-back) already overlaps CPU
dispatch with GPU compute. But the CPU still spends ~3.2 ms setting up tensor
views, computing grid dims, and calling `<<<>>>` for each kernel. This is
partially serialized with GPU execution (the CPU can't dispatch layer 5's
kernels until it finishes dispatching layer 4's).

### vLLM's approach

vLLM captures the entire decode forward pass as a **CUDA Graph** during startup.
At inference time, the graph is replayed with a single `cudaGraphLaunch` call —
zero CPU dispatch overhead. The key trick: **pad all shapes to fixed max sizes**
so the graph's kernel grid dims are compile-time constants. Dynamic data
(expert indices, comp_selected, n_comp) lives in device-memory tensors that the
graph reads at execution time — the graph captures the *pointer* to the tensor,
not the tensor's contents.

### What ds4 would need

The ds4 decode is already a "fixed tape" (43 layers, same kernel sequence every
token). The challenges are:

1. **Dynamic grid sizes:** `n_comp` varies per token (indexer path activates
   only when `n_comp > 512`). `n_selected` varies. `comp_selected` contents
   vary.
2. **Per-layer bookkeeping:** `gpu_graph_encode_decode_layer` swaps
   `cur_hc`/`after_ffn_hc`, increments `layer_n_comp[il]`, updates
   `layer_n_index_comp[il]`.
3. **Conditional branches:** the indexer path (`if (n_comp > decode_top_k)`)
   changes which attention kernel is called.

### Implementation strategy

**Step 1: Static-shape padding.** Always launch the indexed attention kernel
with `n_comp = DS4_N_INDEXER_TOP_K (512)` and `top_k = 512`, padding
`comp_selected` with sentinel indices (e.g., -1 or an out-of-range value that
the kernel's `if (c >= 0 && c < visible_comp)` check rejects). The kernel
already handles this — invalid indices are skipped. This makes the attention
grid shape constant.

**Step 2: Always-turbo4-direct.** When `DS4_KV_TURBO=1`, always use the turbo4
direct kernel (no conditional fallback). The turbo4 kernel handles
`comp_count = 0` (no comp rows) gracefully — it just processes raw rows.

**Step 3: Capture in a side stream.** Use `cudaStreamBeginCapture` /
`cudaStreamEndCapture` on a non-default stream to capture one full decode token
(layers 0–42 + output head). The captured graph references the device-memory
tensor pointers (`g->q->ptr`, `g->heads->ptr`, etc.) which are stable across
tokens. Dynamic values (`pos`, `token`, `selected[]`) are updated via
`cudaGraphExecKernelNodeSetParams` or by writing to a device-memory "args
buffer" that the graph reads.

**Step 4: Per-token update.** Before each `cudaGraphLaunch`, update the dynamic
args (pos, token id, expert selection) in the args buffer. The graph reads
these at execution time. No graph rebuild needed.

### The args buffer approach

Allocate a small device-memory buffer holding per-layer dynamic state:
```c
struct ds4_decode_graph_args {
    uint32_t pos;
    uint32_t token;
    uint32_t raw_row;
    uint32_t n_raw;
    // Per-layer comp state (updated by device-side code, not CPU):
    // layer_n_comp, layer_n_index_comp, raw_rows, comp_rows
};
```

The graph's kernels read `pos`, `token` from this buffer instead of from kernel
parameters. The CPU writes new values before each `cudaGraphLaunch`. Per-layer
state (n_comp increments) is handled by a device-side counter kernel that runs
inside the graph.

### Files to change
- `src/core/ds4_gpu.inc` — new `gpu_graph_capture_decode` / `gpu_graph_replay_decode`
- `src/gpu/ds4_cuda.cu` — `cudaStreamBeginCapture` / `cudaGraphInstantiate` /
  `cudaGraphLaunch` wrappers in `ds4_gpu_begin_commands` / `ds4_gpu_end_commands`
- `src/gpu/ds4_cuda_attention.cuh` — turbo4 kernel: handle `comp_count = 0`
  cleanly (already does, verify)
- `src/gpu/ds4_gpu.h` — new API: `ds4_gpu_decode_graph_capture` /
  `ds4_gpu_decode_graph_launch`

### Verification
- Graph-captured output must be bit-identical to non-graph output
- Test with varying n_comp (short prompt, long prompt) to verify padding works
- Profile: `cudaGraphLaunch` time vs manual dispatch time

### Risk

This is the highest-effort change. The conditional branches (indexer path,
compressor emit every 4 tokens) make the graph non-trivial. A simpler
intermediate step: capture only layers 4–42 (the non-synced portion) as a graph,
keep layers 0–3 on manual dispatch. This captures 90% of the kernels with less
conditional complexity.

---

## Opportunity 6 — Turbo4 attention: shared memory staging

**Impact:** <1% for current workloads. Only matters at extreme context lengths.

### Problem

The turbo4 direct attention kernel (`turbo4_attention.cuh`) reads each turbo4
element directly from global memory via `_t4_elem()`. Each of the 8 warps in the
block reads the same comp row independently → 8× redundant global reads per
element.

### Fix

Cooperatively load each 584-byte turbo4 row into shared memory once, then all 8
warps read from shared memory. Add `cp.async` double-buffering like the FP32
`attention_decode_mixed_heads8_online_kernel`.

```c
__shared__ uint8_t row_shared[584];  // one turbo4 row
// Cooperative load:
for (uint32_t i = threadIdx.x; i < 584; i += blockDim.x)
    row_shared[i] = comp_kv[comp_idx * stride + i];
__syncthreads();
// All warps read from row_shared via _t4_elem(row_shared, ...)
```

### When it matters

Attention is <1% of decode time at current context lengths. This only becomes
relevant if:
- Context grows to 500K+ tokens (comp rows → thousands)
- The raw KV window grows beyond 2304
- The MoE bottleneck is solved and attention becomes the new bottleneck

### Files to change
- `src/gpu/turbo4_attention.cuh` — add shared memory staging + cp.async

### Verification
- Output bit-identical (same data, different load path)
- Benchmark at 1M context to measure attention time fraction

---

## Opportunity 7 — Expert weight streaming buffer (RAM-neutral)

**Impact:** Unknown — needs empirical validation on UMA. Potentially 10–20% if
TLB overhead is the dominant cost.

### Problem

Expert weights (NVFP4, ~13.5 KB/expert/matrix) are accessed from 75 GB managed
memory. Random expert selection across 256 experts × 43 layers causes GPU TLB
misses on the managed address space. On datacenter GPUs (HBM), this isn't a
problem — HBM has its own page tables. On UMA, the GPU shares the CPU's page
table infrastructure.

### vLLM's approach (and why it doesn't apply)

vLLM doesn't face this — all weights sit in HBM at 2–3 TB/s with dedicated GPU
page tables. There's nothing to stream. vLLM's startup cost is `torch.compile`
(kernel fusion) + CUDA graph capture, not weight manipulation.

### The streaming buffer idea

Allocate **one** ~40 KB device-memory buffer (one expert's gate+up+down weights).
Before each expert's MoE computation, `cudaMemcpyAsync` that expert's NVFP4
weights from managed → device buffer. The MoE kernel reads from the device
buffer (GPU page tables only, no 75 GB TLB pressure).

**RAM cost:** 40 KB. Negligible.
**Copy cost:** 7 experts × 43 layers × 40 KB = 12 MB/token. At 273 GB/s
sequential bandwidth: 0.04 ms/token. Negligible.

**The question:** On GB10 UMA, does `cudaMemcpy` from managed→device actually
help? The source pages are already device-resident (ReadMostly +
PreferredLocation=device). The copy might just be a memcpy that warms L2 without
changing the page table story. **This needs empirical testing.**

### How to test

Add a diagnostic mode (`DS4_CUDA_EXPERT_STREAM=1`) that copies each selected
expert's weights to a small device buffer before the MoE kernel, and passes the
buffer pointer instead of the managed pointer. Measure throughput delta.

### Implementation

In `ds4_gpu_routed_moe_one_tensor` (ds4_cuda_moe_dispatch.cuh), after the router
selects experts:
1. Read `selected[]` back to CPU (6 ints — trivial)
2. For each selected expert, `cudaMemcpyAsync` gate+up+down weight ranges to
   the staging buffer
3. Pass the staging buffer pointer to the MoE kernel instead of `gate_base + expert * bytes`

The staging buffer is allocated once at `gpu_graph_alloc` (40 KB × 7 experts =
280 KB, reusable across layers).

### Files to change
- `src/gpu/ds4_cuda_moe_dispatch.cuh` — add staging copy before MoE launch
- `src/gpu/ds4_cuda.cu` — allocate staging buffer in `gpu_graph_alloc`
- `src/gpu/ds4_gpu.h` — new `ds4_gpu_expert_staging_alloc` API

### Verification
- Output bit-identical (same weights, different memory location)
- A/B test: `DS4_CUDA_EXPERT_STREAM=1` vs unset
- If no improvement → UMA memcpy doesn't help TLB → abandon

---

## Opportunity 8 — More kernel fusion (torch.compile-style)

**Impact:** 3–8% (~3–8 ms/token). Medium effort, multiple small wins.

### Problem

`torch.compile` (vLLM's ~10 min startup) fuses kernel chains automatically. ds4
has some fusions but several unfused chains remain:

1. **attn_norm → q_a matmul → q_a_norm → q_b matmul:** 4 kernels, 3
   intermediate buffers (attn_norm, qr, qr_norm). The q_a matmul could fuse with
   the preceding RMS norm (read norm weight + activation, compute norm, write
   directly into q_a's input register).

2. **ffn_norm → router matmul → router_select:** 3 kernels. The router matmul
   (in_dim=4096, out_dim=256) is tiny — could fuse with the preceding norm.

3. **head_rms_norm → rope_tail:** Already fused in
   `head_rms_norm_rope_tail_kernel`. But the Q path does
   `ds4_gpu_head_rms_norm_tensor` (line 1390) then
   `ds4_gpu_rope_tail_tensor` (line 1396) as **separate** calls. The fused
   kernel exists but isn't used here.

4. **HC pre → attn_norm:** Already fused when `fuse_hc_norm` is true
   (`hc_split_weighted_sum_norm_fused_kernel`). But the Q path norm after HC
   is separate.

### Specific fusions to implement

**Fusion A: head_rms_norm + rope_tail for Q path.**
At ds4_gpu.inc:1390–1396, replace:
```c
ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);
ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, ...);
```
with:
```c
ds4_gpu_head_rms_norm_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                         DS4_N_ROT, pos, ...);
```
The fused kernel already exists (`head_rms_norm_rope_tail_kernel` in
ds4_cuda_rope.cuh:3) and is exposed via `ds4_gpu_head_rms_norm_rope_tail_tensor`
(ds4_cuda_dispatch2.cuh:89). This saves 1 kernel launch + 1 intermediate write
of the full Q tensor (q_dim = 32768 floats = 128 KB) per layer.

**Fusion B: q_a matmul + q_a_norm.**
Fuse the Q8_0 matmul (attn_norm → qr) with the subsequent RMS norm (qr → qr_norm)
into one kernel that reads attn_norm, computes the matmul, applies RMS norm
in-place, and writes qr_norm. Saves 1 launch + 1 intermediate buffer (qr, 1024
floats = 4 KB — small, but the launch overhead matters).

This requires a new kernel `matmul_q8_0_rms_norm_fused_kernel` that does:
1. Read weight block + activation block
2. Compute dot product (dp4a)
3. After all blocks: compute RMS norm
4. Write normalized output

**Fusion C: router matmul + router_select.**
The router matmul (4096→256, F16) produces logits. The router_select kernel
does top-k + softmax + bias + weight scaling. These could fuse: the matmul
writes logits to shared memory, then the same block does top-k selection. Saves
1 launch + 1 intermediate buffer (256 floats = 1 KB).

### Files to change
- `src/core/ds4_gpu.inc:1390` — use fused norm+rope for Q path (Fusion A, trivial)
- `src/gpu/ds4_cuda_q8.cuh` — new `matmul_q8_0_rms_norm_fused_kernel` (Fusion B)
- `src/gpu/ds4_cuda_dispatch1.cuh` — dispatch for fused q_a+norm
- `src/gpu/ds4_cuda_router.cuh` — fused router matmul+select (Fusion C)
- `src/core/ds4_gpu.inc` — call fused router

### Verification
- Fusion A: bit-identical (same kernel, just already exists)
- Fusion B/C: compare against unfused path, tolerance < 1e-5 (FP reassociation)

---

## Priority order

| Priority | Opportunity | Impact | Effort | Risk |
|----------|-------------|--------|--------|------|
| 1 | **#2** Software prefetch in MoE | 5–15% | Easy | Zero (hint-only) |
| 2 | **#4** Cache getenv/lookups | 3–5% | Easy | Zero |
| 3 | **#8A** Fused norm+rope for Q | 1–2% | Trivial | Zero (kernel exists) |
| 4 | **#1** MoE down expert parallelism | 10–20% | Medium | Output must match |
| 5 | **#3** F16 matmul half2 loads | 5–10% | Medium | FP rounding |
| 6 | **#7** Expert streaming buffer | ? | Easy-Med | May not help on UMA |
| 7 | **#8B/C** More kernel fusion | 3–8% | Medium | FP rounding |
| 8 | **#5** CUDA Graph capture | 2–5% | Hard | Complex, conditional branches |
| 9 | **#6** Turbo4 shared mem staging | <1% | Easy | Only matters at 500K+ ctx |

**Realistic combined target:** #1+#2+#3+#4+#8A = **~13–15 t/s at medium prompt**
(1.4–1.6× current). The architectural ceiling without MTP is ~27 t/s (weight
bandwidth floor + UMA contention). Closing the remaining gap requires
eliminating all expert TLB overhead (#7 if it works) or MTP (amortizes weight
load across multiple tokens).

**When measuring:** always use the medium prompt (~3500 tokens) as the primary
benchmark. Short-prompt numbers (12+ t/s) are misleading because they don't
exercise the indexer/turbo4 path and have less attention overhead. Long-prompt
numbers (8.8 t/s) are useful for verifying long-context correctness but are
slower due to attention scaling, not MoE inefficiency.

## What won't help (and why)

- **Tensor cores (`mma.sync`) for decode GEMV:** Decode is M=1. `mma.sync`
  needs 16×16 tiles — wastes 15/16 of the M dimension. `__dp4a` is the right
  tool for GEMV. Tensor cores would help prefill (M=2048) but prefill is already
  320 t/s.
- **L2 cache partitioning (`cudaAccessPolicyWindow`):** 24 MB L2 vs 12 MB
  expert traffic/layer — the window is too small. And expert selection changes
  every token, so the persisting window would need constant updates.
- **PagedAttention / continuous batching:** These are multi-request serving
  features. We're single-session.
- **Expert parallelism across GPUs:** We have one GPU.
