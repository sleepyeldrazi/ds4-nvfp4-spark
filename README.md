# DS4 REAP Runtime

This repository is the DS4 runtime fork used to run REAP-pruned
DeepSeek-V4-Flash GGUF files.

The REAP observation and pruning workflow lives in a separate REAP-side repo,
for example:

```text
/path/to/reap-for-ds4
```

This DS4 repo is responsible for the inference engine side:

- open original DS4 DeepSeek-V4-Flash GGUF files for observation
- expose the routed-expert observation hook used by the REAP repo wrapper
- load compact REAP GGUF files
- run `ds4-compact-v1` GGUF files through the existing DS4 Metal/CUDA graph path

## What Changed For REAP

The REAP compact GGUF produced by the REAP repo is not a normal uniform DS4
GGUF.  It stores fewer expert slots in pruned layers and carries explicit REAP
metadata.

The runtime was changed to understand this metadata:

```text
reap.enabled
reap.layout = ds4-compact-v1
reap.layer.policy
reap.layer.expert_count
reap.layer.keep_count
```

Important engine changes:

- `ds4.c` validates `reap.layout=ds4-compact-v1`.
- `ds4.c` reports the runtime marker:
  `REAP runtime metadata enabled: hash_preserved=... router_masked=...`.
- `layer_stored_expert_count(...)` returns the actual stored expert count for a
  layer instead of assuming every layer has 256 physical expert slots.
- Router, hash-routing, and MoE matmul paths use the layer's stored expert count
  when accessing compact tensors.
- The loader accepts compact expert tensors whose expert dimension is reduced in
  layers pruned by REAP.
- The existing graph execution path is reused.  REAP reduces stored/mapped model
  memory; it does not change top-k routing, which remains 6.

The engine also has `--reap-observe-*` options because routed-expert activation
must be collected inside the DS4 runtime that owns the GGUF graph execution.  In
normal use, call those options through the REAP repo's Python wrapper:

```text
tools/ds4_observe_gguf.py
```

## Build

On macOS:

```bash
cd /path/to/ds4_reap
make
```

## Run A REAP-Pruned GGUF

Example compact model path:

```text
/path/to/ds4_reap/gguf/DeepSeek-V4-Flash-REAP50-DS4-compact-IQ2XXS.gguf
```

Run:

```bash
cd /path/to/ds4_reap

export DS4_REAP50_GGUF="gguf/DeepSeek-V4-Flash-REAP50-DS4-compact-IQ2XXS.gguf"

./ds4 \
  -m "$DS4_REAP50_GGUF" \
  --ctx 512 --nothink --temp 0 -n 64 \
  -p 'are you deepseek?'
```

Expected marker:

```text
REAP runtime metadata enabled: hash_preserved=3 router_masked=40 moe_disabled=0 layout=ds4-compact-v1
```

Do not rely on `./ds4` without `-m` when comparing original vs REAP.  The
default `ds4flash.gguf` symlink may point to the original model.  Pass `-m`
explicitly.

## Inspect

```bash
./ds4 \
  -m "$DS4_REAP50_GGUF" \
  --inspect
```

## Current Local Smoke Result

Original DS4 q2-imatrix GGUF:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

REAP50 compact output:

```text
DeepSeek-V4-Flash-REAP50-DS4-compact-IQ2XXS.gguf
```

Observed local result:

```text
original mapped model: about 82697.67 MiB
REAP compact mapped model: about 48097.66 MiB
original GGUF size: about 80.76 GiB
REAP compact GGUF size: about 46.98 GiB
generation smoke: OK
```

Short-generation token/s can look similar to the original because active top-k
is still 6.  The expected local win is model size and mapped memory.

## REAP Workflow Location

Use the REAP repo for observe and prune:

```bash
cd /path/to/reap-for-ds4

.venv/bin/python tools/ds4_observe_gguf.py --help
.venv/bin/python tools/ds4_prune_gguf.py --help
```

The DS4 repo should stay focused on loading and running the resulting compact
GGUF.

## Acknowledgements

Thanks to [antirez/ds4](https://github.com/antirez/ds4) for the original DS4
DeepSeek-V4-Flash inference engine that this fork is based on.
