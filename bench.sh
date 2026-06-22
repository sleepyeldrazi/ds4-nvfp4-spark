#!/bin/bash
# Reproducible benchmark harness. Usage: ./bench.sh [gen-tokens] [ctx]
# Drops OS caches first for stable numbers. Reports 3 runs + median.
set -e
cd "$(dirname "$0")"
GEN="${1:-32}"
CTX="${2:-256}"
MODEL="../DeepSeek-V4-Flash-REAP-K180-hybrid.gguf"
# K180 (98.6 GiB) requires the managed-memory path to load on the 128 GB Spark;
# DS4_KV_TURBO packs both attention + indexer KV so large ctx fits.
export DS4_CUDA_MANAGED_MODEL=1
export DS4_KV_TURBO=1
sync 2>/dev/null || true
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
for i in 1 2 3; do
  echo "=== run $i (gen=$GEN ctx=$CTX) ===" >&2
  ./ds4-bench -m "$MODEL" --prompt-file /tmp/bench_prompt.txt \
    --ctx-start "$CTX" --ctx-max "$CTX" --gen-tokens "$GEN" 2>/dev/null \
    | grep -E '^[0-9]'
done
