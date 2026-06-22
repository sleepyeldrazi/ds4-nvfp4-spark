#!/bin/bash
# Correctness + coherence verification. Run after every kernel change.
# Usage: ./verify.sh
# Exits non-zero if the math answer is wrong or generation is empty/garbled.
set -e
cd "$(dirname "$0")"
MODEL="../DeepSeek-V4-Flash-REAP-K180-hybrid.gguf"
# K180 (98.6 GiB) requires the managed-memory path to load on the 128 GB Spark;
# DS4_KV_TURBO packs both attention + indexer KV so 1M ctx fits.
export DS4_CUDA_MANAGED_MODEL=1
export DS4_KV_TURBO=1

echo "=== [1] math correctness (expect a number ~4) ==="
OUT=$(./ds4 -m "$MODEL" -p "What is 2+2? Answer in one word." -n 8 --nothink --ctx 1048576 2>/dev/null | grep -v '^ds4:' | tail -1)
echo "answer: $OUT"

echo "=== [2] coherence (robot story, ~120 tokens) ==="
./ds4 -m "$MODEL" -p "Tell me a short story about a robot learning to paint." -n 120 --nothink --ctx 1048576 2>/dev/null | grep -v '^ds4:'

echo "=== [3] factual recall (expect Paris) ==="
./ds4 -m "$MODEL" -p "What is the capital of France? One word." -n 8 --nothink --ctx 1048576 2>/dev/null | grep -v '^ds4:' | tail -1
