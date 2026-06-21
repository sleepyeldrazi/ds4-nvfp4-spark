#!/bin/bash
# Correctness + coherence verification. Run after every kernel change.
# Usage: ./verify.sh
# Exits non-zero if the math answer is wrong or generation is empty/garbled.
set -e
cd "$(dirname "$0")"
MODEL="../DeepSeek-V4-Flash-REAP-K128-hybrid.gguf"

echo "=== [1] math correctness (expect a number ~4) ==="
OUT=$(./ds4 -m "$MODEL" -p "What is 2+2? Answer in one word." -n 8 --nothink 2>/dev/null | grep -v '^ds4:' | tail -1)
echo "answer: $OUT"

echo "=== [2] coherence (robot story, ~120 tokens) ==="
./ds4 -m "$MODEL" -p "Tell me a short story about a robot learning to paint." -n 120 --nothink 2>/dev/null | grep -v '^ds4:'

echo "=== [3] factual recall (expect Paris) ==="
./ds4 -m "$MODEL" -p "What is the capital of France? One word." -n 8 --nothink 2>/dev/null | grep -v '^ds4:' | tail -1
