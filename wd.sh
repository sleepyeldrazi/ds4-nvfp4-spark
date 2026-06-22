#!/bin/bash
# Background GPU power/util logger. Samples nvidia-smi at high frequency and
# timestamps each line so the decode steady-state is captured (not just the
# prefill burst or the post-decode idle).
#
# Usage: ./wd.sh <output.csv> [interval_ms]
#        then stop with: kill <pid>  (or `pkill -f wd.sh`)
set -u
OUT="${1:-/tmp/wd_log.csv}"
MS="${2:-100}"
# csv header. Query only numeric fields (clocks.mem returns [N/A] on GB10 iGPU).
echo "t_ms,power_w,gpu_util_pct,mem_util_pct,sm_mhz" > "$OUT"
t0=$(date +%s%3N)
while true; do
  line=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  now=$(date +%s%3N)
  echo "$((now - t0)),${line}" >> "$OUT"
  sleep "$(awk -v ms="$MS" 'BEGIN{printf "%.3f", ms/1000}')"
done
