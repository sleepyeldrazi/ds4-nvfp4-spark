#!/usr/bin/env python3
"""Generate a REAP plan JSON for deepseek4-quantize.c from FP4/FP8 observation data.

Reads calibration_fp4fp8.obs.json (FP4/FP8 forward-hook REAP observation) and
selects top-K experts per routed layer by REAP score. Hash-routed layers (0-2)
preserve all 256 experts.

Output format matches deepseek4-quantize.c:reap_load_plan expectations:
    {"layers": {"3": {"expert_count": 256, "keep_experts": [...]}, ...}}

Usage:
    python3 make_reap_plan.py --obs calibration_fp4fp8.obs.json --k 180 --out reap_k180_plan.json
"""

import argparse
import json
import sys
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", required=True, help="path to calibration_fp4fp8.obs.json")
    parser.add_argument("--k", type=int, default=180, help="experts to keep per routed layer (default 180)")
    parser.add_argument("--out", default="reap_plan.json", help="output plan JSON path")
    args = parser.parse_args()

    with open(args.obs) as f:
        obs = json.load(f)

    n_layers = obs.get("n_layers", 43)
    default_experts = obs.get("default_expert_count", 256)
    k = min(args.k, default_experts)

    plan: Dict = {"layers": {}}

    for il in range(n_layers):
        layer_key = str(il)
        if layer_key not in obs["layers"]:
            continue

        layer_data = obs["layers"][layer_key]
        reap_scores: List[float] = layer_data.get("reap", [])
        if not reap_scores:
            continue

        if il < 3:
            # Hash-routed layers: keep all experts.
            plan["layers"][layer_key] = {
                "expert_count": default_experts,
                "is_hash_routed": True,
                "keep_experts": list(range(default_experts)),
            }
            continue

        # Score-routed: rank by REAP score descending, keep top-K.
        indexed = list(enumerate(reap_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        keep = [eid for eid, _ in indexed[:k]]
        keep.sort()  # sorted for determinism

        plan["layers"][layer_key] = {
            "expert_count": default_experts,
            "keep_experts": keep,
        }

    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"Wrote REAP plan to {args.out}")
    print(f"  Layers: {len(plan['layers'])}")
    for lk, lv in plan["layers"].items():
        n_kept = len(lv.get("keep_experts", []))
        label = "hash" if lv.get("is_hash_routed") else f"top-{n_kept}"
        print(f"  Layer {lk:>2s}: {n_kept}/{lv['expert_count']} experts ({label})")


if __name__ == "__main__":
    main()
