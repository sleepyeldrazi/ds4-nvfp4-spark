#!/usr/bin/env python3
"""Remap an already-materialized mixed per-expert quantization manifest.

Use this when the input GGUF was produced by a separate mixed-quant writer where
some original routed experts are already stored as Q4_K before REAP pruning.
The script maps those original expert ids through a REAP compact pruning plan and
reports which upgraded experts survived and which were pruned.

It does not pick upgrade candidates and it does not create Q4_K bytes. Its input
is a manifest of experts that are already upgraded in the source artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DS4_N_LAYER = 43


def int_list(value: Any, *, field: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [int(item) for item in value]


def source_layer_q4_ids(layer_doc: dict[str, Any]) -> dict[str, list[int]]:
    shared = int_list(layer_doc.get("q4_experts_original_ids"), field="q4_experts_original_ids")
    return {
        "gate": int_list(layer_doc.get("gate_q4_experts_original_ids", shared), field="gate_q4_experts_original_ids"),
        "up": int_list(layer_doc.get("up_q4_experts_original_ids", shared), field="up_q4_experts_original_ids"),
        "down": int_list(layer_doc.get("down_q4_experts_original_ids", shared), field="down_q4_experts_original_ids"),
    }


def compact_map(layer_plan: dict[str, Any]) -> dict[int, int]:
    old_to_new = layer_plan.get("old_to_new")
    if isinstance(old_to_new, dict):
        out: dict[int, int] = {}
        for old, new in old_to_new.items():
            new_i = int(new)
            if new_i >= 0:
                out[int(old)] = new_i
        return out

    keep = layer_plan.get("keep_experts")
    if not isinstance(keep, list):
        raise ValueError("REAP plan layer needs old_to_new or keep_experts")
    return {int(expert): slot for slot, expert in enumerate(keep)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reap-plan", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.reap_plan.open("r", encoding="utf-8") as fp:
        reap_plan = json.load(fp)
    if reap_plan.get("format") != "ds4_reap_prune_plan":
        raise SystemExit(f"{args.reap_plan} is not a ds4_reap_prune_plan JSON")

    with args.source_manifest.open("r", encoding="utf-8") as fp:
        source = json.load(fp)
    if source.get("format") != "ds4_existing_mixed_per_expert_quant_manifest":
        raise SystemExit(f"{args.source_manifest} is not a ds4_existing_mixed_per_expert_quant_manifest JSON")

    source_layers = source.get("layers")
    reap_layers = reap_plan.get("layers")
    if not isinstance(source_layers, dict):
        raise SystemExit("source manifest does not contain a layers object")
    if not isinstance(reap_layers, dict):
        raise SystemExit("REAP plan does not contain a layers object")

    output_layers: dict[str, Any] = {}
    total_survived = 0
    total_pruned = 0
    for layer_key, source_layer in sorted(source_layers.items(), key=lambda item: int(item[0])):
        layer = int(layer_key)
        if layer < 0 or layer >= DS4_N_LAYER:
            raise SystemExit(f"source manifest layer {layer} is outside 0..{DS4_N_LAYER - 1}")
        if not isinstance(source_layer, dict):
            raise SystemExit(f"source manifest layer {layer} must be an object")
        layer_plan = reap_layers.get(str(layer))
        if not isinstance(layer_plan, dict):
            raise SystemExit(f"REAP plan is missing layer {layer}")

        old_to_compact = compact_map(layer_plan)
        part_out: dict[str, Any] = {}
        for part, original_ids in source_layer_q4_ids(source_layer).items():
            survived_original = [expert for expert in original_ids if expert in old_to_compact]
            pruned_original = [expert for expert in original_ids if expert not in old_to_compact]
            survived_slots = [old_to_compact[expert] for expert in survived_original]
            total_survived += len(survived_original)
            total_pruned += len(pruned_original)
            part_out[part] = {
                "q4_experts_original_ids": survived_original,
                "q4_experts_compact_slots": survived_slots,
                "q4_experts_pruned_original_ids": pruned_original,
            }

        output_layers[str(layer)] = {
            "layer": layer,
            "target_quant": source_layer.get("target_quant", source.get("target_quant", "q4_k")),
            "expert_id_space": "original_pre_prune_and_post_prune_compact_slots",
            "parts": part_out,
        }

    doc = {
        "format": "ds4_existing_mixed_per_expert_quant_manifest.remapped",
        "format_version": 1,
        "source_manifest": str(args.source_manifest),
        "reap_plan": str(args.reap_plan),
        "base_source_materialized": True,
        "target_layout_requirement": "ds4-mixed-expert-v1",
        "notes": [
            "This is for inputs where Q4_K experts already exist before REAP pruning.",
            "Surviving original expert ids are mapped to post-prune compact slots.",
            "This manifest alone does not make ds4-compact-v1 mixed per-expert; the GGUF writer/runtime must materialize and consume ds4-mixed-expert-v1.",
        ],
        "total_survived_q4_entries": total_survived,
        "total_pruned_q4_entries": total_pruned,
        "layers": output_layers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(doc, fp, indent=2)
        fp.write("\n")

    print(f"remapped manifest: {args.output}")
    print(f"survived_q4_entries={total_survived} pruned_q4_entries={total_pruned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
