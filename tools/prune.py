#!/usr/bin/env python3
"""Aggregate DS4 REAP observations and physically prune a DS4 GGUF.

This tool works directly on the DS4 GGUF layout. It never dequantizes routed
expert tensors: expert slots are copied as quantized byte ranges, so IQ2_XXS,
Q2_K, and Q4_K routed experts keep their stored quantization type.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable


DS4_N_LAYER = 43
DS4_N_HASH_LAYER = 3
DS4_N_EXPERT = 256
REAP_POLICY_NONE = 0
REAP_POLICY_HASH_PRESERVED = 1
REAP_POLICY_ROUTER_MASK_PRUNED = 2
REAP_POLICY_MOE_DISABLED = 3

GGUF_VALUE_UINT32 = 4
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9

TENSOR_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 40),
    10: ("q2_k", 256, 84),
    11: ("q3_k", 256, 110),
    12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176),
    14: ("q6_k", 256, 210),
    15: ("q8_k", 256, 292),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 110),
    20: ("iq4_nl", 256, 50),
    21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82),
    23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1),
    25: ("i16", 1, 2),
    26: ("i32", 1, 4),
    27: ("i64", 1, 8),
    28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2),
}

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
ROUTER_WEIGHT_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
ROUTER_BIAS_RE = re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def product(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def tensor_nbytes(tensor_type: int, dims: list[int] | tuple[int, ...]) -> int:
    try:
        _, block_elems, block_bytes = TENSOR_TYPES[tensor_type]
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF tensor type {tensor_type}") from exc
    elements = product(dims)
    return ((elements + block_elems - 1) // block_elems) * block_bytes


def tensor_type_name(tensor_type: int) -> str:
    return TENSOR_TYPES.get(tensor_type, (f"type_{tensor_type}", 1, 0))[0]


@dataclass
class KVEntry:
    key: str
    type: int
    start: int
    end: int


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    type: int
    offset: int
    size: int
    new_dims: list[int] = field(default_factory=list)
    new_offset: int = 0
    new_size: int = 0

    def __post_init__(self) -> None:
        self.new_dims = list(self.dims)
        self.new_size = self.size


@dataclass
class GGUFModel:
    path: Path
    version: int
    kvs: list[KVEntry]
    tensors: list[TensorInfo]
    alignment: int
    tensor_data_offset: int
    prefix: mmap.mmap


class Cursor:
    def __init__(self, data: mmap.mmap):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise ValueError("truncated GGUF file")
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u32(self) -> int:
        return struct.unpack_from("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        raw = self.read(n)
        return raw.decode("utf-8")


def skip_value(cur: Cursor, value_type: int, depth: int = 0) -> None:
    if value_type in (0, 1, 7):
        cur.read(1)
    elif value_type in (2, 3):
        cur.read(2)
    elif value_type in (4, 5, 6):
        cur.read(4)
    elif value_type in (10, 11, 12):
        cur.read(8)
    elif value_type == GGUF_VALUE_STRING:
        n = cur.u64()
        cur.read(n)
    elif value_type == GGUF_VALUE_ARRAY:
        if depth > 8:
            raise ValueError("nested metadata array is too deep")
        elem_type = cur.u32()
        n = cur.u64()
        for _ in range(n):
            skip_value(cur, elem_type, depth + 1)
    else:
        raise ValueError(f"unknown GGUF metadata value type {value_type}")


def parse_gguf(path: Path) -> GGUFModel:
    fp = path.open("rb")
    mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    fp.close()
    cur = Cursor(mm)
    if cur.read(4) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    version = cur.u32()
    if version != 3:
        raise ValueError(f"only GGUF v3 is supported, got v{version}")
    n_tensors = cur.u64()
    n_kv = cur.u64()

    kvs: list[KVEntry] = []
    alignment = 32
    for _ in range(n_kv):
        start = cur.pos
        key = cur.string()
        value_type = cur.u32()
        value_pos = cur.pos
        if key == "general.alignment" and value_type == GGUF_VALUE_UINT32:
            alignment = struct.unpack_from("<I", mm, value_pos)[0] or 32
        skip_value(cur, value_type)
        kvs.append(KVEntry(key=key, type=value_type, start=start, end=cur.pos))

    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = cur.string()
        n_dims = cur.u32()
        if n_dims <= 0 or n_dims > 4:
            raise ValueError(f"unsupported tensor rank {n_dims} for {name}")
        dims = [cur.u64() for _ in range(n_dims)]
        tensor_type = cur.u32()
        offset = cur.u64()
        size = tensor_nbytes(tensor_type, dims)
        tensors.append(TensorInfo(name=name, dims=dims, type=tensor_type, offset=offset, size=size))

    tensor_data_offset = align_up(cur.pos, alignment)
    return GGUFModel(
        path=path,
        version=version,
        kvs=kvs,
        tensors=tensors,
        alignment=alignment,
        tensor_data_offset=tensor_data_offset,
        prefix=mm,
    )


def parse_layer_ranges(spec: str) -> set[int]:
    out: set[int] = set()
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"bad layer range {item}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    for layer in out:
        if layer < 0 or layer >= DS4_N_LAYER:
            raise ValueError(f"layer {layer} is outside 0..{DS4_N_LAYER - 1}")
    return out


def load_observation_scores(paths: list[Path]) -> tuple[list[list[float]], list[dict[str, object]]]:
    if not paths:
        raise ValueError("at least one observation JSON is required")
    aggregate: list[list[float]] = [[0.0 for _ in range(DS4_N_EXPERT)] for _ in range(DS4_N_LAYER)]
    summaries: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            doc = json.load(fp)
        if doc.get("format") != "ds4_reap_observation":
            raise ValueError(f"{path} is not a ds4_reap_observation JSON")
        layers = doc.get("layers")
        if not isinstance(layers, dict):
            raise ValueError(f"{path} does not contain a layers object")
        for layer in range(DS4_N_LAYER):
            layer_doc = layers.get(str(layer))
            if not isinstance(layer_doc, dict):
                raise ValueError(f"{path} is missing layer {layer}")
            scores = layer_doc.get("reap")
            if not isinstance(scores, list):
                raise ValueError(f"{path} layer {layer} is missing reap score list")
            if len(scores) > DS4_N_EXPERT:
                raise ValueError(f"{path} layer {layer} has too many experts: {len(scores)}")
            for expert, score in enumerate(scores):
                value = float(score)
                if math.isfinite(value):
                    aggregate[layer][expert] += value
        summaries.append(
            {
                "path": str(path),
                "dataset_path": doc.get("dataset_path"),
                "prompts": doc.get("prompts"),
                "prompt_tokens": doc.get("prompt_tokens"),
                "observed_routes": doc.get("observed_routes"),
                "score_metric": doc.get("score_metric"),
            }
        )
    return aggregate, summaries


def infer_layer_expert_counts(model: GGUFModel) -> list[int]:
    counts = [0 for _ in range(DS4_N_LAYER)]
    for tensor in model.tensors:
        match = EXPERT_RE.match(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        if len(tensor.dims) != 3:
            raise ValueError(f"{tensor.name} is expected to be rank-3, got {tensor.dims}")
        count = int(tensor.dims[2])
        if counts[layer] not in (0, count):
            raise ValueError(f"layer {layer} has inconsistent expert counts")
        counts[layer] = count
    for layer, count in enumerate(counts):
        if count <= 0:
            raise ValueError(f"could not infer routed expert count for layer {layer}")
    return counts


def make_plan(
    model: GGUFModel,
    scores: list[list[float]],
    observation_summaries: list[dict[str, object]],
    compression_ratio: float,
    skip_layers: set[int],
    prune_layers: set[int],
    input_gguf: Path,
    output_gguf: Path,
    source_mixed_quant_manifest: object | None,
) -> tuple[dict[str, object], dict[int, list[int]]]:
    if compression_ratio < 0.0 or compression_ratio >= 1.0:
        raise ValueError("--compression-ratio must be >= 0 and < 1")
    expert_counts = infer_layer_expert_counts(model)
    keep_by_layer: dict[int, list[int]] = {}
    layers_json: dict[str, object] = {}
    for layer in range(DS4_N_LAYER):
        expert_count = expert_counts[layer]
        if layer in skip_layers:
            keep = list(range(expert_count))
            pruned: list[int] = []
            policy = "hash_preserved" if layer < DS4_N_HASH_LAYER else "preserved"
            policy_id = REAP_POLICY_HASH_PRESERVED if layer < DS4_N_HASH_LAYER else REAP_POLICY_NONE
        elif layer in prune_layers:
            n_to_prune = int(expert_count * compression_ratio)
            n_to_prune = min(max(n_to_prune, 0), expert_count - 1)
            ranked = sorted(range(expert_count), key=lambda expert: (scores[layer][expert], expert))
            pruned = ranked[:n_to_prune]
            pruned_set = set(pruned)
            keep = [expert for expert in range(expert_count) if expert not in pruned_set]
            policy = "router_mask_pruned"
            policy_id = REAP_POLICY_ROUTER_MASK_PRUNED
        else:
            keep = list(range(expert_count))
            pruned = []
            policy = "preserved"
            policy_id = REAP_POLICY_NONE
        old_to_new = {str(expert): (-1 if expert in set(pruned) else keep.index(expert)) for expert in range(expert_count)}
        keep_by_layer[layer] = keep
        layers_json[str(layer)] = {
            "layer": layer,
            "activation_policy": policy,
            "policy_id": policy_id,
            "is_hash_routed": layer < DS4_N_HASH_LAYER,
            "expert_count": expert_count,
            "keep_count": len(keep),
            "n_experts_to_prune": len(pruned),
            "compression_ratio": compression_ratio if layer in prune_layers else 0.0,
            "score_metric": "activation_energy_sum2",
            "source_routed_tensor_types": routed_tensor_types_for_layer(model, layer),
            "scores": scores[layer][:expert_count],
            "keep_experts": keep,
            "pruned_experts": pruned,
            "old_to_new": old_to_new,
        }

    plan: dict[str, object] = {
        "format": "ds4_reap_prune_plan",
        "format_version": 1,
        "layout": "ds4-compact-v1",
        "input_gguf": str(input_gguf),
        "output_gguf": str(output_gguf),
        "compression_ratio": compression_ratio,
        "skip_layers": sorted(skip_layers),
        "prune_layers": sorted(prune_layers),
        "observations": observation_summaries,
        "source_mixed_quant_manifest": source_mixed_quant_manifest,
        "notes": [
            "Layers 0-2 are preserved because they are hash-routed.",
            "Layers 3-42 are physically compacted by copying kept quantized expert slots.",
            "Existing routed tensor quantization types are preserved. Whole-tensor Q4_K routed experts remain Q4_K after pruning.",
            "A true per-expert mixed-quant source needs an actual mixed-layout GGUF writer/runtime; this tool can carry its source manifest but does not synthesize Q4_K bytes from Q2 bytes.",
        ],
        "layers": layers_json,
    }
    return plan, keep_by_layer


def routed_tensor_types_for_layer(model: GGUFModel, layer: int) -> dict[str, str]:
    out: dict[str, str] = {}
    prefix = f"blk.{layer}."
    names = {
        "gate": f"{prefix}ffn_gate_exps.weight",
        "up": f"{prefix}ffn_up_exps.weight",
        "down": f"{prefix}ffn_down_exps.weight",
    }
    by_name = {tensor.name: tensor for tensor in model.tensors}
    for part, name in names.items():
        tensor = by_name.get(name)
        out[part] = tensor_type_name(tensor.type) if tensor else "missing"
    return out


def tensor_layer_kind(name: str) -> tuple[str, int] | None:
    match = EXPERT_RE.match(name)
    if match:
        return ("expert", int(match.group(1)))
    match = ROUTER_WEIGHT_RE.match(name)
    if match:
        return ("router_weight", int(match.group(1)))
    match = ROUTER_BIAS_RE.match(name)
    if match:
        return ("router_bias", int(match.group(1)))
    return None


def apply_plan_to_tensors(model: GGUFModel, keep_by_layer: dict[int, list[int]]) -> None:
    for tensor in model.tensors:
        layer_kind = tensor_layer_kind(tensor.name)
        tensor.new_dims = list(tensor.dims)
        if layer_kind is not None:
            kind, layer = layer_kind
            keep_count = len(keep_by_layer[layer])
            if kind == "expert":
                tensor.new_dims[2] = keep_count
            elif kind == "router_weight":
                tensor.new_dims[1] = keep_count
            elif kind == "router_bias":
                tensor.new_dims[0] = keep_count
        tensor.new_size = tensor_nbytes(tensor.type, tensor.new_dims)


def assign_new_offsets(model: GGUFModel) -> int:
    rel = 0
    for tensor in model.tensors:
        rel = align_up(rel, model.alignment)
        tensor.new_offset = rel
        rel += tensor.new_size
    return rel


def gguf_string(raw: str) -> bytes:
    encoded = raw.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def kv_bool(key: str, value: bool) -> bytes:
    return gguf_string(key) + struct.pack("<I?", GGUF_VALUE_BOOL, value)


def kv_string(key: str, value: str) -> bytes:
    return gguf_string(key) + struct.pack("<I", GGUF_VALUE_STRING) + gguf_string(value)


def kv_u32_array(key: str, values: list[int]) -> bytes:
    out = bytearray()
    out += gguf_string(key)
    out += struct.pack("<IIQ", GGUF_VALUE_ARRAY, GGUF_VALUE_UINT32, len(values))
    for value in values:
        out += struct.pack("<I", int(value))
    return bytes(out)


def build_reap_kvs(plan: dict[str, object], plan_path: Path | None) -> list[bytes]:
    layers = plan["layers"]
    assert isinstance(layers, dict)
    policy: list[int] = []
    expert_count: list[int] = []
    keep_count: list[int] = []
    for layer in range(DS4_N_LAYER):
        layer_doc = layers[str(layer)]
        assert isinstance(layer_doc, dict)
        policy.append(int(layer_doc["policy_id"]))
        expert_count.append(int(layer_doc["expert_count"]))
        keep_count.append(int(layer_doc["keep_count"]))
    kvs = [
        kv_bool("reap.enabled", True),
        kv_string("reap.layout", "ds4-compact-v1"),
        kv_u32_array("reap.layer.policy", policy),
        kv_u32_array("reap.layer.expert_count", expert_count),
        kv_u32_array("reap.layer.keep_count", keep_count),
    ]
    if plan_path is not None:
        kvs.insert(2, kv_string("reap.plan.path", str(plan_path)))
    return kvs


def tensor_info_bytes(tensor: TensorInfo) -> bytes:
    out = bytearray()
    out += gguf_string(tensor.name)
    out += struct.pack("<I", len(tensor.new_dims))
    for dim in tensor.new_dims:
        out += struct.pack("<Q", int(dim))
    out += struct.pack("<IQ", tensor.type, tensor.new_offset)
    return bytes(out)


def build_header(model: GGUFModel, reap_kvs: list[bytes]) -> bytes:
    kept_kvs = [kv for kv in model.kvs if not kv.key.startswith("reap.")]
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<IQQ", model.version, len(model.tensors), len(kept_kvs) + len(reap_kvs))
    for kv in kept_kvs:
        out += model.prefix[kv.start:kv.end]
    for kv in reap_kvs:
        out += kv
    for tensor in model.tensors:
        out += tensor_info_bytes(tensor)
    return bytes(out)


def coalesced_runs(indices: list[int]) -> Iterable[tuple[int, int]]:
    if not indices:
        return
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        yield start, prev + 1
        start = prev = value
    yield start, prev + 1


def copy_range(src: BinaryIO, dst: BinaryIO, offset: int, nbytes: int, chunk_size: int = 64 * 1024 * 1024) -> None:
    src.seek(offset)
    remaining = nbytes
    while remaining:
        chunk = src.read(min(chunk_size, remaining))
        if not chunk:
            raise IOError("unexpected EOF while copying tensor data")
        dst.write(chunk)
        remaining -= len(chunk)


def expert_chunk_bytes(tensor: TensorInfo, kind: str) -> int:
    if kind == "expert":
        return tensor_nbytes(tensor.type, tensor.dims[:2])
    if kind == "router_weight":
        return tensor_nbytes(tensor.type, [tensor.dims[0]])
    if kind == "router_bias":
        return tensor_nbytes(tensor.type, [1])
    raise ValueError(f"unknown tensor kind {kind}")


def write_tensor_data(
    src: BinaryIO,
    dst: BinaryIO,
    model: GGUFModel,
    tensor: TensorInfo,
    keep_by_layer: dict[int, list[int]],
) -> None:
    layer_kind = tensor_layer_kind(tensor.name)
    src_base = model.tensor_data_offset + tensor.offset
    if layer_kind is None:
        copy_range(src, dst, src_base, tensor.size)
        return
    kind, layer = layer_kind
    keep = keep_by_layer[layer]
    if len(keep) == tensor.dims[-1] and keep == list(range(tensor.dims[-1])):
        copy_range(src, dst, src_base, tensor.size)
        return
    chunk = expert_chunk_bytes(tensor, kind)
    for start, end in coalesced_runs(keep):
        copy_range(src, dst, src_base + start * chunk, (end - start) * chunk)


def write_pruned_gguf(model: GGUFModel, output: Path, header: bytes, total_tensor_bytes: int, keep_by_layer: dict[int, list[int]]) -> None:
    data_start = align_up(len(header), model.alignment)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    with model.path.open("rb") as src, tmp_output.open("wb") as dst:
        dst.write(header)
        dst.write(b"\0" * (data_start - len(header)))
        written = 0
        for idx, tensor in enumerate(model.tensors, 1):
            rel = dst.tell() - data_start
            if rel > tensor.new_offset:
                raise IOError(f"writer passed planned offset for {tensor.name}")
            dst.write(b"\0" * (tensor.new_offset - rel))
            before = dst.tell()
            write_tensor_data(src, dst, model, tensor, keep_by_layer)
            wrote = dst.tell() - before
            if wrote != tensor.new_size:
                raise IOError(f"{tensor.name}: wrote {wrote} bytes, expected {tensor.new_size}")
            written += wrote
            if idx % 128 == 0 or idx == len(model.tensors):
                print(
                    f"copied {idx}/{len(model.tensors)} tensors, "
                    f"{written / (1024 ** 3):.2f}/{total_tensor_bytes / (1024 ** 3):.2f} GiB",
                    file=sys.stderr,
                )
    tmp_output.replace(output)


def load_manifest_file(path: Path | None) -> object | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def summarize_plan(plan: dict[str, object]) -> str:
    layers = plan["layers"]
    assert isinstance(layers, dict)
    hash_layers = []
    pruned_layers = []
    kept_slots = 0
    original_slots = 0
    for layer in range(DS4_N_LAYER):
        layer_doc = layers[str(layer)]
        assert isinstance(layer_doc, dict)
        policy = layer_doc["activation_policy"]
        if policy == "hash_preserved":
            hash_layers.append(layer)
        if policy == "router_mask_pruned":
            pruned_layers.append(layer)
        kept_slots += int(layer_doc["keep_count"])
        original_slots += int(layer_doc["expert_count"])
    return (
        f"hash_preserved={len(hash_layers)} layers, "
        f"router_pruned={len(pruned_layers)} layers, "
        f"expert_slots={kept_slots}/{original_slots}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gguf", "--model", required=True, type=Path)
    parser.add_argument("--observations", nargs="+", required=True, type=Path)
    parser.add_argument("--output-gguf", "--output", required=True, type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--skip-layers", default="0-2")
    parser.add_argument("--prune-layers", default="3-42")
    parser.add_argument("--source-mixed-quant-manifest", "--upgrade-experts", dest="source_mixed_quant_manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_gguf.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"output exists, pass --overwrite: {args.output_gguf}")
    if args.plan_out is None:
        args.plan_out = args.output_gguf.with_suffix(args.output_gguf.suffix + ".reap_plan.json")

    skip_layers = parse_layer_ranges(args.skip_layers)
    prune_layers = parse_layer_ranges(args.prune_layers)
    overlap = skip_layers & prune_layers
    if overlap:
        raise SystemExit(f"layers cannot be both skipped and pruned: {sorted(overlap)}")
    for layer in range(DS4_N_HASH_LAYER):
        if layer not in skip_layers:
            raise SystemExit("hash-routed layers 0-2 must be included in --skip-layers")

    model = parse_gguf(args.input_gguf)
    scores, observation_summaries = load_observation_scores(args.observations)
    source_mixed_quant_manifest = load_manifest_file(args.source_mixed_quant_manifest)
    plan, keep_by_layer = make_plan(
        model,
        scores,
        observation_summaries,
        args.compression_ratio,
        skip_layers,
        prune_layers,
        args.input_gguf,
        args.output_gguf,
        source_mixed_quant_manifest,
    )
    apply_plan_to_tensors(model, keep_by_layer)
    total_tensor_bytes = assign_new_offsets(model)
    reap_kvs = build_reap_kvs(plan, args.plan_out)
    header = build_header(model, reap_kvs)
    total_file_size = align_up(len(header), model.alignment) + total_tensor_bytes

    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    with args.plan_out.open("w", encoding="utf-8") as fp:
        json.dump(plan, fp, indent=2)
        fp.write("\n")

    print(f"plan: {args.plan_out}")
    print(f"summary: {summarize_plan(plan)}")
    print(f"input_size: {args.input_gguf.stat().st_size / (1024 ** 3):.2f} GiB")
    print(f"expected_output_size: {total_file_size / (1024 ** 3):.2f} GiB")
    if source_mixed_quant_manifest is not None:
        print("source_mixed_quant_manifest: recorded in plan; tensor quantization bytes are preserved by this compact pruner")
    if args.dry_run:
        print("dry_run: did not write GGUF")
        return 0

    args.output_gguf.parent.mkdir(parents=True, exist_ok=True)
    write_pruned_gguf(model, args.output_gguf, header, total_tensor_bytes, keep_by_layer)
    actual = args.output_gguf.stat().st_size
    print(f"wrote: {args.output_gguf}")
    print(f"actual_output_size: {actual / (1024 ** 3):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
