#!/usr/bin/env python3
"""Gate selected real OpenVINO hot/cold full-attention substitutions.

Stock and candidate execute in isolated workers on selected exact 2k, 8k, 16k,
and 32k sentinel prompts.  Each candidate decode consumes the corresponding
stock top-1 token.  The 8k lane performs two decode steps so that it proves
both sides of the hot8192 boundary; the 16k lane proves a material cold-history
prefix, and the 32k lane crosses the bounded dense-history carrier.  This is a
correctness, state-ownership, and codec gate.  The all-ten mode validates
candidate-owned state from phase transitions because upstream custom layers
intentionally make downstream K/V differ from the stock graph.  Wall times are
diagnostic and are not a speedup claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-hot-cold-attention-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
PROMPT_DIR = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts")
PROMPT_2K = PROMPT_DIR / "sentinel_002k.txt"
PROMPT_8K = PROMPT_DIR / "sentinel_008k.txt"
PROMPT_16K = PROMPT_DIR / "sentinel_016k.txt"
PROMPT_32K = PROMPT_DIR / "sentinel_032k.txt"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
CUSTOM_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
CUSTOM_HELPER_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl")
PREFILL_CUSTOM_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl")
LINEAR_CONV_CUSTOM_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_linear_conv_swish.cl")
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
BASE_GATE = ROOT / "tools/intel-qwen36-openvino-full-attention-custom-gate.py"
ABI_EVIDENCE = (
    ROOT / "output/openvino-full-attention-abi-20260714Tseq812-cleanZ")
MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
TARGET_LAYER = 3
KLD_LIMIT = 0.005
FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX = 128
DIRECT_I8_FIXED_COLD_CAPACITY = 32768


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


GRAPH = load_module("iq36_hot_cold_graph", GRAPH_MODULE)
BASE = load_module("iq36_full_attention_base", BASE_GATE)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--prompt-2k", type=Path, default=PROMPT_2K)
  parser.add_argument("--prompt-8k", type=Path, default=PROMPT_8K)
  parser.add_argument("--prompt-16k", type=Path, default=PROMPT_16K)
  parser.add_argument("--prompt-32k", type=Path, default=PROMPT_32K)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
  parser.add_argument(
      "--candidate-gpu-plugin", type=Path,
      help=("candidate-only GPU plugin shared library; stock remains on the "
            "default OpenVINO plugin registry"))
  parser.add_argument("--device", default="GPU")
  parser.add_argument(
      "--lanes", default="2k,8k",
      help="comma-separated subset of 2k,8k,16k,32k (default: 2k,8k)")
  parser.add_argument(
      "--all-ten", action="store_true",
      help="replace all ten full-attention layers instead of layer 3 only")
  parser.add_argument(
      "--target-layers",
      help=("comma-separated full-attention layer subset; mutually exclusive "
            "with --all-ten"))
  parser.add_argument(
      "--capture-attention-outputs", action="store_true",
      help=("diagnostically expose and save each selected attention layer's "
            "last-query output; does not affect gate admission"))
  parser.add_argument(
      "--capture-full-attention-outputs", action="store_true",
      help=("with --capture-attention-outputs, save full selected attention "
            "outputs instead of only the last query"))
  parser.add_argument(
      "--capture-layers",
      help=("comma-separated subset of selected custom layers to expose; "
            "defaults to every selected layer"))
  parser.add_argument(
      "--dump-runtime-graph", action="store_true",
      help="serialize the compiled GPU execution graph for diagnostics")
  parser.add_argument(
      "--capture-full-profile", action="store_true",
      help=("save every OpenVINO profiling row after the final inference; "
            "attention-only rows remain in the ordinary profile field"))
  parser.add_argument(
      "--phase-branch-prefill", action="store_true",
      help=("use the bounded prefill-only operation behind per-layer If "
            "control flow; experimental"))
  parser.add_argument(
      "--stock-prefill-sliced-decode", action="store_true",
      help=("keep stock prefill SDPA, shrink its decode-only history to one "
            "token, and select the custom decode output"))
  parser.add_argument(
      "--direct-i8-fixed-layout", action="store_true",
      help=("candidate-only fixed-state ABI: packed I8 cold K, "
            "dimension-major I8 cold V, and group-major F16 scales"))
  parser.add_argument(
      "--direct-i8-group4-full-cold", action="store_true",
      help=("with the fixed layout, use group-4 cold scales, the full logical "
            "cold prefix, and the dimension-major hot-V decode plane"))
  parser.add_argument(
      "--direct-i8-hybrid-k2-v4", action="store_true",
      help=("with the fixed layout, use group-2 cold-K scales and group-4 "
            "cold-V scales over the full logical cold prefix"))
  parser.add_argument(
      "--fuse-linear-conv-state", action="store_true",
      help=("replace all 30 linear-attention transpose/conv/state/SiLU "
            "boundaries with the fused candidate custom operation"))
  parser.add_argument(
      "--pack-gdn-state", action="store_true",
      help=("store candidate GDN recurrent FP32 state as provider-private "
            "[V,K] rows for coalesced load/store; stock remains unchanged"))
  parser.add_argument(
      "--fc-internal-dynamic-quantize", action="store_true",
      help=("candidate-only graph policy probe: suppress graph-level "
            "DynamicQuantize through model rt-info while preserving the "
            "configured group size for compressed-FC internal quantization"))
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument(
      "--stock-prefill-chunk-tokens", type=int, default=8192,
      help=("maximum stock-worker prefill chunk; only prompts longer than "
            "this value are split (default: 8192)"))
  parser.add_argument(
      "--candidate-prefill-chunk-tokens", type=int, default=8192,
      help=("maximum candidate-worker prefill chunk; must not exceed the "
            "request-owned ring guard (default: 8192)"))
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("memory-stop-gib must be positive")
  if args.stock_prefill_chunk_tokens <= 0:
    parser.error("stock-prefill-chunk-tokens must be positive")
  if args.candidate_prefill_chunk_tokens <= 0:
    parser.error("candidate-prefill-chunk-tokens must be positive")
  if args.candidate_prefill_chunk_tokens > GRAPH.PREFILL_CHUNK_TOKENS:
    parser.error(
        "candidate-prefill-chunk-tokens exceeds the request-owned ring guard "
        f"({GRAPH.PREFILL_CHUNK_TOKENS})")
  if (args.candidate_gpu_plugin is not None and
      not args.candidate_gpu_plugin.is_file()):
    parser.error(
        f"candidate-gpu-plugin is not a file: {args.candidate_gpu_plugin}")
  if args.pack_gdn_state and args.candidate_gpu_plugin is None:
    parser.error("pack-gdn-state requires candidate-gpu-plugin")
  if (args.fc_internal_dynamic_quantize and
      args.candidate_gpu_plugin is None):
    parser.error(
        "fc-internal-dynamic-quantize requires candidate-gpu-plugin")
  if (args.fc_internal_dynamic_quantize and
      (not args.capture_full_profile or not args.dump_runtime_graph)):
    parser.error(
        "fc-internal-dynamic-quantize requires --capture-full-profile and "
        "--dump-runtime-graph for the exact boundary census")
  if args.phase_branch_prefill and args.stock_prefill_sliced_decode:
    parser.error("phase composition modes are mutually exclusive")
  if (args.direct_i8_fixed_layout and
      (args.phase_branch_prefill or args.stock_prefill_sliced_decode)):
    parser.error(
        "direct-i8-fixed-layout requires the unified phase composition")
  if (args.direct_i8_group4_full_cold and
      not args.direct_i8_fixed_layout):
    parser.error(
        "direct-i8-group4-full-cold requires --direct-i8-fixed-layout")
  if args.direct_i8_hybrid_k2_v4 and not args.direct_i8_fixed_layout:
    parser.error(
        "direct-i8-hybrid-k2-v4 requires --direct-i8-fixed-layout")
  if args.direct_i8_group4_full_cold and args.direct_i8_hybrid_k2_v4:
    parser.error("direct-I8 fine-codec modes are mutually exclusive")
  if args.fuse_linear_conv_state and not args.capture_full_profile:
    parser.error(
        "--fuse-linear-conv-state requires --capture-full-profile for the "
        "exact execution census")
  args.lanes = tuple(item.strip() for item in args.lanes.split(",")
                     if item.strip())
  if (not args.lanes or len(set(args.lanes)) != len(args.lanes) or
      any(item not in ("2k", "8k", "16k", "32k") for item in args.lanes)):
    parser.error(
        "lanes must be a unique comma-separated subset of 2k,8k,16k,32k")
  if args.all_ten and args.target_layers:
    parser.error("--all-ten and --target-layers are mutually exclusive")
  if args.target_layers:
    try:
      args.target_layers = tuple(
          int(item.strip()) for item in args.target_layers.split(",")
          if item.strip())
    except ValueError:
      parser.error("target-layers must contain comma-separated integers")
    if (not args.target_layers or
        len(set(args.target_layers)) != len(args.target_layers) or
        any(layer not in GRAPH.FULL_ATTENTION_LAYERS
            for layer in args.target_layers)):
      parser.error(
          f"target-layers must be a unique subset of "
          f"{GRAPH.FULL_ATTENTION_LAYERS}")
  if ((args.direct_i8_group4_full_cold or args.direct_i8_hybrid_k2_v4) and
      (args.all_ten or
       (args.target_layers is not None and len(args.target_layers) != 1))):
    parser.error(
        "direct-I8 fine codecs are admitted for one real layer only")
  if args.capture_layers:
    try:
      args.capture_layers = tuple(
          int(item.strip()) for item in args.capture_layers.split(",")
          if item.strip())
    except ValueError:
      parser.error("capture-layers must contain comma-separated integers")
    if (not args.capture_layers or
        len(set(args.capture_layers)) != len(args.capture_layers) or
        any(layer not in GRAPH.FULL_ATTENTION_LAYERS
            for layer in args.capture_layers)):
      parser.error(
          f"capture-layers must be a unique subset of "
          f"{GRAPH.FULL_ATTENTION_LAYERS}")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-hot-cold-attention-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def safe_name(name: str) -> str:
  return "".join(character if character.isalnum() else "-"
                 for character in name).strip("-")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def mem_available_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("/proc/meminfo does not contain MemAvailable")


def gpu_memory_statistics(core: Any, device: str) -> dict[str, Any]:
  """Capture provider allocation totals without making them a speed claim."""
  try:
    values = core.get_property(device, "GPU_MEMORY_STATISTICS")
    return {str(key): int(value) for key, value in dict(values).items()}
  except Exception as error:  # Property availability varies by GPU runtime.
    return {"error": f"{type(error).__name__}: {error}"}


def handoff_request_states(source: Any, destination: Any) -> dict[str, Any]:
  """Copy every destination state through the public OpenVINO state API."""
  source_states = {str(state.name): state for state in source.query_state()}
  destination_states = {
      str(state.name): state for state in destination.query_state()}
  missing = sorted(set(destination_states) - set(source_states))
  if missing:
    raise RuntimeError(f"prefill request is missing decode states: {missing}")
  rows = []
  started = time.perf_counter_ns()
  for name in sorted(destination_states):
    get_started = time.perf_counter_ns()
    tensor = source_states[name].state
    get_finished = time.perf_counter_ns()
    destination_states[name].state = tensor
    set_finished = time.perf_counter_ns()
    rows.append({
        "name": name,
        "shape": list(tensor.shape),
        "element_type": str(tensor.element_type),
        "bytes": int(tensor.byte_size),
        "get_ms": (get_finished - get_started) / 1_000_000.0,
        "set_ms": (set_finished - get_finished) / 1_000_000.0,
    })
  finished = time.perf_counter_ns()
  return {
      "source_state_count": len(source_states),
      "destination_state_count": len(destination_states),
      "source_only_states": sorted(set(source_states) - set(destination_states)),
      "bytes": sum(row["bytes"] for row in rows),
      "wall_ms": (finished - started) / 1_000_000.0,
      "get_ms": sum(row["get_ms"] for row in rows),
      "set_ms": sum(row["set_ms"] for row in rows),
      "states": rows,
  }


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    layer_type = str(info.get("layerType", ""))
    name = node.get_friendly_name()
    if (layer_type not in (
        "scaled_dot_product_attention", "CustomGPUPrimitive") and
        "iq36_hot_attention" not in name):
      continue
    rows.append({
        "node_name": name,
        "layer_type": layer_type,
        "primitive_type": str(info.get("primitiveType", "")),
        "runtime_precision": str(info.get("runtimePrecision", "")),
        "output_layouts": str(info.get("outputLayouts", "")),
        "output_precisions": str(info.get("outputPrecisions", "")),
    })
  return rows


def profile_rows(
    request: Any, *, attention_only: bool = True,
) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    if (attention_only and row.node_type not in (
        "IndirectSDPA", "IQ36HotAttentionGQA",
        "IQ36DirectI8HotAttentionGQA",
        "IQ36DirectI8Group4HotAttentionGQA") and
        "iq36_hot_attention" not in row.node_name):
      continue
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


def snapshot_states(
    request: Any, phase: str, raw: Path, selected: set[str], capture: set[str],
    np: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
  rows = []
  saved = {}
  arrays = {}
  for state in request.query_state():
    value = np.array(state.state.data, copy=True)
    name = str(state.name)
    row = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": BASE.sha256_array(value, np),
    }
    rows.append(row)
    if name in selected:
      path = raw / f"{phase}-{safe_name(name)}.bin"
      np.ascontiguousarray(value).tofile(path)
      saved[name] = {
          **row,
          "path": relative(path),
          "bytes": path.stat().st_size,
      }
    if name in selected or name in capture:
      arrays[name] = value
  return sorted(rows, key=lambda row: row["name"]), saved, arrays


def logical_cold_length(
    states: dict[str, Any], layer: int, np: Any,
) -> int | None:
  value = states.get(GRAPH.layer_state_names(layer)[2])
  if value is None or value.shape[2] < 1:
    return None
  digits = value[0, 0, 0, :3].astype(np.int64)
  return int(digits[0] + 128 * digits[1] + 16384 * digits[2])


def logical_cold_payloads(
    states: dict[str, Any], layer: int, direct_i8_fixed_layout: bool,
    np: Any, direct_i8_key_quant_group: int = 32,
    direct_i8_value_quant_group: int = 32,
) -> dict[str, Any]:
  """Return the four cold planes in the canonical token-major ABI."""
  names = GRAPH.layer_state_names(layer)
  observed = {name: states[name] for name in names[2:]}
  logical_tokens = logical_cold_length(states, layer, np)
  if logical_tokens is None:
    raise ValueError(f"layer {layer} has no cold-state length sentinel")
  capacity = int(observed[names[2]].shape[2]) - 1
  if not 0 <= logical_tokens <= capacity:
    raise ValueError(
        f"layer {layer} cold length {logical_tokens} exceeds {capacity}")
  if not direct_i8_fixed_layout:
    return {
        name: value[:, :, 1:1 + logical_tokens, :]
        for name, value in observed.items()
    }
  if capacity % GRAPH.KEY_TILE_TOKENS != 0:
    raise ValueError(
        f"layer {layer} direct-I8 capacity {capacity} is not block16 aligned")

  batch = int(observed[names[2]].shape[0])
  heads = int(observed[names[2]].shape[1])
  if (direct_i8_key_quant_group not in (32, 4, 2) or
      direct_i8_value_quant_group not in (32, 4) or
      GRAPH.HEAD_DIM % direct_i8_key_quant_group != 0 or
      GRAPH.HEAD_DIM % direct_i8_value_quant_group != 0):
    raise ValueError(
        "unsupported direct-I8 quant groups "
        f"K{direct_i8_key_quant_group}/V{direct_i8_value_quant_group}")
  key_payload = np.ascontiguousarray(observed[names[2]][:, :, 1:, :])
  key = key_payload.reshape(
      batch, heads, capacity // GRAPH.KEY_TILE_TOKENS,
      GRAPH.HEAD_DIM // 4, GRAPH.KEY_TILE_TOKENS, 4)
  key = np.ascontiguousarray(
      key.transpose(0, 1, 2, 4, 3, 5)
  ).reshape(batch, heads, capacity, GRAPH.HEAD_DIM)

  value_payload = np.ascontiguousarray(observed[names[3]][:, :, 1:, :])
  value = np.ascontiguousarray(
      value_payload.reshape(
          batch, heads, GRAPH.HEAD_DIM, capacity).transpose(0, 1, 3, 2))

  scales = {}
  for name, quant_group in zip(
      names[4:], (direct_i8_key_quant_group, direct_i8_value_quant_group)):
    scale_groups = GRAPH.HEAD_DIM // quant_group
    payload = np.ascontiguousarray(observed[name][:, :, 1:, :])
    group_major = payload.reshape(batch, heads, -1).view(np.float16).reshape(
        batch, heads, scale_groups, capacity)
    token_major = np.ascontiguousarray(
        group_major.transpose(0, 1, 3, 2))
    scales[name] = token_major.view(np.int8).reshape(
        batch, heads, capacity, scale_groups * 2)
  return {
      names[2]: key[:, :, :logical_tokens, :],
      names[3]: value[:, :, :logical_tokens, :],
      names[4]: scales[names[4]][:, :, :logical_tokens, :],
      names[5]: scales[names[5]][:, :, :logical_tokens, :],
  }


def candidate_state_semantics(
    previous: dict[str, Any] | None, current: dict[str, Any],
    start: int, total: int, layer: int, np: Any,
    direct_i8_fixed_layout: bool = False,
    direct_i8_key_quant_group: int = 32,
    direct_i8_value_quant_group: int = 32,
) -> dict[str, Any]:
  names = GRAPH.layer_state_names(layer)
  written_tokens = np.arange(start, total, dtype=np.int64)
  written_tokens = written_tokens[
      (written_tokens < GRAPH.SINK_TOKENS) |
      (written_tokens + GRAPH.HOT_WINDOW >= total)]
  written_slots = np.unique(GRAPH.hot_slots(written_tokens, np))
  all_slots = np.arange(GRAPH.HOT_CAPACITY, dtype=np.int64)
  preserved_slots = np.setdiff1d(
      all_slots, written_slots, assume_unique=True)
  hot_rows = {}
  for name, kind in zip(names[:2], ("key", "value")):
    observed_raw = np.ascontiguousarray(current[name])
    observed = GRAPH.hot_state_rows(observed_raw, kind, np)
    preserved_exact = True
    sink_exact = True
    if previous is not None:
      before_raw = np.ascontiguousarray(previous[name])
      before = GRAPH.hot_state_rows(before_raw, kind, np)
      preserved_exact = bool(np.array_equal(
          before[:, :, preserved_slots, :],
          observed[:, :, preserved_slots, :]))
      sink_exact = bool(np.array_equal(
          before[:, :, :GRAPH.SINK_TOKENS, :],
          observed[:, :, :GRAPH.SINK_TOKENS, :]))
    written = observed[:, :, written_slots, :]
    hot_rows[kind] = {
        "written_global_begin": int(start),
        "written_global_end_exclusive": int(total),
        "written_slot_min": (
            int(written_slots.min()) if written_slots.size else None),
        "written_slot_max": (
            int(written_slots.max()) if written_slots.size else None),
        "written_slots_sha256": hashlib.sha256(
            np.ascontiguousarray(written_slots, dtype="<i8").tobytes()
        ).hexdigest(),
        "written_slot_count": int(len(written_slots)),
        "written_finite": bool(np.isfinite(written).all()),
        "written_nonzero": bool(np.count_nonzero(written) != 0),
        "preserved_slot_count": int(len(preserved_slots)),
        "preserved_slots_exact": preserved_exact,
        "sink_exact_from_previous": sink_exact,
    }

  cold_length = logical_cold_length(current, layer, np)
  expected_cold = max(0, total - GRAPH.HOT_WINDOW)
  observed = {name: current[name] for name in names[2:]}
  logical = logical_cold_payloads(
      current, layer, direct_i8_fixed_layout, np,
      direct_i8_key_quant_group, direct_i8_value_quant_group)
  expected_digits = np.array([
      expected_cold % 128, (expected_cold // 128) % 128,
      (expected_cold // 16384) % 128], dtype=np.int8)
  sentinel_exact = bool(
      np.array_equal(
          observed[names[2]][:, :, 0, :3],
          np.broadcast_to(expected_digits, (1, GRAPH.KV_HEADS, 3))) and
      np.count_nonzero(observed[names[2]][:, :, 0, 3:]) == 0 and
      all(np.count_nonzero(observed[name][:, :, 0, :]) == 0
          for name in names[3:]))
  prefix_exact = True
  previous_cold = 0
  if previous is not None:
    previous_logical = logical_cold_payloads(
        previous, layer, direct_i8_fixed_layout, np,
        direct_i8_key_quant_group, direct_i8_value_quant_group)
    previous_cold = previous_logical[names[2]].shape[2]
    prefix_exact = all(
        np.array_equal(
            previous_logical[name],
            logical[name][:, :, :previous_cold, :])
        for name in names[2:])
  codec = {
      "key_i8": expected_cold == 0,
      "value_i8": expected_cold == 0,
      "key_scale_f16_bytes": expected_cold == 0,
      "value_scale_f16_bytes": expected_cold == 0,
  }
  source_available = expected_cold == 0 or previous is not None
  if expected_cold and previous is not None:
    # Only the newly evicted suffix is guaranteed to remain in the previous
    # hot ring.  Older cold tokens have already fallen out of that carrier;
    # their exact preservation is established independently above.
    cold_tokens = np.arange(previous_cold, expected_cold, dtype=np.int64)
    slots = GRAPH.hot_slots(cold_tokens, np)
    source_key = GRAPH.hot_state_rows(
        previous[names[0]], "key", np)[:, :, slots, :]
    source_value = GRAPH.hot_state_rows(
        previous[names[1]], "value", np)[:, :, slots, :]
    expected_key, expected_key_scale = quantize_group(
        source_key, direct_i8_key_quant_group, np)
    expected_value, expected_value_scale = quantize_group(
        source_value, direct_i8_value_quant_group, np)
    codec = {
        "key_i8": bool(np.array_equal(
            logical[names[2]][:, :, previous_cold:expected_cold, :],
            expected_key)),
        "value_i8": bool(np.array_equal(
            logical[names[3]][:, :, previous_cold:expected_cold, :],
            expected_value)),
        "key_scale_f16_bytes": bool(np.array_equal(
            logical[names[4]][:, :, previous_cold:expected_cold, :],
            expected_key_scale)),
        "value_scale_f16_bytes": bool(np.array_equal(
            logical[names[5]][:, :, previous_cold:expected_cold, :],
            expected_value_scale)),
    }
  cold = {
      "expected_logical_tokens": int(expected_cold),
      "encoded_length": cold_length,
      "physical_rows": int(observed[names[2]].shape[2]),
      "sentinel_exact": sentinel_exact,
      "previous_prefix_exact": prefix_exact,
      "source_available": source_available,
      "codec_exact": codec,
      "all_exact": (
          cold_length == expected_cold and sentinel_exact and prefix_exact and
          source_available and all(codec.values())),
  }
  return {"hot": hot_rows, "cold": cold}


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  cfg = load_json(config_path)
  memory_stop_bytes = int(cfg.get("memory_stop_bytes", 4 * 1024**3))
  if mem_available_bytes() < memory_stop_bytes:
    raise RuntimeError(
        "worker skipped to avoid host OOM: available memory is below "
        f"{memory_stop_bytes} bytes")
  mode = str(cfg["mode"])
  pack_gdn_state = bool(cfg.get("pack_gdn_state", False))
  os.environ.pop("IQ36_GDN_TRANSPOSED_STATE", None)
  if pack_gdn_state and mode == "candidate":
    os.environ["IQ36_GDN_TRANSPOSED_STATE"] = "1"

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  target_layers = tuple(int(layer) for layer in cfg["target_layers"])
  device = str(cfg["device"])
  model_dir = Path(cfg["model_dir"])
  raw = Path(cfg["raw"])
  candidate_gpu_plugin = (
      Path(cfg["candidate_gpu_plugin"])
      if cfg.get("candidate_gpu_plugin") else None)
  if pack_gdn_state and candidate_gpu_plugin is None:
    raise ValueError("packed GDN state requires the candidate GPU plugin")
  candidate_plugin_registry = None
  if candidate_gpu_plugin is not None:
    if not candidate_gpu_plugin.is_file():
      raise FileNotFoundError(candidate_gpu_plugin)
    candidate_plugin_registry = raw / "candidate-plugins.xml"
    candidate_plugin_registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(candidate_gpu_plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(candidate_plugin_registry))
  else:
    core = ov.Core()
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  source_summary = None
  static_phase_separated = bool(cfg.get("static_phase_separated", False))
  if mode == "candidate":
    core.set_property(device, {"CONFIG_FILE": cfg["custom_config"]})
    if static_phase_separated:
      if (bool(cfg.get("phase_branch_prefill", False)) or
          bool(cfg.get("stock_prefill_custom_decode", False)) or
          bool(cfg.get("stock_prefill_sliced_decode", False))):
        raise ValueError(
            "static phase separation is exclusive with dynamic composition")
      if bool(cfg.get("capture_attention_outputs", False)):
        raise ValueError(
            "attention-output capture is not implemented for static phases")
      source, prefill_summary = GRAPH.make_candidate_model(
          core, model_dir, ov, np, target_layers,
          static_phase="prefill",
          fixed_cold_capacity=cfg.get("fixed_cold_capacity"),
          prefill_history_capacity=cfg.get("prefill_history_capacity"),
          fuse_linear_conv_state=bool(
              cfg.get("fuse_linear_conv_state", False)))
      source_summary = {"prefill": prefill_summary, "decode": None}
    else:
      source, source_summary = GRAPH.make_candidate_model(
          core, model_dir, ov, np, target_layers,
          phase_branch_prefill=bool(cfg.get("phase_branch_prefill", False)),
          stock_prefill_custom_decode=bool(
              cfg.get("stock_prefill_custom_decode", False)),
          stock_prefill_sliced_decode=bool(
              cfg.get("stock_prefill_sliced_decode", False)),
          initialize_hot_states=bool(
              cfg.get("initialize_hot_states", False)),
          fixed_cold_capacity=cfg.get("fixed_cold_capacity"),
          prefill_history_capacity=cfg.get("prefill_history_capacity"),
          direct_i8_fixed_layout=bool(
              cfg.get("direct_i8_fixed_layout", False)),
          direct_i8_group4_full_cold=bool(
              cfg.get("direct_i8_group4_full_cold", False)),
          direct_i8_hybrid_k2_v4=bool(
              cfg.get("direct_i8_hybrid_k2_v4", False)),
          relocate_dynamic_split_consumers=bool(
              cfg.get("relocate_dynamic_split_consumers", False)),
          constant_q_gate_split_lengths=bool(
              cfg.get("constant_q_gate_split_lengths", False)),
          fuse_attention_output_gate=bool(
              cfg.get("fuse_attention_output_gate", False)),
          token_major_value_output=bool(
              cfg.get("token_major_value_output", False)),
          attention_gated_dynamic_quantize=bool(
              cfg.get("attention_gated_dynamic_quantize", False)),
          fuse_qk_rope_layout=bool(
              cfg.get("fuse_qk_rope_layout", False)),
          fuse_fixed_fc=bool(cfg.get("fuse_fixed_fc", False)),
          fuse_linear_conv_state=bool(
              cfg.get("fuse_linear_conv_state", False)))
  elif mode == "stock":
    if static_phase_separated:
      raise ValueError("static phase separation is candidate-only")
    source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  else:
    raise ValueError(f"unknown mode: {mode}")
  config_after = str(core.get_property(device, "CONFIG_FILE"))

  # Correctness consumes only the final query position.  Project it in the
  # graph so an 8k diagnostic does not materialize the roughly 7.6 GiB full
  # FP32 logits tensor on the host.
  original_result = source.get_results()[0]
  logits_output = original_result.input_value(0)
  logits_shape = ov.opset13.shape_of(logits_output, "i64")
  sequence_length = ov.opset13.gather(
      logits_shape, ov.opset13.constant(np.array(1, dtype=np.int64)),
      ov.opset13.constant(np.array(0, dtype=np.int64)))
  last_index = ov.opset13.subtract(
      sequence_length, ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_logits = ov.opset13.gather(
      logits_output, last_index,
      ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_logits.set_friendly_name("iq36_gate_last_query_logits")
  source.remove_result(original_result)
  source.add_results([ov.opset13.result(last_logits.output(0))])
  source.validate_nodes_and_infer_types()

  fc_internal_dynamic_quantize = (
      mode == "candidate" and
      bool(cfg.get("fc_internal_dynamic_quantize", False)))
  if fc_internal_dynamic_quantize:
    source.set_rt_info(
        FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX,
        ["runtime_options", "GPU_DYNAMIC_QUANTIZATION_GROUP_SIZE_MAX"])

  captured_attention_outputs = []
  if bool(cfg.get("capture_attention_outputs", False)):
    capture_layers = tuple(
        int(layer) for layer in cfg.get("capture_layers", target_layers))
    if not set(capture_layers).issubset(target_layers):
      raise ValueError(
          f"capture layers {capture_layers} are not a subset of "
          f"target layers {target_layers}")
    for layer in capture_layers:
      if mode == "candidate":
        node = next(
            value for value in source.get_ordered_ops()
            if value.get_friendly_name() ==
               f"iq36_hot_attention_layer{layer}")
        if bool(cfg.get("stock_prefill_custom_decode", False)):
          merge = next(
              value for value in source.get_ordered_ops()
              if value.get_friendly_name() ==
                 f"iq36_hybrid_attention_layer{layer}")
          attention = merge.output(0)
        elif bool(cfg.get("stock_prefill_sliced_decode", False)):
          merge = next(
              value for value in source.get_ordered_ops()
              if value.get_friendly_name() ==
                 f"iq36_sliced_hybrid_attention_layer{layer}")
          attention = merge.output(0)
        else:
          attention = node.output(
              0 if bool(cfg.get("phase_branch_prefill", False)) else 1)
      else:
        node = next(
            value for value in source.get_ordered_ops()
            if value.get_type_name() == "ScaledDotProductAttention" and
               f"layers.{layer}.self_attn" in value.get_friendly_name())
        attention = node.output(0)
      query_axis = ov.opset13.constant(np.array(2, dtype=np.int64))
      scalar_axis = ov.opset13.constant(np.array(0, dtype=np.int64))
      candidate_input_offset = (
          1 if mode == "candidate" and
          bool(cfg.get("phase_branch_prefill", False)) else 0)
      debug_values = {
          "attention": attention,
          "query": node.input_value(candidate_input_offset),
          "key": node.input_value(
              (3 + candidate_input_offset) if mode == "candidate" else 1),
          "value": node.input_value(
              (4 + candidate_input_offset) if mode == "candidate" else 2),
      }
      output_indices = {}
      for role, debug_value in debug_values.items():
        if (role == "attention" and
            bool(cfg.get("capture_full_attention_outputs", False))):
          source.add_outputs(debug_value)
          output_indices[role] = len(source.outputs) - 1
          continue
        debug_shape = ov.opset13.shape_of(debug_value, "i64")
        debug_tokens = ov.opset13.gather(
            debug_shape, query_axis, scalar_axis)
        last_index = ov.opset13.subtract(
            debug_tokens, ov.opset13.constant(np.array(1, dtype=np.int64)))
        last_output = ov.opset13.gather(
            debug_value, last_index, query_axis)
        last_output.set_friendly_name(
            f"iq36_{role}_last_query_layer{layer}_{mode}")
        source.add_outputs(last_output.output(0))
        output_indices[role] = len(source.outputs) - 1
      capture_row = {
          "layer": layer,
          "output_indices": output_indices,
      }
      if (mode == "candidate" and
          not bool(cfg.get("phase_branch_prefill", False))):
        source.add_outputs(node.output(0))
        capture_row["workspace_output_index"] = len(source.outputs) - 1
      captured_attention_outputs.append(capture_row)

  embedding = core.compile_model(
      core.read_model(str(model_dir / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_ids = np.asarray(tokenizer.encode(
      Path(cfg["prompt"]).read_text(encoding="utf-8")
  ).input_ids.data).reshape(-1).astype(np.int64)
  token_path = raw / "prompt-token-ids.u32"
  np.ascontiguousarray(prompt_ids, dtype="<u4").tofile(token_path)

  memory_samples = {
      "memory_stop_bytes": memory_stop_bytes,
      "before_language_compile": mem_available_bytes(),
      "gpu_before_language_compile": gpu_memory_statistics(core, device),
  }
  compile_ms_by_phase = {}
  compiled_by_phase = {}
  request_by_phase = {}
  compile_config = dict(BASE.COMPILE_CONFIG)
  if mode == "candidate":
    # Replacing the last stock K/V-cache pattern makes the GPU plugin's
    # generic LLM detector classify the all-custom graph as a non-LLM.  That
    # would apply the IR's ACTIVATIONS_SCALE_FACTOR=8 runtime option to every
    # compressed FC, unlike the untouched stock graph.  Pin the candidate to
    # the stock LLM policy; this is a candidate-only property and cannot leak
    # into the denominator worker.
    compile_config["ACTIVATIONS_SCALE_FACTOR"] = 0.0
  if static_phase_separated:
    compile_started = time.perf_counter_ns()
    compiled_by_phase["prefill"] = core.compile_model(
        source, device, compile_config)
    compile_ms_by_phase["prefill"] = (
        time.perf_counter_ns() - compile_started) / 1_000_000.0
    del source
    gc.collect()
    memory_samples["after_prefill_compile"] = mem_available_bytes()
    first_compile_delta = max(
        0, memory_samples["before_language_compile"] -
        memory_samples["after_prefill_compile"])
    reserve = int(cfg.get("static_phase_memory_reserve_bytes",
                          6 * 1024 * 1024 * 1024))
    required = first_compile_delta + reserve
    memory_samples["estimated_second_compile_bytes"] = first_compile_delta
    memory_samples["required_before_decode_compile"] = required
    memory_samples["decode_compile_guard_pass"] = (
        memory_samples["after_prefill_compile"] >= required)
    if not memory_samples["decode_compile_guard_pass"]:
      raise RuntimeError(
          "static decode compile skipped to avoid host OOM: "
          f"available={memory_samples['after_prefill_compile']} "
          f"required={required}")
    decode_source, decode_summary = GRAPH.make_candidate_model(
        core, model_dir, ov, np, target_layers,
        static_phase="decode",
        fixed_cold_capacity=cfg.get("fixed_cold_capacity"),
        prefill_history_capacity=cfg.get("prefill_history_capacity"),
        fuse_linear_conv_state=bool(
            cfg.get("fuse_linear_conv_state", False)))
    source_summary["decode"] = decode_summary
    compile_started = time.perf_counter_ns()
    compiled_by_phase["decode"] = core.compile_model(
        decode_source, device, compile_config)
    compile_ms_by_phase["decode"] = (
        time.perf_counter_ns() - compile_started) / 1_000_000.0
    del decode_source
    gc.collect()
    memory_samples["after_decode_compile"] = mem_available_bytes()
    for phase, phase_compiled in compiled_by_phase.items():
      request_by_phase[phase] = phase_compiled.create_infer_request()
      request_by_phase[phase].reset_state()
    hot_bindings = {
        phase: GRAPH.bind_request_owned_hot_states(request, target_layers)
        for phase, request in request_by_phase.items()
    }
    compiled = compiled_by_phase["prefill"]
    request = request_by_phase["prefill"]
  else:
    compile_started = time.perf_counter_ns()
    compiled = core.compile_model(source, device, compile_config)
    compile_ms_by_phase["combined"] = (
        time.perf_counter_ns() - compile_started) / 1_000_000.0
    memory_samples["after_language_compile"] = mem_available_bytes()
    if memory_samples["after_language_compile"] < memory_stop_bytes:
      raise RuntimeError(
          "inference skipped to avoid host OOM after language compile: "
          f"available={memory_samples['after_language_compile']} "
          f"stop={memory_stop_bytes}")
    request = compiled.create_infer_request()
    request.reset_state()
    hot_bindings = (
        GRAPH.bind_request_owned_hot_states(request, target_layers)
        if (mode == "candidate" and
            not bool(cfg.get("skip_hot_state_self_bind", False))) else [])
  memory_samples["gpu_after_language_compile"] = gpu_memory_statistics(
      core, device)
  runtime_graph = None
  if bool(cfg.get("dump_runtime_graph", False)):
    runtime_xml = raw / "runtime-graph.xml"
    runtime_bin = raw / "runtime-graph.bin"
    ov.serialize(compiled.get_runtime_model(), runtime_xml, runtime_bin)
    runtime_graph = {
        "xml": relative(runtime_xml),
        "xml_sha256": sha256_file(runtime_xml),
        "bin": relative(runtime_bin),
        "bin_sha256": sha256_file(runtime_bin),
    }
  compile_ms = sum(compile_ms_by_phase.values())

  candidate_states = set(GRAPH.custom_state_names(target_layers))
  stock_states = {
      name for layer in target_layers for name in GRAPH.stock_state_names(layer)}
  phases = []
  previous_candidate_arrays = None
  state_handoff = None
  collect_states = bool(cfg.get("collect_states", True))
  start = 0
  decode_steps = int(cfg["decode_steps"])
  teacher_tokens = [int(token) for token in cfg.get("decode_tokens", [])]
  trace_marker = (Path(cfg["trace_marker"])
                  if cfg.get("trace_marker") else None)
  if mode == "candidate" and len(teacher_tokens) != decode_steps:
    raise ValueError(
        f"candidate needs {decode_steps} teacher tokens, got "
        f"{len(teacher_tokens)}")
  for index in range(decode_steps + 1):
    if static_phase_separated:
      active_phase = "prefill" if index == 0 else "decode"
      request = request_by_phase[active_phase]
      compiled = compiled_by_phase[active_phase]
    else:
      active_phase = "combined"
    if index == 0:
      tokens = prompt_ids
    elif mode == "stock":
      tokens = [int(phases[-1]["top1"])]
    else:
      tokens = [teacher_tokens[index - 1]]
    total = start + len(tokens)
    marker = (
        f"{cfg['lane']}-{mode}-phase{index}-"
        f"input{len(tokens)}-total{total}")
    started = time.perf_counter_ns()
    prefill_chunk_tokens = int(cfg.get("prefill_chunk_tokens", 0))
    prefill_chunk_state_semantics = []
    if index == 0 and 0 < prefill_chunk_tokens < len(tokens):
      outputs = None
      chunk_rows = []
      previous_chunk_arrays = previous_candidate_arrays
      for chunk_start in range(0, len(tokens), prefill_chunk_tokens):
        chunk_end = min(len(tokens), chunk_start + prefill_chunk_tokens)
        chunk_marker = (
            f"{marker}-chunk{chunk_start}-{chunk_end}")
        if trace_marker is not None:
          trace_marker.write_text(chunk_marker + "\n", encoding="utf-8")
        chunk_started = time.perf_counter_ns()
        outputs = request.infer(
            BASE.make_inputs(
                embedding, tokens[chunk_start:chunk_end], chunk_start,
                chunk_end, np),
            share_outputs=False)
        chunk_infer_ms = (
            time.perf_counter_ns() - chunk_started) / 1_000_000.0
        if mode == "candidate" and collect_states:
          _, _, chunk_arrays = snapshot_states(
              request, f"phase{index}-chunk{chunk_start}-{chunk_end}", raw,
              set(), candidate_states, np)
          chunk_semantics = {
              str(layer): candidate_state_semantics(
                  previous_chunk_arrays, chunk_arrays, chunk_start,
                  chunk_end, layer, np,
                  bool(cfg.get("direct_i8_fixed_layout", False)),
                  2 if cfg.get("direct_i8_hybrid_k2_v4", False) else
                  4 if cfg.get("direct_i8_group4_full_cold", False) else 32,
                  4 if (cfg.get("direct_i8_hybrid_k2_v4", False) or
                        cfg.get("direct_i8_group4_full_cold", False)) else 32)
              for layer in target_layers
          }
          prefill_chunk_state_semantics.append({
              "start": int(chunk_start),
              "end_exclusive": int(chunk_end),
              "layers": chunk_semantics,
          })
          previous_chunk_arrays = chunk_arrays
        chunk_rows.append({
            "start": int(chunk_start),
            "end_exclusive": int(chunk_end),
            "wall_ms_diagnostic": chunk_infer_ms,
        })
      if outputs is None:
        raise RuntimeError("prefill chunk loop produced no outputs")
    else:
      if trace_marker is not None:
        trace_marker.write_text(marker + "\n", encoding="utf-8")
      outputs = request.infer(
          BASE.make_inputs(embedding, tokens, start, total, np),
          share_outputs=False)
      chunk_rows = [{
          "start": int(start),
          "end_exclusive": int(total),
          "wall_ms_diagnostic": (
              time.perf_counter_ns() - started) / 1_000_000.0,
      }]
    wall_ms = sum(row["wall_ms_diagnostic"] for row in chunk_rows)
    if trace_marker is not None:
      trace_marker.unlink(missing_ok=True)
    logits = np.array(
        np.asarray(outputs[compiled.output(0)]).reshape(-1),
        dtype="<f4", copy=True)
    logits_path = raw / f"phase{index}-logits.f32"
    logits.tofile(logits_path)
    attention_outputs = {}
    for row in captured_attention_outputs:
      layer = int(row["layer"])
      tensors = {}
      output_indices = dict(row["output_indices"])
      if "workspace_output_index" in row:
        output_indices["workspace"] = row["workspace_output_index"]
      for role, output_index in output_indices.items():
        value = np.array(
            np.asarray(outputs[compiled.output(int(output_index))]),
            dtype="<f4", copy=True)
        value_path = raw / f"phase{index}-{role}-layer{layer}.f32"
        value.tofile(value_path)
        finite = np.isfinite(value)
        finite_values = value[finite]
        tensors[role] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "path": relative(value_path),
            "sha256": sha256_file(value_path),
            "finite": bool(finite.all()),
            "finite_fraction": float(finite.mean()),
            "finite_min": (
                float(finite_values.min()) if finite_values.size else None),
            "finite_max": (
                float(finite_values.max()) if finite_values.size else None),
            "finite_l2": (
                float(np.linalg.norm(finite_values.astype(np.float64)))
                if finite_values.size else None),
        }
      attention_outputs[str(layer)] = tensors
    if collect_states:
      selected = ((candidate_states if mode == "candidate" else stock_states)
                  if index == decode_steps else set())
      capture = candidate_states if mode == "candidate" else set()
      state_rows, saved_states, arrays = snapshot_states(
          request, f"phase{index}", raw, selected, capture, np)
      cold_lengths = ({
          str(layer): logical_cold_length(arrays, layer, np)
          for layer in target_layers} if mode == "candidate" else {})
      if mode == "candidate" and prefill_chunk_state_semantics:
        self_state = prefill_chunk_state_semantics[-1]["layers"]
      else:
        self_state = ({
            str(layer): candidate_state_semantics(
                previous_candidate_arrays, arrays, start, total, layer, np,
                bool(cfg.get("direct_i8_fixed_layout", False)),
                2 if cfg.get("direct_i8_hybrid_k2_v4", False) else
                4 if cfg.get("direct_i8_group4_full_cold", False) else 32,
                4 if (cfg.get("direct_i8_hybrid_k2_v4", False) or
                      cfg.get("direct_i8_group4_full_cold", False)) else 32)
            for layer in target_layers} if mode == "candidate" else {})
    else:
      state_rows, saved_states, arrays = [], {}, {}
      cold_lengths, self_state = {}, {}
    phases.append({
        "index": index,
        "start": int(start),
        "input_tokens": int(len(tokens)),
        "total_tokens": int(total),
        "trace_marker": marker if trace_marker is not None else None,
        "input_token_ids": [int(value) for value in np.asarray(tokens).reshape(-1)],
        "top1": int(np.argmax(logits)),
        "logits_path": relative(logits_path),
        "logits_sha256": sha256_file(logits_path),
        "logits_finite": bool(np.isfinite(logits).all()),
        "attention_outputs": attention_outputs,
        "prefill_chunks": chunk_rows if index == 0 else [],
        "prefill_chunk_state_semantics": (
            prefill_chunk_state_semantics if index == 0 else []),
        "wall_ms_diagnostic": wall_ms,
        "states": state_rows,
        "saved_states": saved_states,
        "logical_cold_lengths": cold_lengths,
        "self_state_semantics": self_state,
        "compiled_phase": active_phase,
        "full_profile": (
            profile_rows(request, attention_only=False)
            if (bool(cfg.get("capture_full_profile", False)) and
                index in (0, decode_steps)) else None),
    })
    if mode == "candidate" and collect_states:
      previous_candidate_arrays = arrays
    if static_phase_separated and index == 0:
      state_handoff = handoff_request_states(
          request_by_phase["prefill"], request_by_phase["decode"])
      memory_samples["after_state_handoff"] = mem_available_bytes()
    start = total

  memory_samples["gpu_after_final_infer"] = gpu_memory_statistics(
      core, device)

  if static_phase_separated:
    runtime = [
        {**row, "compiled_phase": phase}
        for phase, phase_compiled in compiled_by_phase.items()
        for row in runtime_rows(phase_compiled)
    ]
    profile = [
        {**row, "compiled_phase": phase}
        for phase, phase_request in request_by_phase.items()
        for row in profile_rows(phase_request)
    ]
    full_profile = (
        [
            {**row, "compiled_phase": phase}
            for phase, phase_request in request_by_phase.items()
            for row in profile_rows(phase_request, attention_only=False)
        ] if bool(cfg.get("capture_full_profile", False)) else None)
  else:
    runtime = runtime_rows(compiled)
    profile = profile_rows(request)
    full_profile = (
        profile_rows(request, attention_only=False)
        if bool(cfg.get("capture_full_profile", False)) else None)

  result = {
      "mode": mode,
      "lane": cfg["lane"],
      "target_layers": list(target_layers),
      "capture_attention_outputs": bool(
          cfg.get("capture_attention_outputs", False)),
      "capture_full_attention_outputs": bool(
          cfg.get("capture_full_attention_outputs", False)),
      "logits_projection": "last_query_before_host_output",
      "openvino_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "candidate_gpu_plugin": (
          str(candidate_gpu_plugin.resolve())
          if candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(candidate_gpu_plugin)
          if candidate_gpu_plugin is not None else None),
      "candidate_plugin_registry": (
          relative(candidate_plugin_registry)
          if candidate_plugin_registry is not None else None),
      "gpu_plugin_versions": {
          name: {
              "build_number": version.build_number,
              "description": version.description,
          }
          for name, version in core.get_versions(device).items()
      },
      "config_before": config_before,
      "config_after": config_after,
      "compile_config": compile_config,
      "compile_ms": compile_ms,
      "compile_ms_by_phase": compile_ms_by_phase,
      "memory_samples": memory_samples,
      "compiler_cache": {
          "neo_cache_dir": os.environ.get("NEO_CACHE_DIR"),
          "neo_cache_max_size": os.environ.get("NEO_CACHE_MAX_SIZE"),
          "neo_cache_persistent": os.environ.get("NEO_CACHE_PERSISTENT"),
      },
      "prompt": {
          "path": cfg["prompt"],
          "token_count": int(len(prompt_ids)),
          "token_sha256": sha256_file(token_path),
          "prefill_chunk_tokens": int(cfg.get("prefill_chunk_tokens", 0)),
      },
      "decode_tokens": [int(token) for token in cfg.get("decode_tokens", [])],
      "same_infer_request": not static_phase_separated,
      "static_phase_separated": static_phase_separated,
      "state_handoff": state_handoff,
      "reset_state_called": True,
      "hot_bindings": hot_bindings,
      "hot_state_self_bind_skipped": bool(
          cfg.get("skip_hot_state_self_bind", False)),
      "source_summary": source_summary,
      "direct_i8_fixed_layout": (
          mode == "candidate" and
          bool(cfg.get("direct_i8_fixed_layout", False))),
      "direct_i8_group4_full_cold": (
          mode == "candidate" and
          bool(cfg.get("direct_i8_group4_full_cold", False))),
      "direct_i8_hybrid_k2_v4": (
          mode == "candidate" and
          bool(cfg.get("direct_i8_hybrid_k2_v4", False))),
      "pack_gdn_state": mode == "candidate" and pack_gdn_state,
      "fc_internal_dynamic_quantize": fc_internal_dynamic_quantize,
      "graph_dynamic_quantization_group_size_max": (
          FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX
          if fc_internal_dynamic_quantize else None),
      "runtime": runtime,
      "runtime_graph": runtime_graph,
      "profile": profile,
      "full_profile": full_profile,
      "phases": phases,
  }
  write_json(Path(cfg["result"]), result)
  print(json.dumps({
      "event": "complete", "mode": mode, "lane": cfg["lane"],
      "prompt_tokens": len(prompt_ids),
      "top1": [phase["top1"] for phase in phases],
  }, sort_keys=True))
  return 0


def launch_worker(
    args: argparse.Namespace, lane: str, mode: str, prompt: Path,
    raw: Path, target_layers: tuple[int, ...], decode_steps: int,
    decode_tokens: list[int], capture_layers: tuple[int, ...],
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], list[str]]:
  raw.mkdir(parents=True)
  cache = raw / "neo-cache"
  cache.mkdir()
  config_path = raw / "worker-config.json"
  result_path = raw / "worker-result.json"
  write_json(config_path, {
      "mode": mode,
      "lane": lane,
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(prompt.resolve()),
      "custom_config": str(args.custom_config.resolve()),
      "target_layers": list(target_layers),
      "capture_layers": list(capture_layers),
      "dump_runtime_graph": args.dump_runtime_graph,
      "capture_full_profile": args.capture_full_profile,
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": mode == "candidate" and args.pack_gdn_state,
      "fc_internal_dynamic_quantize": (
          mode == "candidate" and args.fc_internal_dynamic_quantize),
      "decode_steps": decode_steps,
      "decode_tokens": decode_tokens,
      "capture_attention_outputs": args.capture_attention_outputs,
      "capture_full_attention_outputs": args.capture_full_attention_outputs,
      "phase_branch_prefill": args.phase_branch_prefill,
      "stock_prefill_sliced_decode": args.stock_prefill_sliced_decode,
      "direct_i8_fixed_layout": (
          mode == "candidate" and args.direct_i8_fixed_layout),
      "direct_i8_group4_full_cold": (
          mode == "candidate" and args.direct_i8_group4_full_cold),
      "direct_i8_hybrid_k2_v4": (
          mode == "candidate" and args.direct_i8_hybrid_k2_v4),
      "fixed_cold_capacity": (
          DIRECT_I8_FIXED_COLD_CAPACITY
          if mode == "candidate" and args.direct_i8_fixed_layout else None),
      "prefill_history_capacity": (
          GRAPH.RING_CAPACITY
          if mode == "candidate" and args.direct_i8_fixed_layout else None),
      "candidate_gpu_plugin": (
          str(args.candidate_gpu_plugin.resolve())
          if mode == "candidate" and args.candidate_gpu_plugin is not None
          else None),
      "prefill_chunk_tokens": (
          args.stock_prefill_chunk_tokens if mode == "stock" else
          args.candidate_prefill_chunk_tokens),
      "memory_stop_bytes": int(args.memory_stop_gib * 1024**3),
      "raw": str(raw.resolve()),
      "result": str(result_path.resolve()),
  })
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(config_path),
  ]
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("IQ36_GDN_TRANSPOSED_STATE", None)
  environment.update({
      "NEO_CACHE_DIR": str(cache.resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  run = subprocess.run(
      command, cwd=ROOT, env=environment, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace", timeout=args.timeout_s)
  (raw / "worker.stdout").write_text(run.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(run.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": command,
      "environment": {key: environment[key] for key in (
          "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": run.returncode,
  })
  result = load_json(result_path) if result_path.is_file() else {}
  return run, result, command


def load_logits(result: dict[str, Any], index: int, np: Any) -> Any:
  return np.fromfile(
      ROOT / result["phases"][index]["logits_path"], dtype="<f4")


def load_state(
    result: dict[str, Any], index: int, name: str, np: Any,
) -> Any:
  row = result["phases"][index]["saved_states"][name]
  return np.fromfile(
      ROOT / row["path"], dtype=np.dtype(row["dtype"])).reshape(row["shape"])


def shared_state_schema(
    stock_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]],
    target_layers: tuple[int, ...],
) -> dict[str, Any]:
  removed_stock = {
      name for layer in target_layers for name in GRAPH.stock_state_names(layer)}
  custom_states = set(GRAPH.custom_state_names(target_layers))
  stock = {row["name"]: row for row in stock_rows
           if row["name"] not in removed_stock}
  candidate = {row["name"]: row for row in candidate_rows
               if row["name"] not in custom_states}
  names = sorted(set(stock) | set(candidate))
  mismatches = [
      name for name in names
      if name not in stock or name not in candidate or
      stock[name]["shape"] != candidate[name]["shape"] or
      stock[name]["dtype"] != candidate[name]["dtype"] or
      not stock[name]["finite"] or not candidate[name]["finite"]]
  return {
      "stock_shared_count": len(stock),
      "candidate_shared_count": len(candidate),
      "names_match": set(stock) == set(candidate),
      "shape_dtype_finite_match": not mismatches,
      "mismatch_names": mismatches,
  }


def rounded_hot_reference(value: Any, np: Any) -> Any:
  return np.asarray(value, dtype=np.float16).astype(np.float32)


def hot_comparison(
    stock: dict[str, Any], candidate: dict[str, Any], index: int,
    total_tokens: int, target_layers: tuple[int, ...], np: Any,
    direct_i8_group4_full_cold: bool = False,
) -> dict[str, Any]:
  begin = max(0, total_tokens - GRAPH.HOT_WINDOW)
  recent_tokens = np.arange(begin, total_tokens, dtype=np.int64)
  sink_tokens = np.arange(
      min(GRAPH.SINK_TOKENS, total_tokens), dtype=np.int64)
  global_tokens = np.unique(np.concatenate([sink_tokens, recent_tokens]))
  slots = GRAPH.hot_slots(global_tokens, np)
  rows = {}
  for layer in target_layers:
    stock_key, stock_value_name = GRAPH.stock_state_names(layer)
    names = GRAPH.layer_state_names(layer)
    layer_rows = {}
    for stock_name, hot_name, kind in (
        (stock_key, names[0], "key"),
        (stock_value_name, names[1], "value"),
    ):
      stock_value = load_state(stock, index, stock_name, np)
      observed = GRAPH.hot_state_rows(
          load_state(candidate, index, hot_name, np), kind, np
      )[:, :, slots, :]
      expected = rounded_hot_reference(stock_value[:, :, global_tokens, :], np)
      layer_rows[kind] = {
          "global_begin": int(begin),
          "global_end_exclusive": int(total_tokens),
          "logical_tokens": int(len(global_tokens)),
          "exact_sink_tokens": int(len(sink_tokens)),
          "exact_bits": bool(np.array_equal(observed, expected)),
          "numeric": BASE.vector_metrics(expected, observed, np),
      }
    if direct_i8_group4_full_cold:
      stock_value = load_state(stock, index, stock_value_name, np)
      observed = GRAPH.unpack_dimension_major_hot_value(
          load_state(candidate, index, names[0], np), np
      )[:, :, slots, :]
      expected = rounded_hot_reference(
          stock_value[:, :, global_tokens, :], np)
      layer_rows["dimension_major_value"] = {
          "global_begin": int(begin),
          "global_end_exclusive": int(total_tokens),
          "logical_tokens": int(len(global_tokens)),
          "exact_sink_tokens": int(len(sink_tokens)),
          "exact_bits": bool(np.array_equal(observed, expected)),
          "numeric": BASE.vector_metrics(expected, observed, np),
      }
    rows[str(layer)] = layer_rows
  return rows


def quantize_group(
    value: Any, quant_group: int, np: Any,
) -> tuple[Any, Any]:
  rounded = rounded_hot_reference(value, np)
  shape = rounded.shape
  if quant_group not in (32, 4, 2) or GRAPH.HEAD_DIM % quant_group != 0:
    raise ValueError(f"unsupported quantization group {quant_group}")
  scale_groups = GRAPH.HEAD_DIM // quant_group
  blocks = rounded.reshape(
      shape[0], shape[1], shape[2], scale_groups, quant_group)
  maximum = np.max(np.abs(blocks), axis=-1)
  scale = np.where(maximum == 0, 1.0, maximum / 127.0).astype(np.float32)
  quantized = np.clip(
      np.rint(blocks / scale[..., None]), -127, 127).astype(np.int8)
  quantized = quantized.reshape(shape)
  scale_bytes = np.ascontiguousarray(scale.astype(np.float16)).view(
      np.int8).reshape(shape[0], shape[1], shape[2], scale_groups * 2)
  return quantized, scale_bytes


def cold_comparison(
    stock: dict[str, Any], candidate: dict[str, Any], index: int,
    target_layers: tuple[int, ...], np: Any,
    direct_i8_fixed_layout: bool = False,
    direct_i8_key_quant_group: int = 32,
    direct_i8_value_quant_group: int = 32,
) -> dict[str, Any]:
  rows = {}
  for layer in target_layers:
    stock_key_name, stock_value_name = GRAPH.stock_state_names(layer)
    names = GRAPH.layer_state_names(layer)
    stock_key = load_state(stock, index, stock_key_name, np)
    stock_value = load_state(stock, index, stock_value_name, np)
    cold_tokens = max(0, stock_key.shape[2] - GRAPH.HOT_WINDOW)
    expected_key, expected_key_scale = quantize_group(
        stock_key[:, :, :cold_tokens, :], direct_i8_key_quant_group, np)
    expected_value, expected_value_scale = quantize_group(
        stock_value[:, :, :cold_tokens, :], direct_i8_value_quant_group, np)
    observed = {name: load_state(candidate, index, name, np)
                for name in names[2:]}
    logical = logical_cold_payloads(
        observed, layer, direct_i8_fixed_layout, np,
        direct_i8_key_quant_group, direct_i8_value_quant_group)
    digits = observed[names[2]][0, 0, 0, :3].astype(np.int64)
    encoded_length = int(digits[0] + 128 * digits[1] + 16384 * digits[2])
    expected_digits = np.array([
        cold_tokens % 128, (cold_tokens // 128) % 128,
        (cold_tokens // 16384) % 128], dtype=np.int8)
    sentinel_ok = bool(
        np.array_equal(
            observed[names[2]][:, :, 0, :3],
            np.broadcast_to(expected_digits, (1, GRAPH.KV_HEADS, 3))) and
        np.count_nonzero(observed[names[2]][:, :, 0, 3:]) == 0 and
        all(np.count_nonzero(observed[name][:, :, 0, :]) == 0
            for name in names[3:]))
    exact = {
        "key_i8": bool(np.array_equal(logical[names[2]], expected_key)),
        "value_i8": bool(np.array_equal(
            logical[names[3]], expected_value)),
        "key_scale_f16_bytes": bool(np.array_equal(
            logical[names[4]], expected_key_scale)),
        "value_scale_f16_bytes": bool(np.array_equal(
            logical[names[5]], expected_value_scale)),
    }
    rows[str(layer)] = {
        "logical_tokens": int(cold_tokens),
        "encoded_length": encoded_length,
        "physical_rows": int(observed[names[2]].shape[2]),
        "sentinel_exact": sentinel_ok,
        "exact": exact,
        "all_exact": sentinel_ok and encoded_length == cold_tokens and
                     all(exact.values()),
    }
  return rows


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  import numpy as np

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = BASE.git_state(out_dir)
  target_layers = (
      GRAPH.FULL_ATTENTION_LAYERS if args.all_ten else
      args.target_layers if args.target_layers else (TARGET_LAYER,))
  capture_layers = args.capture_layers or target_layers
  if not set(capture_layers).issubset(target_layers):
    raise SystemExit(
        f"capture layers {capture_layers} are not a subset of selected "
        f"layers {target_layers}")
  target_count = len(target_layers)
  expected_stock_sdpa = 10 - target_count
  expected_state_count = 80 + 4 * target_count
  expected_shared_states = 80 - 2 * target_count
  direct_i8_fine_full_cold = (
      args.direct_i8_group4_full_cold or args.direct_i8_hybrid_k2_v4)
  expected_custom_type = (
      "IQ36DirectI8HybridK2V4HotAttentionGQA"
      if args.direct_i8_hybrid_k2_v4 else
      "IQ36DirectI8Group4HotAttentionGQA"
      if args.direct_i8_group4_full_cold else
      "IQ36DirectI8HotAttentionGQA"
      if args.direct_i8_fixed_layout else "IQ36HotAttentionGQA")
  direct_i8_key_quant_group = (
      2 if args.direct_i8_hybrid_k2_v4 else
      4 if args.direct_i8_group4_full_cold else 32)
  direct_i8_value_quant_group = 4 if direct_i8_fine_full_cold else 32
  expected_fixed_cold_capacity = (
      DIRECT_I8_FIXED_COLD_CAPACITY
      if args.direct_i8_fixed_layout else None)
  custom_states = set(GRAPH.custom_state_names(target_layers))
  removed_stock_states = {
      name for layer in target_layers for name in GRAPH.stock_state_names(layer)}
  available_lanes = {
      "2k": {"prompt": args.prompt_2k, "tokens": 2048, "steps": 1},
      "8k": {"prompt": args.prompt_8k, "tokens": 8192, "steps": 2},
      "16k": {"prompt": args.prompt_16k, "tokens": 16384, "steps": 1},
      "32k": {"prompt": args.prompt_32k, "tokens": 32768, "steps": 1},
  }
  lanes = {lane: available_lanes[lane] for lane in args.lanes}
  results = {}
  worker_rows = []
  commands = []
  memory_stop_bytes = int(args.memory_stop_gib * 1024**3)
  for lane, lane_cfg in lanes.items():
    if mem_available_bytes() < memory_stop_bytes:
      raise RuntimeError(
          f"{lane} stock worker skipped to avoid host OOM: available memory "
          f"is below {memory_stop_bytes} bytes")
    stock_run, stock, stock_command = launch_worker(
        args, lane, "stock", lane_cfg["prompt"], raw / lane / "stock",
        target_layers, lane_cfg["steps"], [], capture_layers)
    teacher_tokens = [
        int(phase["top1"])
        for phase in stock.get("phases", [])[:lane_cfg["steps"]]]
    if mem_available_bytes() < memory_stop_bytes:
      raise RuntimeError(
          f"{lane} candidate worker skipped to avoid host OOM: available "
          f"memory is below {memory_stop_bytes} bytes")
    candidate_run, candidate, candidate_command = launch_worker(
        args, lane, "candidate", lane_cfg["prompt"],
        raw / lane / "candidate", target_layers, lane_cfg["steps"],
        teacher_tokens, capture_layers)
    results[lane] = {"stock": stock, "candidate": candidate}
    worker_rows.extend([
        {"lane": lane, "mode": "stock", "returncode": stock_run.returncode,
         "stderr": stock_run.stderr},
        {"lane": lane, "mode": "candidate",
         "returncode": candidate_run.returncode,
         "stderr": candidate_run.stderr},
    ])
    commands.extend([stock_command, candidate_command])

  comparisons = {}
  if all(results[lane][mode] for lane in lanes for mode in ("stock", "candidate")):
    for lane, lane_cfg in lanes.items():
      stock = results[lane]["stock"]
      candidate = results[lane]["candidate"]
      phase_count = lane_cfg["steps"] + 1
      distributions = [
          BASE.distribution_metrics(
              load_logits(stock, index, np),
              load_logits(candidate, index, np), np)
          for index in range(phase_count)]
      schemas = [
          shared_state_schema(
              stock["phases"][index]["states"],
              candidate["phases"][index]["states"], target_layers)
          for index in range(phase_count)]
      total = int(stock["phases"][-1]["total_tokens"])
      comparisons[lane] = {
          "distribution": distributions,
          "stock_top1": [phase["top1"] for phase in stock["phases"]],
          "candidate_top1": [
              phase["top1"] for phase in candidate["phases"]],
          "shared_state_schema": schemas,
          # Once upstream full-attention layers are custom, downstream K/V
          # states are expected to differ from the stock graph.  The direct
          # stock boundary comparison therefore remains an isolated-layer
          # assertion only.  All-layer state correctness is established from
          # the candidate's own phase-to-phase transitions below.
          "isolated_stock_hot": (
              hot_comparison(
                  stock, candidate, phase_count - 1, total,
                  target_layers, np, direct_i8_fine_full_cold)
              if target_count == 1 else {}),
          "isolated_stock_cold": (
              cold_comparison(
                  stock, candidate, phase_count - 1, target_layers, np,
                  args.direct_i8_fixed_layout, direct_i8_key_quant_group,
                  direct_i8_value_quant_group)
              if target_count == 1 else {}),
          "self_state_semantics": [
              phase.get("self_state_semantics", {})
              for phase in candidate["phases"]],
          "cold_length_progression": {
              str(layer): [phase["logical_cold_lengths"][str(layer)]
                           for phase in candidate["phases"]]
              for layer in target_layers},
      }

  candidate_results = [results[lane]["candidate"] for lane in lanes]
  compiler_cache_rows = [{
      "lane": lane,
      "mode": mode,
      **results[lane][mode].get("compiler_cache", {}),
  } for lane in lanes for mode in ("stock", "candidate")]
  source_rows = [result.get("source_summary", {})
                 for result in candidate_results]
  runtime_rows_all = [result.get("runtime", [])
                      for result in candidate_results]
  profile_rows_all = [result.get("profile", [])
                      for result in candidate_results]
  full_profile_rows_all = [result.get("full_profile", [])
                           for result in candidate_results]
  distribution_rows = [
      row for lane in comparisons.values()
      for row in lane.get("distribution", [])]
  schema_rows = [
      row for lane in comparisons.values()
      for row in lane.get("shared_state_schema", [])]
  isolated_hot_rows = [
      row for lane in comparisons.values()
      for layer in lane.get("isolated_stock_hot", {}).values()
      for row in layer.values()]
  isolated_cold_rows = [
      row for lane in comparisons.values()
      for row in lane.get("isolated_stock_cold", {}).values()]
  self_hot_rows = [
      row for lane in comparisons.values()
      for phase in lane.get("self_state_semantics", [])
      for layer in phase.values()
      for row in layer.get("hot", {}).values()]
  self_cold_rows = [
      layer.get("cold", {}) for lane in comparisons.values()
      for phase in lane.get("self_state_semantics", [])
      for layer in phase.values()]
  prefill_chunk_transition_rows = [
      transition
      for result in candidate_results
      for transition in (
          result.get("phases", [{}])[0].get(
              "prefill_chunk_state_semantics", [])
          if result.get("phases") else [])]
  abi_metrics = ABI_EVIDENCE / "metrics.json"

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq812_real_layer3_abi_remains_bound",
            abi_metrics.is_file() and
            load_json(abi_metrics).get("required_checks_passed") is True,
            path=relative(abi_metrics),
            sha256=sha256_file(abi_metrics) if abi_metrics.is_file() else None),
      check("implementation_sources_are_bound",
            all(path.is_file() for path in (
                args.custom_config, args.custom_source, CUSTOM_HELPER_SOURCE,
                PREFILL_CUSTOM_SOURCE, LINEAR_CONV_CUSTOM_SOURCE,
                GRAPH_MODULE, MODEL_CONTRACT)),
            files=[{
                "path": relative(path),
                "sha256": sha256_file(path) if path.is_file() else None,
            } for path in (
                args.custom_config, args.custom_source, CUSTOM_HELPER_SOURCE,
                PREFILL_CUSTOM_SOURCE, LINEAR_CONV_CUSTOM_SOURCE,
                GRAPH_MODULE, MODEL_CONTRACT)]),
      check("all_isolated_workers_complete",
            len(worker_rows) == 2 * len(lanes) and
            all(row["returncode"] == 0 for row in worker_rows),
            workers=worker_rows),
      check("direct_i8_fixed_layout_isolated_to_candidate",
            all(
                results[lane]["stock"].get(
                    "direct_i8_fixed_layout") is False and
                results[lane]["candidate"].get(
                    "direct_i8_fixed_layout") is args.direct_i8_fixed_layout
                for lane in lanes),
            enabled=args.direct_i8_fixed_layout,
            fixed_cold_capacity=expected_fixed_cold_capacity,
            physical_ring_capacity=(
                GRAPH.RING_CAPACITY if args.direct_i8_fixed_layout else None)),
      check("direct_i8_group4_full_cold_isolated_to_candidate",
            all(
                results[lane]["stock"].get(
                    "direct_i8_group4_full_cold") is False and
                results[lane]["candidate"].get(
                    "direct_i8_group4_full_cold") is
                    args.direct_i8_group4_full_cold
                for lane in lanes),
            enabled=args.direct_i8_group4_full_cold,
            key_quant_group=direct_i8_key_quant_group,
            value_quant_group=direct_i8_value_quant_group),
      check("direct_i8_hybrid_k2_v4_isolated_to_candidate",
            all(
                results[lane]["stock"].get(
                    "direct_i8_hybrid_k2_v4") is False and
                results[lane]["candidate"].get(
                    "direct_i8_hybrid_k2_v4") is
                    args.direct_i8_hybrid_k2_v4
                for lane in lanes),
            enabled=args.direct_i8_hybrid_k2_v4,
            key_quant_group=direct_i8_key_quant_group,
            value_quant_group=direct_i8_value_quant_group),
      check("packed_gdn_state_isolated_to_candidate_plugin",
            all(results[lane]["stock"].get("pack_gdn_state") is False and
                results[lane]["candidate"].get("pack_gdn_state") is
                    args.pack_gdn_state
                for lane in lanes),
            enabled=args.pack_gdn_state),
      check("fc_internal_dynamic_quantize_isolated_to_candidate_plugin",
            all(
                results[lane]["stock"].get(
                    "fc_internal_dynamic_quantize") is False and
                results[lane]["stock"].get(
                    "graph_dynamic_quantization_group_size_max") is None and
                results[lane]["candidate"].get(
                    "fc_internal_dynamic_quantize") is
                    args.fc_internal_dynamic_quantize and
                results[lane]["candidate"].get(
                    "graph_dynamic_quantization_group_size_max") == (
                        FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX
                        if args.fc_internal_dynamic_quantize else None)
                for lane in lanes),
            enabled=args.fc_internal_dynamic_quantize,
            graph_group_size_max=FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX),
      check("fc_internal_dynamic_quantize_changes_exact_boundary_census",
            not args.fc_internal_dynamic_quantize or
            (args.capture_full_profile and
             all(
                 sum(row.get("node_type") == "DynamicQuantize"
                     for row in results[lane]["stock"].get(
                         "full_profile", [])) > 0 and
                 sum(row.get("node_type") == "DynamicQuantize"
                     for row in results[lane]["candidate"].get(
                         "full_profile", [])) == 0 and
                 sum(row.get("node_type") == "FullyConnectedCompressed" and
                         row.get("status") == "Status.EXECUTED"
                     for row in results[lane]["candidate"].get(
                         "full_profile", [])) > 0
                 for lane in lanes)),
            rows={lane: {
                mode: {
                    "dynamic_quantize": sum(
                        row.get("node_type") == "DynamicQuantize"
                        for row in (results[lane][mode].get(
                            "full_profile") or [])),
                    "fully_connected_compressed": sum(
                        row.get("node_type") ==
                            "FullyConnectedCompressed" and
                        row.get("status") == "Status.EXECUTED"
                        for row in (results[lane][mode].get(
                            "full_profile") or [])),
                    "moe_fused_compressed": sum(
                        row.get("node_type") ==
                            "MOE3GemmFusedCompressed" and
                        row.get("status") == "Status.EXECUTED"
                        for row in (results[lane][mode].get(
                            "full_profile") or [])),
                } for mode in ("stock", "candidate")
            } for lane in lanes}),
      check("workers_use_fresh_isolated_compiler_caches",
            len(compiler_cache_rows) == 2 * len(lanes) and
            len({row.get("neo_cache_dir")
                 for row in compiler_cache_rows}) == 2 * len(lanes) and
            all(
                row.get("neo_cache_dir") == str(
                    (raw / row["lane"] / row["mode"] /
                     "neo-cache").resolve()) and
                row.get("neo_cache_max_size") == str(
                    4 * 1024 * 1024 * 1024) and
                row.get("neo_cache_persistent") == "1"
                for row in compiler_cache_rows),
            rows=compiler_cache_rows),
      check("exact_selected_prompts_with_stock_teacher_forcing",
            all(
                results[lane][mode].get("prompt", {}).get("token_count") ==
                lane_cfg["tokens"]
                for lane, lane_cfg in lanes.items()
                for mode in ("stock", "candidate")) and
            all(
                results[lane]["stock"].get("prompt", {}).get("token_sha256") ==
                results[lane]["candidate"].get("prompt", {}).get("token_sha256")
                for lane in lanes) and
            all(
                results[lane]["candidate"].get("decode_tokens") ==
                [phase["top1"] for phase in
                 results[lane]["stock"].get("phases", [])[:cfg["steps"]]]
                for lane, cfg in lanes.items()),
            lanes={lane: {
                "stock_prompt": results[lane]["stock"].get("prompt"),
                "candidate_prompt": results[lane]["candidate"].get("prompt"),
                "teacher_tokens": results[lane]["candidate"].get(
                    "decode_tokens"),
            } for lane in lanes}),
      check("source_replaces_exact_selected_sdpa_and_kv_states",
            len(source_rows) == len(lanes) and all(
                row.get("target_layers") == list(target_layers) and
                row.get("stock_sdpa_count_before") == 10 and
                row.get("stock_sdpa_count_after") == expected_stock_sdpa and
                row.get("custom_count_after") == target_count and
                row.get("direct_i8_fixed_layout") is
                    args.direct_i8_fixed_layout and
                row.get("direct_i8_group4_full_cold") is
                    args.direct_i8_group4_full_cold and
                row.get("direct_i8_hybrid_k2_v4") is
                    args.direct_i8_hybrid_k2_v4 and
                row.get("direct_i8_quant_group") ==
                    direct_i8_key_quant_group and
                row.get("direct_i8_key_quant_group") ==
                    direct_i8_key_quant_group and
                row.get("direct_i8_value_quant_group") ==
                    direct_i8_value_quant_group and
                row.get("fixed_cold_capacity") ==
                    expected_fixed_cold_capacity and
                row.get("fuse_linear_conv_state") is
                    args.fuse_linear_conv_state and
                row.get("linear_conv_replacement_count") == (
                    len(GRAPH.LINEAR_ATTENTION_LAYERS)
                    if args.fuse_linear_conv_state else 0) and
                row.get("linear_conv_custom_count_after") == (
                    len(GRAPH.LINEAR_ATTENTION_LAYERS)
                    if args.fuse_linear_conv_state else 0) and
                row.get("state_count_after") == expected_state_count and
                set(row.get("removed_stock_states", [])) ==
                removed_stock_states and
                set(row.get("custom_states", [])) == custom_states and
                row.get("logical_hot_window") == GRAPH.HOT_WINDOW and
                row.get("exact_sink_tokens") == GRAPH.SINK_TOKENS and
                row.get("physical_ring_capacity") == GRAPH.RING_CAPACITY and
                row.get("physical_hot_capacity") == GRAPH.HOT_CAPACITY
                and row.get("hot_key_storage_planes") == (
                    3 if direct_i8_fine_full_cold else 2)
                and row.get("key_scale_bytes") == (
                    GRAPH.GROUP2_SCALE_BYTES
                    if args.direct_i8_hybrid_k2_v4 else
                    GRAPH.GROUP4_SCALE_BYTES
                    if args.direct_i8_group4_full_cold else GRAPH.SCALE_BYTES)
                and row.get("value_scale_bytes") == (
                    GRAPH.GROUP4_SCALE_BYTES
                    if direct_i8_fine_full_cold else GRAPH.SCALE_BYTES)
                for row in source_rows),
            source=source_rows),
      check("runtime_executes_exact_custom_and_stock_attention_counts",
            len(runtime_rows_all) == len(lanes) and all(
                sum(row.get("layer_type") == (
                        "condition" if args.phase_branch_prefill else
                        "CustomGPUPrimitive") and
                    row.get("node_name") in {
                        f"iq36_hot_attention_layer{layer}"
                        for layer in target_layers}
                    for row in runtime) == target_count and
                sum(row.get("layer_type") ==
                    "scaled_dot_product_attention"
                    for row in runtime) == expected_stock_sdpa
                and sum(
                    row.get("layer_type") == "CustomGPUPrimitive" and
                    row.get("node_name") in {
                        f"iq36_linear_conv_swish_layer{layer}"
                        for layer in GRAPH.LINEAR_ATTENTION_LAYERS}
                    for row in runtime) == (
                        len(GRAPH.LINEAR_ATTENTION_LAYERS)
                        if args.fuse_linear_conv_state else 0)
                for runtime in runtime_rows_all),
            runtime=runtime_rows_all),
      check("profile_executes_exact_custom_and_stock_attention_counts",
            len(profile_rows_all) == len(lanes) and all(
                sum(row.get("node_type") == (
                        "If" if args.phase_branch_prefill else
                        expected_custom_type) and
                    row.get("status") == "Status.EXECUTED"
                    for row in profile) == target_count and
                sum(row.get("node_type") == "IndirectSDPA" and
                    row.get("status") == "Status.EXECUTED"
                    for row in profile) == expected_stock_sdpa
                for profile in profile_rows_all),
            profile=profile_rows_all),
      check("profile_executes_exact_linear_conv_count",
            not args.fuse_linear_conv_state or
            (args.capture_full_profile and
             len(full_profile_rows_all) == len(lanes) and
             all(sum(
                 row.get("node_type") == "IQ36LinearConvSwish" and
                 row.get("status") == "Status.EXECUTED"
                 for row in profile) == len(GRAPH.LINEAR_ATTENTION_LAYERS)
                 for profile in full_profile_rows_all)),
            profile=full_profile_rows_all),
      check("all_teacher_forced_distributions_pass",
            len(distribution_rows) == sum(
                cfg["steps"] + 1 for cfg in lanes.values()) and all(
                row.get("finite") is True and
                row.get("top1_match") is True and
                row.get("kld_reference_to_candidate", float("inf")) <=
                KLD_LIMIT for row in distribution_rows),
            kld_limit=KLD_LIMIT, rows=distribution_rows),
      check("exact_greedy_paths_match_stock",
            len(comparisons) == len(lanes) and all(
                row.get("stock_top1") == row.get("candidate_top1")
                for row in comparisons.values()),
            rows={lane: {
                "stock": row.get("stock_top1"),
                "candidate": row.get("candidate_top1"),
            } for lane, row in comparisons.items()}),
      check("all_untouched_state_schemas_remain_finite",
            len(schema_rows) == sum(
                cfg["steps"] + 1 for cfg in lanes.values()) and all(
                row.get("stock_shared_count") == expected_shared_states and
                row.get("candidate_shared_count") == expected_shared_states and
                row.get("names_match") is True and
                row.get("shape_dtype_finite_match") is True
                for row in schema_rows),
            rows=schema_rows),
      check("candidate_hot_state_transitions_are_self_consistent",
            len(self_hot_rows) == 2 * target_count * sum(
                cfg["steps"] + 1 for cfg in lanes.values()) and all(
                row.get("written_slot_count", 0) > 0 and
                row.get("written_finite") is True and
                row.get("written_nonzero") is True and
                row.get("preserved_slots_exact") is True and
                row.get("sink_exact_from_previous") is True
                for row in self_hot_rows),
            expected_rows=2 * target_count * sum(
                cfg["steps"] + 1 for cfg in lanes.values()),
            rows=self_hot_rows),
      check("cold_transition_matches_exact_selected_contexts",
            all(
                len(comparisons.get(lane, {}).get(
                    "cold_length_progression", {})) == target_count and
                all(value == [
                    max(0, cfg["tokens"] + phase - GRAPH.HOT_WINDOW)
                    for phase in range(cfg["steps"] + 1)]
                    for value in comparisons.get(lane, {}).get(
                        "cold_length_progression", {}).values())
                for lane, cfg in lanes.items()),
            rows={lane: row.get("cold_length_progression")
                  for lane, row in comparisons.items()}),
      check("candidate_cold_codec_transitions_are_self_consistent",
            len(self_cold_rows) == target_count * sum(
                cfg["steps"] + 1 for cfg in lanes.values()) and
            all(row.get("all_exact") is True for row in self_cold_rows),
            expected_rows=target_count * sum(
                cfg["steps"] + 1 for cfg in lanes.values()),
            rows=self_cold_rows),
      check("chunked_prefill_state_transitions_are_self_consistent",
            all(
                not result.get("phases") or
                len(result["phases"][0].get("prefill_chunks", [])) <= 1 or
                (len(result["phases"][0].get(
                    "prefill_chunk_state_semantics", [])) ==
                 len(result["phases"][0].get("prefill_chunks", [])) and
                 all(
                     transition.get("start") == chunk.get("start") and
                     transition.get("end_exclusive") ==
                         chunk.get("end_exclusive") and
                     len(transition.get("layers", {})) == target_count and
                     all(
                         all(hot.get("written_slot_count", 0) > 0 and
                             hot.get("written_finite") is True and
                             hot.get("written_nonzero") is True and
                             hot.get("preserved_slots_exact") is True and
                             hot.get("sink_exact_from_previous") is True
                             for hot in layer.get("hot", {}).values()) and
                         layer.get("cold", {}).get("all_exact") is True
                         for layer in transition.get(
                             "layers", {}).values())
                     for transition, chunk in zip(
                         result["phases"][0].get(
                             "prefill_chunk_state_semantics", []),
                         result["phases"][0].get("prefill_chunks", []))))
                for result in candidate_results),
            rows=prefill_chunk_transition_rows),
      check("candidate_owns_bounded_state_without_selected_stock_kv",
            all(
                bool(result.get("phases")) and
                len(result.get("hot_bindings", [])) == 2 * target_count and
                result.get("config_before") == "" and
                result.get("config_after") == str(
                    args.custom_config.resolve()) and
                result.get("target_layers") == list(target_layers) and
                all(len(phase.get("states", [])) == expected_state_count
                    for phase in
                    result.get("phases", [])) and
                all(not removed_stock_states.intersection(
                        row["name"] for row in phase.get("states", [])) and
                    custom_states.issubset(
                        row["name"] for row in phase.get("states", []))
                    for phase in result.get("phases", []))
                and set(result["phases"][-1].get(
                    "saved_states", {})) == custom_states
                and all(not phase.get("saved_states")
                        for phase in result["phases"][:-1])
                for result in candidate_results),
            bindings=[result.get("hot_bindings")
                      for result in candidate_results]),
      check("same_request_reset_to_decode_lifetime",
            all(result.get("same_infer_request") is True and
                result.get("reset_state_called") is True
                for lane in results.values() for result in lane.values())),
  ]
  if target_count == 1:
    checks.extend([
        check("isolated_hot_state_matches_stock_f16_boundary",
              len(isolated_hot_rows) ==
                  (3 if direct_i8_fine_full_cold else 2) * len(lanes)
              and all(
                  row.get("logical_tokens") == min(
                      row.get("global_end_exclusive", 0), GRAPH.HOT_WINDOW) +
                      (GRAPH.SINK_TOKENS if
                       row.get("global_end_exclusive", 0) >
                       GRAPH.HOT_WINDOW else 0)
                  for row in isolated_hot_rows) and
              all(row.get("exact_bits") is True
                  for row in isolated_hot_rows),
              rows=isolated_hot_rows),
        check("isolated_cold_state_matches_stock_codec_boundary",
              len(isolated_cold_rows) == len(lanes) and
              all(row.get("all_exact") is True
                  for row in isolated_cold_rows),
              rows=isolated_cold_rows),
    ])
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "target_layers": list(target_layers),
      "capture_layers": list(capture_layers),
      "dump_runtime_graph": args.dump_runtime_graph,
      "capture_full_profile": args.capture_full_profile,
      "direct_i8_fixed_layout": args.direct_i8_fixed_layout,
      "direct_i8_group4_full_cold": args.direct_i8_group4_full_cold,
      "direct_i8_hybrid_k2_v4": args.direct_i8_hybrid_k2_v4,
      "direct_i8_quant_group": direct_i8_key_quant_group,
      "direct_i8_key_quant_group": direct_i8_key_quant_group,
      "direct_i8_value_quant_group": direct_i8_value_quant_group,
      "fixed_cold_capacity": expected_fixed_cold_capacity,
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": args.pack_gdn_state,
      "fc_internal_dynamic_quantize": args.fc_internal_dynamic_quantize,
      "required_checks_passed": passed,
      "checks": checks,
      "comparisons": comparisons,
      "workers": results,
      "timing_policy": "diagnostic only; no speedup claim",
  }
  correctness = {
      "schema": f"{SCHEMA}-correctness",
      "required_checks_passed": passed,
      "kld_limit": KLD_LIMIT,
      "comparisons": comparisons,
      "checks": checks,
  }
  manifest = {
      "schema": f"{SCHEMA}-manifest",
      "commit": git["commit"],
      "dirty": git["dirty"],
      "target_layers": list(target_layers),
      "commands": commands,
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "openvino_python": str(args.openvino_python.resolve()),
      "host": platform.node(),
      "kernel": platform.release(),
      "platform": platform.platform(),
      "custom_config": relative(args.custom_config),
      "candidate_gpu_plugin": (
          str(args.candidate_gpu_plugin.resolve())
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "phase_branch_prefill": args.phase_branch_prefill,
      "direct_i8_fixed_layout": args.direct_i8_fixed_layout,
      "direct_i8_group4_full_cold": args.direct_i8_group4_full_cold,
      "direct_i8_hybrid_k2_v4": args.direct_i8_hybrid_k2_v4,
      "direct_i8_quant_group": direct_i8_key_quant_group,
      "direct_i8_key_quant_group": direct_i8_key_quant_group,
      "direct_i8_value_quant_group": direct_i8_value_quant_group,
      "fixed_cold_capacity": expected_fixed_cold_capacity,
      "prefill_history_capacity": (
          GRAPH.RING_CAPACITY if args.direct_i8_fixed_layout else None),
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "pack_gdn_state": args.pack_gdn_state,
      "fc_internal_dynamic_quantize": args.fc_internal_dynamic_quantize,
      "graph_dynamic_quantization_group_size_max": (
          FC_INTERNAL_DQ_GRAPH_GROUP_SIZE_MAX
          if args.fc_internal_dynamic_quantize else None),
      "custom_source": relative(args.custom_source),
      "no_speed_claim": True,
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "correctness.json", correctness)
  write_json(out_dir / "manifest.json", manifest)
  status = "PASS" if passed else "FAIL"
  klds = {
      lane: [row.get("kld_reference_to_candidate")
             for row in values.get("distribution", [])]
      for lane, values in comparisons.items()}
  summary = [
      "# OpenVINO hot/cold attention integration gate",
      "",
      f"- status: `{status}`",
      f"- layers: `{list(target_layers)}` "
      f"(`{target_count}` custom + `{expected_stock_sdpa}` stock full attention)",
      f"- logical hot window: `{GRAPH.HOT_WINDOW}` tokens",
      f"- physical hot capacity: `{GRAPH.HOT_CAPACITY}` rows",
      ("- cold codec: packed block16-token/dim4 `I8` K with group-2 "
       "scales, dimension-major `I8` V with group-4 scales, full logical "
       "cold prefix, and dimension-major hot V"
       if args.direct_i8_hybrid_k2_v4 else
      ("- cold codec: packed block16-token/group4-dimension `I8` K, full "
       "logical cold prefix, dimension-major hot/cold V, exact `F16` scales"
       if args.direct_i8_group4_full_cold else
       "- cold codec: packed block16-token/block32-dimension `I8` K, "
       "dimension-major `I8` V, group-major exact `F16` scales"
       if args.direct_i8_fixed_layout else
       "- cold codec: signed block32 `I8` with exact `F16` scale bytes")),
      f"- teacher-forced KLD limit: `{KLD_LIMIT}`",
      f"- observed KLD rows: `{json.dumps(klds, sort_keys=True)}`",
      "- timing: diagnostic only; no speedup claim",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(f"{status}: {out_dir}")
  if not passed:
    for row in checks:
      if not row["pass"]:
        print(f"FAIL: {row['name']}", file=sys.stderr)
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
