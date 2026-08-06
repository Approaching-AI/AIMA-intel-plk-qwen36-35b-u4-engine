#!/usr/bin/env python3
"""Capture real 64-token router shapes before a grouped-DPAS prefill kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-prefill-router-shape-census-gate-v1"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TOKEN_ROOT = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input")
DEFAULT_TOKEN_MANIFEST = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/"
    "token-input-manifest.json")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
DEFAULT_LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
CAPTURE_SOURCE = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
CASES = {
    "prefill_shape_008k": {
        "path": DEFAULT_TOKEN_ROOT / "prefill_shape_008k.tokens.u32",
        "sha256": "8a3554ce47f204926f29b898eee2dd17d3f849f73ab8094c05b4f96a17b35ad8",
    },
    "sentinel_008k": {
        "path": DEFAULT_TOKEN_ROOT / "sentinel_008k.tokens.u32",
        "sha256": "7267c7ad2e8f29947dc3f20d763b8d40229e5f02b7cf6009ef6a6c792daf6185",
    },
}
DEFAULT_TILE_TOKENS = 64
LAYERS = 40
EXPERTS = 256
SELECTED_EXPERTS = 8
TARGET_LAYER_BUDGET_US = 575.33
SELECTED_FUSED_BASELINE_US = 1470.833
NECESSARY_BASELINE_SPEEDUP = SELECTED_FUSED_BASELINE_US / TARGET_LAYER_BUDGET_US
PLANNING_GB_S = 115.0


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--token-manifest", type=Path, default=DEFAULT_TOKEN_MANIFEST)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--tile-tokens", type=int, choices=(64, 1024),
                      default=DEFAULT_TILE_TOKENS)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.threads <= 0 or args.timeout_s <= 0:
    parser.error("threads and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/prefill-router-shape-census-gate-{stamp}"
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      value = json.loads(text)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*parts: str) -> str:
    result = subprocess.run(
        ["git", *parts], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def environment_from_script(path: Path) -> dict[str, str]:
  command = f"source {str(path)!r} >/dev/null 2>&1 && env -0"
  process = subprocess.run(
      ["bash", "-lc", command], cwd=ROOT, check=False, capture_output=True)
  if process.returncode != 0:
    raise SystemExit(f"environment activation failed: {path}")
  environment = dict(os.environ)
  for item in process.stdout.split(b"\0"):
    if b"=" in item:
      key, value = item.split(b"=", 1)
      environment[key.decode()] = value.decode(errors="replace")
  environment["INTEL_FORCE_PROBE"] = "b080"
  return environment


def run_logged(
    command: list[str], *, environment: dict[str, str], timeout_s: int,
    log_path: Path,
) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=ROOT, env=environment, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    returncode = process.returncode
    stdout = process.stdout
    stderr = process.stderr
    timed_out = False
  except subprocess.TimeoutExpired as error:
    returncode = 124
    stdout = error.stdout if isinstance(error.stdout, str) else ""
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    stderr += f"\ntimeout after {timeout_s}s"
    timed_out = True
  log_path.write_text(
      "$ " + " ".join(command) + "\n\n[stdout]\n" + stdout +
      "\n[stderr]\n" + stderr,
      encoding="utf-8")
  return {
      "command": command,
      "log_path": str(log_path),
      "returncode": returncode,
      "timed_out": timed_out,
  }


def compile_capture(
    args: argparse.Namespace, raw_dir: Path, environment: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
  binary = raw_dir / "prefill-router-shape-capture"
  library_dir = args.llama_build / "bin"
  command = [
      "c++", "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}", str(CAPTURE_SOURCE),
      f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1", "-Wl,-l:libggml-base.so.0.13.1",
      "-fopenmp", "-pthread", "-o", str(binary),
  ]
  result = run_logged(
      command, environment=environment, timeout_s=args.timeout_s,
      log_path=raw_dir / "compile.log")
  if result["returncode"] != 0 or not binary.is_file():
    raise SystemExit(f"capture compile failed; see {raw_dir / 'compile.log'}")
  return binary, result


def read_u32_tokens(path: Path) -> list[int]:
  data = path.read_bytes()
  if len(data) % 4 != 0:
    raise SystemExit(f"token file is not uint32 aligned: {path}")
  return [value[0] for value in struct.iter_unpack("<I", data)]


def verify_token_inputs(
    manifest_path: Path, tile_tokens: int,
) -> dict[str, dict[str, Any]]:
  manifest = load_json(manifest_path)
  manifest_cases = manifest.get("cases", {})
  verified: dict[str, dict[str, Any]] = {}
  for case_id, spec in CASES.items():
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
      raise SystemExit(f"locked token file mismatch: {case_id}")
    row = manifest_cases.get(case_id)
    if not isinstance(row, dict) or row.get("prompt_token_count") != 8192:
      raise SystemExit(f"token manifest mismatch: {case_id}")
    tokens = read_u32_tokens(path)
    if len(tokens) != 8192:
      raise SystemExit(f"expected 8192 tokens: {case_id}")
    verified[case_id] = {
        "path": str(path),
        "sha256": spec["sha256"],
        "tile_offset": 0,
        "tile_token_count": tile_tokens,
        "tile_token_ids": tokens[:tile_tokens],
        "total_token_count": len(tokens),
    }
  return verified


def run_capture(
    *, case_id: str, token_path: Path, binary: Path, args: argparse.Namespace,
    raw_dir: Path, environment: dict[str, str],
) -> dict[str, Any]:
  capture_dir = raw_dir / case_id
  capture_dir.mkdir()
  command = [
      str(binary), "--model", str(args.model), "--token-ids-file",
      str(token_path), "--binary-u32-token-file", "--token-count",
      str(args.tile_tokens), "--batch-all", "--router-only", "--out-dir",
      str(capture_dir), "--case-id", f"{case_id}_tile0", "--threads",
      str(args.threads), "--n-ctx", str(max(128, args.tile_tokens * 2)),
      "--ngl", "0", "--top-k", "1",
      "--predicts-generated-position", "0",
  ]
  result = run_logged(
      command, environment=environment, timeout_s=args.timeout_s,
      log_path=raw_dir / f"{case_id}.log")
  if result["returncode"] != 0:
    raise SystemExit(f"capture failed: {case_id}; see {raw_dir / f'{case_id}.log'}")
  result["capture_summary"] = load_json(capture_dir / "capture-summary.json")
  return result


def captured_assignments(
    capture_dir: Path, case_id: str, tile_tokens: int,
) -> list[dict[str, Any]]:
  rows = load_jsonl(capture_dir / "tensor-dumps.jsonl")
  if len(rows) != LAYERS:
    raise SystemExit(f"{case_id}: expected 40 router tensors, got {len(rows)}")
  assignments = []
  observed_layers: set[int] = set()
  for row in rows:
    match = re.fullmatch(r"ffn_moe_topk-(\d+)", str(row.get("tensor_name")))
    if match is None:
      raise SystemExit(f"{case_id}: unexpected tensor {row.get('tensor_name')}")
    layer = int(match.group(1))
    if layer in observed_layers:
      raise SystemExit(f"{case_id}: duplicate layer {layer}")
    observed_layers.add(layer)
    if row.get("tensor_type") != "i32" or row.get("ne") != [8, tile_tokens, 1, 1]:
      raise SystemExit(f"{case_id}: router tensor metadata mismatch at layer {layer}")
    strides = row.get("nb")
    if not isinstance(strides, list) or len(strides) != 4 or strides[0] != 4:
      raise SystemExit(f"{case_id}: router strides missing at layer {layer}")
    payload = capture_dir / str(row["payload_path"])
    data = payload.read_bytes()
    if len(data) != row.get("nbytes"):
      raise SystemExit(f"{case_id}: payload size mismatch at layer {layer}")
    expert_ids_by_token = []
    for token in range(tile_tokens):
      ids = [
          struct.unpack_from("<i", data, token * strides[1] + rank * strides[0])[0]
          for rank in range(SELECTED_EXPERTS)
      ]
      if len(set(ids)) != SELECTED_EXPERTS or not all(
          0 <= expert < EXPERTS for expert in ids):
        raise SystemExit(f"{case_id}: invalid router ids at layer {layer}, token {token}")
      expert_ids_by_token.append(ids)
    assignments.append({
        "case_id": case_id,
        "expert_ids_by_token": expert_ids_by_token,
        "layer": layer,
        "tile_offset": 0,
        "tile_token_count": tile_tokens,
    })
  if observed_layers != set(range(LAYERS)):
    raise SystemExit(f"{case_id}: layer coverage mismatch")
  return sorted(assignments, key=lambda row: int(row["layer"]))


def percentile(values: list[int] | list[float], fraction: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(float(value) for value in values)
  index = max(0, math.ceil(fraction * len(ordered)) - 1)
  return ordered[index]


def weight_inventory(path: Path) -> dict[int, dict[str, Any]]:
  rows = load_jsonl(path)
  if len(rows) != 693:
    raise SystemExit("tensor index does not have 693 rows")
  by_name = {str(row["name"]): row for row in rows}
  inventory: dict[int, dict[str, Any]] = {}
  for layer in range(LAYERS):
    gate = by_name.get(f"blk.{layer}.ffn_gate_up_exps.weight")
    down = by_name.get(f"blk.{layer}.ffn_down_exps.weight")
    if not isinstance(gate, dict) or not isinstance(down, dict):
      raise SystemExit(f"missing expert tensors at layer {layer}")
    if int(gate["nbytes"]) % EXPERTS or int(down["nbytes"]) % EXPERTS:
      raise SystemExit(f"expert tensor bytes are not divisible by 256 at layer {layer}")
    inventory[layer] = {
        "down_expert_bytes": int(down["nbytes"]) // EXPERTS,
        "down_type": down["ggml_type_name"],
        "gate_up_expert_bytes": int(gate["nbytes"]) // EXPERTS,
        "gate_up_type": gate["ggml_type_name"],
        "total_layer_source_bytes": sum(
            int(row["nbytes"]) for row in rows
            if str(row.get("name", "")).startswith(f"blk.{layer}.")),
    }
  return inventory


def shape_rows(
    assignments: list[dict[str, Any]], inventory: dict[int, dict[str, Any]],
    tile_tokens: int,
) -> list[dict[str, Any]]:
  assignments_per_layer = tile_tokens * SELECTED_EXPERTS
  normalization = 64.0 / tile_tokens
  source_input_bytes = tile_tokens * 2048 * 4
  bucket_input_bytes = assignments_per_layer * 2048 * 4
  down_contribution_bytes = assignments_per_layer * 2048 * 4
  final_output_bytes = tile_tokens * 2048 * 4
  permutation_scatter_stream_bytes = (
      source_input_bytes + bucket_input_bytes + down_contribution_bytes * 2 +
      final_output_bytes)
  bucket_working_set_bytes = (
      bucket_input_bytes + assignments_per_layer * 1024 * 4 +
      assignments_per_layer * 512 * 4 + down_contribution_bytes +
      final_output_bytes + assignments_per_layer * 8)
  rows = []
  for assignment in assignments:
    counts = Counter(
        expert for token_ids in assignment["expert_ids_by_token"]
        for expert in token_ids)
    group_sizes = list(counts.values())
    if sum(group_sizes) != assignments_per_layer:
      raise SystemExit("router assignment count mismatch")
    active_experts = len(group_sizes)
    layer = int(assignment["layer"])
    weights = inventory[layer]
    gate_up_unique_bytes = active_experts * int(weights["gate_up_expert_bytes"])
    selected_unique_bytes = active_experts * (
        int(weights["gate_up_expert_bytes"]) + int(weights["down_expert_bytes"]))
    rows.append({
        "active_expert_count": active_experts,
        "assignment_fraction_group_m_ge_8":
            sum(size for size in group_sizes if size >= 8) / assignments_per_layer,
        "assignment_fraction_group_m_ge_16":
            sum(size for size in group_sizes if size >= 16) / assignments_per_layer,
        "assignment_fraction_group_m_ge_32":
            sum(size for size in group_sizes if size >= 32) / assignments_per_layer,
        "assignment_count": assignments_per_layer,
        "bucket_working_set_bytes": bucket_working_set_bytes,
        "case_id": assignment["case_id"],
        "down_expert_bytes": weights["down_expert_bytes"],
        "down_type": weights["down_type"],
        "gate_up_expert_bytes": weights["gate_up_expert_bytes"],
        "gate_up_memory_floor_us_at_115_gb_s":
            gate_up_unique_bytes * normalization / (PLANNING_GB_S * 1000.0),
        "gate_up_type": weights["gate_up_type"],
        "gate_up_unique_weight_bytes": gate_up_unique_bytes,
        "group_m_histogram": {
            str(size): sum(1 for value in group_sizes if value == size)
            for size in sorted(set(group_sizes))
        },
        "group_m_max": max(group_sizes),
        "group_m_mean_active": assignments_per_layer / active_experts,
        "group_m_p50": percentile(group_sizes, 0.50),
        "group_m_p90": percentile(group_sizes, 0.90),
        "group_m_p99": percentile(group_sizes, 0.99),
        "layer": layer,
        "normalized_full_layer_plus_permutation_memory_floor_us_at_115_gb_s":
            (int(weights["total_layer_source_bytes"]) +
             permutation_scatter_stream_bytes) * normalization /
            (PLANNING_GB_S * 1000.0),
        "normalized_full_layer_weight_memory_floor_us_at_115_gb_s":
            int(weights["total_layer_source_bytes"]) * normalization /
            (PLANNING_GB_S * 1000.0),
        "permutation_scatter_stream_bytes": permutation_scatter_stream_bytes,
        "selected_path_memory_floor_us_at_115_gb_s":
            selected_unique_bytes * normalization / (PLANNING_GB_S * 1000.0),
        "selected_path_unique_weight_bytes": selected_unique_bytes,
        "synthetic_max_reuse_gate_up_bytes":
            SELECTED_EXPERTS * int(weights["gate_up_expert_bytes"]),
        "tile_token_count": tile_tokens,
        "total_layer_source_bytes": weights["total_layer_source_bytes"],
        "weight_reuse_assignments_per_active_expert":
            assignments_per_layer / active_experts,
    })
  return rows


def aggregate_shapes(
    rows: list[dict[str, Any]], tile_tokens: int,
) -> dict[str, Any]:
  cases = []
  for case_id in CASES:
    selected = [row for row in rows if row["case_id"] == case_id]
    gate_floor = [float(row["gate_up_memory_floor_us_at_115_gb_s"])
                  for row in selected]
    path_floor = [float(row["selected_path_memory_floor_us_at_115_gb_s"])
                  for row in selected]
    full_floor = [
        float(row["normalized_full_layer_weight_memory_floor_us_at_115_gb_s"])
        for row in selected]
    full_plus_permutation_floor = [
        float(row[
            "normalized_full_layer_plus_permutation_memory_floor_us_at_115_gb_s"])
        for row in selected]
    cases.append({
        "active_expert_count_max": max(row["active_expert_count"] for row in selected),
        "active_expert_count_mean": statistics.fmean(
            row["active_expert_count"] for row in selected),
        "active_expert_count_median": statistics.median(
            row["active_expert_count"] for row in selected),
        "case_id": case_id,
        "gate_up_memory_floor_us_per_layer_mean": statistics.fmean(gate_floor),
        "gate_up_memory_floor_us_per_layer_min": min(gate_floor),
        "gate_up_memory_floor_us_per_layer_max": max(gate_floor),
        "gate_up_memory_floor_within_whole_layer_budget":
            statistics.fmean(gate_floor) <= TARGET_LAYER_BUDGET_US,
        "group_m_mean_active_mean": statistics.fmean(
            row["group_m_mean_active"] for row in selected),
        "normalized_full_layer_plus_permutation_memory_floor_us_max":
            max(full_plus_permutation_floor),
        "normalized_full_layer_plus_permutation_memory_floor_us_mean":
            statistics.fmean(full_plus_permutation_floor),
        "normalized_full_layer_weight_memory_floor_us_max": max(full_floor),
        "normalized_full_layer_weight_memory_floor_us_mean":
            statistics.fmean(full_floor),
        "normalized_full_layer_plus_permutation_within_budget":
            max(full_plus_permutation_floor) <= TARGET_LAYER_BUDGET_US,
        "selected_path_memory_floor_us_per_layer_mean": statistics.fmean(path_floor),
    })
  all_gate_floors = [float(row["gate_up_memory_floor_us_at_115_gb_s"])
                     for row in rows]
  all_full_plus_floors = [
      float(row[
          "normalized_full_layer_plus_permutation_memory_floor_us_at_115_gb_s"])
      for row in rows]
  return {
      "case_rows": cases,
      "census_supports_one_component_kernel": (
          all(row["normalized_full_layer_plus_permutation_within_budget"]
              for row in cases)
          if tile_tokens > 64 else
          all(row["gate_up_memory_floor_within_whole_layer_budget"]
              for row in cases)),
      "gate_up_memory_floor_us_per_layer_mean": statistics.fmean(all_gate_floors),
      "normalized_full_layer_plus_permutation_memory_floor_us_max":
          max(all_full_plus_floors),
      "normalized_full_layer_plus_permutation_memory_floor_us_mean":
          statistics.fmean(all_full_plus_floors),
      "bucket_working_set_bytes": max(
          int(row["bucket_working_set_bytes"]) for row in rows),
      "permutation_scatter_stream_bytes": max(
          int(row["permutation_scatter_stream_bytes"]) for row in rows),
      "necessary_baseline_speedup": NECESSARY_BASELINE_SPEEDUP,
      "planning_gb_s": PLANNING_GB_S,
      "selected_fused_baseline_us": SELECTED_FUSED_BASELINE_US,
      "synthetic_baseline_active_experts": SELECTED_EXPERTS,
      "synthetic_baseline_group_m": 64,
      "target_whole_layer_budget_us": TARGET_LAYER_BUDGET_US,
      "tile_token_count": tile_tokens,
  }


def build_summary(result: dict[str, Any]) -> str:
  aggregate = result["aggregate"]
  case_lines = [
      f"- {row['case_id']}: active experts mean/max "
      f"`{row['active_expert_count_mean']:.2f}/{row['active_expert_count_max']}`, "
      f"mean group M `{row['group_m_mean_active_mean']:.3f}`, gate/up memory "
      f"floor `{row['gate_up_memory_floor_us_per_layer_mean']:.3f} us/layer`, "
      f"full+perm max "
      f"`{row['normalized_full_layer_plus_permutation_memory_floor_us_max']:.3f} us`"
      for row in aggregate["case_rows"]
  ]
  return "\n".join([
      "# Real-router grouped-prefill shape census",
      "",
      f"- tile tokens: `{aggregate['tile_token_count']}`",
      *case_lines,
      f"- target whole-layer budget: `{TARGET_LAYER_BUDGET_US:.2f} us`",
      f"- selected fused baseline: `{SELECTED_FUSED_BASELINE_US:.3f} us`",
      f"- necessary baseline movement: `{NECESSARY_BASELINE_SPEEDUP:.6f}x`",
      "- component kernel admitted: "
      f"`{str(aggregate['census_supports_one_component_kernel']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      "- speedup claim: forbidden; this is router-shape and memory-floor evidence",
      "",
  ])


def main() -> int:
  args = parse_args()
  required = [
      args.model, args.token_manifest, args.tensor_index, args.env_script,
      args.llama_source, args.llama_build, CAPTURE_SOURCE,
      *(Path(spec["path"]) for spec in CASES.values()),
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit(f"missing required paths: {missing}")
  if args.out_dir.exists():
    raise SystemExit(f"output directory already exists: {args.out_dir}")
  args.out_dir.mkdir(parents=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir()

  created_at = iso_now()
  token_inputs = verify_token_inputs(args.token_manifest, args.tile_tokens)
  if sha256_file(args.model) != MODEL_SHA256:
    raise SystemExit("locked model SHA-256 mismatch")
  inventory = weight_inventory(args.tensor_index)
  environment = environment_from_script(args.env_script)
  binary, compile_result = compile_capture(args, raw_dir, environment)
  captures: dict[str, Any] = {}
  assignments: list[dict[str, Any]] = []
  for case_id, spec in CASES.items():
    captures[case_id] = run_capture(
        case_id=case_id, token_path=Path(spec["path"]), binary=binary,
        args=args, raw_dir=raw_dir, environment=environment)
    assignments.extend(captured_assignments(
        raw_dir / case_id, case_id, args.tile_tokens))
  shapes = shape_rows(assignments, inventory, args.tile_tokens)
  aggregate = aggregate_shapes(shapes, args.tile_tokens)

  evidence_checks = [
      {"name": "capture_compiled", "pass": compile_result["returncode"] == 0},
      {"name": "two_locked_8k_prompt_tiles_captured", "pass": len(captures) == 2},
      {"name": "all_80_case_layer_router_tensors_captured",
       "pass": len(assignments) == len(CASES) * LAYERS},
      {"name": "all_router_ids_in_range_and_unique_per_token", "pass": True},
      {"name": "all_layers_have_expected_assignments",
       "pass": all(
           row["assignment_count"] == args.tile_tokens * SELECTED_EXPERTS
           for row in shapes)},
      {"name": "source_weight_inventory_present", "pass": len(inventory) == LAYERS},
  ]
  performance_checks = [
      {"name": "real_tile_memory_floor_fits_whole_layer_budget",
       "pass": aggregate["census_supports_one_component_kernel"]},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  checks = evidence_checks + performance_checks
  evidence_checks_passed = all(bool(row["pass"]) for row in evidence_checks)
  performance_checks_passed = all(bool(row["pass"]) for row in performance_checks)
  required_checks_passed = evidence_checks_passed and performance_checks_passed
  disposition = (
      "admit_one_real_router_expert_bucketed_dpas_component_kernel"
      if required_checks_passed else
      ("reject_64token_grouped_prefill_kernel_on_weight_memory_floor"
       if args.tile_tokens == 64 else
       "reject_context_wide_expert_buckets_on_memory_floor"))
  result = {
      "aggregate": aggregate,
      "captures": captures,
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_checks_passed,
      "git": git_state(),
      "inputs": {
          "capture_source": str(CAPTURE_SOURCE),
          "capture_source_sha256": sha256_file(CAPTURE_SOURCE),
          "model": {"path": str(args.model), "sha256": MODEL_SHA256,
                    "size_bytes": args.model.stat().st_size},
          "tensor_index": str(args.tensor_index),
          "token_inputs": token_inputs,
      },
      "performance_checks_passed": performance_checks_passed,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_jsonl(args.out_dir / "router-assignments.jsonl", assignments)
  write_jsonl(args.out_dir / "layer-shapes.jsonl", shapes)
  write_json(args.out_dir / "correctness.json", {
      "checks": evidence_checks,
      "evidence_checks_passed": evidence_checks_passed,
      "required_checks_passed": evidence_checks_passed,
  })
  write_json(args.out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  write_json(args.out_dir / "result.json", result)
  write_jsonl(args.out_dir / "metrics.jsonl", [
      {"metric": "gate_up_memory_floor_us_per_layer_mean", "phase": "census",
       "value": aggregate["gate_up_memory_floor_us_per_layer_mean"]},
      {"metric": "necessary_baseline_speedup", "phase": "census",
       "value": aggregate["necessary_baseline_speedup"]},
      {"metric": "normalized_full_layer_plus_permutation_memory_floor_us_max",
       "phase": "census",
       "value": aggregate[
           "normalized_full_layer_plus_permutation_memory_floor_us_max"]},
      {"metric": "required_checks_passed", "phase": "gate",
       "value": required_checks_passed},
  ])
  (args.out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "gate_up_memory_floor_us_per_layer_mean":
          aggregate["gate_up_memory_floor_us_per_layer_mean"],
      "normalized_full_layer_plus_permutation_memory_floor_us_max":
          aggregate["normalized_full_layer_plus_permutation_memory_floor_us_max"],
      "out_dir": str(args.out_dir),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
