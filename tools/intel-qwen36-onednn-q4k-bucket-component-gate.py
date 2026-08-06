#!/usr/bin/env python3
"""Run the exact oneDNN Q4_K layer-27 expert-bucket or routed-MoE gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import struct
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-onednn-q4k-bucket-component-gate-v1"
CASE_ID = "prefill_shape_008k"
LAYER = 27
TILE_TOKENS = 1024
SELECTED_EXPERTS = 8
TARGET_LAYER_BUDGET_US_PER_64 = 575.33
PLANNING_GB_S = 115.0
ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
PAYLOAD_SHA256 = {
    f"attn_post_norm-{LAYER}":
        "8d44d06e72ff10a0f827952c02f3370d56288c8de7c482aff3e0554c2ac0395b",
    f"ffn_moe_topk-{LAYER}":
        "76ef4ea4dd7a4385f8d4b18ff00eb181f919d125b7f484c6e7bff3ff473777ba",
    f"ffn_moe_swiglu-{LAYER}":
        "187dd69ae740f39951330fbadb48407f791f9ed5145bfbac53f73c076917b648",
}
ROUTED_PAYLOAD_SHA256 = {
    **PAYLOAD_SHA256,
    f"ffn_moe_weights_norm-{LAYER}":
        "0141a67188d6d8d92e39cac7f646d6af843f4a1ac9411c6505e87cf988cfe2af",
    f"ffn_moe_down-{LAYER}":
        "b6977e220e0dc081a111ddc104607fb6e869888f01f18f852dfc60820b045f26",
    f"ffn_moe_out-{LAYER}":
        "e0dc494a2823ffe10cae0b5bd5c802fb4358b8cbc44b8495fc7c2fc0f8df76f2",
}
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TOKENS = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
DEFAULT_CENSUS = (
    ROOT / "output/prefill-router-shape-census-gate-20260711Tseq639cleanZ")
DEFAULT_CAPTURE = (
    ROOT / "output/expert-bucket-dpas-component-gate-20260711Tseq642cleanZ/"
    "raw/capture")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
DEFAULT_LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
DEFAULT_ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    f"oneDNN-{ONEDNN_COMMIT}")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-lean")
COMPONENT_SOURCE = ROOT / "engine/tools/onednn_q4k_bucket_component.cpp"
CAPTURE_SOURCE = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
TOKEN_SHA256 = "8a3554ce47f204926f29b898eee2dd17d3f849f73ab8094c05b4f96a17b35ad8"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
  parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
  parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
  parser.add_argument("--onednn-source", type=Path,
                      default=DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--routed", action="store_true")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if (args.warmup <= 0 or args.repeat <= 0 or args.threads <= 0 or
      args.timeout_s <= 0):
    parser.error("warmup, repeat, threads, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ("onednn-q4k-routed-moe-component-gate" if args.routed else
            "onednn-q4k-bucket-component-gate")
    args.out_dir = ROOT / f"output/{stem}-{stamp}"
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected a JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected a JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(root: Path, *parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=root, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output(ROOT, "status", "--porcelain")
  return {
      "commit": git_output(ROOT, "rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": process.returncode,
        "stderr": process.stderr,
        "stdout": process.stdout,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }


def shell_run(
    command: list[str], env_script: Path, timeout_s: int,
) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && "
  shell += shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s)


def write_run_logs(raw_dir: Path, name: str, result: dict[str, Any]) -> None:
  (raw_dir / f"{name}.stdout").write_text(
      str(result.get("stdout", "")), encoding="utf-8")
  (raw_dir / f"{name}.stderr").write_text(
      str(result.get("stderr", "")), encoding="utf-8")
  write_json(raw_dir / f"{name}.command.json", {
      "command": result.get("command", []),
      "returncode": result.get("returncode"),
      "timed_out": result.get("timed_out", False),
  })


def compile_capture(
    args: argparse.Namespace, raw_dir: Path,
) -> tuple[Path, dict[str, Any]]:
  binary = raw_dir / "component-capture"
  library_dir = args.llama_build / "bin"
  result = shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}", str(CAPTURE_SOURCE),
      f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1", "-Wl,-l:libggml-base.so.0.13.1",
      "-fopenmp", "-pthread", "-o", str(binary),
  ], args.env_script, args.timeout_s)
  write_run_logs(raw_dir, "capture-build", result)
  return binary, result


def selected_shape(
    census_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
  result = load_json(census_dir / "result.json")
  if (result.get("required_checks_passed") is not True or
      result.get("aggregate", {}).get("tile_token_count") != TILE_TOKENS):
    raise SystemExit("the locked 1024-token census did not pass")
  rows = [
      row for row in load_jsonl(census_dir / "layer-shapes.jsonl")
      if row.get("case_id") == CASE_ID and row.get("layer") == LAYER
  ]
  assignments = [
      row for row in load_jsonl(census_dir / "router-assignments.jsonl")
      if row.get("case_id") == CASE_ID and row.get("layer") == LAYER
  ]
  if len(rows) != 1 or len(assignments) != 1:
    raise SystemExit("the locked layer-27 census data is missing")
  return result, rows[0], assignments[0]


def bucket_for(group_m: int) -> int:
  bucket = 8
  while bucket < group_m:
    bucket *= 2
  if bucket > 512:
    raise SystemExit(f"group M {group_m} exceeds the locked ceiling")
  return bucket


def make_schedule(shape: dict[str, Any]) -> list[dict[str, int]]:
  histogram = shape.get("group_m_histogram")
  if not isinstance(histogram, dict):
    raise SystemExit("layer shape has no group-M histogram")
  buckets: Counter[int] = Counter()
  for group_m_text, expert_count_value in histogram.items():
    buckets[bucket_for(int(group_m_text))] += int(expert_count_value)
  return [
      {"m": bucket, "experts": buckets[bucket]} for bucket in sorted(buckets)
  ]


def derive_budget(
    shape: dict[str, Any], routed: bool,
) -> dict[str, float | int]:
  full_layer = int(shape["total_layer_source_bytes"])
  gate_up = int(shape["gate_up_unique_weight_bytes"])
  permutation = int(shape["permutation_scatter_stream_bytes"])
  down = (int(shape["active_expert_count"]) * 2048 * 2 * 144
          if routed else 0)
  selected_bytes = gate_up + down
  reserved_bytes = (full_layer - selected_bytes if routed else
                    full_layer - gate_up + permutation)
  whole_window_budget_us = (
      TARGET_LAYER_BUDGET_US_PER_64 * TILE_TOKENS / 64)
  reserved_us = reserved_bytes / (PLANNING_GB_S * 1000.0)
  budget: dict[str, float | int] = {
      "full_layer_source_bytes": full_layer,
      "gate_up_unique_weight_bytes": gate_up,
      "down_unique_weight_bytes": down,
      "kernel_cap_us": whole_window_budget_us - reserved_us,
      "permutation_scatter_stream_bytes": permutation,
      "planning_gb_s": PLANNING_GB_S,
      "reserved_noncomponent_bytes": reserved_bytes,
      "reserved_noncomponent_us": reserved_us,
      "selected_component_weight_bytes": selected_bytes,
      "whole_window_budget_us": whole_window_budget_us,
  }
  if routed:
    assignments = int(shape["assignment_count"])
    padded = 12352
    gather_stream = (
        assignments * 2048 * 4 + padded * (2048 + 8 * 4 + 64 * 4))
    post_gather_stream = (
        padded * (1024 * 2 * 4 + 512 * 4 + 512 + 2 * 4 + 16 * 4) +
        assignments * 2048 * (2 * 4 + 4) +
        TILE_TOKENS * 2048 * (SELECTED_EXPERTS * 4 + 4))
    budget.update({
        "current_custom_stream_bytes": gather_stream + post_gather_stream,
        "current_custom_stream_us_at_planning_gb_s":
            (gather_stream + post_gather_stream) / (PLANNING_GB_S * 1000.0),
        "gather_quantize_stream_bytes": gather_stream,
        "post_gather_custom_stream_bytes": post_gather_stream,
        "post_gather_custom_stream_us_at_planning_gb_s":
            post_gather_stream / (PLANNING_GB_S * 1000.0),
    })
  return budget


def tensor_rows(index_path: Path) -> dict[str, dict[str, Any]]:
  rows = load_jsonl(index_path)
  expected = {
      "gate_up": (f"blk.{LAYER}.ffn_gate_up_exps.weight", [2048, 1024, 256]),
      "down": (f"blk.{LAYER}.ffn_down_exps.weight", [512, 2048, 256]),
  }
  result: dict[str, dict[str, Any]] = {}
  for key, (name, dims) in expected.items():
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
      raise SystemExit(f"expected one tensor-index row for {name}")
    if (matches[0].get("dims") != dims or
        matches[0].get("ggml_type_name") != "Q4_K"):
      raise SystemExit(f"locked layer-27 {key} tensor shape or type changed")
    result[key] = matches[0]
  return result


def captured_payloads(
    capture_dir: Path, routed: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
  rows = load_jsonl(capture_dir / "tensor-dumps.jsonl")
  by_name = {str(row["tensor_name"]): row for row in rows}
  expected = {
      f"attn_post_norm-{LAYER}": ("f32", [2048, 1024, 1, 1]),
      f"ffn_moe_topk-{LAYER}": ("i32", [8, 1024, 1, 1]),
      f"ffn_moe_swiglu-{LAYER}": ("f32", [512, 8, 1024, 1]),
  }
  if routed:
    expected.update({
        f"ffn_moe_weights_norm-{LAYER}": ("f32", [8, 1024, 1, 1]),
        f"ffn_moe_down-{LAYER}": ("f32", [2048, 8, 1024, 1]),
        f"ffn_moe_out-{LAYER}": ("f32", [2048, 1024, 1, 1]),
    })
  payload_hashes = ROUTED_PAYLOAD_SHA256 if routed else PAYLOAD_SHA256
  paths: dict[str, Path] = {}
  for name, (tensor_type, shape) in expected.items():
    row = by_name.get(name)
    if row is None or row.get("tensor_type") != tensor_type or row.get("ne") != shape:
      raise SystemExit(f"captured tensor metadata mismatch: {name}")
    path = capture_dir / str(row["payload_path"])
    if (not path.is_file() or path.stat().st_size != int(row["nbytes"]) or
        sha256_file(path) != payload_hashes[name]):
      raise SystemExit(f"captured payload identity mismatch: {name}")
    paths[name] = path
  if len(by_name) != len(expected):
    raise SystemExit(f"capture must contain exactly {len(expected)} tensors")
  return by_name, paths


def captured_router_ids(payload: Path, stride: int) -> list[list[int]]:
  data = payload.read_bytes()
  return [
      [struct.unpack_from("<i", data, token * stride + rank * 4)[0]
       for rank in range(SELECTED_EXPERTS)]
      for token in range(TILE_TOKENS)
  ]


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  lines = [line for line in str(result.get("stdout", "")).splitlines()
           if line.strip()]
  if not lines:
    return {}
  try:
    value = json.loads(lines[-1])
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.model, args.census / "result.json",
      args.census / "layer-shapes.jsonl",
      args.census / "router-assignments.jsonl",
      args.tensor_index,
      args.env_script, args.cxx, args.onednn_source, args.onednn_build,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      COMPONENT_SOURCE,
  ]
  if args.routed:
    required_paths += [
        args.tokens, args.llama_source, args.llama_build, CAPTURE_SOURCE,
        args.llama_build / "bin/libllama.so.0.0.1",
    ]
  else:
    required_paths.append(args.capture / "tensor-dumps.jsonl")
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  if sha256_file(args.model) != MODEL_SHA256:
    raise SystemExit("locked model hash mismatch")
  if args.routed and sha256_file(args.tokens) != TOKEN_SHA256:
    raise SystemExit("locked token input hash mismatch")

  created_at = iso_now()
  census_result, shape, assignments = selected_shape(args.census)
  schedule = make_schedule(shape)
  budget = derive_budget(shape, args.routed)
  tensors = tensor_rows(args.tensor_index)
  capture_dir = args.capture
  capture_build: dict[str, Any] | None = None
  capture_result: dict[str, Any] | None = None
  if args.routed:
    capture_dir = raw_dir / "capture"
    capture_binary, capture_build = compile_capture(args, raw_dir)
    capture_command = [
        str(capture_binary), "--model", str(args.model),
        "--token-ids-file", str(args.tokens), "--binary-u32-token-file",
        "--token-count", str(TILE_TOKENS), "--batch-all",
        "--component-layer", str(LAYER), "--component-through-down",
        "--out-dir", str(capture_dir), "--case-id",
        f"{CASE_ID}_tile1024_layer{LAYER}_routed",
        "--threads", str(args.threads), "--n-ctx", "2048", "--ngl", "0",
        "--top-k", "1", "--predicts-generated-position", "0",
    ]
    capture_result = (
        shell_run(capture_command, args.env_script, args.timeout_s)
        if capture_build["returncode"] == 0 else
        {"command": capture_command, "returncode": 125,
         "stderr": "capture build failed", "stdout": "", "timed_out": False}
    )
    write_run_logs(raw_dir, "capture", capture_result)
    if capture_result["returncode"] != 0:
      raise SystemExit("routed component capture failed; inspect raw logs")
  metadata, payloads = captured_payloads(capture_dir, args.routed)
  topk_name = f"ffn_moe_topk-{LAYER}"
  topk_stride = int(metadata[topk_name]["nb"][1])
  router_ids_match = (
      captured_router_ids(payloads[topk_name], topk_stride) ==
      assignments["expert_ids_by_token"])
  source_commit = git_output(args.onednn_source, "rev-parse", "HEAD")

  binary = raw_dir / "onednn-q4k-bucket-component"
  build_result = shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(COMPONENT_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(binary),
  ], args.env_script, args.timeout_s)
  write_run_logs(raw_dir, "build", build_result)

  command = [
      str(binary), "--model", str(args.model),
      "--weight-offset", str(tensors["gate_up"]["absolute_offset"]),
      "--weight-bytes", str(tensors["gate_up"]["nbytes"]),
      "--input", str(payloads[f"attn_post_norm-{LAYER}"]),
      "--topk", str(payloads[topk_name]),
      "--topk-stride", str(topk_stride),
      "--oracle", str(payloads[f"ffn_moe_swiglu-{LAYER}"]),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--kernel-cap-us", str(budget["kernel_cap_us"]),
  ]
  if args.routed:
    command += [
        "--down-weight-offset", str(tensors["down"]["absolute_offset"]),
        "--down-weight-bytes", str(tensors["down"]["nbytes"]),
        "--router-weights",
        str(payloads[f"ffn_moe_weights_norm-{LAYER}"]),
        "--down-oracle", str(payloads[f"ffn_moe_down-{LAYER}"]),
        "--moe-oracle", str(payloads[f"ffn_moe_out-{LAYER}"]),
    ]
  component_result = (
      shell_run(command, args.env_script, args.timeout_s)
      if build_result["returncode"] == 0 else
      {"command": command, "returncode": 125, "stderr": "build failed",
       "stdout": "", "timed_out": False}
  )
  write_run_logs(raw_dir, "component", component_result)
  probe = parse_probe(component_result)

  observed_schedule = [
      {"experts": row.get("experts"), "m": row.get("m")}
      for row in probe.get("buckets", []) if isinstance(row, dict)
  ]
  evidence_checks = [
      {"name": "locked_census_gate_passed",
       "pass": census_result.get("required_checks_passed") is True},
      {"name": "pinned_onednn_source_commit",
       "pass": source_commit == ONEDNN_COMMIT,
       "observed": source_commit, "required": ONEDNN_COMMIT},
      {"name": "locked_capture_payload_hashes_match", "pass": True},
      {"name": "captured_router_ids_match_seq639", "pass": router_ids_match},
      {"name": "component_build_passed",
       "pass": build_result["returncode"] == 0},
      {"name": "component_execution_completed",
       "pass": component_result["returncode"] in (0, 2) and bool(probe)},
      {"name": "runtime_onednn_hash_matches_source",
       "pass": probe.get("onednn_version", {}).get("hash") == ONEDNN_COMMIT},
      {"name": "arc_b390_selected",
       "pass": "B390" in str(probe.get("device_name"))},
      {"name": "real_layer_schedule_preserved",
       "pass": observed_schedule == schedule and
               probe.get("active_experts") == shape["active_expert_count"] and
               probe.get("assignment_count") == shape["assignment_count"] and
               probe.get("padded_assignments") == 12352},
      {"name": "all_main_and_min_buckets_use_jit_gemm",
       "pass": probe.get("implementations_pass") is True},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  if args.routed:
    evidence_checks += [
        {"name": "routed_capture_build_passed",
         "pass": capture_build is not None and
                 capture_build.get("returncode") == 0},
        {"name": "routed_capture_completed",
         "pass": capture_result is not None and
                 capture_result.get("returncode") == 0},
        {"name": "capture_has_exact_six_tensor_boundary",
         "pass": len(metadata) == 6},
        {"name": "dynamic_routed_moe_mode_selected",
         "pass": probe.get("mode") == "routed_moe"},
    ]
  compare = probe.get("compare", {})
  weighted_down = probe.get("weighted_down_compare", {})
  moe_compare = probe.get("moe_compare", {})
  expected_repacked = 698_351_616 if args.routed else 465_567_744
  correctness_checks = [
      {"name": "all_active_q4_codes_repacked_losslessly",
       "pass": probe.get("repack_pass") is True and
               probe.get("repacked_q4_code_count") == expected_repacked and
               probe.get("repack_mismatch_count") == 0},
      {"name": "exact_q4k_component_correctness_passed",
       "pass": probe.get("correctness_pass") is True},
      {"name": "all_4194304_values_compared",
       "pass": compare.get("compared_value_count") == 4_194_304},
      {"name": "zero_values_above_5e_3",
       "pass": compare.get("mismatch_count") == 0 and
               float(compare.get("max_abs_diff", float("inf"))) <= 5e-3},
  ]
  if args.routed:
    correctness_checks += [
        {"name": "all_16777216_weighted_down_values_compared",
         "pass": weighted_down.get("compared_value_count") == 16_777_216 and
                 weighted_down.get("mismatch_count") == 0},
        {"name": "all_2097152_routed_output_values_compared",
         "pass": moe_compare.get("compared_value_count") == 2_097_152 and
                 moe_compare.get("mismatch_count") == 0},
    ]
  performance_checks = [
      {"name": "complete_runtime_component_below_cap",
       "pass": probe.get("performance_pass") is True and
               float(probe.get("minimum_us", float("inf"))) <=
               float(budget["kernel_cap_us"])},
  ]
  evidence_checks_passed = all(bool(row["pass"]) for row in evidence_checks)
  correctness_checks_passed = all(bool(row["pass"]) for row in correctness_checks)
  performance_checks_passed = all(bool(row["pass"]) for row in performance_checks)
  required_checks_passed = (
      evidence_checks_passed and correctness_checks_passed and
      performance_checks_passed)
  disposition = (
      ("admit_exact_onednn_q4k_routed_moe_layer_integration" if args.routed
       else "admit_exact_onednn_q4k_expert_bucket_layer_integration")
      if required_checks_passed else
      ("reject_exact_onednn_q4k_routed_moe_above_whole_layer_cap"
       if args.routed and correctness_checks_passed else
       "reject_exact_onednn_q4k_component_on_correctness_or_cap"))
  payload_hashes = ROUTED_PAYLOAD_SHA256 if args.routed else PAYLOAD_SHA256
  if args.routed:
    runtime_boundary = {
        "excluded": ["one-time resident Q4_K-to-U4/scale/min repack"],
        "included": [
            "dynamic expert-major input gather and Q8_K quantization",
            "fourteen gate/up and fourteen down oneDNN JIT-GEMMs",
            "exact affine-min compensation and SwiGLU-to-Q8_K quantization",
            "router weighting, contribution stream, inverse scatter",
            "submission and final queue drain",
        ],
        "reason": "whole routed-MoE killer boundary after ADR 0008",
    }
  else:
    runtime_boundary = {
        "excluded": [
            "one-time resident Q4_K-to-U4/scale/min repack",
            "captured F32-to-Q8 preparation",
            "expert-major input permutation",
            "inverse output scatter",
        ],
        "included": [
            "seven grouped-scale S8xU4 oneDNN JIT-GEMMs",
            "seven F32 affine-min compensation oneDNN JIT-GEMMs",
            "seven compensation-plus-SwiGLU OpenCL kernels",
            "submission and final queue drain",
        ],
        "reason": (
            "ADR 0008 component boundary; excluded dynamic work remains "
            "chargeable by the next whole-layer gate"),
    }
  result = {
      "budget": budget,
      "case_id": CASE_ID,
      "checks": evidence_checks + correctness_checks + performance_checks,
      "correctness_checks_passed": correctness_checks_passed,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_checks_passed,
      "git": git_state(),
      "layer": LAYER,
      "performance_checks_passed": performance_checks_passed,
      "probe": probe,
      "required_checks_passed": required_checks_passed,
      "runtime_boundary": runtime_boundary,
      "schedule": {
          "active_experts": shape["active_expert_count"],
          "actual_assignments": shape["assignment_count"],
          "buckets": schedule,
          "padded_assignments": 12352,
          "padding_ratio": 12352 / shape["assignment_count"] - 1,
      },
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "capture": str(capture_dir),
          "capture_payload_sha256": payload_hashes,
          "capture_source": (str(CAPTURE_SOURCE.relative_to(ROOT))
                             if args.routed else None),
          "capture_source_sha256": (sha256_file(CAPTURE_SOURCE)
                                    if args.routed else None),
          "census": str(args.census.relative_to(ROOT)),
          "component": str(COMPONENT_SOURCE.relative_to(ROOT)),
          "component_sha256": sha256_file(COMPONENT_SOURCE),
          "model_path": str(args.model),
          "model_sha256": MODEL_SHA256,
          "onednn_build": str(args.onednn_build),
          "onednn_commit": source_commit,
          "onednn_source": str(args.onednn_source),
      },
      "speedup_claims_allowed": False,
      "tensors": tensors,
      "tile_tokens": TILE_TOKENS,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": evidence_checks + correctness_checks,
      "comparison": compare,
      "moe_comparison": moe_compare if args.routed else None,
      "correctness_checks_passed": correctness_checks_passed,
      "evidence_checks_passed": evidence_checks_passed,
      "weighted_down_comparison": weighted_down if args.routed else None,
  })
  write_json(out_dir / "capture-metadata.json", {
      "case_id": CASE_ID,
      "layer": LAYER,
      "payload_sha256": payload_hashes,
      "routed": args.routed,
      "router_ids_match_seq639": router_ids_match,
      "tensors": list(metadata.values()),
      "tile_tokens": TILE_TOKENS,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": ("single real 1024-token routed-MoE killer boundary"
                 if args.routed else
                 "standalone exact layer-27 component gate"),
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "complete_component_minimum_us",
       "value": probe.get("minimum_us")},
      {"metric": "complete_component_median_us",
       "value": probe.get("median_us")},
      {"metric": "component_cap_us", "value": budget["kernel_cap_us"]},
      {"metric": "cap_fraction",
       "value": (float(probe["minimum_us"]) / float(budget["kernel_cap_us"])
                 if "minimum_us" in probe else None)},
      {"metric": "max_abs_diff", "value": compare.get("max_abs_diff")},
      {"metric": "rmse", "value": compare.get("rmse")},
      {"metric": "weighted_down_max_abs_diff",
       "value": weighted_down.get("max_abs_diff") if args.routed else None},
      {"metric": "moe_max_abs_diff",
       "value": moe_compare.get("max_abs_diff") if args.routed else None},
      {"metric": "required_checks_passed", "value": required_checks_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  minimum_us = probe.get("minimum_us", "unavailable")
  median_us = probe.get("median_us", "unavailable")
  cap_fraction = (
      float(probe["minimum_us"]) / float(budget["kernel_cap_us"])
      if "minimum_us" in probe else float("nan"))
  title = ("# Exact oneDNN Q4_K routed-MoE killer gate" if args.routed else
           "# Exact oneDNN Q4_K expert-bucket component gate")
  summary = [
      title,
      "",
      f"- case/layer: `{CASE_ID}` / `{LAYER}`",
      f"- active experts / assignments: `{shape['active_expert_count']}` / "
      f"`{shape['assignment_count']}`",
      f"- lossless U4 repack: `{probe.get('repacked_q4_code_count')}` codes, "
      f"`{probe.get('repack_mismatch_count')}` mismatches",
      f"- comparison: `{compare.get('compared_value_count')}` values, max abs "
      f"`{compare.get('max_abs_diff')}`, RMSE `{compare.get('rmse')}`",
      f"- complete runtime minimum / median: `{minimum_us} / {median_us} us`",
      f"- component cap / fraction: `{budget['kernel_cap_us']:.3f} us` / "
      f"`{cap_fraction:.3f}`",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
  ]
  if args.routed:
    summary += [
        "The timed boundary includes dynamic gather/Q8_K, exact gate/up,",
        "SwiGLU/Q8_K, exact down, router weighting, contribution streaming,",
        "inverse scatter, submission, and queue drain. Only the one-time",
        "resident weight repack is excluded. This is a route-closing component",
        "result, not a native prefill or product speed claim.",
        "",
    ]
  else:
    summary += [
        "The timed boundary includes all 14 oneDNN JIT-GEMMs, exact affine-min",
        "compensation, SwiGLU, submission, and queue drain. One-time resident",
        "weight repack and the already-budgeted input permutation/scatter remain",
        "outside this component timer and must be charged by whole-layer evidence.",
        "This is not a native prefill or product speed claim.",
        "",
    ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "minimum_us": probe.get("minimum_us"),
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
