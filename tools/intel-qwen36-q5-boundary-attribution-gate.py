#!/usr/bin/env python3
"""Localize the first Q5 correctness amplifier at one fixed teacher-forced edge."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q5-boundary-attribution-gate-v1"
CASE_ID = "router_math_reason_001"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_VARIANT = (
    ROOT / "output/q5-nonhead-surrogate-asset-20260711/"
    "qwen36-nonhead-q6-to-q5.gguf")
DEFAULT_Q5_GATE = (
    ROOT / "output/q5-surrogate-feasibility-gate-20260711Tseq631v1cleanZ/"
    "result.json")
DEFAULT_ORACLE = (
    ROOT / "oracle/r0-oracle-bundle-20260627T060028Z/"
    "teacher-forced-distribution-references.jsonl")
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
VARIANT_SHA256 = "58bfa93fcce9080dfa80d7260875afcefb3c2a2ce2dc87a34e29a03e1e83addf"
EXPECTED_REFERENCE_TOKEN = 25
EXPECTED_Q5_TOKEN = 421
EXPECTED_FIRST_MISMATCH = 8
EXPECTED_LAYER_COUNT = 40
AMPLIFIER_RATIO = 1.25
MATERIAL_RMSE = 1.0e-4
CORRECTION_ACTIVE_BYTE_CAP = 64_000_000
CORRECTION_Q5_CARRIER_CAP_GB_S = 96.0
Q5_BLOCK_BYTES = 176
Q6_BLOCK_BYTES = 210
Q6_CORRECTION_SUFFIXES = {
    "attn_qkv.weight", "attn_v.weight", "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--variant", type=Path, default=DEFAULT_VARIANT)
  parser.add_argument("--q5-gate", type=Path, default=DEFAULT_Q5_GATE)
  parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument("--n-ctx", type=int, default=64)
  parser.add_argument("--top-k", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.threads <= 0 or args.n_ctx <= 0 or args.top_k <= 0:
    parser.error("threads, n-ctx, and top-k must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/q5-boundary-attribution-gate-{stamp}"
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
  started_at = iso_now()
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
      "finished_at": iso_now(),
      "log_path": str(log_path.relative_to(log_path.parents[1])),
      "returncode": returncode,
      "started_at": started_at,
      "timed_out": timed_out,
  }


def find_case(path: Path, case_id: str) -> dict[str, Any]:
  matches = [row for row in load_jsonl(path) if row.get("case_id") == case_id]
  if len(matches) != 1:
    raise SystemExit(f"{path}: expected exactly one {case_id} row")
  return matches[0]


def fixed_prefix(
    oracle_row: dict[str, Any], q5_gate: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
  prompt = oracle_row.get("prompt_token_ids")
  generated = oracle_row.get("generated_token_ids")
  if not isinstance(prompt, list) or not all(isinstance(value, int) for value in prompt):
    raise SystemExit("oracle prompt_token_ids are invalid")
  if not isinstance(generated, list) or not all(
      isinstance(value, int) for value in generated):
    raise SystemExit("oracle generated_token_ids are invalid")
  cases = [
      row for row in q5_gate.get("cases", [])
      if isinstance(row, dict) and row.get("case_id") == CASE_ID
  ]
  if len(cases) != 1:
    raise SystemExit("Q5 gate does not contain the locked failing case")
  q5_case = cases[0]
  mismatch = q5_case.get("first_mismatch_index")
  candidate = q5_case.get("candidate_generated_ids")
  if mismatch != EXPECTED_FIRST_MISMATCH or not isinstance(candidate, list):
    raise SystemExit("Q5 gate first mismatch is not the locked position 8")
  if generated[mismatch] != EXPECTED_REFERENCE_TOKEN:
    raise SystemExit("oracle token at the locked mismatch changed")
  if candidate[mismatch] != EXPECTED_Q5_TOKEN:
    raise SystemExit("Q5 token at the locked mismatch changed")
  if candidate[:mismatch] != generated[:mismatch]:
    raise SystemExit("Q5 prefix does not match the oracle before the locked edge")
  prefix = [*prompt, *generated[:mismatch]]
  return prefix, {
      "candidate_token": candidate[mismatch],
      "generated_prefix_count": mismatch,
      "predicts_generated_position": mismatch,
      "prompt_token_count": len(prompt),
      "reference_token": generated[mismatch],
      "token_count": len(prefix),
  }


def compile_capture(
    args: argparse.Namespace, raw_dir: Path, environment: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
  binary = raw_dir / "q5-teacher-forced-boundary-capture"
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


def run_capture(
    *, label: str, model: Path, binary: Path, token_file: Path,
    prefix_meta: dict[str, Any], args: argparse.Namespace, raw_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
  capture_dir = raw_dir / label
  capture_dir.mkdir()
  command = [
      str(binary), "--model", str(model), "--token-ids-file", str(token_file),
      "--out-dir", str(capture_dir), "--case-id", CASE_ID,
      "--threads", str(args.threads), "--n-ctx", str(args.n_ctx),
      "--ngl", "0", "--top-k", str(args.top_k),
      "--predicts-generated-position",
      str(prefix_meta["predicts_generated_position"]),
      "--watch-token", str(EXPECTED_REFERENCE_TOKEN),
      "--watch-token", str(EXPECTED_Q5_TOKEN),
  ]
  result = run_logged(
      command, environment=environment, timeout_s=args.timeout_s,
      log_path=raw_dir / f"{label}.log")
  if result["returncode"] != 0:
    raise SystemExit(f"{label} capture failed; see {raw_dir / f'{label}.log'}")
  result["capture_dir"] = str(capture_dir.relative_to(args.out_dir))
  result["summary"] = load_json(capture_dir / "capture-summary.json")
  result["topk"] = load_json(capture_dir / "sampler-topk.json")
  return result


def capture_rows(capture_dir: Path) -> dict[str, dict[str, Any]]:
  rows = load_jsonl(capture_dir / "tensor-dumps.jsonl")
  name_counts: dict[str, int] = {}
  for row in rows:
    name = row.get("tensor_name")
    if isinstance(name, str):
      name_counts[name] = name_counts.get(name, 0) + 1
  by_name: dict[str, dict[str, Any]] = {}
  for row in rows:
    name = row.get("tensor_name")
    if not isinstance(name, str):
      raise SystemExit(f"{capture_dir}: missing tensor name")
    comparison_key = name
    if name_counts[name] > 1:
      shape = "x".join(str(value) for value in row.get("ne", []))
      comparison_key = f"{name}::{row.get('tensor_op')}::{shape}"
    if comparison_key in by_name:
      raise SystemExit(
          f"{capture_dir}: duplicate tensor comparison key {comparison_key!r}")
    payload = capture_dir / str(row.get("payload_path"))
    if not payload.is_file() or payload.stat().st_size != row.get("nbytes"):
      raise SystemExit(f"{capture_dir}: payload mismatch for {name}")
    row = dict(row)
    row["absolute_payload_path"] = str(payload)
    row["comparison_key"] = comparison_key
    by_name[comparison_key] = row
  return by_name


def tensor_values(row: dict[str, Any]) -> array.array[Any]:
  tensor_type = str(row.get("tensor_type", "")).lower()
  typecode = {"f32": "f", "i32": "i"}.get(tensor_type)
  if typecode is None:
    raise SystemExit(
        f"unsupported captured tensor type {tensor_type}: {row.get('tensor_name')}")
  values = array.array(typecode)
  values.frombytes(Path(str(row["absolute_payload_path"])).read_bytes())
  if values.itemsize != 4:
    raise SystemExit(f"host array item size is not four bytes for {tensor_type}")
  if sys.byteorder != "little":
    values.byteswap()
  return values


def float_metrics(reference: array.array[Any], candidate: array.array[Any]) -> dict[str, Any]:
  if len(reference) != len(candidate):
    raise SystemExit("captured tensor element counts differ")
  diff_sq = 0.0
  reference_sq = 0.0
  candidate_sq = 0.0
  dot = 0.0
  abs_sum = 0.0
  max_abs = 0.0
  changed = 0
  for ref_value, candidate_value in zip(reference, candidate):
    difference = float(candidate_value) - float(ref_value)
    absolute = abs(difference)
    diff_sq += difference * difference
    reference_sq += float(ref_value) * float(ref_value)
    candidate_sq += float(candidate_value) * float(candidate_value)
    dot += float(ref_value) * float(candidate_value)
    abs_sum += absolute
    max_abs = max(max_abs, absolute)
    changed += difference != 0.0
  count = len(reference)
  rmse = math.sqrt(diff_sq / count) if count else 0.0
  denominator = math.sqrt(reference_sq * candidate_sq)
  return {
      "changed_element_count": changed,
      "cosine": dot / denominator if denominator else None,
      "element_count": count,
      "max_abs": max_abs,
      "mean_abs": abs_sum / count if count else 0.0,
      "relative_l2": math.sqrt(diff_sq / reference_sq) if reference_sq else None,
      "rmse": rmse,
  }


def compare_tensors(
    original: dict[str, dict[str, Any]], q5: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[array.array[Any], array.array[Any]]]]:
  if set(original) != set(q5):
    missing_original = sorted(set(q5) - set(original))
    missing_q5 = sorted(set(original) - set(q5))
    raise SystemExit(
        f"capture tensor sets differ: original missing={missing_original}, "
        f"q5 missing={missing_q5}")
  comparisons: list[dict[str, Any]] = []
  float_pairs: dict[str, tuple[array.array[Any], array.array[Any]]] = {}
  for name in sorted(original):
    original_row = original[name]
    q5_row = q5[name]
    if (
        original_row.get("tensor_type") != q5_row.get("tensor_type") or
        original_row.get("ne") != q5_row.get("ne")
    ):
      raise SystemExit(f"capture metadata differs for {name}")
    original_values = tensor_values(original_row)
    q5_values = tensor_values(q5_row)
    tensor_type = str(original_row["tensor_type"]).lower()
    if tensor_type == "f32":
      metrics = float_metrics(original_values, q5_values)
      float_pairs[name] = (original_values, q5_values)
    else:
      changed = sum(left != right for left, right in zip(original_values, q5_values))
      metrics = {
          "changed_element_count": changed,
          "element_count": len(original_values),
          "exact": changed == 0,
      }
    comparisons.append({
        "comparison_key": name,
        "ne": original_row["ne"],
        "tensor_name": original_row["tensor_name"],
        "tensor_type": original_row["tensor_type"],
        **metrics,
    })
  return comparisons, float_pairs


def layer_rows(
    comparisons: list[dict[str, Any]],
    float_pairs: dict[str, tuple[array.array[Any], array.array[Any]]],
) -> list[dict[str, Any]]:
  comparison_by_name = {str(row["tensor_name"]): row for row in comparisons}
  rows: list[dict[str, Any]] = []
  for layer in range(EXPECTED_LAYER_COUNT):
    input_name = "model.input_embed" if layer == 0 else f"l_out-{layer - 1}"
    output_name = f"l_out-{layer}"
    if input_name not in float_pairs or output_name not in float_pairs:
      raise SystemExit(f"missing layer boundary {input_name} or {output_name}")
    original_input, q5_input = float_pairs[input_name]
    original_output, q5_output = float_pairs[output_name]
    if len(original_input) != len(original_output):
      raise SystemExit(f"layer {layer} residual boundary shapes differ")
    incremental_sq = 0.0
    incremental_max_abs = 0.0
    for ref_in, candidate_in, ref_out, candidate_out in zip(
        original_input, q5_input, original_output, q5_output):
      input_delta = float(candidate_in) - float(ref_in)
      output_delta = float(candidate_out) - float(ref_out)
      incremental = output_delta - input_delta
      incremental_sq += incremental * incremental
      incremental_max_abs = max(incremental_max_abs, abs(incremental))
    input_rmse = float(comparison_by_name[input_name]["rmse"])
    output_rmse = float(comparison_by_name[output_name]["rmse"])
    amplification = output_rmse / input_rmse if input_rmse else None
    rows.append({
        "amplification_ratio": amplification,
        "incremental_max_abs": incremental_max_abs,
        "incremental_rmse": math.sqrt(incremental_sq / len(original_input)),
        "input_boundary": input_name,
        "input_rmse": input_rmse,
        "layer": layer,
        "output_boundary": output_name,
        "output_rmse": output_rmse,
    })
  return rows


def first_router_divergence(comparisons: list[dict[str, Any]]) -> int | None:
  by_name = {str(row["tensor_name"]): row for row in comparisons}
  for layer in range(EXPECTED_LAYER_COUNT):
    row = by_name.get(f"ffn_moe_topk-{layer}")
    if row is not None and int(row.get("changed_element_count", 0)) > 0:
      return layer
  return None


def summarize_attribution(
    layer_boundaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
  first_nonzero = next(
      (row["layer"] for row in layer_boundaries if row["output_rmse"] > 0.0),
      None)
  first_amplifier = next(
      (row["layer"] for row in layer_boundaries
       if row["amplification_ratio"] is not None and
       row["amplification_ratio"] >= AMPLIFIER_RATIO and
       row["output_rmse"] >= MATERIAL_RMSE),
      None)
  largest_increment = max(
      layer_boundaries, key=lambda row: float(row["incremental_rmse"]))
  float_rows = [row for row in comparisons if "rmse" in row]
  component_ranking = sorted(
      float_rows, key=lambda row: float(row["rmse"]), reverse=True)[:20]
  return {
      "amplifier_definition": {
          "minimum_output_rmse": MATERIAL_RMSE,
          "minimum_output_to_input_rmse_ratio": AMPLIFIER_RATIO,
      },
      "first_amplifier_layer": first_amplifier,
      "first_nonzero_output_layer": first_nonzero,
      "first_router_topk_divergence_layer": first_router_divergence(comparisons),
      "largest_incremental_error_layer": largest_increment,
      "largest_tensor_rmse_rows": component_ranking,
  }


def correction_candidate(
    attribution: dict[str, Any], tensor_index_path: Path,
    q5_gate: dict[str, Any],
) -> dict[str, Any]:
  first_amplifier = attribution["first_amplifier_layer"]
  if not isinstance(first_amplifier, int):
    return {
        "eligible_for_single_confirm": False,
        "reason": "no material amplifier met the preregistered threshold",
    }
  candidate_layers = list(range(first_amplifier + 1))
  selected = []
  for row in load_jsonl(tensor_index_path):
    name = str(row.get("name", ""))
    suffix = row.get("suffix")
    if row.get("ggml_type_name") != "Q6_K" or suffix not in Q6_CORRECTION_SUFFIXES:
      continue
    if any(name.startswith(f"blk.{layer}.") for layer in candidate_layers):
      active_bytes = int(row["nbytes"])
      if suffix == "ffn_down_exps.weight":
        active_bytes = active_bytes * 8 // 256
      selected.append({
          "active_q6_bytes": active_bytes,
          "name": name,
          "source_q6_bytes": int(row["nbytes"]),
          "suffix": suffix,
      })
  selected.sort(key=lambda row: str(row["name"]))
  correction_active_bytes = sum(int(row["active_q6_bytes"]) for row in selected)
  traffic = q5_gate.get("traffic_budget", {})
  exact_head_bytes = int(traffic.get("exact_head_refine_q6_bytes", 0))
  all_q5_active_bytes = int(traffic.get("q5_active_bytes", 0))
  raw_q6_gb_s = float(traffic.get("raw_q6_baseline_gb_s", 0.0))
  displaced_q5_bytes = correction_active_bytes * Q5_BLOCK_BYTES // Q6_BLOCK_BYTES
  remaining_q5_bytes = all_q5_active_bytes - displaced_q5_bytes
  rows = []
  for source in traffic.get("target_rows", []):
    budget_ms = float(source["mixed_budget_ms"])
    retained_q6_ms = (
        (exact_head_bytes + correction_active_bytes) / 1e6 / raw_q6_gb_s)
    q5_budget_ms = budget_ms - retained_q6_ms
    required_q5_gb_s = (
        remaining_q5_bytes / 1e6 / q5_budget_ms
        if remaining_q5_bytes > 0 and q5_budget_ms > 0 else None)
    rows.append({
        "bucket": source["bucket"],
        "mixed_budget_ms": budget_ms,
        "q5_budget_ms": q5_budget_ms,
        "required_q5_carrier_gb_s": required_q5_gb_s,
        "retained_q6_ms_at_raw_baseline": retained_q6_ms,
    })
  finite_rows = [
      row for row in rows
      if isinstance(row["required_q5_carrier_gb_s"], (int, float))
  ]
  worst = max(
      finite_rows, key=lambda row: float(row["required_q5_carrier_gb_s"]),
      default=None)
  required_q5 = worst["required_q5_carrier_gb_s"] if worst else None
  active_cap_pass = correction_active_bytes <= CORRECTION_ACTIVE_BYTE_CAP
  carrier_cap_pass = (
      isinstance(required_q5, (int, float)) and
      required_q5 <= CORRECTION_Q5_CARRIER_CAP_GB_S and
      len(finite_rows) == len(rows))
  return {
      "active_q6_byte_cap": CORRECTION_ACTIVE_BYTE_CAP,
      "active_q6_byte_cap_pass": active_cap_pass,
      "candidate_layers": candidate_layers,
      "correction_active_q6_bytes": correction_active_bytes,
      "displaced_q5_active_bytes": displaced_q5_bytes,
      "eligible_for_single_confirm": active_cap_pass and carrier_cap_pass,
      "keep_q6_tensors": [str(row["name"]) for row in selected],
      "q5_carrier_cap_gb_s": CORRECTION_Q5_CARRIER_CAP_GB_S,
      "q5_carrier_cap_pass": carrier_cap_pass,
      "remaining_q5_active_bytes": remaining_q5_bytes,
      "required_q5_carrier_bucket": worst["bucket"] if worst else None,
      "required_q5_carrier_gb_s": required_q5,
      "selected_tensor_rows": selected,
      "selection_rule": (
          "retain the exact Q6 prefix through the first preregistered material "
          "amplifier; run this one candidate without layer enumeration"),
      "target_rows": rows,
  }


def summary_markdown(result: dict[str, Any]) -> str:
  attribution = result["attribution"]
  original_top1 = result["captures"]["original"]["topk"]["top_k"][0]["token_id"]
  q5_top1 = result["captures"]["q5"]["topk"]["top_k"][0]["token_id"]
  largest = attribution["largest_incremental_error_layer"]
  correction = result["correction_candidate"]
  return "\n".join([
      "# Q5 Teacher-forced Boundary Attribution Gate",
      "",
      f"- case: `{CASE_ID}`",
      f"- fixed oracle edge: generated position `{EXPECTED_FIRST_MISMATCH}`",
      f"- original top-1: `{original_top1}`",
      f"- Q5 top-1: `{q5_top1}`",
      f"- captured tensor count: `{result['captured_tensor_count']}` per model",
      f"- first nonzero output layer: `{attribution['first_nonzero_output_layer']}`",
      f"- first material amplifier layer: `{attribution['first_amplifier_layer']}`",
      "- first router top-k divergence layer: "
      f"`{attribution['first_router_topk_divergence_layer']}`",
      "- largest incremental boundary error: layer "
      f"`{largest['layer']}` (`{largest['incremental_rmse']:.9g}` RMSE)",
      "- one bounded correction: exact Q6 layers "
      f"`{correction.get('candidate_layers')}`; active Q6 "
      f"`{correction.get('correction_active_q6_bytes')}` bytes",
      "- correction Q5 carrier requirement: "
      f"`{correction.get('required_q5_carrier_gb_s'):.3f} GB/s`",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      "- performance claim: forbidden; this artifact is correctness attribution only",
      "",
  ])


def main() -> None:
  args = parse_args()
  required_paths = [
      args.model, args.variant, args.q5_gate, args.oracle, args.tensor_index,
      args.env_script,
      args.llama_source, args.llama_build, CAPTURE_SOURCE,
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(f"missing required paths: {missing}")
  if args.out_dir.exists():
    raise SystemExit(f"output directory already exists: {args.out_dir}")
  args.out_dir.mkdir(parents=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir()

  q5_gate = load_json(args.q5_gate)
  oracle_row = find_case(args.oracle, CASE_ID)
  prefix, prefix_meta = fixed_prefix(oracle_row, q5_gate)
  if prefix_meta["token_count"] >= args.n_ctx:
    raise SystemExit("fixed prefix does not fit n-ctx")
  token_file = raw_dir / "token-ids.txt"
  token_file.write_text("\n".join(str(value) for value in prefix) + "\n")

  model_sha256 = sha256_file(args.model)
  variant_sha256 = sha256_file(args.variant)
  if model_sha256 != MODEL_SHA256:
    raise SystemExit("locked source model SHA-256 mismatch")
  if variant_sha256 != VARIANT_SHA256:
    raise SystemExit("locked Q5 variant SHA-256 mismatch")
  if q5_gate.get("variant", {}).get("sha256") != variant_sha256:
    raise SystemExit("Q5 gate variant SHA-256 does not match the capture variant")

  environment = environment_from_script(args.env_script)
  binary, compile_result = compile_capture(args, raw_dir, environment)
  captures = {
      "original": run_capture(
          label="original", model=args.model, binary=binary,
          token_file=token_file, prefix_meta=prefix_meta, args=args,
          raw_dir=raw_dir, environment=environment),
      "q5": run_capture(
          label="q5", model=args.variant, binary=binary,
          token_file=token_file, prefix_meta=prefix_meta, args=args,
          raw_dir=raw_dir, environment=environment),
  }

  original_rows = capture_rows(raw_dir / "original")
  q5_rows = capture_rows(raw_dir / "q5")
  comparisons, float_pairs = compare_tensors(original_rows, q5_rows)
  boundaries = layer_rows(comparisons, float_pairs)
  attribution = summarize_attribution(boundaries, comparisons)
  correction = correction_candidate(attribution, args.tensor_index, q5_gate)
  original_top1 = captures["original"]["topk"]["top_k"][0]["token_id"]
  q5_top1 = captures["q5"]["topk"]["top_k"][0]["token_id"]
  checks = [
      {"name": "capture_compiled", "pass": compile_result["returncode"] == 0},
      {"name": "original_capture_succeeded",
       "pass": captures["original"]["returncode"] == 0},
      {"name": "q5_capture_succeeded", "pass": captures["q5"]["returncode"] == 0},
      {"name": "capture_tensor_sets_match", "pass": set(original_rows) == set(q5_rows)},
      {"name": "all_40_layer_outputs_captured",
       "pass": all(f"l_out-{layer}" in original_rows
                   for layer in range(EXPECTED_LAYER_COUNT))},
      {"name": "fixed_original_edge_reproduced",
       "pass": original_top1 == EXPECTED_REFERENCE_TOKEN},
      {"name": "fixed_q5_edge_reproduced", "pass": q5_top1 == EXPECTED_Q5_TOKEN},
      {"name": "q5_boundary_delta_observed",
       "pass": attribution["first_nonzero_output_layer"] is not None},
      {"name": "single_correction_candidate_within_caps",
       "pass": correction["eligible_for_single_confirm"] is True},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  result = {
      "attribution": attribution,
      "correction_candidate": correction,
      "captures": captures,
      "captured_tensor_count": len(original_rows),
      "checks": checks,
      "compile": compile_result,
      "created_at": iso_now(),
      "fixed_prefix": prefix_meta,
      "git": git_state(),
      "inputs": {
          "capture_source": str(CAPTURE_SOURCE),
          "llama_build": str(args.llama_build),
          "llama_source": str(args.llama_source),
          "model": {"path": str(args.model), "sha256": model_sha256,
                    "size_bytes": args.model.stat().st_size},
          "oracle": str(args.oracle),
          "q5_gate": str(args.q5_gate),
          "tensor_index": str(args.tensor_index),
          "variant": {"path": str(args.variant), "sha256": variant_sha256,
                      "size_bytes": args.variant.stat().st_size},
      },
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_jsonl(args.out_dir / "tensor-comparisons.jsonl", comparisons)
  write_jsonl(args.out_dir / "layer-boundaries.jsonl", boundaries)
  write_json(args.out_dir / "correctness.json", {
      "case_id": CASE_ID,
      "checks": checks,
      "original_top1_token": original_top1,
      "q5_top1_token": q5_top1,
      "required_checks_passed": required_checks_passed,
  })
  write_json(args.out_dir / "manifest.json", {
      "case_id": CASE_ID,
      "created_at": result["created_at"],
      "files": [
          "correctness.json", "layer-boundaries.jsonl", "result.json",
          "summary.md", "tensor-comparisons.jsonl",
      ],
      "raw_payloads_retained_locally": True,
      "schema_version": SCHEMA_VERSION,
  })
  write_json(args.out_dir / "result.json", result)
  (args.out_dir / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
  print(json.dumps({
      "first_amplifier_layer": attribution["first_amplifier_layer"],
      "first_router_topk_divergence_layer":
          attribution["first_router_topk_divergence_layer"],
      "largest_incremental_error_layer":
          attribution["largest_incremental_error_layer"]["layer"],
      "original_top1": original_top1,
      "out_dir": str(args.out_dir),
      "q5_top1": q5_top1,
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  if not required_checks_passed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
