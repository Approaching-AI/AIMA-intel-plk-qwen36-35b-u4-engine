#!/usr/bin/env python3
"""Gate a derived non-head Q5_K surrogate before implementing its carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q5-surrogate-feasibility-gate-v1"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_VARIANT = (
    ROOT / "output/q5-nonhead-surrogate-asset-20260711/"
    "qwen36-nonhead-q6-to-q5.gguf")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_ORACLE = (
    ROOT / "oracle/r0-oracle-bundle-20260627T060028Z/"
    "teacher-forced-distribution-references.jsonl")
DEFAULT_PROMPTS = [
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "deterministic-greedy.jsonl",
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "router-stability.jsonl",
]
DEFAULT_QUANTIZER = Path(
    "/home/intel/llama-cpp/llama-b9518/llama-quantize")
DEFAULT_SERVER = Path("/home/intel/llama-cpp/llama-b9518/llama-server")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_BUDGET = (
    ROOT / "output/router-i8-surrogate-gate-20260711Tseq628cleanZ/result.json")
Q5_BLOCK_BYTES = 176
Q6_BLOCK_BYTES = 210
DEFAULT_Q5_PROMOTION_GB_S = 80.0
RAW_Q6_BASELINE_GB_S = 52.720394412330625
Q6_SUFFIXES = (
    "attn_qkv.weight",
    "attn_v.weight",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
)
GGML_LAYOUTS = {
    0: {"block_size": 1, "name": "F32", "type_size": 4},
    12: {"block_size": 256, "name": "Q4_K", "type_size": 144},
    13: {"block_size": 256, "name": "Q5_K", "type_size": 176},
    14: {"block_size": 256, "name": "Q6_K", "type_size": 210},
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--variant", type=Path)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
  parser.add_argument("--quantizer", type=Path, default=DEFAULT_QUANTIZER)
  parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
  parser.add_argument("--port", type=int, default=18231)
  parser.add_argument("--n-probs", type=int, default=64)
  parser.add_argument("--ready-timeout-s", type=int, default=300)
  parser.add_argument("--request-timeout-s", type=int, default=600)
  parser.add_argument(
      "--keep-q6-suffix", action="append", choices=Q6_SUFFIXES, default=[],
      help="Keep this non-head Q6 tensor class exact for attribution.")
  parser.add_argument(
      "--keep-q6-tensor", action="append", default=[],
      help="Keep this exact non-head Q6 tensor for one evidence-selected correction.")
  parser.add_argument(
      "--q5-promotion-gb-s", type=float, default=DEFAULT_Q5_PROMOTION_GB_S,
      help="Required Q5 carrier ceiling for this route (default 80 GB/s).")
  parser.add_argument(
      "--retained-nonhead-q6-cap-bytes", type=int, default=0,
      help="Optional active-byte cap for an exact-Q6 correction island.")
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  keep_suffixes = sorted(set(args.keep_q6_suffix))
  args.keep_q6_suffix = keep_suffixes
  keep_tensors = sorted(set(args.keep_q6_tensor))
  args.keep_q6_tensor = keep_tensors
  if args.q5_promotion_gb_s <= 0 or args.retained_nonhead_q6_cap_bytes < 0:
    parser.error("Q5 promotion must be positive and the retained-Q6 cap nonnegative")
  if args.variant is None:
    if keep_tensors:
      digest = hashlib.sha256("\n".join(keep_tensors).encode()).hexdigest()[:12]
      args.variant = (
          ROOT / f"output/q5-correction-{digest}-asset-20260711/"
          f"qwen36-q5-correction-{digest}.gguf")
    elif keep_suffixes:
      slug = "-".join(value.removesuffix(".weight").replace("_", "-")
                      for value in keep_suffixes)
      args.variant = (
          ROOT / f"output/q5-attribution-{slug}-asset-20260711/"
          f"qwen36-q5-keep-{slug}-q6.gguf")
    else:
      args.variant = DEFAULT_VARIANT
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
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


def tensor_rows(path: Path) -> list[dict[str, Any]]:
  rows = load_jsonl(path)
  if len(rows) != 693 or len({str(row.get("name")) for row in rows}) != 693:
    raise SystemExit(f"expected 693 unique tensor rows, got {len(rows)}")
  return rows


def q6_nonhead_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  selected = [
      row for row in rows
      if row.get("ggml_type_name") == "Q6_K" and row.get("name") != "output.weight"
  ]
  if len(selected) != 60:
    raise SystemExit(f"expected 60 non-head Q6 tensors, got {len(selected)}")
  return selected


def cli_tensor_type(name: str) -> str:
  if name == "F32":
    return "f32"
  if name.startswith("Q"):
    return "q" + name[1:]
  raise SystemExit(f"unsupported locked tensor type: {name}")


def tensor_type_lines(
    rows: list[dict[str, Any]], selected_names: set[str],
) -> list[str]:
  # llama-quantize treats tensor names as patterns and uses the first match.
  # Longest-first prevents output.weight from shadowing attn_output.weight and
  # ssm_a from shadowing ssm_alpha.weight.
  ordered = sorted(rows, key=lambda row: len(str(row["name"])), reverse=True)
  return [
      f"{row['name']}=" + (
          "q5_K" if row["name"] in selected_names
          else cli_tensor_type(str(row["ggml_type_name"])))
      for row in ordered
  ]


def q5_conversion_names(output: str) -> list[str]:
  names = []
  for line in output.splitlines():
    if "type =" not in line or "q6_K" not in line or "(q5_K)" not in line:
      continue
    prefix = line.split("]", 1)
    if len(prefix) != 2:
      continue
    names.append(prefix[1].split("-", 1)[0].strip())
  return names


def read_u32(handle: Any) -> int:
  return struct.unpack("<I", handle.read(4))[0]


def read_u64(handle: Any) -> int:
  return struct.unpack("<Q", handle.read(8))[0]


def read_gguf_string(handle: Any) -> str:
  return handle.read(read_u64(handle)).decode("utf-8", errors="replace")


def skip_gguf_value(handle: Any, value_type: int) -> None:
  sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
           7: 1, 10: 8, 11: 8, 12: 8}
  if value_type in sizes:
    handle.seek(sizes[value_type], os.SEEK_CUR)
    return
  if value_type == 8:
    handle.seek(read_u64(handle), os.SEEK_CUR)
    return
  if value_type == 9:
    element_type = read_u32(handle)
    length = read_u64(handle)
    for _ in range(length):
      skip_gguf_value(handle, element_type)
    return
  raise SystemExit(f"unsupported GGUF metadata type: {value_type}")


def parse_gguf_index(path: Path) -> dict[str, Any]:
  with path.open("rb") as handle:
    if handle.read(4) != b"GGUF":
      raise SystemExit(f"not a GGUF file: {path}")
    version = read_u32(handle)
    tensor_count = read_u64(handle)
    metadata_count = read_u64(handle)
    alignment = 32
    for _ in range(metadata_count):
      key = read_gguf_string(handle)
      value_type = read_u32(handle)
      if key == "general.alignment" and value_type == 4:
        alignment = read_u32(handle)
      else:
        skip_gguf_value(handle, value_type)
    rows = []
    for index in range(tensor_count):
      name = read_gguf_string(handle)
      dims = [read_u64(handle) for _ in range(read_u32(handle))]
      tensor_type = read_u32(handle)
      offset = read_u64(handle)
      layout = GGML_LAYOUTS.get(tensor_type)
      if layout is None:
        raise SystemExit(f"unsupported GGML tensor type {tensor_type}: {name}")
      elements = math.prod(dims)
      nbytes = (
          math.ceil(elements / int(layout["block_size"]))
          * int(layout["type_size"]))
      rows.append({
          "dims": dims,
          "index": index,
          "name": name,
          "nbytes": nbytes,
          "offset": offset,
          "type": tensor_type,
          "type_name": layout["name"],
      })
    table_end = handle.tell()
  data_offset = math.ceil(table_end / alignment) * alignment
  for row in rows:
    row["absolute_offset"] = data_offset + int(row["offset"])
  return {
      "data_offset": data_offset,
      "metadata_count": metadata_count,
      "rows": rows,
      "tensor_count": tensor_count,
      "version": version,
  }


def isolated_variant_check(
    source: Path, variant: Path, selected_names: set[str],
) -> dict[str, Any]:
  source_index = parse_gguf_index(source)
  variant_index = parse_gguf_index(variant)
  source_by_name = {row["name"]: row for row in source_index["rows"]}
  variant_by_name = {row["name"]: row for row in variant_index["rows"]}
  names_match = set(source_by_name) == set(variant_by_name)
  target_rows_match = names_match and all(
      source_by_name[name]["type_name"] == "Q6_K"
      and variant_by_name[name]["type_name"] == "Q5_K"
      and source_by_name[name]["dims"] == variant_by_name[name]["dims"]
      for name in selected_names)
  non_target_names = sorted(set(source_by_name) - selected_names)
  non_target_layout_match = names_match and all(
      source_by_name[name]["type"] == variant_by_name[name]["type"]
      and source_by_name[name]["dims"] == variant_by_name[name]["dims"]
      and source_by_name[name]["nbytes"] == variant_by_name[name]["nbytes"]
      for name in non_target_names)

  source_digest = hashlib.sha256()
  variant_digest = hashlib.sha256()
  mismatched_payloads = []
  compared_bytes = 0
  if non_target_layout_match:
    with source.open("rb") as source_handle, variant.open("rb") as variant_handle:
      for name in non_target_names:
        source_row = source_by_name[name]
        variant_row = variant_by_name[name]
        source_handle.seek(int(source_row["absolute_offset"]))
        variant_handle.seek(int(variant_row["absolute_offset"]))
        remaining = int(source_row["nbytes"])
        mismatch = False
        name_bytes = name.encode("utf-8") + b"\0"
        source_digest.update(name_bytes)
        variant_digest.update(name_bytes)
        while remaining:
          amount = min(8 * 1024 * 1024, remaining)
          source_chunk = source_handle.read(amount)
          variant_chunk = variant_handle.read(amount)
          if len(source_chunk) != amount or len(variant_chunk) != amount:
            raise SystemExit(f"short tensor payload read: {name}")
          source_digest.update(source_chunk)
          variant_digest.update(variant_chunk)
          mismatch = mismatch or source_chunk != variant_chunk
          remaining -= amount
          compared_bytes += amount
        if mismatch:
          mismatched_payloads.append(name)
  return {
      "compared_non_target_bytes": compared_bytes,
      "mismatched_non_target_payloads": mismatched_payloads,
      "non_target_layout_match": non_target_layout_match,
      "non_target_payload_count": len(non_target_names),
      "non_target_payloads_byte_exact": (
          non_target_layout_match and not mismatched_payloads
          and source_digest.digest() == variant_digest.digest()),
      "source_non_target_payload_sha256": source_digest.hexdigest(),
      "source_tensor_count": source_index["tensor_count"],
      "target_q6_to_q5_rows_match": target_rows_match,
      "tensor_names_match": names_match,
      "variant_non_target_payload_sha256": variant_digest.hexdigest(),
      "variant_tensor_count": variant_index["tensor_count"],
  }


def materialize_variant(
    args: argparse.Namespace, rows: list[dict[str, Any]],
    selected: list[dict[str, Any]], raw_dir: Path,
) -> dict[str, Any]:
  args.variant.parent.mkdir(parents=True, exist_ok=True)
  type_file = args.variant.parent / "locked-nonhead-q6-to-q5.tensor-types.txt"
  selected_names = {str(row["name"]) for row in selected}
  type_file.write_text(
      "\n".join(tensor_type_lines(rows, selected_names)) + "\n",
      encoding="utf-8")
  expected_source_bytes = sum(int(row["nbytes"]) for row in selected)
  expected_q5_bytes = expected_source_bytes * Q5_BLOCK_BYTES // Q6_BLOCK_BYTES
  command = [
      str(args.quantizer), "--allow-requantize", "--tensor-type-file",
      str(type_file), str(args.model), str(args.variant), "Q4_K_M", "16",
  ]
  dry_run_command = command[:1] + ["--dry-run"] + command[1:]
  dry_run_process = subprocess.run(
      dry_run_command, cwd=ROOT, check=False, capture_output=True, text=True,
      timeout=300)
  dry_run_output = dry_run_process.stdout + "\n" + dry_run_process.stderr
  dry_run = {
      "command": dry_run_command,
      "conversion_names": q5_conversion_names(dry_run_output),
      "returncode": dry_run_process.returncode,
      "stderr": dry_run_process.stderr,
      "stdout": dry_run_process.stdout,
  }
  write_json(raw_dir / "quantize-dry-run.json", dry_run)
  if args.variant.is_file():
    run = {
        "command": command,
        "returncode": 0,
        "stderr": "variant already exists; quantization skipped",
        "stdout": "",
        "reused": True,
    }
  else:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        timeout=1800)
    run = {
        "command": command,
        "returncode": process.returncode,
        "stderr": process.stderr,
        "stdout": process.stdout,
        "reused": False,
    }
  write_json(raw_dir / "quantize.json", run)
  return {
      "dry_run": dry_run,
      "expected_q5_bytes": expected_q5_bytes,
      "expected_source_q6_bytes": expected_source_bytes,
      "locked_tensor_count": len(rows),
      "quantize": run,
      "target_tensor_count": len(selected),
      "type_file": str(type_file),
  }


def http_json(
    url: str, payload: dict[str, Any] | None = None, timeout_s: int = 30,
) -> tuple[int, dict[str, Any] | None, str]:
  data = None
  headers = {}
  method = "GET"
  if payload is not None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers["Content-Type"] = "application/json"
    method = "POST"
  request = urllib.request.Request(
      url, data=data, headers=headers, method=method)
  try:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
      text = response.read().decode("utf-8", errors="replace")
      value = json.loads(text) if text else None
      return response.status, value if isinstance(value, dict) else None, text
  except urllib.error.HTTPError as error:
    text = error.read().decode("utf-8", errors="replace")
    return error.code, None, text
  except Exception as error:  # noqa: BLE001 - preserve health diagnostics
    return 0, None, f"{type(error).__name__}: {error}"


def prompt_rows() -> list[dict[str, Any]]:
  rows = []
  for path in DEFAULT_PROMPTS:
    rows.extend(load_jsonl(path))
  if len(rows) != 6:
    raise SystemExit(f"expected six short/router prompts, got {len(rows)}")
  return rows


def oracle_rows(path: Path) -> dict[str, dict[str, Any]]:
  wanted = {row["id"] for row in prompt_rows()}
  result = {
      str(row["case_id"]): row for row in load_jsonl(path)
      if row.get("case_id") in wanted
  }
  if set(result) != wanted:
    raise SystemExit("oracle is missing short/router rows")
  return result


def compare_case(
    prompt: dict[str, Any], oracle: dict[str, Any], response: dict[str, Any],
) -> dict[str, Any]:
  probabilities = response.get("completion_probabilities")
  if not isinstance(probabilities, list):
    probabilities = []
  candidate_ids = [
      int(item["id"]) for item in probabilities
      if isinstance(item, dict) and isinstance(item.get("id"), int)
  ]
  reference_ids = [int(value) for value in oracle.get("generated_token_ids", [])]
  compared = min(len(candidate_ids), len(reference_ids))
  matches = sum(
      candidate_ids[index] == reference_ids[index] for index in range(compared))
  first_mismatch = next((
      index for index in range(compared)
      if candidate_ids[index] != reference_ids[index]
  ), None)
  if first_mismatch is None and len(candidate_ids) != len(reference_ids):
    first_mismatch = compared

  logprob_differences = []
  reference_positions = oracle.get("distribution_positions", [])
  for index in range(min(compared, len(reference_positions), len(probabilities))):
    reference_position = reference_positions[index]
    candidate = probabilities[index]
    if not isinstance(reference_position, dict) or not isinstance(candidate, dict):
      continue
    reference_id = reference_position.get("reference_token_id")
    reference_logprob = reference_position.get("reference_token_logprob")
    top = candidate.get("top_logprobs")
    if not isinstance(top, list) or not isinstance(reference_logprob, (int, float)):
      continue
    matched = next((
        item for item in top
        if isinstance(item, dict) and item.get("id") == reference_id
        and isinstance(item.get("logprob"), (int, float))
    ), None)
    if matched is not None:
      logprob_differences.append(
          abs(float(matched["logprob"]) - float(reference_logprob)))
  return {
      "candidate_generated_count": len(candidate_ids),
      "candidate_generated_ids": candidate_ids,
      "case_id": prompt["id"],
      "exact_sequence_match": candidate_ids == reference_ids,
      "first_mismatch_index": first_mismatch,
      "matched_token_count": matches,
      "max_reference_logprob_abs_diff": (
          max(logprob_differences) if logprob_differences else None),
      "reference_generated_count": len(reference_ids),
      "reference_logprob_compared_count": len(logprob_differences),
  }


def active_q6_bytes(row: dict[str, Any]) -> int:
  value = int(row["nbytes"])
  if row.get("suffix") == "ffn_down_exps.weight":
    value = value * 8 // 256
  return value


def q5_budget(
    source_budget: dict[str, Any], all_q6_rows: list[dict[str, Any]],
    selected_names: set[str], q5_promotion_gb_s: float,
) -> dict[str, Any]:
  traffic = source_budget["traffic_budget"]
  source_q6_bytes = int(traffic["q6_bytes"])
  nonhead_q6_bytes = sum(active_q6_bytes(row) for row in all_q6_rows)
  exact_head_refine_q6_bytes = source_q6_bytes - nonhead_q6_bytes
  if exact_head_refine_q6_bytes < 0:
    raise SystemExit("non-head Q6 byte inventory exceeds source budget")
  selected_source_q6_bytes = sum(
      active_q6_bytes(row) for row in all_q6_rows
      if row["name"] in selected_names)
  q5_bytes = selected_source_q6_bytes * Q5_BLOCK_BYTES // Q6_BLOCK_BYTES
  retained_q6_bytes = source_q6_bytes - selected_source_q6_bytes
  rows = []
  for source in traffic["target_rows"]:
    mixed_budget_ms = float(source["q6_budget_ms"])
    retained_q6_ms = retained_q6_bytes / 1e6 / RAW_Q6_BASELINE_GB_S
    q5_budget_ms = mixed_budget_ms - retained_q6_ms
    required: float | None = (
        q5_bytes / 1e6 / q5_budget_ms
        if q5_bytes > 0 and q5_budget_ms > 0 else None)
    predicted_ms = retained_q6_ms + q5_bytes / 1e6 / q5_promotion_gb_s
    rows.append({
        "bucket": source["bucket"],
        "mixed_budget_ms": mixed_budget_ms,
        "q5_budget_ms": q5_budget_ms,
        "predicted_mixed_ms_at_q5_kill": predicted_ms,
        "retained_q6_ms_at_raw_baseline": retained_q6_ms,
        "required_q5_carrier_gb_s": required,
    })
  worst = max(
      rows,
      key=lambda row: (
          row["required_q5_carrier_gb_s"]
          if row["required_q5_carrier_gb_s"] is not None else math.inf))
  return {
      "exact_head_refine_q6_bytes": exact_head_refine_q6_bytes,
      "mixed_route_fits_budget_at_q5_kill": all(
          row["predicted_mixed_ms_at_q5_kill"] <= row["mixed_budget_ms"]
          for row in rows),
      "nonhead_source_q6_active_bytes": nonhead_q6_bytes,
      "q5_active_bytes": q5_bytes,
      "q5_block_bytes": Q5_BLOCK_BYTES,
      "q5_promotion_gb_s": q5_promotion_gb_s,
      "raw_q6_baseline_gb_s": RAW_Q6_BASELINE_GB_S,
      "retained_q6_active_bytes": retained_q6_bytes,
      "retained_nonhead_q6_active_bytes": (
          nonhead_q6_bytes - selected_source_q6_bytes),
      "selected_source_q6_active_bytes": selected_source_q6_bytes,
      "source_q6_active_bytes": source_q6_bytes,
      "required_q5_carrier_bucket": worst["bucket"],
      "required_q5_carrier_gb_s": worst["required_q5_carrier_gb_s"],
      "target_rows": rows,
  }


def build_summary(result: dict[str, Any]) -> str:
  budget = result["traffic_budget"]
  return "\n".join([
      "# Derived non-head Q5_K surrogate feasibility gate",
      "",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- precision checks passed: "
      f"`{str(result['precision_checks_passed']).lower()}`",
      f"- performance arithmetic passed: "
      f"`{str(result['performance_checks_passed']).lower()}`",
      f"- converted tensors: `{result['variant']['target_tensor_count']}`",
      f"- byte-exact non-target tensors: "
      f"`{result['variant']['isolation']['non_target_payload_count']}`",
      f"- exact greedy cases: `{result['exact_case_count']}/6`",
      f"- exact greedy tokens: `{result['matched_token_count']}/"
      f"{result['reference_token_count']}`",
      f"- selected source Q6 active bytes: "
      f"`{budget['selected_source_q6_active_bytes']}`",
      f"- Q5 active bytes: `{budget['q5_active_bytes']}`",
      f"- retained Q6 active bytes: `{budget['retained_q6_active_bytes']}`",
      f"- required Q5 carrier: `"
      f"{budget['required_q5_carrier_gb_s']:.3f} GB/s`"
      if budget["required_q5_carrier_gb_s"] is not None
      else "- required Q5 carrier: `infeasible with retained raw Q6`",
      f"- Q5 carrier promotion gate: `>={budget['q5_promotion_gb_s']:.1f} GB/s`",
      "",
      "This gate isolates the non-head Q6->Q5 precision change while retaining",
      "the original Q6 output head and F32 routers. Exact six-case greedy",
      "agreement is an early feasibility ruler, not the required whole-model",
      "teacher-forced distribution or a native performance claim.",
      "",
  ])


def main() -> int:
  args = parse_args()
  if args.n_probs < 5:
    raise SystemExit("--n-probs must be at least 5")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (args.out_dir or (
      ROOT / f"output/q5-surrogate-feasibility-gate-{stamp}")).resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  for required in (
      args.model, args.tensor_index, args.oracle, args.quantizer, args.server,
      args.env_script, args.budget, *DEFAULT_PROMPTS):
    if not required.is_file():
      raise SystemExit(f"required input missing: {required}")

  all_rows = tensor_rows(args.tensor_index)
  all_q6_rows = q6_nonhead_rows(all_rows)
  keep_suffixes = set(args.keep_q6_suffix)
  keep_tensors = set(args.keep_q6_tensor)
  all_q6_names = {str(row["name"]) for row in all_q6_rows}
  unknown_keep_tensors = sorted(keep_tensors - all_q6_names)
  if unknown_keep_tensors:
    raise SystemExit(
        f"--keep-q6-tensor names are not non-head Q6 tensors: {unknown_keep_tensors}")
  source_rows = [
      row for row in all_q6_rows
      if row.get("suffix") not in keep_suffixes and row.get("name") not in keep_tensors]
  if not source_rows:
    raise SystemExit("at least one non-head Q6 tensor must be converted")
  selected_names = {str(row["name"]) for row in source_rows}
  variant_info = materialize_variant(args, all_rows, source_rows, raw_dir)
  quantize_ok = (
      variant_info["quantize"]["returncode"] == 0 and args.variant.is_file())
  if not quantize_ok:
    raise SystemExit("Q5 variant quantization failed")
  variant_info.update({
      "converted_suffixes": sorted({str(row.get("suffix")) for row in source_rows}),
      "kept_q6_suffixes": sorted(keep_suffixes),
      "kept_q6_tensors": sorted(keep_tensors),
      "path": str(args.variant),
      "sha256": sha256_file(args.variant),
      "size_bytes": args.variant.stat().st_size,
  })
  variant_info["isolation"] = isolated_variant_check(
      args.model, args.variant, selected_names)

  environment = environment_from_script(args.env_script)
  command = [
      str(args.server), "-m", str(args.variant), "-c", "1024", "-n", "128",
      "-ngl", "0", "--host", "127.0.0.1", "--port", str(args.port),
      "--no-webui", "-np", "1", "--log-file", str(raw_dir / "server.log"),
      "--log-colors", "off",
  ]
  write_json(raw_dir / "server-command.json", {"command": command})
  stdout = (raw_dir / "server.stdout").open("w", encoding="utf-8")
  stderr = (raw_dir / "server.stderr").open("w", encoding="utf-8")
  process = subprocess.Popen(
      command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True,
      encoding="utf-8", errors="replace", env=environment)
  ready = False
  health_attempts = []
  cases = []
  try:
    for attempt in range(args.ready_timeout_s):
      if process.poll() is not None:
        break
      status, _, text = http_json(
          f"http://127.0.0.1:{args.port}/health", timeout_s=5)
      health_attempts.append({
          "attempt": attempt + 1, "status": status, "text": text[:200]})
      if status == 200:
        ready = True
        break
      time.sleep(1)
    write_json(raw_dir / "server-health.json", {
        "attempts": health_attempts, "ready": ready})
    if ready:
      references = oracle_rows(args.oracle)
      for prompt in prompt_rows():
        request = {
            "cache_prompt": False,
            "n_predict": int(prompt["max_new_tokens"]),
            "n_probs": args.n_probs,
            "prompt": prompt["prompt"],
            "seed": 0,
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
        }
        status, response, text = http_json(
            f"http://127.0.0.1:{args.port}/completion", request,
            timeout_s=args.request_timeout_s)
        case_id = str(prompt["id"])
        (raw_dir / f"{case_id}-response.raw").write_text(text, encoding="utf-8")
        write_json(raw_dir / f"{case_id}-request.json", request)
        write_json(raw_dir / f"{case_id}-response.json", response or {})
        comparison = compare_case(prompt, references[case_id], response or {})
        comparison["request_status"] = status
        cases.append(comparison)
  finally:
    if process.poll() is None:
      process.terminate()
      try:
        process.wait(timeout=30)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    stdout.close()
    stderr.close()

  exact_cases = sum(row["exact_sequence_match"] for row in cases)
  matched_tokens = sum(int(row["matched_token_count"]) for row in cases)
  reference_tokens = sum(int(row["reference_generated_count"]) for row in cases)
  budget = q5_budget(
      load_json(args.budget), all_q6_rows, selected_names,
      args.q5_promotion_gb_s)
  isolation = variant_info["isolation"]
  dry_run_names = set(variant_info["dry_run"]["conversion_names"])
  precision_checks = [
      {"name": "variant_quantized", "pass": quantize_ok},
      {"name": "all_693_tensor_types_locked", "pass":
       variant_info["locked_tensor_count"] == 693},
      {"name": "dry_run_converts_only_selected_q6_to_q5", "pass":
       variant_info["dry_run"]["returncode"] == 0
       and dry_run_names == selected_names},
      {"name": "selected_nonhead_q6_tensors_converted", "pass":
       variant_info["target_tensor_count"] == len(source_rows)
       and isolation["target_q6_to_q5_rows_match"]},
      {"name": "variant_tensor_names_preserved", "pass":
       isolation["tensor_names_match"]
       and isolation["source_tensor_count"] == 693
       and isolation["variant_tensor_count"] == 693},
      {"name": "non_target_layouts_preserved", "pass":
       isolation["non_target_layout_match"]},
      {"name": "all_non_target_payloads_byte_exact", "pass":
       isolation["non_target_payload_count"] == 693 - len(source_rows)
       and isolation["non_target_payloads_byte_exact"]},
      {"name": "server_ready", "pass": ready},
      {"name": "six_cases_captured", "pass": len(cases) == 6},
      {"name": "all_requests_succeeded", "pass":
       len(cases) == 6 and all(row["request_status"] == 200 for row in cases)},
      {"name": "all_greedy_sequences_exact", "pass": exact_cases == 6},
      {"name": "all_reference_tokens_exact", "pass":
       reference_tokens > 0 and matched_tokens == reference_tokens},
  ]
  performance_checks = [
      {"name": "q5_carrier_kill_number_bounded", "pass":
       isinstance(budget["required_q5_carrier_gb_s"], (int, float))
       and budget["required_q5_carrier_gb_s"] <= args.q5_promotion_gb_s
       and budget["mixed_route_fits_budget_at_q5_kill"]},
      {"name": "retained_nonhead_q6_active_bytes_bounded", "pass":
       args.retained_nonhead_q6_cap_bytes == 0 or
       budget["retained_nonhead_q6_active_bytes"] <=
       args.retained_nonhead_q6_cap_bytes},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  checks = precision_checks + performance_checks
  precision_checks_passed = all(row["pass"] for row in precision_checks)
  performance_checks_passed = all(row["pass"] for row in performance_checks)
  required_checks_passed = all(row["pass"] for row in checks)
  result = {
      "cases": cases,
      "checks": checks,
      "created_at": created_at,
      "exact_case_count": exact_cases,
      "git": git_state(),
      "matched_token_count": matched_tokens,
      "performance_checks_passed": performance_checks_passed,
      "precision_checks_passed": precision_checks_passed,
      "reference_token_count": reference_tokens,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "traffic_budget": budget,
      "variant": variant_info,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "cases": cases,
      "checks": precision_checks,
      "precision_checks_passed": precision_checks_passed,
      "required_checks_passed": required_checks_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "short/router precision-feasibility gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "variant_sha256": variant_info["sha256"],
      "workstream": WORKSTREAM,
  })
  metrics = {
      "required_checks_passed": required_checks_passed,
      "precision_checks_passed": precision_checks_passed,
      "performance_checks_passed": performance_checks_passed,
      "exact_case_count": exact_cases,
      "matched_token_count": matched_tokens,
      "reference_token_count": reference_tokens,
      "required_q5_carrier_gb_s": budget["required_q5_carrier_gb_s"],
  }
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in metrics.items():
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
      "exact_case_count": exact_cases,
      "matched_token_count": matched_tokens,
      "reference_token_count": reference_tokens,
  }))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
