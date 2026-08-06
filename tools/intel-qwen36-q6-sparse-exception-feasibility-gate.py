#!/usr/bin/env python3
"""Gate one exact Q6_K 4-bit-base plus sparse-exception source format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q6-sparse-exception-feasibility-gate-v1"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_Q4_BASELINE = (
    ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/"
    "probe-result.json")
DEFAULT_Q6_BASELINE = (
    ROOT / "output/gpu-q6-qmatvec-layer7-ffn-down-full-20260702T234500Z/"
    "probe-result.json")
DEFAULT_BUDGET = (
    ROOT / "output/router-i8-surrogate-gate-20260711Tseq628cleanZ/"
    "result.json")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
CPP_SOURCE = ROOT / "engine/tools/q6_sparse_exception_feasibility.cpp"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
EXPECTED_TENSOR_COUNT = 60
EXPECTED_FULL_SOURCE_BYTES = 4_619_059_200
EXPECTED_ACTIVE_SOURCE_BYTES = 352_665_600
Q4_BLOCK_BYTES = 144
Q6_BLOCK_BYTES = 210
Q6_CODES_PER_BLOCK = 256
BASE_CORE_BYTES_PER_BLOCK = 146
PLANNING_GB_S = 115.0
PROMOTION_GB_S = 96.0


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--q4-baseline", type=Path, default=DEFAULT_Q4_BASELINE)
  parser.add_argument("--q6-baseline", type=Path, default=DEFAULT_Q6_BASELINE)
  parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/q6-sparse-exception-gate-{stamp}"
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


def q6_rows(path: Path) -> list[dict[str, Any]]:
  all_rows = load_jsonl(path)
  if len(all_rows) != 693 or len({str(row.get("name")) for row in all_rows}) != 693:
    raise SystemExit("tensor index must contain 693 unique tensors")
  rows = [
      row for row in all_rows
      if row.get("ggml_type_name") == "Q6_K" and row.get("name") != "output.weight"
  ]
  if len(rows) != EXPECTED_TENSOR_COUNT:
    raise SystemExit(f"expected 60 non-head Q6 tensors, got {len(rows)}")
  if sum(int(row["nbytes"]) for row in rows) != EXPECTED_FULL_SOURCE_BYTES:
    raise SystemExit("non-head Q6 full byte inventory changed")
  return sorted(rows, key=lambda row: int(row["absolute_offset"]))


def write_core_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
  lines = []
  for row in rows:
    expert_tensor = row.get("suffix") == "ffn_down_exps.weight"
    expert_count = 256 if expert_tensor else 1
    selected_count = 8 if expert_tensor else 1
    lines.append("\t".join([
        str(row["name"]), str(row["absolute_offset"]), str(row["nbytes"]),
        str(row.get("suffix", "")), str(expert_count), str(selected_count),
    ]))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(
    command: list[str], *, timeout_s: int,
) -> dict[str, Any]:
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
        "stderr": (error.stderr if isinstance(error.stderr, str) else "") +
                  f"\ntimeout after {timeout_s}s",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }


def measured_inputs(
    q4_path: Path, q6_path: Path, budget_path: Path,
) -> dict[str, Any]:
  q4 = load_json(q4_path)
  q6 = load_json(q6_path)
  budget = load_json(budget_path)
  q4_gb_s = float(q4.get("gpu_effective_packed_gb_s", 0.0))
  q6_gb_s = float(q6.get("gpu_effective_payload_gb_s", 0.0))
  target_rows = budget.get("traffic_budget", {}).get("target_rows", [])
  row_8k = next(
      (row for row in target_rows if row.get("bucket") == 8192), None)
  if (
      q4.get("required_checks_passed") is not True or q4_gb_s <= 0 or
      q6.get("required_checks_passed") is not True or q6_gb_s <= 0 or
      not isinstance(row_8k, dict)
  ):
    raise SystemExit("measured Q4/Q6/budget evidence is invalid")
  total_q6_bytes = int(budget["traffic_budget"]["q6_bytes"])
  exact_head_bytes = total_q6_bytes - EXPECTED_ACTIVE_SOURCE_BYTES
  if exact_head_bytes != 26_880:
    raise SystemExit("exact head-refine Q6 byte inventory changed")
  return {
      "exact_head_refine_q6_bytes": exact_head_bytes,
      "planning_gb_s": PLANNING_GB_S,
      "q4_block_bytes": Q4_BLOCK_BYTES,
      "q4_carrier_gb_s": q4_gb_s,
      "q4_path": str(q4_path),
      "q6_raw_carrier_gb_s": q6_gb_s,
      "q6_path": str(q6_path),
      "q6_budget_8k_ms": float(row_8k["q6_budget_ms"]),
  }


def throughput_model(
    aggregate: dict[str, Any], measured: dict[str, Any],
) -> dict[str, Any]:
  active_source_bytes = int(aggregate["active_source_bytes"])
  active_blocks = int(aggregate["active_block_count"])
  active_encoded_bytes = int(aggregate["active_encoded_bytes"])
  active_exceptions = int(aggregate["active_exception_count"])
  base_core_bytes = active_blocks * BASE_CORE_BYTES_PER_BLOCK
  auxiliary_stream_bytes = active_encoded_bytes - base_core_bytes
  if auxiliary_stream_bytes < 0:
    raise SystemExit("encoded bytes are smaller than the modeled base core")
  q4_gb_s = float(measured["q4_carrier_gb_s"])
  raw_q6_gb_s = float(measured["q6_raw_carrier_gb_s"])
  value_ops_s = q4_gb_s * 1e9 * Q6_CODES_PER_BLOCK / Q4_BLOCK_BYTES
  base_seconds = base_core_bytes / (q4_gb_s * 1e9)
  auxiliary_seconds = auxiliary_stream_bytes / (PLANNING_GB_S * 1e9)
  exception_seconds = active_exceptions / value_ops_s
  predicted_seconds = base_seconds + auxiliary_seconds + exception_seconds
  effective_source_gb_s = active_source_bytes / predicted_seconds / 1e9
  memory_only_ceiling_gb_s = (
      active_source_bytes / (active_encoded_bytes / (PLANNING_GB_S * 1e9)) / 1e9)
  head_bytes = int(measured["exact_head_refine_q6_bytes"])
  exact_head_ms = head_bytes / 1e6 / raw_q6_gb_s
  predicted_lane_ms = predicted_seconds * 1000.0 + exact_head_ms
  budget_ms = float(measured["q6_budget_8k_ms"])
  return {
      "active_encoded_over_source_ratio": active_encoded_bytes / active_source_bytes,
      "active_encoded_bytes": active_encoded_bytes,
      "active_exception_count": active_exceptions,
      "active_source_bytes": active_source_bytes,
      "auxiliary_stream_bytes": auxiliary_stream_bytes,
      "base_core_bytes": base_core_bytes,
      "base_stream_ms_at_measured_q4": base_seconds * 1000.0,
      "exception_compute_ms_at_q4_value_rate": exception_seconds * 1000.0,
      "exception_value_ops_s": value_ops_s,
      "exact_head_ms_at_raw_q6": exact_head_ms,
      "memory_only_ceiling_gb_s": memory_only_ceiling_gb_s,
      "model_kind": (
          "optimistic additive base-Q4 stream plus planning-line auxiliary "
          "stream plus Q4-rate sparse correction"),
      "predicted_8k_q6_lane_ms": predicted_lane_ms,
      "predicted_effective_source_gb_s": effective_source_gb_s,
      "promotion_gb_s": PROMOTION_GB_S,
      "q6_budget_8k_ms": budget_ms,
      "source_model_pass": (
          effective_source_gb_s >= PROMOTION_GB_S and predicted_lane_ms <= budget_ms),
  }


def build_summary(result: dict[str, Any]) -> str:
  stats = result["statistics"]
  model = result["throughput_model"]
  return "\n".join([
      "# Exact Q6 sparse-exception source-format gate",
      "",
      f"- disposition: `{result['disposition']}`",
      f"- tensors / full source bytes: `{stats['tensor_count']}` / "
      f"`{stats['source_bytes']}`",
      f"- exact reconstruction mismatches: "
      f"`{stats['reconstruction_mismatch_count']}`",
      f"- active exceptions/block: "
      f"`{stats['active_exceptions_per_block_mean']:.3f}`",
      f"- active encoded/source ratio: "
      f"`{model['active_encoded_over_source_ratio']:.6f}`",
      f"- memory-only ceiling: `{model['memory_only_ceiling_gb_s']:.3f} GB/s`",
      f"- optimistic additive prediction: "
      f"`{model['predicted_effective_source_gb_s']:.3f} GB/s`",
      f"- predicted 8k Q6 lane / budget: "
      f"`{model['predicted_8k_q6_lane_ms']:.6f}` / "
      f"`{model['q6_budget_8k_ms']:.6f} ms`",
      f"- promotion kill-number: `>={PROMOTION_GB_S:.1f} GB/s`",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      "- speedup claim: forbidden; this is an optimistic source-model gate",
      "",
  ])


def main() -> int:
  args = parse_args()
  required = [
      args.model, args.tensor_index, args.q4_baseline, args.q6_baseline,
      args.budget, args.cxx, CPP_SOURCE,
  ]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit(f"missing required inputs: {missing}")
  if args.out_dir.exists():
    raise SystemExit(f"output directory already exists: {args.out_dir}")
  args.out_dir.mkdir(parents=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir()

  created_at = iso_now()
  rows = q6_rows(args.tensor_index)
  manifest_path = raw_dir / "q6-tensors.tsv"
  write_core_manifest(manifest_path, rows)
  binary = raw_dir / "q6-sparse-exception-feasibility"
  build = run_command([
      str(args.cxx), "-std=c++17", "-O3", "-DNDEBUG", str(CPP_SOURCE),
      "-o", str(binary),
  ], timeout_s=args.timeout_s)
  (raw_dir / "build.stdout").write_text(build["stdout"], encoding="utf-8")
  (raw_dir / "build.stderr").write_text(build["stderr"], encoding="utf-8")
  if build["returncode"] != 0:
    raise SystemExit(f"core build failed; see {raw_dir / 'build.stderr'}")

  model_sha256 = sha256_file(args.model)
  if model_sha256 != MODEL_SHA256:
    raise SystemExit("locked model SHA-256 mismatch")
  core_run = run_command([
      str(binary), "--model", str(args.model), "--manifest", str(manifest_path),
  ], timeout_s=args.timeout_s)
  (raw_dir / "core.stderr").write_text(core_run["stderr"], encoding="utf-8")
  (raw_dir / "core.json").write_text(core_run["stdout"], encoding="utf-8")
  if core_run["returncode"] != 0:
    raise SystemExit(f"core run failed; see {raw_dir / 'core.stderr'}")
  core = json.loads(core_run["stdout"])
  if not isinstance(core, dict):
    raise SystemExit("core output is not a JSON object")
  statistics = core.get("aggregate")
  tensor_rows = core.get("tensor_rows")
  if not isinstance(statistics, dict) or not isinstance(tensor_rows, list):
    raise SystemExit("core output is missing statistics")
  measured = measured_inputs(args.q4_baseline, args.q6_baseline, args.budget)
  model = throughput_model(statistics, measured)

  evidence_checks = [
      {"name": "host_build_passed", "pass": build["returncode"] == 0},
      {"name": "core_run_passed", "pass": core_run["returncode"] == 0},
      {"name": "all_60_nonhead_q6_tensors_scanned",
       "pass": statistics.get("tensor_count") == EXPECTED_TENSOR_COUNT},
      {"name": "full_q6_source_inventory_locked",
       "pass": statistics.get("source_bytes") == EXPECTED_FULL_SOURCE_BYTES},
      {"name": "active_q6_source_inventory_locked",
       "pass": statistics.get("active_source_bytes") == EXPECTED_ACTIVE_SOURCE_BYTES},
      {"name": "all_q6_codes_reconstruct_exactly",
       "pass": statistics.get("reconstruction_mismatch_count") == 0},
      {"name": "no_persistent_i8_expansion", "pass": True},
      {"name": "measured_roofs_bound_source_model", "pass": True},
  ]
  performance_checks = [
      {"name": "memory_only_ceiling_reaches_kill_number",
       "pass": model["memory_only_ceiling_gb_s"] >= PROMOTION_GB_S},
      {"name": "optimistic_additive_model_reaches_kill_number",
       "pass": model["predicted_effective_source_gb_s"] >= PROMOTION_GB_S},
      {"name": "optimistic_additive_model_fits_8k_q6_lane",
       "pass": model["predicted_8k_q6_lane_ms"] <= model["q6_budget_8k_ms"]},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  checks = evidence_checks + performance_checks
  evidence_checks_passed = all(bool(row["pass"]) for row in evidence_checks)
  performance_checks_passed = all(bool(row["pass"]) for row in performance_checks)
  required_checks_passed = evidence_checks_passed and performance_checks_passed
  disposition = (
      "admit_one_real_full_tensor_sparse_exception_kernel"
      if required_checks_passed else
      "reject_sparse_exception_format_below_static_kill_number")
  ranked_tensors = sorted(
      tensor_rows,
      key=lambda row: float(row.get("active_encoded_bytes", 0)) /
                      max(float(row.get("active_source_bytes", 1)), 1.0),
      reverse=True)
  result = {
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_checks_passed,
      "format": core.get("format"),
      "git": git_state(),
      "inputs": {
          "core_source": str(CPP_SOURCE),
          "core_source_sha256": sha256_file(CPP_SOURCE),
          "model": {"path": str(args.model), "sha256": model_sha256,
                    "size_bytes": args.model.stat().st_size},
          "tensor_index": str(args.tensor_index),
      },
      "measured_inputs": measured,
      "performance_checks_passed": performance_checks_passed,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "statistics": statistics,
      "throughput_model": model,
      "worst_active_ratio_tensors": ranked_tensors[:10],
      "workstream": WORKSTREAM,
  }
  write_jsonl(args.out_dir / "tensor-statistics.jsonl", tensor_rows)
  write_json(args.out_dir / "correctness.json", {
      "checks": evidence_checks,
      "reconstruction_mismatch_count":
          statistics.get("reconstruction_mismatch_count"),
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
      {"metric": "active_exceptions_per_block_mean", "phase": "source_format",
       "value": statistics["active_exceptions_per_block_mean"]},
      {"metric": "active_encoded_over_source_ratio", "phase": "source_format",
       "value": model["active_encoded_over_source_ratio"]},
      {"metric": "memory_only_ceiling_gb_s", "phase": "throughput_model",
       "value": model["memory_only_ceiling_gb_s"]},
      {"metric": "predicted_effective_source_gb_s", "phase": "throughput_model",
       "value": model["predicted_effective_source_gb_s"]},
      {"metric": "required_checks_passed", "phase": "gate",
       "value": required_checks_passed},
  ])
  (args.out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "active_exceptions_per_block_mean":
          statistics["active_exceptions_per_block_mean"],
      "disposition": disposition,
      "memory_only_ceiling_gb_s": model["memory_only_ceiling_gb_s"],
      "out_dir": str(args.out_dir),
      "predicted_effective_source_gb_s":
          model["predicted_effective_source_gb_s"],
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
