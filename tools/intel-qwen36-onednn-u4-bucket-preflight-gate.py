#!/usr/bin/env python3
"""Gate oneDNN's raw U4 core on the real layer-27 1024-token buckets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-onednn-u4-bucket-preflight-gate-v0"
CASE_ID = "prefill_shape_008k"
LAYER = 27
TILE_TOKENS = 1024
TARGET_LAYER_BUDGET_US_PER_64 = 575.33
PLANNING_GB_S = 115.0
ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
DEFAULT_CENSUS = (
    ROOT / "output/prefill-router-shape-census-gate-20260711Tseq639cleanZ")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    f"oneDNN-{ONEDNN_COMMIT}")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-lean")
COMPONENT_SOURCE = ROOT / "engine/tools/onednn_u4_bucket_preflight.cpp"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.warmup <= 0 or args.repeat <= 0 or args.timeout_s <= 0:
    parser.error("warmup, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/onednn-u4-bucket-preflight-gate-{stamp}"
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
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def run(command: list[str], timeout_s: int, cwd: Path = ROOT) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
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


def selected_shape(census_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  result = load_json(census_dir / "result.json")
  if (result.get("required_checks_passed") is not True or
      result.get("aggregate", {}).get("tile_token_count") != TILE_TOKENS):
    raise SystemExit("the locked 1024-token census did not pass")
  rows = [
      row for row in load_jsonl(census_dir / "layer-shapes.jsonl")
      if row.get("case_id") == CASE_ID and row.get("layer") == LAYER
  ]
  if len(rows) != 1:
    raise SystemExit("the locked layer-27 census row is missing")
  return result, rows[0]


def bucket_for(group_m: int) -> int:
  bucket = 8
  while bucket < group_m:
    bucket *= 2
  if bucket > 512:
    raise SystemExit(f"group M {group_m} exceeds the admitted bucket ceiling")
  return bucket


def make_schedule(shape: dict[str, Any]) -> list[dict[str, int]]:
  histogram = shape.get("group_m_histogram")
  if not isinstance(histogram, dict):
    raise SystemExit("layer shape has no group-M histogram")
  counts: Counter[int] = Counter()
  assignments = 0
  active_experts = 0
  for key, value in histogram.items():
    group_m = int(key)
    expert_count = int(value)
    if group_m <= 0 or expert_count <= 0:
      raise SystemExit("group-M histogram contains a non-positive value")
    counts[bucket_for(group_m)] += expert_count
    assignments += group_m * expert_count
    active_experts += expert_count
  if assignments != int(shape.get("assignment_count", -1)):
    raise SystemExit("group-M histogram assignment total changed")
  if active_experts != int(shape.get("active_expert_count", -1)):
    raise SystemExit("group-M histogram expert total changed")
  return [
      {"m": bucket, "experts": counts[bucket]}
      for bucket in sorted(counts)
  ]


def derive_budget(shape: dict[str, Any]) -> dict[str, float | int]:
  full_layer = int(shape["total_layer_source_bytes"])
  gate_up = int(shape["gate_up_unique_weight_bytes"])
  permutation = int(shape["permutation_scatter_stream_bytes"])
  reserved_bytes = full_layer - gate_up + permutation
  whole_window_budget_us = (
      TARGET_LAYER_BUDGET_US_PER_64 * TILE_TOKENS / 64)
  reserved_us = reserved_bytes / (PLANNING_GB_S * 1000.0)
  return {
      "full_layer_source_bytes": full_layer,
      "gate_up_unique_weight_bytes": gate_up,
      "kernel_cap_us": whole_window_budget_us - reserved_us,
      "permutation_scatter_stream_bytes": permutation,
      "planning_gb_s": PLANNING_GB_S,
      "reserved_noncomponent_bytes": reserved_bytes,
      "reserved_noncomponent_us": reserved_us,
      "whole_window_budget_us": whole_window_budget_us,
  }


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
      args.census, args.env_script, args.cxx, args.onednn_source,
      args.onednn_build, COMPONENT_SOURCE,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  census_result, shape = selected_shape(args.census)
  schedule = make_schedule(shape)
  budget = derive_budget(shape)
  source_commit = git_output(args.onednn_source, "rev-parse", "HEAD")
  binary = raw_dir / "onednn-u4-bucket-preflight"
  build_result = shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}",
      str(COMPONENT_SOURCE), f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(binary),
  ], args.env_script, args.timeout_s)
  write_run_logs(raw_dir, "build", build_result)

  command = [
      str(binary), "--actual-assignments", str(shape["assignment_count"]),
      "--expected-experts", str(shape["active_expert_count"]),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--kernel-cap-us", str(budget["kernel_cap_us"]),
  ]
  for bucket in schedule:
    command.extend(["--bucket", f"{bucket['m']}:{bucket['experts']}"])
  probe_result = (
      shell_run(command, args.env_script, args.timeout_s)
      if build_result["returncode"] == 0 else
      {"command": command, "returncode": 125, "stderr": "build failed",
       "stdout": "", "timed_out": False}
  )
  write_run_logs(raw_dir, "probe", probe_result)
  probe = parse_probe(probe_result)

  padded_assignments = sum(
      bucket["m"] * bucket["experts"] for bucket in schedule)
  expected_schedule = [
      {"experts": row["experts"], "m": row["m"]} for row in schedule
  ]
  observed_schedule = [
      {"experts": row.get("experts"), "m": row.get("m")}
      for row in probe.get("buckets", [])
      if isinstance(row, dict)
  ]
  checks = [
      {"name": "locked_census_gate_passed",
       "pass": census_result.get("required_checks_passed") is True},
      {"name": "pinned_onednn_source_commit",
       "pass": source_commit == ONEDNN_COMMIT,
       "observed": source_commit, "required": ONEDNN_COMMIT},
      {"name": "component_build_passed",
       "pass": build_result["returncode"] == 0},
      {"name": "component_execution_completed",
       "pass": probe_result["returncode"] in (0, 2) and bool(probe)},
      {"name": "runtime_onednn_hash_matches_source",
       "pass": probe.get("onednn_version", {}).get("hash") == ONEDNN_COMMIT},
      {"name": "real_layer_schedule_preserved",
       "pass": observed_schedule == expected_schedule and
               probe.get("actual_assignments") == shape["assignment_count"] and
               probe.get("expected_experts") == shape["active_expert_count"]},
      {"name": "power_of_two_padding_accounted",
       "pass": probe.get("padded_assignments") == padded_assignments},
      {"name": "all_buckets_use_jit_gemm",
       "pass": probe.get("implementations_pass") is True},
      {"name": "raw_u4_core_below_component_cap",
       "pass": probe.get("performance_pass") is True and
               float(probe.get("minimum_us", float("inf"))) <=
               float(budget["kernel_cap_us"])},
      {"name": "raw_core_speedup_claims_forbidden", "pass": True},
  ]
  evidence_checks_passed = all(bool(row["pass"]) for row in checks[:-2])
  performance_checks_passed = bool(checks[-2]["pass"])
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  disposition = (
      "admit_one_exact_q4k_onednn_u4_component"
      if required_checks_passed else
      "reject_onednn_u4_bucket_core_below_kill_number")
  result = {
      "budget": budget,
      "case_id": CASE_ID,
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_checks_passed,
      "git": git_state(),
      "layer": LAYER,
      "performance_checks_passed": performance_checks_passed,
      "probe": probe,
      "required_checks_passed": required_checks_passed,
      "schedule": {
          "actual_assignments": shape["assignment_count"],
          "active_experts": shape["active_expert_count"],
          "buckets": schedule,
          "padded_assignments": padded_assignments,
          "padding_ratio": padded_assignments / shape["assignment_count"] - 1,
      },
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "census": str(args.census),
          "component": str(COMPONENT_SOURCE.relative_to(ROOT)),
          "component_sha256": sha256_file(COMPONENT_SOURCE),
          "onednn_build": str(args.onednn_build),
          "onednn_commit": source_commit,
          "onednn_source": str(args.onednn_source),
      },
      "speedup_claims_allowed": False,
      "tile_tokens": TILE_TOKENS,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "applicable": False,
      "next_required_gate": "exact Q4_K scale/min compensation and SwiGLU",
      "reason": "raw U4 matmul support/performance preflight only",
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "standalone raw U4 component preflight",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "raw_u4_minimum_us", "value": probe.get("minimum_us")},
      {"metric": "raw_u4_median_us", "value": probe.get("median_us")},
      {"metric": "component_cap_us", "value": budget["kernel_cap_us"]},
      {"metric": "cap_fraction",
       "value": (float(probe["minimum_us"]) / float(budget["kernel_cap_us"])
                 if "minimum_us" in probe else None)},
      {"metric": "padding_ratio",
       "value": padded_assignments / shape["assignment_count"] - 1},
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
  (out_dir / "summary.md").write_text("\n".join([
      "# oneDNN raw-U4 real-bucket preflight",
      "",
      f"- case/layer: `{CASE_ID}` / `{LAYER}`",
      f"- real assignments / experts: `{shape['assignment_count']}` / "
      f"`{shape['active_expert_count']}`",
      f"- power-of-two padded assignments: `{padded_assignments}` "
      f"(`{padded_assignments / shape['assignment_count'] - 1:.3%}` overhead)",
      f"- target-facing component cap: `{budget['kernel_cap_us']:.3f} us`",
      f"- raw U4 minimum / median: `{minimum_us} / {median_us} us`",
      f"- cap fraction: `{cap_fraction:.3f}`",
      f"- disposition: `{disposition}`",
      "",
      "This gate times seven JIT GEMM calls over the exact layer-27 expert",
      "histogram, with every active expert's 2048x1024 U4 tensor present once.",
      "It proves route headroom only. Q4_K group scales, min compensation,",
      "SwiGLU, teacher-forced correctness, and whole-engine speed remain open.",
      "",
  ]), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "minimum_us": probe.get("minimum_us"),
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
