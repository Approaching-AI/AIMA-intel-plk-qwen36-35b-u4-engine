#!/usr/bin/env python3
"""Measure exact-context cost of the current packed Level Zero decode backend.

This is a kill-number diagnostic. It reserves the full 512-token output
capacity but may execute fewer sample tokens. Full-attention KV history is
zero-initialized, so the row measures real context-dependent device work and
resident memory without claiming semantic correctness or product speed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ACCEPTANCE = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
TOKEN_FILE = (
    ROOT
    / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z"
    / "token-input/fresh_code_03.tokens.u32"
)
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CORE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
HOST_SUBMIT_BUDGET_MS = 0.100
MIN_INFERENCE_SAMPLES = 20


def parse_buckets(value: str) -> tuple[int, ...]:
  buckets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
  if not buckets or any(bucket not in CORE_BUCKETS for bucket in buckets):
    raise argparse.ArgumentTypeError("buckets must be a non-empty core subset")
  if len(set(buckets)) != len(buckets):
    raise argparse.ArgumentTypeError("buckets must not repeat")
  return buckets


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--buckets", type=parse_buckets, default=CORE_BUCKETS)
  parser.add_argument("--sample-tokens", type=int, default=1)
  parser.add_argument("--profile-buckets", type=parse_buckets, default=())
  parser.add_argument("--int8-block32-kv-gqa", action="store_true")
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def run(
    command: list[str], timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout, env=environment)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def summary(payload: dict[str, Any]) -> str:
  lines = [
      "# Packed-token exact-context gap gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- absolute decode guard passed: `{str(payload['absolute_decode_guard_pass']).lower()}`",
      f"- sample tokens per bucket: `{payload['sample_tokens']}`",
      f"- full KV dtype: `{payload['full_kv_dtype']}`",
      "- reserved output capacity: `512`",
      f"- host-submit 95% UCB budget: `{HOST_SUBMIT_BUDGET_MS:.3f} ms`",
      "- state semantics: `zero_initialized_performance_only`",
      "- correctness / speedup claim: `not applicable / forbidden`",
      "",
      "| input | wall median / UCB ms | host median / UCB / max ms | native tok/s | target tok/s | guard | decode s | resident GiB |",
      "|---:|---:|---:|---:|---:|:---:|---:|---:|",
  ]
  for row in payload["rows"]:
    inference = row.get("decode_target_inference") or {}
    host_inference = row.get("host_submit_inference") or {}
    upper = inference.get("upper_confidence_bound_ms")
    upper_text = f"{upper:.3f}" if isinstance(upper, (int, float)) else "n/a"
    host_median = host_inference.get("point_estimate_ms")
    host_upper = host_inference.get("upper_confidence_bound_ms")
    host_max = row.get("host_submit_ms_max")
    host_text = " / ".join(
        f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
        for value in (host_median, host_upper, host_max))
    guard = "pass" if inference.get("rate_pass") is True else "fail"
    decode_s = (
        row["wall_ms_total"] / 1000.0
        if payload["sample_tokens"] == 512
        else row["output_512_projected_wall_s"])
    lines.append(
        f"| {row['context_tokens']} | {row['wall_ms_median']:.3f} / {upper_text} | "
        f"{host_text} | "
        f"{row['wall_tokens_s']:.3f} | {row['decode_target_tokens_s']:.2f} | "
        f"{guard} | "
        f"{decode_s:.3f} | "
        f"{row['resident_state_gib']:.3f} |")
  if payload["sample_tokens"] == 512:
    lines += [
        "",
        "Decode time is the measured sum of all 512 wall samples. The row",
        "remains zero-state capacity evidence, not semantic product evidence.",
        "",
    ]
  else:
    lines += [
        "",
        "Decode time is a constant-cost projection from the sampled row, not a",
        "measured output-512 claim. A full row is admissible only when this",
        "kill-number says it can materially approach the target.",
        "",
    ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.sample_tokens < 1 or args.sample_tokens > 512:
    raise SystemExit("--sample-tokens must be in 1..512")
  if any(bucket not in args.buckets for bucket in args.profile_buckets):
    raise SystemExit("--profile-buckets must be a subset of --buckets")
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  generated_dir = out_dir / "generated"
  raw_dir.mkdir(parents=True, exist_ok=False)
  generated_dir.mkdir(parents=True, exist_ok=True)
  git = git_state(out_dir)
  acceptance = json.loads(ACCEPTANCE.read_text())
  decode_targets = acceptance["bootstrap_targets"]["decode_tokens_s"]

  module = generated_dir / "iq36_q4x8_all.bin"
  compile_command = [
      "ocloc", "compile", "-file", str(ROOT / "engine/gpu/opencl/q4x8_matvec.cl"),
      "-device", "0xb080", "-output", "iq36_q4x8_all",
      "-out_dir", str(generated_dir), "-output_no_suffix", "--format", "zebin",
      "-options", "-cl-std=CL3.0 -D IQ36_USE_INTEGER_DOT=1", "-q",
  ]
  compile_run = run(compile_command, 300)
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure_run = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target",
      "iq36-packed-token-level-zero-backend-smoke", "-j8",
  ]
  build_run = run(build_command, 600)
  write_json(raw_dir / "build.json", {
      "compile": {"command": compile_command, "returncode": compile_run.returncode,
                  "stdout": compile_run.stdout, "stderr": compile_run.stderr},
      "configure": {"command": configure_command, "returncode": configure_run.returncode,
                    "stdout": configure_run.stdout, "stderr": configure_run.stderr},
      "build": {"command": build_command, "returncode": build_run.returncode,
                "stdout": build_run.stdout, "stderr": build_run.stderr},
  })
  executable = BUILD_DIR / "iq36-packed-token-level-zero-backend-smoke"
  build_ok = all((
      compile_run.returncode == 0, configure_run.returncode == 0,
      build_run.returncode == 0, module.is_file(), executable.is_file(),
  ))

  rows: list[dict[str, Any]] = []
  run_records: list[dict[str, Any]] = []
  if build_ok:
    candidate_env = os.environ.copy()
    if args.int8_block32_kv_gqa:
      candidate_env["IQ36_INT8_BLOCK32_KV_GQA"] = "1"
    for bucket in args.buckets:
      command = [
          str(executable), str(MODEL), str(module), str(TOKEN_FILE),
          str(bucket), str(args.sample_tokens),
      ]
      environment = os.environ.copy()
      environment.update(candidate_env)
      if bucket in args.profile_buckets:
        environment["IQ36_PROFILE_KERNELS"] = "1"
      completed = run(command, args.timeout_s, environment)
      (raw_dir / f"{bucket}.stdout").write_text(completed.stdout)
      (raw_dir / f"{bucket}.stderr").write_text(completed.stderr)
      result = parse_last_json(completed.stdout)
      target = float(decode_targets[str(bucket)])
      wall_tps = result.get("wall_tokens_s")
      wall_samples = result.get("wall_ms_samples")
      host_samples = result.get("host_submit_ms_samples")
      target_inference = (
          latency_cap_inference(
              wall_samples, cap=1000.0 / target,
              min_samples=MIN_INFERENCE_SAMPLES)
          if isinstance(wall_samples, list) and wall_samples else None)
      host_submit_inference = (
          latency_cap_inference(
              host_samples, cap=HOST_SUBMIT_BUDGET_MS,
              min_samples=MIN_INFERENCE_SAMPLES)
          if isinstance(host_samples, list) and host_samples else None)
      row = {
          **result,
          "command": command,
          "decode_target_inference": target_inference,
          "decode_target_ratio": (
              float(wall_tps) / target
              if isinstance(wall_tps, (int, float)) else None),
          "decode_target_tokens_s": target,
          "host_submit_inference": host_submit_inference,
          "profile_enabled": bucket in args.profile_buckets,
          "resident_state_gib": (
              float(result["resident_state_bytes"]) / 1024**3
              if isinstance(result.get("resident_state_bytes"), int) else None),
          "returncode": completed.returncode,
      }
      rows.append(row)
      run_records.append({
          "bucket": bucket, "command": command,
          "returncode": completed.returncode,
          "stdout_size_bytes": len(completed.stdout.encode()),
          "stderr_tail": completed.stderr[-4000:],
      })

  exact_buckets = [row.get("context_tokens") for row in rows] == list(args.buckets)
  expected_kv_dtype = (
      "int8_block32_fp16_scale_f32_hot8192"
      if args.int8_block32_kv_gqa else "f32")
  row_checks = all(
      row.get("returncode") == 0
      and row.get("required_checks_passed") is True
      and row.get("correctness_applicable") is False
      and row.get("speedup_claims_allowed") is False
      and row.get("reserved_output_tokens") == 512
      and row.get("sample_tokens") == args.sample_tokens
      and row.get("state_semantics") == "zero_initialized_performance_only"
      and row.get("full_kv_dtype") == expected_kv_dtype
      and isinstance(row.get("wall_tokens_s"), (int, float))
      and math.isfinite(float(row["wall_tokens_s"]))
      and float(row["wall_tokens_s"]) > 0.0
      for row in rows
  ) and len(rows) == len(args.buckets)
  rate_inference_available = all(
      isinstance(row.get("decode_target_inference"), dict)
      and row["decode_target_inference"].get("sample_count_pass") is True
      for row in rows)
  absolute_decode_guard_pass = rate_inference_available and all(
      row["decode_target_inference"].get("rate_pass") is True for row in rows)
  host_submit_guard_applicable = args.sample_tokens >= MIN_INFERENCE_SAMPLES
  host_submit_guard_pass = (
      all(
          isinstance(row.get("host_submit_inference"), dict)
          and row["host_submit_inference"].get("sample_count_pass") is True
          and row["host_submit_inference"].get("rate_pass") is True
          for row in rows)
      if host_submit_guard_applicable else None)
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "target_module_and_smoke_build", "pass": build_ok},
      {"name": "exact_core_bucket_subset", "pass": exact_buckets},
      {"name": "all_context_rows_executed", "pass": row_checks},
      {"name": "host_submit_budget_inference", "pass": (
          host_submit_guard_pass if host_submit_guard_applicable else True),
       "applicable": host_submit_guard_applicable,
       "budget_ms": HOST_SUBMIT_BUDGET_MS,
       "method": "one_sided_95pct_bootstrap_median_ucb",
       "minimum_sample_count": MIN_INFERENCE_SAMPLES},
      {"name": "full_kv_dtype_selected", "pass": all(
          row.get("full_kv_dtype") == expected_kv_dtype for row in rows),
       "expected": expected_kv_dtype},
      {"name": "semantic_correctness_not_claimed", "pass": all(
          row.get("correctness_applicable") is False for row in rows)},
      {"name": "product_speedup_not_claimed", "pass": all(
          row.get("speedup_claims_allowed") is False for row in rows)},
  ]
  required = all(bool(check["pass"]) for check in checks)
  ratios = [float(row["decode_target_ratio"]) for row in rows
            if isinstance(row.get("decode_target_ratio"), (int, float))]
  payload = {
      "buckets": list(args.buckets),
      "absolute_decode_guard_pass": absolute_decode_guard_pass,
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "full_kv_dtype": expected_kv_dtype,
      "host_submit_budget_ms": HOST_SUBMIT_BUDGET_MS,
      "host_submit_guard_applicable": host_submit_guard_applicable,
      "host_submit_guard_pass": host_submit_guard_pass,
      "profile_buckets": list(args.profile_buckets),
      "required_checks_passed": required,
      "rate_inference_available": rate_inference_available,
      "route_label": (
          "candidate" if required and absolute_decode_guard_pass
          else "diagnostic" if required else "rejected"),
      "rows": rows,
      "sample_tokens": args.sample_tokens,
      "schema_version": "intel-qwen36-packed-token-context-gap-v2",
      "speedup_claims_allowed": False,
      "target_ratio_max": max(ratios) if ratios else None,
      "target_ratio_min": min(ratios) if ratios else None,
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"], "git": git,
      "required_checks_passed": required, "route_label": payload["route_label"],
      "schema_version": payload["schema_version"],
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": payload["workstream"],
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks, "correctness_applicable": False,
      "required_checks_passed": required,
      "state_semantics": "zero_initialized_performance_only",
      "speedup_claims_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  adjacent = []
  for previous, current in zip(rows, rows[1:]):
    adjacent.append({
        "from_bucket": previous["context_tokens"],
        "to_bucket": current["context_tokens"],
        "wall_ms_ratio": current["wall_ms_median"] / previous["wall_ms_median"],
    })
  write_json(out_dir / "smoothness.json", {
      "adjacent_context_cost": adjacent,
      "applicable": True,
      "notes": "performance-only zero-state slope; not product smoothness",
  })
  write_json(raw_dir / "runs.json", run_records)
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
      "row_count": len(rows),
      "target_ratio_min": payload["target_ratio_min"],
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
