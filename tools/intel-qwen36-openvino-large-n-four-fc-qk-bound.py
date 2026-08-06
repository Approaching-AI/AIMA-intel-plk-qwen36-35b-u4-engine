#!/usr/bin/env python3
"""Reject or admit the 30-group large-N four-FC subset without a worker."""

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
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-large-n-four-fc-qk-bound-v0"
OUTCOME = ROOT / (
    "output/openvino-four-fc-qk-bundle-outcome-"
    "20260718Tseq1332-cleanZ/metrics.json")
BOUND = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
QK_WORKER = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
ALL_FOUR_WORKER = ROOT / (
    "output/openvino-four-fc-qk-bundle-component-"
    "20260718Tseq1331-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
PROVIDER_BOUND = ROOT / (
    "output/openvino-attention-output-gate-fusion-bound-"
    "20260717Tseq1311c-cleanZ/metrics.json")
LINEAR_SUFFIXES = (
    "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "linear_attn.in_proj_z/ov_ext::linear/MatMul",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-four-fc-qk-bundle-component.py",
      "tools/intel-qwen36-openvino-four-fc-qk-bundle-outcome.py",
      "tools/intel-qwen36-openvino-large-n-four-fc-qk-bound.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def profile_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
  rows = worker.get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker lacks full_profile")
  return [row for row in rows if row.get("status") == "Status.EXECUTED"]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (OUTCOME, BOUND, QK_WORKER, ALL_FOUR_WORKER, PROVIDER_BOUND)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing large-N bound inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError("memory stop tripped before source bound")
  git = git_state(output)
  outcome = load_json(OUTCOME)
  bound = load_json(BOUND)
  qk = load_json(QK_WORKER)
  all_four = load_json(ALL_FOUR_WORKER)
  provider = load_json(PROVIDER_BOUND)
  qk_rows = profile_rows(qk)
  all_four_rows = profile_rows(all_four)

  original_linear = [
      row for row in qk_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and any(str(row.get("node_name", "")).endswith(suffix)
              for suffix in LINEAR_SUFFIXES)]
  fused_linear = [
      row for row in all_four_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and "_fused_4FCs" in str(row.get("node_name", ""))
      and ".linear_attn." in str(row.get("node_name", ""))]
  original_raw_ms = sum(
      float(row.get("real_time_us") or 0.0) for row in original_linear) / 1000.0
  fused_raw_ms = sum(
      float(row.get("real_time_us") or 0.0) for row in fused_linear) / 1000.0
  raw_event_delta_ms = original_raw_ms - fused_raw_ms
  removed_dispatches = len(original_linear) - len(fused_linear)
  enqueue_us = float(provider["budget"]["max_enqueue_us_per_dispatch"])
  set_args_us = float(
      provider["budget"]["set_arguments_per_boundary_dispatch_us"])
  dispatch_ceiling_ms = removed_dispatches * (enqueue_us + set_args_us) / 1000.0
  favorable_incremental_ceiling_ms = raw_event_delta_ms + dispatch_ceiling_ms
  required_incremental_ms = (
      float(bound["budget"]["current_kill_number_ms"])
      - float(bound["budget"]["seq1327_qk_observed_component_point_ms"]))
  shortfall_ms = required_incremental_ms - favorable_incremental_ceiling_ms
  rms_igc_union_ms = (
      float(bound["budget"]["complete_registered_rms_bucket_ms"])
      + float(bound["budget"]["seq1301_igc_unconfirmed_median_point_ms"]))
  expanded_ceiling_ms = favorable_incremental_ceiling_ms + rms_igc_union_ms
  expanded_shortfall_ms = required_incremental_ms - expanded_ceiling_ms
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1332_admits_only_structurally_distinct_source_bound",
            outcome.get("required_checks_passed") is True
            and outcome.get("all_four_fc_route_closed") is True
            and outcome.get("large_n_subset_source_bound_admitted") is True
            and outcome.get("compiler_build_admitted") is False),
      check("large_n_profile_rows_are_exact",
            len(original_linear) == 120 and len(fused_linear) == 30
            and removed_dispatches == 90,
            original_rows=len(original_linear), fused_rows=len(fused_linear),
            removed_dispatches=removed_dispatches),
      check("large_n_raw_event_delta_is_exact_but_nonadditive",
            math.isclose(original_raw_ms, 8.243, abs_tol=1e-12)
            and math.isclose(fused_raw_ms, 7.849, abs_tol=1e-12)
            and math.isclose(raw_event_delta_ms, 0.394, abs_tol=1e-12),
            original_raw_ms=original_raw_ms,
            fused_raw_ms=fused_raw_ms,
            raw_event_delta_ms=raw_event_delta_ms,
            raw_profile_is_savings_evidence=False),
      check("favorable_raw_plus_provider_ceiling_misses_residual",
            favorable_incremental_ceiling_ms < required_incremental_ms
            and shortfall_ms > 0.0,
            dispatch_ceiling_ms=dispatch_ceiling_ms,
            favorable_incremental_ceiling_ms=favorable_incremental_ceiling_ms,
            required_incremental_ms=required_incremental_ms,
            shortfall_ms=shortfall_ms),
      check("even_rms_and_igc_favorable_union_still_misses",
            expanded_ceiling_ms < required_incremental_ms
            and expanded_shortfall_ms > 0.0,
            rms_igc_union_ms=rms_igc_union_ms,
            expanded_ceiling_ms=expanded_ceiling_ms,
            expanded_shortfall_ms=expanded_shortfall_ms),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            available_start_bytes=available_start, stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_large_n_four_fc_before_build_admit_router_isolated_shared_triple_bound"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": False,
      "plugin_build_admitted": False,
      "gpu_worker_admitted": False,
      "large_n_route_closed": required_checks_passed,
      "budget": {
          "original_linear_fcp_raw_ms": original_raw_ms,
          "fused_linear_fcp_raw_ms": fused_raw_ms,
          "nonadditive_raw_event_delta_ms": raw_event_delta_ms,
          "removed_dispatches": removed_dispatches,
          "favorable_dispatch_ceiling_ms": dispatch_ceiling_ms,
          "favorable_incremental_ceiling_ms": favorable_incremental_ceiling_ms,
          "required_incremental_ms_after_retained_qk": required_incremental_ms,
          "large_n_shortfall_ms": shortfall_ms,
          "favorable_rms_plus_igc_union_ms": rms_igc_union_ms,
          "expanded_favorable_ceiling_ms": expanded_ceiling_ms,
          "expanded_shortfall_ms": expanded_shortfall_ms,
          "interpretation": (
              "raw event times are non-additive and are granted only as a "
              "favorable source screen; even adding provider overhead, the "
              "full RMS bucket, and the unconfirmed IGC median cannot fund "
              "another large-N build"),
      },
      "next_route": {
          "route": "openvino_router_isolated_shared_triple_source_bound",
          "reason": (
              "the three shared-expert branches account for 6.644 ms of the "
              "7.120-ms original router/shared FC event point, while the "
              "256-wide router gate is only 0.476 ms and directly controls "
              "expert selection; bound separating it before source edits"),
          "compiler_or_worker_admitted": False,
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Large-N four-FC plus retained-Q/K bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or worker ran.

The exact 30-group large-N subset removes 90 FC rows, but its registered
non-additive FC event point changes only `{original_raw_ms:.3f} ->
{fused_raw_ms:.3f} ms`. Even granting that full `{raw_event_delta_ms:.3f}-ms`
delta plus `{dispatch_ceiling_ms:.6f} ms` of provider overhead yields only
`{favorable_incremental_ceiling_ms:.6f} ms`, short of the
`{required_incremental_ms:.6f}-ms` post-Q/K residual by
`{shortfall_ms:.6f} ms`. Granting RMS and IGC still leaves
`{expanded_shortfall_ms:.6f} ms`.

Reject before another build. Next bound a different semantic cut: preserve the
256-wide router gate and fuse only the three shared-expert projections, which
carry 6.644 of the original 7.120-ms router/shared FC event point. OOM
observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "large_n_shortfall_ms": shortfall_ms,
      "expanded_shortfall_ms": expanded_shortfall_ms,
      "workers": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
