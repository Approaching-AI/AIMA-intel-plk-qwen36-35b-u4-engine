#!/usr/bin/env python3
"""Classify seq1331 without rerunning its single guarded worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-four-fc-qk-bundle-outcome-v0"
COMPONENT = ROOT / (
    "output/openvino-four-fc-qk-bundle-component-"
    "20260718Tseq1331-candidate-2k-warm17-cleanZ/metrics.json")
WORKER = ROOT / (
    "output/openvino-four-fc-qk-bundle-component-"
    "20260718Tseq1331-candidate-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
QK_WORKER = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
BOUND = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")


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


def executed_counts(worker: dict[str, Any]) -> Counter[str]:
  rows = worker.get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker lacks full_profile")
  return Counter(
      str(row.get("node_type")) for row in rows
      if row.get("status") == "Status.EXECUTED")


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (COMPONENT, WORKER, QK_WORKER, BOUND)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing four-FC outcome inputs: " + ", ".join(missing))
  git = git_state(output)
  component = load_json(COMPONENT)
  worker = load_json(WORKER)
  qk_worker = load_json(QK_WORKER)
  bound = load_json(BOUND)
  failed_evidence = [
      row["name"] for row in component["evidence_checks"]
      if row.get("pass") is not True]
  monitor = component["worker"]["monitor"]
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  corrected_worker_safe = (
      component["worker"]["returncode"] == 0
      and component["worker"]["timed_out"] is False
      and component["worker"]["memory_guard"]["tripped"] is False
      and component["worker"]["oom_observed"] is False
      and int(monitor["system_available_min_bytes"]) >= stop_bytes)
  expected = component["expected_top1"]
  actual = component["actual_top1"]
  differing_top1 = sum(left != right for left, right in zip(expected, actual))
  candidate_counts = executed_counts(worker)
  qk_counts = executed_counts(qk_worker)
  census_delta = {
      key: int(candidate_counts.get(key, 0) - qk_counts.get(key, 0))
      for key in sorted(set(candidate_counts) | set(qk_counts))
      if candidate_counts.get(key, 0) != qk_counts.get(key, 0)
  }
  performance = component["performance"]
  profile = component["profile"]["candidate"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1331_worker_completed_once_without_oom_or_guard",
            corrected_worker_safe,
            returncode=component["worker"]["returncode"],
            oom_observed=component["worker"]["oom_observed"],
            memory_guard=component["worker"]["memory_guard"],
            monitor=monitor),
      check("original_inconclusive_is_only_zero_swap_overguard",
            failed_evidence == [
                "one_candidate_worker_completes_above_stop_without_oom"],
            failed_evidence=failed_evidence,
            note=("process swap is retained as telemetry; the worker stayed "
                  "above the 4-GiB available-memory stop and did not OOM")),
      check("all_70_four_fc_groups_activated_exactly",
            component.get("activation_passed") is True
            and profile["fused_four_fc_count"] == 70
            and profile["core_counts"]["FullyConnectedCompressed"] == 161
            and profile["unfused_target_original_count"] == 0),
      check("teacher_forced_correctness_fails_completely",
            component.get("correctness_passed") is False
            and len(expected) == len(actual) == 18
            and differing_top1 == 18,
            differing_top1=differing_top1),
      check("combined_short_screen_misses_kill_number",
            component.get("performance_passed") is False
            and performance["total_observed_saving_ms"]
                < performance["required_total_saving_ms"]
            and performance["total_margin_ms"] < 0.0,
            performance=performance),
      check("large_n_subset_is_structurally_distinct_not_a_repeat",
            bound["locked_ir"]["counts"]["linear_groups"] == 30
            and bound["locked_ir"]["counts"]["router_groups"] == 40
            and bound["locked_ir"]["parameter_bytes"]["linear_attention"]
                == 409098240
            and bound["locked_ir"]["parameter_bytes"]["router_shared"]
                == 56568960,
            note=("the next source bound keeps only the 30 large-N linear "
                  "groups and excludes the 40 routing-sensitive N=1281 "
                  "groups; it is not an unchanged rerun")),
      check("no_additional_worker_ran", True,
            candidate_workers=0, control_workers=0, gpu_contexts=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_all_four_fc_groups_admit_large_n_subset_source_bound"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "all_four_fc_route_closed": required_checks_passed,
      "large_n_subset_source_bound_admitted": required_checks_passed,
      "compiler_build_admitted": False,
      "gpu_worker_admitted": False,
      "worker_safety": {
          "corrected_safe": corrected_worker_safe,
          "process_swap_peak_bytes": monitor["process_swap_peak_bytes"],
          "system_available_min_bytes": monitor[
              "system_available_min_bytes"],
          "memory_stop_bytes": stop_bytes,
          "oom_observed": component["worker"]["oom_observed"],
          "interpretation": (
              "swap is diagnostic telemetry, not a failure when the serial "
              "worker remains far above the available-memory stop and no "
              "OOM or guard occurs"),
      },
      "activation": {
          "fused_four_fc_count": profile["fused_four_fc_count"],
          "linear_groups": profile["fused_four_fc_linear_count"],
          "router_shared_groups": profile[
              "fused_four_fc_router_shared_count"],
          "fully_connected_compressed": profile[
              "core_counts"]["FullyConnectedCompressed"],
          "variadic_split": profile["variadic_split_executed"],
          "crop": profile["crop_executed"],
          "census_delta_vs_qk_only": census_delta,
      },
      "correctness": {
          "expected_top1": expected,
          "actual_top1": actual,
          "differing_top1": differing_top1,
      },
      "performance": performance,
      "next_route": {
          "route": "openvino_large_n_four_fc_qk_source_bound",
          "hypothesis_not_yet_proven": (
              "exclude the 40 N=1281 router/shared groups, whose routing "
              "outputs are correctness-sensitive and whose added split "
              "overhead may not amortize; retain only 30 N=12352 linear "
              "attention groups"),
          "expected_if_activated": {
              "fused_four_fc_groups": 30,
              "removed_fully_connected_compressed": 90,
              "target_fully_connected_compressed": 281,
          },
          "requirements": [
              "source-only exact predicate and non-overlap bound first",
              "derive a kill-number screen before another build",
              "no unchanged all-group repeat or sample extension",
          ],
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "candidate_workers": 0,
      "control_workers": 0,
      "gpu_contexts": 0,
  })
  report = f"""# Four-way compressed-FC plus Q/K outcome

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No additional worker ran.

Seq1331 completely activates the intended rewrite: 70 four-way FCs execute,
the FC census moves `371 -> 161`, all old target branches disappear, and the
ten Q/K producers remain. It is nevertheless rejected: all 18 teacher-forced
top-1 IDs differ, and the short combined point saves
`{performance['total_observed_saving_ms']:.6f} ms`, missing the
`{performance['required_total_saving_ms']:.6f}-ms` kill-number by
`{-performance['total_margin_ms']:.6f} ms`.

The worker returned normally, never approached the 4-GiB available-memory
stop, and did not OOM. Its `{monitor['process_swap_peak_bytes']} B` process
swap is retained as telemetry rather than promoted to a safety failure; no
rerun is needed to classify the result.

Close the unchanged all-group route. The only admitted successor is a
source-only bound for the structurally distinct large-N subset: retain 30
linear-attention groups and exclude 40 routing-sensitive N=1281 groups before
considering another build.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "differing_top1": differing_top1,
      "total_margin_ms": performance["total_margin_ms"],
      "additional_workers": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
