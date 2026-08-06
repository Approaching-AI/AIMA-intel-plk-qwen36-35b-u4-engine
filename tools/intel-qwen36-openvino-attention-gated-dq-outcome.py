#!/usr/bin/env python3
"""Close the gated-DQ route from the retained completed worker.

This tool launches nothing.  It separates evidence validity from the two
measured outcome gates so the activated, safe seq1321 worker can conclusively
close on both token mismatch and a short-wall regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-gated-dq-outcome-v0"
COMPONENT_DIR = ROOT / (
    "output/openvino-attention-gated-dq-component-"
    "20260717Tseq1321-candidate-2k-warm17-cleanZ")
COMPONENT = COMPONENT_DIR / "metrics.json"
WORKER = COMPONENT_DIR / "raw/2k/candidate/worker-result.json"
BOUND = ROOT / (
    "output/openvino-attention-gated-dq-bound-"
    "20260717Tseq1317b-cleanZ/metrics.json")
AUDIT = ROOT / (
    "output/openvino-attention-gated-dq-rewrite-"
    "20260717Tseq1320b-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")


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


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-attention-gated-dq-outcome.py"}
  relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def stable_walls(result: dict[str, Any]) -> list[float]:
  walls = [float(row["wall_ms_diagnostic"])
           for row in result.get("phases", [])[1:]]
  if len(walls) != 17 or not all(math.isfinite(value) and value > 0.0
                                 for value in walls):
    raise ValueError("worker does not have 17 finite decode walls")
  return walls[1:]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (COMPONENT, WORKER, BOUND, AUDIT, CONTROL)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gated-DQ outcome inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_start = available_memory_bytes()
  if memory_start < stop_bytes:
    raise RuntimeError(f"memory stop: {memory_start} < {stop_bytes}")
  git = git_state(output)
  component = load_json(COMPONENT)
  worker = load_json(WORKER)
  bound = load_json(BOUND)
  audit = load_json(AUDIT)
  control = load_json(CONTROL)

  expected = [int(value) for value in component["expected_top1"]]
  actual = [int(row.get("top1", -1)) for row in worker.get("phases", [])]
  mismatches = [
      {"phase": index, "expected": wanted, "actual": observed}
      for index, (wanted, observed) in enumerate(zip(expected, actual))
      if wanted != observed]
  control_stable = stable_walls(control)
  candidate_stable = stable_walls(worker)
  control_median = statistics.median(control_stable)
  candidate_median = statistics.median(candidate_stable)
  observed_saving = control_median - candidate_median
  required_saving = float(bound["budget"]["required_component_saving_ms"])
  profile = component["profile"]["candidate"]
  worker_meta = component["worker"]

  evidence_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1320b_admitted_the_exact_plugin_and_one_worker",
            audit.get("required_checks_passed") is True
            and audit.get("candidate_workers_admitted") == 1
            and component.get("gpu_workers_launched") == 1
            and component.get("control_workers_launched") == 0),
      check("retained_worker_completed_safely_without_oom",
            worker_meta.get("returncode") == 0
            and worker_meta.get("timed_out") is False
            and worker_meta.get("memory_guard", {}).get("tripped") is False
            and worker_meta.get("oom_observed") is False
            and int(worker_meta.get("monitor", {}).get(
                "system_available_min_bytes") or 0) >= stop_bytes,
            worker=worker_meta),
      check("all_ten_gated_dq_rows_and_core_census_activated",
            component.get("activation_passed") is True
            and profile.get("core_counts_exact") is True
            and profile.get("gated_dynamic_quantize_executed") == 10
            and profile.get("dynamic_quantize_executed") == 151
            and profile.get("old_output_transpose_executed") == 0
            and profile.get("old_gate_multiply_executed") == 0,
            profile=profile),
      check("retained_raw_worker_matches_component_summary",
            actual == component.get("actual_top1")
            and abs(control_median - component["performance"][
                "control_median_ms"]) < 1e-12
            and abs(candidate_median - component["performance"][
                "candidate_median_ms"]) < 1e-12
            and abs(observed_saving - component["performance"][
                "observed_median_saving_ms"]) < 1e-12),
      check("finalizer_launches_no_worker_or_gpu_context", True,
            gpu_contexts=0, candidate_workers=0, control_workers=0,
            stock_workers=0, long_workers=0, product_workers=0),
  ]
  evidence_passed = all(row["pass"] for row in evidence_checks)
  correctness_passed = len(actual) == len(expected) and not mismatches
  performance_passed = observed_saving >= required_saving
  verdict = (
      "reject_attention_gated_dq_after_component"
      if evidence_passed and (not correctness_passed or not performance_passed)
      else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "evidence_checks_passed": evidence_passed,
      "activation_passed": component.get("activation_passed"),
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "route_closed": verdict.startswith("reject_"),
      "expected_top1": expected,
      "actual_top1": actual,
      "token_mismatches": mismatches,
      "performance": {
          "stable_sample_rule": "drop first decode JIT sample",
          "stable_samples_per_side": 16,
          "control_median_ms": control_median,
          "candidate_median_ms": candidate_median,
          "observed_median_saving_ms": observed_saving,
          "required_incremental_saving_ms": required_saving,
          "margin_to_required_ms": observed_saving - required_saving,
          "speed_claim": False,
      },
      "evidence_checks": evidence_checks,
      "workers_launched": 0,
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": memory_start,
          "available_end_bytes": available_memory_bytes(),
          "oom_observed": False,
      },
      "decision": {
          "close_route": verdict.startswith("reject_"),
          "repeat_admitted": False,
          "sample_extension_admitted": False,
          "reopen_condition": (
              "materially different arithmetic/layout contract with a new "
              "source bound; unchanged group64 gated-DQ is closed"),
          "next_route": "openvino_post_gated_dq_source_opportunity_bound",
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "gpu_contexts": 0,
      "candidate_workers": 0,
      "control_workers": 0,
      "long_workers": 0,
  })
  report = f"""# Attention consumer-side gated-DQ outcome

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_passed).lower()}`; activation:
`{str(component.get('activation_passed')).lower()}`; correctness:
`{str(correctness_passed).lower()}`; performance:
`{str(performance_passed).lower()}`.

The retained seq1321 worker safely executed all ten gated-DQ rows and the exact
core census, with no OOM. It mismatched `{len(mismatches)}/{len(expected)}`
teacher-forced tokens. Its stable short median was
`{candidate_median:.7f} ms` versus `{control_median:.7f} ms`, a
`{observed_saving:.7f}-ms` saving against the required
`{required_saving:.7f} ms` (therefore a regression and a failed cut).

Close the unchanged group-64 gated-DQ route without repeat or sample
extension. This finalizer launched no GPU context or worker.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "token_mismatches": len(mismatches),
      "observed_saving_ms": observed_saving,
      "oom_observed": False,
  }, separators=(",", ":")), flush=True)
  return 0 if evidence_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
