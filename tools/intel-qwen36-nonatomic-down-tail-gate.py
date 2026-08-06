#!/usr/bin/env python3
"""Gate a non-atomic Q6 down-to-tail component proof."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-nonatomic-down-tail-gate-v0"
DEFAULT_PROBE = (
    ROOT
    / "output/gpu-selected-down-q6-nonatomic-down-tail-probe-20260707Tseq80Z-l7/probe.json"
)


def utc_stamp() -> str:
  return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"expected object JSON: {path}")
  return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nested(obj: dict[str, Any], *keys: str) -> Any:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return current


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  return nested(obj, *keys) is True


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  value = nested(obj, *keys)
  return float(value) if isinstance(value, (int, float)) else None


def rel(path: Path) -> str:
  resolved = path.resolve()
  try:
    return str(resolved.relative_to(ROOT))
  except ValueError:
    return str(resolved)


def write_metrics(path: Path, rows: list[tuple[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({"phase": "nonatomic_down_tail_gate", "metric": metric, "value": value}) + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--max-ratio-vs-combined-down", type=float, default=1.15)
  return parser.parse_args()


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  metrics = payload["metrics"]
  checks = payload["checks"]
  failed = [item["name"] for item in checks if not item["pass"]]
  lines = [
      "# Non-Atomic Down-Tail Gate",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- source probe: `{payload['source_probe']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- tensor type: `{metrics['tensor_type']}`",
      f"- component kernel: `{metrics['kernel']}`",
      f"- combined down min us: `{metrics['combined_down_min_us']}`",
      f"- contribution min us: `{metrics['contribution_min_us']}`",
      f"- reduce min us: `{metrics['reduce_min_us']}`",
      f"- shell sum min us: `{metrics['shell_sum_min_us']}`",
      f"- shell/combined-down ratio: `{metrics['ratio_vs_combined_down']}`",
      f"- contributor global work items: `{metrics['contribution_global_work_items']}`",
      f"- reduce global work items: `{metrics['reduce_global_work_items']}`",
      f"- max layer-vs-oracle diff: `{metrics['layer_vs_oracle_max_abs_diff']}`",
      f"- failed checks: `{failed}`",
      "",
      "This is component evidence only. It preserves the Q6 selected/shared",
      "down contributor parallelism and replaces the closed atomic tail path with",
      "a contribution scratch plus one non-atomic hidden-row reduce. It is not",
      "decode throughput evidence and does not allow speedup claims.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir or ROOT / f"output/nonatomic-down-tail-gate-{utc_stamp()}"
  out_dir.mkdir(parents=True, exist_ok=False)

  payload = load_json(args.probe)
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe.get("timings"), dict) else {}
  comparisons = (
      probe.get("comparisons", {}).get("q6_nonatomic_down_tail", {})
      if isinstance(probe.get("comparisons"), dict)
      else {}
  )

  rows_per_expert = nested_number(probe, "rows_per_expert")
  expected_contrib_global = int(rows_per_expert * 9) if rows_per_expert is not None else None
  expected_reduce_global = int(rows_per_expert) if rows_per_expert is not None else None
  ratio = nested_number(probe, "candidate_q6_nonatomic_down_tail_ratio_vs_combined_down")
  metrics = {
      "tensor_type": probe.get("tensor_type"),
      "kernel": probe.get("candidate_q6_nonatomic_down_tail_kernel"),
      "measured": probe.get("candidate_q6_nonatomic_down_tail_measured"),
      "preserves_contributor_parallelism": probe.get(
          "candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism"
      ),
      "boundary_reopen_candidate": probe.get(
          "candidate_q6_nonatomic_down_tail_boundary_reopen_candidate"
      ),
      "ratio_vs_combined_down": ratio,
      "max_ratio_vs_combined_down": args.max_ratio_vs_combined_down,
      "combined_down_min_us": nested_number(
          timings, "candidate_q6_selected_shared_combined_kernel_min_us"
      ),
      "contribution_min_us": nested_number(
          timings, "candidate_q6_nonatomic_down_tail_contribution_kernel_min_us"
      ),
      "reduce_min_us": nested_number(
          timings, "candidate_q6_nonatomic_down_tail_reduce_kernel_min_us"
      ),
      "shell_sum_min_us": nested_number(
          timings, "candidate_q6_nonatomic_down_tail_shell_sum_min_us"
      ),
      "contribution_global_work_items": nested_number(
          timings, "candidate_q6_nonatomic_down_tail_contribution_global_work_items"
      ),
      "reduce_global_work_items": nested_number(
          timings, "candidate_q6_nonatomic_down_tail_reduce_global_work_items"
      ),
      "expected_contribution_global_work_items": expected_contrib_global,
      "expected_reduce_global_work_items": expected_reduce_global,
      "contrib_vs_cpu_max_abs_diff": nested_number(
          comparisons, "contrib_vs_cpu", "max_abs_diff"
      ),
      "layer_vs_oracle_max_abs_diff": nested_number(
          comparisons, "layer_vs_oracle", "max_abs_diff"
      ),
  }
  checks = [
      {"name": "probe_required_checks_passed", "pass": payload.get("required_checks_passed") is True},
      {"name": "probe_stdout_required_checks_passed", "pass": probe.get("required_checks_passed") is True},
      {"name": "q6_tensor_measured", "pass": probe.get("tensor_type") == "Q6_K"},
      {
          "name": "nonatomic_tail_measured",
          "pass": probe.get("candidate_q6_nonatomic_down_tail_measured") is True,
      },
      {
          "name": "contributor_parallelism_preserved",
          "pass": probe.get(
              "candidate_q6_nonatomic_down_tail_preserves_contributor_parallelism"
          )
          is True,
      },
      {
          "name": "component_matches_cpu_and_oracle",
          "pass": nested_bool(
              probe,
              "checks",
              "candidate_q6_nonatomic_down_tail_matches_current_and_oracle",
          ),
      },
      {
          "name": "component_timing_positive",
          "pass": nested_bool(
              probe,
              "checks",
              "candidate_q6_nonatomic_down_tail_event_timing_positive",
          ),
      },
      {
          "name": "contribution_global_is_rows_per_expert_times_9",
          "pass": expected_contrib_global is not None
          and metrics["contribution_global_work_items"] == expected_contrib_global,
      },
      {
          "name": "reduce_global_is_rows_per_expert",
          "pass": expected_reduce_global is not None
          and metrics["reduce_global_work_items"] == expected_reduce_global,
      },
      {
          "name": "ratio_within_boundary_threshold",
          "pass": ratio is not None and ratio <= args.max_ratio_vs_combined_down,
      },
      {
          "name": "boundary_reopen_candidate",
          "pass": probe.get("candidate_q6_nonatomic_down_tail_boundary_reopen_candidate")
          is True,
      },
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  disposition = (
      "component_reopen_candidate_decode_wiring"
      if required_checks_passed
      else "rejected_or_incomplete_component_proof"
  )
  result = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": iso_now(),
      "source_probe": rel(args.probe),
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "metrics": metrics,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": result["created_at"],
      "tool": "tools/intel-qwen36-nonatomic-down-tail-gate.py",
      "artifact": rel(out_dir),
      "source_probe": rel(args.probe),
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  write_json(out_dir / "gate.json", result)
  write_json(out_dir / "manifest.json", manifest)
  write_json(
      out_dir / "correctness.json",
      {
          "schema_version": SCHEMA_VERSION,
          "workstream": WORKSTREAM,
          "checks": checks,
          "required_checks_passed": required_checks_passed,
          "speedup_claims_allowed": False,
      },
  )
  write_metrics(out_dir / "metrics.jsonl", sorted(metrics.items()))
  write_summary(out_dir / "summary.md", result)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
