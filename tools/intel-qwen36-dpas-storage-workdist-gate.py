#!/usr/bin/env python3
"""Gate the DPAS selected gate/up storage/work-distribution branch.

This is route-selection arithmetic over the target DPAS local-Q8 workgroup
sharing probe. It cannot set the speed frontier; it only decides whether the
DPAS branch earned another implementation round after seq78.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-dpas-storage-workdist-gate-v0"
DEFAULT_PROBE = ROOT / "output/gpu-dpas-q4-exact-gate-20260707Tlocalq8-workdist-r2/probe.json"
DEFAULT_BUDGET = ROOT / "output/dpas-prefill-budget-20260706TbudgetZ/budget.json"
DEFAULT_TILING_GATE = ROOT / "output/dpas-gateup-tiling-budget-20260706Tseq52Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/dpas-storage-workdist-gate-20260707Tseq79Z"


def load_json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return payload


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def unwrap_probe(payload: dict[str, Any]) -> dict[str, Any]:
  probe = payload.get("probe")
  return probe if isinstance(probe, dict) else payload


def num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def best_result(results: Any) -> dict[str, Any]:
  if not isinstance(results, list):
    return {}
  best: dict[str, Any] = {}
  for row in results:
    if not isinstance(row, dict):
      continue
    value = num(row.get("kernel_min_us"))
    if value <= 0.0:
      continue
    if not best or value < num(best.get("kernel_min_us")):
      best = row
  return best


def check(label: str, passed: bool, detail: dict[str, Any] | None = None) -> dict[str, Any]:
  row: dict[str, Any] = {"label": label, "pass": bool(passed)}
  if detail is not None:
    row["detail"] = detail
  return row


def compute(args: argparse.Namespace) -> dict[str, Any]:
  probe_wrapper = load_json(args.probe)
  probe = unwrap_probe(probe_wrapper)
  budget = load_json(args.budget)
  tiling_gate = load_json(args.tiling_gate)

  baseline_us = num(probe.get("selected_swiglu_fusion_best_kernel_min_us"))
  localq8_us = num(probe.get("selected_swiglu_fusion_localq8_best_kernel_min_us"))
  baseline_best = best_result(probe.get("selected_swiglu_fusion_results"))
  localq8_best = best_result(probe.get("selected_swiglu_fusion_localq8_results"))
  target_layer_us = num(budget.get("target_layer_budget_us"))
  seq50_selected_us = num(budget.get("selected_swiglu_fusion_us"))
  required_from_seq50 = num(
      budget.get("selected_swiglu_required_speedup_to_fit_budget")
  )
  required_from_localq8 = localq8_us / target_layer_us if target_layer_us else 0.0
  same_run_ratio = baseline_us / localq8_us if localq8_us else 0.0
  seq50_ratio = seq50_selected_us / localq8_us if localq8_us else 0.0
  existing_bound = num(
      (tiling_gate.get("derived") or {}).get("max_existing_ratio_bound")
  )
  localq8_improves = same_run_ratio > 1.0
  localq8_beats_existing_bound = same_run_ratio > existing_bound
  reaches_whole_layer_budget = localq8_us <= target_layer_us if target_layer_us else False
  gate_passed = (
      bool(probe_wrapper.get("required_checks_passed"))
      and bool(probe.get("selected_swiglu_fusion_localq8_all_match_cpu"))
      and localq8_improves
      and localq8_beats_existing_bound
      and reaches_whole_layer_budget
  )

  checks = [
      check(
          "target_probe_required_checks_passed",
          bool(probe_wrapper.get("required_checks_passed")),
          {"artifact": display_path(args.probe)},
      ),
      check(
          "localq8_selected_fused_matches_cpu",
          bool(probe.get("selected_swiglu_fusion_localq8_all_match_cpu")),
          {
              "max_abs_diff": ((localq8_best.get("output_vs_cpu") or {}).get("max_abs_diff")
                               if isinstance(localq8_best.get("output_vs_cpu"), dict)
                               else None),
          },
      ),
      check(
          "localq8_timing_positive",
          baseline_us > 0.0 and localq8_us > 0.0,
          {"baseline_us": baseline_us, "localq8_us": localq8_us},
      ),
      check(
          "localq8_improves_same_run_baseline",
          localq8_improves,
          {"same_run_speedup": same_run_ratio},
      ),
      check(
          "localq8_exceeds_existing_ratio_bound",
          localq8_beats_existing_bound,
          {"same_run_speedup": same_run_ratio, "existing_ratio_bound": existing_bound},
      ),
      check(
          "localq8_reaches_whole_layer_budget",
          reaches_whole_layer_budget,
          {"localq8_us": localq8_us, "target_layer_budget_us": target_layer_us},
      ),
  ]

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "speedup_claims_allowed": False,
      "inputs": {
          "probe": display_path(args.probe),
          "budget": display_path(args.budget),
          "tiling_gate": display_path(args.tiling_gate),
      },
      "measurements": {
          "target_required_checks_passed": probe_wrapper.get("required_checks_passed"),
          "layer": probe_wrapper.get("layer"),
          "real_tokens": probe_wrapper.get("real_tokens"),
          "selected_fused_baseline_us": baseline_us,
          "selected_fused_baseline_best": baseline_best,
          "selected_fused_localq8_us": localq8_us,
          "selected_fused_localq8_best": localq8_best,
          "target_layer_budget_us": target_layer_us,
          "seq50_selected_swiglu_us": seq50_selected_us,
          "required_speedup_from_seq50_to_budget": required_from_seq50,
          "required_speedup_from_localq8_to_budget": required_from_localq8,
          "localq8_speedup_vs_same_run_baseline": same_run_ratio,
          "localq8_speedup_vs_seq50_selected": seq50_ratio,
          "existing_ratio_bound": existing_bound,
      },
      "checks": checks,
      "verdict": {
          "dpas_storage_work_distribution_gate_passed": gate_passed,
          "localq8_workgroup_sharing_closed": not gate_passed,
          "reason": (
              "local-Q8 workgroup sharing did not improve the selected fused "
              "gate/up-to-SwiGLU lane and remains far above the 8k whole-layer "
              "budget"
              if not gate_passed
              else "local-Q8 workgroup sharing cleared the selected-lane design gate"
          ),
          "next_route": (
              "Switch to a materially different non-atomic down-to-tail "
              "component proof that preserves rows_per_expert*9 contributor "
              "parallelism; do not spend another DPAS round on local-Q8 "
              "workgroup sharing or occupancy-only selected gate/up fill."
              if not gate_passed
              else "Continue DPAS storage/work-distribution implementation with "
              "promotion-grade prefill correctness and throughput evidence."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "workstream": WORKSTREAM,
      "tool": "tools/intel-qwen36-dpas-storage-workdist-gate.py",
      "artifact": display_path(out_dir),
      "source_artifacts": list(result["inputs"].values()),
      "required_checks_passed": all(row["pass"] for row in result["checks"][:3]),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  m = result["measurements"]
  lines = [
      "# DPAS Storage/Work-Distribution Gate",
      "",
      "This is route-selection evidence only; it is not speed-frontier evidence.",
      "",
      f"- selected fused baseline: `{m['selected_fused_baseline_us']:.3f}` us",
      f"- selected fused local-Q8: `{m['selected_fused_localq8_us']:.3f}` us",
      f"- local-Q8 speedup vs same-run baseline: "
      f"`{m['localq8_speedup_vs_same_run_baseline']:.6f}x`",
      f"- whole-layer 8k budget: `{m['target_layer_budget_us']:.3f}` us",
      f"- local-Q8 speedup still required: "
      f"`{m['required_speedup_from_localq8_to_budget']:.3f}x`",
      f"- existing ratio bound: `{m['existing_ratio_bound']:.3f}x`",
      "",
      "## Verdict",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
  parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
  parser.add_argument("--tiling-gate", type=Path, default=DEFAULT_TILING_GATE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  for key in ("probe", "budget", "tiling_gate", "out_dir"):
    path = getattr(args, key)
    if not path.is_absolute():
      setattr(args, key, ROOT / path)
  result = compute(args)
  write_outputs(result, args.out_dir)
  m = result["measurements"]
  print("dpas storage/work-distribution gate")
  print(f"  artifact: {args.out_dir}")
  print(
      "  selected fused baseline/local-Q8: "
      f"{m['selected_fused_baseline_us']:.3f} / "
      f"{m['selected_fused_localq8_us']:.3f} us"
  )
  print(
      "  local-Q8 same-run speedup: "
      f"{m['localq8_speedup_vs_same_run_baseline']:.6f}x"
  )
  print(f"  verdict: {result['verdict']['reason']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
