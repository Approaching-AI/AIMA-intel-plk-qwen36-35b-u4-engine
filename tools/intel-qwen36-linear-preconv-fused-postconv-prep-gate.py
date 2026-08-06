#!/usr/bin/env python3
"""Score the fused linear-postconv prep component proof.

This is component-gate evidence only. It consumes the seq111 route gate and the
target-side fused postconv-prep probe, then decides whether a decode row is
admissible or whether the fused shape should close.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-linear-preconv-fused-postconv-prep-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_SEQ111 = (
    ROOT / "output/linear-preconv-remaining-kernel-algorithm-gate-20260707Tseq111Z/metrics.json"
)
DEFAULT_PROBE = (
    ROOT / "output/linear-preconv-fused-postconv-prep-probe-20260707Tseq112Z/probe.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/linear-preconv-fused-postconv-prep-gate-20260707Tseq112Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
  }


def _substage_gap(frontier: dict[str, Any], stage: str, substage: str) -> float:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  for row in budget.get("substage_gap_estimates_ms_per_token", []):
    if (
        isinstance(row, dict)
        and row.get("stage") == stage
        and row.get("substage") == substage
    ):
      return _num(row.get("gap_ms_per_token"))
  return 0.0


def _timings(probe: dict[str, Any]) -> dict[str, float]:
  raw = probe.get("probe")
  raw = raw if isinstance(raw, dict) else probe
  timings = raw.get("timings")
  timings = timings if isinstance(timings, dict) else {}
  fused = raw.get("fused_timings")
  fused = fused if isinstance(fused, dict) else {}
  current_sum = (
      _num(timings.get("silu_split_gpu_kernel_min_us"))
      + _num(timings.get("q_l2_gpu_kernel_min_us"))
      + _num(timings.get("k_l2_gpu_kernel_min_us"))
  )
  fused_min = _num(fused.get("fused_gpu_kernel_min_us"))
  return {
      "current_split_kernel_sum_us_per_layer": current_sum,
      "fused_kernel_us_per_layer": fused_min,
      "kernel_delta_us_per_layer": current_sum - fused_min,
      "current_silu_split_us": _num(timings.get("silu_split_gpu_kernel_min_us")),
      "current_q_l2_us": _num(timings.get("q_l2_gpu_kernel_min_us")),
      "current_k_l2_us": _num(timings.get("k_l2_gpu_kernel_min_us")),
      "fused_global_work_items": _num(fused.get("fused_global_work_items")),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  seq111 = _load_json(args.seq111)
  probe = _load_json(args.probe)
  raw_probe = probe.get("probe") if isinstance(probe.get("probe"), dict) else probe
  timings = _timings(probe)
  frontier_state = _frontier_summary(frontier)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  projected_cut = timings["kernel_delta_us_per_layer"] * 30.0 / 1000.0
  postconv_gap = _substage_gap(frontier, "linear_preconv", "postconv_prep")
  alpha_beta_gap = _substage_gap(frontier, "linear_preconv", "alpha_beta")
  component_correct = (
      probe.get("required_checks_passed") is True
      and _nested(raw_probe, "checks", "fused_postconv_prep_matches_oracle") is True
  )
  component_floor_covering = component_correct and projected_cut >= floor_gap
  component_non_regressive = component_correct and projected_cut >= 0.0

  checks = [
      {
          "name": "seq111_authorized_this_component",
          "pass": seq111.get("required_checks_passed") is True
          and seq111.get("selected_next_route")
          == "linear_preconv_fused_postconv_prep_component_probe",
      },
      {
          "name": "fused_component_correct",
          "pass": component_correct,
      },
      {
          "name": "timings_present",
          "pass": timings["current_split_kernel_sum_us_per_layer"] > 0.0
          and timings["fused_kernel_us_per_layer"] > 0.0,
      },
      {
          "name": "fused_component_does_not_clear_floor_gap",
          "pass": not component_floor_covering,
          "detail": {
              "projected_cut_ms_per_token": projected_cut,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "fused_component_is_regressive",
          "pass": not component_non_regressive,
          "detail": {
              "current_split_kernel_sum_us_per_layer": timings[
                  "current_split_kernel_sum_us_per_layer"],
              "fused_kernel_us_per_layer": timings["fused_kernel_us_per_layer"],
              "kernel_delta_us_per_layer": timings["kernel_delta_us_per_layer"],
          },
      },
      {
          "name": "alpha_beta_gap_remains_floor_sized",
          "pass": alpha_beta_gap > floor_gap,
          "detail": {
              "alpha_beta_gap_ms_per_token": alpha_beta_gap,
              "postconv_gap_ms_per_token": postconv_gap,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "component_correct": component_correct,
      "component_probe_allowed": False,
      "decode_probe_allowed": False,
      "component_floor_covering": component_floor_covering,
      "component_non_regressive": component_non_regressive,
      "disposition": (
          "reject_linear_preconv_fused_postconv_prep_component"
          if required else "linear_preconv_fused_postconv_prep_gate_failed"
      ),
      "selected_next_route": (
          "linear_preconv_alpha_beta_remaining_algorithm_gate"
          if required else "manual_review_linear_preconv_fused_postconv_prep"
      ),
      "next_action": (
          "Close the fused postconv-prep component as a speed route. It is exact, "
          "but the fused kernel is slower than the current split/qk path. The "
          "next linear-preconv unit is an alpha/beta remaining-algorithm gate; "
          "do not launch a decode row from the fused postconv component."
          if required else "Review failed fused-postconv checks before selecting the next route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "seq111": _rel(args.seq111),
          "probe": _rel(args.probe),
      },
      "frontier": frontier_state,
      "timings": timings,
      "projected_cut_ms_per_token": projected_cut,
      "substage_gap_ms_per_token": {
          "linear_preconv.alpha_beta": alpha_beta_gap,
          "linear_preconv.postconv_prep": postconv_gap,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if check["pass"] is not True]
  timings = payload["timings"]
  lines = [
      "# Linear-Preconv Fused Postconv-Prep Component Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- component correct: `{str(payload['component_correct']).lower()}`",
      f"- decode probe allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- split kernel sum: `{timings['current_split_kernel_sum_us_per_layer']:.3f}` us/layer",
      f"- fused kernel: `{timings['fused_kernel_us_per_layer']:.3f}` us/layer",
      f"- projected cut: `{payload['projected_cut_ms_per_token']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is component-gate evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--seq111", type=Path, default=DEFAULT_SEQ111)
  parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
