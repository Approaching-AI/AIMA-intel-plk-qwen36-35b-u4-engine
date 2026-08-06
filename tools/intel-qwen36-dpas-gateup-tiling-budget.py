#!/usr/bin/env python3
"""Gate DPAS selected gate/up tiling against proven occupancy ceilings.

This is route-selection arithmetic over existing artifacts. It asks whether
the seq49/seq50 selected gate/up-to-SwiGLU DPAS lane can plausibly reach the
8k prefill per-layer budget through occupancy/local-size fill alone, using the
accepted Q4 selected+shared occupancy4 and full-tensor rows as bounded ratios.
It is not benchmark evidence and cannot set the frontier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_BUDGET = ROOT / "output/dpas-prefill-budget-20260706TbudgetZ/budget.json"
DEFAULT_OCCUPANCY = (
    ROOT
    / "output/gpu-q4x8-selected-shared-gate-up-occupancy4-probe-20260705T203448Z"
)
DEFAULT_FULL_TENSOR = (
    ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z"
)
DEFAULT_OUT_DIR = ROOT / "output/dpas-gateup-tiling-budget-20260706Tseq52Z"
SCHEMA_VERSION = "intel-qwen36-dpas-gateup-tiling-budget-v0"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _json_path(path: Path, name: str) -> Path:
  candidate = path / name if path.is_dir() else path
  if not candidate.is_file():
    raise SystemExit(f"{candidate}: missing JSON artifact")
  return candidate


def _num(value: Any) -> float:
  if isinstance(value, (int, float)):
    return float(value)
  return 0.0


def _ratio(numerator: float, denominator: float) -> float:
  return numerator / denominator if denominator > 0.0 else 0.0


def _budget_summary(path: Path) -> dict[str, Any]:
  source = _json_path(path, "budget.json")
  payload = _load_json(source)
  selected_us = _num(payload.get("selected_swiglu_fusion_us"))
  layer_budget_us = _num(payload.get("target_layer_budget_us"))
  if selected_us <= 0.0 or layer_budget_us <= 0.0:
    raise SystemExit(f"{path}: missing selected_swiglu_fusion_us/target_layer_budget_us")
  return {
      "artifact": _display_path(source),
      "source_probe": payload.get("source_probe"),
      "bucket": int(_num(payload.get("bucket"))),
      "tile_tokens": int(_num(payload.get("tile_tokens"))),
      "layers": int(_num(payload.get("layers"))),
      "target_prefill_tokens_s": _num(payload.get("target_prefill_tokens_s")),
      "target_layer_budget_us": layer_budget_us,
      "selected_swiglu_fusion_us": selected_us,
      "selected_shared_swiglu_us": _num(payload.get("selected_shared_swiglu_us")),
      "fused_component_us": _num(payload.get("fused_component_us")),
      "fused_component_tokens_s": _num(payload.get("fused_component_tokens_s")),
      "required_full_component_speedup_to_target": _num(
          payload.get("required_full_component_speedup_to_target")
      ),
  }


def _occupancy_summary(path: Path) -> dict[str, Any]:
  source = _json_path(path, "probe-result.json")
  payload = _load_json(source)
  timings = payload.get("timings")
  checks = payload.get("checks")
  if not isinstance(timings, dict):
    raise SystemExit(f"{path}: missing timings")
  checks = checks if isinstance(checks, dict) else {}
  current_gbps = _num(timings.get("combined_gate_up_gpu_effective_packed_gb_s"))
  occupancy4_gbps = _num(timings.get("occupancy4_gate_up_gpu_effective_packed_gb_s"))
  occupancy_ratio = _num(timings.get("occupancy4_scaled_vs_combined_speedup"))
  if occupancy_ratio <= 0.0:
    occupancy_ratio = _ratio(occupancy4_gbps, current_gbps)
  return {
      "artifact": _display_path(path),
      "required_checks_passed": bool(payload.get("required_checks_passed")),
      "speedup_claims_allowed": bool(checks.get("speedup_claims_allowed")),
      "current_combined_gateup_gbps": current_gbps,
      "occupancy4_gateup_gbps": occupancy4_gbps,
      "occupancy4_scaled_vs_combined_speedup": occupancy_ratio,
      "combined_kernel_min_us": _num(timings.get("combined_gate_up_gpu_kernel_min_us")),
      "occupancy4_single_group_min_us": _num(
          timings.get("occupancy4_single_group_min_us")
      ),
      "combined_rows": int(_num(payload.get("combined_rows"))),
      "occupancy4_rows": int(_num(payload.get("occupancy4_rows"))),
  }


def _full_tensor_summary(path: Path, baseline_gbps: float) -> dict[str, Any]:
  source = _json_path(path, "probe-result.json")
  payload = _load_json(source)
  full_gbps = _num(payload.get("gpu_effective_packed_gb_s"))
  return {
      "artifact": _display_path(path),
      "required_checks_passed": bool(payload.get("required_checks_passed")),
      "gpu_effective_packed_gb_s": full_gbps,
      "gpu_kernel_min_us": _num(payload.get("gpu_kernel_min_us")),
      "best_variant": payload.get("gpu_best_variant"),
      "ratio_vs_selected_shared_current": _ratio(full_gbps, baseline_gbps),
      "rows": int(_num(payload.get("rows"))),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  budget = _budget_summary(args.budget)
  occupancy = _occupancy_summary(args.occupancy_artifact)
  full_tensor = _full_tensor_summary(
      args.full_tensor_artifact,
      occupancy["current_combined_gateup_gbps"],
  )
  selected_us = budget["selected_swiglu_fusion_us"]
  target_layer_us = budget["target_layer_budget_us"]
  whole_layer_need = _ratio(selected_us, target_layer_us)
  share_targets = {}
  for share in args.selected_budget_share:
    if share <= 0.0 or share > 1.0:
      raise SystemExit("--selected-budget-share values must be in (0, 1]")
    share_budget = target_layer_us * share
    share_targets[f"{share:.3f}"] = {
        "selected_budget_share": share,
        "selected_budget_us": share_budget,
        "required_speedup": _ratio(selected_us, share_budget),
    }

  ratios = {
      "occupancy4_scaled": occupancy["occupancy4_scaled_vs_combined_speedup"],
      "full_tensor_gbps_ratio": full_tensor["ratio_vs_selected_shared_current"],
  }
  optimistic = {
      name: {
          "assumed_speedup": ratio,
          "projected_selected_swiglu_us": _ratio(selected_us, ratio),
          "pct_of_whole_layer_budget": _ratio(_ratio(selected_us, ratio), target_layer_us)
          * 100.0,
          "still_exceeds_whole_layer_budget": _ratio(selected_us, ratio)
          > target_layer_us,
          "shortfall_speedup_vs_whole_layer_need": whole_layer_need - ratio,
      }
      for name, ratio in ratios.items()
  }
  max_existing_ratio = max(ratios.values())
  occupancy_only_closed = max_existing_ratio < whole_layer_need
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "budget_source": budget,
      "occupancy_source": occupancy,
      "full_tensor_source": full_tensor,
      "derived": {
          "selected_swiglu_required_speedup_to_fit_whole_layer_budget": whole_layer_need,
          "selected_swiglu_pct_of_whole_layer_budget": _ratio(
              selected_us, target_layer_us
          )
          * 100.0,
          "selected_budget_share_targets": share_targets,
          "existing_ratio_bounds": ratios,
          "max_existing_ratio_bound": max_existing_ratio,
          "optimistic_selected_swiglu_after_existing_ratios": optimistic,
          "occupancy_only_gateup_tiling_closed": occupancy_only_closed,
      },
      "verdict": {
          "occupancy_only_gateup_tiling_closed": occupancy_only_closed,
          "reason": (
              "selected fused gate/up-to-SwiGLU needs more speedup to fit even "
              "the entire per-layer 8k budget than the existing selected+shared "
              "Q4 occupancy4 and full-tensor ratio bounds can provide"
              if occupancy_only_closed
              else "existing occupancy/full-tensor ratio bounds can cover the "
              "whole-layer selected-lane speedup need"
          ),
          "next_route": (
              "Do not spend another round on occupancy-only/local-size/dispatch "
              "fill for the current selected gate/up DPAS lane. A DPAS follow-up "
              "must change storage, tiling, or work distribution beyond the "
              "existing Q4 occupancy/full-tensor bounds, or switch back to a "
              "different decode proof."
              if occupancy_only_closed
              else "An occupancy-focused DPAS tiling proof remains arithmetically "
              "admissible, but still must leave budget for shared/down/router/"
              "attention and correctness evidence."
          ),
      },
      "speedup_claims_allowed": False,
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "tool": "tools/intel-qwen36-dpas-gateup-tiling-budget.py",
      "workstream": WORKSTREAM,
      "artifact": str(out_dir.relative_to(ROOT)),
      "source_artifacts": [
          result["budget_source"]["artifact"],
          result["occupancy_source"]["artifact"],
          result["full_tensor_source"]["artifact"],
      ],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  derived = result["derived"]
  optimistic = derived["optimistic_selected_swiglu_after_existing_ratios"]
  lines = [
      "# DPAS Gate/Up Tiling Budget",
      "",
      "This is route-selection arithmetic over existing artifacts, not speed evidence.",
      "",
      "## Target",
      "",
      f"- selected fused SwiGLU: `{result['budget_source']['selected_swiglu_fusion_us']:.3f}` us",
      f"- whole per-layer 8k budget: `{result['budget_source']['target_layer_budget_us']:.3f}` us",
      f"- speedup needed to fit whole layer: "
      f"`{derived['selected_swiglu_required_speedup_to_fit_whole_layer_budget']:.3f}x`",
      "",
      "## Existing Ratio Bounds",
      "",
      f"- selected+shared Q4 occupancy4 ratio: "
      f"`{derived['existing_ratio_bounds']['occupancy4_scaled']:.3f}x`",
      f"- full-tensor Q4 ratio vs current selected+shared: "
      f"`{derived['existing_ratio_bounds']['full_tensor_gbps_ratio']:.3f}x`",
      "",
      "## Optimistic Projections",
      "",
  ]
  for key, item in optimistic.items():
    lines.append(
        f"- {key}: `{item['projected_selected_swiglu_us']:.3f}` us, "
        f"`{item['pct_of_whole_layer_budget']:.2f}%` of whole layer budget"
    )
  lines.extend([
      "",
      "## Verdict",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ])
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
  parser.add_argument("--occupancy-artifact", type=Path, default=DEFAULT_OCCUPANCY)
  parser.add_argument("--full-tensor-artifact", type=Path, default=DEFAULT_FULL_TENSOR)
  parser.add_argument(
      "--selected-budget-share",
      action="append",
      type=float,
      default=[1.0, 0.5, 0.333],
      help="Allowed share of per-layer budget for selected gate/up lane.",
  )
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("dpas gate/up tiling budget")
  print(f"  artifact: {out_dir}")
  print(
      "  selected lane need: "
      f"{derived['selected_swiglu_required_speedup_to_fit_whole_layer_budget']:.3f}x "
      "to fit the whole per-layer budget"
  )
  print(
      "  existing ratios: occupancy4 "
      f"{derived['existing_ratio_bounds']['occupancy4_scaled']:.3f}x; "
      "full-tensor "
      f"{derived['existing_ratio_bounds']['full_tensor_gbps_ratio']:.3f}x"
  )
  print(f"  verdict: {result['verdict']['reason']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
