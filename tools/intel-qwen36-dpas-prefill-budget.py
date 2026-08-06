#!/usr/bin/env python3
"""Budget a DPAS prefill component artifact against the 8k prefill target.

This is a route-selection arithmetic gate, not benchmark evidence. It asks
whether the measured DPAS component shape is close enough to the accepted
prefill target to justify more same-shape kernel work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_ACCEPTANCE = Path(
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
)
DEFAULT_LAYERS = 40
DEFAULT_BUCKET = 8192


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(obj: dict[str, Any], *keys: str) -> float:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return 0.0
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else 0.0


def _bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def _probe_from_artifact(path: Path) -> tuple[dict[str, Any], Path]:
  if path.is_dir():
    for name in ("probe.json", "probe-result.json"):
      candidate = path / name
      if candidate.is_file():
        payload = _load_json(candidate)
        if isinstance(payload, dict):
          probe = payload.get("probe") if name == "probe.json" else payload
          if isinstance(probe, dict):
            return probe, candidate
    raise SystemExit(f"{path}: no probe.json or probe-result.json")
  payload = _load_json(path)
  if not isinstance(payload, dict):
    raise SystemExit(f"{path}: expected JSON object")
  probe = payload.get("probe") if "probe" in payload else payload
  if not isinstance(probe, dict):
    raise SystemExit(f"{path}: expected probe object")
  return probe, path


def _target_tps(acceptance: Path, bucket: int) -> float:
  payload = _load_json(acceptance)
  if not isinstance(payload, dict):
    raise SystemExit(f"{acceptance}: expected JSON object")
  targets = payload.get("bootstrap_targets")
  if not isinstance(targets, dict):
    raise SystemExit(f"{acceptance}: missing bootstrap_targets")
  prefill = targets.get("prefill_tokens_s")
  if not isinstance(prefill, dict):
    raise SystemExit(f"{acceptance}: missing prefill targets")
  value = prefill.get(str(bucket))
  if not isinstance(value, (int, float)):
    raise SystemExit(f"{acceptance}: missing prefill target for bucket {bucket}")
  return float(value)


def compute_budget(
    artifact: Path,
    acceptance: Path,
    bucket: int,
    layers: int,
) -> dict[str, Any]:
  probe, source = _probe_from_artifact(artifact)
  target_tps = _target_tps(acceptance, bucket)
  tokens = int(_num(probe, "real_tile_tokens") or _num(probe, "shared_tile_tokens"))
  if tokens <= 0:
    raise SystemExit(f"{source}: missing positive real_tile_tokens")
  if layers <= 0:
    raise SystemExit("--layers must be positive")

  target_layer_us = tokens * 1_000_000.0 / (target_tps * layers)
  baseline_parts = {
      "selected_gateup": _num(probe, "real_tokenreuse_best_kernel_min_us"),
      "shared_gateup": _num(probe, "shared_tokenreuse_best_kernel_min_us"),
      "selected_q8_prep": _num(probe, "selected_down_q8_prep", "kernel_min_us"),
      "shared_q8_prep": _num(probe, "shared_down_q8_prep", "kernel_min_us"),
      "selected_down": _num(probe, "selected_down_device_q8_tokenreuse_best_kernel_min_us"),
      "shared_down": _num(probe, "shared_down_device_q8_tokenreuse_best_kernel_min_us"),
  }
  fused_parts = {
      "selected_swiglu_fusion": _num(probe, "selected_swiglu_fusion_best_kernel_min_us"),
      "shared_swiglu_fusion": _num(probe, "shared_swiglu_fusion_best_kernel_min_us"),
      "selected_q8_prep": _num(probe, "selected_down_fused_q8_prep", "kernel_min_us"),
      "shared_q8_prep": _num(probe, "shared_down_fused_q8_prep", "kernel_min_us"),
      "selected_down": _num(probe, "selected_down_fused_device_q8_tokenreuse_best_kernel_min_us"),
      "shared_down": _num(probe, "shared_down_fused_device_q8_tokenreuse_best_kernel_min_us"),
  }
  baseline_us = sum(baseline_parts.values())
  fused_us = sum(fused_parts.values())
  selected_swiglu_us = fused_parts["selected_swiglu_fusion"]
  selected_shared_swiglu_us = (
      fused_parts["selected_swiglu_fusion"] +
      fused_parts["shared_swiglu_fusion"]
  )
  projected_tps = tokens * 1_000_000.0 / (fused_us * layers) if fused_us > 0 else 0.0
  baseline_tps = tokens * 1_000_000.0 / (baseline_us * layers) if baseline_us > 0 else 0.0
  savings_us = baseline_us - fused_us
  savings_pct = (savings_us / baseline_us * 100.0) if baseline_us > 0 else 0.0
  required_full_speedup = fused_us / target_layer_us if target_layer_us > 0 else 0.0
  selected_only_speedup = selected_swiglu_us / target_layer_us if target_layer_us > 0 else 0.0
  selected_shared_speedup = (
      selected_shared_swiglu_us / target_layer_us if target_layer_us > 0 else 0.0
  )
  selected_only_exceeds_budget = selected_swiglu_us > target_layer_us

  checks = {
      "required_checks_passed": _bool(probe, "required_checks_passed"),
      "selected_swiglu_fusion_all_match_cpu": _bool(
          probe, "selected_swiglu_fusion_all_match_cpu"
      ),
      "shared_swiglu_fusion_all_match_cpu": _bool(
          probe, "shared_swiglu_fusion_all_match_cpu"
      ),
      "selected_down_fused_device_q8_tokenreuse_all_match_cpu": _bool(
          probe, "selected_down_fused_device_q8_tokenreuse_all_match_cpu"
      ),
      "shared_down_fused_device_q8_tokenreuse_all_match_cpu": _bool(
          probe, "shared_down_fused_device_q8_tokenreuse_all_match_cpu"
      ),
  }
  if not checks["required_checks_passed"]:
    # The wrapper payload carries required_checks_passed; probe-result does not.
    checks["required_checks_passed"] = all(
        value for key, value in checks.items() if key != "required_checks_passed"
    )

  verdict = (
      "current DPAS fused component shape is target-blocked by selected gate/up"
      if selected_only_exceeds_budget
      else "current DPAS fused component shape is not blocked by selected gate/up alone"
  )
  next_action = (
      "Do not spend another round on same-shape Q8-prep/down or small fusion "
      "variants; the selected gate/up-to-SwiGLU lane alone is above the entire "
      "per-layer 8k budget. Next DPAS work must change selected gate/up tiling, "
      "storage, or work distribution by a multi-x factor, or switch back to the "
      "decode lifetime route."
      if selected_only_exceeds_budget
      else "A same-shape DPAS follow-up may still be arithmetically admissible."
  )

  return {
      "schema": "intel-qwen36-dpas-prefill-budget-v0",
      "artifact": str(artifact),
      "source_probe": str(source),
      "bucket": bucket,
      "layers": layers,
      "tile_tokens": tokens,
      "target_prefill_tokens_s": round(target_tps, 6),
      "target_layer_budget_us": round(target_layer_us, 3),
      "checks": checks,
      "baseline_component_us": round(baseline_us, 3),
      "baseline_component_tokens_s": round(baseline_tps, 6),
      "baseline_parts_us": {key: round(value, 3) for key, value in baseline_parts.items()},
      "fused_component_us": round(fused_us, 3),
      "fused_component_tokens_s": round(projected_tps, 6),
      "fused_component_pct_of_target": round(projected_tps / target_tps * 100.0, 6),
      "fused_parts_us": {key: round(value, 3) for key, value in fused_parts.items()},
      "fused_savings_us": round(savings_us, 3),
      "fused_savings_pct": round(savings_pct, 6),
      "required_full_component_speedup_to_target": round(required_full_speedup, 6),
      "selected_swiglu_fusion_us": round(selected_swiglu_us, 3),
      "selected_swiglu_pct_of_budget": round(selected_only_speedup * 100.0, 6),
      "selected_swiglu_required_speedup_to_fit_budget": round(selected_only_speedup, 6),
      "selected_shared_swiglu_us": round(selected_shared_swiglu_us, 3),
      "selected_shared_swiglu_pct_of_budget": round(selected_shared_speedup * 100.0, 6),
      "selected_swiglu_alone_exceeds_layer_budget": selected_only_exceeds_budget,
      "route_verdict": verdict,
      "next_action": next_action,
      "speedup_claims_allowed": False,
  }


def render_text(budget: dict[str, Any]) -> str:
  lines = [
      f"DPAS prefill budget - {budget['artifact']}",
      f"  target: {budget['target_prefill_tokens_s']} tok/s @ bucket {budget['bucket']}",
      f"  tile/layers: {budget['tile_tokens']} tokens over {budget['layers']} layers",
      f"  per-layer budget: {budget['target_layer_budget_us']:.3f} us",
      "",
      "  component:",
      f"    baseline device-Q8 FFN       {budget['baseline_component_us']:>9.3f} us"
      f"  ({budget['baseline_component_tokens_s']:>8.3f} tok/s)",
      f"    fused SwiGLU device-Q8       {budget['fused_component_us']:>9.3f} us"
      f"  ({budget['fused_component_tokens_s']:>8.3f} tok/s)",
      f"    fused savings                {budget['fused_savings_us']:>9.3f} us"
      f"  ({budget['fused_savings_pct']:>7.3f}%)",
      f"    full component speedup need  {budget['required_full_component_speedup_to_target']:>9.3f}x",
      "",
      "  lower bounds:",
      f"    selected fused SwiGLU        {budget['selected_swiglu_fusion_us']:>9.3f} us"
      f"  ({budget['selected_swiglu_pct_of_budget']:>7.2f}% of whole layer budget)",
      f"    selected+shared fused SwiGLU {budget['selected_shared_swiglu_us']:>9.3f} us"
      f"  ({budget['selected_shared_swiglu_pct_of_budget']:>7.2f}% of whole layer budget)",
      "",
      f"  verdict: {budget['route_verdict']}",
      f"  next: {budget['next_action']}",
      "  note: arithmetic gate only; no speed or product claim.",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("artifact", type=Path)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--bucket", type=int, default=DEFAULT_BUCKET)
  parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  budget = compute_budget(args.artifact, args.acceptance, args.bucket, args.layers)
  if args.out_dir is not None:
    out_dir = args.out_dir
    if not out_dir.is_absolute():
      out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    created_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    manifest = {
        "schema": "intel-qwen36-dpas-prefill-budget-manifest-v0",
        "workstream": WORKSTREAM,
        "created_at": created_at,
        "tool": "tools/intel-qwen36-dpas-prefill-budget.py",
        "artifact": str(out_dir.relative_to(ROOT)),
        "source_artifact": budget["artifact"],
        "bucket": args.bucket,
        "layers": args.layers,
        "speedup_claims_allowed": False,
    }
    (out_dir / "budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_text(budget) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_rows = [
        ("target_layer_budget_us", budget["target_layer_budget_us"]),
        ("baseline_component_us", budget["baseline_component_us"]),
        ("fused_component_us", budget["fused_component_us"]),
        ("fused_component_tokens_s", budget["fused_component_tokens_s"]),
        ("fused_component_pct_of_target", budget["fused_component_pct_of_target"]),
        ("required_full_component_speedup_to_target", budget["required_full_component_speedup_to_target"]),
        ("selected_swiglu_fusion_us", budget["selected_swiglu_fusion_us"]),
        ("selected_swiglu_pct_of_budget", budget["selected_swiglu_pct_of_budget"]),
        ("selected_swiglu_alone_exceeds_layer_budget", budget["selected_swiglu_alone_exceeds_layer_budget"]),
    ]
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
      for metric, value in metric_rows:
        handle.write(json.dumps({
            "phase": "dpas_prefill_budget",
            "metric": metric,
            "value": value,
        }, sort_keys=True) + "\n")
    print(out_dir)
    return 0
  if args.json:
    print(json.dumps(budget, indent=2, sort_keys=True))
  else:
    print(render_text(budget))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
