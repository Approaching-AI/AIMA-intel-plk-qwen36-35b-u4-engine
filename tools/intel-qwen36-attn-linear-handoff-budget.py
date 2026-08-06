#!/usr/bin/env python3
"""Gate simple linear-final -> attention-front handoff routes.

This is route-selection arithmetic over existing evidence. It compares the
nearby current-stack noqueue row with the scratch-backed linear-final
device-Q8 handoff row, then folds in the F32-input output-projection component
probe. It is not benchmark evidence and cannot set the frontier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_F32_PROBE = (
    ROOT / "output/gpu-q4x8-output-projection-probe-20260706T072447Z"
)
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_BASELINE_TS = "20260705T233559Z"
DEFAULT_HANDOFF_TS = "20260706T061902Z"
DEFAULT_OUT_DIR = ROOT / "output/attn-linear-handoff-budget-20260706Tseq51Z"
SCHEMA_VERSION = "intel-qwen36-attn-linear-handoff-budget-v1"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
  for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    if isinstance(row, dict):
      yield row


def _row_by_ts(path: Path, ts: str) -> dict[str, Any]:
  matches = [row for row in _jsonl(path) if row.get("ts") == ts]
  if not matches:
    raise SystemExit(f"{path}: no explore row with ts={ts}")
  return matches[-1]


def _num(value: Any) -> float:
  if isinstance(value, (int, float)):
    return float(value)
  return 0.0


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("profile_smoke")
  if isinstance(value, dict):
    return value
  return {}


def _tokens(row: dict[str, Any], profile: dict[str, Any]) -> float:
  tokens = row.get("decode_tokens") or profile.get("decode_continuation_output_tokens")
  if not isinstance(tokens, (int, float)) or tokens <= 0:
    raise SystemExit(f"row {row.get('ts')} lacks positive decode token count")
  return float(tokens)


def _ms_per_token(ns: Any, tokens: float) -> float:
  return _num(ns) / tokens / 1_000_000.0


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
  profile = _profile(row)
  wall = profile.get("wall_profile_ns")
  linear_delta = profile.get("linear_delta_wall_profile_ns")
  if not isinstance(wall, dict) or not isinstance(linear_delta, dict):
    raise SystemExit(f"row {row.get('ts')} lacks wall/linear-delta profile")
  tokens = _tokens(row, profile)
  return {
      "ts": row.get("ts"),
      "label": row.get("label"),
      "tokens": tokens,
      "tps": _num(row.get("tps")),
      "top1_matches_native": bool(row.get("top1_matches_native")),
      "required_checks_passed": bool(row.get("required_checks_passed")),
      "wall_ms_per_token": _ms_per_token(row.get("decode_ns"), tokens),
      "attention_front_ms_per_token": _ms_per_token(
          wall.get("attention_front"), tokens
      ),
      "linear_delta_ms_per_token": _ms_per_token(
          wall.get("linear_delta"), tokens
      ),
      "linear_delta_final_read_ms_per_token": _ms_per_token(
          linear_delta.get("final_read"), tokens
      ),
      "linear_delta_kernel_ms_per_token": _ms_per_token(
          linear_delta.get("kernel"), tokens
      ),
  }


def _f32_probe_summary(path: Path) -> dict[str, Any]:
  probe_path = path / "probe-result.json" if path.is_dir() else path
  if not probe_path.is_file():
    raise SystemExit(f"{probe_path}: missing F32-input output-projection probe")
  payload = _load_json(probe_path)
  timings = payload.get("timings")
  comparisons = payload.get("comparisons")
  checks = payload.get("checks")
  if not isinstance(timings, dict) or not isinstance(comparisons, dict):
    raise SystemExit(f"{probe_path}: malformed probe-result")
  f32 = comparisons.get("linear_attn_out_f32input")
  if not isinstance(f32, dict):
    raise SystemExit(f"{probe_path}: missing linear_attn_out_f32input")
  gpu_vs_oracle = f32.get("gpu_vs_oracle")
  if not isinstance(gpu_vs_oracle, dict):
    raise SystemExit(f"{probe_path}: missing f32 gpu_vs_oracle comparison")
  checks = checks if isinstance(checks, dict) else {}
  return {
      "artifact": str(path),
      "q8_rowlane_kernel_us": _num(
          timings.get("output_projection_gpu_kernel_min_us")
      ),
      "f32input_kernel_us": _num(
          timings.get("f32input_output_projection_gpu_kernel_min_us")
      ),
      "f32input_gpu_vs_oracle_cosine": _num(gpu_vs_oracle.get("cosine")),
      "f32input_gpu_vs_oracle_max_abs": _num(gpu_vs_oracle.get("max_abs_diff")),
      "f32input_gpu_vs_oracle_rmse": _num(gpu_vs_oracle.get("rmse")),
      "f32input_strict_component_passed": bool(
          checks.get("f32input_output_projection_matches_oracle")
      ),
  }


def _best_summary(path: Path) -> dict[str, Any]:
  smoke = _load_json(path / "smoke.json" if path.is_dir() else path)
  return {
      "artifact": str(path),
      "tps": _num(smoke.get("gpu_hybrid_decode_tok_s")),
      "required_checks_passed": bool(smoke.get("required_checks_passed")),
      "top1_matches_native": bool(smoke.get("top1_matches_native")),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  baseline = _row_summary(_row_by_ts(args.explore_log, args.baseline_ts))
  handoff = _row_summary(_row_by_ts(args.explore_log, args.handoff_ts))
  f32_probe = _f32_probe_summary(args.f32_probe)
  best = _best_summary(args.current_best)

  final_read_saved = (
      baseline["linear_delta_final_read_ms_per_token"]
      - handoff["linear_delta_final_read_ms_per_token"]
  )
  attention_front_growth = (
      handoff["attention_front_ms_per_token"]
      - baseline["attention_front_ms_per_token"]
  )
  wall_growth = handoff["wall_ms_per_token"] - baseline["wall_ms_per_token"]
  f32_speed_delta_us = (
      f32_probe["f32input_kernel_us"] - f32_probe["q8_rowlane_kernel_us"]
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "baseline_row": baseline,
      "device_q8_handoff_row": handoff,
      "f32input_output_projection_probe": f32_probe,
      "current_best": best,
      "derived": {
          "final_read_saved_ms_per_token": final_read_saved,
          "attention_front_growth_ms_per_token": attention_front_growth,
          "wall_growth_ms_per_token": wall_growth,
          "device_q8_handoff_tps_delta_vs_baseline": (
              handoff["tps"] - baseline["tps"]
          ),
          "device_q8_handoff_tps_delta_vs_best": handoff["tps"] - best["tps"],
          "f32input_kernel_delta_us_vs_q8_rowlane": f32_speed_delta_us,
          "attention_front_noninflation_required": True,
          "device_q8_handoff_noninflation_passed": attention_front_growth <= 0.0,
          "f32input_strict_component_passed": f32_probe[
              "f32input_strict_component_passed"
          ],
      },
      "verdict": {
          "simple_final_output_handoff_closed": (
              attention_front_growth > 0.0
              and not f32_probe["f32input_strict_component_passed"]
          ),
          "reason": (
              "scratch-backed device-Q8 handoff removes the linear-delta final "
              "read but inflates attention-front wall; the F32-input projection "
              "avoids the device-Q8 chain but fails the strict component gate."
          ),
          "next_route": (
              "Do not spend another round on simple final_output handoff. "
              "The remaining attention/linear work must change broader "
              "attention-front/FFN-input ownership or use a different kernel "
              "algorithm; otherwise switch back to a true multi-x DPAS "
              "gate/up tiling/storage proof."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  derived = result["derived"]
  f32 = result["f32input_output_projection_probe"]
  lines = [
      "# Attention/Linear Handoff Budget",
      "",
      "This is route-selection arithmetic over existing evidence, not speed evidence.",
      "",
      "## Device-Q8 Handoff",
      "",
      f"- baseline: `{result['baseline_row']['label']}` "
      f"{result['baseline_row']['tps']:.8f} tok/s",
      f"- handoff: `{result['device_q8_handoff_row']['label']}` "
      f"{result['device_q8_handoff_row']['tps']:.8f} tok/s",
      f"- final-read saved: `{derived['final_read_saved_ms_per_token']:.3f}` ms/token",
      f"- attention-front growth: `{derived['attention_front_growth_ms_per_token']:.3f}` ms/token",
      f"- total wall growth: `{derived['wall_growth_ms_per_token']:.3f}` ms/token",
      "",
      "## F32-Input Projection",
      "",
      f"- Q8 rowlane projection: `{f32['q8_rowlane_kernel_us']:.3f}` us",
      f"- F32-input projection: `{f32['f32input_kernel_us']:.3f}` us",
      f"- F32 gpu-vs-oracle cosine: `{f32['f32input_gpu_vs_oracle_cosine']:.10f}`",
      f"- F32 gpu-vs-oracle max abs: `{f32['f32input_gpu_vs_oracle_max_abs']:.10f}`",
      f"- strict component gate passed: `{str(f32['f32input_strict_component_passed']).lower()}`",
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
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--baseline-ts", default=DEFAULT_BASELINE_TS)
  parser.add_argument("--handoff-ts", default=DEFAULT_HANDOFF_TS)
  parser.add_argument("--f32-probe", type=Path, default=DEFAULT_F32_PROBE)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_BEST)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  write_outputs(result, args.out_dir)
  derived = result["derived"]
  print("attention/linear handoff budget")
  print(f"  artifact: {args.out_dir}")
  print(
      "  device-Q8: final-read saved "
      f"{derived['final_read_saved_ms_per_token']:.3f} ms/token; "
      "attention-front grew "
      f"{derived['attention_front_growth_ms_per_token']:.3f} ms/token; "
      "wall delta "
      f"{derived['wall_growth_ms_per_token']:.3f} ms/token"
  )
  print(
      "  f32-input: strict component passed "
      f"{str(derived['f32input_strict_component_passed']).lower()}; "
      "kernel delta "
      f"{derived['f32input_kernel_delta_us_vs_q8_rowlane']:.3f} us"
  )
  print(
      "  verdict: "
      f"{result['verdict']['reason']} "
      f"{result['verdict']['next_route']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
