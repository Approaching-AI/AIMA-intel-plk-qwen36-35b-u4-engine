#!/usr/bin/env python3
"""Gate Q6 deferred-finish routes by tail-drain accounting.

This is route-selection arithmetic over existing rows, not benchmark evidence.
It separates the useful signal from the seq24 Q6 deferred-finish bundle
(selected down wait collapses) from the promotion failure (the wait moves into
FFN-tail and the speed row stays inside noise).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_FRONTIER = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_PROMOTION = ROOT / "output/r2-gpu-decode-smoke-20260705T150748Z"
DEFAULT_BASELINE_TS = "20260705T143327Z"
DEFAULT_DEFER_TS = "20260705T150703Z"
DEFAULT_DEFER_ONLY_TS = "20260705T150528Z"
DEFAULT_OUT_DIR = ROOT / "output/q6-defer-drain-budget-20260706Tseq53Z"
SCHEMA_VERSION = "intel-qwen36-q6-defer-drain-budget-v0"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
  matches = [row for row in _iter_jsonl(path) if row.get("ts") == ts]
  if not matches:
    raise SystemExit(f"{path}: no row ts={ts}")
  return matches[-1]


def _smoke_artifact(path: Path) -> dict[str, Any]:
  smoke_path = path / "smoke.json" if path.is_dir() else path
  if not smoke_path.is_file():
    raise SystemExit(f"{smoke_path}: missing smoke artifact")
  payload = _load_json(smoke_path)
  if not isinstance(payload, dict):
    raise SystemExit(f"{smoke_path}: expected JSON object")
  return payload


def _frontier_noise_pct(path: Path) -> float:
  payload = _load_json(path)
  no_progress = payload.get("no_progress") if isinstance(payload, dict) else None
  if not isinstance(no_progress, dict):
    return 0.0
  noise = no_progress.get("noise")
  if isinstance(noise, dict):
    rel = noise.get("rel")
    return float(rel) * 100.0 if isinstance(rel, (int, float)) else 0.0
  if isinstance(noise, (int, float)):
    value = float(noise)
    return value * 100.0 if 0.0 < value < 1.0 else value
  return 0.0


def _tps(payload: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = payload.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  smoke = payload.get("smoke")
  if isinstance(smoke, dict):
    return _tps(smoke)
  profile = payload.get("profile_smoke")
  if isinstance(profile, dict):
    value = profile.get("gpu_hybrid_decode_tok_s")
    if isinstance(value, (int, float)):
      return float(value)
  decode_ns = _num(payload.get("decode_ns"))
  tokens = _tokens(payload)
  return tokens * 1_000_000_000.0 / decode_ns if decode_ns > 0.0 else 0.0


def _tokens(row: dict[str, Any]) -> float:
  profile = row.get("profile_smoke")
  tokens = row.get("decode_tokens")
  if not isinstance(tokens, (int, float)) and isinstance(profile, dict):
    tokens = profile.get("decode_continuation_output_tokens")
  if not isinstance(tokens, (int, float)) or tokens <= 0:
    raise SystemExit(f"row {row.get('ts')} lacks positive token count")
  return float(tokens)


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  profile = row.get("profile_smoke")
  if not isinstance(profile, dict):
    raise SystemExit(f"row {row.get('ts')} lacks profile_smoke")
  wall = profile.get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {row.get('ts')} lacks wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(row) / 1_000_000.0


def _selected_ms(row: dict[str, Any], substage: str) -> float:
  profile = row.get("profile_smoke")
  if not isinstance(profile, dict):
    raise SystemExit(f"row {row.get('ts')} lacks profile_smoke")
  selected = profile.get("selected_ffn_wall_profile_ns")
  if not isinstance(selected, dict):
    raise SystemExit(f"row {row.get('ts')} lacks selected FFN profile")
  return _num(selected.get(substage)) / _tokens(row) / 1_000_000.0


def _wall_ms(row: dict[str, Any]) -> float:
  return _num(row.get("decode_ns")) / _tokens(row) / 1_000_000.0


def _pct_delta(candidate: float, baseline: float) -> float:
  return (candidate / baseline - 1.0) * 100.0 if baseline > 0.0 else 0.0


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "ts": row.get("ts"),
      "label": row.get("label"),
      "source_sha": row.get("source_sha"),
      "tps": _tps(row),
      "top1_matches_native": bool(row.get("top1_matches_native")),
      "required_checks_passed": bool(row.get("required_checks_passed")),
      "defer_ffn_down_finish_bundle": bool(row.get("defer_ffn_down_finish_bundle")),
      "wall_ms_per_token": _wall_ms(row),
      "selected_ffn_ms_per_token": _stage_ms(row, "selected_ffn"),
      "ffn_tail_ms_per_token": _stage_ms(row, "ffn_tail"),
      "selected_down_wait_ms_per_token": _selected_ms(row, "down_kernel_wait"),
      "selected_down_ms_per_token": _selected_ms(row, "down"),
      "selected_down_q8_ms_per_token": _selected_ms(row, "down_q8"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  baseline = _row_summary(_row_by_ts(args.explore_log, args.baseline_ts))
  defer_only = _row_summary(_row_by_ts(args.explore_log, args.defer_only_ts))
  defer_tail = _row_summary(_row_by_ts(args.explore_log, args.defer_ts))
  current_best = _smoke_artifact(args.current_best)
  promotion = _smoke_artifact(args.promotion_artifact)
  current_best_tps = _tps(current_best)
  promotion_tps = _tps(promotion)
  noise_pct = _frontier_noise_pct(args.frontier)

  selected_wait_saved = (
      baseline["selected_down_wait_ms_per_token"]
      - defer_tail["selected_down_wait_ms_per_token"]
  )
  selected_ffn_saved = (
      baseline["selected_ffn_ms_per_token"]
      - defer_tail["selected_ffn_ms_per_token"]
  )
  tail_growth = defer_tail["ffn_tail_ms_per_token"] - baseline["ffn_tail_ms_per_token"]
  net_selected_tail_delta = selected_ffn_saved - tail_growth
  projected_wall_without_tail_growth = defer_tail["wall_ms_per_token"] - max(0.0, tail_growth)
  projected_tps_without_tail_growth = (
      1000.0 / projected_wall_without_tail_growth
      if projected_wall_without_tail_growth > 0.0 else 0.0
  )
  promotion_delta_pct = _pct_delta(promotion_tps, current_best_tps)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "source_rows": {
          "baseline_profile": baseline,
          "defer_only_explore": defer_only,
          "defer_tail_rmsnorm_explore": defer_tail,
      },
      "speed_artifacts": {
          "current_best": {
              "artifact": _display_path(args.current_best),
              "tps": current_best_tps,
              "top1_matches_native": bool(current_best.get("top1_matches_native")),
              "required_checks_passed": bool(current_best.get("required_checks_passed")),
          },
          "defer_tail_promotion": {
              "artifact": _display_path(args.promotion_artifact),
              "tps": promotion_tps,
              "top1_matches_native": bool(promotion.get("top1_matches_native")),
              "required_checks_passed": bool(promotion.get("required_checks_passed")),
          },
      },
      "derived": {
          "frontier_noise_pct": noise_pct,
          "promotion_delta_pct_vs_current_best": promotion_delta_pct,
          "promotion_outside_noise": abs(promotion_delta_pct) > noise_pct,
          "selected_down_wait_saved_ms_per_token": selected_wait_saved,
          "selected_ffn_saved_ms_per_token": selected_ffn_saved,
          "ffn_tail_growth_ms_per_token": tail_growth,
          "net_selected_plus_tail_delta_ms_per_token": net_selected_tail_delta,
          "projected_wall_without_tail_growth_ms_per_token": projected_wall_without_tail_growth,
          "projected_tps_without_tail_growth": projected_tps_without_tail_growth,
          "tail_drain_elimination_clears_floor": projected_tps_without_tail_growth >= args.floor_tps,
      },
      "verdict": {
          "q6_defer_tail_bundle_promoted": (
              promotion_tps > current_best_tps
              and abs(promotion_delta_pct) > noise_pct
          ),
          "tail_drain_shift_confirmed": (
              selected_wait_saved > 1.0 and tail_growth > 1.0
          ),
          "reason": (
              "Q6 deferred finish collapses selected down wait, but the "
              "measured bundle shifts the drain into FFN-tail and the promotion "
              "row stays inside the frontier noise band."
          ),
          "next_route": (
              "Do not retry finish deferral or tail read-as-drain alone. The "
              "only admissible continuation on this axis is a broader "
              "tail-output/residual ownership or down/tail fusion design that "
              "removes the tail drain instead of moving it."
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
      "tool": "tools/intel-qwen36-q6-defer-drain-budget.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  derived = result["derived"]
  lines = [
      "# Q6 Defer Drain Budget",
      "",
      "This is route-selection arithmetic over existing rows, not speed evidence.",
      "",
      "## Promotion",
      "",
      f"- current best: `{result['speed_artifacts']['current_best']['tps']:.8f}` tok/s",
      f"- defer bundle promotion: "
      f"`{result['speed_artifacts']['defer_tail_promotion']['tps']:.8f}` tok/s",
      f"- delta vs best: `{derived['promotion_delta_pct_vs_current_best']:.3f}%`",
      f"- frontier noise band: `{derived['frontier_noise_pct']:.3f}%`",
      "",
      "## Drain Accounting",
      "",
      f"- selected down wait saved: "
      f"`{derived['selected_down_wait_saved_ms_per_token']:.3f}` ms/token",
      f"- selected FFN saved: "
      f"`{derived['selected_ffn_saved_ms_per_token']:.3f}` ms/token",
      f"- FFN-tail growth: "
      f"`{derived['ffn_tail_growth_ms_per_token']:.3f}` ms/token",
      f"- projected speed without tail growth: "
      f"`{derived['projected_tps_without_tail_growth']:.3f}` tok/s",
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
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--baseline-ts", default=DEFAULT_BASELINE_TS)
  parser.add_argument("--defer-only-ts", default=DEFAULT_DEFER_ONLY_TS)
  parser.add_argument("--defer-ts", default=DEFAULT_DEFER_TS)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument("--promotion-artifact", type=Path, default=DEFAULT_PROMOTION)
  parser.add_argument("--floor-tps", type=float, default=19.5)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("q6 defer drain budget")
  print(f"  artifact: {out_dir}")
  print(
      "  promotion delta vs best: "
      f"{derived['promotion_delta_pct_vs_current_best']:.3f}% "
      f"(noise {derived['frontier_noise_pct']:.3f}%)"
  )
  print(
      "  drain: selected wait saved "
      f"{derived['selected_down_wait_saved_ms_per_token']:.3f} ms/token; "
      "tail grew "
      f"{derived['ffn_tail_growth_ms_per_token']:.3f} ms/token"
  )
  print(f"  verdict: {result['verdict']['reason']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
