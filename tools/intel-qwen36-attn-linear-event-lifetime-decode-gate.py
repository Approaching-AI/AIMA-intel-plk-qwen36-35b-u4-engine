#!/usr/bin/env python3
"""Gate the combined attention/linear event-lifetime decode explore rows.

This is route evidence, not promotion evidence.  It compares a same-source
accepted-stack baseline to the default-off IQ36_ATTENTION_LINEAR_EVENT_LIFETIME
candidate and records whether the bundle clears the floor gap without growing
attention-front wall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-attn-linear-event-lifetime-decode-gate-v0"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_FRONTIER = ROOT / "doc/active" / WORKSTREAM / "frontier.json"
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_BASELINE_LABEL = "seq84-attn-linear-event-lifetime-current-source-baseline-8tok"
DEFAULT_CANDIDATE_LABEL = "seq84-attn-linear-event-lifetime-combined-8tok"
DEFAULT_OUT_DIR = ROOT / "output/attn-linear-event-lifetime-decode-gate-20260707Tseq84Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


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


def _row_by_label(path: Path, label: str) -> dict[str, Any]:
  rows = [row for row in _iter_jsonl(path) if row.get("label") == label]
  if not rows:
    raise SystemExit(f"{path}: missing explore row label={label!r}")
  return rows[-1]


def _artifact(path: Path) -> dict[str, Any]:
  smoke = path / "smoke.json" if path.is_dir() else path
  payload = _load_json(smoke)
  if not isinstance(payload, dict):
    raise SystemExit(f"{smoke}: expected JSON object")
  return payload


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _tokens(row: dict[str, Any]) -> float:
  value = row.get("decode_tokens")
  if isinstance(value, (int, float)) and value > 0:
    return float(value)
  smoke = row.get("profile_smoke")
  if isinstance(smoke, dict):
    value = smoke.get("decode_continuation_output_tokens")
    if isinstance(value, (int, float)) and value > 0:
      return float(value)
  raise SystemExit(f"row {row.get('label')}: missing positive decode token count")


def _tps(row: dict[str, Any]) -> float:
  value = row.get("tps") or row.get("gpu_hybrid_decode_tok_s")
  if isinstance(value, (int, float)):
    return float(value)
  decode_ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  tokens = _tokens(row)
  return tokens * 1_000_000_000.0 / decode_ns if decode_ns > 0.0 else 0.0


def _wall_ms(row: dict[str, Any]) -> float:
  decode_ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  return decode_ns / _tokens(row) / 1_000_000.0 if decode_ns > 0.0 else 0.0


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  smoke = row.get("profile_smoke")
  return smoke if isinstance(smoke, dict) else row


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  wall = _profile(row).get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {row.get('label')}: missing wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(row) / 1_000_000.0


def _preconv_ms(row: dict[str, Any], stage: str) -> float:
  wall = _profile(row).get("linear_preconv_wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {row.get('label')}: missing linear_preconv_wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(row) / 1_000_000.0


def _delta_ms(row: dict[str, Any], stage: str) -> float:
  wall = _profile(row).get("linear_delta_wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {row.get('label')}: missing linear_delta_wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(row) / 1_000_000.0


def _noise_rel(frontier_path: Path) -> float:
  frontier = _load_json(frontier_path)
  no_progress = frontier.get("no_progress") if isinstance(frontier, dict) else None
  noise = no_progress.get("noise") if isinstance(no_progress, dict) else None
  if isinstance(noise, dict):
    value = _num(noise.get("rel"))
  else:
    value = _num(noise)
  return value if 0.0 < value < 1.0 else value / 100.0


def _floor_gap(frontier_path: Path) -> float:
  frontier = _load_json(frontier_path)
  budget = frontier.get("goal_budget") if isinstance(frontier, dict) else None
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  return max(0.0, _num(per_token.get("wall")) - _num(verdict.get("floor_budget_ms_per_token")))


def _summary(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "ts": row.get("ts"),
      "label": row.get("label"),
      "source_sha": row.get("source_sha"),
      "config_sha": row.get("config_sha"),
      "decode_tokens": _tokens(row),
      "tps": _tps(row),
      "wall_ms_per_token": _wall_ms(row),
      "top1_matches_native": row.get("top1_matches_native") is True,
      "attention_linear_event_lifetime": row.get("attention_linear_event_lifetime") is True,
      "linear_final_device_q8_handoff": row.get("linear_final_device_q8_handoff") is True,
      "attention_front_resident_residual_input": row.get("attention_front_resident_residual_input") is True,
      "attention_front_ms_per_token": _stage_ms(row, "attention_front"),
      "linear_preconv_ms_per_token": _stage_ms(row, "linear_preconv"),
      "linear_delta_ms_per_token": _stage_ms(row, "linear_delta"),
      "layer_input_rmsnorm_ms_per_token": _stage_ms(row, "layer_input_rmsnorm"),
      "full_core_ms_per_token": _stage_ms(row, "full_core"),
      "linear_preconv_input_q8_ms_per_token": _preconv_ms(row, "input_q8"),
      "linear_preconv_alpha_beta_ms_per_token": _preconv_ms(row, "alpha_beta"),
      "linear_preconv_qkv_conv_ms_per_token": _preconv_ms(row, "qkv_conv"),
      "linear_delta_final_read_ms_per_token": _delta_ms(row, "final_read"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  baseline_row = _row_by_label(args.explore_log, args.baseline_label)
  candidate_row = _row_by_label(args.explore_log, args.candidate_label)
  baseline = _summary(baseline_row)
  candidate = _summary(candidate_row)
  current_best = _artifact(args.current_best)
  current_best_tps = _tps(current_best)
  noise_rel = _noise_rel(args.frontier)
  floor_gap_ms = _floor_gap(args.frontier)

  same_source = (
      isinstance(baseline.get("source_sha"), str)
      and baseline.get("source_sha") == candidate.get("source_sha")
  )
  same_tokens = baseline["decode_tokens"] == candidate["decode_tokens"] == 8.0
  candidate_tps = candidate["tps"]
  baseline_tps = baseline["tps"]
  attention_front_delta = (
      candidate["attention_front_ms_per_token"] -
      baseline["attention_front_ms_per_token"]
  )
  wall_delta = candidate["wall_ms_per_token"] - baseline["wall_ms_per_token"]
  tps_delta_pct = (
      (candidate_tps / baseline_tps - 1.0) * 100.0 if baseline_tps > 0.0 else 0.0
  )
  tps_delta_pct_vs_frontier = (
      (candidate_tps / current_best_tps - 1.0) * 100.0
      if current_best_tps > 0.0 else 0.0
  )
  attention_front_non_growth = attention_front_delta <= 0.0
  removes_floor_gap = wall_delta <= -floor_gap_ms
  clears_floor = candidate_tps >= 19.5
  event_lifetime_flags_active = (
      candidate["attention_linear_event_lifetime"]
      and candidate["linear_final_device_q8_handoff"]
      and candidate["attention_front_resident_residual_input"]
      and candidate["linear_preconv_input_q8_ms_per_token"] == 0.0
      and candidate["linear_preconv_alpha_beta_ms_per_token"] == 0.0
      and candidate["linear_delta_final_read_ms_per_token"] == 0.0
  )

  checks = [
      {"name": "same_source_sha", "pass": same_source},
      {"name": "same_8_token_shape", "pass": same_tokens},
      {"name": "baseline_top1", "pass": baseline["top1_matches_native"]},
      {"name": "candidate_top1", "pass": candidate["top1_matches_native"]},
      {"name": "baseline_event_lifetime_disabled", "pass": not baseline["attention_linear_event_lifetime"]},
      {"name": "candidate_event_lifetime_flags_active", "pass": event_lifetime_flags_active},
      {"name": "candidate_below_frontier", "pass": candidate_tps < current_best_tps},
      {"name": "candidate_attention_front_grew", "pass": attention_front_delta > 0.0},
  ]
  required = all(item["pass"] for item in checks)

  if required and not attention_front_non_growth:
    disposition = "rejected_combined_event_lifetime_attention_front_growth"
    next_action = (
        "Close the combined event-lifetime alias as a speed path: it removes "
        "linear-delta final read and the preconv host-Q8/alpha-beta walls, but "
        "grows attention-front and qkv/conv enough to miss the floor. Next route "
        "needs a narrower proof that fixes the attention-front device-Q8 chain "
        "or a different dominant bucket."
    )
  elif required and removes_floor_gap and clears_floor:
    disposition = "candidate_event_lifetime_promote_to_confirm"
    next_action = "Run a non-explore confirm plus paired distribution before any speed claim."
  elif required:
    disposition = "rejected_combined_event_lifetime_does_not_clear_floor"
    next_action = "Do not promote; select the next route from the updated profile."
  else:
    disposition = "event_lifetime_decode_evidence_incomplete"
    next_action = "Fix missing/invalid compare evidence before route closure."

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "next_action": next_action,
      "checks": checks,
      "inputs": {
          "explore_log": _display(args.explore_log),
          "baseline_label": args.baseline_label,
          "candidate_label": args.candidate_label,
          "frontier": _display(args.frontier),
          "current_best": _display(args.current_best),
      },
      "frontier": {
          "current_best_tps": current_best_tps,
          "floor_tps": 19.5,
          "floor_gap_ms_per_token": floor_gap_ms,
          "noise_rel": noise_rel,
      },
      "rows": {
          "baseline": baseline,
          "candidate": candidate,
      },
      "derived": {
          "candidate_tps_delta_vs_baseline": candidate_tps - baseline_tps,
          "candidate_tps_delta_pct_vs_baseline": tps_delta_pct,
          "candidate_tps_delta_pct_vs_frontier": tps_delta_pct_vs_frontier,
          "wall_delta_ms_per_token": wall_delta,
          "attention_front_delta_ms_per_token": attention_front_delta,
          "linear_preconv_delta_ms_per_token": (
              candidate["linear_preconv_ms_per_token"] -
              baseline["linear_preconv_ms_per_token"]
          ),
          "linear_preconv_qkv_conv_delta_ms_per_token": (
              candidate["linear_preconv_qkv_conv_ms_per_token"] -
              baseline["linear_preconv_qkv_conv_ms_per_token"]
          ),
          "linear_delta_delta_ms_per_token": (
              candidate["linear_delta_ms_per_token"] -
              baseline["linear_delta_ms_per_token"]
          ),
          "linear_delta_final_read_delta_ms_per_token": (
              candidate["linear_delta_final_read_ms_per_token"] -
              baseline["linear_delta_final_read_ms_per_token"]
          ),
          "attention_front_non_growth": attention_front_non_growth,
          "removes_floor_gap": removes_floor_gap,
          "clears_floor": clears_floor,
          "route_promotable": required and attention_front_non_growth and removes_floor_gap and clears_floor,
      },
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [item["name"] for item in payload["checks"] if item["pass"] is not True]
  derived = payload["derived"]
  rows = payload["rows"]
  lines = [
      "# Attention/Linear Event-Lifetime Decode Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- baseline: `{rows['baseline']['tps']:.8f}` tok/s",
      f"- candidate: `{rows['candidate']['tps']:.8f}` tok/s",
      f"- delta vs baseline: `{derived['candidate_tps_delta_pct_vs_baseline']:.3f}%`",
      f"- wall delta: `{derived['wall_delta_ms_per_token']:.3f}` ms/token",
      f"- attention-front delta: `{derived['attention_front_delta_ms_per_token']:.3f}` ms/token",
      f"- linear-preconv delta: `{derived['linear_preconv_delta_ms_per_token']:.3f}` ms/token",
      f"- route promotable: `{str(derived['route_promotable']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is decode route evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
  parser.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
