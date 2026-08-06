#!/usr/bin/env python3
"""Gate the carrier full-boundary decode evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-hidden-state-carrier-full-boundary-decode-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ122 = ROOT / "output/resident-hidden-state-carrier-selected-shared-tail-target-compile-gate-20260707Tseq122Z/metrics.json"
DEFAULT_DISTRIBUTION = ROOT / "output/r2-gpu-resident-hidden-state-carrier-full-boundary-distribution-20260707Tseq123Z/result.json"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_EXPLORE_LABEL = "resident-hidden-state-carrier-full-boundary-speed-seq124"
DEFAULT_CURRENT_BEST = ROOT / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
DEFAULT_BASELINE_DISTRIBUTION = ROOT / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-distribution-20260705T143408Z/result.json"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-state-carrier-full-boundary-decode-gate-20260707Tseq124Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  value = payload.get("smoke")
  return value if isinstance(value, dict) else payload


def _find_explore(path: Path, label: str) -> dict[str, Any]:
  found: dict[str, Any] | None = None
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("label") == label:
      found = row
  if found is None:
    raise SystemExit(f"explore label not found: {label}")
  return found


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  profile = row.get("profile_smoke")
  profile = profile if isinstance(profile, dict) else row
  wall = profile.get("wall_profile_ns")
  if not isinstance(wall, dict):
    return 0.0
  tokens = _num(row.get("decode_tokens") or profile.get("decode_continuation_output_tokens"))
  return _num(wall.get(stage)) / tokens / 1_000_000.0 if tokens > 0.0 else 0.0


def _tps(row: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = row.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  tokens = _num(row.get("decode_tokens") or row.get("decode_continuation_output_tokens"))
  ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  return tokens * 1_000_000_000.0 / ns if tokens > 0.0 and ns > 0.0 else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq122 = _load_json(args.seq122)
  distribution_payload = _load_json(args.distribution)
  distribution = _smoke(distribution_payload)
  baseline_distribution = _smoke(_load_json(args.baseline_distribution))
  current_best = _smoke(_load_json(args.current_best))
  explore = _find_explore(args.explore_log, args.explore_label)
  frontier_state = _frontier_state(frontier)
  dist = distribution.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}

  current_best_tps = _tps(current_best)
  explore_tps = _tps(explore)
  rel_vs_best = (
      (explore_tps / current_best_tps) - 1.0 if current_best_tps > 0.0 else 0.0
  )
  distribution_tps = _tps(distribution)
  baseline_distribution_tps = _tps(baseline_distribution)
  rel_vs_distribution_baseline = (
      (distribution_tps / baseline_distribution_tps) - 1.0
      if baseline_distribution_tps > 0.0 else 0.0
  )
  noise_rel = frontier_state["noise_rel"]
  selected_delta_ms = _stage_ms(explore, "selected_ffn") - _stage_ms(
      current_best, "selected_ffn")
  ffn_tail_delta_ms = _stage_ms(explore, "ffn_tail") - _stage_ms(
      current_best, "ffn_tail")
  attention_front_delta_ms = _stage_ms(explore, "attention_front") - _stage_ms(
      current_best, "attention_front")

  checks = [
      {
          "name": "seq122_selected_full_boundary_decode_gate",
          "pass": (
              seq122.get("required_checks_passed") is True
              and seq122.get("selected_next_route")
                  == "resident_hidden_state_carrier_full_boundary_decode_gate"
              and _has_switch(
                  routes,
                  "accept_selected_shared_tail_compile_switch_to_full_boundary_decode_gate",
                  122,
              )
          ),
      },
      {
          "name": "distribution_correctness_passed",
          "pass": (
              distribution_payload.get("required_checks_passed") is True
              and distribution.get("top1_matches_native") is True
              and dist.get("required_checks_passed") is True
              and _num(dist.get("max_kld")) < 0.005
              and _num(dist.get("top1_rate")) >= 0.99
          ),
          "detail": {
              "top1": distribution.get("top1_matches_native"),
              "top1_count": distribution.get("top1_match_count"),
              "max_kld": dist.get("max_kld"),
              "top1_rate": dist.get("top1_rate"),
              "min_logits_cosine": dist.get("min_logits_cosine"),
              "logits_cosine_pass": dist.get("logits_cosine_pass"),
          },
      },
      {
          "name": "distribution_uses_full_carrier_stack",
          "pass": (
              distribution.get("resident_hidden_state_carrier_enabled") is True
              and distribution.get(
                  "resident_hidden_state_carrier_preconv_bundle_enabled") is True
              and distribution.get(
                  "resident_hidden_state_carrier_selected_shared_tail_enabled") is True
              and distribution.get("ffn_tail_resident_input_enabled") is False
          ),
      },
      {
          "name": "explore_preserves_top1_and_uses_full_carrier_stack",
          "pass": (
              explore.get("top1_matches_native") is True
              and explore.get("resident_hidden_state_carrier_selected_shared_tail")
                  is True
              and explore.get("ffn_tail_resident_input") is False
          ),
      },
      {
          "name": "explore_regresses_outside_noise",
          "pass": rel_vs_best < -noise_rel,
          "detail": {
              "explore_tps": explore_tps,
              "current_best_tps": current_best_tps,
              "relative_vs_best": rel_vs_best,
              "noise_rel": noise_rel,
          },
      },
      {
          "name": "distribution_lane_regresses_outside_noise",
          "pass": rel_vs_distribution_baseline < -noise_rel,
          "detail": {
              "distribution_tps": distribution_tps,
              "baseline_distribution_tps": baseline_distribution_tps,
              "relative_vs_distribution_baseline": rel_vs_distribution_baseline,
              "noise_rel": noise_rel,
          },
      },
      {
          "name": "selected_gain_is_consumed_by_tail_or_attention_growth",
          "pass": selected_delta_ms < 0.0
          and (ffn_tail_delta_ms + attention_front_delta_ms) > 0.0,
          "detail": {
              "selected_delta_ms_per_token": selected_delta_ms,
              "ffn_tail_delta_ms_per_token": ffn_tail_delta_ms,
              "attention_front_delta_ms_per_token": attention_front_delta_ms,
          },
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_current_carrier_full_boundary_as_speed_cut"
          if required_checks_passed else "carrier_full_boundary_decode_gate_failed"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_tail_growth_root_gate"
          if required_checks_passed else "resident_hidden_state_carrier_decode_review"
      ),
      "next_action": (
          "Do not promote or repeat the current carrier full-boundary decode "
          "shape. Correctness passes, but the no-distribution explore row "
          "regresses outside the noise band and the selected-FFN gain is "
          "consumed by FFN-tail/attention-front growth. The next unit must be a "
          "root-cause gate for carrier tail growth, source/profile first."
          if required_checks_passed
          else "Fix failed carrier decode gate checks before changing route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq122": _rel(args.seq122),
          "distribution": _rel(args.distribution),
          "explore_log": _rel(args.explore_log),
          "explore_label": args.explore_label,
          "current_best": _rel(args.current_best),
          "baseline_distribution": _rel(args.baseline_distribution),
      },
      "frontier": frontier_state,
      "distribution": {
          "tps": distribution_tps,
          "required_checks_passed": distribution_payload.get(
              "required_checks_passed"),
          "max_kld": dist.get("max_kld"),
          "top1_rate": dist.get("top1_rate"),
          "top1_match_count": distribution.get("top1_match_count"),
          "min_logits_cosine": dist.get("min_logits_cosine"),
      },
      "explore": {
          "ts": explore.get("ts"),
          "label": explore.get("label"),
          "source_sha": explore.get("source_sha"),
          "tps": explore_tps,
          "top1_matches_native": explore.get("top1_matches_native"),
          "required_checks_passed": explore.get("required_checks_passed"),
          "failed_checks": explore.get("failed_checks"),
      },
      "comparison": {
          "current_best_tps": current_best_tps,
          "relative_vs_best": rel_vs_best,
          "baseline_distribution_tps": baseline_distribution_tps,
          "relative_vs_distribution_baseline": rel_vs_distribution_baseline,
          "selected_delta_ms_per_token": selected_delta_ms,
          "ffn_tail_delta_ms_per_token": ffn_tail_delta_ms,
          "attention_front_delta_ms_per_token": attention_front_delta_ms,
      },
      "checks": checks,
  }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Full-Boundary Decode Gate",
      "",
      f"- required_checks_passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected_next_route: `{payload['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq122", type=Path, default=DEFAULT_SEQ122)
  parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--explore-label", default=DEFAULT_EXPLORE_LABEL)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument(
      "--baseline-distribution", type=Path, default=DEFAULT_BASELINE_DISTRIBUTION)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(payload, args.out_dir)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
