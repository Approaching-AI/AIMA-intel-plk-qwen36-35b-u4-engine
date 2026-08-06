#!/usr/bin/env python3
"""Gate the Q6 non-atomic down-to-tail decode wiring evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q6-nonatomic-down-tail-decode-gate-v0"
DEFAULT_FRONTIER = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_ONE_TOKEN = (
    ROOT
    / "output/r2-gpu-q6-nonatomic-down-tail-accepted-stack-1tok-20260707Tseq82Z"
)
DEFAULT_PROFILE = (
    ROOT
    / "output/r2-gpu-q6-nonatomic-down-tail-accepted-stack-8tok-profile-20260707Tseq82Z"
)
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_BASELINE_LABEL = "q6-nonatomic-down-tail-current-source-baseline-noqueue-8tok-seq82"
DEFAULT_EXPLORE_LABEL = "q6-nonatomic-down-tail-accepted-stack-noqueue-8tok-seq82"
DEFAULT_OUT_DIR = ROOT / "output/q6-nonatomic-down-tail-decode-gate-20260707Tseq82-r2Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
  result_path = path / "result.json" if path.is_dir() else path
  payload = _load_json(result_path)
  if not isinstance(payload, dict):
    raise SystemExit(f"{result_path}: expected JSON object")
  return payload


def _smoke(path: Path) -> dict[str, Any]:
  payload = _artifact(path)
  smoke = payload.get("smoke")
  if not isinstance(smoke, dict):
    raise SystemExit(f"{path}: missing smoke object")
  return smoke


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
    raise SystemExit(f"{path}: no row label={label!r}")
  return rows[-1]


def _display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _tokens(row: dict[str, Any]) -> float:
  for key in ("decode_tokens", "decode_continuation_output_tokens"):
    value = row.get(key)
    if isinstance(value, (int, float)) and value > 0:
      return float(value)
  return 0.0


def _tps(row: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = row.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  tokens = _tokens(row)
  decode_ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  return tokens * 1_000_000_000.0 / decode_ns if tokens > 0.0 and decode_ns > 0.0 else 0.0


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  profile = row.get("profile_smoke")
  return profile if isinstance(profile, dict) else row


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  wall = _profile(row).get("wall_profile_ns")
  if not isinstance(wall, dict):
    return 0.0
  tokens = _tokens(row)
  return _num(wall.get(stage)) / tokens / 1_000_000.0 if tokens > 0.0 else 0.0


def _noise_rel(path: Path) -> float:
  frontier = _load_json(path)
  no_progress = frontier.get("no_progress") if isinstance(frontier, dict) else None
  noise = no_progress.get("noise") if isinstance(no_progress, dict) else None
  if isinstance(noise, dict):
    value = _num(noise.get("rel"))
  else:
    value = _num(noise)
  return value if 0.0 < value < 1.0 else value / 100.0


def _failed_checks(payload: dict[str, Any]) -> list[str]:
  checks = payload.get("checks")
  if not isinstance(checks, list):
    return []
  failed: list[str] = []
  for check in checks:
    if isinstance(check, dict) and check.get("pass") is not True:
      name = check.get("name")
      if isinstance(name, str):
        failed.append(name)
  return failed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--one-token-artifact", type=Path, default=DEFAULT_ONE_TOKEN)
  parser.add_argument("--profile-artifact", type=Path, default=DEFAULT_PROFILE)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
  parser.add_argument("--explore-label", default=DEFAULT_EXPLORE_LABEL)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  one_result = _artifact(args.one_token_artifact)
  one = _smoke(args.one_token_artifact)
  profile_result = _artifact(args.profile_artifact)
  profile = _smoke(args.profile_artifact)
  baseline = _row_by_label(args.explore_log, args.baseline_label)
  explore = _row_by_label(args.explore_log, args.explore_label)
  current_best = _smoke(args.current_best)

  current_best_tps = _tps(current_best)
  candidate_tps = _tps(explore)
  noise_rel = _noise_rel(args.frontier)
  delta_pct_vs_frontier = (
      (candidate_tps / current_best_tps - 1.0) * 100.0
      if current_best_tps > 0.0
      else 0.0
  )
  baseline_tps = _tps(baseline)
  delta_pct_vs_same_source = (
      (candidate_tps / baseline_tps - 1.0) * 100.0 if baseline_tps > 0.0 else 0.0
  )

  profile_failed = _failed_checks(profile_result)
  expected_profile_failures = {
      "target_binary_ran",
      "gpu_decode_required_checks_passed",
  }
  checks = [
      {
          "name": "one_token_required_checks_passed",
          "pass": one_result.get("required_checks_passed") is True,
      },
      {"name": "one_token_top1", "pass": one.get("top1_matches_native") is True},
      {"name": "one_token_topk", "pass": one.get("topk_ids_match_native") is True},
      {
          "name": "one_token_nonatomic_enabled",
          "pass": one.get("selected_shared_q6_down_tail_nonatomic_enabled") is True,
      },
      {
          "name": "one_token_nonatomic_layers_20",
          "pass": one.get("selected_shared_q6_down_tail_nonatomic_layers") == 20,
      },
      {"name": "profile_top1", "pass": profile.get("top1_matches_native") is True},
      {"name": "profile_top1_count_8", "pass": profile.get("top1_match_count") == 8},
      {
          "name": "profile_expected_free_run_topk_failure_only",
          "pass": set(profile_failed) == expected_profile_failures,
      },
      {
          "name": "profile_nonatomic_enabled",
          "pass": profile.get("selected_shared_q6_down_tail_nonatomic_enabled") is True,
      },
      {
          "name": "profile_nonatomic_layers_160",
          "pass": profile.get("selected_shared_q6_down_tail_nonatomic_layers") == 160,
      },
      {
          "name": "profile_nonatomic_kernel_timing_positive",
          "pass": _num(profile.get("selected_shared_q6_down_tail_nonatomic_shell_kernel_us"))
          > 0.0,
      },
      {
          "name": "explore_top1",
          "pass": explore.get("top1_matches_native") is True,
      },
      {
          "name": "baseline_top1",
          "pass": baseline.get("top1_matches_native") is True,
      },
      {
          "name": "baseline_nonatomic_disabled",
          "pass": baseline.get("selected_shared_q6_down_tail_nonatomic") is False,
      },
      {
          "name": "explore_nonatomic_enabled",
          "pass": explore.get("selected_shared_q6_down_tail_nonatomic") is True,
      },
      {
          "name": "explore_accepted_stack_defer_enabled",
          "pass": explore.get("defer_ffn_down_finish_bundle") is True,
      },
      {
          "name": "explore_nonatomic_layers_160",
          "pass": explore.get("selected_shared_q6_down_tail_nonatomic_layers") == 160,
      },
  ]
  required_checks_passed = all(item["pass"] for item in checks)

  disposition = (
      "rejected_q6_nonatomic_down_tail_decode_regresses_8tok"
      if candidate_tps < current_best_tps * (1.0 - noise_rel)
      else "q6_nonatomic_down_tail_decode_inside_noise_or_candidate"
  )

  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "speedup_claims_allowed": False,
      "one_token_artifact": _display(args.one_token_artifact),
      "profile_artifact": _display(args.profile_artifact),
      "explore_row": f"{_display(args.explore_log)}#{explore.get('ts')}",
      "current_best_artifact": _display(args.current_best),
      "current_best_tps": current_best_tps,
      "same_source_baseline_row": f"{_display(args.explore_log)}#{baseline.get('ts')}",
      "same_source_baseline_tps": baseline_tps,
      "candidate_tps": candidate_tps,
      "delta_pct_vs_same_source_baseline": delta_pct_vs_same_source,
      "delta_pct_vs_frontier": delta_pct_vs_frontier,
      "frontier_noise_rel": noise_rel,
      "profile_top1_count": profile.get("top1_match_count"),
      "profile_topk_count": profile.get("topk_match_count"),
      "profile_failed_checks": profile_failed,
      "nonatomic_layers_profile": profile.get(
          "selected_shared_q6_down_tail_nonatomic_layers"
      ),
      "nonatomic_contrib_kernel_us_profile": profile.get(
          "selected_shared_q6_down_tail_nonatomic_contrib_kernel_us"
      ),
      "nonatomic_reduce_kernel_us_profile": profile.get(
          "selected_shared_q6_down_tail_nonatomic_reduce_kernel_us"
      ),
      "nonatomic_shell_kernel_us_profile": profile.get(
          "selected_shared_q6_down_tail_nonatomic_shell_kernel_us"
      ),
      "candidate_selected_ffn_ms_per_token": _stage_ms(explore, "selected_ffn"),
      "candidate_ffn_tail_ms_per_token": _stage_ms(explore, "ffn_tail"),
      "candidate_shared_ffn_ms_per_token": _stage_ms(explore, "shared_ffn"),
      "baseline_selected_ffn_ms_per_token": _stage_ms(baseline, "selected_ffn"),
      "baseline_ffn_tail_ms_per_token": _stage_ms(baseline, "ffn_tail"),
      "baseline_shared_ffn_ms_per_token": _stage_ms(baseline, "shared_ffn"),
      "current_best_selected_ffn_ms_per_token": _stage_ms(current_best, "selected_ffn"),
      "current_best_ffn_tail_ms_per_token": _stage_ms(current_best, "ffn_tail"),
      "current_best_shared_ffn_ms_per_token": _stage_ms(current_best, "shared_ffn"),
      "checks": checks,
  }

  args.out_dir.mkdir(parents=True, exist_ok=False)
  (args.out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [item["name"] for item in checks if not item["pass"]]
  summary = [
      "# Q6 Non-Atomic Down-Tail Decode Gate",
      "",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      f"- disposition: `{disposition}`",
      f"- one-token artifact: `{metrics['one_token_artifact']}`",
      f"- profile artifact: `{metrics['profile_artifact']}`",
      f"- explore row: `{metrics['explore_row']}`",
      f"- same-source baseline row: `{metrics['same_source_baseline_row']}`",
      f"- candidate tok/s: `{candidate_tps}`",
      f"- same-source baseline tok/s: `{baseline_tps}`",
      f"- current best tok/s: `{current_best_tps}`",
      f"- delta vs same-source baseline: `{delta_pct_vs_same_source:.6f}%`",
      f"- delta vs frontier: `{delta_pct_vs_frontier:.6f}%`",
      f"- non-atomic profile layers: `{metrics['nonatomic_layers_profile']}`",
      f"- non-atomic shell kernel us: `{metrics['nonatomic_shell_kernel_us_profile']}`",
      f"- failed checks: `{failed}`",
      "",
      "This gate validates the default-off decode wiring evidence. It is not",
      "promotion evidence because the 8-token row regresses versus the accepted",
      "frontier and exact free-run top-k remains diagnostic-only.",
      "",
  ]
  (args.out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
