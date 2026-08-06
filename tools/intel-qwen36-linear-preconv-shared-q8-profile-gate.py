#!/usr/bin/env python3
"""Gate the shared-Q8 linear-preconv decode path with comparable 8-token rows.

This is route-selection evidence, not benchmark promotion evidence. It compares
the current-source accepted-stack baseline against the opt-in shared-Q8
preconv row, then checks whether the removed host-Q8 bridge actually improves
the token-emitting lane outside the frontier noise band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-linear-preconv-shared-q8-profile-gate-v0"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_FRONTIER = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_BASELINE_LABEL = "seq77-current-source-baseline-z-8tok"
DEFAULT_SHARED_Q8_LABEL = "seq77-shared-q8-preconv-8tok-profile"
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
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
  matches = [row for row in _iter_jsonl(path) if row.get("label") == label]
  if not matches:
    raise SystemExit(f"{path}: no row label={label!r}")
  return matches[-1]


def _smoke_artifact(path: Path) -> dict[str, Any]:
  smoke_path = path / "smoke.json" if path.is_dir() else path
  if not smoke_path.is_file():
    raise SystemExit(f"{smoke_path}: missing smoke artifact")
  payload = _load_json(smoke_path)
  if not isinstance(payload, dict):
    raise SystemExit(f"{smoke_path}: expected JSON object")
  return payload


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _tokens(payload: dict[str, Any]) -> float:
  for key in ("decode_tokens", "decode_continuation_output_tokens"):
    value = payload.get(key)
    if isinstance(value, (int, float)) and value > 0:
      return float(value)
  profile = payload.get("profile_smoke")
  if isinstance(profile, dict):
    return _tokens(profile)
  value = payload.get("decode_tokens_per_session")
  if isinstance(value, (int, float)) and value > 0:
    return float(value)
  raise SystemExit(f"row {payload.get('label') or payload.get('ts')} lacks token count")


def _tps(payload: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = payload.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  decode_ns = _num(payload.get("decode_ns") or payload.get("gpu_hybrid_decode_ns"))
  tokens = _tokens(payload)
  return tokens * 1_000_000_000.0 / decode_ns if decode_ns > 0.0 else 0.0


def _profile(payload: dict[str, Any]) -> dict[str, Any]:
  profile = payload.get("profile_smoke")
  if isinstance(profile, dict):
    return profile
  return payload


def _wall_ms(payload: dict[str, Any]) -> float:
  decode_ns = _num(payload.get("decode_ns") or payload.get("gpu_hybrid_decode_ns"))
  return decode_ns / _tokens(payload) / 1_000_000.0 if decode_ns > 0.0 else 0.0


def _stage_ms(payload: dict[str, Any], stage: str) -> float:
  wall = _profile(payload).get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {payload.get('label') or payload.get('ts')} lacks wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(payload) / 1_000_000.0


def _preconv_ms(payload: dict[str, Any], substage: str) -> float:
  wall = _profile(payload).get("linear_preconv_wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(
        f"row {payload.get('label') or payload.get('ts')} lacks linear preconv profile"
    )
  return _num(wall.get(substage)) / _tokens(payload) / 1_000_000.0


def _host_q8_us(payload: dict[str, Any]) -> float:
  kernel = _profile(payload).get("linear_preconv_kernel_profile_us")
  if not isinstance(kernel, dict):
    return 0.0
  return _num(kernel.get("host_q8_bridge"))


def _noise_rel(path: Path) -> float:
  frontier = _load_json(path)
  no_progress = frontier.get("no_progress") if isinstance(frontier, dict) else None
  noise = no_progress.get("noise") if isinstance(no_progress, dict) else None
  if isinstance(noise, dict):
    return _num(noise.get("rel"))
  value = _num(noise)
  return value if 0.0 < value < 1.0 else value / 100.0


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
  return {
      "ts": payload.get("ts"),
      "label": payload.get("label"),
      "source_sha": payload.get("source_sha"),
      "config_sha": payload.get("config_sha"),
      "decode_tokens": _tokens(payload),
      "tps": _tps(payload),
      "wall_ms_per_token": _wall_ms(payload),
      "top1_matches_native": bool(payload.get("top1_matches_native")),
      "required_checks_passed": bool(payload.get("required_checks_passed")),
      "defer_ffn_down_finish_bundle": bool(payload.get("defer_ffn_down_finish_bundle")),
      "reuse_selected_q8_for_shared_ffn": bool(payload.get("reuse_selected_q8_for_shared_ffn")),
      "linear_preconv_ms_per_token": _stage_ms(payload, "linear_preconv"),
      "linear_preconv_input_q8_ms_per_token": _preconv_ms(payload, "input_q8"),
      "linear_preconv_alpha_beta_ms_per_token": _preconv_ms(payload, "alpha_beta"),
      "linear_preconv_qkv_conv_ms_per_token": _preconv_ms(payload, "qkv_conv"),
      "linear_preconv_postconv_prep_ms_per_token": _preconv_ms(payload, "postconv_prep"),
      "linear_preconv_host_q8_bridge_us": _host_q8_us(payload),
      "q4_cpu_order_z_ms_per_token": _stage_ms(payload, "q4_cpu_order_z"),
      "attention_front_ms_per_token": _stage_ms(payload, "attention_front"),
      "selected_ffn_ms_per_token": _stage_ms(payload, "selected_ffn"),
      "ffn_tail_ms_per_token": _stage_ms(payload, "ffn_tail"),
      "full_core_ms_per_token": _stage_ms(payload, "full_core"),
  }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
  return _num(candidate.get(key)) - _num(baseline.get(key))


def compute(args: argparse.Namespace) -> dict[str, Any]:
  baseline_row = _row_by_label(args.explore_log, args.baseline_label)
  candidate_row = _row_by_label(args.explore_log, args.shared_q8_label)
  baseline = _summary(baseline_row)
  candidate = _summary(candidate_row)
  current_best = _smoke_artifact(args.current_best)
  current_best_tps = _tps(current_best)
  noise_rel = _noise_rel(args.frontier)

  same_source = (
      baseline.get("source_sha")
      and candidate.get("source_sha")
      and baseline.get("source_sha") == candidate.get("source_sha")
  )
  same_tokens = baseline["decode_tokens"] == candidate["decode_tokens"] == 8.0
  candidate_tps = candidate["tps"]
  baseline_tps = baseline["tps"]
  candidate_delta_pct = (
      (candidate_tps / baseline_tps - 1.0) * 100.0 if baseline_tps > 0 else 0.0
  )
  best_delta_pct = (
      (candidate_tps / current_best_tps - 1.0) * 100.0
      if current_best_tps > 0 else 0.0
  )
  below_baseline_noise = candidate_tps < baseline_tps * (1.0 - noise_rel)
  below_current_best = candidate_tps < current_best_tps
  host_bridge_removed = (
      candidate["linear_preconv_host_q8_bridge_us"] == 0.0
      and candidate["linear_preconv_input_q8_ms_per_token"] == 0.0
  )

  deltas = {
      "tps_delta_vs_current_source_baseline": candidate_tps - baseline_tps,
      "tps_delta_pct_vs_current_source_baseline": candidate_delta_pct,
      "tps_delta_vs_accepted_frontier": candidate_tps - current_best_tps,
      "tps_delta_pct_vs_accepted_frontier": best_delta_pct,
      "wall_ms_per_token": _delta(candidate, baseline, "wall_ms_per_token"),
      "linear_preconv_ms_per_token": _delta(
          candidate, baseline, "linear_preconv_ms_per_token"
      ),
      "linear_preconv_input_q8_ms_per_token": _delta(
          candidate, baseline, "linear_preconv_input_q8_ms_per_token"
      ),
      "linear_preconv_alpha_beta_ms_per_token": _delta(
          candidate, baseline, "linear_preconv_alpha_beta_ms_per_token"
      ),
      "linear_preconv_qkv_conv_ms_per_token": _delta(
          candidate, baseline, "linear_preconv_qkv_conv_ms_per_token"
      ),
      "linear_preconv_postconv_prep_ms_per_token": _delta(
          candidate, baseline, "linear_preconv_postconv_prep_ms_per_token"
      ),
      "q4_cpu_order_z_ms_per_token": _delta(
          candidate, baseline, "q4_cpu_order_z_ms_per_token"
      ),
      "attention_front_ms_per_token": _delta(
          candidate, baseline, "attention_front_ms_per_token"
      ),
      "selected_ffn_ms_per_token": _delta(
          candidate, baseline, "selected_ffn_ms_per_token"
      ),
      "ffn_tail_ms_per_token": _delta(candidate, baseline, "ffn_tail_ms_per_token"),
      "full_core_ms_per_token": _delta(candidate, baseline, "full_core_ms_per_token"),
  }

  close_as_speed_route = (
      bool(same_source)
      and same_tokens
      and candidate["top1_matches_native"]
      and host_bridge_removed
      and below_baseline_noise
      and below_current_best
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "explore_log": _display_path(args.explore_log),
          "baseline_label": args.baseline_label,
          "shared_q8_label": args.shared_q8_label,
          "current_best": _display_path(args.current_best),
          "frontier": _display_path(args.frontier),
      },
      "frontier": {
          "current_best_tps": current_best_tps,
          "noise_rel": noise_rel,
          "noise_pct": noise_rel * 100.0,
      },
      "rows": {
          "current_source_baseline": baseline,
          "shared_q8_candidate": candidate,
      },
      "checks": [
          {"label": "same_source_sha", "pass": bool(same_source)},
          {"label": "same_8_token_shape", "pass": same_tokens},
          {
              "label": "shared_q8_top1_matches_native",
              "pass": candidate["top1_matches_native"],
          },
          {
              "label": "shared_q8_host_q8_bridge_removed",
              "pass": host_bridge_removed,
          },
          {
              "label": "shared_q8_below_baseline_beyond_noise",
              "pass": below_baseline_noise,
              "detail": {
                  "candidate_tps": candidate_tps,
                  "baseline_tps": baseline_tps,
                  "noise_rel": noise_rel,
              },
          },
          {
              "label": "shared_q8_below_accepted_frontier",
              "pass": below_current_best,
              "detail": {
                  "candidate_tps": candidate_tps,
                  "current_best_tps": current_best_tps,
              },
          },
      ],
      "derived": deltas,
      "verdict": {
          "speedup_claims_allowed": False,
          "shared_q8_profile_closes_speed_route": close_as_speed_route,
          "reason": (
              "The opt-in shared-Q8 path removes the linear-preconv host-Q8 "
              "bridge, but on the same current-source 8-token stack it loses "
              f"{abs(candidate_delta_pct):.3f}% versus baseline and remains "
              "below the accepted frontier."
          ),
          "next_route": (
              "Do not spend more on shared-Q8 preconv unless a specific "
              "qkv_conv regression fix is identified; switch to DPAS beyond "
              "the current Q4 occupancy bounds or a materially different "
              "non-atomic down-to-tail design."
          ),
      },
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "metrics": "metrics.json",
      "speedup_claims_allowed": False,
      "shared_q8_profile_closes_speed_route": metrics["verdict"][
          "shared_q8_profile_closes_speed_route"
      ],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  derived = metrics["derived"]
  rows = metrics["rows"]
  summary = [
      "# Shared-Q8 Linear-Preconv Profile Gate",
      "",
      f"- baseline row: `{rows['current_source_baseline']['label']}` "
      f"({rows['current_source_baseline']['tps']:.8f} tok/s)",
      f"- shared-Q8 row: `{rows['shared_q8_candidate']['label']}` "
      f"({rows['shared_q8_candidate']['tps']:.8f} tok/s)",
      f"- delta vs baseline: `{derived['tps_delta_pct_vs_current_source_baseline']:.3f}%`",
      f"- delta vs accepted frontier: `{derived['tps_delta_pct_vs_accepted_frontier']:.3f}%`",
      f"- linear-preconv delta: `{derived['linear_preconv_ms_per_token']:.3f}` ms/token",
      f"- qkv/conv delta: `{derived['linear_preconv_qkv_conv_ms_per_token']:.3f}` ms/token",
      f"- verdict closes speed route: "
      f"`{str(metrics['verdict']['shared_q8_profile_closes_speed_route']).lower()}`",
      "",
      metrics["verdict"]["reason"],
      "",
      metrics["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
  parser.add_argument("--shared-q8-label", default=DEFAULT_SHARED_Q8_LABEL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()

  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
