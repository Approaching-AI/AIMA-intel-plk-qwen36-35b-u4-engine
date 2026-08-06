#!/usr/bin/env python3
"""Gate the carrier tail-growth root before another token row.

This is route-control evidence only. It consumes the seq124 carrier decode
gate, the paired distribution rows, existing local-carrier/down-tail closures,
and the current source shape. It does not run the target and does not create
speed evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-hidden-state-carrier-tail-growth-root-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ124 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-boundary-decode-gate-20260707Tseq124Z/metrics.json"
)
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_BASELINE_DISTRIBUTION = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-distribution-20260705T143408Z/result.json"
)
DEFAULT_CARRIER_DISTRIBUTION = (
    ROOT
    / "output/r2-gpu-resident-hidden-state-carrier-full-boundary-distribution-20260707Tseq123Z/result.json"
)
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ63_TS = "20260706T094825Z"
DEFAULT_SEQ65_TS = "20260706T102433Z"
DEFAULT_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_SEQ66 = ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z/metrics.json"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-tail-growth-root-gate-20260707Tseq125Z"
)


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


def _tokens(row: dict[str, Any]) -> float:
  for key in ("decode_tokens", "decode_continuation_output_tokens"):
    value = row.get(key)
    if isinstance(value, (int, float)) and value > 0:
      return float(value)
  profile = row.get("profile_smoke")
  if isinstance(profile, dict):
    return _tokens(profile)
  return 0.0


def _tps(row: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = row.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  tokens = _tokens(row)
  ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  return tokens * 1_000_000_000.0 / ns if tokens > 0.0 and ns > 0.0 else 0.0


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  profile = row.get("profile_smoke")
  return profile if isinstance(profile, dict) else row


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  profile = _profile(row)
  wall = profile.get("wall_profile_ns")
  if not isinstance(wall, dict):
    return 0.0
  tokens = _tokens(row)
  return _num(wall.get(stage)) / tokens / 1_000_000.0 if tokens > 0 else 0.0


def _find_explore(path: Path, *, ts: str) -> dict[str, Any]:
  found: dict[str, Any] | None = None
  for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    if isinstance(row, dict) and row.get("ts") == ts:
      found = row
  if found is None:
    raise SystemExit(f"{path}: no explore row ts={ts}")
  return found


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
  goal_budget = frontier.get("goal_budget")
  goal_budget = goal_budget if isinstance(goal_budget, dict) else {}
  verdict = goal_budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  per_token = goal_budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(goal_anchor.get("current_best_tps")),
      "floor_tps": _num(goal_anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"
      ),
  }


def _source_shape(path: Path, engine_source: Path, engine_header: Path) -> dict[str, Any]:
  text = "\n".join([
      path.read_text(encoding="utf-8"),
      engine_source.read_text(encoding="utf-8"),
      engine_header.read_text(encoding="utf-8"),
  ])
  resident_tail_call = re.search(
      r"RunFfnTailFromDownHandlesResidentInputs\(\s*"
      r"shared_gate_handle,\s*ffn_input_handle,\s*"
      r"selected_gpu\.down_handle,\s*router\.normalized_weights,\s*"
      r"shared_gpu\.down_handle,\s*attention_residual_handle,\s*"
      r"kHiddenSize,\s*kExpertUsedCount,\s*repeat\)",
      text,
      re.S,
  )
  no_readback_tail_call = re.search(
      r"RunFfnTailFromDownHandlesResidentInputs\([^;]*"
      r"kHiddenSize,\s*kExpertUsedCount,\s*repeat,\s*false\)",
      text,
      re.S,
  )
  keep_prev_includes_carrier = (
      "DecodeKeepPrevLayerOutputHandle" in text
      and "g_decode_resident_hidden_state_carrier_selected_shared_tail" in text
  )
  rmsnorm_consumer_is_flag_only = re.search(
      r"g_decode_resident_tail_output_rmsnorm_input\s*\?\s*"
      r"g_decode_prev_layer_output_handle\s*:\s*0",
      text,
  )
  attention_consumer_is_flag_only = re.search(
      r"g_decode_attention_front_resident_residual_input\s*\?\s*"
      r"g_decode_prev_layer_output_handle\s*:\s*0",
      text,
  )
  return {
      "source": _rel(path),
      "engine_source": _rel(engine_source),
      "engine_header": _rel(engine_header),
      "carrier_tail_uses_resident_input_primitive": resident_tail_call is not None,
      "carrier_tail_passes_no_readback_false": no_readback_tail_call is not None,
      "tail_helper_returns_host_layer_output": (
          "return std::move(tail_gpu.layer_output);" in text
      ),
      "resident_tail_primitive_has_default_readback": (
          "RunFfnTailFromDownHandlesResidentInputs(" in text
          and "bool readback_layer_output = true" in text
      ),
      "keep_prev_layer_output_includes_carrier_tail": (
          keep_prev_includes_carrier
      ),
      "prev_handle_next_rmsnorm_consumer_requires_existing_flag": (
          rmsnorm_consumer_is_flag_only is not None
      ),
      "prev_handle_attention_residual_consumer_requires_existing_flag": (
          attention_consumer_is_flag_only is not None
      ),
  }


def _closed_route_names(rejected: dict[str, Any], *needles: str) -> list[str]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return []
  lower_needles = tuple(needle.lower() for needle in needles)
  out: list[str] = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("route", "class", "reason", "runtime_cleanup")
    ).lower()
    if any(needle in haystack for needle in lower_needles):
      route = row.get("route")
      if isinstance(route, str):
        out.append(route)
  return out


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _frontier_state(_load_json(args.frontier))
  rejected = _load_json(args.rejected)
  seq124 = _load_json(args.seq124)
  current_best = _smoke(_load_json(args.current_best))
  baseline_dist = _smoke(_load_json(args.baseline_distribution))
  carrier_dist_payload = _load_json(args.carrier_distribution)
  carrier_dist = _smoke(carrier_dist_payload)
  seq63 = _find_explore(args.explore_log, ts=args.seq63_ts)
  seq65 = _find_explore(args.explore_log, ts=args.seq65_ts)
  seq54 = _load_json(args.seq54)
  seq66 = _load_json(args.seq66)
  source = _source_shape(args.source, args.engine_source, args.engine_header)

  cmp124 = seq124.get("comparison")
  cmp124 = cmp124 if isinstance(cmp124, dict) else {}
  selected_delta = _num(cmp124.get("selected_delta_ms_per_token"))
  ffn_tail_delta = _num(cmp124.get("ffn_tail_delta_ms_per_token"))
  attention_delta = _num(cmp124.get("attention_front_delta_ms_per_token"))
  selected_tail_delta = selected_delta + ffn_tail_delta
  selected_tail_savings = -selected_tail_delta

  dist_selected_delta = (
      _stage_ms(carrier_dist, "selected_ffn")
      - _stage_ms(baseline_dist, "selected_ffn")
  )
  dist_tail_delta = (
      _stage_ms(carrier_dist, "ffn_tail")
      - _stage_ms(baseline_dist, "ffn_tail")
  )
  dist_attention_delta = (
      _stage_ms(carrier_dist, "attention_front")
      - _stage_ms(baseline_dist, "attention_front")
  )
  seq63_tail_delta = _stage_ms(seq63, "ffn_tail") - _stage_ms(
      current_best, "ffn_tail"
  )
  seq65_tail_delta = _stage_ms(seq65, "ffn_tail") - _stage_ms(
      current_best, "ffn_tail"
  )
  floor_gap = frontier["floor_gap_ms_per_token"]

  closed_local_tail = _closed_route_names(
      rejected,
      "ffn_tail_resident_input",
      "tail-output",
      "resident residual input",
      "read-as-drain",
  )
  closed_down_tail = _closed_route_names(
      rejected,
      "down-tail",
      "routed_down_fusion",
      "rowgroup",
      "non-atomic",
      "atomic",
  )

  checks = [
      {
          "name": "seq124_decode_gate_valid_negative",
          "pass": seq124.get("required_checks_passed") is True
          and seq124.get("disposition")
          == "reject_current_carrier_full_boundary_as_speed_cut"
          and seq124.get("selected_next_route")
          == "resident_hidden_state_carrier_tail_growth_root_gate",
      },
      {
          "name": "paired_distribution_confirms_same_growth_shape",
          "pass": (
              carrier_dist_payload.get("required_checks_passed") is True
              and dist_tail_delta > floor_gap
              and abs(dist_tail_delta - ffn_tail_delta) <= 0.15
              and abs(dist_attention_delta - attention_delta) <= 0.15
          ),
          "detail": {
              "speed_lane_ffn_tail_delta_ms_per_token": ffn_tail_delta,
              "distribution_ffn_tail_delta_ms_per_token": dist_tail_delta,
              "speed_lane_attention_delta_ms_per_token": attention_delta,
              "distribution_attention_delta_ms_per_token": dist_attention_delta,
          },
      },
      {
          "name": "ffn_tail_growth_matches_existing_resident_input_tail_class",
          "pass": (
              ffn_tail_delta > floor_gap
              and seq63_tail_delta > floor_gap
              and seq65_tail_delta > floor_gap
              and ffn_tail_delta < seq63_tail_delta
          ),
          "detail": {
              "carrier_ffn_tail_delta_ms_per_token": ffn_tail_delta,
              "seq63_ffn_tail_delta_ms_per_token": seq63_tail_delta,
              "seq65_ffn_tail_delta_ms_per_token": seq65_tail_delta,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "selected_tail_savings_alone_do_not_clear_floor",
          "pass": 0.0 < selected_tail_savings < floor_gap,
          "detail": {
              "selected_delta_ms_per_token": selected_delta,
              "ffn_tail_delta_ms_per_token": ffn_tail_delta,
              "selected_tail_savings_ms_per_token": selected_tail_savings,
              "floor_gap_ms_per_token": floor_gap,
          },
      },
      {
          "name": "source_still_reads_back_tail_output_on_carrier_path",
          "pass": (
              source["carrier_tail_uses_resident_input_primitive"]
              and source["resident_tail_primitive_has_default_readback"]
              and not source["carrier_tail_passes_no_readback_false"]
              and source["tail_helper_returns_host_layer_output"]
          ),
          "detail": source,
      },
      {
          "name": "prev_layer_handle_retention_is_not_a_full_consumer_chain",
          "pass": (
              source["keep_prev_layer_output_includes_carrier_tail"]
              and source["prev_handle_next_rmsnorm_consumer_requires_existing_flag"]
              and source[
                  "prev_handle_attention_residual_consumer_requires_existing_flag"
              ]
          ),
          "detail": source,
      },
      {
          "name": "local_tail_and_down_tail_alternates_are_closed",
          "pass": (
              bool(closed_local_tail)
              and bool(closed_down_tail)
              and bool(
                  seq54.get("derived", {}).get(
                      "resident_hidden_state_carrier_or_down_tail_fusion_required"
                  )
              )
              and bool(
                  seq66.get("derived", {}).get(
                      "hidden_row_serial_q6_down_tail_fusion_closed"
                  )
              )
          ),
          "detail": {
              "closed_local_tail_routes": closed_local_tail,
              "closed_down_tail_routes": closed_down_tail,
              "seq54_evidence": _rel(args.seq54),
              "seq66_evidence": _rel(args.seq66),
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  selected_next_route = (
      "resident_hidden_state_carrier_layer_output_handle_loop_contract_gate"
      if required else "resident_hidden_state_carrier_tail_growth_manual_review"
  )
  next_action = (
      "Do not launch another carrier token row from the current selected/shared "
      "tail shape. The growth root is not missing selected/shared handle wiring: "
      "the path still drains FFN-tail into a host layer-output vector, and the "
      "retained previous-layer handle is not yet a full next-layer consumer "
      "chain. The next unit must be a source/design contract for a handle-carried "
      "layer-output loop that can make tail readback optional only when next-layer "
      "RMSNorm, attention residual, preconv, router/FFN input, and final LM-head "
      "handoff have valid handle consumers. Do not reopen local tail-input or "
      "down-tail fusion variants without a new parallelism-preserving component "
      "proof."
      if required else
      "Fix the failed root-cause evidence before selecting another carrier row."
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "carrier_tail_growth_bound_to_host_layer_output_drain"
          if required else "carrier_tail_growth_root_evidence_incomplete"
      ),
      "selected_next_route": selected_next_route,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "rejected": _rel(args.rejected),
          "seq124": _rel(args.seq124),
          "current_best": _rel(args.current_best),
          "baseline_distribution": _rel(args.baseline_distribution),
          "carrier_distribution": _rel(args.carrier_distribution),
          "explore_log": _rel(args.explore_log),
          "seq63_ts": args.seq63_ts,
          "seq65_ts": args.seq65_ts,
          "source": _rel(args.source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "seq54": _rel(args.seq54),
          "seq66": _rel(args.seq66),
      },
      "frontier": frontier,
      "speed_lane": {
          "current_best_tps": _tps(current_best),
          "carrier_tps": _num(seq124.get("explore", {}).get("tps"))
          if isinstance(seq124.get("explore"), dict) else 0.0,
          "selected_delta_ms_per_token": selected_delta,
          "ffn_tail_delta_ms_per_token": ffn_tail_delta,
          "attention_front_delta_ms_per_token": attention_delta,
          "selected_tail_delta_ms_per_token": selected_tail_delta,
          "selected_tail_savings_ms_per_token": selected_tail_savings,
      },
      "distribution_lane": {
          "baseline_tps": _tps(baseline_dist),
          "carrier_tps": _tps(carrier_dist),
          "selected_delta_ms_per_token": dist_selected_delta,
          "ffn_tail_delta_ms_per_token": dist_tail_delta,
          "attention_front_delta_ms_per_token": dist_attention_delta,
      },
      "local_carrier_closure_rows": {
          "seq63": {
              "ts": seq63.get("ts"),
              "label": seq63.get("label"),
              "tps": _tps(seq63),
              "ffn_tail_delta_ms_per_token": seq63_tail_delta,
          },
          "seq65": {
              "ts": seq65.get("ts"),
              "label": seq65.get("label"),
              "tps": _tps(seq65),
              "ffn_tail_delta_ms_per_token": seq65_tail_delta,
          },
      },
      "source_shape": source,
      "checks": checks,
  }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  speed = payload["speed_lane"]
  dist = payload["distribution_lane"]
  lines = [
      "# Resident Hidden-State Carrier Tail-Growth Root Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- speed-lane FFN-tail delta: `{speed['ffn_tail_delta_ms_per_token']:.3f}` ms/token",
      f"- speed-lane attention-front delta: `{speed['attention_front_delta_ms_per_token']:.3f}` ms/token",
      f"- selected+tail savings: `{speed['selected_tail_savings_ms_per_token']:.3f}` ms/token",
      f"- distribution FFN-tail delta: `{dist['ffn_tail_delta_ms_per_token']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq124", type=Path, default=DEFAULT_SEQ124)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument(
      "--baseline-distribution", type=Path, default=DEFAULT_BASELINE_DISTRIBUTION)
  parser.add_argument(
      "--carrier-distribution", type=Path, default=DEFAULT_CARRIER_DISTRIBUTION)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--seq63-ts", default=DEFAULT_SEQ63_TS)
  parser.add_argument("--seq65-ts", default=DEFAULT_SEQ65_TS)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
  parser.add_argument("--seq54", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--seq66", type=Path, default=DEFAULT_SEQ66)
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
