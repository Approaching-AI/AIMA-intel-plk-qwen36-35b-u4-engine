#!/usr/bin/env python3
"""Gate selected/shared down-to-tail fusion shape before kernel work.

This is route-selection arithmetic over existing artifacts and source shape. It
does not benchmark or authorize a speed claim. The gate asks whether the next
down-to-tail route can be a naive hidden-row fused kernel, or whether it must
preserve the current per-expert Q6 down parallelism and change the reduction /
drain ownership instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-down-tail-fusion-budget-v0"
DEFAULT_FRONTIER = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)
DEFAULT_CURRENT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
)
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_SEQ63_TS = "20260706T094825Z"
DEFAULT_SEQ65_TS = "20260706T102433Z"
DEFAULT_OUT_DIR = ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z"


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


def _tokens(payload: dict[str, Any]) -> float:
  for key in ("decode_tokens", "decode_continuation_output_tokens"):
    value = payload.get(key)
    if isinstance(value, (int, float)) and value > 0:
      return float(value)
  profile = payload.get("profile_smoke")
  if isinstance(profile, dict):
    return _tokens(profile)
  raise SystemExit(f"row {payload.get('ts')} lacks positive token count")


def _tps(payload: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = payload.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  decode_ns = _num(payload.get("decode_ns") or payload.get("gpu_hybrid_decode_ns"))
  tokens = _tokens(payload)
  return tokens * 1_000_000_000.0 / decode_ns if decode_ns > 0.0 else 0.0


def _wall_ms_per_token(payload: dict[str, Any]) -> float:
  for key in ("decode_ns", "gpu_hybrid_decode_ns"):
    value = _num(payload.get(key))
    if value > 0:
      return value / _tokens(payload) / 1_000_000.0
  tps = _tps(payload)
  return 1000.0 / tps if tps > 0.0 else 0.0


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  profile = row.get("profile_smoke")
  if not isinstance(profile, dict):
    raise SystemExit(f"row {row.get('ts')} lacks profile_smoke")
  return profile


def _stage_ms(row: dict[str, Any], stage: str) -> float:
  wall = _profile(row).get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit(f"row {row.get('ts')} lacks wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(row) / 1_000_000.0


def _selected_ms(row: dict[str, Any], substage: str) -> float:
  selected = _profile(row).get("selected_ffn_wall_profile_ns")
  if not isinstance(selected, dict):
    raise SystemExit(f"row {row.get('ts')} lacks selected_ffn_wall_profile_ns")
  return _num(selected.get(substage)) / _tokens(row) / 1_000_000.0


def _best_stage_ms(smoke: dict[str, Any], stage: str) -> float:
  wall = smoke.get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit("current best lacks wall_profile_ns")
  return _num(wall.get(stage)) / _tokens(smoke) / 1_000_000.0


def _best_selected_ms(smoke: dict[str, Any], substage: str) -> float:
  selected = smoke.get("selected_ffn_wall_profile_ns")
  if not isinstance(selected, dict):
    raise SystemExit("current best lacks selected_ffn_wall_profile_ns")
  return _num(selected.get(substage)) / _tokens(smoke) / 1_000_000.0


def _frontier_budget(path: Path) -> dict[str, float]:
  frontier = _load_json(path)
  goal_budget = frontier.get("goal_budget") if isinstance(frontier, dict) else None
  goal_anchor = frontier.get("goal_anchor") if isinstance(frontier, dict) else None
  no_progress = frontier.get("no_progress") if isinstance(frontier, dict) else None
  if not isinstance(goal_budget, dict) or not isinstance(goal_anchor, dict):
    raise SystemExit(f"{path}: missing goal_budget/goal_anchor")
  verdict = goal_budget.get("verdict")
  per_token = goal_budget.get("per_token_ms")
  noise = no_progress.get("noise") if isinstance(no_progress, dict) else None
  noise_rel = _num(noise.get("rel")) if isinstance(noise, dict) else 0.0
  return {
      "current_best_tps": _num(goal_anchor.get("current_best_tps")),
      "floor_tps": _num(goal_anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": _num(per_token.get("wall")) if isinstance(per_token, dict) else 0.0,
      "floor_budget_ms_per_token": (
          _num(verdict.get("floor_budget_ms_per_token"))
          if isinstance(verdict, dict) else 0.0
      ),
      "noise_pct": noise_rel * 100.0,
  }


def _source_shape(gpu_source: Path, opencl_source: Path) -> dict[str, Any]:
  gpu_text = gpu_source.read_text(encoding="utf-8")
  opencl_text = opencl_source.read_text(encoding="utf-8")
  combined_global = re.search(
      r"const std::size_t global\s*=\s*static_cast<std::size_t>"
      r"\(rows_per_expert \* 9\)",
      gpu_text,
  )
  tail_loop = re.search(
      r"ffn_tail_fused_output_f32[\s\S]*?"
      r"for \(uint expert = 0; expert < expert_count; \+\+expert\)",
      opencl_text,
  )
  return {
      "gpu_source": _display_path(gpu_source),
      "opencl_source": _display_path(opencl_source),
      "q6_selected_shared_global_rows_per_expert_x9": combined_global is not None,
      "tail_kernel_reduces_expert_count_loop": tail_loop is not None,
      "parallelism_collapse_factor_for_hidden_row_serial_fusion": (
          9 if combined_global is not None else 0
      ),
  }


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "ts": row.get("ts"),
      "label": row.get("label"),
      "tps": _tps(row),
      "top1_matches_native": bool(row.get("top1_matches_native")),
      "required_checks_passed": bool(row.get("required_checks_passed")),
      "wall_ms_per_token": _wall_ms_per_token(row),
      "selected_ffn_ms_per_token": _stage_ms(row, "selected_ffn"),
      "ffn_tail_ms_per_token": _stage_ms(row, "ffn_tail"),
      "selected_down_wait_ms_per_token": _selected_ms(row, "down_kernel_wait"),
      "selected_down_ms_per_token": _selected_ms(row, "down"),
      "linear_preconv_ms_per_token": _stage_ms(row, "linear_preconv"),
      "ffn_tail_resident_input": bool(row.get("ffn_tail_resident_input")),
      "attention_front_resident_residual_input": bool(
          row.get("attention_front_resident_residual_input")
      ),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _frontier_budget(args.frontier)
  best = _smoke_artifact(args.current_best)
  seq63 = _row_summary(_row_by_ts(args.explore_log, args.seq63_ts))
  seq65 = _row_summary(_row_by_ts(args.explore_log, args.seq65_ts))
  source = _source_shape(args.gpu_source, args.opencl_source)
  floor_gap = frontier["wall_ms_per_token"] - frontier["floor_budget_ms_per_token"]
  best_selected_tail = (
      _best_stage_ms(best, "selected_ffn") + _best_stage_ms(best, "ffn_tail")
  )
  seq63_selected_tail = seq63["selected_ffn_ms_per_token"] + seq63["ffn_tail_ms_per_token"]
  seq65_selected_tail = seq65["selected_ffn_ms_per_token"] + seq65["ffn_tail_ms_per_token"]
  local_carrier_best_delta = min(
      seq63_selected_tail - best_selected_tail,
      seq65_selected_tail - best_selected_tail,
  )
  hidden_row_serial_closed = (
      source["parallelism_collapse_factor_for_hidden_row_serial_fusion"] >= 9
      and local_carrier_best_delta > 0.0
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _display_path(args.frontier),
          "current_best": _display_path(args.current_best),
          "explore_log": _display_path(args.explore_log),
          "gpu_source": _display_path(args.gpu_source),
          "opencl_source": _display_path(args.opencl_source),
      },
      "frontier": {
          **frontier,
          "floor_gap_ms_per_token": floor_gap,
      },
      "source_shape": source,
      "current_best_selected_tail": {
          "selected_ffn_ms_per_token": _best_stage_ms(best, "selected_ffn"),
          "ffn_tail_ms_per_token": _best_stage_ms(best, "ffn_tail"),
          "selected_plus_tail_ms_per_token": best_selected_tail,
          "selected_down_wait_ms_per_token": _best_selected_ms(best, "down_kernel_wait"),
          "selected_down_ms_per_token": _best_selected_ms(best, "down"),
      },
      "local_carrier_rows": {
          "seq63_ffn_tail_resident_input": {
              **seq63,
              "selected_plus_tail_ms_per_token": seq63_selected_tail,
              "selected_plus_tail_delta_vs_best_ms_per_token": (
                  seq63_selected_tail - best_selected_tail
              ),
          },
          "seq65_ffn_tail_plus_attention_residual": {
              **seq65,
              "selected_plus_tail_ms_per_token": seq65_selected_tail,
              "selected_plus_tail_delta_vs_best_ms_per_token": (
                  seq65_selected_tail - best_selected_tail
              ),
          },
      },
      "derived": {
          "local_carrier_best_delta_vs_current_best_selected_tail_ms_per_token": (
              local_carrier_best_delta
          ),
          "required_total_wall_cut_to_floor_ms_per_token": max(0.0, floor_gap),
          "hidden_row_serial_q6_down_tail_fusion_closed": hidden_row_serial_closed,
          "minimum_admissible_fusion_shape": (
              "preserve selected/shared Q6 per-expert parallelism and change "
              "the final reduction/drain ownership; do not serialize the 8 "
              "selected experts plus shared down into one hidden-row work item"
          ),
      },
      "verdict": {
          "naive_hidden_row_serial_fusion_promotable": False,
          "speedup_claims_allowed": False,
          "reason": (
              "The live selected+shared Q6 down kernel exposes rows_per_expert*9 "
              "work-items, while the current FFN-tail kernel is only the final "
              "expert/shared reduction. Seq63 and seq65 show local carrier flags "
              "already reduce selected-FFN wall but grow FFN-tail enough to "
              "regress. A naive hidden-row fused kernel would collapse the Q6 "
              "down parallelism by 9x and is not an admissible next kernel."
          ),
          "next_route": (
              "Continue only with a down-to-tail design that preserves Q6 "
              "parallelism while eliminating the final drain, a full hidden-state "
              "carrier across host-vector boundaries, or DPAS work beyond the "
              "existing Q4 occupancy bounds."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Down-Tail Fusion Budget",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- current best: `{result['frontier']['current_best_tps']}` tok/s",
      f"- floor gap: `{result['frontier']['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- source Q6 parallelism collapse factor for naive serial fusion: "
      f"`{result['source_shape']['parallelism_collapse_factor_for_hidden_row_serial_fusion']}`",
      f"- local carrier best selected+tail delta: "
      f"`{result['derived']['local_carrier_best_delta_vs_current_best_selected_tail_ms_per_token']:.3f}` ms/token",
      f"- naive hidden-row serial fusion promotable: "
      f"`{str(result['verdict']['naive_hidden_row_serial_fusion_promotable']).lower()}`",
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--current-best", type=Path, default=DEFAULT_CURRENT_BEST)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--gpu-source", type=Path, default=DEFAULT_GPU_SOURCE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL_SOURCE)
  parser.add_argument("--seq63-ts", default=DEFAULT_SEQ63_TS)
  parser.add_argument("--seq65-ts", default=DEFAULT_SEQ65_TS)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  write_outputs(result, args.out_dir)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
