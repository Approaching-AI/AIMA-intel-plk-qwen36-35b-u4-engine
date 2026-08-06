#!/usr/bin/env python3
"""Budget the selected/shared FFN branch-overlap route from explore rows.

This is a route-selection arithmetic gate, not benchmark evidence. It asks
whether overlapping the independent selected and shared FFN branches after the
router is large enough to clear the 19.5 tok/s floor before spending source
work on a multi-queue/common-Q8 implementation.

Input is normally `output/explore-log.jsonl`; the tool uses the row's
`profile_smoke.wall_profile_ns` stage timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FLOOR_TPS = 19.5
DEFAULT_FRONTIER = Path(
    "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
  if path.suffix == ".jsonl":
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
      if isinstance(row, dict):
        yield row
    return

  payload = _load_json(path)
  if isinstance(payload, dict):
    yield payload
    return
  raise SystemExit(f"{path}: expected JSON object or JSONL rows")


def _smoke(row: dict[str, Any]) -> dict[str, Any]:
  for key in ("profile_smoke", "smoke"):
    value = row.get(key)
    if isinstance(value, dict):
      return value
  return row


def _row_label(row: dict[str, Any], path: Path) -> str:
  return str(row.get("label") or row.get("source_artifact") or path)


def _is_usable(row: dict[str, Any]) -> bool:
  smoke = _smoke(row)
  wall_profile = smoke.get("wall_profile_ns")
  if not isinstance(wall_profile, dict):
    return False
  if not isinstance(wall_profile.get("selected_ffn"), (int, float)):
    return False
  if not isinstance(wall_profile.get("shared_ffn"), (int, float)):
    return False
  return True


def _select_row(path: Path, label: str | None) -> dict[str, Any]:
  rows = [row for row in _iter_rows(path)]
  if label is not None:
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
      raise SystemExit(f"no row with label {label!r} in {path}")
    row = matches[-1]
    if not _is_usable(row):
      raise SystemExit(f"row {label!r} lacks usable FFN wall profile fields")
    return row

  usable = [row for row in rows if _is_usable(row)]
  if not usable:
    raise SystemExit(f"no usable FFN profile row in {path}")
  return usable[-1]


def _tokens(row: dict[str, Any], smoke: dict[str, Any]) -> float:
  tokens = (
      row.get("decode_tokens")
      or smoke.get("decode_continuation_output_tokens")
      or row.get("decode_continuation_output_tokens")
  )
  if not isinstance(tokens, (int, float)) or tokens <= 0:
    raise SystemExit("row lacks a positive decode token count")
  return float(tokens)


def _decode_ns(row: dict[str, Any], smoke: dict[str, Any]) -> float:
  ns = row.get("decode_ns") or smoke.get("gpu_hybrid_decode_ns")
  if not isinstance(ns, (int, float)) or ns <= 0:
    raise SystemExit("row lacks positive decode wall ns")
  return float(ns)


def _q8_penalty_ns(smoke: dict[str, Any], policy: str) -> float:
  selected = smoke.get("selected_ffn_wall_profile_ns")
  shared = smoke.get("shared_ffn_wall_profile_ns")
  selected_input_q8 = (
      _num(selected.get("input_q8"))
      if isinstance(selected, dict) else 0.0
  )
  shared_input_q8 = (
      _num(shared.get("input_q8"))
      if isinstance(shared, dict) else 0.0
  )
  if policy == "none":
    return 0.0
  if policy == "observed-shared":
    return shared_input_q8
  if policy == "selected-input":
    return selected_input_q8
  if policy == "max-observed":
    return max(selected_input_q8, shared_input_q8)
  raise SystemExit(f"unknown q8 penalty policy: {policy}")


def _load_noise_pct(frontier: Path | None) -> float | None:
  if frontier is None or not frontier.is_file():
    return None
  payload = _load_json(frontier)
  if not isinstance(payload, dict):
    return None
  no_progress = payload.get("no_progress")
  if not isinstance(no_progress, dict):
    return None
  noise = no_progress.get("noise")
  if isinstance(noise, (int, float)):
    return float(noise) * 100.0 if 0.0 < float(noise) < 1.0 else float(noise)
  if isinstance(noise, dict):
    rel = noise.get("rel")
    if isinstance(rel, (int, float)):
      return float(rel) * 100.0
  return None


def compute_budget(
    row: dict[str, Any],
    path: Path,
    floor_tps: float,
    q8_penalty_policy: str,
    noise_pct: float | None,
) -> dict[str, Any]:
  smoke = _smoke(row)
  wall_profile = smoke.get("wall_profile_ns")
  if not isinstance(wall_profile, dict):
    raise SystemExit("row lacks wall_profile_ns")
  tokens = _tokens(row, smoke)
  decode_ns = _decode_ns(row, smoke)
  wall_ms_per_token = decode_ns / tokens / 1e6
  floor_ms_per_token = 1e3 / floor_tps
  selected_ns = _num(wall_profile.get("selected_ffn"))
  shared_ns = _num(wall_profile.get("shared_ffn"))
  if selected_ns <= 0 or shared_ns <= 0:
    raise SystemExit("selected/shared FFN wall fields must be positive")

  penalty_ns = _q8_penalty_ns(smoke, q8_penalty_policy)
  serial_ns = selected_ns + shared_ns
  overlapped_ns = max(selected_ns, shared_ns + penalty_ns)
  saved_ns = serial_ns - overlapped_ns
  projected_wall_ms = wall_ms_per_token - saved_ns / tokens / 1e6
  projected_tps = 1e3 / projected_wall_ms if projected_wall_ms > 0 else 0.0
  measured_tps = row.get("tps") or smoke.get("gpu_hybrid_decode_tok_s")
  if not isinstance(measured_tps, (int, float)):
    measured_tps = tokens * 1e9 / decode_ns

  return {
      "schema": "intel-qwen36-ffn-branch-budget-v0",
      "source": str(path),
      "label": _row_label(row, path),
      "floor_tps": floor_tps,
      "noise_pct": noise_pct,
      "q8_penalty_policy": q8_penalty_policy,
      "decode_tokens": int(tokens),
      "measured_tok_s": round(float(measured_tps), 6),
      "ms_per_token": {
          "wall": round(wall_ms_per_token, 3),
          "floor_budget": round(floor_ms_per_token, 3),
          "reduction_needed_to_floor": round(
              max(0.0, wall_ms_per_token - floor_ms_per_token), 3
          ),
          "selected_ffn": round(selected_ns / tokens / 1e6, 3),
          "shared_ffn": round(shared_ns / tokens / 1e6, 3),
          "selected_plus_shared_serial": round(serial_ns / tokens / 1e6, 3),
          "independent_shared_q8_penalty": round(penalty_ns / tokens / 1e6, 3),
          "perfect_branch_overlap_saving": round(saved_ns / tokens / 1e6, 3),
          "projected_wall_after_overlap": round(projected_wall_ms, 3),
          "projected_floor_margin": round(
              floor_ms_per_token - projected_wall_ms, 3
          ),
      },
      "projected_tok_s_after_overlap": round(projected_tps, 6),
      "clears_floor_arithmetically": projected_tps >= floor_tps,
      "route_verdict": (
          "branch-overlap route is arithmetically promotion-sized"
          if projected_tps >= floor_tps
          else "branch-overlap route is not enough by arithmetic"
      ),
      "note": (
          "This is an upper-bound route gate, not a speed claim. It assumes the "
          "selected and shared FFN branches can run concurrently after routing "
          "while preserving resident handles and correctness."
      ),
  }


def render_text(budget: dict[str, Any]) -> str:
  ms = budget["ms_per_token"]
  lines = [
      f"FFN branch budget - {budget['label']}",
      f"  measured: {budget['measured_tok_s']} tok/s over {budget['decode_tokens']} tokens",
      f"  wall/floor: {ms['wall']:.3f} / {ms['floor_budget']:.3f} ms/token",
      f"  needed reduction: {ms['reduction_needed_to_floor']:.3f} ms/token",
      "",
      "  selected/shared FFN serial wall:",
      f"    selected_ffn                {ms['selected_ffn']:>8.3f} ms/token",
      f"    shared_ffn                  {ms['shared_ffn']:>8.3f} ms/token",
      f"    serial total                {ms['selected_plus_shared_serial']:>8.3f} ms/token",
      "",
      f"  branch-overlap model ({budget['q8_penalty_policy']}):",
      f"    independent shared Q8 cost  {ms['independent_shared_q8_penalty']:>8.3f} ms/token",
      f"    projected saving            {ms['perfect_branch_overlap_saving']:>8.3f} ms/token",
      f"    projected wall              {ms['projected_wall_after_overlap']:>8.3f} ms/token",
      f"    projected speed             {budget['projected_tok_s_after_overlap']:>8.3f} tok/s",
      f"    floor margin                {ms['projected_floor_margin']:>8.3f} ms/token",
  ]
  if budget.get("noise_pct") is not None:
    lines.append(f"    frontier noise band         {budget['noise_pct']:>8.3f}%")
  lines += [
      "",
      f"  verdict: {budget['route_verdict']}",
      "  note: arithmetic gate only; no speed claim or correctness claim.",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "path",
      type=Path,
      nargs="?",
      default=Path("output/explore-log.jsonl"),
      help="explore JSONL or result-like JSON file",
  )
  parser.add_argument("--label", help="row label; default uses latest usable row")
  parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR_TPS)
  parser.add_argument(
      "--q8-penalty",
      choices=("none", "observed-shared", "selected-input", "max-observed"),
      default="selected-input",
      help="extra Q8 input cost charged to an independent shared branch",
  )
  parser.add_argument(
      "--frontier",
      type=Path,
      default=DEFAULT_FRONTIER,
      help="frontier JSON for the current noise band; use /dev/null to skip",
  )
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  frontier = None if str(args.frontier) == "/dev/null" else args.frontier
  row = _select_row(args.path, args.label)
  budget = compute_budget(
      row,
      args.path,
      args.floor,
      args.q8_penalty,
      _load_noise_pct(frontier),
  )
  if args.json:
    print(json.dumps(budget, indent=2, sort_keys=True))
  else:
    print(render_text(budget))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
