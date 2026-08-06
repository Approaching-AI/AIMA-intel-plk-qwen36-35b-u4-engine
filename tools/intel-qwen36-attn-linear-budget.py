#!/usr/bin/env python3
"""Budget the attention-front / linear-preconv route from current rows.

This is a route-selection arithmetic gate, not benchmark evidence. It answers
whether launch/gap-only cleanup in the attention-front, full-core, and
linear-preconv buckets can clear the 19.5 tok/s floor, before spending source
work on another isolated finish/readback/buffer variant.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FLOOR_TPS = 19.5
DEFAULT_FRONTIER = Path(
    "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
)
DEFAULT_REJECTED = Path(
    "doc/active/intel-qwen36-35b-a3b-gguf-q4km/rejected-routes.json"
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
  wall = smoke.get("wall_profile_ns")
  return (
      isinstance(wall, dict)
      and isinstance(wall.get("attention_front"), (int, float))
      and isinstance(wall.get("linear_preconv"), (int, float))
  )


def _select_row(path: Path, label: str | None) -> dict[str, Any]:
  rows = [row for row in _iter_rows(path)]
  if label is not None:
    matches = [row for row in rows if row.get("label") == label]
    if not matches:
      raise SystemExit(f"no row with label {label!r} in {path}")
    row = matches[-1]
    if not _is_usable(row):
      raise SystemExit(f"row {label!r} lacks attention/linear wall fields")
    return row
  usable = [row for row in rows if _is_usable(row)]
  if not usable:
    raise SystemExit(f"no usable attention/linear profile row in {path}")
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


def _goal_budget(frontier: Path | None) -> dict[str, Any]:
  if frontier is None or not frontier.is_file():
    return {}
  payload = _load_json(frontier)
  if not isinstance(payload, dict):
    return {}
  budget = payload.get("goal_budget")
  return budget if isinstance(budget, dict) else {}


def _noise_pct(frontier: Path | None) -> float | None:
  if frontier is None or not frontier.is_file():
    return None
  payload = _load_json(frontier)
  if not isinstance(payload, dict):
    return None
  no_progress = payload.get("no_progress")
  if not isinstance(no_progress, dict):
    return None
  noise = no_progress.get("noise")
  if isinstance(noise, dict):
    rel = noise.get("rel")
    if isinstance(rel, (int, float)):
      return float(rel) * 100.0
  if isinstance(noise, (int, float)):
    return float(noise) * 100.0 if 0.0 < float(noise) < 1.0 else float(noise)
  return None


def _gap_by_stage(goal_budget: dict[str, Any]) -> dict[str, float]:
  gaps: dict[str, float] = {}
  rows = goal_budget.get("stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return gaps
  for item in rows:
    if not isinstance(item, dict):
      continue
    stage = item.get("stage")
    gap = item.get("gap_ms_per_token")
    if isinstance(stage, str) and isinstance(gap, (int, float)):
      gaps[stage] = float(gap)
  return gaps


def _closed_counts(path: Path | None) -> dict[str, Any]:
  if path is None or not path.is_file():
    return {"attention_or_linear": 0, "routes": []}
  payload = _load_json(path)
  rejected = payload.get("rejected") if isinstance(payload, dict) else None
  if not isinstance(rejected, list):
    return {"attention_or_linear": 0, "routes": []}
  pattern = re.compile(
      r"attention|fullcore|full_core|linear|preconv|qkv|alpha|beta|rmsnorm",
      re.IGNORECASE,
  )
  routes = [
      str(item.get("route"))
      for item in rejected
      if isinstance(item, dict)
      and (
          pattern.search(str(item.get("route", "")))
          or pattern.search(str(item.get("class", "")))
      )
  ]
  return {"attention_or_linear": len(routes), "routes": routes[-12:]}


def _component_probe(path: Path | None) -> dict[str, Any] | None:
  if path is None:
    return None
  probe_path = path / "probe-result.json" if path.is_dir() else path
  if not probe_path.is_file():
    return None
  payload = _load_json(probe_path)
  return payload if isinstance(payload, dict) else None


def _projection_timings(probe: dict[str, Any] | None) -> dict[str, Any]:
  if not isinstance(probe, dict):
    return {}
  timings = probe.get("projection_timings")
  if not isinstance(timings, dict):
    return {}
  out: dict[str, Any] = {}
  for name in ("linear_attn_qkv_mixed", "alpha", "beta", "z"):
    item = timings.get(name)
    if isinstance(item, dict):
      out[name] = {
          "min_us": _num(item.get("gpu_kernel_min_us")),
          "gb_s": _num(item.get("gpu_effective_packed_gb_s")),
      }
  return out


def _attention_timings(probe: dict[str, Any] | None) -> dict[str, Any]:
  if not isinstance(probe, dict):
    return {}
  timings = probe.get("timings")
  if not isinstance(timings, dict):
    return {}
  keys = (
      "attention_front_kernel_sum_min_us",
      "attention_output_projection_min_us",
      "post_attention_residual_add_min_us",
      "ffn_rmsnorm_min_us",
  )
  return {key: _num(timings.get(key)) for key in keys}


def compute_budget(
    row: dict[str, Any],
    source: Path,
    floor_tps: float,
    frontier: Path | None,
    rejected: Path | None,
    preconv_probe: Path | None,
    attention_probe: Path | None,
) -> dict[str, Any]:
  smoke = _smoke(row)
  wall = smoke.get("wall_profile_ns")
  if not isinstance(wall, dict):
    raise SystemExit("row lacks wall_profile_ns")
  linear_wall = smoke.get("linear_preconv_wall_profile_ns")
  linear_wall = linear_wall if isinstance(linear_wall, dict) else {}

  tokens = _tokens(row, smoke)
  decode_ns = _decode_ns(row, smoke)
  wall_ms = decode_ns / tokens / 1e6
  floor_ms = 1e3 / floor_tps
  needed_ms = max(0.0, wall_ms - floor_ms)
  measured_tps = row.get("tps") or smoke.get("gpu_hybrid_decode_tok_s")
  if not isinstance(measured_tps, (int, float)):
    measured_tps = tokens * 1e9 / decode_ns

  stages = {
      "attention_front": _num(wall.get("attention_front")) / tokens / 1e6,
      "full_core": _num(wall.get("full_core")) / tokens / 1e6,
      "linear_preconv": _num(wall.get("linear_preconv")) / tokens / 1e6,
  }
  linear_substages = {
      "qkv_conv": _num(linear_wall.get("qkv_conv")) / tokens / 1e6,
      "alpha_beta": _num(linear_wall.get("alpha_beta")) / tokens / 1e6,
      "postconv_prep": _num(linear_wall.get("postconv_prep")) / tokens / 1e6,
      "input_q8": _num(linear_wall.get("input_q8")) / tokens / 1e6,
      "host_activation": _num(linear_wall.get("host_activation")) / tokens / 1e6,
  }

  goal = _goal_budget(frontier)
  gaps = _gap_by_stage(goal)
  gap_subset = {
      stage: gaps.get(stage, 0.0)
      for stage in ("attention_front", "full_core", "linear_preconv")
  }
  gap_sum = sum(gap_subset.values())

  preconv = _component_probe(preconv_probe)
  attention = _component_probe(attention_probe)
  closed = _closed_counts(rejected)
  noise = _noise_pct(frontier)

  return {
      "schema": "intel-qwen36-attn-linear-budget-v0",
      "source": str(source),
      "label": _row_label(row, source),
      "floor_tps": floor_tps,
      "decode_tokens": int(tokens),
      "measured_tok_s": round(float(measured_tps), 6),
      "noise_pct": noise,
      "ms_per_token": {
          "wall": round(wall_ms, 3),
          "floor_budget": round(floor_ms, 3),
          "reduction_needed_to_floor": round(needed_ms, 3),
          **{k: round(v, 3) for k, v in stages.items()},
      },
      "linear_preconv_substages_ms_per_token": {
          k: round(v, 3) for k, v in linear_substages.items()
      },
      "same_source_gap_upper_bound_ms_per_token": {
          **{k: round(v, 3) for k, v in gap_subset.items()},
          "attention_fullcore_linear_sum": round(gap_sum, 3),
          "clears_floor": gap_sum >= needed_ms,
      },
      "component_probe": {
          "preconv_projection_timings": _projection_timings(preconv),
          "attention_timings": _attention_timings(attention),
      },
      "closed_route_counts": closed,
      "route_verdict": (
          "gap-only attention/full-core/linear cleanup is insufficient; "
          "next work must be a kernel algorithm or broader resident boundary"
          if gap_sum < needed_ms
          else "gap-only cleanup is arithmetically enough; require a concrete "
          "event/lifetime proof before source work"
      ),
      "note": (
          "This gate uses current wall rows plus paired valid-profile stage gaps. "
          "It does not claim speed; it prevents isolated read/finish/buffer "
          "variants when their full upper bound cannot reach the floor."
      ),
  }


def render_text(budget: dict[str, Any]) -> str:
  ms = budget["ms_per_token"]
  gaps = budget["same_source_gap_upper_bound_ms_per_token"]
  linear = budget["linear_preconv_substages_ms_per_token"]
  lines = [
      f"attention/linear budget - {budget['label']}",
      f"  measured: {budget['measured_tok_s']} tok/s over {budget['decode_tokens']} tokens",
      f"  wall/floor: {ms['wall']:.3f} / {ms['floor_budget']:.3f} ms/token",
      f"  needed reduction: {ms['reduction_needed_to_floor']:.3f} ms/token",
      "",
      "  stage walls:",
      f"    attention_front             {ms['attention_front']:>8.3f} ms/token",
      f"    full_core                   {ms['full_core']:>8.3f} ms/token",
      f"    linear_preconv              {ms['linear_preconv']:>8.3f} ms/token",
      "",
      "  paired-profile gap upper bound:",
      f"    attention_front             {gaps['attention_front']:>8.3f} ms/token",
      f"    full_core                   {gaps['full_core']:>8.3f} ms/token",
      f"    linear_preconv              {gaps['linear_preconv']:>8.3f} ms/token",
      f"    combined                    {gaps['attention_fullcore_linear_sum']:>8.3f} ms/token",
      f"    clears floor                {str(gaps['clears_floor']).lower()}",
      "",
      "  linear-preconv substage walls:",
      f"    qkv_conv                    {linear['qkv_conv']:>8.3f} ms/token",
      f"    alpha_beta                  {linear['alpha_beta']:>8.3f} ms/token",
      f"    postconv_prep               {linear['postconv_prep']:>8.3f} ms/token",
      f"    input_q8                    {linear['input_q8']:>8.3f} ms/token",
      f"    host_activation             {linear['host_activation']:>8.3f} ms/token",
  ]
  preconv = budget["component_probe"].get("preconv_projection_timings") or {}
  if preconv:
    lines += ["", "  preconv component timings:"]
    for name in ("linear_attn_qkv_mixed", "alpha", "beta", "z"):
      item = preconv.get(name)
      if isinstance(item, dict):
        lines.append(
            f"    {name:<27} {item['min_us']:>8.3f} us"
            f"  {item['gb_s']:>8.3f} GB/s"
        )
  attention = budget["component_probe"].get("attention_timings") or {}
  if attention:
    lines += ["", "  attention component timings:"]
    for key, label in (
        ("attention_front_kernel_sum_min_us", "attention_front_sum"),
        ("attention_output_projection_min_us", "output_projection"),
        ("post_attention_residual_add_min_us", "residual_add"),
        ("ffn_rmsnorm_min_us", "ffn_rmsnorm"),
    ):
      if key in attention:
        lines.append(f"    {label:<27} {attention[key]:>8.3f} us")
  if budget.get("noise_pct") is not None:
    lines += ["", f"  frontier noise band: {budget['noise_pct']:.3f}%"]
  lines += [
      "",
      f"  closed attention/linear-like routes: {budget['closed_route_counts']['attention_or_linear']}",
      f"  verdict: {budget['route_verdict']}",
      "  note: arithmetic/component gate only; no speed claim.",
  ]
  return "\n".join(lines)


def write_artifact(out_dir: Path, budget: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "budget.json").write_text(
      json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  (out_dir / "summary.md").write_text(render_text(budget) + "\n", encoding="utf-8")
  metric = {
      "name": "attn_linear_budget",
      "label": budget["label"],
      "measured_tok_s": budget["measured_tok_s"],
      "needed_ms_per_token": budget["ms_per_token"]["reduction_needed_to_floor"],
      "gap_upper_bound_ms_per_token": budget[
          "same_source_gap_upper_bound_ms_per_token"
      ]["attention_fullcore_linear_sum"],
      "gap_only_clears_floor": budget[
          "same_source_gap_upper_bound_ms_per_token"
      ]["clears_floor"],
  }
  (out_dir / "metrics.jsonl").write_text(json.dumps(metric, sort_keys=True) + "\n",
                                         encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "path",
      type=Path,
      nargs="?",
      default=Path("output/explore-log.jsonl"),
      help="JSONL explore log or one JSON result row",
  )
  parser.add_argument("--label", help="select the latest JSONL row with label")
  parser.add_argument("--floor-tps", type=float, default=DEFAULT_FLOOR_TPS)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--preconv-probe", type=Path)
  parser.add_argument("--attention-probe", type=Path)
  parser.add_argument(
      "--out-dir",
      type=Path,
      help="optional output artifact directory; default only prints",
  )
  parser.add_argument("--json", action="store_true", help="emit JSON")
  args = parser.parse_args()

  row = _select_row(args.path, args.label)
  budget = compute_budget(
      row,
      args.path,
      args.floor_tps,
      args.frontier,
      args.rejected,
      args.preconv_probe,
      args.attention_probe,
  )
  if args.out_dir is not None:
    write_artifact(args.out_dir, budget)
  if args.json:
    print(json.dumps(budget, indent=2, sort_keys=True))
  else:
    print(render_text(budget))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
