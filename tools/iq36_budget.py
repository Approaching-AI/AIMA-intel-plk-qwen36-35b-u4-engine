#!/usr/bin/env python3
"""Per-token decode time budget + kill-number verdict (ch.2 §2.1 move #1).

The methodology's first strategic move is "derive the kill-number, then let one
line of arithmetic reject a route before running more experiments". For the
resident/full-GPU decode-loop route the kill question is:

    If every microsecond of non-kernel overhead (host bridges, launches,
    uploads/readbacks, waits) were removed, does the decode loop reach the
    19.5 tok/s same-host Vulkan floor?

A speed-row artifact already carries the answer: `smoke.gpu_hybrid_decode_ns`
is the timed wall, `smoke.gpu_kernel_sum_min_us` is the summed per-kernel
device-busy floor, and `smoke.wall_profile_ns` decomposes the wall by stage.
No new run is needed — this module turns those numbers into a verdict:

    kernel_busy_ms_per_token > floor_budget_ms  =>  overhead-only cuts can
    NEVER reach the floor; kernel-side work (bandwidth/layout/fusion) is
    required and micro overhead-cut candidates are sub-threshold BY BUDGET.

Importable (`import iq36_budget`) for frontier-sync, and runnable as a CLI:

  python3 tools/iq36_budget.py output/r2-gpu-<...>-speed-<ts>Z [--floor 19.5]
  python3 tools/iq36_budget.py <artifact-dir-or-result.json> --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_FLOOR_TPS = 19.5


def _load_result(path: Path) -> dict[str, Any] | None:
  if path.is_dir():
    path = path / "result.json"
  if not path.is_file():
    return None
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return None
  return payload if isinstance(payload, dict) else None


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _gpu_unprofiled_breakdown_ns(smoke: dict[str, Any]) -> dict[str, float] | None:
  gpu_ns = smoke.get("gpu_hybrid_decode_ns")
  profiles = smoke.get("gpu_token_profiles")
  if not isinstance(gpu_ns, (int, float)) or gpu_ns <= 0:
    return None
  if not isinstance(profiles, list) or not profiles:
    return None
  token_wall_ns = 0.0
  token_profiled_ns = 0.0
  for profile in profiles:
    if not isinstance(profile, dict):
      return None
    token_ns = profile.get("gpu_decode_ns")
    profiled_ns = profile.get("profiled_wall_ns")
    if not isinstance(token_ns, (int, float)) or not isinstance(profiled_ns, (int, float)):
      return None
    token_wall_ns += float(token_ns)
    token_profiled_ns += float(profiled_ns)
  return {
      "gpu_loop_bookkeeping_ns": max(0.0, float(gpu_ns) - token_wall_ns),
      "token_core_unprofiled_ns": max(0.0, token_wall_ns - token_profiled_ns),
  }


def _gpu_loop_bookkeeping_profile_ns(smoke: dict[str, Any]) -> dict[str, float] | None:
  profile = smoke.get("gpu_loop_bookkeeping_wall_profile_ns")
  keys = (
      "pre_snapshot", "stats_snapshot", "state_diff", "counter_after",
      "correctness", "profile_assembly", "trace_capture", "finalize",
  )
  if isinstance(profile, dict):
    rows = {
        key: float(profile.get(key, 0.0))
        for key in keys
        if isinstance(profile.get(key), (int, float)) and profile.get(key, 0.0) > 0
    }
    return rows or None

  profiles = smoke.get("gpu_token_profiles")
  if not isinstance(profiles, list) or not profiles:
    return None
  rows = {key: 0.0 for key in keys}
  found = False
  for token_profile in profiles:
    if not isinstance(token_profile, dict):
      continue
    loop_profile = token_profile.get("loop_bookkeeping_wall_profile_ns")
    if not isinstance(loop_profile, dict):
      continue
    found = True
    for key in keys:
      value = loop_profile.get(key)
      if isinstance(value, (int, float)) and value > 0:
        rows[key] += float(value)
  if not found:
    return None
  return {key: value for key, value in rows.items() if value > 0}


def _stage_kernel_gap_rows(smoke: dict[str, Any], tokens: float) -> list[dict[str, Any]]:
  """Best-effort stage wall minus known kernel/substage work.

  These rows are directional profiling, not promotion evidence. They answer:
  "inside the current top wall buckets, where is the largest wall that is not
  already explained by the known kernel/device-min fields?"
  """
  wall_profile = smoke.get("wall_profile_ns")
  if not isinstance(wall_profile, dict):
    return []

  rows: list[dict[str, Any]] = []

  def add(stage: str, known_us: float, known_label: str) -> None:
    wall_ns = wall_profile.get(stage)
    if not isinstance(wall_ns, (int, float)) or wall_ns <= 0:
      return
    wall_ms_total = float(wall_ns) / 1e6
    known_ms_total = max(0.0, known_us / 1e3)
    gap_ms_total = wall_ms_total - known_ms_total
    rows.append({
        "stage": stage,
        "wall_ms_per_token": round(wall_ms_total / tokens, 3),
        "known_work_ms_per_token": round(known_ms_total / tokens, 3),
        "gap_ms_per_token": round(gap_ms_total / tokens, 3),
        "known_work": known_label,
    })

  def add_wall_only(stage: str, wall_ns: Any, known_label: str) -> None:
    if not isinstance(wall_ns, (int, float)) or wall_ns <= 0:
      return
    wall_ms_total = float(wall_ns) / 1e6
    rows.append({
        "stage": stage,
        "wall_ms_per_token": round(wall_ms_total / tokens, 3),
        "known_work_ms_per_token": 0.0,
        "gap_ms_per_token": round(wall_ms_total / tokens, 3),
        "known_work": known_label,
    })

  unprofiled_breakdown = _gpu_unprofiled_breakdown_ns(smoke)
  if unprofiled_breakdown is not None:
    add_wall_only(
        "gpu_loop_bookkeeping",
        unprofiled_breakdown["gpu_loop_bookkeeping_ns"],
        "GPU loop wall outside timed token core",
    )
    add_wall_only(
        "token_core_unprofiled",
        unprofiled_breakdown["token_core_unprofiled_ns"],
        "timed token core wall outside named profile buckets",
    )
  else:
    add_wall_only(
        "unprofiled_wall",
        smoke.get("unprofiled_wall_ns"),
        "wall outside named profile buckets",
    )


  ffn = smoke.get("ffn_kernel_profile_us")
  if isinstance(ffn, dict):
    add(
        "selected_ffn",
        _num(ffn.get("selected_gate_up")) +
        _num(ffn.get("selected_swiglu")) +
        _num(ffn.get("selected_down")),
        "selected gate/up + SwiGLU + down kernel min",
    )
    add(
        "shared_ffn",
        _num(ffn.get("shared_gate")) +
        _num(ffn.get("shared_up")) +
        _num(ffn.get("shared_swiglu")) +
        _num(ffn.get("shared_down")),
        "shared gate/up + SwiGLU + down kernel min",
    )

  linear_preconv = smoke.get("linear_preconv_kernel_profile_us")
  if isinstance(linear_preconv, dict):
    add(
        "linear_preconv",
        sum(
            _num(linear_preconv.get(key))
            for key in (
                "qkv", "conv", "alpha", "beta", "z",
                "postconv_silu_split", "postconv_q_l2", "postconv_k_l2",
            )
        ),
        "linear preconv QKV/conv/alpha-beta/z/postconv kernel min",
    )

  add(
      "attention_front",
      _num(smoke.get("attention_front_handoff_kernel_us")) +
      _num(smoke.get("attention_front_ffn_rmsnorm_min_us")),
      "attention-front handoff + FFN RMSNorm kernel min",
  )
  add(
      "full_core",
      _num(smoke.get("full_core_attention_front_handoff_kernel_us")),
      "full-core attention-front handoff kernel min",
  )
  add(
      "q4_cpu_order_z",
      _num(smoke.get("q4_cpu_order_z_kernel_profile_us")),
      "Q4 CPU-order z kernel min",
  )

  linear_delta = smoke.get("linear_delta_wall_profile_ns")
  if isinstance(linear_delta, dict):
    add(
        "linear_delta",
        _num(linear_delta.get("kernel")) / 1e3,
        "linear-delta kernel wall subprofile",
    )

  lm_head_wall = smoke.get("lm_head_wall_profile_ns")
  if isinstance(lm_head_wall, dict):
    add(
        "lm_head_gpu",
        sum(_num(v) for v in lm_head_wall.values()) / 1e3,
        "LM-head wall subprofile",
    )

  rows.sort(key=lambda row: -row["gap_ms_per_token"])
  return rows


def _substage_gap_rows(smoke: dict[str, Any], tokens: float) -> list[dict[str, Any]]:
  """Fine-grained wall-minus-known-work rows inside stages.

  Stage-level gaps are good for choosing the next region. These substage rows
  are the follow-up question: within that region, which specific wall bucket is
  not already explained by a matching kernel/device-min field?
  """
  rows: list[dict[str, Any]] = []

  def add(stage: str, substage: str, wall_ns: Any, known_us: float,
          known_label: str) -> None:
    if not isinstance(wall_ns, (int, float)) or wall_ns <= 0:
      return
    wall_ms_total = float(wall_ns) / 1e6
    known_ms_total = max(0.0, known_us / 1e3)
    rows.append({
        "stage": stage,
        "substage": substage,
        "wall_ms_per_token": round(wall_ms_total / tokens, 3),
        "known_work_ms_per_token": round(known_ms_total / tokens, 3),
        "gap_ms_per_token": round((wall_ms_total - known_ms_total) / tokens, 3),
        "known_work": known_label,
    })

  ffn = smoke.get("ffn_kernel_profile_us")
  selected = smoke.get("selected_ffn_wall_profile_ns")
  if isinstance(ffn, dict) and isinstance(selected, dict):
    add("selected_ffn", "gate_up", selected.get("gate_up"),
        _num(ffn.get("selected_gate_up")),
        "selected gate/up kernel min")
    add("selected_ffn", "swiglu", selected.get("swiglu"),
        _num(ffn.get("selected_swiglu")),
        "selected SwiGLU kernel min")
    add("selected_ffn", "down", selected.get("down"),
        _num(ffn.get("selected_down")),
        "selected down kernel min")
    add("selected_ffn", "down_q8", selected.get("down_q8"),
        _num(ffn.get("selected_host_q8_bridge")),
        "selected host Q8 bridge")
    add("selected_ffn", "input_q8", selected.get("input_q8"), 0.0,
        "selected input Q8 wall only")
    add("selected_ffn", "raw_setup", selected.get("raw_setup"), 0.0,
        "selected raw/handle setup wall only")
    add("selected_ffn", "down_kernel_wait", selected.get("down_kernel_wait"),
        0.0, "selected down queue wait wall only")

  shared = smoke.get("shared_ffn_wall_profile_ns")
  if isinstance(ffn, dict) and isinstance(shared, dict):
    add("shared_ffn", "gate_up", shared.get("gate_up"),
        _num(ffn.get("shared_gate")) + _num(ffn.get("shared_up")),
        "shared gate/up kernel min")
    add("shared_ffn", "swiglu", shared.get("swiglu"),
        _num(ffn.get("shared_swiglu")),
        "shared SwiGLU kernel min")
    add("shared_ffn", "down", shared.get("down"),
        _num(ffn.get("shared_down")),
        "shared down kernel min")
    add("shared_ffn", "down_q8", shared.get("down_q8"),
        _num(ffn.get("shared_host_q8_bridge")),
        "shared host Q8 bridge")
    add("shared_ffn", "input_q8", shared.get("input_q8"), 0.0,
        "shared input Q8 wall only")
    add("shared_ffn", "raw_setup", shared.get("raw_setup"), 0.0,
        "shared raw/handle setup wall only")

  linear = smoke.get("linear_preconv_wall_profile_ns")
  linear_k = smoke.get("linear_preconv_kernel_profile_us")
  if isinstance(linear, dict) and isinstance(linear_k, dict):
    add("linear_preconv", "qkv_conv", linear.get("qkv_conv"),
        _num(linear_k.get("qkv")) + _num(linear_k.get("conv")),
        "linear qkv + conv kernel min")
    add("linear_preconv", "alpha_beta", linear.get("alpha_beta"),
        _num(linear_k.get("alpha")) + _num(linear_k.get("beta")),
        "linear alpha/beta kernel min")
    add("linear_preconv", "postconv_prep", linear.get("postconv_prep"),
        _num(linear_k.get("postconv_silu_split")) +
        _num(linear_k.get("postconv_q_l2")) +
        _num(linear_k.get("postconv_k_l2")),
        "linear postconv-prep kernel min")
    add("linear_preconv", "input_q8", linear.get("input_q8"),
        _num(linear_k.get("host_q8_bridge")),
        "linear input Q8 bridge")
    add("linear_preconv", "host_activation", linear.get("host_activation"),
        0.0, "linear host activation wall only")

  linear_delta = smoke.get("linear_delta_wall_profile_ns")
  if isinstance(linear_delta, dict):
    add("linear_delta", "kernel", linear_delta.get("kernel"),
        _num(linear_delta.get("kernel")) / 1e3,
        "linear-delta kernel wall subprofile")
    add("linear_delta", "input_upload", linear_delta.get("input_upload"),
        0.0, "linear-delta input upload wall only")
    add("linear_delta", "final_read", linear_delta.get("final_read"),
        0.0, "linear-delta final read wall only")

  loop_profile = _gpu_loop_bookkeeping_profile_ns(smoke)
  if isinstance(loop_profile, dict):
    for substage, wall_ns in loop_profile.items():
      add("gpu_loop_bookkeeping", substage, wall_ns, 0.0,
          "GPU loop bookkeeping wall only")

  rows.sort(key=lambda row: -row["gap_ms_per_token"])
  return rows


def compute_budget_from_payload(payload: dict[str, Any],
                                source_artifact: str | Path,
                                floor_tps: float = DEFAULT_FLOOR_TPS) -> dict[str, Any] | None:
  """Compute the per-token budget table from a result-like payload.

  Returns None when the artifact lacks the timing fields (older runs).
  All times are per generated token, in milliseconds.
  """
  smoke = payload.get("smoke")
  if not isinstance(smoke, dict):
    return None

  wall_ns = smoke.get("gpu_hybrid_decode_ns")
  tokens = smoke.get("decode_continuation_output_tokens") or payload.get("decode_tokens")
  kernel_sum_us = smoke.get("gpu_kernel_sum_min_us")
  kernel_profiles_valid = not (
      smoke.get("opencl_no_queue_profiling") is True
      or smoke.get("skip_opencl_event_profile_readback") is True
      or payload.get("opencl_no_queue_profiling") is True
      or payload.get("skip_opencl_event_profile_readback") is True
  )
  if not kernel_profiles_valid:
    kernel_sum_us = None
  if not isinstance(wall_ns, (int, float)) or not isinstance(tokens, (int, float)) or tokens <= 0:
    return None

  wall_ms = wall_ns / tokens / 1e6
  kernel_ms = kernel_sum_us / tokens / 1e3 if isinstance(kernel_sum_us, (int, float)) else None
  overhead_ms = wall_ms - kernel_ms if kernel_ms is not None else None
  floor_budget_ms = 1e3 / floor_tps

  stage_walls: list[dict[str, Any]] = []
  wall_entries: list[tuple[str, float]] = []
  wall_profile = smoke.get("wall_profile_ns")
  if isinstance(wall_profile, dict):
    for stage, ns in wall_profile.items():
      if isinstance(ns, (int, float)) and ns > 0:
        wall_entries.append((stage, float(ns)))
  unprofiled_breakdown = _gpu_unprofiled_breakdown_ns(smoke)
  if unprofiled_breakdown is not None:
    for stage, ns in unprofiled_breakdown.items():
      if ns > 0:
        wall_entries.append((stage.removesuffix("_ns"), ns))
  else:
    unprofiled_wall_ns = smoke.get("unprofiled_wall_ns")
    if isinstance(unprofiled_wall_ns, (int, float)) and unprofiled_wall_ns > 0:
      wall_entries.append(("unprofiled_wall", float(unprofiled_wall_ns)))
  for stage, ns in sorted(wall_entries, key=lambda item: -item[1]):
    stage_walls.append({"stage": stage, "ms_per_token": round(ns / tokens / 1e6, 3)})
  loop_bookkeeping_profile = _gpu_loop_bookkeeping_profile_ns(smoke)

  verdict: dict[str, Any] = {
      "floor_tps": floor_tps,
      "floor_budget_ms_per_token": round(floor_budget_ms, 3),
  }
  if kernel_ms is not None:
    overhead_only_ceiling = 1e3 / kernel_ms if kernel_ms > 0 else float("inf")
    verdict.update({
        "overhead_only_ceiling_tok_s": round(overhead_only_ceiling, 3),
        "can_reach_floor_without_kernel_work": overhead_only_ceiling >= floor_tps,
        "min_kernel_time_cut_pct_needed": (
            round(max(0.0, (kernel_ms - floor_budget_ms) / kernel_ms) * 100, 1)
            if kernel_ms > 0 else 0.0
        ),
    })

  return {
      "schema": "iq36-decode-budget-v0",
      "source_artifact": str(source_artifact),
      "decode_tokens": int(tokens),
      "measured_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
      "kernel_profiles_valid": kernel_profiles_valid,
      "per_token_ms": {
          "wall": round(wall_ms, 3),
          "gpu_kernel_busy_floor": round(kernel_ms, 3) if kernel_ms is not None else None,
          "non_kernel_overhead": round(overhead_ms, 3) if overhead_ms is not None else None,
      },
      "stage_walls_ms_per_token": stage_walls[:10],
      "unprofiled_wall_breakdown_ms_per_token": (
          {
              key.removesuffix("_ns"): round(value / tokens / 1e6, 3)
              for key, value in unprofiled_breakdown.items()
          }
          if unprofiled_breakdown is not None else None
      ),
      "gpu_loop_bookkeeping_profile_ms_per_token": (
          {
              key: round(value / tokens / 1e6, 3)
              for key, value in loop_bookkeeping_profile.items()
          }
          if loop_bookkeeping_profile is not None else None
      ),
      "stage_kernel_gap_estimates_ms_per_token": (
          _stage_kernel_gap_rows(smoke, tokens)[:10]
          if kernel_profiles_valid else []
      ),
      "substage_gap_estimates_ms_per_token": (
          _substage_gap_rows(smoke, tokens)[:16]
          if kernel_profiles_valid else []
      ),
      "verdict": verdict,
      "note": (
          "gpu_kernel_busy_floor sums per-kernel MIN device times: the best case "
          "for the CURRENT kernels/layout. Overhead-only (host bridge / launch / "
          "upload-readback) cuts cannot push tok/s above overhead_only_ceiling_tok_s; "
          "reaching the floor past that ceiling requires kernel-side work "
          "(bandwidth/layout/fusion — see routes-ledger parked_routes)."
      ),
  }


def compute_budget(artifact: Path, floor_tps: float = DEFAULT_FLOOR_TPS) -> dict[str, Any] | None:
  """Compute the per-token budget table for a speed-row artifact."""
  payload = _load_result(artifact)
  if payload is None:
    return None
  return compute_budget_from_payload(payload, artifact, floor_tps)


def render_table(budget: dict[str, Any]) -> str:
  per = budget["per_token_ms"]
  v = budget["verdict"]
  lines = [
      f"decode budget — {budget['source_artifact']}",
      f"  measured: {budget['measured_tok_s']} tok/s over {budget['decode_tokens']} tokens",
      f"  per-token wall            {per['wall']:>9.3f} ms",
  ]
  if per["gpu_kernel_busy_floor"] is not None:
    lines += [
        f"  gpu kernel busy (floor)   {per['gpu_kernel_busy_floor']:>9.3f} ms",
        f"  non-kernel overhead       {per['non_kernel_overhead']:>9.3f} ms   <- max recoverable by overhead cuts",
    ]
  elif budget.get("kernel_profiles_valid") is False:
    lines.append(
        "  gpu kernel busy (floor)     invalid (OpenCL event profiling disabled/skipped)"
    )
  lines.append(
      f"  floor budget @{v['floor_tps']} tok/s  {v['floor_budget_ms_per_token']:>9.3f} ms"
  )
  if "overhead_only_ceiling_tok_s" in v:
    lines += [
        "",
        f"  overhead-only ceiling: {v['overhead_only_ceiling_tok_s']} tok/s "
        f"(floor {'REACHABLE' if v['can_reach_floor_without_kernel_work'] else 'NOT reachable'} without kernel work)",
    ]
    if not v["can_reach_floor_without_kernel_work"]:
      lines.append(
          f"  kernel device time itself must shrink >= {v['min_kernel_time_cut_pct_needed']}% "
          "even with ALL overhead removed"
      )
  if budget["stage_walls_ms_per_token"]:
    lines += ["", "  top stage walls (ms/token):"]
    for row in budget["stage_walls_ms_per_token"]:
      lines.append(f"    {row['stage']:<28s} {row['ms_per_token']:>9.3f}")
  breakdown = budget.get("unprofiled_wall_breakdown_ms_per_token")
  if isinstance(breakdown, dict) and breakdown:
    lines += ["", "  unprofiled wall split (ms/token):"]
    for key, value in sorted(breakdown.items(), key=lambda item: -_num(item[1])):
      lines.append(f"    {key:<28s} {_num(value):>9.3f}")
  loop_profile = budget.get("gpu_loop_bookkeeping_profile_ms_per_token")
  if isinstance(loop_profile, dict) and loop_profile:
    lines += ["", "  GPU loop bookkeeping profile (ms/token):"]
    for key, value in sorted(loop_profile.items(), key=lambda item: -_num(item[1])):
      lines.append(f"    {key:<28s} {_num(value):>9.3f}")
  gaps = budget.get("stage_kernel_gap_estimates_ms_per_token") or []
  if gaps:
    lines += [
        "",
        "  largest stage wall minus known kernel/subprofile work (ms/token):",
    ]
    for row in gaps[:8]:
      lines.append(
          f"    {row['stage']:<20s} gap {row['gap_ms_per_token']:>7.3f}  "
          f"wall {row['wall_ms_per_token']:>7.3f}  "
          f"known {row['known_work_ms_per_token']:>7.3f}"
      )
  subgaps = budget.get("substage_gap_estimates_ms_per_token") or []
  if subgaps:
    lines += [
        "",
        "  largest substage wall minus known work (ms/token):",
    ]
    for row in subgaps[:10]:
      label = f"{row['stage']}.{row['substage']}"
      lines.append(
          f"    {label:<28s} gap {row['gap_ms_per_token']:>7.3f}  "
          f"wall {row['wall_ms_per_token']:>7.3f}  "
          f"known {row['known_work_ms_per_token']:>7.3f}"
      )
  return "\n".join(lines)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("artifact", type=Path,
                  help="speed-row artifact dir (or its result.json)")
  ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR_TPS)
  ap.add_argument("--json", action="store_true", help="emit machine JSON only")
  args = ap.parse_args()

  budget = compute_budget(args.artifact, args.floor)
  if budget is None:
    print(json.dumps({"error": "artifact lacks decode timing fields", "artifact": str(args.artifact)}))
    return 2
  if args.json:
    print(json.dumps(budget, indent=2, sort_keys=True))
  else:
    print(render_table(budget))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
