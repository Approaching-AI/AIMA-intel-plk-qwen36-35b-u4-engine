#!/usr/bin/env python3
"""Gate the Q6 rowgroup-local down-to-tail component proof."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q6-rowgroup-down-tail-gate-v0"
DEFAULT_PROBE = (
    ROOT
    / "output/gpu-selected-down-q6-rowgroup-down-tail-probe-20260707Tseq93Z/probe.json"
)
DEFAULT_SEQ92 = (
    ROOT
    / "output/selected-ffn-down-wait-drain-removal-gate-20260707Tseq92Z/metrics.json"
)
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = ROOT / "output/q6-rowgroup-down-tail-gate-20260707Tseq93Z"
Q6_LAYER_INVOCATIONS_PER_TOKEN = 20.0


def iso_now() -> str:
  return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"expected object JSON: {path}")
  return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nested(obj: Any, *keys: str) -> Any:
  current = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return current


def nested_bool(obj: Any, *keys: str) -> bool:
  return nested(obj, *keys) is True


def nested_number(obj: Any, *keys: str) -> float | None:
  value = nested(obj, *keys)
  return float(value) if isinstance(value, (int, float)) else None


def rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def extract_kernel(source: str, kernel_name: str) -> str:
  marker = f"__kernel void {kernel_name}("
  start = source.find(marker)
  if start < 0:
    return ""
  brace = source.find("{", start)
  if brace < 0:
    return ""
  depth = 0
  for index in range(brace, len(source)):
    char = source[index]
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
      if depth == 0:
        return source[start:index + 1]
  return ""


def write_metrics(path: Path, rows: list[tuple[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(
          json.dumps(
              {"phase": "q6_rowgroup_down_tail_gate", "metric": metric, "value": value}
          )
          + "\n"
      )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
  parser.add_argument("--seq92", type=Path, default=DEFAULT_SEQ92)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  metrics = payload["metrics"]
  failed = [item["name"] for item in payload["checks"] if not item["pass"]]
  lines = [
      "# Q6 Rowgroup Down-Tail Gate",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- source probe: `{payload['source_probe']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- kernel: `{metrics['kernel']}`",
      f"- rowgroup min us: `{metrics['rowgroup_min_us']}`",
      f"- target us: `{metrics['target_us']}`",
      f"- miss vs target us: `{metrics['miss_vs_target_us']}`",
      f"- rowgroup/non-atomic ratio: `{metrics['ratio_vs_nonatomic']}`",
      f"- rowgroup/target ratio: `{metrics['ratio_vs_target']}`",
      f"- global/local work items: `{metrics['global_work_items']}` / `{metrics['local_work_items']}`",
      f"- layer-vs-oracle max abs diff: `{metrics['layer_vs_oracle_max_abs_diff']}`",
      f"- layer-vs-nonatomic max abs diff: `{metrics['layer_vs_nonatomic_max_abs_diff']}`",
      f"- estimated delta vs non-atomic decode ms/token: `{metrics['estimated_delta_vs_nonatomic_ms_per_token']}`",
      f"- decode probe allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      "This is component evidence only. The rowgroup kernel is exact and avoids",
      "global contribution scratch plus atomics, but it does not clear the",
      "seq92 component-shell target, so it cannot be promoted to decode.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=False)

  payload = load_json(args.probe)
  seq92 = load_json(args.seq92)
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe.get("timings"), dict) else {}
  comparisons = (
      probe.get("comparisons", {}).get("q6_rowgroup_down_tail", {})
      if isinstance(probe.get("comparisons"), dict)
      else {}
  )
  opencl = args.opencl_source.read_text(encoding="utf-8")
  kernel_name = "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_rowgroup_reduce_raw"
  kernel_body = extract_kernel(opencl, kernel_name)

  rowgroup_min = nested_number(timings, "candidate_q6_rowgroup_down_tail_kernel_min_us")
  nonatomic_shell = nested_number(timings, "candidate_q6_nonatomic_down_tail_shell_sum_min_us")
  target = nested_number(probe, "candidate_q6_rowgroup_down_tail_component_shell_target_us")
  if target is None:
    target = nested_number(seq92, "component_shell_target_us_per_layer")
  rows_per_expert = nested_number(probe, "rows_per_expert")
  expected_global = int(rows_per_expert * 16) if rows_per_expert is not None else None
  global_work_items = nested_number(timings, "candidate_q6_rowgroup_down_tail_global_work_items")
  local_work_items = nested_number(timings, "candidate_q6_rowgroup_down_tail_local_work_items")
  target_cleared = rowgroup_min is not None and target is not None and rowgroup_min <= target
  component_exact = nested_bool(
      probe, "checks", "candidate_q6_rowgroup_down_tail_matches_current_and_oracle"
  )
  timing_positive = nested_bool(
      probe, "checks", "candidate_q6_rowgroup_down_tail_event_timing_positive"
  )
  avoids_scratch_and_atomics = (
      bool(kernel_body)
      and "__global float* contrib" not in kernel_body
      and "atomic_" not in kernel_body
      and "atomic_add" not in kernel_body
  )
  has_rowgroup_reduce_shape = (
      "__local float partial[16]" in kernel_body
      and "get_group_id(0)" in kernel_body
      and "barrier(CLK_LOCAL_MEM_FENCE)" in kernel_body
      and "for (uint i = 0; i < 9U; ++i)" in kernel_body
  )

  metrics = {
      "kernel": probe.get("candidate_q6_rowgroup_down_tail_kernel"),
      "measured": probe.get("candidate_q6_rowgroup_down_tail_measured"),
      "target_us": target,
      "rowgroup_min_us": rowgroup_min,
      "rowgroup_mean_us": nested_number(timings, "candidate_q6_rowgroup_down_tail_kernel_mean_us"),
      "nonatomic_shell_min_us": nonatomic_shell,
      "combined_down_min_us": nested_number(
          timings, "candidate_q6_selected_shared_combined_kernel_min_us"
      ),
      "miss_vs_target_us": (
          rowgroup_min - target
          if rowgroup_min is not None and target is not None
          else None
      ),
      "ratio_vs_target": (
          rowgroup_min / target
          if rowgroup_min is not None and target not in (None, 0)
          else None
      ),
      "ratio_vs_nonatomic": nested_number(
          probe, "candidate_q6_rowgroup_down_tail_ratio_vs_nonatomic"
      ),
      "ratio_vs_combined_down": nested_number(
          probe, "candidate_q6_rowgroup_down_tail_ratio_vs_combined_down"
      ),
      "component_exact": component_exact,
      "target_cleared": target_cleared,
      "global_work_items": global_work_items,
      "local_work_items": local_work_items,
      "expected_global_work_items": expected_global,
      "layer_vs_oracle_max_abs_diff": nested_number(
          comparisons, "layer_vs_oracle", "max_abs_diff"
      ),
      "layer_vs_nonatomic_max_abs_diff": nested_number(
          comparisons, "layer_vs_nonatomic", "max_abs_diff"
      ),
      "estimated_delta_vs_nonatomic_ms_per_token": (
          (rowgroup_min - nonatomic_shell) * Q6_LAYER_INVOCATIONS_PER_TOKEN / 1000.0
          if rowgroup_min is not None and nonatomic_shell is not None
          else None
      ),
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
  }
  metrics["decode_probe_allowed"] = bool(
      component_exact
      and timing_positive
      and avoids_scratch_and_atomics
      and has_rowgroup_reduce_shape
      and target_cleared
  )
  checks = [
      {"name": "probe_required_checks_passed", "pass": payload.get("required_checks_passed") is True},
      {"name": "probe_stdout_required_checks_passed", "pass": probe.get("required_checks_passed") is True},
      {"name": "q6_tensor_measured", "pass": probe.get("tensor_type") == "Q6_K"},
      {"name": "rowgroup_tail_measured", "pass": probe.get("candidate_q6_rowgroup_down_tail_measured") is True},
      {"name": "rowgroup_kernel_present_in_opencl", "pass": bool(kernel_body)},
      {"name": "rowgroup_kernel_uses_local_reduce", "pass": has_rowgroup_reduce_shape},
      {"name": "rowgroup_kernel_avoids_global_contrib_scratch_and_atomics", "pass": avoids_scratch_and_atomics},
      {"name": "component_matches_current_and_oracle", "pass": component_exact},
      {"name": "component_timing_positive", "pass": timing_positive},
      {"name": "global_work_items_are_rows_times_16", "pass": expected_global is not None and global_work_items == expected_global},
      {"name": "local_work_items_are_16", "pass": local_work_items == 16},
      {"name": "component_shell_beats_seq92_target", "pass": target_cleared},
      {"name": "boundary_reopen_candidate", "pass": probe.get("candidate_q6_rowgroup_down_tail_boundary_reopen_candidate") is True},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  disposition = (
      "component_reopen_candidate_decode_wiring"
      if required_checks_passed
      else "rejected_rowgroup_local_reduce_component_speed"
  )
  result = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": iso_now(),
      "source_probe": rel(args.probe),
      "seq92_gate": rel(args.seq92),
      "opencl_source": rel(args.opencl_source),
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "selected_next_route": "post_rowgroup_route_selection_gate",
      "metrics": metrics,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": result["created_at"],
      "tool": "tools/intel-qwen36-q6-rowgroup-down-tail-gate.py",
      "artifact": rel(args.out_dir),
      "source_probe": rel(args.probe),
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  write_json(args.out_dir / "gate.json", result)
  write_json(args.out_dir / "metrics.json", metrics)
  write_json(args.out_dir / "manifest.json", manifest)
  write_json(args.out_dir / "correctness.json", correctness)
  write_metrics(args.out_dir / "metrics.jsonl", sorted(metrics.items()))
  write_summary(args.out_dir / "summary.md", result)
  print(args.out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
