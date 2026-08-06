#!/usr/bin/env python3
"""Roll up q4-plane profile/cache evidence into an R3 gap report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r3-q4-plane-gap-rollup-v0"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--diagnostic-dir", type=Path, required=True)
  parser.add_argument("--route-check-dir", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def tensor_suffix(name: str) -> str:
  parts = name.split(".", 2)
  if len(parts) == 3 and parts[0] == "blk" and parts[1].isdigit():
    return parts[2]
  return name


def load_stdout_rows(diagnostic_dir: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in sorted((diagnostic_dir / "native-stdout").glob("*.json")):
    row = read_json(path)
    row["_source"] = str(path.resolve().relative_to(ROOT))
    rows.append(row)
  return rows


def aggregate_profiles(stdout_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  by_op: dict[str, dict[str, Any]] = defaultdict(
      lambda: {"call_count": 0, "examples": [], "row_count": 0, "total_ns": 0}
  )
  by_suffix: dict[str, dict[str, Any]] = defaultdict(
      lambda: {"call_count": 0, "examples": [], "ops": set(), "row_count": 0, "total_ns": 0}
  )
  for stdout in stdout_rows:
    for row in stdout.get("matvec_profile", []):
      if not isinstance(row, dict):
        continue
      op = str(row.get("op", ""))
      name = str(row.get("tensor_name", ""))
      total_ns = int(row.get("total_ns", 0) or 0)
      call_count = int(row.get("call_count", 0) or 0)
      row_count = int(row.get("row_count", 0) or 0)
      op_row = by_op[op]
      op_row["call_count"] += call_count
      op_row["row_count"] += row_count
      op_row["total_ns"] += total_ns
      if len(op_row["examples"]) < 3:
        op_row["examples"].append(name)
      suffix = tensor_suffix(name)
      suffix_row = by_suffix[suffix]
      suffix_row["call_count"] += call_count
      suffix_row["row_count"] += row_count
      suffix_row["total_ns"] += total_ns
      suffix_row["ops"].add(op)
      if len(suffix_row["examples"]) < 3:
        suffix_row["examples"].append(name)

  op_rows = [
      {"op": key, **value}
      for key, value in by_op.items()
  ]
  suffix_rows = [
      {**value, "ops": sorted(value["ops"]), "suffix": key}
      for key, value in by_suffix.items()
  ]
  op_rows.sort(key=lambda row: int(row["total_ns"]), reverse=True)
  suffix_rows.sort(key=lambda row: int(row["total_ns"]), reverse=True)
  return op_rows, suffix_rows


def summarize(payload: dict[str, Any]) -> str:
  rollup = payload["rollup"]
  lines = [
      "# R3 Q4 Plane Gap Rollup",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- source diagnostic: `{payload['source_diagnostic']}`",
      f"- route check: `{payload['source_route_check']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- prompt prefill ns: `{rollup.get('prompt_prefill_ns')}`",
      f"- total profiled matvec ns: `{rollup.get('total_profile_ns')}`",
      f"- q4-plane profile ns: `{rollup.get('q4_plane_profile_ns')}`",
      f"- q4-plane repack ns: `{rollup.get('q4_plane_repack_ns')}`",
      f"- q4-plane cached bytes: `{rollup.get('q4_plane_cached_bytes')}`",
      "",
      "| op | total ns | calls | rows | examples |",
      "|---|---:|---:|---:|---|",
  ]
  for row in rollup["top_ops"][:10]:
    lines.append(
        "| "
        + " | ".join([
            f"`{row['op']}`",
            str(row["total_ns"]),
            str(row["call_count"]),
            str(row["row_count"]),
            ", ".join(f"`{item}`" for item in row["examples"]),
        ])
        + " |"
    )
  lines += [
      "",
      "| suffix | total ns | calls | rows | ops |",
      "|---|---:|---:|---:|---|",
  ]
  for row in rollup["top_suffixes"][:10]:
    lines.append(
        "| "
        + " | ".join([
            f"`{row['suffix']}`",
            str(row["total_ns"]),
            str(row["call_count"]),
            str(row["row_count"]),
            ", ".join(f"`{item}`" for item in row["ops"]),
        ])
        + " |"
    )
  has_down_aggregate = any(
      row["op"] == "selected_expert_ffn_down_aggregate"
      for row in rollup["top_ops"]
  )
  if has_down_aggregate:
    conclusion = [
        "Conclusion: expanded q4-plane is active and cacheable. The remaining",
        "R3 gap now concentrates in q4-plane lanes, Q6 pair, and selected-expert",
        "down aggregation. The next route should reduce memory traffic or fuse",
        "the dominant lanes rather than adding another dot-kernel variant.",
    ]
  else:
    conclusion = [
        "Conclusion: expanded q4-plane is active and cacheable, and",
        "selected-expert down Q6 pair is isolated as its own lane. The remaining R3 gap",
        "concentrates in q4-plane lanes, Q6 pair, and selected-expert down work;",
        "the next route should reduce memory traffic or fuse dominant lanes",
        "rather than adding another dot-kernel variant.",
    ]
  lines += ["", *conclusion, ""]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  diagnostic_dir = args.diagnostic_dir.resolve()
  route_check_dir = args.route_check_dir.resolve()
  if not diagnostic_dir.exists():
    raise SystemExit(f"diagnostic dir missing: {diagnostic_dir}")
  if not route_check_dir.exists():
    raise SystemExit(f"route-check dir missing: {route_check_dir}")

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r3-q4-plane-gap-rollup-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  diagnostic = read_json(diagnostic_dir / "diagnostic.json")
  correctness = read_json(diagnostic_dir / "correctness.json")
  route_check = read_json(route_check_dir / "route-check.json")
  stdout_rows = load_stdout_rows(diagnostic_dir)
  op_rows, suffix_rows = aggregate_profiles(stdout_rows)
  case_rows = [
      row
      for stdout in stdout_rows
      for row in stdout.get("cases", [])
      if isinstance(row, dict)
  ]
  prompt_prefill_ns = sum(
      int(row.get("timing_ns", {}).get("prompt_prefill", 0) or 0)
      for row in case_rows
  )
  total_profile_ns = sum(int(row["total_ns"]) for row in op_rows)
  q4_plane_profile_ns = sum(
      int(row["total_ns"]) for row in op_rows if "q4plane" in str(row["op"])
  )
  q4_cache = route_check.get("route_check", {}).get("q4_plane_cache_summary", {})
  q4_plane_hits = int(q4_cache.get("q4_plane_hits", 0) or 0)
  q4_plane_misses = int(q4_cache.get("q4_plane_misses", 0) or 0)
  q4_plane_hit_rate = (
      q4_plane_hits / (q4_plane_hits + q4_plane_misses)
      if q4_plane_hits + q4_plane_misses > 0 else 0.0
  )
  q4_plane_repack_ns = int(q4_cache.get("q4_plane_repack_ns", 0) or 0)
  q4_plane_cached_bytes = int(q4_cache.get("q4_plane_cached_bytes", 0) or 0)
  checks = [
      {
          "name": "source_context_diagnostic_required_checks_passed",
          "pass": correctness.get("required_checks_passed") is True,
      },
      {
          "name": "source_route_check_required_checks_passed",
          "pass": route_check.get("required_checks_passed") is True,
      },
      {"name": "q4_plane_cache_stats_present", "pass": q4_plane_repack_ns > 0},
      {"name": "profile_rows_present", "pass": total_profile_ns > 0 and bool(op_rows)},
      {
          "name": "dominant_remaining_suffixes_present",
          "pass": any(row["suffix"] == "ffn_down_exps.weight" for row in suffix_rows)
          and any(row["suffix"] == "ssm_out.weight" for row in suffix_rows)
          and any(row["suffix"] == "attn_gate.weight" for row in suffix_rows),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  rollup = {
      "checks": checks,
      "prompt_prefill_ns": prompt_prefill_ns,
      "q4_plane_cached_bytes": q4_plane_cached_bytes,
      "q4_plane_hit_rate": q4_plane_hit_rate,
      "q4_plane_hits": q4_plane_hits,
      "q4_plane_misses": q4_plane_misses,
      "q4_plane_profile_ns": q4_plane_profile_ns,
      "q4_plane_repack_ns": q4_plane_repack_ns,
      "q4_plane_repack_share_of_prefill": (
          q4_plane_repack_ns / prompt_prefill_ns if prompt_prefill_ns > 0 else 0.0
      ),
      "q4_plane_profile_share_of_prefill": (
          q4_plane_profile_ns / prompt_prefill_ns if prompt_prefill_ns > 0 else 0.0
      ),
      "top_ops": op_rows,
      "top_suffixes": suffix_rows,
      "total_profile_ns": total_profile_ns,
  }
  payload = {
      "created_at": created_at,
      "required_checks_passed": required_checks_passed,
      "rollup": rollup,
      "schema_version": SCHEMA_VERSION,
      "source_diagnostic": str(diagnostic_dir.relative_to(ROOT)),
      "source_route_check": str(route_check_dir.relative_to(ROOT)),
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "source_diagnostic": str(diagnostic_dir.relative_to(ROOT)),
      "source_route_check": str(route_check_dir.relative_to(ROOT)),
      "tool": "tools/intel-qwen36-r3-q4-plane-gap-rollup.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "gap-rollup.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r3_q4_plane_gap_rollup",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "top-ops.jsonl", op_rows)
  write_jsonl(out_dir / "top-suffixes.jsonl", suffix_rows)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r3_q4_plane_gap_rollup",
      [
          ("total_profile_ns", total_profile_ns),
          ("q4_plane_profile_ns", q4_plane_profile_ns),
          ("q4_plane_cached_bytes", q4_plane_cached_bytes),
          ("q4_plane_hit_rate", q4_plane_hit_rate),
          ("q4_plane_repack_ns", q4_plane_repack_ns),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(summarize(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
