#!/usr/bin/env python3
"""Validate that q4k_plane_v0 is wired into the native route artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r3-q4-plane-route-check-v0"
EXPECTED_SENTINEL_001K_SIGNATURE = [271, 198, 21134, 3054, 3437]
EXPANDED_DENSE_SUFFIXES = {
    "attn_gate.weight",
    "attn_k.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_v.weight",
    "ffn_down_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ssm_out.weight",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--diagnostic-dir",
      type=Path,
      default=None,
      help="context-ladder-native-diagnostic artifact with --q4-plane-layout.",
  )
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def latest_q4_plane_diagnostic() -> Path:
  candidates = sorted(ROOT.glob("output/context-ladder-native-diagnostic-*"))
  for path in reversed(candidates):
    diag_path = path / "diagnostic.json"
    if not diag_path.exists():
      continue
    try:
      diag = read_json(diag_path)
    except (OSError, json.JSONDecodeError):
      continue
    diagnostic = diag.get("diagnostic", {})
    if isinstance(diagnostic, dict) and diagnostic.get("q4_plane_layout") is True:
      return path
  raise SystemExit("no q4-plane context diagnostic found")


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


def load_stdout_rows(diagnostic_dir: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  stdout_dir = diagnostic_dir / "native-stdout"
  for path in sorted(stdout_dir.glob("*.json")):
    row = read_json(path)
    row["_source"] = str(path.resolve().relative_to(ROOT))
    rows.append(row)
  return rows


def profile_rows(stdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for stdout in stdout_rows:
    source = stdout.get("_source")
    for row in stdout.get("matvec_profile", []):
      if not isinstance(row, dict):
        continue
      item = dict(row)
      item["source"] = source
      rows.append(item)
  return rows


def case_rows(stdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for stdout in stdout_rows:
    source = stdout.get("_source")
    for row in stdout.get("cases", []):
      if not isinstance(row, dict):
        continue
      item = dict(row)
      item["source"] = source
      rows.append(item)
  return rows


def cache_stats_rows(stdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for stdout in stdout_rows:
    stats = stdout.get("resident_tensor_cache_stats")
    if not isinstance(stats, dict):
      continue
    item = dict(stats)
    item["source"] = stdout.get("_source")
    rows.append(item)
  return rows


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
  total = 0
  for row in rows:
    value = row.get(key)
    if isinstance(value, int):
      total += value
  return total


def summarize(payload: dict[str, Any]) -> str:
  checks = payload["route_check"]
  dense = checks["dense_q4plane_rows"]
  expanded_dense = checks["expanded_dense_q4plane_rows"]
  gate = checks["gate_up_q4plane_rows"]
  down = checks["down_q4plane_rows"]
  down_q6pair = checks["down_q6pair_rows"]
  cache = checks["q4_plane_cache_summary"]
  lines = [
      "# R3 Q4 Plane Route Check",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- source diagnostic: `{payload['source_diagnostic']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- dense q4plane profile rows: `{len(dense)}`",
      f"- expanded dense q4plane profile rows: `{len(expanded_dense)}`",
      f"- gate/up q4plane profile rows: `{len(gate)}`",
      f"- selected-down q4plane profile rows: `{len(down)}`",
      f"- selected-down q6pair profile rows: `{len(down_q6pair)}`",
      f"- q4 plane cached bytes: `{cache.get('q4_plane_cached_bytes')}`",
      f"- q4 plane repack ns: `{cache.get('q4_plane_repack_ns')}`",
      f"- q4 plane hits/misses: `{cache.get('q4_plane_hits')}` / `{cache.get('q4_plane_misses')}`",
      "",
      "| route | sample tensor | call count | rows | total ns |",
      "|---|---|---:|---:|---:|",
  ]
  for name, rows in (
      ("dense", dense[:3]),
      ("expanded_dense", expanded_dense[:3]),
      ("gate_up", gate[:3]),
      ("down", down[:3]),
      ("down_q6pair", down_q6pair[:3]),
  ):
    for row in rows:
      lines.append(
          "| "
          + " | ".join([
              name,
              f"`{row.get('tensor_name')}`",
              str(row.get("call_count")),
              str(row.get("row_count")),
              str(row.get("total_ns")),
          ])
          + " |"
      )
  conclusion = [
      "Conclusion: q4k_plane_v0 is wired into the native route for expanded",
      "Q4_K dense lanes plus selected-expert gate/up and down.",
  ]
  if down_q6pair:
    conclusion.append("Selected-expert Q6 down pair is profiled as a separate lane.")
  conclusion += [
      "This is a correctness/profile route check, not a throughput promotion or",
      "speedup claim.",
  ]
  lines += ["", *conclusion, ""]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  diagnostic_dir = (
      latest_q4_plane_diagnostic()
      if args.diagnostic_dir is None
      else args.diagnostic_dir.resolve()
  )
  if not diagnostic_dir.exists():
    raise SystemExit(f"diagnostic dir missing: {diagnostic_dir}")

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r3-q4-plane-route-check-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  diagnostic = read_json(diagnostic_dir / "diagnostic.json")
  correctness = read_json(diagnostic_dir / "correctness.json")
  stdout_rows = load_stdout_rows(diagnostic_dir)
  profiles = profile_rows(stdout_rows)
  cases = case_rows(stdout_rows)
  cache_stats = cache_stats_rows(stdout_rows)
  q4_plane_cache_summary = {
      "q4_plane_cached_bytes": sum_int(cache_stats, "q4_plane_cached_bytes"),
      "q4_plane_hits": sum_int(cache_stats, "q4_plane_hits"),
      "q4_plane_misses": sum_int(cache_stats, "q4_plane_misses"),
      "q4_plane_repack_ns": sum_int(cache_stats, "q4_plane_repack_ns"),
  }
  dense_q4plane = [
      row for row in profiles
      if row.get("op") in (
          "matvec_tensor_dense_q4plane",
          "matvec_tensor_dense_q4plane_pair",
      )
      and str(row.get("tensor_name", "")).endswith("attn_qkv.weight")
  ]
  expanded_dense_q4plane = [
      row for row in profiles
      if row.get("op") in (
          "matvec_tensor_dense_q4plane",
          "matvec_tensor_dense_q4plane_pair",
      )
      and any(str(row.get("tensor_name", "")).endswith(suffix) for suffix in EXPANDED_DENSE_SUFFIXES)
  ]
  gate_up_q4plane = [
      row for row in profiles
      if row.get("op") in (
          "selected_expert_ffn_gate_swiglu_q4plane",
          "selected_expert_ffn_gate_swiglu_q4plane_pair",
      )
      and str(row.get("tensor_name", "")).endswith("ffn_gate_up_exps.weight")
  ]
  down_q4plane = [
      row for row in profiles
      if row.get("op") in (
          "selected_expert_ffn_down_q4plane",
          "selected_expert_ffn_down_q4plane_expert_major",
      )
      and str(row.get("tensor_name", "")).endswith("ffn_down_exps.weight")
  ]
  down_q6pair = [
      row for row in profiles
      if row.get("op") == "selected_expert_ffn_down_q6pair"
      and str(row.get("tensor_name", "")).endswith("ffn_down_exps.weight")
  ]
  selected_down_q6pair_requested = (
      diagnostic.get("diagnostic", {}).get("selected_expert_down_q6_pair_dot")
      is True
  )
  sentinel_cases = [row for row in cases if row.get("case_id") == "sentinel_001k"]
  checks = [
      {
          "name": "source_context_diagnostic_required_checks_passed",
          "pass": correctness.get("required_checks_passed") is True,
      },
      {
          "name": "q4_plane_flag_recorded",
          "pass": correctness.get("q4_plane_layout") is True
          and diagnostic.get("diagnostic", {}).get("q4_plane_layout") is True
          and all(row.get("q4_plane_layout_enabled") is True for row in stdout_rows),
      },
      {
          "name": "attn_qkv_dense_q4plane_profile_present",
          "pass": len(dense_q4plane) > 0,
      },
      {
          "name": "ffn_gate_up_exps_q4plane_profile_present",
          "pass": len(gate_up_q4plane) > 0,
      },
      {
          "name": "expanded_dense_q4plane_profile_present",
          "pass": len(expanded_dense_q4plane) > 0,
      },
      {
          "name": "selected_down_q4plane_profile_present",
          "pass": len(down_q4plane) > 0,
      },
      {
          "name": "selected_down_q6pair_profile_present_when_requested",
          "pass": (not selected_down_q6pair_requested) or len(down_q6pair) > 0,
      },
      {
          "name": "q4_plane_cache_stats_present",
          "pass": bool(cache_stats)
          and q4_plane_cache_summary["q4_plane_misses"] > 0
          and q4_plane_cache_summary["q4_plane_repack_ns"] > 0
          and q4_plane_cache_summary["q4_plane_cached_bytes"] > 0,
      },
      {
          "name": "sentinel_001k_first_token_stable",
          "pass": bool(sentinel_cases)
          and all(row.get("generated_token_ids", [None])[0] == 271 for row in sentinel_cases),
      },
      {
          "name": "sentinel_001k_topk_signature_stable",
          "pass": bool(sentinel_cases)
          and all(
              row.get("first_token_top_logprob_id_signature")
              == EXPECTED_SENTINEL_001K_SIGNATURE
              for row in sentinel_cases
          ),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "created_at": created_at,
      "required_checks_passed": required_checks_passed,
      "route_check": {
          "case_rows": cases,
          "checks": checks,
          "cache_stats_rows": cache_stats,
          "dense_q4plane_rows": dense_q4plane,
          "down_q4plane_rows": down_q4plane,
          "down_q6pair_rows": down_q6pair,
          "expanded_dense_q4plane_rows": expanded_dense_q4plane,
          "gate_up_q4plane_rows": gate_up_q4plane,
          "q4_plane_cache_summary": q4_plane_cache_summary,
      },
      "schema_version": SCHEMA_VERSION,
      "source_diagnostic": str(diagnostic_dir.relative_to(ROOT)),
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "source_diagnostic": str(diagnostic_dir.relative_to(ROOT)),
      "tool": "tools/intel-qwen36-r3-q4-plane-route-check.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "route-check.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r3_q4_plane_route_check",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r3_q4_plane_route_check",
      [
          ("dense_q4plane_profile_rows", len(dense_q4plane)),
          ("expanded_dense_q4plane_profile_rows", len(expanded_dense_q4plane)),
          ("gate_up_q4plane_profile_rows", len(gate_up_q4plane)),
          ("down_q4plane_profile_rows", len(down_q4plane)),
          ("down_q6pair_profile_rows", len(down_q6pair)),
          ("q4_plane_cached_bytes", q4_plane_cache_summary["q4_plane_cached_bytes"]),
          ("q4_plane_hits", q4_plane_cache_summary["q4_plane_hits"]),
          ("q4_plane_misses", q4_plane_cache_summary["q4_plane_misses"]),
          ("q4_plane_repack_ns", q4_plane_cache_summary["q4_plane_repack_ns"]),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(summarize(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
