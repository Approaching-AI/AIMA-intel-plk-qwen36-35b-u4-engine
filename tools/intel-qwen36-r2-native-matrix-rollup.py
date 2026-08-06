#!/usr/bin/env python3
"""Roll up R2 native matrix artifacts into one 1k-8k denominator view."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r2-native-matrix-rollup-v0"
REQUIRED_CASE_IDS = (
    "sentinel_001k",
    "prefill_shape_001k",
    "sentinel_002k",
    "prefill_shape_002k",
    "sentinel_004k",
    "prefill_shape_004k",
    "sentinel_008k",
    "prefill_shape_008k",
)
REQUIRED_BUCKETS = (1024, 2048, 4096, 8192)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("artifacts", nargs="+", type=Path)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      text = line.strip()
      if not text:
        continue
      value = json.loads(text)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def artifact_rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


def build_summary(payload: dict[str, Any]) -> str:
  rollup = payload["rollup"]
  lines = [
      "# R2 Native Matrix Rollup",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- row count: `{len(rollup['case_results'])}`",
      f"- native rows complete: `{str(rollup['native_rows_complete']).lower()}`",
      f"- same-host floor refreshed: `false`",
      f"- r2 exit gate closed: `{str(payload['r2_exit_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | bucket | gen | prefill tok/s | decode tok/s | vs floor | roofline util | source |",
      "|---|---:|---:|---:|---:|---:|---:|---|",
  ]
  for row in rollup["case_results"]:
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("bucket")),
            str(row.get("generated_token_count")),
            str(row.get("prefill_tok_s")),
            str(row.get("decode_continuation_tok_s")),
            str(row.get("decode_vs_floor")),
            str(row.get("decode_roofline_util")),
            f"`{row.get('source_artifact')}`",
        ])
        + " |"
    )
  lines += [
      "",
      "This closes the native-row coverage part of R2 only. R2 remains open until",
      "the same-host llama.cpp floor is refreshed and bound into the acceptance",
      "matrix.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r2-native-matrix-rollup-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  sources: list[dict[str, Any]] = []
  rows_by_case: dict[str, dict[str, Any]] = {}
  for artifact in args.artifacts:
    artifact = artifact.resolve()
    matrix_path = artifact / "matrix.json"
    correctness_path = artifact / "correctness.json"
    rows_path = artifact / "case-results.jsonl"
    matrix = read_json(matrix_path)
    correctness = read_json(correctness_path)
    rows = read_jsonl(rows_path)
    sources.append({
        "artifact": artifact_rel(artifact),
        "case_count": len(rows),
        "required_checks_passed": correctness.get("required_checks_passed"),
        "r2_exit_gate_closed": correctness.get("r2_exit_gate_closed"),
        "schema_version": matrix.get("schema_version"),
        "speedup_claims_allowed": correctness.get("speedup_claims_allowed"),
      })
    for row in rows:
      case_id = row.get("case_id")
      if not isinstance(case_id, str):
        continue
      enriched = dict(row)
      enriched["source_artifact"] = artifact_rel(artifact)
      rows_by_case[case_id] = enriched

  case_results = [rows_by_case[case_id] for case_id in REQUIRED_CASE_IDS if case_id in rows_by_case]
  observed_case_ids = [row["case_id"] for row in case_results]
  observed_buckets = sorted({
      int(row["bucket"])
      for row in case_results
      if isinstance(row.get("bucket"), int)
  })
  missing_cases = [case_id for case_id in REQUIRED_CASE_IDS if case_id not in rows_by_case]
  checks = [
      {
          "name": "source_artifacts_present",
          "pass": all((artifact / "case-results.jsonl").exists() for artifact in args.artifacts),
      },
      {
          "name": "source_artifacts_do_not_allow_speedup_claims",
          "pass": all(source.get("speedup_claims_allowed") is False for source in sources),
      },
      {
          "name": "required_case_ids_present",
          "pass": not missing_cases,
          "missing_cases": missing_cases,
      },
      {
          "name": "required_buckets_present",
          "pass": observed_buckets == list(REQUIRED_BUCKETS),
          "observed_buckets": observed_buckets,
      },
      {
          "name": "all_rows_generated_512_tokens",
          "pass": all(row.get("generated_token_count") == 512 for row in case_results)
          and len(case_results) == len(REQUIRED_CASE_IDS),
      },
      {
          "name": "all_rows_have_prefill_decode_and_alignment",
          "pass": all(
              row.get("prefill_tok_s") is not None
              and row.get("decode_continuation_tok_s") is not None
              and row.get("decode_vs_floor") is not None
              and row.get("decode_roofline_util") is not None
              for row in case_results
          ) and len(case_results) == len(REQUIRED_CASE_IDS),
      },
      {
          "name": "cold_no_prefix_floor_still_bootstrap",
          "pass": all(row.get("floor", {}).get("is_bootstrap_placeholder") is True for row in case_results),
      },
  ]
  native_rows_complete = all(check["pass"] for check in checks[:-1])
  r2_exit_checks = [
      {"name": "native_1k_8k_512_rows_complete", "pass": native_rows_complete},
      {
          "name": "same_host_floor_refreshed_not_bootstrap",
          "pass": False,
          "status": "pending_refresh",
      },
      {
          "name": "acceptance_matrix_bound_to_fresh_floor",
          "pass": False,
          "status": "pending_refresh",
      },
  ]
  r2_exit_gate_closed = native_rows_complete and all(check["pass"] for check in r2_exit_checks)
  payload = {
      "created_at": created_at,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "rollup": {
          "case_ids": observed_case_ids,
          "case_results": case_results,
          "native_rows_complete": native_rows_complete,
          "observed_buckets": observed_buckets,
          "required_case_ids": list(REQUIRED_CASE_IDS),
          "source_artifacts": sources,
      },
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "native_rows_complete": native_rows_complete,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r2-native-matrix-rollup.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "matrix-rollup.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r2_native_speed_denominator_matrix_rollup",
      "native_rows_complete": native_rows_complete,
      "r2_exit_checks": r2_exit_checks,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "required_checks_passed": native_rows_complete,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "case-results.jsonl", case_results)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r2_native_matrix_rollup",
      [
          ("case_count", len(case_results)),
          ("native_rows_complete", native_rows_complete),
          ("r2_exit_gate_closed", r2_exit_gate_closed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if native_rows_complete else 1


if __name__ == "__main__":
  raise SystemExit(main())
