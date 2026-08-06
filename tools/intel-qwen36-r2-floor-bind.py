#!/usr/bin/env python3
"""Bind R2 native rows to fresh same-host floor and acceptance matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r2-floor-bind-v0"
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
)
REQUIRED_BUCKETS = (1024, 2048, 4096, 8192)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--native-rollup", type=Path, required=True)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--floor-artifact", type=Path, action="append", required=True)
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


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


def ratio(num: Any, den: Any) -> float | None:
  if not isinstance(num, (int, float)) or not isinstance(den, (int, float)) or den == 0:
    return None
  return round(float(num) / float(den), 8)


def load_floor_artifacts(paths: list[Path]) -> dict[int, dict[str, Any]]:
  floors: dict[int, dict[str, Any]] = {}
  for artifact in paths:
    artifact = artifact.resolve()
    correctness = read_json(artifact / "correctness.json")
    row = read_json(artifact / "row.json").get("row", {})
    bucket = row.get("bucket")
    if not isinstance(bucket, int):
      raise SystemExit(f"{artifact}: missing bucket")
    floors[bucket] = {
        "artifact": rel(artifact),
        "decode_tok_s": row.get("decode_tokens_s"),
        "mode": row.get("llama_bench_mode"),
        "output_tokens_requested": row.get("output_tokens_requested"),
        "prefill_tok_s": row.get("prefill_tokens_s"),
        "required_checks_passed": correctness.get("required_checks_passed"),
      }
  return floors


def build_summary(payload: dict[str, Any]) -> str:
  rows = payload["bound_rows"]
  lines = [
      "# R2 Floor Bound Denominator",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- native rollup: `{payload['native_rollup']}`",
      f"- acceptance matrix: `{payload['acceptance_matrix']}`",
      f"- acceptance sha256: `{payload['acceptance_matrix_sha256']}`",
      f"- r2 denominator gate closed: `{str(payload['r2_denominator_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | bucket | native decode | fresh floor | vs floor | roofline util |",
      "|---|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    floor = row["fresh_floor"]
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("bucket")),
            str(row.get("decode_continuation_tok_s")),
            str(floor.get("decode_tok_s")),
            str(row.get("decode_vs_fresh_floor")),
            str(row.get("decode_roofline_util")),
        ])
        + " |"
    )
  lines += [
      "",
      "This closes the R2 denominator evidence gate. It is not a speedup claim;",
      "it establishes the floor/roofline judge for the next optimization step.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r2-floor-bind-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  acceptance = read_json(args.acceptance)
  same_host = acceptance.get("same_host_floor", {})
  floors = load_floor_artifacts(args.floor_artifact)
  native_correctness = read_json(args.native_rollup / "correctness.json")
  native_rows = read_jsonl(args.native_rollup / "case-results.jsonl")

  bound_rows: list[dict[str, Any]] = []
  for row in native_rows:
    bucket = row.get("bucket")
    floor = floors.get(bucket)
    if floor is None:
      continue
    enriched = dict(row)
    enriched["fresh_floor"] = floor
    enriched["decode_vs_fresh_floor"] = ratio(
        row.get("decode_continuation_tok_s"), floor.get("decode_tok_s")
    )
    enriched["prefill_vs_fresh_floor"] = ratio(
        row.get("prefill_tok_s"), floor.get("prefill_tok_s")
    )
    bound_rows.append(enriched)

  observed_buckets = sorted({
      int(row["bucket"])
      for row in bound_rows
      if isinstance(row.get("bucket"), int)
  })
  matrix_floor_artifacts = same_host.get("artifacts", {})
  checks = [
      {
          "name": "native_rollup_required_checks_passed",
          "pass": native_correctness.get("required_checks_passed") is True
          and native_correctness.get("native_rows_complete") is True,
      },
      {
          "name": "fresh_floor_artifacts_required_checks_passed",
          "pass": all(floor.get("required_checks_passed") is True for floor in floors.values()),
      },
      {
          "name": "fresh_floor_buckets_complete",
          "pass": sorted(floors) == list(REQUIRED_BUCKETS),
          "observed_buckets": sorted(floors),
      },
      {
          "name": "acceptance_matrix_has_same_host_floor_binding",
          "pass": same_host.get("is_bootstrap_placeholder") is False
          and sorted(int(bucket) for bucket in same_host.get("decode_tokens_s", {})) == list(REQUIRED_BUCKETS),
      },
      {
          "name": "acceptance_artifacts_match_floor_inputs",
          "pass": all(
              matrix_floor_artifacts.get(str(bucket), "").rstrip("/")
              == floor["artifact"].rstrip("/")
              for bucket, floor in floors.items()
          ),
      },
      {
          "name": "bound_native_rows_cover_1k_8k",
          "pass": observed_buckets == list(REQUIRED_BUCKETS) and len(bound_rows) == 8,
      },
      {
          "name": "all_bound_rows_generated_512_tokens",
          "pass": all(row.get("generated_token_count") == 512 for row in bound_rows)
          and len(bound_rows) == 8,
      },
      {
          "name": "all_rows_report_fresh_floor_ratios",
          "pass": all(
              row.get("decode_vs_fresh_floor") is not None
              and row.get("prefill_vs_fresh_floor") is not None
              for row in bound_rows
          ) and len(bound_rows) == 8,
      },
  ]
  r2_denominator_gate_closed = all(check["pass"] for check in checks)
  payload = {
      "acceptance_matrix": rel(args.acceptance),
      "acceptance_matrix_sha256": iq36_local.sha256_file(args.acceptance),
      "bound_rows": bound_rows,
      "created_at": created_at,
      "fresh_floor": {
          "artifacts": {str(bucket): floor["artifact"] for bucket, floor in sorted(floors.items())},
          "decode_tokens_s": {str(bucket): floor["decode_tok_s"] for bucket, floor in sorted(floors.items())},
          "prefill_tokens_s": {str(bucket): floor["prefill_tok_s"] for bucket, floor in sorted(floors.items())},
      },
      "native_rollup": rel(args.native_rollup),
      "r2_denominator_gate_closed": r2_denominator_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "r2_denominator_gate_closed": r2_denominator_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r2-floor-bind.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "floor-bind.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r2_speed_denominator_floor_bound",
      "r2_denominator_gate_closed": r2_denominator_gate_closed,
      "required_checks_passed": r2_denominator_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "case-results.jsonl", bound_rows)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r2_floor_bind",
      [
          ("bound_row_count", len(bound_rows)),
          ("fresh_floor_bucket_count", len(floors)),
          ("r2_denominator_gate_closed", r2_denominator_gate_closed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if r2_denominator_gate_closed else 1


if __name__ == "__main__":
  raise SystemExit(main())
