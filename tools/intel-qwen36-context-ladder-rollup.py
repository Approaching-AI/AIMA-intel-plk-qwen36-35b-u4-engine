#!/usr/bin/env python3
"""Roll up native context-ladder diagnostic artifacts.

The per-run native context-ladder diagnostic intentionally isolates cases in
target processes. This tool combines compatible artifacts so we can extend the
ladder without rerunning earlier buckets, while still checking that all rows use
the same route and cold no-prefix policy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-context-ladder-rollup-v0"
DIAGNOSTIC_SCHEMA = "intel-qwen36-context-ladder-native-diagnostic-v0"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "artifacts",
      nargs="+",
      type=Path,
      help="context-ladder-native-diagnostic artifact directories to roll up.",
  )
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
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


def rel(path: Path) -> str:
  return path.resolve().relative_to(ROOT).as_posix()


def artifact_payload(path: Path) -> dict[str, Any]:
  diag_path = path / "diagnostic.json"
  correctness_path = path / "correctness.json"
  if not diag_path.exists():
    raise SystemExit(f"missing diagnostic.json: {path}")
  if not correctness_path.exists():
    raise SystemExit(f"missing correctness.json: {path}")
  payload = load_json(diag_path)
  correctness = load_json(correctness_path)
  return {
      "artifact": path,
      "correctness": correctness,
      "diagnostic": payload,
  }


def artifact_summary(item: dict[str, Any]) -> dict[str, Any]:
  artifact = item["artifact"]
  payload = item["diagnostic"]
  diag = payload.get("diagnostic", {})
  return {
      "artifact": rel(artifact),
      "case_count": len(diag.get("case_results", [])),
      "case_ids": diag.get("case_ids", []),
      "case_process_isolation": diag.get("case_process_isolation"),
      "created_at": payload.get("created_at"),
      "dense_q6_pair_dot": diag.get("dense_q6_pair_dot"),
      "max_new_tokens": diag.get("max_new_tokens"),
      "prefix_cache_enabled": diag.get("prefix_cache_enabled"),
      "required_checks_passed": payload.get("required_checks_passed"),
      "resident_cache": diag.get("resident_cache"),
      "route": diag.get("route"),
      "schema_version": payload.get("schema_version"),
      "selected_expert_down_q4_pair_dot": diag.get(
          "selected_expert_down_q4_pair_dot", False
      ),
      "speedup_claims_allowed": payload.get("speedup_claims_allowed"),
  }


def normalized_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for item in items:
    artifact = item["artifact"]
    for row in item["diagnostic"].get("diagnostic", {}).get("case_results", []):
      out = dict(row)
      out["artifact"] = rel(artifact)
      rows.append(out)
  rows.sort(key=lambda row: (
      str(row.get("kind")),
      int(row.get("prompt_token_count", -1)),
      str(row.get("case_id")),
      str(row.get("artifact")),
  ))
  return rows


def consistency_checks(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  summaries = [artifact_summary(item) for item in items]
  routes = {summary.get("route") for summary in summaries}
  dense_q6_pair_dot = {summary.get("dense_q6_pair_dot") for summary in summaries}
  selected_expert_down_q4_pair_dot = {
      summary.get("selected_expert_down_q4_pair_dot") for summary in summaries
  }
  max_new_tokens = {summary.get("max_new_tokens") for summary in summaries}
  resident_cache = {summary.get("resident_cache") for summary in summaries}
  case_ids = [row.get("case_id") for row in rows]
  duplicate_case_ids = sorted(
      case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
  )
  return [
      {
          "name": "all_artifacts_have_expected_schema",
          "pass": all(summary.get("schema_version") == DIAGNOSTIC_SCHEMA for summary in summaries),
      },
      {
          "name": "all_artifacts_required_checks_passed",
          "pass": all(summary.get("required_checks_passed") is True for summary in summaries),
      },
      {
          "name": "all_artifacts_disable_speedup_claims",
          "pass": all(summary.get("speedup_claims_allowed") is False for summary in summaries),
      },
      {
          "name": "all_artifacts_are_cold_no_prefix",
          "pass": all(summary.get("prefix_cache_enabled") is False for summary in summaries),
      },
      {
          "name": "all_artifacts_use_case_process_isolation",
          "pass": all(summary.get("case_process_isolation") is True for summary in summaries),
      },
      {
          "name": "route_consistent",
          "pass": len(routes) == 1 and None not in routes,
          "routes": sorted(str(route) for route in routes),
      },
      {
          "name": "dense_q6_pair_dot_policy_consistent",
          "pass": len(dense_q6_pair_dot) == 1,
          "dense_q6_pair_dot": sorted(str(value) for value in dense_q6_pair_dot),
      },
      {
          "name": "selected_expert_down_q4_pair_dot_policy_consistent",
          "pass": len(selected_expert_down_q4_pair_dot) == 1,
          "selected_expert_down_q4_pair_dot": sorted(
              str(value) for value in selected_expert_down_q4_pair_dot
          ),
      },
      {
          "name": "max_new_tokens_consistent",
          "pass": len(max_new_tokens) == 1 and None not in max_new_tokens,
          "max_new_tokens": sorted(str(value) for value in max_new_tokens),
      },
      {
          "name": "resident_cache_policy_consistent",
          "pass": len(resident_cache) == 1 and None not in resident_cache,
          "resident_cache": sorted(str(value) for value in resident_cache),
      },
      {
          "name": "no_duplicate_case_ids",
          "pass": not duplicate_case_ids,
          "duplicates": duplicate_case_ids,
      },
      {
          "name": "all_rows_have_positive_prefill",
          "pass": all(
              isinstance(row.get("prompt_prefill_ns"), int)
              and row["prompt_prefill_ns"] > 0
              and isinstance(row.get("prompt_token_count"), int)
              and row["prompt_token_count"] > 0
              for row in rows
          ),
      },
  ]


def monotonic_checks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  checks: list[dict[str, Any]] = []
  series_rows: list[dict[str, Any]] = []
  kinds = sorted({row.get("kind") for row in rows if row.get("kind")})
  for kind in kinds:
    group = [
        row for row in rows
        if row.get("kind") == kind and isinstance(row.get("prompt_token_count"), int)
    ]
    group.sort(key=lambda row: int(row["prompt_token_count"]))
    prompt_counts = [row.get("prompt_token_count") for row in group]
    prefill_ns = [row.get("prompt_prefill_ns") for row in group]
    monotonic = all(
        isinstance(prefill_ns[i], int)
        and isinstance(prefill_ns[i + 1], int)
        and prefill_ns[i] <= prefill_ns[i + 1]
        for i in range(len(prefill_ns) - 1)
    )
    checks.append({
        "bucket_count": len(group),
        "name": f"{kind}_prefill_monotonic",
        "pass": len(group) >= 2 and monotonic,
        "prompt_prefill_ns": prefill_ns,
        "prompt_token_counts": prompt_counts,
      })
    for row in group:
      per_token = row.get("prompt_prefill_ns_per_token")
      if per_token is None and row.get("prompt_token_count"):
        per_token = row.get("prompt_prefill_ns") / row.get("prompt_token_count")
      series_rows.append({
          "artifact": row.get("artifact"),
          "case_id": row.get("case_id"),
          "first_generated_token_id": row.get("first_generated_token_id"),
          "kind": kind,
          "prompt_prefill_ns": row.get("prompt_prefill_ns"),
          "prompt_prefill_ns_per_token": per_token,
          "prompt_token_count": row.get("prompt_token_count"),
          "top_logprob_id_signature": row.get("top_logprob_id_signature"),
      })
  return checks, series_rows


def build_summary(payload: dict[str, Any]) -> str:
  rollup = payload["rollup"]
  lines = [
      "# Native Context Ladder Rollup",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- artifacts: {len(rollup['artifacts'])}",
      f"- case rows: {len(rollup['case_results'])}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | kind | prompt tokens | prefill ns | prefill ns/token | first token | artifact |",
      "|---|---|---:|---:|---:|---:|---|",
  ]
  for row in rollup["series"]:
    per_token = row.get("prompt_prefill_ns_per_token")
    per_token_text = f"{per_token:.2f}" if isinstance(per_token, (float, int)) else ""
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("kind")),
            str(row.get("prompt_token_count")),
            str(row.get("prompt_prefill_ns")),
            per_token_text,
            str(row.get("first_generated_token_id")),
            str(row.get("artifact")),
        ])
        + " |"
    )
  lines += [
      "",
      "This rollup checks cold no-prefix context-ladder diagnostics only. It is",
      "not a promotion benchmark matrix and must not be used as a speedup claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/context-ladder-rollup-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  items = [artifact_payload(path.resolve()) for path in args.artifacts]
  rows = normalized_rows(items)
  checks = consistency_checks(items, rows)
  monotonic, series = monotonic_checks(rows)
  checks.extend(monotonic)
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "created_at": created_at,
      "required_checks_passed": required_checks_passed,
      "rollup": {
          "artifacts": [artifact_summary(item) for item in items],
          "case_results": rows,
          "series": series,
      },
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-context-ladder-rollup.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "rollup.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "context_ladder_rollup",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "case-results.jsonl", rows)
  write_jsonl(out_dir / "series.jsonl", series)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "context_ladder_rollup",
      [
          ("artifact_count", len(items)),
          ("case_count", len(rows)),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
