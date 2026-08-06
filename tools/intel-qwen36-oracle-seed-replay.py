#!/usr/bin/env python3
"""Replay-check native token outputs against the staged CPU llama.cpp seed."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-oracle-seed-stage-v0"
DEFAULT_ORACLE_JSONL = (
    ROOT
    / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--oracle-jsonl",
      type=Path,
      default=DEFAULT_ORACLE_JSONL,
      help="Staged token/top-k seed JSONL.",
  )
  parser.add_argument(
      "--candidate-jsonl",
      type=Path,
      default=None,
      help="Candidate native output JSONL to compare.",
  )
  parser.add_argument(
      "--fixture-from-oracle",
      action="store_true",
      help="Use a candidate copied from the oracle seed for a verifier smoke test.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-oracle-seed-replay-<UTC>.",
  )
  return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: row must be a JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.write_text(
      "".join(
          json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
          for row in rows
      ),
      encoding="utf-8",
  )


def is_int_list(value: Any) -> bool:
  return isinstance(value, list) and all(isinstance(item, int) for item in value)


def by_case_id(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
  by_id: dict[str, dict[str, Any]] = {}
  duplicates: list[str] = []
  for row in rows:
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
      duplicates.append("<missing-case-id>")
      continue
    if case_id in by_id:
      duplicates.append(case_id)
    by_id[case_id] = row
  return by_id, duplicates


def targets_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
  targets = row.get("generation_targets")
  if not isinstance(targets, list):
    return {}
  out: dict[str, dict[str, Any]] = {}
  for target in targets:
    if not isinstance(target, dict):
      continue
    name = target.get("target")
    if isinstance(name, str) and name:
      out[name] = target
  return out


def first_int(value: Any) -> int | None:
  if is_int_list(value) and value:
    return value[0]
  return None


def top_signature(target: dict[str, Any] | None) -> list[int] | None:
  if not isinstance(target, dict):
    return None
  signature = target.get("top_logprob_id_signature")
  if is_int_list(signature) and signature:
    return signature
  top_logprobs = target.get("top_logprobs")
  if isinstance(top_logprobs, list) and top_logprobs:
    ids: list[int] = []
    for item in top_logprobs:
      if isinstance(item, dict) and isinstance(item.get("id"), int):
        ids.append(item["id"])
    return ids if ids else None
  return None


def candidate_rows_from_oracle(oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for row in oracle_rows:
    candidate_targets: list[dict[str, Any]] = []
    for target in row.get("generation_targets", []):
      if not isinstance(target, dict):
        continue
      candidate_targets.append(
          {
              "generated_token_ids": copy.deepcopy(target.get("generated_token_ids")),
              "generated_token_ids_sha256": target.get("generated_token_ids_sha256"),
              "max_new_tokens": target.get("max_new_tokens"),
              "target": target.get("target"),
              "text_sha256": target.get("text_sha256"),
              "top_logprob_id_signature": copy.deepcopy(
                  target.get("top_logprob_id_signature")
              ),
              "top_logprobs": copy.deepcopy(target.get("top_logprobs")),
          }
      )
    rows.append(
        {
            "case_id": row.get("case_id"),
            "generation_targets": candidate_targets,
            "native_output_source": "oracle-fixture",
            "prompt_token_ids": copy.deepcopy(row.get("prompt_token_ids")),
            "prompt_utf8_sha256": row.get("prompt_utf8_sha256"),
            "schema_version": SCHEMA_VERSION,
            "workstream": WORKSTREAM,
        }
    )
  return rows


def compare_target(
    target_name: str,
    oracle_target: dict[str, Any],
    candidate_target: dict[str, Any] | None,
) -> dict[str, Any]:
  expected_ids = oracle_target.get("generated_token_ids")
  actual_ids = (
      candidate_target.get("generated_token_ids")
      if isinstance(candidate_target, dict)
      else None
  )
  expected_text_sha = oracle_target.get("text_sha256")
  actual_text_sha = (
      candidate_target.get("text_sha256")
      if isinstance(candidate_target, dict)
      else None
  )
  expected_top_signature = top_signature(oracle_target)
  actual_top_signature = top_signature(candidate_target)
  return {
      "actual_generated_token_ids": actual_ids,
      "actual_text_sha256": actual_text_sha,
      "actual_top_logprob_id_signature": actual_top_signature,
      "expected_generated_token_ids": expected_ids,
      "expected_text_sha256": expected_text_sha,
      "expected_top_logprob_id_signature": expected_top_signature,
      "generated_token_ids_match": is_int_list(actual_ids) and actual_ids == expected_ids,
      "generated_token_ids_present": is_int_list(actual_ids) and bool(actual_ids),
      "target": target_name,
      "target_present": isinstance(candidate_target, dict),
      "text_sha256_match": isinstance(actual_text_sha, str)
      and actual_text_sha == expected_text_sha,
      "text_sha256_present": isinstance(actual_text_sha, str) and bool(actual_text_sha),
      "top1_id_match": first_int(actual_top_signature) == first_int(expected_top_signature),
      "top_logprob_id_signature_match": actual_top_signature == expected_top_signature,
      "top_logprob_id_signature_present": is_int_list(actual_top_signature)
      and bool(actual_top_signature),
  }


def compare_case(
    case_id: str,
    oracle: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
  oracle_targets = targets_by_name(oracle)
  candidate_targets = targets_by_name(candidate or {})
  expected_target_names = set(oracle_targets)
  actual_target_names = set(candidate_targets)
  target_results = [
      compare_target(target_name, oracle_targets[target_name], candidate_targets.get(target_name))
      for target_name in sorted(expected_target_names)
  ]
  prompt_hash = candidate.get("prompt_utf8_sha256") if isinstance(candidate, dict) else None
  prompt_token_ids = (
      candidate.get("prompt_token_ids") if isinstance(candidate, dict) else None
  )
  schema_version = candidate.get("schema_version") if isinstance(candidate, dict) else None
  workstream = candidate.get("workstream") if isinstance(candidate, dict) else None
  return {
      "candidate_present": isinstance(candidate, dict),
      "case_id": case_id,
      "extra_targets": sorted(actual_target_names - expected_target_names),
      "missing_targets": sorted(expected_target_names - actual_target_names),
      "prompt_token_ids_match": is_int_list(prompt_token_ids)
      and prompt_token_ids == oracle.get("prompt_token_ids"),
      "prompt_token_ids_present": is_int_list(prompt_token_ids) and bool(prompt_token_ids),
      "prompt_utf8_sha256_match": isinstance(prompt_hash, str)
      and prompt_hash == oracle.get("prompt_utf8_sha256"),
      "prompt_utf8_sha256_present": isinstance(prompt_hash, str) and bool(prompt_hash),
      "schema_version_match": schema_version == SCHEMA_VERSION,
      "schema_version_present": isinstance(schema_version, str) and bool(schema_version),
      "target_results": target_results,
      "workstream_match": workstream == WORKSTREAM,
      "workstream_present": isinstance(workstream, str) and bool(workstream),
  }


def evaluate(
    oracle_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
  oracle_by_id, oracle_duplicates = by_case_id(oracle_rows)
  candidate_by_id, candidate_duplicates = by_case_id(candidate_rows)
  oracle_ids = set(oracle_by_id)
  candidate_ids = set(candidate_by_id)
  missing_cases = sorted(oracle_ids - candidate_ids)
  extra_cases = sorted(candidate_ids - oracle_ids)
  case_results = [
      compare_case(case_id, oracle_by_id[case_id], candidate_by_id.get(case_id))
      for case_id in sorted(oracle_ids)
  ]
  flat_targets = [
      target
      for row in case_results
      for target in row.get("target_results", [])
      if isinstance(target, dict)
  ]
  first_targets = [
      target for target in flat_targets if target.get("target") == "first_token"
  ]
  short_targets = [
      target for target in flat_targets if target.get("target") == "short_generation"
  ]

  def all_cases(field: str) -> bool:
    return bool(case_results) and all(row.get(field) is True for row in case_results)

  def all_targets(targets: list[dict[str, Any]], field: str) -> bool:
    return bool(targets) and all(target.get(field) is True for target in targets)

  checks = [
      {
          "duplicates": oracle_duplicates,
          "name": "oracle_rows_loaded",
          "pass": bool(oracle_rows) and not oracle_duplicates,
          "row_count": len(oracle_rows),
      },
      {
          "duplicates": candidate_duplicates,
          "name": "candidate_rows_loaded",
          "pass": bool(candidate_rows) and not candidate_duplicates,
          "row_count": len(candidate_rows),
      },
      {
          "extra_cases": extra_cases,
          "missing_cases": missing_cases,
          "name": "candidate_case_ids_exact",
          "pass": bool(oracle_ids) and not missing_cases and not extra_cases,
      },
      {"name": "candidate_workstream", "pass": all_cases("workstream_match")},
      {"expected": SCHEMA_VERSION, "name": "candidate_schema_version", "pass": all_cases("schema_version_match")},
      {"name": "prompt_utf8_sha256_match", "pass": all_cases("prompt_utf8_sha256_match")},
      {"name": "prompt_token_ids_match", "pass": all_cases("prompt_token_ids_match")},
      {
          "name": "generation_targets_present",
          "pass": bool(flat_targets)
          and all(
              row.get("missing_targets") == [] and row.get("extra_targets") == []
              for row in case_results
          ),
      },
      {
          "name": "generated_token_ids_present",
          "pass": bool(flat_targets)
          and all_targets(flat_targets, "generated_token_ids_present"),
      },
      {
          "name": "first_token_generated_ids_match",
          "pass": all_targets(first_targets, "generated_token_ids_match"),
      },
      {
          "name": "short_generation_generated_ids_match",
          "pass": all_targets(short_targets, "generated_token_ids_match"),
      },
      {
          "name": "decoded_text_sha256_match",
          "pass": bool(flat_targets) and all_targets(flat_targets, "text_sha256_match"),
      },
      {
          "name": "first_token_top1_id_match",
          "pass": all_targets(first_targets, "top1_id_match"),
      },
      {
          "name": "top_logprob_id_signature_match",
          "pass": all_targets(flat_targets, "top_logprob_id_signature_match"),
      },
  ]
  return {
      "case_results": case_results,
      "checks": checks,
      "created_at": iso_now(),
      "extra_cases": extra_cases,
      "missing_cases": missing_cases,
      "required_checks_passed": all(check["pass"] is True for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }


def build_summary(result: dict[str, Any], out_dir: Path) -> str:
  passed = result["required_checks_passed"]
  failed = [check["name"] for check in result["checks"] if check.get("pass") is not True]
  lines = [
      "# R0 oracle seed replay",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- output: `{out_dir}`",
  ]
  if failed:
    lines.append(f"- failed checks: {', '.join(failed)}")
  lines.extend(
      [
          "",
          "This is a deterministic seed replay check only. It does not close the",
          "teacher-forced distribution or per-boundary tensor gates.",
          "",
      ]
  )
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  if args.candidate_jsonl is None and not args.fixture_from_oracle:
    raise SystemExit("provide --candidate-jsonl or --fixture-from-oracle")
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-oracle-seed-replay-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_rows = load_jsonl(args.oracle_jsonl)
  if args.fixture_from_oracle:
    candidate_rows = candidate_rows_from_oracle(oracle_rows)
    candidate_path = out_dir / "fixture-candidate.jsonl"
    write_jsonl(candidate_path, candidate_rows)
  else:
    candidate_path = args.candidate_jsonl
    candidate_rows = load_jsonl(candidate_path)

  result = evaluate(oracle_rows, candidate_rows)
  result["oracle_jsonl"] = str(args.oracle_jsonl)
  result["candidate_jsonl"] = str(candidate_path)
  write_json(out_dir / "correctness.json", result)
  (out_dir / "summary.md").write_text(build_summary(result, out_dir), encoding="utf-8")
  print(f"oracle seed replay required_checks_passed={result['required_checks_passed']}")
  print(f"replay output: {out_dir}")
  if result["required_checks_passed"] is not True:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
