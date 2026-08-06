#!/usr/bin/env python3
"""Assemble the staged R0 oracle evidence into a full bundle candidate."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-full-bundle-v0"
EDGE_CASE_IDS = {"sentinel_256k", "prefill_shape_256k"}
TOKEN_TOPK_SOURCES = [
    "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl",
    "output/r0-oracle-topk-smoke-20260626T084130Z/topk-smoke.jsonl",
    "output/r0-oracle-topk-smoke-20260626T084753Z/topk-smoke.jsonl",
    "output/r0-oracle-topk-smoke-20260626T085856Z/topk-smoke.jsonl",
    "output/r0-oracle-topk-smoke-20260626T092009Z/topk-smoke.jsonl",
    "output/r0-oracle-topk-smoke-20260626T100409Z/topk-smoke.jsonl",
    "output/r0-oracle-topk-smoke-20260626T121946Z/topk-smoke.jsonl",
]
DISTRIBUTION_SOURCES = [
    "output/r0-distribution-capture-short-router-20260626T080938Z/teacher-forced-distribution-short-router.jsonl",
    "output/r0-distribution-capture-materialized-20260626T213023Z/teacher-forced-distribution-materialized.jsonl",
    "output/r0-distribution-capture-materialized-20260626T215314Z/teacher-forced-distribution-materialized.jsonl",
    "output/r0-distribution-capture-materialized-20260626T221606Z/teacher-forced-distribution-materialized.jsonl",
    "output/r0-distribution-capture-materialized-20260626T234204Z/teacher-forced-distribution-materialized.jsonl",
    "output/r0-distribution-capture-materialized-20260627T013743Z/teacher-forced-distribution-materialized.jsonl",
]
TOKEN_ID_PATH = "output/r0-oracle-token-id-capture-20260626T083347Z/prompt-token-id-references.jsonl"
BOUNDARY_FRAGMENT_DIR = "output/r0-boundary-bundle-fragment-20260627T054948Z"
PROMPT_EDGE_POLICY_DIR = "output/r0-oracle-256k-prompt-edge-policy-20260626T145727Z"

PATH_FIELDS = (
    "reference_input_tensor_path",
    "input_tensor_path",
    "reference_output_tensor_path",
    "output_tensor_path",
    "tensor_path",
)
PATH_MAP_FIELDS = (
    "reference_input_tensor_paths",
    "reference_output_tensor_paths",
    "tensor_paths",
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to oracle/r0-oracle-bundle-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path, base: Path = ROOT) -> str:
  return os.path.relpath(path.resolve(), base.resolve())


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
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


def latest(pattern: str, filename: str) -> Path:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  if not paths:
    raise SystemExit(f"missing latest artifact for {pattern}/{filename}")
  return paths[-1]


def expected_prompt_rows(capture_spec: dict[str, Any]) -> list[dict[str, Any]]:
  rows = capture_spec.get("capture_ladder", {}).get("prompt_rows", [])
  if not isinstance(rows, list):
    raise SystemExit("capture spec missing prompt rows")
  result = [row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]
  if len(result) != 26:
    raise SystemExit(f"expected 26 prompt rows, found {len(result)}")
  return result


def row_id(row: dict[str, Any]) -> str:
  value = row.get("case_id") or row.get("prompt_id") or row.get("id")
  if not isinstance(value, str) or not value:
    raise SystemExit("row missing case_id/prompt_id/id")
  return value


def indexed_rows(paths: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
  rows: dict[str, dict[str, Any]] = {}
  sources: dict[str, str] = {}
  for path_value in paths:
    path = ROOT / path_value
    for row in load_jsonl(path):
      case_id = row_id(row)
      if case_id in rows:
        continue
      rows[case_id] = row
      sources[case_id] = path_value
  return rows, sources


def topk_available(row: dict[str, Any]) -> bool:
  top_logprobs = row.get("top_logprobs")
  if isinstance(top_logprobs, list) and top_logprobs:
    return True
  for target in row.get("generation_targets", []):
    if isinstance(target, dict):
      top_logprobs = target.get("top_logprobs")
      if isinstance(top_logprobs, list) and top_logprobs:
        return True
  first_token = row.get("first_token")
  if isinstance(first_token, dict):
    top_logprobs = first_token.get("top_logprobs")
    if isinstance(top_logprobs, list) and top_logprobs:
      return True
  return False


def clean_limitations(row: dict[str, Any]) -> None:
  limitations = row.pop("limitations", None)
  if isinstance(limitations, dict):
    row["source_limitations"] = limitations


def merge_prompt_token_ids(
    row: dict[str, Any],
    token_row: dict[str, Any],
) -> None:
  token_ids = token_row.get("prompt_token_ids")
  if not isinstance(token_ids, list) or not token_ids:
    raise SystemExit(f"{row_id(token_row)}: token-id reference row missing prompt_token_ids")
  row["prompt_token_ids"] = token_ids
  for field in (
      "prompt_token_count",
      "observed_prompt_tokens",
      "prompt_token_ids_sha256",
      "prompt_utf8_sha256",
      "prompt_file_sha256",
      "materialized_prompt_path",
      "remote_prompt_path",
      "target_prompt_tokens",
      "tokenizer_evidence",
  ):
    if field in token_row and field not in row:
      row[field] = token_row[field]


def assemble_token_topk(
    prompt_rows: list[dict[str, Any]],
    token_id_rows: dict[str, dict[str, Any]],
    edge_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  topk_rows: dict[str, dict[str, Any]] = {}
  topk_sources: dict[str, str] = {}
  for path_value in TOKEN_TOPK_SOURCES:
    path = ROOT / path_value
    for row in load_jsonl(path):
      case_id = row_id(row)
      if case_id in EDGE_CASE_IDS or case_id in topk_rows:
        continue
      if not topk_available(row):
        continue
      topk_rows[case_id] = row
      topk_sources[case_id] = path_value

  policy_id = edge_policy.get("policy_id")
  rows = []
  missing_topk = []
  for prompt_row in prompt_rows:
    case_id = prompt_row["id"]
    token_row = token_id_rows.get(case_id)
    if token_row is None:
      raise SystemExit(f"missing token-id row for {case_id}")
    if case_id in EDGE_CASE_IDS:
      row = {
          "bundle_row_status": "prompt_edge_policy_row",
          "capture_status": "policy_resolved_prompt_edge",
          "case_id": case_id,
          "kind": prompt_row.get("kind"),
          "policy_id": policy_id,
          "prompt_edge_policy_path": PROMPT_EDGE_POLICY_DIR,
          "prompt_set": prompt_row.get("prompt_set"),
          "suite": prompt_row.get("suite"),
          "target_prompt_tokens": prompt_row.get("target_prompt_tokens"),
          "topk_logprobs_available": False,
          "workstream": WORKSTREAM,
      }
      merge_prompt_token_ids(row, token_row)
      rows.append(row)
      continue
    source = topk_rows.get(case_id)
    if source is None:
      missing_topk.append(case_id)
      continue
    row = dict(source)
    clean_limitations(row)
    merge_prompt_token_ids(row, token_row)
    row["bundle_row_status"] = "accepted_token_topk_reference"
    row["source_artifact"] = topk_sources[case_id]
    row["workstream"] = WORKSTREAM
    rows.append(row)
  if missing_topk:
    raise SystemExit(f"missing non-edge top-k rows: {', '.join(missing_topk)}")
  return rows, {
      "prompt_edge_rows": len([row for row in rows if row.get("case_id") in EDGE_CASE_IDS]),
      "source_artifacts": sorted(set(topk_sources.values())),
      "token_topk_rows": len(rows),
  }


def assemble_distribution(
    prompt_rows: list[dict[str, Any]],
    token_id_rows: dict[str, dict[str, Any]],
    edge_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  distribution_rows, distribution_sources = indexed_rows(DISTRIBUTION_SOURCES)
  policy_id = edge_policy.get("policy_id")
  rows = []
  missing = []
  position_count = 0
  for prompt_row in prompt_rows:
    case_id = prompt_row["id"]
    token_row = token_id_rows.get(case_id)
    if token_row is None:
      raise SystemExit(f"missing token-id row for {case_id}")
    if case_id in EDGE_CASE_IDS:
      rows.append({
          "bundle_row_status": "prompt_edge_policy_row",
          "capture_status": "policy_resolved_prompt_edge",
          "case_id": case_id,
          "distribution_available": False,
          "distribution_positions": [],
          "kind": prompt_row.get("kind"),
          "policy_id": policy_id,
          "prompt_edge_policy_path": PROMPT_EDGE_POLICY_DIR,
          "prompt_set": prompt_row.get("prompt_set"),
          "prompt_token_count": token_row.get("prompt_token_count"),
          "prompt_token_ids_sha256": token_row.get("prompt_token_ids_sha256"),
          "prompt_utf8_sha256": token_row.get("prompt_utf8_sha256"),
          "suite": prompt_row.get("suite"),
          "target_prompt_tokens": prompt_row.get("target_prompt_tokens"),
          "workstream": WORKSTREAM,
      })
      continue
    source = distribution_rows.get(case_id)
    if source is None:
      missing.append(case_id)
      continue
    row = dict(source)
    clean_limitations(row)
    positions = row.get("distribution_positions")
    if not isinstance(positions, list) or not positions:
      raise SystemExit(f"{case_id}: distribution row missing positions")
    for position in positions:
      if not isinstance(position, dict) or not position.get("top_logprobs"):
        raise SystemExit(f"{case_id}: distribution position missing top_logprobs")
    position_count += len(positions)
    row["bundle_row_status"] = "accepted_teacher_forced_distribution_reference"
    row["source_artifact"] = distribution_sources[case_id]
    row["workstream"] = WORKSTREAM
    rows.append(row)
  if missing:
    raise SystemExit(f"missing non-edge distribution rows: {', '.join(missing)}")
  return rows, {
      "distribution_position_count": position_count,
      "prompt_edge_rows": len([row for row in rows if row.get("case_id") in EDGE_CASE_IDS]),
      "source_artifacts": sorted(set(distribution_sources.values())),
      "teacher_forced_distribution_rows": len(rows),
  }


def resolve_fragment_path(value: str, source_dir: Path) -> Path:
  path = Path(value)
  if not path.is_absolute():
    path = source_dir / path
  path = path.resolve()
  if not path.is_file():
    raise SystemExit(f"missing boundary payload: {path}")
  return path


def rewrite_boundary_paths(
    row: dict[str, Any],
    source_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
  row = dict(row)
  for field in PATH_FIELDS:
    value = row.get(field)
    if isinstance(value, str) and value:
      row[field] = rel(resolve_fragment_path(value, source_dir), out_dir)
  for field in PATH_MAP_FIELDS:
    values = row.get(field)
    if not isinstance(values, dict):
      continue
    rewritten = {}
    for key, value in values.items():
      if isinstance(value, str) and value:
        rewritten[key] = rel(resolve_fragment_path(value, source_dir), out_dir)
    row[field] = rewritten
  row["source_artifact"] = rel(source_dir)
  return row


def assemble_boundary_rows(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  fragment_dir = ROOT / BOUNDARY_FRAGMENT_DIR
  input_path = fragment_dir / "boundary-references" / "inputs.jsonl"
  output_path = fragment_dir / "boundary-references" / "outputs.jsonl"
  inputs = [
      rewrite_boundary_paths(row, fragment_dir, out_dir)
      for row in load_jsonl(input_path)
  ]
  outputs = [
      rewrite_boundary_paths(row, fragment_dir, out_dir)
      for row in load_jsonl(output_path)
  ]
  return inputs, outputs


def assert_unique_case_ids(rows: list[dict[str, Any]], label: str) -> None:
  seen: set[str] = set()
  duplicates = []
  for row in rows:
    case_id = row_id(row)
    if case_id in seen:
      duplicates.append(case_id)
    seen.add(case_id)
  if duplicates:
    raise SystemExit(f"{label}: duplicate case ids: {', '.join(sorted(set(duplicates)))}")


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      (ROOT / f"oracle/r0-oracle-bundle-{stamp}").resolve()
      if args.out_dir is None
      else (ROOT / args.out_dir).resolve()
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "boundary-references").mkdir(parents=True, exist_ok=True)

  oracle_contract = load_json(ROOT / "oracle/oracle-bundle-contract.json")
  capture_spec_path = latest("r0-oracle-capture-spec-*", "capture-spec.json")
  capture_spec = load_json(capture_spec_path)
  prompt_rows = expected_prompt_rows(capture_spec)
  expected_ids = {row["id"] for row in prompt_rows}
  token_id_rows, _ = indexed_rows([TOKEN_ID_PATH])
  if set(token_id_rows) != expected_ids:
    raise SystemExit("token-id rows do not match capture spec prompt ids")

  policy_path = ROOT / PROMPT_EDGE_POLICY_DIR / "policy.json"
  edge_policy = load_json(policy_path)
  token_topk_rows, token_topk_summary = assemble_token_topk(
      prompt_rows,
      token_id_rows,
      edge_policy,
  )
  distribution_rows, distribution_summary = assemble_distribution(
      prompt_rows,
      token_id_rows,
      edge_policy,
  )
  input_rows, output_rows = assemble_boundary_rows(out_dir)

  assert_unique_case_ids(token_topk_rows, "token_topk")
  assert_unique_case_ids(distribution_rows, "teacher_forced_distribution")
  checks = [
      {
          "name": "prompt_rows_match_capture_spec",
          "pass": len(token_topk_rows) == len(prompt_rows) == len(distribution_rows) == 26,
      },
      {
          "name": "boundary_rows_match_capture_spec",
          "pass": len(input_rows) == 524 and len(output_rows) == 524,
          "input_rows": len(input_rows),
          "output_rows": len(output_rows),
      },
      {
          "name": "prompt_edge_rows_present",
          "pass": token_topk_summary["prompt_edge_rows"] == 2
          and distribution_summary["prompt_edge_rows"] == 2,
      },
      {
          "name": "teacher_forced_distribution_positions_present",
          "pass": distribution_summary["distribution_position_count"] > 91,
          "distribution_position_count": distribution_summary["distribution_position_count"],
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  if not required_checks_passed:
    raise SystemExit("assembled bundle failed internal checks")

  write_jsonl(out_dir / "token-topk-references.jsonl", token_topk_rows)
  write_jsonl(
      out_dir / "teacher-forced-distribution-references.jsonl",
      distribution_rows,
  )
  write_jsonl(out_dir / "boundary-references" / "inputs.jsonl", input_rows)
  write_jsonl(out_dir / "boundary-references" / "outputs.jsonl", output_rows)

  model = oracle_contract["model"]
  manifest = {
      "assembled_at": created_at,
      "evidence": {
          "boundary_fragment": BOUNDARY_FRAGMENT_DIR,
          "capture_spec": rel(capture_spec_path),
          "distribution_sources": distribution_summary["source_artifacts"],
          "prompt_edge_policy": PROMPT_EDGE_POLICY_DIR,
          "token_id_source": TOKEN_ID_PATH,
          "token_topk_sources": token_topk_summary["source_artifacts"],
      },
      "model": model,
      "row_counts": {
          "boundary_input_rows": len(input_rows),
          "boundary_output_rows": len(output_rows),
          "teacher_forced_distribution_rows": len(distribution_rows),
          "token_topk_rows": len(token_topk_rows),
      },
      "schema_version": SCHEMA_VERSION,
      "status": {
          "full_acceptance_bundle": True,
          "r0_oracle_gate_closed": True,
      },
      "tool": "tools/intel-qwen36-r0-oracle-bundle-assemble.py",
      "workstream": WORKSTREAM,
  }
  correctness = {
      "checks": checks,
      "full_acceptance_bundle": True,
      "gate": "r0_oracle_full_bundle",
      "required_checks_passed": True,
      "r0_oracle_gate_closed": True,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", manifest)
  write_json(out_dir / "correctness.json", correctness)
  write_json(out_dir / "assembly-summary.json", {
      "created_at": created_at,
      "output_dir": rel(out_dir),
      "schema_version": SCHEMA_VERSION,
      "teacher_forced_distribution": distribution_summary,
      "token_topk": token_topk_summary,
      "workstream": WORKSTREAM,
  })
  print(rel(out_dir))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
