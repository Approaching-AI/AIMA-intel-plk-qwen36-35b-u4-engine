#!/usr/bin/env python3
"""Audit the R1 native token correctness gate.

This gate intentionally rejects oracle fixtures and reference-runtime rows as
native evidence. A future native token loop must provide a candidate JSONL with
the six short/router rows before the gate can close.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-native-correctness-gate-v0"
ORACLE_SEED = ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
NATIVE_SOURCE = "intel_qwen36_native"
FORBIDDEN_SOURCES = {
    "llama.cpp",
    "llama.cpp CPU",
    "llama.cpp CPU server",
    "openvino",
    "OpenVINO",
    "oracle-fixture",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--candidate-jsonl",
      type=Path,
      default=None,
      help="Native candidate JSONL to validate against the short/router seed.",
  )
  parser.add_argument(
      "--oracle-jsonl",
      type=Path,
      default=ORACLE_SEED,
      help="Short/router token oracle seed JSONL.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r1-native-correctness-gate-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


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
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: row must be object")
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


def is_int_list(value: Any) -> bool:
  return isinstance(value, list) and all(isinstance(item, int) for item in value)


def by_case_id(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
  out: dict[str, dict[str, Any]] = {}
  duplicates: list[str] = []
  for row in rows:
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
      duplicates.append("<missing-case-id>")
      continue
    if case_id in out:
      duplicates.append(case_id)
    out[case_id] = row
  return out, duplicates


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


def top_signature(target: dict[str, Any] | None) -> list[int] | None:
  if not isinstance(target, dict):
    return None
  signature = target.get("top_logprob_id_signature")
  if is_int_list(signature) and signature:
    return signature
  top_logprobs = target.get("top_logprobs")
  if not isinstance(top_logprobs, list):
    return None
  ids = [
      item["id"] for item in top_logprobs
      if isinstance(item, dict) and isinstance(item.get("id"), int)
  ]
  return ids or None


def first_id(signature: list[int] | None) -> int | None:
  return signature[0] if isinstance(signature, list) and signature else None


def native_source_ok(row: dict[str, Any]) -> bool:
  source = row.get("native_output_source")
  if source != NATIVE_SOURCE:
    return False
  for field in ("source_reference_runtime", "reference_runtime", "source_runtime_dependency"):
    value = row.get(field)
    if isinstance(value, str) and value in FORBIDDEN_SOURCES:
      return False
  return True


def compare_case(
    case_id: str,
    oracle: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
  oracle_targets = targets_by_name(oracle)
  candidate_targets = targets_by_name(candidate or {})
  target_results = []
  for name, oracle_target in sorted(oracle_targets.items()):
    candidate_target = candidate_targets.get(name)
    expected_ids = oracle_target.get("generated_token_ids")
    actual_ids = (
        candidate_target.get("generated_token_ids")
        if isinstance(candidate_target, dict)
        else None
    )
    expected_sig = top_signature(oracle_target)
    actual_sig = top_signature(candidate_target)
    target_results.append({
        "actual_generated_token_ids": actual_ids,
        "expected_generated_token_ids": expected_ids,
        "generated_token_ids_match": is_int_list(actual_ids) and actual_ids == expected_ids,
        "generated_token_ids_present": is_int_list(actual_ids) and bool(actual_ids),
        "target": name,
        "target_present": isinstance(candidate_target, dict),
        "top1_id_match": first_id(actual_sig) == first_id(expected_sig),
        "top_logprob_id_signature_match": actual_sig == expected_sig,
        "top_logprob_id_signature_present": is_int_list(actual_sig) and bool(actual_sig),
    })
  prompt_hash = candidate.get("prompt_utf8_sha256") if isinstance(candidate, dict) else None
  prompt_token_ids = candidate.get("prompt_token_ids") if isinstance(candidate, dict) else None
  return {
      "candidate_present": isinstance(candidate, dict),
      "case_id": case_id,
      "native_output_source_ok": isinstance(candidate, dict) and native_source_ok(candidate),
      "prompt_token_ids_match": is_int_list(prompt_token_ids)
      and prompt_token_ids == oracle.get("prompt_token_ids"),
      "prompt_utf8_sha256_match": isinstance(prompt_hash, str)
      and prompt_hash == oracle.get("prompt_utf8_sha256"),
      "target_results": target_results,
      "workstream_match": isinstance(candidate, dict)
      and candidate.get("workstream") == WORKSTREAM,
  }


def all_cases(case_results: list[dict[str, Any]], field: str) -> bool:
  return bool(case_results) and all(row.get(field) is True for row in case_results)


def all_targets(case_results: list[dict[str, Any]], field: str) -> bool:
  targets = [
      target
      for row in case_results
      for target in row.get("target_results", [])
      if isinstance(target, dict)
  ]
  return bool(targets) and all(target.get(field) is True for target in targets)


def build_summary(payload: dict[str, Any]) -> str:
  gate = payload["r1_native_correctness_gate"]
  lines = [
      "# R1 Native Correctness Gate",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- native candidate present: `{str(gate['native_candidate_present']).lower()}`",
      f"- short/router oracle rows: {gate['oracle_seed_row_count']}",
      f"- R1 native correctness gate closed: `{str(gate['r1_native_correctness_gate_closed']).lower()}`",
      "",
  ]
  missing = gate.get("missing_for_gate", [])
  if missing:
    lines.append(f"- missing for gate: `{', '.join(missing)}`")
    lines.append("")
  lines.extend([
      "This gate does not accept oracle fixtures, llama.cpp rows, or OpenVINO",
      "rows as native evidence.",
      "",
  ])
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      (ROOT / f"output/r1-native-correctness-gate-{stamp}").resolve()
      if args.out_dir is None
      else args.out_dir.resolve()
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_rows = load_jsonl(args.oracle_jsonl)
  oracle_by_id, oracle_duplicates = by_case_id(oracle_rows)
  oracle_contract = load_json(ROOT / "oracle/oracle-bundle-contract.json")
  resident_load = oracle_contract.get("capture_plan", {}).get("latest_resident_harness_load", {})

  candidate_rows: list[dict[str, Any]] = []
  candidate_duplicates: list[str] = []
  candidate_by_id: dict[str, dict[str, Any]] = {}
  if args.candidate_jsonl is not None:
    candidate_rows = load_jsonl(args.candidate_jsonl)
    candidate_by_id, candidate_duplicates = by_case_id(candidate_rows)

  expected_ids = set(oracle_by_id)
  candidate_ids = set(candidate_by_id)
  missing_cases = sorted(expected_ids - candidate_ids)
  extra_cases = sorted(candidate_ids - expected_ids)
  case_results = [
      compare_case(case_id, oracle_by_id[case_id], candidate_by_id.get(case_id))
      for case_id in sorted(expected_ids)
  ]
  candidate_present = args.candidate_jsonl is not None
  candidate_shape_ok = (
      candidate_present
      and not candidate_duplicates
      and not missing_cases
      and not extra_cases
  )
  native_candidate_correct = (
      candidate_shape_ok
      and all_cases(case_results, "native_output_source_ok")
      and all_cases(case_results, "workstream_match")
      and all_cases(case_results, "prompt_utf8_sha256_match")
      and all_cases(case_results, "prompt_token_ids_match")
      and all_targets(case_results, "generated_token_ids_present")
      and all_targets(case_results, "generated_token_ids_match")
      and all_targets(case_results, "top1_id_match")
  )
  r0_ready = (
      oracle_contract.get("r0_oracle_gate_closed") is True
      and resident_load.get("r0_resident_harness_gate_closed") is True
  )
  gate_closed = r0_ready and native_candidate_correct
  missing_for_gate = []
  if not candidate_present:
    missing_for_gate.append("native_candidate_jsonl")
  if candidate_present and not native_candidate_correct:
    missing_for_gate.append("native_candidate_exact_replay_match")

  payload = {
      "case_results": case_results,
      "created_at": created_at,
      "evidence": {
          "candidate_jsonl": rel(args.candidate_jsonl),
          "oracle_jsonl": rel(args.oracle_jsonl),
          "oracle_contract": "oracle/oracle-bundle-contract.json",
          "resident_harness_load": resident_load.get("path"),
      },
      "r1_native_correctness_gate": {
          "candidate_duplicate_count": len(candidate_duplicates),
          "extra_cases": extra_cases,
          "missing_cases": missing_cases,
          "missing_for_gate": missing_for_gate,
          "native_candidate_present": candidate_present,
          "native_candidate_source_required": NATIVE_SOURCE,
          "oracle_seed_row_count": len(oracle_rows),
          "r0_ready": r0_ready,
          "r1_native_correctness_gate_closed": gate_closed,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "duplicates": oracle_duplicates,
          "name": "oracle_seed_loaded",
          "pass": len(oracle_rows) == 6 and not oracle_duplicates,
      },
      {
          "name": "r0_oracle_and_resident_harness_ready",
          "pass": r0_ready,
      },
      {
          "name": "native_candidate_required_for_gate",
          "pass": True,
          "candidate_present": candidate_present,
      },
      {
          "name": "oracle_fixture_and_reference_runtime_rejected",
          "pass": True,
          "required_native_output_source": NATIVE_SOURCE,
      },
      {
          "name": "gate_state_recorded",
          "pass": True,
          "r1_native_correctness_gate_closed": gate_closed,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-native-correctness-gate.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "gate.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r1_native_gguf_correctness_first_token_loop",
      "r1_native_correctness_gate_closed": gate_closed,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("oracle_seed_row_count", len(oracle_rows)),
        ("native_candidate_present", candidate_present),
        ("r1_native_correctness_gate_closed", gate_closed),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_native_correctness_gate",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 native correctness gate output: {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
