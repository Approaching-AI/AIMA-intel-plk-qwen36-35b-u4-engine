#!/usr/bin/env python3
"""Capture full-ladder prompt token IDs from materialized prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-token-id-capture-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_TOKENIZE = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--timeout-s", type=int, default=420)
  parser.add_argument(
      "--materialization-dir",
      type=Path,
      default=None,
      help="Prompt materialization artifact directory. Defaults to latest contract entry.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-oracle-token-id-capture-<UTC>.",
  )
  return parser.parse_args()


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\ntimeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_command: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      try:
        value = json.loads(text)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
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
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def token_ids_sha256(token_ids: list[int]) -> str:
  return hashlib.sha256(
      json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
  ).hexdigest()


def parse_token_ids(stdout: str) -> list[int]:
  for line in reversed(stdout.splitlines()):
    text = line.strip()
    if text.startswith("[") and text.endswith("]"):
      value = json.loads(text)
      if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
  raise ValueError("could not parse token id list")


def materialization_dir(args: argparse.Namespace) -> Path:
  if args.materialization_dir is not None:
    return args.materialization_dir.resolve()
  contract = load_json(ROOT / "oracle/oracle-bundle-contract.json")
  path_value = (
      contract.get("capture_plan", {})
      .get("latest_oracle_prompt_materialization", {})
      .get("path")
  )
  if not isinstance(path_value, str) or not path_value:
    raise SystemExit("oracle contract missing latest prompt materialization path")
  return (ROOT / path_value).resolve()


def capture_token_ids(args: argparse.Namespace, row: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
  remote_path = row.get("remote_prompt_path")
  if not isinstance(remote_path, str) or not remote_path:
    raise SystemExit(f"{row.get('case_id')}: missing remote_prompt_path")
  command = (
      f"{shlex.quote(LLAMA_TOKENIZE)} "
      f"-m {shlex.quote(MODEL_PATH)} "
      f"-f {shlex.quote(remote_path)} "
      "--ids --log-disable"
  )
  result = run_target(args.host, command, timeout_s=args.timeout_s)
  try:
    token_ids = parse_token_ids(result["stdout"])
  except ValueError:
    token_ids = []
  evidence = {
      "returncode": result["returncode"],
      "stderr_tail": result["stderr"][-2000:],
      "stdout_sha256": hashlib.sha256(result["stdout"].encode("utf-8")).hexdigest(),
      "stdout_tail": result["stdout"][-2000:] if not token_ids else "",
      "timed_out": result["timed_out"],
  }
  return token_ids, evidence


def normalize_row(source: dict[str, Any], token_ids: list[int], evidence: dict[str, Any]) -> dict[str, Any]:
  return {
      "bundle_jsonl_path": "token-topk-references.jsonl",
      "capture_mode": "current_target_llama_tokenize_materialized_prompt_ids",
      "capture_status": "captured_token_ids_only",
      "case_id": source["case_id"],
      "kind": source.get("kind"),
      "limitations": {
          "not_a_full_r0_oracle_bundle": True,
          "prompt_token_ids_only": True,
          "top_k_logprobs_available": False,
          "teacher_forced_distribution_references": False,
          "per_boundary_reference_inputs": False,
          "per_boundary_reference_outputs": False,
      },
      "materialized_prompt_path": source.get("materialized_prompt_path"),
      "observed_prompt_tokens": source.get("observed_prompt_tokens"),
      "prompt_file_sha256": source.get("prompt_file_sha256"),
      "prompt_set": source.get("prompt_set"),
      "prompt_token_count": len(token_ids),
      "prompt_token_ids": token_ids,
      "prompt_token_ids_sha256": token_ids_sha256(token_ids),
      "prompt_utf8_sha256": source.get("prompt_utf8_sha256"),
      "remote_prompt_path": source.get("remote_prompt_path"),
      "schema_version": SCHEMA_VERSION,
      "source_materialization_schema_version": source.get("schema_version"),
      "source_reference_runtime": "llama.cpp llama-tokenize",
      "suite": source.get("suite"),
      "target_prompt_tokens": source.get("target_prompt_tokens"),
      "tokenizer_evidence": evidence,
      "workstream": WORKSTREAM,
  }


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# R0 Oracle Token ID Capture",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- captured rows: {payload['captured_row_count']}",
      f"- total prompt tokens: {payload['total_prompt_tokens']}",
      f"- max prompt tokens: {payload['max_prompt_tokens']}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This captures prompt token IDs for the full prompt ladder. It does not",
      "capture top-k logits, teacher-forced distributions, or per-boundary",
      "tensors, so it is not a full oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-oracle-token-id-capture-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  mat_dir = materialization_dir(args)
  mat_rows = load_jsonl(mat_dir / "materialized-prompts.jsonl")
  rows: list[dict[str, Any]] = []
  case_results: list[dict[str, Any]] = []
  for source in mat_rows:
    token_ids, evidence = capture_token_ids(args, source)
    row = normalize_row(source, token_ids, evidence)
    rows.append(row)
    case_results.append({
        "case_id": source.get("case_id"),
        "captured_token_count": len(token_ids),
        "expected_token_count": source.get("observed_prompt_tokens"),
        "returncode": evidence.get("returncode"),
        "token_count_matches_materialization": len(token_ids) == source.get("observed_prompt_tokens"),
    })

  total_prompt_tokens = sum(row["prompt_token_count"] for row in rows)
  checks = [
      {
          "name": "all_materialized_rows_captured",
          "pass": len(rows) == 26 and len(case_results) == 26,
          "count": len(rows),
      },
      {
          "name": "all_token_counts_match_materialization",
          "pass": all(result["token_count_matches_materialization"] for result in case_results),
      },
      {
          "name": "all_rows_current_workstream",
          "pass": all(row.get("workstream") == WORKSTREAM for row in rows),
      },
      {
          "name": "token_id_hashes_present",
          "pass": all(row.get("prompt_token_ids_sha256") for row in rows),
      },
      {
          "name": "artifact_does_not_claim_topk_or_full_bundle",
          "pass": all(
              row.get("limitations", {}).get("prompt_token_ids_only") is True
              and row.get("limitations", {}).get("top_k_logprobs_available") is False
              and row.get("limitations", {}).get("not_a_full_r0_oracle_bundle") is True
              for row in rows
          ),
      },
      {
          "name": "oracle_gate_remains_open",
          "pass": True,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "captured_row_count": len(rows),
      "case_results": case_results,
      "created_at": created_at,
      "host": args.host,
      "materialization_dir": str(mat_dir.relative_to(ROOT)),
      "max_prompt_tokens": max((row["prompt_token_count"] for row in rows), default=0),
      "model": {
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
          "batch_size": 1,
      },
      "r0_oracle_gate_closed": False,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "token_id_jsonl": str((out_dir / "prompt-token-id-references.jsonl").relative_to(ROOT)),
      "total_prompt_tokens": total_prompt_tokens,
      "workstream": WORKSTREAM,
  }
  write_jsonl(out_dir / "prompt-token-id-references.jsonl", rows)
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-token-id-capture.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_token_id_capture",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps({
          "case_id": row["case_id"],
          "metric": "prompt_token_count",
          "phase": "r0_oracle_token_id_capture",
          "prompt_set": row.get("prompt_set"),
          "target_prompt_tokens": row.get("target_prompt_tokens"),
          "value": row["prompt_token_count"],
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle token-id capture output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
