#!/usr/bin/env python3
"""Build and run the engine-side six-row seed prompt input check.

This artifact validates the native input path for the six R1 oracle seed
prompts. It converts oracle prompt token IDs into u32le files, checks exact
token-sequence hashes on the PTL target, decodes prompt token embedding rows
through the engine GGUF path, and compares the short_math_001 final prompt
embedding against the R0 oracle embedding payload. It does not run the
40-layer model loop or emit native candidate JSONL rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-engine-seed-prompt-input-check-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-seed-prompt-input-check-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_SEED = ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
EXPECTED_CASE_IDS = [
    "router_code_reason_002",
    "router_instruction_003",
    "router_math_reason_001",
    "short_factual_002",
    "short_math_001",
    "short_transform_003",
]
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/seed_prompt_input_check.cpp", "tests/seed_prompt_input_check.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-seed", type=Path, default=DEFAULT_ORACLE_SEED)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=300)
  return parser.parse_args()


def run(cmd: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    return {
        "cmd": cmd,
        "returncode": 124,
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + "\ntimeout",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
    }
  return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
  }


def run_target(host: str, remote_command: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_to(host: str, local_path: Path, remote_path: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_to(host, local_path, remote_path, timeout_s)


def copy_tree_to(host: str, local_dir: Path, remote_dir: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_tree_to(host, local_dir, remote_dir, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(row)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def fnv64(data: bytes) -> int:
  value = FNV_OFFSET
  for byte in data:
    value ^= byte
    value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
  return value


def is_int_list(value: Any) -> bool:
  return isinstance(value, list) and all(isinstance(item, int) for item in value)


def resolve_embedding_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = next(
      (
          row for row in inputs
          if row.get("boundary_type") == "embedding"
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "embedding"
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing embedding boundary rows")
  token_id = input_row.get("reference_input_tensor", {}).get("token_id")
  if token_id != 30:
    raise SystemExit("oracle embedding row must be short_math_001 final token id 30")
  relative_payload = output_row.get("reference_output_tensor_path")
  if not isinstance(relative_payload, str) or not relative_payload:
    raise SystemExit("oracle embedding row missing payload path")
  payload_path = (oracle_bundle / relative_payload).resolve()
  if not payload_path.exists():
    raise SystemExit(f"oracle embedding payload missing: {payload_path}")
  if payload_path.stat().st_size != 8192:
    raise SystemExit("oracle embedding payload size mismatch")
  return {
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "oracle_payload_path": str(payload_path.relative_to(ROOT)),
      "oracle_payload_sha256": sha256_file(payload_path),
      "oracle_payload_size_bytes": payload_path.stat().st_size,
      "payload_path": payload_path,
      "source_prompt_case_id": input_row.get("source_prompt_case_id"),
      "source_token_position": input_row.get("source_token_position"),
      "token_id": token_id,
  }


def load_seed_rows(oracle_seed: Path) -> list[dict[str, Any]]:
  rows = load_jsonl(oracle_seed.resolve())
  by_case: dict[str, dict[str, Any]] = {}
  for row in rows:
    case_id = row.get("case_id")
    if not isinstance(case_id, str):
      raise SystemExit("seed row missing case_id")
    if case_id in by_case:
      raise SystemExit(f"duplicate seed case_id: {case_id}")
    token_ids = row.get("prompt_token_ids")
    if not is_int_list(token_ids) or not token_ids:
      raise SystemExit(f"seed row {case_id} missing prompt_token_ids")
    if row.get("prompt_token_count") != len(token_ids):
      raise SystemExit(f"seed row {case_id} token count mismatch")
    by_case[case_id] = row
  if set(by_case) != set(EXPECTED_CASE_IDS):
    raise SystemExit("oracle seed case set mismatch")
  return [by_case[case_id] for case_id in EXPECTED_CASE_IDS]


def prepare_token_inputs(rows: list[dict[str, Any]], token_dir: Path) -> dict[str, Any]:
  token_dir.mkdir(parents=True, exist_ok=True)
  case_entries: dict[str, dict[str, Any]] = {}
  lines: list[str] = []
  total_prompt_tokens = 0
  unique_token_ids: set[int] = set()
  for row in rows:
    case_id = row["case_id"]
    token_ids = row["prompt_token_ids"]
    data = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    token_file = f"{case_id}.tokens.u32"
    token_path = token_dir / token_file
    token_path.write_bytes(data)
    token_fnv = f"{fnv64(data):016x}"
    token_sha256 = sha256_bytes(data)
    total_prompt_tokens += len(token_ids)
    unique_token_ids.update(token_ids)
    first_target = next(
        target for target in row["generation_targets"]
        if target.get("target") == "first_token"
    )
    case_entries[case_id] = {
        "first_token_id": token_ids[0],
        "first_token_top_logprob_id_signature": first_target.get(
            "top_logprob_id_signature"
        ),
        "last_token_id": token_ids[-1],
        "prompt_set": row.get("prompt_set"),
        "prompt_token_count": len(token_ids),
        "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
        "source_seed_schema_version": row.get("schema_version"),
        "token_file": token_file,
        "token_file_fnv64": token_fnv,
        "token_file_sha256": token_sha256,
        "token_file_size_bytes": len(data),
    }
    lines.append(
        "\t".join([
            case_id,
            str(len(token_ids)),
            token_fnv,
            str(token_ids[0]),
            str(token_ids[-1]),
            token_file,
        ])
    )
  (token_dir / "cases.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return {
      "case_count": len(rows),
      "cases": case_entries,
      "cases_tsv_sha256": sha256_file(token_dir / "cases.tsv"),
      "total_prompt_tokens": total_prompt_tokens,
      "unique_token_count": len(unique_token_ids),
  }


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 1e-6
      and comparison.get("rmse") <= 1e-7
      and comparison.get("cosine") >= 0.999999
  )


def check_passed(
    parsed: dict[str, Any],
    build: dict[str, Any],
    check: dict[str, Any],
    model_path: str,
    token_input: dict[str, Any],
) -> bool:
  tensor = parsed.get("tensor", {})
  cases = parsed.get("cases", {})
  return (
      build.get("returncode") == 0
      and check.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("passed") is True
      and parsed.get("load_map_ready") is True
      and parsed.get("case_count") == len(EXPECTED_CASE_IDS)
      and parsed.get("case_ids_ok") is True
      and parsed.get("cases_ok") is True
      and parsed.get("total_prompt_tokens") == token_input["total_prompt_tokens"]
      and parsed.get("unique_embedding_rows_decoded") == token_input["unique_token_count"]
      and parsed.get("unique_embeddings_ok") is True
      and tensor.get("name") == "token_embd.weight"
      and tensor.get("type_name") == "Q4_K"
      and tensor.get("dims") == [2048, 248320]
      and tensor.get("shape_ok") is True
      and set(cases) == set(EXPECTED_CASE_IDS)
      and all(case.get("passed") is True for case in cases.values())
      and comparison_passed(parsed.get("short_math_oracle_embedding_compare", {}))
  )


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_seed_prompt_input_check"]
  compare = state["short_math_oracle_embedding_compare"]
  lines = [
      "# R1 Engine Seed Prompt Input Check",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle seed: `{state['oracle_seed']}`",
      f"- case count: {state['case_count']}",
      f"- total prompt tokens: {state['total_prompt_tokens']}",
      f"- unique embedding rows decoded: {state['unique_embedding_rows_decoded']}",
      f"- short_math embedding max abs diff: {compare.get('max_abs_diff')}",
      f"- short_math embedding RMSE: {compare.get('rmse')}",
      f"- seed prompt input check passed: `{str(payload['engine_seed_prompt_input_check_passed']).lower()}`",
      "",
      "This artifact validates the native seed prompt token input path and",
      "embedding row decode for all six R1 oracle seed rows. It still does not",
      "run the 40-layer model loop, emit native candidate JSONL, or allow",
      "speedup claims.",
      "",
  ]
  return "\n".join(lines)


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_seed_prompt_input_check"]
  compare = state["short_math_oracle_embedding_compare"]
  rows = [
      ("engine_seed_prompt_input_check_passed", payload["engine_seed_prompt_input_check_passed"]),
      ("seed_prompt_case_count", state["case_count"]),
      ("seed_prompt_total_tokens", state["total_prompt_tokens"]),
      ("seed_prompt_unique_embedding_rows_decoded", state["unique_embedding_rows_decoded"]),
      ("short_math_embedding_max_abs_diff", compare.get("max_abs_diff")),
      ("short_math_embedding_rmse", compare.get("rmse")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_seed_prompt_input_check",
          "value": value,
      }, sort_keys=True) + "\n")


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-seed-prompt-input-check-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-seed-prompt-input-check-{stamp}"
  remote_token_dir = f"{remote_dir}/tokens"
  remote_oracle_embedding = f"{remote_dir}/oracle/short_math_embedding.bin"

  rows = load_seed_rows(args.oracle_seed)
  token_input_dir = out_dir / "token-input"
  token_input = prepare_token_inputs(rows, token_input_dir)
  embedding = resolve_embedding_reference(args.oracle_bundle)

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "tokens", "oracle")
      ),
      args.timeout_s,
  )
  source_transfers: list[dict[str, Any]] = []
  token_transfer = {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  oracle_transfer = {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    token_transfer = copy_tree_to(
        args.host,
        token_input_dir,
        remote_token_dir,
        args.timeout_s,
    )
    oracle_transfer = copy_to(
        args.host,
        embedding["payload_path"],
        remote_oracle_embedding,
        args.timeout_s,
    )

  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and token_transfer.get("returncode") == 0
      and oracle_transfer.get("returncode") == 0
  )
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/seed_prompt_input_check.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-seed-prompt-input-check')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  check_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-seed-prompt-input-check"),
      shlex.quote(args.model),
      shlex.quote(remote_token_dir),
      shlex.quote(remote_oracle_embedding),
  ])
  check = (
      run_target(args.host, check_command, args.timeout_s)
      if build["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  parsed: dict[str, Any] = {}
  parse_error = None
  if check.get("stdout"):
    try:
      parsed = json.loads(check["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  passed = bool(parsed) and check_passed(
      parsed,
      build,
      check,
      args.model,
      token_input,
  )
  state = {
      "boundary_type": "seed_prompt_input_path",
      "case_count": token_input["case_count"],
      "case_token_counts": {
          case_id: entry["prompt_token_count"]
          for case_id, entry in token_input["cases"].items()
      },
      "cases": token_input["cases"],
      "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
      "model_path": args.model,
      "oracle_bundle": embedding["oracle_bundle"],
      "oracle_seed": str(args.oracle_seed.resolve().relative_to(ROOT)),
      "prompt_token_ids_source": "oracle_seed_prompt_token_ids_u32le",
      "remote_token_dir": remote_token_dir,
      "seed_prompt_input_path_ready": passed,
      "short_math_oracle_embedding_compare": (
          parsed.get("short_math_oracle_embedding_compare", {}) if parsed else {}
      ),
      "source_seed_schema_version": "intel-qwen36-oracle-seed-stage-v0",
      "target_build_returncode": build.get("returncode"),
      "target_check_returncode": check.get("returncode"),
      "tensor": parsed.get("tensor", {}) if parsed else {},
      "total_prompt_tokens": token_input["total_prompt_tokens"],
      "unique_embedding_rows_decoded": (
          parsed.get("unique_embedding_rows_decoded") if parsed else None
      ),
      "unique_token_count": token_input["unique_token_count"],
  }
  payload = {
      "created_at": created_at,
      "engine_seed_prompt_input_check": state,
      "engine_seed_prompt_input_check_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": build,
      "target_check": check,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "token-input-manifest.json", token_input)
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "oracle_embedding_payload": embedding["oracle_payload_path"],
      "oracle_seed": state["oracle_seed"],
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-seed-prompt-input-check.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "oracle_embedding_transfer": oracle_transfer,
      "remote_dir": remote_dir,
      "remote_oracle_embedding": remote_oracle_embedding,
      "remote_token_dir": remote_token_dir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
      "token_transfer": token_transfer,
  })
  write_json(out_dir / "build.json", build)
  write_json(
      out_dir / "seed-prompt-input-stdout.json",
      parsed if parsed else {"parse_error": parse_error},
  )
  write_json(out_dir / "check.json", payload)
  checks = [
      {"name": "oracle_seed_rows_loaded", "pass": token_input["case_count"] == len(EXPECTED_CASE_IDS)},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {"name": "seed_prompt_token_inputs_transferred", "pass": token_transfer.get("returncode") == 0},
      {"name": "short_math_oracle_embedding_transferred", "pass": oracle_transfer.get("returncode") == 0},
      {"name": "target_engine_seed_prompt_input_check_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_seed_prompt_input_check_ran", "pass": check.get("returncode") == 0},
      {"name": "target_engine_seed_prompt_input_check_output_parsed", "pass": bool(parsed)},
      {"name": "seed_prompt_input_path_ready", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_seed_prompt_input_check_passed": passed,
      "gate": "r1_engine_seed_prompt_input_check",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check_item["pass"] for check_item in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine seed prompt input check output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
