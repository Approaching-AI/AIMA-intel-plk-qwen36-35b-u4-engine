#!/usr/bin/env python3
"""Build and run the engine-side embedding boundary compare on the PTL target."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-embedding-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-embedding-compare-v0"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/tests/compare_harness.hpp", "tests/compare_harness.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/embedding_compare.cpp", "tests/embedding_compare.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=120)
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


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_metric(path: Path, rows: list[tuple[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_embedding_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: row must be an object")
      rows.append(row)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_embedding_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = next(
      (
          row
          for row in inputs
          if row.get("boundary_type") == "embedding" and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row
          for row in outputs
          if row.get("boundary_type") == "embedding" and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing embedding boundary input/output rows")
  token_id = input_row.get("reference_input_tensor", {}).get("token_id")
  if not isinstance(token_id, int):
    raise SystemExit("oracle embedding input row missing integer token id")
  relative_payload = output_row.get("reference_output_tensor_path")
  if not isinstance(relative_payload, str) or not relative_payload:
    raise SystemExit("oracle embedding output row missing payload path")
  payload_path = (oracle_bundle / relative_payload).resolve()
  if not payload_path.exists():
    raise SystemExit(f"oracle embedding payload missing: {payload_path}")
  if payload_path.stat().st_size != 8192:
    raise SystemExit("oracle embedding payload must be 8192 bytes")
  shape = output_row.get("shape_metadata", {})
  if shape.get("nbytes") != 8192 or shape.get("ne") != [2048, 1, 1, 1]:
    raise SystemExit("oracle embedding payload shape mismatch")
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


def compare_passed(
    parsed: dict[str, Any],
    build: dict[str, Any],
    compare: dict[str, Any],
    model_path: str,
    token_id: int,
) -> bool:
  state = parsed.get("comparison", {})
  tensor = parsed.get("tensor", {})
  native = parsed.get("native_vector", {})
  oracle = parsed.get("oracle_vector", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("token_id") == token_id
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and tensor.get("name") == "token_embd.weight"
      and tensor.get("type_name") == "Q4_K"
      and tensor.get("dims") == [2048, 248320]
      and tensor.get("shape_ok") is True
      and native.get("count") == 2048
      and native.get("finite") is True
      and native.get("nonzero") is True
      and oracle.get("count") == 2048
      and oracle.get("finite") is True
      and oracle.get("nonzero") is True
      and state.get("same_size") is True
      and state.get("finite") is True
      and state.get("mismatch_count") == 0
      and state.get("max_abs_diff") <= 1e-6
      and state.get("rmse") <= 1e-7
      and state.get("cosine") >= 0.999999
  )


def build_summary(payload: dict[str, Any]) -> str:
  compare = payload["engine_embedding_compare"]
  lines = [
      "# R1 Engine Embedding Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{compare['source_prompt_case_id']}` token position {compare['source_token_position']}",
      f"- token id: {compare['token_id']}",
      f"- target build returncode: {payload['target_build']['returncode']}",
      f"- target compare returncode: {payload['target_compare']['returncode']}",
      f"- max abs diff: {compare.get('max_abs_diff')}",
      f"- RMSE: {compare.get('rmse')}",
      f"- cosine: {compare.get('cosine')}",
      f"- embedding compare passed: `{str(payload['engine_embedding_compare_passed']).lower()}`",
      "",
      "This artifact proves the engine-side C++ GGUF row decoder reproduces",
      "the oracle embedding boundary for the locked prompt token. It does not",
      "run the full model loop, emit native candidate token rows, or allow",
      "speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-embedding-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-embedding-compare-{stamp}"
  embedding = resolve_embedding_reference(args.oracle_bundle)
  remote_payload = f"{remote_dir}/oracle/{embedding['payload_path'].name}"

  mkdir = run_target(
      args.host,
      "mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfer_results = []
  payload_transfer = {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      transfer_results.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    payload_transfer = copy_to(
        args.host,
        embedding["payload_path"],
        remote_payload,
        args.timeout_s,
    )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/embedding_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-embedding-compare')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in transfer_results)
      and payload_transfer["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = (
      f"{shlex.quote(remote_dir + '/build/iq36-embedding-compare')} "
      f"{shlex.quote(args.model)} "
      f"{embedding['token_id']} "
      f"{shlex.quote(remote_payload)}"
  )
  compare = (
      run_target(args.host, compare_command, args.timeout_s)
      if build["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  parsed: dict[str, Any] = {}
  parse_error = None
  if compare.get("stdout"):
    try:
      parsed = json.loads(compare["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  passed = bool(parsed) and compare_passed(
      parsed,
      build,
      compare,
      args.model,
      embedding["token_id"],
  )
  comparison = parsed.get("comparison", {}) if parsed else {}
  tensor = parsed.get("tensor", {}) if parsed else {}
  payload = {
      "created_at": created_at,
      "engine_embedding_compare": {
          "cosine": comparison.get("cosine"),
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "max_abs_diff": comparison.get("max_abs_diff"),
          "mean_abs_diff": comparison.get("mean_abs_diff"),
          "mismatch_count": comparison.get("mismatch_count"),
          "native_value_count": parsed.get("native_vector", {}).get("count") if parsed else None,
          "oracle_payload_path": embedding["oracle_payload_path"],
          "oracle_payload_sha256": embedding["oracle_payload_sha256"],
          "oracle_payload_size_bytes": embedding["oracle_payload_size_bytes"],
          "oracle_value_count": parsed.get("oracle_vector", {}).get("count") if parsed else None,
          "rmse": comparison.get("rmse"),
          "source_prompt_case_id": embedding["source_prompt_case_id"],
          "source_token_position": embedding["source_token_position"],
          "tensor_dims": tensor.get("dims"),
          "tensor_name": tensor.get("name"),
          "tensor_type": tensor.get("type_name"),
          "token_id": embedding["token_id"],
      },
      "engine_embedding_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": embedding["oracle_bundle"],
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": {
          "returncode": build.get("returncode"),
          "stderr": build.get("stderr"),
          "stdout": build.get("stdout"),
      },
      "target_compare": {
          "returncode": compare.get("returncode"),
          "stderr": compare.get("stderr"),
          "stdout": compare.get("stdout"),
      },
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": embedding["oracle_bundle"],
      "oracle_payload_path": embedding["oracle_payload_path"],
      "oracle_payload_sha256": embedding["oracle_payload_sha256"],
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-embedding-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "oracle_payload_transfer": payload_transfer,
      "remote_dir": remote_dir,
      "remote_oracle_payload": remote_payload,
      "source_files": SOURCE_FILES,
      "transfer_results": transfer_results,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "embedding-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfer_results)},
          {"name": "oracle_embedding_payload_transferred", "pass": payload_transfer.get("returncode") == 0},
          {"name": "target_engine_embedding_compare_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_embedding_compare_ran", "pass": compare.get("returncode") == 0},
          {"name": "target_engine_embedding_compare_output_parsed", "pass": bool(parsed)},
          {"name": "embedding_boundary_matches_oracle_payload", "pass": passed},
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_embedding_compare_passed": passed,
      "gate": "r1_engine_embedding_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metric(out_dir / "metrics.jsonl", [
      ("engine_embedding_compare_passed", passed),
      ("token_id", embedding["token_id"]),
      ("native_value_count", payload["engine_embedding_compare"]["native_value_count"]),
      ("oracle_value_count", payload["engine_embedding_compare"]["oracle_value_count"]),
      ("max_abs_diff", comparison.get("max_abs_diff")),
      ("rmse", comparison.get("rmse")),
      ("cosine", comparison.get("cosine")),
      ("mismatch_count", comparison.get("mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ])
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine embedding compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
