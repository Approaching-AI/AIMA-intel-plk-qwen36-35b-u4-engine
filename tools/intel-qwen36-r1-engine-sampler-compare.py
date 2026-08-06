#!/usr/bin/env python3
"""Build and run the engine-side global sampler top-k compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-sampler-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-sampler-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/sampler_compare.cpp", "tests/sampler_compare.cpp"),
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
  parser.add_argument("--timeout-s", type=int, default=240)
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
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


def resolve_payload(oracle_bundle: Path, relative_payload: str, expected_size: int) -> Path:
  payload_path = (oracle_bundle / relative_payload).resolve()
  if not payload_path.exists():
    raise SystemExit(f"oracle sampler payload missing: {payload_path}")
  if expected_size >= 0 and payload_path.stat().st_size != expected_size:
    raise SystemExit(f"oracle sampler payload size mismatch: {payload_path}")
  return payload_path


def _find_row(
    rows: list[dict[str, Any]],
    boundary_type: str,
    tensor_kind: str,
) -> dict[str, Any]:
  row = next(
      (
          item for item in rows
          if item.get("boundary_type") == boundary_type
          and item.get("layer") == "global"
          and item.get("tensor_kind") == tensor_kind
      ),
      None,
  )
  if not isinstance(row, dict):
    raise SystemExit(f"oracle bundle missing global {boundary_type} {tensor_kind} row")
  return row


def _require_logits_shape(row: dict[str, Any]) -> None:
  shape = row.get("shape_metadata", {})
  if shape.get("nbytes") != 993280:
    raise SystemExit("oracle result_output nbytes mismatch")
  if shape.get("ne") != [248320, 1, 1, 1]:
    raise SystemExit("oracle result_output shape mismatch")
  if shape.get("tensor_name") != "result_output":
    raise SystemExit("oracle result_output tensor name mismatch")
  if shape.get("tensor_op") != "MUL_MAT":
    raise SystemExit("oracle result_output op mismatch")


def _require_sampler_shape(row: dict[str, Any]) -> None:
  shape = row.get("shape_metadata", {})
  if shape.get("shape") != ["token_id", "top_k_logprobs"]:
    raise SystemExit("oracle sampler top-k shape mismatch")
  if row.get("capture_status") != "captured_sampler_topk":
    raise SystemExit("oracle sampler top-k capture status mismatch")
  dtype = row.get("dtype_metadata", {})
  if dtype.get("tensor_type") != "sampler_topk_json":
    raise SystemExit("oracle sampler top-k dtype mismatch")


def payload_record(path: Path, label: str) -> dict[str, Any]:
  return {
      f"{label}_payload_path": str(path.relative_to(ROOT)),
      f"{label}_payload_sha256": sha256_file(path),
      f"{label}_payload_size_bytes": path.stat().st_size,
      f"{label}_path": path,
  }


def load_sampler_topk(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    payload = json.load(fh)
  if not isinstance(payload, dict):
    raise SystemExit("sampler top-k JSON must be an object")
  top_k = payload.get("top_k")
  if not isinstance(top_k, list) or len(top_k) != 8:
    raise SystemExit("sampler top-k JSON must contain 8 rows")
  return payload


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = _find_row(inputs, "sampler", "input")
  output_row = _find_row(outputs, "sampler", "output")
  _require_logits_shape(input_row)
  _require_sampler_shape(output_row)

  source_case = output_row.get("source_prompt_case_id")
  source_token = output_row.get("source_token_position")
  if (
      input_row.get("source_prompt_case_id") != source_case
      or input_row.get("source_token_position") != source_token
  ):
    raise SystemExit("oracle global sampler source case/token mismatch")

  logits_path = resolve_payload(
      oracle_bundle,
      input_row["reference_input_tensor_path"],
      993280,
  )
  sampler_path = resolve_payload(
      oracle_bundle,
      output_row["reference_output_tensor_path"],
      -1,
  )
  sampler_json = load_sampler_topk(sampler_path)
  result: dict[str, Any] = {
      "boundary_type": "sampler",
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "source_prompt_case_id": source_case,
      "source_token_position": source_token,
      "sampler_case_id": sampler_json.get("case_id"),
      "sampler_prompt_token_count": sampler_json.get("prompt_token_count"),
      "sampler_logits_present": sampler_json.get("logits_present"),
      "sampler_top_token_id": sampler_json.get("top_k", [{}])[0].get("token_id"),
      "sampler_top_logit": sampler_json.get("top_k", [{}])[0].get("logit"),
      "sampler_top_k_count": len(sampler_json.get("top_k", [])),
  }
  result.update(payload_record(logits_path, "input"))
  result.update(payload_record(sampler_path, "output"))
  return result


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  state = parsed.get("comparison", {})
  sampler_json = parsed.get("sampler_json", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and parsed.get("logits_vector", {}).get("count") == 248320
      and parsed.get("logits_vector", {}).get("finite") is True
      and parsed.get("logits_vector", {}).get("nonzero") is True
      and sampler_json.get("case_id") == "short_math_001"
      and sampler_json.get("source_token_position") == 15
      and sampler_json.get("prompt_token_count") == 16
      and sampler_json.get("logits_present") is True
      and len(parsed.get("native_top_k", [])) == 8
      and len(parsed.get("expected_top_k", [])) == 8
      and state.get("same_size") is True
      and state.get("finite") is True
      and state.get("top1_matches") is True
      and state.get("token_id_mismatch_count") == 0
      and state.get("logits_within_threshold") is True
      and state.get("max_abs_logit_diff") <= 0.0005
  )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_sampler_compare"]
  rows = [
      ("engine_sampler_compare_passed", payload["engine_sampler_compare_passed"]),
      ("logits_value_count", state.get("logits_value_count")),
      ("top_k_count", state.get("top_k_count")),
      ("top_token_id", state.get("top_token_id")),
      ("max_abs_logit_diff", state.get("max_abs_logit_diff")),
      ("token_id_mismatch_count", state.get("token_id_mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_sampler_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_sampler_compare"]
  lines = [
      "# R1 Engine Sampler Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- top token id: `{state.get('top_token_id')}`",
      f"- top-k count: {state.get('top_k_count')}",
      f"- max abs logit diff: {state.get('max_abs_logit_diff')}",
      f"- token-id mismatch count: {state.get('token_id_mismatch_count')}",
      f"- sampler compare passed: `{str(payload['engine_sampler_compare_passed']).lower()}`",
      "",
      "This artifact records whether the engine-side C++ deterministic top-k",
      "sampler path reproduces the captured `sampler-topk.json` rows from the",
      "global `result_output` logits. It does not run the full model loop,",
      "emit native candidate token rows, or allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-sampler-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-sampler-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)

  remote_payloads = {
      "input": f"{remote_dir}/oracle/{ref['input_path'].name}",
      "output": f"{remote_dir}/oracle/{ref['output_path'].name}",
  }
  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      label: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for label in remote_payloads
  }
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    for label, remote_path in remote_payloads.items():
      payload_transfers[label] = copy_to(
          args.host,
          ref[f"{label}_path"],
          remote_path,
          args.timeout_s,
      )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/sampler_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-sampler-compare')}",
  ])
  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = (
      f"{shlex.quote(remote_dir + '/build/iq36-sampler-compare')} "
      f"{shlex.quote(args.model)} "
      f"{shlex.quote(remote_payloads['input'])} "
      f"{shlex.quote(remote_payloads['output'])}"
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

  passed = bool(parsed) and compare_passed(parsed, build, compare, args.model)
  comparison = parsed.get("comparison", {}) if parsed else {}
  native_top_k = parsed.get("native_top_k", []) if parsed else []
  payload = {
      "created_at": created_at,
      "engine_sampler_compare": {
          "boundary_type": ref["boundary_type"],
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "expected_top_k": parsed.get("expected_top_k", []) if parsed else [],
          "input_payload_path": ref["input_payload_path"],
          "input_payload_sha256": ref["input_payload_sha256"],
          "input_payload_size_bytes": ref["input_payload_size_bytes"],
          "logits_value_count": parsed.get("logits_vector", {}).get("count") if parsed else None,
          "max_abs_logit_diff": comparison.get("max_abs_logit_diff"),
          "mean_abs_logit_diff": comparison.get("mean_abs_logit_diff"),
          "native_top_k": native_top_k,
          "output_payload_path": ref["output_payload_path"],
          "output_payload_sha256": ref["output_payload_sha256"],
          "output_payload_size_bytes": ref["output_payload_size_bytes"],
          "sampler_case_id": ref["sampler_case_id"],
          "sampler_logits_present": ref["sampler_logits_present"],
          "sampler_prompt_token_count": ref["sampler_prompt_token_count"],
          "source_prompt_case_id": ref["source_prompt_case_id"],
          "source_token_position": ref["source_token_position"],
          "target_build_returncode": build.get("returncode"),
          "target_compare_returncode": compare.get("returncode"),
          "token_id_mismatch_count": comparison.get("token_id_mismatch_count"),
          "top_k_count": len(native_top_k),
          "top_token_id": native_top_k[0].get("token_id") if native_top_k else None,
      },
      "engine_sampler_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
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
      "input_payload_path": ref["input_payload_path"],
      "input_payload_sha256": ref["input_payload_sha256"],
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "output_payload_path": ref["output_payload_path"],
      "output_payload_sha256": ref["output_payload_sha256"],
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-sampler-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "input_payload_transfer": payload_transfers["input"],
      "mkdir": mkdir,
      "output_payload_transfer": payload_transfers["output"],
      "remote_dir": remote_dir,
      "remote_payloads": remote_payloads,
      "source_files": SOURCE_FILES,
      "transfer_results": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "sampler-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfers)},
          {"name": "oracle_sampler_input_payload_transferred", "pass": payload_transfers["input"].get("returncode") == 0},
          {"name": "oracle_sampler_output_payload_transferred", "pass": payload_transfers["output"].get("returncode") == 0},
          {"name": "target_engine_sampler_compare_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_sampler_compare_ran", "pass": compare.get("returncode") == 0},
          {"name": "target_engine_sampler_compare_output_parsed", "pass": bool(parsed)},
          {"name": "sampler_topk_matches_oracle_payload", "pass": passed},
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_sampler_compare_passed": passed,
      "gate": "r1_engine_sampler_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine sampler compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
