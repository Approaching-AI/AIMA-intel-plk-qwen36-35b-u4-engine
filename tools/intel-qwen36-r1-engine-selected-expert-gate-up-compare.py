#!/usr/bin/env python3
"""Build and run the engine-side L0 selected expert gate/up compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-selected-expert-gate-up-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-selected-expert-gate-up-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/selected_expert_gate_up_compare.cpp", "tests/selected_expert_gate_up_compare.cpp"),
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
    raise SystemExit(f"oracle selected expert gate/up payload missing: {payload_path}")
  if payload_path.stat().st_size != expected_size:
    raise SystemExit(f"oracle selected expert gate/up payload size mismatch: {payload_path}")
  return payload_path


def payload_record(path: Path, label: str) -> dict[str, Any]:
  return {
      f"{label}_payload_path": str(path.relative_to(ROOT)),
      f"{label}_payload_sha256": sha256_file(path),
      f"{label}_payload_size_bytes": path.stat().st_size,
      f"{label}_path": path,
  }


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = next(
      (
          row for row in inputs
          if row.get("boundary_type") == "selected_expert_gate_up"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "selected_expert_gate_up"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  router_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "router_topk"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing L0 selected_expert_gate_up rows")
  if not isinstance(router_row, dict):
    raise SystemExit("oracle bundle missing L0 router_topk output row")

  input_shape = input_row.get("shape_metadata", {})
  output_shape = output_row.get("shape_metadata", {})
  if input_shape.get("nbytes") != 8192 or input_shape.get("ne") != [2048, 1, 1, 1]:
    raise SystemExit("oracle L0 selected expert gate/up input shape mismatch")
  if input_shape.get("tensor_name") != "attn_post_norm-0":
    raise SystemExit("oracle L0 selected expert gate/up input tensor name mismatch")
  if output_shape.get("nbytes") != 32768 or output_shape.get("ne") != [1024, 8, 1, 1]:
    raise SystemExit("oracle L0 selected expert gate/up output shape mismatch")
  if output_shape.get("tensor_name") != "ffn_moe_gate_up-0":
    raise SystemExit("oracle L0 selected expert gate/up output tensor name mismatch")

  router_paths = router_row.get("reference_output_tensor_paths", {})
  if not isinstance(router_paths, dict):
    raise SystemExit("oracle L0 router_topk output payload map missing")

  paths = {
      "input": resolve_payload(oracle_bundle, input_row["reference_input_tensor_path"], 8192),
      "topk": resolve_payload(oracle_bundle, router_paths["ffn_moe_topk-0"], 32),
      "output": resolve_payload(oracle_bundle, output_row["reference_output_tensor_path"], 32768),
  }
  result: dict[str, Any] = {
      "boundary_type": "selected_expert_gate_up",
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "source_prompt_case_id": input_row.get("source_prompt_case_id"),
      "source_token_position": input_row.get("source_token_position"),
  }
  for label, path in paths.items():
    result.update(payload_record(path, label))
  return result


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  tensor = parsed.get("tensor", {})
  input_vector = parsed.get("input_vector", {})
  native_vector = parsed.get("native_vector", {})
  oracle_vector = parsed.get("oracle_vector", {})
  comparison = parsed.get("comparison", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and tensor.get("name") == "blk.0.ffn_gate_up_exps.weight"
      and tensor.get("type_name") == "Q4_K"
      and tensor.get("dims") == [2048, 1024, 256]
      and tensor.get("shape_ok") is True
      and input_vector.get("count") == 2048
      and native_vector.get("count") == 8192
      and oracle_vector.get("count") == 8192
      and len(parsed.get("expert_ids", [])) == 8
      and all(
          vector.get("finite") is True and vector.get("nonzero") is True
          for vector in (input_vector, native_vector, oracle_vector)
      )
      and comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-3
      and comparison.get("rmse") <= 5e-4
      and comparison.get("cosine") >= 0.999
  )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_selected_expert_gate_up_compare"]
  rows = [
      ("engine_selected_expert_gate_up_compare_passed", payload["engine_selected_expert_gate_up_compare_passed"]),
      ("input_value_count", state.get("input_value_count")),
      ("topk_value_count", state.get("topk_value_count")),
      ("native_value_count", state.get("native_value_count")),
      ("oracle_value_count", state.get("oracle_value_count")),
      ("max_abs_diff", state.get("max_abs_diff")),
      ("rmse", state.get("rmse")),
      ("cosine", state.get("cosine")),
      ("mismatch_count", state.get("mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_selected_expert_gate_up_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_selected_expert_gate_up_compare"]
  lines = [
      "# R1 Engine Selected Expert Gate/Up Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- weight tensor: `{state.get('tensor_name')}` {state.get('tensor_type')} {state.get('tensor_dims')}",
      f"- expert ids: `{state.get('expert_ids')}`",
      f"- max abs diff: {state.get('max_abs_diff')}",
      f"- RMSE: {state.get('rmse')}",
      f"- cosine: {state.get('cosine')}",
      f"- selected expert gate/up compare passed: `{str(payload['engine_selected_expert_gate_up_compare_passed']).lower()}`",
      "",
      "This artifact records whether the engine-side C++ selected-expert",
      "gate/up path reproduces the L0 `ffn_moe_gate_up-0` tensor for the",
      "locked prompt token. It does not run the full model loop, emit native",
      "candidate token rows, or allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-selected-expert-gate-up-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-selected-expert-gate-up-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)

  remote_payloads = {
      label: f"{remote_dir}/oracle/{ref[f'{label}_path'].name}"
      for label in ("input", "topk", "output")
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
      f"{shlex.quote(remote_dir + '/tests/selected_expert_gate_up_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-selected-expert-gate-up-compare')}",
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
      f"{shlex.quote(remote_dir + '/build/iq36-selected-expert-gate-up-compare')} "
      f"{shlex.quote(args.model)} "
      f"{shlex.quote(remote_payloads['input'])} "
      f"{shlex.quote(remote_payloads['topk'])} "
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
  payload = {
      "created_at": created_at,
      "engine_selected_expert_gate_up_compare": {
          "boundary_type": ref["boundary_type"],
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "expert_ids": parsed.get("expert_ids") if parsed else None,
          "input_payload_path": ref["input_payload_path"],
          "input_payload_sha256": ref["input_payload_sha256"],
          "input_payload_size_bytes": ref["input_payload_size_bytes"],
          "input_value_count": parsed.get("input_vector", {}).get("count") if parsed else None,
          "max_abs_diff": comparison.get("max_abs_diff"),
          "mean_abs_diff": comparison.get("mean_abs_diff"),
          "mismatch_count": comparison.get("mismatch_count"),
          "native_value_count": parsed.get("native_vector", {}).get("count") if parsed else None,
          "oracle_value_count": parsed.get("oracle_vector", {}).get("count") if parsed else None,
          "output_payload_path": ref["output_payload_path"],
          "output_payload_sha256": ref["output_payload_sha256"],
          "output_payload_size_bytes": ref["output_payload_size_bytes"],
          "rmse": comparison.get("rmse"),
          "cosine": comparison.get("cosine"),
          "source_prompt_case_id": ref["source_prompt_case_id"],
          "source_token_position": ref["source_token_position"],
          "tensor_dims": parsed.get("tensor", {}).get("dims") if parsed else None,
          "tensor_name": parsed.get("tensor", {}).get("name") if parsed else None,
          "tensor_type": parsed.get("tensor", {}).get("type_name") if parsed else None,
          "topk_payload_path": ref["topk_payload_path"],
          "topk_payload_sha256": ref["topk_payload_sha256"],
          "topk_payload_size_bytes": ref["topk_payload_size_bytes"],
          "topk_value_count": len(parsed.get("expert_ids", [])) if parsed else None,
      },
      "engine_selected_expert_gate_up_compare_passed": passed,
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
      "tool": "tools/intel-qwen36-r1-engine-selected-expert-gate-up-compare.py",
      "topk_payload_path": ref["topk_payload_path"],
      "topk_payload_sha256": ref["topk_payload_sha256"],
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "input_payload_transfer": payload_transfers["input"],
      "mkdir": mkdir,
      "output_payload_transfer": payload_transfers["output"],
      "remote_dir": remote_dir,
      "remote_payloads": remote_payloads,
      "source_files": SOURCE_FILES,
      "topk_payload_transfer": payload_transfers["topk"],
      "transfer_results": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "selected-expert-gate-up-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfers)},
          {"name": "oracle_selected_expert_gate_up_input_payload_transferred", "pass": payload_transfers["input"].get("returncode") == 0},
          {"name": "oracle_selected_expert_gate_up_topk_payload_transferred", "pass": payload_transfers["topk"].get("returncode") == 0},
          {"name": "oracle_selected_expert_gate_up_output_payload_transferred", "pass": payload_transfers["output"].get("returncode") == 0},
          {"name": "target_engine_selected_expert_gate_up_compare_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_selected_expert_gate_up_compare_ran", "pass": compare.get("returncode") == 0},
          {"name": "target_engine_selected_expert_gate_up_compare_output_parsed", "pass": bool(parsed)},
          {"name": "selected_expert_gate_up_matches_oracle_payload", "pass": passed},
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_selected_expert_gate_up_compare_passed": passed,
      "gate": "r1_engine_selected_expert_gate_up_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine selected expert gate/up compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
