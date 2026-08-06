#!/usr/bin/env python3
"""Build and run the engine-side L0 router top-k boundary compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-router-topk-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-router-topk-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/router_topk_compare.cpp", "tests/router_topk_compare.cpp"),
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
    raise SystemExit(f"oracle router top-k payload missing: {payload_path}")
  if payload_path.stat().st_size != expected_size:
    raise SystemExit(f"oracle router top-k payload size mismatch: {payload_path}")
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
          if row.get("boundary_type") == "router_topk"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "router_topk"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing L0 router_topk rows")

  input_shape = input_row.get("shape_metadata", {})
  if input_shape.get("nbytes") != 8192 or input_shape.get("ne") != [2048, 1, 1, 1]:
    raise SystemExit("oracle L0 router top-k input shape mismatch")
  if input_shape.get("tensor_name") != "attn_post_norm-0":
    raise SystemExit("oracle L0 router top-k input tensor name mismatch")

  output_paths = output_row.get("reference_output_tensor_paths", {})
  if not isinstance(output_paths, dict):
    raise SystemExit("oracle L0 router top-k output payload map missing")

  paths = {
      "input": resolve_payload(oracle_bundle, input_row["reference_input_tensor_path"], 8192),
      "logits": resolve_payload(oracle_bundle, output_paths["ffn_moe_logits-0"], 1024),
      "topk": resolve_payload(oracle_bundle, output_paths["ffn_moe_topk-0"], 32),
      "weights": resolve_payload(oracle_bundle, output_paths["ffn_moe_weights-0"], 32),
      "weights_norm": resolve_payload(
          oracle_bundle,
          output_paths["ffn_moe_weights_norm-0"],
          32,
      ),
  }
  result: dict[str, Any] = {
      "boundary_type": "router_topk",
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
  native_logits = parsed.get("native_logits_vector", {})
  oracle_logits = parsed.get("oracle_logits_vector", {})
  native_weights = parsed.get("native_weights_vector", {})
  oracle_weights = parsed.get("oracle_weights_vector", {})
  native_weights_norm = parsed.get("native_weights_norm_vector", {})
  oracle_weights_norm = parsed.get("oracle_weights_norm_vector", {})
  logits_cmp = parsed.get("comparison_logits", {})
  topk_cmp = parsed.get("comparison_topk", {})
  weights_cmp = parsed.get("comparison_weights", {})
  weights_norm_cmp = parsed.get("comparison_weights_norm", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and tensor.get("name") == "blk.0.ffn_gate_inp.weight"
      and tensor.get("type_name") == "F32"
      and tensor.get("dims") == [2048, 256]
      and tensor.get("shape_ok") is True
      and input_vector.get("count") == 2048
      and native_logits.get("count") == 256
      and oracle_logits.get("count") == 256
      and native_weights.get("count") == 8
      and oracle_weights.get("count") == 8
      and native_weights_norm.get("count") == 8
      and oracle_weights_norm.get("count") == 8
      and all(
          vector.get("finite") is True and vector.get("nonzero") is True
          for vector in (
              input_vector,
              native_logits,
              oracle_logits,
              native_weights,
              oracle_weights,
              native_weights_norm,
              oracle_weights_norm,
          )
      )
      and parsed.get("native_topk") == parsed.get("oracle_topk")
      and topk_cmp.get("same_size") is True
      and topk_cmp.get("mismatch_count") == 0
      and logits_cmp.get("same_size") is True
      and logits_cmp.get("finite") is True
      and logits_cmp.get("mismatch_count") == 0
      and logits_cmp.get("max_abs_diff") <= 1e-4
      and logits_cmp.get("rmse") <= 1e-5
      and weights_cmp.get("same_size") is True
      and weights_cmp.get("finite") is True
      and weights_cmp.get("mismatch_count") == 0
      and weights_cmp.get("max_abs_diff") <= 2e-5
      and weights_cmp.get("rmse") <= 1e-6
      and weights_norm_cmp.get("same_size") is True
      and weights_norm_cmp.get("finite") is True
      and weights_norm_cmp.get("mismatch_count") == 0
      and weights_norm_cmp.get("max_abs_diff") <= 2e-5
      and weights_norm_cmp.get("rmse") <= 1e-6
  )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_router_topk_compare"]
  rows = [
      ("engine_router_topk_compare_passed", payload["engine_router_topk_compare_passed"]),
      ("input_value_count", state.get("input_value_count")),
      ("logits_value_count", state.get("logits_value_count")),
      ("topk_value_count", state.get("topk_value_count")),
      ("weights_value_count", state.get("weights_value_count")),
      ("weights_norm_value_count", state.get("weights_norm_value_count")),
      ("logits_max_abs_diff", state.get("logits_max_abs_diff")),
      ("weights_max_abs_diff", state.get("weights_max_abs_diff")),
      ("weights_norm_max_abs_diff", state.get("weights_norm_max_abs_diff")),
      ("topk_mismatch_count", state.get("topk_mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_router_topk_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_router_topk_compare"]
  lines = [
      "# R1 Engine Router Top-K Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- weight tensor: `{state.get('tensor_name')}` {state.get('tensor_type')} {state.get('tensor_dims')}",
      f"- native top-k: `{state.get('native_topk')}`",
      f"- oracle top-k: `{state.get('oracle_topk')}`",
      f"- logits max abs diff: {state.get('logits_max_abs_diff')}",
      f"- weights max abs diff: {state.get('weights_max_abs_diff')}",
      f"- weights norm max abs diff: {state.get('weights_norm_max_abs_diff')}",
      f"- router top-k compare passed: `{str(payload['engine_router_topk_compare_passed']).lower()}`",
      "",
      "This artifact records whether the engine-side C++ router path",
      "reproduces the L0 top-k expert ids and router weights for the locked",
      "prompt token. It does not run the full model loop, emit native",
      "candidate token rows, or allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-router-topk-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-router-topk-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)

  remote_payloads = {
      label: f"{remote_dir}/oracle/{ref[f'{label}_path'].name}"
      for label in ("input", "logits", "topk", "weights", "weights_norm")
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
      f"{shlex.quote(remote_dir + '/tests/router_topk_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-router-topk-compare')}",
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
      f"{shlex.quote(remote_dir + '/build/iq36-router-topk-compare')} "
      f"{shlex.quote(args.model)} "
      f"{shlex.quote(remote_payloads['input'])} "
      f"{shlex.quote(remote_payloads['logits'])} "
      f"{shlex.quote(remote_payloads['topk'])} "
      f"{shlex.quote(remote_payloads['weights'])} "
      f"{shlex.quote(remote_payloads['weights_norm'])}"
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
  logits_cmp = parsed.get("comparison_logits", {}) if parsed else {}
  topk_cmp = parsed.get("comparison_topk", {}) if parsed else {}
  weights_cmp = parsed.get("comparison_weights", {}) if parsed else {}
  weights_norm_cmp = parsed.get("comparison_weights_norm", {}) if parsed else {}
  payload = {
      "created_at": created_at,
      "engine_router_topk_compare": {
          "boundary_type": ref["boundary_type"],
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "input_payload_path": ref["input_payload_path"],
          "input_payload_sha256": ref["input_payload_sha256"],
          "input_payload_size_bytes": ref["input_payload_size_bytes"],
          "input_value_count": parsed.get("input_vector", {}).get("count") if parsed else None,
          "logits_max_abs_diff": logits_cmp.get("max_abs_diff"),
          "logits_mean_abs_diff": logits_cmp.get("mean_abs_diff"),
          "logits_payload_path": ref["logits_payload_path"],
          "logits_payload_sha256": ref["logits_payload_sha256"],
          "logits_payload_size_bytes": ref["logits_payload_size_bytes"],
          "logits_rmse": logits_cmp.get("rmse"),
          "logits_value_count": parsed.get("native_logits_vector", {}).get("count") if parsed else None,
          "native_topk": parsed.get("native_topk") if parsed else None,
          "oracle_topk": parsed.get("oracle_topk") if parsed else None,
          "source_prompt_case_id": ref["source_prompt_case_id"],
          "source_token_position": ref["source_token_position"],
          "tensor_dims": parsed.get("tensor", {}).get("dims") if parsed else None,
          "tensor_name": parsed.get("tensor", {}).get("name") if parsed else None,
          "tensor_type": parsed.get("tensor", {}).get("type_name") if parsed else None,
          "topk_mismatch_count": topk_cmp.get("mismatch_count"),
          "topk_payload_path": ref["topk_payload_path"],
          "topk_payload_sha256": ref["topk_payload_sha256"],
          "topk_payload_size_bytes": ref["topk_payload_size_bytes"],
          "topk_value_count": topk_cmp.get("lhs_value_count"),
          "weights_max_abs_diff": weights_cmp.get("max_abs_diff"),
          "weights_mean_abs_diff": weights_cmp.get("mean_abs_diff"),
          "weights_norm_max_abs_diff": weights_norm_cmp.get("max_abs_diff"),
          "weights_norm_mean_abs_diff": weights_norm_cmp.get("mean_abs_diff"),
          "weights_norm_payload_path": ref["weights_norm_payload_path"],
          "weights_norm_payload_sha256": ref["weights_norm_payload_sha256"],
          "weights_norm_payload_size_bytes": ref["weights_norm_payload_size_bytes"],
          "weights_norm_rmse": weights_norm_cmp.get("rmse"),
          "weights_norm_value_count": parsed.get("native_weights_norm_vector", {}).get("count") if parsed else None,
          "weights_payload_path": ref["weights_payload_path"],
          "weights_payload_sha256": ref["weights_payload_sha256"],
          "weights_payload_size_bytes": ref["weights_payload_size_bytes"],
          "weights_rmse": weights_cmp.get("rmse"),
          "weights_value_count": parsed.get("native_weights_vector", {}).get("count") if parsed else None,
      },
      "engine_router_topk_compare_passed": passed,
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
      "logits_payload_path": ref["logits_payload_path"],
      "logits_payload_sha256": ref["logits_payload_sha256"],
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-router-topk-compare.py",
      "topk_payload_path": ref["topk_payload_path"],
      "topk_payload_sha256": ref["topk_payload_sha256"],
      "weights_norm_payload_path": ref["weights_norm_payload_path"],
      "weights_norm_payload_sha256": ref["weights_norm_payload_sha256"],
      "weights_payload_path": ref["weights_payload_path"],
      "weights_payload_sha256": ref["weights_payload_sha256"],
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "input_payload_transfer": payload_transfers["input"],
      "logits_payload_transfer": payload_transfers["logits"],
      "mkdir": mkdir,
      "remote_dir": remote_dir,
      "remote_payloads": remote_payloads,
      "source_files": SOURCE_FILES,
      "topk_payload_transfer": payload_transfers["topk"],
      "transfer_results": transfers,
      "weights_norm_payload_transfer": payload_transfers["weights_norm"],
      "weights_payload_transfer": payload_transfers["weights"],
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "router-topk-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfers)},
          {"name": "oracle_router_topk_input_payload_transferred", "pass": payload_transfers["input"].get("returncode") == 0},
          {"name": "oracle_router_topk_logits_payload_transferred", "pass": payload_transfers["logits"].get("returncode") == 0},
          {"name": "oracle_router_topk_indices_payload_transferred", "pass": payload_transfers["topk"].get("returncode") == 0},
          {"name": "oracle_router_topk_weights_payload_transferred", "pass": payload_transfers["weights"].get("returncode") == 0},
          {"name": "oracle_router_topk_weights_norm_payload_transferred", "pass": payload_transfers["weights_norm"].get("returncode") == 0},
          {"name": "target_engine_router_topk_compare_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_router_topk_compare_ran", "pass": compare.get("returncode") == 0},
          {"name": "target_engine_router_topk_compare_output_parsed", "pass": bool(parsed)},
          {"name": "router_topk_matches_oracle_payloads", "pass": passed},
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_router_topk_compare_passed": passed,
      "gate": "r1_engine_router_topk_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine router top-k compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
