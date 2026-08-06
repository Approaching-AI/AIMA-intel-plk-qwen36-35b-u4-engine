#!/usr/bin/env python3
"""Build and run the engine-side L0 linear attention delta core compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-linear-attn-delta-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-linear-attn-delta-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/linear_attn_delta_compare.cpp", "tests/linear_attn_delta_compare.cpp"),
]

PAYLOAD_SPECS = {
    "q_conv_predelta": {
        "stage_name": "q_conv_predelta.bin",
        "path": PAYLOAD_ROOT / "q_conv_predelta-0__tok15__ord6.bin",
        "size_bytes": 8192,
        "value_count": 2048,
        "tensor_name": "q_conv_predelta-0",
    },
    "k_conv_predelta": {
        "stage_name": "k_conv_predelta.bin",
        "path": PAYLOAD_ROOT / "k_conv_predelta-0__tok15__ord8.bin",
        "size_bytes": 8192,
        "value_count": 2048,
        "tensor_name": "k_conv_predelta-0",
    },
    "v_conv_predelta": {
        "stage_name": "v_conv_predelta.bin",
        "path": PAYLOAD_ROOT / "v_conv_predelta-0__tok15__ord9.bin",
        "size_bytes": 16384,
        "value_count": 4096,
        "tensor_name": "v_conv_predelta-0",
    },
    "gate": {
        "stage_name": "gate.bin",
        "path": PAYLOAD_ROOT / "gate-0__tok15__ord12.bin",
        "size_bytes": 128,
        "value_count": 32,
        "tensor_name": "gate-0",
    },
    "beta_sigmoid": {
        "stage_name": "beta_sigmoid.bin",
        "path": PAYLOAD_ROOT / "beta_sigmoid-0__tok15__ord14.bin",
        "size_bytes": 128,
        "value_count": 32,
        "tensor_name": "beta_sigmoid-0",
    },
    "state_predelta": {
        "stage_name": "state_predelta.bin",
        "path": PAYLOAD_ROOT / "state_predelta-0__tok15__ord15.bin",
        "size_bytes": 2097152,
        "value_count": 524288,
        "tensor_name": "state_predelta-0",
    },
    "z": {
        "stage_name": "z.bin",
        "path": PAYLOAD_ROOT / "z-0__tok15__ord17.bin",
        "size_bytes": 16384,
        "value_count": 4096,
        "tensor_name": "z-0",
    },
    "attention_output": {
        "stage_name": "attn_output.bin",
        "path": PAYLOAD_ROOT / "attn_output-0__tok15__ord16.bin",
        "size_bytes": 16384,
        "value_count": 4096,
        "tensor_name": "attn_output-0",
    },
    "final_output": {
        "stage_name": "final_output.bin",
        "path": PAYLOAD_ROOT / "final_output-0__tok15__ord18.bin",
        "size_bytes": 16384,
        "value_count": 4096,
        "tensor_name": "final_output-0",
    },
}


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


def resolve_payloads() -> dict[str, dict[str, Any]]:
  resolved: dict[str, dict[str, Any]] = {}
  for name, spec in PAYLOAD_SPECS.items():
    path = Path(spec["path"]).resolve()
    if not path.exists():
      raise SystemExit(f"linear attention delta payload missing: {path}")
    if path.stat().st_size != spec["size_bytes"]:
      raise SystemExit(f"linear attention delta payload size mismatch: {path}")
    resolved[name] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "stage_name": spec["stage_name"],
        "tensor_name": spec["tensor_name"],
        "value_count": spec["value_count"],
        "local_path": path,
    }
  return resolved


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = next(
      (
          row for row in inputs
          if row.get("boundary_type") == "attention"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "attention"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  projection_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "attention_output_projection"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing L0 attention boundary rows")
  if not isinstance(projection_row, dict):
    raise SystemExit("oracle bundle missing L0 attention output projection row")

  payloads = resolve_payloads()
  return {
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "payloads": payloads,
      "policy_id": input_row.get("policy_id"),
      "source_prompt_case_id": input_row.get("source_prompt_case_id"),
      "source_token_position": input_row.get("source_token_position"),
      "attention_output_task_id": projection_row.get("task_id"),
      "final_output_task_id": output_row.get("task_id"),
  }


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-5
      and comparison.get("rmse") <= 5e-6
      and comparison.get("cosine") >= 0.999999
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  input_vectors = parsed.get("input_vectors", {})
  native_vectors = parsed.get("native_vectors", {})
  oracle_vectors = parsed.get("oracle_vectors", {})
  norm_tensor = parsed.get("norm_tensor", {})
  comparisons = parsed.get("comparisons", {})
  expected_input_counts = {
      "q_conv_predelta": 2048,
      "k_conv_predelta": 2048,
      "v_conv_predelta": 4096,
      "gate": 32,
      "beta_sigmoid": 32,
      "state_predelta": 524288,
      "z": 4096,
  }
  input_counts_ok = all(
      input_vectors.get(name, {}).get("count") == count
      and input_vectors.get(name, {}).get("finite") is True
      and input_vectors.get(name, {}).get("nonzero") is True
      for name, count in expected_input_counts.items()
  )
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("layer_index") == 0
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and norm_tensor.get("name") == "blk.0.ssm_norm.weight"
      and norm_tensor.get("type_name") == "F32"
      and norm_tensor.get("dims") == [128]
      and norm_tensor.get("shape_ok") is True
      and parsed.get("norm_weight", {}).get("count") == 128
      and parsed.get("norm_weight", {}).get("finite") is True
      and parsed.get("norm_weight", {}).get("nonzero") is True
      and input_counts_ok
      and native_vectors.get("attention_output", {}).get("count") == 4096
      and native_vectors.get("attention_output", {}).get("finite") is True
      and native_vectors.get("attention_output", {}).get("nonzero") is True
      and native_vectors.get("recurrent_state", {}).get("count") == 524288
      and native_vectors.get("recurrent_state", {}).get("finite") is True
      and native_vectors.get("recurrent_state", {}).get("nonzero") is True
      and native_vectors.get("final_output", {}).get("count") == 4096
      and native_vectors.get("final_output", {}).get("finite") is True
      and native_vectors.get("final_output", {}).get("nonzero") is True
      and oracle_vectors.get("attention_output", {}).get("count") == 4096
      and oracle_vectors.get("attention_output", {}).get("finite") is True
      and oracle_vectors.get("attention_output", {}).get("nonzero") is True
      and oracle_vectors.get("final_output", {}).get("count") == 4096
      and oracle_vectors.get("final_output", {}).get("finite") is True
      and oracle_vectors.get("final_output", {}).get("nonzero") is True
      and comparison_passed(comparisons.get("attention_output", {}))
      and comparison_passed(comparisons.get("final_output", {}))
  )


def slim_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      name: {
          key: value
          for key, value in payload.items()
          if key != "local_path"
      }
      for name, payload in payloads.items()
  }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_linear_attn_delta_compare"]
  rows = [
      ("engine_linear_attn_delta_compare_passed", payload["engine_linear_attn_delta_compare_passed"]),
      ("q_value_count", state.get("vector_counts", {}).get("q_conv_predelta")),
      ("k_value_count", state.get("vector_counts", {}).get("k_conv_predelta")),
      ("v_value_count", state.get("vector_counts", {}).get("v_conv_predelta")),
      ("state_value_count", state.get("vector_counts", {}).get("state_predelta")),
      ("attention_output_max_abs_diff", state.get("comparisons", {}).get("attention_output", {}).get("max_abs_diff")),
      ("attention_output_rmse", state.get("comparisons", {}).get("attention_output", {}).get("rmse")),
      ("attention_output_cosine", state.get("comparisons", {}).get("attention_output", {}).get("cosine")),
      ("final_output_max_abs_diff", state.get("comparisons", {}).get("final_output", {}).get("max_abs_diff")),
      ("final_output_rmse", state.get("comparisons", {}).get("final_output", {}).get("rmse")),
      ("final_output_cosine", state.get("comparisons", {}).get("final_output", {}).get("cosine")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_linear_attn_delta_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_linear_attn_delta_compare"]
  attn = state["comparisons"]["attention_output"]
  final = state["comparisons"]["final_output"]
  lines = [
      "# R1 Engine Linear Attention Delta Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- policy id: `{state.get('policy_id')}`",
      f"- target build returncode: {payload['target_build']['returncode']}",
      f"- target compare returncode: {payload['target_compare']['returncode']}",
      f"- attention output max abs diff: {attn.get('max_abs_diff')}",
      f"- attention output RMSE: {attn.get('rmse')}",
      f"- attention output cosine: {attn.get('cosine')}",
      f"- final output max abs diff: {final.get('max_abs_diff')}",
      f"- final output RMSE: {final.get('rmse')}",
      f"- final output cosine: {final.get('cosine')}",
      f"- linear attention delta compare passed: `{str(payload['engine_linear_attn_delta_compare_passed']).lower()}`",
      "",
      "This artifact validates the engine-side scalar-gated delta state update",
      "and gated RMSNorm for the L0 linear attention path using oracle",
      "predelta tensors. It does not implement the convolution/state input",
      "path, full-attention KV updates, native token rows, or speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-linear-attn-delta-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-linear-attn-delta-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)
  remote_payload_dir = f"{remote_dir}/oracle"

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {}
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    for name, payload in ref["payloads"].items():
      remote_path = f"{remote_payload_dir}/{payload['stage_name']}"
      payload_transfers[name] = copy_to(
          args.host,
          payload["local_path"],
          remote_path,
          args.timeout_s,
      )
  else:
    payload_transfers = {
        name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
        for name in ref["payloads"]
    }

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/linear_attn_delta_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-linear-attn-delta-compare')}",
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
      f"{shlex.quote(remote_dir + '/build/iq36-linear-attn-delta-compare')} "
      f"{shlex.quote(args.model)} "
      f"{shlex.quote(remote_payload_dir)}"
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
  comparisons = parsed.get("comparisons", {}) if parsed else {}
  vector_counts = {
      "attention_output": parsed.get("native_vectors", {}).get("attention_output", {}).get("count") if parsed else None,
      "beta_sigmoid": parsed.get("input_vectors", {}).get("beta_sigmoid", {}).get("count") if parsed else None,
      "final_output": parsed.get("native_vectors", {}).get("final_output", {}).get("count") if parsed else None,
      "gate": parsed.get("input_vectors", {}).get("gate", {}).get("count") if parsed else None,
      "k_conv_predelta": parsed.get("input_vectors", {}).get("k_conv_predelta", {}).get("count") if parsed else None,
      "norm_weight": parsed.get("norm_weight", {}).get("count") if parsed else None,
      "q_conv_predelta": parsed.get("input_vectors", {}).get("q_conv_predelta", {}).get("count") if parsed else None,
      "recurrent_state": parsed.get("native_vectors", {}).get("recurrent_state", {}).get("count") if parsed else None,
      "state_predelta": parsed.get("input_vectors", {}).get("state_predelta", {}).get("count") if parsed else None,
      "v_conv_predelta": parsed.get("input_vectors", {}).get("v_conv_predelta", {}).get("count") if parsed else None,
      "z": parsed.get("input_vectors", {}).get("z", {}).get("count") if parsed else None,
  }
  payload = {
      "created_at": created_at,
      "engine_linear_attn_delta_compare": {
          "attention_output_task_id": ref["attention_output_task_id"],
          "boundary_type": "linear_attention_delta_core",
          "comparisons": {
              "attention_output": comparisons.get("attention_output", {}),
              "final_output": comparisons.get("final_output", {}),
          },
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "epsilon": parsed.get("epsilon") if parsed else None,
          "final_output_task_id": ref["final_output_task_id"],
          "layer_index": parsed.get("layer_index") if parsed else None,
          "payloads": slim_payloads(ref["payloads"]),
          "policy_id": ref["policy_id"],
          "source_prompt_case_id": ref["source_prompt_case_id"],
          "source_token_position": ref["source_token_position"],
          "tensor_dims": parsed.get("norm_tensor", {}).get("dims") if parsed else None,
          "tensor_name": parsed.get("norm_tensor", {}).get("name") if parsed else None,
          "tensor_type": parsed.get("norm_tensor", {}).get("type_name") if parsed else None,
          "vector_counts": vector_counts,
      },
      "engine_linear_attn_delta_compare_passed": passed,
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
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "payloads": slim_payloads(ref["payloads"]),
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-linear-attn-delta-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfers": payload_transfers,
      "remote_dir": remote_dir,
      "remote_payload_dir": remote_payload_dir,
      "source_files": SOURCE_FILES,
      "transfer_results": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "linear-attn-delta-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfers)},
          {"name": "oracle_linear_attention_delta_payloads_transferred", "pass": all(item.get("returncode") == 0 for item in payload_transfers.values())},
          {"name": "target_engine_linear_attention_delta_compare_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_linear_attention_delta_compare_ran", "pass": compare.get("returncode") == 0},
          {"name": "target_engine_linear_attention_delta_compare_output_parsed", "pass": bool(parsed)},
          {"name": "linear_attention_delta_core_matches_oracle_payloads", "pass": passed},
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_linear_attn_delta_compare_passed": passed,
      "gate": "r1_engine_linear_attention_delta_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine linear attention delta compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
