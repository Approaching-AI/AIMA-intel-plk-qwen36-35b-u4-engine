#!/usr/bin/env python3
"""Build and run the engine-side all-linear-layer post-conv compare.

This validates token-15 linear-attention post-conv math for every linear layer:
alpha/beta/gate/z projections, q/k/v slicing and norms, gated-delta update
output, RMSNorm output, and the ssm_out projection. It uses oracle
conv_output_raw and state_predelta payloads as inputs, so it is component
evidence only and does not prove convolution history replay or the 40-layer
native generation loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess

import iq36_local
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-engine-linear-attn-all-postconv-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-linear-attn-all-postconv-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
R0_REMOTE_OUTPUT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output"
R0_TENSOR_DUMPS = R0_REMOTE_OUTPUT / "tensor-dumps.jsonl"
SOURCE_TOKEN_POSITION = 15
LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12,
    13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
    26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    (
        "engine/tests/linear_attn_all_postconv_compare.cpp",
        "tests/linear_attn_all_postconv_compare.cpp",
    ),
]

PAYLOAD_ROLES = {
    "attn_norm": {"tensor_prefix": "attn_norm", "size_bytes": 8192, "value_count": 2048},
    "conv_output_raw": {"tensor_prefix": "conv_output_raw", "size_bytes": 32768, "value_count": 8192},
    "conv_output_silu": {"tensor_prefix": "conv_output_silu", "size_bytes": 32768, "value_count": 8192},
    "q_conv": {"tensor_prefix": "q_conv", "size_bytes": 8192, "value_count": 2048},
    "q_conv_predelta": {"tensor_prefix": "q_conv_predelta", "size_bytes": 8192, "value_count": 2048},
    "k_conv": {"tensor_prefix": "k_conv", "size_bytes": 8192, "value_count": 2048},
    "k_conv_predelta": {"tensor_prefix": "k_conv_predelta", "size_bytes": 8192, "value_count": 2048},
    "v_conv_predelta": {"tensor_prefix": "v_conv_predelta", "size_bytes": 16384, "value_count": 4096},
    "alpha": {"tensor_prefix": "alpha", "size_bytes": 128, "value_count": 32},
    "a_softplus": {"tensor_prefix": "a_softplus", "size_bytes": 128, "value_count": 32},
    "gate": {"tensor_prefix": "gate", "size_bytes": 128, "value_count": 32},
    "beta": {"tensor_prefix": "beta", "size_bytes": 128, "value_count": 32},
    "beta_sigmoid": {"tensor_prefix": "beta_sigmoid", "size_bytes": 128, "value_count": 32},
    "state_predelta": {"tensor_prefix": "state_predelta", "size_bytes": 2097152, "value_count": 524288},
    "attention_output": {"tensor_prefix": "attn_output", "size_bytes": 16384, "value_count": 4096},
    "z": {"tensor_prefix": "z", "size_bytes": 16384, "value_count": 4096},
    "final_output": {"tensor_prefix": "final_output", "size_bytes": 16384, "value_count": 4096},
    "linear_attn_out": {"tensor_prefix": "linear_attn_out", "size_bytes": 8192, "value_count": 2048},
}

INPUT_ROLES = {"attn_norm", "conv_output_raw", "state_predelta"}
EXPECTED_COMPARISONS = set(PAYLOAD_ROLES) - INPUT_ROLES


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=900)
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
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def layer_prefix(layer_index: int) -> str:
  return f"l{layer_index:02d}"


def find_tensor_row(rows: list[dict[str, Any]], tensor_name: str) -> dict[str, Any]:
  matches = [
      row for row in rows
      if row.get("case_id") == "short_math_001"
      and row.get("source_token_position") == SOURCE_TOKEN_POSITION
      and row.get("tensor_name") == tensor_name
  ]
  if len(matches) != 1:
    raise SystemExit(
        f"{R0_TENSOR_DUMPS}: expected one token {SOURCE_TOKEN_POSITION} "
        f"row for {tensor_name}, found {len(matches)}"
    )
  return matches[0]


def payload_entry(
    rows: list[dict[str, Any]],
    layer_index: int,
    role: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
  tensor_name = f"{spec['tensor_prefix']}-{layer_index}"
  row = find_tensor_row(rows, tensor_name)
  payload_path = row.get("payload_path")
  if not isinstance(payload_path, str) or not payload_path:
    raise SystemExit(f"{R0_TENSOR_DUMPS}: row missing payload_path for {tensor_name}")
  path = (R0_REMOTE_OUTPUT / payload_path).resolve()
  if not path.exists():
    raise SystemExit(f"missing R0 payload: {path}")
  if path.stat().st_size != spec["size_bytes"]:
    raise SystemExit(f"R0 payload size mismatch for {path}")
  stage_name = f"{layer_prefix(layer_index)}_{role}.bin"
  return {
      "layer_index": layer_index,
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "payload_role": role,
      "sha256": sha256_file(path),
      "size_bytes": spec["size_bytes"],
      "stage_name": stage_name,
      "tensor_name": row.get("tensor_name"),
      "tensor_op": row.get("tensor_op"),
      "token_position": SOURCE_TOKEN_POSITION,
      "value_count": spec["value_count"],
  }


def resolve_payloads() -> dict[str, Any]:
  rows = load_jsonl(R0_TENSOR_DUMPS)
  payloads: dict[str, dict[str, Any]] = {}
  for layer in LINEAR_LAYERS:
    for role, spec in PAYLOAD_ROLES.items():
      entry = payload_entry(rows, layer, role, spec)
      payloads[entry["stage_name"]] = entry
  return {
      "linear_layers": LINEAR_LAYERS,
      "payloads": payloads,
      "r0_tensor_dumps": str(R0_TENSOR_DUMPS.relative_to(ROOT)),
      "source_prompt_case_id": "short_math_001",
      "source_token_position": SOURCE_TOKEN_POSITION,
  }


def slim_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }


def stage_payload_tree(payloads: dict[str, dict[str, Any]], stage_dir: Path) -> None:
  stage_dir.mkdir(parents=True, exist_ok=True)
  for stage_name, payload in payloads.items():
    target = stage_dir / stage_name
    source = payload["local_path"]
    try:
      os.link(source, target)
    except OSError:
      shutil.copy2(source, target)


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-4
      and comparison.get("rmse") <= 5e-5
      and comparison.get("cosine") >= 0.99999
  )


def parsed_layer_passed(layer: dict[str, Any]) -> bool:
  comparisons = layer.get("comparisons", {})
  return (
      layer.get("passed") is True
      and layer.get("counts_ok") is True
      and layer.get("stats_ok") is True
      and layer.get("comparisons_ok") is True
      and layer.get("tensors", {}).get("shape_ok") is True
      and set(comparisons) == EXPECTED_COMPARISONS
      and all(comparison_passed(comparisons[name]) for name in EXPECTED_COMPARISONS)
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  layers = parsed.get("layers", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("prompt_case_id") == "short_math_001"
      and parsed.get("source_token_position") == SOURCE_TOKEN_POSITION
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and parsed.get("layers_ok") is True
      and parsed.get("layer_count") == len(LINEAR_LAYERS)
      and parsed.get("linear_layers") == LINEAR_LAYERS
      and set(layers) == {str(layer) for layer in LINEAR_LAYERS}
      and all(parsed_layer_passed(layer) for layer in layers.values())
  )


def layer_comparison_summary(parsed: dict[str, Any]) -> dict[str, Any]:
  summary: dict[str, Any] = {}
  for layer_key, layer in parsed.get("layers", {}).items():
    layer_summary: dict[str, Any] = {}
    for name, comparison in layer.get("comparisons", {}).items():
      layer_summary[name] = {
          "cosine": comparison.get("cosine"),
          "max_abs_diff": comparison.get("max_abs_diff"),
          "mean_abs_diff": comparison.get("mean_abs_diff"),
          "mismatch_count": comparison.get("mismatch_count"),
          "rmse": comparison.get("rmse"),
      }
    summary[layer_key] = layer_summary
  return summary


def worst_metric(parsed: dict[str, Any], comparison_name: str, metric: str, default: float) -> float:
  values: list[float] = []
  for layer in parsed.get("layers", {}).values():
    value = layer.get("comparisons", {}).get(comparison_name, {}).get(metric)
    if isinstance(value, (int, float)):
      values.append(float(value))
  if not values:
    return default
  if metric == "cosine":
    return min(values)
  return max(values)


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_linear_attn_all_postconv_compare"]
  rows = [
      (
          "engine_linear_attn_all_postconv_compare_passed",
          payload["engine_linear_attn_all_postconv_compare_passed"],
      ),
      ("linear_layer_count", len(state.get("linear_layers", []))),
      ("staged_payload_count", state.get("staged_payload_count")),
      ("attention_output_worst_max_abs_diff", state.get("worst_metrics", {}).get("attention_output_max_abs_diff")),
      ("final_output_worst_max_abs_diff", state.get("worst_metrics", {}).get("final_output_max_abs_diff")),
      ("linear_attn_out_worst_max_abs_diff", state.get("worst_metrics", {}).get("linear_attn_out_max_abs_diff")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_linear_attn_all_postconv_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_linear_attn_all_postconv_compare"]
  worst = state["worst_metrics"]
  lines = [
      "# R1 Engine All-Linear Post-Conv Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- source tensor dump: `{state['r0_tensor_dumps']}`",
      f"- linear layers: `{state['linear_layers']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- staged payloads: {state['staged_payload_count']}",
      f"- worst attention output max abs diff: {worst.get('attention_output_max_abs_diff')}",
      f"- worst final output max abs diff: {worst.get('final_output_max_abs_diff')}",
      f"- worst ssm_out projection max abs diff: {worst.get('linear_attn_out_max_abs_diff')}",
      f"- all-linear post-conv compare passed: `{str(payload['engine_linear_attn_all_postconv_compare_passed']).lower()}`",
      "",
      "This artifact validates token-15 linear-attention post-conv math for",
      "all 30 linear layers using oracle conv_output_raw and state_predelta",
      "inputs. It does not prove convolution history replay, residual chaining,",
      "the integrated 40-layer loop, native candidate JSONL rows, or speedup.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-linear-attn-all-postconv-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-linear-attn-all-postconv-compare-{stamp}"
  remote_payload_dir = f"{remote_dir}/payloads"
  ref = resolve_payloads()

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "payloads")
      ),
      args.timeout_s,
  )
  source_transfers: list[dict[str, Any]] = []
  payload_transfer: dict[str, Any] = {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    if all(item["returncode"] == 0 for item in source_transfers):
      with tempfile.TemporaryDirectory(prefix="iq36-linear-all-postconv-") as temp_dir:
        local_payload_dir = Path(temp_dir) / "payloads"
        stage_payload_tree(ref["payloads"], local_payload_dir)
        payload_transfer = copy_tree_to(
            args.host,
            local_payload_dir,
            remote_payload_dir,
            args.timeout_s,
        )

  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and payload_transfer.get("returncode") == 0
  )
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/linear_attn_all_postconv_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-linear-attn-all-postconv-compare')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-linear-attn-all-postconv-compare"),
      shlex.quote(args.model),
      shlex.quote(remote_payload_dir),
  ])
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
  worst_metrics = {
      "attention_output_max_abs_diff": worst_metric(parsed, "attention_output", "max_abs_diff", 0.0),
      "attention_output_rmse": worst_metric(parsed, "attention_output", "rmse", 0.0),
      "attention_output_cosine": worst_metric(parsed, "attention_output", "cosine", 0.0),
      "final_output_max_abs_diff": worst_metric(parsed, "final_output", "max_abs_diff", 0.0),
      "final_output_rmse": worst_metric(parsed, "final_output", "rmse", 0.0),
      "final_output_cosine": worst_metric(parsed, "final_output", "cosine", 0.0),
      "linear_attn_out_max_abs_diff": worst_metric(parsed, "linear_attn_out", "max_abs_diff", 0.0),
      "linear_attn_out_rmse": worst_metric(parsed, "linear_attn_out", "rmse", 0.0),
      "linear_attn_out_cosine": worst_metric(parsed, "linear_attn_out", "cosine", 0.0),
  }
  state = {
      "boundary_type": "linear_attention_all_postconv_core",
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "layer_comparisons": layer_comparison_summary(parsed) if parsed else {},
      "layer_count": len(LINEAR_LAYERS),
      "linear_layers": LINEAR_LAYERS,
      "payloads": slim_payloads(ref["payloads"]),
      "r0_tensor_dumps": ref["r0_tensor_dumps"],
      "source_prompt_case_id": ref["source_prompt_case_id"],
      "source_token_position": ref["source_token_position"],
      "staged_payload_count": len(ref["payloads"]),
      "target_build_returncode": build.get("returncode"),
      "target_compare_returncode": compare.get("returncode"),
      "worst_metrics": worst_metrics,
  }
  payload = {
      "created_at": created_at,
      "engine_linear_attn_all_postconv_compare": state,
      "engine_linear_attn_all_postconv_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": build,
      "target_compare": compare,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "linear_layers": LINEAR_LAYERS,
      "model_path": args.model,
      "payloads": slim_payloads(ref["payloads"]),
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-linear-attn-all-postconv-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfer": payload_transfer,
      "remote_dir": remote_dir,
      "remote_payload_dir": remote_payload_dir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
      "staged_payload_count": len(ref["payloads"]),
  })
  write_json(out_dir / "build.json", build)
  write_json(
      out_dir / "linear-attn-all-postconv-stdout.json",
      parsed if parsed else {"parse_error": parse_error},
  )
  write_json(out_dir / "compare.json", payload)
  checks = [
      {"name": "r0_tensor_dump_available", "pass": R0_TENSOR_DUMPS.exists()},
      {"name": "expected_all_linear_payloads_resolved", "pass": len(ref["payloads"]) == len(LINEAR_LAYERS) * len(PAYLOAD_ROLES)},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {"name": "linear_attention_all_postconv_payloads_transferred", "pass": payload_transfer.get("returncode") == 0},
      {"name": "target_engine_linear_attention_all_postconv_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_linear_attention_all_postconv_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_linear_attention_all_postconv_compare_output_parsed", "pass": bool(parsed)},
      {"name": "all_linear_attention_postconv_layers_match_oracle_payloads", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_linear_attn_all_postconv_compare_passed": passed,
      "gate": "r1_engine_linear_attention_all_postconv_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine linear-attn all-postconv compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
