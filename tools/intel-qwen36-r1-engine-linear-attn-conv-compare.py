#!/usr/bin/env python3
"""Build and run the engine-side L0 linear attention conv-state compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-linear-attn-conv-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-linear-attn-conv-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/linear_attn_conv_compare.cpp", "tests/linear_attn_conv_compare.cpp"),
]

PAYLOAD_SPECS = {
    "model_input_embed": ("model_input_embed.bin", "model.input_embed__tok15__ord0.bin", 8192, 2048),
    "attn_norm": ("attn_norm.bin", "attn_norm-0__tok15__ord1.bin", 8192, 2048),
    "linear_attn_qkv_mixed": ("linear_attn_qkv_mixed.bin", "linear_attn_qkv_mixed-0__tok15__ord2.bin", 32768, 8192),
    "conv_output_raw": ("conv_output_raw.bin", "conv_output_raw-0__tok15__ord3.bin", 32768, 8192),
}

COMPARISON_NAMES = [
    "model_input_embed",
    "attention_norm",
    "linear_attn_qkv_mixed",
    "conv_output_raw",
]

EXPECTED_PROMPT_TOKEN_IDS = [
    15666, 303, 799, 2716, 11316, 25, 1092, 369,
    220, 16, 22, 5346, 220, 17, 20, 30,
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
  return {"cmd": cmd, "returncode": proc.returncode, "stderr": proc.stderr, "stdout": proc.stdout}


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
  for name, (stage_name, file_name, size_bytes, value_count) in PAYLOAD_SPECS.items():
    path = (PAYLOAD_ROOT / file_name).resolve()
    if not path.exists():
      raise SystemExit(f"linear attention conv payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"linear attention conv payload size mismatch: {path}")
    resolved[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
        "tensor_name": file_name.split("__", 1)[0],
        "value_count": value_count,
    }
  return resolved


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  token_rows = load_jsonl(oracle_bundle / "token-topk-references.jsonl")
  token_row = next((row for row in token_rows if row.get("case_id") == "short_math_001"), None)
  if not isinstance(token_row, dict):
    raise SystemExit("oracle bundle missing short_math_001 token row")
  if token_row.get("prompt_token_ids") != EXPECTED_PROMPT_TOKEN_IDS:
    raise SystemExit("short_math_001 prompt token ids changed")

  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  qkv_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "qkv_projection"
          and row.get("layer") == 0
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(qkv_row, dict):
    raise SystemExit("oracle bundle missing L0 qkv projection output row")

  return {
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "payloads": resolve_payloads(),
      "policy_id": qkv_row.get("policy_id"),
      "prompt_token_count": token_row.get("prompt_token_count"),
      "prompt_token_ids_sha256": token_row.get("prompt_token_ids_sha256"),
      "source_prompt_case_id": qkv_row.get("source_prompt_case_id"),
      "source_token_position": qkv_row.get("source_token_position"),
  }


def slim_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-4
      and comparison.get("rmse") <= 5e-5
      and comparison.get("cosine") >= 0.99999
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  comparisons = parsed.get("comparisons", {})
  native_vectors = parsed.get("native_vectors", {})
  expected_counts = {
      "model_input_embed": 2048,
      "attention_norm": 2048,
      "linear_attn_qkv_mixed": 8192,
      "conv_output_raw": 8192,
      "conv_state_after": 24576,
  }
  vectors_ok = all(
      native_vectors.get(name, {}).get("count") == count
      and native_vectors.get(name, {}).get("finite") is True
      and native_vectors.get(name, {}).get("nonzero") is True
      for name, count in expected_counts.items()
  )
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("prompt_case_id") == "short_math_001"
      and parsed.get("source_token_position") == 15
      and parsed.get("prompt_token_count") == 16
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and parsed.get("tensors", {}).get("shape_ok") is True
      and all(comparison_passed(comparisons.get(name, {})) for name in COMPARISON_NAMES)
      and vectors_ok
  )


def comparison_summary(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
  result = {}
  for name in COMPARISON_NAMES:
    comparison = parsed.get("comparisons", {}).get(name, {})
    result[name] = {
        "max_abs_diff": comparison.get("max_abs_diff"),
        "mean_abs_diff": comparison.get("mean_abs_diff"),
        "rmse": comparison.get("rmse"),
        "cosine": comparison.get("cosine"),
        "mismatch_count": comparison.get("mismatch_count"),
    }
  return result


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_linear_attn_conv_compare"]
  comparisons = state["comparisons"]
  rows = [
      ("engine_linear_attn_conv_compare_passed", payload["engine_linear_attn_conv_compare_passed"]),
      ("attention_norm_max_abs_diff", comparisons.get("attention_norm", {}).get("max_abs_diff")),
      ("linear_attn_qkv_mixed_max_abs_diff", comparisons.get("linear_attn_qkv_mixed", {}).get("max_abs_diff")),
      ("conv_output_raw_max_abs_diff", comparisons.get("conv_output_raw", {}).get("max_abs_diff")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_linear_attn_conv_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  comparisons = payload["engine_linear_attn_conv_compare"]["comparisons"]
  lines = [
      "# R1 Engine Linear Attention Conv-State Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- target build returncode: {payload['target_build']['returncode']}",
      f"- target compare returncode: {payload['target_compare']['returncode']}",
      f"- attention norm max abs diff: {comparisons.get('attention_norm', {}).get('max_abs_diff')}",
      f"- qkv mixed max abs diff: {comparisons.get('linear_attn_qkv_mixed', {}).get('max_abs_diff')}",
      f"- conv output raw max abs diff: {comparisons.get('conv_output_raw', {}).get('max_abs_diff')}",
      f"- conv compare passed: `{str(payload['engine_linear_attn_conv_compare_passed']).lower()}`",
      "",
      "This artifact validates the engine-side L0 recurrent convolution state",
      "input/update path across the 16-token `short_math_001` prompt. It still",
      "does not emit a native candidate token row.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-linear-attn-conv-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-linear-attn-conv-compare-{stamp}"
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
  source_transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in PAYLOAD_SPECS
  }
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    for name, payload in ref["payloads"].items():
      payload_transfers[name] = copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/linear_attn_conv_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-linear-attn-conv-compare')}",
  ])
  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-linear-attn-conv-compare"),
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
  state = {
      "boundary_type": "linear_attention_conv_state",
      "comparisons": comparison_summary(parsed) if parsed else {},
      "conv_kernel_size": parsed.get("conv_kernel_size") if parsed else None,
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "layer_index": 0,
      "payloads": slim_payloads(ref["payloads"]),
      "policy_id": ref["policy_id"],
      "prompt_token_count": ref["prompt_token_count"],
      "prompt_token_ids_sha256": ref["prompt_token_ids_sha256"],
      "source_prompt_case_id": ref["source_prompt_case_id"],
      "source_token_position": ref["source_token_position"],
      "target_build_returncode": build.get("returncode"),
      "target_compare_returncode": compare.get("returncode"),
      "tensors": parsed.get("tensors", {}) if parsed else {},
      "vector_counts": {
          key: value.get("count")
          for key, value in parsed.get("native_vectors", {}).items()
          if isinstance(value, dict)
      } if parsed else {},
  }
  payload = {
      "created_at": created_at,
      "engine_linear_attn_conv_compare": state,
      "engine_linear_attn_conv_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
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
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "payloads": slim_payloads(ref["payloads"]),
      "prompt_token_ids_sha256": ref["prompt_token_ids_sha256"],
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-linear-attn-conv-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfers": payload_transfers,
      "remote_dir": remote_dir,
      "remote_payload_dir": remote_payload_dir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "linear-attn-conv-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  checks = [
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {
          "name": "oracle_linear_attn_conv_payloads_transferred",
          "pass": all(item.get("returncode") == 0 for item in payload_transfers.values()),
      },
      {"name": "target_engine_linear_attn_conv_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_linear_attn_conv_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_linear_attn_conv_compare_output_parsed", "pass": bool(parsed)},
      {"name": "linear_attn_conv_state_matches_oracle_payloads", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_linear_attn_conv_compare_passed": passed,
      "gate": "r1_engine_linear_attn_conv_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine linear attention conv compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
