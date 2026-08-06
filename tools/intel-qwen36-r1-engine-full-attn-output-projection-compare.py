#!/usr/bin/env python3
"""Build and run the engine-side L3 full-attention output projection compare.

This validates the first full-attention layer's output projection from
``attn_gated-3`` to ``attn_output-3`` for ``short_math_001`` token position
15. It is component evidence only: it consumes the oracle attention result,
does not update KV cache, does not run attention, does not emit native
candidate JSONL rows, does not close R1, and does not allow speed claims.
"""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-full-attn-output-projection-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-full-attn-output-projection-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    (
        "engine/tests/full_attn_output_projection_compare.cpp",
        "tests/full_attn_output_projection_compare.cpp",
    ),
]

PAYLOAD_SPECS = {
    "attn_gated": ("attn_gated.bin", "attn_gated-3__tok15__ord128.bin", 16384, 4096),
    "attn_output": ("attn_output.bin", "attn_output-3__tok15__ord129.bin", 8192, 2048),
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
  parser.add_argument("--timeout-s", type=int, default=360)
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
  for name, (stage_name, file_name, size_bytes, value_count) in PAYLOAD_SPECS.items():
    path = (PAYLOAD_ROOT / file_name).resolve()
    if not path.exists():
      raise SystemExit(f"full-attn output projection payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"full-attn output projection payload size mismatch: {path}")
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
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  input_row = next(
      (
          row for row in inputs
          if row.get("boundary_type") == "attention_output_projection"
          and row.get("layer") == 3
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  output_row = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "attention_output_projection"
          and row.get("layer") == 3
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(input_row, dict) or not isinstance(output_row, dict):
    raise SystemExit("oracle bundle missing L3 full-attn output projection rows")
  if input_row.get("source_prompt_case_id") != "short_math_001":
    raise SystemExit("oracle L3 full-attn output prompt case changed")
  if input_row.get("source_token_position") != 15:
    raise SystemExit("oracle L3 full-attn output source token changed")
  input_paths = input_row.get("reference_input_tensor_paths", {})
  output_paths = output_row.get("reference_output_tensor_paths", {})
  if input_paths.get("attn_gated-3") is None:
    raise SystemExit("oracle L3 full-attn output input path missing")
  if output_paths.get("attn_output-3") is None:
    raise SystemExit("oracle L3 full-attn output path missing")
  return {
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "payloads": resolve_payloads(),
      "policy_id": output_row.get("policy_id"),
      "source_prompt_case_id": input_row.get("source_prompt_case_id"),
      "source_token_position": input_row.get("source_token_position"),
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
      and comparison.get("max_abs_diff") <= 5e-3
      and comparison.get("rmse") <= 1e-3
      and comparison.get("cosine") >= 0.99999
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  tensor = parsed.get("tensor", {})
  input_vector = parsed.get("input_vector", {})
  native_vector = parsed.get("native_vector", {})
  oracle_vector = parsed.get("oracle_vector", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("layer_index") == 3
      and parsed.get("prompt_case_id") == "short_math_001"
      and parsed.get("source_token_position") == 15
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and tensor.get("name") == "blk.3.attn_output.weight"
      and tensor.get("type_name") == "Q4_K"
      and tensor.get("dims") == [4096, 2048]
      and tensor.get("shape_ok") is True
      and input_vector.get("count") == 4096
      and input_vector.get("finite") is True
      and input_vector.get("nonzero") is True
      and native_vector.get("count") == 2048
      and native_vector.get("finite") is True
      and native_vector.get("nonzero") is True
      and oracle_vector.get("count") == 2048
      and oracle_vector.get("finite") is True
      and oracle_vector.get("nonzero") is True
      and comparison_passed(parsed.get("comparison", {}))
  )


def comparison_summary(parsed: dict[str, Any]) -> dict[str, Any]:
  comparison = parsed.get("comparison", {})
  return {
      "max_abs_diff": comparison.get("max_abs_diff"),
      "mean_abs_diff": comparison.get("mean_abs_diff"),
      "rmse": comparison.get("rmse"),
      "cosine": comparison.get("cosine"),
      "mismatch_count": comparison.get("mismatch_count"),
  }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_full_attn_output_projection_compare"]
  rows = [
      (
          "engine_full_attn_output_projection_compare_passed",
          payload["engine_full_attn_output_projection_compare_passed"],
      ),
      ("input_value_count", state.get("input_value_count")),
      ("native_value_count", state.get("native_value_count")),
      ("oracle_value_count", state.get("oracle_value_count")),
      ("max_abs_diff", state.get("comparison", {}).get("max_abs_diff")),
      ("rmse", state.get("comparison", {}).get("rmse")),
      ("cosine", state.get("comparison", {}).get("cosine")),
      ("mismatch_count", state.get("comparison", {}).get("mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_full_attn_output_projection_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_full_attn_output_projection_compare"]
  comparison = state["comparison"]
  lines = [
      "# R1 Engine Full-Attention Output Projection Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- layer index: {state['layer_index']}",
      f"- max abs diff: {comparison.get('max_abs_diff')}",
      f"- rmse: {comparison.get('rmse')}",
      f"- cosine: {comparison.get('cosine')}",
      f"- full-attention output projection compare passed: `{str(payload['engine_full_attn_output_projection_compare_passed']).lower()}`",
      "",
      "This artifact validates the first full-attention layer's output",
      "projection from the oracle attention result. It does not update KV",
      "cache, run attention, emit native candidate JSONL rows, or close R1.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-full-attn-output-projection-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-full-attn-output-projection-compare-{stamp}"
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

  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/full_attn_output_projection_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-full-attn-output-projection-compare')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-full-attn-output-projection-compare"),
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
      "boundary_type": "full_attention_output_projection",
      "comparison": comparison_summary(parsed) if parsed else {},
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "input_value_count": parsed.get("input_vector", {}).get("count") if parsed else None,
      "layer_index": 3,
      "native_value_count": parsed.get("native_vector", {}).get("count") if parsed else None,
      "oracle_value_count": parsed.get("oracle_vector", {}).get("count") if parsed else None,
      "payloads": slim_payloads(ref["payloads"]),
      "policy_id": ref["policy_id"],
      "source_prompt_case_id": ref["source_prompt_case_id"],
      "source_token_position": ref["source_token_position"],
      "target_build_returncode": build.get("returncode"),
      "target_compare_returncode": compare.get("returncode"),
      "tensor": parsed.get("tensor", {}) if parsed else {},
  }

  payload = {
      "created_at": created_at,
      "engine_full_attn_output_projection_compare": state,
      "engine_full_attn_output_projection_compare_passed": passed,
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
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-full-attn-output-projection-compare.py",
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
  write_json(
      out_dir / "full-attn-output-projection-stdout.json",
      parsed if parsed else {"parse_error": parse_error},
  )
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
          "name": "oracle_full_attn_output_projection_payloads_transferred",
          "pass": all(item.get("returncode") == 0 for item in payload_transfers.values()),
      },
      {"name": "target_engine_full_attn_output_projection_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_full_attn_output_projection_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_full_attn_output_projection_compare_output_parsed", "pass": bool(parsed)},
      {"name": "full_attention_output_projection_matches_oracle_payload", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_full_attn_output_projection_compare_passed": passed,
      "gate": "r1_engine_full_attn_output_projection_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine full-attn output projection compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
