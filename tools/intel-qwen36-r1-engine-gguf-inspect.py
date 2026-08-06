#!/usr/bin/env python3
"""Build and run the engine-side GGUF inspector on the PTL target."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-engine-gguf-inspect-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/gguf_inspect.cpp", "tests/gguf_inspect.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
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
          "phase": "r1_engine_gguf_inspect",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  inspect = payload["engine_gguf_inspect"]
  payload_smoke = payload["payload_smoke"]
  lines = [
      "# R1 Engine GGUF Inspect",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- target build returncode: {payload['target_build']['returncode']}",
      f"- target inspect returncode: {payload['target_inspect']['returncode']}",
      f"- tensors: {inspect.get('tensor_count')}",
      f"- linear/SSM layers: {inspect.get('linear_ssm_layer_count')}",
      f"- full-attention layers: {inspect.get('full_attention_layer_count')}",
      f"- payload smoke tensors: {payload_smoke.get('decoded_tensor_count')}",
      f"- engine GGUF inspect passed: `{str(payload['engine_gguf_inspect_passed']).lower()}`",
      "",
      "This artifact proves the engine-side C++ GGUF parser can build and read",
      "the locked target model tensor index and decode representative tensor",
      "payload blocks. It does not run inference, generate token candidates, or",
      "allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def inspect_passed(parsed: dict[str, Any], build: dict[str, Any], inspect: dict[str, Any]) -> bool:
  load_map = parsed.get("native_gguf_load_map", {})
  payload_smoke = parsed.get("payload_smoke", {})
  return (
      build.get("returncode") == 0
      and inspect.get("returncode") == 0
      and parsed.get("schema_version") == "intel-qwen36-engine-gguf-inspect-v0"
      and parsed.get("model_path") == DEFAULT_MODEL
      and parsed.get("file_size_bytes") == 21166755168
      and parsed.get("header", {}).get("version") == 3
      and parsed.get("header", {}).get("metadata_kv_count") == 45
      and parsed.get("header", {}).get("tensor_count") == 693
      and load_map.get("native_gguf_load_map_ready") is True
      and load_map.get("failed_check_count") == 0
      and load_map.get("tensor_count") == 693
      and load_map.get("linear_ssm_layer_count") == 30
      and load_map.get("full_attention_layer_count") == 10
      and load_map.get("full_attention_layers") == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
      and load_map.get("tensor_type_counts") == {"F32": 301, "Q4_K": 331, "Q6_K": 61}
      and payload_smoke.get("passed") is True
      and payload_smoke.get("decoded_tensor_count") == 3
      and {
          row.get("type_name")
          for row in payload_smoke.get("tensors", [])
          if isinstance(row, dict)
      } == {"F32", "Q4_K", "Q6_K"}
      and all(
          row.get("finite") is True and row.get("nonzero") is True
          for row in payload_smoke.get("tensors", [])
          if isinstance(row, dict)
      )
  )


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-gguf-inspect-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-gguf-inspect-{stamp}"

  mkdir = run_target(
      args.host,
      "mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build")
      ),
      args.timeout_s,
  )
  transfer_results = []
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      transfer_results.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/gguf_inspect.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-gguf-inspect')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if mkdir["returncode"] == 0 and all(item["returncode"] == 0 for item in transfer_results)
      else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  inspect_command = (
      f"{shlex.quote(remote_dir + '/build/iq36-gguf-inspect')} "
      f"{shlex.quote(args.model)}"
  )
  inspect = (
      run_target(args.host, inspect_command, args.timeout_s)
      if build["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  parsed: dict[str, Any] = {}
  parse_error = None
  if inspect.get("stdout"):
    try:
      parsed = json.loads(inspect["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  passed = bool(parsed) and inspect_passed(parsed, build, inspect)
  load_map = parsed.get("native_gguf_load_map", {}) if parsed else {}
  payload_smoke = parsed.get("payload_smoke", {}) if parsed else {}
  payload = {
      "created_at": created_at,
      "engine_gguf_inspect": {
          "full_attention_layer_count": load_map.get("full_attention_layer_count"),
          "full_attention_layers": load_map.get("full_attention_layers"),
          "linear_ssm_layer_count": load_map.get("linear_ssm_layer_count"),
          "native_gguf_load_map_ready": load_map.get("native_gguf_load_map_ready"),
          "tensor_count": load_map.get("tensor_count"),
          "tensor_type_counts": load_map.get("tensor_type_counts"),
      },
      "engine_gguf_inspect_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "payload_smoke": {
          "decoded_tensor_count": payload_smoke.get("decoded_tensor_count"),
          "passed": payload_smoke.get("passed"),
          "tensor_names": [
              row.get("name")
              for row in payload_smoke.get("tensors", [])
              if isinstance(row, dict)
          ],
          "tensor_types": [
              row.get("type_name")
              for row in payload_smoke.get("tensors", [])
              if isinstance(row, dict)
          ],
      },
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": {
          "returncode": build.get("returncode"),
          "stderr": build.get("stderr"),
          "stdout": build.get("stdout"),
      },
      "target_inspect": {
          "returncode": inspect.get("returncode"),
          "stderr": inspect.get("stderr"),
          "stdout": inspect.get("stdout"),
      },
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-gguf-inspect.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "remote_dir": remote_dir,
      "source_files": SOURCE_FILES,
      "transfer_results": transfer_results,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "inspect-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "inspect.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
          {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfer_results)},
          {"name": "target_engine_inspector_built", "pass": build.get("returncode") == 0},
          {"name": "target_engine_inspector_ran", "pass": inspect.get("returncode") == 0},
          {"name": "target_engine_inspector_output_parsed", "pass": bool(parsed)},
          {"name": "locked_model_shape_validated_by_engine_cpp", "pass": passed},
          {
              "name": "representative_tensor_payload_blocks_decoded",
              "pass": payload_smoke.get("passed") is True,
          },
          {"name": "does_not_close_native_token_correctness", "pass": True},
      ],
      "engine_gguf_inspect_passed": passed,
      "gate": "r1_engine_gguf_inspect",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metric(out_dir / "metrics.jsonl", [
      ("engine_gguf_inspect_passed", passed),
      ("tensor_count", load_map.get("tensor_count")),
      ("linear_ssm_layer_count", load_map.get("linear_ssm_layer_count")),
      ("full_attention_layer_count", load_map.get("full_attention_layer_count")),
      ("payload_smoke_decoded_tensor_count", payload_smoke.get("decoded_tensor_count")),
      ("payload_smoke_passed", payload_smoke.get("passed")),
      ("r1_native_correctness_gate_closed", False),
  ])
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine gguf inspect output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
