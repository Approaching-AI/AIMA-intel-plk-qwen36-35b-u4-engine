#!/usr/bin/env python3
"""Preflight target-side reference runtimes for R0 oracle capture."""

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
SCHEMA_VERSION = "intel-qwen36-r0-oracle-runtime-preflight-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_DIR = "/home/intel/llama-cpp/llama-b9518"
LLAMA_SERVER = f"{LLAMA_DIR}/llama-server"
LLAMA_TOKENIZE = f"{LLAMA_DIR}/llama-tokenize"
LLAMA_CLI = f"{LLAMA_DIR}/llama-cli"
OPENVINO_DIR = "/home/intel/ov"
OPENVINO_MODEL = "/home/intel/Qwen3.6-35B-A3B-ov"
OPENVINO_BENCH = f"{OPENVINO_DIR}/benchmark_vlm_new.py"
PRIOR_ORACLE_ROOT = (
    "/home/intel/intel-box-run/native-llama-generation-oracle-cpu-20260615T133419Z"
)
PRIOR_ORACLE_TOOL = f"{PRIOR_ORACLE_ROOT}/tools/native/llama_generation_oracle.py"
PRIOR_ORACLE_RAW = f"{PRIOR_ORACLE_ROOT}/remote-output/llama-generation-oracle/raw"
INTEL_ENV = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-oracle-runtime-preflight-<UTC>.",
  )
  return parser.parse_args()


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_script: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_script, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def bool_value(values: dict[str, str], key: str) -> bool:
  return values.get(key) == "true"


def contains_any(text: str, needles: list[str]) -> bool:
  lowered = text.lower()
  return any(needle.lower() in lowered for needle in needles)


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


def build_summary(payload: dict[str, Any]) -> str:
  routes = payload["oracle_runtime_routes"]
  inventory = payload["target_inventory"]
  lines = [
      "# R0 Oracle Runtime Preflight",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- locked model present: `{str(inventory['model_present']).lower()}`",
      f"- llama-server present: `{str(inventory['llama_server_present']).lower()}`",
      f"- llama-tokenize present: `{str(inventory['llama_tokenize_present']).lower()}`",
      f"- OpenVINO model present: `{str(inventory['openvino_model_present']).lower()}`",
      f"- prior llama oracle tool present: `{str(inventory['prior_oracle_tool_present']).lower()}`",
      f"- prior completion probability raw files: {inventory['prior_completion_probability_raw_count']}",
      f"- distribution capture candidate: `{routes['teacher_forced_distribution']['route_status']}`",
      f"- boundary tensor capture candidate: `{routes['per_boundary_tensors']['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is a capability preflight only. It does not start a long model",
      "server, capture tensors, or create an oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-oracle-runtime-preflight-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  inventory_script = "\n".join([
      "set -u",
      f"printf 'model_present='; test -f {shlex.quote(MODEL_PATH)} && echo true || echo false",
      f"printf 'model_size_bytes='; stat -c %s {shlex.quote(MODEL_PATH)} 2>/dev/null || echo missing",
      f"printf 'llama_server_present='; test -x {shlex.quote(LLAMA_SERVER)} && echo true || echo false",
      f"printf 'llama_tokenize_present='; test -x {shlex.quote(LLAMA_TOKENIZE)} && echo true || echo false",
      f"printf 'llama_cli_present='; test -x {shlex.quote(LLAMA_CLI)} && echo true || echo false",
      f"printf 'openvino_model_present='; test -d {shlex.quote(OPENVINO_MODEL)} && echo true || echo false",
      f"printf 'openvino_bench_present='; test -f {shlex.quote(OPENVINO_BENCH)} && echo true || echo false",
      f"printf 'intel_env_present='; test -f {shlex.quote(INTEL_ENV)} && echo true || echo false",
      f"printf 'prior_oracle_root_present='; test -d {shlex.quote(PRIOR_ORACLE_ROOT)} && echo true || echo false",
      f"printf 'prior_oracle_tool_present='; test -f {shlex.quote(PRIOR_ORACLE_TOOL)} && echo true || echo false",
      f"printf 'prior_oracle_raw_present='; test -d {shlex.quote(PRIOR_ORACLE_RAW)} && echo true || echo false",
      f"printf 'prior_completion_probability_raw_count='; find {shlex.quote(PRIOR_ORACLE_RAW)} -maxdepth 1 -type f -name '*response.json' 2>/dev/null | wc -l",
      "printf 'hostname='; hostname",
  ])
  commands = {
      "inventory": (inventory_script, 30),
      "llama_server_help": (f"{shlex.quote(LLAMA_SERVER)} --help 2>&1 | head -c 80000", 30),
      "llama_tokenize_help": (f"{shlex.quote(LLAMA_TOKENIZE)} --help 2>&1 | head -c 40000", 30),
      "openvino_import": (
          f"cd {shlex.quote(OPENVINO_DIR)} && "
          ". openvino_env/bin/activate && "
          "python - <<'PY'\n"
          "import openvino_genai\n"
          "print('openvino_genai_import=true')\n"
          "print('openvino_genai_version=' + str(getattr(openvino_genai, '__version__', 'unknown')))\n"
          "PY",
          45,
      ),
      "prior_oracle_tool_head": (
          f"test -f {shlex.quote(PRIOR_ORACLE_TOOL)} && sed -n '1,220p' {shlex.quote(PRIOR_ORACLE_TOOL)} || true",
          30,
      ),
      "boundary_related_files": (
          "find /home/intel/intel-box-run /home/intel/llama-cpp "
          "-maxdepth 6 -type f "
          "\\( -iname '*boundary*' -o -iname '*activation*' -o -iname '*tensor*oracle*' \\) "
          "2>/dev/null | head -200",
          45,
      ),
  }
  results: dict[str, dict[str, Any]] = {}
  for name, (script, timeout_s) in commands.items():
    result = run_target(args.host, script, timeout_s=timeout_s)
    results[name] = {
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
    }
    (raw_dir / f"{name}.stdout").write_text(result["stdout"], encoding="utf-8")
    (raw_dir / f"{name}.stderr").write_text(result["stderr"], encoding="utf-8")

  inventory_values = parse_key_values((raw_dir / "inventory.stdout").read_text(encoding="utf-8"))
  server_help = (raw_dir / "llama_server_help.stdout").read_text(encoding="utf-8")
  tokenizer_help = (raw_dir / "llama_tokenize_help.stdout").read_text(encoding="utf-8")
  prior_tool = (raw_dir / "prior_oracle_tool_head.stdout").read_text(encoding="utf-8")
  boundary_files = [
      line.strip()
      for line in (raw_dir / "boundary_related_files.stdout").read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  openvino_stdout = (raw_dir / "openvino_import.stdout").read_text(encoding="utf-8")
  prior_raw_count_text = inventory_values.get("prior_completion_probability_raw_count", "0")
  try:
    prior_raw_count = int(prior_raw_count_text)
  except ValueError:
    prior_raw_count = 0

  inventory = {
      "hostname": inventory_values.get("hostname"),
      "intel_env_present": bool_value(inventory_values, "intel_env_present"),
      "llama_cli_present": bool_value(inventory_values, "llama_cli_present"),
      "llama_server_present": bool_value(inventory_values, "llama_server_present"),
      "llama_tokenize_present": bool_value(inventory_values, "llama_tokenize_present"),
      "model_present": bool_value(inventory_values, "model_present"),
      "model_size_bytes": inventory_values.get("model_size_bytes"),
      "model_size_matches_contract": inventory_values.get("model_size_bytes") == "21166755168",
      "openvino_bench_present": bool_value(inventory_values, "openvino_bench_present"),
      "openvino_genai_import": "openvino_genai_import=true" in openvino_stdout,
      "openvino_model_present": bool_value(inventory_values, "openvino_model_present"),
      "prior_completion_probability_raw_count": prior_raw_count,
      "prior_oracle_raw_present": bool_value(inventory_values, "prior_oracle_raw_present"),
      "prior_oracle_root_present": bool_value(inventory_values, "prior_oracle_root_present"),
      "prior_oracle_tool_present": bool_value(inventory_values, "prior_oracle_tool_present"),
  }
  server_capabilities = {
      "completion_probabilities_seen_in_prior_tool": "completion_probabilities" in prior_tool,
      "help_mentions_logprobs": contains_any(server_help, ["logprobs", "log-prob"]),
      "help_mentions_slots_or_cache": contains_any(server_help, ["slot", "cache"]),
      "help_mentions_ctx_size": contains_any(server_help, ["ctx-size", "context size"]),
      "help_mentions_seed": contains_any(server_help, ["seed"]),
      "help_mentions_temperature": contains_any(server_help, ["temp"]),
  }
  tokenizer_capabilities = {
      "help_mentions_model": "--model" in tokenizer_help or "-m" in tokenizer_help,
      "help_mentions_prompt": contains_any(tokenizer_help, ["prompt", "text"]),
      "tokenizer_executable": inventory["llama_tokenize_present"],
  }
  distribution_route_ready = (
      inventory["model_present"]
      and inventory["llama_server_present"]
      and inventory["llama_tokenize_present"]
      and inventory["prior_oracle_tool_present"]
      and server_capabilities["completion_probabilities_seen_in_prior_tool"]
  )
  stock_boundary_route_ready = bool(boundary_files)
  payload = {
      "created_at": created_at,
      "evidence": {
          "model_path": MODEL_PATH,
          "model_sha256": MODEL_SHA256,
          "raw_dir": rel(raw_dir),
      },
      "host": args.host,
      "oracle_runtime_routes": {
          "teacher_forced_distribution": {
              "candidate_route_present": distribution_route_ready,
              "limitations": [
                  "preflight did not start llama-server or capture new logprob rows",
                  "prior raw response files are not required because the prior oracle tool can be rerun",
                  "full 26-row ladder still needs bounded capture and replay checks",
              ],
              "route_status": (
                  "candidate_prior_llama_oracle_tool_completion_probabilities_route"
                  if distribution_route_ready
                  else "missing_distribution_capture_route"
              ),
              "source_runtime": "llama.cpp CPU server",
          },
          "tokenization": {
              "candidate_route_present": inventory["llama_tokenize_present"],
              "route_status": (
                  "candidate_llama_tokenize_route"
                  if inventory["llama_tokenize_present"]
                  else "missing_tokenizer_route"
              ),
          },
          "per_boundary_tensors": {
              "candidate_route_present": stock_boundary_route_ready,
              "known_boundary_related_files": boundary_files[:50],
              "limitations": [
                  "stock llama.cpp/OpenVINO paths do not expose the required 17 boundary tensors as bundle JSONL",
                  "requires instrumentation or a reference forward path that can dump each queued boundary input/output",
              ],
              "route_status": (
                  "candidate_boundary_instrumentation_files_found"
                  if stock_boundary_route_ready
                  else "missing_stock_boundary_tensor_capture_route"
              ),
          },
      },
      "r0_oracle_gate_closed": False,
      "raw_command_status": results,
      "schema_version": SCHEMA_VERSION,
      "server_capabilities": server_capabilities,
      "target_inventory": inventory,
      "tokenizer_capabilities": tokenizer_capabilities,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "target_reachable",
          "pass": results["inventory"]["returncode"] == 0 and inventory.get("hostname") is not None,
      },
      {
          "name": "locked_model_present",
          "pass": inventory["model_present"] and inventory["model_size_matches_contract"],
      },
      {
          "name": "llama_server_distribution_candidate_present",
          "pass": distribution_route_ready,
      },
      {
          "name": "llama_tokenizer_present",
          "pass": inventory["llama_tokenize_present"],
      },
      {
          "name": "openvino_import_available_for_sanity",
          "pass": inventory["openvino_genai_import"] and inventory["openvino_model_present"],
      },
      {
          "name": "stock_boundary_tensor_route_not_ready",
          "pass": stock_boundary_route_ready is False,
          "boundary_related_file_count": len(boundary_files),
      },
      {
          "name": "oracle_gate_remains_open",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-runtime-preflight.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "preflight.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_runtime_preflight",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("prior_completion_probability_raw_count", prior_raw_count),
        ("distribution_candidate_route_present", distribution_route_ready),
        ("stock_boundary_route_ready", stock_boundary_route_ready),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_oracle_runtime_preflight",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle runtime preflight output: {out_dir}")


if __name__ == "__main__":
  main()
