#!/usr/bin/env python3
"""Preflight a llama.cpp same-host denominator route on the target."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_BENCH = "/home/intel/llama-cpp/llama-b9518/llama-bench"
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
      help="Output directory. Defaults to output/r0-llama-denominator-preflight-<UTC>.",
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
  parsed: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      parsed[key.strip()] = value.strip()
  return parsed


def parse_devices(stdout: str) -> dict[str, Any]:
  vulkan_visible = "Vulkan0:" in stdout
  device_lines = [
      line.strip()
      for line in stdout.splitlines()
      if line.strip().startswith("Vulkan") or "Vulkan0:" in line
  ]
  memory_mib = None
  free_mib = None
  match = re.search(r"\((\d+) MiB,\s*(\d+) MiB free\)", stdout)
  if match:
    memory_mib = int(match.group(1))
    free_mib = int(match.group(2))
  return {
      "device_lines": device_lines,
      "free_mib": free_mib,
      "memory_mib": memory_mib,
      "vulkan0_visible": vulkan_visible,
  }


def summarize_help(help_text: str) -> dict[str, Any]:
  required_flags = [
      "--model",
      "--n-prompt",
      "--n-gen",
      "--output",
      "--no-warmup",
      "--n-gpu-layers",
      "--device",
      "--cache-type-k",
      "--cache-type-v",
  ]
  return {
      "required_flags": required_flags,
      "required_flags_present": {
          flag: flag in help_text for flag in required_flags
      },
  }


def build_summary(payload: dict[str, Any]) -> str:
  checks = payload["checks"]
  devices = payload["device_modes"]
  lines = [
      "# R0 llama.cpp Denominator Preflight",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- llama-bench: `{LLAMA_BENCH}`",
      f"- model: `{MODEL_PATH}`",
      f"- binary present: `{str(checks['llama_bench_present']).lower()}`",
      f"- model present: `{str(checks['model_present']).lower()}`",
      f"- Vulkan visible without env: `{str(devices['plain']['vulkan0_visible']).lower()}`",
      f"- Vulkan visible with `INTEL_FORCE_PROBE=b080`: `{str(devices['force_probe']['vulkan0_visible']).lower()}`",
      f"- Vulkan visible with Intel env: `{str(devices['intel_env']['vulkan0_visible']).lower()}`",
      f"- force probe required: `{str(checks['force_probe_required']).lower()}`",
      f"- candidate ready for smoke run: `{str(payload['candidate_ready_for_smoke']).lower()}`",
      f"- R0 denominator gate closed: `{str(payload['r0_denominator_gate_closed']).lower()}`",
      "",
      "This is a preflight only. It proves that llama-bench can see a PTL",
      "Vulkan device under the required environment, but it does not produce a",
      "262144 throughput denominator metric.",
      "",
      "Suggested smoke command:",
      "",
      "```bash",
      payload["suggested_commands"]["smoke"],
      "```",
      "",
      "Suggested 262144 denominator command after smoke passes:",
      "",
      "```bash",
      payload["suggested_commands"]["bucket_262144"],
      "```",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or (ROOT / f"output/r0-llama-denominator-preflight-{stamp}")
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  inventory_script = "\n".join([
      "set -u",
      f"printf 'llama_bench_present='; test -x {shlex.quote(LLAMA_BENCH)} && echo true || echo false",
      f"printf 'llama_bench_size_bytes='; stat -c %s {shlex.quote(LLAMA_BENCH)} 2>/dev/null || echo missing",
      f"printf 'model_present='; test -f {shlex.quote(MODEL_PATH)} && echo true || echo false",
      f"printf 'model_size_bytes='; stat -c %s {shlex.quote(MODEL_PATH)} 2>/dev/null || echo missing",
      f"printf 'intel_env_present='; test -f {shlex.quote(INTEL_ENV)} && echo true || echo false",
      "printf 'hostname='; hostname",
  ])
  inventory = run_target(args.host, inventory_script, timeout_s=30)
  raw_inventory_path = raw_dir / "inventory.stdout"
  raw_inventory_path.write_text(inventory["stdout"], encoding="utf-8")
  (raw_dir / "inventory.stderr").write_text(inventory["stderr"], encoding="utf-8")

  help_result = run_target(args.host, f"{shlex.quote(LLAMA_BENCH)} --help 2>&1", timeout_s=30)
  (raw_dir / "llama-bench-help.stdout").write_text(help_result["stdout"], encoding="utf-8")
  (raw_dir / "llama-bench-help.stderr").write_text(help_result["stderr"], encoding="utf-8")

  device_scripts = {
      "plain": f"{shlex.quote(LLAMA_BENCH)} --list-devices 2>&1",
      "force_probe": (
          "export INTEL_FORCE_PROBE=b080; "
          f"{shlex.quote(LLAMA_BENCH)} --list-devices 2>&1"
      ),
      "intel_env": (
          f"source {shlex.quote(INTEL_ENV)} >/tmp/iq36-llama-env.log 2>&1 || true; "
          f"{shlex.quote(LLAMA_BENCH)} --list-devices 2>&1"
      ),
      "intel_env_force_probe": (
          f"source {shlex.quote(INTEL_ENV)} >/tmp/iq36-llama-env.log 2>&1 || true; "
          "export INTEL_FORCE_PROBE=b080; "
          f"{shlex.quote(LLAMA_BENCH)} --list-devices 2>&1"
      ),
  }
  device_results: dict[str, dict[str, Any]] = {}
  for mode, script in device_scripts.items():
    result = run_target(args.host, script, timeout_s=30)
    (raw_dir / f"devices-{mode}.stdout").write_text(result["stdout"], encoding="utf-8")
    (raw_dir / f"devices-{mode}.stderr").write_text(result["stderr"], encoding="utf-8")
    parsed = parse_devices(result["stdout"] + "\n" + result["stderr"])
    device_results[mode] = {
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        **parsed,
    }

  inventory_values = parse_key_values(inventory["stdout"])
  help_summary = summarize_help(help_result["stdout"] + "\n" + help_result["stderr"])
  flags_ok = all(help_summary["required_flags_present"].values())
  llama_bench_present = inventory_values.get("llama_bench_present") == "true"
  model_present = inventory_values.get("model_present") == "true"
  model_size_matches = inventory_values.get("model_size_bytes") == "21166755168"
  intel_env_present = inventory_values.get("intel_env_present") == "true"
  vulkan_without_env = device_results["plain"]["vulkan0_visible"]
  vulkan_with_force = device_results["force_probe"]["vulkan0_visible"]
  vulkan_with_env = device_results["intel_env"]["vulkan0_visible"]
  candidate_ready = (
      llama_bench_present
      and model_present
      and model_size_matches
      and intel_env_present
      and flags_ok
      and (vulkan_with_force or vulkan_with_env)
  )

  env_prefix = (
      f"source {shlex.quote(INTEL_ENV)} >/tmp/iq36-llama-env.log 2>&1 || true; "
      "export INTEL_FORCE_PROBE=b080; "
  )
  base_bench = (
      f"{shlex.quote(LLAMA_BENCH)} "
      f"-m {shlex.quote(MODEL_PATH)} "
      "-dev Vulkan0 -ngl -1 -ctk f16 -ctv f16 "
      "-b 512 -ub 512 -t 16 -r 1 --no-warmup -o json"
  )
  payload = {
      "candidate_ready_for_smoke": candidate_ready,
      "checks": {
          "force_probe_required": (not vulkan_without_env) and vulkan_with_force,
          "help_flags_supported": flags_ok,
          "intel_env_present": intel_env_present,
          "llama_bench_present": llama_bench_present,
          "model_present": model_present,
          "model_size_matches_contract": model_size_matches,
      },
      "created_at": created_at,
      "device_modes": device_results,
      "evidence": {
          "help_stdout": str((raw_dir / "llama-bench-help.stdout").relative_to(ROOT)),
          "inventory_stdout": str(raw_inventory_path.relative_to(ROOT)),
          "plain_devices_stdout": str((raw_dir / "devices-plain.stdout").relative_to(ROOT)),
          "force_probe_devices_stdout": str((raw_dir / "devices-force_probe.stdout").relative_to(ROOT)),
          "intel_env_devices_stdout": str((raw_dir / "devices-intel_env.stdout").relative_to(ROOT)),
      },
      "help": help_summary,
      "host": args.host,
      "inventory": inventory_values,
      "llama": {
          "binary": LLAMA_BENCH,
          "intel_env": INTEL_ENV,
          "model_path": MODEL_PATH,
          "model_sha256": MODEL_SHA256,
      },
      "next_required_output": "local-background llama-bench smoke row before any 262144 denominator run",
      "r0_denominator_gate_closed": False,
      "schema_version": "intel-qwen36-r0-llama-denominator-preflight-v0",
      "suggested_commands": {
          "smoke": env_prefix + base_bench + " -p 128 -n 1",
          "bucket_262144": env_prefix + base_bench + " -p 262144 -n 1",
      },
      "workstream": WORKSTREAM,
  }
  checks = [
      {"name": "llama_bench_present", "pass": llama_bench_present},
      {"name": "model_present", "pass": model_present},
      {"name": "model_size_matches_contract", "pass": model_size_matches},
      {"name": "intel_env_present", "pass": intel_env_present},
      {"name": "help_flags_supported", "pass": flags_ok},
      {
          "name": "vulkan_device_visible_with_target_env",
          "pass": vulkan_with_force or vulkan_with_env,
          "force_probe_visible": vulkan_with_force,
          "intel_env_visible": vulkan_with_env,
      },
      {"name": "preflight_does_not_close_denominator", "pass": True},
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-llama-denominator-preflight.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "preflight.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_llama_denominator_preflight",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": payload["schema_version"],
      "workstream": WORKSTREAM,
  })
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "metric": "candidate_ready_for_smoke",
        "phase": "llama_denominator_preflight",
        "value": candidate_ready,
    }, sort_keys=True) + "\n")
    fh.write(json.dumps({
        "metric": "vulkan_visible_with_force_probe",
        "phase": "llama_denominator_preflight",
        "value": vulkan_with_force,
    }, sort_keys=True) + "\n")
  print(f"llama denominator preflight output: {out_dir}")


if __name__ == "__main__":
  main()
