#!/usr/bin/env python3
"""Gate the fixed exact-128k CPU AVX2/F16C FP16-KV GQA component."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine/tools/cpu_avx2_fp16_gqa_decode.cpp"
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
TARGET = "iq36-cpu-avx2-fp16-gqa-decode"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def environment() -> dict[str, Any]:
  commands = {
      "hostname": ["hostname"],
      "kernel": ["uname", "-a"],
      "cpu": ["lscpu"],
      "bios_version": ["bash", "-lc",
                       "head -n 1 /sys/class/dmi/id/bios_version"],
  }
  result: dict[str, Any] = {}
  for name, command in commands.items():
    completed = run(command, 60)
    result[name] = {
        "command": command, "returncode": completed.returncode,
        "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
    }
  return result


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  return "\n".join([
      "# CPU AVX2/F16C FP16-KV GQA component gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- context / chunk / workers: `{result.get('context_tokens')} / {result.get('chunk_tokens')} / {result.get('worker_count')}`",
      f"- output cosine / relL2: `{result.get('output_cosine')} / {result.get('output_relative_l2')}`",
      f"- repeat / confirm wall medians: `{result.get('repeat_ms')} / {result.get('confirm_ms')} ms`",
      f"- paired spread: `{result.get('spread')}`",
      "- integration/product speed admitted: `false / false`",
      "",
      "Wall timing includes current-token conversion and persistent-worker",
      "synchronization. This component gate admits no product speedup row.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  git = git_state(out_dir)
  source_text = SOURCE.read_text()
  source_checks = {
      "fixed_context_131072": "kContextTokens = 131072" in source_text,
      "fixed_chunk_256": "kChunkTokens = 256" in source_text,
      "fixed_workers_16": "kWorkers = 16" in source_text,
      "avx2_f16c_fma": 'target("avx2,f16c,fma")' in source_text,
      "pinned_affinity": "pthread_setaffinity_np" in source_text,
      "fp16_kv": "std::vector<std::uint16_t> k_history" in source_text
      and "std::vector<std::uint16_t> v_history" in source_text,
      "persistent_workers": "workers_.emplace_back" in source_text,
      "timed_conversion_and_sync":
      '"timed_current_token_conversion\\":true' in source_text
      and '"timed_worker_synchronization\\":true' in source_text,
      "native_runtime_only": "openvino" not in source_text.lower(),
  }
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j8"]
  build = run(build_command, 600)
  executable = BUILD_DIR / TARGET
  build_ok = configure.returncode == 0 and build.returncode == 0 and executable.is_file()
  component_command = [str(executable)]
  component = (
      run(component_command, args.timeout_s)
      if build_ok else subprocess.CompletedProcess(
          component_command, 1, "", "build failed"))
  result = parse_last_json(component.stdout)
  write_json(raw_dir / "build.json", {
      "configure": {"command": configure_command,
                    "returncode": configure.returncode,
                    "stdout": configure.stdout, "stderr": configure.stderr},
      "build": {"command": build_command, "returncode": build.returncode,
                "stdout": build.stdout, "stderr": build.stderr},
  })
  (raw_dir / "component.stdout").write_text(component.stdout)
  (raw_dir / "component.stderr").write_text(component.stderr)
  write_json(raw_dir / "component-command.json", {
      "command": component_command, "returncode": component.returncode,
  })
  write_json(raw_dir / "environment.json", environment())

  numeric_pass = bool(
      result.get("finite") is True
      and float(result.get("output_cosine", 0.0)) >= 0.999
      and float(result.get("output_relative_l2", 1.0)) <= 0.002
      and result.get("numeric_pass") is True)
  timing_pass = bool(
      float(result.get("repeat_ms", 1e9)) <= 2.825
      and float(result.get("confirm_ms", 1e9)) <= 2.825
      and float(result.get("spread", 1.0)) <= 0.005
      and result.get("timing_pass") is True)
  fixed_shape = bool(
      result.get("context_tokens") == 131072
      and result.get("chunk_tokens") == 256
      and result.get("head_dim") == 256
      and result.get("q_head_count") == 16
      and result.get("kv_head_count") == 2
      and result.get("gqa_group") == 8
      and result.get("worker_count") == 16
      and result.get("kv_dtype") == "fp16"
      and result.get("isa") == "avx2_f16c_fma"
      and result.get("algorithm") == "cpu_avx2_fp16_gqa_chunked"
      and result.get("timed_current_token_conversion") is True
      and result.get("timed_worker_synchronization") is True)
  distributions = bool(
      len(result.get("repeat_samples_ms", [])) == 7
      and len(result.get("confirm_samples_ms", [])) == 7)
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "fixed_source_contract", "pass": all(source_checks.values()),
       "details": source_checks},
      {"name": "component_build", "pass": build_ok},
      {"name": "component_execution", "pass": component.returncode == 0},
      {"name": "fixed_128k_shape", "pass": fixed_shape},
      {"name": "pinned_worker_affinity", "pass": result.get("affinity_pass") is True},
      {"name": "paired_seven_sample_distributions", "pass": distributions},
      {"name": "component_numeric", "pass": numeric_pass},
      {"name": "component_repeat_confirm_timing", "pass": timing_pass},
      {"name": "component_self_gate",
       "pass": result.get("required_checks_passed") is True},
  ]
  required = all(bool(check["pass"]) for check in checks)
  source = {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)}
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": required,
      "result": result,
      "route_label": "component_promoted" if required else "rejected",
      "schema_version": "intel-qwen36-cpu-avx2-fp16-gqa-decode-gate-v0",
      "source": source,
      "speedup_claims_allowed": False,
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"], "git": git,
      "required_checks_passed": required, "route_label": payload["route_label"],
      "schema_version": payload["schema_version"], "source": source,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": payload["workstream"],
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "numeric": {
          "cosine": result.get("output_cosine"),
          "relative_l2": result.get("output_relative_l2"),
          "rmse": result.get("output_rmse"), "max_abs": result.get("max_abs")},
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w") as handle:
    for lane in ("repeat", "confirm"):
      handle.write(json.dumps({
          "context_tokens": result.get("context_tokens"), "lane": lane,
          "total_ms": result.get(f"{lane}_ms"),
          "samples_ms": result.get(f"{lane}_samples_ms"),
          "route_label": payload["route_label"],
      }, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "applicable": True, "paired_spread": result.get("spread"),
      "paired_spread_max": 0.005,
      "required_checks_passed": float(result.get("spread", 1.0)) <= 0.005,
  })
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
      "repeat_ms": result.get("repeat_ms"),
      "confirm_ms": result.get("confirm_ms"),
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
