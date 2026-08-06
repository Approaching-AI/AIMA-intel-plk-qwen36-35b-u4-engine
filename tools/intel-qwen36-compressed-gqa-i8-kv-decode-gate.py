#!/usr/bin/env python3
"""Gate the fixed 128k block32-INT8-KV GQA decode component on PTL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine/gpu/opencl/compressed_gqa_i8_kv_decode.cl"
RUNNER_SOURCE = ROOT / "engine/tools/compressed_gqa_i8_kv_decode.cpp"
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
TARGET = "iq36-compressed-gqa-i8-kv-decode"


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
      "bios_version": ["bash", "-lc",
                       "head -n 1 /sys/class/dmi/id/bios_version"],
      "opencl": ["bash", "-lc",
                 f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
                 "clinfo -l"],
  }
  result: dict[str, Any] = {}
  for name, command in commands.items():
    completed = run(command, 60)
    result[name] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
  return result


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  return "\n".join([
      "# Block32 INT8-KV GQA decode component gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- context / chunk: `{result.get('context_tokens')} / {result.get('chunk_tokens')}`",
      f"- KV representation: `{result.get('kv_dtype')}`",
      f"- output cosine / relL2: `{result.get('output_cosine')} / {result.get('output_relative_l2')}`",
      f"- repeat quantize / partial / reduce / total: `{result.get('repeat', {}).get('quantize_ms')} / {result.get('repeat', {}).get('partial_ms')} / {result.get('repeat', {}).get('reduce_ms')} / {result.get('repeat', {}).get('total_ms')} ms`",
      f"- confirm quantize / partial / reduce / total: `{result.get('confirm', {}).get('quantize_ms')} / {result.get('confirm', {}).get('partial_ms')} / {result.get('confirm', {}).get('reduce_ms')} / {result.get('confirm', {}).get('total_ms')} ms`",
      f"- median / one-sided 95% upper bound / cap: `{payload.get('performance_inference', {}).get('point_estimate_ms')} / {payload.get('performance_inference', {}).get('upper_confidence_bound_ms')} / {payload.get('performance_inference', {}).get('cap_ms')} ms`",
      f"- paired spread (diagnostic only): `{result.get('spread')}`",
      f"- robust CV / environment class: `{payload.get('performance_inference', {}).get('dispersion', {}).get('robust_cv')} / {payload.get('performance_inference', {}).get('dispersion', {}).get('classification')}`",
      "- integration/product speed admitted: `false / false`",
      "",
      "This synthetic component gate includes current-token K/V quantization,",
      "attention, and reduction device events. It admits neither a token-loop",
      "speedup nor any output-512 product row.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  git = git_state(out_dir)
  source_text = SOURCE.read_text()
  runner_text = RUNNER_SOURCE.read_text()
  source_checks = {
      "fixed_context_131072": "constexpr cl_uint kContextTokens = 131072;"
      in runner_text,
      "fixed_chunk_256": "#define IQ36_CHUNK_TOKENS 256U" in source_text,
      "fixed_quant_group_32": "#define IQ36_QUANT_GROUP 32U" in source_text,
      "fixed_gqa_8": "#define IQ36_GQA_GROUP 8U" in source_text,
      "subgroup_32": "intel_reqd_sub_group_size(32)" in source_text,
      "signed_i8_kv": "__global const char* k_history" in source_text
      and "__global const char* v_history" in source_text,
      "fp16_scales": "__global const half* k_scales" in source_text
      and "__global const half* v_scales" in source_text,
      "symmetric_rne_clamp": "convert_int_rte(value / scale)" in source_text
      and "-127, 127" in source_text,
      "current_token_quantizer": "iq36_quantize_current_i8_block32"
      in source_text,
      "bounded_partials": "iq36_compressed_gqa_partial_reduce" in source_text,
      "quantization_timed": "result.quantize_ms = EventMs(quantize_event)"
      in runner_text,
      "native_runtime_only": "openvino" not in source_text.lower()
      and "openvino" not in runner_text.lower(),
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
  build_ok = (
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"{shlex.quote(str(executable))} {shlex.quote(str(SOURCE))}")
  component = (
      run(["bash", "-lc", shell_command], args.timeout_s)
      if build_ok else subprocess.CompletedProcess(
          ["bash", "-lc", shell_command], 1, "", "build failed"))
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
      "command": ["bash", "-lc", shell_command],
      "returncode": component.returncode,
  })
  write_json(raw_dir / "environment.json", environment())

  numeric_pass = bool(
      result.get("finite") is True
      and float(result.get("output_cosine", 0.0)) >= 0.999
      and float(result.get("output_relative_l2", 1.0)) <= 0.002
      and result.get("numeric_pass") is True)
  repeat = result.get("repeat", {})
  confirm = result.get("confirm", {})
  total_samples = [
      float(sample.get("total_ms", 1e9))
      for sample in result.get("repeat_samples", [])
      + result.get("confirm_samples", [])]
  performance_inference = latency_cap_inference(
      total_samples, cap=2.825, min_samples=20)
  timing_pass = performance_inference["rate_pass"] is True
  fixed_shape = bool(
      result.get("context_tokens") == 131072
      and result.get("chunk_tokens") == 256
      and result.get("head_dim") == 256
      and result.get("q_head_count") == 16
      and result.get("kv_head_count") == 2
      and result.get("gqa_group") == 8
      and result.get("quant_group") == 32
      and result.get("kv_dtype") == "int8_block32_fp16_scale"
      and result.get("scale_dtype") == "fp16"
      and result.get("algorithm") == "int8_block32_gqa_fused"
      and result.get("subgroup_size") == 32
      and result.get("quantization_included") is True
      and result.get("quantization_rounding") == "rne_clamp_-127_127")
  distribution_shape = bool(
      len(result.get("repeat_samples", [])) == 10
      and len(result.get("confirm_samples", [])) == 10
      and all(
          set(sample) == {"quantize_ms", "partial_ms", "reduce_ms", "total_ms"}
          for sample in result.get("repeat_samples", [])
          + result.get("confirm_samples", [])))
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "fixed_source_contract", "pass": all(source_checks.values()),
       "details": source_checks},
      {"name": "component_build", "pass": build_ok},
      {"name": "component_execution", "pass": component.returncode == 0},
      {"name": "fixed_128k_shape", "pass": fixed_shape},
      {"name": "twenty_sample_timing_distribution",
       "pass": distribution_shape},
      {"name": "component_numeric", "pass": numeric_pass},
      {"name": "component_repeat_confirm_timing", "pass": timing_pass},
      {"name": "component_self_gate",
       "pass": result.get("required_checks_passed") is True},
  ]
  required = all(bool(check["pass"]) for check in checks)
  sources = [
      {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
      {"path": str(RUNNER_SOURCE.relative_to(ROOT)),
       "sha256": sha256(RUNNER_SOURCE)},
  ]
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": required,
      "result": result,
      "performance_inference": performance_inference,
      "route_label": "component_promoted" if required else "rejected",
      "schema_version": "intel-qwen36-compressed-gqa-i8-kv-decode-gate-v1",
      "sources": sources,
      "speedup_claims_allowed": False,
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"], "git": git,
      "required_checks_passed": required, "route_label": payload["route_label"],
      "schema_version": payload["schema_version"], "sources": sources,
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
          **result.get(lane, {}), "route_label": payload["route_label"],
      }, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "applicable": True,
      "paired_spread": result.get("spread"),
      "paired_spread_role": "diagnostic_only",
      "dispersion": performance_inference["dispersion"],
      "required_checks_passed": True,
  })
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
      "repeat_ms": repeat.get("total_ms"),
      "confirm_ms": confirm.get("total_ms"),
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
