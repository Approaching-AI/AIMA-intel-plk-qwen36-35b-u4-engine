#!/usr/bin/env python3
"""Gate the fixed 128k fused-GQA FP16-KV decode component on PTL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCALAR_SOURCE = ROOT / "engine/gpu/opencl/fused_gqa_fp16_kv_decode.cl"
XMX_SOURCE = ROOT / "engine/gpu/opencl/xmx_gqa_fp16_kv_flash_decode.cl"
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
TARGET = "iq36-fused-gqa-fp16-kv-decode"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--mode", choices=("scalar", "xmx"), default="scalar")
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


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  return "\n".join([
      "# Fused GQA FP16-KV decode component gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- context: `{result.get('context_tokens')}`",
      f"- fixed chunk: `{result.get('chunk_tokens')}`",
      f"- output cosine / relL2: `{result.get('output_cosine')} / {result.get('output_relative_l2')}`",
      f"- repeat partial / reduce / total: `{result.get('repeat', {}).get('partial_ms')} / {result.get('repeat', {}).get('reduce_ms')} / {result.get('repeat', {}).get('total_ms')} ms`",
      f"- confirm partial / reduce / total: `{result.get('confirm', {}).get('partial_ms')} / {result.get('confirm', {}).get('reduce_ms')} / {result.get('confirm', {}).get('total_ms')} ms`",
      f"- paired spread: `{result.get('spread')}`",
      "- integration/product speed admitted: `false / false`",
      "",
      "This gate admits only the fixed 128k component when numeric, timing, and",
      "noise checks pass. It does not measure a token loop or native prefill.",
      "",
  ])


def main() -> int:
  args = parse_args()
  source = XMX_SOURCE if args.mode == "xmx" else SCALAR_SOURCE
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  git = git_state(out_dir)
  source_text = source.read_text()
  source_checks = {
      "fixed_chunk_256": "#define IQ36_CHUNK_TOKENS 256U" in source_text,
      "fixed_gqa_8": "#define IQ36_GQA_GROUP 8U" in source_text,
      "native_runtime_only": "openvino" not in source_text.lower(),
  }
  if args.mode == "xmx":
    source_checks.update({
        "fixed_token_tile_16": "#define IQ36_TOKEN_TILE 16U" in source_text,
        "subgroup_16": "intel_reqd_sub_group_size(16)" in source_text,
        "dpas_score_and_value": source_text.count(
            "intel_sub_group_f16_f16_matrix_mad_k16") == 2,
        "dpas_ready_k": "iq36_pack_k_dpas16" in source_text,
        "bounded_partials": "iq36_xmx_gqa_partial_reduce" in source_text,
    })
  else:
    source_checks.update({
        "fp16_kv": "__global const half* k_history" in source_text
        and "__global const half* v_history" in source_text,
        "subgroup_32": "intel_reqd_sub_group_size(32)" in source_text,
        "bounded_partials": "iq36_fused_gqa_partial_reduce" in source_text,
    })
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
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"{shlex.quote(str(executable))} {shlex.quote(str(source))}"
      + (" xmx" if args.mode == "xmx" else "")
  )
  component = (
      run(["bash", "-lc", shell_command], args.timeout_s)
      if build_ok else subprocess.CompletedProcess(
          ["bash", "-lc", shell_command], 1, "", "build failed")
  )
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

  numeric_pass = bool(
      result.get("finite") is True
      and float(result.get("output_cosine", 0.0)) >= 0.999
      and float(result.get("output_relative_l2", 1.0)) <= 0.002
      and result.get("numeric_pass") is True)
  repeat = result.get("repeat", {})
  confirm = result.get("confirm", {})
  timing_pass = bool(
      float(repeat.get("total_ms", 1e9)) <= 2.825
      and float(confirm.get("total_ms", 1e9)) <= 2.825
      and float(result.get("spread", 1.0)) <= 0.005
      and result.get("timing_pass") is True)
  fixed_shape = bool(
      result.get("context_tokens") == 131072
      and result.get("chunk_tokens") == 256
      and result.get("head_dim") == 256
      and result.get("q_head_count") == 16
      and result.get("kv_head_count") == 2
      and result.get("gqa_group") == 8
      and result.get("kv_dtype") == "fp16"
      and result.get("algorithm") == (
          "xmx_gqa_flash" if args.mode == "xmx" else "scalar_gqa_fused")
      and result.get("subgroup_size") == (16 if args.mode == "xmx" else 32)
      and result.get("token_tile") == (16 if args.mode == "xmx" else 0))
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "fixed_source_contract", "pass": all(source_checks.values()),
       "details": source_checks},
      {"name": "component_build", "pass": build_ok},
      {"name": "component_execution", "pass": component.returncode == 0},
      {"name": "fixed_128k_shape", "pass": fixed_shape},
      {"name": "component_numeric", "pass": numeric_pass},
      {"name": "component_repeat_confirm_timing", "pass": timing_pass},
      {"name": "component_self_gate", "pass": result.get("required_checks_passed") is True},
  ]
  required = all(bool(check["pass"]) for check in checks)
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": required,
      "result": result,
      "route_label": "component_promoted" if required else "rejected",
      "mode": args.mode,
      "schema_version": "intel-qwen36-fused-gqa-fp16-kv-decode-gate-v0",
      "source": {"path": str(source.relative_to(ROOT)), "sha256": sha256(source)},
      "speedup_claims_allowed": False,
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"], "git": git,
      "required_checks_passed": required, "route_label": payload["route_label"],
      "schema_version": payload["schema_version"],
      "source": payload["source"],
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": payload["workstream"],
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks, "numeric": {
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
      "applicable": True, "paired_spread": result.get("spread"),
      "paired_spread_max": 0.005,
      "required_checks_passed": float(result.get("spread", 1.0)) <= 0.005,
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
