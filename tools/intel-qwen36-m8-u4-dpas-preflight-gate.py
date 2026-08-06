#!/usr/bin/env python3
"""Prove PTL target codegen for the SIMD16 M8 x U4 DPAS overload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-m8-u4-dpas-preflight-gate-v0"
KERNEL = ROOT / "engine/gpu/opencl/m8_u4_dpas_preflight.cl"
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/m8-u4-dpas-preflight-gate-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": process.returncode,
        "stderr": process.stderr,
        "stdout": process.stdout,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }


def shell_run(
    command: list[str], env_script: Path, timeout_s: int, cwd: Path,
) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s, cwd)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  digest.update(path.read_bytes())
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*parts: str) -> str:
    result = subprocess.run(
        ["git", *parts], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  ocloc_dir = raw_dir / "ocloc"
  disasm_dir = raw_dir / "disasm"
  ocloc_dir.mkdir(parents=True, exist_ok=False)
  disasm_dir.mkdir()
  if not KERNEL.is_file() or not args.env_script.is_file():
    raise SystemExit("kernel or environment activation script is missing")

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  compile_result = shell_run([
      "ocloc", "-file", str(KERNEL), "-device", "0xb080",
      "-options", "-cl-std=CL2.0",
  ], args.env_script, args.timeout_s, ocloc_dir)
  write_json(raw_dir / "compile.json", compile_result)
  native_bins = sorted(ocloc_dir.glob("*.bin"))
  disasm_result = (
      run([
          "ocloc", "disasm", "-file", str(native_bins[0]), "-dump",
          str(disasm_dir), "-device", "0xb080",
      ], args.timeout_s, ROOT)
      if compile_result["returncode"] == 0 and native_bins else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "compile failed",
       "timed_out": False}
  )
  write_json(raw_dir / "disasm.json", disasm_result)
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob("*.asm"))
  ze_info = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob(".ze_info"))
  compile_text = str(compile_result["stdout"]) + str(compile_result["stderr"])
  checks = [
      {"name": "ocloc_compile_passed", "pass": compile_result["returncode"] == 0},
      {"name": "exact_ptl_device_selected", "pass": "ptl-h-a0" in compile_text},
      {"name": "ocloc_disassembly_passed", "pass": disasm_result["returncode"] == 0},
      {"name": "m8_dpas_shape_present", "pass": "dpas.8x8" in assembly.lower()},
      {"name": "u4_source_precision_present", "pass": ":u4" in assembly.lower()},
      {"name": "ze_info_has_dpas", "pass": "has_dpas:true" in ze_info.replace(" ", "")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)
  disposition = (
      "admit_one_real_shape_m8_u4_dpas_component"
      if required_checks_passed else
      "reject_m8_u4_dpas_route_on_target_codegen")
  result = {
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "git": git_state(),
      "kernel": {
          "path": str(KERNEL.relative_to(ROOT)),
          "sha256": sha256_file(KERNEL),
      },
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_device_id": "0xb080",
      "target_platform": "ptl-h-a0",
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "applicable": False,
      "checks": checks,
      "reason": "target-codegen preflight; real component correctness follows",
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "target-codegen preflight",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "metric": "required_checks_passed", "value": required_checks_passed,
    }) + "\n")
  (out_dir / "summary.md").write_text("\n".join([
      "# SIMD16 M8 x U4 DPAS target-codegen preflight",
      "",
      "- target: `0xb080` / `ptl-h-a0`",
      f"- M8 DPAS ISA present: `{str(checks[3]['pass']).lower()}`",
      f"- U4 source precision present: `{str(checks[4]['pass']).lower()}`",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "This only authorizes one real-shape component under the existing",
      "5479.754 us cap. It is not correctness or prefill speed evidence.",
      "",
  ]), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
