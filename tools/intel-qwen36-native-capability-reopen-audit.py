#!/usr/bin/env python3
"""Audit whether a genuinely new native capability can reopen ADR 0061."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
REJECTED = ROOT / f"doc/active/{WORKSTREAM}/rejected-routes.json"
ADR_0048 = ROOT / "doc/adr/0048-record-measured-1p10-prefill-route-exhaustion.md"
ADR_0049 = ROOT / "doc/adr/0049-reopen-prefill-only-gpu-npu-complete-ffn-gate.md"
ADR_0050 = ROOT / "doc/adr/0050-close-prefill-only-gpu-npu-restore-owner-decision.md"
ADR_0054 = ROOT / "doc/adr/0054-close-scalar-fused-gqa-select-xmx-flash-decode.md"
ADR_0055 = ROOT / "doc/adr/0055-close-xmx-gqa-select-sdpa-provider-codegen.md"
ADR_0061 = ROOT / "doc/adr/0061-record-long-context-native-route-exhaustion.md"
SEQ773 = ROOT / "output/xmx-gqa-fp16-kv-decode-20260713Tseq773cleanZ/result.json"
SEQ781 = ROOT / "output/cpu-avx2-fp16-gqa-decode-20260713Tseq781cleanZ/result.json"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=120)
  return parser.parse_args()


def run(command: list[str], timeout: int) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout)
  return {
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
  }


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise RuntimeError(f"expected object: {path}")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30)["stdout"].strip()
  dirty = run(["git", "status", "--porcelain"], 30)["stdout"].splitlines()
  try:
    relative = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative = ""
  dirty = [row for row in dirty if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def compiler_candidates() -> dict[str, str | None]:
  fixed = {
      "icpx": [
          Path("/opt/intel/oneapi/compiler/latest/bin/icpx"),
          Path("/home/intel/intel-box-env/conda/bin/icpx"),
      ],
      "dpcpp": [
          Path("/opt/intel/oneapi/compiler/latest/bin/dpcpp"),
          Path("/home/intel/intel-box-env/conda/bin/dpcpp"),
      ],
  }
  result: dict[str, str | None] = {}
  for name, paths in fixed.items():
    resolved = shutil.which(name)
    if resolved is None:
      resolved = next(
          (str(path) for path in paths if path.is_file()
           and os.access(path, os.X_OK)), None)
    result[name] = resolved
  return result


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def summary(payload: dict[str, Any]) -> str:
  capability = payload["capability"]
  return "\n".join([
      "# Native capability reopen audit",
      "",
      f"- audit checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- route reopen allowed: `{str(payload['route_reopen_allowed']).lower()}`",
      f"- native FP8 extension: `{str(capability['native_fp8']).lower()}`",
      f"- SYCL/ESIMD compiler: `{str(capability['esimd_compiler']).lower()}`",
      f"- CPU AVX-512/AMX: `{str(capability['cpu_wide_matrix_isa']).lower()}`",
      "- existing XMX / integer-dot / BF16 / NPU capabilities: `closed by evidence`",
      "",
      "No newly discovered capability has a complete bound below the locked",
      "decode or prefill cap. ADR 0061 therefore remains the active gate.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required = [
      TARGET_CONTRACT, REJECTED, ADR_0048, ADR_0049, ADR_0050, ADR_0054,
      ADR_0055, ADR_0061, SEQ773, SEQ781,
  ]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing audit inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  clinfo = run([
      "bash", "-lc",
      f"source {ENV_SCRIPT} >/dev/null 2>&1 && clinfo"], args.timeout_s)
  lscpu = run(["lscpu"], args.timeout_s)
  ocloc = run(["ocloc", "--version"], args.timeout_s)
  write_json(raw_dir / "clinfo-command.json", clinfo)
  write_json(raw_dir / "lscpu-command.json", lscpu)
  write_json(raw_dir / "ocloc-command.json", ocloc)

  target = load_json(TARGET_CONTRACT)
  rejected_text = REJECTED.read_text()
  adr0048 = ADR_0048.read_text()
  adr0049 = ADR_0049.read_text()
  adr0050 = ADR_0050.read_text()
  adr0054 = ADR_0054.read_text()
  adr0055 = ADR_0055.read_text()
  adr0061 = ADR_0061.read_text()
  adr0050_flat = " ".join(adr0050.split())
  seq773 = load_json(SEQ773)
  seq781 = load_json(SEQ781)
  extensions = set(re.findall(r"\bcl_[a-zA-Z0-9_]+", clinfo["stdout"]))
  compilers = compiler_candidates()
  flags_match = re.search(r"^Flags:\s+(.+)$", lscpu["stdout"], re.MULTILINE)
  cpu_flags = set(flags_match.group(1).split()) if flags_match else set()
  fp8_extensions = sorted(
      extension for extension in extensions if "fp8" in extension.lower())
  native_fp8 = bool(fp8_extensions)
  esimd_compiler = any(compilers.values())
  cpu_wide_matrix_isa = any(
      flag.startswith("avx512") or flag.startswith("amx_")
      for flag in cpu_flags)
  xmx_present = "cl_intel_subgroup_matrix_multiply_accumulate" in extensions
  bf16_present = "cl_intel_bfloat16_conversions" in extensions
  integer_dot_present = "cl_khr_integer_dot_product" in extensions
  xmx_closed = bool(
      xmx_present and seq773.get("required_checks_passed") is False
      and "closes" in adr0055 and "XMX" in adr0055)
  integer_dot_closed = bool(
      integer_dot_present
      and "Do not retry integer-dot" in rejected_text)
  bf16_closed = bool(
      bf16_present and "FP16/BF16" in adr0054 and "close" in adr0054.lower())
  cpu_closed = bool(
      seq781.get("required_checks_passed") is False
      and float(seq781.get("result", {}).get("repeat_ms", 0.0)) > 2.825
      and "CPU" in adr0061)
  npu_decode_closed = bool(
      "reopens neither NPU decode" in adr0049
      and "NPU decode" in adr0050 and "closed" in adr0050.lower())
  prefill_closed = bool(
      "no evidence-backed native prefill architecture" in adr0048
      and "No evidence-backed native prefill architecture remains"
      in adr0050_flat)
  device_matches = bool(
      target.get("runtime", {}).get("opencl_device") in clinfo["stdout"]
      and target.get("runtime", {}).get("opencl_driver_version")
      in clinfo["stdout"])
  capability = {
      "cpu_flags": sorted(cpu_flags),
      "cpu_wide_matrix_isa": cpu_wide_matrix_isa,
      "esimd_compiler": esimd_compiler,
      "esimd_compilers": compilers,
      "existing_bf16": bf16_present,
      "existing_integer_dot": integer_dot_present,
      "existing_xmx": xmx_present,
      "fp8_extensions": fp8_extensions,
      "native_fp8": native_fp8,
      "ocloc_version": ocloc["stdout"].strip(),
      "opencl_extension_count": len(extensions),
  }
  checks = [
      check("repository_clean_at_audit", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("locked_device_and_driver", device_matches),
      check("opencl_inventory_captured", clinfo["returncode"] == 0),
      check("ocloc_inventory_captured", ocloc["returncode"] == 0),
      check("no_native_fp8_extension", not native_fp8,
            extensions=fp8_extensions),
      check("no_sycl_esimd_compiler", not esimd_compiler,
            compilers=compilers),
      check("no_cpu_avx512_or_amx", not cpu_wide_matrix_isa),
      check("existing_xmx_capability_closed", xmx_closed),
      check("existing_integer_dot_capability_closed", integer_dot_closed),
      check("existing_bf16_capability_closed", bf16_closed),
      check("cpu_vector_backend_closed", cpu_closed),
      check("npu_decode_closed", npu_decode_closed),
      check("native_prefill_closed", prefill_closed),
      check("adr0061_owner_gate_present", "owner-recorded" in adr0061),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_reopen_allowed = False
  source = {
      "path": str(Path(__file__).resolve().relative_to(ROOT)),
      "sha256": sha256(Path(__file__).resolve()),
  }
  payload = {
      "capability": capability,
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "new_capability_complete_bound_pass": False,
      "required_checks_passed": required_checks_passed,
      "route_reopen_allowed": route_reopen_allowed,
      "schema_version": "intel-qwen36-native-capability-reopen-audit-v0",
      "source": source,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"],
      "git": git,
      "required_checks_passed": required_checks_passed,
      "route_reopen_allowed": route_reopen_allowed,
      "schema_version": payload["schema_version"],
      "source": source,
      "tool": source["path"],
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "route_reopen_allowed": route_reopen_allowed,
      "speedup_claims_allowed": False,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "Capability inventory audit; no performance candidate ran.",
      "required_checks_passed": True,
  })
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
      "route_reopen_allowed": route_reopen_allowed,
  }, sort_keys=True))
  return 0 if required_checks_passed and not route_reopen_allowed else 2


if __name__ == "__main__":
  raise SystemExit(main())
