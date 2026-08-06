#!/usr/bin/env python3
"""Run the one-shot compact Q6 low4/high2 DPAS killer gate."""

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
SCHEMA_VERSION = "intel-qwen36-q6-splitplane-dpas-gate-v0"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TENSOR = "blk.7.ffn_down_exps.weight"
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_BASELINE = (
    ROOT / "output/gpu-q6-qmatvec-layer7-ffn-down-full-20260702T234500Z/"
    "probe-result.json")
CPP_SOURCE = ROOT / "engine/tools/q6_splitplane_dpas_gate.cpp"
KERNEL_SOURCE = ROOT / "engine/gpu/opencl/q6_splitplane_dpas.cl"
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
PROMOTION_GB_S = 96.0
EXPECTED_PAYLOAD_BYTES = 220_200_960


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=600)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(
    command: list[str], timeout_s: int, cwd: Path = ROOT,
) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        timeout=timeout_s)
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }
  return {
      "command": command,
      "returncode": process.returncode,
      "stderr": process.stderr,
      "stdout": process.stdout,
      "timed_out": False,
  }


def shell_run(command: list[str], env_script: Path, timeout_s: int) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s)


def build_summary(result: dict[str, Any]) -> str:
  probe = result.get("probe", {})
  baseline = result.get("baseline", {})
  return "\n".join([
      "# Compact Q6 low4/high2 DPAS killer gate",
      "",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      f"- tensor / bytes: `{probe.get('tensor_name')}` / "
      f"`{probe.get('payload_bytes')}`",
      f"- persistent expansion bytes: `{probe.get('persistent_expansion_bytes')}`",
      f"- component correctness: `{str((probe.get('comparison') or {}).get('passed')).lower()}`",
      f"- kernel minimum: `{probe.get('kernel_min_us')} us`",
      f"- compact DPAS source bandwidth: `{probe.get('effective_source_gb_s')} GB/s`",
      f"- current raw-Q6 bandwidth: `{baseline.get('gpu_effective_payload_gb_s')} GB/s`",
      f"- promotion kill-number: `>={PROMOTION_GB_S} GB/s`",
      "",
      "The compact kernel is numerically valid and reads the original 210-byte",
      "Q6_K blocks without persistent I8 expansion, but it misses the route",
      "kill-number. Under ADR 0003 this closes the exact split-plane branch;",
      "the result is not a speedup claim and must not trigger a tuning sweep.",
      "",
  ])


def main() -> int:
  args = parse_args()
  if args.repeat < 1:
    raise SystemExit("--repeat must be positive")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (args.out_dir or (
      ROOT / f"output/q6-splitplane-dpas-gate-{stamp}")).resolve()
  raw_dir = out_dir / "raw"
  ocloc_dir = raw_dir / "ocloc"
  disasm_dir = raw_dir / "disasm"
  ocloc_dir.mkdir(parents=True, exist_ok=False)
  disasm_dir.mkdir()

  for required in (
      args.model, args.env_script, args.cxx, args.baseline, CPP_SOURCE,
      KERNEL_SOURCE, GGUF_SOURCE):
    if not required.is_file():
      raise SystemExit(f"required input missing: {required}")

  binary = raw_dir / "q6-splitplane-dpas-gate"
  build = run([
      str(args.cxx), "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
      "-I", str(ROOT / "engine/include"), str(GGUF_SOURCE),
      str(CPP_SOURCE), "-ldl", "-o", str(binary),
  ], args.timeout_s)
  (raw_dir / "build.stdout").write_text(build["stdout"], encoding="utf-8")
  (raw_dir / "build.stderr").write_text(build["stderr"], encoding="utf-8")

  # ocloc writes into its process working directory; keep all generated files
  # inside the artifact.
  ocloc = run([
      "bash", "-lc",
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      f"ocloc -file {shlex.quote(str(KERNEL_SOURCE))} -device 0xb080 "
      f"-options {shlex.quote('-cl-std=CL3.0')}",
  ], args.timeout_s, ocloc_dir)
  write_json(raw_dir / "ocloc.json", ocloc)
  native_bins = sorted(ocloc_dir.glob("*.bin"))
  disasm = (
      run([
          "ocloc", "disasm", "-file", str(native_bins[0]),
          "-dump", str(disasm_dir), "-device", "0xb080",
      ], args.timeout_s)
      if ocloc["returncode"] == 0 and native_bins else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "ocloc failed",
       "timed_out": False}
  )
  write_json(raw_dir / "disasm.json", disasm)
  assembly_text = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob("*.asm"))
  ze_info_text = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob(".ze_info"))

  probe_run = (
      shell_run([
          str(binary), "--model", str(args.model), "--tensor", args.tensor,
          "--kernel-source", str(KERNEL_SOURCE), "--repeat", str(args.repeat),
      ], args.env_script, args.timeout_s)
      if build["returncode"] == 0 else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
       "timed_out": False}
  )
  (raw_dir / "probe.stdout").write_text(
      probe_run["stdout"], encoding="utf-8")
  (raw_dir / "probe.stderr").write_text(
      probe_run["stderr"], encoding="utf-8")
  probe = json.loads(probe_run["stdout"]) if probe_run["returncode"] == 0 else {}
  baseline = load_json(args.baseline)
  effective = float(probe.get("effective_source_gb_s", 0.0))
  checks = [
      {"name": "host_build_passed", "pass": build["returncode"] == 0},
      {"name": "ocloc_compile_passed", "pass": ocloc["returncode"] == 0},
      {"name": "ocloc_disassembly_passed", "pass": disasm["returncode"] == 0},
      {"name": "dpas_instruction_present", "pass": "dpas." in assembly_text.lower()},
      {"name": "ze_info_has_dpas", "pass":
       "has_dpas:true" in ze_info_text.replace(" ", "")},
      {"name": "probe_passed", "pass": probe_run["returncode"] == 0},
      {"name": "arc_b390_selected", "pass": "B390" in str(probe.get("device_name"))},
      {"name": "real_full_tensor_payload", "pass":
       int(probe.get("payload_bytes", 0)) == EXPECTED_PAYLOAD_BYTES},
      {"name": "no_persistent_i8_expansion", "pass":
       int(probe.get("persistent_expansion_bytes", -1)) == 0},
      {"name": "component_correctness_passed", "pass":
       (probe.get("comparison") or {}).get("passed") is True},
      {"name": "q6_promotion_bandwidth", "pass": effective >= PROMOTION_GB_S},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  disposition = (
      "accept_compact_q6_splitplane_dpas_carrier"
      if required_checks_passed else
      "reject_compact_q6_splitplane_dpas_below_kill_number")
  result = {
      "baseline": {
          "artifact": str(args.baseline.relative_to(ROOT)),
          "gpu_effective_payload_gb_s": baseline.get("gpu_effective_payload_gb_s"),
          "gpu_kernel_min_us": baseline.get("gpu_kernel_min_us"),
      },
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "git": git_state(),
      "kernel_source": {
          "path": str(KERNEL_SOURCE.relative_to(ROOT)),
          "sha256": sha256_file(KERNEL_SOURCE),
      },
      "probe": probe,
      "promotion_gb_s": PROMOTION_GB_S,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "comparison": probe.get("comparison"),
      "required_checks_passed": required_checks_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "standalone Q6 component gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "kernel_source": result["kernel_source"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = {
      "required_checks_passed": required_checks_passed,
      "component_correctness_passed": (probe.get("comparison") or {}).get("passed"),
      "kernel_min_us": probe.get("kernel_min_us"),
      "effective_source_gb_s": probe.get("effective_source_gb_s"),
      "promotion_gb_s": PROMOTION_GB_S,
      "baseline_gb_s": baseline.get("gpu_effective_payload_gb_s"),
  }
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in metrics.items():
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "effective_source_gb_s": probe.get("effective_source_gb_s"),
  }))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
