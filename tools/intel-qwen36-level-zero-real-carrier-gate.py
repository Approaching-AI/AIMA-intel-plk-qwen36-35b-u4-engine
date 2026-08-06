#!/usr/bin/env python3
"""Gate the accepted real Q4/Q6 carriers inside one Level Zero command list."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-level-zero-real-carrier-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-level-zero-real-carrier-smoke"
MODULE_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
STATE_BUDGET = (
    ROOT / "output/packed-token-state-budget-gate-20260712Tseq734cleanZ/result.json")
SCHEDULE_GATE = (
    ROOT / "output/packed-token-schedule-gate-20260712Tseq735-state-cleanZ/result.json")
MECHANISM_GATE = (
    ROOT / "output/packed-token-level-zero-gate-20260712Tseq736-state-cleanZ/result.json")
Q4_TENSOR = "blk.5.ffn_gate_up_exps.weight"
Q6_TENSOR = "blk.7.ffn_down_exps.weight"
Q4_BYTES = 301_989_888
Q6_BYTES = 220_200_960
WALL_RATE_GB_S = 105.99411601919999
INDIVIDUAL_NOISE_FLOOR_GB_S = WALL_RATE_GB_S * (1.0 - 0.005)
KERNEL_RATE_GB_S = 106.524608569878


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=7)
  parser.add_argument("--samples", type=int, default=11)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/level-zero-real-carrier-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(command: list[str], timeout_s: int = 180) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command, "returncode": 124,
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") +
            "\ntimeout",
    }
  return {
      "command": command, "returncode": proc.returncode,
      "stdout": proc.stdout, "stderr": proc.stderr,
  }


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def parse_stdout(row: dict[str, Any]) -> dict[str, Any]:
  try:
    value = json.loads(str(row.get("stdout", "")).strip())
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  row = run(["git", *args], timeout_s=30)
  return row["stdout"].strip() if row["returncode"] == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty), "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def comparison_passes(row: Any) -> bool:
  return (
      isinstance(row, dict) and row.get("same_size") is True and
      row.get("finite") is True and finite(row.get("cosine")) and
      float(row["cosine"]) >= 0.999 and
      finite(row.get("relative_l2")) and
      float(row["relative_l2"]) <= 0.002)


def carrier_row_passes(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True and
      row.get("q4_bytes") == Q4_BYTES and row.get("q6_bytes") == Q6_BYTES and
      comparison_passes(row.get("q4_comparison")) and
      comparison_passes(row.get("q6_comparison")) and
      finite(row.get("q4_effective_gb_s")) and
      float(row["q4_effective_gb_s"]) >= INDIVIDUAL_NOISE_FLOOR_GB_S and
      finite(row.get("q6_effective_gb_s")) and
      float(row["q6_effective_gb_s"]) >= INDIVIDUAL_NOISE_FLOOR_GB_S and
      finite(row.get("combined_effective_gb_s")) and
      float(row["combined_effective_gb_s"]) >= KERNEL_RATE_GB_S and
      finite(row.get("submit_min_us")) and float(row["submit_min_us"]) <= 100.0)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  generated = out / "generated"
  generated.mkdir()
  source_paths = [
      MODULE_SOURCE,
      ROOT / "engine/tools/level_zero_real_carrier_smoke.cpp",
      ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      ROOT / "engine/src/gpu_q4x8_matvec.cpp",
      ROOT / "engine/boundaries.json",
  ]
  required = [args.model, args.cmake, STATE_BUDGET, SCHEDULE_GATE,
              MECHANISM_GATE, *source_paths]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  budget = load_json(STATE_BUDGET)
  schedule = load_json(SCHEDULE_GATE)
  mechanism = load_json(MECHANISM_GATE)
  module_compile = run([
      "ocloc", "compile", "-file", str(MODULE_SOURCE), "-device", "0xb080",
      "-output", "iq36_q4x8_all", "-out_dir", str(generated),
      "-output_no_suffix", "--format", "zebin", "-q",
  ])
  module = generated / "iq36_q4x8_all.bin"
  module_validate = run(["ocloc", "validate", "-file", str(module)]) \
      if module.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "module missing",
      }
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = run([
      str(args.cmake), "--build", str(args.build_dir), "--target", TARGET,
      "-j", "8",
  ]) if configure["returncode"] == 0 else {
      "command": [], "returncode": 125, "stdout": "",
      "stderr": "configure failed",
  }
  binary = args.build_dir / TARGET

  def probe(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = run([
        str(binary), str(args.model), str(module), str(args.warmup),
        str(args.samples),
    ], timeout_s=180)
    parsed = parse_stdout(raw)
    write_json(out / f"{label}-run.json", raw)
    write_json(out / f"{label}.json", parsed)
    return raw, parsed

  repeat_raw, repeat = probe("repeat") if (
      module_validate["returncode"] == 0 and build["returncode"] == 0) else (
          {"returncode": 125, "stdout": "", "stderr": "build missing"}, {})
  confirm_raw, confirm = probe("confirm") if repeat else (
      {"returncode": 125, "stdout": "", "stderr": "repeat missing"}, {})
  link_map = run(["ldd", str(binary)], timeout_s=30) \
      if binary.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "binary missing",
      }
  lower_links = (link_map.get("stdout", "") +
                 link_map.get("stderr", "")).lower()
  source_hashes = {
      str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in source_paths
  }
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("corrected_schedule_and_mechanism_predecessors",
            budget.get("required_checks_passed") is True and
            schedule.get("required_checks_passed") is True and
            mechanism.get("required_checks_passed") is True),
      check("full_native_module_compiles_and_validates",
            module_compile["returncode"] == 0 and
            module_validate["returncode"] == 0 and module.is_file()),
      check("real_carrier_smoke_builds",
            configure["returncode"] == 0 and build["returncode"] == 0),
      check("repeat_real_q4_q6_pair_passes",
            repeat_raw["returncode"] == 0 and carrier_row_passes(repeat),
            q4_gb_s=repeat.get("q4_effective_gb_s"),
            q6_gb_s=repeat.get("q6_effective_gb_s"),
            combined_gb_s=repeat.get("combined_effective_gb_s")),
      check("confirm_real_q4_q6_pair_passes",
            confirm_raw["returncode"] == 0 and carrier_row_passes(confirm),
            q4_gb_s=confirm.get("q4_effective_gb_s"),
            q6_gb_s=confirm.get("q6_effective_gb_s"),
            combined_gb_s=confirm.get("combined_effective_gb_s")),
      check("native_dependency_boundary",
            link_map["returncode"] == 0 and "libze_loader" in lower_links and
            "openvino" not in lower_links and "libdnnl" not in lower_links),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "inputs": {
          "model": str(args.model), "q4_tensor": Q4_TENSOR,
          "q6_tensor": Q6_TENSOR,
          "state_budget": str(STATE_BUDGET.relative_to(ROOT)),
          "schedule_gate": str(SCHEDULE_GATE.relative_to(ROOT)),
          "mechanism_gate": str(MECHANISM_GATE.relative_to(ROOT)),
      },
      "module": {
          "path": str(module.relative_to(ROOT)),
          "sha256": hashlib.sha256(module.read_bytes()).hexdigest()
              if module.is_file() else None,
      },
      "source_sha256": source_hashes,
      "admission": {
          "per_carrier_wall_gb_s_min": WALL_RATE_GB_S,
          "per_carrier_noise_floor_gb_s_min": INDIVIDUAL_NOISE_FLOOR_GB_S,
          "paired_kernel_gb_s_min": KERNEL_RATE_GB_S,
          "component_cosine_min": 0.999,
          "component_relative_l2_max": 0.002,
      },
      "repeat": repeat, "confirm": confirm, "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "admit_real_q4_q6_carriers_to_level_zero_backend"
          if passed else "reject_level_zero_real_carrier_port"),
      "product_promotion_ready": False, "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False, "speedup_claims_allowed": False,
  })
  write_json(out / "build.json", {
      "module_compile": module_compile, "module_validate": module_validate,
      "configure": configure, "build": build, "link_map": link_map,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "artifact": str(out),
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "git": state, "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  with (out / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for phase, row in (("repeat", repeat), ("confirm", confirm)):
      for metric in ("q4_effective_gb_s", "q6_effective_gb_s",
                     "combined_effective_gb_s", "combined_kernel_min_us"):
        fh.write(json.dumps({
            "metric": metric, "phase": phase, "value": row.get(metric),
        }, sort_keys=True) + "\n")
  (out / "summary.md").write_text("\n".join([
      "# Level Zero real Q4/Q6 carrier gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- repeat Q4 / Q6 / paired: `"
      f"{float(repeat.get('q4_effective_gb_s', math.nan)):.3f} / "
      f"{float(repeat.get('q6_effective_gb_s', math.nan)):.3f} / "
      f"{float(repeat.get('combined_effective_gb_s', math.nan)):.3f} GB/s`",
      f"- confirm Q4 / Q6 / paired: `"
      f"{float(confirm.get('q4_effective_gb_s', math.nan)):.3f} / "
      f"{float(confirm.get('q6_effective_gb_s', math.nan)):.3f} / "
      f"{float(confirm.get('combined_effective_gb_s', math.nan)):.3f} GB/s`",
      f"- per-carrier noise / paired kernel floors: `"
      f"{INDIVIDUAL_NOISE_FLOOR_GB_S:.3f} / "
      f"{KERNEL_RATE_GB_S:.3f} GB/s`",
      "- correctness: `both real full tensors pass component contract`", "",
      ("This admits the two accepted real carriers into the full Level Zero "
       "backend port." if passed else
       "This rejects the real-carrier port under the registered paired ruler."),
      "It is component evidence, not a full token or speedup claim.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out), "pass": passed,
      "repeat_combined_gb_s": repeat.get("combined_effective_gb_s"),
      "confirm_combined_gb_s": confirm.get("combined_effective_gb_s"),
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
