#!/usr/bin/env python3
"""Gate the device-indexed, byte-coalesced Q6 selected/shared down carrier."""

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
SCHEMA = "intel-qwen36-level-zero-indexed-q6-down-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-level-zero-indexed-q6-down-smoke"
MODULE_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
PREDECESSORS = [
    ROOT / "output/packed-token-state-budget-gate-20260712Tseq734cleanZ/result.json",
    ROOT / "output/packed-token-schedule-gate-20260712Tseq735-state-cleanZ/result.json",
    ROOT / "output/packed-token-level-zero-gate-20260712Tseq736-state-cleanZ/result.json",
    ROOT / "output/level-zero-real-carrier-gate-20260712Tseq738cleanZ/result.json",
]
SELECTED_TENSOR = "blk.7.ffn_down_exps.weight"
SHARED_TENSOR = "blk.7.ffn_down_shexp.weight"
SELECTED_RESIDENT_BYTES = 220_200_960
SHARED_RESIDENT_BYTES = 860_160
SELECTED_ACTIVE_BYTES = 6_881_280
SHARED_ACTIVE_BYTES = 860_160
ACTIVE_BYTES = SELECTED_ACTIVE_BYTES + SHARED_ACTIVE_BYTES
KERNEL_RATE_GB_S = 106.524608569878
KERNEL_TIME_US_MAX = ACTIVE_BYTES / (KERNEL_RATE_GB_S * 1000.0)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=1000)
  parser.add_argument("--samples", type=int, default=21)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/level-zero-indexed-q6-down-gate-{stamp}"
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


def carrier_correct(row: dict[str, Any]) -> bool:
  return (
      row.get("active_bytes") == ACTIVE_BYTES and
      row.get("selected_active_bytes") == SELECTED_ACTIVE_BYTES and
      row.get("shared_active_bytes") == SHARED_ACTIVE_BYTES and
      row.get("selected_resident_bytes") == SELECTED_RESIDENT_BYTES and
      row.get("shared_resident_bytes") == SHARED_RESIDENT_BYTES and
      row.get("selected_positions") == [0, 1, 7, 31, 63, 127, 191, 255] and
      row.get("group_size") == 64 and
      comparison_passes(row.get("selected_comparison")) and
      comparison_passes(row.get("shared_comparison")) and
      finite(row.get("kernel_min_us")) and
      finite(row.get("effective_gb_s")) and
      finite(row.get("submit_min_us")) and
      float(row["submit_min_us"]) <= 100.0)


def rate_passes(row: dict[str, Any]) -> bool:
  return (
      carrier_correct(row) and
      float(row["kernel_min_us"]) <= KERNEL_TIME_US_MAX and
      float(row["effective_gb_s"]) >= KERNEL_RATE_GB_S)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  generated = out / "generated"
  generated.mkdir()
  source_paths = [
      MODULE_SOURCE,
      ROOT / "engine/tools/level_zero_indexed_q6_down_smoke.cpp",
      ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      ROOT / "engine/src/gpu_q4x8_matvec.cpp",
      ROOT / "engine/boundaries.json",
      Path(__file__).resolve(),
  ]
  required = [args.model, args.cmake, *PREDECESSORS, *source_paths]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  predecessors = [load_json(path) for path in PREDECESSORS]
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

  runnable = module_validate["returncode"] == 0 and build["returncode"] == 0
  repeat_raw, repeat = probe("repeat") if runnable else (
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
  correctness_checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("accepted_predecessors_pass",
            all(row.get("required_checks_passed") is True
                for row in predecessors)),
      check("full_native_module_compiles_and_validates",
            module_compile["returncode"] == 0 and
            module_validate["returncode"] == 0 and module.is_file()),
      check("indexed_q6_smoke_builds",
            configure["returncode"] == 0 and build["returncode"] == 0),
      check("repeat_real_selected_shared_q6_correct",
            repeat_raw["returncode"] in (0, 2) and carrier_correct(repeat),
            effective_gb_s=repeat.get("effective_gb_s"),
            kernel_min_us=repeat.get("kernel_min_us")),
      check("confirm_real_selected_shared_q6_correct",
            confirm_raw["returncode"] in (0, 2) and carrier_correct(confirm),
            effective_gb_s=confirm.get("effective_gb_s"),
            kernel_min_us=confirm.get("kernel_min_us")),
      check("native_dependency_boundary",
            link_map["returncode"] == 0 and "libze_loader" in lower_links and
            "openvino" not in lower_links and "libdnnl" not in lower_links),
  ]
  correctness_passed = all(row["pass"] for row in correctness_checks)
  rate_checks = [
      check("repeat_clears_strict_kernel_rate", rate_passes(repeat),
            effective_gb_s=repeat.get("effective_gb_s"),
            kernel_min_us=repeat.get("kernel_min_us")),
      check("confirm_clears_strict_kernel_rate", rate_passes(confirm),
            effective_gb_s=confirm.get("effective_gb_s"),
            kernel_min_us=confirm.get("kernel_min_us")),
  ]
  rate_passed = all(row["pass"] for row in rate_checks)
  passed = correctness_passed and rate_passed
  created_at = iso_now()
  source_hashes = {
      str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in source_paths
  }
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "inputs": {
          "model": str(args.model), "selected_tensor": SELECTED_TENSOR,
          "shared_tensor": SHARED_TENSOR,
          "predecessors": [str(path.relative_to(ROOT)) for path in PREDECESSORS],
      },
      "layout": {
          "name": "q6_rowstripe_byte_coalesced_tile32",
          "dynamic_device_selected_positions": True,
          "rows_per_tile": 32, "group_size": 64,
          "selected_positions": [0, 1, 7, 31, 63, 127, 191, 255],
          "selected_resident_bytes": SELECTED_RESIDENT_BYTES,
          "shared_resident_bytes": SHARED_RESIDENT_BYTES,
          "selected_active_bytes": SELECTED_ACTIVE_BYTES,
          "shared_active_bytes": SHARED_ACTIVE_BYTES,
      },
      "admission": {
          "strict_kernel_gb_s_min": KERNEL_RATE_GB_S,
          "strict_kernel_us_max": KERNEL_TIME_US_MAX,
          "component_cosine_min": 0.999,
          "component_relative_l2_max": 0.002,
      },
      "module": {
          "path": str(module.relative_to(ROOT)),
          "sha256": hashlib.sha256(module.read_bytes()).hexdigest()
              if module.is_file() else None,
      },
      "source_sha256": source_hashes,
      "repeat": repeat, "confirm": confirm,
      "correctness_checks": correctness_checks,
      "rate_checks": rate_checks,
      "correctness_checks_passed": correctness_passed,
      "strict_component_rate_passed": rate_passed,
      "required_checks_passed": passed,
      "disposition": (
          "admit_indexed_q6_down_to_real_stage_port" if passed else
          "reject_strict_per_stage_admission_retain_aggregate_timing_evidence"
          if correctness_passed else
          "reject_indexed_q6_down_correctness_or_mechanism"),
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": correctness_checks,
      "correctness_checks_passed": correctness_passed,
      "strict_component_rate_passed": rate_passed,
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
      for metric in ("effective_gb_s", "kernel_min_us", "kernel_mean_us",
                     "submit_min_us", "wall_min_us"):
        fh.write(json.dumps({
            "metric": metric, "phase": phase, "value": row.get(metric),
        }, sort_keys=True) + "\n")
  (out / "summary.md").write_text("\n".join([
      "# Level Zero indexed Q6 down gate", "",
      f"- correctness checks passed: `{str(correctness_passed).lower()}`",
      f"- strict component rate passed: `{str(rate_passed).lower()}`",
      f"- repeat / confirm rate: `"
      f"{float(repeat.get('effective_gb_s', math.nan)):.3f} / "
      f"{float(confirm.get('effective_gb_s', math.nan)):.3f} GB/s`",
      f"- repeat / confirm kernel: `"
      f"{float(repeat.get('kernel_min_us', math.nan)):.3f} / "
      f"{float(confirm.get('kernel_min_us', math.nan)):.3f} us`",
      f"- strict admission: `>= {KERNEL_RATE_GB_S:.3f} GB/s`, "
      f"`<= {KERNEL_TIME_US_MAX:.3f} us`",
      "- product promotion ready: `false`",
      "- speedup claims allowed: `false`", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "correctness_checks_passed": correctness_passed,
      "strict_component_rate_passed": rate_passed,
      "required_checks_passed": passed,
      "repeat_gb_s": repeat.get("effective_gb_s"),
      "confirm_gb_s": confirm.get("effective_gb_s"),
  }, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
