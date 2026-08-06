#!/usr/bin/env python3
"""Gate the one admitted complete-linear non-state projection carrier."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-linear-prefill-nonstate-feasibility-gate-v0"
SOURCE = ROOT / "engine/tools/onednn_linear_prefill_nonstate_probe.cpp"
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
DEFAULT_MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_CAPTURE_GATE = (
    ROOT / "output/linear-prefill-whole-stage-boundary-"
    "20260712Tseq757cleanZ/result.json")
DEFAULT_ENV = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "oneDNN-01b479323f794da1a7a41a6fc084c7e11ccc2c3b")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
EXPECTED_ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
NONSTATE_CAP_US = 1840.0
NOISE_FRACTION = 0.005
RELATIVE_L2_MAXIMUM = 0.002
COSINE_MINIMUM = 0.999


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--capture-gate", type=Path, default=DEFAULT_CAPTURE_GATE)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.warmup < 0 or args.repeat < 3 or args.timeout_s <= 0:
    parser.error("warmup/repeat/timeout arguments are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/linear-prefill-nonstate-feasibility-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path = ROOT) -> dict[str, Any]:
  try:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True,
        timeout=timeout_s, check=False)
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True}


def run_intel(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && " +
      shlex.join(command))
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, label: str, result: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps(result["command"], indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def git_output(*args: str, cwd: Path = ROOT) -> str:
  completed = subprocess.run(
      ["git", *args], cwd=cwd, text=True, capture_output=True, check=True)
  return completed.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"expected JSON object: {path}")
  return value


def parse_json_line(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def projection_map(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
  rows = probe.get("projections", [])
  return {
      str(row.get("label")): row for row in rows
      if isinstance(row, dict) and isinstance(row.get("label"), str)}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.model, args.capture_gate, args.env_script, args.cxx,
      args.onednn_source, args.onednn_build, SOURCE, GGUF_SOURCE,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      args.onednn_source / "include/oneapi/dnnl/dnnl.hpp",
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  onednn_commit = git_output("rev-parse", "HEAD", cwd=args.onednn_source)
  capture_gate = load_json(args.capture_gate)
  capture = ROOT / str(capture_gate.get("capture", {}).get("path", "missing"))
  source_text = SOURCE.read_text(encoding="utf-8")

  binary = raw / "onednn-linear-prefill-nonstate-probe"
  compile_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", f"-I{ROOT / 'engine/include'}",
      str(SOURCE), str(GGUF_SOURCE), f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-ldl", "-pthread", "-o", str(binary),
  ]
  compile_result = run_intel(compile_command, args)
  write_run(raw, "compile", compile_result)

  probe_command = [
      str(binary), "--model", str(args.model), "--capture", str(capture),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
  ]
  rows: list[dict[str, Any]] = []
  for label in ("repeat", "confirm"):
    run_result = (
        run_intel(probe_command, args) if compile_result["returncode"] == 0
        else {"command": probe_command, "returncode": 125, "stdout": "",
              "stderr": "compile failed", "timed_out": False})
    write_run(raw, label, run_result)
    rows.append({"label": label, "returncode": run_result["returncode"],
                 "probe": parse_json_line(run_result)})

  medians = [
      float(row["probe"].get("complete_projection_median_us", math.inf))
      for row in rows]
  spread = (
      abs(medians[0] - medians[1]) / min(medians)
      if len(medians) == 2 and all(math.isfinite(value) and value > 0
                                  for value in medians) else math.inf)
  per_row = [projection_map(row["probe"]) for row in rows]
  fixed_shape = all(
      set(projected) == {"qkv", "z", "alpha", "beta", "out"}
      for projected in per_row)
  row_correctness = []
  row_timing = []
  qkv_rel_l2 = []
  for row, projected, median in zip(rows, per_row, medians):
    comparisons = [
        value.get("comparison", {}) for value in projected.values()]
    row_correctness.append(
        row["returncode"] in (0, 2) and len(comparisons) == 5 and
        all(comparison.get("passes") is True and
            float(comparison.get("relative_l2", math.inf)) <=
            RELATIVE_L2_MAXIMUM and
            float(comparison.get("cosine", -math.inf)) >= COSINE_MINIMUM
            for comparison in comparisons))
    row_timing.append(median <= NONSTATE_CAP_US)
    qkv_rel_l2.append(float(
        projected.get("qkv", {}).get("comparison", {}).get(
            "relative_l2", math.inf)))

  fixed_source = (
      "S8Per32Q6Matmul" in source_text and
      "AffineQ4Matmul" in source_text and
      source_text.count("ProjectionSpec") >= 2 and
      "--storage" not in source_text and "--variant" not in source_text)
  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("locked_oneDNN_codegen_commit",
            onednn_commit == EXPECTED_ONEDNN_COMMIT,
            observed=onednn_commit, expected=EXPECTED_ONEDNN_COMMIT),
      check("seq757_real_boundary_and_budget_passed",
            capture_gate.get("required_checks_passed") is True and
            float(capture_gate.get("budget", {}).get(
                "registered_nonstate_cap_us", math.nan)) == NONSTATE_CAP_US,
            capture_gate=relative(args.capture_gate)),
      check("single_fixed_q8_q4_and_s8_per32_q6_projection_shape", fixed_source),
      check("projection_feasibility_probe_builds",
            compile_result["returncode"] == 0),
      check("repeat_and_confirm_complete_fixed_five_projection_shape",
            fixed_shape, labels=[sorted(value) for value in per_row]),
      check("repeat_and_confirm_pass_all_real_projection_boundaries",
            all(row_correctness), row_correctness=row_correctness,
            qkv_relative_l2=qkv_rel_l2),
      check("projection_only_repeat_and_confirm_clear_nonstate_cap",
            all(row_timing), medians_us=medians, cap_us=NONSTATE_CAP_US,
            omitted_required_stages=["convolution_and_controls", "final_norm"]),
      check("repeat_confirm_spread_inside_noise_band",
            spread <= NOISE_FRACTION, spread_fraction=spread,
            noise_fraction=NOISE_FRACTION),
  ]
  required = all(bool(item["pass"]) for item in checks)
  evaluation_completed = (
      dirty == "" and compile_result["returncode"] == 0 and fixed_shape and
      all(row["returncode"] in (0, 2) and bool(row["probe"]) for row in rows))
  disposition = (
      "accept_nonstate_projection_carrier_continue_complete_tile"
      if required else
      ("reject_nonstate_projection_close_whole_stage_reallocation"
       if evaluation_completed else "incomplete_nonstate_projection_gate"))
  selected_next_route = (
      "native_linear_prefill_complete_whole_stage_tile_gate"
      if required else "native_prefill_product_route_reflection_gate")
  reason = (
      "The fixed five-projection carrier clears all real boundaries and the "
      "complete non-state budget twice; attach convolution/control and norm."
      if required else
      "The fixed projection-only subset already fails at least one registered "
      "accuracy, timing, or noise axis before convolution/control and final "
      "normalization are charged. Close whole-linear budget reallocation "
      "without a projection representation or codegen sweep.")
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "inputs": {"model": str(args.model),
                 "capture_gate": relative(args.capture_gate),
                 "capture": relative(capture)},
      "rows": rows, "complete_projection_medians_us": medians,
      "spread_fraction": spread, "checks": checks,
      "required_checks_passed": required,
      "evaluation_completed": evaluation_completed,
      "disposition": disposition, "selected_next_route": selected_next_route,
      "next_route_reason": reason,
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "manifest.json").write_text(json.dumps({
      "schema_version": SCHEMA, "created_at": created_at, "commit": commit,
      "git_dirty": bool(dirty), "required_checks_passed": required,
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [item["name"] for item in checks if not item["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Linear-prefill non-state feasibility gate", "",
      f"- required_checks_passed: `{str(required).lower()}`",
      f"- disposition: `{disposition}`",
      f"- projection-only repeat/confirm: `{medians}` us",
      f"- non-state cap: `{NONSTATE_CAP_US:.0f} us`",
      f"- paired spread: `{spread:.6%}`",
      f"- Q6 QKV relative L2: `{qkv_rel_l2}`",
      f"- failed checks: `{failed}`", "", reason, ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required, "disposition": disposition,
      "complete_projection_medians_us": medians,
      "spread_fraction": spread, "selected_next_route": selected_next_route,
      "out_dir": relative(out)}, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
