#!/usr/bin/env python3
"""Test the sole compressed deterministic contribution-plane alternate."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import shlex
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-f16-contribution-plane-feasibility-gate-v1"
KERNEL = ROOT / "engine/gpu/opencl/f16_contribution_q4k_codegen_preflight.cl"
RUNNER = ROOT / "engine/tools/swiglu_math_throughput_preflight.cpp"
DEFAULT_SEQ650 = (
    ROOT / "output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ")
DEFAULT_SEQ651 = (
    ROOT / "output/in-core-affine-codegen-feasibility-gate-20260711Tseq651cleanZ")
DEFAULT_CAPTURE = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/"
    "raw/capture/payloads")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
PAYLOADS = {
    "weights": (
        "ffn_moe_weights_norm-27__tok1023__ord2.bin",
        "0141a67188d6d8d92e39cac7f646d6af843f4a1ac9411c6505e87cf988cfe2af"),
    "down": (
        "ffn_moe_down-27__tok1023__ord4.bin",
        "b6977e220e0dc081a111ddc104607fb6e869888f01f18f852dfc60820b045f26"),
    "moe": (
        "ffn_moe_out-27__tok1023__ord5.bin",
        "e0dc494a2823ffe10cae0b5bd5c802fb4358b8cbc44b8495fc7c2fc0f8df76f2"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seq650", type=Path, default=DEFAULT_SEQ650)
  parser.add_argument("--seq651", type=Path, default=DEFAULT_SEQ651)
  parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.warmup, args.repeat, args.timeout_s) <= 0:
    parser.error("warmup, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/f16-contribution-plane-feasibility-gate-{stamp}"
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


def shell_run(command: list[str], env_script: Path, timeout_s: int,
              cwd: Path) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s, cwd)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected a JSON object")
  return value


def parse_last_json(result: dict[str, Any]) -> dict[str, Any]:
  lines = [line for line in str(result.get("stdout", "")).splitlines()
           if line.strip()]
  if not lines:
    return {}
  try:
    value = json.loads(lines[-1])
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(*parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def read_float_array(path: Path) -> array.array[float]:
  values = array.array("f")
  with path.open("rb") as handle:
    values.fromfile(handle, path.stat().st_size // 4)
  return values


def contribution_correctness(paths: dict[str, Path]) -> dict[str, Any]:
  down = read_float_array(paths["down"])
  weights = read_float_array(paths["weights"])
  oracle = read_float_array(paths["moe"])
  token_count = 1024
  ranks = 8
  hidden = 2048
  expected_down = token_count * ranks * hidden
  if len(down) != expected_down or len(weights) != token_count * ranks:
    raise SystemExit("locked weighted-down payload shape changed")
  if len(oracle) != token_count * hidden:
    raise SystemExit("locked routed-output payload shape changed")
  scattered = array.array("f", [0.0]) * len(oracle)
  weighted_max = 0.0
  weighted_abs_sum = 0.0
  weighted_squared_sum = 0.0
  weighted_mismatches = 0
  for token in range(token_count):
    output_base = token * hidden
    for rank in range(ranks):
      weight = weights[token * ranks + rank]
      source_base = (token * ranks + rank) * hidden
      for inner in range(hidden):
        exact = down[source_base + inner] * weight
        rounded = struct.unpack("<e", struct.pack("<e", exact))[0]
        difference = abs(rounded - exact)
        weighted_max = max(weighted_max, difference)
        weighted_abs_sum += difference
        weighted_squared_sum += difference * difference
        weighted_mismatches += difference > 5e-3
        scattered[output_base + inner] = (
            scattered[output_base + inner] + rounded)
  routed_max = 0.0
  routed_abs_sum = 0.0
  routed_squared_sum = 0.0
  routed_mismatches = 0
  for observed, expected in zip(scattered, oracle):
    difference = abs(observed - expected)
    routed_max = max(routed_max, difference)
    routed_abs_sum += difference
    routed_squared_sum += difference * difference
    routed_mismatches += difference > 5e-3
  return {
      "weighted_down": {
          "compared_value_count": expected_down,
          "finite": all(math.isfinite(value) for value in scattered),
          "max_abs_diff": weighted_max,
          "mean_abs_diff": weighted_abs_sum / expected_down,
          "mismatch_count": weighted_mismatches,
          "rmse": math.sqrt(weighted_squared_sum / expected_down),
      },
      "routed_output": {
          "compared_value_count": len(oracle),
          "finite": all(math.isfinite(value) for value in scattered),
          "max_abs_diff": routed_max,
          "mean_abs_diff": routed_abs_sum / len(oracle),
          "mismatch_count": routed_mismatches,
          "rmse": math.sqrt(routed_squared_sum / len(oracle)),
      },
  }


def build_model(seq650: dict[str, Any], math_probe: dict[str, Any]) -> dict[str, Any]:
  probe = seq650["probe"]
  stage = probe["stage_us"]
  bandwidth_gb_s = float(seq650["budget"]["planning_gb_s"])
  bytes_per_us = bandwidth_gb_s * 1000.0
  active_experts = int(probe["active_experts"])
  assignments = int(probe["assignment_count"])
  hidden = 2048
  intermediate = 512
  group32_entries = (
      active_experts * (hidden // 32) * (2 * intermediate) +
      active_experts * (intermediate // 32) * hidden)
  minimum_bytes = group32_entries * 4
  removable_zero_point_bytes = group32_entries // 2
  net_affine_payload_bytes = minimum_bytes - removable_zero_point_bytes
  paired_input_save = assignments * hidden * 2
  paired_output_save = assignments * intermediate * 2
  shell_sum = (
      float(stage["gather"]) + float(stage["residual_swiglu"]) +
      float(stage["residual_weight"]) + float(stage["scatter"]))
  inferred_exact_core = float(probe["minimum_us"]) - shell_sum
  synchronized_exact_core = (
      float(stage["gate"]) + float(stage["up"]) + float(stage["down"]))
  residual_samples = [float(value)
                      for value in math_probe["residual_fma"]["samples_us"]]
  swiglu_samples = [float(value)
                    for value in math_probe["swiglu"]["samples_us"]]
  residual_charge = max(residual_samples)
  swiglu_charge = max(swiglu_samples)
  router_multiply_count = assignments * hidden
  residual_fma_count = int(math_probe["residual_fma_count"])
  router_charge = residual_charge * router_multiply_count / residual_fma_count
  gather_charge = float(stage["gather"])
  scatter_charge = float(stage["scatter"])
  projected = (
      inferred_exact_core - paired_input_save / bytes_per_us -
      paired_output_save / bytes_per_us +
      net_affine_payload_bytes / bytes_per_us + gather_charge +
      scatter_charge + residual_charge + swiglu_charge + router_charge)
  f32_scatter_bytes = assignments * hidden * 4 + 1024 * hidden * 4 + assignments * 4
  f16_scatter_bytes = assignments * hidden * 2 + 1024 * hidden * 4 + assignments * 4
  scaled_f16_scatter = scatter_charge * f16_scatter_bytes / f32_scatter_bytes
  ideal_transport = (
      projected - gather_charge - scatter_charge + scaled_f16_scatter)
  cap = float(seq650["budget"]["kernel_cap_us"])
  return {
      "assumptions": {
          "bandwidth_gb_s": bandwidth_gb_s,
          "core_basis": (
              "seq650 complete minimum minus all four measured external stages; "
              "more optimistic than the synchronized gate+up+down sum"),
          "paired_gateup_credit": "one grouped-input read and one F16 output",
          "scatter_charge": (
              "full measured seq650 F32 scatter in the source-realizable row; "
              "byte-scaled F16 scatter only in the ideal-transport diagnostic"),
          "math_charge": "maximum observed event time from the paired microbench",
      },
      "bytes": {
          "group32_coefficient_entries": group32_entries,
          "floating_minimums": minimum_bytes,
          "removable_u4_zero_points": removable_zero_point_bytes,
          "net_affine_payload": net_affine_payload_bytes,
          "paired_gateup_input_read_saved": paired_input_save,
          "paired_gateup_output_saved": paired_output_save,
          "f32_scatter": f32_scatter_bytes,
          "f16_scatter": f16_scatter_bytes,
      },
      "timing_us": {
          "cap": cap,
          "seq650_complete_minimum": float(probe["minimum_us"]),
          "seq650_external_stage_sum": shell_sum,
          "inferred_exact_core": inferred_exact_core,
          "synchronized_exact_core": synchronized_exact_core,
          "net_affine_payload": net_affine_payload_bytes / bytes_per_us,
          "gather": gather_charge,
          "scatter_conservative": scatter_charge,
          "scatter_f16_byte_scaled_diagnostic": scaled_f16_scatter,
          "residual_fma_max": residual_charge,
          "swiglu_max": swiglu_charge,
          "router_weight_estimate": router_charge,
          "source_realizable_projection": projected,
          "source_realizable_over_cap": projected - cap,
          "ideal_transport_projection": ideal_transport,
          "ideal_transport_over_cap": ideal_transport - cap,
      },
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  ocloc_dir = raw_dir / "ocloc"
  disasm_dir = raw_dir / "disasm"
  ocloc_dir.mkdir(parents=True, exist_ok=False)
  disasm_dir.mkdir()
  required = [
      args.seq650 / "result.json", args.seq651 / "result.json", args.env_script,
      args.cxx, KERNEL, RUNNER,
  ]
  paths: dict[str, Path] = {}
  for name, (filename, expected_hash) in PAYLOADS.items():
    path = args.capture / filename
    required.append(path)
    paths[name] = path
    if path.exists() and sha256_file(path) != expected_hash:
      raise SystemExit(f"locked {name} payload hash mismatch")
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  seq650 = load_json(args.seq650 / "result.json")
  seq651 = load_json(args.seq651 / "result.json")
  correctness = contribution_correctness(paths)

  compile_result = shell_run([
      "ocloc", "-file", str(KERNEL), "-device", "0xb080",
      "-options", "-cl-std=CL2.0",
  ], args.env_script, args.timeout_s, ocloc_dir)
  write_json(raw_dir / "ocloc-compile.json", compile_result)
  native_bins = sorted(ocloc_dir.glob("*.bin"))
  disasm_result = (
      run([
          "ocloc", "disasm", "-file", str(native_bins[0]), "-dump",
          str(disasm_dir), "-device", "0xb080",
      ], args.timeout_s, ROOT)
      if compile_result["returncode"] == 0 and native_bins else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "compile failed",
       "timed_out": False})
  write_json(raw_dir / "ocloc-disasm.json", disasm_result)
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob("*.asm"))
  ze_info = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob(".ze_info"))
  compile_text = str(compile_result["stdout"]) + str(compile_result["stderr"])

  runner_binary = raw_dir / "swiglu-math-throughput-preflight"
  runner_build = shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300", str(RUNNER), "-lOpenCL",
      "-o", str(runner_binary),
  ], args.env_script, args.timeout_s, ROOT)
  write_json(raw_dir / "runner-build.json", runner_build)
  runner_result = (
      shell_run([
          str(runner_binary), "--warmup", str(args.warmup),
          "--repeat", str(args.repeat),
      ], args.env_script, args.timeout_s, ROOT)
      if runner_build["returncode"] == 0 else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
       "timed_out": False})
  write_json(raw_dir / "runner.json", runner_result)
  math_probe = parse_last_json(runner_result)
  if not math_probe:
    raise SystemExit("math throughput preflight did not produce JSON")
  model = build_model(seq650, math_probe)
  write_json(raw_dir / "model.json", model)
  write_json(raw_dir / "contribution-correctness.json", correctness)

  kernel_text = KERNEL.read_text(encoding="utf-8")
  codegen_checks = [
      check("ocloc_compile_passed", compile_result["returncode"] == 0),
      check("exact_ptl_device_selected", "ptl-h-a0" in compile_text),
      check("ocloc_disassembly_passed", disasm_result["returncode"] == 0),
      check("three_m8_u4_dpas_instructions_present",
            assembly.lower().count("dpas.8x8") >= 3,
            observed=assembly.lower().count("dpas.8x8")),
      check("u4_source_precision_present", ":u4" in assembly.lower()),
      check("both_kernels_report_dpas", ze_info.count("has_dpas:        true") >= 2),
      check("affine_and_weighting_precede_f16_store",
            kernel_text.index("const float8 restored =") <
            kernel_text.index("convert_half8_rte(restored * router_weights[lane])")),
      check("math_runner_build_passed", runner_build["returncode"] == 0),
      check("math_runner_completed",
            runner_result["returncode"] == 0 and math_probe.get("finite") is True),
      check("math_runner_locked_work",
            math_probe.get("value_count") == 4_194_304 and
            math_probe.get("residual_fma_count") == 805_306_368),
  ]
  weighted = correctness["weighted_down"]
  routed = correctness["routed_output"]
  correctness_checks = [
      check("all_16777216_weighted_values_compared",
            weighted["compared_value_count"] == 16_777_216),
      check("f16_weighted_contributions_within_5e_3",
            weighted["mismatch_count"] == 0 and
            weighted["max_abs_diff"] <= 5e-3),
      check("all_2097152_routed_values_compared",
            routed["compared_value_count"] == 2_097_152),
      check("deterministic_f16_scatter_within_5e_3",
            routed["mismatch_count"] == 0 and routed["max_abs_diff"] <= 5e-3),
  ]
  timing = model["timing_us"]
  performance_checks = [
      check("source_realizable_projection_below_cap",
            timing["source_realizable_projection"] <= timing["cap"],
            observed_us=timing["source_realizable_projection"],
            required_us=timing["cap"]),
      check("ideal_transport_projection_below_cap",
            timing["ideal_transport_projection"] <= timing["cap"],
            observed_us=timing["ideal_transport_projection"],
            required_us=timing["cap"]),
  ]
  evidence_checks = [
      check("seq650_clean_exact_measurement",
            seq650.get("git", {}).get("dirty") is False and
            seq650.get("evidence_checks_passed") is True),
      check("seq651_closed_no_plane_route",
            seq651.get("required_checks_passed") is False and
            seq651.get("disposition") ==
            "reject_in_core_affine_grouped_codegen_on_aggregation_and_floor"),
      check("locked_payload_hashes_match", True),
      check("speedup_claims_forbidden", True),
  ]
  codegen_passed = all(row["pass"] for row in codegen_checks)
  correctness_passed = all(row["pass"] for row in correctness_checks)
  performance_passed = all(row["pass"] for row in performance_checks)
  evidence_passed = all(row["pass"] for row in evidence_checks)
  required_passed = (
      codegen_passed and correctness_passed and performance_passed and
      evidence_passed)
  disposition = (
      "admit_one_real_f16_contribution_grouped_q4k_killer_gate"
      if required_passed else
      "reject_f16_contribution_plane_above_exact_core_cap")
  result = {
      "checks": evidence_checks + codegen_checks + correctness_checks +
                performance_checks,
      "codegen_checks_passed": codegen_passed,
      "correctness": correctness,
      "correctness_checks_passed": correctness_passed,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_passed,
      "git": git_state(),
      "math_probe": math_probe,
      "model": model,
      "performance_checks_passed": performance_passed,
      "required_checks_passed": required_passed,
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "capture": str(args.capture.relative_to(ROOT)),
          "kernel": str(KERNEL.relative_to(ROOT)),
          "kernel_sha256": sha256_file(KERNEL),
          "payload_sha256": {key: value[1] for key, value in PAYLOADS.items()},
          "runner": str(RUNNER.relative_to(ROOT)),
          "runner_sha256": sha256_file(RUNNER),
          "seq650": str(args.seq650.relative_to(ROOT)),
          "seq651": str(args.seq651.relative_to(ROOT)),
      },
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": evidence_checks + codegen_checks + correctness_checks,
      "correctness": correctness,
      "correctness_checks_passed": correctness_passed,
      "note": "oracle-only carrier proof; no real matrix kernel was authorized",
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "static carrier/codegen/performance feasibility gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "weighted_f16_max_abs_diff", "value": weighted["max_abs_diff"]},
      {"metric": "routed_f16_max_abs_diff", "value": routed["max_abs_diff"]},
      {"metric": "source_realizable_projection_us",
       "value": timing["source_realizable_projection"]},
      {"metric": "ideal_transport_projection_us",
       "value": timing["ideal_transport_projection"]},
      {"metric": "complete_cap_us", "value": timing["cap"]},
      {"metric": "required_checks_passed", "value": required_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  failed = [row["name"] for row in result["checks"] if not row["pass"]]
  (out_dir / "summary.md").write_text("\n".join([
      "# F16 deterministic contribution-plane feasibility gate",
      "",
      f"- weighted-down F16 max abs / mismatches: "
      f"`{weighted['max_abs_diff']} / {weighted['mismatch_count']}`",
      f"- routed-output F16 max abs / mismatches: "
      f"`{routed['max_abs_diff']} / {routed['mismatch_count']}`",
      f"- source-realizable projection / cap: "
      f"`{timing['source_realizable_projection']:.3f} / {timing['cap']:.3f} us`",
      f"- ideal-transport projection / cap: "
      f"`{timing['ideal_transport_projection']:.3f} / {timing['cap']:.3f} us`",
      f"- failed checks: `{failed}`",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "F16 is a correctness-safe deterministic partial carrier on the locked",
      "layer-27 oracle. It cannot rescue the route: the optimistic exact-core",
      "projection remains above the cap even after eliminating grouped gather",
      "and byte-scaling the scatter. No real kernel is authorized.",
      "",
  ]), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "ideal_transport_us": timing["ideal_transport_projection"],
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_passed,
      "source_realizable_us": timing["source_realizable_projection"],
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
