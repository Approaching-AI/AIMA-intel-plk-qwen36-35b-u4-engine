#!/usr/bin/env python3
"""Gate exact Q6_K per-16 accuracy and projected mixed-prefill timing."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-q6-exact-per16-prefill-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CAPTURE = (
    ROOT / "output/all-layer-mixed-component-20260711Tseq671cleanZ/"
    "raw/capture/payloads")
BASELINE = (
    ROOT / "output/grouped-s8-u8-q6-prefill-gate-"
    "20260711Tseq669cleanZ")
TENSOR = "blk.39.ffn_down_exps.weight"
LAYER = 39
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002
MIXED_CAP_US = 390_857.440
NOISE_US = 1_954.2872


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--baseline", type=Path, default=BASELINE)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.repeat, args.timeout_s) <= 0:
    parser.error("jobs, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/q6-exact-per16-prefill-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {"commit": git_output("rev-parse", "HEAD"),
          "dirty": bool(dirty), "dirty_paths": dirty.splitlines()}


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True}


def run_env(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      f"export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && "
      f"{shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"], "returncode": result["returncode"],
      "timed_out": result["timed_out"]})
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def payload(capture: Path, stem: str) -> Path:
  matches = sorted(capture.glob(f"{stem}__tok1023__ord*.bin"))
  if len(matches) != 1:
    raise ValueError(f"expected one payload for {stem}, found {matches}")
  return matches[0]


def load_json_line(path: Path) -> dict[str, Any]:
  for line in reversed(path.read_text(encoding="utf-8").splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  raise ValueError(f"no JSON object in {path}")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def accuracy_pass(probe: dict[str, Any]) -> bool:
  comparison = probe.get("comparison", {})
  return (probe.get("correctness_pass") is True and
          comparison.get("compared_value_count") == 16_777_216 and
          comparison.get("finite") is True and
          float(comparison.get("cosine", float("-inf"))) >=
          COMPONENT_COSINE_MIN and
          float(comparison.get("relative_l2", float("inf"))) <=
          COMPONENT_RELATIVE_L2_MAX)


def baseline_row(baseline: Path, label: str) -> dict[str, Any]:
  result = json.loads((baseline / "result.json").read_text(encoding="utf-8"))
  mixed = next(row for row in result["mixed_budget"]["rows"]
               if row["label"] == label)
  envelope = load_json_line(baseline / f"raw/q6-envelope-{label}.stdout")
  q6_layers = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
               22, 25, 28, 31, 34, 35, 36, 37, 38, 39}
  rows = [row for row in envelope["per_layer"]
          if int(row["layer"]) in q6_layers]
  layer = next(row for row in rows if int(row["layer"]) == LAYER)
  return {
      "label": label,
      "mixed_complete_sum_us": float(mixed["mixed_complete_sum_us"]),
      "q6_down_sum_us": sum(float(row["stage_us"]["down"])
                            for row in rows),
      "q6_task_count": sum(int(row["task_count"]) for row in rows),
      "layer39_task_count": int(layer["task_count"]),
      "layer39_surrogate_down_us": float(layer["stage_us"]["down"]),
  }


def projection(probe: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
  exact_us = float(probe.get("kernel_min_us", float("inf")))
  exact_tasks = int(probe.get("work_tile_count", 0))
  per_task_us = exact_us / exact_tasks if exact_tasks else float("inf")
  projected_down = per_task_us * baseline["q6_task_count"]
  projected_mixed = (baseline["mixed_complete_sum_us"] -
                     baseline["q6_down_sum_us"] + projected_down)
  return {
      **baseline,
      "exact_layer39_kernel_us": exact_us,
      "exact_us_per_task": per_task_us,
      "projected_exact_q6_down_sum_us": projected_down,
      "projected_mixed_complete_sum_us": projected_mixed,
      "mixed_cap_us": MIXED_CAP_US,
      "noise_us": NOISE_US,
      "headroom_us": MIXED_CAP_US - projected_mixed,
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  build_dir = raw / "build"
  state = git_state()
  required = [
      args.model, args.env_script, args.cmake,
      args.baseline / "result.json",
      args.baseline / "raw/q6-envelope-primary.stdout",
      args.baseline / "raw/q6-envelope-confirm.stdout",
      ROOT / "engine/tools/grouped_s8_u8_q6_surrogate_down.cpp",
      ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl",
  ]
  payloads = {
      "swiglu": payload(args.capture, f"ffn_moe_swiglu-{LAYER}"),
      "topk": payload(args.capture, f"ffn_moe_topk-{LAYER}"),
      "router": payload(args.capture, f"ffn_moe_weights_norm-{LAYER}"),
      "oracle": payload(args.capture, f"ffn_moe_down-{LAYER}"),
  }
  missing = [str(path) for path in [*required, *payloads.values()]
             if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", "iq36-grouped-s8-u8-q6-surrogate-down"], args) \
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)
  binary = build_dir / "iq36-grouped-s8-u8-q6-surrogate-down"
  command = [
      str(binary), "--model", str(args.model), "--tensor", TENSOR,
      "--kernel", str(
          ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      "--swiglu", str(payloads["swiglu"]), "--topk", str(payloads["topk"]),
      "--topk-stride", "1024", "--router-weights", str(payloads["router"]),
      "--oracle", str(payloads["oracle"]), "--warmup", "3",
      "--repeat", str(args.repeat), "--kernel-cap-us", "10000",
      "--m-tile", "16", "--flatten-output-tasks", "--exact-per16",
  ]
  runs: list[dict[str, Any]] = []
  probes: list[dict[str, Any]] = []
  for label in ("primary", "confirm"):
    run_result = run_env(command, args) if build["returncode"] == 0 else {
        "command": command, "returncode": 125, "stdout": "",
        "stderr": "build failed", "timed_out": False}
    write_run(raw, label, run_result)
    runs.append(run_result)
    probes.append(parse_probe(run_result))
  f16_diagnostic = run_env([*command, "--round-swiglu-f16"], args) \
      if build["returncode"] == 0 else {
          "command": [*command, "--round-swiglu-f16"],
          "returncode": 125, "stdout": "", "stderr": "build failed",
          "timed_out": False}
  write_run(raw, "f16-handoff-diagnostic", f16_diagnostic)
  f16_probe = parse_probe(f16_diagnostic)

  baselines = [baseline_row(args.baseline, label)
               for label in ("primary", "confirm")]
  projections = [projection(probe, baseline)
                 for probe, baseline in zip(probes, baselines)]
  accuracy_checks = [
      check(f"{label}_all_values_meet_component_contract",
            run_result["returncode"] == 0 and accuracy_pass(probe),
            comparison=probe.get("comparison"),
            prepack=probe.get("prepack"))
      for label, run_result, probe in zip(
          ("primary", "confirm"), runs, probes)]
  timing_checks = [
      check(f"{row['label']}_projected_mixed_sum_clears_cap_beyond_noise",
            row["headroom_us"] > NOISE_US, projection=row)
      for row in projections]
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False),
      check("locked_model_and_worst_q6_layer",
            args.model.resolve() == MODEL.resolve() and TENSOR.startswith(
                f"blk.{LAYER}.")),
      check("clean_exact_per16_build",
            configure["returncode"] == build["returncode"] == 0),
      check("same_flat_task_count_as_layer39_schedule",
            all(int(probe.get("work_tile_count", -1)) ==
                    baseline["layer39_task_count"]
                for probe, baseline in zip(probes, baselines))),
      check("exact_q6_weight_reconstruction",
            all(float((probe.get("prepack") or {}).get(
                "relative_l2_weight_error", float("inf"))) == 0.0
                for probe in probes)),
      check("f16_swiglu_handoff_is_contract_blocker",
            f16_diagnostic["returncode"] == 2 and
            (f16_probe.get("comparison") or {}).get("finite") is True and
            float((f16_probe.get("comparison") or {}).get(
                "relative_l2", 0.0)) > COMPONENT_RELATIVE_L2_MAX,
            comparison=f16_probe.get("comparison")),
      *accuracy_checks,
      *timing_checks,
  ]
  accuracy_passed = all(row["pass"] for row in accuracy_checks)
  timing_passed = all(row["pass"] for row in timing_checks)
  accepted = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": iso_now(), "git": state, "model": str(args.model),
      "tensor": TENSOR, "capture": str(args.capture),
      "baseline": str(args.baseline),
      "component_accuracy_contract": {
          "cosine_min": COMPONENT_COSINE_MIN,
          "relative_l2_max": COMPONENT_RELATIVE_L2_MAX,
          "finite_outputs_required": True,
      },
      "representation": {
          "name": "exact Q6_K signed values plus F32 per-16 scales",
          "resident_bytes": 335_544_320,
          "dpas_k32_calls_per_source_k32": 2,
      },
      "probes": {"primary": probes[0], "confirm": probes[1]},
      "f16_swiglu_handoff_diagnostic": f16_probe,
      "mixed_timing_projection": projections, "checks": checks,
      "accuracy_checks_passed": accuracy_passed,
      "timing_checks_passed": timing_passed,
      "required_checks_passed": accepted,
      "disposition": (
          "accept_exact_per16_q6_prefill_carrier" if accepted else
          "reject_exact_per16_as_product_timing_carrier_keep_accuracy_reference"
          if accuracy_passed else "reject_exact_per16_accuracy"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": accuracy_checks,
      "required_checks_passed": accuracy_passed,
      "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": result["created_at"],
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": accepted,
      "speedup_claims_allowed": False})
  lines = [
      "# Exact Q6_K per-16 grouped-prefill gate", "",
      f"- component accuracy passed: `{str(accuracy_passed).lower()}`",
      f"- mixed timing passed: `{str(timing_passed).lower()}`",
      f"- F16 SwiGLU diagnostic relL2: `"
      f"{(f16_probe.get('comparison') or {}).get('relative_l2')}`",
  ]
  for row in projections:
    lines.append(
        f"- {row['label']}: exact layer-39 `{row['exact_layer39_kernel_us']:.3f} us`; "
        f"projected mixed `{row['projected_mixed_complete_sum_us']:.3f} us`; "
        f"headroom `{row['headroom_us']:.3f} us`")
  lines.extend(["", "Exact per-16 is an accuracy reference. Projection uses",
                "the measured flat-task count and the clean seq669 paired",
                "non-down stage sums; it is not a product speed claim.", ""])
  (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "accepted": accepted,
                    "accuracy_passed": accuracy_passed,
                    "timing_passed": timing_passed,
                    "projection": projections}, sort_keys=True))
  return 0 if accepted else 2


if __name__ == "__main__":
  raise SystemExit(main())
