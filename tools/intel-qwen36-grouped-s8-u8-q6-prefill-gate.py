#!/usr/bin/env python3
"""Gate the grouped Q6-down prefill carrier and mixed 40-layer budget."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-grouped-s8-u8-q6-prefill-gate-v1"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
Q4_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-"
    "20260711Tseq673cleanZ")
CAPTURE = ROOT / "output/q6-layer7-prefill-capture-20260711Tseq669trialZ"
Q4_LAYERS = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18,
             20, 21, 23, 24, 26, 27, 29, 30, 32, 33]
Q6_LAYERS = [0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
             22, 25, 28, 31, 34, 35, 36, 37, 38, 39]
ROUTED_LAYER_CAP_US = 9_771.436
MIXED_40_LAYER_CAP_US = 40 * ROUTED_LAYER_CAP_US
NOISE_FRACTION = 0.005
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--tensor-index", type=Path, default=TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--q4-artifact", type=Path, default=Q4_ARTIFACT)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--envelope-repeat", type=int, default=5)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.repeat, args.envelope_repeat, args.jobs, args.timeout_s) <= 0:
    parser.error("repeat, envelope-repeat, jobs, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/grouped-s8-u8-q6-prefill-gate-{stamp}"
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


def run(command: list[str], timeout_s: int,
        cwd: Path = ROOT) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True}


def run_env(command: list[str], args: argparse.Namespace,
            extra_env: dict[str, str] | None = None,
            cwd: Path = ROOT) -> dict[str, Any]:
  exports = {"INTEL_FORCE_PROBE": "b080", "DNNL_VERBOSE": "0",
             **(extra_env or {})}
  export_text = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in exports.items())
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      f"export {export_text} && {shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s, cwd)


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


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def tensor_rows(path: Path) -> dict[str, dict[str, Any]]:
  rows: dict[str, dict[str, Any]] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      row = json.loads(line)
      rows[str(row["name"])] = row
  return rows


def payload(capture: Path, stem: str) -> Path:
  matches = sorted((capture / "payloads").glob(f"{stem}__tok1023__ord*.bin"))
  if len(matches) != 1:
    raise ValueError(f"expected one payload for {stem}, found {matches}")
  return matches[0]


def q4_sum(probe: dict[str, Any]) -> float:
  wanted = set(Q4_LAYERS)
  return sum(float(row["full_complete_minimum_us"])
             for row in probe.get("per_layer", [])
             if int(row["layer"]) in wanted)


def compare_pass(probe: dict[str, Any], key: str, count: int) -> bool:
  row = probe.get(key, {})
  return (row.get("compared_value_count") == count and
          float(row.get("cosine", float("-inf"))) >=
          COMPONENT_COSINE_MIN and
          float(row.get("relative_l2", float("inf"))) <=
          COMPONENT_RELATIVE_L2_MAX and
          row.get("finite") is True)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  prepack = raw / "prepacked"
  prepack.mkdir()
  build_dir = raw / "build"
  q4_raw = args.q4_artifact / "raw"
  required = [
      args.model, args.tensor_index, args.env_script, args.cmake, args.cxx,
      args.capture / "capture-summary.json",
      args.capture / "tensor-dumps.jsonl",
      q4_raw / "gateup.0.bin", q4_raw / "down.0.bin",
      q4_raw / "offline-prepack-generator", q4_raw / "prepacked",
      q4_raw / "schedule-probes",
      ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl",
      ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  state = git_state()
  rows = tensor_rows(args.tensor_index)
  gateup = rows["blk.7.ffn_gate_up_exps.weight"]
  q6_down = rows["blk.7.ffn_down_exps.weight"]
  dummy_q4_down = rows["blk.27.ffn_down_exps.weight"]
  capture_summary = json.loads(
      (args.capture / "capture-summary.json").read_text(encoding="utf-8"))
  input_path = payload(args.capture, "attn_post_norm-7")
  topk_path = payload(args.capture, "ffn_moe_topk-7")
  router_path = payload(args.capture, "ffn_moe_weights_norm-7")
  swiglu_path = payload(args.capture, "ffn_moe_swiglu-7")
  down_oracle = payload(args.capture, "ffn_moe_down-7")
  moe_oracle = payload(args.capture, "ffn_moe_out-7")
  topk_stride = 1024

  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  targets = [
      "iq36-grouped-s8-u8-q6-prefill-resident-smoke",
      "iq36-grouped-s8-u8-q6-prefill-schedule-envelope-smoke",
      "iq36-grouped-s8-u4-prefill-schedule-envelope-smoke",
      "iq36-grouped-s8-u8-q6-surrogate-down",
  ]
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", *targets], args) if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)

  q6_tool = raw / "grouped-s8-u8-q6-surrogate-down"
  tool_build = run_env([
      str(args.cxx), "-std=c++17", "-O3", "-fopenmp",
      str(ROOT / "engine/tools/grouped_s8_u8_q6_surrogate_down.cpp"),
      f"-I{ROOT / 'engine/include'}", "-lOpenCL", "-ldl", "-lpthread",
      str(ROOT / "engine/src/gguf_loader.cpp"), "-o", str(q6_tool)], args)
  write_run(raw, "q6-tool-build", tool_build)

  q4_prep_command = [
      str(q4_raw / "offline-prepack-generator"),
      "--model", str(args.model),
      "--weight-offset", str(gateup["absolute_offset"]),
      "--weight-bytes", str(gateup["nbytes"]),
      "--down-weight-offset", str(dummy_q4_down["absolute_offset"]),
      "--down-weight-bytes", str(dummy_q4_down["nbytes"]),
      "--grouped-gateup-binary", str(q4_raw / "gateup.0.bin"),
      "--grouped-down-binary", str(q4_raw / "down.0.bin"),
      "--dump-prepacked-dir", str(prepack), "--prepack-only",
  ]
  q4_prep = run_env(q4_prep_command, args, {
      "IQ36_GENERATE_S8_GROUPED": "1",
      "DNNL_PRIMITIVE_CACHE_CAPACITY": "0"})
  write_run(raw, "q4-prepack", q4_prep)

  q6_base_command = [
      str(q6_tool), "--model", str(args.model),
      "--kernel", str(
          ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      "--swiglu", str(swiglu_path), "--topk", str(topk_path),
      "--topk-stride", str(topk_stride),
      "--router-weights", str(router_path), "--oracle", str(down_oracle),
      "--warmup", "3", "--repeat", str(args.repeat),
      "--kernel-cap-us", "5000", "--m-tile", "16",
      "--f16-weight-scales", "--flatten-output-tasks",
  ]
  q6_primary = run_env(
      [*q6_base_command, "--dump-prepacked-dir", str(prepack)], args) \
      if tool_build["returncode"] == 0 and q4_prep["returncode"] == 0 else {
          "command": q6_base_command, "returncode": 125, "stdout": "",
          "stderr": "Q6 tool build or Q4 prepack failed", "timed_out": False}
  write_run(raw, "q6-primary", q6_primary)
  q6_confirm = run_env(q6_base_command, args) \
      if q6_primary["returncode"] == 0 else {
          "command": q6_base_command, "returncode": 125, "stdout": "",
          "stderr": "Q6 primary failed", "timed_out": False}
  write_run(raw, "q6-confirm", q6_confirm)
  q6_primary_probe = parse_probe(q6_primary)
  q6_confirm_probe = parse_probe(q6_confirm)

  resident = build_dir / "iq36-grouped-s8-u8-q6-prefill-resident-smoke"
  resident_command = [
      str(resident), str(prepack), str(q4_raw / "gateup.0.bin"),
      str(q4_raw / "down.0.bin"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      str(input_path), str(topk_path), str(topk_stride), str(router_path),
      str(swiglu_path), str(down_oracle), str(moe_oracle), str(args.repeat),
  ]
  resident_runs = []
  for label in ("primary", "confirm"):
    result = run_env(resident_command, args) \
        if build["returncode"] == 0 and q6_primary["returncode"] == 0 else {
            "command": resident_command, "returncode": 125, "stdout": "",
            "stderr": "build or prepack failed", "timed_out": False}
    resident_runs.append(result)
    write_run(raw, f"resident-{label}", result)
  resident_probes = [parse_probe(row) for row in resident_runs]

  q4_envelope_binary = (
      build_dir / "iq36-grouped-s8-u4-prefill-schedule-envelope-smoke")
  q4_envelope_command = [
      str(q4_envelope_binary), str(q4_raw / "prepacked"),
      str(q4_raw / "gateup.0.bin"), str(q4_raw / "down.0.bin"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(input_path), str(q4_raw / "schedule-probes"), str(router_path),
      str(args.envelope_repeat),
  ]
  q6_envelope_binary = (
      build_dir / "iq36-grouped-s8-u8-q6-prefill-schedule-envelope-smoke")
  q6_envelope_command = [
      str(q6_envelope_binary), str(prepack),
      str(q4_raw / "gateup.0.bin"), str(q4_raw / "down.0.bin"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      str(input_path), str(q4_raw / "schedule-probes"), str(router_path),
      str(args.envelope_repeat),
  ]
  envelope_rows: list[dict[str, Any]] = []
  for label in ("primary", "confirm"):
    q4_run = run_env(q4_envelope_command, args)
    q6_run = run_env(q6_envelope_command, args)
    write_run(raw, f"q4-envelope-{label}", q4_run)
    write_run(raw, f"q6-envelope-{label}", q6_run)
    q4_probe = parse_probe(q4_run)
    q6_probe = parse_probe(q6_run)
    q4_complete = q4_sum(q4_probe)
    q6_complete = float(q6_probe.get("complete_sum_us", float("inf")))
    mixed = q4_complete + q6_complete
    envelope_rows.append({
        "label": label, "q4_complete_sum_us": q4_complete,
        "q6_complete_sum_us": q6_complete,
        "mixed_complete_sum_us": mixed,
        "headroom_us": MIXED_40_LAYER_CAP_US - mixed,
        "q4_probe": q4_probe, "q6_probe": q6_probe,
        "returncodes": [q4_run["returncode"], q6_run["returncode"]],
    })

  noise_us = MIXED_40_LAYER_CAP_US * NOISE_FRACTION
  q6_checks = [
      check(f"q6_{label}_all_values_correct",
            result["returncode"] == 0 and
            probe.get("correctness_pass") is True and
            compare_pass(probe, "comparison", 16_777_216),
            kernel_min_us=probe.get("kernel_min_us"))
      for label, result, probe in zip(
          ("primary", "confirm"), (q6_primary, q6_confirm),
          (q6_primary_probe, q6_confirm_probe))]
  resident_checks = [
      check(f"resident_{label}_all_values_correct",
            result["returncode"] == 0 and
            probe.get("integration_pass") is True and
            probe.get("maps_native_only") is True and
            compare_pass(probe, "swiglu_compare", 4_194_304) and
            compare_pass(probe, "weighted_down_compare", 16_777_216) and
            compare_pass(probe, "moe_compare", 2_097_152),
            complete_minimum_us=probe.get("complete_minimum_us"))
      for label, result, probe in zip(
          ("primary", "confirm"), resident_runs, resident_probes)]
  envelope_checks = [
      check(f"mixed_40_layer_{row['label']}_clears_cap_beyond_noise",
            row["returncodes"] == [0, 0] and
            row["q4_probe"].get("schedule_envelope_pass") is True and
            row["q6_probe"].get("q6_schedule_envelope_pass") is True and
            row["headroom_us"] > noise_us,
            mixed_complete_sum_us=row["mixed_complete_sum_us"],
            cap_us=MIXED_40_LAYER_CAP_US,
            headroom_us=row["headroom_us"], noise_us=noise_us)
      for row in envelope_rows]
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False),
      check("locked_model_and_layer7_q6_tensor",
            args.model.resolve() == MODEL.resolve() and
            q6_down.get("type") == 14 and
            q6_down.get("nbytes") == 220_200_960),
      check("locked_codec_split", len(Q4_LAYERS) == len(Q6_LAYERS) == 20 and
            sorted(Q4_LAYERS + Q6_LAYERS) == list(range(40))),
      check("layer7_1024_token_capture",
            capture_summary.get("component_layer") == 7 and
            capture_summary.get("token_count") == 1024 and
            capture_summary.get("component_through_down") is True),
      check("clean_build_and_capture_free_prepack",
            configure["returncode"] == build["returncode"] ==
            tool_build["returncode"] == q4_prep["returncode"] == 0),
      *q6_checks, *resident_checks, *envelope_checks,
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "model": str(args.model), "capture": str(args.capture),
      "representation": {
          "name": "Q6_K to centered-U8 plus F16 group scales",
          "resident_down_bytes": 285_212_672,
          "m_tile": 16, "task_encoding_bytes": 4,
      },
      "component_accuracy_contract": {
          "cosine_min": COMPONENT_COSINE_MIN,
          "relative_l2_max": COMPONENT_RELATIVE_L2_MAX,
          "finite_outputs_required": True,
      },
      "q6_component": {"primary": q6_primary_probe,
                        "confirm": q6_confirm_probe},
      "resident_component": {"primary": resident_probes[0],
                              "confirm": resident_probes[1]},
      "mixed_budget": {
          "layer_cap_us": ROUTED_LAYER_CAP_US,
          "mixed_40_layer_cap_us": MIXED_40_LAYER_CAP_US,
          "noise_fraction": NOISE_FRACTION, "noise_us": noise_us,
          "rows": [{key: value for key, value in row.items()
                    if key not in {"q4_probe", "q6_probe"}}
                   for row in envelope_rows],
      },
      "checks": checks, "required_checks_passed": passed,
      "disposition": (
          "accept_grouped_q6_prefill_and_mixed_40_layer_budget"
          if passed else "reject_grouped_q6_prefill"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False})
  lines = [
      "# Grouped Q6 prefill and mixed-codec gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- mixed 40-layer cap: `{MIXED_40_LAYER_CAP_US:.3f} us`",
      f"- noise guard: `{noise_us:.3f} us`",
  ]
  for row in envelope_rows:
    lines.append(
        f"- {row['label']}: `{row['mixed_complete_sum_us']:.3f} us`, "
        f"headroom `{row['headroom_us']:.3f} us`")
  lines.extend(["", "This promotes only the grouped Q6 carrier and the mixed",
                "40-layer feasibility boundary. It is not live model-state,",
                "teacher-forced, token, context-ladder, or product speed evidence.",
                ""])
  (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "pass": passed,
                    "mixed_budget": result["mixed_budget"]},
                   sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
