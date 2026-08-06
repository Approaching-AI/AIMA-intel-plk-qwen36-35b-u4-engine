#!/usr/bin/env python3
"""Compare all 40 resident mixed-codec MoE boundaries from one live capture."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-all-layer-mixed-component-gate-v5"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
TOKEN_FILE = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
Q4_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-"
    "20260711Tseq673cleanZ")
PAYLOAD_ARTIFACT = (
    ROOT / "output/all-layer-exact-block-q4q6-prepack-load-"
    "20260711Tseq712cleanZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--llama-source", type=Path, default=LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=LLAMA_BUILD)
  parser.add_argument("--token-file", type=Path, default=TOKEN_FILE)
  parser.add_argument("--q4-artifact", type=Path, default=Q4_ARTIFACT)
  parser.add_argument("--payload-artifact", type=Path,
                      default=PAYLOAD_ARTIFACT)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.repeat, args.timeout_s) <= 0:
    parser.error("jobs, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/all-layer-mixed-component-{stamp}"
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


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def comparison_pass(value: dict[str, Any], count: int) -> bool:
  return (value.get("compared_value_count") == count and
          float(value.get("cosine", float("-inf"))) >=
          COMPONENT_COSINE_MIN and
          float(value.get("relative_l2", float("inf"))) <=
          COMPONENT_RELATIVE_L2_MAX and
          value.get("finite") is True)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  capture = raw / "capture"
  build_dir = raw / "build"
  q4_raw = args.q4_artifact / "raw"
  prep_root = args.payload_artifact / "raw/layers"
  required = [
      args.model, args.env_script, args.cmake, args.cxx, args.token_file,
      args.llama_source / "include/llama.h",
      args.llama_source / "ggml/include/ggml.h",
      args.llama_build / "bin/libllama.so.0.0.1",
      args.llama_build / "bin/libggml.so.0.13.1",
      q4_raw / "gateup.0.bin", q4_raw / "down.0.bin", prep_root,
      ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp",
      ROOT / "engine/tools/grouped_mixed_prefill_all_layer_compare.cpp",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  state = git_state()
  capture_binary = raw / "all-layer-component-capture"
  capture_build = run_env([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}",
      str(ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"),
      f"-L{args.llama_build / 'bin'}",
      f"-Wl,-rpath,{args.llama_build / 'bin'}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1",
      "-Wl,-l:libggml-base.so.0.13.1", "-fopenmp", "-pthread",
      "-o", str(capture_binary)], args)
  write_run(raw, "capture-build", capture_build)
  capture_command = [
      str(capture_binary), "--model", str(args.model),
      "--token-ids-file", str(args.token_file), "--binary-u32-token-file",
      "--token-count", "1024", "--batch-all", "--component-all-layers",
      "--component-through-down", "--out-dir", str(capture),
      "--case-id", "prefill_shape_008k_tile1024_all40_routed",
      "--threads", "16", "--n-ctx", "2048", "--ngl", "0",
      "--top-k", "1", "--predicts-generated-position", "0",
  ]
  capture_run = run_env(capture_command, args) \
      if capture_build["returncode"] == 0 else {
          "command": capture_command, "returncode": 125, "stdout": "",
          "stderr": "capture build failed", "timed_out": False}
  write_run(raw, "capture", capture_run)
  capture_summary = {}
  if (capture / "capture-summary.json").is_file():
    capture_summary = json.loads(
        (capture / "capture-summary.json").read_text(encoding="utf-8"))

  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", "iq36-grouped-mixed-prefill-all-layer-compare"], args) \
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)
  compare_binary = (
      build_dir / "iq36-grouped-mixed-prefill-all-layer-compare")
  compare_command = [
      str(compare_binary), str(q4_raw / "gateup.0.bin"),
      str(q4_raw / "down.0.bin"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      str(prep_root), str(capture), str(args.repeat),
  ]
  compare = run_env(compare_command, args) \
      if capture_run["returncode"] == 0 and build["returncode"] == 0 else {
          "command": compare_command, "returncode": 125, "stdout": "",
          "stderr": "capture or build failed", "timed_out": False}
  write_run(raw, "compare", compare)
  probe = parse_probe(compare)

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False),
      check("locked_model_path", args.model.resolve() == MODEL.resolve()),
      check("single_live_1024_token_all40_capture",
            capture_run["returncode"] == 0 and
            capture_summary.get("token_count") == 1024 and
            capture_summary.get("batch_all") is True and
            capture_summary.get("component_all_layers") is True and
            capture_summary.get("component_through_down") is True and
            capture_summary.get("captured_tensor_count") == 240,
            summary=capture_summary),
      check("all_40_resident_layer_runs_passed",
            compare["returncode"] == 0 and
            probe.get("all_layer_compare_pass") is True and
            probe.get("layer_count") == 40 and
            len(probe.get("per_layer", [])) == 40 and
            sum(row.get("codec") == "Q6_K_EXACT_BLOCK"
                for row in probe.get("per_layer", [])) == 20),
      check("all_167772160_swiglu_values_pass",
            comparison_pass(
                probe.get("all_swiglu_compare", {}), 167_772_160),
            comparison=probe.get("all_swiglu_compare")),
      check("all_671088640_weighted_down_values_pass",
            comparison_pass(
                probe.get("all_weighted_down_compare", {}), 671_088_640),
            comparison=probe.get("all_weighted_down_compare")),
      check("all_83886080_routed_output_values_pass",
            comparison_pass(
                probe.get("all_routed_output_compare", {}), 83_886_080),
            comparison=probe.get("all_routed_output_compare")),
      check("one_native_context_owns_all_real_weights",
            probe.get("context_create_count") == 1 and
            probe.get("program_load_count") == 4 and
            probe.get("resident_weight_bytes") == 21_726_494_720 and
            probe.get("run_count") == 40),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "model": str(args.model), "capture": str(capture),
      "payload_artifact": str(args.payload_artifact),
      "q4_gateup_artifact": str(args.q4_artifact),
      "q4_down_artifact": str(args.q4_artifact),
      "component_accuracy_contract": {
          "cosine_min": COMPONENT_COSINE_MIN,
          "relative_l2_max": COMPONENT_RELATIVE_L2_MAX,
          "finite_outputs_required": True,
      },
      "capture_summary": capture_summary, "component_probe": probe,
      "checks": checks, "required_checks_passed": passed,
      "disposition": (
          "accept_all40_live_input_component_correctness"
          if passed else "reject_all40_mixed_component"),
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
  (out / "summary.md").write_text("\n".join([
      "# All-layer mixed component correctness", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- captured tensors: `{capture_summary.get('captured_tensor_count')}`",
      f"- resident layer runs: `{probe.get('run_count')}`",
      f"- SwiGLU values: `{(probe.get('all_swiglu_compare') or {}).get('compared_value_count')}`",
      f"- weighted-down values: `{(probe.get('all_weighted_down_compare') or {}).get('compared_value_count')}`",
      f"- routed-output values: `{(probe.get('all_routed_output_compare') or {}).get('compared_value_count')}`",
      "", "All inputs came from one live 1024-token model evaluation, but",
      "native component outputs are not yet chained into the next layer.",
      "Teacher-forced tokens and product speed remain open.", "",
  ]), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "pass": passed,
                    "capture_summary": capture_summary,
                    "aggregate": {
                        "swiglu": probe.get("all_swiglu_compare"),
                        "weighted_down": probe.get(
                            "all_weighted_down_compare"),
                        "routed_output": probe.get(
                            "all_routed_output_compare")}}, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
