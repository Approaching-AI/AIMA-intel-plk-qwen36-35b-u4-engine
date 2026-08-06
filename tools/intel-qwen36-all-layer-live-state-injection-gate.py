#!/usr/bin/env python3
"""Inject all 40 native routed-MoE outputs into one live reference graph."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-all-layer-live-state-injection-gate-v4"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
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
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ")
PAYLOAD_ARTIFACT = (
    ROOT / "output/all-layer-exact-block-q4q6-prepack-load-"
    "20260711Tseq712cleanZ")
COMPONENT_ARTIFACT = (
    ROOT / "output/all-layer-exact-block-q4q6-crdiv-component-"
    "20260711Tseq720cleanZ")
TEACHER_TOKENS = [264, 264, 271, 248068, 198, 8160, 579, 264]
VOCABULARY_SIZE = 248_320
INJECTION_VALUE_COUNT = 83_886_080
RESIDENT_BYTES = 21_726_494_720
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002
KL_DIVERGENCE_MAX = 0.005


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--llama-source", type=Path, default=LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=LLAMA_BUILD)
  parser.add_argument("--token-file", type=Path, default=TOKEN_FILE)
  parser.add_argument("--q4-artifact", type=Path, default=Q4_ARTIFACT)
  parser.add_argument("--q4-down-artifact", type=Path, default=Q4_ARTIFACT)
  parser.add_argument("--q4-down-binary-name", default="down.0.bin")
  parser.add_argument("--payload-artifact", type=Path,
                      default=PAYLOAD_ARTIFACT)
  parser.add_argument("--component-artifact", type=Path,
                      default=COMPONENT_ARTIFACT)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--apply-codec", choices=["all", "q4", "q6"],
                      default="all")
  parser.add_argument("--teacher-forced", action="store_true")
  parser.add_argument("--q4-f32-contributions", action="store_true")
  parser.add_argument("--q4-exact-block", action=argparse.BooleanOptionalAction,
                      default=True)
  parser.add_argument("--layer-start", type=int, default=0)
  parser.add_argument("--layer-end", type=int, default=40)
  parser.add_argument("--reference-swiglu", action="store_true")
  parser.add_argument("--reference-down", action="store_true")
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if Path(args.q4_down_binary_name).name != args.q4_down_binary_name:
    parser.error("q4 down binary name must be a basename")
  if not 0 <= args.layer_start <= args.layer_end <= 40:
    parser.error("layer range must satisfy 0 <= start <= end <= 40")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/all-layer-live-state-injection-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected object")
  return value


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {"commit": git_output("rev-parse", "HEAD"),
          "dirty": bool(dirty), "dirty_paths": dirty.splitlines()}


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  begin = time.monotonic()
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False, "wall_s": time.monotonic() - begin}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True, "wall_s": time.monotonic() - begin}


def run_env(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && "
      f"{shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"], "returncode": result["returncode"],
      "timed_out": result["timed_out"], "wall_s": result["wall_s"]})
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def read_floats(path: Path) -> array[float]:
  values = array("f")
  with path.open("rb") as stream:
    values.fromfile(stream, path.stat().st_size // values.itemsize)
  if sys.byteorder != "little":
    values.byteswap()
  return values


def compare_logits(baseline_path: Path, injected_path: Path) -> dict[str, Any]:
  baseline = read_floats(baseline_path)
  injected = read_floats(injected_path)
  if len(baseline) != len(injected) or not baseline:
    return {"finite": False, "value_count": 0}
  finite = all(math.isfinite(value) for value in baseline) and all(
      math.isfinite(value) for value in injected)
  baseline_max = max(baseline)
  injected_max = max(injected)
  baseline_exp = [math.exp(value - baseline_max) for value in baseline]
  baseline_sum = math.fsum(baseline_exp)
  injected_sum = math.fsum(
      math.exp(value - injected_max) for value in injected)
  baseline_log_z = baseline_max + math.log(baseline_sum)
  injected_log_z = injected_max + math.log(injected_sum)
  kld = math.fsum(
      (weight / baseline_sum) *
      ((left - baseline_log_z) - (right - injected_log_z))
      for weight, left, right in zip(baseline_exp, baseline, injected))
  error_squared = math.fsum(
      (left - right) ** 2 for left, right in zip(baseline, injected))
  baseline_squared = math.fsum(value * value for value in baseline)
  injected_squared = math.fsum(value * value for value in injected)
  dot = math.fsum(left * right for left, right in zip(baseline, injected))
  def top_two(values: array[float]) -> tuple[int, int]:
    first = -1
    second = -1
    for index, value in enumerate(values):
      if first < 0 or value > values[first]:
        second = first
        first = index
      elif second < 0 or value > values[second]:
        second = index
    return first, second

  baseline_top1, baseline_top2 = top_two(baseline)
  injected_top1, injected_top2 = top_two(injected)
  return {
      "value_count": len(baseline), "finite": finite,
      "baseline_top1": baseline_top1,
      "baseline_top2": baseline_top2,
      "baseline_top1_margin": (
          baseline[baseline_top1] - baseline[baseline_top2]),
      "injected_top1": injected_top1,
      "injected_top2": injected_top2,
      "injected_top1_margin": (
          injected[injected_top1] - injected[injected_top2]),
      "kl_divergence": max(0.0, kld),
      "cosine": dot / math.sqrt(baseline_squared * injected_squared),
      "relative_l2": math.sqrt(error_squared / baseline_squared),
      "max_abs_diff": max(
          abs(left - right) for left, right in zip(baseline, injected)),
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  build_dir = raw / "build"
  build_dir.mkdir()
  baseline_dir = raw / "baseline"
  injected_dir = raw / "injected"
  q4_raw = args.q4_artifact / "raw"
  q4_down_raw = args.q4_down_artifact / "raw"
  q4_down_binary = q4_down_raw / args.q4_down_binary_name
  prep_root = args.payload_artifact / "raw/layers"
  source = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
  runtime_source = ROOT / "engine/src/grouped_s8_u4_prefill_runtime.cpp"
  support_kernel = (
      ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl")
  q6_kernel = ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"
  required = [
      args.model, args.env_script, args.cxx, args.token_file,
      args.llama_source / "include/llama.h",
      args.llama_source / "ggml/include/ggml.h",
      args.llama_build / "bin/libllama.so.0.0.1", source, runtime_source,
      support_kernel, q6_kernel, q4_raw / "gateup.0.bin",
      q4_down_binary, prep_root,
      args.component_artifact / "result.json"]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  state = git_state()
  component = load_json(args.component_artifact / "result.json")
  binary = build_dir / "live-injection-capture"
  build = run_env([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DIQ36_GROUPED_LIVE_INJECTION", "-DGGML_BACKEND_SHARED",
      "-DGGML_SHARED", "-DGGML_USE_CPU", "-DLLAMA_SHARED",
      f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}",
      f"-I{ROOT / 'engine/include'}", str(source), str(runtime_source),
      f"-L{args.llama_build / 'bin'}",
      f"-Wl,-rpath,{args.llama_build / 'bin'}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1",
      "-Wl,-l:libggml-base.so.0.13.1", "-fopenmp", "-pthread",
      "-lOpenCL", "-o", str(binary)], args)
  write_run(raw, "build", build)

  common = [
      str(binary), "--model", str(args.model), "--token-ids-file",
      str(args.token_file), "--binary-u32-token-file", "--token-count",
      "1024", "--batch-all", "--threads", "16", "--n-ctx", "2048",
      "--ngl", "0", "--top-k", "16", "--predicts-generated-position",
      "0", "--live-injection-boundaries", "--no-tensor-dumps",
      "--full-logits", "--generate-count", "8"]
  if args.teacher_forced:
    for token in TEACHER_TOKENS:
      common.extend(["--generate-teacher-token", str(token)])
  if args.q4_f32_contributions:
    common.append("--inject-q4-f32-contributions")
  if args.q4_exact_block:
    common.append("--inject-q4-exact-block")
  common.extend(["--inject-layer-start", str(args.layer_start),
                 "--inject-layer-end", str(args.layer_end)])
  if args.reference_swiglu:
    common.append("--inject-reference-swiglu")
  if args.reference_down:
    common.append("--inject-reference-down")
  baseline_command = [
      *common, "--out-dir", str(baseline_dir), "--case-id",
      "live-state-baseline-1024"]
  baseline = run_env(baseline_command, args) if build["returncode"] == 0 else {
      "command": baseline_command, "returncode": 125, "stdout": "",
      "stderr": "build failed", "timed_out": False, "wall_s": 0.0}
  write_run(raw, "baseline", baseline)
  injected_command = [
      *common, "--out-dir", str(injected_dir), "--case-id",
      "live-state-injected-1024", "--inject-prep-root", str(prep_root),
      "--inject-gateup-binary", str(q4_raw / "gateup.0.bin"),
      "--inject-down-binary", str(q4_down_binary),
      "--inject-support-kernel", str(support_kernel),
      "--inject-q6-kernel", str(q6_kernel),
      "--inject-apply-codec", args.apply_codec]
  injected = run_env(injected_command, args) \
      if baseline["returncode"] == 0 else {
          "command": injected_command, "returncode": 125, "stdout": "",
          "stderr": "baseline failed", "timed_out": False, "wall_s": 0.0}
  write_run(raw, "injected", injected)

  injection_summary = {}
  if (injected_dir / "live-injection-summary.json").is_file():
    injection_summary = load_json(
        injected_dir / "live-injection-summary.json")
  distribution = {}
  baseline_logits = baseline_dir / "full-logits.f32.bin"
  injected_logits = injected_dir / "full-logits.f32.bin"
  if baseline_logits.is_file() and injected_logits.is_file():
    distribution = compare_logits(baseline_logits, injected_logits)
  ldd = run(["ldd", str(binary)], args.timeout_s) if binary.is_file() else {
      "command": ["ldd", str(binary)], "returncode": 125, "stdout": "",
      "stderr": "binary missing", "timed_out": False, "wall_s": 0.0}
  write_run(raw, "ldd", ldd)
  ldd_lower = str(ldd["stdout"]).lower()
  per_layer = injection_summary.get("per_layer", [])
  q6_layers = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
               22, 25, 28, 31, 34, 35, 36, 37, 38, 39}
  expected_applied_layers = sum(
      args.apply_codec == "all" or
      (args.apply_codec == "q6" and layer in q6_layers) or
      (args.apply_codec == "q4" and layer not in q6_layers)
      for layer in range(args.layer_start, args.layer_end))
  baseline_tokens = {}
  injected_tokens = {}
  if (baseline_dir / "generated-tokens.json").is_file():
    baseline_tokens = load_json(baseline_dir / "generated-tokens.json")
  if (injected_dir / "generated-tokens.json").is_file():
    injected_tokens = load_json(injected_dir / "generated-tokens.json")
  teacher_forced_ladder: list[dict[str, Any]] = []
  if args.teacher_forced:
    for position in range(len(TEACHER_TOKENS)):
      name = f"continuation-logits-{position:03d}.f32.bin"
      baseline_step = baseline_dir / name
      injected_step = injected_dir / name
      comparison = (
          compare_logits(baseline_step, injected_step)
          if baseline_step.is_file() and injected_step.is_file() else {})
      teacher_forced_ladder.append({
          "position": position,
          "fed_token": TEACHER_TOKENS[position],
          **comparison,
      })
  first_unstable_position = next((
      row["position"] for row in teacher_forced_ladder
      if (row.get("finite") is not True or
          row.get("baseline_top1") != row.get("injected_top1") or
          float(row.get("kl_divergence", float("inf"))) >
              KL_DIVERGENCE_MAX)), None)
  teacher_forced_contract_passed = (
      len(teacher_forced_ladder) == len(TEACHER_TOKENS) and
      first_unstable_position is None)

  continuation_check = (
      check("teacher_forced_shared_prefix_distribution_ladder_passes",
            teacher_forced_contract_passed and
            baseline_tokens.get("teacher_forced") is True and
            injected_tokens.get("teacher_forced") is True and
            baseline_tokens.get("fed_token_ids") == TEACHER_TOKENS and
            injected_tokens.get("fed_token_ids") == TEACHER_TOKENS,
            required_kl_divergence_max=KL_DIVERGENCE_MAX,
            first_unstable_position=first_unstable_position,
            baseline_predictions=baseline_tokens.get("token_ids"),
            injected_predictions=injected_tokens.get("token_ids"),
            ladder=teacher_forced_ladder)
      if args.teacher_forced else
      check("eight_greedy_continuation_tokens_match_exactly",
            baseline_tokens.get("count") == 8 and
            injected_tokens.get("count") == 8 and
            baseline_tokens.get("token_ids") ==
                injected_tokens.get("token_ids"),
            baseline=baseline_tokens.get("token_ids"),
            injected=injected_tokens.get("token_ids")))

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("locked_model_and_clean_all40_component_prerequisite",
            args.model.resolve() == MODEL.resolve() and
            component.get("required_checks_passed") is True and
            component.get("git", {}).get("dirty") is False),
      check("live_injection_harness_builds", build["returncode"] == 0),
      check("paired_reference_evaluations_complete",
            baseline["returncode"] == 0 and injected["returncode"] == 0,
            baseline_wall_s=baseline["wall_s"],
            injected_wall_s=injected["wall_s"]),
      check("all_40_live_same_state_components_pass",
            injection_summary.get("required_checks_passed") is True and
            injection_summary.get("injection_count") == 40 and
            injection_summary.get("applied_codec") == args.apply_codec and
            injection_summary.get("applied_layer_start") == args.layer_start and
            injection_summary.get("applied_layer_end") == args.layer_end and
            injection_summary.get("reference_swiglu") ==
                args.reference_swiglu and
            injection_summary.get("reference_down") ==
                args.reference_down and
            injection_summary.get("applied_layer_count") ==
                expected_applied_layers and
            [row.get("layer") for row in per_layer] == list(range(40)) and
            all(row.get("routed_output_compare", {}).get("finite") is True and
                float(row.get("routed_output_compare", {}).get(
                    "cosine", -1.0)) >= COMPONENT_COSINE_MIN and
                float(row.get("routed_output_compare", {}).get(
                    "relative_l2", 1e30)) <= COMPONENT_RELATIVE_L2_MAX
                for row in per_layer)),
      check("all_live_injected_values_aggregated",
            injection_summary.get(
                "aggregate_routed_output_compare", {}).get(
                    "compared_value_count") == INJECTION_VALUE_COUNT,
            comparison=injection_summary.get(
                "aggregate_routed_output_compare")),
      check("final_full_vocabulary_distribution_passes",
            distribution.get("value_count") == VOCABULARY_SIZE and
            distribution.get("finite") is True and
            distribution.get("baseline_top1") ==
                distribution.get("injected_top1") and
            float(distribution.get("kl_divergence", 1e30)) <=
                KL_DIVERGENCE_MAX,
            required_kl_divergence_max=KL_DIVERGENCE_MAX,
            comparison=distribution),
      continuation_check,
      check("one_native_runtime_owns_all_real_payloads",
            injection_summary.get("context_create_count") == 1 and
            injection_summary.get("program_load_count") == 4 and
            injection_summary.get("layer_count") == 40 and
            injection_summary.get("run_count") == 40 and
            injection_summary.get("resident_weight_bytes") == RESIDENT_BYTES),
      check("hybrid_maps_and_links_no_onednn_or_openvino",
            ldd["returncode"] == 0 and "dnnl" not in ldd_lower and
            "openvino" not in ldd_lower and
            injection_summary.get("maps_exclude_onednn_openvino") is True),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state, "model": str(args.model),
      "component_prerequisite": str(args.component_artifact),
      "payload_artifact": str(args.payload_artifact),
      "q4_gateup_artifact": str(args.q4_artifact),
      "q4_down_artifact": str(args.q4_down_artifact),
      "q4_down_binary": str(q4_down_binary),
      "applied_codec": args.apply_codec,
      "teacher_forced": args.teacher_forced,
      "q4_f32_contributions": args.q4_f32_contributions,
      "q4_exact_block": args.q4_exact_block,
      "applied_layer_start": args.layer_start,
      "applied_layer_end": args.layer_end,
      "reference_swiglu": args.reference_swiglu,
      "reference_down": args.reference_down,
      "hybrid_reference_host_only": True,
      "injection_summary": injection_summary,
      "final_distribution": distribution,
      "greedy_continuation": {
          "baseline": baseline_tokens.get("token_ids"),
          "injected": injected_tokens.get("token_ids")},
      "teacher_forced_ladder": teacher_forced_ladder,
      "teacher_forced_contract_passed": teacher_forced_contract_passed,
      "first_unstable_position": first_unstable_position,
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          (f"accept_1024_token_{args.apply_codec}_teacher_forced_state"
           if args.teacher_forced else
           f"accept_1024_token_{args.apply_codec}_live_state_injection")
          if passed else
          (f"reject_1024_token_{args.apply_codec}_teacher_forced_state"
           if args.teacher_forced else
           f"reject_1024_token_{args.apply_codec}_live_state_injection")),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  (out / "summary.md").write_text("\n".join([
      "# All-40 live-state routed-MoE injection", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- injected layers: `{injection_summary.get('injection_count')}`",
      f"- aggregate routed relative L2: "
      f"`{(injection_summary.get('aggregate_routed_output_compare') or {}).get('relative_l2')}`",
      f"- full-vocabulary KLD: `{distribution.get('kl_divergence')}`",
      f"- baseline/injected top-1: `{distribution.get('baseline_top1')}` / "
      f"`{distribution.get('injected_top1')}`", "",
      f"- greedy tokens: `{injected_tokens.get('token_ids')}`", "",
      f"- teacher-forced: `{str(args.teacher_forced).lower()}`",
      f"- first unstable position: `{first_unstable_position}`", "",
      "The CPU/llama host supplies unported graph stages and is correctness",
      "evidence only. Its wall time is not a native product speed row.", "",
  ]), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "pass": passed,
                    "distribution": distribution,
                    "greedy_tokens": injected_tokens.get("token_ids"),
                    "first_unstable_position": first_unstable_position,
                    "aggregate": injection_summary.get(
                        "aggregate_routed_output_compare")}, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
