#!/usr/bin/env python3
"""Capture the missing linear-prefill input and lock the whole-stage budget."""

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
SCHEMA = "intel-qwen36-linear-prefill-whole-stage-boundary-gate-v0"
CAPTURE_SOURCE = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
DEFAULT_MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TOKENS = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
DEFAULT_LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
DEFAULT_LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
DEFAULT_ENV = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_PROFILE = (
    ROOT / "output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/"
    "profile.json")
DEFAULT_STATE = (
    ROOT / "output/linear-attention-prefill-state-20260712Tseq753cleanZ/"
    "result.json")
EXPECTED_LLAMA_COMMIT = "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
EXPECTED_CATEGORY_US = {
    "linear_attention_projection": 50_171.0,
    "linear_attention_conv_reorder": 12_118.0,
    "linear_attention_gated_delta": 71_428.0,
}
LAYER_COUNT = 30
PRODUCT_RATIO = 1.10
NONSTATE_CAP_US = 1840.0
EXPECTED_TENSORS = {
    "attn_norm-0",
    "linear_attn_qkv_mixed-0",
    "conv_output_raw-0",
    "q_conv_predelta-0",
    "k_conv_predelta-0",
    "v_conv_predelta-0",
    "alpha-0",
    "a_softplus-0",
    "gate-0",
    "beta-0",
    "beta_sigmoid-0",
    "state_predelta-0",
    "new_state-0",
    "attn_output-0",
    "z-0",
    "final_output-0",
    "linear_attn_out-0",
    "conv_states_reshaped-0",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
  parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV)
  parser.add_argument("--openvino-profile", type=Path, default=DEFAULT_PROFILE)
  parser.add_argument("--state-result", type=Path, default=DEFAULT_STATE)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=180)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.threads <= 0 or args.timeout_s <= 0:
    parser.error("threads and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/linear-prefill-whole-stage-boundary-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> dict[str, Any]:
  try:
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
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
      + shlex.join(command))
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    value = json.loads(line)
    if not isinstance(value, dict):
      raise SystemExit(f"expected JSONL object: {path}")
    rows.append(value)
  return rows


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  capture = raw / "capture-output"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.model, args.tokens, args.llama_source, args.llama_build,
      args.env_script, args.openvino_profile, args.state_result, CAPTURE_SOURCE,
      args.llama_build / "bin/libllama.so.0.0.1",
      args.llama_build / "bin/libggml.so.0.13.1",
      args.llama_build / "bin/libggml-cpu.so.0.13.1",
      args.llama_build / "bin/libggml-base.so.0.13.1",
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  llama_commit = git_output("rev-parse", "HEAD", cwd=args.llama_source)

  library_dir = args.llama_build / "bin"
  binary = raw / "capture"
  compile_command = [
      "c++", "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}", str(CAPTURE_SOURCE),
      f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1", "-Wl,-l:libggml-base.so.0.13.1",
      "-fopenmp", "-pthread", "-o", str(binary),
  ]
  compile_result = run_intel(compile_command, args)
  write_run(raw, "compile", compile_result)

  capture_command = [
      str(binary), "--model", str(args.model), "--token-ids-file",
      str(args.tokens), "--binary-u32-token-file", "--token-count", "1024",
      "--batch-all", "--linear-component-layer", "0", "--out-dir",
      str(capture), "--case-id", "prefill_shape_008k_tile1024_linear_layer0",
      "--threads", str(args.threads), "--n-ctx", "2048", "--ngl", "0",
      "--top-k", "1", "--predicts-generated-position", "1024",
  ]
  capture_result = (
      run_intel(capture_command, args) if compile_result["returncode"] == 0
      else {"command": capture_command, "returncode": 125, "stdout": "",
            "stderr": "compile failed", "timed_out": False})
  write_run(raw, "capture", capture_result)

  summary_path = capture / "capture-summary.json"
  tensor_index_path = capture / "tensor-dumps.jsonl"
  summary = load_json(summary_path) if summary_path.is_file() else {}
  tensor_rows = load_jsonl(tensor_index_path) if tensor_index_path.is_file() else []
  tensor_by_name = {
      str(row.get("tensor_name")): row for row in tensor_rows
      if isinstance(row.get("tensor_name"), str)}
  captured_names = set(tensor_by_name)
  input_row = tensor_by_name.get("attn_norm-0", {})
  input_payload = capture / str(input_row.get("payload_path", "missing"))

  profile = load_json(args.openvino_profile)
  profile_runs = profile.get("runs", [])
  category_ms = (
      profile_runs[0].get("category_ms", {})
      if isinstance(profile_runs, list) and len(profile_runs) == 1 and
      isinstance(profile_runs[0], dict) else {})
  category_us = {
      name: float(category_ms.get(name, math.nan)) * 1000.0
      for name in EXPECTED_CATEGORY_US}
  denominator_total_us = sum(category_us.values())
  denominator_per_layer_us = denominator_total_us / LAYER_COUNT
  whole_stage_cap_us = denominator_per_layer_us / PRODUCT_RATIO

  state_result = load_json(args.state_result)
  state_rows = state_result.get("rows", [])
  state_medians = [
      float(row.get("probe", {}).get("state_core_median_us", math.nan))
      for row in state_rows if isinstance(row, dict)
  ]
  slower_state_us = max(state_medians, default=math.nan)
  residual_us = whole_stage_cap_us - slower_state_us

  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("locked_llama_source_commit",
            llama_commit == EXPECTED_LLAMA_COMMIT,
            observed=llama_commit, expected=EXPECTED_LLAMA_COMMIT),
      check("capture_program_builds", compile_result["returncode"] == 0),
      check("real_layer0_capture_completes", capture_result["returncode"] == 0),
      check("capture_is_one_1024_token_linear_layer0_batch",
            summary.get("token_count") == 1024 and
            summary.get("batch_all") is True and
            summary.get("linear_component_layer") == 0),
      check("normalized_hidden_and_existing_boundaries_captured_once",
            captured_names == EXPECTED_TENSORS and
            len(tensor_rows) == len(EXPECTED_TENSORS),
            captured=sorted(captured_names), expected=sorted(EXPECTED_TENSORS)),
      check("normalized_hidden_shape_and_payload_locked",
            input_row.get("tensor_type") == "f32" and
            input_row.get("ne") == [2048, 1024, 1, 1] and
            input_row.get("nbytes") == 1024 * 2048 * 4 and
            input_payload.stat().st_size == 1024 * 2048 * 4,
            row=input_row, payload=relative(input_payload)),
      check("openvino_complete_linear_categories_locked",
            all(math.isclose(category_us[name], expected, abs_tol=0.5)
                for name, expected in EXPECTED_CATEGORY_US.items()),
            category_us=category_us),
      check("seq753_slower_state_row_locked",
            len(state_medians) == 2 and
            math.isclose(slower_state_us, 2211.04101562, abs_tol=1.0e-6),
            state_medians_us=state_medians),
      check("whole_stage_and_nonstate_caps_preregistered",
            math.isclose(denominator_per_layer_us, 4457.233333333333,
                         abs_tol=1.0e-9) and
            math.isclose(whole_stage_cap_us, 4052.0303030303025,
                         abs_tol=1.0e-9) and
            residual_us >= NONSTATE_CAP_US and residual_us < 1841.0,
            denominator_per_layer_us=denominator_per_layer_us,
            whole_stage_cap_us=whole_stage_cap_us,
            slower_state_us=slower_state_us, residual_us=residual_us,
            registered_nonstate_cap_us=NONSTATE_CAP_US),
  ]
  required = all(bool(item["pass"]) for item in checks)
  disposition = (
      "accept_real_linear_input_and_nonstate_whole_stage_design_gate"
      if required else "reject_incomplete_whole_stage_boundary_gate")
  selected_next_route = (
      "native_linear_prefill_nonstate_projection_conv_output_tile_gate"
      if required else "native_prefill_route_reflection_gate")
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "inputs": {"model": str(args.model), "tokens": relative(args.tokens),
                 "openvino_profile": relative(args.openvino_profile),
                 "state_result": relative(args.state_result)},
      "capture": {"path": relative(capture), "summary": summary,
                  "tensor_names": sorted(captured_names),
                  "normalized_hidden_payload": relative(input_payload)},
      "budget": {"category_us": category_us,
                 "denominator_total_us": denominator_total_us,
                 "denominator_per_layer_us": denominator_per_layer_us,
                 "product_ratio": PRODUCT_RATIO,
                 "whole_stage_cap_us": whole_stage_cap_us,
                 "state_medians_us": state_medians,
                 "slower_state_us": slower_state_us,
                 "residual_us": residual_us,
                 "registered_nonstate_cap_us": NONSTATE_CAP_US},
      "checks": checks, "required_checks_passed": required,
      "disposition": disposition, "selected_next_route": selected_next_route,
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "manifest.json").write_text(json.dumps({
      "schema_version": SCHEMA, "created_at": created_at, "commit": commit,
      "git_dirty": bool(dirty), "required_checks_passed": required,
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [item["name"] for item in checks if not item["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Linear-prefill whole-stage boundary gate", "",
      f"- required_checks_passed: `{str(required).lower()}`",
      f"- disposition: `{disposition}`",
      f"- complete linear denominator/cap: "
      f"`{denominator_per_layer_us:.3f} / {whole_stage_cap_us:.3f} us`",
      f"- slower state charge / non-state residual: "
      f"`{slower_state_us:.3f} / {residual_us:.3f} us`",
      f"- registered non-state cap: `{NONSTATE_CAP_US:.0f} us`",
      f"- normalized hidden payload: `{relative(input_payload)}`",
      f"- failed checks: `{failed}`", "",
      ("The missing real normalized-hidden boundary is locked. Implement one "
       "native non-state tile under the registered residual; do not vary the "
       "rejected state kernels." if required else
       "The boundary/budget gate is incomplete; inspect the failed axis and "
       "do not start a non-state implementation."), ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required, "disposition": disposition,
      "whole_stage_cap_us": whole_stage_cap_us,
      "registered_nonstate_cap_us": NONSTATE_CAP_US,
      "selected_next_route": selected_next_route,
      "out_dir": relative(out)}, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
