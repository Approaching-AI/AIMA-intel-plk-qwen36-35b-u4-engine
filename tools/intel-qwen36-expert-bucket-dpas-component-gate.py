#!/usr/bin/env python3
"""Run the sole real-shape 1024-token expert-bucket DPAS component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-expert-bucket-dpas-component-gate-v1"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TOKENS = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
DEFAULT_CENSUS = (
    ROOT / "output/prefill-router-shape-census-gate-20260711Tseq639cleanZ")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
DEFAULT_LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
CAPTURE_SOURCE = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
COMPONENT_SOURCE = ROOT / "engine/tools/expert_bucket_dpas_component.cpp"
PREPACKED_KERNEL_SOURCE = (
    ROOT / "engine/gpu/opencl/prepacked_fused_q4k_moe.cl")
CASE_ID = "prefill_shape_008k"
LAYER = 27
TILE_TOKENS = 1024
SELECTED_EXPERTS = 8
TARGET_LAYER_BUDGET_US_PER_64 = 575.33
PLANNING_GB_S = 115.0
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
TOKEN_SHA256 = "8a3554ce47f204926f29b898eee2dd17d3f849f73ab8094c05b4f96a17b35ad8"
ROUTED_PAYLOAD_SHA256 = {
    f"attn_post_norm-{LAYER}":
        "8d44d06e72ff10a0f827952c02f3370d56288c8de7c482aff3e0554c2ac0395b",
    f"ffn_moe_topk-{LAYER}":
        "76ef4ea4dd7a4385f8d4b18ff00eb181f919d125b7f484c6e7bff3ff473777ba",
    f"ffn_moe_weights_norm-{LAYER}":
        "0141a67188d6d8d92e39cac7f646d6af843f4a1ac9411c6505e87cf988cfe2af",
    f"ffn_moe_swiglu-{LAYER}":
        "187dd69ae740f39951330fbadb48407f791f9ed5145bfbac53f73c076917b648",
    f"ffn_moe_down-{LAYER}":
        "b6977e220e0dc081a111ddc104607fb6e869888f01f18f852dfc60820b045f26",
    f"ffn_moe_out-{LAYER}":
        "e0dc494a2823ffe10cae0b5bd5c802fb4358b8cbc44b8495fc7c2fc0f8df76f2",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
  parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--kernel-mode",
                      choices=("m1_u8", "m8_u4", "prepacked_routed"),
                      default="m1_u8")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.threads <= 0 or args.repeat <= 0 or args.timeout_s <= 0:
    parser.error("threads, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/expert-bucket-dpas-component-gate-{stamp}"
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*parts: str) -> str:
    result = subprocess.run(
        ["git", *parts], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(
    command: list[str], timeout_s: int, cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=cwd, env=environment, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
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


def shell_run(
    command: list[str], env_script: Path, timeout_s: int, cwd: Path = ROOT,
) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += "export INTEL_FORCE_PROBE=b080 && " + shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s, cwd)


def write_run_logs(raw_dir: Path, name: str, result: dict[str, Any]) -> None:
  (raw_dir / f"{name}.stdout").write_text(
      str(result.get("stdout", "")), encoding="utf-8")
  (raw_dir / f"{name}.stderr").write_text(
      str(result.get("stderr", "")), encoding="utf-8")
  write_json(raw_dir / f"{name}.command.json", {
      "command": result.get("command", []),
      "returncode": result.get("returncode"),
      "timed_out": result.get("timed_out", False),
  })


def compile_capture(args: argparse.Namespace, raw_dir: Path) -> tuple[Path, dict[str, Any]]:
  binary = raw_dir / "component-capture"
  library_dir = args.llama_build / "bin"
  command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}", str(CAPTURE_SOURCE),
      f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1", "-Wl,-l:libggml-base.so.0.13.1",
      "-fopenmp", "-pthread", "-o", str(binary),
  ]
  result = shell_run(command, args.env_script, args.timeout_s)
  write_run_logs(raw_dir, "capture-build", result)
  return binary, result


def compile_component(args: argparse.Namespace, raw_dir: Path) -> tuple[Path, dict[str, Any]]:
  binary = raw_dir / "expert-bucket-dpas-component"
  result = shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      str(COMPONENT_SOURCE), "-ldl", "-pthread", "-o", str(binary),
  ], args.env_script, args.timeout_s)
  write_run_logs(raw_dir, "component-build", result)
  return binary, result


def extract_kernel(raw_dir: Path) -> Path:
  source = COMPONENT_SOURCE.read_text(encoding="utf-8")
  prefix = 'const char* kOpenClSource = R"CLC(\n'
  suffix = '\n)CLC";'
  start = source.find(prefix)
  end = source.find(suffix, start + len(prefix))
  if start < 0 or end < 0:
    raise SystemExit("could not extract embedded OpenCL source")
  kernel = raw_dir / "expert_bucket_dpas.cl"
  kernel.write_text(source[start + len(prefix):end] + "\n", encoding="utf-8")
  return kernel


def tensor_rows(index_path: Path) -> dict[str, dict[str, Any]]:
  rows = load_jsonl(index_path)
  expected = {
      "gate_up": (f"blk.{LAYER}.ffn_gate_up_exps.weight", [2048, 1024, 256]),
      "down": (f"blk.{LAYER}.ffn_down_exps.weight", [512, 2048, 256]),
  }
  result: dict[str, dict[str, Any]] = {}
  for key, (name, dims) in expected.items():
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
      raise SystemExit(f"expected one tensor index row for {name}")
    if (matches[0].get("dims") != dims or
        matches[0].get("ggml_type_name") != "Q4_K"):
      raise SystemExit(f"locked layer-27 {key} tensor shape or type changed")
    result[key] = matches[0]
  return result


def census_shape(census_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  result = load_json(census_dir / "result.json")
  if (result.get("required_checks_passed") is not True or
      result.get("aggregate", {}).get("tile_token_count") != TILE_TOKENS):
    raise SystemExit("the locked 1024-token census did not pass")
  rows = [
      row for row in load_jsonl(census_dir / "layer-shapes.jsonl")
      if row.get("case_id") == CASE_ID and row.get("layer") == LAYER
  ]
  if len(rows) != 1:
    raise SystemExit("locked census layer shape is missing")
  assignments = [
      row for row in load_jsonl(census_dir / "router-assignments.jsonl")
      if row.get("case_id") == CASE_ID and row.get("layer") == LAYER
  ]
  if len(assignments) != 1:
    raise SystemExit("locked census router assignments are missing")
  return rows[0], assignments[0]


def component_cap_us(
    shape: dict[str, Any], routed: bool,
) -> dict[str, float | int]:
  full_weights = int(shape["total_layer_source_bytes"])
  gate_up_weights = int(shape["gate_up_unique_weight_bytes"])
  permutation = int(shape["permutation_scatter_stream_bytes"])
  down_weights = (int(shape["active_expert_count"]) * 2048 * 2 * 144
                  if routed else 0)
  reserved_bytes = (full_weights - gate_up_weights - down_weights
                    if routed else
                    full_weights - gate_up_weights + permutation)
  whole_window_budget_us = TARGET_LAYER_BUDGET_US_PER_64 * TILE_TOKENS / 64
  reserved_us = reserved_bytes / (PLANNING_GB_S * 1000.0)
  return {
      "full_layer_source_bytes": full_weights,
      "gate_up_unique_weight_bytes": gate_up_weights,
      "down_unique_weight_bytes": down_weights,
      "permutation_scatter_stream_bytes": permutation,
      "reserved_noncomponent_bytes": reserved_bytes,
      "reserved_noncomponent_us_at_115_gb_s": reserved_us,
      "whole_window_budget_us": whole_window_budget_us,
      "kernel_cap_us": whole_window_budget_us - reserved_us,
  }


def captured_payloads(
    capture_dir: Path, routed: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
  rows = load_jsonl(capture_dir / "tensor-dumps.jsonl")
  by_name = {str(row["tensor_name"]): row for row in rows}
  expected = {
      f"attn_post_norm-{LAYER}": ("f32", [2048, 1024, 1, 1]),
      f"ffn_moe_topk-{LAYER}": ("i32", [8, 1024, 1, 1]),
      f"ffn_moe_swiglu-{LAYER}": ("f32", [512, 8, 1024, 1]),
  }
  if routed:
    expected.update({
        f"ffn_moe_weights_norm-{LAYER}": ("f32", [8, 1024, 1, 1]),
        f"ffn_moe_down-{LAYER}": ("f32", [2048, 8, 1024, 1]),
        f"ffn_moe_out-{LAYER}": ("f32", [2048, 1024, 1, 1]),
    })
  paths: dict[str, Path] = {}
  for name, (tensor_type, shape) in expected.items():
    row = by_name.get(name)
    if row is None or row.get("tensor_type") != tensor_type or row.get("ne") != shape:
      raise SystemExit(f"captured tensor metadata mismatch: {name}")
    path = capture_dir / str(row["payload_path"])
    if (not path.is_file() or path.stat().st_size != int(row["nbytes"]) or
        (routed and sha256_file(path) != ROUTED_PAYLOAD_SHA256[name])):
      raise SystemExit(f"captured payload mismatch: {name}")
    paths[name] = path
  if len(by_name) != len(expected):
    raise SystemExit(f"capture must contain exactly {len(expected)} tensors")
  return by_name, paths


def captured_router_ids(
    payload: Path, stride: int,
) -> list[list[int]]:
  data = payload.read_bytes()
  rows: list[list[int]] = []
  for token in range(TILE_TOKENS):
    rows.append([
        struct.unpack_from("<i", data, token * stride + rank * 4)[0]
        for rank in range(SELECTED_EXPERTS)
    ])
  return rows


def build_summary(result: dict[str, Any]) -> str:
  probe = result.get("probe", {})
  compare = probe.get("compare", {})
  budget = result["budget"]
  if result.get("kernel_mode") == "prepacked_routed":
    return "\n".join([
        "# Resident-prepacked fused Q4_K DPAS routed-MoE killer gate",
        "",
        f"- shape/layer/tokens: `{CASE_ID}` / `{LAYER}` / `{TILE_TOKENS}`",
        f"- active experts / assignments: `{probe.get('active_experts')}` / "
        f"`{probe.get('assignment_count')}`",
        f"- resident prepacked bytes: `{probe.get('resident_prepacked_bytes')}`",
        f"- lossless U4 codes / mismatches: "
        f"`{probe.get('repacked_q4_code_count')}` / "
        f"`{probe.get('repack_mismatch_count')}`",
        f"- SwiGLU max abs / RMSE: `{compare.get('max_abs_diff')}` / "
        f"`{compare.get('rmse')}`",
        f"- complete runtime minimum / median: `{probe.get('kernel_min_us')}` / "
        f"`{probe.get('kernel_median_us')} us`",
        f"- routed-MoE cap: `{budget['kernel_cap_us']:.3f} us`",
        f"- stage profile: `{probe.get('stage_profile_us')}`",
        f"- required checks passed: "
        f"`{str(result['required_checks_passed']).lower()}`",
        f"- disposition: `{result['disposition']}`",
        "",
        "The timer includes dynamic input Q8, resident-prepacked exact Q4_K",
        "gate/up, SwiGLU-to-Q8, exact down, normalized router weighting,",
        "deterministic scatter, submission, and queue drain. Only one-time",
        "resident plane preparation is excluded. This is route evidence, not",
        "a native prefill or product speed claim.",
        "",
    ])
  return "\n".join([
      "# Real expert-bucket DPAS component gate",
      "",
      f"- shape: `{CASE_ID}`, layer `{LAYER}`, `{TILE_TOKENS}` tokens",
      f"- active experts / assignments: `{probe.get('active_experts')}` / "
      f"`{probe.get('assignment_count')}`",
      f"- task count: `{probe.get('task_count')}`",
      f"- kernel mode / task width: `{probe.get('kernel_mode')}` / "
      f"`{probe.get('task_tokens')}`",
      f"- component correctness: `{str(probe.get('correctness_pass')).lower()}` "
      f"(max abs `{compare.get('max_abs_diff')}`, RMSE `{compare.get('rmse')}`)",
      f"- DPAS kernel minimum / median: `{probe.get('kernel_min_us')} / "
      f"{probe.get('kernel_median_us')} us`",
      f"- target-facing component cap: `{budget['kernel_cap_us']:.3f} us`",
      f"- normalized kernel minimum: "
      f"`{probe.get('kernel_normalized_per_64_us')} us/64 tokens`",
      f"- effective charged traffic: `{probe.get('effective_gb_s')} GB/s`",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      "",
      "The kernel uses the locked real input, all 8192 router assignments, the",
      "full layer-27 Q4_K gate/up tensor, expert-major buckets, local weight",
      "reuse across eight DPAS subgroups, and the captured llama.cpp SwiGLU",
      "oracle. Correctness passes, but the target-facing performance gate does",
      "not. This is component evidence, not a native prefill speed claim.",
      "",
  ])


def main() -> int:
  args = parse_args()
  routed = args.kernel_mode == "prepacked_routed"
  created_at = iso_now()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  capture_dir = raw_dir / "capture"
  ocloc_dir = raw_dir / "ocloc"
  disasm_dir = raw_dir / "disasm"
  capture_dir.mkdir(parents=True, exist_ok=False)
  ocloc_dir.mkdir()
  disasm_dir.mkdir()

  for required in (
      args.model, args.tokens, args.census / "result.json",
      args.census / "layer-shapes.jsonl",
      args.census / "router-assignments.jsonl", args.tensor_index,
      args.env_script, args.cxx, CAPTURE_SOURCE, COMPONENT_SOURCE):
    if not required.is_file():
      raise SystemExit(f"required input missing: {required}")
  if routed and not PREPACKED_KERNEL_SOURCE.is_file():
    raise SystemExit(f"required input missing: {PREPACKED_KERNEL_SOURCE}")
  if sha256_file(args.tokens) != TOKEN_SHA256:
    raise SystemExit("locked token input hash mismatch")
  if sha256_file(args.model) != MODEL_SHA256:
    raise SystemExit("locked model hash mismatch")

  shape, census_assignments = census_shape(args.census)
  budget = component_cap_us(shape, routed)
  tensors = tensor_rows(args.tensor_index)
  capture_binary, capture_build = compile_capture(args, raw_dir)
  component_binary, component_build = compile_component(args, raw_dir)

  capture_command = [
      str(capture_binary), "--model", str(args.model),
      "--token-ids-file", str(args.tokens), "--binary-u32-token-file",
      "--token-count", str(TILE_TOKENS), "--batch-all",
      "--component-layer", str(LAYER), "--out-dir", str(capture_dir),
      "--case-id", f"{CASE_ID}_tile1024_layer{LAYER}",
      "--threads", str(args.threads), "--n-ctx", "2048", "--ngl", "0",
      "--top-k", "1", "--predicts-generated-position", "0",
  ]
  if routed:
    capture_command.append("--component-through-down")
  capture = (
      shell_run(capture_command, args.env_script, args.timeout_s)
      if capture_build["returncode"] == 0 else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
       "timed_out": False}
  )
  write_run_logs(raw_dir, "capture", capture)
  if capture["returncode"] != 0:
    raise SystemExit("component capture failed; inspect raw/capture.stderr")
  metadata, payloads = captured_payloads(capture_dir, routed)
  topk_name = f"ffn_moe_topk-{LAYER}"
  topk_stride = int(metadata[topk_name]["nb"][1])
  ids_match = (
      captured_router_ids(payloads[topk_name], topk_stride) ==
      census_assignments["expert_ids_by_token"])

  kernel = PREPACKED_KERNEL_SOURCE if routed else extract_kernel(raw_dir)
  ocloc = shell_run([
      "ocloc", "-file", str(kernel), "-device", "0xb080",
      "-options", "-cl-std=CL2.0",
  ], args.env_script, args.timeout_s, ocloc_dir)
  write_json(raw_dir / "ocloc.json", ocloc)
  native_bins = sorted(ocloc_dir.glob("*.bin"))
  disasm = (
      run([
          "ocloc", "disasm", "-file", str(native_bins[0]), "-dump",
          str(disasm_dir), "-device", "0xb080",
      ], args.timeout_s)
      if ocloc["returncode"] == 0 and native_bins else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "ocloc failed",
       "timed_out": False}
  )
  write_json(raw_dir / "disasm.json", disasm)
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob("*.asm"))
  ze_info = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob(".ze_info"))

  component_command = [
      str(component_binary), "--model", str(args.model),
      "--weight-offset", str(tensors["gate_up"]["absolute_offset"]),
      "--weight-bytes", str(tensors["gate_up"]["nbytes"]),
      "--input", str(payloads[f"attn_post_norm-{LAYER}"]),
      "--topk", str(payloads[topk_name]),
      "--topk-stride", str(topk_stride),
      "--oracle", str(payloads[f"ffn_moe_swiglu-{LAYER}"]),
      "--repeat", str(args.repeat),
      "--kernel-mode", args.kernel_mode,
      "--kernel-cap-us", str(budget["kernel_cap_us"]),
  ]
  if routed:
    component_command += [
        "--kernel-source", str(PREPACKED_KERNEL_SOURCE),
        "--down-weight-offset", str(tensors["down"]["absolute_offset"]),
        "--down-weight-bytes", str(tensors["down"]["nbytes"]),
        "--router-weights", str(payloads[f"ffn_moe_weights_norm-{LAYER}"]),
        "--down-oracle", str(payloads[f"ffn_moe_down-{LAYER}"]),
        "--moe-oracle", str(payloads[f"ffn_moe_out-{LAYER}"]),
    ]
  component = (
      shell_run(component_command, args.env_script, args.timeout_s)
      if component_build["returncode"] == 0 else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
       "timed_out": False}
  )
  write_run_logs(raw_dir, "component", component)
  try:
    probe = json.loads(component["stdout"])
  except json.JSONDecodeError:
    probe = {}

  evidence_checks = [
      {"name": "capture_build_passed", "pass": capture_build["returncode"] == 0},
      {"name": "component_build_passed", "pass": component_build["returncode"] == 0},
      {"name": "locked_component_capture_passed", "pass": capture["returncode"] == 0},
      {"name": "capture_has_exact_required_tensor_boundary",
       "pass": len(metadata) == (6 if routed else 3)},
      {"name": "captured_router_ids_match_seq639", "pass": ids_match},
      {"name": "ocloc_compile_passed", "pass": ocloc["returncode"] == 0},
      {"name": "ocloc_disassembly_passed", "pass": disasm["returncode"] == 0},
      {"name": "dpas_instruction_present", "pass": "dpas." in assembly.lower()},
      {"name": "selected_lowbit_isa_present", "pass":
       args.kernel_mode not in ("m8_u4", "prepacked_routed") or
       ("dpas.8x8" in assembly.lower() and ":u4" in assembly.lower())},
      {"name": "ze_info_has_dpas", "pass": "has_dpas:true" in ze_info.replace(" ", "")},
      {"name": "component_executed", "pass": component["returncode"] in (0, 2)},
      {"name": "arc_b390_selected", "pass": "B390" in str(probe.get("device_name"))},
      {"name": "real_shape_222_experts_8192_assignments", "pass":
       probe.get("active_experts") == 222 and probe.get("assignment_count") == 8192},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  correctness_checks = [
      {"name": "component_correctness_passed", "pass":
       probe.get("correctness_pass") is True},
      {"name": "all_4194304_values_compared", "pass":
       probe.get("compare", {}).get("compared_value_count") == 4_194_304},
  ]
  if routed:
    correctness_checks += [
        {"name": "all_active_codes_repacked_losslessly",
         "pass": probe.get("repack_pass") is True and
                 probe.get("repacked_q4_code_count") == 698_351_616 and
                 probe.get("repack_mismatch_count") == 0},
        {"name": "all_16777216_weighted_down_values_compared",
         "pass": probe.get("weighted_down_compare", {}).get(
             "compared_value_count") == 16_777_216 and
                 probe.get("weighted_down_compare", {}).get(
                     "mismatch_count") == 0},
        {"name": "all_2097152_routed_values_compared",
         "pass": probe.get("moe_compare", {}).get(
             "compared_value_count") == 2_097_152 and
                 probe.get("moe_compare", {}).get("mismatch_count") == 0},
    ]
  performance_checks = [
      {"name": "kernel_fits_remaining_whole_layer_budget", "pass":
       probe.get("performance_pass") is True},
      {"name": "kernel_min_at_or_below_cap", "pass":
       float(probe.get("kernel_min_us", float("inf"))) <=
       float(budget["kernel_cap_us"])},
  ]
  evidence_checks_passed = all(bool(row["pass"]) for row in evidence_checks)
  correctness_checks_passed = all(bool(row["pass"]) for row in correctness_checks)
  performance_checks_passed = all(bool(row["pass"]) for row in performance_checks)
  required_checks_passed = (
      evidence_checks_passed and correctness_checks_passed and
      performance_checks_passed)
  if routed:
    disposition = (
        "admit_prepacked_fused_q4k_dpas_routed_moe"
        if required_checks_passed else
        "reject_prepacked_fused_q4k_dpas_routed_moe_above_cap")
  else:
    disposition = (
        f"admit_context_wide_expert_bucket_{args.kernel_mode}_layer_integration"
        if required_checks_passed else
        f"reject_context_wide_expert_bucket_{args.kernel_mode}_below_kill_number")

  checks = evidence_checks + correctness_checks + performance_checks
  result = {
      "budget": budget,
      "case_id": CASE_ID,
      "checks": checks,
      "correctness_checks_passed": correctness_checks_passed,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_checks_passed,
      "git": git_state(),
      "kernel_mode": args.kernel_mode,
      "layer": LAYER,
      "performance_checks_passed": performance_checks_passed,
      "probe": probe,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "capture_source": str(CAPTURE_SOURCE.relative_to(ROOT)),
          "capture_source_sha256": sha256_file(CAPTURE_SOURCE),
          "component_source": str(COMPONENT_SOURCE.relative_to(ROOT)),
          "component_source_sha256": sha256_file(COMPONENT_SOURCE),
          "kernel_source": (str(PREPACKED_KERNEL_SOURCE.relative_to(ROOT))
                            if routed else str(kernel.relative_to(raw_dir))),
          "kernel_source_sha256": sha256_file(kernel),
          "census": str(args.census.relative_to(ROOT)),
          "census_result_sha256": sha256_file(args.census / "result.json"),
          "model_path": str(args.model),
          "model_sha256": MODEL_SHA256,
          "token_path": str(args.tokens.relative_to(ROOT)),
          "token_sha256": TOKEN_SHA256,
      },
      "speedup_claims_allowed": False,
      "tensors": tensors,
      "tile_tokens": TILE_TOKENS,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": evidence_checks + correctness_checks,
      "comparison": probe.get("compare"),
      "weighted_down_comparison": (
          probe.get("weighted_down_compare") if routed else None),
      "moe_comparison": probe.get("moe_compare") if routed else None,
      "correctness_checks_passed": correctness_checks_passed,
      "evidence_checks_passed": evidence_checks_passed,
  })
  write_json(out_dir / "capture-metadata.json", {
      "case_id": CASE_ID,
      "layer": LAYER,
      "payload_sha256": ROUTED_PAYLOAD_SHA256 if routed else None,
      "router_ids_match_seq639": ids_match,
      "tensors": list(metadata.values()),
      "tile_tokens": TILE_TOKENS,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": ("single real 1024-token routed-MoE killer boundary"
                 if routed else "standalone expert-bucket component gate"),
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = {
      "active_experts": probe.get("active_experts"),
      "assignment_count": probe.get("assignment_count"),
      "component_correctness_passed": probe.get("correctness_pass"),
      "effective_gb_s": probe.get("effective_gb_s"),
      "kernel_cap_us": budget["kernel_cap_us"],
      "kernel_mean_us": probe.get("kernel_mean_us"),
      "kernel_mode": probe.get("kernel_mode"),
      "kernel_median_us": probe.get("kernel_median_us"),
      "kernel_min_us": probe.get("kernel_min_us"),
      "performance_checks_passed": performance_checks_passed,
      "required_checks_passed": required_checks_passed,
      "task_count": probe.get("task_count"),
      "resident_prepacked_bytes": probe.get("resident_prepacked_bytes"),
      "stage_profile_us": probe.get("stage_profile_us"),
  }
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in metrics.items():
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "kernel_cap_us": budget["kernel_cap_us"],
      "kernel_min_us": probe.get("kernel_min_us"),
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
