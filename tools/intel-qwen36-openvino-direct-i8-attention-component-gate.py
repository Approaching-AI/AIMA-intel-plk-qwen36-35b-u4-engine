#!/usr/bin/env python3
"""Gate one admitted 32k one-layer direct-I8 hot/cold component."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
RUNNER_SOURCE = ROOT / "engine/tools/direct_i8_hotcold_gqa_decode.cpp"
BOUNDARIES = ROOT / "engine/boundaries.json"
ROUTES = ROOT / "doc/active" / WS / "routes-ledger.json"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CAP_MS = 0.5618915
SOURCE_BOUNDS = {
    32: ROOT / (
        "output/openvino-direct-i8-attention-bound-"
        "20260715Tseq1244-cleanZ/metrics.json"),
    4: ROOT / (
        "output/openvino-direct-i8-refinement-bound-"
        "20260715Tseq1261-cleanZ/metrics.json"),
    2: ROOT / (
        "output/openvino-direct-i8-hybrid-k2-v4-bound-"
        "20260715Tseq1268-cleanZ/metrics.json"),
}
SPLIT_SOURCE_BOUND = ROOT / (
    "output/openvino-split-state-owner-hot16k-k2-v4-bound-"
    "20260715Tseq1274-cleanZ/metrics.json")
SPECIALIZED_CODEGEN = ROOT / (
    "output/openvino-hot-cold-partial-storage-codegen-"
    "20260715Tseq1277-cleanZ/metrics.json")
TARGETS = {
    32: "iq36-direct-i8-hotcold-gqa-decode",
    4: "iq36-direct-i8-group4-hotcold-gqa-decode",
    2: "iq36-direct-i8-hybrid-k2-v4-hotcold-gqa-decode",
}
SPLIT_TARGET = (
    "iq36-direct-i8-hybrid-k2-v4-hot16k-split-state-owner-gqa-decode")
SPECIALIZED_TARGET = (
    "iq36-direct-i8-hybrid-k2-v4-hot16k-storage-specialized-gqa-decode")
ALGORITHMS = {
    32: "direct_i8_block32_hot8192_f16_dpas",
    4: "direct_i8_group4_full_cold_hot8192_f16_dpas",
    2: "direct_i8_hybrid_k2_v4_full_cold_hot8192_f16_dpas",
}
SPLIT_ALGORITHM = "direct_i8_hybrid_k2_v4_hot16384_split_state_owner_dpas"
SPECIALIZED_ALGORITHM = (
    "direct_i8_hybrid_k2_v4_hot16384_storage_specialized_dpas")
COLD_K_LAYOUTS = {
    32: "token16_block32_packed_i8",
    4: "token16_group4_packed_i8",
    2: "token16_dim4_packed_i8_group2_fp16_scale",
}
COLD_V_LAYOUTS = {
    32: "dimension_major_token16_i8",
    4: "dimension_major_token16_i8",
    2: "dimension_major_token16_i8_group4_fp16_scale",
}
VALUE_GROUPS = {32: 32, 4: 4, 2: 4}
CONTEXT_TOKENS = 32768
BASE_HOT_TOKENS = 8192
SPLIT_HOT_TOKENS = 16384
HEAD_DIM = 256
KV_HEADS = 2


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument(
      "--quant-group", type=int, choices=(32, 4, 2), default=32,
      help="2 selects the admitted asymmetric K2/V4 codec")
  parser.add_argument(
      "--hot-tokens", type=int, choices=(8192, 16384), default=8192,
      help="16384 selects the admitted split-state-owner K2/V4 component")
  parser.add_argument(
      "--storage-specialized", action="store_true",
      help="select the seq1277-admitted cold-only/hot-only component")
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  if args.hot_tokens == SPLIT_HOT_TOKENS and args.quant_group != 2:
    parser.error("--hot-tokens 16384 is admitted only with --quant-group 2")
  if args.storage_specialized and not (
      args.hot_tokens == SPLIT_HOT_TOKENS and args.quant_group == 2):
    parser.error(
        "--storage-specialized is admitted only with K2/V4 hot16384")
  return args


def expected_state_bytes(
    key_group: int, value_group: int, hot_tokens: int,
) -> int:
  cold_tokens = CONTEXT_TOKENS - hot_tokens
  hot_kv = hot_tokens * KV_HEADS * HEAD_DIM * 2 * 2
  cold_kv = cold_tokens * KV_HEADS * HEAD_DIM * 2
  cold_k_scales = cold_tokens * KV_HEADS * (HEAD_DIM // key_group) * 2
  cold_v_scales = cold_tokens * KV_HEADS * (HEAD_DIM // value_group) * 2
  return hot_kv + cold_kv + cold_k_scales + cold_v_scales


def run(
    command: list[str], timeout: int,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "swaps": r"Swaps: (\d+)",
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def environment() -> dict[str, Any]:
  commands = {
      "hostname": ["hostname"],
      "kernel": ["uname", "-a"],
      "bios_version": [
          "bash", "-lc", "head -n 1 /sys/class/dmi/id/bios_version"],
      "opencl": [
          "bash", "-lc",
          f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
          "clinfo -l"],
  }
  result: dict[str, Any] = {}
  for name, command in commands.items():
    completed = run(command, 60)
    result[name] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
  return result


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  inference = payload["performance_inference"]
  return "\n".join([
      f"# Direct-I8 K{payload['quant_group']}/V{payload['value_quant_group']} "
      "hot/cold component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- context / hot / cold: `{result.get('context_tokens')} / "
      f"{result.get('hot_tokens')} / {result.get('cold_tokens')}`",
      f"- key/value quantization groups / state bytes: "
      f"`{result.get('key_quant_group')} / "
      f"{result.get('value_quant_group')} / {result.get('state_bytes')}`",
      f"- storage-specialized / partial dispatches: "
      f"`{result.get('storage_specialized')} / "
      f"{result.get('partial_dispatches')}`",
      f"- complete median / one-sided 95% UCB / cap: "
      f"`{inference.get('point_estimate_ms')} / "
      f"{inference.get('upper_confidence_bound_ms')} / "
      f"{inference.get('cap_ms')} ms`",
      f"- output cosine / relative L2: "
      f"`{result.get('output_cosine')} / "
      f"{result.get('output_relative_l2')}`",
      f"- peak RSS / swaps: "
      f"`{payload.get('worker_resources', {}).get('maximum_resident_kib')} "
      f"KiB / {payload.get('worker_resources', {}).get('swaps')}`",
      "",
      "The timed distribution includes state append/quantization, QK, softmax,",
      "PV, partial workspace, and reduction. It admits the component only;",
      "graph integration and every full-model, long, ABBA, and product worker",
      "remain blocked behind a separate exact integration gate.",
      "",
  ])


def main() -> int:
  args = parse_args()
  quant_group = args.quant_group
  hot_tokens = args.hot_tokens
  cold_tokens = CONTEXT_TOKENS - hot_tokens
  split_state_owner = (
      quant_group == 2 and hot_tokens == SPLIT_HOT_TOKENS)
  storage_specialized = split_state_owner and args.storage_specialized
  source_bound_path = (
      SPLIT_SOURCE_BOUND if split_state_owner else SOURCE_BOUNDS[quant_group])
  target = (
      SPECIALIZED_TARGET if storage_specialized else
      SPLIT_TARGET if split_state_owner else TARGETS[quant_group])
  algorithm = (
      SPECIALIZED_ALGORITHM if storage_specialized else
      SPLIT_ALGORITHM if split_state_owner else ALGORITHMS[quant_group])
  state_bytes = expected_state_bytes(
      quant_group, VALUE_GROUPS[quant_group], hot_tokens)
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = [
      SOURCE, RUNNER_SOURCE, BOUNDARIES, ROUTES, STATUS,
      source_bound_path, CMAKE, ENV_SCRIPT]
  if storage_specialized:
    required_paths.append(SPECIALIZED_CODEGEN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))
  git = git_state(out_dir)
  source_bound = json.loads(source_bound_path.read_text(encoding="utf-8"))
  codegen = (
      json.loads(SPECIALIZED_CODEGEN.read_text(encoding="utf-8"))
      if storage_specialized else {})
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER_SOURCE.read_text(encoding="utf-8")
  boundaries = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
  routes = json.loads(ROUTES.read_text(encoding="utf-8"))
  status_text = STATUS.read_text(encoding="utf-8")
  registered_rows = [
      row for row in boundaries.get("infra_targets", [])
      if row.get("target") == target
      and row.get("source") == "tools/direct_i8_hotcold_gqa_decode.cpp"]
  expected_definitions = (
      ["IQ36_COMPONENT_KEY_QUANT_GROUP=2",
       "IQ36_COMPONENT_VALUE_QUANT_GROUP=4",
       "IQ36_COMPONENT_HOT_TOKENS=16384",
       "IQ36_COMPONENT_UPDATE_AFTER_ATTENTION=1",
       "IQ36_COMPONENT_STORAGE_SPECIALIZED=1"]
      if storage_specialized else
      ["IQ36_COMPONENT_KEY_QUANT_GROUP=2",
       "IQ36_COMPONENT_VALUE_QUANT_GROUP=4",
       "IQ36_COMPONENT_HOT_TOKENS=16384",
       "IQ36_COMPONENT_UPDATE_AFTER_ATTENTION=1"]
      if split_state_owner else
      ["IQ36_COMPONENT_QUANT_GROUP=4"] if quant_group == 4 else
      ["IQ36_COMPONENT_KEY_QUANT_GROUP=2",
       "IQ36_COMPONENT_VALUE_QUANT_GROUP=4"]
      if quant_group == 2 else None)
  registered = bool(
      len(registered_rows) == 1
      and registered_rows[0].get("compile_definitions") ==
          expected_definitions)
  bound_admitted = (
      source_bound.get("component_admitted") is True
      and source_bound.get("git", {}).get("dirty") is False
      and (
          (split_state_owner
           and source_bound.get("verdict") ==
               "admit_one_split_state_owner_hot16k_k2_v4_component"
           and source_bound.get("selected_component", {}).get(
               "logical_hot_tokens") == SPLIT_HOT_TOKENS
           and source_bound.get("selected_component", {}).get(
               "logical_cold_tokens") == cold_tokens
           and source_bound.get("selected_component", {}).get(
               "state_bytes", {}).get("total") == state_bytes
           and source_bound.get("selected_component", {}).get(
               "execution_order") == "partial_then_reduce_then_update"
           and source_bound.get("selected_component", {}).get(
               "state_writer_count") == 1
           and source_bound.get("graph_source_admitted") is False
           and source_bound.get("long_worker_admitted") is False)
          or (not split_state_owner and quant_group == 32
           and source_bound.get("verdict") ==
               "admit_one_direct_i8_attention_component")
          or (not split_state_owner and quant_group == 4
              and source_bound.get("verdict") ==
                  "admit_one_direct_i8_group4_full_cold_component"
              and source_bound.get("pareto", {}).get("4", {}).get(
                  "state_bytes", {}).get("total") == state_bytes
              and source_bound.get("selected_component", {}).get(
                  "quant_group") == 4
              and source_bound.get("graph_integration_admitted") is False
              and source_bound.get("long_worker_admitted") is False)
          or (not split_state_owner and quant_group == 2
              and source_bound.get("verdict") ==
                  "admit_one_direct_i8_hybrid_k2_v4_component"
              and source_bound.get("selected_component", {}).get(
                  "key_quant_group") == 2
              and source_bound.get("selected_component", {}).get(
                  "value_quant_group") == 4
              and source_bound.get("selected_component", {}).get(
                  "state_bytes") == state_bytes
              and source_bound.get("graph_integration_admitted") is False
              and source_bound.get("long_worker_admitted") is False)))
  active_route = routes.get("active_route", {}).get("id")
  seq1277_selected = any(
      row.get("seq") == 1277 and row.get("selected_next_route") ==
          "openvino_hot_cold_partial_storage_specialized_component"
      for row in routes.get("candidate_history", [])
      if isinstance(row, dict))
  codegen_programs = {
      row.get("storage"): row for row in codegen.get("programs", [])
      if isinstance(row, dict)}
  codegen_admitted = bool(
      not storage_specialized or (
          codegen.get("required_checks_passed") is True
          and codegen.get("verdict") ==
              "admit_one_20_sample_specialized_partial_component"
          and codegen.get("git", {}).get("dirty") is False
          and codegen.get("gpu_kernel_executed") is False
          and set(codegen_programs) == {"cold", "hot"}
          and codegen_programs["cold"].get("grf") == 96
          and codegen_programs["hot"].get("grf") == 64
          and all(row.get("simd") == 16
                  and row.get("slm_bytes") == 18496
                  and row.get("spill_bytes") == 0
                  for row in codegen_programs.values())))
  source_checks = {
      "selected_bound_admits_one_component": bound_admitted,
      "selected_codegen_admits_one_specialized_component": codegen_admitted,
      "active_route_admits_this_exact_specialized_component":
          not storage_specialized
          or (active_route ==
              "openvino_hot_cold_partial_storage_specialized_component"
              and seq1277_selected
              and "one 20-sample" in status_text),
      "locked_32k_parameterized_hot_cold_window":
          "#define IQ36_CONTEXT_TOKENS 32768U" in source_text
          and "#ifndef IQ36_HOT_TOKENS" in source_text
          and "#define IQ36_HOT_TOKENS 8192U" in source_text
          and ("#define IQ36_COLD_TOKENS "
               "(IQ36_CONTEXT_TOKENS - IQ36_HOT_TOKENS)") in source_text
          and "IQ36_HOT_TOKENS != 8192U" in source_text
          and "IQ36_HOT_TOKENS != 16384U" in source_text,
      "fixed_group32_default_and_k4_v4_or_k2_v4_refinement":
          "#ifndef IQ36_QUANT_GROUP" in source_text
          and "#define IQ36_QUANT_GROUP 32" in source_text
          and "#if IQ36_KEY_QUANT_GROUP == 32" in source_text
          and "IQ36_KEY_QUANT_GROUP must be 32, 4, or 2" in source_text
          and "IQ36_VALUE_QUANT_GROUP must be 32 or 4" in source_text
          and "#ifndef IQ36_COMPONENT_QUANT_GROUP" in runner_text
          and "-DIQ36_KEY_QUANT_GROUP=" in runner_text
          and "-DIQ36_VALUE_QUANT_GROUP=" in runner_text
          and "__global const half* cold_k_scales" in source_text
          and "__global const half* cold_v_scales" in source_text,
      "dpas_f16_consumption":
          source_text.count("intel_sub_group_f16_f16_matrix_mad_k16(") >= 4,
      "dpas_friendly_k_and_v_tiles":
          COLD_K_LAYOUTS[quant_group] in runner_text
          and COLD_V_LAYOUTS[quant_group] in runner_text,
      "refinement_assembles_required_scale_fragments":
          quant_group == 32
          or ("const uint packed3 = intel_sub_group_block_read(" in source_text
              and "IQ36_LOAD_KEY_SCALE(3U)" in source_text
              and "const half16 scales = (half16)(" in source_text
              and (quant_group == 4
                   or "IQ36_LOAD_KEY_SCALE(7U)" in source_text)),
      "hybrid_decouples_key_scale_group_from_dim4_pack":
          quant_group != 2
          or ("#define IQ36_KEY_PACK_WORDS (IQ36_HEAD_DIM / 4U)" in source_text
              and "(dim & 3U)] = quantized" in source_text
              and "kKeyPackWords = kHeadDim / 4U" in runner_text),
      "scalar_local_f32_reconstruction_absent":
          "__local float local_k" not in source_text
          and "__local float local_v" not in source_text,
      "complete_timed_scope":
          "append_quantize_qk_softmax_pv_workspace_reduce" in runner_text
          and "result.update_ms = EventMs(update_event)" in runner_text
          and "result.partial_ms = EventMs(partial_event)" in runner_text
          and "result.reduce_ms = EventMs(reduce_event)" in runner_text,
      "selected_execution_order_is_bounded":
          not split_state_owner
          or ("IQ36_COMPONENT_UPDATE_AFTER_ATTENTION" in runner_text
              and "partial_then_reduce_then_update" in runner_text
              and "IQ36_COMPONENT_HOT_TOKENS" in runner_text),
      "storage_specialized_runtime_preserves_workspace_and_tail":
          not storage_specialized
          or ("iq36_direct_i8_cold_partial" in source_text
              and "iq36_direct_f16_hot_partial" in source_text
              and "IQ36_COMPONENT_STORAGE_SPECIALIZED" in runner_text
              and "kColdChunkCount" in runner_text
              and "kHotChunkCount" in runner_text
              and "EventMs(partial_event) + EventMs(hot_partial_event)"
                  in runner_text
              and "Run(queue, update, partial, hot_partial, reduce)"
                  in runner_text),
      "twenty_samples_and_exact_cap":
          "constexpr int kSamples = 20;" in runner_text
          and "constexpr double kComponentCapMs = 0.5618915;" in runner_text,
      "boundary_target_registered": registered,
  }
  sample_memory("after-source-audit", stop_bytes, memory)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  sample_memory("after-configure", stop_bytes, memory)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", target, "-j1"]
  build = run(build_command, 600)
  sample_memory("after-build", stop_bytes, memory)
  executable = BUILD_DIR / target
  build_ok = (
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())
  time_path = raw_dir / "worker.time.txt"
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} "
      f"{shlex.quote(str(executable))} {shlex.quote(str(SOURCE))}")
  sample_memory("before-component-worker", stop_bytes, memory)
  component = (
      run(["bash", "-lc", shell_command], args.timeout_s)
      if build_ok else subprocess.CompletedProcess(
          ["bash", "-lc", shell_command], 1, "", "build failed"))
  sample_memory("after-component-worker", stop_bytes, memory)
  result = parse_last_json(component.stdout)
  resources = parse_time(time_path)

  write_json(raw_dir / "build.json", {
      "configure": {
          "command": configure_command,
          "returncode": configure.returncode,
          "stdout": configure.stdout,
          "stderr": configure.stderr,
      },
      "build": {
          "command": build_command,
          "returncode": build.returncode,
          "stdout": build.stdout,
          "stderr": build.stderr,
      },
  })
  (raw_dir / "component.stdout").write_text(
      component.stdout, encoding="utf-8")
  (raw_dir / "component.stderr").write_text(
      component.stderr, encoding="utf-8")
  write_json(raw_dir / "component-command.json", {
      "command": ["bash", "-lc", shell_command],
      "returncode": component.returncode,
  })
  write_json(raw_dir / "environment.json", environment())

  total_samples = [
      float(row.get("total_ms", 1e9))
      for row in result.get("samples", [])]
  performance_inference = latency_cap_inference(
      total_samples, cap=CAP_MS, min_samples=20)
  numeric_pass = bool(
      result.get("finite") is True
      and result.get("numeric_pass") is True
      and float(result.get("output_cosine", 0.0)) >= 0.999
      and float(result.get("output_relative_l2", 1.0)) <= 0.002)
  fixed_shape = bool(
      result.get("algorithm") == algorithm
      and result.get("context_tokens") == CONTEXT_TOKENS
      and result.get("hot_tokens") == hot_tokens
      and result.get("cold_tokens") == cold_tokens
      and result.get("chunk_tokens") == 512
      and result.get("head_dim") == 256
      and result.get("q_head_count") == 16
      and result.get("kv_head_count") == 2
      and result.get("quant_group") == quant_group
      and result.get("key_quant_group") == quant_group
      and result.get("value_quant_group") == VALUE_GROUPS[quant_group]
      and result.get("key_pack_dimensions") == 4
      and result.get("cold_k_layout") == COLD_K_LAYOUTS[quant_group]
      and result.get("cold_v_layout") == COLD_V_LAYOUTS[quant_group]
      and result.get("scale_dtype") == "fp16"
      and result.get("state_bytes") == state_bytes
      and result.get("execution_order") == (
          "partial_then_reduce_then_update" if split_state_owner else
          "update_then_partial_then_reduce")
      and result.get("storage_specialized") is storage_specialized
      and result.get("partial_dispatches") == (
          2 if storage_specialized else 1)
      and result.get("subgroup_size") == 16)
  distribution_shape = bool(
      len(result.get("samples", [])) == 20
      and all(
          set(row) == {"update_ms", "partial_ms", "reduce_ms", "total_ms"}
          for row in result.get("samples", [])))
  peak_rss_kib = int(resources.get("maximum_resident_kib", 1 << 62))
  swaps = int(resources.get("swaps", -1))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("fixed_source_contract", all(source_checks.values()),
            source_checks=source_checks),
      check("component_build_serial_j1", build_ok,
            build_command=build_command),
      check("component_execution", component.returncode == 0),
      check("fixed_32k_one_layer_shape", fixed_shape),
      check("twenty_sample_timing_distribution", distribution_shape),
      check("component_numeric", numeric_pass),
      check("one_sided_95pct_ucb_clears_complete_cap",
            performance_inference.get("rate_pass") is True,
            performance_inference=performance_inference),
      check("component_self_gate",
            result.get("required_checks_passed") is True),
      check("worker_rss_and_swap_are_bounded",
            peak_rss_kib < 4 * 1024 * 1024 and swaps == 0,
            maximum_resident_kib=peak_rss_kib, swaps=swaps),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  promoted_verdict = (
      "promote_hot16k_k2_v4_storage_specialized_component"
      if storage_specialized else
      "promote_split_state_owner_hot16k_k2_v4_component"
      if split_state_owner else {
      32: "promote_direct_i8_attention_component",
      4: "promote_direct_i8_group4_full_cold_component",
      2: "promote_direct_i8_hybrid_k2_v4_component",
  }[quant_group])
  rejected_verdict = (
      "reject_hot16k_k2_v4_storage_specialized_component"
      if storage_specialized else
      "reject_split_state_owner_hot16k_k2_v4_component"
      if split_state_owner else {
      32: "reject_direct_i8_attention_component",
      4: "reject_direct_i8_group4_full_cold_component",
      2: "reject_direct_i8_hybrid_k2_v4_component",
  }[quant_group])
  verdict = promoted_verdict if required else rejected_verdict
  source_paths = [SOURCE, RUNNER_SOURCE, BOUNDARIES, source_bound_path]
  if storage_specialized:
    source_paths.extend((SPECIALIZED_CODEGEN, ROUTES, STATUS))
  sources = [
      {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
      for path in source_paths]
  payload = {
      "schema_version":
          "intel-qwen36-openvino-direct-i8-attention-component-gate-v2",
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "quant_group": quant_group,
      "value_quant_group": VALUE_GROUPS[quant_group],
      "hot_tokens": hot_tokens,
      "cold_tokens": cold_tokens,
      "split_state_owner": split_state_owner,
      "storage_specialized": storage_specialized,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_promoted": required,
      "graph_integration_admitted": False,
      "full_model_worker_admitted": False,
      "long_worker_admitted": False,
      "product_claim_allowed": False,
      "checks": checks,
      "result": result,
      "performance_inference": performance_inference,
      "worker_resources": resources,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"],
      "git": git,
      "quant_group": quant_group,
      "hot_tokens": hot_tokens,
      "storage_specialized": storage_specialized,
      "required_checks_passed": required,
      "schema_version": payload["schema_version"],
      "sources": sources,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "verdict": verdict,
      "workstream": WS,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "numeric": {
          "cosine": result.get("output_cosine"),
          "relative_l2": result.get("output_relative_l2"),
          "rmse": result.get("output_rmse"),
          "max_abs": result.get("max_abs"),
      },
      "required_checks_passed": required,
      "quant_group": quant_group,
      "hot_tokens": hot_tokens,
      "storage_specialized": storage_specialized,
      "product_claim_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
    for index, row in enumerate(result.get("samples", [])):
      stream.write(json.dumps({
          "context_tokens": result.get("context_tokens"),
          "quant_group": quant_group,
          "hot_tokens": hot_tokens,
          "storage_specialized": storage_specialized,
          "sample": index,
          **row,
          "verdict": verdict,
      }, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "applicable": True,
      "dispersion": performance_inference["dispersion"],
      "role": "component_environment_telemetry_only",
      "quant_group": quant_group,
      "hot_tokens": hot_tokens,
      "storage_specialized": storage_specialized,
      "required_checks_passed": required,
  })
  (out_dir / "summary.md").write_text(summary(payload), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "verdict": verdict,
      "required_checks_passed": required,
      "median_ms": performance_inference.get("point_estimate_ms"),
      "ucb_ms": performance_inference.get("upper_confidence_bound_ms"),
      "cap_ms": CAP_MS,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
