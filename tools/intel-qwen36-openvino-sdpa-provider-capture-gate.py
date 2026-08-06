#!/usr/bin/env python3
"""Capture the pinned exact-128k OpenVINO optimized-SDPA provider program."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-sdpa-provider-capture-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_MODEL = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
OCLOC = Path("/usr/bin/ocloc")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
EXPECTED_EXEC_TYPE = "ocl::sdpa::opt__f16"
EXPECTED_LAYERS = 10
COMPONENT_CAP_US = 2825.0


WORKER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


LM_HEAD_NAME = "__module.model.lm_head/ov_ext::linear/MatMul"


def cache_files(cache_dir: Path) -> set[str]:
  return {
      str(path.relative_to(cache_dir))
      for path in cache_dir.rglob("*.cl_cache") if path.is_file()
  }


def make_inputs(start: int, token_count: int, total: int):
  positions = np.arange(start, start + token_count, dtype=np.int64)
  return {
      "attention_mask": np.ones((1, total), dtype=np.int64),
      "inputs_embeds": np.zeros((1, token_count, 2048), dtype=np.float32),
      "position_ids": np.tile(positions, (4, 1)).reshape(
          4, 1, token_count),
      "beam_idx": np.zeros((1,), dtype=np.int32),
  }


def attention_profile(request):
  rows = []
  for item in request.get_profiling_info():
    if item.node_type != "IndirectSDPA" and (
        "scaled_dot_product_attention" not in item.node_name):
      continue
    rows.append({
        "exec_type": item.exec_type,
        "node_name": item.node_name,
        "node_type": item.node_type,
        "real_time_us": item.real_time.total_seconds() * 1_000_000.0,
        "status": str(item.status),
    })
  rows.sort(key=lambda row: row["node_name"])
  return rows


def state_shapes(request):
  rows = []
  for state in request.query_state():
    rows.append({
        "element_type": str(state.state.element_type),
        "name": state.name,
        "shape": list(state.state.shape),
    })
  return rows


def main():
  config_path = Path(sys.argv[1])
  cfg = json.loads(config_path.read_text(encoding="utf-8"))
  cache_dir = Path(cfg["cache_dir"])
  core = ov.Core()
  source = core.read_model(cfg["model"])
  lm_head = next(
      node for node in source.get_ops()
      if node.get_friendly_name() == LM_HEAD_NAME)
  hidden_result = ov.opset13.result(lm_head.input_value(0))
  hidden_result.set_friendly_name("hidden_states_result")
  hidden = ov.Model(
      [hidden_result], source.get_sinks(), source.get_parameters(),
      "language_model_exact_context_sdpa_capture")
  compile_config = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
  }
  compile_started = time.perf_counter()
  compiled = core.compile_model(hidden, cfg["device"], compile_config)
  compile_wall_ms = (time.perf_counter() - compile_started) * 1000.0
  request = compiled.create_infer_request()
  request.reset_state()

  context_tokens = int(cfg["context_tokens"])
  chunk_tokens = int(cfg["chunk_tokens"])
  # Seed one token short. The first query-one request then has an exact
  # source/KV length of context_tokens, matching the registered component.
  seed_tokens = context_tokens - 1
  chunks = []
  start = 0
  seed_started = time.perf_counter()
  while start < seed_tokens:
    count = min(chunk_tokens, seed_tokens - start)
    wall_started = time.perf_counter()
    outputs = request.infer(make_inputs(start, count, start + count))
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    chunks.append({
        "end_token": start + count,
        "output_shapes": [list(value.shape) for value in outputs.values()],
        "start_token": start,
        "token_count": count,
        "wall_ms": wall_ms,
    })
    start += count
    if len(chunks) % 16 == 0 or start == seed_tokens:
      print(json.dumps({
          "event": "seed_progress", "seeded_tokens": start,
          "seed_tokens": seed_tokens, "wall_ms": wall_ms,
      }), flush=True)
  seed_wall_ms = (time.perf_counter() - seed_started) * 1000.0

  cache_before_exact = cache_files(cache_dir)
  cache_before_exact_snapshot = sorted(cache_before_exact)
  probes = []
  for probe_index in range(int(cfg["decode_probes"])):
    source_tokens = context_tokens + probe_index
    if cfg.get("trace_marker"):
      Path(cfg["trace_marker"]).write_text(
          str(source_tokens) + "\n", encoding="utf-8")
    wall_started = time.perf_counter()
    outputs = request.infer(make_inputs(
        source_tokens - 1, 1, source_tokens))
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    cache_after = cache_files(cache_dir)
    rows = attention_profile(request)
    times = [float(row["real_time_us"]) for row in rows]
    probe = {
        "attention_max_us": max(times) if times else None,
        "attention_median_us": statistics.median(times) if times else None,
        "attention_min_us": min(times) if times else None,
        "attention_rows": rows,
        "cache_delta": sorted(cache_after - cache_before_exact),
        "output_shapes": [list(value.shape) for value in outputs.values()],
        "probe_index": probe_index,
        "source_tokens": source_tokens,
        "wall_ms": wall_ms,
    }
    probes.append(probe)
    print(json.dumps({
        "event": "decode_probe", "source_tokens": source_tokens,
        "attention_median_us": probe["attention_median_us"],
        "attention_max_us": probe["attention_max_us"],
        "wall_ms": wall_ms,
    }), flush=True)
    cache_before_exact = cache_after

  result = {
      "cache_after": sorted(cache_files(cache_dir)),
      "cache_before_exact": cache_before_exact_snapshot,
      "chunk_count": len(chunks),
      "chunk_last": chunks[-1] if chunks else None,
      "chunk_tokens": chunk_tokens,
      "compile_config": compile_config,
      "compile_wall_ms": compile_wall_ms,
      "compiled_properties": {
          "DYNAMIC_QUANTIZATION_GROUP_SIZE": compiled.get_property(
              "DYNAMIC_QUANTIZATION_GROUP_SIZE"),
          "PERFORMANCE_HINT": str(compiled.get_property("PERFORMANCE_HINT")),
          "PERF_COUNT": compiled.get_property("PERF_COUNT"),
      },
      "context_tokens": context_tokens,
      "decode_probes": probes,
      "device": cfg["device"],
      "openvino_version": ov.get_version(),
      "seed_tokens": seed_tokens,
      "seed_wall_ms": seed_wall_ms,
      "state_shapes_after_probes": state_shapes(request),
  }
  Path(cfg["result_path"]).write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
'''


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--context-tokens", type=int, default=131072)
  parser.add_argument("--chunk-tokens", type=int, default=1024)
  parser.add_argument("--decode-probes", type=int, default=3)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--model", type=Path, default=OV_MODEL)
  parser.add_argument("--timeout-s", type=int, default=1800)
  return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state(out_dir: Path) -> dict[str, Any]:
  def command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else ""

  dirty = command("status", "--porcelain").splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def disassemble(binary: Path, dump_dir: Path, timeout_s: int) -> dict[str, Any]:
  dump_dir.mkdir(parents=True, exist_ok=False)
  command = [str(OCLOC), "disasm", "-file", str(binary), "-dump", str(dump_dir)]
  run = subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout_s)
  ze_info = dump_dir / ".ze_info"
  build_options = dump_dir / ".misc.buildOptions"
  return {
      "build_options": (
          build_options.read_text(encoding="utf-8", errors="replace")
          if build_options.is_file() else ""),
      "command": command,
      "returncode": run.returncode,
      "stderr": run.stderr,
      "stdout": run.stdout,
      "ze_info": (
          ze_info.read_text(encoding="utf-8", errors="replace")
          if ze_info.is_file() else ""),
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  cache_dir = raw_dir / "neo-cache"
  raw_dir.mkdir(parents=True, exist_ok=False)
  cache_dir.mkdir()
  git = git_state(out_dir)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure = subprocess.run(
      configure_command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET, "-j8"]
  build = subprocess.run(
      build_command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=600)
  trace_build_ok = bool(
      configure.returncode == 0 and build.returncode == 0
      and TRACE_LIBRARY.is_file())
  write_json(raw_dir / "trace-build.json", {
      "build": {"command": build_command, "returncode": build.returncode,
                "stderr": build.stderr, "stdout": build.stdout},
      "configure": {"command": configure_command,
                    "returncode": configure.returncode,
                    "stderr": configure.stderr, "stdout": configure.stdout},
      "library": str(TRACE_LIBRARY),
  })

  worker_path = raw_dir / "openvino-sdpa-worker.py"
  worker_path.write_text(WORKER)
  worker_result_path = raw_dir / "worker-result.json"
  worker_config = {
      "cache_dir": str(cache_dir),
      "chunk_tokens": args.chunk_tokens,
      "context_tokens": args.context_tokens,
      "decode_probes": args.decode_probes,
      "device": args.device,
      "model": str(args.model),
      "result_path": str(worker_result_path),
      "trace_marker": str(raw_dir / "trace-active"),
  }
  config_path = raw_dir / "worker-config.json"
  write_json(config_path, worker_config)
  command = [str(args.openvino_python), str(worker_path), str(config_path)]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_TRACE_FILTER": "sdpa_",
      "IQ36_OPENCL_TRACE_MARKER": str(raw_dir / "trace-active"),
      "IQ36_OPENCL_TRACE_PATH": str(raw_dir / "dispatch-trace.jsonl"),
      "IQ36_OPENCL_TRACE_TIMING": "1",
      "LD_AUDIT": str(TRACE_LIBRARY),
      "NEO_CACHE_DIR": str(cache_dir),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  worker = (
      subprocess.run(
          command, cwd=ROOT, env=env, check=False, capture_output=True,
          text=True, encoding="utf-8", errors="replace",
          timeout=args.timeout_s)
      if trace_build_ok else subprocess.CompletedProcess(
          command, 1, "", "dispatch trace build failed"))
  (raw_dir / "worker.stdout").write_text(worker.stdout)
  (raw_dir / "worker.stderr").write_text(worker.stderr)
  write_json(raw_dir / "worker-command.json", {
      "command": command,
      "environment": {
          key: env[key] for key in (
              "IQ36_OPENCL_TRACE_FILTER", "IQ36_OPENCL_TRACE_MARKER",
              "IQ36_OPENCL_TRACE_PATH", "IQ36_OPENCL_TRACE_TIMING",
              "LD_AUDIT", "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE",
              "NEO_CACHE_PERSISTENT")
      },
      "returncode": worker.returncode,
  })
  worker_result = (
      json.loads(worker_result_path.read_text())
      if worker_result_path.is_file() else {})
  dispatch_trace_path = raw_dir / "dispatch-trace.jsonl"
  dispatch_rows = []
  if dispatch_trace_path.is_file():
    for line in dispatch_trace_path.read_text().splitlines():
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(row, dict):
        dispatch_rows.append(row)

  probes = worker_result.get("decode_probes", [])
  exact = probes[0] if probes else {}
  # NEO may compile the dynamic-shape SDPA bundle during the seed requests,
  # before the first exact query-one dispatch. The cache directory started
  # empty and contains programs from this one pinned model only, so retain the
  # complete optimized-SDPA bundle and separately report the exact-request
  # delta. Exact provider selection is proved by PERF_COUNT below.
  exact_delta = exact.get("cache_delta", [])
  captures = []
  cache_paths = sorted(cache_dir.rglob("*.cl_cache"))
  for binary in cache_paths:
    relative = str(binary.relative_to(cache_dir))
    data = binary.read_bytes()
    if b"sdpa_" not in data:
      continue
    digest = sha256(binary)
    dump_dir = raw_dir / "sdpa-disassembly" / digest[:16]
    disasm = disassemble(binary, dump_dir, min(args.timeout_s, 300))
    kernel_names = sorted(set(
        token.decode("ascii", errors="ignore")
        for token in re.findall(rb"sdpa_[A-Za-z0-9_]+", data)))
    captures.append({
        "build_options": disasm["build_options"],
        "disassembly_returncode": disasm["returncode"],
        "kernel_names": kernel_names,
        "relative_path": relative,
        "sha256": digest,
        "size_bytes": binary.stat().st_size,
        "ze_info_has_finalization": "sdpa_opt__finalization" in disasm["ze_info"],
        "ze_info_has_single_reg": "sdpa_opt__single_reg" in disasm["ze_info"],
    })
    write_json(dump_dir / "command.json", {
        key: disasm[key] for key in ("command", "returncode", "stdout", "stderr")
    })

  attention_rows = exact.get("attention_rows", [])
  times = [float(row.get("real_time_us", 0.0)) for row in attention_rows]
  selection_pass = bool(
      len(attention_rows) == EXPECTED_LAYERS
      and all(row.get("exec_type") == EXPECTED_EXEC_TYPE for row in attention_rows)
      and all(row.get("node_type") == "IndirectSDPA" for row in attention_rows)
      and all(value > 0.0 for value in times))
  capture_pass = bool(
      captures
      and all(row["disassembly_returncode"] == 0 for row in captures))
  shape_pass = bool(
      args.context_tokens == 131072
      and args.chunk_tokens == 1024
      and worker_result.get("seed_tokens") == 131071
      and exact.get("source_tokens") == 131072)
  source_pass = bool(
      OV_SOURCE.is_dir()
      and args.model.resolve() == OV_MODEL.resolve()
      and args.openvino_python.resolve() == OV_PYTHON.resolve())
  exact_dispatches = [
      row for row in dispatch_rows
      if row.get("marker") == str(args.context_tokens)
  ]
  dispatch_names = sorted(set(
      str(row.get("kernel", "")) for row in exact_dispatches))
  captured_names = {
      name for capture in captures for name in capture.get("kernel_names", [])
  }
  dispatch_binary_mapping_pass = bool(
      dispatch_names and all(name in captured_names for name in dispatch_names))
  dispatch_metadata_pass = bool(
      exact_dispatches
      and all(row.get("status") == 0 for row in exact_dispatches)
      and all(row.get("timing_status") == 0 for row in exact_dispatches)
      and all(int(row.get("duration_ns", 0)) > 0 for row in exact_dispatches)
      and all(row.get("global_size") for row in exact_dispatches)
      and all(row.get("local_size") for row in exact_dispatches)
      and all(len(row.get("args", [])) == 9 for row in exact_dispatches)
      and all(all("mem_bytes" in arg for arg in row["args"][:6])
              for row in exact_dispatches)
      and all("head_hex" in row["args"][0] for row in exact_dispatches)
      and any("sdpa_" in name for name in dispatch_names)
      and dispatch_binary_mapping_pass)
  dispatch_durations_ns = [
      int(row.get("duration_ns", 0)) for row in exact_dispatches]
  dispatch_cap_direction = bool(
      dispatch_durations_ns
      and max(dispatch_durations_ns) <= COMPONENT_CAP_US * 1000.0)
  profile_cap_direction = bool(times and max(times) <= COMPONENT_CAP_US)
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "pinned_provider_source", "pass": source_pass,
       "openvino_commit": OV_COMMIT},
      {"name": "dispatch_trace_build", "pass": trace_build_ok},
      {"name": "worker_execution", "pass": worker.returncode == 0},
      {"name": "fixed_exact_context_shape", "pass": shape_pass},
      {"name": "optimized_indirect_sdpa_selected", "pass": selection_pass},
      {"name": "exact_dispatch_binary_captured", "pass": capture_pass},
      {"name": "exact_dispatch_metadata_captured",
       "pass": dispatch_metadata_pass},
  ]
  required = all(bool(check["pass"]) for check in checks)
  result = {
      "binary_captures": captures,
      "component_cap_us": COMPONENT_CAP_US,
      "context_tokens": args.context_tokens,
      "exact_request_cache_delta": exact_delta,
      "exact_dispatch_count": len(exact_dispatches),
      "exact_dispatch_duration_ns": dispatch_durations_ns,
      "exact_dispatch_max_us": (
          max(dispatch_durations_ns) / 1000.0
          if dispatch_durations_ns else None),
      "exact_dispatch_median_us": (
          statistics.median(dispatch_durations_ns) / 1000.0
          if dispatch_durations_ns else None),
      "exact_dispatch_min_us": (
          min(dispatch_durations_ns) / 1000.0
          if dispatch_durations_ns else None),
      "exact_dispatch_cap_direction_pass": dispatch_cap_direction,
      "exact_dispatch_kernel_names": dispatch_names,
      "exact_dispatch_binary_mapping_pass": dispatch_binary_mapping_pass,
      "exact_dispatch_metadata_pass": dispatch_metadata_pass,
      "exact_profile": exact,
      "expected_exec_type": EXPECTED_EXEC_TYPE,
      "openvino_commit": OV_COMMIT,
      "profile_cap_direction_pass": profile_cap_direction,
      "profile_max_us": max(times) if times else None,
      "profile_median_us": statistics.median(times) if times else None,
      "profile_min_us": min(times) if times else None,
      "provider_selection_pass": selection_pass,
      "seed_wall_ms": worker_result.get("seed_wall_ms"),
  }
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": required,
      "result": result,
      "route_label": "provider_source_captured" if required else "rejected",
      "schema_version": SCHEMA,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"],
      "git": git,
      "required_checks_passed": required,
      "route_label": payload["route_label"],
      "schema_version": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "applicable": False,
      "reason": "provider selection/capture only; native replay numeric gate remains open",
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w") as handle:
    for probe in probes:
      handle.write(json.dumps({
          "attention_max_us": probe.get("attention_max_us"),
          "attention_median_us": probe.get("attention_median_us"),
          "attention_min_us": probe.get("attention_min_us"),
          "provider": EXPECTED_EXEC_TYPE,
          "source_tokens": probe.get("source_tokens"),
          "speedup_claims_allowed": False,
          "wall_ms": probe.get("wall_ms"),
      }, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "exact-shape source capture is not a context-ladder claim",
      "required_checks_passed": required,
  })
  summary = [
      "# Exact-128k optimized-SDPA provider capture gate",
      "",
      f"- required checks passed: `{str(required).lower()}`",
      f"- provider: `{EXPECTED_EXEC_TYPE}`",
      f"- exact layer rows: `{len(attention_rows)}`",
      f"- provider profile min / median / max: `{result['profile_min_us']} / {result['profile_median_us']} / {result['profile_max_us']} us`",
      f"- profile direction under 2.825 ms: `{str(profile_cap_direction).lower()}`",
      f"- exact dispatch binaries: `{len(captures)}`",
      f"- exact traced dispatches: `{len(exact_dispatches)}`",
      f"- exact traced kernels: `{', '.join(dispatch_names)}`",
      f"- exact event min / median / max: `{result['exact_dispatch_min_us']} / {result['exact_dispatch_median_us']} / {result['exact_dispatch_max_us']} us`",
      "- native replay / product speed admitted: `false / false`",
      "",
      "This artifact accepts only exact provider selection and reproducible",
      "offline program capture. Native-only replay, numeric, repeat/confirm,",
      "integration, output-512, and product correctness remain open.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary))
  print(json.dumps({
      "binary_captures": len(captures),
      "out_dir": str(out_dir.relative_to(ROOT)),
      "profile_max_us": result["profile_max_us"],
      "profile_median_us": result["profile_median_us"],
      "required_checks_passed": required,
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
