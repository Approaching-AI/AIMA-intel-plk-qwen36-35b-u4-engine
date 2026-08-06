#!/usr/bin/env python3
"""Capture one real decode FC/conv/GDN boundary without graph observers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-provider-aware-linear-capture-v0"
WORKER = REPO / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = REPO / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
PROMPT = REPO / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_002k.txt")
CUSTOM_CONFIG = REPO / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
CANDIDATE_PLUGIN = Path(
    "/home/intel/ov/openvino_env/lib/python3.12/site-packages/openvino/libs/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_PLUGIN_SHA = (
    "58b5d38e711b6ed36e169b8958e74e9b65c464b91d874dd7588b5355a285f8fa")
EXPECTED_DIAGNOSTIC_TOP1 = (271, 64)
EXPECTED_LAYER0_FC_OUTPUT_SHA = (
    "e61b186df196bf7c5df571e98c85466ac53cbb065620384da084f932e4ee1fd2")
TARGET_LAYERS = tuple(range(3, 40, 4))
MARKER = "2k-candidate-phase1-input1-total2049"
MAX_CAPTURE_BYTES = 1024 * 1024

CAPTURES = (
    {
        "label": "linear-conv",
        "filter": "iq36_linear_conv_swish",
        "before_begin": 0,
        "before_end": 5,
        "after_begin": 3,
        "after_end": 5,
        "expected_before": tuple(range(5)),
        "expected_after": (3, 4),
    },
    {
        "label": "gdn",
        "filter": "gated_delta_net_ref",
        "before_begin": 0,
        "before_end": 8,
        "after_begin": 6,
        "after_end": 8,
        "expected_before": tuple(range(8)),
        "expected_after": (6, 7),
    },
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.memory_stop_gib <= 0.0:
    parser.error("timeout and memory stop must be positive")
  return args


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


def run(
    command: list[str], timeout_s: int, environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=REPO, env=environment, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    return {
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command, "returncode": 124,
        "stdout": str(error.stdout or ""), "stderr": str(error.stderr or ""),
        "timed_out": True,
    }


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows = []
  for line in path.read_text(encoding="utf-8").splitlines():
    value = json.loads(line)
    if isinstance(value, dict):
      rows.append(value)
  return rows


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def capture_signature(rows: list[dict[str, Any]]) -> list[list[Any]]:
  return sorted([
      [row.get("phase"), int(row.get("arg_index", -1)),
       int(row.get("bytes", 0)), int(row.get("status", -1))]
      for row in rows if row.get("event") == "capture"])


def run_capture(
    spec: dict[str, Any], raw: Path, timeout_s: int, stop_bytes: int,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
  label = str(spec["label"])
  directory = raw / label
  capture = directory / "capture"
  cache = directory / "neo-cache"
  directory.mkdir(parents=True)
  capture.mkdir()
  cache.mkdir()
  marker_path = directory / "trace-active"
  trace_path = directory / "opencl-trace.jsonl"
  result_path = directory / "worker-result.json"
  config_path = directory / "worker-config.json"
  config = {
      "collect_states": False,
      "custom_config": str(CUSTOM_CONFIG),
      "candidate_gpu_plugin": str(CANDIDATE_PLUGIN),
      "decode_steps": 1,
      "decode_tokens": [271],
      "device": "GPU",
      "lane": "2k",
      "mode": "candidate",
      "model_dir": str(MODEL_DIR),
      "prompt": str(PROMPT),
      "prefill_chunk_tokens": 8192,
      "fixed_cold_capacity": 2048,
      "initialize_hot_states": True,
      "skip_hot_state_self_bind": True,
      "dump_runtime_graph": False,
      "capture_full_profile": False,
      "fuse_linear_conv_state": True,
      "pack_gdn_state": False,
      "prefill_history_capacity": 16384,
      "phase_branch_prefill": False,
      "stock_prefill_custom_decode": False,
      "stock_prefill_sliced_decode": False,
      "static_phase_separated": False,
      "raw": str(directory),
      "result": str(result_path),
      "target_layers": list(TARGET_LAYERS),
      "trace_marker": str(marker_path),
  }
  write_json(config_path, config)
  environment = os.environ.copy()
  for name in (
      "OV_GPU_CONFIG_FILE", "IQ36_GDN_TRANSPOSED_STATE",
      "IQ36_OPENCL_CAPTURE_GEMM_M", "IQ36_OPENCL_CAPTURE_GEMM_N",
      "IQ36_OPENCL_CAPTURE_GEMM_K"):
    environment.pop(name, None)
  environment.update({
      "OV_GPU_USM_POLICY": "0",
      "IQ36_OPENCL_TRACE_FILTER": str(spec["filter"]),
      "IQ36_OPENCL_TRACE_MARKER": str(marker_path),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path),
      "IQ36_OPENCL_TRACE_TIMING": "1",
      "IQ36_OPENCL_CAPTURE_DIR": str(capture),
      "IQ36_OPENCL_CAPTURE_MARKER_FILTER": MARKER,
      "IQ36_OPENCL_CAPTURE_MAX_BYTES": str(MAX_CAPTURE_BYTES),
      "IQ36_OPENCL_CAPTURE_BEFORE_BEGIN": str(spec["before_begin"]),
      "IQ36_OPENCL_CAPTURE_BEFORE_END": str(spec["before_end"]),
      "IQ36_OPENCL_CAPTURE_AFTER_BEGIN": str(spec["after_begin"]),
      "IQ36_OPENCL_CAPTURE_AFTER_END": str(spec["after_end"]),
      "LD_AUDIT": str(TRACE_LIBRARY),
      "NEO_CACHE_DIR": str(cache),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  sample_memory(f"before-{label}-worker", stop_bytes, memory)
  worker = run(
      [str(OV_PYTHON), str(WORKER), "--worker-config", str(config_path)],
      timeout_s, environment)
  sample_memory(f"after-{label}-worker", stop_bytes, memory)
  (directory / "worker.stdout").write_text(
      worker["stdout"], encoding="utf-8")
  (directory / "worker.stderr").write_text(
      worker["stderr"], encoding="utf-8")
  write_json(directory / "worker-command.json", {
      "command": worker["command"],
      "environment": {name: environment[name] for name in (
          "OV_GPU_USM_POLICY", "IQ36_OPENCL_TRACE_FILTER",
          "IQ36_OPENCL_TRACE_MARKER", "IQ36_OPENCL_TRACE_PATH",
          "IQ36_OPENCL_TRACE_TIMING", "IQ36_OPENCL_CAPTURE_DIR",
          "IQ36_OPENCL_CAPTURE_MARKER_FILTER",
          "IQ36_OPENCL_CAPTURE_MAX_BYTES",
          "IQ36_OPENCL_CAPTURE_BEFORE_BEGIN",
          "IQ36_OPENCL_CAPTURE_BEFORE_END",
          "IQ36_OPENCL_CAPTURE_AFTER_BEGIN",
          "IQ36_OPENCL_CAPTURE_AFTER_END", "LD_AUDIT", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": worker["returncode"],
  })
  result = load_json(result_path) if result_path.is_file() else {}
  trace = load_jsonl(trace_path)
  selected = [
      row for row in trace if row.get("event") == "ndrange" and
      row.get("marker") == MARKER and
      str(spec["filter"]) in str(row.get("kernel", ""))]
  captured = [
      row for row in trace if row.get("event") == "capture" and
      row.get("marker") == MARKER]
  return {
      "label": label,
      "filter": spec["filter"],
      "returncode": worker["returncode"],
      "result": result,
      "selected_dispatches": selected,
      "capture_rows": captured,
      "capture_signature": capture_signature(captured),
      "expected_before": list(spec["expected_before"]),
      "expected_after": list(spec["expected_after"]),
      "directory": str(directory.relative_to(REPO)),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)
  required = (WORKER, OV_PYTHON, CMAKE, MODEL_DIR, PROMPT, CUSTOM_CONFIG,
              CANDIDATE_PLUGIN)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing capture inputs: " + ", ".join(missing))

  git = git_state()
  configure = run([
      str(CMAKE), "-S", str(REPO / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"], min(args.timeout_s, 600))
  build = run([
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET,
      "-j1"], min(args.timeout_s, 600))
  write_json(raw / "trace-build.json", {"configure": configure, "build": build})
  sample_memory("after-serial-trace-build", stop_bytes, memory)

  runs = []
  if (configure["returncode"] == 0 and build["returncode"] == 0 and
      TRACE_LIBRARY.is_file()):
    for spec in CAPTURES:
      runs.append(run_capture(
          spec, raw, args.timeout_s, stop_bytes, memory))

  diagnostic_phase_rows = [
      [{"top1": phase.get("top1"),
        "logits_sha256": phase.get("logits_sha256")}
       for phase in row["result"].get("phases", [])]
      for row in runs]
  layer0_fc_capture = (
      raw / "linear-conv/capture/dispatch000-arg0-before.bin")

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("capture_plugin_is_bound_stock_opencl_diagnostic",
            sha256(CANDIDATE_PLUGIN) == EXPECTED_PLUGIN_SHA,
            path=str(CANDIDATE_PLUGIN), sha256=sha256(CANDIDATE_PLUGIN)),
      check("trace_library_builds_serially",
            configure["returncode"] == 0 and build["returncode"] == 0 and
            TRACE_LIBRARY.is_file(), configure=configure, build=build),
      check("two_isolated_workers_complete",
            len(runs) == 2 and all(row["returncode"] == 0 for row in runs),
            runs=[{"label": row["label"], "returncode": row["returncode"]}
                  for row in runs]),
      check("bound_opencl_diagnostic_is_reproducible",
            len(diagnostic_phase_rows) == 2 and
            all(tuple(phase["top1"] for phase in row) ==
                EXPECTED_DIAGNOSTIC_TOP1 for row in diagnostic_phase_rows) and
            diagnostic_phase_rows[0] == diagnostic_phase_rows[1],
            expected_top1=list(EXPECTED_DIAGNOSTIC_TOP1),
            phases=diagnostic_phase_rows,
            note=("the bound stock OpenCL diagnostic backend has a different "
                  "phase-1 token from the accepted production Level Zero "
                  "carrier; this gate admits only component numeric evidence")),
      check("captured_linear_conv_input_is_locked_layer0_fc_output",
            layer0_fc_capture.is_file() and
            sha256(layer0_fc_capture) == EXPECTED_LAYER0_FC_OUTPUT_SHA,
            expected_sha256=EXPECTED_LAYER0_FC_OUTPUT_SHA,
            observed_sha256=(sha256(layer0_fc_capture)
                             if layer0_fc_capture.is_file() else None)),
      check("all_thirty_decode_dispatches_are_live_per_capture",
            len(runs) == 2 and all(
                len(row["selected_dispatches"]) == 30 and
                all(int(item.get("duration_ns", 0)) > 0
                    for item in row["selected_dispatches"])
                for row in runs)),
      check("all_requested_usm_arguments_captured",
            len(runs) == 2 and all(
                sorted(int(item["arg_index"]) for item in row["capture_rows"]
                       if item.get("phase") == "before") ==
                    row["expected_before"] and
                sorted(int(item["arg_index"]) for item in row["capture_rows"]
                       if item.get("phase") == "after") ==
                    row["expected_after"] and
                all(item.get("status") == 0 and item.get("bytes", 0) > 0
                    for item in row["capture_rows"])
                for row in runs),
            signatures={row["label"]: row["capture_signature"] for row in runs}),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "host": {"target_alias": "intel-ptl-local", "kernel": platform.release()},
      "verdict": (
          "admit_real_layer0_component_inputs" if required_checks_passed else
          "reject_capture_before_component"),
      "required_checks_passed": required_checks_passed,
      "component_execution_admitted": required_checks_passed,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "runs": runs,
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": f"{SCHEMA}-manifest-v0",
      "created_at": metrics["created_at"],
      "artifact": str(output.relative_to(REPO)),
      "git": git,
      "candidate_plugin": str(CANDIDATE_PLUGIN),
      "candidate_plugin_sha256": sha256(CANDIDATE_PLUGIN),
      "trace_library": str(TRACE_LIBRARY.relative_to(REPO)),
      "trace_library_sha256": sha256(TRACE_LIBRARY),
      "worker": str(WORKER.relative_to(REPO)),
      "worker_sha256": sha256(WORKER),
      "memory_stop_gib": args.memory_stop_gib,
  })
  durations = {
      row["label"]: int(row["selected_dispatches"][0]["duration_ns"])
      if row["selected_dispatches"] else None for row in runs}
  summary = f"""# OpenVINO provider-aware linear capture

Verdict: **{metrics['verdict']}**. Required checks:
`{str(required_checks_passed).lower()}`.

Two isolated 2k candidate-graph workers captured the first real layer-0 decode
`IQ36LinearConvSwish` and stock GatedDeltaNet dispatches under the bound stock
plugin's OpenCL-USM diagnostic path. Event durations were `{durations}`;
timing is capture telemetry, not component admission evidence.

The diagnostic backend reproducibly emitted top-1
`{list(EXPECTED_DIAGNOSTIC_TOP1)}` in both workers. This is not the production
Level Zero token oracle; its scope is exact component numeric comparison only.

All requested USM input/output arguments are stored under `raw/`. The next gate
may execute one parameterized component against these exact outputs and state.
No graph integration, long row, ABBA, or output512 is admitted here.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(REPO)),
      "verdict": metrics["verdict"],
      "required_checks_passed": required_checks_passed,
      "durations_ns": durations,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
