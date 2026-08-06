#!/usr/bin/env python3
"""Gate one real decode-only provider-aware linear/GDN component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-provider-aware-linear-component-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
PLUGIN = Path(
    "/home/intel/ov/openvino_env/lib/python3.12/site-packages/openvino/libs/"
    "libopenvino_intel_gpu_plugin.so")
PLUGIN_SHA256 = (
    "58b5d38e711b6ed36e169b8958e74e9b65c464b91d874dd7588b5355a285f8fa")
CONFIG = ROOT / (
    "engine/openvino/custom/iq36_provider_aware_linear_decode.xml")
SOURCE = ROOT / (
    "engine/openvino/custom/iq36_provider_aware_linear_decode.cl")
CAPTURE = ROOT / (
    "output/openvino-provider-aware-linear-capture-"
    "20260715Tseq1236c-cleanZ")

FEATURES = 8192
CONV_STATE = 4
VALUE_HEADS = 32
HEAD_SIZE = 128
GDN_STATE_ELEMENTS = VALUE_HEADS * HEAD_SIZE * HEAD_SIZE
GDN_STATE_BYTES = GDN_STATE_ELEMENTS * 2
ATTENTION_ELEMENTS = VALUE_HEADS * HEAD_SIZE
CONV_STATE_ELEMENTS = FEATURES * CONV_STATE
PACKED_ELEMENTS = (
    ATTENTION_ELEMENTS + CONV_STATE_ELEMENTS + GDN_STATE_ELEMENTS)
ALL_LAYER_COUNT = 30
ADJACENT_TARGET_MS = 0.8273600000025967
PER_LAYER_TARGET_US = ADJACENT_TARGET_MS * 1000.0 / ALL_LAYER_COUNT
REQUIRED_STATE_GBPS = 76.04254496205104

INPUTS = (
    ("fc_output", "linear-conv", 0, (1, 1, 1, FEATURES)),
    ("previous_conv_state", "linear-conv", 1,
     (1, 1, FEATURES, CONV_STATE)),
    ("conv_weights", "linear-conv", 2,
     (1, 1, FEATURES, CONV_STATE)),
    ("initial_gdn_state", "gdn", 3,
     (1, VALUE_HEADS, HEAD_SIZE, HEAD_SIZE)),
    ("gate", "gdn", 4, (1, 1, VALUE_HEADS, 1)),
    ("beta", "gdn", 5, (1, 1, VALUE_HEADS, 1)),
)
REFERENCES = (
    ("attention", "gdn", 6, (1, 1, VALUE_HEADS, HEAD_SIZE)),
    ("next_conv_state", "linear-conv", 4,
     (1, 1, FEATURES, CONV_STATE)),
    ("next_gdn_state", "gdn", 7,
     (1, VALUE_HEADS, HEAD_SIZE, HEAD_SIZE)),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--warmup", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=31)
  parser.add_argument("--trials", type=int, default=2)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if (args.warmup < 1 or args.repeat < 5 or not 1 <= args.trials <= 3 or
      args.timeout_s <= 0 or args.memory_stop_gib <= 0.0):
    parser.error("warmup/repeat/trials/timeout/memory arguments are invalid")
  if args.out_dir is None and args.worker_config is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / (
        f"output/openvino-provider-aware-linear-component-{stamp}")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


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


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      text=True, capture_output=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      text=True, capture_output=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def capture_path(label: str, index: int, phase: str) -> Path:
  return CAPTURE / (
      f"raw/{label}/capture/dispatch000-arg{index}-{phase}.bin")


def custom_class(ov: Any) -> type:
  class IQ36ProviderAwareLinearDecode(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(1)
      self.set_output_type(
          0, self.get_input_element_type(0),
          ov.PartialShape([1, 1, 1, PACKED_ELEMENTS]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36ProviderAwareLinearDecode(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36ProviderAwareLinearDecode


def make_model(ov: Any) -> Any:
  parameters = [
      ov.opset13.parameter(shape, ov.Type.f16, name=name)
      for name, _label, _index, shape in INPUTS]
  operation = custom_class(ov)([
      parameter.output(0) for parameter in parameters])
  operation.set_friendly_name("iq36_provider_aware_linear_decode")
  return ov.Model(
      [operation.output(0)], parameters,
      "iq36_provider_aware_linear_decode_component")


def profile_rows(request: Any) -> list[dict[str, Any]]:
  return [{
      "node_name": row.node_name,
      "node_type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()]


def compare(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  lhs = np.asarray(reference)
  rhs = np.asarray(candidate)
  delta = rhs.astype(np.float32) - lhs.astype(np.float32)
  return {
      "count": int(lhs.size),
      "same_shape": bool(lhs.shape == rhs.shape),
      "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
      "exact_bits": bool(np.array_equal(lhs, rhs)),
      "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
      "reference_sha256": hashlib.sha256(
          np.ascontiguousarray(lhs).tobytes()).hexdigest(),
      "candidate_sha256": hashlib.sha256(
          np.ascontiguousarray(rhs).tobytes()).hexdigest(),
  }


def worker_main(config_path: Path) -> int:
  import numpy as np
  import openvino as ov

  cfg = load_json(config_path)
  raw = Path(cfg["raw"])
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(PLUGIN.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  core.set_property("GPU", {"CONFIG_FILE": str(CONFIG)})
  compiled = core.compile_model(
      make_model(ov), "GPU",
      {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})

  values = {}
  for name, label, index, shape in INPUTS:
    values[name] = np.fromfile(
        capture_path(label, index, "before"), dtype="<f2").reshape(shape)
  references = [
      np.fromfile(capture_path(label, index, "after"), dtype="<f2")
      .reshape(shape)
      for _name, label, index, shape in REFERENCES]
  feed = {compiled.input(name): value for name, value in values.items()}

  trials = []
  for trial_index in range(int(cfg["trials"])):
    request = compiled.create_infer_request()
    for _ in range(int(cfg["warmup"])):
      request.infer(feed, share_outputs=False)
    kernel_samples = []
    wall_samples = []
    custom_counts = []
    final_profiles: list[dict[str, Any]] = []
    for _ in range(int(cfg["repeat"])):
      started = time.perf_counter_ns()
      request.infer(feed, share_outputs=False)
      wall_samples.append((time.perf_counter_ns() - started) / 1000.0)
      profiles = profile_rows(request)
      custom = [
          row for row in profiles
          if row["status"] == "Status.EXECUTED" and
          (row["node_type"] == "IQ36ProviderAwareLinearDecode" or
           "provider_aware_linear_decode" in row["node_name"].lower())]
      custom_counts.append(len(custom))
      kernel_samples.append(sum(row["real_time_us"] for row in custom))
      final_profiles = profiles

    packed = np.array(request.get_output_tensor(0).data, copy=True).reshape(-1)
    outputs = []
    offset = 0
    for reference in references:
      count = int(reference.size)
      outputs.append(packed[offset:offset + count].reshape(reference.shape))
      offset += count
    if offset != packed.size:
      raise RuntimeError(
          f"packed output size mismatch: consumed {offset}, got {packed.size}")
    comparisons = [
        {"name": name, **compare(reference, candidate, np)}
        for (name, _label, _index, _shape), reference, candidate in
        zip(REFERENCES, references, outputs)]
    for (name, _label, _index, _shape), value in zip(REFERENCES, outputs):
      np.ascontiguousarray(value).tofile(
          raw / f"trial{trial_index}-{name}.f16")
    trials.append({
        "trial": trial_index,
        "comparisons": comparisons,
        "custom_profile_counts": custom_counts,
        "kernel_us_samples": kernel_samples,
        "kernel_us_median": statistics.median(kernel_samples),
        "wall_us_samples": wall_samples,
        "wall_us_median": statistics.median(wall_samples),
        "final_profile": final_profiles,
    })

  write_json(Path(cfg["result"]), {
      "openvino_version": ov.get_version(),
      "plugin": str(PLUGIN),
      "plugin_sha256": sha256(PLUGIN),
      "trials": trials,
  })
  return 0


def run_worker(
    config: Path, timeout_s: int, stdout: Path, stderr: Path,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        [str(OV_PYTHON), str(Path(__file__).resolve()),
         "--worker-config", str(config)],
        cwd=ROOT, text=True, capture_output=True, timeout=timeout_s,
        check=False, encoding="utf-8", errors="replace")
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    return {"returncode": result.returncode, "timed_out": False}
  except subprocess.TimeoutExpired as error:
    stdout.write_text(str(error.stdout or ""), encoding="utf-8")
    stderr.write_text(str(error.stderr or ""), encoding="utf-8")
    return {"returncode": 124, "timed_out": True}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)
  required = [OV_PYTHON, PLUGIN, CONFIG, SOURCE, CAPTURE / "metrics.json"]
  required.extend(capture_path(label, index, "before")
                  for _name, label, index, _shape in INPUTS)
  required.extend(capture_path(label, index, "after")
                  for _name, label, index, _shape in REFERENCES)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  git = git_state()
  capture_metrics = load_json(CAPTURE / "metrics.json")
  activated = capture_path("linear-conv", 3, "after")
  gdn_input = capture_path("gdn", 0, "before")
  config = raw / "worker-config.json"
  result_path = raw / "worker-result.json"
  write_json(config, {
      "raw": str(raw), "result": str(result_path),
      "warmup": args.warmup, "repeat": args.repeat,
      "trials": args.trials,
  })
  sample_memory("before-component-worker", stop_bytes, memory)
  worker = run_worker(
      config, args.timeout_s, raw / "worker.stdout", raw / "worker.stderr")
  sample_memory("after-component-worker", stop_bytes, memory)
  result = load_json(result_path) if result_path.is_file() else {}
  trials = result.get("trials", [])
  medians = [float(row.get("kernel_us_median", float("inf")))
             for row in trials]
  best_kernel_us = min(medians, default=0.0)
  timing_decisive = (
      len(medians) == args.trials and best_kernel_us > 0.0 and
      math.isfinite(best_kernel_us))
  all_layer_ms = (
      best_kernel_us * ALL_LAYER_COUNT / 1000.0
      if timing_decisive else 0.0)
  state_gbps = (
      (2 * GDN_STATE_BYTES) / (best_kernel_us * 1000.0)
      if timing_decisive else 0.0)
  all_exact = (
      len(trials) == args.trials and all(
          len(row.get("comparisons", [])) == len(REFERENCES) and
          all(item.get("finite") and item.get("same_shape") and
              item.get("exact_bits")
              for item in row.get("comparisons", []))
          for row in trials))
  profile_exact = (
      len(trials) == args.trials and all(
          len(row.get("custom_profile_counts", [])) == args.repeat and
          all(count == 1 for count in row["custom_profile_counts"]) and
          all(float(value) > 0.0 for value in row["kernel_us_samples"])
          for row in trials))
  input_sizes = {
      name: capture_path(label, index, "before").stat().st_size
      for name, label, index, _shape in INPUTS}
  expected_input_sizes = {
      name: int(math.prod(shape) * 2)
      for name, _label, _index, shape in INPUTS}

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("capture_gate_admits_real_layer0_component_inputs",
            capture_metrics.get("required_checks_passed") is True and
            capture_metrics.get("component_execution_admitted") is True and
            capture_metrics.get("graph_integration_admitted") is False,
            capture_git=capture_metrics.get("git"),
            capture_verdict=capture_metrics.get("verdict")),
      check("captured_conv_output_is_exact_gdn_input",
            sha256(activated) == sha256(gdn_input),
            linear_conv_output_sha256=sha256(activated),
            gdn_input_sha256=sha256(gdn_input)),
      check("all_capture_input_sizes_are_exact",
            input_sizes == expected_input_sizes,
            observed=input_sizes, expected=expected_input_sizes),
      check("bound_stock_opencl_plugin_is_exact",
            sha256(PLUGIN) == PLUGIN_SHA256,
            path=str(PLUGIN), sha256=sha256(PLUGIN)),
      check("single_isolated_component_worker_completes",
            worker["returncode"] == 0, worker=worker),
      check("fused_attention_and_both_states_are_bit_exact",
            all_exact, trials=[row.get("comparisons") for row in trials]),
      check("exactly_one_custom_dispatch_executes_per_sample",
            profile_exact,
            counts=[row.get("custom_profile_counts") for row in trials]),
      check("component_timing_produces_decisive_bound",
            timing_decisive,
            best_kernel_us=best_kernel_us,
            per_layer_target_us=PER_LAYER_TARGET_US,
            all_layer_ms=all_layer_ms,
            adjacent_target_ms=ADJACENT_TARGET_MS,
            measured_state_gbps=state_gbps,
            required_state_gbps=REQUIRED_STATE_GBPS),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  performance_passed = (
      timing_decisive and all_exact and profile_exact and
      all_layer_ms <= ADJACENT_TARGET_MS and
      state_gbps >= REQUIRED_STATE_GBPS)
  verdict = (
      "admit_provider_aware_graph_integration" if performance_passed else
      "reject_provider_aware_route_before_graph_integration")
  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  metrics = {
      "schema": SCHEMA, "created_at": created_at,
      "workstream": WORKSTREAM, "git": git,
      "host": {"target_alias": "intel-ptl-local",
               "kernel": platform.release()},
      "capture": str(CAPTURE.relative_to(ROOT)),
      "component": {
          "source": str(SOURCE.relative_to(ROOT)),
          "config": str(CONFIG.relative_to(ROOT)),
          "source_sha256": sha256(SOURCE),
          "config_sha256": sha256(CONFIG),
          "warmup": args.warmup, "repeat": args.repeat,
          "trials": args.trials,
          "trial_kernel_us_medians": medians,
          "best_kernel_us": best_kernel_us,
          "all_layer_ms": all_layer_ms,
          "adjacent_target_ms": ADJACENT_TARGET_MS,
          "state_read_write_gbps": state_gbps,
      },
      "worker": worker, "worker_result": result,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "component_correctness_passed": all_exact,
      "component_performance_passed": performance_passed,
      "graph_integration_admitted": performance_passed,
      "long_worker_admitted": False,
      "verdict": verdict,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  write_json(out / "metrics.json", metrics)
  write_json(out / "manifest.json", {
      "schema": f"{SCHEMA}-manifest-v0", "created_at": created_at,
      "artifact": str(out.relative_to(ROOT)), "git": git,
      "plugin": str(PLUGIN), "plugin_sha256": sha256(PLUGIN),
      "source_sha256": sha256(SOURCE), "config_sha256": sha256(CONFIG),
      "capture": str(CAPTURE.relative_to(ROOT)),
      "memory_stop_gib": args.memory_stop_gib,
  })
  summary = f"""# OpenVINO provider-aware linear decode component

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The fused layer-0 component preserves captured attention, conv state, and GDN
state bits: `{str(all_exact).lower()}`. The optimistic faster trial median is
`{best_kernel_us:.3f} us/layer`, or `{all_layer_ms:.6f} ms/token` for all
thirty layers, against the `{ADJACENT_TARGET_MS:.6f} ms/token` bound. Its
state-only rate is `{state_gbps:.3f} GB/s` versus `{REQUIRED_STATE_GBPS:.3f}`
required. No graph integration or long worker executes in this gate.
"""
  (out / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": str(out.relative_to(ROOT)), "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "exact": all_exact, "best_kernel_us": best_kernel_us,
      "all_layer_ms": all_layer_ms,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
