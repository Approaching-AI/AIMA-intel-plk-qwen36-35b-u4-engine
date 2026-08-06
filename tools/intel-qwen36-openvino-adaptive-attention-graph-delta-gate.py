#!/usr/bin/env python3
"""Measure matched 32k-to-64k device deltas for the adaptive graph boundary.

The worker compiles the isolated top-512 and top-256 graph nodes once each,
then runs twenty paired device-profile samples per top-k in ABBA order.  Every
sample uses a fresh InferRequest and request-owned remote buffers; host input
copies and output reads are outside the measured request.  OpenVINO exposes an
integer cumulative event average across those requests, so the worker records
and finite-differences it with an explicit at-most-42-us rounding bound.  The
opt-in plugin trace also records each request's four exact execution intervals;
their sum drives the cap test and independently checks the reconstructed
aggregate.  This is a component inference gate, not an end-to-end speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-adaptive-attention-graph-delta-gate-v2"
ROUTE = "openvino_attention_adaptive_graph_matched_delta"
PASS_ROUTE = "openvino_attention_adaptive_all10_graph_compile_boundary"
FAIL_ROUTE = "openvino_attention_adaptive_graph_stage_attribution"

BASE_TOOL = ROOT / "tools/intel-qwen36-openvino-adaptive-attention-layer3-gate.py"
CAPTURED_GATE = ROOT / (
    "output/openvino-adaptive-attention-captured-layer3-gate-"
    "20260721Tseq1728-block2d-all512-clean/gate.json")
COMPILE_GATE = ROOT / (
    "output/openvino-adaptive-attention-compile-gate-"
    "20260721Tseq1727-block2d-all512-clean/gate.json")
ALL512_BOUND = ROOT / (
    "output/openvino-adaptive-attention-all512-bound-"
    "20260721Tseq1724-clean/bound.json")
SEQ1673 = ROOT / (
    "output/openvino-adaptive-attention-component-"
    "20260720Tseq1673-clean/result.json")

BASE_HISTORY_TOKENS = 32768
TARGET_HISTORY_TOKENS = 65536
PAIR_BLOCKS = 10
MIN_SAMPLES = 20
PREFLIGHT_BYTES = 8 * 1024 ** 3
CAP_MS = {512: 0.7923170134317261, 256: 0.7247750157917522}
ACTIVE_TOPK = 512
ACTIVE_LAYER_COUNT = 10
WEIGHTED_CAP_MS = 7.38283415319747
STAGE_ENTRIES = (
    "iq36_adaptive_attention_partial",
    "iq36_adaptive_attention_select_reduce_union",
    "iq36_adaptive_attention_correct_normalize",
    "iq36_adaptive_attention_ordered_update",
)


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module("iq36_adaptive_graph_delta_base", BASE_TOOL)
PERF = load_module("iq36_adaptive_graph_delta_perf", ROOT / "tools/iq36_perf_inference.py")
OV_PYTHON = BASE.OV_PYTHON
DEFAULT_PLUGIN = BASE.DEFAULT_PLUGIN


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  parser.add_argument("--candidate-plugin", type=Path, default=DEFAULT_PLUGIN)
  parser.add_argument(
      "--value-quant-group", type=int, choices=(16, 32), default=32)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--allow-dirty", action="store_true")
  args = parser.parse_args()
  if args.worker_config is None and args.out_dir is None:
    parser.error("--out-dir is required outside worker mode")
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def configure_geometry(history_tokens: int, topk: int) -> None:
  if history_tokens not in {BASE_HISTORY_TOKENS, TARGET_HISTORY_TOKENS}:
    raise ValueError(f"unsupported history geometry: {history_tokens}")
  BASE.TOPK = topk
  BASE.HISTORY_TOKENS = history_tokens
  BASE.KEY_TOKENS = history_tokens + 1
  BASE.HOT_WINDOW = 16384
  BASE.COLD_TOKENS = history_tokens - BASE.HOT_WINDOW
  BASE.COLD_CHUNKS = (
      BASE.COLD_TOKENS + BASE.CHUNK_TOKENS - 1) // BASE.CHUNK_TOKENS


def make_fixtures(np: Any) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
  configure_geometry(BASE_HISTORY_TOKENS, 512)
  base = BASE.make_fixture(np)
  configure_geometry(TARGET_HISTORY_TOKENS, 512)
  target = BASE.make_fixture(np)
  return {"base": base, "target": target}


def profile_average_us(request: Any) -> float:
  rows = BASE.profile_rows(request)
  if len(rows) != 1:
    raise RuntimeError(f"expected one custom profile row, observed {len(rows)}")
  value = float(rows[0]["real_time_us"])
  if not math.isfinite(value) or value <= 0.0:
    raise RuntimeError(f"invalid custom profile duration: {value}")
  return value


def read_stage_trace(
    path: Path, expected_count: int, *, timeout_s: float = 5.0,
) -> dict[str, Any]:
  """Read the just-completed request's four device execution intervals."""
  deadline = time.monotonic() + timeout_s
  last_error = "trace file is absent"
  while time.monotonic() < deadline:
    if path.is_file():
      text = path.read_text(encoding="utf-8")
      lines = text.splitlines()
      if text and not text.endswith("\n"):
        lines = lines[:-1]
      if len(lines) > expected_count:
        raise RuntimeError(
            f"stage trace advanced past request {expected_count}: "
            f"{len(lines)} rows")
      if len(lines) == expected_count:
        try:
          row = json.loads(lines[-1])
          stages = row.get("stages", [])
          entries = tuple(stage.get("entry") for stage in stages)
          executing_ns = [int(stage.get("executing_ns", -1))
                          for stage in stages]
          if entries != STAGE_ENTRIES:
            raise RuntimeError(
                f"unexpected stage order: {entries!r}")
          if any(value <= 0 for value in executing_ns):
            raise RuntimeError(
                f"non-positive stage duration: {executing_ns!r}")
          stage_us = {
              entry: value / 1000.0
              for entry, value in zip(entries, executing_ns)
          }
          return {
              "stage_executing_ns": dict(zip(entries, executing_ns)),
              "stage_sum_us": sum(executing_ns) / 1000.0,
              "stage_us": stage_us,
          }
        except (
            json.JSONDecodeError, RuntimeError, TypeError, ValueError,
        ) as error:
          last_error = f"{type(error).__name__}: {error}"
    time.sleep(0.001)
  raise RuntimeError(
      f"stage trace request {expected_count} did not complete: {last_error}")


def signed_median_summary(values: list[float], *, seed: int) -> dict[str, Any]:
  """Summarize signed stage deltas without treating them as promotion gates."""
  samples = [float(value) for value in values]
  if len(samples) != MIN_SAMPLES or any(
      not math.isfinite(value) for value in samples):
    raise ValueError("stage attribution requires twenty finite samples")
  rng = random.Random(seed)
  medians = sorted(
      statistics.median(rng.choices(samples, k=len(samples)))
      for _ in range(PERF.DEFAULT_BOOTSTRAP_RESAMPLES))
  rank = max(1, math.ceil(PERF.DEFAULT_CONFIDENCE * len(medians)))
  return {
      "bootstrap_resamples": PERF.DEFAULT_BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": seed,
      "confidence": PERF.DEFAULT_CONFIDENCE,
      "difference_samples_ms": samples,
      "point_estimate_ms": statistics.median(samples),
      "promotion_gate": False,
      "sample_count": len(samples),
      "upper_confidence_bound_ms": medians[min(rank - 1, len(medians) - 1)],
  }


def execute_once(
    compiled: Any, context: Any, feed: dict[str, Any], reference: dict[str, Any],
    geometry: str, topk: int, ov: Any, np: Any, *, validate: bool,
    stage_trace_path: Path, expected_trace_count: int,
) -> dict[str, Any]:
  history_tokens = (
      BASE_HISTORY_TOKENS if geometry == "base" else TARGET_HISTORY_TOKENS)
  configure_geometry(history_tokens, topk)
  request = compiled.create_infer_request()
  remote_outputs = []
  output_shapes = []
  owned = []
  try:
    for port in compiled.inputs:
      name = port.get_any_name()
      shape = (
          port.shape if port.partial_shape.is_static else
          ov.Shape(list(feed[name].shape)))
      remote = context.create_tensor(port.element_type, shape, {})
      remote.copy_from(ov.Tensor(feed[name]))
      request.set_tensor(name, remote)
      owned.append(remote)
    for port in compiled.outputs[:6]:
      shape = port.shape
      remote = context.create_tensor(port.element_type, shape, {})
      request.set_tensor(port.get_any_name(), remote)
      remote_outputs.append(remote)
      output_shapes.append(shape)

    started = time.perf_counter_ns()
    request.start_async()
    request.wait()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    average_us = profile_average_us(request)
    stage_trace = read_stage_trace(stage_trace_path, expected_trace_count)
    result = {
        "profile_average_us": average_us,
        **stage_trace,
        "wall_ms": wall_ms,
    }
    if validate:
      workspace_array = BASE.copy_remote(
          remote_outputs[0], compiled.output(0).element_type,
          output_shapes[0], context, np)
      attention = BASE.copy_remote(
          remote_outputs[1], compiled.output(1).element_type,
          output_shapes[1], context, np)
      workspace = BASE.analyze_workspace(workspace_array, np)
      independent = BASE.adaptive_reference(feed, reference, workspace, np)
      metrics = BASE.vector_metrics(
          independent, attention.astype(np.float32), np)
      result["validation"] = {
          "attention_metrics": metrics,
          "completion_exact": bool(np.array_equal(
              workspace["completion"],
              np.full((BASE.KV_HEADS,), BASE.COLD_CHUNKS, dtype=np.uint32))),
          "local_candidate_scores_exact": (
              workspace["local_candidate_score_identity_pass"]),
          "local_candidate_shape_exact": (
              workspace["local_candidate_shape_pass"]),
          "union_counts": workspace["union_counts"],
          "union_exact": workspace["union_exact"],
          "workspace_output_exact": bool(np.array_equal(
              workspace["attention"].astype(np.float16), attention)),
      }
    return result
  finally:
    del remote_outputs
    del owned
    del request
    gc.collect()


def run_topk(
    topk: int, core: Any, context: Any,
    fixtures: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    raw: Path, ov: Any, np: Any,
) -> dict[str, Any]:
  trace_path = raw / f"top{topk}-stages.jsonl"
  trace_path.unlink(missing_ok=True)
  os.environ["IQ36_ADAPTIVE_STAGE_PROFILE_PATH"] = str(trace_path.resolve())
  configure_geometry(TARGET_HISTORY_TOKENS, topk)
  model = BASE.make_model(ov)
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(model, context, {
      "ACTIVATIONS_SCALE_FACTOR": 0.0,
      "PERFORMANCE_HINT": "LATENCY",
      "PERF_COUNT": True,
  })
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0

  profile_count = 0
  prior_average_us = 0.0
  trace_count = 0

  def recover_device_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Invert OpenVINO GPU's integer cumulative profile average.

    The plugin shares one PerfCounter across InferRequests created from a
    compiled model.  get_profiling_info() therefore returns floor(S_n / n),
    not the latest event.  The finite difference below recovers the event to
    within n microseconds; the bound is retained with every sample.
    """
    nonlocal profile_count, prior_average_us
    profile_count += 1
    average_us = float(row["profile_average_us"])
    device_us = (
        average_us if profile_count == 1 else
        profile_count * average_us -
        (profile_count - 1) * prior_average_us)
    prior_average_us = average_us
    row["device_us"] = device_us
    row["profile_stage_abs_error_us"] = abs(
        device_us - float(row["stage_sum_us"]))
    row["profile_count"] = profile_count
    row["profile_rounding_error_bound_us"] = float(profile_count)
    return row

  def execute_traced(
      feed: dict[str, Any], reference: dict[str, Any], geometry: str, *,
      validate: bool,
  ) -> dict[str, Any]:
    nonlocal trace_count
    trace_count += 1
    return execute_once(
        compiled, context, feed, reference, geometry, topk, ov, np,
        validate=validate, stage_trace_path=trace_path,
        expected_trace_count=trace_count)

  validation = {}
  for geometry in ("base", "target"):
    feed, reference = fixtures[geometry]
    validation[geometry] = recover_device_sample(execute_traced(
        feed, reference, geometry, validate=True))

  rows = []
  profile_differences_ms = []
  stage_sum_differences_ms = []
  stage_differences_ms: dict[str, list[float]] = {
      entry: [] for entry in STAGE_ENTRIES}
  for block in range(PAIR_BLOCKS):
    block_profile_values: dict[str, list[float]] = {
        "base": [], "target": []}
    block_stage_sum_values: dict[str, list[float]] = {
        "base": [], "target": []}
    block_stage_values = {
        entry: {"base": [], "target": []} for entry in STAGE_ENTRIES}
    for order_index, geometry in enumerate(("base", "target", "target", "base")):
      feed, reference = fixtures[geometry]
      observed = recover_device_sample(execute_traced(
          feed, reference, geometry, validate=False))
      row = {
          "block": block,
          "device_ms": float(observed["device_us"]) / 1000.0,
          "geometry": geometry,
          "history_tokens": (
              BASE_HISTORY_TOKENS if geometry == "base" else
              TARGET_HISTORY_TOKENS),
          "order_index": order_index,
          "profile_average_us": observed["profile_average_us"],
          "profile_count": observed["profile_count"],
          "profile_rounding_error_bound_us": observed[
              "profile_rounding_error_bound_us"],
          "profile_stage_abs_error_us": observed[
              "profile_stage_abs_error_us"],
          "stage_executing_ns": observed["stage_executing_ns"],
          "stage_sum_ms": float(observed["stage_sum_us"]) / 1000.0,
          "stage_us": observed["stage_us"],
          "topk": topk,
          "wall_ms": observed["wall_ms"],
      }
      rows.append(row)
      block_profile_values[geometry].append(row["device_ms"])
      block_stage_sum_values[geometry].append(row["stage_sum_ms"])
      for entry in STAGE_ENTRIES:
        block_stage_values[entry][geometry].append(
            float(row["stage_us"][entry]) / 1000.0)
    profile_differences_ms.extend([
        block_profile_values["target"][0] - block_profile_values["base"][0],
        block_profile_values["target"][1] - block_profile_values["base"][1],
    ])
    stage_sum_differences_ms.extend([
        block_stage_sum_values["target"][0] -
            block_stage_sum_values["base"][0],
        block_stage_sum_values["target"][1] -
            block_stage_sum_values["base"][1],
    ])
    for entry in STAGE_ENTRIES:
      values = block_stage_values[entry]
      stage_differences_ms[entry].extend([
          values["target"][0] - values["base"][0],
          values["target"][1] - values["base"][1],
      ])

  with (raw / f"top{topk}-samples.jsonl").open(
      "w", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row, sort_keys=True) + "\n")
  inference = PERF.latency_cap_inference(
      stage_sum_differences_ms, cap=CAP_MS[topk], min_samples=MIN_SAMPLES,
      seed=PERF.DEFAULT_BOOTSTRAP_SEED + topk)
  stage_attribution = {}
  for stage_index, entry in enumerate(STAGE_ENTRIES):
    summary = signed_median_summary(
        stage_differences_ms[entry],
        seed=PERF.DEFAULT_BOOTSTRAP_SEED + topk * 10 + stage_index)
    summary.update({
        "median_base_ms": statistics.median(
            float(row["stage_us"][entry]) / 1000.0
            for row in rows if row["geometry"] == "base"),
        "median_target_ms": statistics.median(
            float(row["stage_us"][entry]) / 1000.0
            for row in rows if row["geometry"] == "target"),
    })
    stage_attribution[entry] = summary
  validation_pass = all(
      row.get("validation", {}).get("union_exact") is True and
      row.get("validation", {}).get("completion_exact") is True and
      row.get("validation", {}).get("local_candidate_scores_exact") is True and
      row.get("validation", {}).get("local_candidate_shape_exact") is True and
      row.get("validation", {}).get("workspace_output_exact") is True and
      row.get("validation", {}).get(
          "attention_metrics", {}).get("finite") is True and
      float(row["validation"]["attention_metrics"]["relative_l2"]) <= 0.002 and
      float(row["validation"]["attention_metrics"]["max_abs"]) <= 0.002
      for row in validation.values())
  result = {
      "compile_ms": compile_ms,
      "difference_samples_ms": stage_sum_differences_ms,
      "inference": inference,
      "median_base_device_ms": statistics.median(
          row["stage_sum_ms"] for row in rows if row["geometry"] == "base"),
      "median_target_device_ms": statistics.median(
          row["stage_sum_ms"] for row in rows if row["geometry"] == "target"),
      "median_base_profile_reconstructed_ms": statistics.median(
          row["device_ms"] for row in rows if row["geometry"] == "base"),
      "median_target_profile_reconstructed_ms": statistics.median(
          row["device_ms"] for row in rows if row["geometry"] == "target"),
      "profile_reconstructed_difference_samples_ms": profile_differences_ms,
      "profile_reconstruction_max_error_bound_us": max(
          row["profile_rounding_error_bound_us"] for row in rows),
      "profile_stage_max_abs_error_us": max(
          row["profile_stage_abs_error_us"] for row in rows),
      "sample_rows": rows,
      "stage_attribution": stage_attribution,
      "stage_trace_count": trace_count,
      "stage_trace_path": BASE.relative(trace_path),
      "validation": validation,
      "validation_pass": validation_pass,
  }
  os.environ.pop("IQ36_ADAPTIVE_STAGE_PROFILE_PATH", None)
  del compiled
  del model
  gc.collect()
  return result


def worker_main(config_path: Path) -> int:
  started = time.perf_counter_ns()
  result_path: Path | None = None
  try:
    if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
      raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")
    import numpy as np
    import openvino as ov

    config = BASE.load_json(config_path)
    value_quant_group = int(config.get("value_quant_group", 32))
    BASE.configure_quantization(
        32, value_quant_group, False, False, packed_kv_variant=None)
    topks = (512,) if value_quant_group == 16 else (512, 256)
    raw = Path(config["raw"])
    plugin = Path(config["plugin"])
    result_path = raw / "worker-result.json"
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property("GPU", {"CONFIG_FILE": str(BASE.CUSTOM_XML.resolve())})
    context = core.get_default_context("GPU")
    fixtures = make_fixtures(np)
    results = {}
    for topk in topks:
      results[str(topk)] = run_topk(
          topk, core, context, fixtures, raw, ov, np)
    weighted_ucb = ACTIVE_LAYER_COUNT * float(
        results[str(ACTIVE_TOPK)]["inference"]["upper_confidence_bound_ms"])
    weighted_stage_point_ms = {
        entry: ACTIVE_LAYER_COUNT * float(
            results[str(ACTIVE_TOPK)]["stage_attribution"][entry][
                "point_estimate_ms"])
        for entry in STAGE_ENTRIES}
    dominant_stage = max(
        STAGE_ENTRIES, key=lambda entry: weighted_stage_point_ms[entry])
    performance_pass = (
        results[str(ACTIVE_TOPK)]["inference"]["rate_pass"] is True and
        weighted_ucb <= WEIGHTED_CAP_MS)
    required_checks = {
        "both_geometries_match_independent_reference": all(
            results[str(topk)]["validation_pass"] for topk in topks),
        "both_topk_have_twenty_paired_samples": all(
            len(results[str(topk)]["difference_samples_ms"]) == MIN_SAMPLES
            for topk in topks),
        "device_samples_are_positive_and_target_deltas_are_finite": all(
            all(math.isfinite(value) and value > 0.0
                for value in result["difference_samples_ms"])
            for result in results.values()),
        "profile_reconstruction_rounding_bound_is_below_50us": all(
            float(results[str(topk)][
                "profile_reconstruction_max_error_bound_us"]) < 50.0
            for topk in topks),
        "all_forty_two_stage_trace_rows_are_complete": all(
            int(results[str(topk)]["stage_trace_count"]) ==
                2 + 4 * PAIR_BLOCKS
            for topk in topks),
        "stage_sums_match_aggregate_profile_rounding_bound": all(
            float(results[str(topk)]["profile_stage_max_abs_error_us"]) <=
                float(results[str(topk)][
                    "profile_reconstruction_max_error_bound_us"]) + 2.0
            for topk in topks),
    }
    payload = {
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "openvino_version": ov.get_version(),
        "performance_cap_pass": performance_pass,
        "dominant_weighted_stage": dominant_stage,
        "plugin": str(plugin.resolve()),
        "plugin_sha256": BASE.sha256(plugin),
        "quantization": {"key_group": 32, "value_group": value_quant_group},
        "required_checks": required_checks,
        "required_checks_passed": all(required_checks.values()),
        "results": results,
        "active_topk_by_layer": {
            str(layer): ACTIVE_TOPK for layer in range(3, 40, 4)},
        "weighted_stage_point_ms": weighted_stage_point_ms,
        "weighted_ucb_ms": weighted_ucb,
        "weighted_cap_ms": WEIGHTED_CAP_MS,
    }
    BASE.write_json(result_path, payload)
    print(json.dumps({
        "performance_cap_pass": performance_pass,
        "required_checks_passed": payload["required_checks_passed"],
        "top256_ucb_ms": results.get("256", {}).get(
            "inference", {}).get("upper_confidence_bound_ms"),
        "top512_ucb_ms": results["512"]["inference"][
            "upper_confidence_bound_ms"],
        "weighted_ucb_ms": weighted_ucb,
    }, sort_keys=True), flush=True)
    return 0 if payload["required_checks_passed"] else 2
  except Exception as error:
    payload = {
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "error": f"{type(error).__name__}: {error}",
        "required_checks_passed": False,
        "traceback": traceback.format_exc(),
    }
    if result_path is not None:
      BASE.write_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 2


def summary_markdown(payload: dict[str, Any]) -> str:
  worker = payload.get("worker", {})
  results = worker.get("results", {})
  lines = [
      "# Adaptive-attention graph matched-delta gate",
      "",
      f"- Verdict: `{payload['verdict']}`",
      f"- Measurement checks passed: "
      f"`{str(payload['required_checks_passed']).lower()}`",
      f"- Performance cap passed: "
      f"`{str(worker.get('performance_cap_pass')).lower()}`",
      "",
      "| top-k | 32k median ms | 64k median ms | delta median ms | delta UCB ms | cap ms |",
      "|---:|---:|---:|---:|---:|---:|",
  ]
  for topk in sorted((int(key) for key in results), reverse=True):
    result = results.get(str(topk), {})
    inference = result.get("inference", {})
    lines.append(
        f"| {topk} | {result.get('median_base_device_ms')} | "
        f"{result.get('median_target_device_ms')} | "
        f"{inference.get('point_estimate_ms')} | "
        f"{inference.get('upper_confidence_bound_ms')} | "
        f"{inference.get('cap_ms')} |")
  lines.extend([
      "",
      f"- Dominant all-512 weighted growth stage: "
      f"`{worker.get('dominant_weighted_stage')}`",
      "",
      "| stage | top-512 delta ms | top-256 diagnostic ms | all-512 10-layer ms |",
      "|---|---:|---:|---:|",
  ])
  weighted = worker.get("weighted_stage_point_ms", {})
  for entry in STAGE_ENTRIES:
    short = entry.removeprefix("iq36_adaptive_attention_")
    top512 = results.get("512", {}).get(
        "stage_attribution", {}).get(entry, {}).get("point_estimate_ms")
    top256 = results.get("256", {}).get(
        "stage_attribution", {}).get(entry, {}).get("point_estimate_ms")
    lines.append(
        f"| {short} | {top512} | {top256} | {weighted.get(entry)} |")
  lines.extend([
      "",
      "All timings are sums of the four traced device execution intervals.",
      "The aggregate OpenVINO profile is retained as an independent rounding",
      "cross-check. Host copies, stage-trace file writes,",
      "request construction, model execution, and token generation are out of",
      "scope; this artifact cannot support an end-to-end speed claim.",
      "",
  ])
  return "\n".join(lines)


def orchestrator_main(args: argparse.Namespace) -> int:
  assert args.out_dir is not None
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  plugin = args.candidate_plugin.resolve()
  stop_bytes = int(args.memory_stop_gib * 1024 ** 3)
  memory = [BASE.memory_sample("start")]
  if int(memory[0]["available_bytes"]) < PREFLIGHT_BYTES:
    raise RuntimeError(
        f"8-GiB preflight failed: {memory[0]['available_bytes']} bytes")
  git = BASE.git_state(out)
  captured_gate = BASE.load_json(CAPTURED_GATE)
  compile_gate = BASE.load_json(COMPILE_GATE)
  all512_bound = BASE.load_json(ALL512_BOUND)
  seq1673 = BASE.load_json(SEQ1673)
  expected_plugin_sha = str(compile_gate.get("plugin_sha256", ""))
  plugin_sha = BASE.sha256(plugin) if plugin.is_file() else ""

  config = raw / "worker-config.json"
  BASE.write_json(config, {
      "plugin": str(plugin),
      "raw": str(raw),
      "value_quant_group": args.value_quant_group,
  })
  time_path = raw / "worker.time.txt"
  command = [
      "/usr/bin/time", "-v", "-o", str(time_path),
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config),
  ]
  environment = os.environ.copy()
  environment["NEO_CACHE_DIR"] = str((raw / "compiler-cache").resolve())
  (raw / "compiler-cache").mkdir()
  worker_run = BASE.run_monitored_worker(
      command, environment, args.timeout_s, stop_bytes, memory,
      raw / "worker.stdout", raw / "worker.stderr")
  memory.append(BASE.memory_sample("finish"))
  worker = BASE.load_json(raw / "worker-result.json") \
      if (raw / "worker-result.json").is_file() else {}
  resources = BASE.parse_time(time_path)
  BASE.write_json(raw / "worker-command.json", {
      **worker_run,
      "environment": {"NEO_CACHE_DIR": environment["NEO_CACHE_DIR"]},
      "resources": resources,
  })
  source_caps = {
      topk: float(seq1673.get("performance_inference", {}).get(
          str(topk), {}).get("cap_ms", math.inf))
      for topk in (512, 256)}
  memory_pass = (
      not worker_run["guard_tripped"] and
      all(int(row["available_bytes"]) >= stop_bytes for row in memory))
  checks = [
      BASE.check("repository_clean_at_gate",
                 not git["dirty"] or args.allow_dirty,
                 git=git, allow_dirty=args.allow_dirty),
      BASE.check("seq1727_plugin_is_exact", plugin.is_file() and
                 plugin_sha == expected_plugin_sha,
                 plugin=BASE.relative(plugin), sha256=plugin_sha),
      BASE.check("seq1728_captured_boundary_is_admitted",
                 captured_gate.get("required_checks_passed") is True and
                 captured_gate.get("verdict") ==
                     "admit_adaptive_attention_captured_layer3_boundary"),
      BASE.check("seq1724_all512_numeric_and_traffic_bound_is_admitted",
                 all512_bound.get("all_required_checks_pass") is True and
                 all(all512_bound.get("checks", {}).values()) and
                 tuple(all512_bound.get("rule", {}).get(
                     "high_topk_layers", ())) == tuple(range(3, 40, 4)) and
                 int(all512_bound.get("rule", {}).get(
                     "topk_per_query", 0)) == ACTIVE_TOPK),
      BASE.check("seq1673_caps_are_exact",
                 all(abs(source_caps[topk] - CAP_MS[topk]) < 1e-12
                     for topk in (512, 256)),
                 observed=source_caps, expected=CAP_MS),
      BASE.check("single_serial_worker_completes",
                 worker_run["returncode"] == 0),
      BASE.check("matched_graph_delta_measurement_is_complete",
                 worker.get("required_checks_passed") is True,
                 worker_checks=worker.get("required_checks", {})),
      BASE.check("worker_peak_rss_is_bounded",
                 0 < int(resources.get("maximum_resident_kib", 0)) <
                     8 * 1024 * 1024,
                 maximum_resident_kib=resources.get("maximum_resident_kib")),
      BASE.check("worker_does_not_swap",
                 int(resources.get("swaps", -1)) == 0,
                 swaps=resources.get("swaps")),
      BASE.check("memory_guard_never_tripped", memory_pass,
                 stop_bytes=stop_bytes,
                 minimum_available_bytes=min(
                     int(row["available_bytes"]) for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  performance_pass = worker.get("performance_cap_pass") is True
  verdict = (
      "admit_adaptive_attention_all10_graph_compile_boundary"
      if required and performance_pass else
      "profile_adaptive_attention_graph_stage_gap"
      if required else "repair_adaptive_attention_graph_delta_measurement")
  payload = {
      "all10_compile_worker_admitted": bool(required and performance_pass),
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "long_worker_admitted": False,
      "memory": memory,
      "model_worker_admitted": False,
      "next_route": (
          PASS_ROUTE if required and performance_pass else
          FAIL_ROUTE if required else ROUTE),
      "product_worker_admitted": False,
      "quantization": {"key_group": 32, "value_group": args.value_quant_group},
      "required_checks_passed": required,
      "route": ROUTE,
      "schema_version": SCHEMA,
      "sources": {
          "captured_gate": BASE.relative(CAPTURED_GATE),
          "compile_gate": BASE.relative(COMPILE_GATE),
          "all512_bound": BASE.relative(ALL512_BOUND),
          "seq1673": BASE.relative(SEQ1673),
      },
      "verdict": verdict,
      "worker": worker,
      "worker_resources": resources,
      "worker_run": worker_run,
      "workstream": WS,
  }
  BASE.write_json(out / "gate.json", payload)
  (out / "summary.md").write_text(
      summary_markdown(payload), encoding="utf-8")
  print(json.dumps({
      "all10_compile_worker_admitted": payload[
          "all10_compile_worker_admitted"],
      "output": BASE.relative(out),
      "required_checks_passed": required,
      "top256_ucb_ms": worker.get("results", {}).get(
          "256", {}).get("inference", {}).get("upper_confidence_bound_ms"),
      "top512_ucb_ms": worker.get("results", {}).get(
          "512", {}).get("inference", {}).get("upper_confidence_bound_ms"),
      "verdict": verdict,
      "weighted_ucb_ms": worker.get("weighted_ucb_ms"),
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  parsed = parse_args()
  if parsed.worker_config is not None:
    raise SystemExit(worker_main(parsed.worker_config.resolve()))
  raise SystemExit(orchestrator_main(parsed))
