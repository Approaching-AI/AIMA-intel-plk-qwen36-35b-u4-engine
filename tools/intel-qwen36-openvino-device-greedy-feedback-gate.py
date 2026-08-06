#!/usr/bin/env python3
"""Bound device greedy feedback against full-logit host argmax.

The control and candidate share one SimpleGPU identity producer over the
locked 248320-row F32 logits shape.  The control returns the complete tensor
and runs NumPy argmax, matching the product worker's timing boundary.  The
candidate performs a two-pass GPU reduction and returns one I32 token.  Twenty
interleaved ABBA blocks decide whether the paired saving clears the complete
64k decode admission number before any full-model integration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-device-greedy-feedback-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2055/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONFIG = ROOT / "engine/openvino/custom/iq36_greedy_top1.xml"
SOURCES = (
    ROOT / "engine/openvino/custom/iq36_identity.cl",
    ROOT / "engine/openvino/custom/iq36_greedy_top1.cl",
    ROOT / "engine/openvino/custom/iq36_greedy_top1_merge.cl",
)
DEFAULT_LOGITS = ROOT / (
    "output/openvino-prefill-q32-splitn-20260723Tseq2093-all10-64k-"
    "o512-abba1/raw/sentinel_064k/correctness/candidate/"
    "step0000-logits.f32")
VOCABULARY = 248320
PARTIAL_COUNT = 64
MIN_SAMPLES = 20
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0x5136
REQUIRED_SAVING_MS = 0.16560487656


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--samples", type=int, default=MIN_SAMPLES)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.out_dir is None:
    parser.error("--out-dir is required")
  if (args.warmup < 1 or args.samples != MIN_SAMPLES or
      args.min_available_gib <= args.abort_below_available_gib or
      args.abort_below_available_gib <= 0.0 or args.timeout_s <= 0):
    parser.error("invalid warmup/sample/memory/timeout configuration")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is unavailable")


def process_memory_bytes() -> dict[str, int]:
  values = {"rss_bytes": 0, "swap_bytes": 0}
  for line in Path("/proc/self/status").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("VmRSS:"):
      values["rss_bytes"] = int(line.split()[1]) * 1024
    elif line.startswith("VmSwap:"):
      values["swap_bytes"] = int(line.split()[1]) * 1024
  return values


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.splitlines()
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def lower_bootstrap_median(values: list[float]) -> float:
  if len(values) < 1 or any(not math.isfinite(value) for value in values):
    raise ValueError("paired savings must be finite and non-empty")
  rng = random.Random(BOOTSTRAP_SEED)
  medians = sorted(
      statistics.median(rng.choices(values, k=len(values)))
      for _ in range(BOOTSTRAP_RESAMPLES))
  rank = max(1, math.ceil(0.05 * len(medians)))
  return float(medians[rank - 1])


def dispersion(values: list[float]) -> dict[str, Any]:
  median = statistics.median(values)
  mad = statistics.median(abs(value - median) for value in values)
  robust_cv = math.inf if median == 0.0 else 1.4826 * mad / abs(median)
  if robust_cv <= 0.01:
    classification = "normal"
  elif robust_cv <= 0.02:
    classification = "warning"
  else:
    classification = "investigate"
  return {
      "classification": classification,
      "metric": "1.4826_mad_over_abs_median",
      "promotion_gate": False,
      "robust_cv": robust_cv,
  }


def custom_classes(ov: Any) -> tuple[type, type, type]:
  class IQ36Identity(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36Identity(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  class IQ36GreedyTop1Partials(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, ov.Type.i64, ov.PartialShape([1, 1, 1, PARTIAL_COUNT]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GreedyTop1Partials(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  class IQ36GreedyTop1Merge(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(0, ov.Type.i32, ov.PartialShape([1, 1, 1, 1]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GreedyTop1Merge(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36Identity, IQ36GreedyTop1Partials, IQ36GreedyTop1Merge


def make_model(ov: Any, device_top1: bool) -> Any:
  identity_class, partials_class, merge_class = custom_classes(ov)
  parameter = ov.opset13.parameter(
      [1, 1, 1, VOCABULARY], ov.Type.f32, name="logits")
  identity = identity_class([parameter.output(0)])
  identity.set_friendly_name("iq36_greedy_logit_producer")
  output = identity.output(0)
  if device_top1:
    partials = partials_class([output])
    partials.set_friendly_name("iq36_greedy_top1_partials")
    merge = merge_class([partials.output(0)])
    merge.set_friendly_name("iq36_greedy_top1_merge")
    output = merge.output(0)
  return ov.Model(
      [output], [parameter],
      "iq36_device_greedy" if device_top1 else "iq36_host_greedy_control")


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def worker_main(config_path: Path) -> int:
  import numpy as np
  import openvino as ov

  cfg = load_json(config_path)
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(
        f"worker requires {OV_PYTHON}, observed {sys.executable}")
  raw = Path(cfg["raw"])
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(PLUGIN.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  logits_path = Path(cfg["logits"])
  source = np.fromfile(logits_path, dtype="<f4")
  if source.size != VOCABULARY or not bool(np.isfinite(source).all()):
    raise RuntimeError(
        f"invalid logits row: shape={source.shape}, finite="
        f"{bool(np.isfinite(source).all())}")
  source = np.ascontiguousarray(source.reshape(1, 1, 1, VOCABULARY))
  expected_token = int(np.argmax(source.reshape(-1)))

  core = ov.Core(str(registry))
  config_before = str(core.get_property("GPU", "CONFIG_FILE"))
  core.set_property("GPU", {"CONFIG_FILE": str(CONFIG.resolve())})
  config_after = str(core.get_property("GPU", "CONFIG_FILE"))
  # The product LM head publishes F32 logits.  The generic GPU hint otherwise
  # narrows this isolated custom identity to F16 and undercounts the host
  # result boundary by half.
  compile_config = {
      "INFERENCE_PRECISION_HINT": ov.Type.f32,
      "PERFORMANCE_HINT": "LATENCY",
  }
  control = core.compile_model(make_model(ov, False), "GPU", compile_config)
  candidate = core.compile_model(make_model(ov, True), "GPU", compile_config)
  control_request = control.create_infer_request()
  candidate_request = candidate.create_infer_request()

  def infer(compiled: Any, request: Any, device_top1: bool) -> tuple[int, float]:
    started = time.perf_counter_ns()
    outputs = request.infer({compiled.input(0): source})
    output = np.asarray(outputs[compiled.output(0)])
    token = (
        int(output.reshape(-1)[-1]) if device_top1
        else int(np.argmax(np.asarray(output, dtype=np.float32).reshape(-1))))
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return token, elapsed_ms

  for _ in range(int(cfg["warmup"])):
    control_token, _ = infer(control, control_request, False)
    candidate_token, _ = infer(candidate, candidate_request, True)
    if control_token != expected_token or candidate_token != expected_token:
      raise RuntimeError("warmup token mismatch")

  stop_bytes = int(float(cfg["abort_below_available_gib"]) * (1024 ** 3))
  memory_rows = []
  blocks = []
  exact = True
  for block_index in range(int(cfg["samples"])):
    available = available_memory_bytes()
    memory_rows.append({
        "available_bytes": available,
        "block": block_index,
        **process_memory_bytes(),
    })
    if available < stop_bytes:
      raise RuntimeError(
          f"memory guard at block {block_index}: {available} < {stop_bytes}")
    rows = []
    for label, compiled, request, device_top1 in (
        ("control-a1", control, control_request, False),
        ("candidate-b1", candidate, candidate_request, True),
        ("candidate-b2", candidate, candidate_request, True),
        ("control-a2", control, control_request, False),
    ):
      token, wall_ms = infer(compiled, request, device_top1)
      exact = exact and token == expected_token
      rows.append({"label": label, "token": token, "wall_ms": wall_ms})
    control_ms = statistics.median(
        row["wall_ms"] for row in rows if row["label"].startswith("control"))
    candidate_ms = statistics.median(
        row["wall_ms"] for row in rows
        if row["label"].startswith("candidate"))
    blocks.append({
        "block": block_index,
        "candidate_ms": candidate_ms,
        "control_ms": control_ms,
        "rows": rows,
        "saving_ms": control_ms - candidate_ms,
    })

  profile_compiled = core.compile_model(
      make_model(ov, True), "GPU",
      {**compile_config, "PERF_COUNT": True})
  profile_request = profile_compiled.create_infer_request()
  profile_token, _ = infer(profile_compiled, profile_request, True)
  profile = [{
      "exec_type": row.exec_type,
      "node_name": row.node_name,
      "node_type": row.node_type,
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
      "status": str(row.status),
  } for row in profile_request.get_profiling_info()]
  runtime = []
  for node in profile_compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    if str(info.get("layerType")) != "CustomGPUPrimitive":
      continue
    runtime.append({
        "layer_type": str(info.get("layerType")),
        "node_name": node.get_friendly_name(),
        "output_layouts": str(info.get("outputLayouts")),
        "output_precisions": str(info.get("outputPrecisions")),
        "primitive_type": str(info.get("primitiveType")),
    })

  savings = [float(row["saving_ms"]) for row in blocks]
  lower = lower_bootstrap_median(savings)
  inference = {
      "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
      "bootstrap_seed": BOOTSTRAP_SEED,
      "confidence": 0.95,
      "dispersion": dispersion(savings),
      "lower_confidence_bound_saving_ms": lower,
      "method": "paired_one_sided_percentile_bootstrap_median_difference",
      "minimum_sample_count": MIN_SAMPLES,
      "point_estimate_saving_ms": statistics.median(savings),
      "required_saving_ms": REQUIRED_SAVING_MS,
      "sample_count": len(savings),
      "sample_count_pass": len(savings) >= MIN_SAMPLES,
      "saving_pass": len(savings) >= MIN_SAMPLES and lower >= REQUIRED_SAVING_MS,
  }
  write_json(Path(cfg["result"]), {
      "blocks": blocks,
      "candidate_median_ms": statistics.median(
          float(row["candidate_ms"]) for row in blocks),
      "config_after": config_after,
      "config_before": config_before,
      "control_median_ms": statistics.median(
          float(row["control_ms"]) for row in blocks),
      "device_tokens_exact": exact and profile_token == expected_token,
      "expected_token": expected_token,
      "inference": inference,
      "input": {
          "byte_count": logits_path.stat().st_size,
          "path": str(logits_path.resolve()),
          "sha256": sha256_file(logits_path),
          "vocabulary": VOCABULARY,
      },
      "memory_rows": memory_rows,
      "openvino_version": ov.get_version(),
      "profile": profile,
      "profile_token": profile_token,
      "runtime": runtime,
  })
  return 0


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  out_dir = args.out_dir.resolve()
  if out_dir.exists():
    raise SystemExit(f"output directory exists: {out_dir}")
  raw = out_dir / "raw"
  raw.mkdir(parents=True)
  logits = args.logits.resolve()
  for path in (PLUGIN, CONFIG, *SOURCES, logits):
    if not path.is_file():
      raise SystemExit(f"missing required file: {path}")
  preflight = available_memory_bytes()
  preflight_floor = int(args.min_available_gib * (1024 ** 3))
  if preflight < preflight_floor:
    raise SystemExit(
        f"preflight memory {preflight} below {preflight_floor} bytes")

  result_path = raw / "worker-result.json"
  worker_config = raw / "worker-config.json"
  write_json(worker_config, {
      "abort_below_available_gib": args.abort_below_available_gib,
      "logits": str(logits),
      "raw": str(raw),
      "result": str(result_path),
      "samples": args.samples,
      "warmup": args.warmup,
  })
  command = [
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(worker_config),
  ]
  started = time.perf_counter_ns()
  run = subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True,
      timeout=args.timeout_s, check=False)
  elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  (raw / "worker.stdout").write_text(run.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(run.stderr, encoding="utf-8")
  result = load_json(result_path) if result_path.is_file() else {}
  memory_rows = result.get("memory_rows", [])
  minimum_available = min(
      (int(row["available_bytes"]) for row in memory_rows),
      default=available_memory_bytes())
  peak_rss = max(
      (int(row["rss_bytes"]) for row in memory_rows), default=0)
  peak_swap = max(
      (int(row["swap_bytes"]) for row in memory_rows), default=0)
  runtime_names = {str(row.get("node_name"))
                   for row in result.get("runtime", [])}
  checks = [
      check("worker_passed", run.returncode == 0,
            returncode=run.returncode),
      check("device_tokens_exact", result.get("device_tokens_exact") is True),
      check("paired_samples_exact", len(result.get("blocks", [])) == MIN_SAMPLES,
            actual=len(result.get("blocks", [])), expected=MIN_SAMPLES),
      check("custom_runtime_nodes_exact", len(runtime_names) == 3,
            nodes=sorted(runtime_names)),
      check("memory_guard_not_tripped",
            minimum_available >= int(
                args.abort_below_available_gib * (1024 ** 3)),
            minimum_available_bytes=minimum_available),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  performance_pass = bool(
      required_checks_passed and
      result.get("inference", {}).get("saving_pass") is True)
  git = git_state()
  manifest = {
      "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "command": " ".join([
          str(OV_PYTHON), str(Path(__file__).relative_to(ROOT)),
          "--out-dir", str(args.out_dir), "--logits", str(args.logits),
          "--warmup", str(args.warmup), "--samples", str(args.samples),
          "--min-available-gib", str(args.min_available_gib),
          "--abort-below-available-gib",
          str(args.abort_below_available_gib),
      ]),
      "config": {"path": str(CONFIG.relative_to(ROOT)),
                 "sha256": sha256_file(CONFIG)},
      "git": git,
      "host": {
          "machine": platform.machine(),
          "node": platform.node(),
          "platform": platform.platform(),
      },
      "plugin": {"path": str(PLUGIN), "sha256": sha256_file(PLUGIN)},
      "schema_version": SCHEMA,
      "sources": [{"path": str(path.relative_to(ROOT)),
                   "sha256": sha256_file(path)} for path in SOURCES],
      "workstream": WORKSTREAM,
  }
  gate = {
      "checks": checks,
      "performance_pass": performance_pass,
      "required_checks_passed": required_checks_passed,
      "route_label": "component_promoted" if performance_pass else "rejected",
      "schema_version": SCHEMA,
      "worker_elapsed_ms": elapsed_ms,
  }
  memory = {
      "minimum_available_bytes": minimum_available,
      "peak_process_rss_bytes": peak_rss,
      "peak_process_swap_bytes": peak_swap,
      "preflight_available_bytes": preflight,
  }
  write_json(out_dir / "manifest.json", manifest)
  write_json(out_dir / "gate.json", gate)
  write_json(out_dir / "memory.json", memory)
  write_json(out_dir / "result.json", result)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
    for row in result.get("blocks", []):
      stream.write(json.dumps(row, sort_keys=True) + "\n")
  inference = result.get("inference", {})
  summary = [
      "# OpenVINO device greedy feedback component gate",
      "",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      f"- component performance pass: `{str(performance_pass).lower()}`",
      f"- control median: `{result.get('control_median_ms')} ms`",
      f"- candidate median: `{result.get('candidate_median_ms')} ms`",
      f"- paired median saving: "
      f"`{inference.get('point_estimate_saving_ms')} ms`",
      f"- one-sided 95% saving LCB: "
      f"`{inference.get('lower_confidence_bound_saving_ms')} ms`",
      f"- required saving: `{REQUIRED_SAVING_MS} ms`",
      "",
      "The control returns all 248320 F32 logits and runs host NumPy argmax. "
      "The candidate returns one I32 token after a two-pass GPU reduction.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir),
      "performance_pass": performance_pass,
      "required_checks_passed": required_checks_passed,
      "saving_lcb_ms": inference.get("lower_confidence_bound_saving_ms"),
      "saving_median_ms": inference.get("point_estimate_saving_ms"),
  }, indent=2, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
