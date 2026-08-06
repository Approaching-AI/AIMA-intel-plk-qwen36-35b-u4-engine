#!/usr/bin/env python3
"""Measure one OpenVINO MoE prefill route on the fixed 1024-token body."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


LM_HEAD_NAME = "__module.model.lm_head/ov_ext::linear/MatMul"
MODES = {
    "default_grouped": {},
    "micro": {
        "MOE_USE_GROUPED_GEMM_PREFILL": "0",
        "MOE_USE_MICRO_GEMM_PREFILL": "1",
    },
    "onednn_loop": {
        "MOE_USE_GROUPED_GEMM_PREFILL": "0",
        "MOE_USE_MICRO_GEMM_PREFILL": "0",
    },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=sorted(MODES), required=True)
  parser.add_argument(
      "--model", type=Path,
      default=Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml"))
  parser.add_argument("--seq-len", type=int, default=1024)
  parser.add_argument("--repeat", type=int, default=6)
  parser.add_argument("--dynamic-quantization-group-size", type=int, default=256)
  args = parser.parse_args()
  if args.seq_len < 2 or args.repeat < 4:
    parser.error("seq-len must be >=2 and repeat must be >=4")
  return args


def hidden_profile_model(ov: Any, model: Any) -> Any:
  lm_head = next(
      (op for op in model.get_ops()
       if op.get_friendly_name() == LM_HEAD_NAME), None)
  if lm_head is None:
    raise RuntimeError(f"lm_head node not found: {LM_HEAD_NAME}")
  result = ov.opset13.result(lm_head.input_value(0))
  result.set_friendly_name("hidden_states_result")
  return ov.Model(
      [result], model.get_sinks(), model.get_parameters(),
      "language_model_hidden_prefill_route_probe")


def make_inputs(np: Any, seq_len: int) -> dict[str, Any]:
  return {
      "attention_mask": np.ones((1, seq_len), dtype=np.int64),
      "inputs_embeds": np.zeros((1, seq_len, 2048), dtype=np.float32),
      "position_ids": np.tile(
          np.arange(seq_len, dtype=np.int64), (4, 1)).reshape(4, 1, seq_len),
      "beam_idx": np.zeros((1,), dtype=np.int32),
  }


def run_once(request: Any, inputs: dict[str, Any]) -> dict[str, Any]:
  request.reset_state()
  started = time.perf_counter()
  outputs = request.infer(inputs)
  wall_ms = (time.perf_counter() - started) * 1000.0

  node_type_us: dict[str, float] = defaultdict(float)
  exec_types: set[str] = set()
  profiled_sum_us = 0.0
  for item in request.get_profiling_info():
    real_us = item.real_time.total_seconds() * 1_000_000.0
    if real_us <= 0:
      continue
    profiled_sum_us += real_us
    node_type_us[item.node_type] += real_us
    if item.node_type == "MOE3GemmFusedCompressed":
      exec_types.add(item.exec_type)
  return {
      "wall_ms": wall_ms,
      "profiled_sum_ms": profiled_sum_us / 1000.0,
      "moe3gemm_profiled_ms":
          node_type_us.get("MOE3GemmFusedCompressed", 0.0) / 1000.0,
      "fully_connected_profiled_ms":
          node_type_us.get("FullyConnectedCompressed", 0.0) / 1000.0,
      "moe_exec_types": sorted(exec_types),
      "output_shapes": [list(value.shape) for value in outputs.values()],
  }


def main() -> int:
  args = parse_args()
  for name in ("MOE_USE_GROUPED_GEMM_PREFILL", "MOE_USE_MICRO_GEMM_PREFILL"):
    os.environ.pop(name, None)
  os.environ.update(MODES[args.mode])

  # Import only after the route environment is fixed. The plugin reads these
  # variables while constructing the compiled primitive.
  import numpy as np  # pylint: disable=import-outside-toplevel
  import openvino as ov  # pylint: disable=import-outside-toplevel

  core = ov.Core()
  source_model = core.read_model(str(args.model))
  profile_model = hidden_profile_model(ov, source_model)
  config = {
      "PERF_COUNT": True,
      "PERFORMANCE_HINT": "LATENCY",
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": args.dynamic_quantization_group_size,
  }
  compiled = core.compile_model(profile_model, "GPU", config)
  request = compiled.create_infer_request()
  inputs = make_inputs(np, args.seq_len)
  rows = [run_once(request, inputs) for _ in range(args.repeat)]
  warm_rows = rows[1:]
  walls = [float(row["wall_ms"]) for row in warm_rows]
  result = {
      "schema_version": "intel-qwen36-openvino-moe-route-probe-v0",
      "tool": "tools/intel-qwen36-openvino-moe-route-measure.py",
      "mode": args.mode,
      "route_environment": MODES[args.mode],
      "openvino_version": ov.get_version(),
      "python": platform.python_version(),
      "model": str(args.model.resolve()),
      "device": "GPU",
      "seq_len": args.seq_len,
      "repeat": args.repeat,
      "warmup_rows_excluded": 1,
      "warm_wall_median_ms": statistics.median(walls),
      "warm_wall_minimum_ms": min(walls),
      "warm_wall_maximum_ms": max(walls),
      "warm_wall_range_fraction": (max(walls) - min(walls)) / min(walls),
      "rows": rows,
      "notes": [
          "The first row is a same-process warmup and is excluded from the route median.",
          "Synthetic zero embeddings and PERF_COUNT make this an architecture differential, not a product denominator or correctness row.",
          "The exact installed OpenVINO source defines how the environment selects grouped, micro, and per-expert-loop paths.",
      ],
  }
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
