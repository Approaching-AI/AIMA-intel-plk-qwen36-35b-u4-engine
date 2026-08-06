#!/usr/bin/env python3
"""Gate the one-layer OpenVINO hot-ring/cold-append state topology.

The target topology keeps an exact 8192-token F32 hot K/V ring in static
Variable state and overwrites it through a custom GPU operation that echoes
only the update payload. Older K/V and their signed block32 scales live in four
append-only Variable states. This gate uses the real Qwen full-attention
dimensions (2 KV heads, head size 256), fills a 16k-equivalent split, then
performs one decode update in the same InferRequest without importing state
data from the host.

This is a state-ownership and data-movement gate.  It is not an attention
correctness or end-to-end speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-hot-cold-state-topology-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_SOURCE = Path("/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_ring_update.xml"
CUSTOM_SOURCE = ROOT / "engine/openvino/custom/iq36_hot_ring_update.cl"

KV_HEADS = 2
HEAD_DIM = 256
BLOCK_SIZE = 32
SCALE_DIM = HEAD_DIM // BLOCK_SIZE
SCALE_BYTES_DIM = SCALE_DIM * 2
HOT_WINDOW = 8192
COLD_PREFILL = 8192
RING_SLOT = 137

HOT_STATE_NAMES = ("hot_key", "hot_value")
COLD_STATE_NAMES = ("cold_key", "cold_value")
SCALE_STATE_NAMES = ("cold_key_scale", "cold_value_scale")
ALL_STATE_NAMES = HOT_STATE_NAMES + COLD_STATE_NAMES + SCALE_STATE_NAMES


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-hot-cold-state-topology-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  if not path.is_file():
    return rows
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"{path}:{line_number}: expected JSON object")
    rows.append(value)
  return rows


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def sha256_array(value: Any, np: Any) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*arguments: str) -> str:
    run = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative_out = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    relative_out = ""
  dirty = [row for row in dirty
           if not relative_out or relative_out not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def build_trace(raw: Path, timeout_s: int) -> dict[str, Any]:
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure = subprocess.run(
      configure_command, cwd=ROOT, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET,
      "-j8",
  ]
  build = subprocess.run(
      build_command, cwd=ROOT, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
  result = {
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
      "library": str(TRACE_LIBRARY),
      "pass": bool(
          configure.returncode == 0 and build.returncode == 0 and
          TRACE_LIBRARY.is_file()),
  }
  write_json(raw / "trace-build.json", result)
  return result


def make_hot_values(np: Any, kind: str, tokens: int) -> Any:
  token = np.arange(tokens, dtype=np.float32).reshape(1, 1, tokens, 1)
  head = np.arange(KV_HEADS, dtype=np.float32).reshape(1, KV_HEADS, 1, 1)
  dim = np.arange(HEAD_DIM, dtype=np.float32).reshape(1, 1, 1, HEAD_DIM)
  if kind == "key":
    return (((token * 3 + head * 17 + dim * 5) % 1021) - 510) / 128
  return (((token * 7 + head * 29 + dim * 11) % 1013) - 506) / 256


def make_cold_values(np: Any, kind: str, tokens: int) -> Any:
  token = np.arange(tokens, dtype=np.int32).reshape(1, 1, tokens, 1)
  head = np.arange(KV_HEADS, dtype=np.int32).reshape(1, KV_HEADS, 1, 1)
  dim = np.arange(HEAD_DIM, dtype=np.int32).reshape(1, 1, 1, HEAD_DIM)
  factor = 3 if kind == "key" else 7
  offset = 11 if kind == "key" else 37
  return (((token * factor + head * 19 + dim + offset) % 255) - 127
          ).astype(np.int8)


def make_scales(np: Any, kind: str, tokens: int) -> Any:
  token = np.arange(tokens, dtype=np.float32).reshape(1, 1, tokens, 1)
  head = np.arange(KV_HEADS, dtype=np.float32).reshape(1, KV_HEADS, 1, 1)
  block = np.arange(SCALE_DIM, dtype=np.float32).reshape(
      1, 1, 1, SCALE_DIM)
  offset = 1 if kind == "key" else 9
  scales = (0.00390625 *
            (1 + ((token + head * 3 + block + offset) % 31))).astype(
                np.float16)
  return np.ascontiguousarray(scales).view(np.int8).reshape(
      1, KV_HEADS, tokens, SCALE_BYTES_DIM)


def state_arrays(
    request: Any, np: Any, names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
  rows = {}
  for state in request.query_state():
    if names is not None and state.name not in names:
      continue
    rows[state.name] = np.asarray(state.state.data).copy()
  return rows


def bind_request_owned_hot_states(request: Any) -> list[dict[str, Any]]:
  rows = []
  for state in request.query_state():
    if state.name not in HOT_STATE_NAMES:
      continue
    tensor = state.state
    before_shape = list(tensor.shape)
    before_type = str(tensor.element_type)
    # Assigning the request-owned Tensor handle to itself marks the static
    # Variable initialized without importing or materializing host data.
    state.state = tensor
    rebound = state.state
    rows.append({
        "name": state.name,
        "before_shape": before_shape,
        "after_shape": list(rebound.shape),
        "before_type": before_type,
        "after_type": str(rebound.element_type),
        "bytes": int(rebound.byte_size),
    })
  return rows


def state_summaries(states: dict[str, Any], np: Any) -> dict[str, Any]:
  return {
      name: {
          "shape": list(value.shape),
          "dtype": str(value.dtype),
          "sha256": sha256_array(value, np),
      }
      for name, value in sorted(states.items())
  }


def matched_outputs(
    outputs: dict[Any, Any], expected: dict[str, Any], np: Any,
) -> dict[str, Any]:
  candidates = [np.asarray(value).copy() for value in outputs.values()]
  matched = {}
  for name, target in expected.items():
    for index, candidate in enumerate(candidates):
      if (candidate.shape == target.shape and
          candidate.dtype == target.dtype and
          np.array_equal(candidate, target)):
        matched[name] = candidates.pop(index)
        break
  return matched


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    rows.append({
        "node_name": node.get_friendly_name(),
        "layer_type": str(info.get("layerType", "")),
        "primitive_type": str(info.get("primitiveType", "")),
        "runtime_precision": str(info.get("runtimePrecision", "")),
        "output_layouts": str(info.get("outputLayouts", "")),
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


def make_model(ov: Any) -> Any:
  class IQ36HotRingUpdate(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(1)
      self.set_output_type(
          0, self.get_input_element_type(1),
          self.get_input_partial_shape(1))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36HotRingUpdate(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  parameter = ov.opset13.parameter
  hot_key_update = parameter(
      [1, KV_HEADS, -1, HEAD_DIM], ov.Type.f32,
      name="hot_key_update")
  hot_value_update = parameter(
      [1, KV_HEADS, -1, HEAD_DIM], ov.Type.f32,
      name="hot_value_update")
  hot_base_slot = parameter(
      [1, 1, 1, 1], ov.Type.i32, name="hot_base_slot")
  cold_key_append = parameter(
      [1, KV_HEADS, -1, HEAD_DIM], ov.Type.i8,
      name="cold_key_append")
  cold_value_append = parameter(
      [1, KV_HEADS, -1, HEAD_DIM], ov.Type.i8,
      name="cold_value_append")
  cold_key_scale_append = parameter(
      [1, KV_HEADS, -1, SCALE_BYTES_DIM], ov.Type.i8,
      name="cold_key_scale_append")
  cold_value_scale_append = parameter(
      [1, KV_HEADS, -1, SCALE_BYTES_DIM], ov.Type.i8,
      name="cold_value_scale_append")

  results = []
  sinks = []
  for name, update in (
      (HOT_STATE_NAMES[0], hot_key_update),
      (HOT_STATE_NAMES[1], hot_value_update),
  ):
    past = ov.opset13.read_value(
        name, ov.Type.f32,
        ov.PartialShape([1, KV_HEADS, HOT_WINDOW, HEAD_DIM]),
        f"{name}_past")
    ring_update = IQ36HotRingUpdate([past, update, hot_base_slot])
    ring_update.set_friendly_name(f"{name}_ring_update")
    result = ov.opset13.result(ring_update)
    result.set_friendly_name(f"{name}_update_echo")
    results.append(result)

  for name, append, element_type, width in (
      (COLD_STATE_NAMES[0], cold_key_append, ov.Type.i8, HEAD_DIM),
      (COLD_STATE_NAMES[1], cold_value_append, ov.Type.i8, HEAD_DIM),
      (SCALE_STATE_NAMES[0], cold_key_scale_append, ov.Type.i8,
       SCALE_BYTES_DIM),
      (SCALE_STATE_NAMES[1], cold_value_scale_append, ov.Type.i8,
       SCALE_BYTES_DIM),
  ):
    past = ov.opset13.read_value(
        name, element_type, ov.PartialShape([1, KV_HEADS, -1, width]),
        f"{name}_past")
    present = ov.opset13.concat([past, append], 2)
    present.set_friendly_name(f"{name}_append")
    sinks.append(ov.opset13.assign(present, name, f"{name}_assign"))
    result = ov.opset13.result(present)
    result.set_friendly_name(f"{name}_state_result")
    results.append(result)

  parameters = [
      hot_key_update, hot_value_update, hot_base_slot,
      cold_key_append, cold_value_append,
      cold_key_scale_append, cold_value_scale_append,
  ]
  return ov.Model(results, sinks, parameters, "iq36_hot_cold_state")


def worker_main(args: argparse.Namespace) -> int:
  import numpy as np
  import openvino as ov

  config = load_json(args.worker_config)
  marker = Path(config["marker"])
  result_path = Path(config["result"])
  device = str(config["device"])
  if Path(sys.prefix).resolve() != args.openvino_python.parent.parent.resolve():
    raise RuntimeError(
        f"worker requires {args.openvino_python}, observed {sys.executable}")

  model = make_model(ov)
  raw_graph = [{
      "node_name": node.get_friendly_name(),
      "type_name": node.get_type_name(),
      "version": node.get_type_info().version_id,
      "output_shape": (str(node.get_output_partial_shape(0))
                       if node.get_output_size() else ""),
  } for node in model.get_ordered_ops()]
  core = ov.Core()
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  core.set_property(device, {"CONFIG_FILE": str(CUSTOM_CONFIG.resolve())})
  config_after = str(core.get_property(device, "CONFIG_FILE"))
  compiled = core.compile_model(
      model, device,
      {"INFERENCE_PRECISION_HINT": ov.Type.f32,
       "PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  runtime = runtime_rows(compiled)
  request = compiled.create_infer_request()
  request.reset_state()
  hot_self_bind = bind_request_owned_hot_states(request)
  reset_states = state_arrays(request, np, HOT_STATE_NAMES)

  hot_key = make_hot_values(np, "key", HOT_WINDOW).astype(np.float32)
  hot_value = make_hot_values(np, "value", HOT_WINDOW).astype(np.float32)
  cold_key = make_cold_values(np, "key", COLD_PREFILL)
  cold_value = make_cold_values(np, "value", COLD_PREFILL)
  cold_key_scale = make_scales(np, "key", COLD_PREFILL)
  cold_value_scale = make_scales(np, "value", COLD_PREFILL)
  prefill_inputs = {
      "hot_key_update": hot_key,
      "hot_value_update": hot_value,
      "hot_base_slot": np.zeros((1, 1, 1, 1), dtype=np.int32),
      "cold_key_append": cold_key,
      "cold_value_append": cold_value,
      "cold_key_scale_append": cold_key_scale,
      "cold_value_scale_append": cold_value_scale,
  }
  prefill_outputs = request.infer(prefill_inputs, share_outputs=False)
  prefill_states = state_arrays(request, np, HOT_STATE_NAMES)
  prefill_states.update(matched_outputs(prefill_outputs, {
      COLD_STATE_NAMES[0]: cold_key,
      COLD_STATE_NAMES[1]: cold_value,
      SCALE_STATE_NAMES[0]: cold_key_scale,
      SCALE_STATE_NAMES[1]: cold_value_scale,
  }, np))

  decode_hot_key = make_hot_values(np, "key", 1).astype(np.float32) + 64
  decode_hot_value = make_hot_values(np, "value", 1).astype(np.float32) - 32
  decode_cold_key = make_cold_values(np, "key", 1) + np.int8(1)
  decode_cold_value = make_cold_values(np, "value", 1) - np.int8(1)
  decode_key_scale = make_scales(np, "key", 1)
  decode_value_scale = make_scales(np, "value", 1)
  decode_inputs = {
      "hot_key_update": decode_hot_key,
      "hot_value_update": decode_hot_value,
      "hot_base_slot": np.full(
          (1, 1, 1, 1), RING_SLOT, dtype=np.int32),
      "cold_key_append": decode_cold_key,
      "cold_value_append": decode_cold_value,
      "cold_key_scale_append": decode_key_scale,
      "cold_value_scale_append": decode_value_scale,
  }
  marker.write_text("decode-hot-cold\n", encoding="utf-8")
  decode_outputs = request.infer(decode_inputs, share_outputs=False)
  marker.unlink(missing_ok=True)
  decode_profile = profile_rows(request)
  decode_states = state_arrays(request, np, HOT_STATE_NAMES)
  expected_decode_cold = {
      COLD_STATE_NAMES[0]: np.concatenate(
          [cold_key, decode_cold_key], axis=2),
      COLD_STATE_NAMES[1]: np.concatenate(
          [cold_value, decode_cold_value], axis=2),
      SCALE_STATE_NAMES[0]: np.concatenate(
          [cold_key_scale, decode_key_scale], axis=2),
      SCALE_STATE_NAMES[1]: np.concatenate(
          [cold_value_scale, decode_value_scale], axis=2),
  }
  decode_states.update(matched_outputs(
      decode_outputs, expected_decode_cold, np))

  hot_key_unchanged = np.concatenate(
      [hot_key[:, :, :RING_SLOT, :],
       hot_key[:, :, RING_SLOT + 1:, :]], axis=2)
  hot_value_unchanged = np.concatenate(
      [hot_value[:, :, :RING_SLOT, :],
       hot_value[:, :, RING_SLOT + 1:, :]], axis=2)
  decode_hot_key_unchanged = np.concatenate(
      [decode_states[HOT_STATE_NAMES[0]][:, :, :RING_SLOT, :],
       decode_states[HOT_STATE_NAMES[0]][:, :, RING_SLOT + 1:, :]], axis=2)
  decode_hot_value_unchanged = np.concatenate(
      [decode_states[HOT_STATE_NAMES[1]][:, :, :RING_SLOT, :],
       decode_states[HOT_STATE_NAMES[1]][:, :, RING_SLOT + 1:, :]], axis=2)
  semantics = {
      "reset_hot_zero": all(
          name in reset_states and reset_states[name].shape ==
          (1, KV_HEADS, HOT_WINDOW, HEAD_DIM) and
          bool(np.count_nonzero(reset_states[name]) == 0)
          for name in HOT_STATE_NAMES),
      "prefill_hot_exact": bool(
          np.array_equal(prefill_states[HOT_STATE_NAMES[0]], hot_key) and
          np.array_equal(prefill_states[HOT_STATE_NAMES[1]], hot_value)),
      "prefill_cold_exact": bool(
          np.array_equal(prefill_states[COLD_STATE_NAMES[0]], cold_key) and
          np.array_equal(prefill_states[COLD_STATE_NAMES[1]], cold_value) and
          np.array_equal(
              prefill_states[SCALE_STATE_NAMES[0]], cold_key_scale) and
          np.array_equal(
              prefill_states[SCALE_STATE_NAMES[1]], cold_value_scale)),
      "decode_hot_slot_exact": bool(
          np.array_equal(
              decode_states[HOT_STATE_NAMES[0]][:, :, RING_SLOT:RING_SLOT + 1, :],
              decode_hot_key) and
          np.array_equal(
              decode_states[HOT_STATE_NAMES[1]][:, :, RING_SLOT:RING_SLOT + 1, :],
              decode_hot_value)),
      "decode_hot_other_slots_exact": bool(
          np.array_equal(decode_hot_key_unchanged, hot_key_unchanged) and
          np.array_equal(decode_hot_value_unchanged, hot_value_unchanged)),
      "decode_cold_prefix_exact": bool(
          np.array_equal(
              decode_states[COLD_STATE_NAMES[0]][:, :, :-1, :], cold_key) and
          np.array_equal(
              decode_states[COLD_STATE_NAMES[1]][:, :, :-1, :], cold_value) and
          np.array_equal(
              decode_states[SCALE_STATE_NAMES[0]][:, :, :-1, :],
              cold_key_scale) and
          np.array_equal(
              decode_states[SCALE_STATE_NAMES[1]][:, :, :-1, :],
              cold_value_scale)),
      "decode_cold_append_exact": bool(
          np.array_equal(
              decode_states[COLD_STATE_NAMES[0]][:, :, -1:, :],
              decode_cold_key) and
          np.array_equal(
              decode_states[COLD_STATE_NAMES[1]][:, :, -1:, :],
              decode_cold_value) and
          np.array_equal(
              decode_states[SCALE_STATE_NAMES[0]][:, :, -1:, :],
              decode_key_scale) and
          np.array_equal(
              decode_states[SCALE_STATE_NAMES[1]][:, :, -1:, :],
              decode_value_scale)),
      "hot_update_echoes_exact": bool(
          sum(np.asarray(value).shape == hot_key.shape and
              np.array_equal(np.asarray(value), hot_key)
              for value in prefill_outputs.values()) == 1 and
          sum(np.asarray(value).shape == hot_value.shape and
              np.array_equal(np.asarray(value), hot_value)
              for value in prefill_outputs.values()) == 1 and
          sum(np.asarray(value).shape == decode_hot_key.shape and
              np.array_equal(np.asarray(value), decode_hot_key)
              for value in decode_outputs.values()) == 1 and
          sum(np.asarray(value).shape == decode_hot_value.shape and
              np.array_equal(np.asarray(value), decode_hot_value)
              for value in decode_outputs.values()) == 1),
  }
  result = {
      "openvino_version": ov.get_version(),
      "config_before": config_before,
      "config_after": config_after,
      "raw_graph": raw_graph,
      "runtime": runtime,
      "decode_profile": decode_profile,
      "hot_self_bind": hot_self_bind,
      "reset_states": state_summaries(reset_states, np),
      "prefill_states": state_summaries(prefill_states, np),
      "decode_states": state_summaries(decode_states, np),
      "semantics": semantics,
  }
  write_json(result_path, result)
  return 0


def work_items(rows: list[dict[str, Any]]) -> int:
  size = rows[0].get("global_size") if rows else None
  if not isinstance(size, list) or not size:
    return 0
  product = 1
  for value in size:
    if not isinstance(value, int):
      return 0
    product *= value
  return product


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args)

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  marker = raw / "trace-active"
  trace_path = raw / "dispatch-trace.jsonl"
  worker_result_path = raw / "worker-result.json"
  worker_config_path = raw / "worker-config.json"
  write_json(worker_config_path, {
      "device": args.device,
      "marker": str(marker),
      "result": str(worker_result_path),
  })
  trace_build = build_trace(raw, args.timeout_s)
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(worker_config_path),
      "--openvino-python", str(args.openvino_python),
  ]
  env = os.environ.copy()
  env.update({
      "LD_PRELOAD": str(TRACE_LIBRARY),
      "IQ36_OPENCL_TRACE_MARKER": str(marker),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path),
      "IQ36_OPENCL_TRACE_FILTER": "_",
      "IQ36_OPENCL_TRACE_TIMING": "1",
  })
  worker = subprocess.run(
      command, cwd=ROOT, env=env, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=args.timeout_s)
  (raw / "worker.stdout").write_text(worker.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(worker.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": command,
      "returncode": worker.returncode,
      "trace_filter": env["IQ36_OPENCL_TRACE_FILTER"],
  })

  result = (
      load_json(worker_result_path) if worker_result_path.is_file() else {})
  trace = load_jsonl(trace_path)
  dispatches = [row for row in trace if row.get("event") == "ndrange"]
  hot_dispatches = [
      row for row in dispatches
      if "iq36_hot_ring_update" in str(row.get("kernel", ""))]
  concat_dispatches = [
      row for row in dispatches
      if "concat" in str(row.get("kernel", "")).lower()]
  hot_kernel_us = sum(
      float(row.get("duration_ns", 0)) for row in hot_dispatches) / 1000.0
  cold_append_kernel_us = sum(
      float(row.get("duration_ns", 0)) for row in concat_dispatches) / 1000.0
  runtime = result.get("runtime", [])
  hot_runtime = [
      row for row in runtime
      if row.get("node_name") in {
          "hot_key_ring_update", "hot_value_ring_update"}]
  cold_runtime = [
      row for row in runtime
      if row.get("layer_type") == "kv_cache" and
      str(row.get("node_name", "")).startswith("cold_")]
  profile = result.get("decode_profile", [])
  hot_profile = [
      row for row in profile
      if row.get("node_name") in {
          "hot_key_ring_update", "hot_value_ring_update"}]
  cold_profile = [
      row for row in profile
      if row.get("node_type") == "KVCache" and
      str(row.get("node_name", "")).startswith("cold_")]
  semantics = result.get("semantics", {})
  git = git_state(out_dir)
  source_head = ""
  if OV_SOURCE.is_dir():
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=OV_SOURCE, check=False,
        capture_output=True, text=True).stdout.strip()

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_openvino_source_commit", source_head == OV_COMMIT,
            source=str(OV_SOURCE), expected=OV_COMMIT, observed=source_head),
      check("trace_library_builds", bool(trace_build.get("pass")),
            build=trace_build),
      check("worker_completed", worker.returncode == 0,
            returncode=worker.returncode, stderr=worker.stderr),
      check("custom_hot_ring_config_is_exclusively_bound",
            CUSTOM_CONFIG.is_file() and CUSTOM_SOURCE.is_file() and
            result.get("config_before") == "" and
            result.get("config_after") == str(CUSTOM_CONFIG.resolve()),
            config=str(CUSTOM_CONFIG),
            config_sha256=(sha256_file(CUSTOM_CONFIG)
                           if CUSTOM_CONFIG.is_file() else None),
            source=str(CUSTOM_SOURCE),
            source_sha256=(sha256_file(CUSTOM_SOURCE)
                           if CUSTOM_SOURCE.is_file() else None),
            before=result.get("config_before"),
            after=result.get("config_after")),
      check("real_one_layer_16k_split_shape",
            HOT_WINDOW == 8192 and COLD_PREFILL == 8192 and
            KV_HEADS == 2 and HEAD_DIM == 256 and SCALE_DIM == 8,
            hot_window=HOT_WINDOW, cold_prefill=COLD_PREFILL,
            kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE, scale_dim=SCALE_DIM),
      check("six_variable_states_have_locked_types_and_shapes",
            set(result.get("decode_states", {})) == set(ALL_STATE_NAMES) and
            all(result["decode_states"][name].get("shape") ==
                [1, KV_HEADS, HOT_WINDOW, HEAD_DIM] and
                result["decode_states"][name].get("dtype") == "float32"
                for name in HOT_STATE_NAMES) and
            all(result["decode_states"][name].get("shape") ==
                [1, KV_HEADS, COLD_PREFILL + 1, HEAD_DIM] and
                result["decode_states"][name].get("dtype") == "int8"
                for name in COLD_STATE_NAMES) and
            all(result["decode_states"][name].get("shape") ==
                [1, KV_HEADS, COLD_PREFILL + 1, SCALE_BYTES_DIM] and
                result["decode_states"][name].get("dtype") == "int8"
                for name in SCALE_STATE_NAMES),
            states=result.get("decode_states", {})),
      check("same_request_state_semantics_are_bit_exact",
            bool(semantics) and all(semantics.values()),
            semantics=semantics),
      check("request_owned_hot_states_are_self_bound_once_without_import",
            len(result.get("hot_self_bind", [])) == 2 and
            {row.get("name") for row in result.get("hot_self_bind", [])} ==
            set(HOT_STATE_NAMES) and
            all(row.get("before_shape") ==
                [1, KV_HEADS, HOT_WINDOW, HEAD_DIM] and
                row.get("after_shape") == row.get("before_shape") and
                "float32" in str(row.get("before_type")) and
                "float32" in str(row.get("after_type"))
                for row in result.get("hot_self_bind", [])),
            bindings=result.get("hot_self_bind", [])),
      check("runtime_lowers_hot_ring_to_custom_f32_update",
            len(hot_runtime) == 2 and
            all(row.get("layer_type") == "CustomGPUPrimitive" and
                row.get("runtime_precision") == "f32"
                for row in hot_runtime), runtime=hot_runtime),
      check("runtime_lowers_cold_states_to_append_only_kv_cache",
            len(cold_runtime) == 4 and
            all(row.get("layer_type") == "kv_cache"
                for row in cold_runtime) and
            sorted(row.get("runtime_precision") for row in cold_runtime) ==
            ["i8", "i8", "i8", "i8"], runtime=cold_runtime),
      check("hot_custom_update_writes_only_one_slot_payload",
            len(hot_dispatches) == 2 and
            all(work_items([row]) == KV_HEADS * HEAD_DIM and
                any(arg.get("mem_bytes") ==
                    KV_HEADS * HOT_WINDOW * HEAD_DIM * 4
                    for arg in row.get("args", [])
                    if isinstance(arg, dict)) and
                not any(work_items([candidate]) ==
                        KV_HEADS * HOT_WINDOW * HEAD_DIM
                        for candidate in hot_dispatches)
                for row in hot_dispatches),
            dispatches=hot_dispatches),
      check("no_full_history_cold_concat_dispatch",
            len(concat_dispatches) >= 4 and
            max((work_items([row]) for row in concat_dispatches), default=0)
            <= KV_HEADS * HEAD_DIM,
            dispatches=concat_dispatches),
      check("profile_executes_two_hot_updates_and_four_cold_appends",
            len(hot_profile) == 2 and
            all(row.get("status") == "Status.EXECUTED"
                for row in hot_profile) and
            len(cold_profile) == 4 and
            all(row.get("status") == "Status.EXECUTED"
                for row in cold_profile),
            hot=hot_profile, cold=cold_profile),
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "required_checks_passed": passed,
      "checks": checks,
      "openvino_version": result.get("openvino_version"),
      "topology": {
          "hot": {
              "storage": "F32 static Variable ring mutated by custom GPU op",
              "shape_per_k_or_v": [1, KV_HEADS, HOT_WINDOW, HEAD_DIM],
              "decode_update_values_per_k_or_v": KV_HEADS * HEAD_DIM,
              "request_owned_self_bind_once": True,
          },
          "cold": {
              "storage": (
                  "signed I8 append-only Variable plus exact F16 block32 "
                  "scale bytes in I8 Variable"),
              "prefill_tokens": COLD_PREFILL,
              "decode_append_tokens": 1,
          },
          "same_infer_request": True,
          "external_state_import": False,
      },
      "runtime": runtime,
      "decode_profile": profile,
      "dispatch_trace": dispatches,
      "state_update_kernel_us": {
          "hot_custom_total": hot_kernel_us,
          "cold_append_total": cold_append_kernel_us,
          "one_layer_total": hot_kernel_us + cold_append_kernel_us,
          "trace_timing_instrumented": True,
          "excludes_gate_only_state_result_reorders": True,
      },
  }
  correctness = {
      "schema": f"{SCHEMA}-correctness",
      "required_checks_passed": passed,
      "semantics": semantics,
      "reset_states": result.get("reset_states", {}),
      "prefill_states": result.get("prefill_states", {}),
      "decode_states": result.get("decode_states", {}),
  }
  manifest = {
      "schema": f"{SCHEMA}-manifest",
      "commit": git["commit"],
      "dirty": git["dirty"],
      "command": command,
      "device": args.device,
      "openvino_python": str(args.openvino_python),
      "openvino_source": str(OV_SOURCE),
      "openvino_commit": source_head,
      "trace_library": str(TRACE_LIBRARY),
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "correctness.json", correctness)
  write_json(out_dir / "manifest.json", manifest)
  status = "PASS" if passed else "FAIL"
  summary = [
      "# OpenVINO hot/cold state topology gate",
      "",
      f"- status: `{status}`",
      f"- hot ring: `F32 [1,2,{HOT_WINDOW},{HEAD_DIM}]` for each K/V",
      "- cold state: signed `I8` K/V plus exact `F16` block32 scale bytes "
      "in `I8` Variable state",
      f"- represented prompt split: `{HOT_WINDOW + COLD_PREFILL}` tokens",
      f"- decode ring slot: `{RING_SLOT}`",
      f"- traced one-layer state-update kernels: "
      f"`{hot_kernel_us + cold_append_kernel_us:.3f} us` "
      f"(`{hot_kernel_us:.3f} us` hot + "
      f"`{cold_append_kernel_us:.3f} us` cold append)",
      "- state lifetime: one InferRequest from reset through prefill and decode",
      "- hot state activation: one request-owned Tensor self-bind; no host data",
      "- external state import: `false`",
      "",
      "This gate proves state ownership and bounded decode movement only. It "
      "does not claim attention correctness or end-to-end speed.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(f"{status}: {out_dir}")
  if not passed:
    for row in checks:
      if not row["pass"]:
        print(f"FAIL: {row['name']}", file=sys.stderr)
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
