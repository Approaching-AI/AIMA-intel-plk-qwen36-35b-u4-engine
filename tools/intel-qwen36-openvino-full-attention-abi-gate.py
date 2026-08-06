#!/usr/bin/env python3
"""Lock one real full-attention layer's same-request OpenVINO ABI.

The worker runs the untouched product graph from reset through an exact 2k
prompt and one greedy decode token.  It records the raw graph state mapping,
the compiled IndirectSDPA/KV-cache path, logical state append ownership, and a
non-invasive OpenCL capture of layer 3's first decode dispatch.  This is an ABI
and state-lifetime gate, not a custom-kernel or performance claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-full-attention-abi-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
PROMPT = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_002k.txt")
TARGET_LAYER = 3
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
EXPECTED_INPUT_TOKENS = 2048
EXPECTED_STATE_COUNT = 80
KEY_VARIABLE = "cache_params.past.key.0cache_params.present.key.0"
VALUE_VARIABLE = "cache_params.past.value.0cache_params.present.value.0"
EXPECTED_SDPA_EXEC_TYPE = "ocl::sdpa::opt__f16"
MULTI_OUTPUT_EVIDENCE = (
    ROOT / "output/openvino-multi-output-custom-20260714Tseq811-cleanZ/"
    "metrics.json")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
COMPILE_CONFIG = {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
    "PERFORMANCE_HINT": "LATENCY",
    "PERF_COUNT": True,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--prompt", type=Path, default=PROMPT)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-full-attention-abi-{stamp}"
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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_array(value: Any, np: Any) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


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


def locked_file_rows(
    model_dir: Path, contract: dict[str, Any],
) -> list[dict[str, Any]]:
  rows = []
  locked = contract.get("product_model", {}).get("locked_files", {})
  for name, expected in sorted(locked.items()):
    path = model_dir / name
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    rows.append({
        "name": name,
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "sha256": digest,
        "expected_bytes": expected.get("bytes"),
        "expected_sha256": expected.get("sha256"),
        "pass": bool(
            exists and size == expected.get("bytes") and
            digest == expected.get("sha256")),
    })
  return rows


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


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def graph_value_row(label: str, value: Any) -> dict[str, Any]:
  node = value.get_node()
  return {
      "label": label,
      "element_type": str(value.get_element_type()),
      "partial_shape": str(value.get_partial_shape()),
      "producer_name": node.get_friendly_name(),
      "producer_output": int(value.get_index()),
      "producer_type": node.get_type_name(),
      "tensor_names": sorted(value.get_names()),
  }


def inspect_graph(model: Any, np: Any) -> dict[str, Any]:
  sdpa_nodes = [node for node in model.get_ordered_ops()
                if node.get_type_name() == "ScaledDotProductAttention"]
  layer_rows = []
  for node in sdpa_nodes:
    match = re.search(r"layers\.(\d+)\.self_attn", node.get_friendly_name())
    layer_rows.append({
        "layer": int(match.group(1)) if match else None,
        "name": node.get_friendly_name(),
        "input_count": node.get_input_size(),
        "output_count": node.get_output_size(),
    })
  target = next(
      node for node in sdpa_nodes
      if f"layers.{TARGET_LAYER}.self_attn" in node.get_friendly_name())
  assigns = {
      node.get_variable_id(): node for node in model.get_sinks()
      if node.get_type_name() == "Assign"
  }
  key_assign = assigns[KEY_VARIABLE]
  value_assign = assigns[VALUE_VARIABLE]
  present_key = key_assign.input_value(0)
  present_value = value_assign.input_value(0)
  key_concat = present_key.get_node()
  value_concat = present_value.get_node()

  output_chain = []
  cursor = target.output(0)
  for _ in range(3):
    consumers = list(cursor.get_target_inputs())
    if len(consumers) != 1:
      break
    node = consumers[0].get_node()
    output_chain.append({
        "type": node.get_type_name(),
        "name": node.get_friendly_name(),
        "output_shape": str(node.output(0).get_partial_shape()),
    })
    cursor = node.output(0)

  scale_node = target.input_value(4).get_node()
  scale_source = (
      scale_node.input_value(0).get_node()
      if scale_node.get_input_size() == 1 else None)
  scale_value = None
  if scale_source is not None and scale_source.get_type_name() == "Constant":
    scale_value = float(np.asarray(scale_source.get_data()).reshape(-1)[0])

  values = [
      graph_value_row("query_rotary_gqa16", target.input_value(0)),
      graph_value_row("key_repeated_gqa16", target.input_value(1)),
      graph_value_row("value_repeated_gqa16", target.input_value(2)),
      graph_value_row("attention_mask", target.input_value(3)),
      graph_value_row("attention_scale", target.input_value(4)),
      graph_value_row("attention_output_gqa16", target.output(0)),
      graph_value_row("past_key_after_beam_gather", key_concat.input_value(0)),
      graph_value_row("current_key_rope_gqa2", key_concat.input_value(1)),
      graph_value_row("present_key_state_gqa2", present_key),
      graph_value_row(
          "past_value_after_beam_gather", value_concat.input_value(0)),
      graph_value_row("current_value_gqa2", value_concat.input_value(1)),
      graph_value_row("present_value_state_gqa2", present_value),
  ]
  return {
      "operation_count": len(model.get_ordered_ops()),
      "sdpa_count": len(sdpa_nodes),
      "sdpa_layers": layer_rows,
      "target_layer": TARGET_LAYER,
      "target_name": target.get_friendly_name(),
      "target_inputs_and_state": values,
      "scale_value_f16_source": scale_value,
      "output_chain": output_chain,
      "key_state": {
          "variable_id": key_assign.get_variable_id(),
          "assign_name": key_assign.get_friendly_name(),
          "concat_name": key_concat.get_friendly_name(),
          "concat_axis": int(key_concat.get_axis()),
      },
      "value_state": {
          "variable_id": value_assign.get_variable_id(),
          "assign_name": value_assign.get_friendly_name(),
          "concat_name": value_concat.get_friendly_name(),
          "concat_axis": int(value_concat.get_axis()),
      },
  }


def make_inputs(
    embedding: Any, token_ids: Any, start: int, total: int, np: Any,
) -> dict[str, Any]:
  ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
  embedded = np.asarray(
      embedding({embedding.input(0): ids})[embedding.output(0)])
  positions = np.arange(start, start + ids.shape[1], dtype=np.int64)
  return {
      "attention_mask": np.ones((1, total), dtype=np.int64),
      "beam_idx": np.zeros((1,), dtype=np.int32),
      "inputs_embeds": embedded.astype(np.float32, copy=False),
      "position_ids": np.tile(positions, (4, 1)).reshape(
          4, 1, ids.shape[1]),
  }


def state_snapshot(request: Any, np: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  rows = []
  values = {}
  for state in request.query_state():
    value = np.array(state.state.data, copy=True)
    name = str(state.name)
    rows.append({
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": sha256_array(value, np),
    })
    if name in (KEY_VARIABLE, VALUE_VARIABLE):
      values[name] = value
  return sorted(rows, key=lambda row: row["name"]), values


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    layer_type = str(info.get("layerType"))
    is_attention = (
        "scaled_dot_product_attention" in node.get_friendly_name())
    if (not is_attention and
        layer_type not in ("IndirectSDPA", "KVCache", "ReadValue")):
      continue
    rows.append({
      "node_name": node.get_friendly_name(),
      "role": "sdpa" if is_attention else "state",
        "layer_type": layer_type,
        "primitive_type": str(info.get("primitiveType")),
        "runtime_precision": str(info.get("runtimePrecision")),
        "output_layouts": str(info.get("outputLayouts")),
        "output_precisions": str(info.get("outputPrecisions")),
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    if row.node_type not in ("IndirectSDPA", "KVCache"):
      continue
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  cfg = load_json(config_path)
  model_dir = Path(cfg["model_dir"])
  raw = Path(cfg["raw_dir"])
  core = ov.Core()
  config_before = str(core.get_property(cfg["device"], "CONFIG_FILE"))
  source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  graph = inspect_graph(source, np)
  embedding = core.compile_model(
      core.read_model(
          str(model_dir / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_ids = np.asarray(tokenizer.encode(
      Path(cfg["prompt"]).read_text(encoding="utf-8")
  ).input_ids.data).reshape(-1).astype(np.int64)
  np.ascontiguousarray(prompt_ids, dtype="<u4").tofile(
      raw / "prompt-token-ids.u32")

  compiled = core.compile_model(source, cfg["device"], COMPILE_CONFIG)
  request = compiled.create_infer_request()
  request.reset_state()
  prefill_outputs = request.infer(
      make_inputs(
          embedding, prompt_ids, 0, len(prompt_ids), np),
      share_outputs=False)
  prefill_logits = np.array(
      np.asarray(prefill_outputs[compiled.output(0)])[0, -1],
      dtype=np.float32, copy=True)
  next_token = int(np.argmax(prefill_logits))
  prefill_states, prefill_target = state_snapshot(request, np)
  for name, value in prefill_target.items():
    stem = "key" if name == KEY_VARIABLE else "value"
    np.ascontiguousarray(value).tofile(raw / f"prefill-{stem}-state.bin")

  Path(cfg["trace_marker"]).write_text(
      f"decode{len(prompt_ids) + 1}\n", encoding="utf-8")
  decode_outputs = request.infer(
      make_inputs(
          embedding, [next_token], len(prompt_ids), len(prompt_ids) + 1,
          np),
      share_outputs=False)
  decode_logits = np.array(
      np.asarray(decode_outputs[compiled.output(0)])[0, -1],
      dtype=np.float32, copy=True)
  decode_states, decode_target = state_snapshot(request, np)
  for name, value in decode_target.items():
    stem = "key" if name == KEY_VARIABLE else "value"
    np.ascontiguousarray(value).tofile(raw / f"decode-{stem}-state.bin")

  append_rows = []
  for name in (KEY_VARIABLE, VALUE_VARIABLE):
    before = prefill_target[name]
    after = decode_target[name]
    append_rows.append({
        "name": name,
        "prefill_shape": list(before.shape),
        "decode_shape": list(after.shape),
        "dtype": str(after.dtype),
        "prefix_exact_bits": bool(np.array_equal(
            before, after[:, :, :before.shape[2], :])),
        "appended_tokens": int(after.shape[2] - before.shape[2]),
        "appended_finite": bool(np.isfinite(
            after[:, :, before.shape[2]:, :]).all()),
        "prefill_sha256": sha256_array(before, np),
        "decode_sha256": sha256_array(after, np),
        "appended_sha256": sha256_array(
            after[:, :, before.shape[2]:, :], np),
    })

  result = {
      "config_before": config_before,
      "config_after": str(core.get_property(cfg["device"], "CONFIG_FILE")),
      "device": cfg["device"],
      "openvino_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "compile_config": COMPILE_CONFIG,
      "graph_abi": graph,
      "prompt": {
          "path": cfg["prompt"],
          "token_count": int(len(prompt_ids)),
          "token_sha256": sha256_file(raw / "prompt-token-ids.u32"),
      },
      "state_lifetime": {
          "reset_state_called": True,
          "external_state_set": False,
          "request_count": 2,
          "prefill_state_count": len(prefill_states),
          "decode_state_count": len(decode_states),
          "append_rows": append_rows,
      },
      "prefill": {
          "logits_finite": bool(np.isfinite(prefill_logits).all()),
          "logits_sha256": sha256_array(prefill_logits, np),
          "top1": next_token,
          "states": prefill_states,
      },
      "decode": {
          "input_token": next_token,
          "logits_finite": bool(np.isfinite(decode_logits).all()),
          "logits_sha256": sha256_array(decode_logits, np),
          "top1": int(np.argmax(decode_logits)),
          "states": decode_states,
          "profile": profile_rows(request),
      },
      "runtime": runtime_rows(compiled),
  }
  write_json(Path(cfg["result_path"]), result)
  print(json.dumps({
      "event": "complete",
      "prompt_tokens": len(prompt_ids),
      "prefill_top1": next_token,
      "decode_top1": int(np.argmax(decode_logits)),
      "state_count": len(decode_states),
  }), flush=True)
  return 0


def little_u32(argument: dict[str, Any]) -> int | None:
  value = str(argument.get("hex", ""))
  if len(value) != 8:
    return None
  return int.from_bytes(bytes.fromhex(value), byteorder="little")


def capture_summary(
    trace_rows: list[dict[str, Any]], capture_dir: Path,
) -> dict[str, Any]:
  dispatches = [row for row in trace_rows
                if row.get("event") == "ndrange"]
  captures = [row for row in trace_rows
              if row.get("event") == "capture"]
  first = dispatches[0] if dispatches else {}
  arguments = {int(row.get("index", -1)): row
               for row in first.get("args", [])}
  capture_rows = []
  for row in captures:
    path = Path(str(row.get("path", "")))
    capture_rows.append({
        **row,
        "exists": path.is_file(),
        "file_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "relative_path": relative(path) if path.is_file() else str(path),
    })
  logical_bytes = {
      "key": 1 * 2 * (EXPECTED_INPUT_TOKENS + 1) * 256 * 2,
      "query": 1 * 16 * 1 * 256 * 2,
      "value": 1 * 2 * (EXPECTED_INPUT_TOKENS + 1) * 256 * 2,
      "attention_output": 1 * 16 * 1 * 256 * 2,
      "mask": 1 * 1 * 1 * (EXPECTED_INPUT_TOKENS + 1) * 2,
  }
  return {
      "dispatch_count": len(dispatches),
      "capture_event_count": len(captures),
      "capture_directory": relative(capture_dir),
      "first_dispatch": first,
      "kernel_argument_order": {
          "1": "K f16 [batch,2,context,256]",
          "2": "Q f16 [batch,16,1,256]",
          "3": "V f16 [batch,2,context,256]",
          "4": "attention output f16 [batch,16,1,256]",
          "5": "attention mask f16 [batch,1,1,context]",
          "6": "head dimension d",
          "7": "source length k",
          "8": "query length q",
      },
      "scalar_arguments": {
          "d": little_u32(arguments.get(6, {})),
          "k": little_u32(arguments.get(7, {})),
          "q": little_u32(arguments.get(8, {})),
      },
      "buffer_arguments": {
          str(index): arguments.get(index, {}) for index in range(1, 6)
      },
      "minimum_logical_bytes": logical_bytes,
      "captures": capture_rows,
  }


def summary_markdown(metrics: dict[str, Any]) -> str:
  worker = metrics.get("worker", {})
  state = worker.get("state_lifetime", {})
  capture = metrics.get("opencl_capture", {})
  return "\n".join([
      "# OpenVINO full-attention same-request ABI gate",
      "",
      f"- required checks passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- target layer: `{TARGET_LAYER}`",
      f"- prompt/decode source length: `{worker.get('prompt', {}).get('token_count')} / {capture.get('scalar_arguments', {}).get('k')}`",
      f"- state count before/after decode: `{state.get('prefill_state_count')} / {state.get('decode_state_count')}`",
      f"- captured SDPA dispatches: `{capture.get('dispatch_count')}`",
      "- graph state: `[B,2,T,256]` K and V, append axis `2`",
      "- compiled decode ABI: `K,Q,V,A,mask,d,k,q` in F16",
      "- custom output packing required: `false` (dynamic multi-output proven by seq811)",
      "- speedup claim allowed: `false`",
      "",
      "The next route unit is the graph-owned hot/cold state update pattern;",
      "kernel work is not admitted until append/trim ownership compiles without",
      "full-history materialization.",
      "",
  ])


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  capture_dir = raw / "capture"
  cache_dir = raw / "neo-cache"
  raw.mkdir(parents=True, exist_ok=False)
  capture_dir.mkdir()
  cache_dir.mkdir()
  git = git_state(out_dir)
  contract = load_json(args.model_contract)
  locked = locked_file_rows(args.model_dir, contract)
  trace_build = build_trace(raw, min(args.timeout_s, 600))

  worker_result_path = raw / "worker-result.json"
  config = {
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(args.prompt.resolve()),
      "raw_dir": str(raw),
      "result_path": str(worker_result_path),
      "trace_marker": str(raw / "trace-active"),
  }
  config_path = raw / "worker-config.json"
  write_json(config_path, config)
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(config_path),
  ]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_TRACE_FILTER": "sdpa_micro__generate",
      "IQ36_OPENCL_TRACE_MARKER": str(raw / "trace-active"),
      "IQ36_OPENCL_TRACE_PATH": str(raw / "dispatch-trace.jsonl"),
      "IQ36_OPENCL_CAPTURE_DIR": str(capture_dir),
      "IQ36_OPENCL_CAPTURE_BEFORE_BEGIN": "1",
      "IQ36_OPENCL_CAPTURE_BEFORE_END": "6",
      "IQ36_OPENCL_CAPTURE_AFTER_BEGIN": "4",
      "IQ36_OPENCL_CAPTURE_AFTER_END": "5",
      "LD_AUDIT": str(TRACE_LIBRARY),
      "NEO_CACHE_DIR": str(cache_dir),
      "NEO_CACHE_MAX_SIZE": str(1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  worker = (
      subprocess.run(
          command, cwd=ROOT, env=env, check=False, capture_output=True,
          text=True, encoding="utf-8", errors="replace",
          timeout=args.timeout_s)
      if trace_build["pass"] else subprocess.CompletedProcess(
          command, 1, "", "trace build failed"))
  (raw / "worker.stdout").write_text(worker.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(worker.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": command,
      "environment": {key: env[key] for key in (
          "IQ36_OPENCL_TRACE_FILTER", "IQ36_OPENCL_TRACE_MARKER",
          "IQ36_OPENCL_TRACE_PATH", "IQ36_OPENCL_CAPTURE_DIR",
          "IQ36_OPENCL_CAPTURE_BEFORE_BEGIN",
          "IQ36_OPENCL_CAPTURE_BEFORE_END",
          "IQ36_OPENCL_CAPTURE_AFTER_BEGIN",
          "IQ36_OPENCL_CAPTURE_AFTER_END", "LD_AUDIT", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": worker.returncode,
  })
  result = (
      load_json(worker_result_path) if worker_result_path.is_file() else {})
  trace_rows = load_jsonl(raw / "dispatch-trace.jsonl")
  capture = capture_summary(trace_rows, capture_dir)
  graph = result.get("graph_abi", {})
  state = result.get("state_lifetime", {})
  append_rows = state.get("append_rows", [])
  runtime = result.get("runtime", [])
  runtime_sdpa = [row for row in runtime if row.get("role") == "sdpa"]
  profile_sdpa = [row for row in result.get("decode", {}).get("profile", [])
                  if row.get("node_type") == "IndirectSDPA"]
  first_args = capture.get("buffer_arguments", {})
  minimum = capture.get("minimum_logical_bytes", {})
  capture_events = capture.get("captures", [])
  captured_pairs = {(row.get("phase"), row.get("arg_index")): row
                    for row in capture_events}
  multi_output = (
      load_json(MULTI_OUTPUT_EVIDENCE)
      if MULTI_OUTPUT_EVIDENCE.is_file() else {})
  graph_rows = {row.get("label"): row
                for row in graph.get("target_inputs_and_state", [])}
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_product_model_matches_contract",
            bool(locked) and all(row["pass"] for row in locked),
            rows=locked),
      check("trace_library_builds", trace_build["pass"],
            build=trace_build),
      check("untouched_stock_worker_completed", worker.returncode == 0,
            returncode=worker.returncode, stderr=worker.stderr),
      check("stock_worker_never_loads_candidate_config",
            result.get("config_before") == "" and
            result.get("config_after") == "",
            before=result.get("config_before"),
            after=result.get("config_after")),
      check("exact_2k_prompt_and_finite_logits",
            result.get("prompt", {}).get("token_count") ==
            EXPECTED_INPUT_TOKENS and
            result.get("prefill", {}).get("logits_finite") is True and
            result.get("decode", {}).get("logits_finite") is True,
            prompt=result.get("prompt"),
            prefill_top1=result.get("prefill", {}).get("top1"),
            decode_top1=result.get("decode", {}).get("top1")),
      check("raw_graph_exposes_exact_ten_full_attention_layers",
            graph.get("sdpa_count") == len(FULL_ATTENTION_LAYERS) and
            [row.get("layer") for row in graph.get("sdpa_layers", [])] ==
            list(FULL_ATTENTION_LAYERS),
            layers=graph.get("sdpa_layers", [])),
      check("layer3_graph_abi_is_gqa16_over_persistent_gqa2",
            graph_rows.get("query_rotary_gqa16", {}).get("partial_shape") ==
            "[?,16,?,256]" and
            graph_rows.get("present_key_state_gqa2", {}).get(
                "partial_shape") == "[?,2,?,256]" and
            graph_rows.get("present_value_state_gqa2", {}).get(
                "partial_shape") == "[?,2,?,256]" and
            graph.get("key_state", {}).get("variable_id") == KEY_VARIABLE and
            graph.get("value_state", {}).get("variable_id") ==
            VALUE_VARIABLE and
            graph.get("key_state", {}).get("concat_axis") == 2 and
            graph.get("value_state", {}).get("concat_axis") == 2 and
            graph.get("scale_value_f16_source") == 0.0625,
            graph_abi=graph),
      check("same_request_builds_and_appends_prompt_state_from_reset",
            state.get("reset_state_called") is True and
            state.get("external_state_set") is False and
            state.get("request_count") == 2 and
            state.get("prefill_state_count") == EXPECTED_STATE_COUNT and
            state.get("decode_state_count") == EXPECTED_STATE_COUNT and
            len(append_rows) == 2 and all(
                row.get("prefill_shape") == [1, 2, 2048, 256] and
                row.get("decode_shape") == [1, 2, 2049, 256] and
                row.get("dtype") == "float32" and
                row.get("prefix_exact_bits") is True and
                row.get("appended_tokens") == 1 and
                row.get("appended_finite") is True
                for row in append_rows),
            state_lifetime=state),
      check("compiled_runtime_keeps_ten_f16_sdpa_nodes",
            len(runtime_sdpa) == len(FULL_ATTENTION_LAYERS) and
            len(profile_sdpa) == len(FULL_ATTENTION_LAYERS) and
            all(row.get("primitive_type") == EXPECTED_SDPA_EXEC_TYPE
                for row in runtime_sdpa) and
            all(row.get("exec_type") == EXPECTED_SDPA_EXEC_TYPE and
                row.get("status") == "Status.EXECUTED"
                for row in profile_sdpa),
            runtime=runtime_sdpa, profile=profile_sdpa),
      check("layer3_decode_dispatch_has_exact_d_k_q",
            capture.get("dispatch_count") == len(FULL_ATTENTION_LAYERS) and
            capture.get("first_dispatch", {}).get("marker") == "decode2049" and
            capture.get("scalar_arguments") == {"d": 256, "k": 2049, "q": 1},
            dispatch_count=capture.get("dispatch_count"),
            first_kernel=capture.get("first_dispatch", {}).get("kernel"),
            scalars=capture.get("scalar_arguments")),
      check("decode_k_q_v_output_mask_buffers_are_real_and_sized",
            int(first_args.get("1", {}).get("mem_bytes", 0)) >=
            int(minimum.get("key", math.inf)) and
            int(first_args.get("2", {}).get("mem_bytes", 0)) >=
            int(minimum.get("query", math.inf)) and
            int(first_args.get("3", {}).get("mem_bytes", 0)) >=
            int(minimum.get("value", math.inf)) and
            int(first_args.get("4", {}).get("mem_bytes", 0)) >=
            int(minimum.get("attention_output", math.inf)) and
            int(first_args.get("5", {}).get("mem_bytes", 0)) >=
            int(minimum.get("mask", math.inf)),
            buffers=first_args, minimum_logical_bytes=minimum),
      check("noninvasive_decode_buffers_captured_before_and_after",
            all(
                captured_pairs.get(key, {}).get("status") == 0 and
                captured_pairs.get(key, {}).get("exists") is True and
                captured_pairs.get(key, {}).get("file_bytes") ==
                captured_pairs.get(key, {}).get("bytes")
                for key in (
                    ("before", 1), ("before", 2), ("before", 3),
                    ("before", 4), ("before", 5), ("after", 4))),
            captures=capture_events),
      check("dynamic_multi_output_state_ports_are_admitted",
            multi_output.get("required_checks_passed") is True and
            multi_output.get("git", {}).get("commit") ==
            "c1f24489cf9984657a79826216ea6238ee39849b",
            evidence=relative(MULTI_OUTPUT_EVIDENCE),
            evidence_commit=multi_output.get("git", {}).get("commit")),
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
      "git": git,
      "worker": result,
      "opencl_capture": capture,
      "checks": checks,
      "conclusion": (
          "Layer 3 has a proven same-request GQA16/Q over GQA2 K/V append "
          "boundary. Dynamic SimpleGPU can expose attention and state on "
          "separate ports; the next risk is graph-owned hot/cold append and "
          "trim without full-history materialization."
          if passed else
          "The stock full-attention graph/state/dispatch ABI is not yet "
          "closed."),
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "correctness.json", {
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
      "checks": checks,
  })
  write_json(out_dir / "manifest.json", {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "tool": relative(Path(__file__)),
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
      "metrics": "metrics.json",
      "correctness": "correctness.json",
      "raw": "raw/",
      "git": git,
  })
  (out_dir / "summary.md").write_text(
      summary_markdown(metrics), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir),
      "required_checks_passed": passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
  }, indent=2))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
