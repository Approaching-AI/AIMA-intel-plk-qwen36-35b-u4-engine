#!/usr/bin/env python3
"""Gate one real OpenVINO full-attention custom arithmetic substitution.

The component lane replays layer 3's clean seq812 decode boundary.  The
language-model lane replaces only layer 3's stock SDPA with the same
parameterized GQA custom operation, then runs an exact 2k prompt and one
teacher-forced decode token in one InferRequest.  Stock and candidate execute
in isolated workers.  The stock F32 append state is intentionally retained in
this arithmetic subgate; bounded hot/cold state integration is the next gate.

This is correctness and substitution evidence, not a speedup claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-full-attention-custom-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
PROMPT = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_002k.txt")
ABI_EVIDENCE = (
    ROOT / "output/openvino-full-attention-abi-20260714Tseq812-cleanZ")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_full_attention_gqa.xml"
CUSTOM_SOURCE = ROOT / "engine/openvino/custom/iq36_full_attention_gqa.cl"
TARGET_LAYER = 3
EXPECTED_INPUT_TOKENS = 2048
EXPECTED_STATE_COUNT = 80
EXPECTED_SDPA_COUNT = 10
KEY_VARIABLE = "cache_params.past.key.0cache_params.present.key.0"
VALUE_VARIABLE = "cache_params.past.value.0cache_params.present.value.0"
COMPILE_CONFIG = {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
    "PERFORMANCE_HINT": "LATENCY",
    "PERF_COUNT": True,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--prompt", type=Path, default=PROMPT)
  parser.add_argument("--abi-evidence", type=Path, default=ABI_EVIDENCE)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
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
    args.out_dir = ROOT / f"output/openvino-full-attention-custom-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_array(value: Any, np: Any) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*arguments: str) -> str:
    run = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    output_relative = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  dirty = [row for row in dirty
           if not output_relative or output_relative not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def vector_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  if ref.shape != cand.shape:
    return {
        "count": int(ref.size), "candidate_count": int(cand.size),
        "shape_match": False, "finite": False, "exact_bits": False,
        "max_abs": float("inf"), "relative_l2": float("inf"),
        "cosine": 0.0,
    }
  difference = cand - ref
  ref_norm = float(np.linalg.norm(ref))
  cand_norm = float(np.linalg.norm(cand))
  denominator = ref_norm * cand_norm
  return {
      "count": int(ref.size),
      "candidate_count": int(cand.size),
      "shape_match": True,
      "finite": bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
      "exact_bits": bool(np.array_equal(reference, candidate)),
      "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
      "relative_l2": (
          float(np.linalg.norm(difference) / ref_norm)
          if ref_norm else float(np.linalg.norm(difference))),
      "cosine": (
          float(np.dot(ref, cand) / denominator) if denominator else 1.0),
  }


def distribution_metrics(
    reference: Any, candidate: Any, np: Any,
) -> dict[str, Any]:
  result = vector_metrics(reference, candidate, np)
  if not result["shape_match"] or not result["finite"]:
    return {
        **result, "kld_reference_to_candidate": float("inf"),
        "reference_top1": None, "candidate_top1": None,
        "top1_match": False,
    }
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  ref_probability = np.exp(ref - float(np.max(ref)))
  candidate_probability = np.exp(cand - float(np.max(cand)))
  ref_probability /= float(ref_probability.sum())
  candidate_probability /= float(candidate_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  reference_top1 = int(np.argmax(ref))
  candidate_top1 = int(np.argmax(cand))
  return {
      **result,
      "kld_reference_to_candidate": float(np.sum(
          ref_probability * (
              np.log(np.maximum(ref_probability, epsilon)) -
              np.log(np.maximum(candidate_probability, epsilon))))),
      "reference_top1": reference_top1,
      "candidate_top1": candidate_top1,
      "top1_match": reference_top1 == candidate_top1,
  }


def custom_class(ov: Any) -> type:
  class IQ36FullAttentionGQA(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, self.get_input_element_type(0),
          self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36FullAttentionGQA(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36FullAttentionGQA


def target_nodes(model: Any) -> tuple[Any, Any, Any]:
  target = next(
      node for node in model.get_ordered_ops()
      if node.get_type_name() == "ScaledDotProductAttention" and
      f"layers.{TARGET_LAYER}.self_attn" in node.get_friendly_name())
  assigns = {
      node.get_variable_id(): node for node in model.get_sinks()
      if node.get_type_name() == "Assign"
  }
  return target, assigns[KEY_VARIABLE], assigns[VALUE_VARIABLE]


def make_candidate_model(
    core: Any, model_dir: Path, ov: Any,
) -> tuple[Any, dict[str, Any]]:
  model = core.read_model(str(model_dir / "openvino_language_model.xml"))
  before = model.get_ordered_ops()
  target, key_assign, value_assign = target_nodes(model)
  present_key = key_assign.input_value(0)
  present_value = value_assign.input_value(0)
  operation = custom_class(ov)([
      target.input_value(0), present_key, present_value,
      target.input_value(3)])
  operation.set_friendly_name(f"iq36_full_attention_layer{TARGET_LAYER}")
  target.output(0).replace(operation.output(0))
  model.validate_nodes_and_infer_types()
  after = model.get_ordered_ops()
  summary = {
      "target_layer": TARGET_LAYER,
      "target_name": target.get_friendly_name(),
      "source_operation_count_before": len(before),
      "source_operation_count_after": len(after),
      "stock_sdpa_count_before": sum(
          node.get_type_name() == "ScaledDotProductAttention"
          for node in before),
      "stock_sdpa_count_after": sum(
          node.get_type_name() == "ScaledDotProductAttention"
          for node in after),
      "custom_count_after": sum(
          node.get_type_name() == "IQ36FullAttentionGQA" for node in after),
      "query_shape": str(operation.input_value(0).get_partial_shape()),
      "key_state_shape": str(operation.input_value(1).get_partial_shape()),
      "value_state_shape": str(operation.input_value(2).get_partial_shape()),
      "mask_shape": str(operation.input_value(3).get_partial_shape()),
      "output_shape": str(operation.output(0).get_partial_shape()),
      "state_carrier": "stock F32 append state retained for arithmetic gate",
  }
  return model, summary


def make_component_model(ov: Any) -> Any:
  shapes = (
      [1, 16, 1, 256],
      [1, 2, EXPECTED_INPUT_TOKENS + 1, 256],
      [1, 2, EXPECTED_INPUT_TOKENS + 1, 256],
      [1, 1, 1, EXPECTED_INPUT_TOKENS + 1],
  )
  parameters = [
      ov.opset13.parameter(shape, ov.Type.f32, name=f"boundary_{index}")
      for index, shape in enumerate(shapes)
  ]
  operation = custom_class(ov)([parameter.output(0)
                                for parameter in parameters])
  operation.set_friendly_name("iq36_full_attention_component")
  return ov.Model([operation.output(0)], parameters, "iq36_attention_component")


def runtime_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    layer_type = str(info.get("layerType", ""))
    if (layer_type not in (
        "scaled_dot_product_attention", "CustomGPUPrimitive") and
        "full_attention" not in node.get_friendly_name()):
      continue
    rows.append({
        "node_name": node.get_friendly_name(),
        "layer_type": layer_type,
        "primitive_type": str(info.get("primitiveType", "")),
        "runtime_precision": str(info.get("runtimePrecision", "")),
        "output_layouts": str(info.get("outputLayouts", "")),
        "output_precisions": str(info.get("outputPrecisions", "")),
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    if (row.node_type not in (
        "IndirectSDPA", "IQ36FullAttentionGQA") and
        "full_attention" not in row.node_name):
      continue
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


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


def state_snapshot(
    request: Any, phase: str, raw: Path, np: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  rows = []
  targets = {}
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
      kind = "key" if name == KEY_VARIABLE else "value"
      path = raw / f"{phase}-{kind}-state.bin"
      np.ascontiguousarray(value, dtype="<f4").tofile(path)
      targets[name] = {
          "path": relative(path),
          "shape": list(value.shape),
          "dtype": str(value.dtype),
          "sha256": sha256_file(path),
      }
  return sorted(rows, key=lambda row: row["name"]), targets


def run_component(
    core: Any, abi_dir: Path, device: str, ov: Any, np: Any,
) -> dict[str, Any]:
  model = make_component_model(ov)
  compiled = core.compile_model(model, device, COMPILE_CONFIG)
  raw = abi_dir / "raw"
  values = [
      np.fromfile(
          raw / "capture/dispatch000-arg2-before.bin", dtype="<f2"
      ).astype(np.float32).reshape(1, 16, 1, 256),
      np.fromfile(raw / "decode-key-state.bin", dtype="<f4").reshape(
          1, 2, EXPECTED_INPUT_TOKENS + 1, 256),
      np.fromfile(raw / "decode-value-state.bin", dtype="<f4").reshape(
          1, 2, EXPECTED_INPUT_TOKENS + 1, 256),
      np.fromfile(
          raw / "capture/dispatch000-arg5-before.bin", dtype="<f2"
      ).astype(np.float32).reshape(
          1, 1, 1, EXPECTED_INPUT_TOKENS + 1),
  ]
  request = compiled.create_infer_request()
  request.infer(dict(zip(compiled.inputs, values)), share_outputs=False)
  started = time.perf_counter_ns()
  outputs = request.infer(
      dict(zip(compiled.inputs, values)), share_outputs=False)
  wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  candidate = np.asarray(outputs[compiled.output(0)], dtype=np.float32)
  reference = np.fromfile(
      raw / "capture/dispatch000-arg4-after.bin", dtype="<f2"
  ).astype(np.float32).reshape(1, 16, 1, 256)
  return {
      "numeric": vector_metrics(reference, candidate, np),
      "candidate_shape": list(candidate.shape),
      "candidate_finite": bool(np.isfinite(candidate).all()),
      "runtime": runtime_rows(compiled),
      "profile": profile_rows(request),
      "wall_ms_diagnostic": wall_ms,
  }


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  cfg = load_json(config_path)
  mode = str(cfg["mode"])
  raw = Path(cfg["raw"])
  model_dir = Path(cfg["model_dir"])
  device = str(cfg["device"])
  core = ov.Core()
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  source_summary = None
  component = None
  no_config_error = ""
  if mode == "candidate":
    core.set_property(device, {"CONFIG_FILE": cfg["custom_config"]})
    source, source_summary = make_candidate_model(
        core, model_dir, ov)
  elif mode == "stock":
    source = core.read_model(
        str(model_dir / "openvino_language_model.xml"))
  else:
    raise ValueError(f"unknown worker mode: {mode}")
  config_after = str(core.get_property(device, "CONFIG_FILE"))

  embedding = core.compile_model(
      core.read_model(
          str(model_dir / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_ids = np.asarray(tokenizer.encode(
      Path(cfg["prompt"]).read_text(encoding="utf-8")
  ).input_ids.data).reshape(-1).astype(np.int64)
  token_path = raw / "prompt-token-ids.u32"
  np.ascontiguousarray(prompt_ids, dtype="<u4").tofile(token_path)

  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(source, device, COMPILE_CONFIG)
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
  request = compiled.create_infer_request()
  request.reset_state()

  prefill_started = time.perf_counter_ns()
  prefill_outputs = request.infer(
      make_inputs(
          embedding, prompt_ids, 0, len(prompt_ids), np),
      share_outputs=False)
  prefill_wall_ms = (
      time.perf_counter_ns() - prefill_started) / 1_000_000.0
  prefill_logits = np.array(
      np.asarray(prefill_outputs[compiled.output(0)])[0, -1],
      dtype="<f4", copy=True)
  prefill_logits_path = raw / "prefill-logits.f32"
  prefill_logits.tofile(prefill_logits_path)
  prefill_states, prefill_targets = state_snapshot(
      request, "prefill", raw, np)

  decode_token = int(cfg.get("decode_token", int(np.argmax(prefill_logits))))
  decode_started = time.perf_counter_ns()
  decode_outputs = request.infer(
      make_inputs(
          embedding, [decode_token], len(prompt_ids), len(prompt_ids) + 1,
          np),
      share_outputs=False)
  decode_wall_ms = (
      time.perf_counter_ns() - decode_started) / 1_000_000.0
  decode_logits = np.array(
      np.asarray(decode_outputs[compiled.output(0)])[0, -1],
      dtype="<f4", copy=True)
  decode_logits_path = raw / "decode-logits.f32"
  decode_logits.tofile(decode_logits_path)
  decode_states, decode_targets = state_snapshot(request, "decode", raw, np)

  append_rows = []
  for name in (KEY_VARIABLE, VALUE_VARIABLE):
    before = np.fromfile(
        ROOT / prefill_targets[name]["path"], dtype="<f4").reshape(
            prefill_targets[name]["shape"])
    after = np.fromfile(
        ROOT / decode_targets[name]["path"], dtype="<f4").reshape(
            decode_targets[name]["shape"])
    append_rows.append({
        "name": name,
        "prefix_exact_bits": bool(np.array_equal(
            before, after[:, :, :before.shape[2], :])),
        "appended_tokens": int(after.shape[2] - before.shape[2]),
        "appended_finite": bool(np.isfinite(
            after[:, :, before.shape[2]:, :]).all()),
    })

  runtime = runtime_rows(compiled)
  profile = profile_rows(request)
  if mode == "candidate":
    component = run_component(
        core, Path(cfg["abi_evidence"]), device, ov, np)
    try:
      ov.Core().compile_model(make_component_model(ov), device)
    except Exception as exc:
      no_config_error = repr(exc)

  result = {
      "mode": mode,
      "openvino_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "config_before": config_before,
      "config_after": config_after,
      "compile_config": COMPILE_CONFIG,
      "compile_ms": compile_ms,
      "prompt": {
          "path": cfg["prompt"],
          "token_count": int(len(prompt_ids)),
          "token_sha256": sha256_file(token_path),
      },
      "source_summary": source_summary,
      "runtime": runtime,
      "profile": profile,
      "prefill": {
          "wall_ms_diagnostic": prefill_wall_ms,
          "logits_path": relative(prefill_logits_path),
          "logits_sha256": sha256_file(prefill_logits_path),
          "logits_finite": bool(np.isfinite(prefill_logits).all()),
          "top1": int(np.argmax(prefill_logits)),
          "states": prefill_states,
          "target_states": prefill_targets,
      },
      "decode": {
          "input_token": decode_token,
          "wall_ms_diagnostic": decode_wall_ms,
          "logits_path": relative(decode_logits_path),
          "logits_sha256": sha256_file(decode_logits_path),
          "logits_finite": bool(np.isfinite(decode_logits).all()),
          "top1": int(np.argmax(decode_logits)),
          "states": decode_states,
          "target_states": decode_targets,
          "append_rows": append_rows,
      },
      "component": component,
      "no_config_error": no_config_error,
  }
  write_json(Path(cfg["result"]), result)
  print(json.dumps({
      "event": "complete", "mode": mode,
      "prompt_tokens": len(prompt_ids),
      "prefill_top1": int(np.argmax(prefill_logits)),
      "decode_input": decode_token,
      "decode_top1": int(np.argmax(decode_logits)),
      "state_count": len(decode_states),
  }), flush=True)
  return 0


def launch_worker(
    args: argparse.Namespace, mode: str, raw: Path,
    decode_token: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any], list[str]]:
  raw.mkdir()
  config_path = raw / "worker-config.json"
  result_path = raw / "worker-result.json"
  config = {
      "mode": mode,
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(args.prompt.resolve()),
      "abi_evidence": str(args.abi_evidence.resolve()),
      "custom_config": str(args.custom_config.resolve()),
      "raw": str(raw.resolve()),
      "result": str(result_path.resolve()),
  }
  if decode_token is not None:
    config["decode_token"] = decode_token
  write_json(config_path, config)
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(config_path),
  ]
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  run = subprocess.run(
      command, cwd=ROOT, env=environment, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace", timeout=args.timeout_s)
  (raw / "worker.stdout").write_text(run.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(run.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": command, "returncode": run.returncode,
  })
  result = load_json(result_path) if result_path.is_file() else {}
  return run, result, command


def load_logits(result: dict[str, Any], phase: str, np: Any) -> Any:
  path = ROOT / result[phase]["logits_path"]
  return np.fromfile(path, dtype="<f4")


def load_target_state(
    result: dict[str, Any], phase: str, name: str, np: Any,
) -> Any:
  row = result[phase]["target_states"][name]
  return np.fromfile(ROOT / row["path"], dtype="<f4").reshape(row["shape"])


def state_schema_comparison(
    stock: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> dict[str, Any]:
  stock_rows = {row["name"]: row for row in stock}
  candidate_rows = {row["name"]: row for row in candidate}
  common = sorted(set(stock_rows) & set(candidate_rows))
  return {
      "stock_count": len(stock_rows),
      "candidate_count": len(candidate_rows),
      "names_match": set(stock_rows) == set(candidate_rows),
      "all_finite": all(
          stock_rows[name]["finite"] and candidate_rows[name]["finite"]
          for name in common),
      "shape_dtype_match": all(
          stock_rows[name]["shape"] == candidate_rows[name]["shape"] and
          stock_rows[name]["dtype"] == candidate_rows[name]["dtype"]
          for name in common),
      "exact_state_count": sum(
          stock_rows[name]["sha256"] == candidate_rows[name]["sha256"]
          for name in common),
      "mismatch_names": [
          name for name in common
          if stock_rows[name]["sha256"] != candidate_rows[name]["sha256"]],
  }


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  import numpy as np

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = git_state(out_dir)
  abi_metrics_path = args.abi_evidence / "metrics.json"
  abi_metrics = load_json(abi_metrics_path) if abi_metrics_path.is_file() else {}

  stock_run, stock, stock_command = launch_worker(
      args, "stock", raw / "stock")
  stock_decode_token = (
      int(stock.get("prefill", {}).get("top1")) if stock else None)
  candidate_run, candidate, candidate_command = launch_worker(
      args, "candidate", raw / "candidate", stock_decode_token)

  comparisons = {}
  if stock and candidate:
    comparisons = {
        "prefill_logits": distribution_metrics(
            load_logits(stock, "prefill", np),
            load_logits(candidate, "prefill", np), np),
        "decode_logits": distribution_metrics(
            load_logits(stock, "decode", np),
            load_logits(candidate, "decode", np), np),
        "prefill_state_schema": state_schema_comparison(
            stock["prefill"]["states"], candidate["prefill"]["states"]),
        "decode_state_schema": state_schema_comparison(
            stock["decode"]["states"], candidate["decode"]["states"]),
        "target_states": {},
    }
    for phase in ("prefill", "decode"):
      for name in (KEY_VARIABLE, VALUE_VARIABLE):
        comparisons["target_states"][f"{phase}:{name}"] = vector_metrics(
            load_target_state(stock, phase, name, np),
            load_target_state(candidate, phase, name, np), np)

  source = candidate.get("source_summary", {})
  runtime = candidate.get("runtime", [])
  custom_runtime = [
      row for row in runtime
      if row.get("layer_type") == "CustomGPUPrimitive" and
      row.get("node_name") == f"iq36_full_attention_layer{TARGET_LAYER}"]
  stock_runtime = [
      row for row in runtime
      if row.get("layer_type") == "scaled_dot_product_attention"]
  custom_profile = [
      row for row in candidate.get("profile", [])
      if row.get("node_type") == "IQ36FullAttentionGQA"]
  component = candidate.get("component") or {}
  component_numeric = component.get("numeric", {})
  target_metrics = list(comparisons.get("target_states", {}).values())
  prefill_distribution = comparisons.get("prefill_logits", {})
  decode_distribution = comparisons.get("decode_logits", {})
  state_schemas = [
      comparisons.get("prefill_state_schema", {}),
      comparisons.get("decode_state_schema", {}),
  ]

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq812_abi_evidence_is_bound_and_passed",
            abi_metrics.get("required_checks_passed") is True,
            path=relative(abi_metrics_path),
            sha256=(sha256_file(abi_metrics_path)
                    if abi_metrics_path.is_file() else None)),
      check("custom_config_and_source_are_bound",
            args.custom_config.is_file() and args.custom_source.is_file(),
            config=relative(args.custom_config),
            config_sha256=(sha256_file(args.custom_config)
                           if args.custom_config.is_file() else None),
            source=relative(args.custom_source),
            source_sha256=(sha256_file(args.custom_source)
                           if args.custom_source.is_file() else None)),
      check("isolated_workers_complete",
            stock_run.returncode == 0 and candidate_run.returncode == 0,
            stock_returncode=stock_run.returncode,
            stock_stderr=stock_run.stderr,
            candidate_returncode=candidate_run.returncode,
            candidate_stderr=candidate_run.stderr),
      check("stock_worker_never_loads_candidate_config",
            stock.get("config_before") == "" and
            stock.get("config_after") == "",
            before=stock.get("config_before"), after=stock.get("config_after")),
      check("candidate_worker_loads_only_requested_config",
            candidate.get("config_before") == "" and
            candidate.get("config_after") == str(args.custom_config.resolve()),
            before=candidate.get("config_before"),
            after=candidate.get("config_after")),
      check("custom_operation_requires_bound_config",
            bool(candidate.get("no_config_error")),
            error=candidate.get("no_config_error")),
      check("exact_2k_prompt_and_same_teacher_forced_decode_token",
            stock.get("prompt", {}).get("token_count") ==
            EXPECTED_INPUT_TOKENS and
            candidate.get("prompt", {}).get("token_count") ==
            EXPECTED_INPUT_TOKENS and
            stock.get("prompt", {}).get("token_sha256") ==
            candidate.get("prompt", {}).get("token_sha256") and
            candidate.get("decode", {}).get("input_token") ==
            stock.get("prefill", {}).get("top1"),
            stock_prompt=stock.get("prompt"),
            candidate_prompt=candidate.get("prompt"),
            candidate_decode_input=candidate.get("decode", {}).get(
                "input_token")),
      check("source_replaces_only_real_layer3_sdpa",
            source.get("target_layer") == TARGET_LAYER and
            source.get("stock_sdpa_count_before") == EXPECTED_SDPA_COUNT and
            source.get("stock_sdpa_count_after") == EXPECTED_SDPA_COUNT - 1 and
            source.get("custom_count_after") == 1,
            summary=source),
      check("runtime_executes_one_custom_and_nine_stock_sdpa",
            len(custom_runtime) == 1 and
            len(stock_runtime) == EXPECTED_SDPA_COUNT - 1 and
            len(custom_profile) == 1 and
            custom_profile[0].get("status") == "Status.EXECUTED",
            custom_runtime=custom_runtime, stock_runtime=stock_runtime,
            custom_profile=custom_profile),
      check("seq812_decode_component_matches",
            component.get("candidate_finite") is True and
            component_numeric.get("finite") is True and
            component_numeric.get("cosine", 0.0) >= 0.999 and
            component_numeric.get("relative_l2", 1.0) <= 0.002,
            numeric=component_numeric,
            wall_ms_diagnostic=component.get("wall_ms_diagnostic")),
      check("prefill_and_decode_logits_pass_distribution_gate",
            all(row.get("finite") is True and
                row.get("kld_reference_to_candidate", 1.0) <= 0.005 and
                row.get("top1_match") is True
                for row in (prefill_distribution, decode_distribution)),
            prefill=prefill_distribution, decode=decode_distribution),
      check("exact_two_token_greedy_path_matches_stock",
            candidate.get("prefill", {}).get("top1") ==
            stock.get("prefill", {}).get("top1") and
            candidate.get("decode", {}).get("top1") ==
            stock.get("decode", {}).get("top1"),
            stock=[stock.get("prefill", {}).get("top1"),
                   stock.get("decode", {}).get("top1")],
            candidate=[candidate.get("prefill", {}).get("top1"),
                       candidate.get("decode", {}).get("top1")]),
      check("same_request_target_state_append_is_preserved",
            all(row.get("prefix_exact_bits") is True and
                row.get("appended_tokens") == 1 and
                row.get("appended_finite") is True
                for row in candidate.get("decode", {}).get(
                    "append_rows", [])) and
            len(candidate.get("decode", {}).get("append_rows", [])) == 2,
            rows=candidate.get("decode", {}).get("append_rows", [])),
      check("target_state_numeric_matches_stock",
            len(target_metrics) == 4 and
            all(row.get("finite") is True and
                row.get("cosine", 0.0) >= 0.999 and
                row.get("relative_l2", 1.0) <= 0.002
                for row in target_metrics),
            rows=comparisons.get("target_states", {})),
      check("all_80_state_schemas_remain_finite_and_compatible",
            len(state_schemas) == 2 and
            all(row.get("stock_count") == EXPECTED_STATE_COUNT and
                row.get("candidate_count") == EXPECTED_STATE_COUNT and
                row.get("names_match") is True and
                row.get("all_finite") is True and
                row.get("shape_dtype_match") is True
                for row in state_schemas),
            prefill=state_schemas[0], decode=state_schemas[1]),
  ]
  passed = all(row["pass"] for row in checks)
  created = dt.datetime.now(dt.timezone.utc).isoformat()
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": created,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
      "git": git,
      "checks": checks,
      "comparisons": comparisons,
      "stock": stock,
      "candidate": candidate,
  }
  correctness = {
      "schema": f"{SCHEMA}-correctness",
      "required_checks_passed": passed,
      "component": component_numeric,
      "prefill_logits": prefill_distribution,
      "decode_logits": decode_distribution,
      "target_states": comparisons.get("target_states", {}),
      "state_schema": state_schemas,
      "greedy_tokens": {
          "stock": [stock.get("prefill", {}).get("top1"),
                    stock.get("decode", {}).get("top1")],
          "candidate": [candidate.get("prefill", {}).get("top1"),
                        candidate.get("decode", {}).get("top1")],
      },
  }
  manifest = {
      "schema": f"{SCHEMA}-manifest",
      "workstream": WORKSTREAM,
      "created_utc": created,
      "commit": git["commit"],
      "dirty": git["dirty"],
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(args.prompt.resolve()),
      "stock_command": stock_command,
      "candidate_command": candidate_command,
      "custom_config": relative(args.custom_config),
      "custom_config_sha256": (
          sha256_file(args.custom_config)
          if args.custom_config.is_file() else None),
      "custom_source": relative(args.custom_source),
      "custom_source_sha256": (
          sha256_file(args.custom_source)
          if args.custom_source.is_file() else None),
      "abi_evidence": relative(args.abi_evidence),
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "correctness.json", correctness)
  write_json(out_dir / "manifest.json", manifest)
  status = "PASS" if passed else "FAIL"
  (out_dir / "summary.md").write_text("\n".join([
      "# OpenVINO layer-3 full-attention custom arithmetic gate", "",
      f"- status: `{status}`",
      f"- prompt/decode input: `{EXPECTED_INPUT_TOKENS} / 1`",
      f"- stock greedy: `{correctness['greedy_tokens']['stock']}`",
      f"- candidate greedy: `{correctness['greedy_tokens']['candidate']}`",
      f"- component relative L2: "
      f"`{component_numeric.get('relative_l2')}`",
      f"- prefill KLD: "
      f"`{prefill_distribution.get('kld_reference_to_candidate')}`",
      f"- decode KLD: "
      f"`{decode_distribution.get('kld_reference_to_candidate')}`",
      "- state carrier: stock F32 append state retained for arithmetic isolation",
      "- speedup claim allowed: `false`", "",
      "This gate accepts one-real-layer attention arithmetic only. Bounded "
      "hot/cold state integration, all-ten expansion, and performance remain "
      "open.", "",
  ]), encoding="utf-8")
  print(f"{status}: {out_dir}")
  if not passed:
    for row in checks:
      if not row["pass"]:
        print(f"FAIL: {row['name']}", file=sys.stderr)
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
