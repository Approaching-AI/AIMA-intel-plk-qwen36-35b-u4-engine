#!/usr/bin/env python3
"""Capture one real seq1024 stock-OpenVINO GatedDeltaNet boundary.

The raw IR represents every linear-attention recurrence as an OpenVINO Loop.
The stock GPU plugin recognizes that Loop and lowers it to the internal
``ocl::gated_delta_net::ref___f16`` primitive.  This gate audits layer 0's
actual OpenCL arguments without adding graph consumers to semantic inputs,
exposes only the two proven-safe Loop outputs as Results, and cross-checks the
observer graph against the untouched model at logits and all 80 state tensors.

The resulting tensors are the OV1 component oracle.  This gate neither loads a
candidate CONFIG_FILE nor makes a performance claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-gdn-boundary-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
PROMPT = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_008k.txt")
LOOP_NAME = "Loop_1520"
SEQ_LEN = 1024
EXPECTED_PROMPT_TOKENS = 8192
EXPECTED_GDN_COUNT = 30
EXPECTED_STATE_COUNT = 80
EXPECTED_PRIMITIVE = "ocl::gated_delta_net::ref___f16"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
COMPILE_CONFIG = {
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
    "PERFORMANCE_HINT": "LATENCY",
    "PERF_COUNT": True,
}
CAPTURE_BOUNDARIES = (
    # q/k/v are three views into one token-major allocation.  Scalars 9..11
    # select head offsets 0/16/32; scalars 12..14 lock the 8192-value stride.
    ("qkv", "before", 0, (1, SEQ_LEN, 64, 128)),
    ("initial_state", "before", 3, (1, 32, 128, 128)),
    ("gate", "before", 4, (1, SEQ_LEN, 32)),
    ("beta", "before", 5, (1, SEQ_LEN, 32)),
    ("attention_output", "after", 6, (1, SEQ_LEN, 32, 128)),
    ("final_state", "after", 7, (1, 32, 128, 128)),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--prompt", type=Path, default=PROMPT)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--worker-config", type=Path,
                      help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-gdn-boundary-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*args: str) -> str:
    run = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative = ""
  dirty = [line for line in dirty if not relative or relative not in line]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


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
      "pass": (
          configure.returncode == 0 and build.returncode == 0 and
          TRACE_LIBRARY.is_file()),
  }
  write_json(raw / "trace-build.json", result)
  return result


def locked_file_rows(model_dir: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
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
        "pass": (
            exists and size == expected.get("bytes") and
            digest == expected.get("sha256")),
    })
  return rows


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def runtime_gdn_rows(compiled: Any) -> list[dict[str, Any]]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    if str(info.get("layerType")) != "GatedDeltaNet":
      continue
    rows.append({
        "node_name": node.get_friendly_name(),
        "primitive_type": str(info.get("primitiveType")),
        "runtime_precision": str(info.get("runtimePrecision")),
        "output_layouts": str(info.get("outputLayouts")),
        "output_precisions": str(info.get("outputPrecisions")),
    })
  return rows


def profile_rows(request: Any) -> list[dict[str, Any]]:
  rows = []
  for row in request.get_profiling_info():
    if row.node_type != "GatedDeltaNet":
      continue
    rows.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })
  return rows


def make_observer_model(core: Any, model_path: Path, ov: Any) -> tuple[Any, list[str]]:
  model = core.read_model(str(model_path))
  loop = next(
      (node for node in model.get_ordered_ops()
       if node.get_friendly_name() == LOOP_NAME), None)
  if loop is None or loop.get_type_name() != "Loop":
    raise RuntimeError(f"stock graph does not contain {LOOP_NAME} Loop")
  names = []
  # Directly observing any of the six semantic inputs changes GPU precision /
  # liveness decisions and is therefore not an oracle.  The dispatch audit
  # copies those buffers without adding graph consumers.  The two Loop outputs
  # are safe observers and cross-check the audited internal output layout.
  for label, index in (("attention_output", 0), ("final_state", 1)):
    value = loop.output(index)
    name = f"iq36_ov1_layer0_{label}"
    value.get_tensor().add_names({name})
    result = ov.opset13.result(value)
    result.set_friendly_name(name + "_result")
    model.add_results([result])
    names.append(name)
  model.validate_nodes_and_infer_types()
  return model, names


def make_inputs(
    embedding: Any, token_ids: Any, np: Any,
) -> dict[str, Any]:
  ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
  embedded = np.asarray(
      embedding({embedding.input(0): ids})[embedding.output(0)])
  embedded = embedded.astype(np.float32, copy=False)
  positions = np.arange(ids.shape[1], dtype=np.int64)
  return {
      "attention_mask": np.ones((1, ids.shape[1]), dtype=np.int64),
      "beam_idx": np.zeros((1,), dtype=np.int32),
      "inputs_embeds": embedded,
      "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, -1),
  }


def state_snapshot(request: Any, np: Any) -> dict[str, Any]:
  result = {}
  for state in request.query_state():
    result[str(state.name)] = np.array(state.state.data, copy=True)
  return result


def run_graph(
    compiled: Any, inputs: dict[str, Any], np: Any,
    observer_names: list[str] | None = None,
) -> dict[str, Any]:
  request = compiled.create_infer_request()
  request.reset_state()
  started = time.perf_counter_ns()
  outputs = request.infer(inputs, share_outputs=False)
  wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  logits_value = np.asarray(outputs[compiled.output(0)], dtype=np.float32)
  logits = np.array(logits_value[0, -1], dtype=np.float32, copy=True)
  boundaries = {}
  for name in observer_names or []:
    boundaries[name] = np.array(outputs[compiled.output(name)], copy=True)
  return {
      "boundaries": boundaries,
      "gdn_profile": profile_rows(request),
      "logits": logits,
      "states": state_snapshot(request, np),
      "wall_ms_diagnostic": wall_ms,
  }


def vector_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  diff = cand - ref
  ref_norm = float(np.linalg.norm(ref))
  cand_norm = float(np.linalg.norm(cand))
  denom = ref_norm * cand_norm
  return {
      "count": int(ref.size),
      "finite": bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
      "exact_bits": bool(np.array_equal(reference, candidate)),
      "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
      "relative_l2": (
          float(np.linalg.norm(diff) / ref_norm)
          if ref_norm else float(np.linalg.norm(diff))),
      "cosine": float(np.dot(ref, cand) / denom) if denom else 1.0,
  }


def distribution_metrics(reference: Any, candidate: Any, np: Any) -> dict[str, Any]:
  ref = np.asarray(reference, dtype=np.float64).reshape(-1)
  cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
  ref_shifted = ref - float(np.max(ref))
  cand_shifted = cand - float(np.max(cand))
  ref_prob = np.exp(ref_shifted)
  cand_prob = np.exp(cand_shifted)
  ref_prob /= float(ref_prob.sum())
  cand_prob /= float(cand_prob.sum())
  eps = np.finfo(np.float64).tiny
  numeric = vector_metrics(reference, candidate, np)
  return {
      **numeric,
      "kld_reference_to_observer": float(np.sum(
          ref_prob * (np.log(np.maximum(ref_prob, eps)) -
                      np.log(np.maximum(cand_prob, eps))))),
      "reference_top1": int(np.argmax(ref)),
      "observer_top1": int(np.argmax(cand)),
      "top1_match": bool(int(np.argmax(ref)) == int(np.argmax(cand))),
  }


def compare_states(reference: dict[str, Any], observer: dict[str, Any], np: Any) -> dict[str, Any]:
  names_match = set(reference) == set(observer)
  rows = []
  for name in sorted(set(reference) & set(observer)):
    metrics = vector_metrics(reference[name], observer[name], np)
    rows.append({
        "name": name,
        "shape": list(reference[name].shape),
        "dtype": str(reference[name].dtype),
        "reference_sha256": sha256_array(reference[name], np),
        "observer_sha256": sha256_array(observer[name], np),
        **metrics,
    })
  return {
      "names_match": names_match,
      "reference_count": len(reference),
      "observer_count": len(observer),
      "all_exact_bits": names_match and all(row["exact_bits"] for row in rows),
      "max_abs": max((row["max_abs"] for row in rows), default=math.inf),
      "max_relative_l2": max(
          (row["relative_l2"] for row in rows), default=math.inf),
      "min_cosine": min((row["cosine"] for row in rows), default=-math.inf),
      "rows": rows,
  }


def captured_boundary_rows(
    capture_dir: Path, np: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  rows = []
  values = {}
  for label, phase, index, expected_shape in CAPTURE_BOUNDARIES:
    payload = capture_dir / f"dispatch000-arg{index}-{phase}.bin"
    value = np.fromfile(payload, dtype="<f2") if payload.is_file() else np.array([])
    expected_count = math.prod(expected_shape)
    shaped = value.reshape(expected_shape) if value.size == expected_count else value
    values[label] = shaped
    rows.append({
        "label": label,
        "phase": phase,
        "kernel_arg": index,
        "shape": list(shaped.shape),
        "expected_shape": list(expected_shape),
        "dtype": str(shaped.dtype),
        "count": int(value.size),
        "expected_count": expected_count,
        "byte_count": payload.stat().st_size if payload.is_file() else None,
        "finite": bool(value.size and np.isfinite(value).all()),
        "all_zero": bool(value.size and np.count_nonzero(value) == 0),
        "minimum": float(np.min(value)) if value.size else None,
        "maximum": float(np.max(value)) if value.size else None,
        "l2_norm": (
            float(np.linalg.norm(value.astype(np.float64)))
            if value.size else None),
        "sha256": sha256_file(payload) if payload.is_file() else None,
        "payload": relative(payload),
    })
  return rows, values


def internal_output_crosscheck(
    captured: dict[str, Any], observed: dict[str, Any], np: Any,
) -> dict[str, Any]:
  internal_attention = np.asarray(captured["attention_output"]).transpose(
      0, 2, 1, 3).astype(np.float32)
  internal_state = np.asarray(captured["final_state"]).astype(np.float32)
  observed_attention = observed["iq36_ov1_layer0_attention_output"]
  observed_state = observed["iq36_ov1_layer0_final_state"]
  return {
      "attention": vector_metrics(
          observed_attention, internal_attention, np),
      "state": vector_metrics(observed_state, internal_state, np),
      "observed_attention_shape": list(observed_attention.shape),
      "internal_attention_layout": "[batch,token,value_head,value_dim]",
      "observed_attention_layout": "[batch,value_head,token,value_dim]",
  }


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise SystemExit(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  cfg = load_json(config_path)
  model_dir = Path(cfg["model_dir"])
  prompt = Path(cfg["prompt"])
  capture_dir = Path(cfg["capture_dir"])
  marker = Path(cfg["marker"])
  trace_path = Path(cfg["trace_path"])
  result_path = Path(cfg["result_path"])
  device = str(cfg["device"])

  core = ov.Core()
  custom_config = core.get_property(device, "CONFIG_FILE")
  embedding = core.compile_model(
      core.read_model(str(model_dir / "openvino_text_embeddings_model.xml")),
      "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_ids = np.asarray(
      tokenizer.encode(prompt.read_text(encoding="utf-8")).input_ids.data
  ).reshape(-1).astype(np.int64)
  tile_ids = prompt_ids[:SEQ_LEN]
  inputs = make_inputs(embedding, tile_ids, np)
  token_payload = Path(cfg["token_payload"])
  np.ascontiguousarray(tile_ids, dtype="<u4").tofile(token_payload)

  untouched_source = core.read_model(
      str(model_dir / "openvino_language_model.xml"))
  started = time.perf_counter_ns()
  untouched_compiled = core.compile_model(
      untouched_source, device, COMPILE_CONFIG)
  untouched_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  untouched_runtime = runtime_gdn_rows(untouched_compiled)
  untouched = run_graph(untouched_compiled, inputs, np)
  del untouched_compiled, untouched_source
  gc.collect()

  observer_source, observer_names = make_observer_model(
      core, model_dir / "openvino_language_model.xml", ov)
  started = time.perf_counter_ns()
  observer_compiled = core.compile_model(
      observer_source, device, COMPILE_CONFIG)
  observer_compile_ms = (time.perf_counter_ns() - started) / 1_000_000.0
  observer_runtime = runtime_gdn_rows(observer_compiled)
  marker.write_text("seq1024_layer0\n", encoding="utf-8")
  try:
    observer = run_graph(observer_compiled, inputs, np, observer_names)
  finally:
    marker.unlink(missing_ok=True)

  trace = load_jsonl(trace_path)
  dispatches = [row for row in trace if row.get("event") == "ndrange"]
  captures = [row for row in trace if row.get("event") == "capture"]
  first_dispatch = dispatches[0] if dispatches else {}
  scalar_args = {}
  for row in first_dispatch.get("args", []):
    index = int(row.get("index", -1))
    if index < 8 or row.get("size") != 4:
      continue
    scalar_args[str(index)] = int.from_bytes(
        bytes.fromhex(str(row.get("hex", ""))), "little", signed=True)

  boundaries, captured = captured_boundary_rows(capture_dir, np)
  crosscheck = internal_output_crosscheck(
      captured, observer["boundaries"], np)
  logits = distribution_metrics(untouched["logits"], observer["logits"], np)
  state = compare_states(untouched["states"], observer["states"], np)
  untouched_layer0_profile = [
      row for row in untouched["gdn_profile"] if row["node_name"] == LOOP_NAME]
  observer_layer0_profile = [
      row for row in observer["gdn_profile"] if row["node_name"] == LOOP_NAME]
  runtime_pass = (
      len(untouched_runtime) == EXPECTED_GDN_COUNT and
      len(observer_runtime) == EXPECTED_GDN_COUNT and
      all(row["primitive_type"] == EXPECTED_PRIMITIVE
          for row in untouched_runtime + observer_runtime) and
      any(row["node_name"] == LOOP_NAME for row in untouched_runtime) and
      any(row["node_name"] == LOOP_NAME for row in observer_runtime))
  profile_pass = (
      len(untouched["gdn_profile"]) == EXPECTED_GDN_COUNT and
      len(observer["gdn_profile"]) == EXPECTED_GDN_COUNT and
      len(untouched_layer0_profile) == 1 and
      len(observer_layer0_profile) == 1 and
      all(row["exec_type"] == EXPECTED_PRIMITIVE and
          row["status"] == "Status.EXECUTED"
          for row in untouched["gdn_profile"] + observer["gdn_profile"]))
  boundary_pass = all(
      row["shape"] == row["expected_shape"] and
      row["dtype"] == "float16" and row["finite"] and
      row["count"] == row["expected_count"] and
      row["byte_count"] == row["count"] * 2
      for row in boundaries)
  alias_paths = [
      capture_dir / f"dispatch000-arg{index}-before.bin"
      for index in (0, 1, 2)]
  alias_hashes = [sha256_file(path) if path.is_file() else None
                  for path in alias_paths]
  expected_scalars = {
      "8": SEQ_LEN, "9": 0, "10": 16, "11": 32,
      "12": 8192, "13": 8192, "14": 8192,
  }
  worker_checks = [
      check("stock_worker_has_no_candidate_custom_config",
            str(custom_config) == "", observed=str(custom_config)),
      check("exact_sentinel_8k_first_tile",
            len(prompt_ids) == EXPECTED_PROMPT_TOKENS and
            len(tile_ids) == SEQ_LEN,
            prompt_tokens=int(len(prompt_ids)), tile_tokens=int(len(tile_ids)),
            tile_sha256=sha256_file(token_payload)),
      check("safe_output_observer_preserves_all_stock_gdn_fusions",
            runtime_pass, untouched=untouched_runtime,
            observer=observer_runtime),
      check("stock_layer0_gdn_executes_in_both_graphs", profile_pass,
            untouched_layer0=untouched_layer0_profile,
            observer_layer0=observer_layer0_profile),
      check("dispatch_audit_captures_first_real_gdn_without_graph_input_observers",
            len(dispatches) == EXPECTED_GDN_COUNT and len(captures) == 8 and
            all(row.get("status") == 0 for row in captures),
            dispatch_count=len(dispatches), captures=captures),
      check("gdn_argument_alias_offsets_and_strides_locked",
            len(set(alias_hashes)) == 1 and None not in alias_hashes and
            scalar_args == expected_scalars,
            qkv_alias_sha256=alias_hashes, scalar_args=scalar_args,
            expected_scalars=expected_scalars),
      check("all_internal_f16_boundary_shapes_and_payloads_pass",
            boundary_pass, rows=boundaries),
      check("initial_state_is_zero",
            next(row for row in boundaries
                 if row["label"] == "initial_state")["all_zero"]),
      check("audited_internal_outputs_match_safe_graph_outputs_exactly",
            crosscheck["attention"]["exact_bits"] and
            crosscheck["state"]["exact_bits"], crosscheck=crosscheck),
      check("safe_output_observer_logits_match_untouched_stock",
            logits["finite"] and logits["kld_reference_to_observer"] <= 0.005 and
            logits["cosine"] >= 0.999 and logits["top1_match"],
            metrics=logits),
      check("safe_output_observer_all_80_states_match_untouched_stock",
            state["names_match"] and
            state["reference_count"] == EXPECTED_STATE_COUNT and
            state["observer_count"] == EXPECTED_STATE_COUNT and
            state["max_relative_l2"] <= 0.002 and
            state["min_cosine"] >= 0.999,
            summary={key: value for key, value in state.items()
                     if key != "rows"}),
  ]
  passed = all(row["pass"] for row in worker_checks)
  result = {
      "worker_checks": worker_checks,
      "worker_checks_passed": passed,
      "boundaries": boundaries,
      "crosscheck": crosscheck,
      "logits_comparison": logits,
      "state_comparison": state,
      "untouched_runtime": untouched_runtime,
      "observer_runtime": observer_runtime,
      "untouched_profile": untouched["gdn_profile"],
      "observer_profile": observer["gdn_profile"],
      "dispatch": first_dispatch,
      "capture_rows": captures,
      "scalar_args": scalar_args,
      "untouched_compile_ms": untouched_compile_ms,
      "observer_compile_ms": observer_compile_ms,
      "untouched_wall_ms_diagnostic": untouched["wall_ms_diagnostic"],
      "observer_wall_ms_diagnostic": observer["wall_ms_diagnostic"],
      "openvino_runtime_version": ov.get_version(),
      "openvino_genai_version": ov_genai.__version__,
      "custom_config": str(custom_config),
      "prompt_tokens": int(len(prompt_ids)),
      "tile_tokens": int(len(tile_ids)),
      "token_payload": relative(token_payload),
  }
  write_json(result_path, result)
  print(json.dumps({
      "event": "worker_complete", "worker_checks_passed": passed,
      "boundary_count": len(boundaries),
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config.resolve())

  out = args.out_dir.resolve()
  raw = out / "raw"
  capture_dir = raw / "boundary"
  cache_dir = raw / "neo-cache"
  capture_dir.mkdir(parents=True, exist_ok=False)
  cache_dir.mkdir()
  required = [args.model_contract, args.prompt, CMAKE, OV_PYTHON,
              args.model_dir / "openvino_language_model.xml",
              args.model_dir / "openvino_text_embeddings_model.xml"]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = dt.datetime.now(dt.timezone.utc).isoformat()
  git = git_state(out)
  contract = load_json(args.model_contract)
  locked_files = locked_file_rows(args.model_dir, contract)
  contract_sha = sha256_file(args.model_contract)
  prompt_sha = sha256_file(args.prompt)
  trace_build = build_trace(raw, args.timeout_s)
  marker = raw / "trace-active"
  trace_path = raw / "dispatch-trace.jsonl"
  worker_result_path = raw / "worker-result.json"
  token_payload = raw / "sentinel-008k-first-1024.u32"
  worker_config = {
      "model_dir": str(args.model_dir.resolve()),
      "prompt": str(args.prompt.resolve()),
      "device": args.device,
      "capture_dir": str(capture_dir),
      "marker": str(marker),
      "trace_path": str(trace_path),
      "result_path": str(worker_result_path),
      "token_payload": str(token_payload),
  }
  worker_config_path = raw / "worker-config.json"
  write_json(worker_config_path, worker_config)
  worker_command = [
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(worker_config_path),
  ]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_TRACE_FILTER": "gated_delta_net_ref",
      "IQ36_OPENCL_TRACE_MARKER": str(marker),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path),
      "IQ36_OPENCL_TRACE_TIMING": "0",
      "IQ36_OPENCL_CAPTURE_DIR": str(capture_dir),
      "LD_AUDIT": str(TRACE_LIBRARY),
      "NEO_CACHE_DIR": str(cache_dir),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if trace_build["pass"]:
    try:
      worker = subprocess.run(
          worker_command, cwd=ROOT, env=env, check=False,
          capture_output=True, text=True, encoding="utf-8",
          errors="replace", timeout=args.timeout_s)
    except subprocess.TimeoutExpired as exc:
      worker = subprocess.CompletedProcess(
          worker_command, 124, str(exc.stdout or ""), str(exc.stderr or ""))
  else:
    worker = subprocess.CompletedProcess(
        worker_command, 125, "", "trace build failed")
  (raw / "worker.stdout").write_text(worker.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(worker.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": worker_command,
      "environment": {key: env[key] for key in (
          "IQ36_OPENCL_TRACE_FILTER", "IQ36_OPENCL_TRACE_MARKER",
          "IQ36_OPENCL_TRACE_PATH", "IQ36_OPENCL_TRACE_TIMING",
          "IQ36_OPENCL_CAPTURE_DIR", "LD_AUDIT", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": worker.returncode,
  })
  worker_result = (
      load_json(worker_result_path) if worker_result_path.is_file() else {})
  runtime_contract = contract["runtime_contract"]["baseline"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_locked_model_files_match_contract",
            bool(locked_files) and all(row["pass"] for row in locked_files),
            rows=locked_files),
      check("dispatch_audit_builds", bool(trace_build["pass"])),
      check("isolated_stock_worker_completes", worker.returncode == 0,
            returncode=worker.returncode, stderr=worker.stderr[-2000:]),
      check("runtime_versions_match_contract",
            worker_result.get("openvino_runtime_version") ==
            runtime_contract["openvino_runtime_version"] and
            worker_result.get("openvino_genai_version") ==
            runtime_contract["openvino_genai_version"],
            openvino_runtime=worker_result.get("openvino_runtime_version"),
            openvino_genai=worker_result.get("openvino_genai_version")),
      check("worker_component_oracle_checks_pass",
            worker_result.get("worker_checks_passed") is True,
            failed=[row.get("name") for row in
                    worker_result.get("worker_checks", [])
                    if not row.get("pass")]),
  ]
  required_passed = all(row["pass"] for row in checks)
  correctness = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_passed,
      "checks": checks,
      "worker_checks": worker_result.get("worker_checks", []),
      "logits_comparison": worker_result.get("logits_comparison"),
      "state_comparison": worker_result.get("state_comparison"),
      "internal_output_crosscheck": worker_result.get("crosscheck"),
      "claim_boundary": "one-real-layer stock component oracle only",
      "product_speedup_claim": False,
  }
  write_json(out / "correctness.json", correctness)
  boundaries = worker_result.get("boundaries", [])
  write_jsonl(out / "metrics.jsonl", [
      {"metric_scope": "boundary", **row} for row in boundaries
  ] + [
      {"metric_scope": "profile", "graph": graph, **row}
      for graph, profile in (
          ("untouched", worker_result.get("untouched_profile", [])),
          ("observer", worker_result.get("observer_profile", [])))
      for row in profile
  ])
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": relative(Path(__file__)),
      "command": sys.argv,
      "git": git,
      "model_contract": relative(args.model_contract),
      "model_contract_sha256": contract_sha,
      "model_dir": str(args.model_dir),
      "locked_files": locked_files,
      "prompt": relative(args.prompt),
      "prompt_sha256": prompt_sha,
      "token_payload": relative(token_payload),
      "compile_config": COMPILE_CONFIG,
      "trace_library": relative(TRACE_LIBRARY),
      "worker_command": worker_command,
      "worker_returncode": worker.returncode,
      "untouched_compile_ms": worker_result.get("untouched_compile_ms"),
      "observer_compile_ms": worker_result.get("observer_compile_ms"),
      "untouched_wall_ms_diagnostic": worker_result.get(
          "untouched_wall_ms_diagnostic"),
      "observer_wall_ms_diagnostic": worker_result.get(
          "observer_wall_ms_diagnostic"),
      "openvino_runtime_version": worker_result.get(
          "openvino_runtime_version"),
      "openvino_genai_version": worker_result.get(
          "openvino_genai_version"),
      "required_checks_passed": required_passed,
      "product_speedup_claim": False,
  })
  failed = [row["name"] for row in checks if not row["pass"]]
  logits = worker_result.get("logits_comparison", {})
  state = worker_result.get("state_comparison", {})
  crosscheck = worker_result.get("crosscheck", {})
  (out / "summary.md").write_text("\n".join([
      "# OpenVINO GatedDeltaNet real-boundary gate", "",
      f"- required checks: **{'PASS' if required_passed else 'FAIL'}**",
      f"- layer: `{LOOP_NAME}` / model layer 0 / sequence `{SEQ_LEN}`",
      f"- stock primitive: `{EXPECTED_PRIMITIVE}`",
      f"- audited internal F16 tensors: `{len(boundaries)}`",
      f"- logits KLD / cosine / top-1: "
      f"`{logits.get('kld_reference_to_observer')}` / "
      f"`{logits.get('cosine')}` / `{logits.get('top1_match')}`",
      f"- state tensors / max relative L2 / minimum cosine: "
      f"`{state.get('reference_count')}` / "
      f"`{state.get('max_relative_l2')}` / `{state.get('min_cosine')}`",
      f"- internal output/state exact cross-check: "
      f"`{crosscheck.get('attention', {}).get('exact_bits')}` / "
      f"`{crosscheck.get('state', {}).get('exact_bits')}`",
      f"- failed checks: `{failed}`", "",
      "This artifact is the one-real-layer stock component oracle for OV1. ",
      "The dispatch audit adds no semantic-input graph consumer; its single ",
      "sequential wall rows are diagnostics, not a speed claim.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_passed,
      "boundary_count": len(boundaries),
      "gdn_count": len(worker_result.get("observer_runtime", [])),
      "logits_kld": logits.get("kld_reference_to_observer"),
      "state_count": state.get("reference_count"),
      "failed_checks": failed,
      "out_dir": relative(out),
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
