#!/usr/bin/env python3
"""Gate the fused OpenVINO linear-attention conv/state boundary.

Stock and candidate run in isolated processes. The stock worker executes the
exact graph sequence Transpose -> Concat -> GroupConvolution -> Slice -> SiLU
-> Transpose plus the last-four state slice. The candidate worker executes one
dynamic, two-output SimpleGPU operation over identical deterministic F16
inputs. This is a component correctness/timing gate, not a token-speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-linear-conv-custom-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
CUSTOM_SOURCE = ROOT / "engine/openvino/custom/iq36_linear_conv_swish.cl"
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
FEATURES = 8192
STATE = 4
LENGTHS = (1, 32, 1024)
REPEATS = {1: 21, 32: 11, 1024: 5}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
  parser.add_argument(
      "--candidate-gpu-plugin", type=Path, default=CANDIDATE_PLUGIN)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-linear-conv-custom-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state(out_dir: Path) -> dict[str, Any]:
  run = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=False,
      capture_output=True, text=True, encoding="utf-8", errors="replace")
  dirty = run.stdout.splitlines()
  try:
    relative_out = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    relative_out = ""
  dirty = [row for row in dirty
           if not relative_out or relative_out not in row]
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
      capture_output=True, text=True).stdout.strip()
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def build_stock_model(ov: Any, np: Any, weights_value: Any) -> Any:
  qkv = ov.opset13.parameter(
      ov.PartialShape([1, 1, -1, FEATURES]), ov.Type.f16, name="qkv")
  state = ov.opset13.parameter(
      [1, 1, FEATURES, STATE], ov.Type.f16, name="state")
  qkv3 = ov.opset13.reshape(
      qkv, ov.opset13.constant(np.array([1, -1, FEATURES], np.int64)),
      False)
  feature_major = ov.opset13.transpose(
      qkv3, ov.opset13.constant(np.array([0, 2, 1], np.int64)))
  state3 = ov.opset13.reshape(
      state, ov.opset13.constant(np.array([1, FEATURES, STATE], np.int64)),
      False)
  history = ov.opset13.concat([state3, feature_major], 2)
  # Product convolution weights are constants. Keeping them constant here is
  # semantically important: a dynamic 5-D GroupConvolution weights parameter
  # exercises a different GPU reorder path that is absent from the model.
  weights5 = ov.opset13.constant(
      weights_value.reshape(FEATURES, 1, 1, STATE))
  convolved = ov.opset13.group_convolution(
      history, weights5, [1], [0], [0], [1])
  # GroupConvolution produces T+1 rows. Stock drops row zero.
  causal = ov.opset13.slice(
      convolved,
      ov.opset13.constant(np.array([1], np.int64)),
      ov.opset13.constant(np.array([2**63 - 1], np.int64)),
      ov.opset13.constant(np.array([1], np.int64)),
      ov.opset13.constant(np.array([2], np.int64)))
  activated = ov.opset13.swish(
      causal, ov.opset13.constant(np.array(1.0, np.float16)))
  token_major = ov.opset13.transpose(
      activated, ov.opset13.constant(np.array([0, 2, 1], np.int64)))
  output = ov.opset13.reshape(
      token_major,
      ov.opset13.constant(np.array([1, 1, -1, FEATURES], np.int64)),
      False)
  next_state3 = ov.opset13.slice(
      history,
      ov.opset13.constant(np.array([-STATE], np.int64)),
      ov.opset13.constant(np.array([2**63 - 1], np.int64)),
      ov.opset13.constant(np.array([1], np.int64)),
      ov.opset13.constant(np.array([2], np.int64)))
  next_state = ov.opset13.reshape(
      next_state3,
      ov.opset13.constant(np.array([1, 1, FEATURES, STATE], np.int64)),
      False)
  return ov.Model(
      [output.output(0), next_state.output(0)], [qkv, state],
      "iq36_linear_conv_stock")


def build_candidate_model(ov: Any, weights_value: Any) -> Any:
  class IQ36LinearConvSwish(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(2)
      self.set_output_type(
          0, self.get_input_element_type(0),
          self.get_input_partial_shape(0))
      self.set_output_type(
          1, self.get_input_element_type(1),
          self.get_input_partial_shape(1))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36LinearConvSwish(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  qkv = ov.opset13.parameter(
      ov.PartialShape([1, 1, -1, FEATURES]), ov.Type.f16, name="qkv")
  state = ov.opset13.parameter(
      [1, 1, FEATURES, STATE], ov.Type.f16, name="state")
  weights = ov.opset13.constant(weights_value)
  operation = IQ36LinearConvSwish([qkv, state, weights])
  operation.set_friendly_name("iq36_linear_conv_swish")
  return ov.Model(
      [operation.output(0), operation.output(1)], [qkv, state],
      "iq36_linear_conv_candidate")


def deterministic_inputs(np: Any, length: int) -> tuple[Any, Any]:
  rng = np.random.default_rng(36035 + length)
  qkv = rng.normal(0.0, 0.55, (1, 1, length, FEATURES)).astype(np.float16)
  state = rng.normal(0.0, 0.55, (1, 1, FEATURES, STATE)).astype(np.float16)
  return qkv, state


def deterministic_weights(np: Any) -> Any:
  rng = np.random.default_rng(36035004)
  return rng.normal(
      0.0, 0.12, (1, 1, FEATURES, STATE)).astype(np.float16)


def profile_rows(request: Any) -> list[dict[str, Any]]:
  return [{
      "node_name": row.node_name,
      "node_type": row.node_type,
      "exec_type": row.exec_type,
      "status": str(row.status),
      "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
  } for row in request.get_profiling_info()]


def worker_main(config_path: Path) -> int:
  import numpy as np
  import openvino as ov

  cfg = load_json(config_path)
  mode = str(cfg["mode"])
  raw = Path(cfg["raw"])
  weights = deterministic_weights(np)
  if mode == "candidate":
    plugin = Path(cfg["candidate_gpu_plugin"])
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property(cfg["device"], {"CONFIG_FILE": cfg["custom_config"]})
    model = build_candidate_model(ov, weights)
  elif mode == "stock":
    core = ov.Core()
    model = build_stock_model(ov, np, weights)
  else:
    raise ValueError(mode)
  compiled = core.compile_model(
      model, cfg["device"], {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  request = compiled.create_infer_request()
  rows = []
  for length in LENGTHS:
    qkv, state = deterministic_inputs(np, length)
    feed = {
        compiled.input("qkv"): qkv,
        compiled.input("state"): state,
    }
    request.infer(feed, share_outputs=False)
    walls = []
    outputs = None
    for _ in range(REPEATS[length]):
      started = time.perf_counter_ns()
      outputs = request.infer(feed, share_outputs=False)
      walls.append((time.perf_counter_ns() - started) / 1_000.0)
    output0 = np.ascontiguousarray(outputs[compiled.output(0)])
    output1 = np.ascontiguousarray(outputs[compiled.output(1)])
    output0_path = raw / f"{mode}-t{length}-output.f16"
    output1_path = raw / f"{mode}-t{length}-state.f16"
    output0.tofile(output0_path)
    output1.tofile(output1_path)
    profile = profile_rows(request)
    rows.append({
        "length": length,
        "output_shape": list(output0.shape),
        "state_shape": list(output1.shape),
        "output": str(output0_path),
        "state": str(output1_path),
        "wall_us_median": statistics.median(walls),
        "wall_us_samples": walls,
        "profile_total_us": sum(
            row["real_time_us"] for row in profile
            if row["status"] == "Status.EXECUTED"),
        "profile": profile,
    })
  write_json(Path(cfg["result"]), {
      "mode": mode, "openvino_version": ov.get_version(), "rows": rows})
  return 0


def compare_arrays(np: Any, reference: Path, candidate: Path) -> dict[str, Any]:
  lhs = np.fromfile(reference, dtype="<f2").astype(np.float32)
  rhs = np.fromfile(candidate, dtype="<f2").astype(np.float32)
  delta = rhs - lhs
  denom = float(np.linalg.norm(lhs.astype(np.float64)))
  return {
      "count": int(lhs.size),
      "same_shape": bool(lhs.shape == rhs.shape),
      "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
      "exact": bool(np.array_equal(lhs, rhs)),
      "max_abs": float(np.max(np.abs(delta))),
      "relative_l2": float(
          np.linalg.norm(delta.astype(np.float64)) / max(denom, 1e-30)),
  }


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  import numpy as np

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  results: dict[str, Any] = {}
  workers = []
  for mode in ("stock", "candidate"):
    mode_raw = raw / mode
    mode_raw.mkdir()
    config = {
        "mode": mode,
        "raw": str(mode_raw),
        "result": str(mode_raw / "result.json"),
        "device": args.device,
        "custom_config": str(args.custom_config.resolve()),
        "candidate_gpu_plugin": str(args.candidate_gpu_plugin.resolve()),
    }
    config_path = mode_raw / "worker-config.json"
    write_json(config_path, config)
    command = [str(args.openvino_python), str(Path(__file__).resolve()),
               "--worker-config", str(config_path)]
    run = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=args.timeout_s)
    (mode_raw / "stdout").write_text(run.stdout, encoding="utf-8")
    (mode_raw / "stderr").write_text(run.stderr, encoding="utf-8")
    workers.append({"mode": mode, "returncode": run.returncode,
                    "stderr": run.stderr})
    results[mode] = (
        load_json(mode_raw / "result.json")
        if (mode_raw / "result.json").is_file() else {})

  comparisons = []
  for length in LENGTHS:
    stock = next((row for row in results["stock"].get("rows", [])
                  if row["length"] == length), {})
    candidate = next((row for row in results["candidate"].get("rows", [])
                      if row["length"] == length), {})
    if not stock or not candidate:
      comparisons.append({
          "length": length,
          "available": False,
          "stock_present": bool(stock),
          "candidate_present": bool(candidate),
      })
      continue
    output = compare_arrays(
        np, Path(stock["output"]), Path(candidate["output"]))
    state = compare_arrays(
        np, Path(stock["state"]), Path(candidate["state"]))
    comparisons.append({
        "length": length,
        "available": True,
        "output": output,
        "state": state,
        "stock_wall_us": stock["wall_us_median"],
        "candidate_wall_us": candidate["wall_us_median"],
        "wall_ratio": candidate["wall_us_median"] / stock["wall_us_median"],
        "stock_profile_us": stock["profile_total_us"],
        "candidate_profile_us": candidate["profile_total_us"],
    })

  git = git_state(out_dir)
  checks = [
      check("isolated_workers_complete",
            all(row["returncode"] == 0 for row in workers), workers=workers),
      check("custom_sources_bound",
            args.custom_config.is_file() and args.custom_source.is_file() and
            args.candidate_gpu_plugin.is_file(),
            config_sha256=sha256(args.custom_config),
            source_sha256=sha256(args.custom_source),
            plugin_sha256=sha256(args.candidate_gpu_plugin)),
      check("component_outputs_match_stock",
            len(comparisons) == len(LENGTHS) and
            all(row.get("available", False) and
                row["output"]["finite"] and
                row["output"]["exact"]
                for row in comparisons), comparisons=comparisons),
      check("conv_state_is_bit_exact",
            len(comparisons) == len(LENGTHS) and
            all(row.get("available", False) and row["state"]["exact"]
                for row in comparisons),
            comparisons=comparisons),
      check("candidate_moves_both_decode_and_prefill_component",
            len(comparisons) == len(LENGTHS) and
            comparisons[0].get("available", False) and
            comparisons[-1].get("available", False) and
            comparisons[0]["wall_ratio"] < 0.8 and
            comparisons[-1]["wall_ratio"] < 0.8,
            comparisons=comparisons),
      check("repository_clean_at_gate", not git["dirty"], git=git),
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": passed,
      "checks": checks,
      "workers": results,
      "comparisons": comparisons,
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "manifest.json", {
      "schema": SCHEMA, "workstream": WORKSTREAM,
      "required_checks_passed": passed, "metrics": "metrics.json"})
  print(json.dumps({
      "out_dir": str(out_dir), "required_checks_passed": passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
      "comparisons": comparisons,
  }, indent=2))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
