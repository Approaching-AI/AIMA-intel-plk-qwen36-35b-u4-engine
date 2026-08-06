#!/usr/bin/env python3
"""Gate multi-output SimpleGPU support in the pinned OpenVINO GPU runtime.

The legacy static-shape path rejects custom operations with more than one
output.  A dynamic operation selects the new shape-inference path, where the
GPU plugin can bind multiple output ports.  This gate locks that distinction
on the target machine before the full-attention state carrier depends on it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-multi-output-custom-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CUSTOM_CONFIG = (
    ROOT / "engine/openvino/custom/iq36_multi_output_probe.xml")
CUSTOM_SOURCE = (
    ROOT / "engine/openvino/custom/iq36_multi_output_probe.cl")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
PROBE_LENGTHS = (16, 37)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument("--custom-source", type=Path, default=CUSTOM_SOURCE)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None and args.worker_result is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-multi-output-custom-{stamp}"
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


def any_value(value: Any) -> Any:
  try:
    return value.value
  except Exception:
    return str(value)


def worker_main(args: argparse.Namespace) -> int:
  import numpy as np
  import openvino as ov

  if Path(sys.prefix).resolve() != args.openvino_python.parent.parent.resolve():
    raise RuntimeError(
        f"worker requires {args.openvino_python}, observed {sys.executable}")

  class IQ36MultiOutputProbe(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      shape = self.get_input_partial_shape(0)
      self.set_output_size(2)
      self.set_output_type(
          0, self.get_input_element_type(0), shape)
      self.set_output_type(
          1, self.get_input_element_type(0),
          ov.PartialShape([shape[0], shape[1], shape[3], shape[2]]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36MultiOutputProbe(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  def make_model(dynamic: bool) -> Any:
    shape = ov.PartialShape([1, 1, 1, -1 if dynamic else 16])
    value = ov.opset13.parameter(shape, ov.Type.f32, name="input")
    # The pinned plugin's output-port validation compares the output index to
    # input count.  A real attention op has many inputs; this second probe input
    # locks the same valid binding shape for output port one.
    dummy = ov.opset13.parameter(shape, ov.Type.f32, name="dummy")
    operation = IQ36MultiOutputProbe([value.output(0), dummy.output(0)])
    operation.set_friendly_name("iq36_multi_output_probe")
    return ov.Model(
        [operation.output(0), operation.output(1)], [value, dummy],
        f"iq36_multi_output_probe_{'dynamic' if dynamic else 'static'}")

  no_config_error = ""
  try:
    ov.Core().compile_model(make_model(True), args.device)
  except Exception as exc:
    no_config_error = repr(exc)

  core = ov.Core()
  config_before = str(core.get_property(args.device, "CONFIG_FILE"))
  core.set_property(
      args.device, {"CONFIG_FILE": str(args.custom_config.resolve())})
  config_after = str(core.get_property(args.device, "CONFIG_FILE"))

  static_error = ""
  try:
    core.compile_model(make_model(False), args.device)
  except Exception as exc:
    static_error = repr(exc)

  compiled = core.compile_model(
      make_model(True), args.device,
      {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  request = compiled.create_infer_request()
  probes = []
  for length in PROBE_LENGTHS:
    source = np.arange(length, dtype=np.float32).reshape(1, 1, 1, length)
    outputs = request.infer({
        compiled.input("input"): source,
        compiled.input("dummy"): source,
    }, share_outputs=False)
    copied = np.asarray(outputs[compiled.output(0)])
    doubled = np.asarray(outputs[compiled.output(1)])
    probes.append({
        "length": length,
        "input_shape": list(source.shape),
        "output_shapes": [list(copied.shape), list(doubled.shape)],
        "output_dtypes": [str(copied.dtype), str(doubled.dtype)],
        "copied_exact": bool(np.array_equal(copied, source)),
        "doubled_exact": bool(np.array_equal(
            doubled.reshape(-1), (source * 2).reshape(-1))),
    })

  profile = []
  for row in request.get_profiling_info():
    if (row.node_type != "IQ36MultiOutputProbe" and
        "iq36_multi_output_probe" not in row.node_name.lower()):
      continue
    profile.append({
        "node_name": row.node_name,
        "node_type": row.node_type,
        "exec_type": row.exec_type,
        "status": str(row.status),
        "real_time_us": row.real_time.total_seconds() * 1_000_000.0,
    })

  runtime = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): any_value(value)
            for key, value in node.get_rt_info().items()}
    if str(info.get("layerType")) != "CustomGPUPrimitive":
      continue
    runtime.append({
        "node_name": node.get_friendly_name(),
        "layer_type": str(info.get("layerType")),
        "primitive_type": str(info.get("primitiveType")),
        "runtime_precision": str(info.get("runtimePrecision")),
        "output_layouts": str(info.get("outputLayouts")),
        "output_precisions": str(info.get("outputPrecisions")),
    })

  write_json(args.worker_result, {
      "config_after": config_after,
      "config_before": config_before,
      "dynamic_model": {
          "input_count": 2,
          "output_count": len(compiled.outputs),
          "probes": probes,
          "profile": profile,
          "runtime": runtime,
      },
      "no_config_error": no_config_error,
      "openvino_version": ov.get_version(),
      "static_model_error": static_error,
  })
  return 0


def main() -> int:
  args = parse_args()
  if args.worker_result is not None:
    return worker_main(args)

  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  worker_result_path = raw / "worker-result.json"
  command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-result", str(worker_result_path),
      "--device", args.device,
      "--openvino-python", str(args.openvino_python),
      "--custom-config", str(args.custom_config.resolve()),
      "--custom-source", str(args.custom_source.resolve()),
  ]
  worker = subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=args.timeout_s)
  (raw / "worker.stdout").write_text(worker.stdout, encoding="utf-8")
  (raw / "worker.stderr").write_text(worker.stderr, encoding="utf-8")
  write_json(raw / "worker-command.json", {
      "command": command,
      "returncode": worker.returncode,
  })
  result = (
      load_json(worker_result_path) if worker_result_path.is_file() else {})
  dynamic = result.get("dynamic_model", {})
  probes = dynamic.get("probes", [])
  executed = [row for row in dynamic.get("profile", [])
              if row.get("node_type") == "IQ36MultiOutputProbe" and
              row.get("status") == "Status.EXECUTED"]
  runtime = dynamic.get("runtime", [])
  static_error = str(result.get("static_model_error", ""))
  checks = [
      check("worker_completed", worker.returncode == 0,
            returncode=worker.returncode, stderr=worker.stderr),
      check("pinned_openvino_source_commit_exists",
            OV_SOURCE.is_dir() and subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=OV_SOURCE, check=False,
                capture_output=True, text=True).stdout.strip() == OV_COMMIT,
            source=str(OV_SOURCE), expected_commit=OV_COMMIT),
      check("custom_config_and_source_are_bound",
            args.custom_config.is_file() and args.custom_source.is_file(),
            config=str(args.custom_config.resolve()),
            config_sha256=(sha256(args.custom_config)
                           if args.custom_config.is_file() else None),
            source=str(args.custom_source.resolve()),
            source_sha256=(sha256(args.custom_source)
                           if args.custom_source.is_file() else None)),
      check("worker_starts_without_custom_config",
            result.get("config_before") == "",
            observed=result.get("config_before")),
      check("worker_loads_only_probe_config",
            result.get("config_after") == str(args.custom_config.resolve()),
            observed=result.get("config_after")),
      check("custom_operation_requires_bound_config",
            bool(result.get("no_config_error")),
            error=result.get("no_config_error")),
      check("static_multi_output_path_rejects_by_documented_limit",
            "static model only support one output" in static_error,
            error=static_error),
      check("dynamic_custom_model_exposes_two_outputs",
            dynamic.get("input_count") == 2 and
            dynamic.get("output_count") == 2,
            input_count=dynamic.get("input_count"),
            output_count=dynamic.get("output_count")),
      check("same_request_reallocates_both_dynamic_outputs_exactly",
            [row.get("length") for row in probes] == list(PROBE_LENGTHS) and
            all(row.get("output_shapes") == [
                    [1, 1, 1, row["length"]],
                    [1, 1, row["length"], 1]] and
                row.get("copied_exact") and row.get("doubled_exact")
                for row in probes), probes=probes),
      check("multi_output_custom_kernel_executes",
            len(executed) == 1 and len(runtime) == 1,
            profile=dynamic.get("profile", []), runtime=runtime),
  ]
  git = git_state(out_dir)
  checks.append(check(
      "clean_commit_for_promotion", not git["dirty"], git=git))
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "required_checks_passed": passed,
      "git": git,
      "openvino_source": {"path": str(OV_SOURCE), "commit": OV_COMMIT},
      "worker": result,
      "checks": checks,
      "conclusion": (
          "The pinned GPU runtime supports distinct dynamic outputs from one "
          "SimpleGPU node; the one-output assumption applies only to the "
          "legacy static-shape path."
          if passed else
          "The pinned runtime has not yet proven the dynamic multi-output "
          "SimpleGPU boundary."),
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "manifest.json", {
      "schema": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": passed,
      "metrics": "metrics.json",
      "raw": "raw/",
  })
  print(json.dumps({
      "out_dir": str(out_dir),
      "required_checks_passed": passed,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
  }, indent=2))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
