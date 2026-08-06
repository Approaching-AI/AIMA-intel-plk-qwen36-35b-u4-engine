#!/usr/bin/env python3
"""Reconcile CPU/GPU/NPU product architectures before another native kernel.

This is ADR 0012's bounded product-level gate.  OpenVINO is used only as a
compiler/hardware probe.  A passing result may select one repository-owned
native architecture for a later exact component gate; it is never a product
speed or correctness claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-product-architecture-feasibility-v1"
DEFAULT_OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
DEFAULT_OPENVINO_MODEL = Path("/home/intel/Qwen3.6-35B-A3B-ov")
DEFAULT_SEQ650 = (
    ROOT / "output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ")
DEFAULT_CENSUS = (
    ROOT / "output/prefill-router-shape-census-gate-20260711Tseq639cleanZ")
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json")
DEFAULT_FRONTIER = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json")
DEFAULT_ADR3 = ROOT / "doc/adr/0003-surrogate-refine-splitplane-dual-phase-engine.md"

MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
OPENVINO_LANGUAGE_MODEL_BYTES = 18_646_205_498
HEADLINE_PREFILL_TPS = 2781.0
HEADLINE_DECODE_TPS = 52.79
M64_PROXY_ROWS = 248_320
HIDDEN = 2_048
TILE_TOKENS = 1_024
ASSIGNMENTS = 8_192
MOE_INTERMEDIATE = 512
Q6_ACTIVE_BYTES = 352_665_600
ACCEPTED_ACTIVE_BYTES = 1_786_959_744
KV_BYTES_PER_TOKEN_AT_8K = 8_192 * 20_480
PACKED_Q4_GB_S = 110.522
Q6_KILL_GB_S = 96.0
ROUTER_BUDGET_US = 450.0
SCHEDULE_BUDGET_US = 350.0


WORKER = r'''#!/usr/bin/env python3
import hashlib
import json
import math
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import openvino_genai as ov_genai
from openvino import Core, Model, Type, get_version
from openvino import opset15 as ov


cfg = json.loads(Path(__import__("sys").argv[1]).read_text(encoding="utf-8"))
out = {
    "device": {},
    "full_model": {},
    "openvino_genai_version": ov_genai.__version__,
    "openvino_runtime_version": get_version(),
}


def serial(value):
  if isinstance(value, dict):
    return {str(k): serial(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [serial(v) for v in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


def execution_devices(compiled):
  value = compiled.get_property("EXECUTION_DEVICES")
  return [value] if isinstance(value, str) else list(value)


def compare(lhs, rhs):
  a = np.asarray(lhs, dtype=np.float32).ravel()
  b = np.asarray(rhs, dtype=np.float32).ravel()
  delta = a - b
  denom = float(np.linalg.norm(a) * np.linalg.norm(b))
  return {
      "compared": int(a.size),
      "cosine": float(np.dot(a, b) / denom) if denom else 1.0,
      "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
      "max_abs": float(np.max(np.abs(delta))),
      "rmse": float(np.sqrt(np.mean(delta * delta))),
  }


def submodel(core, m):
  full = core.read_model(str(Path(cfg["model"]) / "openvino_language_model.xml"))
  mm = next(
      node for node in full.get_ops()
      if node.get_friendly_name() == "__module.model.lm_head/ov_ext::linear/MatMul")
  param = ov.parameter([1, m, 2048], Type.f32, name=f"hidden_m{m}")
  mm.input(0).replace_source_output(param.output(0))
  return Model([mm.output(0)], [param], f"qwen36_real_lm_head_m{m}")


def timed_infer(request, data, count):
  walls = []
  for _ in range(count):
    started = time.perf_counter()
    request.infer({0: data})
    walls.append((time.perf_counter() - started) * 1e6)
  return walls


def run_shape(core, m, include_cpu):
  model = submodel(core, m)
  data = np.linspace(-1.0, 1.0, m * 2048, dtype=np.float32).reshape(1, m, 2048)
  devices = ["CPU", "GPU", "NPU"] if include_cpu else ["GPU", "NPU"]
  compiled = {}
  requests = {}
  row = {
      "logical_ops_per_device": 2 * m * 2048 * 248320,
      "m": m,
      "shape": [1, m, 2048],
      "weight_bytes_per_device": 248320 * 2048,
  }
  for device in devices:
    started = time.perf_counter()
    compiled[device] = core.compile_model(model, device)
    requests[device] = compiled[device].create_infer_request()
    row[device] = {
        "compile_s": time.perf_counter() - started,
        "execution_devices": execution_devices(compiled[device]),
    }
    for _ in range(3):
      requests[device].infer({0: data})
    walls = timed_infer(requests[device], data, 7 if m == 1 else 5)
    row[device]["solo_pre_us"] = walls
    row[device]["solo_pre_min_us"] = min(walls)
    row[device]["solo_pre_median_us"] = float(np.median(walls))

  concurrent = []
  for _ in range(12 if m == 1 else 8):
    started = time.perf_counter()
    requests["GPU"].start_async({0: data})
    requests["NPU"].start_async({0: data})
    requests["GPU"].wait()
    gpu_wait_us = (time.perf_counter() - started) * 1e6
    requests["NPU"].wait()
    concurrent.append({
        "gpu_wait_us": gpu_wait_us,
        "wall_us": (time.perf_counter() - started) * 1e6,
    })
  concurrent_walls = [sample["wall_us"] for sample in concurrent]
  row["concurrent"] = concurrent
  row["concurrent_max_us"] = max(concurrent_walls)
  row["concurrent_median_us"] = float(np.median(concurrent_walls))
  row["concurrent_min_us"] = min(concurrent_walls)

  outputs = {}
  for device in devices:
    requests[device].infer({0: data})
    outputs[device] = np.array(
        requests[device].get_output_tensor(0).data, dtype=np.float32, copy=True)
    walls = timed_infer(requests[device], data, 5)
    row[device]["solo_post_us"] = walls
    row[device]["solo_post_median_us"] = float(np.median(walls))
    row[device]["output_abs_sum"] = float(np.abs(outputs[device]).sum())
    row[device]["output_finite"] = bool(np.isfinite(outputs[device]).all())
  row["npu_vs_gpu"] = compare(outputs["NPU"], outputs["GPU"])
  if include_cpu:
    row["gpu_vs_cpu"] = compare(outputs["GPU"], outputs["CPU"])
    row["npu_vs_cpu"] = compare(outputs["NPU"], outputs["CPU"])
    row["output_sha256"] = {
        device: hashlib.sha256(outputs[device].tobytes()).hexdigest()
        for device in devices
    }
  return row


try:
  core = Core()
  out["available_devices"] = list(core.available_devices)
  for device in core.available_devices:
    props = {}
    for key in (
        "FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "DEVICE_GOPS",
        "OPTIMIZATION_CAPABILITIES", "NPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_MAX_TILES", "NPU_COMPILER_VERSION", "NPU_DRIVER_VERSION"):
      try:
        props[key] = serial(core.get_property(device, key))
      except Exception as exc:
        props[key] = {"unavailable": repr(exc)}
    out["device"][device] = props

  language_model = core.read_model(
      str(Path(cfg["model"]) / "openvino_language_model.xml"))
  ops = list(language_model.get_ops())
  started = time.perf_counter()
  supported = core.query_model(language_model, "NPU")
  unsupported = [
      {"name": node.get_friendly_name(), "type": node.get_type_name()}
      for node in ops if node.get_friendly_name() not in supported
  ]
  out["full_model"].update({
      "op_count": len(ops),
      "query_s": time.perf_counter() - started,
      "supported_count": len(supported),
      "unsupported": unsupported[:100],
      "unsupported_count": len(unsupported),
  })

  started = time.perf_counter()
  try:
    pipe = ov_genai.VLMPipeline(cfg["model"], "NPU")
    out["full_model"]["compile"] = {
        "error": None,
        "success": True,
        "wall_s": time.perf_counter() - started,
    }
    del pipe
  except Exception as exc:
    text = repr(exc)
    duplicate = __import__("re").search(r"Found ([0-9]+) duplicated names", text)
    out["full_model"]["compile"] = {
        "duplicate_name_count": int(duplicate.group(1)) if duplicate else None,
        "error": text,
        "success": False,
        "wall_s": time.perf_counter() - started,
    }

  out["components"] = {
      "m1": run_shape(core, 1, True),
      "m64": run_shape(core, 64, False),
  }
except Exception as exc:
  out["fatal_error"] = repr(exc)
  out["fatal_traceback"] = traceback.format_exc()

out["maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("IQ36_RESULT_JSON=" + json.dumps(out, sort_keys=True), flush=True)
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--openvino-python", type=Path,
                      default=DEFAULT_OPENVINO_PYTHON)
  parser.add_argument("--openvino-model", type=Path,
                      default=DEFAULT_OPENVINO_MODEL)
  parser.add_argument("--seq650", type=Path, default=DEFAULT_SEQ650)
  parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/product-architecture-feasibility-{stamp}"
  return args


def read_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def selected_shape(census: Path) -> dict[str, Any]:
  with (census / "layer-shapes.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
      row = json.loads(line)
      if row.get("case_id") == "prefill_shape_008k" and row.get("layer") == 27:
        return row
  raise SystemExit("seq639 layer-27 shape is missing")


def multiple8_padding(shape: dict[str, Any]) -> dict[str, Any]:
  histogram = {int(key): int(value)
               for key, value in shape["group_m_histogram"].items()}
  actual = sum(m * count for m, count in histogram.items())
  padded = sum(((m + 7) // 8) * 8 * count for m, count in histogram.items())
  return {
      "actual_assignments": actual,
      "bucket_count": len({((m + 7) // 8) * 8 for m in histogram}),
      "padded_assignments": padded,
      "padding_ratio": padded / actual,
  }


def parse_worker(stdout: str) -> dict[str, Any]:
  prefix = "IQ36_RESULT_JSON="
  lines = [line[len(prefix):] for line in stdout.splitlines()
           if line.startswith(prefix)]
  if not lines:
    return {}
  return json.loads(lines[-1])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required = [
      args.openvino_python,
      args.openvino_model / "openvino_language_model.xml",
      args.openvino_model / "openvino_language_model.bin",
      args.seq650 / "result.json",
      args.census / "result.json",
      args.census / "layer-shapes.jsonl",
      args.acceptance,
      args.frontier,
      DEFAULT_ADR3,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  language_bin = args.openvino_model / "openvino_language_model.bin"
  if language_bin.stat().st_size != OPENVINO_LANGUAGE_MODEL_BYTES:
    raise SystemExit("OpenVINO language-model byte size mismatch")

  worker_path = raw_dir / "npu-product-feasibility-worker.py"
  config_path = raw_dir / "worker-config.json"
  worker_path.write_text(WORKER, encoding="utf-8")
  write_json(config_path, {"model": str(args.openvino_model)})
  created_at = iso_now()
  try:
    worker_run = subprocess.run(
        [str(args.openvino_python), "-u", str(worker_path), str(config_path)],
        check=False, capture_output=True, text=True, timeout=args.timeout_s,
        encoding="utf-8", errors="replace")
    timed_out = False
    stdout, stderr, returncode = (
        worker_run.stdout, worker_run.stderr, worker_run.returncode)
  except subprocess.TimeoutExpired as exc:
    timed_out = True
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    returncode = 124
  (raw_dir / "worker.stdout").write_text(stdout, encoding="utf-8")
  (raw_dir / "worker.stderr").write_text(stderr, encoding="utf-8")
  worker = parse_worker(stdout)
  compile_row = worker.get("full_model", {}).get("compile", {})
  if compile_row.get("success") is False and not isinstance(
      compile_row.get("duplicate_name_count"), int):
    duplicate = re.search(r"Found ([0-9]+) duplicated names", stderr)
    if duplicate:
      compile_row["duplicate_name_count"] = int(duplicate.group(1))
  shape = selected_shape(args.census)
  padding = multiple8_padding(shape)
  seq650 = read_json(args.seq650 / "result.json")
  acceptance = read_json(args.acceptance)
  frontier = read_json(args.frontier)

  m1 = worker.get("components", {}).get("m1", {})
  m64 = worker.get("components", {}).get("m64", {})
  m1_wall = float(m1.get("concurrent_median_us", math.inf))
  m64_wall = float(m64.get("concurrent_median_us", math.inf))
  proxy_weight_bytes = int(m1.get("weight_bytes_per_device", 0))
  proxy_ops = int(m64.get("logical_ops_per_device", 0))
  aggregate_stream_gb_s = (
      2.0 * proxy_weight_bytes / (m1_wall * 1e3)
      if math.isfinite(m1_wall) and m1_wall > 0 else 0.0)
  aggregate_m64_tops = (
      2.0 * proxy_ops / (m64_wall * 1e6)
      if math.isfinite(m64_wall) and m64_wall > 0 else 0.0)
  npu_m64_us = float(
      m64.get("NPU", {}).get("solo_pre_median_us", math.inf))
  npu_m64_tops = (
      proxy_ops / (npu_m64_us * 1e6)
      if math.isfinite(npu_m64_us) and npu_m64_us > 0 else 0.0)

  real_matrix_ops = (
      2 * ASSIGNMENTS * HIDDEN * MOE_INTERMEDIATE * 2 +
      2 * ASSIGNMENTS * MOE_INTERMEDIATE * HIDDEN)
  stages = seq650["probe"]["stage_us"]
  external_shell_us = sum(float(stages[name]) for name in (
      "gather", "residual_swiglu", "residual_weight", "scatter"))
  cap_us = float(seq650["budget"]["kernel_cap_us"])
  hybrid_matrix_us = (
      real_matrix_ops / (aggregate_m64_tops * 1e6) * padding["padding_ratio"]
      if aggregate_m64_tops > 0 else math.inf)
  hybrid_complete_us = hybrid_matrix_us + external_shell_us
  npu_only_matrix_us = (
      real_matrix_ops / (npu_m64_tops * 1e6) * padding["padding_ratio"]
      if npu_m64_tops > 0 else math.inf)
  npu_only_complete_us = npu_only_matrix_us + external_shell_us

  decode_budget_us = 1e6 / HEADLINE_DECODE_TPS
  mixed_decode_us = (
      (ACCEPTED_ACTIVE_BYTES - Q6_ACTIVE_BYTES + KV_BYTES_PER_TOKEN_AT_8K) /
      (PACKED_Q4_GB_S * 1e3) +
      Q6_ACTIVE_BYTES / (aggregate_stream_gb_s * 1e3) +
      ROUTER_BUDGET_US + SCHEDULE_BUDGET_US
      if aggregate_stream_gb_s > 0 else math.inf)
  mixed_decode_tps = 1e6 / mixed_decode_us if mixed_decode_us > 0 else 0.0
  strict_required_gb_s = (
      (ACCEPTED_ACTIVE_BYTES + KV_BYTES_PER_TOKEN_AT_8K) *
      HEADLINE_DECODE_TPS / 1e9)

  full_model = worker.get("full_model", {})
  full_compile = full_model.get("compile", {})
  available = worker.get("available_devices", [])
  evidence_checks = [
      check("worker_completed", returncode == 0 and not timed_out,
            returncode=returncode, timed_out=timed_out),
      check("worker_returned_structured_result", bool(worker) and
            "fatal_error" not in worker),
      check("cpu_gpu_npu_available", all(device in available
                                         for device in ("CPU", "GPU", "NPU")),
            available=available),
      check("full_language_model_query_supported_all_ops",
            full_model.get("op_count") == 16_051 and
            full_model.get("supported_count") == 16_051 and
            full_model.get("unsupported_count") == 0,
            op_count=full_model.get("op_count"),
            supported_count=full_model.get("supported_count")),
      check("full_npu_compile_outcome_recorded",
            isinstance(full_compile.get("success"), bool),
            success=full_compile.get("success"),
            duplicate_name_count=full_compile.get("duplicate_name_count")),
      check("parameterized_m1_m64_compile_on_gpu_and_npu",
            all(m.get(device, {}).get("execution_devices") in
                (["GPU.0"], ["GPU"]) if device == "GPU" else
                m.get(device, {}).get("execution_devices") == ["NPU"]
                for m in (m1, m64) for device in ("GPU", "NPU"))),
      check("m1_npu_numeric_proxy_matches_cpu",
            m1.get("npu_vs_cpu", {}).get("finite") is True and
            float(m1.get("npu_vs_cpu", {}).get("cosine", 0.0)) >= 0.999 and
            float(m1.get("npu_vs_cpu", {}).get("max_abs", math.inf)) <= 0.01,
            comparison=m1.get("npu_vs_cpu")),
      check("m64_npu_numeric_proxy_matches_gpu",
            m64.get("npu_vs_gpu", {}).get("finite") is True and
            float(m64.get("npu_vs_gpu", {}).get("cosine", 0.0)) >= 0.999 and
            float(m64.get("npu_vs_gpu", {}).get("max_abs", math.inf)) <= 0.01,
            comparison=m64.get("npu_vs_gpu")),
      check("locked_decode_anchor_present",
            float(frontier["goal_anchor"]["current_best_tps"]) > 0 and
            float(acceptance["bootstrap_targets"]["decode_tokens_s"]["8192"]) ==
            HEADLINE_DECODE_TPS),
      check("speedup_claims_forbidden", True),
  ]
  architecture_checks = [
      check("cpu_decode_architecture_fails_product_floor",
            float(frontier["goal_anchor"]["cpu_native_denominator_tps"]) <
            HEADLINE_DECODE_TPS,
            cpu_tps=frontier["goal_anchor"]["cpu_native_denominator_tps"],
            required_tps=HEADLINE_DECODE_TPS),
      check("gpu_exact_grouped_prefill_is_closed",
            float(seq650["probe"]["minimum_us"]) > cap_us,
            observed_us=seq650["probe"]["minimum_us"], required_us=cap_us),
      check("npu_only_prefill_projection_fails_cap",
            npu_only_complete_us > cap_us,
            projected_us=npu_only_complete_us, required_us=cap_us),
      check("gpu_npu_prefill_projection_clears_cap",
            hybrid_complete_us <= cap_us,
            projected_us=hybrid_complete_us, required_us=cap_us),
      check("gpu_npu_exact_q6_carrier_clears_kill_number",
            aggregate_stream_gb_s >= Q6_KILL_GB_S,
            observed_gb_s=aggregate_stream_gb_s,
            required_gb_s=Q6_KILL_GB_S),
      check("gpu_npu_mixed_decode_projection_clears_headline",
            mixed_decode_us <= decode_budget_us,
            projected_us=mixed_decode_us, required_us=decode_budget_us,
            projected_tps=mixed_decode_tps),
  ]
  evidence_passed = all(row["pass"] for row in evidence_checks)
  architecture_passed = all(row["pass"] for row in architecture_checks)
  required_passed = evidence_passed and architecture_passed
  disposition = (
      "select_gpu_npu_parameterized_exact_component_gate"
      if required_passed else
      "request_owner_decision_no_complete_architecture_bound")

  result = {
      "architecture_checks_passed": architecture_passed,
      "bounds": {
          "decode": {
              "accepted_active_bytes": ACCEPTED_ACTIVE_BYTES,
              "aggregate_proxy_stream_gb_s": aggregate_stream_gb_s,
              "budget_us": decode_budget_us,
              "headline_target_tps": HEADLINE_DECODE_TPS,
              "kv_bytes_at_8k": KV_BYTES_PER_TOKEN_AT_8K,
              "mixed_projection_tps": mixed_decode_tps,
              "mixed_projection_us": mixed_decode_us,
              "packed_q4_gb_s": PACKED_Q4_GB_S,
              "q6_active_bytes": Q6_ACTIVE_BYTES,
              "q6_kill_gb_s": Q6_KILL_GB_S,
              "router_budget_us": ROUTER_BUDGET_US,
              "schedule_budget_us": SCHEDULE_BUDGET_US,
              "strict_candidate_required_gb_s": strict_required_gb_s,
          },
          "prefill": {
              "aggregate_m64_tops": aggregate_m64_tops,
              "cap_us": cap_us,
              "external_exact_shell_us": external_shell_us,
              "headline_target_tps": HEADLINE_PREFILL_TPS,
              "hybrid_complete_us": hybrid_complete_us,
              "hybrid_matrix_us": hybrid_matrix_us,
              "multiple8": padding,
              "npu_only_complete_us": npu_only_complete_us,
              "npu_only_matrix_us": npu_only_matrix_us,
              "npu_proxy_tops": npu_m64_tops,
              "real_matrix_ops": real_matrix_ops,
          },
      },
      "checks": evidence_checks + architecture_checks,
      "created_at": created_at,
      "disposition": disposition,
      "evidence_checks_passed": evidence_passed,
      "git": git_state(),
      "required_checks_passed": required_passed,
      "schema_version": SCHEMA_VERSION,
      "selected_route": (
          "gpu_npu_parameterized_exact_q6_variable_m_component_v1"
          if required_passed else None),
      "sources": {
          "acceptance": str(args.acceptance.relative_to(ROOT)),
          "adr3": str(DEFAULT_ADR3.relative_to(ROOT)),
          "census": str(args.census.relative_to(ROOT)),
          "frontier": str(args.frontier.relative_to(ROOT)),
          "openvino_language_model_bytes": language_bin.stat().st_size,
          "openvino_model": str(args.openvino_model),
          "seq650": str(args.seq650.relative_to(ROOT)),
      },
      "speedup_claims_allowed": False,
      "worker": worker,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": evidence_checks,
      "evidence_checks_passed": evidence_passed,
      "m1_npu_vs_cpu": m1.get("npu_vs_cpu"),
      "m64_npu_vs_gpu": m64.get("npu_vs_gpu"),
      "required_checks_passed": required_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "product architecture component feasibility gate only",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "aggregate_m1_stream_gb_s", "value": aggregate_stream_gb_s},
      {"metric": "aggregate_m64_tops", "value": aggregate_m64_tops},
      {"metric": "prefill_hybrid_complete_us", "value": hybrid_complete_us},
      {"metric": "prefill_cap_us", "value": cap_us},
      {"metric": "decode_mixed_projection_us", "value": mixed_decode_us},
      {"metric": "decode_budget_us", "value": decode_budget_us},
      {"metric": "required_checks_passed", "value": required_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  compile_note = (
      "passed" if full_compile.get("success") else
      f"failed ({full_compile.get('duplicate_name_count')} duplicated names)")
  summary = [
      "# Product architecture feasibility reconciliation",
      "",
      f"- full OpenVINO NPU model compile: `{compile_note}`",
      f"- full-model NPU query: `{full_model.get('supported_count')} / "
      f"{full_model.get('op_count')}` ops supported",
      f"- M=1 GPU+NPU aggregate stream: `{aggregate_stream_gb_s:.3f} GB/s`",
      f"- M=64 GPU+NPU aggregate compute: `{aggregate_m64_tops:.3f} TOPS`",
      f"- NPU-only prefill projection: `{npu_only_complete_us:.3f} us`",
      f"- hybrid prefill projection / cap: `{hybrid_complete_us:.3f} / "
      f"{cap_us:.3f} us`",
      f"- hybrid mixed decode projection / floor: `{mixed_decode_tps:.3f} / "
      f"{HEADLINE_DECODE_TPS:.2f} tok/s`",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "CPU, GPU-only grouped prefill, and NPU-only prefill do not clear the",
      "product bound.  The only admitted successor is a parameterized GPU+NPU",
      "partition, and only for one exact Q6/variable-M component gate.  The",
      "OpenVINO graph is compiler/hardware evidence; a promoted runtime may not",
      "link OpenVINO or oneDNN.  No native speedup or product claim is made.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_passed,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
