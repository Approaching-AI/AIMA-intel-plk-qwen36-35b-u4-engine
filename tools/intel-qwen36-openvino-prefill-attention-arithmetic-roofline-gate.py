#!/usr/bin/env python3
"""Run the one admitted fixed KQ+PV prefill arithmetic roofline component."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

from iq36_perf_inference import latency_cap_inference


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-prefill-attention-arithmetic-roofline-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
PLUGIN_SHA256 = (
    "432f4ebb1802b619ed347e1ba6344492177da884c9cf4d8a815e360122e0f876")
CONFIG = REPO / (
    "engine/openvino/custom/iq36_prefill_attention_arithmetic_roofline.xml")
SOURCE = REPO / (
    "engine/openvino/custom/iq36_prefill_attention_tiled.cl")
HELPERS = REPO / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
SHIMS = REPO / "engine/openvino/custom/iq36_prefill_microkernel_shims.cl"
SOURCE_BOUND = REPO / (
    "output/openvino-prefill-context-attention-bound-"
    "20260715Tseq1258-cleanZ/metrics.json")

BATCH = 48
Q_HEADS = 16
KV_HEADS = 2
GROUPS = BATCH * Q_HEADS
REPEATS = 16
KEY_TOKENS = 128
QUERY_TOKENS = 32
HEAD_DIM = 256
LOCAL_SIZE = 128
HOT_TOKENS = 129
HOT_KEY_WORDS_PER_BLOCK = 2048
HOT_KEY_STORAGE_BLOCKS = 2 * ((HOT_TOKENS + 15) // 16) + 1
OUTPUT_TILE_WIDTH = QUERY_TOKENS * HEAD_DIM
MACS_PER_GROUP_REPEAT = 2 * KEY_TOKENS * QUERY_TOKENS * HEAD_DIM
TOTAL_MACS = GROUPS * REPEATS * MACS_PER_GROUP_REPEAT
EXPECTED_REQUIRED_TMAC_S = 17.682446822490522
EXPECTED_SAMPLES = 20


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--samples", type=int, default=EXPECTED_SAMPLES)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required")
  if (args.warmup < 1 or args.samples != EXPECTED_SAMPLES or
      args.timeout_s <= 0 or args.memory_stop_gib <= 0.0):
    parser.error("warmup/samples/timeout/memory arguments are invalid")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
      capture_output=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, check=True, text=True,
      capture_output=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def custom_class(ov: Any) -> type:
  class IQ36PrefillAttentionArithmeticRoofline(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_size(5)
      query = self.get_input_partial_shape(0)
      query_tiles = ov.Dimension.dynamic()
      if query[2].is_static:
        query_tiles = ov.Dimension(
            (query[2].get_length() + QUERY_TOKENS - 1) // QUERY_TOKENS)
      self.set_output_type(
          0, ov.Type.f16,
          ov.PartialShape([
              query[0], query[1], query_tiles, OUTPUT_TILE_WIDTH]))
      scratch = self.get_input_partial_shape(10)
      self.set_output_type(1, ov.Type.i8, scratch)
      self.set_output_type(2, ov.Type.i8, scratch)
      scale_shape = ov.PartialShape([
          scratch[0], scratch[1], scratch[2], 16])
      self.set_output_type(3, ov.Type.i8, scale_shape)
      self.set_output_type(4, ov.Type.i8, scale_shape)

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36PrefillAttentionArithmeticRoofline(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36PrefillAttentionArithmeticRoofline


def make_model(ov: Any) -> Any:
  parameters = [
      ov.opset13.parameter(
          [BATCH, Q_HEADS, -1, HEAD_DIM],
          ov.Type.f16, name="query"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, HOT_KEY_STORAGE_BLOCKS,
           HOT_KEY_WORDS_PER_BLOCK], ov.Type.i32, name="hot_key"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, HOT_TOKENS, HEAD_DIM],
          ov.Type.f16, name="hot_value"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, -1, HEAD_DIM],
          ov.Type.f16, name="current_key"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, -1, HEAD_DIM],
          ov.Type.f16, name="current_value"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, 1, HEAD_DIM], ov.Type.i8, name="cold_key"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, 1, HEAD_DIM], ov.Type.i8, name="cold_value"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, 1, 16], ov.Type.i8, name="key_scale"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, 1, 16], ov.Type.i8, name="value_scale"),
      ov.opset13.parameter(
          [BATCH, 1, -1, KEY_TOKENS],
          ov.Type.f16, name="mask"),
      ov.opset13.parameter(
          [BATCH, KV_HEADS, 1, HEAD_DIM],
          ov.Type.i8, name="eviction_template"),
      ov.opset13.parameter(
          [1, 1, 1, 1], ov.Type.i32, name="eviction_count"),
      ov.opset13.parameter(
          [1, 1, 1, 1], ov.Type.i32, name="length_carrier"),
  ]
  operation = custom_class(ov)([
      parameter.output(0) for parameter in parameters])
  operation.set_friendly_name("iq36_prefill_attention_arithmetic_roofline")
  return ov.Model(
      [operation.output(0)], parameters,
      "iq36_prefill_attention_arithmetic_roofline_component")


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
  raw = Path(cfg["raw"])
  registry = raw / "candidate-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location="
      f"{quoteattr(str(PLUGIN.resolve()))}/></plugins></ie>\n",
      encoding="utf-8")
  core = ov.Core(str(registry))
  core.set_property("GPU", {"CONFIG_FILE": str(CONFIG.resolve())})
  compile_started = time.perf_counter_ns()
  compiled = core.compile_model(
      make_model(ov), "GPU",
      {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True})
  compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0

  rng = np.random.default_rng(0x5136)
  values = {
      "query": (rng.standard_normal(
          (BATCH, Q_HEADS, QUERY_TOKENS, HEAD_DIM)) *
          0.02).astype(np.float16),
      "hot_key": np.zeros(
          (BATCH, KV_HEADS, HOT_KEY_STORAGE_BLOCKS,
           HOT_KEY_WORDS_PER_BLOCK), dtype=np.int32),
      "hot_value": np.zeros(
          (BATCH, KV_HEADS, HOT_TOKENS, HEAD_DIM), dtype=np.float16),
      "current_key": (rng.standard_normal(
          (BATCH, KV_HEADS, KEY_TOKENS, HEAD_DIM)) *
          0.02).astype(np.float16),
      "current_value": (rng.standard_normal(
          (BATCH, KV_HEADS, KEY_TOKENS, HEAD_DIM)) *
          0.02).astype(np.float16),
      "cold_key": np.zeros(
          (BATCH, KV_HEADS, 1, HEAD_DIM), dtype=np.int8),
      "cold_value": np.zeros(
          (BATCH, KV_HEADS, 1, HEAD_DIM), dtype=np.int8),
      "key_scale": np.zeros(
          (BATCH, KV_HEADS, 1, 16), dtype=np.int8),
      "value_scale": np.zeros(
          (BATCH, KV_HEADS, 1, 16), dtype=np.int8),
      "mask": np.zeros(
          (BATCH, 1, QUERY_TOKENS, KEY_TOKENS), dtype=np.float16),
      "eviction_template": np.zeros(
          (BATCH, KV_HEADS, 1, HEAD_DIM), dtype=np.int8),
      "eviction_count": np.full(
          (1, 1, 1, 1), REPEATS, dtype=np.int32),
      "length_carrier": np.full(
          (1, 1, 1, 1), QUERY_TOKENS, dtype=np.int32),
  }
  feed = {compiled.input(name): value for name, value in values.items()}
  request = compiled.create_infer_request()
  for _ in range(int(cfg["warmup"])):
    request.infer(feed, share_outputs=True)

  samples = []
  final_profile: list[dict[str, Any]] = []
  for index in range(int(cfg["samples"])):
    started = time.perf_counter_ns()
    request.infer(feed, share_outputs=True)
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    profile = profile_rows(request)
    custom = [
        row for row in profile
        if row["status"] == "Status.EXECUTED" and
        (row["node_type"] == "IQ36PrefillAttentionArithmeticRoofline" or
         "prefill_attention_arithmetic_roofline" in
         row["node_name"].lower())]
    output = np.asarray(request.get_output_tensor(0).data)
    samples.append({
        "sample": index,
        "kernel_ms": sum(row["real_time_us"] for row in custom) / 1000.0,
        "wall_ms": wall_ms,
        "custom_profile_count": len(custom),
        "output_sha256": hashlib.sha256(
            np.ascontiguousarray(output).tobytes()).hexdigest(),
    })
    final_profile = profile

  output = np.array(request.get_output_tensor(0).data, copy=True)
  write_json(Path(cfg["result"]), {
      "openvino_version": ov.get_version(),
      "plugin": str(PLUGIN.resolve()),
      "plugin_sha256": sha256(PLUGIN),
      "compile_ms": compile_ms,
      "batch": BATCH,
      "q_heads": Q_HEADS,
      "groups": GROUPS,
      "repeats": REPEATS,
      "total_macs": TOTAL_MACS,
      "samples": samples,
      "output": {
          "shape": list(output.shape),
          "finite": bool(np.isfinite(output).all()),
          "nonzero": bool(np.any(output != 0.0)),
          "minimum": float(np.min(output)),
          "maximum": float(np.max(output)),
      },
      "final_profile": final_profile,
  })
  return 0


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "swaps": r"Swaps: (\d+)",
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def run_worker(
    config: Path, timeout_s: int, raw: Path,
) -> dict[str, Any]:
  time_path = raw / "worker.time.txt"
  command = [
      "/usr/bin/time", "-v", "-o", str(time_path),
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config)]
  environment = os.environ.copy()
  environment["NEO_CACHE_DIR"] = str((raw / "neo-cache").resolve())
  try:
    completed = subprocess.run(
        command, cwd=REPO, text=True, capture_output=True,
        timeout=timeout_s, check=False, env=environment,
        encoding="utf-8", errors="replace")
    (raw / "worker.stdout").write_text(
        completed.stdout, encoding="utf-8")
    (raw / "worker.stderr").write_text(
        completed.stderr, encoding="utf-8")
    return {
        "command": command, "returncode": completed.returncode,
        "timed_out": False, "resources": parse_time(time_path)}
  except subprocess.TimeoutExpired as error:
    (raw / "worker.stdout").write_text(
        str(error.stdout or ""), encoding="utf-8")
    (raw / "worker.stderr").write_text(
        str(error.stderr or ""), encoding="utf-8")
    return {
        "command": command, "returncode": 124, "timed_out": True,
        "resources": parse_time(time_path)}


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config)

  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      OV_PYTHON, PLUGIN, CONFIG, SOURCE, HELPERS, SHIMS, SOURCE_BOUND)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))
  git = git_state()
  bound = load_json(SOURCE_BOUND)
  source = SOURCE.read_text(encoding="utf-8")
  config_text = CONFIG.read_text(encoding="utf-8")
  shims = SHIMS.read_text(encoding="utf-8")
  required_tmac_s = float(
      bound.get("projection", {}).get("tightest_required_tmac_s", 0.0))
  cap_ms = TOTAL_MACS / required_tmac_s / 1e9
  source_checks = {
      "component_branch_is_compile_time_isolated": all(
          marker in source for marker in (
              "#if defined(IQ36_PREFILL_ARITHMETIC_ROOFLINE)",
              "product XML never defines this branch")),
      "runtime_repeat_control": (
          "(uint)eviction_count[INPUT11_OFFSET]" in source),
      "roofline_kq_and_pv_calls_are_present": (
          source.count("iq36_micro_score_tile score = ugemm_kq(") == 2 and
          source.count("iq36_micro_output_tile product = ugemm_vs(") == 1),
      "fixed_work_sizes": (
          '<WorkSizes global="128,Y,B*F" local="128,1,1"/>' in
          config_text),
      "roofline_compiler_flag_is_component_only": (
          "-DIQ36_PREFILL_ARITHMETIC_ROOFLINE=1" in config_text),
      "exact_shims": all(marker in shims for marker in (
          "#define ugemm_kq_wg_tile_m 128",
          "#define ugemm_kq_wg_tile_n 32",
          "#define ugemm_vs_wg_tile_m 256",
          "#define ugemm_vs_wg_tile_n 32",
          "#define ugemm_kq_systolic  1",
          "#define ugemm_vs_systolic  1")),
  }
  sample_memory("after-source-audit", stop_bytes, memory)

  config = raw / "worker-config.json"
  result_path = raw / "worker-result.json"
  write_json(config, {
      "raw": str(raw), "result": str(result_path),
      "warmup": args.warmup, "samples": args.samples})
  sample_memory("before-component-worker", stop_bytes, memory)
  worker = run_worker(config, args.timeout_s, raw)
  sample_memory("after-component-worker", stop_bytes, memory)
  result = load_json(result_path) if result_path.is_file() else {}

  samples = result.get("samples", [])
  kernel_ms = [float(row.get("kernel_ms", 0.0)) for row in samples]
  valid_timing = (
      len(kernel_ms) == EXPECTED_SAMPLES and
      all(math.isfinite(value) and value > 0.0 for value in kernel_ms))
  inference = (
      latency_cap_inference(kernel_ms, cap=cap_ms, min_samples=EXPECTED_SAMPLES)
      if valid_timing else {})
  ucb_ms = float(inference.get("upper_confidence_bound_ms", math.inf))
  lower_tmac_s = TOTAL_MACS / ucb_ms / 1e9 if ucb_ms > 0.0 else 0.0
  digests = {str(row.get("output_sha256", "")) for row in samples}
  resources = worker.get("resources", {})
  peak_rss_kib = int(resources.get("maximum_resident_kib", 1 << 62))
  swaps = int(resources.get("swaps", -1))
  bound_contract = bool(
      bound.get("required_checks_passed") is True and
      bound.get("verdict") ==
      "admit_one_bounded_current_microkernel_arithmetic_roofline_component" and
      bound.get("gpu_worker_admitted") is True and
      bound.get("long_worker_admitted") is False and
      required_tmac_s == EXPECTED_REQUIRED_TMAC_S)
  shape_contract = bool(
      result.get("batch") == BATCH and result.get("q_heads") == Q_HEADS and
      result.get("groups") == GROUPS and result.get("repeats") == REPEATS and
      result.get("total_macs") == TOTAL_MACS and
      result.get("output", {}).get("shape") ==
      [BATCH, Q_HEADS, 1, OUTPUT_TILE_WIDTH])
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1258_component_contract_is_exact", bound_contract,
            required_tmac_s=required_tmac_s, cap_ms=cap_ms),
      check("fixed_single_source_geometry", all(source_checks.values()),
            source_checks=source_checks),
      check("candidate_plugin_is_exact",
            sha256(PLUGIN) == PLUGIN_SHA256 and
            result.get("plugin_sha256") == PLUGIN_SHA256,
            plugin=str(PLUGIN), plugin_sha256=sha256(PLUGIN)),
      check("single_serial_component_worker_completes",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False, worker=worker),
      check("fixed_component_shape_and_mac_count", shape_contract),
      check("twenty_complete_device_timing_samples",
            valid_timing and
            all(row.get("custom_profile_count") == 1 for row in samples)),
      check("component_output_is_live_and_deterministic",
            result.get("output", {}).get("finite") is True and
            result.get("output", {}).get("nonzero") is True and
            len(digests) == 1 and "" not in digests,
            output=result.get("output"), digest_count=len(digests)),
      check("one_sided_95pct_lower_rate_clears_target",
            inference.get("rate_pass") is True and
            lower_tmac_s >= required_tmac_s,
            performance_inference=inference,
            lower_tmac_s=lower_tmac_s,
            required_tmac_s=required_tmac_s),
      check("worker_rss_and_swap_are_bounded",
            peak_rss_kib < 2 * 1024 * 1024 and swaps == 0,
            maximum_resident_kib=peak_rss_kib, swaps=swaps),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "promote_current_microkernel_arithmetic_roofline" if
      required_checks_passed else
      "reject_current_microkernel_arithmetic_roofline")
  sources = {
      str(path.relative_to(REPO)): sha256(path)
      for path in (CONFIG, SOURCE, HELPERS, SHIMS, SOURCE_BOUND)}
  payload = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_promoted": required_checks_passed,
      "online_softmax_carrier_component_admitted": required_checks_passed,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "product_claim_allowed": False,
      "fixed_geometry": {
          "batch": BATCH, "q_heads": Q_HEADS,
          "groups": GROUPS, "repeats": REPEATS,
          "key_tokens": KEY_TOKENS, "query_tokens": QUERY_TOKENS,
          "head_dim": HEAD_DIM, "total_macs": TOTAL_MACS,
      },
      "performance_inference": inference,
      "lower_confidence_bound_tmac_s": lower_tmac_s,
      "required_tmac_s": required_tmac_s,
      "cap_ms": cap_ms,
      "checks": checks,
      "worker": worker,
      "worker_result": result,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(output / "metrics.json", payload)
  with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
    for row in samples:
      stream.write(json.dumps({
          **row,
          "tmac_s": TOTAL_MACS / float(row["kernel_ms"]) / 1e9,
          "verdict": verdict,
      }, sort_keys=True) + "\n")
  summary = f"""# Prefill-attention arithmetic roofline component

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The fixed component executes `{TOTAL_MACS}` KQ+PV MACs per sample using the
exact extracted 128x32 and 256x32 systolic microkernels.  Device median/UCB is
`{inference.get('point_estimate_ms')} / {inference.get('upper_confidence_bound_ms')}`
ms against the `{cap_ms:.6f}`-ms cap; the one-sided 95% rate lower bound is
`{lower_tmac_s:.6f} TMAC/s` versus `{required_tmac_s:.6f}` required.  This
omits softmax, state, and graph overhead and is not product evidence.

Peak worker RSS is `{peak_rss_kib} KiB`, swaps are `{swaps}`, and the 4-GiB
available-memory stop did not trip.  A pass admits one exact online-softmax
carrier component only; graph integration and long workers remain blocked.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(REPO)),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "median_ms": inference.get("point_estimate_ms"),
      "ucb_ms": inference.get("upper_confidence_bound_ms"),
      "cap_ms": cap_ms,
      "lower_tmac_s": lower_tmac_s,
      "required_tmac_s": required_tmac_s,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
