#!/usr/bin/env python3
"""Gate direct row-major metadata for the fused router fixed-FC providers.

The stock GPU transform horizontally fuses the three shared-expert projections
to M=1025 and keeps the M=256 router separate.  This gate uses those real U4,
F16-scale, and U4-zero-point constants to compare the admitted group-major
gemmstone provider with an otherwise identical package that consumes the
stock [M,G,1] metadata directly.  It is an isolated component decision, not a
product speed claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iq36_perf_inference import paired_speedup_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-row-major-provider-gate-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-openvino-fixed-fc-prefill-component-gate.py"
GRAPH_TOOL = ROOT / "tools/intel_qwen36_openvino_fixed_fc.py"
CAPTURE = ROOT / (
    "output/openvino-fc-boundary-capture-20260715Tseq1227-"
    "layer0-qkv-2k-o1-dirtyZ/raw/capture/dispatch000-arg1-before.bin")
CAPTURE_SHA256 = "5916d74c73811c7ad0bd54f6610842329ab570ed05e191a8875f55081be46f4c"
HORIZONTAL_FUSION = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/src/plugin/transformations/fc_horizontal_fusion.cpp")
SEQ1361 = ROOT / (
    "output/openvino-fixed-fc-prefill-20260718Tseq1361-cleanZ/metrics.json")
ALLOWED_UNCOMMITTED = {
    "engine/gpu/opencl/openvino_fc_micro_host.cl",
    "engine/tools/openvino_fc_multi_output_runtime.cpp",
    "engine/tools/openvino_moe_micro_codegen.cpp",
    "tools/intel-qwen36-openvino-fixed-fc-row-major-provider-gate.py",
}
FUSED_ROWS = (
    {"name": "shared1025", "indices": (0, 1, 2), "width": 1025},
    {"name": "router256", "indices": (3,), "width": 256},
)


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--tokens", type=int, default=128)
  parser.add_argument("--warmup", type=int, default=32)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--blocks", type=int, default=8)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  args = parser.parse_args()
  if (args.tokens < 2 or args.warmup < 1 or args.repeat < 5 or
      args.blocks < 8 or args.timeout_s < 1 or
      args.min_available_gib <= 0.0 or args.abort_below_available_gib <= 0.0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("token, timing, timeout, or memory arguments are invalid")
  if args.output is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output = ROOT / f"output/openvino-fixed-fc-row-major-{stamp}"
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def meminfo() -> dict[str, int]:
  values: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, rest = line.split(":", 1)
    if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
      values[key] = int(rest.split()[0]) * 1024
  return values


def pack_u4(values: Any, np: Any) -> Any:
  flat = np.asarray(values, dtype=np.uint8).reshape(-1)
  if flat.size % 2:
    raise ValueError("U4 stream is not byte aligned")
  return ((flat[0::2] & np.uint8(15)) |
          ((flat[1::2] & np.uint8(15)) << np.uint8(4)))


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = [row for row in rows
           if row[3:] not in ALLOWED_UNCOMMITTED and
           not row[3:].startswith(output_relative)]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty,
          "allowed_uncommitted": sorted(ALLOWED_UNCOMMITTED)}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def make_real_fused_data(output: Path, tokens: int, graph: Any,
                         ov: Any, np: Any) -> list[dict[str, Any]]:
  source = ov.Core().read_model(str(graph.MODEL_XML))
  group = next(
      row for row in graph.discover_fixed_fc_groups(source, ov)
      if row["layer"] == 0 and row["cohort"] == "router_shared_input")
  if tuple(group["widths"]) != (1, 512, 512, 256) or group["k"] != 2048:
    raise RuntimeError("locked router/shared group shape differs")
  base = np.fromfile(CAPTURE, dtype="<f2")
  if base.size != 2048:
    raise RuntimeError("locked activation capture size differs")
  factors = (
      1.0 + ((np.arange(tokens, dtype=np.float32) % 17.0) - 8.0) *
      np.float32(0.001953125))
  activation = np.ascontiguousarray(
      factors[:, None] * base.astype(np.float32)[None, :], dtype=np.float16)
  rows = []
  for fused in FUSED_ROWS:
    directory = output / fused["name"]
    directory.mkdir(parents=True)
    activation.tofile(directory / "input.f16")
    weights = []
    scales = []
    zero_points = []
    projection_widths = []
    for index in fused["indices"]:
      projection = group["projections"][index]
      m = int(projection["m"])
      groups = int(projection["groups"])
      projection_widths.append(m)
      weights.append(np.asarray(
          projection["weight"].get_data(), dtype=np.uint8
      ).reshape(m, 2048 // 2).copy())
      scales.append(np.asarray(
          projection["scale"].get_data(), dtype=np.float16
      ).reshape(m, groups).copy())
      packed = np.asarray(
          projection["zero_point"].get_data(), dtype=np.uint8).reshape(-1)
      logical = np.empty(packed.size * 2, dtype=np.uint8)
      logical[0::2] = packed & np.uint8(15)
      logical[1::2] = packed >> np.uint8(4)
      zero_points.append(logical[:m * groups].reshape(m, groups).copy())
    row_weights = np.concatenate(weights, axis=0)
    row_scales = np.concatenate(scales, axis=0)
    row_zero_points = np.concatenate(zero_points, axis=0)
    if row_weights.shape != (fused["width"], 1024):
      raise RuntimeError(f"{fused['name']} fused weight shape differs")
    row_weights.tofile(directory / "weights.u4")
    row_scales.tofile(directory / "row-scales.f16")
    pack_u4(row_zero_points, np).tofile(directory / "row-zps.u4")
    row_zero_points.tofile(directory / "row-zps.u8")
    np.ascontiguousarray(row_scales.T).tofile(
        directory / "group-scales.f16")
    np.ascontiguousarray(row_zero_points.T).tofile(
        directory / "group-zps.u8")
    pack_u4(np.ascontiguousarray(row_zero_points.T), np).tofile(
        directory / "group-zps.u4")
    rows.append({
        **fused, "projection_widths": projection_widths,
        "k": 2048, "groups": 32, "tokens": tokens,
        "weight_bytes": int(row_weights.nbytes),
        "scale_bytes": int(row_scales.nbytes),
        "zero_point_bytes": int(row_zero_points.size // 2),
        "hashes": {
            name: sha256(directory / name) for name in (
                "input.f16", "weights.u4", "row-scales.f16",
                "row-zps.u4", "row-zps.u8", "group-scales.f16",
                "group-zps.u4", "group-zps.u8")},
    })
  return rows


def main() -> int:
  args = parse_args()
  import numpy as np
  import openvino as ov

  base = load_module(BASE_TOOL, "iq36_fixed_fc_prefill_base")
  graph = load_module(GRAPH_TOOL, "iq36_fixed_fc_row_major_graph")
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      BASE_TOOL, GRAPH_TOOL, CAPTURE, HORIZONTAL_FUSION, SEQ1361,
      base.CODEGEN_SOURCE, base.SINGLE_HOST, base.RUNTIME_SOURCE,
      base.ONEDNN_BUILD / "src/libdnnl.a")
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing row-major provider inputs: " + ", ".join(missing))
  if sha256(CAPTURE) != CAPTURE_SHA256:
    raise RuntimeError("locked activation capture hash differs")
  preflight_bytes = int(args.min_available_gib * 1024**3)
  stop_bytes = int(args.abort_below_available_gib * 1024**3)
  start_memory = meminfo()
  if start_memory["MemAvailable"] < preflight_bytes:
    raise RuntimeError("eight-GiB preflight did not clear")
  git = git_state(output)
  onednn_status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=base.ONEDNN_SOURCE,
      check=True, capture_output=True, text=True).stdout.strip()
  anchor = base.load_json(SEQ1361)
  horizontal_source = HORIZONTAL_FUSION.read_text(encoding="utf-8")
  data_rows = make_real_fused_data(raw / "data", args.tokens, graph, ov, np)
  del graph

  configure = base.run([
      str(base.CMAKE), "-S", str(ROOT / "engine"), "-B",
      str(base.BUILD_DIR)], args.timeout_s)
  base.write_run(raw, "configure", configure)
  runtime_build = base.run([
      str(base.CMAKE), "--build", str(base.BUILD_DIR), "--target",
      "iq36-openvino-fc-multi-output-runtime", "--", "-j1"],
      args.timeout_s)
  base.write_run(raw, "runtime-build", runtime_build)
  runtime = base.BUILD_DIR / "iq36-openvino-fc-multi-output-runtime"
  codegen = raw / "openvino-fc-row-major-codegen"
  codegen_build = base.run(base.codegen_build_command(codegen), args.timeout_s)
  base.write_run(raw, "codegen-build", codegen_build)
  prerequisites = all(row["returncode"] == 0 for row in (
      configure, runtime_build, codegen_build))

  results = []
  memory_samples = [{"label": "start", **start_memory}]
  for shape in data_rows:
    current = meminfo()
    memory_samples.append({"label": f"before-{shape['name']}", **current})
    if current["MemAvailable"] < stop_bytes:
      raise RuntimeError(f"memory stop before {shape['name']}")
    packages = raw / "packages" / shape["name"]
    baseline_dir = packages / "group-major"
    candidate_dir = packages / "row-major"
    common = [
        str(codegen), "--prefill-fc", "--shape-name", shape["name"],
        "--m", str(shape["width"]), "--n", str(args.tokens),
        "--k", "2048", "--host-source", str(base.SINGLE_HOST)]
    baseline_command = [
        *common, "--u8-zero-point", "--dump-dir", str(baseline_dir)]
    candidate_command = [
        *common, "--row-major-metadata", "--u8-zero-point",
        "--dump-dir", str(candidate_dir)]
    baseline_run = base.run_intel(baseline_command, args.timeout_s) \
        if prerequisites else {
            "command": baseline_command, "returncode": 125, "stdout": "",
            "stderr": "prerequisite failed", "timed_out": False}
    base.write_run(raw, f"{shape['name']}-group-major-codegen", baseline_run)
    candidate_run = base.run_intel(candidate_command, args.timeout_s) \
        if baseline_run["returncode"] == 0 else {
            "command": candidate_command, "returncode": 125, "stdout": "",
            "stderr": "baseline codegen failed", "timed_out": False}
    base.write_run(raw, f"{shape['name']}-row-major-codegen", candidate_run)
    baseline_codegen = base.parse_json_line(baseline_run)
    candidate_codegen = base.parse_json_line(candidate_run)
    baseline_package = (baseline_codegen.get("packages") or [{}])[0]
    candidate_package = (candidate_codegen.get("packages") or [{}])[0]
    baseline_settings = baseline_package.get("settings", {})
    candidate_settings = candidate_package.get("settings", {})
    data = raw / "data" / shape["name"]
    runtime_command = [
        str(runtime), "--candidate-single", "--baseline-zp-u8",
        "--candidate-zp-u8",
        "--baseline-binary",
        str(baseline_dir / f"{shape['name']}.program.bin"),
        "--candidate-binary",
        str(candidate_dir / f"{shape['name']}.program.bin"),
        "--kernel", f"iq36_moe_micro_{shape['name']}",
        "--input", str(data / "input.f16"),
        "--baseline-weights", str(data / "weights.u4"),
        "--baseline-scales", str(data / "group-scales.f16"),
        "--baseline-zps", str(data / "group-zps.u8"),
        "--weights", str(data / "weights.u4"),
        "--scales", str(data / "row-scales.f16"),
        "--zps", str(data / "row-zps.u8"),
        "--widths", str(shape["width"]), "--k", "2048",
        "--n", str(args.tokens), "--quant-group", "64",
        "--sg-per-wg-m", str(candidate_settings.get("sg_per_wg_m", 0)),
        "--sg-per-wg-n", str(candidate_settings.get("sg_per_wg_n", 0)),
        "--sg-per-wg-k", str(candidate_settings.get("sg_per_wg_k", 0)),
        "--wg-tile-m", str(candidate_settings.get("wg_tile_m", 0)),
        "--wg-tile-n", str(candidate_settings.get("wg_tile_n", 0)),
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--blocks", str(args.blocks), "--actual-prefix",
        str(raw / f"{shape['name']}-actual")]
    matching_settings = baseline_settings == candidate_settings
    runtime_run = base.run_intel(runtime_command, args.timeout_s) \
        if candidate_run["returncode"] == 0 and matching_settings else {
            "command": runtime_command, "returncode": 125, "stdout": "",
            "stderr": "codegen failed or package settings differ",
            "timed_out": False}
    base.write_run(raw, f"{shape['name']}-runtime", runtime_run)
    results.append({
        **shape, "baseline_codegen": baseline_codegen,
        "candidate_codegen": candidate_codegen,
        "baseline_codegen_returncode": baseline_run["returncode"],
        "candidate_codegen_returncode": candidate_run["returncode"],
        "matching_settings": matching_settings,
        "runtime_returncode": runtime_run["returncode"],
        "runtime": base.parse_json_line(runtime_run),
        "package_hashes": {
            "group_major_micro": sha256(
                baseline_dir / f"{shape['name']}.micro.bin")
            if (baseline_dir / f"{shape['name']}.micro.bin").is_file()
            else None,
            "row_major_micro": sha256(
                candidate_dir / f"{shape['name']}.micro.bin")
            if (candidate_dir / f"{shape['name']}.micro.bin").is_file()
            else None,
        },
    })

  memory_samples.append({"label": "finish", **meminfo()})
  baseline_schedule = []
  candidate_schedule = []
  if all(len(row["runtime"].get("blocks", [])) == args.blocks
         for row in results):
    for index in range(args.blocks):
      baseline_schedule.append(40.0 * sum(
          float(row["runtime"]["blocks"][index]["baseline_us"])
          for row in results))
      candidate_schedule.append(40.0 * sum(
          float(row["runtime"]["blocks"][index]["candidate_us"])
          for row in results))
  inference = (
      paired_speedup_inference(
          baseline_schedule, candidate_schedule, target_ratio=1.0,
          min_blocks=args.blocks)
      if baseline_schedule else {})
  source_contract = {
      "locked_four_projection_comment":
          "fanout has four K=2048 compressed FCs" in horizontal_source,
      "shared_three_fusion":
          "return shared_expert_candidates;" in horizontal_source,
      "weight_concat_axis_zero":
          "Concat>(weight_nodes_as_output_vector, 0)" in horizontal_source,
      "scale_concat_axis_zero":
          "Concat>(scales_as_output_vector, 0)" in horizontal_source,
      "zero_point_concat_axis_zero":
          "Concat>(zp_nodes_as_output_vector, 0)" in horizontal_source,
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_onednn_source_is_clean", onednn_status == "",
            dirty_paths=onednn_status.splitlines()),
      check("retained_group_major_prefill_anchor_is_admitted",
            anchor.get("required_checks_passed") is True and
            anchor.get("component_admission_passed") is True,
            artifact=str(SEQ1361.relative_to(ROOT))),
      check("stock_horizontal_fusion_contract_is_exact",
            all(source_contract.values()), contract=source_contract),
      check("real_fused_shapes_and_stream_sizes_are_exact",
            [row["width"] for row in data_rows] == [1025, 256] and
            all(row["k"] == 2048 and row["groups"] == 32 and
                row["weight_bytes"] == row["width"] * 1024 and
                row["scale_bytes"] == row["width"] * 32 * 2 and
                row["zero_point_bytes"] == row["width"] * 16
                for row in data_rows), rows=data_rows),
      check("serial_build_and_codegen_passed",
            prerequisites and all(
                row["baseline_codegen_returncode"] == 0 and
                row["candidate_codegen_returncode"] == 0 and
                row["matching_settings"] for row in results)),
      check("package_metadata_layouts_are_exact",
            all((row["baseline_codegen"].get("packages") or [{}])[0].get(
                    "metadata_layout") == "group_major_group_m" and
                (row["baseline_codegen"].get("packages") or [{}])[0].get(
                    "zero_point_type") == "u8" and
                (row["candidate_codegen"].get("packages") or [{}])[0].get(
                    "metadata_layout") == "row_major_m_group" and
                (row["candidate_codegen"].get("packages") or [{}])[0].get(
                    "zero_point_type") == "u8"
                for row in results)),
      check("both_real_fused_row_major_outputs_are_bit_exact",
            all(row["runtime_returncode"] == 0 and
                row["runtime"].get("baseline_candidate_compare", {}).get(
                    "finite") is True and
                row["runtime"].get("baseline_candidate_compare", {}).get(
                    "exact_rate") == 1.0 for row in results),
            comparisons=[row["runtime"].get("baseline_candidate_compare")
                         for row in results]),
      check("row_major_packages_are_systolic_and_spill_free",
            all((row["candidate_codegen"].get("packages") or [{}])[0].get(
                    "systolic") is True and
                row["runtime"].get("candidate_spill_memory_bytes") == 0
                for row in results)),
      check("paired_component_inference_has_required_samples",
            len(baseline_schedule) == args.blocks and
            inference.get("sample_count_pass") is True,
            baseline_schedule_us=baseline_schedule,
            candidate_schedule_us=candidate_schedule),
      check("memory_preflight_and_stop_held_without_oom",
            start_memory["MemAvailable"] >= preflight_bytes and
            min(row["MemAvailable"] for row in memory_samples) >= stop_bytes and
            all(row["runtime_returncode"] not in (-9, 137)
                for row in results), memory_samples=memory_samples,
            note="swap is telemetry only; GPU work stayed serial"),
  ]
  required_passed = all(row["pass"] for row in checks)
  direct_admitted = required_passed and inference.get("rate_pass") is True
  verdict = (
      "admit_direct_row_major_metadata_plugin_internal_provider"
      if direct_admitted else
      "select_plugin_internal_one_time_group_major_metadata_reorder"
      if required_passed else
      "row_major_fused_provider_gate_inconclusive")
  result = {
      "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS, "required_checks_passed": required_passed,
      "direct_row_major_provider_admitted": direct_admitted,
      "verdict": verdict, "git": git,
      "inputs": {
          "tokens": args.tokens, "capture": str(CAPTURE.relative_to(ROOT)),
          "capture_sha256": sha256(CAPTURE),
          "horizontal_fusion_source": str(HORIZONTAL_FUSION),
      },
      "rows": results,
      "complete_router_schedule": {
          "baseline_samples_us": baseline_schedule,
          "candidate_samples_us": candidate_schedule,
          "baseline_median_us": (
              statistics.median(baseline_schedule)
              if baseline_schedule else math.inf),
          "candidate_median_us": (
              statistics.median(candidate_schedule)
              if candidate_schedule else math.inf),
          "candidate_to_baseline_ratio": (
              statistics.median(candidate_schedule) /
              statistics.median(baseline_schedule)
              if baseline_schedule else math.inf),
          "paired_non_regression_inference": inference,
          "timing_role": "paired isolated all-40 router component only",
      },
      "checks": checks,
      "next_action": (
          "Wire the row-major packages into a static fully-connected manager "
          "for T>1 while preserving oneDNN for T=1; prove isolated selection."
          if direct_admitted else
          "Use one plugin-internal constant reorder to the admitted group-major "
          "metadata layout, then prove static T>1 selection without graph If."
          if required_passed else
          "Repair the fused row-major source or evidence boundary before a "
          "plugin build."),
  }
  write_json(output / "metrics.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA, "workstream": WS,
      "required_checks_passed": required_passed,
      "direct_row_major_provider_admitted": direct_admitted,
      "metrics": "metrics.json", "raw": "raw/"})
  print(json.dumps({
      "output": str(output), "verdict": verdict,
      "required_checks_passed": required_passed,
      "direct_row_major_provider_admitted": direct_admitted,
      "failed_checks": [row["name"] for row in checks if not row["pass"]],
      "complete_router_schedule": result["complete_router_schedule"],
  }, indent=2, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
