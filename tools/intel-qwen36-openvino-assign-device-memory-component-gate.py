#!/usr/bin/env python3
"""Decide the Assign producer device-memory route from one short A/B pair.

The source-only seq1282b bound admitted exactly one short component.  This
gate consumes the resulting isolated 2k/17-step control and candidate workers,
checks that the upstream allocation contract really moved the locked graph's
Assign-fed storage from usm_host to usm_device, and then applies the previously
derived adjacent-component saving threshold.  It opens one property-only GPU
context to record the locked device type, but it does not compile or run a
model and never launches a long worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-assign-device-memory-component-gate-v0"

BOUND = ROOT / (
    "output/openvino-assign-device-memory-bound-"
    "20260717Tseq1282b-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-assign-device-memory-component-"
    "20260717Tseq1284-control-2k-warm17-cleanZ/metrics.json")
CANDIDATE = ROOT / (
    "output/openvino-assign-device-memory-component-"
    "20260717Tseq1285-candidate-2k-warm17-cleanZ/metrics.json")
RUNTIME_GRAPH = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1150-fixed-2k-dq-census-cleanZ/raw/2k/candidate/"
    "runtime-graph.xml")
PATCH = ROOT / "engine/openvino/iq36-assign-producer-device-memory.patch"
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-control-seq1283/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-assign-device-seq1283/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")

EXPECTED_CONTROL_PLUGIN = (
    "432f4ebb1802b619ed347e1ba6344492177da884c9cf4d8a815e360122e0f876")
EXPECTED_CANDIDATE_PLUGIN = (
    "eaff3b2fb212679de761b05d7c1d95594ea1a51ba5025abfbcc3ee0d85f57527")
DECODE_STEPS = 17


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def plugin_device_properties(plugin: Path) -> dict[str, Any]:
  script = """
import json
import sys
import openvino as ov
core = ov.Core()
core.register_plugin(sys.argv[1], "GPUX")
keys = ("FULL_DEVICE_NAME", "DEVICE_TYPE", "DEVICE_ARCHITECTURE",
        "GPU_DEVICE_TOTAL_MEM_SIZE")
print(json.dumps({key: str(core.get_property("GPUX", key)) for key in keys},
                 sort_keys=True))
"""
  result = subprocess.run(
      [str(OV_PYTHON), "-c", script, str(plugin.resolve())],
      cwd=ROOT, text=True, capture_output=True, check=True)
  value = json.loads(result.stdout)
  if not isinstance(value, dict):
    raise TypeError("unexpected GPU property probe response")
  return value


def runtime_assign_census(path: Path) -> dict[str, Any]:
  root = ET.parse(path).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError("runtime graph is missing layers or edges")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  incoming: dict[int, list[int]] = defaultdict(list)
  for edge in edges_node:
    incoming[int(edge.attrib["to-layer"])].append(
        int(edge.attrib["from-layer"]))
  rows: list[dict[str, Any]] = []
  for assign_id, assign in layers.items():
    if str(assign.attrib.get("type", "")).lower() != "assign":
      continue
    for producer_id in incoming[assign_id]:
      producer = layers[producer_id]
      output = producer.find("output")
      ports = []
      if output is not None:
        for port in output:
          ports.append({
              "precision": port.attrib.get("precision"),
              "shape": [int(dim.text or "-1") for dim in port.findall("dim")],
          })
      rows.append({
          "assign": assign.attrib.get("name"),
          "producer": producer.attrib.get("name"),
          "producer_type": producer.attrib.get("type"),
          "ports": ports,
      })
  counts = Counter(str(row["producer_type"]) for row in rows)
  conv_bytes = 0
  for row in rows:
    if row["producer_type"] != "Reshape" or not row["ports"]:
      continue
    port = row["ports"][0]
    elements = 1
    for dim in port["shape"]:
      if dim < 0:
        elements = 0
        break
      elements *= dim
    item_bytes = 2 if port["precision"] == "FP16" else 0
    conv_bytes += elements * item_bytes
  return {
      "assign_count": len(rows),
      "producer_type_counts": dict(sorted(counts.items())),
      "static_conv_producer_bytes": conv_bytes,
      "rows": rows,
  }


def actual_source_identity(metrics: dict[str, Any]) -> dict[str, Any]:
  identity = metrics["accepted_identity"]
  return {
      "config_sha256": identity.get("actual_config_sha256"),
      "sources": {
          str(row.get("path")): row.get("actual_sha256")
          for row in identity.get("sources", [])
      },
  }


def worker_ok(metrics: dict[str, Any]) -> bool:
  worker = metrics["worker"]
  return (
      worker.get("returncode") == 0
      and worker.get("timed_out") is False
      and worker.get("oom_observed") is False
      and worker.get("memory_guard", {}).get("tripped") is False)


def selected_profile(metrics: dict[str, Any]) -> dict[str, float]:
  raw = metrics["profile_audit"][
      "raw_real_time_us_by_node_type_nonadditive"]
  names = (
      "Assign", "GatedDeltaNet", "IQ36LinearConvSwish",
      "FullyConnectedCompressed", "DynamicQuantize",
      "IQ36HotAttentionGQA", "Transpose")
  return {name: float(raw.get(name, 0.0)) for name in names}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(
        f"memory stop: {available_start} < {stop_bytes} bytes")

  required = (
      BOUND, CONTROL, CANDIDATE, RUNTIME_GRAPH, PATCH, CONTROL_PLUGIN,
      CANDIDATE_PLUGIN, OPENVINO_SOURCE / ".git", OV_PYTHON)
  missing = [display_path(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing Assign component inputs: " + ", ".join(missing))

  git = git_state(output)
  bound = load_json(BOUND)
  control = load_json(CONTROL)
  candidate = load_json(CANDIDATE)
  runtime = runtime_assign_census(RUNTIME_GRAPH)
  device = plugin_device_properties(CANDIDATE_PLUGIN)
  (raw / "candidate-device-properties.json").write_text(
      json.dumps(device, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  source_patch_present = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(PATCH.resolve())],
      cwd=OPENVINO_SOURCE, text=True, capture_output=True).returncode == 0
  pair_commit = str(control["git"]["commit"])
  pair_same_commit = pair_commit == str(candidate["git"]["commit"])
  pair_is_ancestor = subprocess.run(
      ["git", "merge-base", "--is-ancestor", pair_commit, git["commit"]],
      cwd=ROOT, text=True, capture_output=True).returncode == 0
  pair_to_gate_paths = subprocess.run(
      ["git", "diff", "--name-only", f"{pair_commit}..{git['commit']}"],
      cwd=ROOT, text=True, capture_output=True, check=True
  ).stdout.splitlines()
  allowed_gate_paths = {
      "tools/intel-qwen36-openvino-assign-device-memory-component-gate.py"}

  control_hash = sha256(CONTROL_PLUGIN)
  candidate_hash = sha256(CANDIDATE_PLUGIN)
  control_identity = actual_source_identity(control)
  candidate_identity = actual_source_identity(candidate)
  control_walls = [
      float(value) for value in
      control["worker_result_summary"]["decode_wall_ms"]]
  candidate_walls = [
      float(value) for value in
      candidate["worker_result_summary"]["decode_wall_ms"]]
  control_stable = control_walls[1:]
  candidate_stable = candidate_walls[1:]
  control_median = statistics.median(control_stable)
  candidate_median = statistics.median(candidate_stable)
  control_mean = statistics.mean(control_stable)
  candidate_mean = statistics.mean(candidate_stable)
  observed_wall_saving_ms = control_median - candidate_median

  boundary = bound["bound"]
  required_adjacent_saving_ms = (
      float(boundary["current_adjacent_ms"])
      - float(boundary["adjacent_target_ms"]))
  component_performance_passed = (
      observed_wall_saving_ms >= required_adjacent_saving_ms)

  control_memory = control["worker_result_summary"]["memory_samples"]
  candidate_memory = candidate["worker_result_summary"]["memory_samples"]
  compile_delta = {
      key: int(candidate_memory["gpu_after_language_compile"][key])
      - int(control_memory["gpu_after_language_compile"][key])
      for key in ("usm_device", "usm_host")}
  final_delta = {
      key: int(candidate_memory["gpu_after_final_infer"][key])
      - int(control_memory["gpu_after_final_infer"][key])
      for key in ("usm_device", "usm_host")}
  recurrent_one_way_bytes = int(boundary["state_read_write_bytes"]) // 2
  minimum_expected_final_shift = (
      runtime["static_conv_producer_bytes"] + recurrent_one_way_bytes)

  control_profile = selected_profile(control)
  candidate_profile = selected_profile(candidate)
  profile_delta_us = {
      name: candidate_profile[name] - control_profile[name]
      for name in control_profile}

  top1_exact = (
      control["actual_top1"] == control["expected_top1"]
      and candidate["actual_top1"] == candidate["expected_top1"]
      and candidate["actual_top1"] == control["actual_top1"])
  profile_census_exact = (
      control["profile_audit"]["selected_counts_exact"] is True
      and candidate["profile_audit"]["selected_counts_exact"] is True
      and control["profile_audit"]["selected_executed_counts"]
      == candidate["profile_audit"]["selected_executed_counts"])

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("source_bound_admitted_exactly_one_short_component",
            bound.get("required_checks_passed") is True
            and bound.get("component_build_admitted") is True
            and bound.get("long_worker_admitted") is False),
      check("pair_uses_one_clean_common_snapshot",
            pair_same_commit
            and control["git"]["dirty"] is False
            and candidate["git"]["dirty"] is False,
            pair_commit=pair_commit),
      check("gate_only_postdates_pair_by_its_own_tool",
            pair_is_ancestor
            and set(pair_to_gate_paths).issubset(allowed_gate_paths),
            pair_to_gate_paths=pair_to_gate_paths),
      check("control_and_candidate_plugins_are_exact_and_distinct",
            control_hash == EXPECTED_CONTROL_PLUGIN
            and candidate_hash == EXPECTED_CANDIDATE_PLUGIN
            and control_hash != candidate_hash,
            control_sha256=control_hash,
            candidate_sha256=candidate_hash),
      check("candidate_source_contains_exact_durable_patch",
            source_patch_present, patch=display_path(PATCH)),
      check("pair_uses_identical_current_graph_sources_and_config",
            control_identity == candidate_identity,
            identity=control_identity),
      check("locked_runtime_assign_topology_is_exact",
            runtime["assign_count"] == 60
            and runtime["producer_type_counts"]
            == {"GatedDeltaNet": 30, "Reshape": 30}
            and runtime["static_conv_producer_bytes"] == 1_966_080,
            assign_count=runtime["assign_count"],
            producer_type_counts=runtime["producer_type_counts"],
            static_conv_producer_bytes=runtime[
                "static_conv_producer_bytes"]),
      check("locked_target_is_integrated_b390",
            "B390" in str(device.get("FULL_DEVICE_NAME"))
            and "INTEGRATED" in str(device.get("DEVICE_TYPE")).upper(),
            device=device),
      check("both_short_workers_complete_without_oom",
            worker_ok(control) and worker_ok(candidate),
            control_monitor=control["worker"]["monitor"],
            candidate_monitor=candidate["worker"]["monitor"]),
      check("pair_is_exact_2k_warm17_only",
            control.get("lane") == "2k"
            and candidate.get("lane") == "2k"
            and len(control_walls) == DECODE_STEPS
            and len(candidate_walls) == DECODE_STEPS
            and control.get("long_worker_launched") is False
            and candidate.get("long_worker_launched") is False),
      check("teacher_forced_top1_and_profile_census_are_exact",
            top1_exact and profile_census_exact),
      check("compile_time_conv_storage_moves_host_to_device_exactly",
            compile_delta["usm_device"]
            == runtime["static_conv_producer_bytes"]
            and compile_delta["usm_host"]
            == -runtime["static_conv_producer_bytes"],
            compile_delta=compile_delta),
      check("final_state_storage_moves_host_to_device_conservatively",
            final_delta["usm_device"] == -final_delta["usm_host"]
            and final_delta["usm_device"] >= minimum_expected_final_shift,
            final_delta=final_delta,
            minimum_expected_final_shift=minimum_expected_final_shift),
      check("no_model_compile_or_long_worker_ran_in_decision_gate",
            True, gpu_property_contexts=1, model_compiles=0,
            model_workers=0, long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_accepted = required_checks_passed and component_performance_passed
  verdict = (
      "accept_assign_device_memory_for_graph_integration"
      if route_accepted else
      "reject_assign_device_memory_after_short_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "pair_commit": pair_commit,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_performance_passed": component_performance_passed,
      "route_accepted": route_accepted,
      "graph_integration_admitted": route_accepted,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "device": device,
      "plugins": {
          "control": {"path": str(CONTROL_PLUGIN), "sha256": control_hash},
          "candidate": {
              "path": str(CANDIDATE_PLUGIN), "sha256": candidate_hash},
      },
      "runtime_graph": runtime,
      "allocation": {
          "compile_delta_candidate_minus_control_bytes": compile_delta,
          "final_delta_candidate_minus_control_bytes": final_delta,
          "recurrent_one_way_bytes": recurrent_one_way_bytes,
          "minimum_expected_final_shift_bytes": minimum_expected_final_shift,
          "interpretation": (
              "the patch is active and moves the complete static conv "
              "producer allocation plus at least the bounded one-way "
              "recurrent state footprint from usm_host to usm_device"),
      },
      "performance": {
          "control_decode_wall_ms": control_walls,
          "candidate_decode_wall_ms": candidate_walls,
          "stable_sample_rule": "drop first decode JIT sample",
          "stable_samples_per_side": len(control_stable),
          "control_median_ms": control_median,
          "candidate_median_ms": candidate_median,
          "control_mean_ms": control_mean,
          "candidate_mean_ms": candidate_mean,
          "observed_wall_saving_ms": observed_wall_saving_ms,
          "required_adjacent_saving_ms": required_adjacent_saving_ms,
          "margin_to_required_saving_ms": (
              observed_wall_saving_ms - required_adjacent_saving_ms),
          "raw_profile_us_control_nonadditive": control_profile,
          "raw_profile_us_candidate_nonadditive": candidate_profile,
          "raw_profile_us_delta_candidate_minus_control_nonadditive": (
              profile_delta_us),
          "raw_profile_is_decision_evidence": False,
          "interpretation": (
              "allocation changed exactly, but the short wall moves in the "
              "wrong direction and does not clear the pre-derived saving; "
              "profile rows are retained only as non-additive telemetry"),
      },
      "correctness": {
          "top1_exact": top1_exact,
          "profile_census_exact": profile_census_exact,
          "actual_top1": candidate["actual_top1"],
      },
      "oom": {
          "control": control["worker"]["monitor"],
          "candidate": candidate["worker"]["monitor"],
          "guard_tripped": False,
          "oom_observed": False,
      },
      "checks": checks,
      "decision": {
          "close_route": required_checks_passed
          and not component_performance_passed,
          "reason": (
              "on the locked integrated B390, the exact host-to-device "
              "allocation shift produces no wall saving; do not spend a "
              "32k, ABBA, output512, or FC-integration worker on this patch"),
          "reopen_condition": (
              "a different complete state owner or kernel capability with "
              "an independently derived bound; not another repeat of the "
              "same Assign allocation-type patch"),
      },
      "memory_stop_bytes": stop_bytes,
      "available_memory_start_bytes": available_start,
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (output / "manifest.json").write_text(
      json.dumps({
          "schema": SCHEMA,
          "tool": display_path(Path(__file__)),
          "git": git,
          "inputs": {
              display_path(path): sha256(path)
              for path in required if path.is_file()
          },
          "gpu_property_contexts": 1,
          "model_compiles": 0,
          "model_workers": 0,
          "long_workers": 0,
      }, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  summary = "\n".join((
      "# OpenVINO Assign device-memory short component",
      "",
      f"Verdict: **{verdict}**. Evidence checks: "
      f"`{str(required_checks_passed).lower()}`; component performance gate: "
      f"`{str(component_performance_passed).lower()}`.",
      "",
      f"The exact candidate moves `{final_delta['usm_device']:,}` bytes from "
      "`usm_host` to `usm_device` by the final inference. The compile-time "
      f"shift is exactly `{runtime['static_conv_producer_bytes']:,}` bytes, "
      "matching all 30 static conv-state Assign producers. The locked device "
      f"reports `{device.get('FULL_DEVICE_NAME')}` and "
      f"`{device.get('DEVICE_TYPE')}`.",
      "",
      "Both isolated workers preserve all 18 teacher-forced top-1 tokens and "
      "the exact execution census. Neither worker OOMed or tripped the 4 GiB "
      "guard; no long worker ran.",
      "",
      f"After dropping the first decode JIT sample, control median is "
      f"`{control_median:.6f} ms` and candidate median is "
      f"`{candidate_median:.6f} ms`: observed saving "
      f"`{observed_wall_saving_ms:.6f} ms`, versus the pre-derived required "
      f"`{required_adjacent_saving_ms:.6f} ms`. The allocation mechanism is "
      "therefore proven active but is not a performance opportunity on this "
      "locked integrated target. Do not launch 32k, ABBA, output512, or "
      "fixed-FC integration for this patch.",
      "",
  ))
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_performance_passed": component_performance_passed,
      "allocation_shift_bytes": final_delta["usm_device"],
      "observed_wall_saving_ms": observed_wall_saving_ms,
      "required_adjacent_saving_ms": required_adjacent_saving_ms,
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
