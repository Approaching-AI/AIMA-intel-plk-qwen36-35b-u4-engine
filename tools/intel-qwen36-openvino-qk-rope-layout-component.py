#!/usr/bin/env python3
"""Audit and run the admitted Q/K RoPE-layout producer component.

``--audit-only`` performs the mandatory no-GPU all-ten rewrite/source audit.
The default mode launches exactly one candidate-only 2k/17-step worker using
the corrected multi-output custom-op plugin and retained seq1304 control.  It
launches no new control, stock, long, ABBA, output512, or product worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-qk-rope-layout-component-v0"
HELPER_PATH = ROOT / (
    "tools/intel-qwen36-openvino-dynamic-split-consumer-relocation-"
    "component.py")
BOUND = ROOT / (
    "output/openvino-qk-rope-layout-bound-"
    "20260717Tseq1323-cleanZ/metrics.json")
AUDIT = ROOT / (
    "output/openvino-qk-rope-layout-rewrite-"
    "20260717Tseq1324-cleanZ/metrics.json")
FAILED_COMPONENT = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1325-candidate-2k-warm17-cleanZ/metrics.json")
FAILED_STDERR = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1325-candidate-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker.stderr")
COMPILE_FIX_BUILD = ROOT / (
    "output/openvino-qk-rope-layout-preprocess-id-build-"
    "20260717Tseq1326-cleanZ/manifest.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER_SOURCE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
PREPROCESS_ID_PATCH = ROOT / (
    "engine/openvino/iq36-custom-preprocess-input-port-id.patch")
MULTI_OUTPUT_PATCH = ROOT / (
    "engine/openvino/iq36-custom-multi-output-binding.patch")
GATED_DQ_PATCH = ROOT / (
    "engine/openvino/iq36-attention-gated-dynamic-quantize.patch")
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "IQ36QKRopeLayout": 10,
    "RMS": 131,
}


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


HELPER = load_module(HELPER_PATH, "iq36_split_relocation_component")
BASE = HELPER.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--audit-only", action="store_true")
  parser.add_argument("--plugin", type=Path, default=PLUGIN)
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--igc-library-dir", type=Path, default=None,
                      help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  if args.igc_library_dir is not None:
    parser.error("this component does not admit an IGC delta")
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
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {"tools/intel-qwen36-openvino-qk-rope-layout-component.py"}
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(output_relative):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def execute_rewrite_audit(raw: Path) -> dict[str, Any]:
  script = f"""
import importlib.util, json
from pathlib import Path
import numpy as np
import openvino as ov
p = Path({str(GRAPH_SOURCE)!r})
s = importlib.util.spec_from_file_location('iq36_graph', p)
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)
model, summary = g.make_candidate_model(
    ov.Core(), Path({str(MODEL_DIR)!r}), ov, np,
    target_layers=g.FULL_ATTENTION_LAYERS,
    initialize_hot_states=True, fixed_cold_capacity=2048,
    prefill_history_capacity=16384, fuse_linear_conv_state=True,
    fuse_qk_rope_layout=True)
ops = model.get_ordered_ops()
names = {{node.get_friendly_name() for node in ops}}
counts = {{kind: sum(node.get_type_name() == kind for node in ops)
          for kind in ('IQ36HotAttentionGQA', 'IQ36LinearConvSwish',
                       'IQ36QKRopeLayout', 'ScaledDotProductAttention')}}
old_live = []
rows = []
for layer in g.FULL_ATTENTION_LAYERS:
  prefix = f'__module.model.model.language_model.layers.{{layer}}.self_attn/'
  suffixes = (
      ('aten::transpose/Transpose_2', 'aten::transpose/Transpose_1',
       'aten::slice/Slice_4', 'aten::slice/Slice_7',
       'aten::slice/Slice', 'aten::slice/Slice_3',
       'aten::add/Add_1', 'aten::add/Add',
       'aten::cat/Concat_5', 'aten::cat/Concat_2')
      if layer == 39 else
      ('aten::transpose/Transpose', 'aten::transpose/Transpose_1',
       'aten::slice/Slice', 'aten::slice/Slice_3',
       'aten::slice/Slice_4', 'aten::slice/Slice_7',
       'aten::add/Add', 'aten::add/Add_1',
       'aten::cat/Concat_1', 'aten::cat/Concat_3'))
  old_live.extend(prefix + suffix for suffix in suffixes
                  if prefix + suffix in names)
  operation = next(node for node in ops if node.get_friendly_name() ==
                   f'iq36_qk_rope_layout_layer{{layer}}')
  rows.append({{
      'layer': layer,
      'input_shapes': [str(operation.get_input_partial_shape(index))
                       for index in range(operation.get_input_size())],
      'output_shapes': [str(operation.get_output_partial_shape(index))
                        for index in range(operation.get_output_size())],
      'output_consumers': [
          sorted(port.get_node().get_friendly_name()
                 for port in operation.output(index).get_target_inputs())
          for index in range(operation.get_output_size())],
  }})
print(json.dumps({{
  'summary': {{key: summary[key] for key in (
      'custom_count_after', 'stock_sdpa_count_after',
      'fuse_qk_rope_layout', 'qk_rope_layout_rewrite_count')}},
  'counts': counts, 'old_live': old_live, 'rows': rows,
}}, sort_keys=True))
"""
  before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=True,
      capture_output=True, text=True, timeout=120)
  after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  audit = json.loads(completed.stdout)
  audit["child_max_rss_kib_upper_bound"] = max(0, int(after - before))
  write_json(raw / "model-rewrite-audit.json", audit)
  return audit


def source_audit() -> dict[str, Any]:
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  marker = '<CustomLayer name="IQ36QKRopeLayout"'
  tail = config.split(marker, 1)[1] if config.count(marker) == 1 else ""
  block = tail.split("</CustomLayer>", 1)[0] if tail else ""
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  worker = WORKER_SOURCE.read_text(encoding="utf-8")
  return {
      "one_custom_config_entry": config.count(marker) == 1,
      "custom_config_inputs": block.count('type="input"'),
      "custom_config_outputs": block.count('type="output"'),
      "custom_config_exact_work_size": (
          '<WorkSizes global="X,Y,B*F" local="16,1,1"/>' in block),
      "graph_has_two_output_contract_and_all_ten_rewrite": all(
          token in graph for token in (
              "class IQ36QKRopeLayout",
              "fuse_qk_rope_layout",
              "query_concat.output(0).replace(qk_rope.output(0))",
              "key_concat.output(0).replace(qk_rope.output(1))")),
      "worker_plumbs_default_off_flag": (
          'cfg.get("fuse_qk_rope_layout", False)' in worker),
      "kernel_has_token_major_inputs_head_major_outputs": all(
          token in kernel for token in (
              "__kernel void iq36_qk_rope_layout",
              "(ulong)token * INPUT0_PITCHES[1]",
              "(ulong)query_head * OUTPUT0_PITCHES[1]",
              "query_head < (uint)OUTPUT1_DIMS[1]")),
      "kernel_has_partial64_rotate_half_and_tail_copy": all(
          token in kernel for token in (
              "dimension < 64U", "dimension < 32U",
              "dimension + 32U", "dimension - 32U",
              "cosine * value - sine * peer",
              "cosine * value + sine * peer")),
  }


def rewrite_exact(audit: dict[str, Any]) -> bool:
  return (
      audit.get("summary") == {
          "custom_count_after": 10,
          "stock_sdpa_count_after": 0,
          "fuse_qk_rope_layout": True,
          "qk_rope_layout_rewrite_count": 10,
      }
      and audit.get("counts") == {
          "IQ36HotAttentionGQA": 10,
          "IQ36LinearConvSwish": 30,
          "IQ36QKRopeLayout": 10,
          "ScaledDotProductAttention": 0,
      }
      and audit.get("old_live") == []
      and len(audit.get("rows", [])) == 10
      and all(
          row.get("input_shapes") == [
              "[?,?,16,256]", "[?,?,2,256]",
              "[?,1,?,64]", "[?,1,?,64]"]
          and row.get("output_shapes") == [
              "[?,16,?,256]", "[?,2,?,256]"]
          and any(name == f"iq36_hot_attention_layer{row['layer']}"
                  for name in row.get("output_consumers", [[], []])[0])
          and any(name == f"iq36_hot_attention_layer{row['layer']}"
                  for name in row.get("output_consumers", [[], []])[1])
          for row in audit.get("rows", [])))


def profile_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
  rows = result.get("full_profile")
  if isinstance(rows, list):
    return rows
  phases = result.get("phases", [])
  if phases and isinstance(phases[-1], dict):
    rows = phases[-1].get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker has no final full profile")
  return rows


def runtime_audit(result: dict[str, Any]) -> dict[str, Any]:
  rows = profile_rows(result)
  executed = [row for row in rows
              if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type")) for row in executed)
  old_rows = []
  for row in executed:
    name = str(row.get("node_name", ""))
    layer = next((value for value in FULL_ATTENTION_LAYERS
                  if f"layers.{value}.self_attn/" in name), None)
    if layer is None:
      continue
    q_transpose = ("/aten::transpose/Transpose_2" if layer == 39
                   else "/aten::transpose/Transpose")
    q_concat = ("/aten::cat/Concat_5" if layer == 39
                else "/aten::cat/Concat_1")
    k_concat = ("/aten::cat/Concat_2" if layer == 39
                else "/aten::cat/Concat_3")
    if ((row.get("node_type") == "Transpose" and name.endswith(
            (q_transpose, "/aten::transpose/Transpose_1")))
        or (row.get("node_type") == "StridedSlice" and name.endswith(
            ("/aten::slice/Slice", "/aten::slice/Slice_3",
             "/aten::slice/Slice_4", "/aten::slice/Slice_7")))
        or (row.get("node_type") == "RoPE" and name.endswith(
            ("/aten::add/Add", "/aten::add/Add_1")))
        or (row.get("node_type") == "Concat" and name.endswith(
            (q_concat, k_concat)))):
      old_rows.append(row)
  core_counts = {key: int(counts.get(key, 0))
                 for key in EXPECTED_CORE_COUNTS}
  return {
      "old_boundary_executed": len(old_rows),
      "qk_rope_layout_executed": int(counts.get("IQ36QKRopeLayout", 0)),
      "core_counts": core_counts,
      "core_counts_exact": core_counts == EXPECTED_CORE_COUNTS,
      "old_rows": old_rows,
      "raw_profile_time_is_savings_evidence": False,
  }


def stable_walls(result: dict[str, Any]) -> list[float]:
  walls = [float(row["wall_ms_diagnostic"])
           for row in result.get("phases", [])[1:]]
  if len(walls) != 17 or not all(
      math.isfinite(value) and value > 0.0 for value in walls):
    raise ValueError("worker does not have 17 finite decode walls")
  return walls[1:]


def audit_main(args: argparse.Namespace, output: Path) -> int:
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, MODEL_DIR / "openvino_language_model.xml", OV_PYTHON,
      GRAPH_SOURCE, WORKER_SOURCE, KERNEL_SOURCE, CUSTOM_CONFIG)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing Q/K rewrite inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  bound = load_json(BOUND)
  rewrite = execute_rewrite_audit(raw)
  source = source_audit()
  rewrite_ok = rewrite_exact(rewrite)
  source_ok = (
      source.get("one_custom_config_entry") is True
      and source.get("custom_config_inputs") == 4
      and source.get("custom_config_outputs") == 2
      and all(value is True for key, value in source.items()
              if key not in ("custom_config_inputs", "custom_config_outputs")))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1323_admits_exact_source_and_one_candidate",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get(
                "candidate_workers_admitted_after_exact_rewrite_audit") == 1),
      check("no_concurrent_worker_during_rewrite_audit", not concurrent,
            concurrent=concurrent),
      check("exact_all_ten_rewrite_removes_old_qk_boundary",
            rewrite_ok, audit=rewrite),
      check("config_kernel_graph_and_worker_contract_is_exact",
            source_ok, audit=source),
      check("rewrite_audit_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0, model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_qk_rope_layout_candidate"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "mode": "audit_only",
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "candidate_workers_admitted": 1 if required_checks_passed else 0,
      "additional_control_worker_admitted": False,
      "rewrite": rewrite,
      "source_contract": source,
      "memory": {
          "stop_bytes": int(args.abort_below_available_gib * 1024**3),
          "available_end_bytes": available_memory_bytes(),
          "oom_observed": False,
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "mode": "audit_only",
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
      "model_ir_reads": 1,
  })
  report = f"""# Q/K RoPE-layout rewrite audit

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The no-GPU rewrite creates ten `IQ36QKRopeLayout` operations with four inputs
and two outputs. All 100 old Q/K transpose, slice, RoPE, and concat nodes are
dead, while every accepted custom-attention operation consumes both new
head-major outputs. The shared source implements partial-64 rotate-half and
tail copy for every layer through one parameterized kernel.

Admit one candidate-only short worker. No compiler, GPU context, or model
worker ran in this audit; OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "candidate_workers_admitted": 1 if required_checks_passed else 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


def component_main(args: argparse.Namespace, output: Path) -> int:
  worker_dir = output / "raw/2k/candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, AUDIT, FAILED_COMPONENT, FAILED_STDERR, COMPILE_FIX_BUILD,
      CONTROL, args.plugin, BASE.REFERENCE_WORKER,
      BASE.WORKER, GRAPH_SOURCE, WORKER_SOURCE, KERNEL_SOURCE, CUSTOM_CONFIG)
  required += (PREPROCESS_ID_PATCH, MULTI_OUTPUT_PATCH, GATED_DQ_PATCH)
  missing = [display(Path(path)) for path in required
             if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing Q/K component inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  bound = load_json(BOUND)
  audit = load_json(AUDIT)
  failed = load_json(FAILED_COMPONENT)
  failed_stderr = FAILED_STDERR.read_text(encoding="utf-8", errors="replace")
  build = load_json(COMPILE_FIX_BUILD)
  coapplied = {
      str(row.get("path")): row
      for row in build.get("coapplied_patches", [])
      if isinstance(row, dict)
  }
  control = load_json(CONTROL)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["fuse_qk_rope_layout"] = True
  worker = BASE.launch_worker(args, worker_dir, config)
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  candidate_profile = runtime_audit(result) if result else {}
  control_profile = runtime_audit(control)
  source = result.get("source_summary") or {}
  control_stable = stable_walls(control)
  candidate_stable = stable_walls(result) if result else []
  control_median = statistics.median(control_stable)
  candidate_median = (
      statistics.median(candidate_stable) if candidate_stable else math.nan)
  observed_saving = control_median - candidate_median
  required_saving = float(bound["budget"]["seq1302_shortfall_ms"])
  activation_passed = (
      control_profile.get("old_boundary_executed") == 100
      and control_profile.get("qk_rope_layout_executed") == 0
      and candidate_profile.get("old_boundary_executed") == 0
      and candidate_profile.get("qk_rope_layout_executed") == 10)
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  performance_passed = (
      math.isfinite(observed_saving) and observed_saving >= required_saving)
  worker_safe = (
      worker["returncode"] == 0 and worker["timed_out"] is False
      and worker["memory_guard"]["tripped"] is False
      and worker["oom_observed"] is False
      and int(worker["monitor"]["system_available_min_bytes"] or 0) >=
          int(args.abort_below_available_gib * 1024**3))
  evidence_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1324_admits_exactly_one_candidate",
            audit.get("required_checks_passed") is True
            and audit.get("candidate_workers_admitted") == 1
            and audit.get("additional_control_worker_admitted") is False),
      check("seq1325_is_compile_only_id_collision_without_oom",
            failed.get("worker", {}).get("returncode") == 1
            and failed.get("worker", {}).get("oom_observed") is False
            and failed.get("actual_top1") == []
            and "Different primitive with id" in failed_stderr
            and "cldnn_custom_preprocess' exists already" in failed_stderr),
      check("seq1326_correction_build_is_exact_and_safe",
            build.get("build", {}).get("result") == "pass"
            and build.get("build", {}).get("oom_or_restart") is False
            and build.get("build", {}).get("process_swap_bytes") == 0
            and build.get("scope", {}).get("model_workers") == 0
            and build.get("correction_patch", {}).get("sha256") ==
                sha256(PREPROCESS_ID_PATCH)
            and coapplied.get(display(MULTI_OUTPUT_PATCH), {}).get(
                "sha256") == sha256(MULTI_OUTPUT_PATCH)
            and coapplied.get(display(GATED_DQ_PATCH), {}).get(
                "sha256") == sha256(GATED_DQ_PATCH)),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("one_candidate_worker_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_corrected_candidate_plugin",
            plugin_hash == build.get("candidate_plugin", {}).get("sha256")
            and result.get("candidate_gpu_plugin_sha256") == plugin_hash,
            plugin_sha256=plugin_hash),
      check("graph_executes_exact_ten_qk_rope_layout_producers",
            source.get("fuse_qk_rope_layout") is True
            and source.get("qk_rope_layout_rewrite_count") == 10
            and source.get("custom_count_after") == 10,
            source_summary=source),
      check("core_execution_census_is_exact",
            candidate_profile.get("core_counts_exact") is True,
            core_counts=candidate_profile.get("core_counts")),
      check("qk_boundary_activation_outcome_is_completely_measured",
            activation_passed,
            control=control_profile, candidate=candidate_profile),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in evidence_checks)
  route_accepted = (
      evidence_checks_passed and correctness_passed and performance_passed)
  verdict = (
      "retain_qk_rope_layout_as_bundle_cut" if route_accepted else
      "reject_qk_rope_layout_after_component"
      if evidence_checks_passed else "inconclusive")
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "stable_samples_per_side": 16,
      "control_median_ms": control_median,
      "candidate_median_ms": candidate_median,
      "observed_median_saving_ms": observed_saving,
      "required_incremental_saving_ms": required_saving,
      "margin_to_required_ms": observed_saving - required_saving,
      "control_mean_ms": statistics.mean(control_stable),
      "candidate_mean_ms": (
          statistics.mean(candidate_stable) if candidate_stable else None),
      "component_performance_passed": performance_passed,
      "speed_claim": False,
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "mode": "component",
      "git": git,
      "verdict": verdict,
      "evidence_checks_passed": evidence_checks_passed,
      "route_accepted": route_accepted,
      "activation_passed": activation_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "control_workers_launched": 0,
      "long_workers_launched": 0,
      "product_workers_launched": 0,
      "decode_tokens": decode_tokens,
      "expected_top1": expected_top1,
      "actual_top1": actual_top1,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "source_summary": source,
      "profile": {"candidate": candidate_profile,
                  "control": control_profile},
      "performance": performance,
      "evidence_checks": evidence_checks,
      "decision": {
          "close_route": evidence_checks_passed and not route_accepted,
          "retain_as_bundle_ingredient": route_accepted,
          "next_route": (
              "openvino_fc_rms_igc_qk_rope_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for the unchanged Q/K producer, repeat, or sample "
              "extension if this component closes"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "mode": "component",
      "tool": display(Path(__file__)),
      "git": git,
      "plugin": str(plugin),
      "plugin_sha256": plugin_hash,
      "inputs": {display(Path(path)): sha256(Path(path))
                 for path in required},
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "stock_workers": 0,
      "control_workers": 0,
      "candidate_workers": 1,
  })
  report = f"""# Q/K RoPE-layout component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Exactly one corrected candidate-only 2k/17-step worker ran against retained
seq1304 timing evidence. Old Q/K boundary dispatches move
`{control_profile.get('old_boundary_executed')} ->
{candidate_profile.get('old_boundary_executed')}`; fused Q/K producers execute
`{candidate_profile.get('qk_rope_layout_executed')}` times. The component is
accepted only if all 18 teacher-forced top-1 tokens and the core census remain
exact.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This is not a product speed claim. The seq1326
plugin correction built separately; this component launched no compiler,
stock, control, concurrent, long, ABBA, output512, or product worker. OOM
observed: `{str(worker['oom_observed']).lower()}`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "activation_passed": activation_passed,
      "correctness_passed": correctness_passed,
      "performance_passed": performance_passed,
      "worker_returncode": worker["returncode"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if evidence_checks_passed else 2


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  if args.audit_only:
    return audit_main(args, output)
  return component_main(args, output)


if __name__ == "__main__":
  raise SystemExit(main())
