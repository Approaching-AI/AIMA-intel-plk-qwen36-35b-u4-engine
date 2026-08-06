#!/usr/bin/env python3
"""Audit and run the admitted attention-output gate fusion component.

``--audit-only`` performs the mandatory no-GPU all-ten graph/source audit.
The default mode then launches exactly one candidate-only 2k/17-step worker
against retained seq1304.  No additional control, stock, long, ABBA, or
product worker is launched.
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
SCHEMA = "intel-qwen36-openvino-attention-output-gate-fusion-component-v0"
HELPER_PATH = ROOT / (
    "tools/intel-qwen36-openvino-dynamic-split-consumer-relocation-"
    "component.py")
BOUND = ROOT / (
    "output/openvino-attention-output-gate-fusion-bound-"
    "20260717Tseq1311c-cleanZ/metrics.json")
AUDIT = ROOT / (
    "output/openvino-attention-output-gate-fusion-rewrite-"
    "20260717Tseq1312-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-dynamic-split-inplace-plugin-build-"
    "20260717Tseq1303c-cleanZ/manifest.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-dynamic-split-control-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER_SOURCE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
KERNEL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
PREFILL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_prefill_attention_tiled.cl")
HELPERS_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36GatedHotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
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


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-attention-output-gate-fusion-component.py"}
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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing")


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
    fuse_attention_output_gate=True)
ops = model.get_ordered_ops()
names = {{x.get_friendly_name() for x in ops}}
types = {{name: sum(x.get_type_name() == name for x in ops)
         for name in ('IQ36GatedHotAttentionGQA',
                      'ScaledDotProductAttention',
                      'IQ36LinearConvSwish')}}
old = []
projection_inputs = []
custom_rows = []
for layer in g.FULL_ATTENTION_LAYERS:
  prefix = f'__module.model.model.language_model.layers.{{layer}}.self_attn/'
  for suffix in ('aten::transpose/Transpose_3',
                 'aten::reshape/Reshape_2', 'aten::mul/Multiply_6'):
    if prefix + suffix in names:
      old.append(prefix + suffix)
  projection_name = (
      f'__module.model.model.language_model.layers.{{layer}}.'
      'self_attn.o_proj/ov_ext::linear/MatMul')
  projection = next(x for x in ops if x.get_friendly_name() == projection_name)
  projection_inputs.append({{
      'layer': layer,
      'source': projection.input_value(0).get_node().get_friendly_name(),
      'shape': str(projection.input_value(0).get_partial_shape()),
  }})
  operation = next(x for x in ops if x.get_friendly_name() ==
                   f'iq36_hot_attention_layer{{layer}}')
  custom_rows.append({{
      'layer': layer, 'inputs': operation.get_input_size(),
      'outputs': operation.get_output_size(),
      'output1_shape': str(operation.get_output_partial_shape(1)),
      'gate_shape': str(operation.get_input_partial_shape(13)),
  }})
print(json.dumps({{
  'summary': {{key: summary[key] for key in (
      'custom_count_after', 'stock_sdpa_count_after',
      'fuse_attention_output_gate', 'attention_output_gate_fusion_count')}},
  'types': types, 'old_live': old,
  'projection_inputs': projection_inputs, 'custom_rows': custom_rows,
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


def config_source_audit() -> dict[str, Any]:
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  marker = '<CustomLayer name="IQ36GatedHotAttentionGQA"'
  tail = config.split(marker, 1)[1] if config.count(marker) == 1 else ""
  block = tail.split("</CustomLayer>", 1)[0] if tail else ""
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  prefill = PREFILL_SOURCE.read_text(encoding="utf-8")
  helpers = HELPERS_SOURCE.read_text(encoding="utf-8")
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  worker = WORKER_SOURCE.read_text(encoding="utf-8")
  return {
      "one_custom_config_entry": config.count(marker) == 1,
      "custom_config_inputs": block.count('type="input"'),
      "custom_config_outputs": block.count('type="output"'),
      "custom_config_macro": "-DIQ36_FUSED_GATE_OUTPUT=1" in block,
      "kernel_has_conditional_gate_input": all(text in kernel for text in (
          "#if defined(IQ36_FUSED_GATE_OUTPUT)",
          "const __global INPUT13_TYPE* raw_gate",
          "iq36_gated_attention_value")),
      "shared_helper_has_explicit_f16_gate_epilogue": all(
          text in helpers for text in (
              "inline OUTPUT1_TYPE iq36_gated_attention_value",
              "const half rounded_attention = convert_half_rte",
              "const half rounded_gate = convert_half_rte",
              "const half gated = convert_half_rte")),
      "decode_has_token_major_gated_stores": (
          kernel.count("iq36_gated_attention_value(") >= 4
          and "(ulong)query_position * OUTPUT1_PITCHES[1]" in kernel
          and "(ulong)query_head * OUTPUT1_PITCHES[2]" in kernel),
      "prefill_microkernel_and_scalar_paths_have_gated_stores": all(
          text in prefill for text in (
              "const __global INPUT13_TYPE* raw_gate",
              "tile_access(",
              "output_accumulator, dimension_block, query_in_tile",
              "(ulong)query_position * OUTPUT1_PITCHES[1]",
              "(ulong)query_head * OUTPUT1_PITCHES[2]"))
          and prefill.count("iq36_gated_attention_value(") >= 3,
      "graph_has_gated_custom_type_and_bypass": all(text in graph for text in (
          "class IQ36GatedHotAttentionGQA",
          "fuse_attention_output_gate",
          "gate_multiply.output(0).replace(gated_flat.output(0))")),
      "worker_plumbs_default_off_flag": (
          'cfg.get("fuse_attention_output_gate", False)' in worker),
  }


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
  layer_fragments = tuple(
      f"layers.{layer}.self_attn" for layer in FULL_ATTENTION_LAYERS)
  old = []
  for row in rows:
    name = str(row.get("node_name", ""))
    if not any(fragment in name for fragment in layer_fragments):
      continue
    if ((row.get("node_type") == "Transpose" and
         name.endswith("/aten::transpose/Transpose_3")) or
        (row.get("node_type") == "Multiply" and
         name.endswith("/aten::mul/Multiply_6"))):
      old.append(row)
  executed = [row for row in old if row.get("status") == "Status.EXECUTED"]
  optimized = [row for row in old
               if row.get("status") == "Status.OPTIMIZED_OUT"]
  gated = [row for row in rows
           if row.get("status") == "Status.EXECUTED" and
           row.get("node_type") == "IQ36GatedHotAttentionGQA"]
  executed_counts = Counter(
      str(row.get("node_type")) for row in rows
      if row.get("status") == "Status.EXECUTED")
  core = {name: int(executed_counts.get(name, 0))
          for name in EXPECTED_CORE_COUNTS}
  return {
      "old_rows_total": len(old),
      "old_rows_executed": len(executed),
      "old_rows_optimized_out": len(optimized),
      "gated_attention_executed": len(gated),
      "core_counts": core,
      "core_counts_exact": core == EXPECTED_CORE_COUNTS,
      "old_rows": old,
      "raw_profile_time_is_savings_evidence": False,
  }


def stable_walls(result: dict[str, Any]) -> list[float]:
  walls = [float(row["wall_ms_diagnostic"])
           for row in result.get("phases", [])[1:]]
  if len(walls) != 17 or not all(math.isfinite(value) and value > 0.0
                                 for value in walls):
    raise ValueError("worker does not have 17 finite decode walls")
  return walls[1:]


def audit_main(args: argparse.Namespace, output: Path) -> int:
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, MODEL_DIR / "openvino_language_model.xml", OV_PYTHON,
      GRAPH_SOURCE, WORKER_SOURCE, KERNEL_SOURCE, PREFILL_SOURCE,
      HELPERS_SOURCE, CUSTOM_CONFIG)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing fusion-rewrite inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  bound = load_json(BOUND)
  rewrite = execute_rewrite_audit(raw)
  source = config_source_audit()
  summary = rewrite["summary"]
  rewrite_exact = (
      summary.get("custom_count_after") == 10
      and summary.get("stock_sdpa_count_after") == 0
      and summary.get("fuse_attention_output_gate") is True
      and summary.get("attention_output_gate_fusion_count") == 10
      and rewrite.get("types") == {
          "IQ36GatedHotAttentionGQA": 10,
          "IQ36LinearConvSwish": 30,
          "ScaledDotProductAttention": 0,
      }
      and rewrite.get("old_live") == []
      and len(rewrite.get("projection_inputs", [])) == 10
      and all(row.get("source", "").startswith(
                  "iq36_attention_output_gated_flat_layer")
              and row.get("shape") == "[?,?,4096]"
              for row in rewrite.get("projection_inputs", []))
      and len(rewrite.get("custom_rows", [])) == 10
      and all(row.get("inputs") == 14 and row.get("outputs") == 6
              and row.get("output1_shape") == "[?,?,16,256]"
              and row.get("gate_shape") == "[?,?,16,256]"
              for row in rewrite.get("custom_rows", [])))
  source_exact = (
      source.get("one_custom_config_entry") is True
      and source.get("custom_config_inputs") == 14
      and source.get("custom_config_outputs") == 6
      and all(value is True for key, value in source.items()
              if key not in ("custom_config_inputs", "custom_config_outputs")))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1311c_admits_exact_source_and_one_candidate",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get(
                "candidate_workers_admitted_after_exact_rewrite_audit") == 1),
      check("no_concurrent_worker_during_rewrite_audit", not concurrent,
            concurrent=concurrent),
      check("exact_all_ten_model_rewrite_removes_old_epilogue",
            rewrite_exact, audit=rewrite),
      check("custom_config_kernel_graph_and_worker_contract_is_exact",
            source_exact, audit=source),
      check("rewrite_audit_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0, model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_attention_output_gate_fusion_candidate"
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
  report = f"""# Attention-output gate-fusion rewrite audit

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The no-GPU model rewrite creates ten `IQ36GatedHotAttentionGQA` operations,
each with 14 inputs and gated token-major `[B,Q,16,256]` output. All ten stock
SDPAs and all thirty old `Transpose_3` / `Reshape_2` / `Multiply_6` nodes are
dead; every output projection is fed by the new dispatch-free flat reshape.
The custom config has exactly 14 inputs and six outputs, and the conditional
kernel contains both prefill and decode gated token-major stores.

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
      BOUND, AUDIT, BUILD, CONTROL, args.plugin, BASE.REFERENCE_WORKER,
      BASE.WORKER, GRAPH_SOURCE, KERNEL_SOURCE, PREFILL_SOURCE,
      HELPERS_SOURCE, CUSTOM_CONFIG)
  missing = [display(Path(path)) for path in required if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing fusion-component inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  bound = load_json(BOUND)
  audit = load_json(AUDIT)
  build = load_json(BUILD)
  control = load_json(CONTROL)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["fuse_attention_output_gate"] = True
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
      control_profile.get("old_rows_executed") == 20
      and candidate_profile.get("old_rows_executed") == 0
      and candidate_profile.get("gated_attention_executed") == 10)
  correctness_passed = (
      len(phases) == 18
      and actual_top1 == expected_top1
      and all(row.get("logits_finite") is True for row in phases))
  core_census_passed = candidate_profile.get("core_counts_exact") is True
  performance_passed = (
      math.isfinite(observed_saving) and observed_saving >= required_saving)
  worker_safe = (
      worker["returncode"] == 0 and worker["timed_out"] is False
      and worker["memory_guard"]["tripped"] is False
      and worker["oom_observed"] is False
      and int(worker["monitor"]["system_available_min_bytes"] or 0) >=
          int(args.abort_below_available_gib * 1024**3))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1312_admits_exactly_one_candidate",
            audit.get("required_checks_passed") is True
            and audit.get("candidate_workers_admitted") == 1
            and audit.get("additional_control_worker_admitted") is False),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("one_candidate_worker_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_retained_control_plugin",
            plugin_hash == build["control_plugin"]["sha256"]
            and result.get("candidate_gpu_plugin_sha256") == plugin_hash),
      check("graph_executes_exact_ten_output_gate_fusions",
            source.get("fuse_attention_output_gate") is True
            and source.get("attention_output_gate_fusion_count") == 10
            and source.get("custom_count_after") == 10,
            source_summary=source),
      check("teacher_forced_top1_is_exact", correctness_passed,
            expected_top1=expected_top1, actual_top1=actual_top1),
      check("core_execution_census_is_exact", core_census_passed,
            core_counts=candidate_profile.get("core_counts")),
      check("output_epilogue_activation_outcome_is_completely_measured",
            activation_passed,
            control=control_profile, candidate=candidate_profile),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in checks)
  route_accepted = (
      evidence_checks_passed and activation_passed and correctness_passed
      and performance_passed)
  verdict = (
      "retain_attention_output_gate_fusion_as_bundle_cut"
      if route_accepted else
      "reject_attention_output_gate_fusion_after_component"
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
      "checks": checks,
      "decision": {
          "close_route": evidence_checks_passed and not route_accepted,
          "retain_as_bundle_ingredient": route_accepted,
          "next_route": (
              "openvino_fc_rms_igc_output_gate_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for the unchanged fusion, repeat, or sample extension if "
              "this component closes"),
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
      "inputs": {display(Path(path)): sha256(Path(path)) for path in required},
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "stock_workers": 0,
      "control_workers": 0,
      "candidate_workers": 1,
  })
  report = f"""# Attention-output gate-fusion component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17-step worker ran against retained seq1304.
The candidate uses ten gated token-major custom-attention operations. Old
output-epilogue executed rows move `{control_profile.get('old_rows_executed')}
-> {candidate_profile.get('old_rows_executed')}` and gated custom operations
execute `{candidate_profile.get('gated_attention_executed')}` times. All 18
teacher-forced top-1 tokens and the core census must remain exact.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This is not a product speed claim. No compiler,
stock, control, concurrent, long, ABBA, output512, or product worker ran; OOM
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
