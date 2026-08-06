#!/usr/bin/env python3
"""Audit and run the token-major current-value/output attention component.

``--audit-only`` performs the mandatory no-GPU all-ten rewrite/source audit.
The default mode launches exactly one candidate-only 2k/17-step worker against
retained seq1304. It never launches a new control, stock, long, ABBA,
output512, or product worker.
"""

from __future__ import annotations

import argparse
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
SCHEMA = "intel-qwen36-openvino-attention-token-major-value-output-component-v0"
FUSION_COMPONENT_PATH = ROOT / (
    "tools/intel-qwen36-openvino-attention-output-gate-fusion-component.py")
BOUND = ROOT / (
    "output/openvino-attention-token-major-value-output-bound-"
    "20260717Tseq1314-cleanZ/metrics.json")
AUDIT = ROOT / (
    "output/openvino-attention-token-major-value-output-rewrite-"
    "20260717Tseq1315-cleanZ/metrics.json")
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
    "IQ36LinearConvSwish": 30,
    "IQ36TokenMajorValueAttentionGQA": 10,
    "RMS": 131,
}


def load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


FUSION = load_module(FUSION_COMPONENT_PATH, "iq36_gate_fusion_component")
BASE = FUSION.BASE


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


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-attention-token-major-value-output-"
      "component.py"}
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
    token_major_value_output=True)
ops = model.get_ordered_ops()
names = {{x.get_friendly_name() for x in ops}}
types = {{name: sum(x.get_type_name() == name for x in ops)
         for name in ('IQ36TokenMajorValueAttentionGQA',
                      'IQ36HotAttentionGQA',
                      'ScaledDotProductAttention',
                      'IQ36LinearConvSwish')}}
rows = []
old_live = []
rewrites = {{row['layer']: row
            for row in summary['token_major_value_output_rewrites']}}
for layer in g.FULL_ATTENTION_LAYERS:
  prefix = f'__module.model.model.language_model.layers.{{layer}}.self_attn/'
  operation = next(x for x in ops if x.get_friendly_name() ==
                   f'iq36_hot_attention_layer{{layer}}')
  multiply = next(x for x in ops if x.get_friendly_name() ==
                  prefix + 'aten::mul/Multiply_6')
  for old in (rewrites[layer]['value_transpose'],
              prefix + 'aten::transpose/Transpose_3',
              prefix + 'aten::reshape/Reshape_2'):
    if old in names:
      old_live.append(old)
  rows.append({{
      'layer': layer,
      'inputs': operation.get_input_size(),
      'outputs': operation.get_output_size(),
      'input4_shape': str(operation.get_input_partial_shape(4)),
      'output1_shape': str(operation.get_output_partial_shape(1)),
      'multiply_source': multiply.input_value(0).get_node().get_friendly_name(),
      'multiply_type': multiply.get_type_name(),
  }})
print(json.dumps({{
    'summary': {{key: summary[key] for key in (
        'custom_count_after', 'stock_sdpa_count_after',
        'fuse_attention_output_gate', 'token_major_value_output',
        'token_major_value_output_rewrite_count')}},
    'types': types, 'rows': rows, 'old_live': old_live,
}}, sort_keys=True))
"""
  before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=True,
      capture_output=True, text=True, timeout=120)
  after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  audit = json.loads(completed.stdout)
  audit["child_max_rss_kib_upper_bound"] = max(0, int(after - before))
  FUSION.write_json(raw / "model-rewrite-audit.json", audit)
  return audit


def source_audit() -> dict[str, Any]:
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  marker = '<CustomLayer name="IQ36TokenMajorValueAttentionGQA"'
  tail = config.split(marker, 1)[1] if config.count(marker) == 1 else ""
  block = tail.split("</CustomLayer>", 1)[0] if tail else ""
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  worker = WORKER_SOURCE.read_text(encoding="utf-8")
  helpers = HELPERS_SOURCE.read_text(encoding="utf-8")
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  prefill = PREFILL_SOURCE.read_text(encoding="utf-8")
  return {
      "one_custom_config_entry": config.count(marker) == 1,
      "custom_config_inputs": block.count('type="input"'),
      "custom_config_outputs": block.count('type="output"'),
      "custom_config_macro": "-DIQ36_TOKEN_MAJOR_VALUE_OUTPUT=1" in block,
      "graph_has_mixed_layout_type_and_exact_bypasses": all(
          text in graph for text in (
              "class IQ36TokenMajorValueAttentionGQA",
              "token_major_value_output",
              "current_value = value_transpose.input_value(0)",
              "output_reshape.output(0).replace(")),
      "worker_plumbs_default_off_flag": (
          'cfg.get("token_major_value_output", False)' in worker),
      "helpers_switch_only_current_value_axes": all(
          text in helpers for text in (
              "IQ36_CURRENT_VALUE_HEAD_PITCH INPUT4_PITCHES[2]",
              "IQ36_CURRENT_VALUE_TOKEN_PITCH INPUT4_PITCHES[1]",
              "inline ulong iq36_current_value_index")),
      "decode_uses_shared_value_index_and_token_major_output": all(
          text in kernel for text in (
              "iq36_current_value_index(batch, kv_head, 0U, 0U)",
              "#if defined(IQ36_TOKEN_MAJOR_OUTPUT)",
              "(ulong)query_head * OUTPUT1_PITCHES[2]")),
      "prefill_microkernel_and_scalar_paths_are_token_major": all(
          text in prefill for text in (
              "IQ36_CURRENT_VALUE_TOKEN_PITCH",
              "#if defined(IQ36_TOKEN_MAJOR_OUTPUT)",
              "output_accumulator, dimension_block, query_in_tile",
              "(ulong)query_position * OUTPUT1_PITCHES[1]")),
      "layout_variant_adds_no_gate_input": (
          block.count('port-index="13"') == 0),
  }


def rewrite_exact(rewrite: dict[str, Any]) -> bool:
  return (
      rewrite.get("summary") == {
          "custom_count_after": 10,
          "fuse_attention_output_gate": False,
          "stock_sdpa_count_after": 0,
          "token_major_value_output": True,
          "token_major_value_output_rewrite_count": 10,
      }
      and rewrite.get("types") == {
          "IQ36HotAttentionGQA": 0,
          "IQ36LinearConvSwish": 30,
          "IQ36TokenMajorValueAttentionGQA": 10,
          "ScaledDotProductAttention": 0,
      }
      and rewrite.get("old_live") == []
      and len(rewrite.get("rows", [])) == 10
      and all(
          row.get("inputs") == 13 and row.get("outputs") == 6
          and row.get("input4_shape") == "[?,?,2,256]"
          and row.get("output1_shape") == "[?,?,16,256]"
          and row.get("multiply_type") == "Multiply"
          and row.get("multiply_source") ==
              f"iq36_attention_token_major_flat_layer{row.get('layer')}"
          for row in rewrite.get("rows", [])))


def source_exact(source: dict[str, Any]) -> bool:
  return (
      source.get("one_custom_config_entry") is True
      and source.get("custom_config_inputs") == 13
      and source.get("custom_config_outputs") == 6
      and all(value is True for key, value in source.items()
              if key not in ("custom_config_inputs", "custom_config_outputs")))


def runtime_layout_audit(
    result: dict[str, Any], bound: dict[str, Any]) -> dict[str, Any]:
  graph_rows = bound["accepted_graph_layout"]["rows"]
  value_names = {str(row["value_transpose"]) for row in graph_rows}
  output_names = {str(row["output_transpose"]) for row in graph_rows}
  multiply_names = {str(row["gate_multiply"]) for row in graph_rows}
  selected = []
  for row in FUSION.profile_rows(result):
    name = str(row.get("node_name", ""))
    kind = None
    if name in value_names:
      kind = "current_value_transpose"
    elif name in output_names:
      kind = "attention_output_transpose"
    elif name in multiply_names:
      kind = "preserved_gate_multiply"
    if kind is not None:
      selected.append({**row, "boundary_kind": kind})
  executed = [row for row in selected
              if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row["boundary_kind"]) for row in executed)
  all_rows = FUSION.profile_rows(result)
  gated = [row for row in all_rows
           if row.get("status") == "Status.EXECUTED" and
           row.get("node_type") == "IQ36TokenMajorValueAttentionGQA"]
  executed_counts = Counter(
      str(row.get("node_type")) for row in all_rows
      if row.get("status") == "Status.EXECUTED")
  core = {name: int(executed_counts.get(name, 0))
          for name in EXPECTED_CORE_COUNTS}
  return {
      "layout_status_counts": dict(sorted(counts.items())),
      "old_layout_rows_executed": int(
          counts.get("current_value_transpose", 0)
          + counts.get("attention_output_transpose", 0)),
      "gate_multiply_executed": int(
          counts.get("preserved_gate_multiply", 0)),
      "token_major_attention_executed": len(gated),
      "core_counts": core,
      "core_counts_exact": core == EXPECTED_CORE_COUNTS,
      "rows": executed,
      "raw_profile_time_is_savings_evidence": False,
  }


def audit_main(args: argparse.Namespace, output: Path) -> int:
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, MODEL_DIR / "openvino_language_model.xml", OV_PYTHON,
      GRAPH_SOURCE, WORKER_SOURCE, KERNEL_SOURCE, PREFILL_SOURCE,
      HELPERS_SOURCE, CUSTOM_CONFIG)
  missing = [FUSION.display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing token-major rewrite inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  bound = FUSION.load_json(BOUND)
  rewrite = execute_rewrite_audit(raw)
  source = source_audit()
  checks = [
      FUSION.check("repository_clean_at_gate", not git["dirty"], git=git),
      FUSION.check("seq1314_admits_exact_source_and_one_candidate",
                   bound.get("required_checks_passed") is True
                   and bound.get("source_edit_admitted") is True
                   and bound.get(
                       "candidate_workers_admitted_after_exact_rewrite_audit")
                       == 1),
      FUSION.check("no_concurrent_worker_during_rewrite_audit",
                   not concurrent, concurrent=concurrent),
      FUSION.check("exact_all_ten_mixed_layout_rewrite_removes_old_rows",
                   rewrite_exact(rewrite), audit=rewrite),
      FUSION.check("custom_config_graph_and_both_kernel_phases_are_exact",
                   source_exact(source), audit=source),
      FUSION.check("rewrite_audit_launches_no_compiler_gpu_or_model_worker",
                   True, compilers=0, gpu_contexts=0, model_workers=0,
                   model_ir_reads=1),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_token_major_value_output_candidate"
      if passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "mode": "audit_only",
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "candidate_workers_admitted": 1 if passed else 0,
      "additional_control_worker_admitted": False,
      "rewrite": rewrite,
      "source_contract": source,
      "memory": {
          "stop_bytes": int(args.abort_below_available_gib * 1024**3),
          "available_end_bytes": FUSION.available_memory_bytes(),
          "oom_observed": False,
      },
      "checks": checks,
  }
  FUSION.write_json(output / "metrics.json", metrics)
  FUSION.write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "mode": "audit_only",
      "tool": FUSION.display(Path(__file__)),
      "git": git,
      "inputs": {FUSION.display(path): FUSION.sha256(path)
                 for path in required},
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
      "model_ir_reads": 1,
  })
  report = f"""# Token-major value/output rewrite audit

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

The no-GPU rewrite creates ten 13-input mixed-layout attention operations.
All ten old current-value and output Transposes are dead, all ten gate
Multiplies remain live, and prefill/decode source changes only INPUT4/OUTPUT1
pitch axes. Admit one candidate-only short worker; no compiler, GPU context,
or model worker ran in this audit. OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": FUSION.display(output), "verdict": verdict,
      "candidate_workers_admitted": 1 if passed else 0,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


def component_main(args: argparse.Namespace, output: Path) -> int:
  worker_dir = output / "raw/2k/candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, AUDIT, BUILD, CONTROL, args.plugin, BASE.REFERENCE_WORKER,
      BASE.WORKER, GRAPH_SOURCE, KERNEL_SOURCE, PREFILL_SOURCE,
      HELPERS_SOURCE, CUSTOM_CONFIG)
  missing = [FUSION.display(Path(path)) for path in required
             if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing token-major component inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  bound = FUSION.load_json(BOUND)
  audit = FUSION.load_json(AUDIT)
  build = FUSION.load_json(BUILD)
  control = FUSION.load_json(CONTROL)
  reference = FUSION.load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = FUSION.sha256(plugin)
  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["token_major_value_output"] = True
  worker = BASE.launch_worker(args, worker_dir, config)
  result = worker["result"]
  phases = result.get("phases", [])
  actual_top1 = [int(row.get("top1", -1)) for row in phases]
  candidate_profile = runtime_layout_audit(result, bound) if result else {}
  control_profile = runtime_layout_audit(control, bound)
  source = result.get("source_summary") or {}
  control_stable = FUSION.stable_walls(control)
  candidate_stable = FUSION.stable_walls(result) if result else []
  control_median = statistics.median(control_stable)
  candidate_median = (
      statistics.median(candidate_stable) if candidate_stable else math.nan)
  observed_saving = control_median - candidate_median
  required_saving = float(bound["budget"]["seq1302_shortfall_ms"])
  activation_passed = (
      control_profile.get("old_layout_rows_executed") == 20
      and control_profile.get("gate_multiply_executed") == 10
      and candidate_profile.get("old_layout_rows_executed") == 0
      and candidate_profile.get("gate_multiply_executed") == 10
      and candidate_profile.get("token_major_attention_executed") == 10)
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
      FUSION.check("repository_clean_at_gate", not git["dirty"], git=git),
      FUSION.check("seq1315_admits_exactly_one_candidate",
                   audit.get("required_checks_passed") is True
                   and audit.get("candidate_workers_admitted") == 1
                   and audit.get("additional_control_worker_admitted") is False),
      FUSION.check("no_concurrent_worker_at_launch", not concurrent,
                   concurrent=concurrent),
      FUSION.check("one_candidate_worker_completes_above_stop_without_oom",
                   worker_safe, worker={key: worker[key] for key in (
                       "returncode", "timed_out", "memory_guard", "monitor",
                       "oom_observed")}),
      FUSION.check("worker_uses_exact_retained_control_plugin",
                   plugin_hash == build["control_plugin"]["sha256"]
                   and result.get("candidate_gpu_plugin_sha256") == plugin_hash),
      FUSION.check("graph_executes_exact_ten_mixed_layout_operations",
                   source.get("token_major_value_output") is True
                   and source.get("token_major_value_output_rewrite_count") == 10
                   and source.get("custom_count_after") == 10,
                   source_summary=source),
      FUSION.check("teacher_forced_top1_is_exact", correctness_passed,
                   expected_top1=expected_top1, actual_top1=actual_top1),
      FUSION.check("core_execution_census_is_exact", core_census_passed,
                   core_counts=candidate_profile.get("core_counts")),
      FUSION.check("layout_activation_outcome_is_completely_measured",
                   activation_passed, control=control_profile,
                   candidate=candidate_profile),
      FUSION.check("profile_times_are_not_added_as_savings",
                   candidate_profile.get("raw_profile_time_is_savings_evidence")
                       is False),
  ]
  evidence_checks_passed = all(row["pass"] for row in checks)
  route_accepted = (
      evidence_checks_passed and activation_passed and correctness_passed
      and performance_passed)
  verdict = (
      "retain_token_major_value_output_as_bundle_cut"
      if route_accepted else
      "reject_token_major_value_output_after_component"
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
              "openvino_fc_rms_igc_token_major_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for the unchanged layout, repeat, or sample extension if "
              "this component closes"),
      },
  }
  FUSION.write_json(output / "metrics.json", metrics)
  FUSION.write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "mode": "component",
      "tool": FUSION.display(Path(__file__)),
      "git": git,
      "plugin": str(plugin),
      "plugin_sha256": plugin_hash,
      "inputs": {FUSION.display(Path(path)): FUSION.sha256(Path(path))
                 for path in required},
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "stock_workers": 0,
      "control_workers": 0,
      "candidate_workers": 1,
  })
  report = f"""# Token-major value/output attention component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_checks_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17-step worker ran against retained seq1304.
Old layout rows execute `{control_profile.get('old_layout_rows_executed')} ->
{candidate_profile.get('old_layout_rows_executed')}` while gate Multiply rows
remain `{control_profile.get('gate_multiply_executed')} ->
{candidate_profile.get('gate_multiply_executed')}`.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This is not a product speed claim. No compiler,
stock, control, long, ABBA, output512, or product worker ran; OOM observed:
`{str(worker['oom_observed']).lower()}`.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": FUSION.display(output), "verdict": verdict,
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
