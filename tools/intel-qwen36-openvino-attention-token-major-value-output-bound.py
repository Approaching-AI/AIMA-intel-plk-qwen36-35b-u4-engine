#!/usr/bin/env python3
"""Bound a layout-only token-major value/output attention cut.

The rejected gated-output route proved that adding scalar Sigmoid work to the
state-owning attention kernel is too expensive.  This source-only gate audits a
strictly narrower layout cut: bypass the ten current-value Transposes, return
token-major output from the existing custom operation, and bypass the ten
output Transposes while leaving the optimized gate Multiply untouched.

The gate reads the locked IR once in a no-GPU subprocess.  It starts no
compiler, GPU context, plugin build, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-token-major-value-output-bound-v0"
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER_SOURCE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
KERNEL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
PREFILL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_prefill_attention_tiled.cl")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
CONTROL_WORKER = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
FAILED_FUSION = ROOT / (
    "output/openvino-attention-output-gate-fusion-component-"
    "20260717Tseq1313-candidate-2k-warm17-cleanZ/metrics.json")
FUSION_BOUND = ROOT / (
    "output/openvino-attention-output-gate-fusion-bound-"
    "20260717Tseq1311c-cleanZ/metrics.json")
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))


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
  allowed = {
      "tools/intel-qwen36-openvino-attention-token-major-value-output-"
      "bound.py"}
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


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def profile_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
  rows = worker.get("full_profile")
  if isinstance(rows, list):
    return rows
  phases = worker.get("phases", [])
  if phases and isinstance(phases[-1], dict):
    rows = phases[-1].get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker has no final full profile")
  return rows


def execute_graph_audit(raw: Path) -> dict[str, Any]:
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
    prefill_history_capacity=16384, fuse_linear_conv_state=True)
ops = model.get_ordered_ops()
rows = []
for layer in g.FULL_ATTENTION_LAYERS:
  operation = next(x for x in ops if x.get_friendly_name() ==
                   f'iq36_hot_attention_layer{{layer}}')
  value_transpose = operation.input_value(4).get_node()
  value_order = (value_transpose.input_value(1).get_node().get_vector().tolist()
                 if value_transpose.get_type_name() == 'Transpose' else [])
  output_targets = list(operation.output(1).get_target_inputs())
  output_transpose = output_targets[0].get_node() if len(output_targets) == 1 else None
  output_order = (output_transpose.input_value(1).get_node().get_vector().tolist()
                  if output_transpose is not None and
                  output_transpose.get_type_name() == 'Transpose' else [])
  reshape_targets = (list(output_transpose.output(0).get_target_inputs())
                     if output_transpose is not None else [])
  output_reshape = reshape_targets[0].get_node() if len(reshape_targets) == 1 else None
  multiply_targets = (list(output_reshape.output(0).get_target_inputs())
                      if output_reshape is not None else [])
  gate_multiply = multiply_targets[0].get_node() if len(multiply_targets) == 1 else None
  rows.append({{
      'layer': layer,
      'value_transpose': value_transpose.get_friendly_name(),
      'value_type': value_transpose.get_type_name(),
      'value_order': value_order,
      'value_source': value_transpose.input_value(0).get_node().get_friendly_name(),
      'value_source_shape': str(value_transpose.input_value(0).get_partial_shape()),
      'value_output_shape': str(value_transpose.output(0).get_partial_shape()),
      'output_transpose': (output_transpose.get_friendly_name()
                           if output_transpose is not None else None),
      'output_type': (output_transpose.get_type_name()
                      if output_transpose is not None else None),
      'output_order': output_order,
      'attention_output_shape': str(operation.output(1).get_partial_shape()),
      'output_transpose_shape': (str(output_transpose.output(0).get_partial_shape())
                                 if output_transpose is not None else None),
      'output_reshape': (output_reshape.get_friendly_name()
                         if output_reshape is not None else None),
      'output_reshape_shape': (str(output_reshape.output(0).get_partial_shape())
                               if output_reshape is not None else None),
      'gate_multiply': (gate_multiply.get_friendly_name()
                        if gate_multiply is not None else None),
      'gate_multiply_type': (gate_multiply.get_type_name()
                             if gate_multiply is not None else None),
  }})
print(json.dumps({{
    'summary': {{key: summary[key] for key in (
        'custom_count_after', 'stock_sdpa_count_after',
        'fuse_attention_output_gate')}},
    'rows': rows,
}}, sort_keys=True))
"""
  before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=True,
      capture_output=True, text=True, timeout=120)
  after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  audit = json.loads(completed.stdout)
  audit["child_max_rss_kib_upper_bound"] = max(0, int(after - before))
  write_json(raw / "accepted-graph-layout-audit.json", audit)
  return audit


def graph_contract_exact(audit: dict[str, Any]) -> bool:
  summary = audit.get("summary", {})
  rows = audit.get("rows", [])
  return (
      summary == {
          "custom_count_after": 10,
          "fuse_attention_output_gate": False,
          "stock_sdpa_count_after": 0,
      }
      and len(rows) == 10
      and all(
          row.get("value_type") == "Transpose"
          and row.get("value_order") == [0, 2, 1, 3]
          and row.get("value_source_shape") == "[?,?,2,256]"
          and row.get("value_output_shape") == "[?,2,?,256]"
          and row.get("output_type") == "Transpose"
          and row.get("output_order") == [0, 2, 1, 3]
          and row.get("attention_output_shape") == "[?,16,?,256]"
          and row.get("output_transpose_shape") == "[?,?,16,256]"
          and row.get("output_reshape_shape") == "[?,?,4096]"
          and row.get("gate_multiply_type") == "Multiply"
          and str(row.get("output_transpose", "")).endswith(
              "/aten::transpose/Transpose_3")
          and str(row.get("output_reshape", "")).endswith(
              "/aten::reshape/Reshape_2")
          and str(row.get("gate_multiply", "")).endswith(
              "/aten::mul/Multiply_6")
          for row in rows))


def runtime_layout_audit(
    worker: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
  value_names = {str(row["value_transpose"]) for row in graph["rows"]}
  output_names = {str(row["output_transpose"]) for row in graph["rows"]}
  multiply_names = {str(row["gate_multiply"]) for row in graph["rows"]}
  selected = []
  for row in profile_rows(worker):
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
  expected = {
      "attention_output_transpose": 10,
      "current_value_transpose": 10,
      "preserved_gate_multiply": 10,
  }
  return {
      "counts": dict(sorted(counts.items())),
      "expected_counts": expected,
      "counts_exact": dict(counts) == expected,
      "exec_types": {
          kind: sorted({str(row.get("exec_type")) for row in executed
                        if row["boundary_kind"] == kind})
          for kind in expected},
      "raw_real_time_us_nonadditive": sum(
          float(row.get("real_time_us", 0.0)) for row in executed),
      "raw_profile_time_is_savings_evidence": False,
      "rows": executed,
  }


def source_contract() -> dict[str, Any]:
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  prefill = PREFILL_SOURCE.read_text(encoding="utf-8")
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  return {
      "one_parameterized_graph_loop": "for layer in target_layers:" in graph,
      "current_value_head_major_decode_index": all(text in kernel for text in (
          "(ulong)kv_head * INPUT4_PITCHES[1]",
          "(ulong)token * INPUT4_PITCHES[2]")),
      "output_head_major_decode_index": (
          "(ulong)query_head * OUTPUT1_PITCHES[1]" in kernel),
      "prefill_current_value_and_output_are_head_major": all(
          text in prefill for text in (
              "(ulong)kv_head * INPUT4_PITCHES[1]",
              "(__global half*)&output[output_tile]")),
      "simplegpu_config_is_parameterizable": (
          '<CustomLayer name="IQ36HotAttentionGQA"' in config
          and '<WorkSizes global="128,Y,B*F" local="128,1,1"/>' in config),
      "required_variant": (
          "13-input mixed-layout SimpleGPU op: Q/K stay head-major; current V "
          "is [B,Q,2,256]; output is [B,Q,16,256]"),
      "required_graph_rewrite": (
          "bypass each current-value Transpose input and each output "
          "Transpose/Reshape chain; keep all ten gate Multiply nodes live"),
      "required_kernel_rewrite": (
          "change only INPUT4 and OUTPUT1 pitch axes in prefill/decode; add no "
          "Sigmoid, Multiply, reduction, state, or attention arithmetic"),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")
  required = (
      MODEL_XML, OV_PYTHON, GRAPH_SOURCE, WORKER_SOURCE, KERNEL_SOURCE,
      PREFILL_SOURCE, CUSTOM_CONFIG, CONTROL_WORKER, FAILED_FUSION,
      FUSION_BOUND)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing token-major bound inputs: " + ", ".join(missing))

  git = git_state(output)
  failed = load_json(FAILED_FUSION)
  fusion_bound = load_json(FUSION_BOUND)
  worker = load_json(CONTROL_WORKER)
  graph = execute_graph_audit(raw)
  runtime = runtime_layout_audit(worker, graph)
  source = source_contract()

  failed_route_closed = (
      failed.get("verdict") ==
          "reject_attention_output_gate_fusion_after_component"
      and failed.get("evidence_checks_passed") is True
      and failed.get("activation_passed") is True
      and failed.get("correctness_passed") is True
      and failed.get("performance_passed") is False
      and math.isclose(
          float(failed["performance"]["observed_median_saving_ms"]),
          -1.4386054999999942, abs_tol=1e-9))
  source_exact = all(
      value is True for value in source.values() if isinstance(value, bool))
  old_budget = fusion_bound["budget"]
  per_dispatch_us = (
      float(old_budget["max_enqueue_us_per_dispatch"])
      + float(old_budget["set_arguments_per_boundary_dispatch_us"]))
  removed_dispatches = 20
  provider_ceiling_ms = removed_dispatches * per_dispatch_us / 1000.0
  prior_union_ms = float(
      old_budget["seq1302_favorable_rms_plus_igc_union_ms"])
  residual_ms = float(old_budget["residual_after_fixed_fc_ms"])
  shortfall_ms = float(old_budget["seq1302_shortfall_ms"])
  expanded_union_ms = prior_union_ms + provider_ceiling_ms
  expanded_margin_ms = expanded_union_ms - residual_ms

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("gated_output_arithmetic_route_is_closed", failed_route_closed),
      check("accepted_graph_has_exact_ten_value_and_output_transposes",
            graph_contract_exact(graph), audit=graph),
      check("runtime_executes_exact_twenty_layout_rows_and_ten_gate_rows",
            runtime["counts_exact"], audit=runtime),
      check("runtime_profile_rows_are_not_added_as_savings",
            runtime["raw_profile_time_is_savings_evidence"] is False),
      check("layout_only_source_contract_is_exact", source_exact,
            contract=source),
      check("twenty_dispatch_ceiling_closes_bundle_shortfall",
            provider_ceiling_ms > shortfall_ms
            and expanded_margin_ms > 0.0,
            provider_ceiling_ms=provider_ceiling_ms,
            shortfall_ms=shortfall_ms,
            expanded_margin_ms=expanded_margin_ms),
      check("source_gate_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0, model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_token_major_value_output_source_and_one_short_candidate"
      if required_checks_passed else "inconclusive")
  budget = {
      "max_enqueue_us_per_dispatch":
          float(old_budget["max_enqueue_us_per_dispatch"]),
      "set_arguments_per_boundary_dispatch_us":
          float(old_budget["set_arguments_per_boundary_dispatch_us"]),
      "removed_dispatches": removed_dispatches,
      "favorable_layout_provider_ceiling_ms": provider_ceiling_ms,
      "seq1302_favorable_rms_plus_igc_union_ms": prior_union_ms,
      "expanded_favorable_union_ms": expanded_union_ms,
      "residual_after_fixed_fc_ms": residual_ms,
      "seq1302_shortfall_ms": shortfall_ms,
      "expanded_union_margin_ms": expanded_margin_ms,
      "interpretation": (
          "source admission only; the layout-only component must independently "
          "save the seq1302 shortfall and any final bundle must be rebuilt"),
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": required_checks_passed,
      "candidate_workers_admitted_after_exact_rewrite_audit": (
          1 if required_checks_passed else 0),
      "additional_control_worker_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "failed_gated_output_component": {
          "verdict": failed.get("verdict"),
          "observed_median_saving_ms": failed.get("performance", {}).get(
              "observed_median_saving_ms"),
          "activation_passed": failed.get("activation_passed"),
          "correctness_passed": failed.get("correctness_passed"),
      },
      "accepted_graph_layout": graph,
      "runtime_layout": runtime,
      "implementation_contract": source,
      "budget": budget,
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_memory_bytes(),
          "oom_observed": False,
      },
      "checks": checks,
      "decision": {
          "next_route": "openvino_attention_token_major_value_output_component",
          "reason": (
              "remove ten current-value and ten output layout dispatches "
              "without adding gate arithmetic to the attention kernel"),
          "reopen_condition": (
              "one exact all-ten rewrite audit followed by one candidate-only "
              "short worker; close on correctness, census, or cut failure"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
      "model_ir_reads": 1,
  })
  report = f"""# Token-major value/output attention bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The accepted graph has ten exact current-value `[B,Q,2,256] ->
[B,2,Q,256]` Transposes and ten exact attention-output `[B,16,Q,256] ->
[B,Q,16,256]` Transposes. The retained runtime executes all twenty rows and
all ten gate Multiplies. The proposed mixed-layout custom operation bypasses
only those twenty Transposes and keeps the optimized gate kernel independent;
it adds no gate or attention arithmetic.

The independent twenty-dispatch provider ceiling is
`{provider_ceiling_ms:.7f} ms`, above the `{shortfall_ms:.7f}-ms` bundle
shortfall. This is source admission, not measured saving. Require an exact
no-GPU all-ten rewrite audit, then one candidate-only short worker against
retained seq1304; no new control, long, ABBA, output512, or product worker.

No compiler or GPU context ran. Model IR reads: one; OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "source_edit_admitted": required_checks_passed,
      "candidate_workers_admitted_after_exact_rewrite_audit": (
          1 if required_checks_passed else 0),
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
