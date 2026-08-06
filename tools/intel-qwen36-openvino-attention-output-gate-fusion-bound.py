#!/usr/bin/env python3
"""Bound an attention-output transpose/gate epilogue fusion.

This source-only gate closes the failed split optimization and audits a
distinct output-side boundary: ten custom-attention outputs each execute an
F32 Transpose followed by a gate Multiply before the output projection.  It
derives the exact bundle admission arithmetic and, if every contract holds,
admits only the source implementation plus one candidate-only short worker.
It creates no GPU context, compiler, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-output-gate-fusion-bound-v0"
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
KERNEL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
FAILED_COMPONENT = ROOT / (
    "output/openvino-constant-q-gate-split-component-"
    "20260717Tseq1310-candidate-2k-warm17-cleanZ/metrics.json")
FAILED_WORKER = ROOT / (
    "output/openvino-constant-q-gate-split-component-"
    "20260717Tseq1310-candidate-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
SPLIT_BOUND = ROOT / (
    "output/openvino-dynamic-split-inplace-bound-"
    "20260717Tseq1303b-cleanZ/metrics.json")
POST_IGC_BOUND = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
IGC_COMPONENT = ROOT / (
    "output/openvino-igc2382-component-gate-"
    "20260717Tseq1301-cleanZ/metrics.json")
PROJECTION_BOUND = ROOT / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")
ELEMENTWISE_BOUND = ROOT / (
    "output/openvino-decode-elementwise-residual-bound-"
    "20260715Tseq1239-cleanZ/metrics.json")
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
      "tools/intel-qwen36-openvino-attention-output-gate-fusion-bound.py"}
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


def port_shapes(node: ET.Element, section: str) -> list[list[int]]:
  parent = node.find(section)
  if parent is None:
    return []
  return [[int(dim.text) for dim in port.findall("dim")]
          for port in parent.findall("port")]


def locked_output_epilogue_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  by_name = {str(node.attrib.get("name")): node_id
             for node_id, node in layers.items()}
  outgoing: dict[int, list[int]] = {node_id: [] for node_id in layers}
  incoming: dict[int, list[int]] = {node_id: [] for node_id in layers}
  for edge in edges_node:
    source = int(edge.attrib["from-layer"])
    target = int(edge.attrib["to-layer"])
    outgoing[source].append(target)
    incoming[target].append(source)

  def member(layer: int, suffix: str) -> str:
    return (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/{suffix}")

  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    names = {
        "split": member(layer, "prim::ListUnpack/VariadicSplit"),
        "gate_reshape": member(layer, "aten::reshape/Reshape_3"),
        "gate_sigmoid": member(layer, "aten::sigmoid/Sigmoid"),
        "sdpa": member(
            layer,
            "aten::scaled_dot_product_attention/ScaledDotProductAttention"),
        "output_transpose": member(layer, "aten::transpose/Transpose_3"),
        "output_reshape": member(layer, "aten::reshape/Reshape_2"),
        "gate_multiply": member(layer, "aten::mul/Multiply_6"),
        "output_projection": (
            "__module.model.model.language_model.layers."
            f"{layer}.self_attn.o_proj/ov_ext::linear/MatMul"),
    }
    ids = {key: by_name.get(name) for key, name in names.items()}
    direct_pairs = (
        ("sdpa", "output_transpose"),
        ("output_transpose", "output_reshape"),
        ("output_reshape", "gate_multiply"),
        ("gate_sigmoid", "gate_multiply"),
        ("gate_multiply", "output_projection"),
        ("gate_reshape", "gate_sigmoid"),
    )
    direct = {
        f"{source}->{target}": (
            ids[source] is not None and ids[target] is not None and
            ids[target] in outgoing[int(ids[source])])
        for source, target in direct_pairs
    }
    split_id = ids["split"]
    gate_reshape_id = ids["gate_reshape"]
    split_to_gate = (
        split_id is not None and gate_reshape_id is not None and
        gate_reshape_id in outgoing[int(split_id)])
    shapes = {
        key: port_shapes(layers[int(node_id)], "output")
        for key, node_id in ids.items() if node_id is not None}
    shape_exact = (
        shapes.get("sdpa") == [[-1, 16, -1, 256]]
        and shapes.get("output_transpose") == [[-1, -1, 16, 256]]
        and shapes.get("output_reshape") == [[-1, -1, 4096]]
        and shapes.get("gate_reshape") == [[-1, -1, 4096]]
        and shapes.get("gate_multiply") == [[-1, -1, 4096]])
    multiply_inputs = sorted(
        str(layers[source].attrib.get("type"))
        for source in incoming[int(ids["gate_multiply"])])
    rows.append({
        "layer": layer,
        "names": names,
        "names_present": all(value is not None for value in ids.values()),
        "direct_edges": direct,
        "split_to_gate_reshape": split_to_gate,
        "multiply_input_types": multiply_inputs,
        "shapes": shapes,
        "shape_exact": shape_exact,
    })
  exact = all(
      row["names_present"] and all(row["direct_edges"].values())
      and row["split_to_gate_reshape"]
      and row["multiply_input_types"] == ["Reshape", "Sigmoid"]
      and row["shape_exact"] for row in rows)
  return {"rows": rows, "exact_output_epilogue_contract": exact}


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


def runtime_output_epilogue_audit(worker: dict[str, Any]) -> dict[str, Any]:
  layer_fragments = tuple(
      f"layers.{layer}.self_attn" for layer in FULL_ATTENTION_LAYERS)
  selected = []
  for row in profile_rows(worker):
    name = str(row.get("node_name", ""))
    if not any(fragment in name for fragment in layer_fragments):
      continue
    kind = None
    if (row.get("node_type") == "Transpose" and
        name.endswith("/aten::transpose/Transpose_3")):
      kind = "attention_output_transpose"
    elif (row.get("node_type") == "Multiply" and
          name.endswith("/aten::mul/Multiply_6")):
      kind = "attention_gate_multiply"
    if kind is not None:
      selected.append({**row, "boundary_kind": kind})
  executed = [row for row in selected
              if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row["boundary_kind"]) for row in executed)
  exec_types = {
      kind: sorted({str(row.get("exec_type")) for row in executed
                    if row["boundary_kind"] == kind})
      for kind in counts}
  return {
      "counts": dict(sorted(counts.items())),
      "expected_counts": {
          "attention_gate_multiply": 10,
          "attention_output_transpose": 10,
      },
      "counts_exact": dict(counts) == {
          "attention_gate_multiply": 10,
          "attention_output_transpose": 10,
      },
      "exec_types": exec_types,
      "raw_real_time_us_nonadditive": sum(
          float(row.get("real_time_us", 0.0)) for row in executed),
      "raw_profile_time_is_savings_evidence": False,
      "rows": executed,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")
  required = (
      MODEL_XML, GRAPH_SOURCE, KERNEL_SOURCE, CUSTOM_CONFIG,
      FAILED_COMPONENT, FAILED_WORKER, SPLIT_BOUND, POST_IGC_BOUND,
      IGC_COMPONENT, PROJECTION_BOUND, ELEMENTWISE_BOUND)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing output-gate bound inputs: " + ", ".join(missing))

  git = git_state(output)
  failed = load_json(FAILED_COMPONENT)
  failed_worker = load_json(FAILED_WORKER)
  split_bound = load_json(SPLIT_BOUND)
  post_igc = load_json(POST_IGC_BOUND)
  igc_component = load_json(IGC_COMPONENT)
  projection = load_json(PROJECTION_BOUND)
  elementwise = load_json(ELEMENTWISE_BOUND)
  locked = locked_output_epilogue_audit()
  runtime = runtime_output_epilogue_audit(failed_worker)

  failed_route_closed = (
      failed.get("verdict") == "reject_constant_q_gate_split_after_component"
      and failed.get("evidence_checks_passed") is True
      and failed.get("activation_passed") is False
      and failed.get("correctness_passed") is True
      and failed.get("route_accepted") is False
      and failed.get("profile", {}).get("target_split", {}).get(
          "status_counts") == {"Status.EXECUTED": 20})
  older_bounds_closed_before_igc = (
      projection.get("required_checks_passed") is True
      and projection.get("source_edit_admitted") is False
      and elementwise.get("required_checks_passed") is True
      and elementwise.get("source_edit_admitted") is False)
  igc_exact_but_unconfirmed = (
      igc_component.get("required_checks_passed") is True
      and igc_component.get("route_accepted") is False
      and math.isclose(
          float(igc_component["performance"]["observed_median_saving_ms"]),
          0.2793365, abs_tol=1e-9))

  split_budget = split_bound["budget"]
  per_dispatch_us = (
      float(split_budget["max_enqueue_us_per_dispatch"])
      + float(split_budget["set_arguments_per_boundary_dispatch_us"]))
  removed_dispatches = 20
  provider_ceiling_ms = removed_dispatches * per_dispatch_us / 1000.0
  prior_union_ms = float(
      split_budget["seq1302_favorable_rms_plus_igc_union_ms"])
  residual_ms = float(split_budget["residual_after_fixed_fc_ms"])
  prior_shortfall_ms = float(split_budget["seq1302_shortfall_ms"])
  expanded_union_ms = prior_union_ms + provider_ceiling_ms
  expanded_margin_ms = expanded_union_ms - residual_ms
  post_igc_consistent = math.isclose(
      float(post_igc["budget"]["favorable_rms_plus_igc_union_ms"]),
      prior_union_ms,
      abs_tol=1e-12)

  graph_text = GRAPH_SOURCE.read_text(encoding="utf-8")
  kernel_text = KERNEL_SOURCE.read_text(encoding="utf-8")
  implementation_contract = {
      "custom_attention_currently_returns_head_major_query_shape": all(
          text in graph_text for text in (
              "1, self.get_input_element_type(0), query)",
              "attention_output = operation.output(1)")),
      "kernel_has_separate_attention_output_store": all(
          text in kernel_text for text in (
              "__global OUTPUT1_TYPE* output",
              "const uint query_head = kv_head * IQ36_GQA_GROUP + head;",
              "output[output_base + (ulong)dim0 * OUTPUT1_PITCHES[3]]")),
      "required_shape": "custom output [B,Q,16,256], then reshape [B,Q,4096]",
      "required_gate_input": (
          "pre-Sigmoid gate [B,Q,16,256]; compute Sigmoid and Multiply at "
          "the attention output store with explicit F16 round points"),
      "required_graph_bypass": (
          "replace the original gate Multiply output, leaving the failed "
          "Q/gate split path otherwise unchanged"),
      "required_runtime_census": (
          "ten Transpose_3 and ten Multiply_6 rows optimized out or absent"),
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("constant_split_and_relocation_route_is_closed",
            failed_route_closed),
      check("locked_ir_has_ten_exact_output_transpose_gate_chains",
            locked["exact_output_epilogue_contract"], audit=locked),
      check("runtime_executes_exact_twenty_output_epilogue_rows",
            runtime["counts_exact"], audit=runtime),
      check("runtime_profile_rows_are_not_added_as_savings",
            runtime["raw_profile_time_is_savings_evidence"] is False),
      check("older_projection_and_elementwise_bounds_are_not_relitigated",
            older_bounds_closed_before_igc,
            note=("this route is the narrower post-attention epilogue and is "
                  "reopened only by the later IGC/RMS bundle arithmetic")),
      check("official_igc_component_identity_is_exact_but_unconfirmed",
            igc_exact_but_unconfirmed),
      check("seq1302_bundle_arithmetic_is_consistent",
            post_igc_consistent and prior_shortfall_ms > 0.0),
      check("independent_twenty_dispatch_ceiling_closes_bundle_shortfall",
            provider_ceiling_ms > prior_shortfall_ms
            and expanded_margin_ms > 0.0,
            provider_ceiling_ms=provider_ceiling_ms,
            prior_shortfall_ms=prior_shortfall_ms,
            expanded_union_ms=expanded_union_ms,
            expanded_margin_ms=expanded_margin_ms),
      check("implementation_contract_is_source_complete",
            implementation_contract[
                "custom_attention_currently_returns_head_major_query_shape"]
            and implementation_contract[
                "kernel_has_separate_attention_output_store"],
            contract=implementation_contract),
      check("decision_gate_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0, model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_attention_output_gate_fusion_source_and_one_short_candidate"
      if required_checks_passed else "inconclusive")
  available_end = available_memory_bytes()
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
      "compiler_build_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "failed_split_component": {
          "closed": failed_route_closed,
          "verdict": failed.get("verdict"),
          "activation_passed": failed.get("activation_passed"),
          "correctness_passed": failed.get("correctness_passed"),
          "performance": failed.get("performance"),
          "memory": failed.get("worker", {}).get("monitor"),
          "oom_observed": failed.get("worker", {}).get("oom_observed"),
      },
      "locked_output_epilogue": locked,
      "runtime_output_epilogue": runtime,
      "implementation_contract": implementation_contract,
      "budget": {
          "residual_after_fixed_fc_ms": residual_ms,
          "seq1302_favorable_rms_plus_igc_union_ms": prior_union_ms,
          "seq1302_shortfall_ms": prior_shortfall_ms,
          "removed_dispatches": removed_dispatches,
          "max_enqueue_us_per_dispatch": split_budget[
              "max_enqueue_us_per_dispatch"],
          "set_arguments_per_boundary_dispatch_us": split_budget[
              "set_arguments_per_boundary_dispatch_us"],
          "favorable_output_epilogue_provider_ceiling_ms":
              provider_ceiling_ms,
          "expanded_favorable_union_ms": expanded_union_ms,
          "expanded_union_margin_ms": expanded_margin_ms,
          "interpretation": (
              "source admission only; the output fusion must independently "
              "save at least the seq1302 shortfall, and any final IGC/RMS/FC "
              "bundle must be rebuilt and measured together"),
      },
      "overlap_discipline": {
          "failed_split_dispatches_counted": False,
          "raw_profile_times_counted": False,
          "igc_point_saving_confirmed": False,
          "final_bundle_additivity_claimed": False,
          "required_component_cut_ms": prior_shortfall_ms,
          "required_final_action": (
              "measure the output fusion alone on the current compiler; if "
              "it clears, rebuild the eventual combined bundle rather than "
              "adding independent point estimates"),
      },
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "oom_observed": False,
      },
      "checks": checks,
      "decision": {
          "next_route": (
              "openvino_attention_output_gate_fusion_component"
              if required_checks_passed else None),
          "reason": (
              "write gated token-major attention output in the existing "
              "custom operation and remove ten output transposes plus ten "
              "gate multiplies"),
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
  report = f"""# Attention-output gate-fusion bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

Seq1310 closes constant split lengths plus consumer relocation: all ten folds
and relocations execute and correctness is exact, but all twenty split/Crop
rows remain live. Its `0.2765330-ms` short median movement is not interpretable
because the registered activation condition fails. No repeat is admitted.

The distinct output boundary is exact across all ten full-attention layers:
`SDPA -> Transpose_3 -> Reshape_2 -> Multiply_6 -> output projection`, with
the other Multiply input supplied by the Q/gate Sigmoid. The retained runtime
executes ten F32 output Transposes and ten gate Multiplies. A fused custom
attention output can consume the pre-Sigmoid gate, reproduce the F16 round
points, write `[B,Q,16,256]`, reshape without a dispatch, and replace the
Multiply output. It does not count or retry the failed split optimization.

The exact twenty-dispatch provider ceiling is `{provider_ceiling_ms:.7f} ms`.
Added only as favorable source admission to seq1302's
`{prior_union_ms:.7f}-ms` RMS-plus-IGC union, it reaches
`{expanded_union_ms:.7f} ms` versus the `{residual_ms:.7f}-ms` residual, a
`{expanded_margin_ms:.7f}-ms` margin. Raw profile times are not savings. The
fusion must independently save at least `{prior_shortfall_ms:.7f} ms`; an
eventual FC/RMS/IGC/fusion bundle must be rebuilt and measured together.

Admit the exact source implementation and, only after a no-GPU rewrite audit,
one candidate-only 2k/17-step worker against retained seq1304. No new control,
compiler build, repeat, long, ABBA, output512, or product worker is funded.
This gate launches no GPU context or model worker; OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_edit_admitted": required_checks_passed,
      "provider_ceiling_ms": provider_ceiling_ms,
      "expanded_union_margin_ms": expanded_margin_ms,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
