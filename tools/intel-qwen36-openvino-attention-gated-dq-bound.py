#!/usr/bin/env python3
"""Bound consumer-side attention transpose/gate/dynamic-quantize fusion.

The two producer-side attention layout experiments activated exactly but
regressed the short wall.  This source-only gate moves the same data movement
to the output-projection consumer: one custom operation must transpose the
accepted head-major attention output, apply the already-computed F16 gate,
and emit the exact group-64 I8 activation, F16 scales, and I32 precomputed
reductions consumed by the compressed output projection.  The group size is
derived from the locked U4 output-projection scale/zero-point tensors rather
than copied from the requested plugin property.  It invokes no compiler, GPU
context, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-gated-dq-bound-v1"

SEQ1302 = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
SEQ1311 = ROOT / (
    "output/openvino-attention-output-gate-fusion-bound-"
    "20260717Tseq1311c-cleanZ/metrics.json")
SEQ1313 = ROOT / (
    "output/openvino-attention-output-gate-fusion-component-"
    "20260717Tseq1313-candidate-2k-warm17-cleanZ/metrics.json")
SEQ1316 = ROOT / (
    "output/openvino-attention-token-major-value-output-component-"
    "20260717Tseq1316-candidate-2k-warm17-cleanZ/metrics.json")
CONTROL_WORKER = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
DQ_TRANSFORM = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "dynamic_quantize_fully_connected.cpp")
DQ_KERNEL = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
    "dynamic_quantize_gpu_opt.cl")
MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
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
      "tools/intel-qwen36-openvino-attention-gated-dq-bound.py"}
  relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative):
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
  if not isinstance(rows, list):
    raise TypeError("control worker has no full_profile")
  return rows


def control_census(worker: dict[str, Any]) -> dict[str, Any]:
  rows = [row for row in profile_rows(worker)
          if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type")) for row in rows)
  selected: list[dict[str, Any]] = []
  for row in rows:
    name = str(row.get("node_name", ""))
    for layer in FULL_ATTENTION_LAYERS:
      prefix = (
          "__module.model.model.language_model.layers."
          f"{layer}.self_attn/")
      if name == prefix + "aten::transpose/Transpose_3":
        selected.append({**row, "layer": layer,
                         "boundary_kind": "output_transpose"})
      elif name == prefix + "aten::mul/Multiply_6":
        selected.append({**row, "layer": layer,
                         "boundary_kind": "gate_multiply"})
      elif name == (
          "__module.model.model.language_model.layers."
          f"{layer}.self_attn.o_proj/ov_ext::linear/MatMul"):
        selected.append({**row, "layer": layer,
                         "boundary_kind": "output_projection"})
  boundary_counts = Counter(
      str(row["boundary_kind"]) for row in selected)
  return {
      "executed_counts": dict(sorted(counts.items())),
      "boundary_counts": dict(sorted(boundary_counts.items())),
      "boundary_exec_types": {
          kind: sorted({str(row.get("exec_type")) for row in selected
                        if row["boundary_kind"] == kind})
          for kind in sorted(boundary_counts)
      },
      "dynamic_quantize_count": counts["DynamicQuantize"],
      "rows": selected,
      "raw_profile_time_is_savings_evidence": False,
  }


def locked_output_projection_quantization() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  by_name = {layer.get("name"): layer for layer in root.find("layers")}
  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    stem = (
        "self.model.model.language_model.layers."
        f"{layer}.self_attn.o_proj.weight")
    tensors = {}
    for kind, suffix in (("weight", ""), ("zero_point", "/zero_point"),
                         ("scale", "/scale")):
      node = by_name[stem + suffix]
      data = node.find("data")
      shape = [int(value) for value in data.get("shape").split(",")]
      tensors[kind] = {
          "element_type": data.get("element_type"),
          "shape": shape,
      }
    innermost = tensors["weight"]["shape"][1] * tensors["weight"]["shape"][2]
    zp_groups = tensors["zero_point"]["shape"][1]
    scale_groups = tensors["scale"]["shape"][1]
    required_group_size = min(
        innermost // zp_groups, innermost // scale_groups)
    rows.append({
        "layer": layer,
        "innermost_size": innermost,
        "zero_point_group_count": zp_groups,
        "scale_group_count": scale_groups,
        "required_group_size": required_group_size,
        "tensors": tensors,
    })
  return {
      "rows": rows,
      "required_group_sizes": sorted({
          row["required_group_size"] for row in rows}),
      "all_have_static_weight_zero_point": all(
          row["tensors"]["zero_point"]["element_type"] == "u4"
          for row in rows),
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
      SEQ1302, SEQ1311, SEQ1313, SEQ1316, CONTROL_WORKER,
      GRAPH_SOURCE, CUSTOM_CONFIG, DQ_TRANSFORM, DQ_KERNEL, MODEL_XML)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gated-DQ inputs: " + ", ".join(missing))

  git = git_state(output)
  seq1302 = load_json(SEQ1302)
  seq1311 = load_json(SEQ1311)
  seq1313 = load_json(SEQ1313)
  seq1316 = load_json(SEQ1316)
  worker = load_json(CONTROL_WORKER)
  census = control_census(worker)
  quantization = locked_output_projection_quantization()
  graph_text = GRAPH_SOURCE.read_text(encoding="utf-8")
  config_text = CUSTOM_CONFIG.read_text(encoding="utf-8")
  transform_text = DQ_TRANSFORM.read_text(encoding="utf-8")
  dq_kernel_text = DQ_KERNEL.read_text(encoding="utf-8")

  producer_routes_closed = (
      seq1313.get("verdict")
          == "reject_attention_output_gate_fusion_after_component"
      and seq1313.get("evidence_checks_passed") is True
      and seq1313.get("activation_passed") is True
      and seq1313.get("correctness_passed") is True
      and seq1313.get("performance_passed") is False
      and seq1316.get("verdict")
          == "reject_token_major_value_output_after_component"
      and seq1316.get("evidence_checks_passed") is True
      and seq1316.get("activation_passed") is True
      and seq1316.get("correctness_passed") is True
      and seq1316.get("performance_passed") is False)

  locked = seq1311["locked_output_epilogue"]
  locked_exact = (
      seq1311.get("required_checks_passed") is True
      and locked.get("exact_output_epilogue_contract") is True
      and len(locked.get("rows", [])) == 10
      and all(row["direct_edges"]["gate_multiply->output_projection"]
              for row in locked["rows"]))
  runtime_exact = census["boundary_counts"] == {
      "gate_multiply": 10,
      "output_projection": 10,
      "output_transpose": 10,
  }

  dq_transform_contract = all(token in transform_text for token in (
      "const auto activation = m_fc->input_value(0);",
      "dyn_quan = std::make_shared<DynamicQuantize>(activation, config);",
      "dyn_quan->output(0)",
      "dyn_quan->output(1)",
      "same_attributes(cached.attrs, config)",
  ))
  dq_kernel_contract = all(token in dq_kernel_text for token in (
      "QUANTIZE_GROUP_SIZE",
      "ACT_MIN_VAL 0.003h",
      "sub_group_reduce_max",
      "convert_char4_rte",
      "output_scale",
  ))
  dq_group64_contract = (
      quantization["required_group_sizes"] == [64]
      and quantization["all_have_static_weight_zero_point"]
      and all(token in transform_text for token in (
          "adj_group_size = required_group_size;",
          "config.precomputed_reduction_dt = element::i32;",
          "config.precomputed_reduction = true;",
          "dyn_quan->output(dyn_quan_output_idx++)")))
  custom_multi_output_contract = all(token in graph_text for token in (
      "self.set_output_size(6)",
      "self.set_output_type(2, ov.Type.i8",
      "operation.output(1)",
  )) and all(token in config_text for token in (
      'type="output" port-index="0"',
      'type="output" port-index="1"',
      'type="output" port-index="2"',
  ))

  prior_union_ms = float(
      seq1302["budget"]["favorable_rms_plus_igc_union_ms"])
  residual_ms = float(seq1302["budget"]["residual_after_fixed_fc_ms"])
  required_component_ms = residual_ms - prior_union_ms
  prior_budget = seq1311["budget"]
  per_dispatch_us = (
      float(prior_budget["max_enqueue_us_per_dispatch"])
      + float(prior_budget["set_arguments_per_boundary_dispatch_us"]))
  # Three current dispatches become one consumer-side kernel per layer.
  current_dispatches = 30
  replacement_dispatches = 10
  removed_dispatches = current_dispatches - replacement_dispatches
  provider_ceiling_ms = removed_dispatches * per_dispatch_us / 1000.0
  expanded_favorable_union_ms = prior_union_ms + provider_ceiling_ms
  expanded_margin_ms = expanded_favorable_union_ms - residual_ms

  implementation_contract = {
      "operation": "IQ36GatedTransposeDynamicQuantize",
      "inputs": [
          "accepted head-major F16 attention output [B,16,Q,256]",
          "already-computed F16 sigmoid gate [B,Q,4096]",
      ],
      "outputs": [
          "shape-carrier F16 [B,Q,4096] used only before GPU lowering",
          "group-64 symmetric I8 activation [B,Q,4096]",
          "F16 reciprocal quantization scales [B,Q,64]",
          "I32 precomputed activation reductions [B,Q,64]",
      ],
      "gpu_lowering": (
          "teach DynamicQuantizeFullyConnected to consume the custom I8 and "
          "scale outputs instead of inserting a separate DynamicQuantize"),
      "arithmetic": (
          "transpose addressing, F16 gate multiply and round point, then the "
          "existing ACT_MIN_VAL/max/127/convert_char_rte group-64 contract"),
      "activation_gate": (
          "exactly ten custom fused-DQ rows; zero old Transpose_3, "
          "Multiply_6, and their ten associated DynamicQuantize rows"),
      "correctness_gate": "18/18 top-1 and unchanged core operation census",
      "performance_gate_ms": required_component_ms,
      "scope": "one no-GPU rewrite audit, then one candidate-only 2k/17 worker",
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("both_producer_side_layout_routes_are_closed",
            producer_routes_closed,
            seq1313_saving_ms=seq1313["performance"][
                "observed_median_saving_ms"],
            seq1316_saving_ms=seq1316["performance"][
                "observed_median_saving_ms"]),
      check("locked_ir_has_exact_ten_gate_to_output_projection_chains",
            locked_exact),
      check("control_executes_exact_ten_output_boundaries",
            runtime_exact, census=census),
      check("gpu_lowering_inserts_one_unique_dq_per_unique_fc_activation",
            dq_transform_contract
            and census["dynamic_quantize_count"] == 161,
            note=("the ten gate Multiply activations are unique; cached DQ is "
                  "shared only for identical activation node/output pairs")),
      check("locked_output_projection_requires_group64_with_reduction",
            dq_kernel_contract and dq_group64_contract,
            quantization=quantization,
            note=("requested group256 is adjusted to the U4 scale/zero-point "
                  "group64; static asymmetric weights require I32 reduction")),
      check("custom_operation_infrastructure_supports_multi_output_i8",
            custom_multi_output_contract),
      check("net_twenty_dispatch_ceiling_clears_component_cut",
            removed_dispatches == 20
            and provider_ceiling_ms > required_component_ms
            and expanded_margin_ms > 0.0,
            current_dispatches=current_dispatches,
            replacement_dispatches=replacement_dispatches,
            provider_ceiling_ms=provider_ceiling_ms,
            required_component_ms=required_component_ms,
            expanded_margin_ms=expanded_margin_ms),
      check("raw_profile_and_failed_route_times_are_not_added", True,
            raw_profile_time_counted=False,
            failed_route_point_estimates_counted=False,
            final_bundle_additivity_claimed=False),
      check("source_gate_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_compiles=0,
            model_workers=0, long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_consumer_side_gated_dynamic_quantize_source"
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
      "plugin_build_admitted_after_exact_rewrite_audit": (
          required_checks_passed),
      "candidate_workers_admitted_after_exact_rewrite_audit": (
          1 if required_checks_passed else 0),
      "additional_control_worker_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "producer_route_closure": {
          "seq1313": {
              "verdict": seq1313.get("verdict"),
              "activation_passed": seq1313.get("activation_passed"),
              "correctness_passed": seq1313.get("correctness_passed"),
              "performance": seq1313.get("performance"),
          },
          "seq1316": {
              "verdict": seq1316.get("verdict"),
              "activation_passed": seq1316.get("activation_passed"),
              "correctness_passed": seq1316.get("correctness_passed"),
              "performance": seq1316.get("performance"),
          },
      },
      "control_census": census,
      "source_contract": {
          "locked_output_epilogue_exact": locked_exact,
          "dq_transform_contract_exact": dq_transform_contract,
          "dq_kernel_contract_exact": dq_kernel_contract,
          "dq_group64_contract_exact": dq_group64_contract,
          "locked_output_projection_quantization": quantization,
          "custom_multi_output_contract_exact": custom_multi_output_contract,
      },
      "implementation_contract": implementation_contract,
      "budget": {
          "residual_after_fixed_fc_ms": residual_ms,
          "seq1302_favorable_rms_plus_igc_union_ms": prior_union_ms,
          "required_component_saving_ms": required_component_ms,
          "current_dispatches": current_dispatches,
          "replacement_dispatches": replacement_dispatches,
          "removed_dispatches": removed_dispatches,
          "max_enqueue_us_per_dispatch": prior_budget[
              "max_enqueue_us_per_dispatch"],
          "set_arguments_per_boundary_dispatch_us": prior_budget[
              "set_arguments_per_boundary_dispatch_us"],
          "favorable_provider_ceiling_ms": provider_ceiling_ms,
          "expanded_favorable_union_ms": expanded_favorable_union_ms,
          "expanded_union_margin_ms": expanded_margin_ms,
          "interpretation": (
              "source admission only; the component must independently save "
              "the required cut, and the eventual FC/RMS/IGC bundle must be "
              "rebuilt and measured together"),
      },
      "checks": checks,
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "oom_observed": False,
      },
      "decision": {
          "next_route": (
              "openvino_attention_gated_dynamic_quantize_rewrite"
              if required_checks_passed else None),
          "stop_rule": (
              "after an exact no-GPU all-ten rewrite, run one candidate-only "
              "short worker; close on activation, correctness, or "
              f"{required_component_ms:.7f}-ms cut failure"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_compiles": 0,
      "model_workers": 0,
      "long_workers": 0,
  })
  report = f"""# Attention consumer-side gated dynamic-quantize bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

Seq1313 and seq1316 both activate their exact all-ten producer-side layout
cuts and preserve 18/18 top-1 plus the core census, but regress the stable
short median by `{abs(float(seq1313['performance']['observed_median_saving_ms'])):.6f}`
and `{abs(float(seq1316['performance']['observed_median_saving_ms'])):.6f} ms`.
They are closed without repeat.

The distinct consumer boundary is exact: ten head-major attention outputs run
through `Transpose_3 -> Reshape_2 -> Multiply_6 -> compressed o_proj`, and the
GPU lowering inserts DynamicQuantize for each unique FC activation. Implement
one parameterized custom operation that performs transpose addressing, the F16
gate multiply round point, and the existing group-64 symmetric I8 quantizer,
then wire its I8/scales/I32 reductions directly into FullyConnectedCompressed.
The requested group-256 property is not the effective contract here: the
locked U4 scale and zero-point tensors force group 64 in all ten layers.

This replaces 30 current dispatches with ten fused rows, for a source-only
provider ceiling of `{provider_ceiling_ms:.7f} ms` versus the required
`{required_component_ms:.7f} ms`. Raw PERF_COUNT times and the two failed point
estimates are not added. The component must clear the cut on its own; any final
FC/RMS/IGC bundle must be rebuilt and measured together.

Admit only the source rewrite and exact no-GPU all-ten audit. If that passes,
admit one candidate-only 2k/17 worker against retained seq1304 behind the
4-GiB stop. No new control, repeat, long, ABBA, output512, product, or sweep
worker is funded. OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_edit_admitted": required_checks_passed,
      "provider_ceiling_ms": provider_ceiling_ms,
      "required_component_ms": required_component_ms,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
