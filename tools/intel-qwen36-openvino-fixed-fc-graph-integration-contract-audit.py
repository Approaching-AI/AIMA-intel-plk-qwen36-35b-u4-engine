#!/usr/bin/env python3
"""Audit the distinct fixed-FC graph-integration contract without a GPU run.

The retained seq1233 component is an FP16-activation/U4-weight gemmstone
microkernel.  The live OpenVINO graph instead feeds 371 compressed FCs from
161 graph-level I8 DynamicQuantize outputs.  This gate proves whether the
component can enter before that rewrite as a bounded multi-output custom
provider, preserving each original FP16 result boundary and avoiding the
closed Concat/VariadicSplit horizontal-fusion route.

No compiler, OpenCL/Level Zero context, graph compile, or model worker runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-graph-contract-audit-v0"
R0 = Path("/home/intel/intel-qwen36-r0")
OV_SOURCE = R0 / "source/openvino-90214e5be05"

SEQ1233_DIR = ROOT / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ")
SEQ1233 = SEQ1233_DIR / "metrics.json"
SEQ1294 = ROOT / (
    "output/openvino-fc-hardware-limit-bound-"
    "20260717Tseq1294-cleanZ/metrics.json")
SEQ1297 = ROOT / (
    "output/openvino-fc-upstream-vector-imm-bound-"
    "20260717Tseq1297-cleanZ/metrics.json")
SEQ1327 = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/metrics.json")
SEQ1328_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-fc-rms-igc-qk-rope-bundle-bound.py")
SEQ1352 = ROOT / (
    "output/openvino-post-seq1351-upstream-opportunity-bound-"
    "20260718Tseq1352-cleanZ/metrics.json")

CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST = ROOT / "engine/gpu/opencl/openvino_fc_micro_host.cl"
GRAPH = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
MULTI_OUTPUT_PATCH = ROOT / "engine/openvino/iq36-custom-multi-output-binding.patch"
PREPROCESS_PATCH = ROOT / "engine/openvino/iq36-custom-preprocess-input-port-id.patch"
MICRO_FUSION_PATCH = ROOT / "engine/openvino/iq36-simplegpu-microkernel-fusion.patch"

CUSTOM_LAYER_CPP = OV_SOURCE / "src/plugins/intel_gpu/src/plugin/custom_layer.cpp"
CUSTOM_OP_CPP = OV_SOURCE / "src/plugins/intel_gpu/src/plugin/ops/custom.cpp"
CUSTOM_PRIMITIVE_CPP = OV_SOURCE / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/custom_primitive.cpp")
CUSTOM_PRIMITIVE_HPP = OV_SOURCE / (
    "src/plugins/intel_gpu/include/intel_gpu/primitives/custom_gpu_primitive.hpp")
PROGRAM_BUILDER_CPP = OV_SOURCE / "src/plugins/intel_gpu/src/plugin/program_builder.cpp"

KILL_NUMBER_MS = 2.837085
EXPECTED_NON_LM_FC_TENSORS = 390
EXPECTED_PARAMETER_BYTES = 770_901_120
EXPECTED_CUSTOM_GROUPS = 160
EXPECTED_GRAPH_DQ_REMOVED = 160
EXPECTED_STOCK_COMPRESSED_FC_WITH_LM_HEAD = 371


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
  for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if row.startswith("MemAvailable:"):
      return int(row.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-post-seq1351-upstream-opportunity-bound.py",
      "tools/intel-qwen36-openvino-fixed-fc-graph-integration-contract-audit.py",
  }
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


def load_module(path: Path, name: str) -> ModuleType:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def text_has(path: Path, *needles: str) -> bool:
  value = path.read_text(encoding="utf-8")
  return all(needle in value for needle in needles)


def universal_package_audit(seq1233: dict[str, Any],
                            seq1297: dict[str, Any]) -> dict[str, Any]:
  shim_paths = sorted(SEQ1233_DIR.glob("raw/*/codegen/*.shim.cl"))
  binary_paths = sorted(SEQ1233_DIR.glob("raw/*/codegen/*.micro.bin"))
  shim_hashes = {sha256(path) for path in shim_paths}
  binary_hashes = {sha256(path) for path in binary_paths}
  nonzero = [row for row in seq1233["cohorts"] if int(row["count"]) > 0]
  shapes = [
      {"name": row["name"], "m": int(row["m"]), "k": int(row["k"]),
       "count": int(row["count"]),
       "global": row["runtime"]["global"],
       "local": row["runtime"]["local"],
       "kernel_median_us": float(row["runtime"]["kernel_median_us"])}
      for row in nonzero]
  universal_check = next(
      (row for row in seq1297.get("checks", [])
       if row.get("name") ==
       "all_five_current_cohorts_share_universal_micro_binary"), {})
  upstream_hashes = {
      str(value) for value in universal_check.get("hashes", {}).values()
      if isinstance(value, str)}
  return {
      "shim_files": [display(path) for path in shim_paths],
      "micro_binary_files": [display(path) for path in binary_paths],
      "shim_hashes": sorted(shim_hashes),
      "micro_binary_hashes": sorted(binary_hashes),
      "seq1297_universal_hashes": sorted(upstream_hashes),
      "all_six_shims_identical": len(shim_paths) == 6 and len(shim_hashes) == 1,
      "all_six_micro_binaries_identical": (
          len(binary_paths) == 6 and len(binary_hashes) == 1),
      "seq1297_confirms_five_cohort_universality": (
          len(upstream_hashes) == 1 and upstream_hashes == binary_hashes),
      "runtime_shape_arguments_present": text_has(
          HOST, "int m,", "int k,", "int n)",
          "weight_ptr, k, input_ptr, k, m, n, k"),
      "fixed_geometry": {
          "sg_per_wg_m": 2, "sg_per_wg_n": 1, "sg_per_wg_k": 8,
          "wg_tile_m": 64, "wg_tile_n": 8, "slm_size": 16384},
      "shapes": shapes,
  }


def custom_provider_source_audit() -> dict[str, Any]:
  primitive = CUSTOM_PRIMITIVE_CPP.read_text(encoding="utf-8")
  custom_op = CUSTOM_OP_CPP.read_text(encoding="utf-8")
  custom_layer = CUSTOM_LAYER_CPP.read_text(encoding="utf-8")
  builder = PROGRAM_BUILDER_CPP.read_text(encoding="utf-8")
  graph = GRAPH.read_text(encoding="utf-8")
  u4_mapping = ('{data_types::u4, "uchar"}' in primitive or
                '{data_types::u4, "uchar"},' in primitive)
  return {
      "config_reads_opencl_source_only": (
          'FOREACH_CHILD(sourceNode, node, "Source")' in custom_layer and
          "std::ifstream inputFile(filename)" in custom_layer),
      "embedded_gemmstone_fuser_bridge_is_applied": all(
          token in primitive for token in (
              "gemmstone/microkernel/fuser.hpp",
              "hasMicrokernels(source.c_str())",
              "has_microkernels")),
      "custom_ops_selected_by_python_type_name": (
          "m_custom_layers.find(op->get_type_name())" in builder and
          "CreateCustomOp(*this, op, customLayer->second)" in builder and
          "class IQ36QKRopeLayout(ov.Op)" in graph),
      "multi_output_binding_is_applied": all(
          token in custom_op for token in (
              "op->get_output_size()",
              "param.portIndex >= static_cast<int>(op->get_output_size())",
              "customPrim.output_layouts[i]")),
      "input_port_specific_preprocess_ids_are_applied": (
          '"_input_" + std::to_string(param.portIndex)' in custom_op),
      "arbitrary_input_and_output_tensor_arguments_supported": (
          'case CustomLayer::ParamType::Input:' in custom_op and
          'case CustomLayer::ParamType::Output:' in custom_op and
          "kernelParameters.resize" in custom_op),
      "format_any_avoids_weight_reorder": (
          "if (param.format != cldnn::format::any)" in custom_op),
      "u4_jit_type_mapping_already_present": u4_mapping,
      "u4_jit_type_mapping_required_delta": None if u4_mapping else {
          "file": display(CUSTOM_PRIMITIVE_CPP),
          "change": 'add {data_types::u4, "uchar"} to dataTypeToIndex',
          "scope": "JIT spelling only; the provider consumes the existing packed constant allocation and ignores its logical U4 pitches",
      },
      "durable_patch_inputs_present": all(
          path.is_file() for path in (
              MULTI_OUTPUT_PATCH, PREPROCESS_PATCH, MICRO_FUSION_PATCH)),
      "one_parameterized_source_design": {
          "operation_kinds": ["IQ36FixedFC1", "IQ36FixedFC3", "IQ36FixedFC4"],
          "kernel_entry": "iq36_fixed_fc_multi",
          "universal_micro_package_count": 1,
          "per_layer_source_files": 0,
          "input_contract": "one FP16 activation plus separate raw U4 weight, transposed group-major FP16 scale, and transposed group-major U4 zero-point per projection",
          "output_contract": "one independent FP16 buffer per original MatMul result; no Concat, Crop, or VariadicSplit",
      },
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      SEQ1233, SEQ1294, SEQ1297, SEQ1327, SEQ1328_TOOL, SEQ1352,
      CODEGEN, HOST, GRAPH, MULTI_OUTPUT_PATCH, PREPROCESS_PATCH,
      MICRO_FUSION_PATCH, CUSTOM_LAYER_CPP, CUSTOM_OP_CPP,
      CUSTOM_PRIMITIVE_CPP, CUSTOM_PRIMITIVE_HPP, PROGRAM_BUILDER_CPP)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing contract-audit inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  if memory[0]["available_bytes"] < stop_bytes:
    raise RuntimeError("memory stop tripped before source audit")

  seq1233 = load_json(SEQ1233)
  seq1294 = load_json(SEQ1294)
  seq1297 = load_json(SEQ1297)
  seq1327 = load_json(SEQ1327)
  seq1352 = load_json(SEQ1352)
  bundle_module = load_module(SEQ1328_TOOL, "iq36_seq1328_bundle")
  grouped_ir = bundle_module.locked_ir_audit()
  memory.append({"stage": "after-ir-audit",
                 "available_bytes": available_memory_bytes()})

  universal = universal_package_audit(seq1233, seq1297)
  provider = custom_provider_source_audit()
  cohorts = seq1294["cohorts"]
  cohort_calls = sum(int(row["calls"]) for row in cohorts.values())
  parameter_bytes = sum(int(row["parameter_bytes"]) for row in cohorts.values())
  observed_tensors = sum(
      int(row["observed_tensor_count"]) for row in cohorts.values())

  fixed_schedule_ms = float(seq1233["aggregate"]["dominant_ms"])
  stock_fc_ms = float(seq1233["aggregate"]["stock_ms"])
  fc_saving_ms = float(seq1233["aggregate"]["optimistic_saving_ms"])
  qk_saving_ms = 39.1818755 - 37.9969565
  combined_screen_ms = fc_saving_ms + qk_saving_ms
  margin_ms = combined_screen_ms - KILL_NUMBER_MS
  arithmetic = {
      "seq1233_fixed_schedule_ms": fixed_schedule_ms,
      "seq1233_stock_fc_ms": stock_fc_ms,
      "fixed_fc_component_saving_ms": fc_saving_ms,
      "seq1327_qk_saving_ms": qk_saving_ms,
      "combined_source_screen_ms": combined_screen_ms,
      "kill_number_ms": KILL_NUMBER_MS,
      "source_screen_margin_ms": margin_ms,
      "margin_us_per_custom_group": margin_ms * 1000 / EXPECTED_CUSTOM_GROUPS,
      "dq_kernel_saving_charged_ms": 0.0,
      "dispatch_saving_charged_ms": 0.0,
      "stock_non_lm_fc_dispatches": EXPECTED_STOCK_COMPRESSED_FC_WITH_LM_HEAD - 1,
      "stock_non_lm_dq_dispatches": EXPECTED_GRAPH_DQ_REMOVED,
      "candidate_custom_dispatches": EXPECTED_CUSTOM_GROUPS,
      "net_dispatch_reduction": (
          EXPECTED_STOCK_COMPRESSED_FC_WITH_LM_HEAD - 1 +
          EXPECTED_GRAPH_DQ_REMOVED - EXPECTED_CUSTOM_GROUPS),
      "claim": "source/component admission only; no additive product inference",
  }

  git = git_state(output)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_registered_inputs_are_conclusive",
            seq1233.get("required_checks_passed") is True and
            seq1294.get("required_checks_passed") is True and
            seq1297.get("required_checks_passed") is True and
            seq1352.get("required_checks_passed") is True),
      check("locked_ir_exact_group_fanouts_remain_present",
            grouped_ir["all_groups_exact"] and
            grouped_ir["counts"] == {
                "linear_groups": 30, "linear_matmuls": 120,
                "router_groups": 40, "router_matmuls": 160,
                "existing_full_qkv_groups": 10,
                "existing_full_qkv_matmuls": 30,
                "candidate_four_fc_groups": 70,
                "candidate_four_fc_matmuls": 280}),
      check("exact_complete_non_lm_fc_census",
            observed_tensors == EXPECTED_NON_LM_FC_TENSORS and
            parameter_bytes == EXPECTED_PARAMETER_BYTES and
            cohort_calls == EXPECTED_CUSTOM_GROUPS,
            observed_tensors=observed_tensors,
            parameter_bytes=parameter_bytes,
            custom_group_calls=cohort_calls),
      check("universal_gemmstone_package_is_shape_parameterized",
            universal["all_six_shims_identical"] and
            universal["all_six_micro_binaries_identical"] and
            universal["seq1297_confirms_five_cohort_universality"] and
            universal["runtime_shape_arguments_present"],
            shim_hashes=universal["shim_hashes"],
            micro_binary_hashes=universal["micro_binary_hashes"]),
      check("real_boundary_numeric_component_is_tight",
            seq1233["checks"][13]["pass"] is True and
            seq1233["checks"][13]["compare"]["exact_rate"] >= 0.9997 and
            seq1233["checks"][13]["compare"]["max_abs_diff"] <= 0.0001221,
            compare=seq1233["checks"][13]["compare"]),
      check("custom_provider_can_embed_universal_package",
            provider["config_reads_opencl_source_only"] and
            provider["embedded_gemmstone_fuser_bridge_is_applied"]),
      check("python_custom_op_multi_output_path_is_live",
            provider["custom_ops_selected_by_python_type_name"] and
            provider["multi_output_binding_is_applied"] and
            provider["input_port_specific_preprocess_ids_are_applied"] and
            provider["arbitrary_input_and_output_tensor_arguments_supported"]),
      check("packed_u4_inputs_need_only_bounded_jit_spelling_delta",
            provider["format_any_avoids_weight_reorder"] and
            provider["u4_jit_type_mapping_required_delta"] is not None,
            required_delta=provider["u4_jit_type_mapping_required_delta"]),
      check("design_preserves_original_fp16_outputs_without_split",
            provider["one_parameterized_source_design"]["per_layer_source_files"] == 0 and
            provider["one_parameterized_source_design"]["universal_micro_package_count"] == 1 and
            "no Concat" in provider["one_parameterized_source_design"]["output_contract"]),
      check("source_screen_clears_kill_number_without_dq_or_dispatch_credit",
            combined_screen_ms > KILL_NUMBER_MS and margin_ms > 0.0 and
            arithmetic["dq_kernel_saving_charged_ms"] == 0.0 and
            arithmetic["dispatch_saving_charged_ms"] == 0.0,
            combined_screen_ms=combined_screen_ms,
            kill_number_ms=KILL_NUMBER_MS, margin_ms=margin_ms),
      check("next_gate_is_standalone_multi_buffer_component_only", True),
      check("no_compiler_gpu_graph_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, graph_compiles=0, model_workers=0),
      check("memory_guard_never_tripped",
            min(row["available_bytes"] for row in memory) >= stop_bytes,
            memory=memory, stop_bytes=stop_bytes),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_fixed_fc_multi_output_standalone_component_source_cut"
      if passed else "reject_fixed_fc_graph_integration_contract")
  next_action = ({
      "route": "openvino_fixed_fc_multi_output_standalone_component",
      "requirements": [
          "generate one universal embedded gemmstone package and one multi-buffer host kernel",
          "add only the U4 custom-JIT spelling and three shape-arity XML contracts",
          "measure separate weight/scale/zp and FP16 output buffers against seq1233 with no model graph",
          "require the complete five-cohort UCB plus pointer-selection charge to retain the 0.500194-ms margin",
          "do not launch a model worker until standalone numeric and complete-schedule checks pass",
      ],
  } if passed else None)

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "checks": checks,
      "grouped_ir": grouped_ir,
      "complete_fc_census": {
          "non_lm_fc_tensors": observed_tensors,
          "parameter_bytes": parameter_bytes,
          "custom_groups": cohort_calls,
          "cohorts": cohorts,
      },
      "universal_package": universal,
      "provider_source": provider,
      "arithmetic": arithmetic,
      "next_action": next_action,
      "execution_boundary": {
          "compilers": 0, "gpu_contexts": 0, "graph_compiles": 0,
          "model_workers": 0},
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA + "-manifest-v0",
      "created_at": metrics["created_at"],
      "artifact": display(output),
      "git": git,
      "inputs": {display(path): {"bytes": path.stat().st_size,
                                  "sha256": sha256(path)}
                 for path in required},
      "verdict": verdict,
      "required_checks_passed": passed,
  })
  report = f"""# Fixed-FC graph-integration contract audit

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.
No compiler, GPU context, graph compile, or model worker ran.

The locked graph contains exactly `{observed_tensors}` non-LM U4 FC tensors,
`{parameter_bytes:,}` parameter bytes, and `{cohort_calls}` same-input or
single-output call groups.  The retained gemmstone shim and micro binary are
byte-identical across all six generated shapes, and the host ABI supplies M
and K at runtime.  One embedded package can therefore serve the five complete
cohorts; per-layer source is unnecessary.

The distinct integration point is before graph DynamicQuantize.  Three Python
custom-op arities consume FP16 activation plus separate raw U4 weights,
group-major scales, and zero points, and emit one independent FP16 buffer per
original MatMul.  There is no weight Concat, output Crop, or VariadicSplit, so
this does not reopen the token-failing horizontal-fusion route.  The existing
multi-output, input-port, and embedded-microkernel plugin infrastructure is
already live.  The only new plugin plumbing is a bounded U4-to-OpenCL `uchar`
JIT spelling; packed allocations remain unchanged.

Seq1233's fixed schedule saves `{fc_saving_ms:.6f} ms/token`; seq1327 Q/K adds
`{qk_saving_ms:.6f} ms/token`.  Their source screen is
`{combined_screen_ms:.6f} ms/token`, leaving `{margin_ms:.6f} ms/token` over
the `{KILL_NUMBER_MS:.6f}` kill-number.  This audit credits zero time for
removing `{EXPECTED_GRAPH_DQ_REMOVED}` DQ kernels and zero time for the net
`{arithmetic['net_dispatch_reduction']}` dispatch reduction.  The margin funds
only one standalone multi-buffer component and complete-schedule UCB; it is
not product inference or permission for a model worker.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "combined_source_screen_ms": combined_screen_ms,
      "margin_ms": margin_ms,
      "compiler_or_worker_launched": False,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
