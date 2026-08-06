#!/usr/bin/env python3
"""Bound one split-state-owner hot16k/cold16k K2/V4 component.

This gate is source/evidence only.  It closes the current monolithic
OpenVINO integration, derives the exact state/traffic change from the promoted
K2/V4 partial/reduce/update component, and admits at most one fixed 20-sample
standalone component.  It launches no compiler or GPU worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-split-state-owner-hot16k-k2-v4-bound-v0"

STATUS = REPO / "doc/active" / WS / "STATUS.md"
ROUTES = REPO / "doc/active" / WS / "routes-ledger.json"
COMPONENT = REPO / (
    "output/openvino-direct-i8-hybrid-k2-v4-attention-component-"
    "20260715Tseq1269-cleanZ/result.json")
CORRECTNESS = REPO / (
    "output/openvino-direct-i8-hybrid-k2-v4-integration-"
    "20260715Tseq1272-layer3-32k-cleanZ/metrics.json")
PROFILE = REPO / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1273-hybrid-k2-v4-layer3-32k-warm25-cleanZ/metrics.json")
AUDIT = REPO / (
    "output/openvino-direct-i8-group4-dispatch-audit-"
    "20260715Tseq1266-diagnostic")
INTEGRATED_ZE = AUDIT / "integrated-decode/.ze_info"
STANDALONE_ZE = AUDIT / "standalone-disasm/.ze_info"
INTEGRATED_ASM = AUDIT / (
    "integrated-decode/.text.iq36_hot_attention_single_owner.asm")
PARTIAL_ASM = AUDIT / (
    "standalone-disasm/.text.iq36_direct_i8_hotcold_partial.asm")
REDUCE_ASM = AUDIT / (
    "standalone-disasm/.text.iq36_direct_i8_hotcold_reduce.asm")
UPDATE_ASM = AUDIT / (
    "standalone-disasm/.text.iq36_direct_i8_update_state.asm")
COMPONENT_SOURCE = REPO / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
COMPONENT_RUNNER = REPO / "engine/tools/direct_i8_hotcold_gqa_decode.cpp"
GRAPH_BUILDER = REPO / "tools/intel_qwen36_openvino_hot_cold_attention.py"

CONTEXT = 32768
BASE_HOT = 8192
BASE_COLD = CONTEXT - BASE_HOT
NEXT_HOT = 16384
NEXT_COLD = CONTEXT - NEXT_HOT
HEAD_DIM = 256
KV_HEADS = 2
KEY_GROUP = 2
VALUE_GROUP = 4
F16_BYTES = 2
I8_BYTES = 1
CURRENT_KV_READ_BYTES = KV_HEADS * HEAD_DIM * F16_BYTES * 2
COMPONENT_CAP_MS = 0.5618915


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
  try:
    return str(path.relative_to(REPO))
  except ValueError:
    return str(path)


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


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    output_relative = str(output.relative_to(REPO))
  except ValueError:
    output_relative = ""
  status = [
      row for row in status
      if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def state_bytes(hot: int, cold: int) -> dict[str, int]:
  hot_kv = hot * KV_HEADS * HEAD_DIM * F16_BYTES * 2
  cold_k_i8 = cold * KV_HEADS * HEAD_DIM * I8_BYTES
  cold_v_i8 = cold * KV_HEADS * HEAD_DIM * I8_BYTES
  cold_k_scales = cold * KV_HEADS * (HEAD_DIM // KEY_GROUP) * F16_BYTES
  cold_v_scales = cold * KV_HEADS * (HEAD_DIM // VALUE_GROUP) * F16_BYTES
  return {
      "hot_kv": hot_kv,
      "cold_k_i8": cold_k_i8,
      "cold_v_i8": cold_v_i8,
      "cold_k_scales": cold_k_scales,
      "cold_v_scales": cold_v_scales,
      "total": (
          hot_kv + cold_k_i8 + cold_v_i8 + cold_k_scales +
          cold_v_scales),
  }


def line_count(path: Path) -> int:
  return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def kernel_execution_block(text: str, name: str) -> str:
  match = re.search(
      rf"  - name:\s+{re.escape(name)}\n(.*?)(?=\n  - name:|\n"
      r"kernels_misc_info:|\Z)", text, re.DOTALL)
  return match.group(1) if match else ""


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      STATUS, ROUTES, COMPONENT, CORRECTNESS, PROFILE, INTEGRATED_ZE,
      STANDALONE_ZE, INTEGRATED_ASM, PARTIAL_ASM, REDUCE_ASM, UPDATE_ASM,
      COMPONENT_SOURCE, COMPONENT_RUNNER, GRAPH_BUILDER)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing split-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  routes = load_json(ROUTES)
  component = load_json(COMPONENT)
  correctness = load_json(CORRECTNESS)
  profile = load_json(PROFILE)
  status_text = STATUS.read_text(encoding="utf-8")
  source_text = COMPONENT_SOURCE.read_text(encoding="utf-8")
  runner_text = COMPONENT_RUNNER.read_text(encoding="utf-8")
  graph_text = GRAPH_BUILDER.read_text(encoding="utf-8")
  integrated_ze = INTEGRATED_ZE.read_text(encoding="utf-8")
  standalone_ze = STANDALONE_ZE.read_text(encoding="utf-8")
  sample_memory("after-evidence-load", stop_bytes, memory)

  component_result = component["result"]
  component_inference = component["performance_inference"]
  component_ucb = float(component_inference["upper_confidence_bound_ms"])
  base_bytes = state_bytes(BASE_HOT, BASE_COLD)
  next_bytes = state_bytes(NEXT_HOT, NEXT_COLD)
  conservative_next_bytes = next_bytes["total"] + CURRENT_KV_READ_BYTES
  conservative_ratio = conservative_next_bytes / base_bytes["total"]
  scaled_ucb = component_ucb * conservative_ratio
  margin = COMPONENT_CAP_MS - scaled_ucb

  integrated_block = kernel_execution_block(
      integrated_ze, "iq36_hot_attention_single_owner")
  partial_block = kernel_execution_block(
      standalone_ze, "iq36_direct_i8_hotcold_partial")
  reduce_block = kernel_execution_block(
      standalone_ze, "iq36_direct_i8_hotcold_reduce")
  update_block = kernel_execution_block(
      standalone_ze, "iq36_direct_i8_update_state")
  codegen = {
      "integrated": {
          "asm_lines": line_count(INTEGRATED_ASM),
          "simd16": "simd_size:       16" in integrated_block,
          "grf128": "grf_count:       128" in integrated_block,
          "slm18528": "slm_size:        18528" in integrated_block,
      },
      "partial": {
          "asm_lines": line_count(PARTIAL_ASM),
          "simd16": "simd_size:       16" in partial_block,
          "grf128": "grf_count:       128" in partial_block,
          "slm18496": "slm_size:        18496" in partial_block,
      },
      "reduce": {
          "asm_lines": line_count(REDUCE_ASM),
          "simd32": "simd_size:       32" in reduce_block,
          "grf128": "grf_count:       128" in reduce_block,
      },
      "update": {
          "asm_lines": line_count(UPDATE_ASM),
          "simd32": "simd_size:       32" in update_block,
          "grf64": "grf_count:       64" in update_block,
      },
  }

  current_topology = {
      "three_component_kernels_exist": all(
          name in source_text for name in (
              "iq36_direct_i8_update_state",
              "iq36_direct_i8_hotcold_partial",
              "iq36_direct_i8_hotcold_reduce")),
      "component_runner_is_in_order": (
          "clCreateCommandQueue(" in runner_text and
          "CL_QUEUE_PROFILING_ENABLE" in runner_text and
          "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" not in runner_text),
      "current_component_order_is_update_partial_reduce": (
          runner_text.index("clEnqueueNDRangeKernel(queue, update") <
          runner_text.index("clEnqueueNDRangeKernel(queue, partial") <
          runner_text.index("clEnqueueNDRangeKernel(queue, reduce")),
      "fixed_graph_has_one_combined_state_owner": (
          "operation_inputs = [" in graph_text and
          "query, reads[0].output(0), reads[1].output(0)" in graph_text and
          "fixed-capacity in-place" in graph_text),
      "fixed_graph_does_not_fan_out_assign_writers": (
          "if fixed_cold_capacity is None:" in graph_text and
          "new_sinks.append(ov.opset13.assign(" in graph_text),
  }

  failed_profile_checks = {
      row["name"] for row in profile["checks"] if row.get("pass") is False}
  comparisons = correctness.get("comparisons", {}).get("32k", {})
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("active_gate_selects_exact_split_hot16k_bound",
            routes.get("active_route", {}).get("id") ==
                "openvino_split_state_owner_hot16k_k2_v4_bound"
            and "split-state-owner hot16k/cold16k K2/V4 component bound" in
                status_text),
      check("hybrid_k2_v4_component_is_clean_and_promoted",
            component.get("required_checks_passed") is True
            and component.get("component_promoted") is True
            and component.get("graph_integration_admitted") is False
            and component_result.get("key_quant_group") == KEY_GROUP
            and component_result.get("value_quant_group") == VALUE_GROUP
            and component_result.get("state_bytes") == base_bytes["total"]
            and component_inference.get("sample_count") == 20
            and component_ucb == 0.540728),
      check("hybrid_one_layer_32k_semantics_are_exact",
            correctness.get("required_checks_passed") is True
            and correctness.get("direct_i8_hybrid_k2_v4") is True
            and correctness.get("target_layers") == [3]
            and comparisons.get("candidate_top1") == [271, 248068]
            and all(
                row.get("kld_reference_to_candidate", 1.0) <= 0.005
                for row in comparisons.get("distribution", []))
            and comparisons.get("isolated_stock_cold", {}).get(
                "3", {}).get("all_exact") is True),
      check("current_single_owner_integration_is_closed",
            profile.get("attribution_checks_passed") is False
            and profile.get("carrier_admission_passed") is True
            and profile.get("hybrid_k2_v4_integration_inference", {}).get(
                "upper_confidence_bound_ms") == 1.121354
            and profile.get("hybrid_k2_v4_max_kld") ==
                0.019071362398716374
            and failed_profile_checks == {
                "hybrid_k2_v4_integrated_decode_ucb_clears_component_cap",
                "hybrid_k2_v4_all_profile_distributions_pass"}),
      check("dispatch_codegen_proves_material_topology_gap",
            codegen["integrated"] == {
                "asm_lines": 6625, "simd16": True, "grf128": True,
                "slm18528": True}
            and codegen["partial"] == {
                "asm_lines": 2992, "simd16": True, "grf128": True,
                "slm18496": True}
            and codegen["reduce"] == {
                "asm_lines": 324, "simd32": True, "grf128": True}
            and codegen["update"] == {
                "asm_lines": 219, "simd32": True, "grf64": True},
            codegen=codegen),
      check("existing_sources_support_one_bounded_topology_change",
            all(current_topology.values()), current_topology=current_topology),
      check("hot16k_cold16k_state_accounting_is_exact",
            base_bytes["total"] == 60_817_408
            and next_bytes == {
                "hot_kv": 33_554_432,
                "cold_k_i8": 8_388_608,
                "cold_v_i8": 8_388_608,
                "cold_k_scales": 8_388_608,
                "cold_v_scales": 4_194_304,
                "total": 62_914_560}),
      check("conservative_byte_scaled_ucb_clears_cap",
            conservative_next_bytes == 62_916_608
            and scaled_ucb < COMPONENT_CAP_MS
            and margin > 0.0,
            basis_ucb_ms=component_ucb,
            basis_state_bytes=base_bytes["total"],
            next_state_bytes=next_bytes["total"],
            duplicate_current_kv_read_bytes=CURRENT_KV_READ_BYTES,
            conservative_ratio=conservative_ratio,
            scaled_ucb_ms=scaled_ucb,
            cap_ms=COMPONENT_CAP_MS,
            margin_ms=margin),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed
  verdict = (
      "admit_one_split_state_owner_hot16k_k2_v4_component"
      if admitted else "reject_split_state_owner_hot16k_before_source")

  component_contract = {
      "context_tokens": CONTEXT,
      "logical_hot_tokens": NEXT_HOT,
      "logical_cold_tokens": NEXT_COLD,
      "key_quant_group": KEY_GROUP,
      "value_quant_group": VALUE_GROUP,
      "key_pack_dimensions": 4,
      "state_bytes": next_bytes,
      "timed_topology": [
          "stateless_hotcold_partial", "stateless_reduce", "state_update"],
      "execution_order": "partial_then_reduce_then_update",
      "state_writer_count": 1,
      "future_graph_dependency": (
          "the updater consumes a reduce-completion carrier and is the only "
          "operation allowed to mutate all six state planes"),
      "workspace_shape_change": "none; 64 chunks of 512 tokens",
      "maximum_one_sided_95_latency_bound_ms": COMPONENT_CAP_MS,
      "warmup_samples": 5,
      "measured_samples": 20,
      "build_parallelism": 1,
      "serial_gpu_worker": True,
      "memory_stop_bytes": stop_bytes,
      "variant_sweep_allowed": False,
      "graph_source_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "passing_action": (
          "run a source-only split OpenVINO topology gate before any graph "
          "edit or correctness worker"),
      "failing_action": (
          "close the split component and enter route-exhaustion reflection"),
  }
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": admitted,
      "compile_admitted": admitted,
      "gpu_worker_admitted": admitted,
      "component_admitted": admitted,
      "graph_source_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "gpu_worker_launched": False,
      "product_claim_allowed": False,
      "basis_component": {
          "ucb_ms": component_ucb,
          "state_bytes": base_bytes,
          "timed_scope": component_result.get("timed_scope"),
      },
      "closed_integration": {
          "ucb_ms": profile.get(
              "hybrid_k2_v4_integration_inference", {}).get(
                  "upper_confidence_bound_ms"),
          "max_kld": profile.get("hybrid_k2_v4_max_kld"),
      },
      "byte_scaled_timing": {
          "next_state_bytes": next_bytes,
          "duplicate_current_kv_read_bytes": CURRENT_KV_READ_BYTES,
          "conservative_bytes": conservative_next_bytes,
          "ratio": conservative_ratio,
          "scaled_ucb_ms": scaled_ucb,
          "cap_ms": COMPONENT_CAP_MS,
          "margin_ms": margin,
      },
      "codegen": codegen,
      "current_topology": current_topology,
      "selected_component": component_contract,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Split-state-owner hot16k/cold16k K2/V4 bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The current single-owner integration is closed at UCB
`{result['closed_integration']['ucb_ms']} ms` and maximum KLD
`{result['closed_integration']['max_kld']}`.  Stored codegen is materially
different: the monolith is 6625 assembly lines, while the promoted carrier is
a 2992-line partial kernel plus 324-line reduce and 219-line update kernels.

Logical hot16k/cold16k K2/V4 state is `{next_bytes['total']}` bytes.  Charging
an additional `{CURRENT_KV_READ_BYTES}` bytes for a separate current-K/V read
and scaling the measured `{component_ucb} ms` UCB yields `{scaled_ucb:.9f} ms`,
only `{margin:.9f} ms` below the exact `{COMPONENT_CAP_MS} ms` cap.

This admits exactly one fixed component: 64 unchanged 512-token chunks,
stateless partial then reduce, followed by one state updater.  It permits five
warmups and twenty measured samples, `-j1`, one serial GPU worker, no variant
sweep, and the 4-GiB available-memory stop.  Graph source, correctness, long,
product, ABBA, and speed claims remain blocked.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "scaled_ucb_ms": scaled_ucb,
      "margin_ms": margin,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
