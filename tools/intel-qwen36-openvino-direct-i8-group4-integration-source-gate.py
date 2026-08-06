#!/usr/bin/env python3
"""Gate one real-layer source integration of the promoted group-4 component."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-direct-i8-group4-integration-source-v0"
COMPONENT = ROOT / (
    "output/openvino-direct-i8-group4-attention-component-"
    "20260715Tseq1262-cleanZ/result.json")
REFINEMENT = ROOT / (
    "output/openvino-direct-i8-refinement-bound-"
    "20260715Tseq1261-cleanZ/metrics.json")
GRAPH = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
GATE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
XML = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
HELPERS = ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
PREFILL = ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl"
DECODE = ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl"

HOT_CAPACITY = 16_385
HOT_KEY_BLOCKS = (HOT_CAPACITY + 15) // 16
HOT_KEY_WORDS_PER_BLOCK = 2_048
KV_HEADS = 2
HEAD_DIM = 256
COLD_CAPACITY = 32_768
GROUP4_SCALE_BYTES = 128
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


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


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


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def state_bytes(planes: int, scale_bytes: int) -> dict[str, int]:
  hot_key = (
      KV_HEADS * (planes * HOT_KEY_BLOCKS + 1) *
      HOT_KEY_WORDS_PER_BLOCK * 4)
  hot_value = KV_HEADS * HOT_CAPACITY * HEAD_DIM * 2
  cold_kv = KV_HEADS * (COLD_CAPACITY + 1) * HEAD_DIM * 2
  cold_scales = (
      KV_HEADS * (COLD_CAPACITY + 1) * scale_bytes * 2)
  return {
      "hot_key": hot_key,
      "hot_value": hot_value,
      "cold_kv": cold_kv,
      "cold_scales": cold_scales,
      "total": hot_key + hot_value + cold_kv + cold_scales,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (COMPONENT, REFINEMENT, GRAPH, GATE, XML, HELPERS, PREFILL, DECODE)
  missing = [relative(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing integration inputs: " + ", ".join(missing))

  git = git_state()
  component = load_json(COMPONENT)
  refinement = load_json(REFINEMENT)
  graph = GRAPH.read_text(encoding="utf-8")
  gate = GATE.read_text(encoding="utf-8")
  xml_text = XML.read_text(encoding="utf-8")
  helpers = HELPERS.read_text(encoding="utf-8")
  prefill = PREFILL.read_text(encoding="utf-8")
  decode = DECODE.read_text(encoding="utf-8")
  ast.parse(graph)
  ast.parse(gate)
  xml_root = ElementTree.fromstring(f"<Root>{xml_text}</Root>")
  sample_memory("after-source-load", stop_bytes, memory)

  component_result = component.get("result", {})
  inference = component.get("performance_inference", {})
  component_exact = (
      component.get("verdict") ==
          "promote_direct_i8_group4_full_cold_component"
      and component.get("required_checks_passed") is True
      and component.get("graph_integration_admitted") is False
      and component_result.get("quant_group") == 4
      and component_result.get("state_bytes") == 54_525_952
      and float(inference.get("upper_confidence_bound_ms", 1.0)) <
          COMPONENT_CAP_MS)
  refinement_exact = (
      refinement.get("verdict") ==
          "admit_one_direct_i8_group4_full_cold_component"
      and refinement.get("component_admitted") is True
      and refinement.get("graph_integration_admitted") is False
      and refinement.get("pareto", {}).get("4", {}).get(
          "state_bytes", {}).get("total") == 54_525_952)

  layers = {
      node.attrib.get("name"): node
      for node in xml_root.findall("CustomLayer")}
  group4_layer = layers.get("IQ36DirectI8Group4HotAttentionGQA")
  group32_layer = layers.get("IQ36DirectI8HotAttentionGQA")
  group4_options = (
      group4_layer.find("CompilerOptions").attrib.get("options", "")
      if group4_layer is not None else "")
  group32_options = (
      group32_layer.find("CompilerOptions").attrib.get("options", "")
      if group32_layer is not None else "")

  graph_contract = {
      "separate_group4_operation_identity":
          "IQ36DirectI8Group4HotAttentionGQA" in graph,
      "group4_requires_fixed_layout":
          "direct_i8_group4_full_cold requires direct_i8_fixed_layout" in graph,
      "group4_scale_state_is_128_bytes":
          "GROUP4_SCALE_BYTES = 128" in graph,
      "group4_adds_third_hot_key_storage_plane":
          "hot_key_storage_planes = 3 if direct_i8_group4_full_cold else 2"
          in graph,
      "group32_default_remains_distinct":
          "direct_i8_custom_class(ov) if direct_i8_fixed_layout" in graph,
  }
  kernel_contract = {
      "fixed_group4_quantization":
          "#define IQ36_QUANT_GROUP 4U" in helpers
          and "#define IQ36_SCALE_GROUPS 64U" in helpers,
      "four_scale_cold_k_fragment":
          "const uint packed3 = intel_sub_group_block_read(" in helpers
          and "const half scale3 = iq36_direct_cold_key_scale(" in helpers,
      "dimension_major_hot_v_plane":
          "iq36_hot_value_dimension_i32_base" in helpers
          and "iq36_direct_hot_value_fragment" in decode,
      "full_logical_cold_decode":
          "const uint attention_cold_tokens = cold_tokens;" in decode,
      "group4_prefill_and_decode_quantizers":
          "pass * 32U + subgroup * 4U + (lane >> 2U)" in prefill
          and "pass * 32U + subgroup * 4U + (lane >> 2U)" in decode
          and prefill.count("sub_group_shuffle_xor") >= 4
          and decode.count("sub_group_shuffle_xor") >= 4,
      "both_hot_v_planes_are_published":
          "iq36_direct_store_hot_value_dimension(" in prefill
          and "iq36_direct_store_hot_value_dimension(" in decode,
  }
  xml_contract = {
      "group4_macro_isolated_to_new_type":
          "-DIQ36_DIRECT_I8_GROUP4_FULL_COLD=1" in group4_options
          and "-DIQ36_DIRECT_I8_FIXED_LAYOUT=1" in group4_options,
      "group32_type_is_unchanged":
          "-DIQ36_DIRECT_I8_FIXED_LAYOUT=1" in group32_options
          and "IQ36_DIRECT_I8_GROUP4_FULL_COLD" not in group32_options,
      "buffer_abi_is_still_13_inputs_6_outputs":
          group4_layer is not None
          and len(group4_layer.findall("./Buffers/Tensor")) == 19,
  }
  gate_contract = {
      "one_layer_only":
          "direct-i8-group4-full-cold is admitted for one real layer only"
          in gate,
      "four_gib_stop_is_worker_enforced":
          'parser.add_argument("--memory-stop-gib"' in gate
          and "worker skipped to avoid host OOM" in gate,
      "group4_codec_evidence_is_parameterized":
          "def quantize_group(" in gate
          and "direct_i8_quant_group" in gate
          and "unpack_dimension_major_hot_value" in graph,
  }

  group32_state = state_bytes(2, 16)
  group4_state = state_bytes(3, GROUP4_SCALE_BYTES)
  state_contract = {
      "group32_bytes": group32_state,
      "group4_bytes": group4_state,
      "incremental_bytes": group4_state["total"] - group32_state["total"],
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("promoted_component_is_exact", component_exact),
      check("refinement_bound_is_exact", refinement_exact),
      check("graph_abi_is_parameterized_and_isolated",
            all(graph_contract.values()), contract=graph_contract),
      check("kernel_contract_is_full_cold_group4",
            all(kernel_contract.values()), contract=kernel_contract),
      check("custom_layer_identity_preserves_group32",
            all(xml_contract.values()), contract=xml_contract),
      check("integration_worker_is_bounded_to_one_layer",
            all(gate_contract.values()), contract=gate_contract),
      check("one_layer_resident_state_is_bounded",
            group4_state["total"] == 117_508_608
            and state_contract["incremental_bytes"] == 31_474_112
            and group4_state["total"] < 128 * 1024**2,
            state=state_contract),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_layer_group4_2k_compile_and_correctness"
      if required_checks_passed else
      "reject_group4_integration_before_openvino_compile")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_integration_admitted": required_checks_passed,
      "openvino_compile_admitted": required_checks_passed,
      "short_2k_worker_admitted": required_checks_passed,
      "one_layer_32k_worker_admitted": False,
      "all_ten_worker_admitted": False,
      "product_worker_admitted": False,
      "gpu_worker_launched": False,
      "selected_contract": {
          "target_layers": [3],
          "lanes": ["2k"],
          "direct_i8_fixed_layout": True,
          "direct_i8_group4_full_cold": True,
          "quant_group": 4,
          "build_or_worker_parallelism": 1,
          "memory_stop_bytes": stop_bytes,
          "passing_action": (
              "admit one clean layer-3 32k teacher-forced worker"),
          "failing_action": (
              "close group-4 integration without a 32k or product worker"),
      },
      "state_bytes": state_contract,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {relative(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Direct-I8 group-4 real-layer source gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The promoted component UCB is
`{inference.get('upper_confidence_bound_ms')} ms` against the exact
`{COMPONENT_CAP_MS}-ms` cap. The real-layer carrier preserves the group-32
operation identity, adds one dimension-major hot-V plane, expands scale state
to 128 bytes per row, and attends the full logical cold prefix. Its physical
resident state is `{group4_state['total']}` bytes for one layer, an increment
of `{state_contract['incremental_bytes']}` bytes over the group-32 carrier.

Passing admits one isolated layer-3 2k compile/correctness worker with the
4-GiB stop. It does not admit 32k, all-ten, ABBA, or product execution.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": relative(output),
      "verdict": verdict,
      "group4_state_bytes": group4_state["total"],
      "incremental_state_bytes": state_contract["incremental_bytes"],
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
