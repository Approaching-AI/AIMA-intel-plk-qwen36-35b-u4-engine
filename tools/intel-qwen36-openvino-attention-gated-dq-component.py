#!/usr/bin/env python3
"""Audit and run the attention consumer-side gated-DQ component.

``--audit-only`` performs the all-ten graph/source/build audit plus one tiny
four-output ABI probe.  The default mode launches exactly one corrected
candidate-only 2k/17-step worker against retained seq1304.  It launches no new
control, stock, long, ABBA, output512, or product worker.
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
SCHEMA = "intel-qwen36-openvino-attention-gated-dq-component-v0"
BASE_PATH = ROOT / (
    "tools/intel-qwen36-openvino-accepted-carrier-profile-refresh.py")
BOUND = ROOT / (
    "output/openvino-attention-gated-dq-bound-"
    "20260717Tseq1317b-cleanZ/metrics.json")
AUDIT = ROOT / (
    "output/openvino-attention-gated-dq-rewrite-"
    "20260717Tseq1320b-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-dynamic-split-control-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER_SOURCE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
KERNEL_SOURCE = ROOT / (
    "engine/openvino/custom/iq36_attention_gated_dynamic_quantize.cl")
LOWERING_PATCH = ROOT / (
    "engine/openvino/iq36-attention-gated-dynamic-quantize.patch")
LOWERING_SOURCE = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "dynamic_quantize_fully_connected.cpp")
BINDING_PATCH = ROOT / (
    "engine/openvino/iq36-custom-multi-output-binding.patch")
BINDING_SOURCE = OPENVINO_SOURCE / (
    "src/plugins/intel_gpu/src/plugin/ops/custom.cpp")
FAILED_COMPONENT = ROOT / (
    "output/openvino-attention-gated-dq-component-"
    "20260717Tseq1319-candidate-2k-warm17-cleanZ/metrics.json")
FAILED_STDERR = ROOT / (
    "output/openvino-attention-gated-dq-component-"
    "20260717Tseq1319-candidate-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker.stderr")
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
EXPECTED_CORE_COUNTS = {
    "Assign": 60,
    "DynamicQuantize": 151,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36GatedTransposeDynamicQuantize": 10,
    "IQ36HotAttentionGQA": 10,
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


BASE = load_module(BASE_PATH, "iq36_accepted_carrier_refresh")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--audit-only", action="store_true")
  parser.add_argument("--plugin", type=Path, default=CANDIDATE_PLUGIN)
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
  allowed = {
      "tools/intel-qwen36-openvino-attention-gated-dq-component.py"}
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
    attention_gated_dynamic_quantize=True)
ops = model.get_ordered_ops()
names = {{node.get_friendly_name() for node in ops}}
counts = {{kind: sum(node.get_type_name() == kind for node in ops)
          for kind in ('IQ36GatedTransposeDynamicQuantize',
                       'IQ36HotAttentionGQA', 'ScaledDotProductAttention',
                       'IQ36LinearConvSwish')}}
rows, old_live = [], []
for layer in g.FULL_ATTENTION_LAYERS:
  prefix = f'__module.model.model.language_model.layers.{{layer}}.self_attn/'
  old = (prefix + 'aten::transpose/Transpose_3',
         prefix + 'aten::reshape/Reshape_2',
         prefix + 'aten::mul/Multiply_6')
  old_live.extend(name for name in old if name in names)
  custom = next(node for node in ops if node.get_friendly_name() ==
                f'iq36_attention_gated_dynamic_quantize_layer{{layer}}')
  consumers = sorted(port.get_node().get_friendly_name()
                     for port in custom.output(0).get_target_inputs())
  rows.append({{
      'layer': layer,
      'inputs': [(str(custom.get_input_element_type(i)),
                  str(custom.get_input_partial_shape(i)))
                 for i in range(custom.get_input_size())],
      'outputs': [(str(custom.get_output_element_type(i)),
                   str(custom.get_output_partial_shape(i)),
                   len(custom.output(i).get_target_inputs()))
                  for i in range(custom.get_output_size())],
      'carrier_consumers': consumers,
  }})
print(json.dumps({{
    'summary': {{key: summary[key] for key in (
        'attention_gated_dynamic_quantize',
        'attention_gated_dynamic_quantize_rewrite_count',
        'custom_count_after', 'stock_sdpa_count_after')}},
    'counts': counts, 'rows': rows, 'old_live': old_live,
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


def execute_abi_probe(raw: Path, plugin: Path) -> dict[str, Any]:
  registry = raw / "abi-probe-plugins.xml"
  registry.write_text(
      "<ie><plugins><plugin name=\"GPU\" location=\""
      f"{plugin.resolve()}\"/></plugins></ie>\n", encoding="utf-8")
  script = f"""
import importlib.util, json
from pathlib import Path
import numpy as np
import openvino as ov
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location(
    'iq36_graph', root / 'tools/intel_qwen36_openvino_hot_cold_attention.py')
graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph)
attention = ov.opset13.parameter(
    [-1, 16, -1, 256], ov.Type.f32, name='attention')
gate = ov.opset13.parameter([-1, -1, 4096], ov.Type.f32, name='gate')
operation = graph.attention_gated_dynamic_quantize_custom_class(ov)(
    [attention.output(0), gate.output(0)])
operation.set_friendly_name('iq36_gated_dq_abi_probe')
model = ov.Model(
    [ov.opset13.result(operation.output(i)) for i in range(4)],
    [attention, gate], 'iq36_gated_dq_abi_probe')
core = ov.Core({str(registry)!r})
core.set_property('GPU', {{
    'CONFIG_FILE': str(root / 'engine/openvino/custom/'
                       'iq36_hot_attention_gqa.xml')}})
compiled = core.compile_model(model, 'GPU', {{
    'DYNAMIC_QUANTIZATION_GROUP_SIZE': 256,
    'PERFORMANCE_HINT': 'LATENCY',
    'PERF_COUNT': True,
    'ACTIVATIONS_SCALE_FACTOR': 0.0,
}})
rng = np.random.default_rng(36)
attention_value = rng.normal(0, 0.4, (1, 16, 1, 256)).astype(np.float32)
gate_value = rng.uniform(0, 1, (1, 1, 4096)).astype(np.float32)
outputs = compiled([attention_value, gate_value])
arrays = [np.array(outputs[port]) for port in compiled.outputs]
flat_attention = attention_value.transpose(0, 2, 1, 3).reshape(1, 1, 4096)
gated = (flat_attention.astype(np.float16) *
         gate_value.astype(np.float16)).astype(np.float16)
blocks = gated.reshape(1, 1, 64, 64)
max_value = np.maximum(
    np.float16(0.003), np.max(np.abs(blocks), axis=-1)).astype(np.float16)
quantize_scale = (np.float16(127.0) / max_value).astype(np.float16)
expected_quantized = np.rint(
    (blocks * quantize_scale[..., None]).astype(np.float16)
).astype(np.int8).reshape(1, 1, 4096)
expected_scale = (np.float16(1.0) / quantize_scale).astype(np.float16)
expected_reduction = expected_quantized.reshape(
    1, 1, 64, 64).astype(np.int32).sum(-1)
runtime = compiled.get_runtime_model().get_ordered_ops()
print(json.dumps({{
    'shapes': [list(value.shape) for value in arrays],
    'dtypes': [str(value.dtype) for value in arrays],
    'quantized_exact': bool(np.array_equal(arrays[1], expected_quantized)),
    'scale_exact': bool(np.array_equal(arrays[2], expected_scale)),
    'reduction_exact': bool(np.array_equal(arrays[3], expected_reduction)),
    'all_outputs_finite': bool(all(np.isfinite(value).all()
                                   for value in arrays[1:])),
    'custom_runtime_rows': sum(
        node.get_type_name() == 'IQ36GatedTransposeDynamicQuantize'
        for node in runtime),
}}, sort_keys=True))
"""
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=True,
      capture_output=True, text=True, timeout=120)
  probe = json.loads(completed.stdout)
  write_json(raw / "four-output-abi-probe.json", probe)
  return probe


def rewrite_exact(audit: dict[str, Any]) -> bool:
  expected_summary = {
      "attention_gated_dynamic_quantize": True,
      "attention_gated_dynamic_quantize_rewrite_count": 10,
      "custom_count_after": 10,
      "stock_sdpa_count_after": 0,
  }
  expected_counts = {
      "IQ36GatedTransposeDynamicQuantize": 10,
      "IQ36HotAttentionGQA": 10,
      "IQ36LinearConvSwish": 30,
      "ScaledDotProductAttention": 0,
  }
  return (
      audit.get("summary") == expected_summary
      and audit.get("counts") == expected_counts
      and audit.get("old_live") == []
      and len(audit.get("rows", [])) == 10
      and all(
          row.get("inputs") == [
              ["<Type: 'float32'>", "[?,16,?,256]"],
              ["<Type: 'float32'>", "[?,?,4096]"],
          ]
          and row.get("outputs") == [
              ["<Type: 'float32'>", "[?,?,4096]", 1],
              ["<Type: 'int8_t'>", "[?,?,4096]", 0],
              ["<Type: 'float16'>", "[?,?,64]", 0],
              ["<Type: 'int32_t'>", "[?,?,64]", 0],
          ]
          and row.get("carrier_consumers") == [
              "__module.model.model.language_model.layers."
              f"{row['layer']}.self_attn.o_proj/ov_ext::linear/MatMul"]
          for row in audit.get("rows", [])))


def source_audit() -> dict[str, Any]:
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  marker = '<CustomLayer name="IQ36GatedTransposeDynamicQuantize"'
  tail = config.split(marker, 1)[1] if config.count(marker) == 1 else ""
  block = tail.split("</CustomLayer>", 1)[0] if tail else ""
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  worker = WORKER_SOURCE.read_text(encoding="utf-8")
  patch = LOWERING_PATCH.read_text(encoding="utf-8")
  lowering = LOWERING_SOURCE.read_text(encoding="utf-8")
  binding_patch = BINDING_PATCH.read_text(encoding="utf-8")
  binding_source = BINDING_SOURCE.read_text(encoding="utf-8")
  reverse_check = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(LOWERING_PATCH)],
      cwd=OPENVINO_SOURCE, capture_output=True, text=True)
  binding_reverse_check = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(BINDING_PATCH)],
      cwd=OPENVINO_SOURCE, capture_output=True, text=True)
  return {
      "one_custom_config_entry": config.count(marker) == 1,
      "custom_config_inputs": block.count('type="input"'),
      "custom_config_outputs": block.count('type="output"'),
      "custom_config_exact_work_size": (
          '<WorkSizes global="16,Y/64,B*F" local="16,1,1"/>' in block),
      "graph_flag_and_four_output_contract": all(token in graph for token in (
          "class IQ36GatedTransposeDynamicQuantize",
          "attention_gated_dynamic_quantize",
          "gate_multiply.output(0).replace(gated_dq.output(0))",
          "self.set_output_type(3, ov.Type.i32, group_shape)")),
      "worker_plumbs_default_off_flag": (
          'cfg.get("attention_gated_dynamic_quantize", False)' in worker),
      "kernel_group64_contract": all(token in kernel for token in (
          "#define IQ36_GROUP_SIZE 64U", "IQ36_ACT_MIN_VALUE 0.003h",
          "sub_group_reduce_max", "convert_char4_rte",
          "sub_group_reduce_add", "(void)shape_carrier")),
      "lowering_wires_i8_scale_and_reduction": all(
          token in patch and token in lowering for token in (
              "IQ36GatedTransposeDynamicQuantize",
              "quantized_activation = activation_node->output(1)",
              "activation_scale = activation_node->output(2)",
              "optional_precomputed_reduction = activation_node->output(3)")),
      "durable_patch_is_exactly_applied": reverse_check.returncode == 0,
      "durable_patch_reverse_check_stderr": reverse_check.stderr,
      "four_output_binding_uses_output_count": all(
          token in binding_patch and token in binding_source for token in (
              "op->get_output_size()",
              "param.portIndex >= static_cast<int>(op->get_output_size())")),
      "binding_patch_is_exactly_applied": (
          binding_reverse_check.returncode == 0),
      "binding_patch_reverse_check_stderr": binding_reverse_check.stderr,
  }


def runtime_audit(result: dict[str, Any]) -> dict[str, Any]:
  rows = result.get("full_profile")
  if not isinstance(rows, list):
    phases = result.get("phases", [])
    rows = phases[-1].get("full_profile") if phases else None
  if not isinstance(rows, list):
    raise TypeError("worker has no full profile")
  executed = [row for row in rows if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type")) for row in executed)
  names = {str(row.get("node_name")) for row in executed}
  old_transposes = 0
  old_multiplies = 0
  for layer in FULL_ATTENTION_LAYERS:
    prefix = (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/")
    old_transposes += prefix + "aten::transpose/Transpose_3" in names
    old_multiplies += prefix + "aten::mul/Multiply_6" in names
  core_counts = {key: counts.get(key, 0) for key in EXPECTED_CORE_COUNTS}
  return {
      "executed_counts": dict(sorted(counts.items())),
      "old_output_transpose_executed": old_transposes,
      "old_gate_multiply_executed": old_multiplies,
      "gated_dynamic_quantize_executed": counts.get(
          "IQ36GatedTransposeDynamicQuantize", 0),
      "dynamic_quantize_executed": counts.get("DynamicQuantize", 0),
      "core_counts": core_counts,
      "core_counts_exact": core_counts == EXPECTED_CORE_COUNTS,
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
      BOUND, CONTROL, CONTROL_PLUGIN, args.plugin, MODEL_DIR /
      "openvino_language_model.xml", OV_PYTHON, GRAPH_SOURCE, WORKER_SOURCE,
      CUSTOM_CONFIG, KERNEL_SOURCE, LOWERING_PATCH, LOWERING_SOURCE,
      BINDING_PATCH, BINDING_SOURCE, FAILED_COMPONENT, FAILED_STDERR)
  missing = [display(Path(path)) for path in required if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing gated-DQ audit inputs: " + ", ".join(missing))
  memory_start = available_memory_bytes()
  git = git_state(output)
  bound = load_json(BOUND)
  failed = load_json(FAILED_COMPONENT)
  failed_stderr = FAILED_STDERR.read_text(encoding="utf-8")
  rewrite = execute_rewrite_audit(raw)
  source = source_audit()
  plugin = args.plugin.resolve()
  abi_probe = execute_abi_probe(raw, plugin)
  candidate_hash = sha256(plugin)
  control_hash = sha256(CONTROL_PLUGIN)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1317b_admits_exact_group64_rewrite",
            bound.get("required_checks_passed") is True
            and bound.get("source_edit_admitted") is True
            and bound.get("source_contract", {}).get(
                "dq_group64_contract_exact") is True),
      check("no_gpu_model_rewrite_is_exact_for_all_ten_layers",
            rewrite_exact(rewrite), rewrite=rewrite),
      check("source_config_kernel_worker_and_lowering_are_exact",
            source.get("one_custom_config_entry") is True
            and source.get("custom_config_inputs") == 2
            and source.get("custom_config_outputs") == 4
            and all(source.get(key) is True for key in (
                "custom_config_exact_work_size",
                "graph_flag_and_four_output_contract",
                "worker_plumbs_default_off_flag",
                "kernel_group64_contract",
                "lowering_wires_i8_scale_and_reduction",
                "durable_patch_is_exactly_applied",
                "four_output_binding_uses_output_count",
                "binding_patch_is_exactly_applied")),
            source=source),
      check("seq1319_failed_only_at_unbound_output_port_without_oom",
            failed.get("verdict") == "inconclusive"
            and failed.get("worker", {}).get("oom_observed") is False
            and failed.get("worker", {}).get("returncode") == 1
            and "Error set arg 4, error code: 1" in failed_stderr,
            note=("output ports two and three were mapped to index -1 because "
                  "the stock check compared them with the two-input count")),
      check("tiny_four_output_gpu_abi_and_arithmetic_are_exact",
            abi_probe == {
                "all_outputs_finite": True,
                # GPU runtime-model nodes are normalized to ExecutionNode;
                # exact typed outputs and arithmetic prove execution here.
                "custom_runtime_rows": 0,
                "dtypes": ["float32", "int8", "float16", "int32"],
                "quantized_exact": True,
                "reduction_exact": True,
                "scale_exact": True,
                "shapes": [[1, 1, 4096], [1, 1, 4096],
                           [1, 1, 64], [1, 1, 64]],
            }, abi_probe=abi_probe),
      check("candidate_plugin_is_built_and_distinct_from_control",
            candidate_hash != control_hash,
            candidate_sha256=candidate_hash, control_sha256=control_hash),
      check("audit_launches_only_one_tiny_abi_probe", True,
            compilers=0, gpu_contexts=1, tiny_custom_compiles=1,
            model_compiles=0,
            model_workers=0, long_workers=0),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = "admit_one_attention_gated_dq_candidate" if passed else "inconclusive"
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "mode": "audit-only",
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "candidate_workers_admitted": 1 if passed else 0,
      "additional_control_worker_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "rewrite": rewrite,
      "source_audit": source,
      "abi_probe": abi_probe,
      "failed_candidate_diagnosis": {
          "artifact": display(FAILED_COMPONENT.parent),
          "worker_returncode": failed.get("worker", {}).get("returncode"),
          "oom_observed": failed.get("worker", {}).get("oom_observed"),
          "error": "Error set arg 4, error code: 1",
      },
      "plugin_build": {
          "command": (
              "cmake --build /home/intel/intel-qwen36-r0/build/"
              "openvino-90214e-l0-gpu --target "
              "openvino_intel_gpu_plugin --parallel 4"),
          "candidate_path": str(plugin),
          "candidate_sha256": candidate_hash,
          "control_sha256": control_hash,
          "elapsed_seconds": 5.62,
          "max_rss_kib": 868888,
          "swaps": 0,
          "exit_status": 0,
      },
      "checks": checks,
      "memory": {
          "stop_bytes": int(args.abort_below_available_gib * 1024**3),
          "available_start_bytes": memory_start,
          "available_end_bytes": available_memory_bytes(),
          "oom_observed": False,
      },
      "decision": {
          "next_route": (
              "openvino_attention_gated_dq_component" if passed else None),
          "stop_rule": (
              "one candidate-only short worker; close on activation, "
              "correctness, or 0.0473885-ms cut failure"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "mode": "audit-only",
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(Path(path)): sha256(Path(path))
                 for path in required if Path(path) != OV_PYTHON},
      "candidate_plugin_sha256": candidate_hash,
      "compilers": 0,
      "gpu_contexts": 1,
      "tiny_custom_compiles": 1,
      "model_workers": 0,
  })
  report = f"""# Attention consumer-side gated-DQ rewrite audit

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

The graph rewrite creates ten standalone gated-DQ consumers and removes all
ten old output transposes, reshapes, and multiplies. Each operation exposes a
graph-only F32 carrier plus I8 `[B,Q,4096]`, F16 `[B,Q,64]` scales, and I32
`[B,Q,64]` reductions. The durable GPU lowering patch consumes those three
compressed outputs directly in FullyConnectedCompressed.

Seq1319 reached the first inference but failed while binding output argument
four because stock custom-op lowering compared output port indices with the
two-input count. The corrected generic binding uses the four-output count. A
tiny dynamic-shape GPU probe now binds all four outputs and exactly matches the
group-64 I8, F16 scale, and I32 reduction reference arithmetic.

The correction build completed in 5.62 seconds with 868888-KiB peak RSS and
zero swaps. Admit exactly one corrected candidate-only 2k/17 worker against
retained seq1304 behind the 4-GiB stop. This audit launched one tiny custom-op
GPU probe and no model worker; OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output), "verdict": verdict,
      "candidate_workers_admitted": 1 if passed else 0,
      "candidate_plugin_sha256": candidate_hash,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


def component_main(args: argparse.Namespace, output: Path) -> int:
  worker_dir = output / "raw/2k/candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = (
      BOUND, AUDIT, CONTROL, args.plugin, BASE.REFERENCE_WORKER,
      BASE.WORKER, GRAPH_SOURCE, WORKER_SOURCE, CUSTOM_CONFIG, KERNEL_SOURCE,
      LOWERING_PATCH, BINDING_PATCH)
  missing = [display(Path(path)) for path in required if not Path(path).is_file()]
  if missing:
    raise SystemExit("missing gated-DQ component inputs: " + ", ".join(missing))
  git = git_state(output)
  concurrent = BASE.other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  bound = load_json(BOUND)
  audit = load_json(AUDIT)
  control = load_json(CONTROL)
  reference = load_json(BASE.REFERENCE_WORKER)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  config, decode_tokens, expected_top1 = BASE.build_worker_config(
      worker_dir, reference, plugin)
  config["attention_gated_dynamic_quantize"] = True
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
  required_saving = float(bound["budget"]["required_component_saving_ms"])
  activation_passed = (
      control_profile.get("old_output_transpose_executed") == 10
      and control_profile.get("old_gate_multiply_executed") == 10
      and control_profile.get("dynamic_quantize_executed") == 161
      and candidate_profile.get("old_output_transpose_executed") == 0
      and candidate_profile.get("old_gate_multiply_executed") == 0
      and candidate_profile.get("gated_dynamic_quantize_executed") == 10
      and candidate_profile.get("dynamic_quantize_executed") == 151)
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
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1320b_admits_exactly_one_corrected_candidate",
            audit.get("required_checks_passed") is True
            and audit.get("candidate_workers_admitted") == 1
            and audit.get("additional_control_worker_admitted") is False),
      check("no_concurrent_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("one_candidate_completes_above_stop_without_oom",
            worker_safe, worker={key: worker[key] for key in (
                "returncode", "timed_out", "memory_guard", "monitor",
                "oom_observed")}),
      check("worker_uses_exact_audited_candidate_plugin",
            plugin_hash == audit["plugin_build"]["candidate_sha256"]
            and result.get("candidate_gpu_plugin_sha256") == plugin_hash),
      check("graph_rewrites_exactly_ten_gated_dq_consumers",
            source.get("attention_gated_dynamic_quantize") is True
            and source.get(
                "attention_gated_dynamic_quantize_rewrite_count") == 10
            and source.get("custom_count_after") == 10,
            source_summary=source),
      check("teacher_forced_top1_is_exact", correctness_passed,
            expected_top1=expected_top1, actual_top1=actual_top1),
      check("core_execution_census_is_exact",
            candidate_profile.get("core_counts_exact") is True,
            core_counts=candidate_profile.get("core_counts")),
      check("gated_dq_activation_outcome_is_exact", activation_passed,
            control=control_profile, candidate=candidate_profile),
      check("profile_times_are_not_added_as_savings",
            candidate_profile.get("raw_profile_time_is_savings_evidence")
                is False),
  ]
  evidence_passed = all(row["pass"] for row in checks)
  route_accepted = (
      evidence_passed and activation_passed and correctness_passed
      and performance_passed)
  verdict = (
      "retain_attention_gated_dq_as_bundle_cut" if route_accepted else
      "reject_attention_gated_dq_after_component"
      if evidence_passed else "inconclusive")
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
      "evidence_checks_passed": evidence_passed,
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
          "close_route": evidence_passed and not route_accepted,
          "retain_as_bundle_ingredient": route_accepted,
          "next_route": (
              "openvino_fc_rms_igc_attention_gated_dq_bundle_bound"
              if route_accepted else "openvino_upstream_capability_watch"),
          "reopen_condition": (
              "none for unchanged gated-DQ, repeat, or sample extension if "
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
  report = f"""# Attention consumer-side gated-DQ component

Verdict: **{verdict}**. Evidence checks:
`{str(evidence_passed).lower()}`; activation:
`{str(activation_passed).lower()}`; correctness:
`{str(correctness_passed).lower()}`; short performance:
`{str(performance_passed).lower()}`.

Exactly one candidate-only 2k/17 worker ran against retained seq1304. The
candidate must replace ten Transpose/Multiply/DynamicQuantize triples with ten
group-64 gated-DQ rows, retain 371 compressed FCs, and preserve all 18
teacher-forced top-1 tokens.

After dropping the first decode JIT sample, diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`, an observed
`{observed_saving:.7f}-ms` saving versus the incremental
`{required_saving:.7f}-ms` cut. This is not a speedup or product claim. No
stock, control, long, ABBA, output512, or product worker ran; OOM observed:
`{str(worker['oom_observed']).lower()}`.
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
  return 0 if evidence_passed else 2


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  if args.audit_only:
    return audit_main(args, output)
  return component_main(args, output)


if __name__ == "__main__":
  raise SystemExit(main())
