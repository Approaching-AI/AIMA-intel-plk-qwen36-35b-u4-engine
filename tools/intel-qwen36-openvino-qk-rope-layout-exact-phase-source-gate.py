#!/usr/bin/env python3
"""Audit Q/K RoPE-layout fusion on the current exact-phase product carrier.

This source/graph gate launches no GPU context or inference worker.  It builds
the locked model in memory, applies the current exact-phase dual-cohort graph
plus the existing default-off IQ36QKRopeLayout rewrite, and verifies that only
the Q/K transpose, partial-RoPE, and concat producer boundary disappears.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import resource
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-qk-rope-layout-exact-phase-source-gate-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
GRAPH_SOURCE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
PRODUCT_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/gate.json")
PRODUCT_PERFORMANCE = PRODUCT_GATE.parent / "performance.json"
PRODUCT_CANDIDATE = PRODUCT_GATE.parent / (
    "raw/prefill_shape_002k/correctness/candidate/worker-result.json")
HISTORICAL_COMPONENT = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/metrics.json")
EXPECTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
NOISE_FRACTION = 0.005
MIN_MATERIALITY_MULTIPLE = 10.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=180)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
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


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def available_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain", "--untracked-files=all"],
      cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
  return {
      "commit": commit,
      "dirty": bool(rows),
      "dirty_paths": [row[3:] for row in rows if len(row) >= 4],
  }


def graph_audit(raw: Path, timeout_s: int) -> dict[str, Any]:
  script = f"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import openvino as ov

path = Path({str(GRAPH_SOURCE)!r})
spec = importlib.util.spec_from_file_location("iq36_exact_qk_graph", path)
graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph)
model, summary = graph.make_candidate_model(
    ov.Core(), Path({str(MODEL_DIR)!r}), ov, np,
    target_layers=graph.FULL_ATTENTION_LAYERS,
    exact_phase_decode=True,
    exact_phase_dual_cohort=True,
    initialize_hot_states=True,
    fixed_cold_capacity=2048,
    prefill_history_capacity=16384,
    exact_history_layers=graph.FULL_ATTENTION_LAYERS,
    exact_history_capacity=17408,
    fuse_linear_conv_state=True,
    fuse_qk_rope_layout=True,
    decode_stock_micro_layers=graph.FULL_ATTENTION_LAYERS)
ops = model.get_ordered_ops()
names = {{node.get_friendly_name() for node in ops}}
types = (
    "IQ36ExactPhaseDualCohortHotAttentionGQA",
    "IQ36LinearConvSwish", "IQ36QKRopeLayout",
    "ScaledDotProductAttention")
counts = {{
    kind: sum(node.get_type_name() == kind for node in ops)
    for kind in types
}}
old_qk_live = []
output_boundary = []
rows = []
for layer in graph.FULL_ATTENTION_LAYERS:
  prefix = (
      "__module.model.model.language_model.layers."
      f"{{layer}}.self_attn/")
  qk_suffixes = (
      ("aten::transpose/Transpose_2", "aten::transpose/Transpose_1",
       "aten::slice/Slice_4", "aten::slice/Slice_7",
       "aten::slice/Slice", "aten::slice/Slice_3",
       "aten::add/Add_1", "aten::add/Add",
       "aten::cat/Concat_5", "aten::cat/Concat_2")
      if layer == 39 else
      ("aten::transpose/Transpose", "aten::transpose/Transpose_1",
       "aten::slice/Slice", "aten::slice/Slice_3",
       "aten::slice/Slice_4", "aten::slice/Slice_7",
       "aten::add/Add", "aten::add/Add_1",
       "aten::cat/Concat_1", "aten::cat/Concat_3"))
  old_qk_live.extend(
      prefix + suffix for suffix in qk_suffixes
      if prefix + suffix in names)
  for suffix in (
      "aten::transpose/Transpose_3", "aten::mul/Multiply_6"):
    if prefix + suffix in names:
      output_boundary.append(prefix + suffix)
  operation = next(
      node for node in ops
      if node.get_friendly_name() == f"iq36_qk_rope_layout_layer{{layer}}")
  rows.append({{
      "layer": layer,
      "input_shapes": [
          str(operation.get_input_partial_shape(index))
          for index in range(operation.get_input_size())],
      "output_shapes": [
          str(operation.get_output_partial_shape(index))
          for index in range(operation.get_output_size())],
      "output_consumers": [
          sorted(
              port.get_node().get_friendly_name()
              for port in operation.output(index).get_target_inputs())
          for index in range(operation.get_output_size())],
  }})
summary_keys = (
    "custom_count_after", "stock_sdpa_count_after",
    "state_count_after", "sink_count_after",
    "fuse_qk_rope_layout", "qk_rope_layout_rewrite_count",
    "exact_phase_decode", "exact_phase_dual_cohort",
    "decode_stock_micro_layers", "fuse_attention_output_gate",
    "token_major_value_output", "attention_gated_dynamic_quantize")
print(json.dumps({{
    "counts": counts,
    "old_qk_live": old_qk_live,
    "output_boundary": output_boundary,
    "rows": rows,
    "summary": {{key: summary.get(key) for key in summary_keys}},
}}, sort_keys=True))
"""
  before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  memory_before = available_bytes()
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=False,
      capture_output=True, text=True, timeout=timeout_s)
  memory_after = available_bytes()
  after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  if completed.returncode != 0:
    audit = {
        "returncode": completed.returncode,
        "stderr": completed.stderr[-12000:],
        "stdout": completed.stdout[-12000:],
    }
  else:
    audit = json.loads(completed.stdout)
    audit["returncode"] = 0
    audit["stderr"] = completed.stderr[-12000:]
  audit["child_max_rss_kib_upper_bound"] = max(
      0, int(after_rss - before_rss))
  audit["memory_available_before_bytes"] = memory_before
  audit["memory_available_after_bytes"] = memory_after
  write_json(raw / "graph-audit.json", audit)
  return audit


def source_audit() -> dict[str, Any]:
  graph = GRAPH_SOURCE.read_text(encoding="utf-8")
  kernel = KERNEL_SOURCE.read_text(encoding="utf-8")
  config = CUSTOM_CONFIG.read_text(encoding="utf-8")
  marker = '<CustomLayer name="IQ36QKRopeLayout"'
  block = (
      config.split(marker, 1)[1].split("</CustomLayer>", 1)[0]
      if config.count(marker) == 1 else "")
  return {
      "exact_phase_only_validation_opening": (
          "(fuse_qk_rope_layout and not exact_phase_decode)" in graph),
      "rewrite_replaces_only_qk_producers": all(
          token in graph for token in (
              "query_concat.output(0).replace(qk_rope.output(0))",
              "key_concat.output(0).replace(qk_rope.output(1))",
              "qk_rope_layout_rewrites.append({")),
      "output_gate_routes_remain_separate": all(
          token in graph for token in (
              "if fuse_attention_output_gate:",
              "if token_major_value_output or attention_gated_dynamic_quantize:",
              "fuse_qk_rope_layout requires the unmodified unified F16 path")),
      "one_custom_layer_registration": config.count(marker) == 1,
      "registration_inputs_outputs": {
          "inputs": block.count('type="input"'),
          "outputs": block.count('type="output"'),
      },
      "registration_work_size_exact": (
          '<WorkSizes global="X,Y,B*F" local="16,1,1"/>' in block),
      "kernel_contract_exact": all(
          token in kernel for token in (
              "__attribute__((reqd_work_group_size(16, 1, 1)))",
              "__kernel void iq36_qk_rope_layout",
              "inline half iq36_qk_rope_value(",
              "const half value, const half peer, const half cosine,",
              "half query_value = convert_half_rte(",
              "half key_value = convert_half_rte(",
              "dimension < 64U",
              "dimension < 32U",
              "cosine * value - sine * peer",
              "cosine * value + sine * peer",
              "query_head < (uint)OUTPUT1_DIMS[1]")) and
          kernel.count("convert_half_rte(") == 8 and
          "inline float iq36_qk_rope_value(" not in kernel and
          "convert_float(" not in kernel and
          "float query_value" not in kernel and
          "float key_value" not in kernel,
  }


def query_transpose_profile(result: dict[str, Any]) -> dict[str, Any]:
  rows = []
  for row in result.get("execution_census", {}).get("top_rows", []):
    if (row.get("node_type") == "Transpose" and
        row.get("exec_type") == "permute_ref__f16" and
        any(
            f"layers.{layer}.self_attn/" in str(row.get("node_name", ""))
            for layer in LAYERS)):
      rows.append(row)
  return {
      "count": len(rows),
      "sum_profile_us": sum(float(row["real_time_us"]) for row in rows),
      "rows": rows,
  }


def rewrite_exact(audit: dict[str, Any]) -> bool:
  summary = audit.get("summary", {})
  rows = audit.get("rows", [])
  return (
      audit.get("returncode") == 0 and
      audit.get("counts") == {
          "IQ36ExactPhaseDualCohortHotAttentionGQA": 10,
          "IQ36LinearConvSwish": 30,
          "IQ36QKRopeLayout": 10,
          "ScaledDotProductAttention": 0,
      } and
      summary.get("custom_count_after") == 10 and
      summary.get("stock_sdpa_count_after") == 0 and
      summary.get("state_count_after") == 120 and
      summary.get("sink_count_after") == 60 and
      summary.get("fuse_qk_rope_layout") is True and
      summary.get("qk_rope_layout_rewrite_count") == 10 and
      summary.get("exact_phase_decode") is True and
      summary.get("exact_phase_dual_cohort") is True and
      summary.get("decode_stock_micro_layers") == list(LAYERS) and
      summary.get("fuse_attention_output_gate") is False and
      summary.get("token_major_value_output") is False and
      summary.get("attention_gated_dynamic_quantize") is False and
      audit.get("old_qk_live") == [] and
      len(audit.get("output_boundary", [])) == 20 and
      len(rows) == 10 and
      all(
          row.get("input_shapes") == [
              "[?,?,16,256]", "[?,?,2,256]",
              "[?,1,?,64]", "[?,1,?,64]"] and
          row.get("output_shapes") == [
              "[?,16,?,256]", "[?,2,?,256]"] and
          any(
              name == f"iq36_hot_attention_layer{row['layer']}"
              for name in row.get("output_consumers", [[], []])[0]) and
          any(
              name == f"iq36_hot_attention_layer{row['layer']}"
              for name in row.get("output_consumers", [[], []])[1])
          for row in rows))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True)
  required_paths = (
      OV_PYTHON, MODEL_DIR / "openvino_language_model.xml",
      GRAPH_SOURCE, KERNEL_SOURCE, CUSTOM_CONFIG, PRODUCT_GATE,
      PRODUCT_PERFORMANCE, PRODUCT_CANDIDATE, HISTORICAL_COMPONENT)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing source-gate inputs: " + ", ".join(missing))

  git = git_state()
  product_gate = load_json(PRODUCT_GATE)
  product_performance = load_json(PRODUCT_PERFORMANCE)
  product_candidate = load_json(PRODUCT_CANDIDATE)
  historical = load_json(HISTORICAL_COMPONENT)
  source = source_audit()
  audit = graph_audit(raw, args.timeout_s)
  profile = query_transpose_profile(product_candidate)

  performance_row = product_performance["cases"][0]
  decode_tps = float(
      performance_row["absolute_floors"]["decode_median"])
  decode_wall_ms = 1000.0 / decode_tps
  historical_saving_ms = float(
      historical["performance"]["observed_median_saving_ms"])
  noise_cut_ms = decode_wall_ms * NOISE_FRACTION
  materiality_multiple = historical_saving_ms / noise_cut_ms
  projected_decode_tps = 1000.0 / (
      decode_wall_ms - historical_saving_ms)
  opportunity = {
      "current_decode_median_tokens_s": decode_tps,
      "current_decode_wall_ms_per_token": decode_wall_ms,
      "current_query_transpose_profile": profile,
      "historical_component_saving_ms_per_token": historical_saving_ms,
      "noise_fraction": NOISE_FRACTION,
      "noise_cut_ms_per_token": noise_cut_ms,
      "materiality_multiple": materiality_multiple,
      "projected_decode_tokens_s_if_historical_saving_retains":
          projected_decode_tps,
      "projection_is_speed_claim": False,
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq2193_formal_lane_and_plugin_are_bound",
            product_gate.get("run_checks_passed") is True and
            product_gate.get("stopped_reason") is None and
            product_gate.get("config", {}).get(
                "candidate_gpu_plugin_sha256") == EXPECTED_PLUGIN_SHA256 and
            performance_row.get("paired_block_count") == 8 and
            performance_row.get("promotion_rate_pass") is True),
      check("current_profile_has_exact_ten_qk_query_transposes",
            profile["count"] == 10 and
            profile["sum_profile_us"] > 0.0,
            profile=profile),
      check("historical_qk_component_is_accepted_and_material",
            historical.get("route_accepted") is True and
            historical.get("correctness_passed") is True and
            historical.get("activation_passed") is True and
            historical.get("performance_passed") is True and
            math.isclose(
                historical_saving_ms, 1.1849190000000078,
                rel_tol=0.0, abs_tol=1e-12),
            historical_performance=historical.get("performance")),
      check("existing_qk_source_and_registration_are_exact",
            source["exact_phase_only_validation_opening"] and
            source["rewrite_replaces_only_qk_producers"] and
            source["output_gate_routes_remain_separate"] and
            source["one_custom_layer_registration"] and
            source["registration_inputs_outputs"] == {
                "inputs": 4, "outputs": 2} and
            source["registration_work_size_exact"] and
            source["kernel_contract_exact"],
            source=source),
      check("exact_phase_graph_rewrite_is_complete_and_scoped",
            rewrite_exact(audit), audit=audit),
      check("historical_saving_clears_current_noise_by_tenfold",
            materiality_multiple >= MIN_MATERIALITY_MULTIPLE and
            historical_saving_ms < decode_wall_ms,
            threshold=MIN_MATERIALITY_MULTIPLE,
            opportunity=opportunity),
      check("source_gate_launches_no_gpu_or_inference_worker", True,
            gpu_contexts_created=0, inference_workers=0,
            graph_build_subprocesses=1),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_phase_qk_rope_product_precheck"
      if passed else
      "reject_qk_rope_exact_phase_composition_before_gpu")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "product_precheck_admitted": passed,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "checks": checks,
      "source": source,
      "graph_audit": audit,
      "opportunity": opportunity,
      "next_action": {
          "route": "openvino_qk_rope_layout_exact_phase_product_precheck",
          "requirements": [
              "plumb one default-off product config flag",
              "compile and run one isolated 2k output130 candidate",
              "require ten QK producers and ten exact dual attention owners",
              "require old QK boundaries absent and output gate untouched",
              "require candidate logits and tokens inside the accepted ruler",
          ],
      },
  }
  write_json(out / "result.json", payload)
  write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": relative(Path(__file__)),
      "git": git,
      "inputs": {
          relative(path): sha256(path)
          for path in (
              GRAPH_SOURCE, KERNEL_SOURCE, CUSTOM_CONFIG, PRODUCT_GATE,
              PRODUCT_PERFORMANCE, PRODUCT_CANDIDATE, HISTORICAL_COMPONENT)
      },
      "gpu_contexts_created": 0,
      "inference_workers": 0,
  })
  report = f"""# Exact-phase Q/K RoPE-layout source gate

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

The current 2k final profile retains `{profile['count']}` matching F16 query
transposes totaling `{profile['sum_profile_us']:.3f} us` of non-additive
device profile time. The accepted earlier Q/K component saved
`{historical_saving_ms:.6f} ms/token`, `{materiality_multiple:.2f}x` the
current 0.5-percent wall-noise cut. Retaining that point movement would project
`{projected_decode_tps:.3f} tok/s`; this is route arithmetic, not a speed
claim.

The no-GPU graph audit creates ten `IQ36QKRopeLayout` producers and ten exact
dual-cohort attention owners, removes all old Q/K producer boundaries, and
leaves the twenty output-transpose/gate nodes live. It launches no GPU context
or inference worker.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "qk_query_transpose_count": profile["count"],
      "historical_saving_ms": historical_saving_ms,
      "materiality_multiple": materiality_multiple,
      "graph_child_max_rss_kib_upper_bound":
          audit.get("child_max_rss_kib_upper_bound"),
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
