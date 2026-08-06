#!/usr/bin/env python3
"""Audit the single-owner four-kernel adaptive OpenVINO source cut."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-adaptive-attention-source-abi-gate-v0"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
HIGH_TOPK = LAYERS
MAX_CHUNKS = 129
PACKED_F32 = 2_228_336
PACKED_BYTES = 8_913_344

GRAPH = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
PRODUCT = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
XML = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
HELPERS = ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
PREFILL = ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl"
OWNER = ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl"
ADAPTIVE = ROOT / "engine/openvino/custom/iq36_adaptive_attention_decode.cl"
PLUGIN_PATCH = ROOT / (
    "engine/openvino/iq36-custom-adaptive-attention-multikernel.patch")
OV_ROOT = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_IMPL = OV_ROOT / (
    "src/plugins/intel_gpu/src/graph/impls/ocl/custom_primitive.cpp")
OV_HEAD_PREFIX = "90214e5be05"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument(
      "--allow-dirty", action="store_true",
      help="development-only: report expected source edits without admitting compile")
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


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


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    out_rel = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    out_rel = ""
  rows = [row for row in rows if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def balanced_preprocessor(text: str) -> tuple[bool, int]:
  depth = 0
  for line in text.splitlines():
    directive = line.lstrip()
    if directive.startswith(("#if ", "#if\t", "#ifdef", "#ifndef")):
      depth += 1
    elif directive.startswith("#endif"):
      depth -= 1
      if depth < 0:
        return False, depth
  return depth == 0, depth


def load_graph() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_adaptive_graph", GRAPH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {GRAPH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def xml_contract(text: str) -> dict[str, Any]:
  root = ET.fromstring("<root>" + text + "</root>")
  rows = {}
  classes = {
      "IQ36AdaptiveTop2048HotAttentionGQA": (2048, ()),
      "IQ36AdaptiveTop1024HotAttentionGQA": (1024, ()),
      "IQ36AdaptiveV16Top512HotAttentionGQA": (
          512, ("-DIQ36_DIRECT_I8_VALUE16=1",)),
      "IQ36AdaptiveTop512HotAttentionGQA": (512, ()),
      "IQ36AdaptiveVResidual1Top512HotAttentionGQA": (
          512, ("-DIQ36_VALUE_RESIDUAL1=1",)),
      "IQ36AdaptiveKResidual1Top512HotAttentionGQA": (
          512, ("-DIQ36_KEY_RESIDUAL1=1",)),
      "IQ36AdaptiveKVResidual1Top512HotAttentionGQA": (
          512, (
              "-DIQ36_KEY_RESIDUAL1=1", "-DIQ36_VALUE_RESIDUAL1=1")),
      "IQ36AdaptiveVResidual1Top256HotAttentionGQA": (
          256, ("-DIQ36_VALUE_RESIDUAL1=1",)),
      "IQ36AdaptiveKResidual1Top256HotAttentionGQA": (
          256, ("-DIQ36_KEY_RESIDUAL1=1",)),
      "IQ36AdaptiveKVResidual1Top256HotAttentionGQA": (
          256, (
              "-DIQ36_KEY_RESIDUAL1=1", "-DIQ36_VALUE_RESIDUAL1=1")),
      "IQ36AdaptiveKeyExactTop256HotAttentionGQA": (
          256, ("-DIQ36_ADAPTIVE_KEY_EXACT=1",)),
      "IQ36AdaptiveK6V7Top256HotAttentionGQA": (
          256, ("-DIQ36_ADAPTIVE_PACKED_K6V7=1",)),
      "IQ36AdaptiveK7V7Top256HotAttentionGQA": (
          256, ("-DIQ36_ADAPTIVE_PACKED_K7V7=1",)),
      "IQ36AdaptiveK7V8Top256HotAttentionGQA": (
          256, ("-DIQ36_ADAPTIVE_PACKED_K7V8=1",)),
      "IQ36AdaptiveK7V8Top512HotAttentionGQA": (
          512, ("-DIQ36_ADAPTIVE_PACKED_K7V8=1",)),
      "IQ36AdaptiveK8V7Top256HotAttentionGQA": (
          256, ("-DIQ36_ADAPTIVE_PACKED_K8V7=1",)),
      "IQ36AdaptiveTop256HotAttentionGQA": (256, ()),
      "IQ36AdaptiveTop252HotAttentionGQA": (252, ()),
      "IQ36AdaptiveTop128HotAttentionGQA": (128, ()),
  }
  for name, (topk, extra_options) in classes.items():
    matches = [node for node in root if node.attrib.get("name") == name]
    if len(matches) != 1:
      rows[name] = {"count": len(matches), "pass": False}
      continue
    node = matches[0]
    kernels = node.findall("Kernel")
    buffers = node.find("Buffers")
    tensors = [] if buffers is None else buffers.findall("Tensor")
    options = node.find("CompilerOptions")
    option_text = "" if options is None else options.attrib.get("options", "")
    sources = [] if not kernels else [
        source.attrib.get("filename") for source in kernels[0].findall("Source")]
    passed = bool(
        len(kernels) == 1
        and kernels[0].attrib.get("entry") == "iq36_adaptive_attention"
        and len(tensors) == 19
        and [int(row.attrib["arg-index"]) for row in tensors] == list(range(19))
        and sources.count("iq36_adaptive_attention_decode.cl") == 1
        and "-DIQ36_PREFILL_FULL_HISTORY=1" in option_text
        and "-DIQ36_PREFILL_USE_MICROKERNEL=1" in option_text
        and "-DIQ36_DIRECT_I8_FIXED_LAYOUT=1" in option_text
        and "-DIQ36_HOT_WINDOW=16384U" in option_text
        and "-DIQ36_DIMENSION_MAJOR_VALUE_PLANE=1" in option_text
        and f"-DIQ36_ADAPTIVE_TOPK={topk}U" in option_text
        and all(option in option_text for option in extra_options))
    rows[name] = {
        "pass": passed, "kernel_count": len(kernels),
        "tensor_count": len(tensors), "sources": sources,
        "options": option_text}
  return rows


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  if out_dir.exists():
    raise SystemExit(f"output already exists: {out_dir}")
  required = (
      GRAPH, PRODUCT, XML, HELPERS, PREFILL, OWNER, ADAPTIVE,
      PLUGIN_PATCH, OV_IMPL)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-gate inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_before = available_memory_bytes()
  if memory_before < stop_bytes:
    raise SystemExit(f"memory stop: {memory_before} < {stop_bytes} bytes")
  out_dir.mkdir(parents=True, exist_ok=False)

  repository = git_state(out_dir)
  texts = {path: path.read_text(encoding="utf-8") for path in required}
  graph = texts[GRAPH]
  product = texts[PRODUCT]
  xml = texts[XML]
  helpers = texts[HELPERS]
  prefill = texts[PREFILL]
  owner = texts[OWNER]
  adaptive = texts[ADAPTIVE]
  plugin_patch = texts[PLUGIN_PATCH]
  graph_module = load_graph()
  workspace_f32 = int(graph_module.adaptive_workspace_f32_elements(MAX_CHUNKS))
  workspace_bytes = workspace_f32 * 4

  xml_rows = xml_contract(xml)
  entrypoints = re.findall(
      r"__kernel\s+void\s+(iq36_adaptive_attention_[a-z_]+)", adaptive)
  unique_entrypoints = list(dict.fromkeys(entrypoints))
  expected_entries = [
      "iq36_adaptive_attention_partial",
      "iq36_adaptive_attention_select_reduce_union",
      "iq36_adaptive_attention_correct_normalize",
      "iq36_adaptive_attention_ordered_update",
  ]
  abi_match = re.search(
      r"#define IQ36_ADAPTIVE_ABI \\\n(.*?)\n\ninline", adaptive, re.S)
  abi_text = "" if abi_match is None else abi_match.group(1)
  abi_ports = re.findall(r"\b(?:INPUT|OUTPUT)(\d+)_TYPE\b", abi_text)
  first_three = adaptive.split(
      "#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_UPDATE", 1)[0]
  partial = adaptive.split(
      "#if IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_PARTIAL", 1)[-1]
  partial = partial.split(
      "#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_SELECT", 1)[0]
  correction = adaptive.split(
      "#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_CORRECT", 1)[-1]
  correction = correction.split(
      "#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_UPDATE", 1)[0]
  update = adaptive.split(
      "#elif IQ36_ADAPTIVE_STAGE == IQ36_ADAPTIVE_STAGE_UPDATE", 1)[-1]
  state_writer_tokens = (
      "iq36_direct_store_cold_key(",
      "iq36_direct_store_cold_value(",
      "iq36_direct_store_cold_key_scale(",
      "iq36_direct_store_cold_value_scale(",
      "iq36_direct_store_hot_value_dimension(",
      "hot_key_bits[hot_key_index] =",
      "hot_value[hot_value_base +",
  )
  preprocessor_rows = {
      str(path.relative_to(ROOT)): balanced_preprocessor(texts[path])
      for path in (HELPERS, PREFILL, OWNER, ADAPTIVE)
  }
  patch_check = subprocess.run(
      ["git", "apply", "--check", str(PLUGIN_PATCH)], cwd=OV_ROOT,
      capture_output=True, text=True)
  ov_head = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=OV_ROOT, check=True,
      capture_output=True, text=True).stdout.strip()

  checks = [
      check("repository_clean_at_gate", not repository["dirty"],
            git=repository, allow_dirty=args.allow_dirty),
      check("pinned_openvino_source_is_exact",
            ov_head.startswith(OV_HEAD_PREFIX), observed=ov_head,
            expected_prefix=OV_HEAD_PREFIX),
      check("graph_route_is_all_layer_block32_exact_history",
            all(token in graph for token in (
                "adaptive_attention_layers",
                "adaptive attention must own every selected",
                "adaptive attention admits exactly block32 I8 K/V",
                "set(exact_history_layers) != set(adaptive_attention_layers)",
                "ADAPTIVE_HOT_WINDOW = 16384",
                "ADAPTIVE_HIGH_TOPK_LAYERS = FULL_ATTENTION_LAYERS",
                "adaptive_attention_topk",
                "IQ36AdaptiveTop2048HotAttentionGQA",
                "IQ36AdaptiveTop1024HotAttentionGQA",
                "IQ36AdaptiveTop512HotAttentionGQA",
                "IQ36AdaptiveTop256HotAttentionGQA",
                "IQ36AdaptiveTop252HotAttentionGQA",
                "IQ36AdaptiveTop128HotAttentionGQA",
                "adaptive_attention_packed_kv_layers",
                "adaptive_attention_packed_kv_variant",
                "IQ36AdaptiveK6V7Top256HotAttentionGQA",
                "IQ36AdaptiveK7V7Top256HotAttentionGQA",
                "IQ36AdaptiveK7V8Top256HotAttentionGQA",
                "IQ36AdaptiveK7V8Top512HotAttentionGQA",
                "IQ36AdaptiveK8V7Top256HotAttentionGQA"))),
      check("graph_workspace_matches_partition_rebuild_layout_exactly",
            workspace_f32 == PACKED_F32
            and workspace_bytes == PACKED_BYTES,
            observed_f32=workspace_f32, observed_bytes=workspace_bytes),
      check("xml_has_one_kernel_nineteen_port_adaptive_classes",
            all(row.get("pass") for row in xml_rows.values()),
            classes=xml_rows),
      check("packed_kv_source_has_exactly_four_bounded_variants",
            all(token in helpers for token in (
                "defined(IQ36_ADAPTIVE_PACKED_K6V7)",
                "defined(IQ36_ADAPTIVE_PACKED_K7V7)",
                "defined(IQ36_ADAPTIVE_PACKED_K7V8)",
                "defined(IQ36_ADAPTIVE_PACKED_K8V7)",
                "#define IQ36_ADAPTIVE_PACKED_KV 1",
                "#define IQ36_KEY_QUANT_BITS 6U",
                "#define IQ36_KEY_QUANT_BITS 7U",
                "#define IQ36_KEY_QUANT_BITS 8U",
                "#define IQ36_VALUE_QUANT_BITS 7U",
                "#define IQ36_VALUE_QUANT_BITS 8U",
                "inline uint8 iq36_direct_cold_packed_subgroup_words6(",
                "inline uint8 iq36_direct_cold_packed_subgroup_words7(",
                "inline uint8 iq36_direct_cold_packed_subgroup_words8("))
            and all(token in adaptive for token in (
                "defined(IQ36_ADAPTIVE_PACKED_KV)",
                "IQ36_KEY_QUANT_BITS",
                "IQ36_VALUE_QUANT_BITS"))
            and "defined(IQ36_ADAPTIVE_PACKED_K6V7)" not in adaptive),
      check("packed_k7v8_uses_dimension_major_v_without_changing_append_abi",
            all(token in helpers for token in (
                "#define IQ36_ADAPTIVE_DIMENSION_MAJOR_V8 1",
                "!defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)"))
            and all(token in prefill for token in (
                "#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)",
                "iq36_direct_store_cold_value("))
            and all(token in adaptive for token in (
                "!defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)",
                "iq36_direct_cold_value_group32_fragments_unscaled(",
                "#if defined(IQ36_ADAPTIVE_DIMENSION_MAJOR_V8)",
                "iq36_direct_store_cold_value("))
            and "cold_value_append[value_append_base +" in adaptive
            and "cold_value_append[cold_value_base +" in prefill),
      check("decode_source_has_exactly_four_ordered_entrypoints",
            unique_entrypoints == expected_entries
            and entrypoints.count(
                "iq36_adaptive_attention_correct_normalize") == 2
            and all(entrypoints.count(entry) == 1
                    for entry in expected_entries
                    if entry != "iq36_adaptive_attention_correct_normalize"),
            observed=entrypoints, expected=expected_entries),
      check("all_four_kernels_share_the_full_19_port_abi",
            abi_match is not None and len(abi_ports) == 19
            and abi_ports == [str(index) for index in range(13)] +
                [str(index) for index in range(6)],
            observed_ports=abi_ports),
      check("only_ordered_update_writes_request_state",
            not any(token in first_three for token in state_writer_tokens)
            and all(token in update for token in state_writer_tokens),
            forbidden_before_update=[
                token for token in state_writer_tokens if token in first_three],
            present_in_update=[
                token for token in state_writer_tokens if token in update]),
      check("partial_handles_dynamic_boundary_and_current_row",
            all(token in adaptive for token in (
                "const uint cold_tokens = iq36_adaptive_cold_tokens",
                "const bool direct_cold_block",
                "const bool exact_history_block",
                "token < key_tokens",
                "current_key, cold_key",
                "current_value, cold_value"))),
      check("partial_scores_with_group32_key_and_keeps_radix_in_registers",
            all(token in partial for token in (
                "iq36_direct_cold_key_group32_fragments(",
                "approximate_score[",
                "uint16 lane_counts = (uint16)(0U);",
                "IQ36_ADAPTIVE_RADIX_DIGIT(sf, 15U)",
                "IQ36_ADAPTIVE_RADIX_DIGIT(s0, 0U)"))
            and all(token in helpers for token in (
                "inline int16 iq36_direct_cold_key_group32_fragments(",
                "const uint8 raw = intel_sub_group_block_read8(",
                "as_int8(convert_half16(as_char16(raw.s0123)) * scale)",
                "as_int8(convert_half16(as_char16(raw.s4567)) * scale)",
                "__builtin_IB_subgroup_block_read_cacheopts_transpose_u32_m32k4(",
                "as_char16(packed.s0246)",
                "as_char16(packed.s1357)"))
            and "uint lane_counts[16]" not in partial),
      check("correction_rebuilds_partitions_with_exact_k_and_v_sidecars",
            all(token in correction for token in (
                "IQ36_ADAPTIVE_PARTITION_TOKENS",
                "iq36_hot_key_dense_i32_base(batch, kv_head)",
                "exact_key[index] = dense_key",
                "token - begin_token] = exact_score",
                "iq36_direct_cold_value_group32_fragments_unscaled(",
                "iq36_hot_value_dimension_i32_base(batch, kv_head)",
                "values0[row] = dense_value[",
                "(ulong)dim0 * (uint)INPUT2_DIMS[2] + slot"))),
      check("prefill_and_update_publish_dimension_major_exact_v",
            "#if defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)" in prefill
            and "#if defined(IQ36_DIMENSION_MAJOR_VALUE_PLANE)" in owner
            and "iq36_direct_store_hot_value_dimension" in update
            and "#ifndef IQ36_HOT_WINDOW" in helpers),
      check("opencl_preprocessor_nesting_is_balanced",
            all(row[0] for row in preprocessor_rows.values()),
            files=preprocessor_rows),
      check("plugin_patch_applies_to_pinned_current_source",
            patch_check.returncode == 0,
            stdout=patch_check.stdout, stderr=patch_check.stderr),
      check("plugin_patch_compiles_one_or_four_sources_chains_and_groups_events",
            all(token in plugin_patch for token in (
                "std::vector<std::shared_ptr<kernel_selector::cl_kernel_data>>",
                "iq36_adaptive_attention_partial",
                "iq36_adaptive_attention_select_reduce_union",
                "iq36_adaptive_attention_correct_normalize",
                "iq36_adaptive_attention_ordered_update",
                "{128, max_chunks, batch * kv_heads}",
                "{256, query_heads, batch}",
                "{128, max_partitions, batch * kv_heads}",
                "{128, kv_heads, batch}",
                "dependencies = {last_event}",
                "stage_events.push_back(last_event)",
                "stream.aggregate_events(stage_events, stage_events.size() > 1)",
                "get_cached_kernel_ids"))
            and not any(token in plugin_patch for token in (
                ".wait(", "stream.finish(", "read_buffer("))),
      check("plugin_patch_exposes_opt_in_nonblocking_stage_trace",
            all(token in plugin_patch for token in (
                "IQ36_ADAPTIVE_STAGE_PROFILE_PATH",
                "last_event->add_event_handler(",
                "get_profiling_info()",
                "instrumentation::profiling_stage::executing",
                "std::chrono::duration_cast<",
                "instance.id()",
                "\\\"node\\\"",
                "std::ofstream output(trace->path, std::ios::app)"))
            and not any(token in plugin_patch for token in (
                ".wait(", "stream.finish(", "read_buffer("))),
      check("product_gate_exposes_and_hashes_adaptive_composition",
            all(token in product for token in (
                '"adaptive_i8_fixed"',
                'adaptive_attention_layers=(',
                'adaptive_attention_topk=adaptive_attention_topk',
                'iq36_adaptive_attention_decode.cl',
                'iq36-custom-adaptive-attention-multikernel.patch',
                'compile_only = bool(cfg.get("compile_only", False))',
                '"worker_created_infer_request": False',
                '"worker_executed_inference": False'))),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            available_before_bytes=memory_before,
            available_after_bytes=available_memory_bytes(),
            stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  compile_admitted = required_checks_passed and not args.allow_dirty
  verdict = (
      "admit_adaptive_attention_source_compile"
      if compile_admitted else
      "development_source_checks_only"
      if args.allow_dirty and all(
          row["pass"] or row["name"] == "repository_clean_at_gate"
          for row in checks) else
      "inconclusive")
  payload = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "compile_admitted": compile_admitted,
      "one_layer_worker_admitted": False,
      "product_worker_admitted": False,
      "long_worker_admitted": False,
      "git": repository,
      "openvino_source": {"path": str(OV_ROOT), "head": ov_head},
      "workspace": {
          "max_chunks": MAX_CHUNKS,
          "packed_f32_elements": workspace_f32,
          "aligned_bytes": workspace_bytes,
      },
      "topk_by_layer": {
          str(layer): 512 if layer in HIGH_TOPK else 256 for layer in LAYERS},
      "entrypoints": entrypoints,
      "source_files": [
          {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
          for path in (GRAPH, PRODUCT, XML, HELPERS, PREFILL, OWNER,
                       ADAPTIVE, PLUGIN_PATCH)],
      "checks": checks,
  }
  (out_dir / "gate.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  summary = [
      "# Adaptive attention source/ABI gate",
      "",
      f"- verdict: `{verdict}`",
      f"- required checks: `{'pass' if required_checks_passed else 'fail'}`",
      f"- compile admitted: `{str(compile_admitted).lower()}`",
      f"- packed output0: `{workspace_bytes}` bytes/layer",
      "- graph owners: `1/layer`",
      "- decode device dispatches: `4/layer`",
      "- state writer: `ordered_update only`",
      "",
  ]
  (out_dir / "summary.md").write_text(
      "\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "compile_admitted": compile_admitted,
      "out_dir": str(out_dir),
  }, sort_keys=True))
  return 0 if (required_checks_passed or args.allow_dirty and
               verdict == "development_source_checks_only") else 1


if __name__ == "__main__":
  raise SystemExit(main())
