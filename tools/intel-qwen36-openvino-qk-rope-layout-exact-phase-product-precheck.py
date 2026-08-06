#!/usr/bin/env python3
"""Run one corrected 2k/output130 Q/K RoPE exact-phase product precheck.

The sole candidate worker retains the accepted seq2193 exact-phase,
dual-cohort, parallel block-top8 carrier and enables only the already accepted
ten-layer IQ36QKRopeLayout producer with the seq2196 stock-half arithmetic
correction.  It is teacher-forced from the accepted stock row.  This gate
checks product logits/tokens, exact graph execution, provider isolation,
unchanged state/core census, post-run Q/K machine code, and memory safety.  It
does not make a performance claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-qk-rope-layout-exact-phase-product-"
    "precheck-v2")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
SOURCE_GATE = ROOT / (
    "output/openvino-qk-rope-layout-exact-phase-source-"
    "20260731Tseq2194-clean/result.json")
ARITHMETIC_GATE = ROOT / (
    "output/openvino-qk-rope-layout-half-arithmetic-reopen-"
    "20260731Tseq2196-clean/result.json")
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
OCLOC = Path("/usr/bin/ocloc")
BASE_PRECHECK = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-precheck-"
    "20260731Tseq2191-clean/result.json")
FORMAL_GATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/gate.json")
BASE_ROOT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-precheck-"
    "20260731Tseq2191-clean/raw/candidate")
BASE_CONFIG = BASE_ROOT / "worker-config.json"
BASE_CANDIDATE = BASE_ROOT / "worker-result.json"
STOCK = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness/"
    "stock/worker-result.json")
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    SOURCE_GATE: (
        "8a372177df85fc014114e4421b079526761fc2e765fa345095729ee0cc0a2006"),
    ARITHMETIC_GATE: (
        "58741f2284b160cf952a8bd767d0dcbdecb5567f191288e00c23cd0dd0589d3c"),
    KERNEL_SOURCE: (
        "be2b1105df7503a24636615a94255e0683d0b8a73bbecd1c7b70d0b9f5306863"),
    BASE_PRECHECK: (
        "7cfc2b851d16024cee475937d074bf16e5d91dc3eb25970e2572682f6c23917d"),
    FORMAL_GATE: (
        "c125f51dde39d6080ed1b4a8698cb3864874fcf31e3acb5a38fffbae9c86ceee"),
    BASE_CONFIG: (
        "52d91d95864de33eccc07ba2c40f17979551b98a0134e3793f73ff6011abaf24"),
    BASE_CANDIDATE: (
        "fc36dcfdcd8132831f5275abfbd922296bbb4fcd330d53196625aa0ca37a1822"),
    STOCK: (
        "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215"),
    PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
}
EXPECTED_SOURCE_COMMIT = "6d5cc7c207324f594488656ebc0f2c42274ad0ad"
EXPECTED_ARITHMETIC_COMMIT = "e4dfc11eef6dfc569aac35d147f995e14bf8d81e"
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
OUTPUT_TOKENS = 130
KLD_MAX = 0.005
TOP1_MIN = 0.99
PREFLIGHT_GIB = 8.0
MEMORY_STOP_GIB = 4.0
EXPECTED_PROVIDER = "+".join((
    "iq36_lm_head_q8_group256_f16_sums",
    "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16",
    "iq36_lm_head_i8_exact_local_top12_correction_f16",
    "iq36_lm_head_output_topk8_f16",
    "iq36_lm_head_topk8_merge_f32",
    "iq36_lm_head_i8_direct_topk8_correction_f16",
    "iq36_lm_head_i8q1_gated_exact_reset_f16",
    "iq36_lm_head_i8q1_gated_exact_collect_f16",
    "iq36_lm_head_i8_gated_exact_matvec_f16",
    "iq36_lm_head_i8q1_gated_exact_output_topk8_f16",
    "iq36_lm_head_i8q1_gated_exact_topk8_merge_f32",
    "iq36_lm_head_i8_gated_exact_topk8_correction_f16",
))


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_qk_rope_exact_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def contains_bytes(path: Path, needle: bytes) -> bool:
  overlap = b""
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      value = overlap + chunk
      if needle in value:
        return True
      overlap = value[-max(0, len(needle) - 1):]
  return False


def assembly_metrics(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8", errors="replace").lower()
  lines = text.splitlines()

  def count(operation: str, data_type: str) -> int:
    suffix = r"(?<!h):f(?:\s|$)" if data_type == "f" else r":hf(?:\s|$)"
    pattern = re.compile(
        rf"\b{operation} \(32\|m0\).*{suffix}", re.IGNORECASE)
    return sum(bool(pattern.search(line)) for line in lines)

  return {
      "path": PRODUCT.relative(path),
      "sha256": sha256(path),
      "bytes": path.stat().st_size,
      "instruction_lines": sum(
          bool(re.search(r"\([0-9]+\|m[0-9]+\)", line)) for line in lines),
      "simd32_float_mul": count("mul", "f"),
      "simd32_float_mad": count("mad", "f"),
      "simd32_half_mul": count("mul", "hf"),
      "simd32_half_mad": count("mad", "hf"),
      "simd32_float_to_half_moves": sum(
          "mov (32|m0)" in line and ":f" in line and ":hf" in line
          for line in lines),
  }


def qk_codegen_audit(
    cache_root: Path, destination: Path, timeout_s: int,
) -> dict[str, Any]:
  matches = [
      path for path in sorted(cache_root.rglob("*.l0_cache"))
      if contains_bytes(path, b"iq36_qk_rope_layout")]
  unique: dict[str, Path] = {}
  for path in matches:
    unique.setdefault(sha256(path), path)
  rows = []
  for digest, program in sorted(unique.items()):
    dump = destination / digest[:16]
    dump.mkdir(parents=True)
    command = [
        str(OCLOC), "disasm", "-file", str(program), "-dump", str(dump)]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        timeout=timeout_s)
    assemblies = sorted(dump.glob(".text.iq36_qk_rope_layout.asm"))
    rows.append({
        "program": PRODUCT.relative(program),
        "program_sha256": digest,
        "matching_cache_copies": sum(
            sha256(path) == digest for path in matches),
        "command": [
            PRODUCT.relative(Path(value)) if value.startswith(str(ROOT))
            else value for value in command],
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "assemblies": [
            assembly_metrics(path) for path in assemblies],
    })
  return {
      "cache_root": PRODUCT.relative(cache_root),
      "matching_program_count": len(matches),
      "unique_program_count": len(rows),
      "programs": rows,
  }


def distribution_summary(
    reference: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
  rows = PRODUCT.ATTENTION_DIAGNOSTICS.distribution_rows(
      reference, candidate, ROOT)
  klds = [
      float(row["kld_stock_to_candidate"]) for row in rows
      if isinstance(row.get("kld_stock_to_candidate"), (int, float)) and
      math.isfinite(float(row["kld_stock_to_candidate"]))
  ]
  return {
      "finite": bool(rows) and all(row.get("finite") is True for row in rows),
      "max_kld": max(klds) if klds else None,
      "row_count": len(rows),
      "top1_rate": (
          sum(row.get("top1_match") is True for row in rows) / len(rows)
          if rows else 0.0),
  }


def without_qk_summary(source: dict[str, Any]) -> dict[str, Any]:
  value = dict(source)
  for key in (
      "fuse_qk_rope_layout", "qk_rope_layout_rewrite_count",
      "qk_rope_layout_rewrites"):
    value.pop(key, None)
  return value


def without_qk_execution(counts: dict[str, Any]) -> dict[str, Any]:
  value = dict(counts)
  for key in (
      "Concat", "Gather", "IQ36QKRopeLayout",
      "RoPE", "StridedSlice", "Transpose"):
    value.pop(key, None)
  return value


def boundary_audit(result: dict[str, Any]) -> dict[str, Any]:
  rows = (
      result.get("execution_census", {}).get("attention_boundary_rows") or [])
  old_rows = []
  output_transposes = []
  output_gates = []
  for row in rows:
    name = str(row.get("node_name", ""))
    layer = next((value for value in FULL_ATTENTION_LAYERS
                  if f"layers.{value}.self_attn/" in name), None)
    if layer is None:
      continue
    q_transpose = ("/aten::transpose/Transpose_2" if layer == 39
                   else "/aten::transpose/Transpose")
    q_concat = ("/aten::cat/Concat_5" if layer == 39
                else "/aten::cat/Concat_1")
    k_concat = ("/aten::cat/Concat_2" if layer == 39
                else "/aten::cat/Concat_3")
    if ((row.get("node_type") == "Transpose" and name.endswith(
            (q_transpose, "/aten::transpose/Transpose_1"))) or
        (row.get("node_type") == "StridedSlice" and name.endswith(
            ("/aten::slice/Slice", "/aten::slice/Slice_3",
             "/aten::slice/Slice_4", "/aten::slice/Slice_7"))) or
        (row.get("node_type") == "RoPE" and name.endswith(
            ("/aten::add/Add", "/aten::add/Add_1"))) or
        (row.get("node_type") == "Concat" and name.endswith(
            (q_concat, k_concat)))):
      old_rows.append(row)
    if (row.get("node_type") == "Transpose" and
        name.endswith("/aten::transpose/Transpose_3")):
      output_transposes.append(row)
    if (row.get("node_type") == "Multiply" and
        name.endswith("/aten::mul/Multiply_6")):
      output_gates.append(row)
  return {
      "attention_boundary_row_count": len(rows),
      "old_qk_rows": old_rows,
      "output_gate_rows": output_gates,
      "output_transpose_rows": output_transposes,
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = (PRODUCT_TOOL, OCLOC, *EXPECTED_SHA256)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing Q/K product-precheck inputs: " + ", ".join(missing))

  input_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  source_gate = PRODUCT.load_json(SOURCE_GATE)
  arithmetic_gate = PRODUCT.load_json(ARITHMETIC_GATE)
  base_precheck = PRODUCT.load_json(BASE_PRECHECK)
  formal_gate = PRODUCT.load_json(FORMAL_GATE)
  base_config = PRODUCT.load_json(BASE_CONFIG)
  base_candidate = PRODUCT.load_json(BASE_CANDIDATE)
  stock = PRODUCT.load_json(STOCK)
  git = PRODUCT.BOOT.git_state(out)
  expected_tokens = [
      int(value) for value in stock["generated_token_ids"][:OUTPUT_TOKENS]]
  reference_path = out / "reference-output130.json"
  PRODUCT.write_json(reference_path, {
      "generated_token_ids": expected_tokens,
      "source": PRODUCT.relative(STOCK),
  })

  config = dict(base_config)
  config.update({
      "candidate_gpu_plugin": str(PLUGIN),
      "case_id": "sentinel_002k_qk_rope_stock_half_output130",
      "capture_execution_census": True,
      "capture_logits": True,
      "checkpoint_steps": list(range(OUTPUT_TOKENS)),
      "fuse_qk_rope_layout": True,
      "output_tokens": OUTPUT_TOKENS,
      "purpose": "teacher_forced_correctness",
      "reference_result": str(reference_path.resolve()),
  })
  config.pop("compile_only", None)
  config.pop("instantiate_only", None)
  worker_args = SimpleNamespace(
      abort_below_available_gib=MEMORY_STOP_GIB,
      candidate_gpu_plugin=PLUGIN,
      candidate_impls_cache_capacity=None,
      custom_config=PRODUCT.CUSTOM_CONFIG,
      device="GPU",
      min_available_gib=PREFLIGHT_GIB,
      model_dir=PRODUCT.MODEL_DIR,
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=False,
      poll_interval_s=1.0,
      prime_candidate_exact_decode_shape=False,
      resume=False,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  worker = PRODUCT.run_worker(worker_args, raw / "candidate", config)
  result = worker.get("result") or {}
  codegen = qk_codegen_audit(
      raw / "candidate/neo-cache", raw / "qk-codegen", args.timeout_s)
  PRODUCT.write_json(raw / "qk-codegen.json", codegen)
  stock_distribution = distribution_summary(stock, result) if result else {}
  carrier_distribution = (
      distribution_summary(base_candidate, result) if result else {})
  source = result.get("source_summary") or {}
  baseline_source = base_candidate.get("source_summary") or {}
  execution = result.get("execution_census") or {}
  counts = execution.get("executed_type_counts") or {}
  baseline_counts = (
      base_candidate.get("execution_census", {}).get(
          "executed_type_counts") or {})
  boundaries = boundary_audit(result)
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  prepacks = trace.get("weight_prepack_rows") or []
  provider_exact = (
      len(selections) == 2 and
      all(
          row.get("provider") == EXPECTED_PROVIDER and
          row.get("tokens") == 1 and row.get("rows") == 248320 and
          row.get("columns") == 2048 and row.get("correction_passes") == 2
          for row in selections) and
      len(prepacks) == 2 and
      prepacks[0].get("process_cache_hit") is False and
      prepacks[1].get("process_cache_hit") is True)
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  minimum_available = int(
      monitor.get("system_available_min_bytes") or 0)
  stop_bytes = int(MEMORY_STOP_GIB * 1024**3)
  gather_delta_exact = (
      baseline_counts.get("Gather") == 12 and counts.get("Gather") == 11)
  codegen_rows = [
      row["assemblies"][0] for row in codegen["programs"]
      if row["returncode"] == 0 and len(row["assemblies"]) == 1]
  qk_half_codegen_exact = (
      codegen["matching_program_count"] >= 2 and
      codegen["unique_program_count"] == 2 and
      len(codegen_rows) == 2 and
      all(
          row["simd32_half_mul"] == 4 and
          row["simd32_half_mad"] == 4 and
          row["simd32_float_mul"] == 0 and
          row["simd32_float_mad"] == 0 and
          row["simd32_float_to_half_moves"] == 0
          for row in codegen_rows))

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_frozen_inputs_have_exact_hashes",
            all(input_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                PRODUCT.relative(path): value
                for path, value in input_hashes.items()}),
      check("seq2194_scope_seq2196_arithmetic_and_seq2193_carrier_admit",
            source_gate.get("required_checks_passed") is True and
            source_gate.get("product_precheck_admitted") is True and
            source_gate.get("git", {}).get("commit") ==
                EXPECTED_SOURCE_COMMIT and
            arithmetic_gate.get("required_checks_passed") is True and
            arithmetic_gate.get("corrected_product_precheck_admitted")
                is True and
            arithmetic_gate.get("verdict") ==
                "admit_one_stock_half_order_qk_rope_product_precheck" and
            arithmetic_gate.get("git", {}).get("commit") ==
                EXPECTED_ARITHMETIC_COMMIT and
            base_precheck.get("required_checks_passed") is True and
            formal_gate.get("run_checks_passed") is True and
            formal_gate.get("product_promotion_ready") is False and
            formal_gate.get("speedup_claims_allowed") is False),
      check("single_serial_candidate_worker_completes_without_oom",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker.get("reused") is not True and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True,
            worker={
                key: worker.get(key) for key in (
                    "returncode", "timed_out", "oom_observed",
                    "elapsed_seconds", "worker_transient_scope")}),
      check("isolated_exact_phase_parallel_carrier_is_unchanged",
            result.get("candidate_path") == "hot_cold_custom" and
            result.get("custom_composition") == "exact_phase" and
            result.get("target_layers") == list(FULL_ATTENTION_LAYERS) and
            result.get("decode_stock_micro_layers") ==
                list(FULL_ATTENTION_LAYERS) and
            result.get("exact_phase_dual_cohort") is True and
            result.get("candidate_gpu_plugin_sha256") ==
                EXPECTED_SHA256[PLUGIN] and
            result.get("lm_head_i8q1") is True and
            result.get("lm_head_i8q1_gated_exact") is True and
            result.get("lm_head_i8q1_gated_q4") is False and
            provider_exact),
      check("qk_rope_source_state_and_nonboundary_execution_are_exact",
            result.get("fuse_qk_rope_layout") is True and
            source.get("fuse_qk_rope_layout") is True and
            source.get("qk_rope_layout_rewrite_count") ==
                len(FULL_ATTENTION_LAYERS) and
            without_qk_summary(source) ==
                without_qk_summary(baseline_source) and
            result.get("state_schema_after") ==
                base_candidate.get("state_schema_after") and
            gather_delta_exact and
            without_qk_execution(counts) ==
                without_qk_execution(baseline_counts),
            gather_delta={
                "baseline": baseline_counts.get("Gather"),
                "candidate": counts.get("Gather"),
                "classification": "removed QK producer-side index gather"}),
      check("exact_ten_qk_producers_replace_only_old_qk_boundaries",
            counts.get("IQ36QKRopeLayout") ==
                len(FULL_ATTENTION_LAYERS) and
            counts.get("IQ36ExactPhaseDualCohortHotAttentionGQA") ==
                len(FULL_ATTENTION_LAYERS) and
            not boundaries["old_qk_rows"],
            executed_type_counts=counts,
            boundary_audit=boundaries),
      check("rejected_output_transpose_and_gate_routes_remain_live",
            len(boundaries["output_transpose_rows"]) ==
                len(FULL_ATTENTION_LAYERS) and
            len(boundaries["output_gate_rows"]) ==
                len(FULL_ATTENTION_LAYERS) and
            len(source_gate.get("graph_audit", {}).get(
                "output_boundary", [])) == 2 * len(FULL_ATTENTION_LAYERS)),
      check("every_qk_cache_shape_uses_stock_half_mul_mad_order",
            qk_half_codegen_exact, codegen=codegen),
      check("all_130_stock_relative_distributions_pass",
            stock_distribution.get("row_count") == OUTPUT_TOKENS and
            stock_distribution.get("finite") is True and
            float(stock_distribution.get("max_kld", math.inf)) <= KLD_MAX and
            float(stock_distribution.get("top1_rate", 0.0)) >= TOP1_MIN,
            distribution=stock_distribution,
            kld_threshold=KLD_MAX, top1_threshold=TOP1_MIN),
      check("exact_output130_tokens_are_preserved",
            result.get("generated_token_count") == OUTPUT_TOKENS and
            result.get("generated_token_ids") == expected_tokens and
            result.get("teacher_forced_from_stock") is True,
            generated_token_ids_sha256=result.get(
                "generated_token_ids_sha256"),
            note=(
                "the accepted 2k answer first appears after token 130; "
                "output512 sentinel truth remains required before promotion")),
      check("memory_guard_never_trips",
            guard.get("tripped") is False and
            minimum_available >= stop_bytes and
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0,
            stop_bytes=stop_bytes, monitor=monitor),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_qk_rope_layout_for_one_2k_abba_precheck"
      if required else
      "reject_qk_rope_layout_before_performance")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "abba_precheck_admitted": required,
      "formal_product_promotion_admitted": False,
      "performance_claim_admitted": False,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "candidate_workers_launched": 1,
      "workers_concurrent": False,
      "checks": checks,
      "plugin": {"path": str(PLUGIN), "sha256": input_hashes[PLUGIN]},
      "correctness": {
          "stock_relative": stock_distribution,
          "current_carrier_relative": carrier_distribution,
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
          "sentinel_pass": result.get("sentinel_pass"),
      },
      "execution": {
          "boundary_audit": boundaries,
          "executed_type_counts": counts,
          "qk_codegen": codegen,
      },
      "worker": worker,
      "next_action": {
          "route": "openvino_qk_rope_layout_stock_half_2k_abba1",
          "requirements": [
              "run one serial control-candidate-candidate-control block",
              "keep the seq2189 plugin and every non-QK carrier field exact",
              "require correctness, no OOM, and a paired incremental win",
          ],
      },
  }
  PRODUCT.write_json(out / "result.json", payload)
  PRODUCT.write_json(out / "manifest.json", {
      "schema": SCHEMA,
      "tool": PRODUCT.relative(Path(__file__)),
      "git": git,
      "inputs": {
          PRODUCT.relative(path): value
          for path, value in input_hashes.items()
      },
      "gpu_workers": 1,
      "stock_workers": 0,
      "candidate_workers": 1,
      "workers_concurrent": False,
  })
  report = f"""# Exact-phase Q/K RoPE product precheck

Verdict: **{verdict}**. Required checks: `{str(required).lower()}`.

One isolated 2k/output130 candidate worker retains the accepted exact-phase
dual-cohort and parallel block-top8 carrier. It executes
`{counts.get('IQ36QKRopeLayout', 0)}` fused Q/K producers and
`{counts.get('IQ36ExactPhaseDualCohortHotAttentionGQA', 0)}` exact attention
owners. Old Q/K boundary executions are
`{len(boundaries['old_qk_rows'])}`; the ten rejected output transposes and ten
output gates remain live.

Stock-relative max KLD/top-1 are
`{stock_distribution.get('max_kld')}/{stock_distribution.get('top1_rate')}`.
The output130 token SHA is `{result.get('generated_token_ids_sha256')}`.
The post-run cache contains `{codegen['matching_program_count']}` matching Q/K
program copies across `{codegen['unique_program_count']}` unique shapes; the
stock-half codegen check is `{str(qk_half_codegen_exact).lower()}`.

Peak RSS/swap telemetry is
`{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`; minimum available memory
is `{minimum_available} B`. This gate launches no stock worker and makes no
performance or promotion claim.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "qk_execution_count": counts.get("IQ36QKRopeLayout", 0),
      "old_qk_boundary_count": len(boundaries["old_qk_rows"]),
      "max_kld": stock_distribution.get("max_kld"),
      "top1_rate": stock_distribution.get("top1_rate"),
      "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
      "oom_observed": worker.get("oom_observed"),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
