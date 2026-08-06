#!/usr/bin/env python3
"""Prove the Q/K RoPE arithmetic mismatch and admit one corrected precheck.

This gate launches no GPU context, compiler, model, or inference worker.  It
binds the failed seq2195 product row, disassembles its frozen custom programs
and an accepted stock RoPE program offline, and checks the sole source change:
the fused producer now keeps the stock GPU's half multiply/mad arithmetic
instead of promoting every operand to float.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import resource
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-qk-rope-layout-half-arithmetic-"
    "reopen-gate-v1")
OCLoc = Path("/usr/bin/ocloc")
KERNEL_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
SOURCE_GATE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-qk-rope-layout-exact-phase-source-gate.py")
REJECTED_ROUTES = ROOT / (
    "doc/active/intel-qwen36-35b-a3b-gguf-q4km/rejected-routes.json")
SEQ2195_ROOT = ROOT / (
    "output/openvino-qk-rope-layout-exact-phase-product-precheck-"
    "20260731Tseq2195-clean")
SEQ2195_RESULT = SEQ2195_ROOT / "result.json"
SEQ2195_CANDIDATE = SEQ2195_ROOT / "raw/candidate/worker-result.json"
BASE_CANDIDATE = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-precheck-"
    "20260731Tseq2191-clean/raw/candidate/worker-result.json")
STOCK = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness/"
    "stock/worker-result.json")
OLD_CUSTOM_PROGRAMS = (
    SEQ2195_ROOT / (
        "raw/candidate/neo-cache/2/d/2ddbc24ecfe41d7f.l0_cache"),
    SEQ2195_ROOT / (
        "raw/candidate/neo-cache/2/3/2303c7a635b11227.l0_cache"),
)
STOCK_PROGRAM = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-product-precheck-"
    "20260731Tseq2191-clean/raw/candidate/neo-cache/5/8/"
    "5872046ec0f4a4e8.l0_cache")
EXPECTED_SHA256 = {
    SEQ2195_RESULT: (
        "4ec9a1414b4cfd47f8b76828e674d8e1730d1c14455e5554c7283fc0073dd0f7"),
    SEQ2195_CANDIDATE: (
        "3a3e72e5abb1c07d7d6ec37188826eb445db5ab6f88f4aa51d73ca4fa6857bc6"),
    BASE_CANDIDATE: (
        "fc36dcfdcd8132831f5275abfbd922296bbb4fcd330d53196625aa0ca37a1822"),
    STOCK: (
        "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215"),
    OLD_CUSTOM_PROGRAMS[0]: (
        "9db34640e86acb7aeeaab4d211faee5457cc7a16e1c074480cfb583c64ef8ba1"),
    OLD_CUSTOM_PROGRAMS[1]: (
        "23f1f649e5fbc7548634fb255bfa8a9209cd28973009d87078adf9df2a9b6b7e"),
    STOCK_PROGRAM: (
        "8f76f6cab56457d868322392fcb3ed3a9c9428263c6009d050eab061083d91c2"),
}
LAYERS = tuple(range(3, 40, 4))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=120)
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


def assembly_metrics(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8", errors="replace").lower()
  lines = text.splitlines()

  def count(op: str, data_type: str) -> int:
    suffix = r"(?<!h):f(?:\s|$)" if data_type == "f" else r":hf(?:\s|$)"
    pattern = re.compile(
        rf"\b{op} \(32\|m0\).*{suffix}", re.IGNORECASE)
    return sum(bool(pattern.search(line)) for line in lines)

  return {
      "path": relative(path),
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


def disassemble(
    program: Path, destination: Path, kernel_pattern: str, timeout_s: int,
) -> dict[str, Any]:
  destination.mkdir(parents=True)
  command = [
      str(OCLoc), "disasm", "-file", str(program),
      "-dump", str(destination)]
  completed = subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      timeout=timeout_s)
  assemblies = sorted(destination.glob(kernel_pattern))
  return {
      "program": relative(program),
      "program_sha256": sha256(program),
      "command": [relative(Path(value)) if value.startswith(str(ROOT))
                  else value for value in command],
      "returncode": completed.returncode,
      "stdout": completed.stdout[-12000:],
      "stderr": completed.stderr[-12000:],
      "assemblies": [assembly_metrics(path) for path in assemblies],
  }


def source_audit() -> dict[str, Any]:
  source = KERNEL_SOURCE.read_text(encoding="utf-8")
  source_gate = SOURCE_GATE_TOOL.read_text(encoding="utf-8")
  return {
      "half_helper_signature_exact": all(
          token in source for token in (
              "inline half iq36_qk_rope_value(",
              "const half value, const half peer, const half cosine,",
              "const half sine, const bool first_half) {")),
      "half_live_values_exact": (
          source.count("half query_value = convert_half_rte(") == 1 and
          source.count("half key_value = convert_half_rte(") == 1 and
          source.count("convert_half_rte(") == 8),
      "float_promotion_removed": all(
          token not in source for token in (
              "inline float iq36_qk_rope_value(",
              "convert_float(", "float query_value", "float key_value")),
      "arithmetic_order_retained": (
          source.count("cosine * value - sine * peer") == 1 and
          source.count("cosine * value + sine * peer") == 1),
      "layout_indexing_retained": all(
          token in source for token in (
              "query_head = batch_head % (uint)OUTPUT0_DIMS[1]",
              "query_head < (uint)OUTPUT1_DIMS[1]",
              "peer_dimension = first_half",
              "dimension + 32U : dimension - 32U",
              "(ulong)token * INPUT2_PITCHES[2]",
              "(ulong)token * INPUT3_PITCHES[2]")),
      "canonical_source_gate_requires_half_order": all(
          token in source_gate for token in (
              '"inline half iq36_qk_rope_value("',
              '"half query_value = convert_half_rte("',
              '"half key_value = convert_half_rte("',
              'kernel.count("convert_half_rte(") == 8',
              '"convert_float(" not in kernel')),
  }


def rejected_route_audit(value: dict[str, Any]) -> dict[str, Any]:
  routes = value.get("rejected", [])
  historical = next(
      (row for row in routes if row.get("route") ==
       "openvino_attention_qk_rope_fusion_plus_layer3_chunk256_v30q"), {})
  current = next(
      (row for row in routes if row.get("route") ==
       "openvino_qk_rope_layout_exact_phase_float_arithmetic_v1"), {})
  return {
      "historical": historical,
      "current": current,
      "historical_requires_new_arithmetic_order_proof": (
          "new arithmetic-order proof" in str(
              historical.get("reopen_condition", ""))),
      "current_allows_only_stock_half_codegen_correction": (
          "stock-half" in str(current.get("reopen_condition", "")).lower() and
          "no unchanged" in str(current.get("reopen_condition", "")).lower()),
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists():
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True)
  required = (
      OCLoc, KERNEL_SOURCE, SOURCE_GATE_TOOL, REJECTED_ROUTES,
      *EXPECTED_SHA256)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing arithmetic-gate inputs: " + ", ".join(missing))

  git = git_state()
  frozen_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  failure = load_json(SEQ2195_RESULT)
  candidate = load_json(SEQ2195_CANDIDATE)
  baseline = load_json(BASE_CANDIDATE)
  stock = load_json(STOCK)
  rejected = rejected_route_audit(load_json(REJECTED_ROUTES))
  source = source_audit()
  before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  custom_disassembly = [
      disassemble(
          program, raw / "old-custom" / sha256(program)[:16],
          ".text.iq36_qk_rope_layout.asm", args.timeout_s)
      for program in OLD_CUSTOM_PROGRAMS
  ]
  stock_disassembly = disassemble(
      STOCK_PROGRAM, raw / "stock" / sha256(STOCK_PROGRAM)[:16],
      ".text.rope_opt__*.asm", args.timeout_s)
  after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  write_json(raw / "disassembly.json", {
      "old_custom": custom_disassembly,
      "stock": stock_disassembly,
  })

  old_assemblies = [
      row["assemblies"][0] for row in custom_disassembly
      if row["returncode"] == 0 and len(row["assemblies"]) == 1]
  stock_assemblies = stock_disassembly["assemblies"]
  old_float_exact = (
      len(old_assemblies) == 2 and
      all(
          row["simd32_float_mul"] == 4 and
          row["simd32_float_mad"] == 4 and
          row["simd32_half_mul"] == 0 and
          row["simd32_half_mad"] == 0 and
          row["simd32_float_to_half_moves"] >= 4
          for row in old_assemblies))
  stock_half_exact = (
      len(stock_assemblies) == 2 and
      all(
          row["simd32_float_mul"] == 0 and
          row["simd32_float_mad"] == 0 and
          row["simd32_half_mul"] == 32 and
          row["simd32_half_mad"] == 32
          for row in stock_assemblies))

  stock_tokens = [int(value) for value in stock["generated_token_ids"][:130]]
  candidate_tokens = [
      int(value) for value in candidate["generated_token_ids"][:130]]
  token_mismatches = [
      {"step": index, "stock": expected, "candidate": observed}
      for index, (expected, observed) in enumerate(
          zip(stock_tokens, candidate_tokens))
      if expected != observed]
  counts = failure.get("execution", {}).get("executed_type_counts", {})
  boundaries = failure.get("execution", {}).get("boundary_audit", {})
  base_profile_rows = [
      row for row in baseline.get("execution_census", {}).get("top_rows", [])
      if row.get("node_type") == "Transpose" and
      row.get("exec_type") == "permute_ref__f16" and
      any(
          f"layers.{layer}.self_attn/" in str(row.get("node_name", ""))
          for layer in LAYERS)]
  failed_checks = {
      row.get("name") for row in failure.get("checks", [])
      if row.get("pass") is False}

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_frozen_inputs_have_exact_hashes",
            all(frozen_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                relative(path): digest
                for path, digest in frozen_hashes.items()}),
      check("seq2195_is_scoped_execution_pass_correctness_rejection",
            failure.get("git", {}).get("commit") ==
                "9219906a3094baa9bd2881da48a3d2661ce16411" and
            failure.get("verdict") ==
                "reject_qk_rope_layout_before_performance" and
            failure.get("required_checks_passed") is False and
            counts.get("IQ36QKRopeLayout") == len(LAYERS) and
            counts.get("IQ36ExactPhaseDualCohortHotAttentionGQA") ==
                len(LAYERS) and
            not boundaries.get("old_qk_rows") and
            len(boundaries.get("output_transpose_rows", [])) == len(LAYERS) and
            len(boundaries.get("output_gate_rows", [])) == len(LAYERS) and
            failed_checks == {
                "qk_rope_source_state_and_nonboundary_execution_are_exact",
                "all_130_stock_relative_distributions_pass",
                "exact_output130_tokens_are_preserved"},
            failed_checks=sorted(failed_checks), counts=counts),
      check("seq2195_failure_is_exactly_bound",
            failure.get("correctness", {}).get(
                "generated_token_ids_sha256") ==
                "462a1f9e14734cfb0c1312f631d19380fd70cebb7edaea1bbdf6fcdd2b0c63d7" and
            failure.get("correctness", {}).get(
                "stock_relative", {}).get("max_kld") ==
                0.06734803267242616 and
            failure.get("correctness", {}).get(
                "stock_relative", {}).get("top1_rate") ==
                0.9923076923076923 and
            token_mismatches == [
                {"step": 42, "stock": 8211, "candidate": 5435}],
            token_mismatches=token_mismatches,
            correctness=failure.get("correctness")),
      check("stock_product_profile_executes_f16_qk_boundary",
            len(base_profile_rows) == len(LAYERS),
            rows=base_profile_rows),
      check("old_custom_programs_prove_float_mul_mad_and_half_store",
            len(custom_disassembly) == 2 and
            all(row["returncode"] == 0 for row in custom_disassembly) and
            old_float_exact,
            disassembly=custom_disassembly),
      check("stock_program_proves_half_mul_mad",
            stock_disassembly["returncode"] == 0 and stock_half_exact,
            disassembly=stock_disassembly),
      check("corrected_source_keeps_stock_half_arithmetic_order",
            all(source.values()), source=source),
      check("rejected_route_reopen_conditions_are_satisfied_only_by_new_proof",
            rejected["historical_requires_new_arithmetic_order_proof"] and
            rejected["current_allows_only_stock_half_codegen_correction"],
            rejected_routes=rejected),
      check("offline_gate_launches_no_gpu_compiler_model_or_inference_worker",
            True, gpu_contexts_created=0, compiler_processes=0,
            model_workers=0, inference_workers=0,
            offline_disassembly_processes=3),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_stock_half_order_qk_rope_product_precheck"
      if passed else
      "close_qk_rope_layout_without_corrected_product_run")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": passed,
      "corrected_product_precheck_admitted": passed,
      "performance_claim_admitted": False,
      "formal_product_promotion_admitted": False,
      "source": source,
      "seq2195": {
          "correctness": failure.get("correctness"),
          "token_mismatches": token_mismatches,
          "executed_type_counts": counts,
      },
      "codegen": {
          "old_custom": custom_disassembly,
          "stock": stock_disassembly,
      },
      "checks": checks,
      "child_max_rss_kib_upper_bound": max(0, after_rss - before_rss),
      "gpu_contexts_created": 0,
      "compiler_processes": 0,
      "model_workers": 0,
      "inference_workers": 0,
      "next_action": {
          "route": "openvino_qk_rope_layout_stock_half_product_precheck",
          "requirements": [
              "push the half-order source and corrected precheck gate first",
              "run one serial 2k output130 candidate and no stock worker",
              "require exact output130 tokens and stock KLD at most 0.005",
              "disassemble every executed QK cache shape after the worker",
              "require SIMD32 half mul/mad and no QK SIMD32 float mul/mad",
              "keep 8-GiB preflight and 4-GiB abort guards",
              "run no ABBA unless every correctness and codegen check passes",
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
              KERNEL_SOURCE, SOURCE_GATE_TOOL, REJECTED_ROUTES,
              *EXPECTED_SHA256)
      },
      "gpu_contexts_created": 0,
      "compiler_processes": 0,
      "model_workers": 0,
      "inference_workers": 0,
      "offline_disassembly_processes": 3,
  })
  report = f"""# Q/K RoPE stock-half arithmetic reopen gate

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

Seq2195 activates all ten fused Q/K producers but changes one output130 token
at step 42 and reaches stock-relative max KLD
`{failure.get('correctness', {}).get('stock_relative', {}).get('max_kld')}`.
Both frozen custom program shapes use four SIMD32 F32 multiply/mad pairs and
convert back to half. The accepted stock RoPE program's two kernels each use
32 SIMD32 half multiply/mad pairs and no SIMD32 float pair.

The corrected source changes only live values and operands from float to half;
the rotate-half expression order and layout indexing remain exact. This
offline gate launches no GPU context, compiler, model, or inference worker and
admits at most one corrected product correctness/codegen precheck.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": relative(out),
      "verdict": verdict,
      "required_checks_passed": passed,
      "token_mismatches": token_mismatches,
      "old_custom_float_pairs": [
          [row["simd32_float_mul"], row["simd32_float_mad"]]
          for row in old_assemblies],
      "stock_half_pairs": [
          [row["simd32_half_mul"], row["simd32_half_mad"]]
          for row in stock_assemblies],
      "child_max_rss_kib_upper_bound": max(0, after_rss - before_rss),
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
