#!/usr/bin/env python3
"""Bound one official-address K-prefetch correction for exact attention.

This gate is source and artifact only.  It compares the accepted dual/triple
carrier's K-prefetch calls with pinned official oneDNN GPU SDPA source,
quantifies the fixed repeated address coverage, and verifies the emitted
prefetch sends in the accepted triple binary.  It never compiles, creates a
GPU context, or launches a model worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "k-prefetch-address-bound-gate-v1")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
OFFICIAL_MICRO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu/src/gpu/intel/sdpa/micro.cl")
ACCEPTED_FUSED = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-"
    "20260724Tseq2145-clean/raw/triple-cohort/existing_shim.fused.cl")
ACCEPTED_ASM = ROOT / (
    "output/openvino-exact-attention-hardware-limit-opportunity-"
    "20260724Tseq2150a-clean/raw/triple-disassembly/"
    ".text.iq36_exact_score_triple_cohort.asm")
TRIPLE_COMPONENT = ROOT / (
    "output/openvino-exact-attention-triple-cohort-component-"
    "20260724Tseq2146-clean/result.json")
TRAFFIC_CEILING = ROOT / (
    "output/openvino-exact-attention-two-workgroup-traffic-"
    "20260724Tseq2151-clean/result.json")
HARDWARE_AUDIT = ROOT / (
    "output/openvino-exact-attention-hardware-limit-opportunity-"
    "20260724Tseq2150a-clean/result.json")

CONTEXT = 131_072
HEAD_DIM = 256
KV_HEADS = 2
KEY_BLOCK = 256
BLOCKS = CONTEXT // KEY_BLOCK
CURRENT_PREFETCH_COLUMNS = 64
ELEMENT_BYTES = 2
MANDATORY_K_BYTES = KV_HEADS * CONTEXT * HEAD_DIM * ELEMENT_BYTES
MANDATORY_KV_BYTES = 2 * MANDATORY_K_BYTES
CURRENT_PREFETCH_BYTES_PER_CALL = (
    KEY_BLOCK * CURRENT_PREFETCH_COLUMNS * ELEMENT_BYTES)
OFFICIAL_PREFETCH_BYTES_PER_CALL = KEY_BLOCK * HEAD_DIM * ELEMENT_BYTES
CURRENT_LOGICAL_REQUEST_BYTES = (
    KV_HEADS * BLOCKS * CURRENT_PREFETCH_BYTES_PER_CALL)
CURRENT_UNIQUE_ADDRESS_BYTES = (
    KV_HEADS * CURRENT_PREFETCH_BYTES_PER_CALL)
CURRENT_REPEATED_ADDRESS_BYTES = (
    CURRENT_LOGICAL_REQUEST_BYTES - CURRENT_UNIQUE_ADDRESS_BYTES)
REGISTERED_DELTA_CAP_MS = 0.1175998
REQUIRED_TRIPLE_LATENCY_MS = 2.7375042
EXPECTED_PREFETCH_SENDS = 4
EXPECTED_BLOCK2D_LOADS = 48


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


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
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  rows = [row for row in rows if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def section(text: str, begin: str, end: str) -> str:
  if text.count(begin) != 1:
    return ""
  remainder = text.split(begin, 1)[1]
  if end not in remainder:
    return ""
  return remainder.split(end, 1)[0]


def median(values: list[float]) -> float:
  ordered = sorted(values)
  if not ordered:
    raise ValueError("median requires values")
  middle = len(ordered) // 2
  return (
      ordered[middle]
      if len(ordered) % 2 else
      (ordered[middle - 1] + ordered[middle]) / 2.0)


def summary(payload: dict[str, Any]) -> str:
  bound = payload["address_bound"]
  opportunity = payload["performance_opportunity"]
  return "\n".join([
      "# Exact-attention K-prefetch address bound",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- current call / unique coverage: "
      f"`{bound['current_prefetch_bytes_per_call']} / "
      f"{bound['current_unique_address_bytes']} B`",
      f"- current logical / repeated-address requests: "
      f"`{bound['current_logical_request_bytes']} / "
      f"{bound['current_repeated_address_bytes']} B`",
      f"- intended K payload / official call: "
      f"`{bound['mandatory_k_bytes']} / "
      f"{bound['official_prefetch_bytes_per_call']} B`",
      f"- accepted ISA prefetch / block2D sends: "
      f"`{bound['accepted_static_prefetch_send_count']} / "
      f"{bound['accepted_static_block2d_load_count']}`",
      f"- triple median / traffic UCB / gap: "
      f"`{opportunity['triple_median_ms']} / "
      f"{opportunity['traffic_latency_ucb_ms']} / "
      f"{opportunity['triple_minus_traffic_ucb_ms']} ms/layer`",
      "",
      "A pass admits one triple-carrier correction to the pinned official",
      "K_next/remaining-context addressing only. Prefetch enablement,",
      "distance, cache policy, geometry, and register variants remain closed.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_memory = available_memory_bytes()
  if start_memory < stop_bytes:
    raise SystemExit(
        f"memory stop: {start_memory} < {stop_bytes} bytes")

  required_paths = (
      SOURCE, SHIMS, OFFICIAL_MICRO, ACCEPTED_FUSED, ACCEPTED_ASM,
      TRIPLE_COMPONENT, TRAFFIC_CEILING, HARDWARE_AUDIT)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing K-prefetch address inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  source_text = SOURCE.read_text(encoding="utf-8")
  shim_text = SHIMS.read_text(encoding="utf-8")
  official_text = OFFICIAL_MICRO.read_text(encoding="utf-8")
  fused_text = ACCEPTED_FUSED.read_text(encoding="utf-8")
  asm_text = ACCEPTED_ASM.read_text(encoding="utf-8")
  triple = load_json(TRIPLE_COMPONENT)
  traffic = load_json(TRAFFIC_CEILING)
  audit = load_json(HARDWARE_AUDIT)

  triple_source = section(
      source_text,
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_COHORT",
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT")
  dual_source = section(
      source_text,
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT",
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_CAPTURE")
  accepted_triple_source = section(
      fused_text,
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_COHORT",
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT")
  current_address_pattern = (
      "key_base, IQ36_D, IQ36_CONTEXT,\n"
      "        ugemm_kq_wg_tile_m, 64, IQ36_D,")
  current_address_pattern_indented = (
      "key_base, IQ36_D, IQ36_CONTEXT,\n"
      "            ugemm_kq_wg_tile_m, 64, IQ36_D,")
  official_next_pattern = (
      "const global KEY_DATA_T *K_next = K + (knext)*stride_k;")
  official_remaining_pattern = (
      "/* r */ k0end - k0 - ugemm_kq_wg_tile_m,")
  official_dimension_pattern = "/* c */ d,"
  official_pointer_pattern = "/* ptr */ K_next,"
  shim_mapping_pattern = (
      "(r)*sizeof(*(ptr)),c,(rmax)*sizeof(*(ptr)),cmax,"
      "(ld)*sizeof(*(ptr))")

  prefetch_send_lines = [
      line.strip() for line in asm_text.splitlines()
      if "send.ugm" in line and "load.ugm.d8u32.a64.ca.ca" in line]
  block2d_lines = [
      line.strip() for line in asm_text.splitlines()
      if "send.ugm" in line
      and "load_block2d.ugm" in line
      and ".ca.ca" in line]

  triple_rows = triple.get("result", {}).get("paired_samples", [])
  triple_values = [
      float(row["triple_ms"]) for row in triple_rows
      if isinstance(row, dict) and "triple_ms" in row]
  triple_median_ms = median(triple_values)
  traffic_inference = traffic.get("performance_inference", {})
  traffic_ucb_ms = float(
      traffic_inference.get("upper_confidence_bound_ms", math.nan))
  traffic_lcb_gb_s = float(
      traffic.get("bandwidth_lcb_gb_s", math.nan))
  triple_minus_traffic_ucb_ms = triple_median_ms - traffic_ucb_ms
  triple_deficit_ms = triple_median_ms - REQUIRED_TRIPLE_LATENCY_MS

  address_bound = {
      "context_tokens": CONTEXT,
      "head_dim": HEAD_DIM,
      "kv_heads": KV_HEADS,
      "key_block": KEY_BLOCK,
      "block_count": BLOCKS,
      "current_prefetch_columns": CURRENT_PREFETCH_COLUMNS,
      "current_prefetch_bytes_per_call": CURRENT_PREFETCH_BYTES_PER_CALL,
      "official_prefetch_bytes_per_call": OFFICIAL_PREFETCH_BYTES_PER_CALL,
      "current_calls_per_workgroup": BLOCKS,
      "current_logical_request_bytes": CURRENT_LOGICAL_REQUEST_BYTES,
      "current_unique_address_bytes": CURRENT_UNIQUE_ADDRESS_BYTES,
      "current_repeated_address_bytes": CURRENT_REPEATED_ADDRESS_BYTES,
      "current_repeated_address_fraction_of_mandatory_kv":
          CURRENT_REPEATED_ADDRESS_BYTES / MANDATORY_KV_BYTES,
      "mandatory_k_bytes": MANDATORY_K_BYTES,
      "mandatory_kv_bytes": MANDATORY_KV_BYTES,
      "official_coverage_equals_full_k_payload":
          OFFICIAL_PREFETCH_BYTES_PER_CALL * BLOCKS * KV_HEADS ==
          MANDATORY_K_BYTES,
      "accepted_static_prefetch_send_count": len(prefetch_send_lines),
      "accepted_static_block2d_load_count": len(block2d_lines),
      "accepted_static_prefetch_send_lines": prefetch_send_lines,
  }
  performance_opportunity = {
      "triple_median_ms": triple_median_ms,
      "traffic_latency_ucb_ms": traffic_ucb_ms,
      "traffic_bandwidth_lcb_gb_s": traffic_lcb_gb_s,
      "triple_minus_traffic_ucb_ms": triple_minus_traffic_ucb_ms,
      "registered_delta_cap_ms": REGISTERED_DELTA_CAP_MS,
      "triple_required_latency_ms": REQUIRED_TRIPLE_LATENCY_MS,
      "triple_deficit_ms": triple_deficit_ms,
      "physical_headroom_multiple_of_registered_cut":
          triple_minus_traffic_ucb_ms / REGISTERED_DELTA_CAP_MS,
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check(
          "seq2151_proves_two_workgroup_physical_headroom",
          traffic.get("verdict") ==
              "admit_one_source_bound_package_or_synchronization_cut"
          and traffic.get("required_checks_passed") is True
          and traffic.get("measurement_valid") is True
          and traffic.get("traffic_capacity_pass") is True
          and traffic.get(
              "exact_kernel_implementation_admitted") is False),
      check(
          "accepted_triple_remains_exact_but_subthreshold",
          triple.get("verdict") ==
              "reject_exact_attention_triple_cohort_component"
          and triple.get("required_checks_passed") is False
          and len(triple_values) == 20
          and math.isclose(
              triple_median_ms, 2.7618225,
              rel_tol=0.0, abs_tol=1.0e-12)),
      check(
          "working_and_accepted_triple_repeat_base_address",
          triple_source.count(current_address_pattern) == 1
          and triple_source.count(current_address_pattern_indented) == 1
          and accepted_triple_source.count(current_address_pattern) == 1
          and accepted_triple_source.count(
              current_address_pattern_indented) == 1
          and "key_base + (ulong)next_block" not in triple_source),
      check(
          "accepted_dual_shares_the_same_address_divergence",
          dual_source.count(current_address_pattern) == 1
          and dual_source.count(current_address_pattern_indented) == 1
          and "key_base + (ulong)next_block" not in dual_source),
      check(
          "pinned_official_source_advances_k_next_and_remaining_context",
          official_next_pattern in official_text
          and official_pointer_pattern in official_text
          and official_remaining_pattern in official_text
          and official_dimension_pattern in official_text),
      check(
          "shim_argument_mapping_proves_transposed_fixed_coverage",
          shim_mapping_pattern in re.sub(r"\s+", "", shim_text)),
      check(
          "fixed_address_arithmetic_is_exact",
          BLOCKS == 512
          and CURRENT_PREFETCH_BYTES_PER_CALL == 32_768
          and OFFICIAL_PREFETCH_BYTES_PER_CALL == 131_072
          and CURRENT_LOGICAL_REQUEST_BYTES == 33_554_432
          and CURRENT_UNIQUE_ADDRESS_BYTES == 65_536
          and CURRENT_REPEATED_ADDRESS_BYTES == 33_488_896
          and address_bound["official_coverage_equals_full_k_payload"]),
      check(
          "accepted_isa_contains_the_fixed_prefetch_send_shape",
          len(prefetch_send_lines) == EXPECTED_PREFETCH_SENDS
          and len(block2d_lines) == EXPECTED_BLOCK2D_LOADS
          and audit.get("accepted_triple_isa", {}).get(
              "load_block2d_ca_ca_count") == EXPECTED_BLOCK2D_LOADS),
      check(
          "physical_headroom_exceeds_registered_cut",
          math.isfinite(traffic_ucb_ms)
          and triple_minus_traffic_ucb_ms >= REGISTERED_DELTA_CAP_MS
          and triple_deficit_ms > 0.0
          and traffic_lcb_gb_s >= 98.05846361806495,
          performance_opportunity=performance_opportunity),
      check("no_compiler_gpu_plugin_or_model_worker_launched", True),
      check(
          "memory_stop_not_crossed",
          available_memory_bytes() >= stop_bytes,
          start_available_bytes=start_memory,
          stop_bytes=stop_bytes),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_official_address_triple_prefetch_correction_codegen"
      if required else
      "reject_official_address_triple_prefetch_correction")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in required_paths
  ]
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "compiler_resource_probe_admitted": required,
      "standalone_component_admitted": False,
      "kernel_enqueue_admitted": False,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "compiler_workers_launched": 0,
      "gpu_workers_launched": 0,
      "plugin_workers_launched": 0,
      "model_workers_launched": 0,
      "checks": checks,
      "address_bound": address_bound,
      "performance_opportunity": performance_opportunity,
      "fixed_correction_contract": {
          "scope": "triple carrier K prefetch addresses only",
          "initial_pointer": "key_base",
          "initial_rows": "IQ36_CONTEXT",
          "loop_pointer":
              "key_base + next_block * ugemm_kq_wg_tile_m * IQ36_D",
          "loop_rows":
              "IQ36_CONTEXT - next_block * ugemm_kq_wg_tile_m",
          "columns": "IQ36_D",
          "maximum_rows": "ugemm_kq_wg_tile_m",
          "maximum_columns": "IQ36_D",
          "leading_dimension": "IQ36_D",
          "must_not_change": [
              "prefetch enablement",
              "prefetch distance",
              "cache policy",
              "cohort geometry",
              "SLM buffers",
              "barriers",
              "generated KQ or VS packages",
              "register-file mode",
          ],
      },
      "memory_stop_bytes": stop_bytes,
      "memory_samples": {
          "start_available_bytes": start_memory,
          "end_available_bytes": available_memory_bytes(),
      },
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": ["result.json", "summary.md", "manifest.json"],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "current_prefetch_bytes_per_call":
          CURRENT_PREFETCH_BYTES_PER_CALL,
      "current_repeated_address_bytes":
          CURRENT_REPEATED_ADDRESS_BYTES,
      "official_prefetch_bytes_per_call":
          OFFICIAL_PREFETCH_BYTES_PER_CALL,
      "triple_minus_traffic_ucb_ms": triple_minus_traffic_ucb_ms,
      "registered_delta_cap_ms": REGISTERED_DELTA_CAP_MS,
      "compiler_workers_launched": 0,
      "gpu_workers_launched": 0,
      "model_workers_launched": 0,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
