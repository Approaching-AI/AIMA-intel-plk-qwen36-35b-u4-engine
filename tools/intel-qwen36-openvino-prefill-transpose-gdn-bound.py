#!/usr/bin/env python3
"""Consolidate the priority-prefill Transpose+GDN boundary source gate.

The gate only audits stored exact component/codegen evidence and the registered
priority-prefill arithmetic.  It never compiles or launches a GPU worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-prefill-transpose-gdn-bound-v0"

ACTIVE = REPO / "doc/active" / WS
ROADMAP = ACTIVE / (
    "intel-qwen36-35b-a3b-gguf-q4km-"
    "openvino-specialization-roadmap-2026-07-13.md")
STATUS = ACTIVE / "STATUS.md"
REJECTED = ACTIVE / "rejected-routes.json"
ACCEPTANCE = REPO / "benchmarks" / WS / "acceptance-matrix.json"

SEQ807 = REPO / (
    "output/openvino-gdn-codegen-20260714Tseq807-"
    "scalar-index-cleanZ")
SEQ1109 = REPO / (
    "output/openvino-gdn-custom-20260715Tseq1109-"
    "qkv-transpose-fused-dirtyZ")
SEQ1110 = REPO / (
    "output/openvino-gdn-custom-20260715Tseq1110-"
    "qkv-tile-fused-dirtyZ")

SEQ807_CORRECTNESS = SEQ807 / "correctness.json"
SEQ807_NUMERIC_CORRECTNESS = SEQ807 / "raw/numeric/correctness.json"
SEQ807_NUMERIC_METRICS = SEQ807 / "raw/numeric/metrics.jsonl"
SEQ807_CODEGEN_METRICS = SEQ807 / "metrics.jsonl"
SEQ1109_CORRECTNESS = SEQ1109 / "correctness.json"
SEQ1109_METRICS = SEQ1109 / "metrics.jsonl"
SEQ1110_CORRECTNESS = SEQ1110 / "correctness.json"
SEQ1110_METRICS = SEQ1110 / "metrics.jsonl"


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise TypeError(f"expected JSON object: {path}:{line_number}")
    rows.append(value)
  return rows


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.relative_to(REPO))
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
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def failed_checks(value: dict[str, Any]) -> list[str]:
  return sorted(
      str(row.get("name")) for row in value.get("checks", [])
      if row.get("pass") is not True)


def all_nested_checks_pass(value: dict[str, Any]) -> bool:
  return all(
      row.get("pass") is True
      for key in ("stock_checks", "candidate_checks")
      for row in value.get(key, []))


def exact_numeric_evidence(value: dict[str, Any]) -> bool:
  component = value.get("component_comparison", {})
  real = value.get("real_model_comparison", {})
  attention = component.get("attention", {})
  state = component.get("final_state", {})
  logits = real.get("logits", {})
  states = real.get("states", {})
  return (
      attention.get("exact_bits") is True and
      attention.get("finite") is True and
      float(attention.get("max_abs", -1.0)) == 0.0 and
      state.get("exact_bits") is True and
      state.get("finite") is True and
      float(state.get("max_abs", -1.0)) == 0.0 and
      logits.get("exact_bits") is True and
      logits.get("finite") is True and
      logits.get("top1_match") is True and
      float(logits.get("kld_reference_to_candidate", -1.0)) == 0.0 and
      states.get("all_exact_bits") is True and
      int(states.get("candidate_count", -1)) == 80 and
      states.get("mismatch_names") == [])


def profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  profile = [row for row in rows if row.get("metric_scope") == "profile"]
  workers: dict[str, dict[str, Any]] = {}
  for worker in ("stock", "candidate"):
    selected = [row for row in profile if row.get("worker") == worker]
    workers[worker] = {
        "rows": len(selected),
        "executed_rows": sum(
            row.get("status") == "Status.EXECUTED" for row in selected),
        "optimized_out_rows": sum(
            row.get("status") == "Status.OPTIMIZED_OUT" for row in selected),
        "real_time_us": sum(float(row.get("real_time_us", 0.0))
                            for row in selected),
        "node_types": sorted({str(row.get("node_type")) for row in selected}),
    }
  return {"profile_rows": len(profile), "workers": workers}


def codegen_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for kind in ("stock", "custom"):
    row = next((item for item in rows if item.get("kind") == kind), {})
    env = row.get("execution_env", {})
    result[kind] = {
        "simd_size": int(env.get("simd_size", -1)),
        "grf_count": int(env.get("grf_count", -1)),
        "eu_thread_count": int(env.get("eu_thread_count", -1)),
        "indirect_stateless_count": int(
            env.get("indirect_stateless_count", -1)),
        "spill_mem_size": int(env.get("spill_mem_size", -1)),
        "private_size": int(env.get("private_size", -1)),
    }
  return result


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      ROADMAP, STATUS, REJECTED, ACCEPTANCE,
      SEQ807_CORRECTNESS, SEQ807_NUMERIC_CORRECTNESS,
      SEQ807_NUMERIC_METRICS, SEQ807_CODEGEN_METRICS,
      SEQ1109_CORRECTNESS, SEQ1109_METRICS,
      SEQ1110_CORRECTNESS, SEQ1110_METRICS)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  acceptance = load_json(ACCEPTANCE)
  rejected = load_json(REJECTED)
  seq807 = load_json(SEQ807_CORRECTNESS)
  seq807_numeric = load_json(SEQ807_NUMERIC_CORRECTNESS)
  seq1109 = load_json(SEQ1109_CORRECTNESS)
  seq1110 = load_json(SEQ1110_CORRECTNESS)
  seq807_profile = profile_summary(load_jsonl(SEQ807_NUMERIC_METRICS))
  seq1109_profile = profile_summary(load_jsonl(SEQ1109_METRICS))
  seq1110_profile = profile_summary(load_jsonl(SEQ1110_METRICS))
  seq807_codegen = codegen_summary(load_jsonl(SEQ807_CODEGEN_METRICS))
  roadmap = ROADMAP.read_text(encoding="utf-8")
  status = STATUS.read_text(encoding="utf-8")
  sample_memory("after-stored-evidence", stop_bytes, memory)

  route = acceptance["candidate_runtime"]["first_prefill_route"]
  registered_envelope_ms = float(
      route["stock_profiled_transpose_plus_gated_delta_ms_per_1024"])
  priority_cuts = {
      str(key): float(value) for key, value in
      route["priority_end_to_end_cut_ms_per_1024"].items()}

  seq807_stock_ms = (
      seq807_profile["workers"]["stock"]["real_time_us"] / 1000.0)
  seq807_custom_ms = (
      seq807_profile["workers"]["candidate"]["real_time_us"] / 1000.0)
  seq1109_stock_ms = (
      seq1109_profile["workers"]["stock"]["real_time_us"] / 1000.0)
  seq1109_custom_ms = (
      seq1109_profile["workers"]["candidate"]["real_time_us"] / 1000.0)
  seq1110_stock_ms = (
      seq1110_profile["workers"]["stock"]["real_time_us"] / 1000.0)
  seq1110_custom_ms = (
      seq1110_profile["workers"]["candidate"]["real_time_us"] / 1000.0)

  # This deliberately optimistic subtraction crosses stored protocols. It is
  # useful only as a design ceiling: delete every registered Transpose and
  # retain the exact seq807 custom GDN cost. It is not promotion arithmetic.
  diagnostic_delete_all_transpose_saving_ms = (
      registered_envelope_ms - seq807_custom_ms)
  diagnostic_margin_by_context_ms = {
      context: diagnostic_delete_all_transpose_saving_ms - cut
      for context, cut in priority_cuts.items()}

  # Seq1110 is the faster of the two exact QKV-to-GDN implementations. Its
  # component time already exceeds the entire registered stock boundary, so
  # the implemented mechanism cannot admit a matching long profile.
  exact_best_fused_ms = min(seq1109_custom_ms, seq1110_custom_ms)
  exact_best_fused_variant = (
      "seq1109_direct" if seq1109_custom_ms < seq1110_custom_ms
      else "seq1110_tiled")
  exact_regression_vs_registered_envelope_ms = (
      exact_best_fused_ms - registered_envelope_ms)
  exact_implemented_saving_ms = (
      registered_envelope_ms - exact_best_fused_ms)

  rejected_routes = rejected.get("rejected", [])
  prior_rejection = next(
      (row for row in rejected_routes
       if row.get("route") ==
           "openvino_qkv_transpose_to_gdn_adjacent_fusion_v28a"), None)

  seq807_worker_shape = seq807_profile["workers"]
  seq1109_worker_shape = seq1109_profile["workers"]
  seq1110_worker_shape = seq1110_profile["workers"]
  dirty_only_expected = ["repository_clean_at_gate"]
  seq1109_dirty_only = failed_checks(seq1109) == dirty_only_expected
  seq1110_dirty_only = failed_checks(seq1110) == dirty_only_expected
  exact_route_regresses = exact_best_fused_ms > registered_envelope_ms

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("registered_priority_prefill_contract_is_exact",
            route.get("id") == "openvino_gated_delta_transpose_fusion" and
            abs(registered_envelope_ms - 149.931) < 1e-9 and
            priority_cuts == {
                "32768": 58.027, "65536": 77.947,
                "131072": 115.486} and
            route.get("must_combine_with_context_attention_cut") is True and
            route.get("component_profile_is_product_evidence") is False,
            registered_envelope_ms_per_1024=registered_envelope_ms,
            priority_cuts_ms_per_1024=priority_cuts),
      check("roadmap_registers_matching_envelope_cuts_and_claim_boundary",
            all(marker in roadmap for marker in (
                "149.931 ms", "58.027", "77.947", "115.486",
                "component attribution only", "end-to-end paired"))),
      check("status_selects_source_only_priority_prefill_gate",
            "priority prefill" in status and
            "Transpose+GDN boundary" in status and
            "Admit a long profile only if the bound is material" in status),
      check("seq807_clean_codegen_and_numeric_gates_pass",
            seq807.get("required_checks_passed") is True and
            failed_checks(seq807) == [] and
            seq807_numeric.get("required_checks_passed") is True and
            failed_checks(seq807_numeric) == []),
      check("seq807_fixed_index_codegen_is_exact",
            seq807_codegen == {
                "stock": {
                    "simd_size": 16, "grf_count": 128,
                    "eu_thread_count": 8,
                    "indirect_stateless_count": 0,
                    "spill_mem_size": 0, "private_size": 0},
                "custom": {
                    "simd_size": 16, "grf_count": 96,
                    "eu_thread_count": 10,
                    "indirect_stateless_count": 0,
                    "spill_mem_size": 0, "private_size": 0}},
            codegen=seq807_codegen),
      check("seq807_same_run_all30_profile_is_exact",
            seq807_worker_shape["stock"]["rows"] == 30 and
            seq807_worker_shape["stock"]["executed_rows"] == 30 and
            seq807_worker_shape["candidate"]["rows"] == 180 and
            seq807_worker_shape["candidate"]["executed_rows"] == 30 and
            seq807_worker_shape["candidate"]["optimized_out_rows"] == 150 and
            abs(seq807_stock_ms - 39.287) < 1e-9 and
            abs(seq807_custom_ms - 49.170) < 1e-9,
            profile=seq807_profile),
      check("seq1109_dirty_artifact_has_only_hygiene_failure",
            seq1109.get("required_checks_passed") is False and
            seq1109_dirty_only and all_nested_checks_pass(seq1109) and
            exact_numeric_evidence(seq1109),
            failed_checks=failed_checks(seq1109)),
      check("seq1110_dirty_artifact_has_only_hygiene_failure",
            seq1110.get("required_checks_passed") is False and
            seq1110_dirty_only and all_nested_checks_pass(seq1110) and
            exact_numeric_evidence(seq1110),
            failed_checks=failed_checks(seq1110)),
      check("exact_direct_and_tiled_profiles_are_all30",
            seq1109_worker_shape["stock"]["rows"] == 30 and
            seq1109_worker_shape["candidate"]["rows"] == 180 and
            seq1110_worker_shape["stock"]["rows"] == 30 and
            seq1110_worker_shape["candidate"]["rows"] == 180 and
            abs(seq1109_stock_ms - 39.422) < 1e-9 and
            abs(seq1109_custom_ms - 253.598) < 1e-9 and
            abs(seq1110_stock_ms - 39.507) < 1e-9 and
            abs(seq1110_custom_ms - 212.569) < 1e-9,
            seq1109_profile=seq1109_profile,
            seq1110_profile=seq1110_profile),
      check("prior_exact_adjacent_route_rejection_is_registered",
            prior_rejection is not None and
            prior_rejection.get("reopen_condition") == (
                "none for QKV-to-GDN-only direct/tiled variants; require a "
                "broader provider-aware FC-to-conv/GDN boundary with a "
                "complete timing bound"),
            prior_rejection=prior_rejection),
      check("cross_protocol_optimism_cannot_cover_all_priority_rows",
            abs(diagnostic_delete_all_transpose_saving_ms - 100.761) < 1e-9 and
            diagnostic_margin_by_context_ms["32768"] > 0.0 and
            diagnostic_margin_by_context_ms["65536"] > 0.0 and
            abs(diagnostic_margin_by_context_ms["131072"] + 14.725) < 1e-9,
            diagnostic_saving_ms_per_1024=
                diagnostic_delete_all_transpose_saving_ms,
            margin_by_context_ms_per_1024=diagnostic_margin_by_context_ms),
      check("best_exact_adjacent_fusion_regresses_vs_entire_envelope",
            exact_route_regresses and
            exact_best_fused_variant == "seq1110_tiled" and
            abs(exact_regression_vs_registered_envelope_ms - 62.638) < 1e-9,
            exact_best_fused_variant=exact_best_fused_variant,
            exact_best_fused_ms_per_1024=exact_best_fused_ms,
            registered_envelope_ms_per_1024=registered_envelope_ms,
            regression_ms_per_1024=exact_regression_vs_registered_envelope_ms),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_fundable = not exact_route_regresses
  verdict = (
      "reject_prefill_transpose_gdn_adjacent_fusion_before_long_profile"
      if required_checks_passed and not route_fundable else
      "admit_one_matching_long_prefill_profile"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": required_checks_passed and route_fundable,
      "source_edit_admitted": False,
      "compile_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": required_checks_passed and route_fundable,
      "registered_contract": {
          "stock_transpose_plus_gdn_ms_per_1024": registered_envelope_ms,
          "priority_end_to_end_cuts_ms_per_1024": priority_cuts,
          "must_combine_with_context_attention_cut": True,
          "component_profile_is_product_evidence": False,
      },
      "seq807_exact_fixed_index_carrier": {
          "stock_gdn_ms_per_1024": seq807_stock_ms,
          "custom_gdn_ms_per_1024": seq807_custom_ms,
          "profile": seq807_profile,
          "codegen": seq807_codegen,
      },
      "exact_adjacent_implementations": {
          "seq1109_direct": {
              "stock_gdn_ms_per_1024": seq1109_stock_ms,
              "custom_boundary_ms_per_1024": seq1109_custom_ms,
              "stored_artifact_clean": False,
              "non_hygiene_required_checks_pass": seq1109_dirty_only,
          },
          "seq1110_tiled": {
              "stock_gdn_ms_per_1024": seq1110_stock_ms,
              "custom_boundary_ms_per_1024": seq1110_custom_ms,
              "stored_artifact_clean": False,
              "non_hygiene_required_checks_pass": seq1110_dirty_only,
          },
          "best_variant": exact_best_fused_variant,
          "best_ms_per_1024": exact_best_fused_ms,
          "implemented_saving_ms_per_1024": exact_implemented_saving_ms,
          "regression_vs_registered_envelope_ms_per_1024":
              exact_regression_vs_registered_envelope_ms,
      },
      "diagnostic_cross_protocol_ceiling": {
          "label": "design_optimism_only_not_admission_or_product_evidence",
          "assumption": (
              "delete the entire registered Transpose share while retaining "
              "the exact seq807 custom GDN cost"),
          "saving_ms_per_1024": diagnostic_delete_all_transpose_saving_ms,
          "margin_by_context_ms_per_1024": diagnostic_margin_by_context_ms,
      },
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Priority-prefill Transpose+GDN boundary

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The registered component envelope is `{registered_envelope_ms:.3f} ms/1024`
for stock Transpose+GDN. The product-row cuts are
`{priority_cuts['32768']:.3f}`, `{priority_cuts['65536']:.3f}`, and
`{priority_cuts['131072']:.3f} ms/1024` at 32k/64k/128k. The registered
profile is component attribution, not product evidence, and the route must
also combine with a context-attention cut.

Clean seq807 is exact and spill-free: stock GDN is `{seq807_stock_ms:.3f}`
and fixed-index custom GDN is `{seq807_custom_ms:.3f} ms/1024`. Deleting every
registered Transpose while retaining seq807 therefore gives a deliberately
cross-protocol design optimism of
`{diagnostic_delete_all_transpose_saving_ms:.3f} ms/1024`. It clears the 32k
and 64k cuts on paper but misses the 128k cut by
`{-diagnostic_margin_by_context_ms['131072']:.3f} ms/1024`; it cannot admit a
long profile.

The exact implemented adjacent boundary is decisive. Seq1109 direct and
seq1110 tiled are component-, logits-, and all-80-state exact apart from their
recorded repository-hygiene failure. They cost `{seq1109_custom_ms:.3f}` and
`{seq1110_custom_ms:.3f} ms/1024`. The faster tiled implementation is already
`{exact_regression_vs_registered_envelope_ms:.3f} ms/1024` slower than the
entire registered stock Transpose+GDN envelope. Source, compile, matching long
profile, ABBA, and product rows are not admitted for this mechanism.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "diagnostic_cross_protocol_saving_ms":
          diagnostic_delete_all_transpose_saving_ms,
      "diagnostic_128k_shortfall_ms":
          -diagnostic_margin_by_context_ms["131072"],
      "exact_best_fused_ms": exact_best_fused_ms,
      "exact_regression_vs_envelope_ms":
          exact_regression_vs_registered_envelope_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
