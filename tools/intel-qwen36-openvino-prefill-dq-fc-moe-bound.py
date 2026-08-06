#!/usr/bin/env python3
"""Close or admit the OV3 long-prefill DQ/compressed-FC/MoE route.

This is a source/evidence-only gate.  It never compiles or launches a GPU
worker.  The exact current graph census is joined to the stored clean 32k
resident-chunk profile, then the entire DynamicQuantize, compressed-FC, and
fused-MoE envelope is made free.  That deliberately favorable ceiling must
clear every registered priority-row cut before implementation is admissible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-prefill-dq-fc-moe-bound-v0"

ACCEPTANCE = REPO / "benchmarks" / WS / "acceptance-matrix.json"
STATUS = REPO / "doc/active" / WS / "STATUS.md"
ROADMAP = REPO / "doc/active" / WS / (
    f"{WS}-openvino-specialization-roadmap-2026-07-13.md")
ROUTES = REPO / "doc/active" / WS / "routes-ledger.json"
ACCEPTED = REPO / "doc/active" / WS / "accepted-cuts.json"
OPENCL_METRICS = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1136-"
    "dq-subgroup-32k-warm17-cleanZ/metrics.json")
OPENCL_PROFILE = OPENCL_METRICS.parent / (
    "raw/32k/candidate/worker-result.json")
LEVEL_ZERO_METRICS = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1172-"
    "l0-dq-restored-32k-warm17-cleanZ/metrics.json")
LEVEL_ZERO_PROFILE = LEVEL_ZERO_METRICS.parent / (
    "raw/32k/candidate/worker-result.json")
CURRENT_PROFILE = REPO / (
    "output/openvino-accepted-carrier-profile-refresh-20260715Tseq1240-"
    "2k-warm17-cleanZ/metrics.json")
SHARED_DQ_PATCH = REPO / "engine/openvino/iq36-shared-dynamic-quantize.patch"
SUBGROUP_DQ_PATCH = REPO / (
    "engine/openvino/iq36-dynamic-quantize-subgroup64.patch")

ROUTE_TYPES = (
    "DynamicQuantize", "FullyConnectedCompressed",
    "MOE3GemmFusedCompressed")
EXPECTED_OPENCL_COUNTS = {
    "DynamicQuantize": 161,
    "FullyConnectedCompressed": 371,
    "MOE3GemmFusedCompressed": 40,
}
EXPECTED_LEVEL_ZERO_EXECUTED_COUNTS = {
    "DynamicQuantize": 0,
    "FullyConnectedCompressed": 371,
    "MOE3GemmFusedCompressed": 40,
}
EXPECTED_LEVEL_ZERO_OPTIMIZED_COUNTS = {
    "DynamicQuantize": 161,
}
EXPECTED_OPENCL_US = {
    "DynamicQuantize": 78_941.0,
    "FullyConnectedCompressed": 603_222.0,
    "MOE3GemmFusedCompressed": 139_459.0,
}
EXPECTED_LEVEL_ZERO_US = {
    "FullyConnectedCompressed": 497_301.0,
    "MOE3GemmFusedCompressed": 110_884.0,
}
PREFILL_CHUNK_TOKENS = 8192


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


def phase_zero(value: dict[str, Any]) -> dict[str, Any]:
  phases = value.get("phases", [])
  if not phases:
    raise ValueError("stored worker result has no phases")
  phase = phases[0]
  if not isinstance(phase, dict):
    raise TypeError("stored phase zero is not an object")
  return phase


def profile_summary(value: dict[str, Any]) -> dict[str, Any]:
  phase = phase_zero(value)
  rows = phase.get("full_profile", [])
  if not isinstance(rows, list):
    raise TypeError("phase-zero full_profile is not a list")
  executed = [row for row in rows
              if row.get("status") == "Status.EXECUTED"]
  optimized = [row for row in rows
               if row.get("status") == "Status.OPTIMIZED_OUT"]
  executed_counts = Counter(str(row.get("node_type", ""))
                            for row in executed)
  optimized_counts = Counter(str(row.get("node_type", ""))
                             for row in optimized)
  executed_us: defaultdict[str, float] = defaultdict(float)
  for row in executed:
    executed_us[str(row.get("node_type", ""))] += float(
        row.get("real_time_us", 0.0))
  route_counts = {kind: int(executed_counts[kind]) for kind in ROUTE_TYPES}
  route_us = {kind: float(executed_us[kind]) for kind in ROUTE_TYPES}
  chunks = phase.get("prefill_chunks", [])
  return {
      "input_tokens": int(phase.get("input_tokens", -1)),
      "total_tokens": int(phase.get("total_tokens", -1)),
      "prefill_chunks": chunks,
      "profile_rows": len(rows),
      "executed_rows": len(executed),
      "route_executed_counts": route_counts,
      "route_optimized_counts": {
          kind: int(optimized_counts[kind]) for kind in ROUTE_TYPES},
      "route_executed_us": route_us,
  }


def exact_chunks(rows: Any) -> bool:
  if not isinstance(rows, list) or len(rows) != 4:
    return False
  expected = [
      {"start": index * PREFILL_CHUNK_TOKENS,
       "end_exclusive": (index + 1) * PREFILL_CHUNK_TOKENS}
      for index in range(4)]
  observed = [
      {"start": int(row.get("start", -1)),
       "end_exclusive": int(row.get("end_exclusive", -1))}
      for row in rows if isinstance(row, dict)]
  return observed == expected


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      ACCEPTANCE, STATUS, ROADMAP, ROUTES, ACCEPTED, OPENCL_METRICS,
      OPENCL_PROFILE, LEVEL_ZERO_METRICS, LEVEL_ZERO_PROFILE,
      CURRENT_PROFILE, SHARED_DQ_PATCH, SUBGROUP_DQ_PATCH)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  acceptance = load_json(ACCEPTANCE)
  routes = load_json(ROUTES)
  accepted = load_json(ACCEPTED)
  opencl_metrics = load_json(OPENCL_METRICS)
  level_zero_metrics = load_json(LEVEL_ZERO_METRICS)
  current = load_json(CURRENT_PROFILE)
  opencl = profile_summary(load_json(OPENCL_PROFILE))
  level_zero = profile_summary(load_json(LEVEL_ZERO_PROFILE))
  status = STATUS.read_text(encoding="utf-8")
  roadmap = ROADMAP.read_text(encoding="utf-8")
  shared_patch = SHARED_DQ_PATCH.read_text(encoding="utf-8")
  subgroup_patch = SUBGROUP_DQ_PATCH.read_text(encoding="utf-8")
  sample_memory("after-stored-evidence", stop_bytes, memory)

  route_contract = acceptance["candidate_runtime"]["first_prefill_route"]
  priority_cuts = {
      str(key): float(value) for key, value in
      route_contract["priority_end_to_end_cut_ms_per_1024"].items()}

  # Profiling was captured after the fourth resident 8k infer.  These three
  # node families consume the current chunk, not prior KV history, so the
  # exact 8k envelope normalizes directly to the contract's ms/1024 unit.
  opencl_ms_per_1024 = {
      kind: opencl["route_executed_us"][kind] / 1000.0 /
      (PREFILL_CHUNK_TOKENS / 1024.0)
      for kind in ROUTE_TYPES}
  gross_delete_all_ms = sum(opencl_ms_per_1024.values())
  gross_margins = {
      context: gross_delete_all_ms - cut
      for context, cut in priority_cuts.items()}

  # The stored Level Zero row optimizes graph DQ out.  Keep this as the exact
  # visible-materialization envelope, while the larger OpenCL sum above grants
  # the route the DQ work anyway.  The verdict uses the larger value.
  level_zero_ms_per_1024 = {
      kind: level_zero["route_executed_us"][kind] / 1000.0 /
      (PREFILL_CHUNK_TOKENS / 1024.0)
      for kind in ROUTE_TYPES}
  visible_delete_all_ms = sum(level_zero_ms_per_1024.values())
  visible_margins = {
      context: visible_delete_all_ms - cut
      for context, cut in priority_cuts.items()}

  current_counts = current.get("profile_audit", {}).get(
      "executed_counts", {})
  current_route_counts = {
      kind: int(current_counts.get(kind, 0)) for kind in ROUTE_TYPES}
  shared_cut = next(
      (row for row in accepted.get("accepted", [])
       if row.get("id") ==
       "openvino_identical_activation_shared_dynamic_quantize"), None)
  parked = routes.get("parked_routes", [])
  parked_rank_one = next(
      (row for row in parked if row.get("rank") == 1), None)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("registered_priority_prefill_contract_is_exact",
            priority_cuts == {
                "32768": 58.027, "65536": 77.947,
                "131072": 115.486} and
            route_contract.get("must_combine_with_context_attention_cut")
            is True and
            route_contract.get("component_profile_is_product_evidence")
            is False,
            priority_cuts_ms_per_1024=priority_cuts),
      check("route_is_registered_rank_one_after_direct_i8",
            isinstance(parked_rank_one, dict) and
            parked_rank_one.get("id") ==
            "openvino_long_prefill_dq_fc_moe_complete_bound" and
            routes.get("active_route", {}).get("id") ==
            "openvino_direct_i8_attention_all_ten_correctness"),
      check("shared_group64_quantization_is_already_accepted",
            shared_cut is not None and
            "371->161" in str(shared_cut.get("note", "")) and
            "SharedDynamicQuantize" in shared_patch and
            "params.group_sizes.back() == 64" in subgroup_patch),
      check("stored_32k_profiles_are_clean",
            opencl_metrics.get("git", {}).get("dirty") is False and
            level_zero_metrics.get("git", {}).get("dirty") is False,
            opencl_git=opencl_metrics.get("git"),
            level_zero_git=level_zero_metrics.get("git")),
      check("stored_profiles_use_exact_resident_8k_chunk_shape",
            opencl["input_tokens"] == 32768 and
            opencl["total_tokens"] == 32768 and
            exact_chunks(opencl["prefill_chunks"]) and
            level_zero["input_tokens"] == 32768 and
            level_zero["total_tokens"] == 32768 and
            exact_chunks(level_zero["prefill_chunks"])),
      check("opencl_complete_route_census_is_exact",
            opencl["route_executed_counts"] == EXPECTED_OPENCL_COUNTS and
            opencl["route_executed_us"] == EXPECTED_OPENCL_US,
            profile=opencl),
      check("level_zero_visible_materialization_census_is_exact",
            level_zero["route_executed_counts"] ==
            EXPECTED_LEVEL_ZERO_EXECUTED_COUNTS and
            {"DynamicQuantize": level_zero[
                "route_optimized_counts"]["DynamicQuantize"]} ==
            EXPECTED_LEVEL_ZERO_OPTIMIZED_COUNTS and
            {kind: level_zero["route_executed_us"][kind]
             for kind in EXPECTED_LEVEL_ZERO_US} == EXPECTED_LEVEL_ZERO_US,
            profile=level_zero),
      check("refreshed_current_graph_census_matches_route",
            current.get("required_checks_passed") is True and
            current.get("profile_audit", {}).get(
                "raw_profile_time_is_savings_evidence") is False and
            current_route_counts == EXPECTED_OPENCL_COUNTS,
            current_route_counts=current_route_counts),
      check("roadmap_and_status_bind_current_gate",
            "## OV3" in roadmap and
            "DynamicQuantize with compressed FC" in roadmap and
            "all-ten fixed-state direct-I8 short correctness" in status),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  gross_clears_all = all(margin >= 0.0 for margin in gross_margins.values())
  route_fundable = required_checks_passed and gross_clears_all
  verdict = (
      "admit_one_ov3_source_component"
      if route_fundable else
      "reject_standalone_dq_fc_moe_before_source_or_long_profile"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "route_fundable": route_fundable,
      "source_edit_admitted": route_fundable,
      "compile_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "registered_contract": {
          "priority_end_to_end_cuts_ms_per_1024": priority_cuts,
          "must_combine_with_context_attention_cut": True,
          "component_profile_is_product_evidence": False,
      },
      "matching_census": {
          "current_graph_route_counts": current_route_counts,
          "opencl_32k_phase_zero": opencl,
          "level_zero_32k_phase_zero": level_zero,
      },
      "gross_delete_all_ceiling": {
          "label": (
              "favorable_component_telemetry_not_product_or_additive_"
              "provider_evidence"),
          "assumption": (
              "make every current shared DynamicQuantize, all 371 compressed "
              "FC nodes, and all 40 fused MoE nodes free"),
          "ms_per_1024_by_type": opencl_ms_per_1024,
          "saving_ms_per_1024": gross_delete_all_ms,
          "margin_by_context_ms_per_1024": gross_margins,
          "clears_all_priority_rows": gross_clears_all,
      },
      "level_zero_visible_delete_all_ceiling": {
          "ms_per_1024_by_type": level_zero_ms_per_1024,
          "saving_ms_per_1024": visible_delete_all_ms,
          "margin_by_context_ms_per_1024": visible_margins,
      },
      "required_context_attention_complement_ms_per_1024": max(
          0.0, -gross_margins["131072"]),
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

  summary = f"""# OV3 prefill DQ/compressed-FC/MoE complete bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The current graph census is exactly `161` shared DynamicQuantize, `371`
FullyConnectedCompressed, and `40` MOE3GemmFusedCompressed nodes.  The clean
32k resident worker exposes the last 8k chunk, so its context-independent
envelope normalizes to `{gross_delete_all_ms:.6f} ms/1024` even after making
all three node families free.  This is deliberately favorable component
telemetry, not additive provider timing or product evidence.

The registered priority-row cuts are `{priority_cuts['32768']:.3f}`,
`{priority_cuts['65536']:.3f}`, and `{priority_cuts['131072']:.3f} ms/1024`.
The delete-all ceiling clears 32k and 64k on paper but misses 128k by
`{-gross_margins['131072']:.6f} ms/1024`.  The matching Level Zero visible
materialization envelope is smaller at `{visible_delete_all_ms:.6f} ms/1024`;
DynamicQuantize is optimized out there.  OV3 is therefore not independently
fundable and cannot admit source, compile, or a long profile.  Any future use
must be bundled behind a context-attention bound that first covers at least
`{-gross_margins['131072']:.6f} ms/1024` at 128k.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "gross_delete_all_ms_per_1024": gross_delete_all_ms,
      "gross_128k_shortfall_ms_per_1024": -gross_margins["131072"],
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
