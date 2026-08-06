#!/usr/bin/env python3
"""Bound the long-prefill context-attention route before a GPU probe.

This gate joins the exact current tiled-attention source to the stored clean
32k all-ten dispatch trace.  It projects only the source-mandated causal tile
count, derives the arithmetic rate needed to clear each registered priority
cut, and admits at most one bounded KQ+PV arithmetic-roofline component.  It
does not compile, launch a GPU worker, or treat component timing as product
evidence.
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
SCHEMA = "intel-qwen36-openvino-prefill-context-attention-bound-v0"

ACCEPTANCE = REPO / "benchmarks" / WS / "acceptance-matrix.json"
STATUS = REPO / "doc/active" / WS / "STATUS.md"
ROUTES = REPO / "doc/active" / WS / "routes-ledger.json"
PROFILE_METRICS = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1105-"
    "all10-32k-cleanZ/metrics.json")
PROFILE_WORKER = PROFILE_METRICS.parent / (
    "raw/32k/candidate/worker-result.json")
HELPERS = REPO / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"
PREFILL_SOURCE = REPO / "engine/openvino/custom/iq36_prefill_attention_tiled.cl"
MICROKERNEL_SHIMS = REPO / (
    "engine/openvino/custom/iq36_prefill_microkernel_shims.cl")
CUSTOM_XML = REPO / "engine/openvino/custom/iq36_hot_attention_gqa.xml"

EXPECTED_CUTS = {32768: 58.027, 65536: 77.947, 131072: 115.486}
PROFILE_COMMIT = "9097a83bb553356ac27502974653e514e3a630e9"
PROFILE_CONTEXT = 32768
PROFILE_ATTENTION_MS = 4353.818423
RESIDENT_CHUNK = 8192
LAYERS = 10
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
QUERY_TILE = 32
KEY_TILE = 128
MACS_PER_PAIR = LAYERS * Q_HEADS * HEAD_DIM * 2


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


def named_check(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  matches = [row for row in metrics.get("checks", [])
             if isinstance(row, dict) and row.get("name") == name]
  if len(matches) != 1:
    raise ValueError(f"expected one stored check {name!r}, got {len(matches)}")
  return matches[0]


def exact_chunks(rows: Any) -> bool:
  if not isinstance(rows, list) or len(rows) != 4:
    return False
  observed = [(int(row.get("start", -1)),
               int(row.get("end_exclusive", -1)))
              for row in rows if isinstance(row, dict)]
  expected = [(index * RESIDENT_CHUNK, (index + 1) * RESIDENT_CHUNK)
              for index in range(4)]
  return observed == expected


def padded_causal_pairs(context: int) -> int:
  if context % KEY_TILE != 0 or KEY_TILE % QUERY_TILE != 0:
    raise ValueError("context and tile geometry are not aligned")
  key_blocks = context // KEY_TILE
  query_groups_per_key_block = KEY_TILE // QUERY_TILE
  return (QUERY_TILE * KEY_TILE * query_groups_per_key_block *
          key_blocks * (key_blocks + 1) // 2)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      ACCEPTANCE, STATUS, ROUTES, PROFILE_METRICS, PROFILE_WORKER, HELPERS,
      PREFILL_SOURCE, MICROKERNEL_SHIMS, CUSTOM_XML)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  acceptance = load_json(ACCEPTANCE)
  routes = load_json(ROUTES)
  metrics = load_json(PROFILE_METRICS)
  worker = load_json(PROFILE_WORKER)
  status = STATUS.read_text(encoding="utf-8")
  helpers = HELPERS.read_text(encoding="utf-8")
  prefill = PREFILL_SOURCE.read_text(encoding="utf-8")
  shims = MICROKERNEL_SHIMS.read_text(encoding="utf-8")
  xml = CUSTOM_XML.read_text(encoding="utf-8")
  sample_memory("after-stored-evidence", stop_bytes, memory)

  route_contract = acceptance["candidate_runtime"]["first_prefill_route"]
  cuts = {int(key): float(value) for key, value in
          route_contract["priority_end_to_end_cut_ms_per_1024"].items()}
  trace_check = named_check(
      metrics, "32k_candidate_has_ten_timed_dispatches_per_phase")
  finite_check = named_check(
      metrics, "32k_candidate_all_phase_logits_finite")
  trace_phase = trace_check.get("phases", [{}])[0]
  finite_phase = finite_check.get("phases", [{}])[0]

  profile_rows = [row for row in worker.get("profile", [])
                  if isinstance(row, dict) and
                  row.get("node_type") == "IQ36HotAttentionGQA" and
                  row.get("status") == "Status.EXECUTED"]
  profile_us = sum(float(row.get("real_time_us", 0.0))
                   for row in profile_rows)
  source_summary = worker.get("source_summary", {})
  phase_zero = worker.get("phases", [{}])[0]

  source_markers = {
      "head_dim_256": "#define IQ36_HEAD_DIM 256U" in helpers,
      "q_heads_16": "#define IQ36_Q_HEADS 16U" in helpers,
      "kv_heads_2": "#define IQ36_KV_HEADS 2U" in helpers,
      "key_chunk_128": "#define IQ36_PREFILL_CHUNK_TOKENS 128U" in helpers,
      "query_tile_32": "#define IQ36_PREFILL_QUERY_TILE 32U" in helpers,
      "full_history": "#define IQ36_PREFILL_FULL_HISTORY 1" in helpers,
      "kq_call": "iq36_micro_score_tile score = ugemm_kq(" in prefill,
      "pv_call": "iq36_micro_output_tile chunk_output = ugemm_vs(" in prefill,
      "causal_chunk_loop": (
          "chunk_begin < causal_tokens" in prefill and
          "chunk_begin += IQ36_PREFILL_CHUNK_TOKENS" in prefill),
      "kq_128x32_systolic": all(marker in shims for marker in (
          "#define ugemm_kq_wg_tile_m 128",
          "#define ugemm_kq_wg_tile_n 32",
          "#define ugemm_kq_systolic  1")),
      "pv_256x32_systolic": all(marker in shims for marker in (
          "#define ugemm_vs_wg_tile_m 256",
          "#define ugemm_vs_wg_tile_n 32",
          "#define ugemm_vs_systolic  1")),
      "xml_enables_extracted_microkernels": (
          "iq36_prefill_microkernel_shims.cl" in xml and
          "-DIQ36_PREFILL_USE_MICROKERNEL=1" in xml),
  }

  macs_32k = padded_causal_pairs(PROFILE_CONTEXT) * MACS_PER_PAIR
  observed_tmac_s = macs_32k / (PROFILE_ATTENTION_MS / 1000.0) / 1e12
  rows: dict[str, dict[str, Any]] = {}
  for context, cut in sorted(cuts.items()):
    padded_pairs = padded_causal_pairs(context)
    logical_pairs = context * (context + 1) // 2
    macs = padded_pairs * MACS_PER_PAIR
    projected_total_ms = PROFILE_ATTENTION_MS * macs / macs_32k
    units = context / 1024.0
    current_ms_per_1024 = projected_total_ms / units
    target_ms_per_1024 = current_ms_per_1024 - cut
    target_total_ms = target_ms_per_1024 * units
    required_tmac_s = (
        macs / (target_total_ms / 1000.0) / 1e12
        if target_total_ms > 0.0 else float("inf"))
    rows[str(context)] = {
        "registered_cut_ms_per_1024": cut,
        "logical_causal_pairs_per_q_head": logical_pairs,
        "current_tile_padded_pairs_per_q_head": padded_pairs,
        "padding_ratio": padded_pairs / logical_pairs,
        "current_tile_kq_plus_pv_macs_all_ten": macs,
        "projected_current_attention_total_ms": projected_total_ms,
        "projected_current_attention_ms_per_1024": current_ms_per_1024,
        "delete_all_margin_ms_per_1024": current_ms_per_1024 - cut,
        "target_attention_ms_per_1024": target_ms_per_1024,
        "required_current_tile_arithmetic_tmac_s": required_tmac_s,
        "required_rate_over_observed": required_tmac_s / observed_tmac_s,
    }

  tightest_context = max(
      rows, key=lambda key: rows[key]["required_current_tile_arithmetic_tmac_s"])
  required_rate = rows[tightest_context][
      "required_current_tile_arithmetic_tmac_s"]
  delete_all_clears = all(
      row["delete_all_margin_ms_per_1024"] >= 0.0
      for row in rows.values())

  parked_context = next(
      (row for row in routes.get("parked_routes", [])
       if row.get("rank") == 2), None)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("registered_priority_prefill_contract_is_exact",
            cuts == EXPECTED_CUTS and
            route_contract.get("must_combine_with_context_attention_cut")
            is True and
            route_contract.get("component_profile_is_product_evidence")
            is False,
            cuts=cuts),
      check("context_attention_route_is_registered_after_ov3",
            isinstance(parked_context, dict) and
            parked_context.get("id") ==
            "openvino_long_prefill_context_attention_complete_bound" and
            routes.get("active_route", {}).get("id") ==
            "openvino_direct_i8_attention_all_ten_correctness"),
      check("stored_32k_attention_profile_is_clean_and_admitted",
            metrics.get("git") == {
                "commit": PROFILE_COMMIT, "dirty": False,
                "dirty_paths": []} and
            metrics.get("attribution_checks_passed") is True and
            metrics.get("carrier_admission_passed") is True,
            profile_git=metrics.get("git")),
      check("stored_32k_trace_shape_and_time_are_exact",
            trace_check.get("pass") is True and
            trace_phase.get("input_tokens") == PROFILE_CONTEXT and
            trace_phase.get("total_tokens") == PROFILE_CONTEXT and
            trace_phase.get("dispatch_count") == 40 and
            float(trace_phase.get("duration_total_ms", -1.0)) ==
            PROFILE_ATTENTION_MS and
            trace_phase.get("kernels") == [
                "iq36_hot_attention_single_owner"] and
            trace_phase.get("global_sizes") == [[128, 256, 16]] and
            trace_phase.get("local_sizes") == [[128, 1, 1]] and
            exact_chunks(trace_phase.get("prefill_chunks")),
            phase=trace_phase),
      check("stored_32k_trace_is_numerically_live",
            finite_check.get("pass") is True and
            finite_phase.get("logits_finite") is True and
            finite_phase.get("top1") == 271),
      check("worker_binds_exact_all_ten_resident_carrier",
            source_summary.get("custom_count_after") == LAYERS and
            source_summary.get("prefill_query_tile") == QUERY_TILE and
            source_summary.get("prefill_history_capacity") ==
            PROFILE_CONTEXT and
            len(source_summary.get("target_layers", [])) == LAYERS and
            exact_chunks(phase_zero.get("prefill_chunks")) and
            len(profile_rows) == LAYERS and profile_us == 730639.0,
            executed_attention_rows=len(profile_rows),
            last_chunk_profile_us=profile_us),
      check("current_source_geometry_and_microkernels_are_exact",
            all(source_markers.values()), markers=source_markers),
      check("delete_all_attention_ceiling_clears_every_priority_cut",
            delete_all_clears, rows=rows),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  component_admitted = required_checks_passed and delete_all_clears
  verdict = (
      "admit_one_bounded_current_microkernel_arithmetic_roofline_component"
      if component_admitted else
      "reject_context_attention_before_compile_or_long_worker"
      if required_checks_passed else "inconclusive")

  component_contract = {
      "purpose": (
          "optimistic KQ+PV-only roofline for the exact extracted current "
          "128x32 and 256x32 systolic microkernels"),
      "single_source_spelling": True,
      "tile_or_workgroup_sweep_allowed": False,
      "context_buffer_allocation_required": False,
      "minimum_one_sided_95_lower_tmac_s": required_rate,
      "equivalent_maximum_one_sided_95_latency_bound": (
          "derive from the component's fixed registered MAC count"),
      "warmup_samples": 20,
      "measured_samples": 20,
      "build_parallelism": 1,
      "serial_gpu_worker": True,
      "memory_stop_bytes": stop_bytes,
      "component_profile_is_product_evidence": False,
      "passing_action": (
          "admit one exact online-softmax/context-state carrier component; "
          "do not admit graph integration or a long worker"),
      "failing_action": (
          "close the current extracted-microkernel family before carrier "
          "source or a long worker"),
  }
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "route_fundable": delete_all_clears,
      "source_edit_admitted": component_admitted,
      "compile_admitted": component_admitted,
      "gpu_worker_admitted": component_admitted,
      "long_worker_admitted": False,
      "graph_integration_admitted": False,
      "gpu_worker_launched": False,
      "observed_32k": {
          "attention_ms_all_ten": PROFILE_ATTENTION_MS,
          "effective_current_tile_tmac_s": observed_tmac_s,
          "dispatch_count": 40,
          "resident_chunk_tokens": RESIDENT_CHUNK,
      },
      "projection": {
          "assumption": (
              "hold the clean 32k effective current-tile arithmetic rate "
              "constant and scale only exact source-derived causal tile MACs"),
          "mac_definition": (
              "one QK plus one PV MAC for every current padded causal pair, "
              "all 10 attention layers and 16 query heads"),
          "rows": rows,
          "tightest_context": int(tightest_context),
          "tightest_required_tmac_s": required_rate,
      },
      "component_contract": component_contract,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")

  ratio = rows[tightest_context]["required_rate_over_observed"]
  summary = f"""# Long-prefill context-attention complete bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The clean all-ten 32k trace executes exactly 40 resident-chunk attention
dispatches in `{PROFILE_ATTENTION_MS:.6f} ms`.  The exact current 32-query by
128-key KQ and 256-value PV tiles therefore sustain an effective
`{observed_tmac_s:.6f} TMAC/s`, including softmax, state, barriers, launch, and
output overhead.  Source-derived causal-tile scaling gives attention ceilings
of `{rows['32768']['projected_current_attention_ms_per_1024']:.6f}`,
`{rows['65536']['projected_current_attention_ms_per_1024']:.6f}`, and
`{rows['131072']['projected_current_attention_ms_per_1024']:.6f} ms/1024`.
Deleting attention entirely clears every registered priority cut.

The tightest row is `{tightest_context}`: after its
`{rows[tightest_context]['registered_cut_ms_per_1024']:.3f} ms/1024` cut, all
current padded KQ+PV arithmetic must fit at at least `{required_rate:.6f}
TMAC/s`, `{ratio:.6f}x` the observed effective rate.  Because this remains an
attainability question, admit exactly one bounded KQ+PV-only roofline using the
already extracted systolic microkernels.  It has one source spelling, no tile
or workgroup sweep, 20 warmups plus 20 measured samples, serial execution,
`-j1`, and the 4-GiB stop.  Passing does not admit graph integration or a long
worker; failure closes this exact microkernel family before either.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "observed_tmac_s": observed_tmac_s,
      "tightest_context": int(tightest_context),
      "required_tmac_s": required_rate,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
