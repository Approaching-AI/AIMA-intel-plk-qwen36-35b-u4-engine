#!/usr/bin/env python3
"""Close or admit OV4 exact-bucket specialization without launching a worker.

The accepted 32k carrier already contains several OV4 mechanisms.  This gate
audits those mechanisms from a clean product artifact and the current source,
then forms an intentionally optimistic union bound from:

* eliminating the complete first-decode transition penalty; and
* eliminating the entire registered Level Zero setup/append envelope.

If that union still misses the current kill-number, compiling seven variants
or running another long row cannot be justified by exact-bucket work alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_CARRIER = (
    REPO / "output" /
    "openvino-hot-cold-product-20260715Tseq1204-"
    "alias-fused-linear-state-32k-o64-cleanZ"
)
PRODUCT_GATE = REPO / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
GRAPH_BUILDER = REPO / "tools/intel_qwen36_openvino_hot_cold_attention.py"
REJECTED = REPO / "doc/active" / WS / "rejected-routes.json"
PREPARE_ROUTE = "openvino_level_zero_prepare_and_assign_microcuts_v28f"
COLD_I8_PHASE = (
    REPO / "output" /
    "openvino-attention-phase-profile-20260715Tseq1087-"
    "prefix9-unified-fixedsignature-nobind-32k-dirtyZ" / "metrics.json"
)
DENSE_F16_PHASE = (
    REPO / "output" /
    "openvino-attention-phase-profile-20260715Tseq1089-"
    "prefix9-densefirstdecode-32k-dirtyZ" / "metrics.json"
)
BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
HOT_WINDOW = 8192
PREFILL_CHUNK = 8192
MIN_RING_CAPACITY = 2 * PREFILL_CHUNK
KV_HEADS = 2
HEAD_DIM = 256
SCALE_BYTES = 16
KEY_TILE = 16


def load_json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"expected JSON object: {path}")
  return payload


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(*args: str) -> str:
  return subprocess.run(
      ["git", *args], cwd=REPO, check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def find_candidate_result(carrier: Path) -> Path:
  matches = sorted(carrier.glob("raw/*/correctness/candidate/worker-result.json"))
  if len(matches) != 1:
    raise ValueError(
        f"expected one candidate worker result under {carrier}, got {len(matches)}")
  return matches[0]


def custom_state_bytes(bucket: int, ring_capacity: int) -> int:
  physical_hot = ring_capacity + 1
  key_blocks = math.ceil(physical_hot / KEY_TILE)
  key_storage_blocks = 2 * key_blocks + 1
  hot_key = KV_HEADS * key_storage_blocks * 2048 * 4
  hot_value = KV_HEADS * physical_hot * HEAD_DIM * 2
  cold_key_value = 2 * KV_HEADS * (bucket + 1) * HEAD_DIM
  cold_scales = 2 * KV_HEADS * (bucket + 1) * SCALE_BYTES
  return 10 * (hot_key + hot_value + cold_key_value + cold_scales)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def candidate_phase_ms(payload: dict[str, Any], phase: int) -> float:
  phases = payload["lanes"]["32k"]["modes"]["candidate"]["phases"]
  matches = [row for row in phases if int(row["index"]) == phase]
  if len(matches) != 1:
    raise ValueError(f"expected one candidate phase {phase}")
  return float(matches[0]["duration_total_ms"])


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--kill-number-ms", type=float, default=2.837)
  parser.add_argument("--decode-cap-ms", type=float, default=26.910657)
  parser.add_argument("--product-output-tokens", type=int, default=512)
  parser.add_argument(
      "--provider-envelope-ms", type=float, default=1.0,
      help="optimistic complete Level Zero setup/append envelope from v28f")
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  carrier = args.carrier.resolve()
  output = args.output.resolve()
  if not carrier.is_dir():
    raise FileNotFoundError(carrier)
  if output.exists():
    raise FileExistsError(output)
  if args.product_output_tokens < 2:
    raise ValueError("product output must contain at least two tokens")
  if args.kill_number_ms <= 0 or args.provider_envelope_ms < 0:
    raise ValueError("invalid timing envelope")

  mem_available = available_memory_bytes()
  memory_stop = int(args.memory_stop_gib * (1024 ** 3))
  if mem_available < memory_stop:
    raise RuntimeError(
        f"memory stop: available={mem_available} required={memory_stop}")

  status = git_output("status", "--porcelain")
  if status:
    raise RuntimeError("repository must be clean before the preflight")
  commit = git_output("rev-parse", "HEAD")

  manifest_path = carrier / "manifest.json"
  plan_path = carrier / "plan.json"
  memory_path = carrier / "memory.json"
  result_path = find_candidate_result(carrier)
  manifest = load_json(manifest_path)
  plan = load_json(plan_path)
  memory = load_json(memory_path)
  result = load_json(result_path)
  rejected = load_json(REJECTED)
  cold_i8_phase = load_json(COLD_I8_PHASE)
  dense_f16_phase = load_json(DENSE_F16_PHASE)
  product_source = PRODUCT_GATE.read_text(encoding="utf-8")
  graph_source = GRAPH_BUILDER.read_text(encoding="utf-8")

  prepare_rows = [
      row for row in rejected.get("rejected", [])
      if row.get("route") == PREPARE_ROUTE
  ]
  if len(prepare_rows) != 1:
    raise ValueError(f"missing unique rejected route {PREPARE_ROUTE}")
  prepare_row = prepare_rows[0]

  cases = [
      row for row in plan.get("cases", [])
      if row.get("candidate_path") == "hot_cold_custom"
  ]
  if len(cases) != 1:
    raise ValueError("carrier must have one custom candidate case")
  case = cases[0]
  bucket = int(case["bucket"])
  source = result.get("source_summary", {})
  walls = [float(value) for value in result.get("decode_wall_ms", [])]
  if len(walls) < 32:
    raise ValueError("carrier needs at least 32 measured decode calls")
  stable = walls[16:]
  stable_median = statistics.median(stable)
  first_transition_excess = max(0.0, walls[0] - stable_median)
  product_decode_calls = args.product_output_tokens - 1
  transition_ceiling = first_transition_excess / product_decode_calls
  optimistic_union = transition_ceiling + args.provider_envelope_ms
  residual = args.kill_number_ms - optimistic_union
  cold_i8_decode_ms = candidate_phase_ms(cold_i8_phase, 1)
  dense_f16_decode_ms = candidate_phase_ms(dense_f16_phase, 1)
  dense_f16_saving_ms = cold_i8_decode_ms - dense_f16_decode_ms
  cold_layers = tuple(int(value) for value in cold_i8_phase["target_layers"])
  dense_layers = tuple(int(value) for value in dense_f16_phase["target_layers"])

  state_summary = result.get("state_summary_after", {})
  actual_state_bytes = int(state_summary.get("byte_count", 0))
  current_custom_bytes = custom_state_bytes(
      bucket, int(source.get("physical_ring_capacity", 0)))
  non_custom_state_bytes = actual_state_bytes - current_custom_bytes
  if non_custom_state_bytes < 0:
    raise ValueError("custom-state formula exceeds measured state bytes")

  capacity_rows = []
  for logical_bucket in BUCKETS:
    current_ring = max(MIN_RING_CAPACITY, logical_bucket)
    current_bytes = (
        non_custom_state_bytes + custom_state_bytes(logical_bucket, current_ring))
    minimum_bytes = (
        non_custom_state_bytes +
        custom_state_bytes(logical_bucket, MIN_RING_CAPACITY))
    capacity_rows.append({
        "bucket": logical_bucket,
        "current_policy_ring_capacity": current_ring,
        "current_policy_state_bytes": current_bytes,
        "smaller_hot_ring_state_bytes": minimum_bytes,
        "resident_state_reduction_bytes": current_bytes - minimum_bytes,
    })

  before_gpu = result.get("gpu_memory", {}).get("before", {})
  after_gpu = result.get("gpu_memory", {}).get("after_decode", {})
  device_growth = int(after_gpu.get("usm_device", 0)) - int(
      before_gpu.get("usm_device", 0))
  host_growth = int(after_gpu.get("usm_host", 0)) - int(
      before_gpu.get("usm_host", 0))
  candidate_memory_rows = [
      row for row in memory.get("rows", [])
      if row.get("mode") == "candidate"
  ]
  if len(candidate_memory_rows) != 1:
    raise ValueError("carrier memory ledger lacks one candidate row")
  candidate_memory = candidate_memory_rows[0]

  compile_pos = product_source.find("language = core.compile_model")
  request_pos = product_source.find("request = language.create_infer_request")
  prefill_pos = product_source.find(
      "for chunk_start in range(0, len(prompt_ids), chunk_tokens)")
  decode_pos = product_source.find("for step in range(1, output_tokens)")
  source_order = 0 <= compile_pos < request_pos < prefill_pos < decode_pos

  checks = [
      check("clean_carrier", not manifest.get("git", {}).get("dirty", True),
            carrier_commit=manifest.get("git", {}).get("commit")),
      check("exact_32k_candidate_path_frozen",
            bucket == 32768 and case.get("candidate_path") == "hot_cold_custom"),
      check("fixed_cold_capacity_matches_bucket",
            source.get("fixed_cold_capacity") == bucket),
      check("fixed_prefill_history_capacity_matches_bucket",
            source.get("prefill_history_capacity") == bucket),
      check("fixed_decode_shape_carrier",
            "fixed product buckets" in str(source.get("custom_attention_mask")) and
            "ceil(total/512)" in str(source.get("length_carrier"))),
      check("one_compiled_model_and_infer_request_source_order", source_order),
      check("same_infer_request_for_prefill_and_decode",
            result.get("same_infer_request") is True),
      check("resident_8k_chunk_schedule",
            result.get("prefill_chunk_tokens") == PREFILL_CHUNK and
            result.get("prefill_chunk_count") == bucket // PREFILL_CHUNK),
      check("all_states_materialized_before_measurement",
            state_summary.get("count") == state_summary.get("materialized_count") == 120),
      check("no_steady_state_capacity_growth",
            device_growth == 0 and abs(host_growth) <= 1024 * 1024,
            device_growth_bytes=device_growth, host_growth_bytes=host_growth),
      check("candidate_oom_safe_at_carrier",
            candidate_memory.get("oom_observed") is False and
            candidate_memory.get("memory_guard_tripped") is False and
            int(candidate_memory.get("system_available_min_bytes", 0)) >= memory_stop),
      check("provider_prepare_route_already_closed",
            "near one millisecond per token" in str(prepare_row.get("reason")) and
            "does not improve the matched complete wall" in
            str(prepare_row.get("reason"))),
      check("full_prompt_dense_f16_beats_cold_i8",
            cold_layers == dense_layers and dense_f16_decode_ms < cold_i8_decode_ms,
            target_layers=list(dense_layers),
            cold_i8_decode_ms=cold_i8_decode_ms,
            dense_f16_decode_ms=dense_f16_decode_ms),
      check("separate_phase_request_route_stays_closed",
            "none for cross-compiled-model state import" in
            REJECTED.read_text(encoding="utf-8")),
      check("exact_bucket_union_ceiling_clears_kill_number",
            optimistic_union > args.kill_number_ms,
            optimistic_union_ms=optimistic_union,
            kill_number_ms=args.kill_number_ms),
  ]
  evidence_checks_passed = all(
      row["pass"] for row in checks
      if row["name"] != "exact_bucket_union_ceiling_clears_kill_number")
  admitted = evidence_checks_passed and optimistic_union > args.kill_number_ms
  verdict = (
      "admit_one_exact_32k_implementation" if admitted
      else "reject_exact_bucket_route_before_compile"
  )

  metrics = {
      "schema": "intel-qwen36-openvino-exact-bucket-preflight-v0",
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "verdict": verdict,
      "required_evidence_checks_passed": evidence_checks_passed,
      "checks": checks,
      "carrier": {
          "artifact": str(carrier.relative_to(REPO)),
          "bucket": bucket,
          "measured_decode_calls": len(walls),
          "stable_skip": 16,
          "stable_median_ms": stable_median,
          "first_decode_ms": walls[0],
          "first_transition_excess_ms": first_transition_excess,
          "state_bytes": actual_state_bytes,
          "candidate_system_available_min_bytes": int(
              candidate_memory["system_available_min_bytes"]),
          "candidate_process_swap_peak_bytes": int(
              candidate_memory["process_swap_peak_bytes"]),
      },
      "optimistic_union_bound": {
          "product_output_tokens": args.product_output_tokens,
          "product_decode_calls": product_decode_calls,
          "first_transition_amortized_ceiling_ms_per_token": transition_ceiling,
          "entire_provider_setup_append_envelope_ms_per_token":
              args.provider_envelope_ms,
          "total_optimistic_saving_ms_per_token": optimistic_union,
          "kill_number_ms_per_token": args.kill_number_ms,
          "residual_miss_ms_per_token": residual,
          "decode_cap_ms_per_token": args.decode_cap_ms,
          "optimism": [
              "charges the complete first-decode excess as removable",
              "charges the entire registered Level Zero setup/append envelope",
              "ignores overlap between the transition and provider envelopes",
              "charges no implementation or integration overhead",
          ],
      },
      "capacity_projection": {
          "measured_non_custom_state_bytes": non_custom_state_bytes,
          "logical_hot_window": HOT_WINDOW,
          "smaller_ring_capacity": MIN_RING_CAPACITY,
          "rows": capacity_rows,
          "speed_disposition": (
              "reject as a speed cut: it re-enables the slower cold-I8 read "
              "path for the prompt prefix"),
          "matched_phase_context": {
              "target_layers": list(dense_layers),
              "cold_i8_decode_ms": cold_i8_decode_ms,
              "dense_f16_decode_ms": dense_f16_decode_ms,
              "dense_f16_saving_ms": dense_f16_saving_ms,
              "cold_i8_artifact": str(COLD_I8_PHASE.parent.relative_to(REPO)),
              "dense_f16_artifact": str(DENSE_F16_PHASE.parent.relative_to(REPO)),
          },
      },
      "source_inputs": {
          "product_gate": {
              "path": str(PRODUCT_GATE.relative_to(REPO)),
              "sha256": sha256(PRODUCT_GATE),
          },
          "graph_builder": {
              "path": str(GRAPH_BUILDER.relative_to(REPO)),
              "sha256": sha256(GRAPH_BUILDER),
          },
          "rejected_routes": {
              "path": str(REJECTED.relative_to(REPO)),
              "sha256": sha256(REJECTED),
          },
          "cold_i8_phase": {
              "path": str(COLD_I8_PHASE.relative_to(REPO)),
              "sha256": sha256(COLD_I8_PHASE),
          },
          "dense_f16_phase": {
              "path": str(DENSE_F16_PHASE.relative_to(REPO)),
              "sha256": sha256(DENSE_F16_PHASE),
          },
      },
  }

  output.mkdir(parents=True)
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest_out = {
      "schema": "intel-qwen36-openvino-exact-bucket-preflight-v0-manifest-v0",
      "created_at": metrics["created_at"],
      "artifact": str(output.relative_to(REPO)),
      "git": {"commit": commit, "dirty": False, "status": status},
      "host": {"target_alias": "intel-ptl-local"},
      "memory": {
          "available_bytes": mem_available,
          "stop_bytes": memory_stop,
          "worker_launched": False,
      },
      "verdict": verdict,
      "required_evidence_checks_passed": evidence_checks_passed,
  }
  (output / "manifest.json").write_text(
      json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  row128 = next(row for row in capacity_rows if row["bucket"] == 131072)
  summary = f"""# OpenVINO exact-bucket specialization preflight

Verdict: **{verdict}**. Required evidence checks: `{str(evidence_checks_passed).lower()}`.

The clean 32k carrier already fixes candidate path, cold/cache capacity, the
decode length carrier, one `InferRequest`, resident 8k prefill chunks, state,
and the serial single-token schedule. Stable decode is
`{stable_median:.6f} ms/token`; the first decode transition is
`{walls[0]:.6f} ms`.

Even deleting that complete transition and amortizing it over the 511 product
decode calls saves only `{transition_ceiling:.6f} ms/token`. Deleting the entire
registered `{args.provider_envelope_ms:.3f} ms/token` Level Zero setup/append
envelope as well gives an overlap-free optimistic union of
`{optimistic_union:.6f} ms/token`, below the `{args.kill_number_ms:.3f}` kill-number
by `{residual:.6f} ms/token`. Seven-variant compilation and another long row are
therefore not admissible.

The current 128k capacity policy projects
`{row128['current_policy_state_bytes'] / (1024**3):.3f} GiB` of materialized model
state. A 16k ring would lower this to
`{row128['smaller_hot_ring_state_bytes'] / (1024**3):.3f} GiB`, but it re-enables
the slower cold-I8 prefix path. Matched 9-layer 32k attribution moved decode
attention from `{dense_f16_decode_ms:.3f}` with dense F16 to
`{cold_i8_decode_ms:.3f} ms` with cold I8. It is a memory option, not a speed
continuation. No worker was launched by this preflight.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(REPO)),
      "event": "openvino_exact_bucket_preflight_complete",
      "optimistic_union_ms": optimistic_union,
      "verdict": verdict,
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as error:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)
