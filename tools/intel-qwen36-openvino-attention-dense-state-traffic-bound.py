#!/usr/bin/env python3
"""Bound arithmetic-preserving decode attention by mandatory dense-state traffic.

This gate is source-only.  It audits the locked model, the accepted attention
source, stored decode IGC attribution, and the exact accepted-carrier refresh.
It never compiles or launches a GPU worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-dense-state-traffic-bound-v0"

MODEL_CONFIG = Path("/home/intel/Qwen3.6-35B-A3B-ov/config.json")
ATTENTION_SOURCE = (
    REPO / "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
ATTENTION_HELPERS = (
    REPO / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl")
STATUS = REPO / "doc/active" / WS / "STATUS.md"
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
PERFORMANCE_TARGET = REPO / "doc/reference" / WS / (
    "performance-target-2026-07-11.md")
PROFILE_REFRESH = REPO / (
    "output/openvino-accepted-carrier-profile-refresh-"
    "20260715Tseq1240-2k-warm17-cleanZ/metrics.json")
CODEGEN = REPO / (
    "output/openvino-attention-codegen-"
    "20260715Tseq954-decode-only-2k-16k-cleanZ/metrics.jsonl")
REJECTED_ROUTES = REPO / "doc/active" / WS / "rejected-routes.json"

CONTEXT_TOKENS = 32768
F16_BYTES = 2
REGISTERED_ATTENTION_MS = 8.456
PLANNING_GB_S = 115.0
RAW_LPDDR_GB_S = 136.5


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


def parse_define(text: str, name: str) -> int:
  match = re.search(
      rf"^#define\s+{re.escape(name)}\s+(\d+)U?\s*$", text, re.MULTILINE)
  if match is None:
    raise ValueError(f"missing numeric define {name}")
  return int(match.group(1))


def codegen_audit() -> dict[str, Any]:
  rows = []
  for line in CODEGEN.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if (
        row.get("mode") == "candidate" and
        row.get("lane") == "16k" and
        row.get("kernel_name") == "iq36_hot_attention_single_owner" and
        "phase1" in str(row.get("build_marker", ""))
    ):
      rows.append(row)
  environments = [row.get("execution_env", {}) for row in rows]
  unique_environments = {
      json.dumps(environment, sort_keys=True) for environment in environments}
  expected = {
      "simd_size": 16,
      "grf_count": 128,
      "eu_thread_count": 8,
      "spill_mem_size": 0,
      "scratch_size": 0,
      "private_size": 0,
      "reported_spill_mem_size": 0,
      "reported_spill_size": 0,
  }
  selected = environments[0] if environments else {}
  return {
      "matching_rows": len(rows),
      "unique_execution_environments": len(unique_environments),
      "execution_environment": selected,
      "expected_execution_environment_subset": expected,
      "spill_free_128_grf_eight_threads": (
          bool(rows) and len(unique_environments) == 1 and
          all(selected.get(key) == value for key, value in expected.items())),
      "scope": (
          "stored decode-core IGC attribution; current phase specialization "
          "changes state/cold selection but retains the chunk512 decode core"),
  }


def source_audit() -> dict[str, Any]:
  source = ATTENTION_SOURCE.read_text(encoding="utf-8")
  helpers = ATTENTION_HELPERS.read_text(encoding="utf-8")
  markers = (
      "Decode: one work-group owns one 512-token partial for one KV head.",
      "Product buckets retain the entire prompt in the dense F16 ring.",
      "for (uint block = subgroup; block < IQ36_BLOCKS_PER_CHUNK;",
      "intel_sub_group_f16_f16_matrix_mad_k16(",
      "const __global half* state_base =",
      "for (uint block = 0U; block < valid_blocks; ++block)",
      "workspace[head_base] =",
      "for (uint partial_chunk = 0U;",
  )
  helper_markers = (
      "Hot K is F16x2-packed in I32 block16 planes for XMX; hot V is direct F16.",
      "#define IQ36_GQA_GROUP 8U",
      "#define IQ36_CHUNK_TOKENS 512U",
      "#define IQ36_TOKEN_TILE 16U",
  )
  return {
      "source_markers": {marker: marker in source for marker in markers},
      "helper_markers": {
          marker: marker in helpers for marker in helper_markers},
      "all_markers_present": (
          all(marker in source for marker in markers) and
          all(marker in helpers for marker in helper_markers)),
      "head_dim": parse_define(helpers, "IQ36_HEAD_DIM"),
      "kv_heads": parse_define(helpers, "IQ36_KV_HEADS"),
      "gqa_group": parse_define(helpers, "IQ36_GQA_GROUP"),
      "chunk_tokens": parse_define(helpers, "IQ36_CHUNK_TOKENS"),
      "token_tile": parse_define(helpers, "IQ36_TOKEN_TILE"),
      "dense_state_contract": (
          "each full-attention layer must consume every prompt K and V F16 "
          "element once; the source already reuses each KV plane across all "
          "eight GQA heads and excludes query/workspace/provider traffic from "
          "the mandatory-byte floor"),
  }


def rejected_route_audit(routes: dict[str, Any]) -> dict[str, Any]:
  rows = routes.get(
      "rejected", routes.get("routes", routes.get("rejected_routes", [])))
  if not isinstance(rows, list):
    rows = []
  selected = []
  for row in rows:
    text = " ".join(str(row.get(key, "")) for key in (
        "id", "reason", "evidence", "reopen_condition"))
    if "seq1205" in text or "chunk256" in text:
      selected.append(row)
  combined = json.dumps(selected, sort_keys=True)
  return {
      "matching_rows": len(selected),
      "records_chunk256_rejection": "31.588" in combined,
      "records_current_chunk512_spill_free": (
          "128 GRF" in combined and "eight" in combined and
          "spill" in combined),
      "rows": selected,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_CONFIG, ATTENTION_SOURCE, ATTENTION_HELPERS, STATUS, FRONTIER,
      PERFORMANCE_TARGET, PROFILE_REFRESH, CODEGEN, REJECTED_ROUTES)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  git = git_state()
  config = load_json(MODEL_CONFIG)
  frontier = load_json(FRONTIER)
  refresh = load_json(PROFILE_REFRESH)
  rejected = load_json(REJECTED_ROUTES)
  source = source_audit()
  igc = codegen_audit()
  closed_shapes = rejected_route_audit(rejected)
  sample_memory("after-source-and-stored-evidence", stop_bytes, memory)

  text_config = config.get("text_config", {})
  model = {
      "layers": int(text_config.get("num_hidden_layers", -1)),
      "full_attention_interval": int(
          text_config.get("full_attention_interval", -1)),
      "attention_heads": int(text_config.get("num_attention_heads", -1)),
      "kv_heads": int(text_config.get("num_key_value_heads", -1)),
      "head_dim": int(text_config.get("head_dim", -1)),
  }
  full_attention_layers = (
      model["layers"] // model["full_attention_interval"]
      if model["full_attention_interval"] > 0 else -1)
  model_exact = model == {
      "layers": 40,
      "full_attention_interval": 4,
      "attention_heads": 16,
      "kv_heads": 2,
      "head_dim": 256,
  }

  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  target_attention_ms = REGISTERED_ATTENTION_MS - kill_number_ms
  plane_bytes_per_layer = (
      CONTEXT_TOKENS * model["kv_heads"] * model["head_dim"] * F16_BYTES)
  mandatory_dense_kv_bytes = (
      plane_bytes_per_layer * 2 * full_attention_layers)
  mandatory_ms_at_planning = (
      mandatory_dense_kv_bytes / (PLANNING_GB_S * 1_000_000_000.0) * 1000.0)
  optimistic_saving_ms = REGISTERED_ATTENTION_MS - mandatory_ms_at_planning
  shortfall_ms = kill_number_ms - optimistic_saving_ms
  required_gb_s = (
      mandatory_dense_kv_bytes / (target_attention_ms / 1000.0) /
      1_000_000_000.0)
  mandatory_ms_at_raw_peak = (
      mandatory_dense_kv_bytes / (RAW_LPDDR_GB_S * 1_000_000_000.0) * 1000.0)
  raw_peak_sensitivity_saving_ms = (
      REGISTERED_ATTENTION_MS - mandatory_ms_at_raw_peak)
  route_fundable_at_planning = optimistic_saving_ms >= kill_number_ms

  status = STATUS.read_text(encoding="utf-8")
  performance_target = PERFORMANCE_TARGET.read_text(encoding="utf-8")
  accepted_sources = {
      str(row.get("path")): row for row in
      refresh.get("accepted_identity", {}).get("sources", [])}
  current_source_hash_matches_refresh = (
      accepted_sources.get(display_path(ATTENTION_SOURCE), {}).get(
          "actual_sha256") == sha256(ATTENTION_SOURCE) and
      accepted_sources.get(display_path(ATTENTION_SOURCE), {}).get(
          "match") is True)
  profile_exact = (
      refresh.get("required_checks_passed") is True and
      refresh.get("verdict") ==
          "select_refreshed_custom_attention_algorithm_bound" and
      refresh.get("profile_audit", {}).get("selected_counts_exact") is True and
      refresh.get("profile_audit", {}).get(
          "selected_executed_counts", {}).get("IQ36HotAttentionGQA") == 10)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_model_has_exact_ten_full_attention_layers",
            model_exact and full_attention_layers == 10, model=model),
      check("accepted_source_encodes_dense_f16_kv_and_gqa_reuse",
            source["all_markers_present"] and
            source["head_dim"] == 256 and source["kv_heads"] == 2 and
            source["gqa_group"] == 8 and source["chunk_tokens"] == 512 and
            source["token_tile"] == 16, source=source),
      check("accepted_carrier_refresh_matches_current_attention_source",
            profile_exact and current_source_hash_matches_refresh,
            source_hash=sha256(ATTENTION_SOURCE)),
      check("stored_decode_igc_is_spill_free_128_grf_eight_threads",
            igc["spill_free_128_grf_eight_threads"], igc=igc),
      check("closed_chunk_shape_record_matches_current_codegen",
            closed_shapes["records_chunk256_rejection"] and
            closed_shapes["records_current_chunk512_spill_free"],
            matching_rows=closed_shapes["matching_rows"]),
      check("registered_attention_bucket_and_kill_number_are_current",
            "8.456" in status and "2.837" in status and
            abs(kill_number_ms - 2.837085) < 1e-9),
      check("planning_and_raw_bandwidth_lines_are_registered",
            "raw LPDDR estimate is 136.5 GB/s" in performance_target and
            "planning line\n  is 115 GB/s" in performance_target),
      check("mandatory_dense_state_floor_misses_kill_number_at_planning_line",
            not route_fundable_at_planning and required_gb_s > PLANNING_GB_S,
            optimistic_saving_ms_per_token=optimistic_saving_ms,
            kill_number_ms_per_token=kill_number_ms,
            required_gb_s=required_gb_s,
            planning_gb_s=PLANNING_GB_S,
            shortfall_ms_per_token=shortfall_ms),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_dense_f16_attention_algorithm_before_source"
      if required_checks_passed and not route_fundable_at_planning else
      "admit_one_materially_different_attention_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": required_checks_passed and route_fundable_at_planning,
      "source_edit_admitted": False,
      "compile_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "budget": {
          "context_tokens": CONTEXT_TOKENS,
          "registered_attention_ms_per_token": REGISTERED_ATTENTION_MS,
          "kill_number_ms_per_token": kill_number_ms,
          "target_attention_ms_per_token": target_attention_ms,
          "full_attention_layers": full_attention_layers,
          "kv_heads": model["kv_heads"],
          "head_dim": model["head_dim"],
          "element_bytes": F16_BYTES,
          "one_k_or_v_plane_bytes_per_layer": plane_bytes_per_layer,
          "mandatory_dense_k_plus_v_bytes_per_token": mandatory_dense_kv_bytes,
          "planning_gb_s": PLANNING_GB_S,
          "mandatory_dense_kv_ms_at_planning": mandatory_ms_at_planning,
          "optimistic_complete_saving_ms_at_planning": optimistic_saving_ms,
          "shortfall_ms_per_token": shortfall_ms,
          "required_dense_kv_gb_s_to_fund_kill": required_gb_s,
          "raw_lpddr_gb_s": RAW_LPDDR_GB_S,
          "mandatory_dense_kv_ms_at_raw_peak": mandatory_ms_at_raw_peak,
          "raw_peak_sensitivity_saving_ms": raw_peak_sensitivity_saving_ms,
          "bound_rule": (
              "charge only one read of prompt K and V F16 state across ten "
              "full-attention layers; make query, score/workspace, softmax, "
              "DPAS, state update, synchronization, launch, and output free"),
          "interpretation": (
              "planning-line rejection, not a physical-impossibility claim; "
              "reopen only with a correctness-safe state-byte reduction or "
              "an independently demonstrated complete K+V carrier above the "
              "required bandwidth before attention source integration"),
      },
      "model": model,
      "source_audit": source,
      "stored_codegen_audit": igc,
      "closed_shape_audit": closed_shapes,
      "accepted_refresh_route_selection": refresh.get("route_selection", {}),
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Dense-F16 attention state-traffic bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The locked model has ten full-attention layers, two KV heads, head dimension
256, and a dense F16 K/V carrier at the accepted 32k decode boundary. The
current chunk512 source already reads each K and V plane once while reusing it
across all eight GQA heads. Stored decode IGC is SIMD16, 128 GRF, eight
threads/EU, and spill/scratch free; closed chunk256 doubled scheduling work and
regressed the accepted carrier. Query/workspace reuse, branch, chunk,
workgroup, SLM, subgroup, and projection-consumer shapes remain closed.

Even an impossible free kernel must read at least
`{mandatory_dense_kv_bytes:,}` bytes/token: prompt-only K plus V across ten
layers. To remove the full `{kill_number_ms:.6f} ms/token` from the registered
`{REGISTERED_ATTENTION_MS:.3f}` bucket, attention must fall to
`{target_attention_ms:.6f} ms/token`, requiring `{required_gb_s:.6f} GB/s` for
mandatory state alone. That exceeds the registered `{PLANNING_GB_S:.0f} GB/s`
planning line. At that line the mandatory floor is
`{mandatory_ms_at_planning:.6f} ms/token`, so the maximally favorable saving is
only `{optimistic_saving_ms:.6f}`, short by `{shortfall_ms:.6f}`.

The `{RAW_LPDDR_GB_S:.1f} GB/s` raw-LPDDR sensitivity would save
`{raw_peak_sensitivity_saving_ms:.6f} ms/token`; this is not an attainable
kernel bound and is not used to admit code. Reopen only with a correctness-safe
state-byte reduction or an independently demonstrated complete K+V carrier
above `{required_gb_s:.6f} GB/s`. Source, compile, 32k, ABBA, and output512 are
not admitted for the current arithmetic-preserving dense-F16 attention route.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "mandatory_dense_kv_bytes": mandatory_dense_kv_bytes,
      "required_gb_s": required_gb_s,
      "planning_gb_s": PLANNING_GB_S,
      "optimistic_saving_ms": optimistic_saving_ms,
      "shortfall_ms": shortfall_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
