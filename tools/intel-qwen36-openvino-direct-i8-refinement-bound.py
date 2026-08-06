#!/usr/bin/env python3
"""Admit or close one accuracy-preserving direct-I8 refinement component.

This gate is source/evidence only.  It joins the failed integrated group-32
product row to the clean direct-I8 component, audits the saved real 32k K/V
state, and evaluates the exact byte/error Pareto points for power-of-two
quantization groups.  It launches no compiler or GPU worker and admits at
most one pre-registered group-4/full-logical-cold component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-direct-i8-refinement-bound-v0"

ACCEPTANCE = REPO / "benchmarks" / WS / "acceptance-matrix.json"
STATUS = REPO / "doc/active" / WS / "STATUS.md"
COMPONENT = REPO / (
    "output/openvino-direct-i8-attention-component-"
    "20260715Tseq1245-cleanZ/result.json")
ABI_BOUND = REPO / (
    "output/openvino-direct-i8-integration-abi-bound-"
    "20260715Tseq1246b-cleanZ/metrics.json")
ACCEPTED_WORKER = REPO / (
    "output/openvino-hot-cold-product-20260715Tseq1204-"
    "alias-fused-linear-state-32k-o64-cleanZ/raw/sentinel_032k/"
    "correctness/candidate/worker-result.json")
DIRECT_WORKER = REPO / (
    "output/openvino-direct-i8-product-20260715Tseq1256-"
    "all10-32k-o45-divergence-cleanZ/raw/sentinel_032k/"
    "correctness/candidate/worker-result.json")
DIRECT_COMPARISON = DIRECT_WORKER.parent.parent / "comparison.json"
STATE_WORKER = REPO / (
    "output/openvino-direct-i8-integration-20260715Tseq1251-"
    "layer3-32k-cleanZ/raw/32k/stock/worker-result.json")
COMPONENT_SOURCE = REPO / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
INTEGRATED_SOURCE = REPO / (
    "engine/openvino/custom/iq36_hot_attention_single_owner.cl")
HELPERS = REPO / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl"

CONTEXT = 32768
HOT = 8192
COLD = CONTEXT - HOT
HEAD_DIM = 256
KV_HEADS = 2
GROUPS = (32, 16, 8, 4)
F16_BYTES = 2
I8_BYTES = 1
LAYERS = 10
ATTENTION_MS = 8.456
KILL_MS = 2.837085
COMPONENT_CAP_MS = (ATTENTION_MS - KILL_MS) / LAYERS
MEMORY_CHUNK_TOKENS = 1024


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


def state_path(metadata: dict[str, Any], kind: str) -> tuple[Path, tuple[int, ...]]:
  needle = f".past.{kind}."
  rows = [row for name, row in metadata.items() if needle in name]
  if len(rows) != 1:
    raise ValueError(f"expected one saved {kind} state, got {len(rows)}")
  row = rows[0]
  path = REPO / str(row["path"])
  shape = tuple(int(value) for value in row["shape"])
  return path, shape


def quantization_audit(path: Path, shape: tuple[int, ...]) -> dict[str, Any]:
  if shape != (1, KV_HEADS, CONTEXT + 1, HEAD_DIM):
    raise ValueError(f"unexpected state shape {shape}: {path}")
  source = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
  accumulators = {
      group: {"error_sq": 0.0, "signal_sq": 0.0,
              "maximum_abs_error": 0.0, "values": 0}
      for group in GROUPS}
  for begin in range(0, COLD, MEMORY_CHUNK_TOKENS):
    end = min(COLD, begin + MEMORY_CHUNK_TOKENS)
    rounded = np.asarray(
        source[:, :, begin:end, :], dtype=np.float16).astype(np.float32)
    for group, totals in accumulators.items():
      blocks = rounded.reshape(
          *rounded.shape[:-1], HEAD_DIM // group, group)
      maximum = np.max(np.abs(blocks), axis=-1)
      scale = np.where(maximum == 0.0, 1.0, maximum / 127.0).astype(
          np.float32)
      quantized = np.clip(
          np.rint(blocks / scale[..., None]), -127, 127).astype(np.int8)
      stored_scale = scale.astype(np.float16).astype(np.float32)
      reconstructed = quantized.astype(np.float32) * stored_scale[..., None]
      error = reconstructed - blocks
      totals["error_sq"] += float(np.sum(error.astype(np.float64) ** 2))
      totals["signal_sq"] += float(np.sum(blocks.astype(np.float64) ** 2))
      totals["maximum_abs_error"] = max(
          float(totals["maximum_abs_error"]),
          float(np.max(np.abs(error))))
      totals["values"] += int(error.size)
  result: dict[str, Any] = {}
  for group, totals in accumulators.items():
    error_sq = float(totals["error_sq"])
    result[str(group)] = {
        "relative_l2": (error_sq / float(totals["signal_sq"])) ** 0.5,
        "rmse": (error_sq / int(totals["values"])) ** 0.5,
        "maximum_abs_error": float(totals["maximum_abs_error"]),
    }
  return result


def state_bytes(group: int) -> dict[str, int]:
  scale_groups = HEAD_DIM // group
  hot_kv = HOT * KV_HEADS * HEAD_DIM * F16_BYTES * 2
  cold_i8 = COLD * KV_HEADS * HEAD_DIM * I8_BYTES * 2
  cold_scales = COLD * KV_HEADS * scale_groups * F16_BYTES * 2
  return {
      "hot_kv": hot_kv,
      "cold_i8": cold_i8,
      "cold_scales": cold_scales,
      "total": hot_kv + cold_i8 + cold_scales,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      ACCEPTANCE, STATUS, COMPONENT, ABI_BOUND, ACCEPTED_WORKER,
      DIRECT_WORKER, DIRECT_COMPARISON, STATE_WORKER, COMPONENT_SOURCE,
      INTEGRATED_SOURCE, HELPERS)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing refinement-bound inputs: " + ", ".join(missing))

  git = git_state()
  acceptance = load_json(ACCEPTANCE)
  component = load_json(COMPONENT)
  abi = load_json(ABI_BOUND)
  accepted = load_json(ACCEPTED_WORKER)
  direct = load_json(DIRECT_WORKER)
  comparison = load_json(DIRECT_COMPARISON)
  state_worker = load_json(STATE_WORKER)
  component_source = COMPONENT_SOURCE.read_text(encoding="utf-8")
  integrated_source = INTEGRATED_SOURCE.read_text(encoding="utf-8")
  helpers = HELPERS.read_text(encoding="utf-8")
  sample_memory("after-evidence-load", stop_bytes, memory)

  saved = state_worker["phases"][1]["saved_states"]
  key_path, key_shape = state_path(saved, "key")
  value_path, value_shape = state_path(saved, "value")
  key_audit = quantization_audit(key_path, key_shape)
  sample_memory("after-key-audit", stop_bytes, memory)
  value_audit = quantization_audit(value_path, value_shape)
  sample_memory("after-value-audit", stop_bytes, memory)

  accepted_median = statistics.median(
      float(value) for value in accepted["decode_wall_ms"][16:])
  direct_median = statistics.median(
      float(value) for value in direct["decode_wall_ms"][16:])
  direct_saving = accepted_median - direct_median
  direct_residual = direct_median - 26.911

  component_result = component["result"]
  component_ucb = float(
      component["performance_inference"]["upper_confidence_bound_ms"])
  group32_bytes = state_bytes(32)
  pareto: dict[str, Any] = {}
  for group in GROUPS:
    bytes_row = state_bytes(group)
    scaled_ucb = component_ucb * bytes_row["total"] / group32_bytes["total"]
    pareto[str(group)] = {
        "state_bytes": bytes_row,
        "scaled_one_layer_ucb_ms": scaled_ucb,
        "scaled_all_layer_ucb_ms": scaled_ucb * LAYERS,
        "margin_below_component_cap_ms": COMPONENT_CAP_MS - scaled_ucb,
        "key": key_audit[str(group)],
        "value": value_audit[str(group)],
        "key_relative_l2_vs_group32": (
            key_audit[str(group)]["relative_l2"] /
            key_audit["32"]["relative_l2"]),
        "value_relative_l2_vs_group32": (
            value_audit[str(group)]["relative_l2"] /
            value_audit["32"]["relative_l2"]),
    }

  group4 = pareto["4"]
  source_mismatch = {
      "component_hot_v_is_dimension_major": (
          "((ulong)kv_head * IQ36_HEAD_DIM + dim) * IQ36_HOT_TOKENS" in
          component_source),
      "integrated_hot_v_is_token_major": (
          "iq36_block2d_load_f16_16x8(" in integrated_source and
          "const __global half* state_base" in integrated_source),
      "integrated_direct_policy_attends_only_wrapped_cold_rows": (
          "min(cold_tokens, dense_history_begin)" in integrated_source),
      "component_attends_full_logical_cold_prefix": (
          "const bool cold_chunk = chunk_begin < IQ36_COLD_TOKENS" in
          component_source),
      "current_scale_group_is_32": (
          "#define IQ36_SCALE_GROUPS 8U" in helpers and
          "dim >> 5U" in helpers),
  }

  route_contract = acceptance["candidate_runtime"]["first_prefill_route"]
  distribution = comparison.get("distribution_rows", [])
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("product_contract_and_decode_kill_are_exact",
            route_contract.get("component_profile_is_product_evidence") is False
            and COMPONENT_CAP_MS == 0.5618915,
            component_cap_ms=COMPONENT_CAP_MS),
      check("clean_group32_component_is_promoted_but_not_product_evidence",
            component.get("verdict") ==
                "promote_direct_i8_attention_component" and
            component.get("component_promoted") is True and
            component.get("graph_integration_admitted") is False and
            component_result.get("quant_group") == 32 and
            component_result.get("cold_tokens") == COLD and
            component_result.get("hot_tokens") == HOT),
      check("prior_abi_bound_is_admitted_but_layout_transfer_is_inexact",
            abi.get("verdict") == "admit_direct_i8_openvino_integration_source"
            and all(source_mismatch.values()),
            source_mismatch=source_mismatch),
      check("integrated_group32_product_is_closed_on_correctness_and_speed",
            comparison.get("required_checks_passed") is False and
            comparison.get("kld_max") == 0.15350662840040744 and
            comparison.get("top1_rate") == 0.6 and
            [row.get("step") for row in distribution] == [0, 1, 7, 36, 44]
            and direct_residual > 0.0,
            accepted_median_after_skip16_ms=accepted_median,
            direct_median_after_skip16_ms=direct_median,
            realized_saving_ms=direct_saving,
            residual_to_cap_ms=direct_residual),
      check("saved_real_32k_state_identity_is_exact",
            key_shape == value_shape == (1, KV_HEADS, CONTEXT + 1, HEAD_DIM)
            and key_path.stat().st_size == value_path.stat().st_size ==
                67_110_912,
            key_path=display_path(key_path), key_sha256=sha256(key_path),
            value_path=display_path(value_path), value_sha256=sha256(value_path)),
      check("group4_materially_reduces_real_state_reconstruction_error",
            group4["key_relative_l2_vs_group32"] <= 0.50 and
            group4["value_relative_l2_vs_group32"] <= 0.50,
            group4=group4),
      check("group4_full_cold_byte_ceiling_clears_component_cap",
            group4["scaled_one_layer_ucb_ms"] < COMPONENT_CAP_MS and
            group4["margin_below_component_cap_ms"] >= 0.10,
            group4=group4),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed
  verdict = (
      "admit_one_direct_i8_group4_full_cold_component"
      if admitted else "reject_direct_i8_refinement_before_compile")

  component_contract = {
      "quant_group": 4,
      "context_tokens": CONTEXT,
      "logical_hot_tokens": HOT,
      "logical_cold_tokens": COLD,
      "cold_policy": "attend_the_full_logical_cold_prefix",
      "hot_k_layout": "token16_dim2_packed_f16",
      "hot_v_layout": "dimension_major_token16_f16",
      "cold_k_layout": "token16_group4_packed_i8",
      "cold_v_layout": "dimension_major_i8",
      "scale_layout": "group_major_fp16",
      "maximum_one_sided_95_latency_bound_ms": COMPONENT_CAP_MS,
      "warmup_samples": 5,
      "measured_samples": 20,
      "build_parallelism": 1,
      "serial_gpu_worker": True,
      "memory_stop_bytes": stop_bytes,
      "tile_or_workgroup_sweep_allowed": False,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "passing_action": (
          "admit one real-layer short integration source gate; retain the "
          "32k product and every ABBA/long row behind correctness"),
      "failing_action": (
          "close direct-I8 quant-group refinements and do not integrate"),
  }
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admitted": admitted,
      "source_edit_admitted": admitted,
      "compile_admitted": admitted,
      "gpu_worker_admitted": admitted,
      "graph_integration_admitted": False,
      "full_model_worker_admitted": False,
      "long_worker_admitted": False,
      "gpu_worker_launched": False,
      "product_claim_allowed": False,
      "integrated_group32": {
          "accepted_median_after_skip16_ms": accepted_median,
          "direct_median_after_skip16_ms": direct_median,
          "realized_saving_ms": direct_saving,
          "residual_to_32k_cap_ms": direct_residual,
          "kld_max": comparison.get("kld_max"),
          "top1_rate": comparison.get("top1_rate"),
      },
      "source_mismatch": source_mismatch,
      "pareto": pareto,
      "selected_component": component_contract,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Direct-I8 refinement complete bound

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The exact integrated group-32 route is closed: its stable 32k median saves
only `{direct_saving:.6f} ms/token`, still misses the cap by
`{direct_residual:.6f} ms/token`, and the registered teacher-forced row reaches
KLD `{comparison.get('kld_max')}` with top-1 rate
`{comparison.get('top1_rate')}`.  The earlier component-to-product transfer
also crossed two real source mismatches: dimension-major versus token-major
hot V, and full logical-cold versus wrapped-only cold selection.

On the saved real 32k layer-3 state, group 4 reduces key/value reconstruction
relative L2 to `{key_audit['4']['relative_l2']:.9f}` and
`{value_audit['4']['relative_l2']:.9f}`, respectively
`{group4['key_relative_l2_vs_group32']:.6f}x` and
`{group4['value_relative_l2_vs_group32']:.6f}x` group 32.  Charging every
extra F16 scale byte scales the clean component UCB to
`{group4['scaled_one_layer_ucb_ms']:.6f} ms`, leaving
`{group4['margin_below_component_cap_ms']:.6f} ms` below the exact
`{COMPONENT_CAP_MS:.7f}-ms` one-layer cap.

This admits one fixed group-4, full-logical-cold, dimension-major-hot-V
component only.  It has no tile/workgroup sweep, five warmups plus twenty
samples, serial execution, `-j1`, and the 4-GiB stop.  Passing admits one
real-layer short source integration; it does not admit a full-model or long
worker.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "group4_scaled_ucb_ms": group4["scaled_one_layer_ucb_ms"],
      "group4_key_relative_l2_ratio":
          group4["key_relative_l2_vs_group32"],
      "group4_value_relative_l2_ratio":
          group4["value_relative_l2_vs_group32"],
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
