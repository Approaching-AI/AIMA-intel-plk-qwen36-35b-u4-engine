#!/usr/bin/env python3
"""Bound one K2/V4 direct-I8 component without compiling or using the GPU.

The integrated K4/V4 profile is closed on both latency and distribution
correctness.  This gate uses its clean evidence, the promoted standalone
K4/V4 component, and the saved real layer-3 state to select at most one
strictly narrower refinement: group-2 scales for cold K while cold V remains
group 4.  The physical K byte packing stays dim-4, so scale grouping and byte
packing must be independent in any admitted source edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-direct-i8-hybrid-k2-v4-bound-v0"

GROUP4_BOUND = REPO / (
    "output/openvino-direct-i8-refinement-bound-"
    "20260715Tseq1261-cleanZ/metrics.json")
GROUP4_COMPONENT = REPO / (
    "output/openvino-direct-i8-group4-attention-component-"
    "20260715Tseq1262-cleanZ/result.json")
GROUP4_CORRECTNESS = REPO / (
    "output/openvino-direct-i8-group4-integration-"
    "20260715Tseq1265-layer3-32k-cleanZ/metrics.json")
GROUP4_PROFILE = REPO / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1267-group4-layer3-32k-warm25-cleanZ/metrics.json")
STATE_WORKER = REPO / (
    "output/openvino-direct-i8-integration-20260715Tseq1251-"
    "layer3-32k-cleanZ/raw/32k/stock/worker-result.json")
SOURCE = REPO / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
RUNNER = REPO / "engine/tools/direct_i8_hotcold_gqa_decode.cpp"

CONTEXT = 32768
HOT = 8192
COLD = CONTEXT - HOT
HEAD_DIM = 256
KV_HEADS = 2
F16_BYTES = 2
I8_BYTES = 1
KEY_GROUP = 2
VALUE_GROUP = 4
PACK_DIMS = 4
MEMORY_CHUNK_TOKENS = 1024
COMPONENT_CAP_MS = 0.5618915


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


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    output_relative = str(output.relative_to(REPO))
  except ValueError:
    output_relative = ""
  status = [
      row for row in status
      if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def state_path(metadata: dict[str, Any], kind: str) -> tuple[Path, tuple[int, ...]]:
  needle = f".past.{kind}."
  rows = [row for name, row in metadata.items() if needle in name]
  if len(rows) != 1:
    raise ValueError(f"expected one saved {kind} state, got {len(rows)}")
  row = rows[0]
  return REPO / str(row["path"]), tuple(int(value) for value in row["shape"])


def quantization_audit(
    path: Path, shape: tuple[int, ...], group: int,
) -> dict[str, Any]:
  if shape != (1, KV_HEADS, CONTEXT + 1, HEAD_DIM):
    raise ValueError(f"unexpected state shape {shape}: {path}")
  source = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
  error_sq = 0.0
  signal_sq = 0.0
  maximum_abs_error = 0.0
  values = 0
  for begin in range(0, COLD, MEMORY_CHUNK_TOKENS):
    end = min(COLD, begin + MEMORY_CHUNK_TOKENS)
    rounded = np.asarray(
        source[:, :, begin:end, :], dtype=np.float16).astype(np.float32)
    blocks = rounded.reshape(*rounded.shape[:-1], HEAD_DIM // group, group)
    maximum = np.max(np.abs(blocks), axis=-1)
    scale = np.where(maximum == 0.0, 1.0, maximum / 127.0).astype(np.float32)
    quantized = np.clip(
        np.rint(blocks / scale[..., None]), -127, 127).astype(np.int8)
    stored_scale = scale.astype(np.float16).astype(np.float32)
    error = quantized.astype(np.float32) * stored_scale[..., None] - blocks
    error64 = error.astype(np.float64)
    block64 = blocks.astype(np.float64)
    error_sq += float(np.sum(error64 * error64))
    signal_sq += float(np.sum(block64 * block64))
    maximum_abs_error = max(maximum_abs_error, float(np.max(np.abs(error))))
    values += int(error.size)
  del source
  return {
      "relative_l2": (error_sq / signal_sq) ** 0.5,
      "rmse": (error_sq / values) ** 0.5,
      "maximum_abs_error": maximum_abs_error,
      "values": values,
  }


def state_bytes(key_group: int, value_group: int) -> dict[str, int]:
  hot_kv = HOT * KV_HEADS * HEAD_DIM * F16_BYTES * 2
  cold_k_i8 = COLD * KV_HEADS * HEAD_DIM * I8_BYTES
  cold_v_i8 = COLD * KV_HEADS * HEAD_DIM * I8_BYTES
  cold_k_scales = COLD * KV_HEADS * (HEAD_DIM // key_group) * F16_BYTES
  cold_v_scales = COLD * KV_HEADS * (HEAD_DIM // value_group) * F16_BYTES
  return {
      "hot_kv": hot_kv,
      "cold_k_i8": cold_k_i8,
      "cold_v_i8": cold_v_i8,
      "cold_k_scales": cold_k_scales,
      "cold_v_scales": cold_v_scales,
      "total": (
          hot_kv + cold_k_i8 + cold_v_i8 + cold_k_scales +
          cold_v_scales),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      GROUP4_BOUND, GROUP4_COMPONENT, GROUP4_CORRECTNESS, GROUP4_PROFILE,
      STATE_WORKER, SOURCE, RUNNER)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing K2/V4 bound inputs: " + ", ".join(missing))

  git = git_state(output)
  group4_bound = load_json(GROUP4_BOUND)
  group4_component = load_json(GROUP4_COMPONENT)
  group4_correctness = load_json(GROUP4_CORRECTNESS)
  group4_profile = load_json(GROUP4_PROFILE)
  state_worker = load_json(STATE_WORKER)
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER.read_text(encoding="utf-8")
  sample_memory("after-evidence-load", stop_bytes, memory)

  saved = state_worker["phases"][1]["saved_states"]
  key_path, key_shape = state_path(saved, "key")
  value_path, value_shape = state_path(saved, "value")
  key_group2 = quantization_audit(key_path, key_shape, KEY_GROUP)
  sample_memory("after-key-group2-audit", stop_bytes, memory)
  value_group2 = quantization_audit(value_path, value_shape, KEY_GROUP)
  sample_memory("after-value-group2-audit", stop_bytes, memory)

  group4_row = group4_bound["pareto"]["4"]
  group4_ucb = float(
      group4_component["performance_inference"]["upper_confidence_bound_ms"])
  group4_bytes = state_bytes(4, 4)
  hybrid_bytes = state_bytes(KEY_GROUP, VALUE_GROUP)
  both2_bytes = state_bytes(2, 2)
  hybrid_scaled_ucb = group4_ucb * hybrid_bytes["total"] / group4_bytes["total"]
  both2_scaled_ucb = group4_ucb * both2_bytes["total"] / group4_bytes["total"]
  profile_inference = group4_profile["group4_integration_inference"]
  profile_failures = {
      row["name"]: row for row in group4_profile["checks"]
      if row.get("pass") is False}
  source_layout = {
      "physical_key_pack_is_four_dimensions":
          "IQ36_COLD_K_WORDS_PER_HEAD" in source_text
          and "(lane >> 2U)" in source_text
          and "(lane & 3U)" in source_text,
      "current_source_conflates_scale_and_pack_groups":
          "#define IQ36_WORDS_PER_GROUP (IQ36_QUANT_GROUP / 4U)" in
          source_text,
      "runner_conflates_scale_and_pack_groups":
          "constexpr cl_uint kWordsPerGroup = kQuantGroup / 4U;" in
          runner_text,
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("group4_standalone_component_is_clean_and_promoted",
            group4_component.get("required_checks_passed") is True
            and group4_component.get("component_promoted") is True
            and group4_component.get("graph_integration_admitted") is False
            and group4_component.get("result", {}).get("quant_group") == 4
            and group4_ucb == 0.487186),
      check("group4_one_layer_32k_state_semantics_are_exact",
            group4_correctness.get("required_checks_passed") is True
            and group4_correctness.get("direct_i8_group4_full_cold") is True
            and group4_correctness.get("direct_i8_quant_group") == 4
            and group4_correctness.get("target_layers") == [3]),
      check("group4_integrated_profile_is_closed_on_latency_and_kld",
            group4_profile.get("attribution_checks_passed") is False
            and group4_profile.get("carrier_admission_passed") is True
            and profile_inference.get("rate_pass") is False
            and profile_inference.get("upper_confidence_bound_ms") == 1.010833
            and group4_profile.get("group4_max_kld") ==
                0.005622454731719582
            and set(profile_failures) == {
                "group4_integrated_decode_ucb_clears_component_cap",
                "group4_all_profile_distributions_pass"},
            profile_inference=profile_inference,
            max_kld=group4_profile.get("group4_max_kld")),
      check("saved_real_32k_state_identity_is_exact",
            key_shape == value_shape == (1, KV_HEADS, CONTEXT + 1, HEAD_DIM)
            and key_path.stat().st_size == value_path.stat().st_size ==
                67_110_912,
            key_path=display_path(key_path), key_sha256=sha256(key_path),
            value_path=display_path(value_path), value_sha256=sha256(value_path)),
      check("group2_key_materially_improves_real_state_error",
            key_group2["relative_l2"] <=
                0.70 * float(group4_row["key"]["relative_l2"]),
            group2=key_group2, group4=group4_row["key"]),
      check("hybrid_byte_scaled_component_ucb_remains_below_cap",
            hybrid_bytes["total"] == 60_817_408
            and hybrid_scaled_ucb < COMPONENT_CAP_MS,
            state_bytes=hybrid_bytes, scaled_ucb_ms=hybrid_scaled_ucb,
            cap_ms=COMPONENT_CAP_MS,
            margin_ms=COMPONENT_CAP_MS - hybrid_scaled_ucb),
      check("both_group2_byte_scaled_component_ucb_is_rejected",
            both2_bytes["total"] == 67_108_864
            and both2_scaled_ucb > COMPONENT_CAP_MS,
            state_bytes=both2_bytes, scaled_ucb_ms=both2_scaled_ucb,
            cap_ms=COMPONENT_CAP_MS),
      check("key_scale_group_must_be_decoupled_from_dim4_pack",
            all(source_layout.values()), source_layout=source_layout),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed
  verdict = (
      "admit_one_direct_i8_hybrid_k2_v4_component"
      if admitted else "reject_direct_i8_hybrid_before_source")

  selected_component = {
      "key_quant_group": KEY_GROUP,
      "value_quant_group": VALUE_GROUP,
      "key_pack_dimensions": PACK_DIMS,
      "context_tokens": CONTEXT,
      "logical_hot_tokens": HOT,
      "logical_cold_tokens": COLD,
      "cold_policy": "attend_the_full_logical_cold_prefix",
      "cold_k_layout": "token16_dim4_packed_i8_group2_fp16_scale",
      "cold_v_layout": "dimension_major_i8_group4_fp16_scale",
      "hot_k_layout": "token16_dim2_packed_f16",
      "hot_v_layout": "dimension_major_token16_f16",
      "state_bytes": hybrid_bytes["total"],
      "maximum_one_sided_95_latency_bound_ms": COMPONENT_CAP_MS,
      "warmup_samples": 5,
      "measured_samples": 20,
      "build_parallelism": 1,
      "serial_gpu_worker": True,
      "memory_stop_bytes": stop_bytes,
      "tile_or_workgroup_sweep_allowed": False,
      "graph_integration_admitted": False,
      "full_model_worker_admitted": False,
      "long_worker_admitted": False,
      "passing_action": (
          "admit one source-only one-layer integration gate, then one 2k "
          "and one 32k teacher-forced correctness worker"),
      "failing_action": (
          "close K2/V4 and do not run an integrated or product worker"),
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
      "closed_group4_integration": {
          "one_layer_ucb_ms": profile_inference.get(
              "upper_confidence_bound_ms"),
          "component_cap_ms": COMPONENT_CAP_MS,
          "max_kld": group4_profile.get("group4_max_kld"),
      },
      "real_state_error": {
          "key_group2": key_group2,
          "key_group4": group4_row["key"],
          "key_group2_vs_group4": (
              key_group2["relative_l2"] /
              float(group4_row["key"]["relative_l2"])),
          "value_group2": value_group2,
          "value_group4": group4_row["value"],
      },
      "byte_scaled_timing": {
          "basis_group4_measured_ucb_ms": group4_ucb,
          "group4": {"state_bytes": group4_bytes, "scaled_ucb_ms": group4_ucb},
          "hybrid_k2_v4": {
              "state_bytes": hybrid_bytes,
              "scaled_ucb_ms": hybrid_scaled_ucb,
              "margin_below_cap_ms": COMPONENT_CAP_MS - hybrid_scaled_ucb,
          },
          "both_k2_v2": {
              "state_bytes": both2_bytes,
              "scaled_ucb_ms": both2_scaled_ucb,
              "margin_below_cap_ms": COMPONENT_CAP_MS - both2_scaled_ucb,
          },
      },
      "source_layout": source_layout,
      "selected_component": selected_component,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Direct-I8 K2/V4 hybrid source bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The current integrated K4/V4 layer is closed: its warm one-layer decode UCB
is `{profile_inference.get('upper_confidence_bound_ms')} ms` against the exact
`{COMPONENT_CAP_MS} ms` cap, and its maximum teacher-forced KLD is
`{group4_profile.get('group4_max_kld')}`.  The standalone K4/V4 component
remains a valid measured basis, not product evidence.

On the saved real 32k layer-3 state, group-2 cold K reaches relative L2
`{key_group2['relative_l2']:.12f}` versus
`{float(group4_row['key']['relative_l2']):.12f}` for group 4
(`{key_group2['relative_l2'] / float(group4_row['key']['relative_l2']):.6f}x`).
Keeping V at group 4 yields `{hybrid_bytes['total']}` logical state bytes.
Charging those bytes to the measured group-4 UCB gives
`{hybrid_scaled_ucb:.9f} ms`, only
`{COMPONENT_CAP_MS - hybrid_scaled_ucb:.9f} ms` below the cap.  K2/V2 scales
to `{both2_scaled_ucb:.9f} ms` and is rejected before source or compile.

This admits exactly one fixed K2/V4 standalone component.  K remains physically
packed four dimensions per word while its F16 scales cover two dimensions;
the implementation must therefore separate scale grouping from byte packing.
It permits no sweep, uses five warmups plus twenty measured samples, serial
GPU execution, `-j1`, and the 4-GiB available-memory stop.  Model integration,
long workers, product rows, and speed claims remain blocked.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "key_group2_relative_l2": key_group2["relative_l2"],
      "hybrid_scaled_ucb_ms": hybrid_scaled_ucb,
      "both2_scaled_ucb_ms": both2_scaled_ucb,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
