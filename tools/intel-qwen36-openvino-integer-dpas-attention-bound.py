#!/usr/bin/env python3
"""Bound integer-DPAS cold QK before any source, compiler, or GPU work.

The gate consumes the clean real OpenVINO query/key boundary capture and the
already measured hot16k K2/V4 split component.  It builds an offline symmetric
I8 error ruler, accounts for the per-group scale separability required by a
K32 integer DPAS, and charges every retained attention stage.  It never creates
an OpenCL context, compiles a kernel, or starts a model worker.
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


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-integer-dpas-attention-bound-v0"
CURRENT_ROUTE = "openvino_integer_dpas_attention_arithmetic_bound"
NEXT_ROUTE = "openvino_locked_target_infeasibility_record"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
CAPTURE = ROOT / (
    "output/openvino-hot-cold-attention-"
    "20260714Tseq848-single-owner-capture-cleanZ")
CAPTURE_MANIFEST = CAPTURE / "manifest.json"
SPLIT_COMPONENT = ROOT / (
    "output/openvino-split-state-owner-hot16k-k2-v4-attention-component-"
    "20260715Tseq1275-cleanZ/result.json")
REFLECTION = ROOT / (
    "output/openvino-route-exhaustion-reflection-"
    "20260715Tseq1276-cleanZ/metrics.json")
GROUP32_CORRECTNESS = ROOT / (
    "output/openvino-direct-i8-product-"
    "20260715Tseq1256-all10-32k-o45-divergence-cleanZ/correctness.json")
PARTIAL_SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
INTEGER_DPAS_SOURCE = ROOT / "engine/gpu/opencl/q6_splitplane_dpas.cl"

TARGET_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
LANE_PHASES = {"2k": 2, "8k": 3}
MODES = ("stock", "candidate")
HEAD_DIM = 256
Q_HEADS = 16
KV_HEADS = 2
GQA_GROUP = Q_HEADS // KV_HEADS
CURRENT_KEY_GROUP = 2
F16_DPAS_K = 16
INTEGER_DPAS_K = 32
GROUPS = (2, 4, 8, 16, 32, 64, 128, 256)
NUMERIC_COSINE_MIN = 0.999
NUMERIC_RELATIVE_L2_MAX = 0.002
CAP_MS = 0.5618915
GROUP32_ROUTE = "openvino_all_ten_fixed_direct_i8_group32_product_v28v"


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


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


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
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
  denominator = float(np.linalg.norm(reference))
  numerator = float(np.linalg.norm(actual - reference))
  return numerator / max(denominator, 1.0e-30)


def cosine(actual: np.ndarray, reference: np.ndarray) -> float:
  denominator = float(np.linalg.norm(actual) * np.linalg.norm(reference))
  return float(np.dot(actual.reshape(-1), reference.reshape(-1))) / max(
      denominator, 1.0e-30)


def symmetric_i8_dequantize(values: np.ndarray, group: int) -> np.ndarray:
  shaped = values.reshape(values.shape[0], -1, group)
  scales = np.max(np.abs(shaped), axis=2, keepdims=True) / 127.0
  divisors = np.where(scales == 0.0, 1.0, scales)
  quantized = np.clip(np.rint(shaped / divisors), -127.0, 127.0)
  return (quantized * scales).reshape(values.shape)


def capture_rows() -> tuple[list[dict[str, Any]], list[Path]]:
  rows: list[dict[str, Any]] = []
  bound_files: list[Path] = []
  for lane, expected_phases in LANE_PHASES.items():
    for mode in MODES:
      worker_path = CAPTURE / "raw" / lane / mode / "worker-result.json"
      worker = load_json(worker_path)
      bound_files.append(worker_path)
      if worker.get("lane") != lane or worker.get("mode") != mode:
        raise ValueError(f"worker lane/mode mismatch: {worker_path}")
      phases = worker.get("phases", [])
      if len(phases) != expected_phases:
        raise ValueError(f"unexpected phase count: {worker_path}")
      for phase_index, phase in enumerate(phases):
        outputs = phase.get("attention_outputs", {})
        if sorted(int(layer) for layer in outputs) != TARGET_LAYERS:
          raise ValueError(
              f"unexpected attention layer set: {worker_path} phase {phase_index}")
        for layer in TARGET_LAYERS:
          tensors = outputs[str(layer)]
          query_meta = tensors["query"]
          key_meta = tensors["key"]
          query_path = ROOT / query_meta["path"]
          key_path = ROOT / key_meta["path"]
          if sha256(query_path) != query_meta["sha256"]:
            raise ValueError(f"query hash mismatch: {query_path}")
          if sha256(key_path) != key_meta["sha256"]:
            raise ValueError(f"key hash mismatch: {key_path}")
          if query_meta["shape"] != [1, Q_HEADS, HEAD_DIM]:
            raise ValueError(f"unexpected query shape: {query_path}")
          expected_key_heads = Q_HEADS if mode == "stock" else KV_HEADS
          if key_meta["shape"] != [1, expected_key_heads, HEAD_DIM]:
            raise ValueError(f"unexpected key shape: {key_path}")
          query = np.fromfile(query_path, dtype="<f4").reshape(
              Q_HEADS, HEAD_DIM).astype(np.float64)
          key = np.fromfile(key_path, dtype="<f4").reshape(
              expected_key_heads, HEAD_DIM).astype(np.float64)
          if expected_key_heads == KV_HEADS:
            key = key[np.arange(Q_HEADS) // GQA_GROUP]
          if not np.isfinite(query).all() or not np.isfinite(key).all():
            raise ValueError(f"non-finite capture: {query_path} / {key_path}")
          rows.append({
              "id": f"{lane}/{mode}/phase{phase_index}/layer{layer}",
              "lane": lane,
              "mode": mode,
              "phase": phase_index,
              "layer": layer,
              "query": query,
              "key": key,
              "query_path": display_path(query_path),
              "key_path": display_path(key_path),
          })
  return rows, bound_files


def worst(
    values: list[tuple[float, str]], *, minimum: bool = False,
) -> dict[str, Any]:
  value, row_id = (min(values) if minimum else max(values))
  return {"value": value, "row": row_id}


def build_ruler(rows: list[dict[str, Any]]) -> dict[str, Any]:
  schemes: dict[str, Any] = {}
  for group in GROUPS:
    query_relative: list[tuple[float, str]] = []
    query_cosine: list[tuple[float, str]] = []
    score_relative: list[tuple[float, str]] = []
    score_cosine: list[tuple[float, str]] = []
    score_max_abs: list[tuple[float, str]] = []
    for row in rows:
      query = row["query"]
      key = row["key"]
      query_dequantized = symmetric_i8_dequantize(query, group)
      key_dequantized = symmetric_i8_dequantize(key, group)
      reference_scores = np.sum(query * key, axis=1)
      integer_scores = np.sum(
          query_dequantized * key_dequantized, axis=1)
      row_id = row["id"]
      query_relative.append((
          relative_l2(query_dequantized, query), row_id))
      query_cosine.append((cosine(query_dequantized, query), row_id))
      score_relative.append((
          relative_l2(integer_scores, reference_scores), row_id))
      score_cosine.append((cosine(integer_scores, reference_scores), row_id))
      score_max_abs.append((
          float(np.max(np.abs(integer_scores - reference_scores))), row_id))

    qk_relative = worst(score_relative)
    qk_cosine = worst(score_cosine, minimum=True)
    numeric_proxy_pass = (
        qk_relative["value"] <= NUMERIC_RELATIVE_L2_MAX
        and qk_cosine["value"] >= NUMERIC_COSINE_MIN)
    scale_segments = HEAD_DIM // group
    # A group below K32 needs one zero-padded call per independently scaled
    # segment.  A group at or above K32 still needs one call per K32 slice.
    integer_dpas_calls = HEAD_DIM // min(group, INTEGER_DPAS_K)
    current_f16_dpas_calls = HEAD_DIM // F16_DPAS_K
    schemes[str(group)] = {
        "scale_group": group,
        "captured_rows": len(rows),
        "query_relative_l2_max": worst(query_relative),
        "query_cosine_min": worst(query_cosine, minimum=True),
        "qk_relative_l2_max": qk_relative,
        "qk_cosine_min": qk_cosine,
        "qk_max_abs_max": worst(score_max_abs),
        "numeric_proxy_relative_l2_max": NUMERIC_RELATIVE_L2_MAX,
        "numeric_proxy_cosine_min": NUMERIC_COSINE_MIN,
        "numeric_proxy_pass": numeric_proxy_pass,
        "independent_scale_segments_per_score": scale_segments,
        "scale_separable_k32_integer_dpas_calls_per_score_tile":
            integer_dpas_calls,
        "current_k16_f16_dpas_calls_per_score_tile": current_f16_dpas_calls,
        "integer_to_current_dpas_call_ratio": (
            integer_dpas_calls / current_f16_dpas_calls),
        "strictly_fewer_dpas_calls": (
            integer_dpas_calls < current_f16_dpas_calls),
    }
  return schemes


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      STATUS, ROUTES, REJECTED, MODEL_CONTRACT, CAPTURE_MANIFEST,
      SPLIT_COMPONENT, REFLECTION, GROUP32_CORRECTNESS, PARTIAL_SOURCE,
      INTEGER_DPAS_SOURCE)
  missing = [display_path(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing integer-DPAS bound inputs: " + ", ".join(missing))

  git = git_state(output)
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  model_contract = load_json(MODEL_CONTRACT)
  manifest = load_json(CAPTURE_MANIFEST)
  split = load_json(SPLIT_COMPONENT)
  reflection = load_json(REFLECTION)
  group32_correctness = load_json(GROUP32_CORRECTNESS)
  status_text = STATUS.read_text(encoding="utf-8")
  partial_text = PARTIAL_SOURCE.read_text(encoding="utf-8")
  integer_text = INTEGER_DPAS_SOURCE.read_text(encoding="utf-8")
  rows, capture_files = capture_rows()
  ruler = build_ruler(rows)
  sample_memory("after-offline-ruler", stop_bytes, memory)

  architecture = model_contract["product_model"]["architecture"]
  group32_rejection = next(
      (row for row in rejected.get("rejected", [])
       if row.get("route") == GROUP32_ROUTE), None)
  group32_case = group32_correctness["cases"][0]
  source_shape = {
      "head_dim_256": "#define IQ36_HEAD_DIM 256U" in partial_text,
      "current_key_group_accepts_2": (
          "IQ36_KEY_QUANT_GROUP != 2" in partial_text),
      "current_cold_key_is_signed_i8": (
          "__global const uint* cold_k" in partial_text
          and "convert_half16(quantized) * scales" in partial_text),
      "current_k2_loads_eight_scales_per_k16": all(
          f"const half scale{index} = IQ36_LOAD_KEY_SCALE({index}U);"
          in partial_text for index in range(8)),
      "current_score_uses_f16_k16_dpas": (
          "intel_sub_group_f16_f16_matrix_mad_k16" in partial_text),
      "registered_integer_intrinsic_is_i8_u8_k32": (
          "intel_sub_group_i8_u8_matrix_mad_k32" in integer_text),
  }

  accurate_groups = [
      int(group) for group, row in ruler.items()
      if row["numeric_proxy_pass"]]
  fewer_call_groups = [
      int(group) for group, row in ruler.items()
      if row["strictly_fewer_dpas_calls"]]
  admissible_groups = sorted(set(accurate_groups) & set(fewer_call_groups))
  current_scheme = ruler[str(CURRENT_KEY_GROUP)]

  latency = reflection["latency_accounting"]
  partial_ucb = float(latency["partial"]["upper_confidence_bound_ms"])
  reduce_ucb = float(latency["reduce"]["upper_confidence_bound_ms"])
  update_ucb = float(latency["update"]["upper_confidence_bound_ms"])
  total_ucb = float(latency["total"]["upper_confidence_bound_ms"])
  additive_projection = partial_ucb + reduce_ucb + update_ucb
  timing = {
      "component_cap_ms": CAP_MS,
      "measured_total_ucb_ms": total_ucb,
      "measured_partial_ucb_ms": partial_ucb,
      "measured_reduce_ucb_ms": reduce_ucb,
      "measured_update_ucb_ms": update_ucb,
      "required_total_cut_ms": total_ucb - CAP_MS,
      "partial_budget_with_measured_tail_ms": CAP_MS - reduce_ucb - update_ucb,
      "free_query_quantization_ms": 0.0,
      "free_reduce_and_update_projection_ms": partial_ucb,
      "free_reduce_and_update_miss_ms": partial_ucb - CAP_MS,
      "unchanged_stage_additive_projection_ms": additive_projection,
      "unchanged_stage_additive_miss_ms": additive_projection - CAP_MS,
      "overlap_credit_ms": 0.0,
      "stage_accounting": {
          "query_quantization": (
              "new serial work; charged as zero in the optimistic rejection"),
          "cold_qk": (
              "K2 has 128 independently scaled dot2 segments; exact K32 "
              "integer accumulation needs 128 isolated/zero-padded calls "
              "versus 16 current K16 F16 calls, before scale and signed-K "
              "bias correction"),
          "hot_qk": "unchanged inside the measured partial UCB",
          "softmax": "unchanged inside the measured partial UCB",
          "pv": "unchanged inside the measured partial UCB",
          "workspace": "unchanged inside the measured partial UCB",
          "reduce": "measured UCB charged separately",
          "update": "measured UCB charged separately",
      },
  }

  expected_rows = sum(LANE_PHASES.values()) * len(MODES) * len(TARGET_LAYERS)
  no_runtime_evidence = not any(
      (output / name).exists()
      for name in ("run.json", "probe.json", "worker.time", "compile.log"))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("active_gate_selects_integer_dpas_source_bound",
            routes.get("active_route", {}).get("id") == CURRENT_ROUTE
            and "source-only integer-DPAS attention arithmetic bound" in
                status_text),
      check("locked_model_architecture_is_exact",
            architecture.get("full_attention_layers") == 10
            and architecture.get("attention_heads") == Q_HEADS
            and architecture.get("kv_heads") == KV_HEADS
            and architecture.get("head_dim") == HEAD_DIM,
            architecture=architecture),
      check("clean_real_openvino_boundary_capture_is_bound",
            manifest.get("dirty") is False
            and manifest.get("model_dir") == "/home/intel/Qwen3.6-35B-A3B-ov"
            and manifest.get("target_layers") == TARGET_LAYERS
            and manifest.get("commit") ==
                "11f2f669724aa176aa8585fe37a83b12272bf9f3",
            manifest=manifest),
      check("offline_ruler_covers_all_real_rows",
            len(rows) == expected_rows == 100
            and {row["lane"] for row in rows} == set(LANE_PHASES)
            and {row["mode"] for row in rows} == set(MODES)
            and {row["layer"] for row in rows} == set(TARGET_LAYERS),
            observed_rows=len(rows), expected_rows=expected_rows),
      check("fine_scale_numeric_proxy_passes_but_has_no_dpas_cut",
            accurate_groups == [2, 4]
            and current_scheme["numeric_proxy_pass"] is True
            and current_scheme[
                "scale_separable_k32_integer_dpas_calls_per_score_tile"] == 128
            and current_scheme["integer_to_current_dpas_call_ratio"] == 8.0,
            accurate_groups=accurate_groups, current_scheme=current_scheme),
      check("every_lower_call_integer_scheme_fails_real_qk_proxy",
            fewer_call_groups == [32, 64, 128, 256]
            and not admissible_groups
            and all(not ruler[str(group)]["numeric_proxy_pass"]
                    for group in fewer_call_groups),
            fewer_call_groups=fewer_call_groups,
            admissible_groups=admissible_groups),
      check("current_source_scale_separability_blocks_one_k32_accumulator",
            all(source_shape.values())
            and HEAD_DIM // CURRENT_KEY_GROUP == 128
            and INTEGER_DPAS_K // CURRENT_KEY_GROUP == 16,
            source_shape=source_shape,
            current_scale_segments_per_score=HEAD_DIM // CURRENT_KEY_GROUP,
            current_scale_segments_per_integer_dpas=
                INTEGER_DPAS_K // CURRENT_KEY_GROUP),
      check("group32_compatible_codec_is_already_closed",
            group32_rejection is not None
            and group32_rejection.get("reopen_condition", "").startswith(
                "none for another output length")
            and group32_case.get("required_checks_passed") is False
            and group32_case.get("kld_max") == 0.15350662840040744
            and group32_case.get("top1_rate") == 0.6
            and ruler["32"]["numeric_proxy_pass"] is False,
            rejection=group32_rejection,
            offline_group32=ruler["32"]),
      check("complete_optimistic_timing_cannot_clear_cap",
            split.get("performance_inference", {}).get(
                "upper_confidence_bound_ms") == total_ucb == 0.587707
            and partial_ucb == 0.576354
            and partial_ucb > CAP_MS
            and additive_projection > CAP_MS
            and not admissible_groups,
            timing=timing),
      check("gate_created_no_runtime_or_compile_evidence",
            no_runtime_evidence),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_admitted = False
  verdict = (
      "reject_integer_dpas_attention_before_source"
      if required_checks_passed else "integer_dpas_bound_audit_failed")

  all_inputs = list(required) + capture_files
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "route_admitted": route_admitted,
      "source_edit_admitted": False,
      "compile_admitted": False,
      "gpu_worker_admitted": False,
      "model_worker_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "speedup_claim_allowed": False,
      "selected_next_route": NEXT_ROUTE if required_checks_passed else CURRENT_ROUTE,
      "capture_coverage": {
          "rows": len(rows),
          "lanes": LANE_PHASES,
          "modes": list(MODES),
          "layers": TARGET_LAYERS,
          "query_shape": [1, Q_HEADS, HEAD_DIM],
          "candidate_key_shape": [1, KV_HEADS, HEAD_DIM],
          "stock_key_shape": [1, Q_HEADS, HEAD_DIM],
      },
      "offline_quantization_ruler": {
          "method": (
              "symmetric signed-I8 maxabs/127 per group on real F32 boundary "
              "values; scales retained in float64, making this optimistic "
              "relative to F16 scale storage"),
          "role": "source-selection proxy only; not component or token correctness",
          "schemes": ruler,
          "accurate_groups": accurate_groups,
          "strictly_lower_dpas_call_groups": fewer_call_groups,
          "admissible_groups": admissible_groups,
      },
      "source_shape": source_shape,
      "arithmetic_bound": {
          "current_key_scale_group": CURRENT_KEY_GROUP,
          "head_dim": HEAD_DIM,
          "current_f16_dpas_k": F16_DPAS_K,
          "registered_integer_dpas_k": INTEGER_DPAS_K,
          "current_k2_scale_segments_per_score":
              HEAD_DIM // CURRENT_KEY_GROUP,
          "current_f16_dpas_calls_per_score_tile": HEAD_DIM // F16_DPAS_K,
          "exact_k2_integer_dpas_calls_per_score_tile":
              HEAD_DIM // CURRENT_KEY_GROUP,
          "exact_k2_integer_to_f16_call_ratio": (
              F16_DPAS_K / CURRENT_KEY_GROUP),
          "signed_k_bias_correction_required_by_registered_i8_u8_intrinsic": True,
          "conclusion": (
              "no tested scale group both clears the real QK proxy and uses "
              "fewer DPAS calls than the current F16-dequantized score path"),
      },
      "timing_bound": timing,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in all_inputs},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# Integer-DPAS attention arithmetic bound

Verdict: **{verdict}**. Required audit checks:
`{str(required_checks_passed).lower()}`. No compiler, OpenCL context, GPU, or
model worker ran.

The clean real-boundary ruler covers `{len(rows)}` query/key rows: both stock
and candidate at 2k/8k, every captured phase, and all ten full-attention
layers. Group2 and group4 clear the optimistic QK proxy; every group that would
reduce the DPAS call count (32/64/128/256) fails it. Group32 reaches worst QK
relative L2 `{ruler['32']['qk_relative_l2_max']['value']:.12f}` versus
`{NUMERIC_RELATIVE_L2_MAX}`, before the already recorded long group32 failure
at KLD `0.15350662840040744` and top-1 rate `0.6`.

Keeping the accurate K2 codec leaves 128 independently scaled dot2 segments
per score. A K32 integer DPAS cannot combine segments with different scales;
an exact isolated decomposition needs 128 integer calls versus 16 current K16
F16 calls, before query quantization, scale application, and signed-K bias
correction. Therefore no cold-QK cut is independently bounded.

Even granting query quantization, reduce, and update for free leaves the
measured partial UCB at `{partial_ucb} ms`, above the exact `{CAP_MS} ms` full
component cap. Charging the unchanged measured tail gives
`{additive_projection:.9f} ms`. Source, compilation, and workers remain
blocked; the next canonical action is the locked-target infeasibility record,
which is not project completion and does not relax acceptance.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "captured_rows": len(rows),
      "admissible_groups": admissible_groups,
      "free_tail_projection_ms": partial_ucb,
      "cap_ms": CAP_MS,
      "compiler_launched": False,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
