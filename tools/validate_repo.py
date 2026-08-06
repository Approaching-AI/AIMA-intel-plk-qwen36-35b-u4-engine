#!/usr/bin/env python3
"""Validate the day-0 repository scaffold."""

from __future__ import annotations

import json
import hashlib
import re
import statistics
import subprocess
import sys
from pathlib import Path

from iq36_perf_inference import latency_cap_inference, paired_speedup_inference


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "AGENTS.md",
    ".meta-agent/AGENT-RUNTIME.md",
    ".gitmodules",
    "meta-agent",
    "contracts/intel-qwen36-target-contract.json",
    "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json",
    "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json",
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json",
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompt-suites.json",
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/README.md",
    "oracle/oracle-bundle-contract.json",
    "doc/README.md",
    "doc/WORKSTREAMS.md",
    "doc/adr/0001-r0-262144-denominator-unavailable.md",
    "doc/adr/0002-r0-256k-prompt-edge-topk-policy.md",
    "doc/adr/0003-surrogate-refine-splitplane-dual-phase-engine.md",
    "doc/adr/0004-close-direct-q6-q5-select-boundary-attribution.md",
    "doc/adr/0005-close-derived-q5-select-exact-q6-sparse-exception-gate.md",
    "doc/adr/0006-close-sparse-exception-q6-select-grouped-prefill-census.md",
    "doc/adr/0007-close-64token-grouping-select-1024token-expert-buckets.md",
    "doc/adr/0070-adopt-openvino-u4-specialization-runtime.md",
    "doc/active/intel-qwen36-35b-a3b-gguf-q4km/intel-qwen36-35b-a3b-gguf-q4km-openvino-specialization-roadmap-2026-07-13.md",
    "goals/intel-qwen36-35b-a3b-q4km-engine.md",
    "tools/intel-qwen36-r0-recapture.py",
    "tools/iq36_local.py",
    "tools/intel-qwen36-r0-reference-artifact-audit.py",
    "tools/intel-qwen36-r0-performance-artifact-audit.py",
    "tools/iq36_perf_inference.py",
    "tools/intel-qwen36-r0-target-denominator-preflight.py",
    "tools/intel-qwen36-r0-materialize-262144-prompt.py",
    "tools/intel-qwen36-r0-openvino-denominator-run.py",
    "tools/intel-qwen36-r0-openvino-denominator-matrix.py",
    "tools/intel-qwen36-r0-source-stream-roof-run.py",
    "tools/intel-qwen36-r0-qmatvec-probe-run.py",
    "tools/intel-qwen36-r0-kv-read-pressure.py",
    "tools/intel-qwen36-r0-route-feasibility.py",
    "tools/intel-qwen36-r0-denominator-oracle-boundary-resolution.py",
    "tools/intel-qwen36-r0-denominator-unavailable-policy.py",
    "tools/intel-qwen36-r0-llama-denominator-preflight.py",
    "tools/intel-qwen36-r0-llama-denominator-run.py",
    "tools/intel-qwen36-r0-oracle-capture-spec.py",
    "tools/intel-qwen36-r0-oracle-runtime-preflight.py",
    "tools/intel-qwen36-r0-boundary-capture-route-preflight.py",
    "tools/intel-qwen36-r0-llama-source-build-route.py",
    "tools/intel-qwen36-r0-llama-instrumentation-map.py",
    "tools/intel-qwen36-r0-boundary-capture-instrumentation-patch.py",
    "tools/intel-qwen36-r0-boundary-capture-build.py",
    "tools/intel-qwen36-r0-boundary-capture-run.py",
    "tools/intel-qwen36-r0-boundary-capture-coverage.py",
    "tools/intel-qwen36-r0-boundary-bundle-fragment-assemble.py",
    "tools/intel-qwen36-r0-distribution-capture-smoke.py",
    "tools/intel-qwen36-r0-distribution-capture-short-router.py",
    "tools/intel-qwen36-r0-distribution-capture-materialized.py",
    "tools/intel-qwen36-r0-oracle-capture-queue.py",
    "tools/intel-qwen36-r0-oracle-prompt-materialize.py",
    "tools/intel-qwen36-r0-oracle-token-id-capture.py",
    "tools/intel-qwen36-r0-oracle-topk-smoke.py",
    "tools/intel-qwen36-r0-oracle-256k-prompt-edge-policy.py",
    "tools/intel-qwen36-r0-oracle-bundle-assemble.py",
    "tools/intel-qwen36-r0-oracle-bundle-validate.py",
    "tools/intel-qwen36-r0-resident-harness-load.py",
    "tools/intel-qwen36-r0-resident-harness-gate-audit.py",
    "tools/intel-qwen36-r1-native-correctness-gate.py",
    "tools/intel-qwen36-r1-native-candidate-route.py",
    "tools/intel-qwen36-r1-native-candidate-jsonl.py",
    "tools/intel-qwen36-post-r1-resident-timed.py",
    "tools/intel-qwen36-lm-head-topk-thread-sweep.py",
    "tools/intel-qwen36-lm-head-q4-surrogate-gate.py",
    "tools/intel-qwen36-router-i8-surrogate-gate.py",
    "tools/intel-qwen36-q6-splitplane-dpas-gate.py",
    "tools/intel-qwen36-q6-sparse-exception-feasibility-gate.py",
    "tools/intel-qwen36-prefill-router-shape-census-gate.py",
    "tools/intel-qwen36-q5-surrogate-feasibility-gate.py",
    "tools/intel-qwen36-q5-boundary-attribution-gate.py",
    "tools/intel-qwen36-selected-gate-q4-thread-sweep.py",
    "tools/intel-qwen36-r1-native-gguf-load-map.py",
    "tools/intel-qwen36-r1-engine-gguf-inspect.py",
    "tools/intel-qwen36-r1-engine-embedding-compare.py",
    "tools/intel-qwen36-r1-engine-seed-prompt-input-check.py",
    "tools/intel-qwen36-r1-engine-rmsnorm-compare.py",
    "tools/intel-qwen36-r1-engine-qkv-compare.py",
    "tools/intel-qwen36-r1-engine-attn-output-compare.py",
    "tools/intel-qwen36-r1-engine-linear-attn-preconv-compare.py",
    "tools/intel-qwen36-r1-engine-linear-attn-conv-compare.py",
    "tools/intel-qwen36-r1-engine-linear-attn-delta-compare.py",
    "tools/intel-qwen36-r1-engine-linear-attn-postconv-compare.py",
    "tools/intel-qwen36-r1-engine-linear-attn-all-postconv-compare.py",
    "tools/intel-qwen36-r1-engine-layer-postconv-compare.py",
    "tools/intel-qwen36-r1-engine-layer1-postconv-compare.py",
    "tools/intel-qwen36-r1-engine-layer-stateful-linear-attn-compare.py",
    "tools/intel-qwen36-r1-engine-two-linear-layers-stateful-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-qkv-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-rope-compare.py",
    "tools/intel-qwen36-r1-full-attn-history-capture.py",
    "tools/intel-qwen36-r1-full-attn-all-history-capture.py",
    "tools/intel-qwen36-r1-engine-full-attn-core-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-stateful-layer-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-all-stateful-layers-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-gate-compare.py",
    "tools/intel-qwen36-r1-engine-full-attn-output-projection-compare.py",
    "tools/intel-qwen36-r1-engine-attn-residual-compare.py",
    "tools/intel-qwen36-r1-engine-ffn-rmsnorm-compare.py",
    "tools/intel-qwen36-r1-engine-router-logits-compare.py",
    "tools/intel-qwen36-r1-engine-router-topk-compare.py",
    "tools/intel-qwen36-r1-engine-selected-expert-gate-up-compare.py",
    "tools/intel-qwen36-r1-engine-swiglu-compare.py",
    "tools/intel-qwen36-r1-engine-selected-expert-down-compare.py",
    "tools/intel-qwen36-r1-engine-shared-expert-compare.py",
    "tools/intel-qwen36-r1-engine-moe-residual-compare.py",
    "tools/intel-qwen36-r1-engine-ffn-block-compare.py",
    "tools/intel-qwen36-r1-engine-layer-shell-compare.py",
    "tools/intel-qwen36-r1-engine-loop-shell-compare.py",
    "tools/intel-qwen36-r1-engine-final-norm-compare.py",
    "tools/intel-qwen36-r1-engine-lm-head-compare.py",
    "tools/intel-qwen36-r1-engine-sampler-compare.py",
    "tools/intel-qwen36-oracle-seed-stage.py",
    "tools/intel-qwen36-oracle-seed-replay.py",
    "tools/intel-qwen36-teacher-forced-seed-stage.py",
    "tools/intel-qwen36-native-carrier-loop-gate.py",
    "tools/intel-qwen36-all-layer-prepack-feasibility-gate.py",
    "tools/intel-qwen36-reference-consensus-matrix.py",
    "tools/intel-qwen36-native-consensus-gate.py",
    "tools/intel-qwen36-product-decode-route-gate.py",
    "engine/CMakeLists.txt",
    "engine/include/intel_qwen36/gguf_loader.hpp",
    "engine/include/intel_qwen36/grouped_s8_u4_prefill_runtime.hpp",
    "engine/include/intel_qwen36/native_carrier_loop.hpp",
    "engine/include/intel_qwen36/resident_harness.hpp",
    "engine/src/gguf_loader.cpp",
    "engine/src/grouped_s8_u4_prefill_runtime.cpp",
    "engine/gpu/opencl/q6_splitplane_dpas.cl",
    "engine/tools/lm_head_q4_surrogate_gate.cpp",
    "engine/tools/grouped_s8_u4_prefill_runtime.cpp",
    "engine/tools/grouped_s8_u4_prefill_api_smoke.cpp",
    "engine/tools/grouped_s8_u4_prefill_resident_smoke.cpp",
    "engine/tools/grouped_s8_u4_prefill_multilayer_load_smoke.cpp",
    "engine/tools/grouped_s8_u4_prefill_schedule_envelope_smoke.cpp",
    "engine/tools/grouped_s8_u8_q6_surrogate_down.cpp",
    "engine/tools/grouped_s8_u8_q6_prefill_resident_smoke.cpp",
    "engine/tools/grouped_s8_u8_q6_prefill_schedule_envelope_smoke.cpp",
    "engine/tools/grouped_mixed_prefill_all_layer_load_smoke.cpp",
    "engine/tools/grouped_mixed_prefill_all_layer_compare.cpp",
    "tools/intel-qwen36-grouped-s8-u8-q6-prefill-gate.py",
    "tools/intel-qwen36-all-layer-mixed-prepack-load-gate.py",
    "tools/intel-qwen36-all-layer-mixed-component-gate.py",
    "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl",
    "engine/tools/native_carrier_loop_smoke.cpp",
    "engine/tools/q6_splitplane_dpas_gate.cpp",
    "engine/tools/router_i8_surrogate_gate.cpp",
    "engine/tools/q5_teacher_forced_boundary_capture.cpp",
    "engine/tools/q6_sparse_exception_feasibility.cpp",
    "engine/tests/embedding_compare.cpp",
    "engine/tests/seed_prompt_input_check.cpp",
    "engine/tests/gguf_inspect.cpp",
    "engine/tests/attn_output_compare.cpp",
    "engine/tests/linear_attn_preconv_compare.cpp",
    "engine/tests/linear_attn_conv_compare.cpp",
    "engine/tests/linear_attn_delta_compare.cpp",
    "engine/tests/linear_attn_postconv_compare.cpp",
    "engine/tests/linear_attn_all_postconv_compare.cpp",
    "engine/tests/layer_postconv_compare.cpp",
    "engine/tests/layer_stateful_linear_attn_compare.cpp",
    "engine/tests/two_linear_layers_stateful_compare.cpp",
    "engine/tests/full_attn_qkv_compare.cpp",
    "engine/tests/full_attn_rope_compare.cpp",
    "engine/tests/full_attn_core_compare.cpp",
    "engine/tests/full_attn_stateful_layer_compare.cpp",
    "engine/tests/full_attn_all_stateful_layers_compare.cpp",
    "engine/tests/full_attn_gate_compare.cpp",
    "engine/tests/full_attn_output_projection_compare.cpp",
    "engine/tests/attn_residual_compare.cpp",
    "engine/tests/ffn_rmsnorm_compare.cpp",
    "engine/tests/router_logits_compare.cpp",
    "engine/tests/router_topk_compare.cpp",
    "engine/tests/selected_expert_gate_up_compare.cpp",
    "engine/tests/swiglu_compare.cpp",
    "engine/tests/selected_expert_down_compare.cpp",
    "engine/tests/shared_expert_compare.cpp",
    "engine/tests/moe_residual_compare.cpp",
    "engine/tests/ffn_block_compare.cpp",
    "engine/tests/layer_shell_compare.cpp",
    "engine/tests/loop_shell_compare.cpp",
    "engine/tests/final_norm_compare.cpp",
    "engine/tests/lm_head_compare.cpp",
    "engine/tests/sampler_compare.cpp",
    "engine/tests/native_candidate_jsonl.cpp",
    "engine/tests/load_bundle.cpp",
    "engine/tests/qkv_compare.cpp",
    "engine/tests/rmsnorm_compare.cpp",
]

JSON_PATHS = [
    "contracts/intel-qwen36-target-contract.json",
    "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json",
    "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json",
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json",
    "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompt-suites.json",
    "oracle/oracle-bundle-contract.json",
]


def load_json(path: str) -> dict:
  with (ROOT / path).open("r", encoding="utf-8") as fh:
    return json.load(fh)


def load_json_path(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as fh:
    return json.load(fh)


def require_expected_fields(actual: dict, expected: dict, label: str) -> None:
  for key, expected_value in expected.items():
    if actual.get(key) != expected_value:
      raise SystemExit(f"{label} {key} mismatch")


def load_jsonl(path: Path) -> list[dict]:
  rows = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: row must be a JSON object")
      rows.append(row)
  return rows


def sha256_canonical_json(value: object) -> str:
  data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_prompt_suites(parsed: dict[str, dict]) -> None:
  matrix = parsed["benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"]
  manifest = parsed["benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompt-suites.json"]
  workstream = "intel-qwen36-35b-a3b-gguf-q4km"
  if manifest.get("workstream") != workstream:
    raise SystemExit("prompt suite workstream mismatch")

  model = manifest.get("model", {})
  expected_model = matrix.get("model", {})
  for key in ("path", "sha256", "batch_size"):
    if model.get(key) != expected_model.get(key):
      raise SystemExit(f"prompt suite model {key} mismatch")

  suites = manifest.get("suites", {})
  if not isinstance(suites, dict):
    raise SystemExit("prompt suite manifest suites must be an object")

  base = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km"
  required_buckets = set(matrix["matrix"]["input_buckets"])
  diagnostic_buckets = set(
      matrix.get("diagnostic_scope", {}).get("input_buckets_not_in_core", [])
  )
  expected_prompt_buckets = required_buckets | diagnostic_buckets
  seen_ids = set()
  prompt_set_counts = {}
  sentinel_buckets = set()
  prefill_buckets = set()

  for suite_name, suite in suites.items():
    if not isinstance(suite, dict):
      raise SystemExit(f"prompt suite {suite_name} must be an object")
    path_value = suite.get("path")
    if not isinstance(path_value, str):
      raise SystemExit(f"prompt suite {suite_name} missing path")
    rows = load_jsonl(base / path_value)
    if not rows:
      raise SystemExit(f"prompt suite {suite_name} is empty")
    for row in rows:
      row_id = row.get("id")
      if not isinstance(row_id, str) or not row_id:
        raise SystemExit(f"prompt suite {suite_name} row missing id")
      if row_id in seen_ids:
        raise SystemExit(f"duplicate prompt row id: {row_id}")
      seen_ids.add(row_id)
      prompt_set = row.get("prompt_set")
      if isinstance(prompt_set, str):
        prompt_set_counts[prompt_set] = prompt_set_counts.get(prompt_set, 0) + 1
      kind = row.get("kind")
      if kind == "token_exact":
        if row.get("temperature") != 0.0:
          raise SystemExit(f"{row_id} token_exact prompt must use temperature 0.0")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
          raise SystemExit(f"{row_id} token_exact prompt missing prompt text")
      elif kind == "sentinel_retrieval":
        if row.get("bucket") != row.get("target_prompt_tokens"):
          raise SystemExit(f"{row_id} sentinel bucket/token target mismatch")
        sentinel_buckets.add(row.get("bucket"))
      elif kind == "prefill_shape":
        if row.get("bucket") != row.get("target_prompt_tokens"):
          raise SystemExit(f"{row_id} prefill bucket/token target mismatch")
        if row.get("cache_mode") != "cold_no_prefix":
          raise SystemExit(f"{row_id} prefill row must be cold_no_prefix")
        prefill_buckets.add(row.get("bucket"))
      else:
        raise SystemExit(f"{row_id} unsupported prompt kind: {kind}")

    min_cases = suite.get("min_cases")
    if isinstance(min_cases, int) and len(rows) < min_cases:
      raise SystemExit(f"prompt suite {suite_name} has fewer than {min_cases} rows")

  if prompt_set_counts.get("short", 0) < 3:
    raise SystemExit("prompt suite must contain at least 3 short cases")
  if prompt_set_counts.get("router-stability", 0) < 3:
    raise SystemExit("prompt suite must contain at least 3 router-stability cases")
  if sentinel_buckets != expected_prompt_buckets:
    raise SystemExit(f"sentinel buckets mismatch: {sorted(sentinel_buckets)}")
  if prefill_buckets != expected_prompt_buckets:
    raise SystemExit(f"prefill buckets mismatch: {sorted(prefill_buckets)}")


def main() -> None:
  missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
  if missing:
    raise SystemExit(f"missing required paths: {missing}")

  parsed = {path: load_json(path) for path in JSON_PATHS}
  workstreams = {
      parsed["contracts/intel-qwen36-target-contract.json"]["workstream"],
      parsed["contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"]["workstream"],
      parsed["contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"]["workstream"],
      parsed["benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"]["workstream"],
  }
  if workstreams != {"intel-qwen36-35b-a3b-gguf-q4km"}:
    raise SystemExit(f"workstream mismatch: {sorted(workstreams)}")

  model = parsed["contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"]["model"]
  if model["layers"] != 40 or model["active_experts"] != 8:
    raise SystemExit("unexpected model contract shape")
  product_contract = parsed[
      "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
  ]
  product_model = product_contract.get("product_model", {})
  product_arch = product_model.get("architecture", {})
  if product_arch.get("layers") != 40 or product_arch.get("active_experts") != 8:
    raise SystemExit("unexpected OpenVINO product model contract shape")
  if product_model.get("path") != "/home/intel/Qwen3.6-35B-A3B-ov":
    raise SystemExit("unexpected OpenVINO product model path")
  product_files = product_model.get("locked_files", {})
  if product_files.get("openvino_language_model.bin", {}).get("sha256") != (
      "46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4"
  ):
    raise SystemExit("unexpected OpenVINO product language-model digest")
  if product_files.get("openvino_text_embeddings_model.bin", {}).get("sha256") != (
      "21b75aed439e3c5a19daedff1c3d564e91a972061f29c100285f97bceb264bf0"
  ):
    raise SystemExit("unexpected OpenVINO product text-embedding digest")
  runtime_contract = product_contract.get("runtime_contract", {})
  if runtime_contract.get("candidate", {}).get("final_runtime_dependency") != (
      "OpenVINO GPU"
  ):
    raise SystemExit("OpenVINO GPU must be the allowed final runtime dependency")
  if runtime_contract.get("baseline", {}).get("custom_gpu_config_allowed") is not False:
    raise SystemExit("stock OpenVINO baseline must reject candidate custom config")

  matrix = parsed["benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"]
  if matrix.get("contract_version") != "0.8":
    raise SystemExit("acceptance matrix contract_version must be 0.8")
  if matrix.get("model", {}).get("path") != product_model.get("path"):
    raise SystemExit("acceptance matrix must target the OpenVINO product model")
  if matrix.get("model_contract", {}).get("manifest_path") != (
      "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
  ):
    raise SystemExit("acceptance matrix product model contract mismatch")
  if not matrix["r0_target_policy"]["refresh_required_before_product_claim"]:
    raise SystemExit("acceptance matrix must require R0 refresh")
  expected_core_buckets = [
      2048, 4096, 8192, 16384, 32768, 65536, 131072
  ]
  if matrix["matrix"]["input_buckets"] != expected_core_buckets:
    raise SystemExit("acceptance matrix core input buckets mismatch")
  if matrix["matrix"]["output_tokens"] != [512]:
    raise SystemExit("acceptance matrix core output lane must be exactly 512")
  diagnostic_scope = matrix.get("diagnostic_scope", {})
  if diagnostic_scope.get("input_buckets_not_in_core") != [
      1024, 102400, 262144
  ]:
    raise SystemExit("acceptance matrix diagnostic input scope mismatch")
  if diagnostic_scope.get("output_token_counts_not_in_core") != [1024]:
    raise SystemExit("acceptance matrix diagnostic output scope mismatch")
  if diagnostic_scope.get("promotion_gating") is not False:
    raise SystemExit("acceptance matrix diagnostics must not gate promotion")
  r0_policy = matrix["r0_target_policy"]
  if r0_policy.get("minimum_openvino_speedup_ratio") != 1.1:
    raise SystemExit("priority rows must require at least 1.10x OpenVINO")
  if r0_policy.get("minimum_openvino_speedup_ratio_applies_to_buckets") != [
      32768, 65536, 131072
  ]:
    raise SystemExit("acceptance matrix priority ratio scope mismatch")
  if r0_policy.get("regression_guard_minimum_ratio") != 0.98:
    raise SystemExit("short-row non-inferiority ratio must be 0.98x")
  if r0_policy.get("regression_guard_buckets") != [
      2048, 4096, 8192, 16384
  ]:
    raise SystemExit("acceptance matrix regression guard scope mismatch")
  if r0_policy.get("stretch_openvino_speedup_ratio") != 1.125:
    raise SystemExit("acceptance matrix stretch target must be 1.125x OpenVINO")
  if r0_policy.get("both_prefill_and_decode_required") is not True:
    raise SystemExit("acceptance matrix must require both prefill and decode")
  if r0_policy.get("route_rejection_satisfies_goal") is not False:
    raise SystemExit("route rejection must not satisfy the project goal")
  candidate_runtime = matrix.get("candidate_runtime", {})
  if candidate_runtime.get("final_runtime_dependency") != (
      "OpenVINO GPU plus candidate-specific custom OpenCL GPU operations"
  ):
    raise SystemExit("acceptance matrix candidate runtime contract mismatch")
  if candidate_runtime.get("baseline_configuration_leakage_allowed") is not False:
    raise SystemExit("candidate configuration must not leak into stock OpenVINO")
  candidate_bootstrap = candidate_runtime.get("bootstrap", {})
  if candidate_bootstrap.get("stage") != "OV0":
    raise SystemExit("OpenVINO specialization bootstrap must remain bound to OV0")
  if candidate_bootstrap.get("status") != (
      "passed_clean_baseline_oracle_and_noop_custom_op_bundle"
  ):
    raise SystemExit("OpenVINO OV0 bootstrap must be closed before OV1")
  if candidate_bootstrap.get("evidence") != (
      "output/openvino-specialization-bootstrap-20260713Tov0contractZ/"
  ):
    raise SystemExit("OpenVINO OV0 bootstrap evidence mismatch")
  if candidate_bootstrap.get("optimization_source_allowed") is not True:
    raise SystemExit("closed OV0 must admit OV1 optimization source")
  if candidate_bootstrap.get("speedup_claims_allowed") is not False:
    raise SystemExit("OV0 closure alone must not allow speedup claims")
  first_prefill = candidate_runtime.get("first_prefill_route", {})
  if first_prefill.get("priority_end_to_end_cut_ms_per_1024") != {
      "32768": 58.027,
      "65536": 77.947,
      "131072": 115.486,
  }:
    raise SystemExit("OpenVINO priority-row prefill cuts mismatch")
  if first_prefill.get("must_combine_with_context_attention_cut") is not True:
    raise SystemExit("prefill fusion must be charged with context attention")
  if first_prefill.get("component_profile_is_product_evidence") is not False:
    raise SystemExit("component profile must not count as product evidence")
  inference_policy = matrix.get("performance_promotion", {}).get("inference", {})
  if inference_policy.get("confidence") != 0.95:
    raise SystemExit("performance inference must use 95% confidence")
  if inference_policy.get("product_minimum_paired_blocks") != 8:
    raise SystemExit("product inference must require eight paired blocks")
  if inference_policy.get("component_minimum_samples") != 20:
    raise SystemExit("component inference must require twenty samples")
  if inference_policy.get("repeat_confirm_median_spread_is_promotion_gate") is not False:
    raise SystemExit("repeat/confirm median spread must not gate promotion")
  component_self_test = latency_cap_inference(
      [2.40 + 0.001 * index for index in range(20)], cap=2.825)
  if component_self_test.get("rate_pass") is not True:
    raise SystemExit("component confidence-bound inference self-test failed")
  product_self_test = paired_speedup_inference(
      [112.0] * 8, [100.0] * 8, target_ratio=1.1)
  if product_self_test.get("rate_pass") is not True:
    raise SystemExit("paired speedup confidence-bound inference self-test failed")
  if r0_policy.get("r0_pending_buckets") != []:
    raise SystemExit("acceptance matrix R0 pending buckets must be policy-resolved")
  if r0_policy.get("r0_unavailable_denominator_buckets") != []:
    raise SystemExit("every core bucket must have an OpenVINO denominator")
  if r0_policy.get("historical_unavailable_denominator_buckets") != [262144]:
    raise SystemExit("acceptance matrix must retain historical 256k unavailability")
  unavailable_policy_path = r0_policy.get("r0_unavailable_policy_path")
  if not isinstance(unavailable_policy_path, str) or not (
      ROOT / unavailable_policy_path
  ).exists():
    raise SystemExit("acceptance matrix unavailable policy path missing")
  prompt_edge_policy_path = r0_policy.get("r0_prompt_edge_policy_path")
  if not isinstance(prompt_edge_policy_path, str) or not (
      ROOT / prompt_edge_policy_path
  ).exists():
    raise SystemExit("acceptance matrix prompt-edge policy path missing")
  bucket_keys = {str(bucket) for bucket in matrix["matrix"]["input_buckets"]}
  targets = matrix.get("bootstrap_targets", {})
  for phase in ("prefill_tokens_s", "decode_tokens_s"):
    phase_targets = targets.get(phase, {})
    if set(phase_targets) != bucket_keys:
      raise SystemExit(f"acceptance matrix {phase} must cover every bucket")
  openvino = matrix.get("openvino_q4_denominator", {})
  comparison_protocol = openvino.get("comparison_protocol", {})
  if comparison_protocol.get("apply_chat_template") is not False:
    raise SystemExit("OpenVINO product rows must disable chat templates")
  if comparison_protocol.get("prefix_caching") is not False:
    raise SystemExit("OpenVINO product rows must disable prefix caching")
  if comparison_protocol.get("fresh_isolated_worker_required") is not True:
    raise SystemExit("stock OpenVINO product rows require an isolated worker")
  if comparison_protocol.get("candidate_custom_config_forbidden") is not True:
    raise SystemExit("stock OpenVINO worker must reject candidate custom config")
  calibration = openvino.get("target_calibration_512", {})
  if calibration.get("status") != "raw_prompt_refresh_complete":
    raise SystemExit("OpenVINO raw-prompt target refresh status mismatch")
  if calibration.get("input_buckets") != matrix["matrix"]["input_buckets"]:
    raise SystemExit("OpenVINO 512-token calibration buckets mismatch")
  if calibration.get("prompt_sets") != [
      "prefill_shape", "sentinel", "filler"
  ]:
    raise SystemExit("OpenVINO target calibration must cover three prompt sets")
  if calibration.get("measured_rows_per_prompt", 0) < 3:
    raise SystemExit("OpenVINO target calibration needs at least three rows/prompt")
  if calibration.get("warmup_rows_per_prompt", 0) < 1:
    raise SystemExit("OpenVINO target calibration needs at least one warmup/prompt")
  if calibration.get("fixed_output_tokens") != 512:
    raise SystemExit("OpenVINO target calibration must bind the 512-token lane")
  if calibration.get("raw_prompt_sets_complete") != [
      "prefill_shape", "sentinel", "filler"
  ]:
    raise SystemExit("OpenVINO raw-prompt prompt-set coverage mismatch")
  if calibration.get("raw_prompt_anchor_prompt_sets") != [
      "prefill_shape", "sentinel"
  ]:
    raise SystemExit("OpenVINO raw-prompt anchor prompt sets mismatch")
  if calibration.get("raw_prompt_anchor_buckets") != [8192]:
    raise SystemExit("OpenVINO raw-prompt anchor buckets mismatch")
  remaining_raw = calibration.get(
      "remaining_raw_prompt_buckets_by_prompt_set", {})
  if remaining_raw != {
      "prefill_shape": [],
      "sentinel": [],
  }:
    raise SystemExit("OpenVINO remaining raw-prompt buckets mismatch")
  if calibration.get("remaining_product_denominator_output_lanes") != []:
    raise SystemExit("OpenVINO target refresh must close every product lane")
  accuracy = matrix.get("accuracy", {})
  distribution = accuracy.get("teacher_forced_distribution", {})
  if distribution.get("product_reference") != "locked stock OpenVINO GPU U4":
    raise SystemExit("stock OpenVINO must be the product distribution reference")
  tokens = accuracy.get("tokens", {})
  if tokens.get("reference_consensus_required_before_candidate_scoring") is not False:
    raise SystemExit("legacy reference consensus must not gate the OpenVINO candidate")
  evidence = openvino.get("evidence", {})
  derivation = evidence.get("derivation")
  if not isinstance(derivation, str) or not (ROOT / derivation).is_file():
    raise SystemExit("OpenVINO target derivation document missing")
  raw_evidence = [
      value for key, value in evidence.items() if key.startswith("raw_512_")
  ]
  if len(raw_evidence) < 3:
    raise SystemExit("OpenVINO raw 512-token evidence map is incomplete")
  for evidence_path in raw_evidence:
    if not isinstance(evidence_path, str) or not (ROOT / evidence_path).is_dir():
      raise SystemExit(f"OpenVINO target evidence missing: {evidence_path}")
  core_refresh_path = evidence.get("raw_512_core_remaining_prefill_sentinel")
  if not isinstance(core_refresh_path, str):
    raise SystemExit("OpenVINO core raw-prompt refresh evidence missing")
  core_refresh_dir = ROOT / core_refresh_path
  core_refresh = load_json_path(core_refresh_dir / "matrix.json")
  core_refresh_correctness = load_json_path(core_refresh_dir / "correctness.json")
  core_refresh_rows = load_jsonl(core_refresh_dir / "metrics.jsonl")
  expected_refresh_buckets = [2048, 4096, 16384, 32768, 65536, 131072]
  expected_refresh_cases = {
      f"{prompt_set}_{bucket}"
      for prompt_set in ("prefill_shape", "sentinel")
      for bucket in ("002k", "004k", "016k", "032k", "064k", "128k")
  }
  refresh_config = core_refresh.get("config", {})
  if refresh_config.get("buckets") != expected_refresh_buckets:
    raise SystemExit("OpenVINO core raw-prompt refresh bucket mismatch")
  if refresh_config.get("prompt_set") != "both":
    raise SystemExit("OpenVINO core raw-prompt refresh prompt-set mismatch")
  if refresh_config.get("output_tokens") != 512:
    raise SystemExit("OpenVINO core raw-prompt refresh output mismatch")
  if refresh_config.get("num_warmup") != 1 or refresh_config.get("num_iter") != 3:
    raise SystemExit("OpenVINO core raw-prompt refresh repeat protocol mismatch")
  if core_refresh.get("git", {}).get("dirty") is not False:
    raise SystemExit("OpenVINO core raw-prompt refresh must be clean")
  if core_refresh.get("required_checks_passed") is not True:
    raise SystemExit("OpenVINO core raw-prompt refresh matrix must pass")
  if core_refresh_correctness.get("required_checks_passed") is not True:
    raise SystemExit("OpenVINO core raw-prompt refresh correctness must pass")
  if len(core_refresh_rows) != 36:
    raise SystemExit("OpenVINO core raw-prompt refresh row count mismatch")
  if {row.get("case_id") for row in core_refresh_rows} != expected_refresh_cases:
    raise SystemExit("OpenVINO core raw-prompt refresh case coverage mismatch")
  for case_id in expected_refresh_cases:
    case_rows = [row for row in core_refresh_rows if row.get("case_id") == case_id]
    if len(case_rows) != 3:
      raise SystemExit(f"{case_id}: OpenVINO refresh repeat count mismatch")
    if any(
        row.get("input_tokens") != row.get("bucket")
        or row.get("tokenizer_input_tokens") != row.get("bucket")
        or row.get("output_tokens") != 512
        for row in case_rows
    ):
      raise SystemExit(f"{case_id}: OpenVINO refresh token count mismatch")
  promoted_refresh_decode = {
      4096: 44.02436065673828,
      16384: 39.032554626464844,
      65536: 26.61721420288086,
  }
  for bucket, expected_value in promoted_refresh_decode.items():
    medians = []
    for prompt_set in ("prefill_shape", "sentinel"):
      case_id = f"{prompt_set}_{bucket // 1024:03d}k"
      medians.append(statistics.median(
          float(row["decode_tokens_s"])
          for row in core_refresh_rows if row.get("case_id") == case_id
      ))
    if abs(max(medians) - expected_value) > 1e-9:
      raise SystemExit(f"OpenVINO refresh decode median mismatch at {bucket}")
  observed = openvino.get("best_observed_tokens_s", {})
  known_keys = bucket_keys
  for observed_phase, target_phase in (
      ("prefill", "prefill_tokens_s"),
      ("decode", "decode_tokens_s"),
  ):
    phase_observed = observed.get(observed_phase, {})
    if set(phase_observed) != known_keys:
      raise SystemExit(
          f"OpenVINO {observed_phase} denominator must cover every core bucket"
      )
    for bucket in map(str, r0_policy[
        "minimum_openvino_speedup_ratio_applies_to_buckets"
    ]):
      minimum = (
          float(phase_observed[bucket]) *
          float(r0_policy["minimum_openvino_speedup_ratio"])
      )
      if float(targets[target_phase][bucket]) + 1e-9 < minimum:
        raise SystemExit(
            f"{target_phase} target at {bucket} is below the OpenVINO ratio"
        )
  token_policy = matrix.get("accuracy", {}).get("tokens", {})
  if token_policy.get("reference_runtimes") != ["stock OpenVINO GPU U4"]:
    raise SystemExit("token policy must use stock OpenVINO as product reference")
  if token_policy.get("min_reference_consensus_prompt_cases") != 0:
    raise SystemExit("OpenVINO product token policy must not require consensus")
  if token_policy.get("reference_consensus_required_before_candidate_scoring") is not False:
    raise SystemExit("legacy reference consensus must not gate candidate scoring")
  if token_policy.get("legacy_reference_disagreement_policy") != (
      "diagnostic_not_candidate_pass_or_failure"
  ):
    raise SystemExit("legacy token reference-disagreement disposition mismatch")
  promotion = matrix.get("performance_promotion", {})
  if promotion.get("all_input_and_output_lanes_must_pass_assigned_role") is not True:
    raise SystemExit("every product lane must pass its assigned performance role")
  if promotion.get("route_rejection_completes_project_goal") is not False:
    raise SystemExit("performance promotion must reject route-only goal closure")
  if promotion.get("route_selection_priority_input_buckets") != [
      32768, 65536, 131072
  ]:
    raise SystemExit("performance promotion long-context route priority mismatch")
  if promotion.get("regression_guard_input_buckets") != [
      2048, 4096, 8192, 16384
  ]:
    raise SystemExit("performance promotion regression guards mismatch")
  context_gap_dir = (
      ROOT / "output/packed-token-context-gap-20260713Tseq770cleanZ"
  )
  context_gap = load_json_path(context_gap_dir / "result.json")
  if context_gap.get("required_checks_passed") is not True:
    raise SystemExit("packed-token exact-context gap gate must pass")
  if context_gap.get("git", {}).get("dirty") is not False:
    raise SystemExit("packed-token exact-context gap gate must be clean")
  if context_gap.get("speedup_claims_allowed") is not False:
    raise SystemExit("packed-token exact-context gap must not claim speedup")
  if context_gap.get("buckets") != expected_core_buckets:
    raise SystemExit("packed-token exact-context gap bucket mismatch")
  if context_gap.get("sample_tokens") != 1:
    raise SystemExit("packed-token exact-context gap sample count mismatch")
  context_rows = context_gap.get("rows", [])
  if len(context_rows) != len(expected_core_buckets):
    raise SystemExit("packed-token exact-context gap row count mismatch")
  expected_context_ratios = {
      2048: 0.7715073103617677,
      4096: 0.6014164047016313,
      8192: 0.4209553933991417,
      16384: 0.2861750022519795,
      32768: 0.15051940215177612,
      65536: 0.1031157791454918,
      131072: 0.07571835641855969,
  }
  for row in context_rows:
    bucket = row.get("context_tokens")
    if bucket not in expected_context_ratios:
      raise SystemExit("packed-token exact-context gap unexpected bucket")
    if row.get("required_checks_passed") is not True:
      raise SystemExit(f"packed-token context row failed at {bucket}")
    if row.get("correctness_applicable") is not False:
      raise SystemExit(f"packed-token context row must be nonsemantic at {bucket}")
    if row.get("state_semantics") != "zero_initialized_performance_only":
      raise SystemExit(f"packed-token context state semantics mismatch at {bucket}")
    if row.get("reserved_output_tokens") != 512:
      raise SystemExit(f"packed-token context output capacity mismatch at {bucket}")
    if abs(float(row.get("decode_target_ratio")) - expected_context_ratios[bucket]) > 1e-12:
      raise SystemExit(f"packed-token context target ratio mismatch at {bucket}")
  context_profile_dir = (
      ROOT / "output/packed-token-context-gap-20260713Tseq771-profile-128k-cleanZ"
  )
  context_profile = load_json_path(context_profile_dir / "result.json")
  if context_profile.get("required_checks_passed") is not True:
    raise SystemExit("packed-token 128k context profile must pass")
  if context_profile.get("git", {}).get("dirty") is not False:
    raise SystemExit("packed-token 128k context profile must be clean")
  if context_profile.get("profile_buckets") != [131072]:
    raise SystemExit("packed-token context profile bucket mismatch")
  profile_rows = context_profile.get("rows", [])
  if len(profile_rows) != 1 or profile_rows[0].get("context_tokens") != 131072:
    raise SystemExit("packed-token 128k context profile row mismatch")
  profile_kernels = {
      row.get("kernel"): float(row.get("device_ms"))
      for row in profile_rows[0].get("kernel_profile", [])
  }
  if abs(profile_kernels.get("full_attn_apply_score_gate_control_f32", 0.0) - 542.949585952) > 1e-9:
    raise SystemExit("packed-token 128k serial apply profile mismatch")
  if abs(profile_kernels.get("full_attn_score_control_f32", 0.0) - 66.2050485725) > 1e-9:
    raise SystemExit("packed-token 128k score profile mismatch")
  if not (ROOT / "doc/adr/0053-select-fused-gqa-fp16-kv-long-context-decode.md").is_file():
    raise SystemExit("long-context fused-GQA route ADR missing")
  scalar_gqa_dir = (
      ROOT / "output/fused-gqa-fp16-kv-decode-20260713Tseq772cleanZ"
  )
  scalar_gqa = load_json_path(scalar_gqa_dir / "result.json")
  if scalar_gqa.get("git", {}).get("dirty") is not False:
    raise SystemExit("scalar fused-GQA component gate must be clean")
  if scalar_gqa.get("required_checks_passed") is not False:
    raise SystemExit("scalar fused-GQA component must remain rejected")
  scalar_result = scalar_gqa.get("result", {})
  if scalar_result.get("numeric_pass") is not True:
    raise SystemExit("scalar fused-GQA numeric evidence mismatch")
  if scalar_result.get("timing_pass") is not False:
    raise SystemExit("scalar fused-GQA timing evidence must fail")
  if abs(float(scalar_result.get("output_cosine")) - 0.99999999946) > 1e-12:
    raise SystemExit("scalar fused-GQA cosine mismatch")
  if abs(float(scalar_result.get("output_relative_l2")) - 8.16663311186e-05) > 1e-15:
    raise SystemExit("scalar fused-GQA relative L2 mismatch")
  if abs(float(scalar_result.get("repeat", {}).get("total_ms")) - 2.844375) > 1e-9:
    raise SystemExit("scalar fused-GQA repeat timing mismatch")
  if abs(float(scalar_result.get("confirm", {}).get("total_ms")) - 2.847395) > 1e-9:
    raise SystemExit("scalar fused-GQA confirm timing mismatch")
  if not (ROOT / "doc/adr/0054-close-scalar-fused-gqa-select-xmx-flash-decode.md").is_file():
    raise SystemExit("XMX GQA flash-decode route ADR missing")
  xmx_gqa_dir = (
      ROOT / "output/xmx-gqa-fp16-kv-decode-20260713Tseq773cleanZ"
  )
  xmx_gqa = load_json_path(xmx_gqa_dir / "result.json")
  if xmx_gqa.get("git", {}).get("dirty") is not False:
    raise SystemExit("XMX fused-GQA component gate must be clean")
  if xmx_gqa.get("mode") != "xmx":
    raise SystemExit("XMX fused-GQA component mode mismatch")
  if xmx_gqa.get("required_checks_passed") is not False:
    raise SystemExit("XMX fused-GQA component must remain rejected")
  xmx_result = xmx_gqa.get("result", {})
  if xmx_result.get("numeric_pass") is not True:
    raise SystemExit("XMX fused-GQA numeric evidence mismatch")
  if xmx_result.get("timing_pass") is not False:
    raise SystemExit("XMX fused-GQA timing evidence must fail")
  if abs(float(xmx_result.get("output_cosine")) - 0.999999999413) > 1e-12:
    raise SystemExit("XMX fused-GQA cosine mismatch")
  if abs(float(xmx_result.get("output_relative_l2")) - 9.06535618574e-05) > 1e-15:
    raise SystemExit("XMX fused-GQA relative L2 mismatch")
  if abs(float(xmx_result.get("repeat", {}).get("total_ms")) - 6.114270) > 1e-9:
    raise SystemExit("XMX fused-GQA repeat timing mismatch")
  if abs(float(xmx_result.get("confirm", {}).get("total_ms")) - 6.149061) > 1e-9:
    raise SystemExit("XMX fused-GQA confirm timing mismatch")
  if abs(float(xmx_result.get("spread")) - 0.00565793704112) > 1e-15:
    raise SystemExit("XMX fused-GQA spread mismatch")
  if not (ROOT / "doc/adr/0055-close-xmx-gqa-select-sdpa-provider-codegen.md").is_file():
    raise SystemExit("optimized-SDPA provider route ADR missing")
  sdpa_capture_dir = (
      ROOT / "output/openvino-sdpa-provider-capture-20260713Tseq774cleanZ"
  )
  sdpa_capture = load_json_path(sdpa_capture_dir / "result.json")
  if sdpa_capture.get("git", {}).get("dirty") is not False:
    raise SystemExit("optimized-SDPA provider capture must be clean")
  if sdpa_capture.get("required_checks_passed") is not True:
    raise SystemExit("optimized-SDPA provider capture must pass")
  sdpa_result = sdpa_capture.get("result", {})
  if sdpa_result.get("context_tokens") != 131072:
    raise SystemExit("optimized-SDPA provider capture context mismatch")
  if sdpa_result.get("expected_exec_type") != "ocl::sdpa::opt__f16":
    raise SystemExit("optimized-SDPA provider selection mismatch")
  if sdpa_result.get("provider_selection_pass") is not True:
    raise SystemExit("optimized-SDPA exact provider selection must pass")
  if len(sdpa_result.get("exact_profile", {}).get("attention_rows", [])) != 10:
    raise SystemExit("optimized-SDPA exact layer count mismatch")
  if len(sdpa_result.get("binary_captures", [])) != 3:
    raise SystemExit("optimized-SDPA captured program count mismatch")
  if abs(float(sdpa_result.get("profile_median_us")) - 56508.5) > 1e-9:
    raise SystemExit("optimized-SDPA profile median mismatch")
  if sdpa_result.get("profile_cap_direction_pass") is not False:
    raise SystemExit("optimized-SDPA PERF_COUNT aggregate must not be promoted")
  sdpa_trace_dir = (
      ROOT / "output/openvino-sdpa-provider-capture-20260713Tseq775trace-cleanZ"
  )
  sdpa_trace = load_json_path(sdpa_trace_dir / "result.json")
  if sdpa_trace.get("git", {}).get("dirty") is not False:
    raise SystemExit("optimized-SDPA dispatch trace must be clean")
  if sdpa_trace.get("required_checks_passed") is not True:
    raise SystemExit("optimized-SDPA dispatch trace must pass capture checks")
  trace_result = sdpa_trace.get("result", {})
  if trace_result.get("exact_dispatch_count") != 10:
    raise SystemExit("optimized-SDPA exact dispatch count mismatch")
  if trace_result.get("exact_dispatch_kernel_names") != [
      "sdpa_micro__generate_5781906426501558618__sa"]:
    raise SystemExit("optimized-SDPA exact program mapping mismatch")
  if trace_result.get("exact_dispatch_metadata_pass") is not True:
    raise SystemExit("optimized-SDPA exact dispatch metadata must pass")
  if len(trace_result.get("binary_captures", [])) != 5:
    raise SystemExit("optimized-SDPA traced program count mismatch")
  if abs(float(trace_result.get("exact_dispatch_median_us")) - 3704.2185) > 1e-9:
    raise SystemExit("optimized-SDPA exact event median mismatch")
  if abs(float(trace_result.get("exact_dispatch_max_us")) - 3984.895) > 1e-9:
    raise SystemExit("optimized-SDPA exact event maximum mismatch")
  if trace_result.get("exact_dispatch_cap_direction_pass") is not False:
    raise SystemExit("optimized-SDPA exact event cap must remain failed")
  if not (ROOT / "doc/adr/0056-close-stateful-sdpa-select-product-paged-gqa.md").is_file():
    raise SystemExit("product paged-GQA route ADR missing")
  paged_dir = (
      ROOT / "output/openvino-paged-gqa-provider-20260713Tseq777cleanZ"
  )
  paged = load_json_path(paged_dir / "result.json")
  if paged.get("git", {}).get("dirty") is not False:
    raise SystemExit("product paged-GQA source gate must be clean")
  if paged.get("required_checks_passed") is not False:
    raise SystemExit("product paged-GQA source must remain rejected")
  paged_result = paged.get("result", {})
  if paged_result.get("trace_rows") != 60:
    raise SystemExit("product paged-GQA trace count mismatch")
  if len(paged_result.get("binary_captures", [])) != 3:
    raise SystemExit("product paged-GQA program count mismatch")
  if len(paged_result.get("layer_rows", [])) != 30:
    raise SystemExit("product paged-GQA layer pairing mismatch")
  if abs(float(paged_result.get("repeat_attention_ms")) - 31.229054) > 1e-9:
    raise SystemExit("product paged-GQA repeat timing mismatch")
  if abs(float(paged_result.get("confirm_attention_ms")) - 27.951553) > 1e-9:
    raise SystemExit("product paged-GQA confirm timing mismatch")
  if abs(float(paged_result.get("paired_spread")) - 0.11725649018499977) > 1e-15:
    raise SystemExit("product paged-GQA spread mismatch")
  if paged_result.get("rate_pass") is not False:
    raise SystemExit("product paged-GQA rate must remain failed")
  if not (ROOT / "doc/adr/0057-close-paged-provider-select-int8-kv-gqa.md").is_file():
    raise SystemExit("INT8 block32-KV GQA route ADR missing")
  int8_kv_dir = (
      ROOT / "output/compressed-gqa-i8-kv-decode-20260713Tseq778cleanZ"
  )
  int8_kv = load_json_path(int8_kv_dir / "result.json")
  if int8_kv.get("git", {}).get("dirty") is not False:
    raise SystemExit("INT8 block32-KV component gate must be clean")
  if int8_kv.get("required_checks_passed") is not False:
    raise SystemExit("INT8 block32-KV component must remain rejected")
  int8_result = int8_kv.get("result", {})
  if int8_result.get("numeric_pass") is not True:
    raise SystemExit("INT8 block32-KV numeric evidence mismatch")
  if int8_result.get("timing_pass") is not False:
    raise SystemExit("INT8 block32-KV timing/noise gate must fail")
  if len(int8_result.get("repeat_samples", [])) != 7 or len(
      int8_result.get("confirm_samples", [])) != 7:
    raise SystemExit("INT8 block32-KV distribution shape mismatch")
  if abs(float(int8_result.get("output_cosine")) - 0.999999971822) > 1e-12:
    raise SystemExit("INT8 block32-KV cosine mismatch")
  if abs(float(int8_result.get("output_relative_l2")) - 0.000324519357589) > 1e-15:
    raise SystemExit("INT8 block32-KV relative L2 mismatch")
  if abs(float(int8_result.get("repeat", {}).get("total_ms")) - 2.459061) > 1e-9:
    raise SystemExit("INT8 block32-KV repeat timing mismatch")
  if abs(float(int8_result.get("confirm", {}).get("total_ms")) - 2.471978) > 1e-9:
    raise SystemExit("INT8 block32-KV confirm timing mismatch")
  if abs(float(int8_result.get("spread")) - 0.0052253701287) > 1e-15:
    raise SystemExit("INT8 block32-KV paired spread mismatch")
  if not (ROOT / "doc/adr/0058-close-int8-noise-select-packed-int6-kv.md").is_file():
    raise SystemExit("packed INT6 block32-KV GQA route ADR missing")
  if not (ROOT / "doc/adr/0062-replace-fixed-noise-veto-reopen-int8-gqa.md").is_file():
    raise SystemExit("confidence-bound measurement ADR missing")
  int8_ci_dir = (
      ROOT / "output/compressed-gqa-i8-kv-decode-20260713Tseq783-ci-confirm-cleanZ"
  )
  int8_ci = load_json_path(int8_ci_dir / "result.json")
  if int8_ci.get("git", {}).get("dirty") is not False:
    raise SystemExit("INT8 confidence-bound component gate must be clean")
  if int8_ci.get("git", {}).get("commit") != (
      "d9cfa0bdd5ed1ee51d905c129c31ce3df2cc53cb"
  ):
    raise SystemExit("INT8 confidence-bound component commit mismatch")
  if int8_ci.get("schema_version") != (
      "intel-qwen36-compressed-gqa-i8-kv-decode-gate-v1"
  ):
    raise SystemExit("INT8 confidence-bound schema mismatch")
  if int8_ci.get("required_checks_passed") is not True:
    raise SystemExit("INT8 confidence-bound component gate must pass")
  ci_result = int8_ci.get("result", {})
  if len(ci_result.get("repeat_samples", [])) != 10 or len(
      ci_result.get("confirm_samples", [])) != 10:
    raise SystemExit("INT8 confidence-bound sample shape mismatch")
  ci_inference = int8_ci.get("performance_inference", {})
  if ci_inference.get("sample_count") != 20:
    raise SystemExit("INT8 confidence-bound sample count mismatch")
  if abs(float(ci_inference.get("point_estimate_ms")) - 2.440468) > 1e-9:
    raise SystemExit("INT8 confidence-bound median mismatch")
  if abs(float(ci_inference.get("upper_confidence_bound_ms")) - 2.452708) > 1e-9:
    raise SystemExit("INT8 confidence-bound upper limit mismatch")
  if ci_inference.get("rate_pass") is not True:
    raise SystemExit("INT8 confidence-bound rate must pass")
  if ci_inference.get("dispersion", {}).get("promotion_gate") is not False:
    raise SystemExit("INT8 dispersion must remain diagnostic")
  int6_kv_dir = ROOT / "output/packed-gqa-i6-kv-decode-20260713Tseq779cleanZ"
  int6_kv = load_json_path(int6_kv_dir / "result.json")
  if int6_kv.get("git", {}).get("dirty") is not False:
    raise SystemExit("packed INT6 block32-KV component gate must be clean")
  if int6_kv.get("required_checks_passed") is not False:
    raise SystemExit("packed INT6 block32-KV component must remain rejected")
  int6_result = int6_kv.get("result", {})
  if int6_result.get("numeric_pass") is not True:
    raise SystemExit("packed INT6 block32-KV numeric evidence mismatch")
  if int6_result.get("timing_pass") is not False:
    raise SystemExit("packed INT6 block32-KV timing gate must fail")
  if len(int6_result.get("repeat_samples", [])) != 7 or len(
      int6_result.get("confirm_samples", [])) != 7:
    raise SystemExit("packed INT6 block32-KV distribution shape mismatch")
  if abs(float(int6_result.get("output_cosine")) - 0.999999876270) > 1e-12:
    raise SystemExit("packed INT6 block32-KV cosine mismatch")
  if abs(float(int6_result.get("output_relative_l2")) - 0.000612114163602) > 1e-15:
    raise SystemExit("packed INT6 block32-KV relative L2 mismatch")
  if abs(float(int6_result.get("repeat", {}).get("total_ms")) - 3.134270) > 1e-9:
    raise SystemExit("packed INT6 block32-KV repeat timing mismatch")
  if abs(float(int6_result.get("confirm", {}).get("total_ms")) - 3.132083) > 1e-9:
    raise SystemExit("packed INT6 block32-KV confirm timing mismatch")
  if abs(float(int6_result.get("spread")) - 0.000697770134673) > 1e-15:
    raise SystemExit("packed INT6 block32-KV paired spread mismatch")
  if not (ROOT / "doc/adr/0059-close-packed-int6-select-scaled-e4m3-kv.md").is_file():
    raise SystemExit("scaled E4M3 block32-KV GQA route ADR missing")
  e4m3_dir = ROOT / "output/scaled-e4m3-gqa-kv-decode-20260713Tseq780cleanZ"
  e4m3 = load_json_path(e4m3_dir / "result.json")
  if e4m3.get("git", {}).get("dirty") is not False:
    raise SystemExit("scaled E4M3 block32-KV component gate must be clean")
  if e4m3.get("required_checks_passed") is not False:
    raise SystemExit("scaled E4M3 block32-KV component must remain rejected")
  e4m3_result = e4m3.get("result", {})
  if e4m3_result.get("numeric_pass") is not True:
    raise SystemExit("scaled E4M3 block32-KV numeric evidence mismatch")
  if e4m3_result.get("timing_pass") is not False:
    raise SystemExit("scaled E4M3 block32-KV noise gate must fail")
  if len(e4m3_result.get("repeat_samples", [])) != 7 or len(
      e4m3_result.get("confirm_samples", [])) != 7:
    raise SystemExit("scaled E4M3 block32-KV distribution shape mismatch")
  if abs(float(e4m3_result.get("output_cosine")) - 0.999999472859) > 1e-12:
    raise SystemExit("scaled E4M3 block32-KV cosine mismatch")
  if abs(float(e4m3_result.get("output_relative_l2")) - 0.0010350004974) > 1e-15:
    raise SystemExit("scaled E4M3 block32-KV relative L2 mismatch")
  if abs(float(e4m3_result.get("repeat", {}).get("total_ms")) - 2.747708) > 1e-9:
    raise SystemExit("scaled E4M3 block32-KV repeat timing mismatch")
  if abs(float(e4m3_result.get("confirm", {}).get("total_ms")) - 2.767082) > 1e-9:
    raise SystemExit("scaled E4M3 block32-KV confirm timing mismatch")
  if abs(float(e4m3_result.get("spread")) - 0.00700159951892) > 1e-15:
    raise SystemExit("scaled E4M3 block32-KV paired spread mismatch")
  if not (ROOT / "doc/adr/0060-close-gpu-codecs-select-cpu-avx2-gqa.md").is_file():
    raise SystemExit("CPU AVX2/F16C GQA route ADR missing")
  cpu_gqa_dir = ROOT / "output/cpu-avx2-fp16-gqa-decode-20260713Tseq781cleanZ"
  cpu_gqa = load_json_path(cpu_gqa_dir / "result.json")
  if cpu_gqa.get("git", {}).get("dirty") is not False:
    raise SystemExit("CPU AVX2/F16C GQA component gate must be clean")
  if cpu_gqa.get("required_checks_passed") is not False:
    raise SystemExit("CPU AVX2/F16C GQA component must remain rejected")
  cpu_result = cpu_gqa.get("result", {})
  if cpu_result.get("affinity_pass") is not True:
    raise SystemExit("CPU AVX2/F16C GQA affinity evidence mismatch")
  if cpu_result.get("numeric_pass") is not True:
    raise SystemExit("CPU AVX2/F16C GQA numeric evidence mismatch")
  if cpu_result.get("timing_pass") is not False:
    raise SystemExit("CPU AVX2/F16C GQA timing gate must fail")
  if len(cpu_result.get("repeat_samples_ms", [])) != 7 or len(
      cpu_result.get("confirm_samples_ms", [])) != 7:
    raise SystemExit("CPU AVX2/F16C GQA distribution shape mismatch")
  if abs(float(cpu_result.get("output_cosine")) - 0.999999999995) > 1e-12:
    raise SystemExit("CPU AVX2/F16C GQA cosine mismatch")
  if abs(float(cpu_result.get("output_relative_l2")) - 0.000149519231509) > 1e-15:
    raise SystemExit("CPU AVX2/F16C GQA relative L2 mismatch")
  if abs(float(cpu_result.get("repeat_ms")) - 31.563737) > 1e-9:
    raise SystemExit("CPU AVX2/F16C GQA repeat timing mismatch")
  if abs(float(cpu_result.get("confirm_ms")) - 31.697967) > 1e-9:
    raise SystemExit("CPU AVX2/F16C GQA confirm timing mismatch")
  if abs(float(cpu_result.get("spread")) - 0.00423465643711) > 1e-15:
    raise SystemExit("CPU AVX2/F16C GQA paired spread mismatch")
  if not (ROOT / "doc/adr/0061-record-long-context-native-route-exhaustion.md").is_file():
    raise SystemExit("long-context native route-exhaustion ADR missing")
  capability_dir = (
      ROOT / "output/native-capability-reopen-audit-20260713Tseq782cleanZ"
  )
  capability = load_json_path(capability_dir / "result.json")
  if capability.get("git", {}).get("dirty") is not False:
    raise SystemExit("native capability reopen audit must be clean")
  if capability.get("required_checks_passed") is not True:
    raise SystemExit("native capability reopen audit checks must pass")
  if capability.get("route_reopen_allowed") is not False:
    raise SystemExit("native capability audit must not reopen a route")
  if capability.get("new_capability_complete_bound_pass") is not False:
    raise SystemExit("native capability complete bound must remain failed")
  capability_result = capability.get("capability", {})
  if capability_result.get("native_fp8") is not False:
    raise SystemExit("native FP8 capability inventory mismatch")
  if capability_result.get("esimd_compiler") is not False:
    raise SystemExit("ESIMD compiler capability inventory mismatch")
  if capability_result.get("cpu_wide_matrix_isa") is not False:
    raise SystemExit("CPU wide-matrix ISA inventory mismatch")
  if capability_result.get("existing_xmx") is not True:
    raise SystemExit("existing XMX capability inventory mismatch")
  if capability_result.get("existing_integer_dot") is not True:
    raise SystemExit("existing integer-dot capability inventory mismatch")
  if capability_result.get("existing_bf16") is not True:
    raise SystemExit("existing BF16 capability inventory mismatch")
  if len(capability.get("checks", [])) != 14 or not all(
      row.get("pass") is True for row in capability.get("checks", [])):
    raise SystemExit("native capability audit check census mismatch")
  goal_text = (
      ROOT / "goals/intel-qwen36-35b-a3b-q4km-engine.md"
  ).read_text(encoding="utf-8")
  for marker in (
      "## Performance Target",
      "`1.10x`",
      "does **not** complete",
      "this project goal",
  ):
    if marker not in goal_text:
      raise SystemExit(f"goal document missing quantitative marker: {marker}")
  target_contract = parsed["contracts/intel-qwen36-target-contract.json"]
  execution = target_contract.get("execution", {})
  if execution.get("mode") != "local":
    raise SystemExit("target contract execution mode must be local")
  if execution.get("network_transport_required") is not False:
    raise SystemExit("local target must not require network transport")
  if target_contract.get("target", {}).get("host_alias") != "local":
    raise SystemExit("target contract host alias must be local")
  if "ssh_user" in target_contract.get("target", {}):
    raise SystemExit("local target contract must not contain a login user")
  legacy_helper = ROOT / "tools" / "iq36_remote.py"
  if legacy_helper.exists():
    raise SystemExit("legacy remote experiment helper must be removed")
  forbidden_commands = ('"ssh"', "'ssh'", '"scp"', "'scp'")
  for tool_path in sorted((ROOT / "tools").glob("*.py")):
    if tool_path.name == Path(__file__).name:
      continue
    tool_text = tool_path.read_text(encoding="utf-8", errors="replace")
    if any(token in tool_text for token in forbidden_commands):
      raise SystemExit(f"{tool_path.name}: network transport command is forbidden")
    if "ptl-cls-dvt2-008" in tool_text:
      raise SystemExit(f"{tool_path.name}: legacy remote target alias is forbidden")
  r0_refresh = target_contract.get("r0_refresh", {})
  completed_items = r0_refresh.get("completed_items", [])
  pending_items = r0_refresh.get("pending_items", [])
  if not any("R0 policy accepted 262144 denominator lane" in item for item in completed_items):
    raise SystemExit("target contract must record denominator unavailable policy")
  if not any("exact 262144-token top-k smoke attempted" in item for item in completed_items):
    raise SystemExit("target contract must record 256k top-k exact-context attempt")
  if not any("R0 prompt-edge policy accepted exact 262144-token" in item for item in completed_items):
    raise SystemExit("target contract must record 256k prompt-edge policy")
  if not any("boundary capture route preflight confirmed" in item for item in completed_items):
    raise SystemExit("target contract must record boundary capture route preflight")
  if not any("llama.cpp source build route resolved" in item for item in completed_items):
    raise SystemExit("target contract must record llama.cpp source build route")
  if not any("llama.cpp instrumentation map resolved all 17 oracle boundary types" in item for item in completed_items):
    raise SystemExit("target contract must record llama.cpp instrumentation map")
  if not any("boundary capture instrumentation patch added" in item for item in completed_items):
    raise SystemExit("target contract must record boundary capture instrumentation patch")
  if not any("boundary capture executable built successfully" in item for item in completed_items):
    raise SystemExit("target contract must record boundary capture build")
  if not any("enhanced boundary capture run succeeded on the locked model" in item for item in completed_items):
    raise SystemExit("target contract must record boundary capture run")
  if not any("hybrid-aware boundary capture coverage preflight found effective policy coverage" in item for item in completed_items):
    raise SystemExit("target contract must record boundary capture coverage")
  if not any("boundary bundle fragment assembled 524 input rows and 524 output rows" in item for item in completed_items):
    raise SystemExit("target contract must record boundary bundle fragment")
  if not any("full oracle bundle assembled and validated" in item for item in completed_items):
    raise SystemExit("target contract must record full oracle bundle validation")
  if any("denominator" in item for item in pending_items):
    raise SystemExit("target contract must not leave denominator item pending")
  if "full oracle bundle capture including full acceptance teacher-forced distributions and per-boundary tensors" in pending_items:
    raise SystemExit("target contract must not keep full oracle bundle pending after validation")
  if "resident harness load(model, oracle_bundle)" in pending_items:
    raise SystemExit("target contract must not keep resident harness load pending after load artifact")
  if not any("resident harness load(model, oracle_bundle) executed successfully" in item for item in completed_items):
    raise SystemExit("target contract must record successful resident harness load")

  validate_prompt_suites(parsed)

  oracle = parsed["oracle/oracle-bundle-contract.json"]
  if oracle.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
    raise SystemExit("oracle contract workstream mismatch")
  required_fields = set(oracle.get("required_bundle_fields", []))
  for field in (
      "token_ids",
      "top_k_logprobs",
      "teacher_forced_distribution_references",
      "per_boundary_reference_inputs",
      "per_boundary_reference_outputs",
  ):
    if field not in required_fields:
      raise SystemExit(f"oracle contract missing required field: {field}")
  if oracle.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("oracle gate must be closed after full bundle validation")
  if oracle.get("missing_for_r0_close") != []:
    raise SystemExit("oracle contract must not list missing oracle outputs after full bundle validation")
  required_bundle_paths = oracle.get("required_bundle_paths", [])
  if required_bundle_paths != [
      "manifest.json",
      "correctness.json",
      "token-topk-references.jsonl",
      "teacher-forced-distribution-references.jsonl",
      "boundary-references/inputs.jsonl",
      "boundary-references/outputs.jsonl",
  ]:
    raise SystemExit("oracle contract required bundle paths mismatch")
  seed = oracle.get("available_seed_artifacts", {}).get(
      "deterministic_cpu_llama_cpp_token_topk_seed", {}
  )
  if seed.get("available") is not True:
    raise SystemExit("oracle contract must record the deterministic CPU seed")
  staging_tool = seed.get("staging_tool")
  if staging_tool != "tools/intel-qwen36-oracle-seed-stage.py":
    raise SystemExit("oracle seed staging tool mismatch")
  if not (ROOT / staging_tool).exists():
    raise SystemExit("oracle seed staging tool missing")
  replay_tool = seed.get("replay_tool")
  if replay_tool != "tools/intel-qwen36-oracle-seed-replay.py":
    raise SystemExit("oracle seed replay tool mismatch")
  if not (ROOT / replay_tool).exists():
    raise SystemExit("oracle seed replay tool missing")
  if seed.get("staged_schema_version") != "intel-qwen36-oracle-seed-stage-v0":
    raise SystemExit("oracle seed staged schema mismatch")
  replay_self_check = seed.get("latest_replay_self_check", {})
  if replay_self_check.get("required_checks_passed") is not True:
    raise SystemExit("oracle replay self-check must pass before registration")
  distribution_tool = seed.get("teacher_forced_distribution_seed_tool")
  if distribution_tool != "tools/intel-qwen36-teacher-forced-seed-stage.py":
    raise SystemExit("teacher-forced distribution seed tool mismatch")
  if not (ROOT / distribution_tool).exists():
    raise SystemExit("teacher-forced distribution seed tool missing")
  if (
      seed.get("teacher_forced_distribution_seed_schema_version")
      != "intel-qwen36-teacher-forced-distribution-seed-v0"
  ):
    raise SystemExit("teacher-forced distribution seed schema mismatch")
  distribution_seed = seed.get("latest_teacher_forced_distribution_seed", {})
  if distribution_seed.get("required_checks_passed") is not True:
    raise SystemExit("teacher-forced distribution seed checks must pass")
  if distribution_seed.get("full_acceptance_bundle") is not False:
    raise SystemExit("teacher-forced distribution seed must not claim full acceptance")
  if distribution_seed.get("distribution_positions") != 91:
    raise SystemExit("teacher-forced distribution seed position count mismatch")
  for subgate in (
      "prompt_token_ids",
      "first_token_top_k_logprobs",
      "short_greedy_generated_token_ids",
      "short_router_teacher_forced_distribution_seed",
  ):
    if subgate not in seed.get("available_subgates", []):
      raise SystemExit(f"oracle seed missing available subgate: {subgate}")
  capture_plan = oracle.get("capture_plan", {})
  required_outputs = capture_plan.get("next_required_outputs", [])
  if required_outputs != []:
    raise SystemExit("oracle capture plan must have no remaining R0 oracle outputs")
  latest_resolution = capture_plan.get("latest_resolution", {})
  if (
      latest_resolution.get("tool")
      != "tools/intel-qwen36-r0-denominator-oracle-boundary-resolution.py"
  ):
    raise SystemExit("latest denominator/oracle resolution tool mismatch")
  if latest_resolution.get("required_checks_passed") is not True:
    raise SystemExit("latest denominator/oracle resolution checks must pass")
  if (
      latest_resolution.get("denominator_interpretation")
      != "openvino_262144_resource_failure_not_metric"
  ):
    raise SystemExit("latest denominator interpretation mismatch")
  if (
      latest_resolution.get("llama_denominator_interpretation")
      != "llama_vulkan_262144_timeout_no_metric_cleanup_complete"
  ):
    raise SystemExit("latest llama denominator interpretation mismatch")
  if latest_resolution.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest denominator/oracle resolution must keep oracle gate open")
  if latest_resolution.get("boundary_count") != len(oracle.get("boundary_types", [])):
    raise SystemExit("latest denominator/oracle resolution boundary count mismatch")
  latest_policy = capture_plan.get("latest_denominator_unavailable_policy", {})
  if (
      latest_policy.get("tool")
      != "tools/intel-qwen36-r0-denominator-unavailable-policy.py"
  ):
    raise SystemExit("latest denominator unavailable policy tool mismatch")
  if latest_policy.get("denominator_policy_gate_closed") is not True:
    raise SystemExit("latest denominator unavailable policy gate must be closed")
  if latest_policy.get("denominator_metric_available") is not False:
    raise SystemExit("latest denominator unavailable policy must not claim metric")
  if latest_policy.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest denominator unavailable policy must forbid speedups")
  policy_path = latest_policy.get("path")
  if not isinstance(policy_path, str) or not policy_path:
    raise SystemExit("latest denominator unavailable policy path missing")
  policy_dir = ROOT / policy_path
  policy_json_path = policy_dir / "policy.json"
  policy_correctness_path = policy_dir / "correctness.json"
  if not policy_json_path.exists() or not policy_correctness_path.exists():
    raise SystemExit("latest denominator unavailable policy artifact missing")
  policy_json = json.loads(policy_json_path.read_text(encoding="utf-8"))
  if (
      policy_json.get("policy", {}).get("r0_denominator_gate_status")
      != "closed_by_unavailable_lane_policy"
  ):
    raise SystemExit("latest denominator unavailable policy status mismatch")
  if policy_json.get("policy", {}).get("r0_closed") is not False:
    raise SystemExit("latest denominator unavailable policy must keep R0 open")
  policy_correctness = json.loads(policy_correctness_path.read_text(encoding="utf-8"))
  if policy_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest denominator unavailable policy checks failed")
  latest_capture_spec = capture_plan.get("latest_oracle_capture_spec", {})
  if latest_capture_spec.get("tool") != "tools/intel-qwen36-r0-oracle-capture-spec.py":
    raise SystemExit("latest oracle capture spec tool mismatch")
  if latest_capture_spec.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle capture spec checks must pass")
  if latest_capture_spec.get("boundary_type_count") != len(oracle.get("boundary_types", [])):
    raise SystemExit("latest oracle capture spec boundary count mismatch")
  if latest_capture_spec.get("per_layer_boundary_record_count") != 520:
    raise SystemExit("latest oracle capture spec per-layer count mismatch")
  if latest_capture_spec.get("prompt_row_count") != 26:
    raise SystemExit("latest oracle capture spec prompt row count mismatch")
  if latest_capture_spec.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle capture spec must keep oracle gate open")
  capture_spec_path = latest_capture_spec.get("path")
  if not isinstance(capture_spec_path, str) or not capture_spec_path:
    raise SystemExit("latest oracle capture spec path missing")
  capture_spec_dir = ROOT / capture_spec_path
  capture_spec_json_path = capture_spec_dir / "capture-spec.json"
  capture_spec_correctness_path = capture_spec_dir / "correctness.json"
  if not capture_spec_json_path.exists() or not capture_spec_correctness_path.exists():
    raise SystemExit("latest oracle capture spec artifact missing")
  capture_spec_json = json.loads(capture_spec_json_path.read_text(encoding="utf-8"))
  if capture_spec_json.get("r0_oracle_gate_status", {}).get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle capture spec must not close oracle gate")
  if capture_spec_json.get("coverage", {}).get("per_layer_boundary_record_count") != 520:
    raise SystemExit("latest oracle capture spec artifact per-layer count mismatch")
  capture_spec_correctness = json.loads(
      capture_spec_correctness_path.read_text(encoding="utf-8")
  )
  if capture_spec_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle capture spec artifact checks failed")
  latest_runtime_preflight = capture_plan.get("latest_oracle_runtime_preflight", {})
  if (
      latest_runtime_preflight.get("tool")
      != "tools/intel-qwen36-r0-oracle-runtime-preflight.py"
  ):
    raise SystemExit("latest oracle runtime preflight tool mismatch")
  if latest_runtime_preflight.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle runtime preflight checks must pass")
  if (
      latest_runtime_preflight.get("teacher_forced_distribution_route_status")
      != "candidate_prior_llama_oracle_tool_completion_probabilities_route"
  ):
    raise SystemExit("latest oracle runtime preflight distribution route mismatch")
  if latest_runtime_preflight.get("tokenization_route_status") != "candidate_llama_tokenize_route":
    raise SystemExit("latest oracle runtime preflight tokenization route mismatch")
  if (
      latest_runtime_preflight.get("per_boundary_tensor_route_status")
      != "missing_stock_boundary_tensor_capture_route"
  ):
    raise SystemExit("latest oracle runtime preflight boundary route mismatch")
  if latest_runtime_preflight.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle runtime preflight must keep oracle gate open")
  runtime_preflight_path = latest_runtime_preflight.get("path")
  if not isinstance(runtime_preflight_path, str) or not runtime_preflight_path:
    raise SystemExit("latest oracle runtime preflight path missing")
  runtime_preflight_dir = ROOT / runtime_preflight_path
  runtime_preflight_json_path = runtime_preflight_dir / "preflight.json"
  runtime_preflight_correctness_path = runtime_preflight_dir / "correctness.json"
  if (
      not runtime_preflight_json_path.exists()
      or not runtime_preflight_correctness_path.exists()
  ):
    raise SystemExit("latest oracle runtime preflight artifact missing")
  runtime_preflight_json = json.loads(
      runtime_preflight_json_path.read_text(encoding="utf-8")
  )
  runtime_routes = runtime_preflight_json.get("oracle_runtime_routes", {})
  if (
      runtime_routes.get("teacher_forced_distribution", {}).get("candidate_route_present")
      is not True
  ):
    raise SystemExit("latest oracle runtime preflight artifact missing distribution route")
  if runtime_routes.get("tokenization", {}).get("candidate_route_present") is not True:
    raise SystemExit("latest oracle runtime preflight artifact missing tokenization route")
  if (
      runtime_routes.get("per_boundary_tensors", {}).get("candidate_route_present")
      is not False
  ):
    raise SystemExit("latest oracle runtime preflight artifact must reject stock boundary route")
  if runtime_preflight_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle runtime preflight artifact must keep gate open")
  runtime_inventory = runtime_preflight_json.get("target_inventory", {})
  if runtime_inventory.get("model_size_matches_contract") is not True:
    raise SystemExit("latest oracle runtime preflight artifact model mismatch")
  runtime_preflight_correctness = json.loads(
      runtime_preflight_correctness_path.read_text(encoding="utf-8")
  )
  if runtime_preflight_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle runtime preflight artifact checks failed")
  latest_boundary_route_preflight = capture_plan.get(
      "latest_boundary_capture_route_preflight", {}
  )
  if (
      latest_boundary_route_preflight.get("tool")
      != "tools/intel-qwen36-r0-boundary-capture-route-preflight.py"
  ):
    raise SystemExit("latest boundary capture route preflight tool mismatch")
  if latest_boundary_route_preflight.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture route preflight checks must pass")
  if latest_boundary_route_preflight.get("boundary_input_task_count") != 524:
    raise SystemExit("latest boundary capture route preflight input count mismatch")
  if latest_boundary_route_preflight.get("boundary_output_task_count") != 524:
    raise SystemExit("latest boundary capture route preflight output count mismatch")
  if latest_boundary_route_preflight.get("boundary_type_count") != len(
      oracle.get("boundary_types", [])
  ):
    raise SystemExit("latest boundary capture route preflight type count mismatch")
  if (
      latest_boundary_route_preflight.get("selected_next_route")
      != "stage_exact_llama_cpp_source_commit_and_build_with_intel_env"
  ):
    raise SystemExit("latest boundary capture route preflight next route mismatch")
  if (
      latest_boundary_route_preflight.get("route_status")
      != "missing_instrumentable_llama_cpp_source_tree_on_target"
  ):
    raise SystemExit("latest boundary capture route preflight status mismatch")
  if latest_boundary_route_preflight.get("llama_install_binary_only") is not True:
    raise SystemExit("latest boundary capture route preflight must record binary-only install")
  if latest_boundary_route_preflight.get("llama_source_tree_present") is not False:
    raise SystemExit("latest boundary capture route preflight must record missing source tree")
  if latest_boundary_route_preflight.get("intel_env_build_tools_present") is not True:
    raise SystemExit("latest boundary capture route preflight must record Intel env toolchain")
  if latest_boundary_route_preflight.get("llama_runtime_build") != 9518:
    raise SystemExit("latest boundary capture route preflight build mismatch")
  if latest_boundary_route_preflight.get("llama_runtime_commit_short") != "7c158fbb4":
    raise SystemExit("latest boundary capture route preflight commit mismatch")
  if (
      latest_boundary_route_preflight.get(
          "current_environment_can_capture_boundary_bundle_now"
      )
      is not False
  ):
    raise SystemExit("latest boundary capture route preflight must not claim bundle route ready")
  if latest_boundary_route_preflight.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture route preflight must keep oracle gate open")
  boundary_route_path = latest_boundary_route_preflight.get("path")
  if not isinstance(boundary_route_path, str) or not boundary_route_path:
    raise SystemExit("latest boundary capture route preflight path missing")
  boundary_route_dir = ROOT / boundary_route_path
  boundary_route_json_path = boundary_route_dir / "preflight.json"
  boundary_route_correctness_path = boundary_route_dir / "correctness.json"
  if (
      not boundary_route_json_path.exists()
      or not boundary_route_correctness_path.exists()
  ):
    raise SystemExit("latest boundary capture route preflight artifact missing")
  boundary_route_json = json.loads(
      boundary_route_json_path.read_text(encoding="utf-8")
  )
  boundary_requirements = boundary_route_json.get("boundary_capture_requirements", {})
  if boundary_requirements.get("boundary_input_task_count") != 524:
    raise SystemExit("latest boundary capture route preflight artifact input mismatch")
  if boundary_requirements.get("boundary_output_task_count") != 524:
    raise SystemExit("latest boundary capture route preflight artifact output mismatch")
  if boundary_requirements.get("boundary_type_count") != len(
      oracle.get("boundary_types", [])
  ):
    raise SystemExit("latest boundary capture route preflight artifact type mismatch")
  if boundary_requirements.get("source_prompt_case_ids") != ["short_math_001"]:
    raise SystemExit("latest boundary capture route preflight source case mismatch")
  if boundary_requirements.get("source_token_positions") != [15]:
    raise SystemExit("latest boundary capture route preflight source position mismatch")
  boundary_target = boundary_route_json.get("target_footholds", {})
  if boundary_target.get("llama_install_binary_only") is not True:
    raise SystemExit("latest boundary capture route preflight artifact binary-only mismatch")
  if boundary_target.get("llama_source_tree_present") is not False:
    raise SystemExit("latest boundary capture route preflight artifact source-tree mismatch")
  if boundary_target.get("intel_env_build_tools_present") is not True:
    raise SystemExit("latest boundary capture route preflight artifact toolchain mismatch")
  llama_runtime = boundary_target.get("llama_runtime_version", {})
  if (
      llama_runtime.get("build_number") != 9518
      or llama_runtime.get("commit_short") != "7c158fbb4"
  ):
    raise SystemExit("latest boundary capture route preflight artifact version mismatch")
  if boundary_target.get("model_size_matches_contract") is not True:
    raise SystemExit("latest boundary capture route preflight artifact model mismatch")
  if boundary_target.get("locked_model_process_present") is not False:
    raise SystemExit("latest boundary capture route preflight artifact found locked server")
  boundary_route_decision = boundary_route_json.get("route_decision", {})
  if (
      boundary_route_decision.get("stock_boundary_route_status")
      != "missing_stock_boundary_tensor_capture_route"
  ):
    raise SystemExit("latest boundary capture route preflight stock route mismatch")
  if (
      boundary_route_decision.get("current_environment_can_capture_boundary_bundle_now")
      is not False
  ):
    raise SystemExit("latest boundary capture route preflight artifact must keep route unavailable")
  if boundary_route_decision.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture route preflight artifact must keep oracle gate open")
  boundary_route_correctness = json.loads(
      boundary_route_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_route_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture route preflight artifact checks failed")
  model_boundary_route = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_capture_route_preflight", {}
  )
  if model_boundary_route != latest_boundary_route_preflight:
    raise SystemExit("model/oracle latest boundary capture route preflight mismatch")
  latest_source_build_route = capture_plan.get("latest_llama_source_build_route", {})
  if (
      latest_source_build_route.get("tool")
      != "tools/intel-qwen36-r0-llama-source-build-route.py"
  ):
    raise SystemExit("latest llama source build route tool mismatch")
  if latest_source_build_route.get("required_checks_passed") is not True:
    raise SystemExit("latest llama source build route checks must pass")
  if latest_source_build_route.get("target_llama_build") != 9518:
    raise SystemExit("latest llama source build route build mismatch")
  if latest_source_build_route.get("target_llama_commit_short") != "7c158fbb4":
    raise SystemExit("latest llama source build route commit mismatch")
  if (
      latest_source_build_route.get("upstream_commit_sha")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest llama source build route upstream commit mismatch")
  if latest_source_build_route.get("upstream_commit_resolved") is not True:
    raise SystemExit("latest llama source build route must resolve upstream")
  if latest_source_build_route.get("target_source_stage_attempted") is not True:
    raise SystemExit("latest llama source build route must stage target source")
  if latest_source_build_route.get("source_ready_for_instrumentation") is not True:
    raise SystemExit("latest llama source build route source must be ready")
  if (
      latest_source_build_route.get("instrumentation_route_status")
      != "source_staged_ready_for_instrumentation"
  ):
    raise SystemExit("latest llama source build route status mismatch")
  if latest_source_build_route.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest llama source build route must keep oracle gate open")
  source_route_path = latest_source_build_route.get("path")
  if not isinstance(source_route_path, str) or not source_route_path:
    raise SystemExit("latest llama source build route path missing")
  source_route_dir = ROOT / source_route_path
  source_route_json_path = source_route_dir / "source-route.json"
  source_route_correctness_path = source_route_dir / "correctness.json"
  if not source_route_json_path.exists() or not source_route_correctness_path.exists():
    raise SystemExit("latest llama source build route artifact missing")
  source_route_json = json.loads(source_route_json_path.read_text(encoding="utf-8"))
  source_route = source_route_json.get("source_route", {})
  upstream_commit = source_route.get("upstream_commit", {})
  if upstream_commit.get("resolved") is not True:
    raise SystemExit("latest llama source build route artifact upstream unresolved")
  if (
      upstream_commit.get("sha")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest llama source build route artifact upstream SHA mismatch")
  source_stage = source_route_json.get("target_source_stage", {})
  if source_stage.get("source_ready_for_instrumentation") is not True:
    raise SystemExit("latest llama source build route artifact source not ready")
  if (
      source_stage.get("source_rev_parse")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest llama source build route artifact stage SHA mismatch")
  if source_stage.get("cmakelists_present") is not True:
    raise SystemExit("latest llama source build route artifact missing CMakeLists")
  if source_stage.get("llama_cpp_file_present") is not True:
    raise SystemExit("latest llama source build route artifact missing src/llama.cpp")
  if source_stage.get("ggml_dir_present") is not True:
    raise SystemExit("latest llama source build route artifact missing ggml")
  if source_route_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest llama source build route artifact must keep oracle gate open")
  source_route_correctness = json.loads(
      source_route_correctness_path.read_text(encoding="utf-8")
  )
  if source_route_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest llama source build route artifact checks failed")
  model_source_route = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_llama_source_build_route", {}
  )
  if model_source_route != latest_source_build_route:
    raise SystemExit("model/oracle latest llama source build route mismatch")
  latest_instrumentation_map = capture_plan.get("latest_llama_instrumentation_map", {})
  if (
      latest_instrumentation_map.get("tool")
      != "tools/intel-qwen36-r0-llama-instrumentation-map.py"
  ):
    raise SystemExit("latest llama instrumentation map tool mismatch")
  if latest_instrumentation_map.get("required_checks_passed") is not True:
    raise SystemExit("latest llama instrumentation map checks must pass")
  if (
      latest_instrumentation_map.get("schema_version")
      != "intel-qwen36-r0-llama-instrumentation-map-v0"
  ):
    raise SystemExit("latest llama instrumentation map schema mismatch")
  if (
      latest_instrumentation_map.get("source_rev_parse")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest llama instrumentation map source SHA mismatch")
  if latest_instrumentation_map.get("source_status_short_count") != 0:
    raise SystemExit("latest llama instrumentation map source must be clean")
  if (
      latest_instrumentation_map.get("target_source_stage_dir")
      != "/home/intel/intel-qwen36-r0/source/llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest llama instrumentation map source dir mismatch")
  if latest_instrumentation_map.get("required_boundary_type_count") != len(
      oracle.get("boundary_types", [])
  ):
    raise SystemExit("latest llama instrumentation map required count mismatch")
  if latest_instrumentation_map.get("mapped_boundary_type_count") != len(
      oracle.get("boundary_types", [])
  ):
    raise SystemExit("latest llama instrumentation map mapped count mismatch")
  if (
      latest_instrumentation_map.get("route_status")
      != "source_mapped_ready_for_instrumentation_patch"
  ):
    raise SystemExit("latest llama instrumentation map status mismatch")
  if latest_instrumentation_map.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest llama instrumentation map must keep oracle gate open")
  instrumentation_map_path = latest_instrumentation_map.get("path")
  if not isinstance(instrumentation_map_path, str) or not instrumentation_map_path:
    raise SystemExit("latest llama instrumentation map path missing")
  instrumentation_map_dir = ROOT / instrumentation_map_path
  instrumentation_map_json_path = instrumentation_map_dir / "instrumentation-map.json"
  instrumentation_map_correctness_path = instrumentation_map_dir / "correctness.json"
  if (
      not instrumentation_map_json_path.exists()
      or not instrumentation_map_correctness_path.exists()
  ):
    raise SystemExit("latest llama instrumentation map artifact missing")
  instrumentation_map_json = json.loads(
      instrumentation_map_json_path.read_text(encoding="utf-8")
  )
  if instrumentation_map_json.get("schema_version") != latest_instrumentation_map.get(
      "schema_version"
  ):
    raise SystemExit("latest llama instrumentation map artifact schema mismatch")
  source_stage = instrumentation_map_json.get("source_stage", {})
  if source_stage.get("source_matches_expected_sha") is not True:
    raise SystemExit("latest llama instrumentation map artifact source SHA mismatch")
  if source_stage.get("source_matches_source_route") is not True:
    raise SystemExit("latest llama instrumentation map artifact source-route mismatch")
  if source_stage.get("source_rev_parse") != latest_instrumentation_map.get(
      "source_rev_parse"
  ):
    raise SystemExit("latest llama instrumentation map artifact source rev mismatch")
  if source_stage.get("source_status_short_count") != 0:
    raise SystemExit("latest llama instrumentation map artifact source must be clean")
  for key in (
      "qwen35moe_cpp_present",
      "llama_graph_cpp_present",
      "llama_graph_h_present",
      "llama_model_cpp_present",
      "llama_sampler_cpp_present",
  ):
    if source_stage.get(key) is not True:
      raise SystemExit(f"latest llama instrumentation map artifact missing {key}")
  coverage = instrumentation_map_json.get("coverage", {})
  if coverage.get("required_boundary_type_count") != len(oracle.get("boundary_types", [])):
    raise SystemExit("latest llama instrumentation map artifact required count mismatch")
  if coverage.get("mapped_boundary_type_count") != len(oracle.get("boundary_types", [])):
    raise SystemExit("latest llama instrumentation map artifact mapped count mismatch")
  if coverage.get("missing_boundary_types") != [] or coverage.get("extra_boundary_types") != []:
    raise SystemExit("latest llama instrumentation map artifact coverage mismatch")
  if coverage.get("unmapped_location_boundary_types") != []:
    raise SystemExit("latest llama instrumentation map artifact has unmapped locations")
  architecture_route = instrumentation_map_json.get("architecture_route", {})
  for key in ("dispatch_location", "main_graph_location", "moe_helper_location"):
    location = architecture_route.get(key, {})
    if not isinstance(location.get("line_start"), int):
      raise SystemExit(f"latest llama instrumentation map artifact missing {key}")
  callback_surface = instrumentation_map_json.get("graph_callback_surface", {})
  for key in (
      "callback_type_location",
      "context_callback_location",
      "graph_context_callback_location",
  ):
    location = callback_surface.get(key, {})
    if not isinstance(location.get("line_start"), int):
      raise SystemExit(f"latest llama instrumentation map artifact missing callback {key}")
  expected_boundary_types = set(oracle.get("boundary_types", []))
  mappings = instrumentation_map_json.get("boundary_mappings", [])
  if {item.get("boundary_type") for item in mappings} != expected_boundary_types:
    raise SystemExit("latest llama instrumentation map boundary set mismatch")
  if len(mappings) != len(expected_boundary_types):
    raise SystemExit("latest llama instrumentation map boundary row count mismatch")
  for item in mappings:
    boundary_type = item.get("boundary_type")
    if not item.get("input_tensor_cues"):
      raise SystemExit(f"{boundary_type}: instrumentation map missing input cues")
    if not item.get("output_tensor_cues"):
      raise SystemExit(f"{boundary_type}: instrumentation map missing output cues")
    locations = item.get("source_locations", [])
    if not isinstance(locations, list) or not locations:
      raise SystemExit(f"{boundary_type}: instrumentation map missing locations")
    for location in locations:
      if not isinstance(location.get("file"), str) or not location.get("file"):
        raise SystemExit(f"{boundary_type}: instrumentation map location file missing")
      if not isinstance(location.get("line_start"), int):
        raise SystemExit(f"{boundary_type}: instrumentation map location line missing")
  if instrumentation_map_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest llama instrumentation map artifact must keep gate open")
  instrumentation_map_correctness = json.loads(
      instrumentation_map_correctness_path.read_text(encoding="utf-8")
  )
  if instrumentation_map_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest llama instrumentation map correctness failed")
  model_instrumentation_map = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_llama_instrumentation_map", {}
  )
  if model_instrumentation_map != latest_instrumentation_map:
    raise SystemExit("model/oracle latest llama instrumentation map mismatch")
  latest_boundary_patch = capture_plan.get(
      "latest_boundary_capture_instrumentation_patch", {}
  )
  if (
      latest_boundary_patch.get("tool")
      != "tools/intel-qwen36-r0-boundary-capture-instrumentation-patch.py"
  ):
    raise SystemExit("latest boundary capture instrumentation patch tool mismatch")
  if (
      latest_boundary_patch.get("schema_version")
      != "intel-qwen36-r0-boundary-capture-instrumentation-patch-v0"
  ):
    raise SystemExit("latest boundary capture instrumentation patch schema mismatch")
  if latest_boundary_patch.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture instrumentation patch checks must pass")
  if latest_boundary_patch.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture instrumentation patch must keep gate open")
  if (
      latest_boundary_patch.get("target_source_stage_dir")
      != "/home/intel/intel-qwen36-r0/source/llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest boundary capture instrumentation patch source dir mismatch")
  if (
      latest_boundary_patch.get("source_rev_parse")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest boundary capture instrumentation patch source SHA mismatch")
  if latest_boundary_patch.get("source_status_short_count_after_patch") != 2:
    raise SystemExit("latest boundary capture instrumentation patch dirty count mismatch")
  if (
      latest_boundary_patch.get("route_status")
      != "target_source_patched_ready_for_build"
  ):
    raise SystemExit("latest boundary capture instrumentation patch status mismatch")
  if latest_boundary_patch.get("capture_tool_registered") is not True:
    raise SystemExit("latest boundary capture instrumentation patch must register tool")
  if latest_boundary_patch.get("capture_tool_present") is not True:
    raise SystemExit("latest boundary capture instrumentation patch tool missing")
  boundary_patch_path = latest_boundary_patch.get("path")
  if not isinstance(boundary_patch_path, str) or not boundary_patch_path:
    raise SystemExit("latest boundary capture instrumentation patch path missing")
  boundary_patch_dir = ROOT / boundary_patch_path
  boundary_patch_json_path = boundary_patch_dir / "patch-route.json"
  boundary_patch_correctness_path = boundary_patch_dir / "correctness.json"
  boundary_patch_diff_path = boundary_patch_dir / "boundary-capture-tool.patch"
  if (
      not boundary_patch_json_path.exists()
      or not boundary_patch_correctness_path.exists()
      or not boundary_patch_diff_path.exists()
  ):
    raise SystemExit("latest boundary capture instrumentation patch artifact missing")
  boundary_patch_json = json.loads(
      boundary_patch_json_path.read_text(encoding="utf-8")
  )
  if boundary_patch_json.get("schema_version") != latest_boundary_patch.get(
      "schema_version"
  ):
    raise SystemExit("latest boundary capture instrumentation patch artifact schema mismatch")
  if boundary_patch_json.get("route_status") != latest_boundary_patch.get(
      "route_status"
  ):
    raise SystemExit("latest boundary capture instrumentation patch artifact status mismatch")
  if boundary_patch_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture instrumentation patch artifact must keep gate open")
  patch_route = boundary_patch_json.get("patch_route", {})
  if patch_route.get("apply_target_patch") is not True:
    raise SystemExit("latest boundary capture instrumentation patch must be target-applied")
  if patch_route.get("git_apply_check_passed") is not True:
    raise SystemExit("latest boundary capture instrumentation patch apply-check failed")
  if patch_route.get("git_apply_passed") is not True:
    raise SystemExit("latest boundary capture instrumentation patch apply failed")
  patch_before = boundary_patch_json.get("target_source_before", {})
  if patch_before.get("source_status_short_count") != 0:
    raise SystemExit("latest boundary capture instrumentation patch before state not clean")
  if (
      patch_before.get("source_rev_parse")
      != "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
  ):
    raise SystemExit("latest boundary capture instrumentation patch before SHA mismatch")
  patch_after = boundary_patch_json.get("target_source_after", {})
  if patch_after.get("source_status_short_count") != 2:
    raise SystemExit("latest boundary capture instrumentation patch after dirty count mismatch")
  if patch_after.get("capture_cpp_present") is not True:
    raise SystemExit("latest boundary capture instrumentation patch missing capture cpp")
  if patch_after.get("capture_cmake_present") is not True:
    raise SystemExit("latest boundary capture instrumentation patch missing capture cmake")
  if patch_after.get("tools_cmake_registered") is not True:
    raise SystemExit("latest boundary capture instrumentation patch not registered")
  patch_status = patch_after.get("source_status_short", "")
  if "M tools/CMakeLists.txt" not in patch_status:
    raise SystemExit("latest boundary capture instrumentation patch missing CMakeLists change")
  if "?? tools/qwen36-boundary-capture/" not in patch_status:
    raise SystemExit("latest boundary capture instrumentation patch missing tool directory")
  boundary_patch_correctness = json.loads(
      boundary_patch_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_patch_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture instrumentation patch correctness failed")
  model_boundary_patch = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_capture_instrumentation_patch", {}
  )
  if model_boundary_patch != latest_boundary_patch:
    raise SystemExit("model/oracle latest boundary capture instrumentation patch mismatch")
  latest_boundary_build = capture_plan.get("latest_boundary_capture_build", {})
  if latest_boundary_build.get("tool") != "tools/intel-qwen36-r0-boundary-capture-build.py":
    raise SystemExit("latest boundary capture build tool mismatch")
  if (
      latest_boundary_build.get("schema_version")
      != "intel-qwen36-r0-boundary-capture-build-v0"
  ):
    raise SystemExit("latest boundary capture build schema mismatch")
  if latest_boundary_build.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture build checks must pass")
  if latest_boundary_build.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture build must keep gate open")
  if latest_boundary_build.get("target") != "llama-qwen36-boundary-capture":
    raise SystemExit("latest boundary capture build target mismatch")
  if (
      latest_boundary_build.get("remote_build_dir")
      != "/home/intel/intel-qwen36-r0/build/llama-qwen36-boundary-capture-20260627T051710Z"
  ):
    raise SystemExit("latest boundary capture build dir mismatch")
  if (
      latest_boundary_build.get("executable_path")
      != "/home/intel/intel-qwen36-r0/build/llama-qwen36-boundary-capture-20260627T051710Z/bin/llama-qwen36-boundary-capture"
  ):
    raise SystemExit("latest boundary capture build executable path mismatch")
  for key in ("configure_returncode", "build_returncode", "help_returncode"):
    if latest_boundary_build.get(key) != 0:
      raise SystemExit(f"latest boundary capture build {key} mismatch")
  if latest_boundary_build.get("executable_present") is not True:
    raise SystemExit("latest boundary capture build executable missing")
  if (
      latest_boundary_build.get("route_status")
      != "boundary_capture_executable_built"
  ):
    raise SystemExit("latest boundary capture build status mismatch")
  boundary_build_path = latest_boundary_build.get("path")
  if not isinstance(boundary_build_path, str) or not boundary_build_path:
    raise SystemExit("latest boundary capture build path missing")
  boundary_build_dir = ROOT / boundary_build_path
  boundary_build_json_path = boundary_build_dir / "build.json"
  boundary_build_correctness_path = boundary_build_dir / "correctness.json"
  if not boundary_build_json_path.exists() or not boundary_build_correctness_path.exists():
    raise SystemExit("latest boundary capture build artifact missing")
  boundary_build_json = json.loads(boundary_build_json_path.read_text(encoding="utf-8"))
  if boundary_build_json.get("schema_version") != latest_boundary_build.get(
      "schema_version"
  ):
    raise SystemExit("latest boundary capture build artifact schema mismatch")
  if boundary_build_json.get("route_status") != latest_boundary_build.get("route_status"):
    raise SystemExit("latest boundary capture build artifact status mismatch")
  build_route = boundary_build_json.get("build_route", {})
  if build_route.get("target") != latest_boundary_build.get("target"):
    raise SystemExit("latest boundary capture build artifact target mismatch")
  if build_route.get("remote_build_dir") != latest_boundary_build.get("remote_build_dir"):
    raise SystemExit("latest boundary capture build artifact dir mismatch")
  if build_route.get("executable_path") != latest_boundary_build.get("executable_path"):
    raise SystemExit("latest boundary capture build artifact executable mismatch")
  for key in ("configure_returncode", "build_returncode", "help_returncode"):
    if build_route.get(key) != 0:
      raise SystemExit(f"latest boundary capture build artifact {key} mismatch")
  if build_route.get("executable_present") is not True:
    raise SystemExit("latest boundary capture build artifact executable missing")
  build_source_state = boundary_build_json.get("source_state", {})
  if build_source_state.get("source_status_short_count") != 2:
    raise SystemExit("latest boundary capture build artifact source dirty count mismatch")
  build_status = build_source_state.get("source_status_short", "")
  if "M tools/CMakeLists.txt" not in build_status:
    raise SystemExit("latest boundary capture build artifact missing CMakeLists change")
  if "?? tools/qwen36-boundary-capture/" not in build_status:
    raise SystemExit("latest boundary capture build artifact missing tool directory")
  boundary_build_correctness = json.loads(
      boundary_build_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_build_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture build correctness failed")
  model_boundary_build = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_capture_build", {}
  )
  if model_boundary_build != latest_boundary_build:
    raise SystemExit("model/oracle latest boundary capture build mismatch")
  latest_boundary_run = capture_plan.get("latest_boundary_capture_run", {})
  if latest_boundary_run.get("tool") != "tools/intel-qwen36-r0-boundary-capture-run.py":
    raise SystemExit("latest boundary capture run tool mismatch")
  if (
      latest_boundary_run.get("schema_version")
      != "intel-qwen36-r0-boundary-capture-run-v0"
  ):
    raise SystemExit("latest boundary capture run schema mismatch")
  if latest_boundary_run.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture run checks must pass")
  if latest_boundary_run.get("full_oracle_bundle") is not False:
    raise SystemExit("latest boundary capture run must not claim full bundle")
  if latest_boundary_run.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture run must keep gate open")
  expected_capture_run_scalars = {
      "case_id": "short_math_001",
      "source_token_position": 15,
      "prompt_token_count": 16,
      "captured_tensor_count": 1493,
      "tensor_jsonl_row_count": 1493,
      "payload_file_count": 1493,
      "payload_bytes_total": 83628864,
      "unique_tensor_name_count": 1473,
      "logits_present": True,
      "extra_filter_count": 16,
      "route_status": "boundary_capture_run_succeeded",
  }
  for key, expected in expected_capture_run_scalars.items():
    if latest_boundary_run.get(key) != expected:
      raise SystemExit(f"latest boundary capture run {key} mismatch")
  boundary_run_path = latest_boundary_run.get("path")
  if not isinstance(boundary_run_path, str) or not boundary_run_path:
    raise SystemExit("latest boundary capture run path missing")
  boundary_run_dir = ROOT / boundary_run_path
  boundary_run_json_path = boundary_run_dir / "capture-run.json"
  boundary_run_correctness_path = boundary_run_dir / "correctness.json"
  boundary_run_tensor_jsonl_path = boundary_run_dir / "remote-output" / "tensor-dumps.jsonl"
  boundary_run_summary_path = boundary_run_dir / "remote-output" / "capture-summary.json"
  boundary_run_topk_path = boundary_run_dir / "remote-output" / "sampler-topk.json"
  boundary_run_payload_dir = boundary_run_dir / "remote-output" / "payloads"
  if (
      not boundary_run_json_path.exists()
      or not boundary_run_correctness_path.exists()
      or not boundary_run_tensor_jsonl_path.exists()
      or not boundary_run_summary_path.exists()
      or not boundary_run_topk_path.exists()
      or not boundary_run_payload_dir.is_dir()
  ):
    raise SystemExit("latest boundary capture run artifact missing")
  boundary_run_json = json.loads(boundary_run_json_path.read_text(encoding="utf-8"))
  if boundary_run_json.get("schema_version") != latest_boundary_run.get(
      "schema_version"
  ):
    raise SystemExit("latest boundary capture run artifact schema mismatch")
  if boundary_run_json.get("route_status") != latest_boundary_run.get("route_status"):
    raise SystemExit("latest boundary capture run artifact status mismatch")
  if boundary_run_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture run artifact must keep gate open")
  run_route = boundary_run_json.get("capture_run", {})
  if run_route.get("case_id") != "short_math_001":
    raise SystemExit("latest boundary capture run artifact case mismatch")
  if run_route.get("source_token_position") != 15:
    raise SystemExit("latest boundary capture run artifact source position mismatch")
  if run_route.get("prompt_observed_tokens") != 16:
    raise SystemExit("latest boundary capture run artifact prompt count mismatch")
  if run_route.get("n_ctx") != 32:
    raise SystemExit("latest boundary capture run artifact n_ctx mismatch")
  if run_route.get("ngl") != 0:
    raise SystemExit("latest boundary capture run artifact ngl mismatch")
  if run_route.get("max_tensors") != 0:
    raise SystemExit("latest boundary capture run artifact max_tensors mismatch")
  if len(run_route.get("extra_filters", [])) != 16:
    raise SystemExit("latest boundary capture run artifact filter count mismatch")
  if run_route.get("returncode") != 0 or run_route.get("timed_out") is not False:
    raise SystemExit("latest boundary capture run artifact command failed")
  if (
      run_route.get("executable_path")
      != latest_boundary_build.get("executable_path")
  ):
    raise SystemExit("latest boundary capture run executable mismatch")
  model_ref = run_route.get("model", {})
  if (
      model_ref.get("path")
      != "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
      or model_ref.get("sha256")
      != "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
  ):
    raise SystemExit("latest boundary capture run model mismatch")
  analysis = boundary_run_json.get("capture_analysis", {})
  if analysis.get("captured_tensor_count") != 1493:
    raise SystemExit("latest boundary capture run captured count mismatch")
  if analysis.get("tensor_jsonl_row_count") != 1493:
    raise SystemExit("latest boundary capture run JSONL count mismatch")
  if analysis.get("payload_file_count") != 1493:
    raise SystemExit("latest boundary capture run payload count mismatch")
  if analysis.get("payload_bytes_total") != 83628864:
    raise SystemExit("latest boundary capture run payload bytes mismatch")
  if analysis.get("tensor_bytes_total") != 83628864:
    raise SystemExit("latest boundary capture run tensor bytes mismatch")
  if analysis.get("unique_tensor_name_count") != 1473:
    raise SystemExit("latest boundary capture run unique tensor count mismatch")
  if analysis.get("logits_present") is not True:
    raise SystemExit("latest boundary capture run logits missing")
  if analysis.get("observed_positions") != [15]:
    raise SystemExit("latest boundary capture run observed position mismatch")
  boundary_run_correctness = json.loads(
      boundary_run_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_run_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture run correctness failed")
  boundary_run_summary = json.loads(
      boundary_run_summary_path.read_text(encoding="utf-8")
  )
  if (
      boundary_run_summary.get("captured_tensor_count") != 1493
      or boundary_run_summary.get("prompt_token_count") != 16
      or boundary_run_summary.get("source_token_position") != 15
      or boundary_run_summary.get("logits_present") is not True
  ):
    raise SystemExit("latest boundary capture run summary mismatch")
  boundary_run_topk = json.loads(boundary_run_topk_path.read_text(encoding="utf-8"))
  topk_rows = boundary_run_topk.get("top_k", [])
  if not isinstance(topk_rows, list) or len(topk_rows) < 5:
    raise SystemExit("latest boundary capture run top-k too small")
  if topk_rows[0].get("token_id") != 271:
    raise SystemExit("latest boundary capture run top-k token mismatch")
  tensor_rows = load_jsonl(boundary_run_tensor_jsonl_path)
  if len(tensor_rows) != 1493:
    raise SystemExit("latest boundary capture run tensor JSONL row count mismatch")
  payload_count = 0
  payload_bytes_total = 0
  observed_positions = set()
  tensor_names = set()
  for row in tensor_rows:
    observed_positions.add(row.get("observed_token_position"))
    tensor_name = row.get("tensor_name")
    if isinstance(tensor_name, str):
      tensor_names.add(tensor_name)
    payload_path = row.get("payload_path")
    if not isinstance(payload_path, str) or not payload_path:
      raise SystemExit("latest boundary capture run row missing payload path")
    payload_file = boundary_run_dir / "remote-output" / payload_path
    if not payload_file.exists():
      raise SystemExit(f"latest boundary capture run missing payload: {payload_path}")
    payload_count += 1
    payload_bytes_total += payload_file.stat().st_size
    if row.get("nbytes") != payload_file.stat().st_size:
      raise SystemExit("latest boundary capture run payload byte mismatch")
  if observed_positions != {15}:
    raise SystemExit("latest boundary capture run JSONL observed positions mismatch")
  if payload_count != 1493 or payload_bytes_total != 83628864:
    raise SystemExit("latest boundary capture run payload aggregate mismatch")
  for required_tensor_name in (
      "model.input_embed",
      "attn_norm-0",
      "linear_attn_qkv_mixed-0",
      "linear_attn_out-0",
      "ffn_moe_topk-0",
      "l_out-39",
      "ffn_out-39",
      "result_norm",
      "result_output",
  ):
    if required_tensor_name not in tensor_names:
      raise SystemExit(f"latest boundary capture run missing {required_tensor_name}")
  model_boundary_run = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_capture_run", {}
  )
  if model_boundary_run != latest_boundary_run:
    raise SystemExit("model/oracle latest boundary capture run mismatch")
  latest_boundary_coverage = capture_plan.get("latest_boundary_capture_coverage", {})
  if (
      latest_boundary_coverage.get("tool")
      != "tools/intel-qwen36-r0-boundary-capture-coverage.py"
  ):
    raise SystemExit("latest boundary capture coverage tool mismatch")
  if (
      latest_boundary_coverage.get("schema_version")
      != "intel-qwen36-r0-boundary-capture-coverage-v0"
  ):
    raise SystemExit("latest boundary capture coverage schema mismatch")
  if latest_boundary_coverage.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture coverage checks must pass")
  if latest_boundary_coverage.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture coverage must keep gate open")
  if latest_boundary_coverage.get("raw_capture_ready_for_bundle_mapping") is not True:
    raise SystemExit("latest boundary capture coverage must claim policy-ready raw capture")
  if latest_boundary_coverage.get("requires_hybrid_policy_for_bundle_mapping") is not True:
    raise SystemExit("latest boundary capture coverage must require hybrid policy")
  expected_coverage_scalars = {
      "input_task_count": 524,
      "output_task_count": 524,
      "input_direct_or_derived_match_count": 434,
      "output_direct_or_derived_match_count": 394,
      "input_all_cues_matched_count": 133,
      "output_all_cues_matched_count": 381,
      "input_effective_policy_match_count": 524,
      "output_effective_policy_match_count": 524,
      "policy_not_applicable_count": 60,
      "policy_equivalent_count": 150,
      "policy_derived_count": 40,
      "route_status": "raw_boundary_capture_effectively_covers_queue_with_hybrid_policy",
  }
  for key, expected in expected_coverage_scalars.items():
    if latest_boundary_coverage.get(key) != expected:
      raise SystemExit(f"latest boundary capture coverage {key} mismatch")
  boundary_coverage_path = latest_boundary_coverage.get("path")
  if not isinstance(boundary_coverage_path, str) or not boundary_coverage_path:
    raise SystemExit("latest boundary capture coverage path missing")
  boundary_coverage_dir = ROOT / boundary_coverage_path
  boundary_coverage_json_path = boundary_coverage_dir / "coverage.json"
  boundary_coverage_correctness_path = boundary_coverage_dir / "correctness.json"
  boundary_input_coverage_path = boundary_coverage_dir / "input-coverage.jsonl"
  boundary_output_coverage_path = boundary_coverage_dir / "output-coverage.jsonl"
  if (
      not boundary_coverage_json_path.exists()
      or not boundary_coverage_correctness_path.exists()
      or not boundary_input_coverage_path.exists()
      or not boundary_output_coverage_path.exists()
  ):
    raise SystemExit("latest boundary capture coverage artifact missing")
  boundary_coverage_json = json.loads(
      boundary_coverage_json_path.read_text(encoding="utf-8")
  )
  if boundary_coverage_json.get("schema_version") != latest_boundary_coverage.get(
      "schema_version"
  ):
    raise SystemExit("latest boundary capture coverage artifact schema mismatch")
  if boundary_coverage_json.get("route_status") != latest_boundary_coverage.get(
      "route_status"
  ):
    raise SystemExit("latest boundary capture coverage artifact status mismatch")
  if boundary_coverage_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary capture coverage artifact must keep gate open")
  if (
      boundary_coverage_json.get("evidence", {}).get("boundary_capture_run")
      != latest_boundary_run.get("path")
  ):
    raise SystemExit("latest boundary capture coverage run evidence mismatch")
  coverage = boundary_coverage_json.get("coverage", {})
  raw_capture = coverage.get("raw_capture", {})
  if raw_capture.get("tensor_jsonl_row_count") != 1493:
    raise SystemExit("latest boundary capture coverage raw row count mismatch")
  if raw_capture.get("unique_tensor_name_count") != 1473:
    raise SystemExit("latest boundary capture coverage unique tensor count mismatch")
  if raw_capture.get("sampler_topk_present") is not True:
    raise SystemExit("latest boundary capture coverage sampler evidence missing")
  if raw_capture.get("full_attention_layers") != [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]:
    raise SystemExit("latest boundary capture coverage full attention layers mismatch")
  if len(raw_capture.get("linear_attention_layers", [])) != 30:
    raise SystemExit("latest boundary capture coverage linear attention layer count mismatch")
  coverage_inputs = coverage.get("inputs", {})
  coverage_outputs = coverage.get("outputs", {})
  if coverage_inputs.get("task_count") != 524:
    raise SystemExit("latest boundary capture coverage input task count mismatch")
  if coverage_outputs.get("task_count") != 524:
    raise SystemExit("latest boundary capture coverage output task count mismatch")
  if coverage_inputs.get("direct_or_derived_match_count") != 434:
    raise SystemExit("latest boundary capture coverage input match count mismatch")
  if coverage_outputs.get("direct_or_derived_match_count") != 394:
    raise SystemExit("latest boundary capture coverage output match count mismatch")
  if coverage_inputs.get("all_cues_matched_count") != 133:
    raise SystemExit("latest boundary capture coverage input all-cues count mismatch")
  if coverage_outputs.get("all_cues_matched_count") != 381:
    raise SystemExit("latest boundary capture coverage output all-cues count mismatch")
  if coverage_inputs.get("effective_policy_match_count") != 524:
    raise SystemExit("latest boundary capture coverage input effective count mismatch")
  if coverage_outputs.get("effective_policy_match_count") != 524:
    raise SystemExit("latest boundary capture coverage output effective count mismatch")
  if coverage_inputs.get("effective_policy_match_complete") is not True:
    raise SystemExit("latest boundary capture coverage inputs must be policy-covered")
  if coverage_outputs.get("effective_policy_match_complete") is not True:
    raise SystemExit("latest boundary capture coverage outputs must be policy-covered")
  if coverage_inputs.get("direct_or_derived_match_complete") is not False:
    raise SystemExit("latest boundary capture coverage must mark input gaps")
  if coverage_outputs.get("direct_or_derived_match_complete") is not False:
    raise SystemExit("latest boundary capture coverage must mark output gaps")
  if coverage_inputs.get("missing_direct_or_derived_by_boundary") != {
      "attention": 30,
      "attention_output_projection": 30,
      "rope": 30,
  }:
    raise SystemExit("latest boundary capture coverage input gap distribution mismatch")
  if coverage_outputs.get("missing_direct_or_derived_by_boundary") != {
      "attention": 30,
      "moe_residual": 40,
      "qkv_projection": 30,
      "rope": 30,
  }:
    raise SystemExit("latest boundary capture coverage output gap distribution mismatch")
  boundary_coverage_correctness = json.loads(
      boundary_coverage_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_coverage_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary capture coverage correctness failed")
  input_coverage_rows = load_jsonl(boundary_input_coverage_path)
  output_coverage_rows = load_jsonl(boundary_output_coverage_path)
  if len(input_coverage_rows) != 524 or len(output_coverage_rows) != 524:
    raise SystemExit("latest boundary capture coverage row count mismatch")
  input_missing_any = sum(
      1 for row in input_coverage_rows
      if row.get("direct_or_derived_match") is False
  )
  output_missing_any = sum(
      1 for row in output_coverage_rows
      if row.get("direct_or_derived_match") is False
  )
  if input_missing_any != 90 or output_missing_any != 130:
    raise SystemExit("latest boundary capture coverage missing row count mismatch")
  input_missing_effective = sum(
      1 for row in input_coverage_rows
      if row.get("effective_policy_match") is False
  )
  output_missing_effective = sum(
      1 for row in output_coverage_rows
      if row.get("effective_policy_match") is False
  )
  if input_missing_effective != 0 or output_missing_effective != 0:
    raise SystemExit("latest boundary capture coverage effective row mismatch")
  model_boundary_coverage = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_capture_coverage", {}
  )
  if model_boundary_coverage != latest_boundary_coverage:
    raise SystemExit("model/oracle latest boundary capture coverage mismatch")
  latest_boundary_fragment = capture_plan.get("latest_boundary_bundle_fragment", {})
  if (
      latest_boundary_fragment.get("tool")
      != "tools/intel-qwen36-r0-boundary-bundle-fragment-assemble.py"
  ):
    raise SystemExit("latest boundary bundle fragment tool mismatch")
  if (
      latest_boundary_fragment.get("schema_version")
      != "intel-qwen36-r0-boundary-bundle-fragment-v0"
  ):
    raise SystemExit("latest boundary bundle fragment schema mismatch")
  if latest_boundary_fragment.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary bundle fragment checks must pass")
  if latest_boundary_fragment.get("full_oracle_bundle") is not False:
    raise SystemExit("latest boundary bundle fragment must not claim full bundle")
  if latest_boundary_fragment.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary bundle fragment must keep gate open")
  expected_fragment_scalars = {
      "input_rows": 524,
      "output_rows": 524,
      "policy_not_applicable_rows": 60,
      "derived_output_rows": 40,
      "captured_inline_rows": 1,
      "sampler_rows": 1,
      "route_status": "boundary_reference_fragment_assembled",
  }
  for key, expected in expected_fragment_scalars.items():
    if latest_boundary_fragment.get(key) != expected:
      raise SystemExit(f"latest boundary bundle fragment {key} mismatch")
  boundary_fragment_path = latest_boundary_fragment.get("path")
  if not isinstance(boundary_fragment_path, str) or not boundary_fragment_path:
    raise SystemExit("latest boundary bundle fragment path missing")
  boundary_fragment_dir = ROOT / boundary_fragment_path
  boundary_fragment_json_path = boundary_fragment_dir / "fragment.json"
  boundary_fragment_correctness_path = boundary_fragment_dir / "correctness.json"
  boundary_fragment_inputs_path = boundary_fragment_dir / "boundary-references" / "inputs.jsonl"
  boundary_fragment_outputs_path = boundary_fragment_dir / "boundary-references" / "outputs.jsonl"
  if (
      not boundary_fragment_json_path.exists()
      or not boundary_fragment_correctness_path.exists()
      or not boundary_fragment_inputs_path.exists()
      or not boundary_fragment_outputs_path.exists()
  ):
    raise SystemExit("latest boundary bundle fragment artifact missing")
  boundary_fragment_json = json.loads(
      boundary_fragment_json_path.read_text(encoding="utf-8")
  )
  if boundary_fragment_json.get("schema_version") != latest_boundary_fragment.get(
      "schema_version"
  ):
    raise SystemExit("latest boundary bundle fragment artifact schema mismatch")
  if boundary_fragment_json.get("route_status") != latest_boundary_fragment.get(
      "route_status"
  ):
    raise SystemExit("latest boundary bundle fragment artifact status mismatch")
  if boundary_fragment_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest boundary bundle fragment artifact must keep gate open")
  if (
      boundary_fragment_json.get("evidence", {}).get("boundary_capture_coverage")
      != latest_boundary_coverage.get("path")
  ):
    raise SystemExit("latest boundary bundle fragment coverage evidence mismatch")
  fragment_counts = boundary_fragment_json.get("fragment_counts", {})
  for key, expected in expected_fragment_scalars.items():
    if key == "route_status":
      continue
    if fragment_counts.get(key) != expected:
      raise SystemExit(f"latest boundary bundle fragment artifact {key} mismatch")
  boundary_fragment_correctness = json.loads(
      boundary_fragment_correctness_path.read_text(encoding="utf-8")
  )
  if boundary_fragment_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest boundary bundle fragment correctness failed")
  if boundary_fragment_correctness.get("full_oracle_bundle") is not False:
    raise SystemExit("latest boundary bundle fragment correctness must not claim full bundle")
  fragment_input_rows = load_jsonl(boundary_fragment_inputs_path)
  fragment_output_rows = load_jsonl(boundary_fragment_outputs_path)
  if len(fragment_input_rows) != 524 or len(fragment_output_rows) != 524:
    raise SystemExit("latest boundary bundle fragment JSONL row count mismatch")
  fragment_status_counts = {}
  for row in fragment_input_rows + fragment_output_rows:
    status = row.get("capture_status")
    fragment_status_counts[status] = fragment_status_counts.get(status, 0) + 1
    if row.get("capture_status") == "policy_not_applicable":
      if row.get("policy_id") != "qwen35moe_linear_attention_no_rope":
        raise SystemExit("latest boundary bundle fragment policy id mismatch")
      continue
    has_inline = (
        "reference_input_tensor" in row
        or "reference_output_tensor" in row
    )
    path_value = row.get("reference_input_tensor_path") or row.get(
        "reference_output_tensor_path"
    )
    if has_inline:
      continue
    if not isinstance(path_value, str) or not path_value:
      raise SystemExit("latest boundary bundle fragment row missing tensor path")
    payload = boundary_fragment_dir / path_value
    if not payload.exists():
      raise SystemExit(f"latest boundary bundle fragment missing payload: {path_value}")
  expected_status_counts = {
      "captured": 796,
      "captured_inline": 1,
      "captured_linear_attention_equivalent": 150,
      "captured_sampler_topk": 1,
      "derived_from_captured_tensors": 40,
      "policy_not_applicable": 60,
  }
  if fragment_status_counts != expected_status_counts:
    raise SystemExit("latest boundary bundle fragment status counts mismatch")
  derived_payload_count = len(list((boundary_fragment_dir / "payloads" / "derived").glob("*.bin")))
  if derived_payload_count != 40:
    raise SystemExit("latest boundary bundle fragment derived payload count mismatch")
  model_boundary_fragment = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {}).get(
      "latest_r0_boundary_bundle_fragment", {}
  )
  if model_boundary_fragment != latest_boundary_fragment:
    raise SystemExit("model/oracle latest boundary bundle fragment mismatch")
  latest_distribution_smoke = capture_plan.get("latest_distribution_capture_smoke", {})
  if (
      latest_distribution_smoke.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-smoke.py"
  ):
    raise SystemExit("latest distribution capture smoke tool mismatch")
  if latest_distribution_smoke.get("required_checks_passed") is not True:
    raise SystemExit("latest distribution capture smoke checks must pass")
  if latest_distribution_smoke.get("case_id") != "short_math_001":
    raise SystemExit("latest distribution capture smoke case mismatch")
  if latest_distribution_smoke.get("generated_token_count") != 1:
    raise SystemExit("latest distribution capture smoke token count mismatch")
  if latest_distribution_smoke.get("request_status") != 200:
    raise SystemExit("latest distribution capture smoke request status mismatch")
  if latest_distribution_smoke.get("top_logprobs_present") is not True:
    raise SystemExit("latest distribution capture smoke missing top-logprobs")
  if latest_distribution_smoke.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest distribution capture smoke must not claim full acceptance")
  if latest_distribution_smoke.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest distribution capture smoke must keep oracle gate open")
  distribution_smoke_path = latest_distribution_smoke.get("path")
  if not isinstance(distribution_smoke_path, str) or not distribution_smoke_path:
    raise SystemExit("latest distribution capture smoke path missing")
  distribution_smoke_dir = ROOT / distribution_smoke_path
  distribution_smoke_json_path = distribution_smoke_dir / "smoke.json"
  distribution_smoke_correctness_path = distribution_smoke_dir / "correctness.json"
  distribution_smoke_jsonl_path = distribution_smoke_dir / "distribution-smoke.jsonl"
  if (
      not distribution_smoke_json_path.exists()
      or not distribution_smoke_correctness_path.exists()
      or not distribution_smoke_jsonl_path.exists()
  ):
    raise SystemExit("latest distribution capture smoke artifact missing")
  distribution_smoke_json = json.loads(
      distribution_smoke_json_path.read_text(encoding="utf-8")
  )
  if distribution_smoke_json.get("required_checks_passed") is not True:
    raise SystemExit("latest distribution capture smoke artifact checks failed")
  if distribution_smoke_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest distribution capture smoke artifact must keep gate open")
  smoke_rows = load_jsonl(distribution_smoke_jsonl_path)
  if len(smoke_rows) != 1:
    raise SystemExit("latest distribution capture smoke JSONL must contain one row")
  smoke_row = smoke_rows[0]
  if smoke_row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
    raise SystemExit("latest distribution capture smoke row workstream mismatch")
  if smoke_row.get("case_id") != "short_math_001":
    raise SystemExit("latest distribution capture smoke row case mismatch")
  if smoke_row.get("generated_token_count") != 1:
    raise SystemExit("latest distribution capture smoke row token count mismatch")
  positions = smoke_row.get("distribution_positions", [])
  if not isinstance(positions, list) or len(positions) != 1:
    raise SystemExit("latest distribution capture smoke row position count mismatch")
  if not positions[0].get("top_logprobs"):
    raise SystemExit("latest distribution capture smoke row missing top-logprobs")
  if smoke_row.get("limitations", {}).get("smoke_only") is not True:
    raise SystemExit("latest distribution capture smoke row must be marked smoke-only")
  distribution_smoke_correctness = json.loads(
      distribution_smoke_correctness_path.read_text(encoding="utf-8")
  )
  if distribution_smoke_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest distribution capture smoke correctness failed")
  latest_short_router_capture = capture_plan.get("latest_short_router_distribution_capture", {})
  if (
      latest_short_router_capture.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-short-router.py"
  ):
    raise SystemExit("latest short/router distribution capture tool mismatch")
  if latest_short_router_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest short/router distribution capture checks must pass")
  if latest_short_router_capture.get("captured_case_count") != 6:
    raise SystemExit("latest short/router distribution capture case count mismatch")
  if latest_short_router_capture.get("selected_case_count") != 6:
    raise SystemExit("latest short/router distribution capture selected count mismatch")
  if latest_short_router_capture.get("total_distribution_positions") != 429:
    raise SystemExit("latest short/router distribution capture position count mismatch")
  if latest_short_router_capture.get("stopped_before_request_count") != 2:
    raise SystemExit("latest short/router distribution capture early-stop count mismatch")
  if latest_short_router_capture.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest short/router distribution capture must not claim full acceptance")
  if latest_short_router_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest short/router distribution capture must keep oracle gate open")
  short_router_path = latest_short_router_capture.get("path")
  if not isinstance(short_router_path, str) or not short_router_path:
    raise SystemExit("latest short/router distribution capture path missing")
  short_router_dir = ROOT / short_router_path
  short_router_capture_path = short_router_dir / "capture.json"
  short_router_correctness_path = short_router_dir / "correctness.json"
  short_router_jsonl_path = short_router_dir / "teacher-forced-distribution-short-router.jsonl"
  if (
      not short_router_capture_path.exists()
      or not short_router_correctness_path.exists()
      or not short_router_jsonl_path.exists()
  ):
    raise SystemExit("latest short/router distribution capture artifact missing")
  short_router_capture = json.loads(short_router_capture_path.read_text(encoding="utf-8"))
  if short_router_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest short/router distribution capture artifact checks failed")
  if short_router_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest short/router distribution capture artifact must keep gate open")
  if short_router_capture.get("total_distribution_positions") != 429:
    raise SystemExit("latest short/router distribution capture artifact position mismatch")
  if short_router_capture.get("stopped_before_request_count") != 2:
    raise SystemExit("latest short/router distribution capture artifact early-stop mismatch")
  short_router_rows = load_jsonl(short_router_jsonl_path)
  if len(short_router_rows) != 6:
    raise SystemExit("latest short/router distribution capture JSONL row count mismatch")
  expected_case_counts = {
      "short_math_001": 48,
      "short_factual_002": 11,
      "short_transform_003": 18,
      "router_math_reason_001": 96,
      "router_code_reason_002": 128,
      "router_instruction_003": 128,
  }
  for row in short_router_rows:
    case_id = row.get("case_id")
    if case_id not in expected_case_counts:
      raise SystemExit(f"unexpected short/router case: {case_id}")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: short/router row workstream mismatch")
    if row.get("capture_status") != "captured_short_router_subset":
      raise SystemExit(f"{case_id}: short/router row status mismatch")
    generated_count = row.get("generated_token_count")
    target_max = row.get("target_max_new_tokens")
    if generated_count != expected_case_counts[case_id]:
      raise SystemExit(f"{case_id}: short/router generated token count mismatch")
    if not isinstance(target_max, int) or not (0 < generated_count <= target_max):
      raise SystemExit(f"{case_id}: short/router generated count outside request")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != generated_count:
      raise SystemExit(f"{case_id}: short/router position count mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: short/router row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("short_router_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: short/router limitations mismatch")
  latest_materialized_distribution = capture_plan.get(
      "latest_materialized_distribution_capture", {}
  )
  if (
      latest_materialized_distribution.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution capture tool mismatch")
  if latest_materialized_distribution.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution capture checks must pass")
  expected_materialized_cases = [
      "sentinel_001k",
      "prefill_shape_001k",
  ]
  if latest_materialized_distribution.get("cases") != expected_materialized_cases:
    raise SystemExit("latest materialized distribution capture cases mismatch")
  if latest_materialized_distribution.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution capture case count mismatch")
  if latest_materialized_distribution.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution capture request count mismatch")
  if latest_materialized_distribution.get("total_distribution_positions") != 541:
    raise SystemExit("latest materialized distribution capture position count mismatch")
  if latest_materialized_distribution.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution capture early-stop count mismatch")
  if latest_materialized_distribution.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution capture must not claim full acceptance")
  if latest_materialized_distribution.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution capture must keep oracle gate open")
  materialized_distribution_path = latest_materialized_distribution.get("path")
  if not isinstance(materialized_distribution_path, str) or not materialized_distribution_path:
    raise SystemExit("latest materialized distribution capture path missing")
  materialized_distribution_dir = ROOT / materialized_distribution_path
  materialized_capture_path = materialized_distribution_dir / "capture.json"
  materialized_correctness_path = materialized_distribution_dir / "correctness.json"
  materialized_jsonl_path = (
      materialized_distribution_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_capture_path.exists()
      or not materialized_correctness_path.exists()
      or not materialized_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution capture artifact missing")
  materialized_capture = json.loads(materialized_capture_path.read_text(encoding="utf-8"))
  if materialized_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution capture artifact checks failed")
  if materialized_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution capture artifact must keep gate open")
  if materialized_capture.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution capture artifact case count mismatch")
  if materialized_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution capture artifact request mismatch")
  if materialized_capture.get("total_distribution_positions") != 541:
    raise SystemExit("latest materialized distribution capture artifact position mismatch")
  expected_materialized_counts = {
      "sentinel_001k": 512,
      "prefill_shape_001k": 29,
  }
  expected_materialized_early_stop = {
      "sentinel_001k": False,
      "prefill_shape_001k": True,
  }
  materialized_case_results = materialized_capture.get("case_results", [])
  if [case.get("case_id") for case in materialized_case_results] != expected_materialized_cases:
    raise SystemExit("latest materialized distribution capture artifact case order mismatch")
  for case in materialized_case_results:
    case_id = case.get("case_id")
    if case.get("expected_prompt_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution expected prompt count mismatch")
    if case.get("prompt_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution result must pass")
    if case.get("captured_positions") != expected_materialized_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution missing top-logprobs")
  materialized_rows = load_jsonl(materialized_jsonl_path)
  if len(materialized_rows) != 2:
    raise SystemExit("latest materialized distribution JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_rows] != expected_materialized_cases:
    raise SystemExit("latest materialized distribution JSONL case order mismatch")
  for row in materialized_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution row status mismatch")
    if row.get("prompt_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution limitations mismatch")
  materialized_correctness = json.loads(
      materialized_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution capture correctness failed")
  latest_materialized_extension = capture_plan.get(
      "latest_materialized_distribution_capture_extension", {}
  )
  if (
      latest_materialized_extension.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution extension tool mismatch")
  if latest_materialized_extension.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution extension checks must pass")
  expected_extension_cases = [
      "sentinel_002k",
      "prefill_shape_002k",
      "sentinel_004k",
      "prefill_shape_004k",
  ]
  if latest_materialized_extension.get("cases") != expected_extension_cases:
    raise SystemExit("latest materialized distribution extension cases mismatch")
  if latest_materialized_extension.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution extension case count mismatch")
  if latest_materialized_extension.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution extension request count mismatch")
  if latest_materialized_extension.get("total_distribution_positions") != 2048:
    raise SystemExit("latest materialized distribution extension position count mismatch")
  if latest_materialized_extension.get("stopped_before_request_count") != 0:
    raise SystemExit("latest materialized distribution extension early-stop count mismatch")
  if latest_materialized_extension.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution extension must not claim full acceptance")
  if latest_materialized_extension.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution extension must keep oracle gate open")
  materialized_extension_path = latest_materialized_extension.get("path")
  if not isinstance(materialized_extension_path, str) or not materialized_extension_path:
    raise SystemExit("latest materialized distribution extension path missing")
  materialized_extension_dir = ROOT / materialized_extension_path
  materialized_extension_capture_path = materialized_extension_dir / "capture.json"
  materialized_extension_correctness_path = materialized_extension_dir / "correctness.json"
  materialized_extension_jsonl_path = (
      materialized_extension_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_extension_capture_path.exists()
      or not materialized_extension_correctness_path.exists()
      or not materialized_extension_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution extension artifact missing")
  materialized_extension_capture = json.loads(
      materialized_extension_capture_path.read_text(encoding="utf-8")
  )
  if materialized_extension_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution extension artifact checks failed")
  if materialized_extension_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution extension artifact must keep gate open")
  if materialized_extension_capture.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution extension artifact case count mismatch")
  if materialized_extension_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution extension artifact request mismatch")
  if materialized_extension_capture.get("total_distribution_positions") != 2048:
    raise SystemExit("latest materialized distribution extension artifact position mismatch")
  expected_extension_prompt_counts = {
      "sentinel_002k": 2048,
      "prefill_shape_002k": 2048,
      "sentinel_004k": 4096,
      "prefill_shape_004k": 4096,
  }
  materialized_extension_case_results = materialized_extension_capture.get("case_results", [])
  if [case.get("case_id") for case in materialized_extension_case_results] != expected_extension_cases:
    raise SystemExit("latest materialized distribution extension artifact case order mismatch")
  for case in materialized_extension_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_extension_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution extension expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution extension observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution extension status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution extension result must pass")
    if case.get("captured_positions") != 512:
      raise SystemExit(f"{case_id}: materialized distribution extension position count mismatch")
    if case.get("stopped_before_request_limit") is not False:
      raise SystemExit(f"{case_id}: materialized distribution extension early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution extension missing top-logprobs")
  materialized_extension_rows = load_jsonl(materialized_extension_jsonl_path)
  if len(materialized_extension_rows) != 4:
    raise SystemExit("latest materialized distribution extension JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_extension_rows] != expected_extension_cases:
    raise SystemExit("latest materialized distribution extension JSONL case order mismatch")
  for row in materialized_extension_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution extension row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution extension capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution extension row status mismatch")
    if row.get("prompt_token_count") != expected_extension_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution extension row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution extension row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != 512:
      raise SystemExit(f"{case_id}: materialized distribution extension row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution extension row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution extension limitations mismatch")
  materialized_extension_correctness = json.loads(
      materialized_extension_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_extension_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution extension correctness failed")
  latest_materialized_extension_8k16k = capture_plan.get(
      "latest_materialized_distribution_capture_extension_8k16k", {}
  )
  if (
      latest_materialized_extension_8k16k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 8k/16k extension tool mismatch")
  if latest_materialized_extension_8k16k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 8k/16k extension checks must pass")
  expected_extension_8k16k_cases = [
      "sentinel_008k",
      "prefill_shape_008k",
      "sentinel_016k",
      "prefill_shape_016k",
  ]
  if latest_materialized_extension_8k16k.get("cases") != expected_extension_8k16k_cases:
    raise SystemExit("latest materialized distribution 8k/16k extension cases mismatch")
  if latest_materialized_extension_8k16k.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 8k/16k extension case count mismatch")
  if latest_materialized_extension_8k16k.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 8k/16k extension request count mismatch")
  if latest_materialized_extension_8k16k.get("total_distribution_positions") != 1956:
    raise SystemExit("latest materialized distribution 8k/16k extension position count mismatch")
  if latest_materialized_extension_8k16k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 8k/16k extension early-stop count mismatch")
  if latest_materialized_extension_8k16k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 8k/16k extension must not claim full acceptance")
  if latest_materialized_extension_8k16k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 8k/16k extension must keep oracle gate open")
  materialized_extension_8k16k_path = latest_materialized_extension_8k16k.get("path")
  if not isinstance(materialized_extension_8k16k_path, str) or not materialized_extension_8k16k_path:
    raise SystemExit("latest materialized distribution 8k/16k extension path missing")
  materialized_extension_8k16k_dir = ROOT / materialized_extension_8k16k_path
  materialized_extension_8k16k_capture_path = materialized_extension_8k16k_dir / "capture.json"
  materialized_extension_8k16k_correctness_path = materialized_extension_8k16k_dir / "correctness.json"
  materialized_extension_8k16k_jsonl_path = (
      materialized_extension_8k16k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_extension_8k16k_capture_path.exists()
      or not materialized_extension_8k16k_correctness_path.exists()
      or not materialized_extension_8k16k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 8k/16k extension artifact missing")
  materialized_extension_8k16k_capture = json.loads(
      materialized_extension_8k16k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_extension_8k16k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact checks failed")
  if materialized_extension_8k16k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact must keep gate open")
  if materialized_extension_8k16k_capture.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact case count mismatch")
  if materialized_extension_8k16k_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact request mismatch")
  if materialized_extension_8k16k_capture.get("total_distribution_positions") != 1956:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact position mismatch")
  expected_extension_8k16k_prompt_counts = {
      "sentinel_008k": 8192,
      "prefill_shape_008k": 8192,
      "sentinel_016k": 16384,
      "prefill_shape_016k": 16384,
  }
  expected_extension_8k16k_position_counts = {
      "sentinel_008k": 512,
      "prefill_shape_008k": 420,
      "sentinel_016k": 512,
      "prefill_shape_016k": 512,
  }
  expected_extension_8k16k_early_stop = {
      "sentinel_008k": False,
      "prefill_shape_008k": True,
      "sentinel_016k": False,
      "prefill_shape_016k": False,
  }
  materialized_extension_8k16k_case_results = materialized_extension_8k16k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_extension_8k16k_case_results] != expected_extension_8k16k_cases:
    raise SystemExit("latest materialized distribution 8k/16k extension artifact case order mismatch")
  for case in materialized_extension_8k16k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_extension_8k16k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension result must pass")
    if case.get("captured_positions") != expected_extension_8k16k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_extension_8k16k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension missing top-logprobs")
  materialized_extension_8k16k_rows = load_jsonl(materialized_extension_8k16k_jsonl_path)
  if len(materialized_extension_8k16k_rows) != 4:
    raise SystemExit("latest materialized distribution 8k/16k extension JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_extension_8k16k_rows] != expected_extension_8k16k_cases:
    raise SystemExit("latest materialized distribution 8k/16k extension JSONL case order mismatch")
  for row in materialized_extension_8k16k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row status mismatch")
    if row.get("prompt_token_count") != expected_extension_8k16k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_extension_8k16k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution 8k/16k extension limitations mismatch")
  materialized_extension_8k16k_correctness = json.loads(
      materialized_extension_8k16k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_extension_8k16k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 8k/16k extension correctness failed")
  latest_materialized_extension_32k64k = capture_plan.get(
      "latest_materialized_distribution_capture_extension_32k64k", {}
  )
  if (
      latest_materialized_extension_32k64k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 32k/64k extension tool mismatch")
  if latest_materialized_extension_32k64k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 32k/64k extension checks must pass")
  expected_extension_32k64k_cases = [
      "sentinel_032k",
      "prefill_shape_032k",
      "sentinel_064k",
      "prefill_shape_064k",
  ]
  if latest_materialized_extension_32k64k.get("cases") != expected_extension_32k64k_cases:
    raise SystemExit("latest materialized distribution 32k/64k extension cases mismatch")
  if latest_materialized_extension_32k64k.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 32k/64k extension case count mismatch")
  if latest_materialized_extension_32k64k.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 32k/64k extension request count mismatch")
  if latest_materialized_extension_32k64k.get("total_distribution_positions") != 1551:
    raise SystemExit("latest materialized distribution 32k/64k extension position count mismatch")
  if latest_materialized_extension_32k64k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 32k/64k extension early-stop count mismatch")
  if latest_materialized_extension_32k64k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 32k/64k extension must not claim full acceptance")
  if latest_materialized_extension_32k64k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 32k/64k extension must keep oracle gate open")
  materialized_extension_32k64k_path = latest_materialized_extension_32k64k.get("path")
  if not isinstance(materialized_extension_32k64k_path, str) or not materialized_extension_32k64k_path:
    raise SystemExit("latest materialized distribution 32k/64k extension path missing")
  materialized_extension_32k64k_dir = ROOT / materialized_extension_32k64k_path
  materialized_extension_32k64k_capture_path = materialized_extension_32k64k_dir / "capture.json"
  materialized_extension_32k64k_correctness_path = materialized_extension_32k64k_dir / "correctness.json"
  materialized_extension_32k64k_jsonl_path = (
      materialized_extension_32k64k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_extension_32k64k_capture_path.exists()
      or not materialized_extension_32k64k_correctness_path.exists()
      or not materialized_extension_32k64k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 32k/64k extension artifact missing")
  materialized_extension_32k64k_capture = json.loads(
      materialized_extension_32k64k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_extension_32k64k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact checks failed")
  if materialized_extension_32k64k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact must keep gate open")
  if materialized_extension_32k64k_capture.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact case count mismatch")
  if materialized_extension_32k64k_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact request mismatch")
  if materialized_extension_32k64k_capture.get("total_distribution_positions") != 1551:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact position mismatch")
  expected_extension_32k64k_prompt_counts = {
      "sentinel_032k": 32768,
      "prefill_shape_032k": 32768,
      "sentinel_064k": 65536,
      "prefill_shape_064k": 65536,
  }
  expected_extension_32k64k_position_counts = {
      "sentinel_032k": 512,
      "prefill_shape_032k": 512,
      "sentinel_064k": 15,
      "prefill_shape_064k": 512,
  }
  expected_extension_32k64k_early_stop = {
      "sentinel_032k": False,
      "prefill_shape_032k": False,
      "sentinel_064k": True,
      "prefill_shape_064k": False,
  }
  materialized_extension_32k64k_case_results = materialized_extension_32k64k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_extension_32k64k_case_results] != expected_extension_32k64k_cases:
    raise SystemExit("latest materialized distribution 32k/64k extension artifact case order mismatch")
  for case in materialized_extension_32k64k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_extension_32k64k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension result must pass")
    if case.get("captured_positions") != expected_extension_32k64k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_extension_32k64k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension missing top-logprobs")
  materialized_extension_32k64k_rows = load_jsonl(materialized_extension_32k64k_jsonl_path)
  if len(materialized_extension_32k64k_rows) != 4:
    raise SystemExit("latest materialized distribution 32k/64k extension JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_extension_32k64k_rows] != expected_extension_32k64k_cases:
    raise SystemExit("latest materialized distribution 32k/64k extension JSONL case order mismatch")
  for row in materialized_extension_32k64k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row status mismatch")
    if row.get("prompt_token_count") != expected_extension_32k64k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_extension_32k64k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution 32k/64k extension limitations mismatch")
  materialized_extension_32k64k_correctness = json.loads(
      materialized_extension_32k64k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_extension_32k64k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 32k/64k extension correctness failed")
  latest_materialized_extension_100k = capture_plan.get(
      "latest_materialized_distribution_capture_extension_100k", {}
  )
  if (
      latest_materialized_extension_100k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 100k extension tool mismatch")
  if latest_materialized_extension_100k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 100k extension checks must pass")
  expected_extension_100k_cases = [
      "sentinel_100k",
      "prefill_shape_100k",
  ]
  if latest_materialized_extension_100k.get("cases") != expected_extension_100k_cases:
    raise SystemExit("latest materialized distribution 100k extension cases mismatch")
  if latest_materialized_extension_100k.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 100k extension case count mismatch")
  if latest_materialized_extension_100k.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 100k extension request count mismatch")
  if latest_materialized_extension_100k.get("total_distribution_positions") != 529:
    raise SystemExit("latest materialized distribution 100k extension position count mismatch")
  if latest_materialized_extension_100k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 100k extension early-stop count mismatch")
  if latest_materialized_extension_100k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 100k extension must not claim full acceptance")
  if latest_materialized_extension_100k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 100k extension must keep oracle gate open")
  materialized_extension_100k_path = latest_materialized_extension_100k.get("path")
  if not isinstance(materialized_extension_100k_path, str) or not materialized_extension_100k_path:
    raise SystemExit("latest materialized distribution 100k extension path missing")
  materialized_extension_100k_dir = ROOT / materialized_extension_100k_path
  materialized_extension_100k_capture_path = materialized_extension_100k_dir / "capture.json"
  materialized_extension_100k_correctness_path = materialized_extension_100k_dir / "correctness.json"
  materialized_extension_100k_jsonl_path = (
      materialized_extension_100k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_extension_100k_capture_path.exists()
      or not materialized_extension_100k_correctness_path.exists()
      or not materialized_extension_100k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 100k extension artifact missing")
  materialized_extension_100k_capture = json.loads(
      materialized_extension_100k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_extension_100k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 100k extension artifact checks failed")
  if materialized_extension_100k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 100k extension artifact must keep gate open")
  if materialized_extension_100k_capture.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 100k extension artifact case count mismatch")
  if materialized_extension_100k_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 100k extension artifact request mismatch")
  if materialized_extension_100k_capture.get("total_distribution_positions") != 529:
    raise SystemExit("latest materialized distribution 100k extension artifact position mismatch")
  expected_extension_100k_prompt_counts = {
      "sentinel_100k": 102400,
      "prefill_shape_100k": 102400,
  }
  expected_extension_100k_position_counts = {
      "sentinel_100k": 17,
      "prefill_shape_100k": 512,
  }
  expected_extension_100k_early_stop = {
      "sentinel_100k": True,
      "prefill_shape_100k": False,
  }
  materialized_extension_100k_case_results = materialized_extension_100k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_extension_100k_case_results] != expected_extension_100k_cases:
    raise SystemExit("latest materialized distribution 100k extension artifact case order mismatch")
  for case in materialized_extension_100k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_extension_100k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension result must pass")
    if case.get("captured_positions") != expected_extension_100k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_extension_100k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension missing top-logprobs")
  materialized_extension_100k_rows = load_jsonl(materialized_extension_100k_jsonl_path)
  if len(materialized_extension_100k_rows) != 2:
    raise SystemExit("latest materialized distribution 100k extension JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_extension_100k_rows] != expected_extension_100k_cases:
    raise SystemExit("latest materialized distribution 100k extension JSONL case order mismatch")
  for row in materialized_extension_100k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 100k extension capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row status mismatch")
    if row.get("prompt_token_count") != expected_extension_100k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_extension_100k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 100k extension row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution 100k extension limitations mismatch")
  materialized_extension_100k_correctness = json.loads(
      materialized_extension_100k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_extension_100k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 100k extension correctness failed")
  latest_materialized_extension_128k = capture_plan.get(
      "latest_materialized_distribution_capture_extension_128k", {}
  )
  if (
      latest_materialized_extension_128k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 128k extension tool mismatch")
  if latest_materialized_extension_128k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 128k extension checks must pass")
  expected_extension_128k_cases = [
      "sentinel_128k",
      "prefill_shape_128k",
  ]
  if latest_materialized_extension_128k.get("cases") != expected_extension_128k_cases:
    raise SystemExit("latest materialized distribution 128k extension cases mismatch")
  if latest_materialized_extension_128k.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 128k extension case count mismatch")
  if latest_materialized_extension_128k.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 128k extension request count mismatch")
  if latest_materialized_extension_128k.get("total_distribution_positions") != 529:
    raise SystemExit("latest materialized distribution 128k extension position count mismatch")
  if latest_materialized_extension_128k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 128k extension early-stop count mismatch")
  if latest_materialized_extension_128k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 128k extension must not claim full acceptance")
  if latest_materialized_extension_128k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 128k extension must keep oracle gate open")
  materialized_extension_128k_path = latest_materialized_extension_128k.get("path")
  if not isinstance(materialized_extension_128k_path, str) or not materialized_extension_128k_path:
    raise SystemExit("latest materialized distribution 128k extension path missing")
  materialized_extension_128k_dir = ROOT / materialized_extension_128k_path
  materialized_extension_128k_capture_path = materialized_extension_128k_dir / "capture.json"
  materialized_extension_128k_correctness_path = materialized_extension_128k_dir / "correctness.json"
  materialized_extension_128k_jsonl_path = (
      materialized_extension_128k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_extension_128k_capture_path.exists()
      or not materialized_extension_128k_correctness_path.exists()
      or not materialized_extension_128k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 128k extension artifact missing")
  materialized_extension_128k_capture = json.loads(
      materialized_extension_128k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_extension_128k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 128k extension artifact checks failed")
  if materialized_extension_128k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 128k extension artifact must keep gate open")
  if materialized_extension_128k_capture.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 128k extension artifact case count mismatch")
  if materialized_extension_128k_capture.get("requested_output_token_count") != 512:
    raise SystemExit("latest materialized distribution 128k extension artifact request mismatch")
  if materialized_extension_128k_capture.get("total_distribution_positions") != 529:
    raise SystemExit("latest materialized distribution 128k extension artifact position mismatch")
  expected_extension_128k_prompt_counts = {
      "sentinel_128k": 131072,
      "prefill_shape_128k": 131072,
  }
  expected_extension_128k_position_counts = {
      "sentinel_128k": 17,
      "prefill_shape_128k": 512,
  }
  expected_extension_128k_early_stop = {
      "sentinel_128k": True,
      "prefill_shape_128k": False,
  }
  materialized_extension_128k_case_results = materialized_extension_128k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_extension_128k_case_results] != expected_extension_128k_cases:
    raise SystemExit("latest materialized distribution 128k extension artifact case order mismatch")
  for case in materialized_extension_128k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_extension_128k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension result must pass")
    if case.get("captured_positions") != expected_extension_128k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_extension_128k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension missing top-logprobs")
  materialized_extension_128k_rows = load_jsonl(materialized_extension_128k_jsonl_path)
  if len(materialized_extension_128k_rows) != 2:
    raise SystemExit("latest materialized distribution 128k extension JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_extension_128k_rows] != expected_extension_128k_cases:
    raise SystemExit("latest materialized distribution 128k extension JSONL case order mismatch")
  for row in materialized_extension_128k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 128k extension capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row status mismatch")
    if row.get("prompt_token_count") != expected_extension_128k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row prompt count mismatch")
    if row.get("requested_output_token_count") != 512:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row request mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_extension_128k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 128k extension row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: materialized distribution 128k extension limitations mismatch")
  materialized_extension_128k_correctness = json.loads(
      materialized_extension_128k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_extension_128k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 128k extension correctness failed")
  materialized_coverage = capture_plan.get("materialized_distribution_capture_coverage", {})
  expected_coverage_cases = [
      "sentinel_001k",
      "prefill_shape_001k",
      "sentinel_002k",
      "prefill_shape_002k",
      "sentinel_004k",
      "prefill_shape_004k",
      "sentinel_008k",
      "prefill_shape_008k",
      "sentinel_016k",
      "prefill_shape_016k",
      "sentinel_032k",
      "prefill_shape_032k",
      "sentinel_064k",
      "prefill_shape_064k",
      "sentinel_100k",
      "prefill_shape_100k",
      "sentinel_128k",
      "prefill_shape_128k",
  ]
  if materialized_coverage.get("artifact_count") != 6:
    raise SystemExit("materialized distribution coverage artifact count mismatch")
  if materialized_coverage.get("paths") != [
      materialized_distribution_path,
      materialized_extension_path,
      materialized_extension_8k16k_path,
      materialized_extension_32k64k_path,
      materialized_extension_100k_path,
      materialized_extension_128k_path,
  ]:
    raise SystemExit("materialized distribution coverage paths mismatch")
  if materialized_coverage.get("captured_case_count") != 18:
    raise SystemExit("materialized distribution coverage case count mismatch")
  if materialized_coverage.get("cases") != expected_coverage_cases:
    raise SystemExit("materialized distribution coverage cases mismatch")
  if materialized_coverage.get("requested_output_token_count") != 512:
    raise SystemExit("materialized distribution coverage request mismatch")
  if materialized_coverage.get("total_distribution_positions") != 7154:
    raise SystemExit("materialized distribution coverage position mismatch")
  if materialized_coverage.get("stopped_before_request_count") != 5:
    raise SystemExit("materialized distribution coverage early-stop mismatch")
  if materialized_coverage.get("full_acceptance_bundle") is not False:
    raise SystemExit("materialized distribution coverage must not claim full acceptance")
  if materialized_coverage.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("materialized distribution coverage must keep oracle gate open")
  latest_materialized_1024 = capture_plan.get(
      "latest_materialized_distribution_capture_1024_1k4k", {}
  )
  if (
      latest_materialized_1024.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 1024 1k/4k tool mismatch")
  if latest_materialized_1024.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 1k/4k checks must pass")
  expected_materialized_1024_cases = [
      "sentinel_001k",
      "prefill_shape_001k",
      "sentinel_002k",
      "prefill_shape_002k",
      "sentinel_004k",
      "prefill_shape_004k",
  ]
  if latest_materialized_1024.get("cases") != expected_materialized_1024_cases:
    raise SystemExit("latest materialized distribution 1024 1k/4k cases mismatch")
  if latest_materialized_1024.get("captured_case_count") != 6:
    raise SystemExit("latest materialized distribution 1024 1k/4k case count mismatch")
  if latest_materialized_1024.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 1k/4k request count mismatch")
  if latest_materialized_1024.get("total_distribution_positions") != 4262:
    raise SystemExit("latest materialized distribution 1024 1k/4k position count mismatch")
  if latest_materialized_1024.get("stopped_before_request_count") != 4:
    raise SystemExit("latest materialized distribution 1024 1k/4k early-stop count mismatch")
  if latest_materialized_1024.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 1024 1k/4k must not claim full acceptance")
  if latest_materialized_1024.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 1k/4k must keep oracle gate open")
  materialized_1024_path = latest_materialized_1024.get("path")
  if not isinstance(materialized_1024_path, str) or not materialized_1024_path:
    raise SystemExit("latest materialized distribution 1024 1k/4k path missing")
  materialized_1024_dir = ROOT / materialized_1024_path
  materialized_1024_capture_path = materialized_1024_dir / "capture.json"
  materialized_1024_correctness_path = materialized_1024_dir / "correctness.json"
  materialized_1024_jsonl_path = (
      materialized_1024_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_1024_capture_path.exists()
      or not materialized_1024_correctness_path.exists()
      or not materialized_1024_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact missing")
  materialized_1024_capture = json.loads(
      materialized_1024_capture_path.read_text(encoding="utf-8")
  )
  if materialized_1024_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact checks failed")
  if materialized_1024_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact must keep gate open")
  if materialized_1024_capture.get("captured_case_count") != 6:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact case count mismatch")
  if materialized_1024_capture.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact request mismatch")
  if materialized_1024_capture.get("total_distribution_positions") != 4262:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact position mismatch")
  expected_materialized_1024_prompt_counts = {
      "sentinel_001k": 1024,
      "prefill_shape_001k": 1024,
      "sentinel_002k": 2048,
      "prefill_shape_002k": 2048,
      "sentinel_004k": 4096,
      "prefill_shape_004k": 4096,
  }
  expected_materialized_1024_position_counts = {
      "sentinel_001k": 665,
      "prefill_shape_001k": 29,
      "sentinel_002k": 873,
      "prefill_shape_002k": 1024,
      "sentinel_004k": 647,
      "prefill_shape_004k": 1024,
  }
  expected_materialized_1024_early_stop = {
      "sentinel_001k": True,
      "prefill_shape_001k": True,
      "sentinel_002k": True,
      "prefill_shape_002k": False,
      "sentinel_004k": True,
      "prefill_shape_004k": False,
  }
  materialized_1024_case_results = materialized_1024_capture.get("case_results", [])
  if [case.get("case_id") for case in materialized_1024_case_results] != expected_materialized_1024_cases:
    raise SystemExit("latest materialized distribution 1024 1k/4k artifact case order mismatch")
  for case in materialized_1024_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_materialized_1024_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k result must pass")
    if case.get("captured_positions") != expected_materialized_1024_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_1024_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k missing top-logprobs")
  materialized_1024_command_results = materialized_1024_capture.get("command_results", [])
  if [case.get("case_id") for case in materialized_1024_command_results] != expected_materialized_1024_cases:
    raise SystemExit("latest materialized distribution 1024 1k/4k command order mismatch")
  if any(case.get("timed_out") is not False for case in materialized_1024_command_results):
    raise SystemExit("latest materialized distribution 1024 1k/4k command timed out")
  materialized_1024_rows = load_jsonl(materialized_1024_jsonl_path)
  if len(materialized_1024_rows) != 6:
    raise SystemExit("latest materialized distribution 1024 1k/4k JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_1024_rows] != expected_materialized_1024_cases:
    raise SystemExit("latest materialized distribution 1024 1k/4k JSONL case order mismatch")
  for row in materialized_1024_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row status mismatch")
    if row.get("prompt_token_count") != expected_materialized_1024_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row prompt count mismatch")
    if row.get("requested_output_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row request mismatch")
    if row.get("generated_token_count") != expected_materialized_1024_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row generated count mismatch")
    if row.get("stopped_before_request_limit") is not expected_materialized_1024_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row early-stop mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_1024_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
        or limitations.get("full_acceptance_context_ladder") is not False
    ):
      raise SystemExit(f"{case_id}: materialized distribution 1024 1k/4k limitations mismatch")
  materialized_1024_correctness = json.loads(
      materialized_1024_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_1024_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 1k/4k correctness failed")
  model_reference_artifacts = parsed[
      "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  ].get("oracle_bundle", {}).get("reference_artifacts", {})
  model_materialized_1024 = model_reference_artifacts.get(
      "latest_r0_materialized_distribution_capture_1024_1k4k", {}
  )
  if model_materialized_1024 != latest_materialized_1024:
    raise SystemExit("model/oracle materialized distribution 1024 1k/4k registration mismatch")
  latest_materialized_1024_8k16k = capture_plan.get(
      "latest_materialized_distribution_capture_1024_8k16k", {}
  )
  if (
      latest_materialized_1024_8k16k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 1024 8k/16k tool mismatch")
  if latest_materialized_1024_8k16k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 8k/16k checks must pass")
  expected_materialized_1024_8k16k_cases = [
      "sentinel_008k",
      "prefill_shape_008k",
      "sentinel_016k",
      "prefill_shape_016k",
  ]
  if latest_materialized_1024_8k16k.get("cases") != expected_materialized_1024_8k16k_cases:
    raise SystemExit("latest materialized distribution 1024 8k/16k cases mismatch")
  if latest_materialized_1024_8k16k.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 1024 8k/16k case count mismatch")
  if latest_materialized_1024_8k16k.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 8k/16k request count mismatch")
  if latest_materialized_1024_8k16k.get("total_distribution_positions") != 3000:
    raise SystemExit("latest materialized distribution 1024 8k/16k position count mismatch")
  if latest_materialized_1024_8k16k.get("stopped_before_request_count") != 3:
    raise SystemExit("latest materialized distribution 1024 8k/16k early-stop count mismatch")
  if latest_materialized_1024_8k16k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 1024 8k/16k must not claim full acceptance")
  if latest_materialized_1024_8k16k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 8k/16k must keep oracle gate open")
  materialized_1024_8k16k_path = latest_materialized_1024_8k16k.get("path")
  if not isinstance(materialized_1024_8k16k_path, str) or not materialized_1024_8k16k_path:
    raise SystemExit("latest materialized distribution 1024 8k/16k path missing")
  materialized_1024_8k16k_dir = ROOT / materialized_1024_8k16k_path
  materialized_1024_8k16k_capture_path = materialized_1024_8k16k_dir / "capture.json"
  materialized_1024_8k16k_correctness_path = materialized_1024_8k16k_dir / "correctness.json"
  materialized_1024_8k16k_jsonl_path = (
      materialized_1024_8k16k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_1024_8k16k_capture_path.exists()
      or not materialized_1024_8k16k_correctness_path.exists()
      or not materialized_1024_8k16k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact missing")
  materialized_1024_8k16k_capture = json.loads(
      materialized_1024_8k16k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_1024_8k16k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact checks failed")
  if materialized_1024_8k16k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact must keep gate open")
  if materialized_1024_8k16k_capture.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact case count mismatch")
  if materialized_1024_8k16k_capture.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact request mismatch")
  if materialized_1024_8k16k_capture.get("total_distribution_positions") != 3000:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact position mismatch")
  expected_materialized_1024_8k16k_prompt_counts = {
      "sentinel_008k": 8192,
      "prefill_shape_008k": 8192,
      "sentinel_016k": 16384,
      "prefill_shape_016k": 16384,
  }
  expected_materialized_1024_8k16k_position_counts = {
      "sentinel_008k": 718,
      "prefill_shape_008k": 420,
      "sentinel_016k": 838,
      "prefill_shape_016k": 1024,
  }
  expected_materialized_1024_8k16k_early_stop = {
      "sentinel_008k": True,
      "prefill_shape_008k": True,
      "sentinel_016k": True,
      "prefill_shape_016k": False,
  }
  materialized_1024_8k16k_case_results = materialized_1024_8k16k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_1024_8k16k_case_results] != expected_materialized_1024_8k16k_cases:
    raise SystemExit("latest materialized distribution 1024 8k/16k artifact case order mismatch")
  for case in materialized_1024_8k16k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_materialized_1024_8k16k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k result must pass")
    if case.get("captured_positions") != expected_materialized_1024_8k16k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_1024_8k16k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k missing top-logprobs")
  materialized_1024_8k16k_command_results = materialized_1024_8k16k_capture.get(
      "command_results", []
  )
  if [case.get("case_id") for case in materialized_1024_8k16k_command_results] != expected_materialized_1024_8k16k_cases:
    raise SystemExit("latest materialized distribution 1024 8k/16k command order mismatch")
  if any(case.get("timed_out") is not False for case in materialized_1024_8k16k_command_results):
    raise SystemExit("latest materialized distribution 1024 8k/16k command timed out")
  materialized_1024_8k16k_rows = load_jsonl(materialized_1024_8k16k_jsonl_path)
  if len(materialized_1024_8k16k_rows) != 4:
    raise SystemExit("latest materialized distribution 1024 8k/16k JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_1024_8k16k_rows] != expected_materialized_1024_8k16k_cases:
    raise SystemExit("latest materialized distribution 1024 8k/16k JSONL case order mismatch")
  for row in materialized_1024_8k16k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row status mismatch")
    if row.get("prompt_token_count") != expected_materialized_1024_8k16k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row prompt count mismatch")
    if row.get("requested_output_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row request mismatch")
    if row.get("generated_token_count") != expected_materialized_1024_8k16k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row generated count mismatch")
    if row.get("stopped_before_request_limit") is not expected_materialized_1024_8k16k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row early-stop mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_1024_8k16k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
        or limitations.get("full_acceptance_context_ladder") is not False
    ):
      raise SystemExit(f"{case_id}: materialized distribution 1024 8k/16k limitations mismatch")
  materialized_1024_8k16k_correctness = json.loads(
      materialized_1024_8k16k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_1024_8k16k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 8k/16k correctness failed")
  model_materialized_1024_8k16k = model_reference_artifacts.get(
      "latest_r0_materialized_distribution_capture_1024_8k16k", {}
  )
  if model_materialized_1024_8k16k != latest_materialized_1024_8k16k:
    raise SystemExit("model/oracle materialized distribution 1024 8k/16k registration mismatch")
  latest_materialized_1024_32k64k = capture_plan.get(
      "latest_materialized_distribution_capture_1024_32k64k", {}
  )
  if (
      latest_materialized_1024_32k64k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 1024 32k/64k tool mismatch")
  if latest_materialized_1024_32k64k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 32k/64k checks must pass")
  expected_materialized_1024_32k64k_cases = [
      "sentinel_032k",
      "prefill_shape_032k",
      "sentinel_064k",
      "prefill_shape_064k",
  ]
  if latest_materialized_1024_32k64k.get("cases") != expected_materialized_1024_32k64k_cases:
    raise SystemExit("latest materialized distribution 1024 32k/64k cases mismatch")
  if latest_materialized_1024_32k64k.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 1024 32k/64k case count mismatch")
  if latest_materialized_1024_32k64k.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 32k/64k request count mismatch")
  if latest_materialized_1024_32k64k.get("total_distribution_positions") != 2971:
    raise SystemExit("latest materialized distribution 1024 32k/64k position count mismatch")
  if latest_materialized_1024_32k64k.get("stopped_before_request_count") != 2:
    raise SystemExit("latest materialized distribution 1024 32k/64k early-stop count mismatch")
  if latest_materialized_1024_32k64k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 1024 32k/64k must not claim full acceptance")
  if latest_materialized_1024_32k64k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 32k/64k must keep oracle gate open")
  materialized_1024_32k64k_path = latest_materialized_1024_32k64k.get("path")
  if not isinstance(materialized_1024_32k64k_path, str) or not materialized_1024_32k64k_path:
    raise SystemExit("latest materialized distribution 1024 32k/64k path missing")
  materialized_1024_32k64k_dir = ROOT / materialized_1024_32k64k_path
  materialized_1024_32k64k_capture_path = materialized_1024_32k64k_dir / "capture.json"
  materialized_1024_32k64k_correctness_path = materialized_1024_32k64k_dir / "correctness.json"
  materialized_1024_32k64k_jsonl_path = (
      materialized_1024_32k64k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_1024_32k64k_capture_path.exists()
      or not materialized_1024_32k64k_correctness_path.exists()
      or not materialized_1024_32k64k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact missing")
  materialized_1024_32k64k_capture = json.loads(
      materialized_1024_32k64k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_1024_32k64k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact checks failed")
  if materialized_1024_32k64k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact must keep gate open")
  if materialized_1024_32k64k_capture.get("captured_case_count") != 4:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact case count mismatch")
  if materialized_1024_32k64k_capture.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact request mismatch")
  if materialized_1024_32k64k_capture.get("total_distribution_positions") != 2971:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact position mismatch")
  expected_materialized_1024_32k64k_prompt_counts = {
      "sentinel_032k": 32768,
      "prefill_shape_032k": 32768,
      "sentinel_064k": 65536,
      "prefill_shape_064k": 65536,
  }
  expected_materialized_1024_32k64k_position_counts = {
      "sentinel_032k": 908,
      "prefill_shape_032k": 1024,
      "sentinel_064k": 15,
      "prefill_shape_064k": 1024,
  }
  expected_materialized_1024_32k64k_early_stop = {
      "sentinel_032k": True,
      "prefill_shape_032k": False,
      "sentinel_064k": True,
      "prefill_shape_064k": False,
  }
  materialized_1024_32k64k_case_results = materialized_1024_32k64k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_1024_32k64k_case_results] != expected_materialized_1024_32k64k_cases:
    raise SystemExit("latest materialized distribution 1024 32k/64k artifact case order mismatch")
  for case in materialized_1024_32k64k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_materialized_1024_32k64k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k result must pass")
    if case.get("captured_positions") != expected_materialized_1024_32k64k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_1024_32k64k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k missing top-logprobs")
  materialized_1024_32k64k_command_results = materialized_1024_32k64k_capture.get(
      "command_results", []
  )
  if [case.get("case_id") for case in materialized_1024_32k64k_command_results] != expected_materialized_1024_32k64k_cases:
    raise SystemExit("latest materialized distribution 1024 32k/64k command order mismatch")
  if any(case.get("timed_out") is not False for case in materialized_1024_32k64k_command_results):
    raise SystemExit("latest materialized distribution 1024 32k/64k command timed out")
  materialized_1024_32k64k_rows = load_jsonl(materialized_1024_32k64k_jsonl_path)
  if len(materialized_1024_32k64k_rows) != 4:
    raise SystemExit("latest materialized distribution 1024 32k/64k JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_1024_32k64k_rows] != expected_materialized_1024_32k64k_cases:
    raise SystemExit("latest materialized distribution 1024 32k/64k JSONL case order mismatch")
  for row in materialized_1024_32k64k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row status mismatch")
    if row.get("prompt_token_count") != expected_materialized_1024_32k64k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row prompt count mismatch")
    if row.get("requested_output_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row request mismatch")
    if row.get("generated_token_count") != expected_materialized_1024_32k64k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row generated count mismatch")
    if row.get("stopped_before_request_limit") is not expected_materialized_1024_32k64k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row early-stop mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_1024_32k64k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
        or limitations.get("full_acceptance_context_ladder") is not False
    ):
      raise SystemExit(f"{case_id}: materialized distribution 1024 32k/64k limitations mismatch")
  materialized_1024_32k64k_correctness = json.loads(
      materialized_1024_32k64k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_1024_32k64k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 32k/64k correctness failed")
  model_materialized_1024_32k64k = model_reference_artifacts.get(
      "latest_r0_materialized_distribution_capture_1024_32k64k", {}
  )
  if model_materialized_1024_32k64k != latest_materialized_1024_32k64k:
    raise SystemExit("model/oracle materialized distribution 1024 32k/64k registration mismatch")
  latest_materialized_1024_100k = capture_plan.get(
      "latest_materialized_distribution_capture_1024_100k", {}
  )
  if (
      latest_materialized_1024_100k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 1024 100k tool mismatch")
  if latest_materialized_1024_100k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 100k checks must pass")
  expected_materialized_1024_100k_cases = [
      "sentinel_100k",
      "prefill_shape_100k",
  ]
  if latest_materialized_1024_100k.get("cases") != expected_materialized_1024_100k_cases:
    raise SystemExit("latest materialized distribution 1024 100k cases mismatch")
  if latest_materialized_1024_100k.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 1024 100k case count mismatch")
  if latest_materialized_1024_100k.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 100k request count mismatch")
  if latest_materialized_1024_100k.get("total_distribution_positions") != 1041:
    raise SystemExit("latest materialized distribution 1024 100k position count mismatch")
  if latest_materialized_1024_100k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 1024 100k early-stop count mismatch")
  if latest_materialized_1024_100k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 1024 100k must not claim full acceptance")
  if latest_materialized_1024_100k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 100k must keep oracle gate open")
  materialized_1024_100k_path = latest_materialized_1024_100k.get("path")
  if not isinstance(materialized_1024_100k_path, str) or not materialized_1024_100k_path:
    raise SystemExit("latest materialized distribution 1024 100k path missing")
  materialized_1024_100k_dir = ROOT / materialized_1024_100k_path
  materialized_1024_100k_capture_path = materialized_1024_100k_dir / "capture.json"
  materialized_1024_100k_correctness_path = materialized_1024_100k_dir / "correctness.json"
  materialized_1024_100k_jsonl_path = (
      materialized_1024_100k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_1024_100k_capture_path.exists()
      or not materialized_1024_100k_correctness_path.exists()
      or not materialized_1024_100k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 1024 100k artifact missing")
  materialized_1024_100k_capture = json.loads(
      materialized_1024_100k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_1024_100k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 100k artifact checks failed")
  if materialized_1024_100k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 100k artifact must keep gate open")
  if materialized_1024_100k_capture.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 1024 100k artifact case count mismatch")
  if materialized_1024_100k_capture.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 100k artifact request mismatch")
  if materialized_1024_100k_capture.get("total_distribution_positions") != 1041:
    raise SystemExit("latest materialized distribution 1024 100k artifact position mismatch")
  expected_materialized_1024_100k_prompt_counts = {
      "sentinel_100k": 102400,
      "prefill_shape_100k": 102400,
  }
  expected_materialized_1024_100k_position_counts = {
      "sentinel_100k": 17,
      "prefill_shape_100k": 1024,
  }
  expected_materialized_1024_100k_early_stop = {
      "sentinel_100k": True,
      "prefill_shape_100k": False,
  }
  materialized_1024_100k_case_results = materialized_1024_100k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_1024_100k_case_results] != expected_materialized_1024_100k_cases:
    raise SystemExit("latest materialized distribution 1024 100k artifact case order mismatch")
  for case in materialized_1024_100k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_materialized_1024_100k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k result must pass")
    if case.get("captured_positions") != expected_materialized_1024_100k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_1024_100k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k missing top-logprobs")
  materialized_1024_100k_command_results = materialized_1024_100k_capture.get(
      "command_results", []
  )
  if [case.get("case_id") for case in materialized_1024_100k_command_results] != expected_materialized_1024_100k_cases:
    raise SystemExit("latest materialized distribution 1024 100k command order mismatch")
  if any(case.get("timed_out") is not False for case in materialized_1024_100k_command_results):
    raise SystemExit("latest materialized distribution 1024 100k command timed out")
  materialized_1024_100k_rows = load_jsonl(materialized_1024_100k_jsonl_path)
  if len(materialized_1024_100k_rows) != 2:
    raise SystemExit("latest materialized distribution 1024 100k JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_1024_100k_rows] != expected_materialized_1024_100k_cases:
    raise SystemExit("latest materialized distribution 1024 100k JSONL case order mismatch")
  for row in materialized_1024_100k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row status mismatch")
    if row.get("prompt_token_count") != expected_materialized_1024_100k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row prompt count mismatch")
    if row.get("requested_output_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row request mismatch")
    if row.get("generated_token_count") != expected_materialized_1024_100k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row generated count mismatch")
    if row.get("stopped_before_request_limit") is not expected_materialized_1024_100k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row early-stop mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_1024_100k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
        or limitations.get("full_acceptance_context_ladder") is not False
    ):
      raise SystemExit(f"{case_id}: materialized distribution 1024 100k limitations mismatch")
  materialized_1024_100k_correctness = json.loads(
      materialized_1024_100k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_1024_100k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 100k correctness failed")
  model_materialized_1024_100k = model_reference_artifacts.get(
      "latest_r0_materialized_distribution_capture_1024_100k", {}
  )
  if model_materialized_1024_100k != latest_materialized_1024_100k:
    raise SystemExit("model/oracle materialized distribution 1024 100k registration mismatch")
  latest_materialized_1024_128k = capture_plan.get(
      "latest_materialized_distribution_capture_1024_128k", {}
  )
  if (
      latest_materialized_1024_128k.get("tool")
      != "tools/intel-qwen36-r0-distribution-capture-materialized.py"
  ):
    raise SystemExit("latest materialized distribution 1024 128k tool mismatch")
  if latest_materialized_1024_128k.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 128k checks must pass")
  expected_materialized_1024_128k_cases = [
      "sentinel_128k",
      "prefill_shape_128k",
  ]
  if latest_materialized_1024_128k.get("cases") != expected_materialized_1024_128k_cases:
    raise SystemExit("latest materialized distribution 1024 128k cases mismatch")
  if latest_materialized_1024_128k.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 1024 128k case count mismatch")
  if latest_materialized_1024_128k.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 128k request count mismatch")
  if latest_materialized_1024_128k.get("total_distribution_positions") != 1041:
    raise SystemExit("latest materialized distribution 1024 128k position count mismatch")
  if latest_materialized_1024_128k.get("stopped_before_request_count") != 1:
    raise SystemExit("latest materialized distribution 1024 128k early-stop count mismatch")
  if latest_materialized_1024_128k.get("full_acceptance_bundle") is not False:
    raise SystemExit("latest materialized distribution 1024 128k must not claim full acceptance")
  if latest_materialized_1024_128k.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 128k must keep oracle gate open")
  materialized_1024_128k_path = latest_materialized_1024_128k.get("path")
  if not isinstance(materialized_1024_128k_path, str) or not materialized_1024_128k_path:
    raise SystemExit("latest materialized distribution 1024 128k path missing")
  materialized_1024_128k_dir = ROOT / materialized_1024_128k_path
  materialized_1024_128k_capture_path = materialized_1024_128k_dir / "capture.json"
  materialized_1024_128k_correctness_path = materialized_1024_128k_dir / "correctness.json"
  materialized_1024_128k_jsonl_path = (
      materialized_1024_128k_dir / "teacher-forced-distribution-materialized.jsonl"
  )
  if (
      not materialized_1024_128k_capture_path.exists()
      or not materialized_1024_128k_correctness_path.exists()
      or not materialized_1024_128k_jsonl_path.exists()
  ):
    raise SystemExit("latest materialized distribution 1024 128k artifact missing")
  materialized_1024_128k_capture = json.loads(
      materialized_1024_128k_capture_path.read_text(encoding="utf-8")
  )
  if materialized_1024_128k_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 128k artifact checks failed")
  if materialized_1024_128k_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest materialized distribution 1024 128k artifact must keep gate open")
  if materialized_1024_128k_capture.get("captured_case_count") != 2:
    raise SystemExit("latest materialized distribution 1024 128k artifact case count mismatch")
  if materialized_1024_128k_capture.get("requested_output_token_count") != 1024:
    raise SystemExit("latest materialized distribution 1024 128k artifact request mismatch")
  if materialized_1024_128k_capture.get("total_distribution_positions") != 1041:
    raise SystemExit("latest materialized distribution 1024 128k artifact position mismatch")
  expected_materialized_1024_128k_prompt_counts = {
      "sentinel_128k": 131072,
      "prefill_shape_128k": 131072,
  }
  expected_materialized_1024_128k_position_counts = {
      "sentinel_128k": 17,
      "prefill_shape_128k": 1024,
  }
  expected_materialized_1024_128k_early_stop = {
      "sentinel_128k": True,
      "prefill_shape_128k": False,
  }
  materialized_1024_128k_case_results = materialized_1024_128k_capture.get(
      "case_results", []
  )
  if [case.get("case_id") for case in materialized_1024_128k_case_results] != expected_materialized_1024_128k_cases:
    raise SystemExit("latest materialized distribution 1024 128k artifact case order mismatch")
  for case in materialized_1024_128k_case_results:
    case_id = case.get("case_id")
    expected_prompt_count = expected_materialized_1024_128k_prompt_counts[case_id]
    if case.get("expected_prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k expected prompt count mismatch")
    if case.get("prompt_token_count") != expected_prompt_count:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k observed prompt count mismatch")
    if case.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k status mismatch")
    if case.get("result_ok") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k result must pass")
    if case.get("captured_positions") != expected_materialized_1024_128k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k position count mismatch")
    if case.get("stopped_before_request_limit") is not expected_materialized_1024_128k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k early-stop mismatch")
    if case.get("top_logprobs_present") is not True:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k missing top-logprobs")
  materialized_1024_128k_command_results = materialized_1024_128k_capture.get(
      "command_results", []
  )
  if [case.get("case_id") for case in materialized_1024_128k_command_results] != expected_materialized_1024_128k_cases:
    raise SystemExit("latest materialized distribution 1024 128k command order mismatch")
  if any(case.get("timed_out") is not False for case in materialized_1024_128k_command_results):
    raise SystemExit("latest materialized distribution 1024 128k command timed out")
  materialized_1024_128k_rows = load_jsonl(materialized_1024_128k_jsonl_path)
  if len(materialized_1024_128k_rows) != 2:
    raise SystemExit("latest materialized distribution 1024 128k JSONL row count mismatch")
  if [row.get("case_id") for row in materialized_1024_128k_rows] != expected_materialized_1024_128k_cases:
    raise SystemExit("latest materialized distribution 1024 128k JSONL case order mismatch")
  for row in materialized_1024_128k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row workstream mismatch")
    if row.get("capture_status") != "captured_materialized_prompt_distribution_subset":
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row status mismatch")
    if row.get("prompt_token_count") != expected_materialized_1024_128k_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row prompt count mismatch")
    if row.get("requested_output_token_count") != 1024:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row request mismatch")
    if row.get("generated_token_count") != expected_materialized_1024_128k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row generated count mismatch")
    if row.get("stopped_before_request_limit") is not expected_materialized_1024_128k_early_stop[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row early-stop mismatch")
    positions = row.get("distribution_positions", [])
    if not isinstance(positions, list) or len(positions) != expected_materialized_1024_128k_position_counts[case_id]:
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row position mismatch")
    if any(not position.get("top_logprobs") for position in positions):
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k row missing top-logprobs")
    limitations = row.get("limitations", {})
    if (
        limitations.get("materialized_prompt_subset_only") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
        or limitations.get("not_a_per_boundary_tensor_bundle") is not True
        or limitations.get("full_acceptance_context_ladder") is not False
    ):
      raise SystemExit(f"{case_id}: materialized distribution 1024 128k limitations mismatch")
  materialized_1024_128k_correctness = json.loads(
      materialized_1024_128k_correctness_path.read_text(encoding="utf-8")
  )
  if materialized_1024_128k_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest materialized distribution 1024 128k correctness failed")
  model_materialized_1024_128k = model_reference_artifacts.get(
      "latest_r0_materialized_distribution_capture_1024_128k", {}
  )
  if model_materialized_1024_128k != latest_materialized_1024_128k:
    raise SystemExit("model/oracle materialized distribution 1024 128k registration mismatch")
  materialized_1024_coverage = capture_plan.get(
      "materialized_distribution_capture_1024_coverage", {}
  )
  expected_materialized_1024_coverage_cases = (
      expected_materialized_1024_cases
      + expected_materialized_1024_8k16k_cases
      + expected_materialized_1024_32k64k_cases
      + expected_materialized_1024_100k_cases
      + expected_materialized_1024_128k_cases
  )
  if materialized_1024_coverage.get("artifact_count") != 5:
    raise SystemExit("materialized distribution 1024 coverage artifact count mismatch")
  if materialized_1024_coverage.get("paths") != [
      materialized_1024_path,
      materialized_1024_8k16k_path,
      materialized_1024_32k64k_path,
      materialized_1024_100k_path,
      materialized_1024_128k_path,
  ]:
    raise SystemExit("materialized distribution 1024 coverage paths mismatch")
  if materialized_1024_coverage.get("captured_case_count") != 18:
    raise SystemExit("materialized distribution 1024 coverage case count mismatch")
  if materialized_1024_coverage.get("cases") != expected_materialized_1024_coverage_cases:
    raise SystemExit("materialized distribution 1024 coverage cases mismatch")
  if materialized_1024_coverage.get("requested_output_token_count") != 1024:
    raise SystemExit("materialized distribution 1024 coverage request mismatch")
  if materialized_1024_coverage.get("total_distribution_positions") != 12315:
    raise SystemExit("materialized distribution 1024 coverage position mismatch")
  if materialized_1024_coverage.get("stopped_before_request_count") != 11:
    raise SystemExit("materialized distribution 1024 coverage early-stop mismatch")
  if materialized_1024_coverage.get("full_acceptance_bundle") is not False:
    raise SystemExit("materialized distribution 1024 coverage must not claim full acceptance")
  if materialized_1024_coverage.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("materialized distribution 1024 coverage must keep oracle gate open")
  model_materialized_1024_coverage = model_reference_artifacts.get(
      "materialized_distribution_capture_1024_coverage", {}
  )
  if model_materialized_1024_coverage != materialized_1024_coverage:
    raise SystemExit("model/oracle materialized distribution 1024 coverage mismatch")
  latest_capture_queue = capture_plan.get("latest_oracle_capture_queue", {})
  if latest_capture_queue.get("tool") != "tools/intel-qwen36-r0-oracle-capture-queue.py":
    raise SystemExit("latest oracle capture queue tool mismatch")
  if latest_capture_queue.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle capture queue checks must pass")
  if latest_capture_queue.get("token_topk_task_count") != 26:
    raise SystemExit("latest oracle capture queue token/top-k task count mismatch")
  if latest_capture_queue.get("teacher_forced_distribution_task_count") != 26:
    raise SystemExit("latest oracle capture queue distribution task count mismatch")
  if latest_capture_queue.get("boundary_input_task_count") != 524:
    raise SystemExit("latest oracle capture queue boundary input count mismatch")
  if latest_capture_queue.get("boundary_output_task_count") != 524:
    raise SystemExit("latest oracle capture queue boundary output count mismatch")
  if latest_capture_queue.get("total_bundle_jsonl_rows") != 1100:
    raise SystemExit("latest oracle capture queue total row count mismatch")
  if latest_capture_queue.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle capture queue must keep oracle gate open")
  capture_queue_path = latest_capture_queue.get("path")
  if not isinstance(capture_queue_path, str) or not capture_queue_path:
    raise SystemExit("latest oracle capture queue path missing")
  capture_queue_dir = ROOT / capture_queue_path
  capture_queue_json_path = capture_queue_dir / "capture-queue.json"
  capture_queue_correctness_path = capture_queue_dir / "correctness.json"
  if not capture_queue_json_path.exists() or not capture_queue_correctness_path.exists():
    raise SystemExit("latest oracle capture queue artifact missing")
  capture_queue_json = json.loads(capture_queue_json_path.read_text(encoding="utf-8"))
  capture_queue_totals = capture_queue_json.get("task_totals", {})
  if capture_queue_totals.get("token_topk_tasks") != 26:
    raise SystemExit("latest oracle capture queue artifact token/top-k count mismatch")
  if capture_queue_totals.get("teacher_forced_distribution_tasks") != 26:
    raise SystemExit("latest oracle capture queue artifact distribution count mismatch")
  if capture_queue_totals.get("boundary_input_tasks") != 524:
    raise SystemExit("latest oracle capture queue artifact boundary input count mismatch")
  if capture_queue_totals.get("boundary_output_tasks") != 524:
    raise SystemExit("latest oracle capture queue artifact boundary output count mismatch")
  if capture_queue_totals.get("total_bundle_jsonl_rows") != 1100:
    raise SystemExit("latest oracle capture queue artifact total count mismatch")
  if capture_queue_json.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle capture queue artifact must keep gate open")
  task_file_counts = {
      "token-topk-tasks.jsonl": 26,
      "teacher-forced-distribution-tasks.jsonl": 26,
      "boundary-input-tasks.jsonl": 524,
      "boundary-output-tasks.jsonl": 524,
  }
  for task_file, expected_count in task_file_counts.items():
    rows = load_jsonl(capture_queue_dir / task_file)
    if len(rows) != expected_count:
      raise SystemExit(f"{task_file} row count mismatch: {len(rows)}")
    if any(row.get("capture_status") != "missing" for row in rows):
      raise SystemExit(f"{task_file} must not claim captured rows")
  capture_queue_correctness = json.loads(
      capture_queue_correctness_path.read_text(encoding="utf-8")
  )
  if capture_queue_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle capture queue artifact checks failed")
  latest_prompt_materialization = capture_plan.get("latest_oracle_prompt_materialization", {})
  if (
      latest_prompt_materialization.get("tool")
      != "tools/intel-qwen36-r0-oracle-prompt-materialize.py"
  ):
    raise SystemExit("latest oracle prompt materialization tool mismatch")
  if latest_prompt_materialization.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle prompt materialization checks must pass")
  if latest_prompt_materialization.get("prompt_row_count") != 26:
    raise SystemExit("latest oracle prompt materialization row count mismatch")
  if latest_prompt_materialization.get("generated_prompt_row_count") != 20:
    raise SystemExit("latest oracle prompt materialization generated row count mismatch")
  if latest_prompt_materialization.get("exact_generated_prompt_row_count") != 20:
    raise SystemExit("latest oracle prompt materialization exact row count mismatch")
  if latest_prompt_materialization.get("max_target_prompt_tokens") != 262144:
    raise SystemExit("latest oracle prompt materialization max target mismatch")
  if latest_prompt_materialization.get("full_oracle_bundle") is not False:
    raise SystemExit("latest oracle prompt materialization must not claim full bundle")
  if latest_prompt_materialization.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle prompt materialization must keep oracle gate open")
  prompt_materialization_path = latest_prompt_materialization.get("path")
  if not isinstance(prompt_materialization_path, str) or not prompt_materialization_path:
    raise SystemExit("latest oracle prompt materialization path missing")
  prompt_materialization_dir = ROOT / prompt_materialization_path
  prompt_materialization_json_path = prompt_materialization_dir / "materialization.json"
  prompt_materialization_correctness_path = prompt_materialization_dir / "correctness.json"
  prompt_materialization_jsonl_path = prompt_materialization_dir / "materialized-prompts.jsonl"
  if (
      not prompt_materialization_json_path.exists()
      or not prompt_materialization_correctness_path.exists()
      or not prompt_materialization_jsonl_path.exists()
  ):
    raise SystemExit("latest oracle prompt materialization artifact missing")
  prompt_materialization = json.loads(
      prompt_materialization_json_path.read_text(encoding="utf-8")
  )
  if prompt_materialization.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle prompt materialization artifact checks failed")
  if prompt_materialization.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle prompt materialization artifact must keep gate open")
  if prompt_materialization.get("prompt_row_count") != 26:
    raise SystemExit("latest oracle prompt materialization artifact row count mismatch")
  if prompt_materialization.get("generated_prompt_row_count") != 20:
    raise SystemExit("latest oracle prompt materialization artifact generated count mismatch")
  if prompt_materialization.get("exact_generated_prompt_row_count") != 20:
    raise SystemExit("latest oracle prompt materialization artifact exact count mismatch")
  prompt_rows = load_jsonl(prompt_materialization_jsonl_path)
  if len(prompt_rows) != 26:
    raise SystemExit("latest oracle prompt materialization JSONL row count mismatch")
  queue_token_rows = load_jsonl(capture_queue_dir / "token-topk-tasks.jsonl")
  queue_case_order = [row.get("case_id") for row in queue_token_rows]
  if [row.get("case_id") for row in prompt_rows] != queue_case_order:
    raise SystemExit("latest oracle prompt materialization case order mismatch")
  expected_targets = {
      "sentinel_001k": 1024,
      "sentinel_002k": 2048,
      "sentinel_004k": 4096,
      "sentinel_008k": 8192,
      "sentinel_016k": 16384,
      "sentinel_032k": 32768,
      "sentinel_064k": 65536,
      "sentinel_100k": 102400,
      "sentinel_128k": 131072,
      "sentinel_256k": 262144,
      "prefill_shape_001k": 1024,
      "prefill_shape_002k": 2048,
      "prefill_shape_004k": 4096,
      "prefill_shape_008k": 8192,
      "prefill_shape_016k": 16384,
      "prefill_shape_032k": 32768,
      "prefill_shape_064k": 65536,
      "prefill_shape_100k": 102400,
      "prefill_shape_128k": 131072,
      "prefill_shape_256k": 262144,
  }
  generated_count = 0
  for row in prompt_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: materialized prompt workstream mismatch")
    if row.get("model", {}).get("path") != model["gguf_model_path"]:
      raise SystemExit(f"{case_id}: materialized prompt model path mismatch")
    if row.get("model", {}).get("sha256") != model["gguf_sha256"]:
      raise SystemExit(f"{case_id}: materialized prompt model sha mismatch")
    prompt_path_value = row.get("materialized_prompt_path")
    if not isinstance(prompt_path_value, str) or not (ROOT / prompt_path_value).exists():
      raise SystemExit(f"{case_id}: materialized prompt file missing")
    prompt_path = ROOT / prompt_path_value
    file_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if file_hash != row.get("prompt_file_sha256"):
      raise SystemExit(f"{case_id}: materialized prompt file sha mismatch")
    if hashlib.sha256(prompt_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != row.get("prompt_utf8_sha256"):
      raise SystemExit(f"{case_id}: materialized prompt text sha mismatch")
    observed = row.get("observed_prompt_tokens")
    if not isinstance(observed, int) or observed <= 0:
      raise SystemExit(f"{case_id}: materialized prompt missing observed token count")
    target = row.get("target_prompt_tokens")
    if case_id in expected_targets:
      generated_count += 1
      if target != expected_targets[case_id]:
        raise SystemExit(f"{case_id}: materialized prompt target mismatch")
      if observed != target:
        raise SystemExit(f"{case_id}: materialized prompt observed count mismatch")
      if row.get("exact_target_prompt_tokens") is not True:
        raise SystemExit(f"{case_id}: materialized prompt must be exact")
    else:
      if target is not None:
        raise SystemExit(f"{case_id}: token-exact materialized prompt should not have target")
  if generated_count != 20:
    raise SystemExit("latest oracle prompt materialization generated row coverage mismatch")
  prompt_materialization_correctness = json.loads(
      prompt_materialization_correctness_path.read_text(encoding="utf-8")
  )
  if prompt_materialization_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle prompt materialization correctness failed")
  latest_token_id_capture = capture_plan.get("latest_oracle_token_id_capture", {})
  if (
      latest_token_id_capture.get("tool")
      != "tools/intel-qwen36-r0-oracle-token-id-capture.py"
  ):
    raise SystemExit("latest oracle token-id capture tool mismatch")
  if latest_token_id_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle token-id capture checks must pass")
  if latest_token_id_capture.get("captured_row_count") != 26:
    raise SystemExit("latest oracle token-id capture row count mismatch")
  if latest_token_id_capture.get("total_prompt_tokens") != 1251478:
    raise SystemExit("latest oracle token-id capture total token count mismatch")
  if latest_token_id_capture.get("max_prompt_tokens") != 262144:
    raise SystemExit("latest oracle token-id capture max token count mismatch")
  if latest_token_id_capture.get("top_k_logprobs_available") is not False:
    raise SystemExit("latest oracle token-id capture must not claim top-k logits")
  if latest_token_id_capture.get("full_oracle_bundle") is not False:
    raise SystemExit("latest oracle token-id capture must not claim full bundle")
  if latest_token_id_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle token-id capture must keep oracle gate open")
  token_id_capture_path = latest_token_id_capture.get("path")
  if not isinstance(token_id_capture_path, str) or not token_id_capture_path:
    raise SystemExit("latest oracle token-id capture path missing")
  token_id_capture_dir = ROOT / token_id_capture_path
  token_id_capture_json_path = token_id_capture_dir / "capture.json"
  token_id_capture_correctness_path = token_id_capture_dir / "correctness.json"
  token_id_capture_jsonl_path = token_id_capture_dir / "prompt-token-id-references.jsonl"
  if (
      not token_id_capture_json_path.exists()
      or not token_id_capture_correctness_path.exists()
      or not token_id_capture_jsonl_path.exists()
  ):
    raise SystemExit("latest oracle token-id capture artifact missing")
  token_id_capture = json.loads(token_id_capture_json_path.read_text(encoding="utf-8"))
  if token_id_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle token-id capture artifact checks failed")
  if token_id_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle token-id capture artifact must keep gate open")
  if token_id_capture.get("captured_row_count") != 26:
    raise SystemExit("latest oracle token-id capture artifact row count mismatch")
  if token_id_capture.get("total_prompt_tokens") != 1251478:
    raise SystemExit("latest oracle token-id capture artifact total mismatch")
  if token_id_capture.get("max_prompt_tokens") != 262144:
    raise SystemExit("latest oracle token-id capture artifact max mismatch")
  token_rows = load_jsonl(token_id_capture_jsonl_path)
  if len(token_rows) != 26:
    raise SystemExit("latest oracle token-id capture JSONL row count mismatch")
  if [row.get("case_id") for row in token_rows] != [row.get("case_id") for row in prompt_rows]:
    raise SystemExit("latest oracle token-id capture case order mismatch")
  materialized_by_case = {row.get("case_id"): row for row in prompt_rows}
  total_token_count = 0
  for row in token_rows:
    case_id = row.get("case_id")
    source = materialized_by_case.get(case_id)
    if not isinstance(source, dict):
      raise SystemExit(f"{case_id}: token-id capture missing materialization source")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: token-id row workstream mismatch")
    token_ids = row.get("prompt_token_ids")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
      raise SystemExit(f"{case_id}: token-id row missing integer token ids")
    token_count = row.get("prompt_token_count")
    if token_count != len(token_ids):
      raise SystemExit(f"{case_id}: token-id row count does not match ids")
    if token_count != source.get("observed_prompt_tokens"):
      raise SystemExit(f"{case_id}: token-id count does not match materialization")
    if row.get("observed_prompt_tokens") != source.get("observed_prompt_tokens"):
      raise SystemExit(f"{case_id}: token-id observed count mismatch")
    if row.get("prompt_utf8_sha256") != source.get("prompt_utf8_sha256"):
      raise SystemExit(f"{case_id}: token-id prompt sha mismatch")
    computed_hash = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if computed_hash != row.get("prompt_token_ids_sha256"):
      raise SystemExit(f"{case_id}: token-id hash mismatch")
    limitations = row.get("limitations", {})
    if (
        limitations.get("prompt_token_ids_only") is not True
        or limitations.get("top_k_logprobs_available") is not False
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: token-id limitations mismatch")
    total_token_count += token_count
  if total_token_count != 1251478:
    raise SystemExit("latest oracle token-id capture computed total mismatch")
  token_id_capture_correctness = json.loads(
      token_id_capture_correctness_path.read_text(encoding="utf-8")
  )
  if token_id_capture_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle token-id capture correctness failed")
  latest_topk_smoke = capture_plan.get("latest_oracle_topk_smoke", {})
  if latest_topk_smoke.get("tool") != "tools/intel-qwen36-r0-oracle-topk-smoke.py":
    raise SystemExit("latest oracle top-k smoke tool mismatch")
  if latest_topk_smoke.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle top-k smoke checks must pass")
  if latest_topk_smoke.get("captured_row_count") != 2:
    raise SystemExit("latest oracle top-k smoke row count mismatch")
  expected_topk_cases = [
      "sentinel_128k",
      "prefill_shape_128k",
  ]
  if latest_topk_smoke.get("cases") != expected_topk_cases:
    raise SystemExit("latest oracle top-k smoke cases mismatch")
  if latest_topk_smoke.get("top_logprobs_per_row") != 5:
    raise SystemExit("latest oracle top-k smoke top-k count mismatch")
  if latest_topk_smoke.get("full_ladder_topk") is not False:
    raise SystemExit("latest oracle top-k smoke must not claim full ladder top-k")
  if latest_topk_smoke.get("full_oracle_bundle") is not False:
    raise SystemExit("latest oracle top-k smoke must not claim full bundle")
  if latest_topk_smoke.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle top-k smoke must keep oracle gate open")
  topk_smoke_path = latest_topk_smoke.get("path")
  if not isinstance(topk_smoke_path, str) or not topk_smoke_path:
    raise SystemExit("latest oracle top-k smoke path missing")
  topk_smoke_dir = ROOT / topk_smoke_path
  topk_smoke_capture_path = topk_smoke_dir / "capture.json"
  topk_smoke_correctness_path = topk_smoke_dir / "correctness.json"
  topk_smoke_jsonl_path = topk_smoke_dir / "topk-smoke.jsonl"
  if (
      not topk_smoke_capture_path.exists()
      or not topk_smoke_correctness_path.exists()
      or not topk_smoke_jsonl_path.exists()
  ):
    raise SystemExit("latest oracle top-k smoke artifact missing")
  topk_smoke_capture = json.loads(topk_smoke_capture_path.read_text(encoding="utf-8"))
  if topk_smoke_capture.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle top-k smoke artifact checks failed")
  if topk_smoke_capture.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest oracle top-k smoke artifact must keep gate open")
  if topk_smoke_capture.get("captured_row_count") != 2:
    raise SystemExit("latest oracle top-k smoke artifact row count mismatch")
  topk_rows = load_jsonl(topk_smoke_jsonl_path)
  if len(topk_rows) != 2:
    raise SystemExit("latest oracle top-k smoke JSONL row count mismatch")
  expected_prompt_counts = {
      "sentinel_128k": 131072,
      "prefill_shape_128k": 131072,
  }
  if [row.get("case_id") for row in topk_rows] != expected_topk_cases:
    raise SystemExit("latest oracle top-k smoke row case order mismatch")
  for row in topk_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: top-k smoke workstream mismatch")
    if row.get("capture_status") != "captured_first_token_topk_smoke":
      raise SystemExit(f"{case_id}: top-k smoke capture status mismatch")
    if row.get("request_status") != 200:
      raise SystemExit(f"{case_id}: top-k smoke request status mismatch")
    if row.get("prompt_token_count") != expected_prompt_counts[case_id]:
      raise SystemExit(f"{case_id}: top-k smoke prompt count mismatch")
    first_token = row.get("first_token", {})
    top_logprobs = first_token.get("top_logprobs")
    if not isinstance(top_logprobs, list) or len(top_logprobs) != 5:
      raise SystemExit(f"{case_id}: top-k smoke top-logprobs mismatch")
    if first_token.get("top1_id") != first_token.get("reference_token_id"):
      raise SystemExit(f"{case_id}: top-k smoke top1/reference mismatch")
    limitations = row.get("limitations", {})
    if (
        limitations.get("first_token_topk_smoke_only") is not True
        or limitations.get("not_full_ladder_topk") is not True
        or limitations.get("not_a_full_r0_oracle_bundle") is not True
    ):
      raise SystemExit(f"{case_id}: top-k smoke limitations mismatch")
  topk_smoke_correctness = json.loads(topk_smoke_correctness_path.read_text(encoding="utf-8"))
  if topk_smoke_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle top-k smoke correctness failed")
  latest_topk_256k_attempt = capture_plan.get(
      "latest_oracle_topk_256k_exact_context_attempt", {}
  )
  if (
      latest_topk_256k_attempt.get("tool")
      != "tools/intel-qwen36-r0-oracle-topk-smoke.py"
  ):
    raise SystemExit("latest 256k top-k exact-context attempt tool mismatch")
  if latest_topk_256k_attempt.get("required_checks_passed") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt must be a failed attempt")
  expected_256k_cases = [
      "sentinel_256k",
      "prefill_shape_256k",
  ]
  if latest_topk_256k_attempt.get("cases") != expected_256k_cases:
    raise SystemExit("latest 256k top-k exact-context attempt cases mismatch")
  if latest_topk_256k_attempt.get("prompt_tokens_per_row") != 262144:
    raise SystemExit("latest 256k top-k exact-context attempt prompt count mismatch")
  if latest_topk_256k_attempt.get("ctx_size") != 262144:
    raise SystemExit("latest 256k top-k exact-context attempt ctx mismatch")
  if latest_topk_256k_attempt.get("request_status") != 400:
    raise SystemExit("latest 256k top-k exact-context attempt status mismatch")
  if latest_topk_256k_attempt.get("failure_type") != "exceed_context_size_error":
    raise SystemExit("latest 256k top-k exact-context attempt failure type mismatch")
  if latest_topk_256k_attempt.get("full_ladder_topk") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt must not claim full top-k")
  if latest_topk_256k_attempt.get("full_oracle_bundle") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt must not claim full bundle")
  if latest_topk_256k_attempt.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt must keep gate open")
  topk_256k_attempt_path = latest_topk_256k_attempt.get("path")
  if not isinstance(topk_256k_attempt_path, str) or not topk_256k_attempt_path:
    raise SystemExit("latest 256k top-k exact-context attempt path missing")
  topk_256k_attempt_dir = ROOT / topk_256k_attempt_path
  topk_256k_capture_path = topk_256k_attempt_dir / "capture.json"
  topk_256k_correctness_path = topk_256k_attempt_dir / "correctness.json"
  topk_256k_jsonl_path = topk_256k_attempt_dir / "topk-smoke.jsonl"
  if (
      not topk_256k_capture_path.exists()
      or not topk_256k_correctness_path.exists()
      or not topk_256k_jsonl_path.exists()
  ):
    raise SystemExit("latest 256k top-k exact-context attempt artifact missing")
  topk_256k_capture = json.loads(topk_256k_capture_path.read_text(encoding="utf-8"))
  if topk_256k_capture.get("required_checks_passed") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt artifact must fail")
  if topk_256k_capture.get("captured_row_count") != 2:
    raise SystemExit("latest 256k top-k exact-context attempt artifact row count mismatch")
  case_results = topk_256k_capture.get("case_results", [])
  if [case.get("case_id") for case in case_results] != expected_256k_cases:
    raise SystemExit("latest 256k top-k exact-context attempt artifact case order mismatch")
  for case in case_results:
    case_id = case.get("case_id")
    if case.get("expected_prompt_token_count") != 262144:
      raise SystemExit(f"{case_id}: 256k top-k attempt expected count mismatch")
    if case.get("prompt_token_count") != 262144:
      raise SystemExit(f"{case_id}: 256k top-k attempt observed count mismatch")
    if case.get("request_status") != 400:
      raise SystemExit(f"{case_id}: 256k top-k attempt status mismatch")
    if case.get("result_ok") is not False:
      raise SystemExit(f"{case_id}: 256k top-k attempt result must fail")
    if case.get("top_logprob_count") != 0:
      raise SystemExit(f"{case_id}: 256k top-k attempt must not have top-logprobs")
  topk_256k_rows = load_jsonl(topk_256k_jsonl_path)
  if len(topk_256k_rows) != 2:
    raise SystemExit("latest 256k top-k exact-context attempt JSONL row count mismatch")
  if [row.get("case_id") for row in topk_256k_rows] != expected_256k_cases:
    raise SystemExit("latest 256k top-k exact-context attempt row case order mismatch")
  for row in topk_256k_rows:
    case_id = row.get("case_id")
    if row.get("workstream") != "intel-qwen36-35b-a3b-gguf-q4km":
      raise SystemExit(f"{case_id}: 256k top-k attempt row workstream mismatch")
    if row.get("request_status") != 400:
      raise SystemExit(f"{case_id}: 256k top-k attempt row request status mismatch")
    if row.get("prompt_token_count") != 262144:
      raise SystemExit(f"{case_id}: 256k top-k attempt row prompt count mismatch")
    first_token = row.get("first_token", {})
    if first_token.get("reference_token_id") is not None:
      raise SystemExit(f"{case_id}: 256k top-k attempt must not capture reference token")
    top_logprobs = first_token.get("top_logprobs")
    if top_logprobs != []:
      raise SystemExit(f"{case_id}: 256k top-k attempt must not capture top-logprobs")
  topk_256k_correctness = json.loads(
      topk_256k_correctness_path.read_text(encoding="utf-8")
  )
  if topk_256k_correctness.get("required_checks_passed") is not False:
    raise SystemExit("latest 256k top-k exact-context attempt correctness must fail")
  for case_number, case_id in enumerate(expected_256k_cases, start=1):
    raw_response_path = (
        topk_256k_attempt_dir
        / f"case-{case_number:02d}-{case_id}"
        / "raw"
        / "remote"
        / "completion_response.raw"
    )
    if not raw_response_path.exists():
      raise SystemExit(f"{case_id}: 256k top-k attempt raw response missing")
    raw_response = json.loads(raw_response_path.read_text(encoding="utf-8"))
    error = raw_response.get("error", {})
    if error.get("code") != 400:
      raise SystemExit(f"{case_id}: 256k top-k attempt raw error code mismatch")
    if error.get("type") != "exceed_context_size_error":
      raise SystemExit(f"{case_id}: 256k top-k attempt raw error type mismatch")
    if error.get("n_prompt_tokens") != 262144 or error.get("n_ctx") != 262144:
      raise SystemExit(f"{case_id}: 256k top-k attempt raw context mismatch")
  latest_256k_prompt_edge_policy = capture_plan.get(
      "latest_oracle_256k_prompt_edge_policy", {}
  )
  if (
      latest_256k_prompt_edge_policy.get("tool")
      != "tools/intel-qwen36-r0-oracle-256k-prompt-edge-policy.py"
  ):
    raise SystemExit("latest 256k prompt-edge policy tool mismatch")
  if latest_256k_prompt_edge_policy.get("required_checks_passed") is not True:
    raise SystemExit("latest 256k prompt-edge policy checks must pass")
  if (
      latest_256k_prompt_edge_policy.get("policy_id")
      != "r0_256k_exact_prompt_first_token_topk_edge"
  ):
    raise SystemExit("latest 256k prompt-edge policy id mismatch")
  if (
      latest_256k_prompt_edge_policy.get("decision")
      != "accept_exact_262144_prompt_first_token_topk_as_context_edge_for_r0"
  ):
    raise SystemExit("latest 256k prompt-edge policy decision mismatch")
  if latest_256k_prompt_edge_policy.get("exact_prompt_token_count") != 262144:
    raise SystemExit("latest 256k prompt-edge policy exact prompt count mismatch")
  if (
      latest_256k_prompt_edge_policy.get(
          "context_safe_max_prompt_tokens_for_first_token_prediction"
      )
      != 262143
  ):
    raise SystemExit("latest 256k prompt-edge policy safe prompt count mismatch")
  if latest_256k_prompt_edge_policy.get("prompt_edge_policy_gate_closed") is not True:
    raise SystemExit("latest 256k prompt-edge policy gate must be closed")
  if latest_256k_prompt_edge_policy.get("topk_logprobs_available") is not False:
    raise SystemExit("latest 256k prompt-edge policy must not claim logits")
  if latest_256k_prompt_edge_policy.get("full_oracle_bundle") is not False:
    raise SystemExit("latest 256k prompt-edge policy must not claim full bundle")
  if latest_256k_prompt_edge_policy.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest 256k prompt-edge policy must keep oracle gate open")
  prompt_edge_policy_path = latest_256k_prompt_edge_policy.get("path")
  if not isinstance(prompt_edge_policy_path, str) or not prompt_edge_policy_path:
    raise SystemExit("latest 256k prompt-edge policy path missing")
  prompt_edge_policy_dir = ROOT / prompt_edge_policy_path
  prompt_edge_policy_json_path = prompt_edge_policy_dir / "policy.json"
  prompt_edge_policy_correctness_path = prompt_edge_policy_dir / "correctness.json"
  if (
      not prompt_edge_policy_json_path.exists()
      or not prompt_edge_policy_correctness_path.exists()
  ):
    raise SystemExit("latest 256k prompt-edge policy artifact missing")
  prompt_edge_policy = json.loads(prompt_edge_policy_json_path.read_text(encoding="utf-8"))
  policy = prompt_edge_policy.get("policy", {})
  if policy.get("prompt_edge_policy_gate_closed") is not True:
    raise SystemExit("latest 256k prompt-edge policy artifact gate mismatch")
  if policy.get("topk_logprobs_available") is not False:
    raise SystemExit("latest 256k prompt-edge policy artifact must not claim logits")
  if policy.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("latest 256k prompt-edge policy artifact must keep gate open")
  if policy.get("context_length") != 262144:
    raise SystemExit("latest 256k prompt-edge policy artifact context mismatch")
  if policy.get("context_safe_max_prompt_tokens_for_first_token_prediction") != 262143:
    raise SystemExit("latest 256k prompt-edge policy artifact safe count mismatch")
  evidence = prompt_edge_policy.get("evidence", {})
  if evidence.get("exact_context_attempt") != topk_256k_attempt_path:
    raise SystemExit("latest 256k prompt-edge policy evidence attempt mismatch")
  prompt_edge_correctness = json.loads(
      prompt_edge_policy_correctness_path.read_text(encoding="utf-8")
  )
  if prompt_edge_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest 256k prompt-edge policy correctness failed")
  check_names = {
      check.get("name")
      for check in prompt_edge_correctness.get("checks", [])
      if check.get("pass") is True
  }
  for required_check in (
      "model_context_contract_locked",
      "materialized_256k_prompts_are_exact",
      "token_id_capture_has_exact_256k_rows",
      "exact_context_topk_attempt_failed_as_expected",
      "llama_server_reported_context_edge",
      "policy_does_not_close_oracle_gate",
  ):
    if required_check not in check_names:
      raise SystemExit(f"latest 256k prompt-edge policy missing check: {required_check}")
  latest_harness_load = capture_plan.get("latest_resident_harness_load", {})
  if latest_harness_load.get("tool") != "tools/intel-qwen36-r0-resident-harness-load.py":
    raise SystemExit("latest resident harness load tool mismatch")
  if latest_harness_load.get("schema_version") != "intel-qwen36-r0-resident-harness-load-v0":
    raise SystemExit("latest resident harness load schema mismatch")
  if latest_harness_load.get("executable") != "build/engine/iq36-load-bundle":
    raise SystemExit("latest resident harness load executable mismatch")
  if latest_harness_load.get("oracle_bundle") != "oracle/r0-oracle-bundle-20260627T060028Z":
    raise SystemExit("latest resident harness load bundle mismatch")
  if latest_harness_load.get("returncode") != 0:
    raise SystemExit("latest resident harness load return code mismatch")
  if latest_harness_load.get("resident_harness_loaded") is not True:
    raise SystemExit("latest resident harness load must enter loaded state")
  if latest_harness_load.get("token_topk_rows") != 26:
    raise SystemExit("latest resident harness load token/top-k row count mismatch")
  if latest_harness_load.get("teacher_forced_distribution_rows") != 26:
    raise SystemExit("latest resident harness load distribution row count mismatch")
  if latest_harness_load.get("boundary_input_rows") != 524:
    raise SystemExit("latest resident harness load boundary input row count mismatch")
  if latest_harness_load.get("boundary_output_rows") != 524:
    raise SystemExit("latest resident harness load boundary output row count mismatch")
  if latest_harness_load.get("r0_resident_harness_gate_closed") is not True:
    raise SystemExit("latest resident harness load must close resident gate")
  if latest_harness_load.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest resident harness load must not allow speedup claims")
  harness_load_path = latest_harness_load.get("path")
  if not isinstance(harness_load_path, str) or not harness_load_path:
    raise SystemExit("latest resident harness load path missing")
  harness_load_dir = ROOT / harness_load_path
  harness_load_json_path = harness_load_dir / "load.json"
  harness_load_correctness_path = harness_load_dir / "correctness.json"
  if not harness_load_json_path.exists() or not harness_load_correctness_path.exists():
    raise SystemExit("latest resident harness load artifact missing")
  harness_load_json = json.loads(harness_load_json_path.read_text(encoding="utf-8"))
  load_gate = harness_load_json.get("resident_harness_load_gate", {})
  if load_gate.get("returncode") != 0:
    raise SystemExit("latest resident harness load artifact return code mismatch")
  if load_gate.get("resident_harness_loaded") is not True:
    raise SystemExit("latest resident harness load artifact must enter loaded state")
  if load_gate.get("r0_resident_harness_gate_closed") is not True:
    raise SystemExit("latest resident harness load artifact must close gate")
  if load_gate.get("oracle_bundle_path") != latest_harness_load.get("oracle_bundle"):
    raise SystemExit("latest resident harness load artifact bundle mismatch")
  stdout = load_gate.get("stdout", "")
  for expected_fragment in (
      "token_topk_rows=26",
      "teacher_forced_distribution_rows=26",
      "boundary_input_rows=524",
      "boundary_output_rows=524",
  ):
    if expected_fragment not in stdout:
      raise SystemExit(f"latest resident harness load stdout missing {expected_fragment}")
  harness_load_correctness = json.loads(
      harness_load_correctness_path.read_text(encoding="utf-8")
  )
  if harness_load_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest resident harness load correctness failed")
  if harness_load_correctness.get("r0_resident_harness_gate_closed") is not True:
    raise SystemExit("latest resident harness load correctness must close gate")

  latest_harness_audit = capture_plan.get("latest_resident_harness_gate_audit", {})
  if latest_harness_audit.get("tool") != "tools/intel-qwen36-r0-resident-harness-gate-audit.py":
    raise SystemExit("latest resident harness gate audit tool mismatch")
  if latest_harness_audit.get("required_checks_passed") is not True:
    raise SystemExit("latest resident harness gate audit checks must pass")
  if latest_harness_audit.get("required_bundle_path_count") != len(required_bundle_paths):
    raise SystemExit("latest resident harness gate audit path count mismatch")
  if latest_harness_audit.get("candidate_real_bundle_count") != 1:
    raise SystemExit("latest resident harness audit must find one structurally loadable bundle")
  if latest_harness_audit.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("latest resident harness audit must see closed oracle gate")
  if latest_harness_audit.get("resident_harness_load_artifact") != "output/r0-resident-harness-load-20260627T061911Z/load.json":
    raise SystemExit("latest resident harness audit load artifact mismatch")
  if latest_harness_audit.get("resident_harness_load_executed") is not True:
    raise SystemExit("latest resident harness audit must record executed load")
  if latest_harness_audit.get("r0_resident_harness_gate_closed") is not True:
    raise SystemExit("latest resident harness audit must close gate")
  harness_audit_path = latest_harness_audit.get("path")
  if not isinstance(harness_audit_path, str) or not harness_audit_path:
    raise SystemExit("latest resident harness gate audit path missing")
  harness_audit_dir = ROOT / harness_audit_path
  harness_audit_json_path = harness_audit_dir / "audit.json"
  harness_audit_correctness_path = harness_audit_dir / "correctness.json"
  if not harness_audit_json_path.exists() or not harness_audit_correctness_path.exists():
    raise SystemExit("latest resident harness gate audit artifact missing")
  harness_audit_json = json.loads(harness_audit_json_path.read_text(encoding="utf-8"))
  if harness_audit_json.get("required_oracle_bundle_paths") != required_bundle_paths:
    raise SystemExit("latest resident harness gate audit required paths mismatch")
  if (
      harness_audit_json.get("resident_harness_gate", {}).get("r0_resident_harness_gate_closed")
      is not True
  ):
    raise SystemExit("latest resident harness gate audit must close gate")
  if (
      harness_audit_json.get("resident_harness_gate", {}).get("candidate_real_bundle_count")
      != 1
  ):
    raise SystemExit("latest resident harness gate audit artifact bundle count mismatch")
  if (
      harness_audit_json.get("resident_harness_gate", {}).get("resident_harness_load_executed")
      is not True
  ):
    raise SystemExit("latest resident harness gate audit artifact must record executed load")
  harness_audit_correctness = json.loads(
      harness_audit_correctness_path.read_text(encoding="utf-8")
  )
  if harness_audit_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest resident harness gate audit artifact checks failed")
  if "resident_harness_gate_closed" not in {
      check.get("name")
      for check in harness_audit_correctness.get("checks", [])
      if check.get("pass") is True
  }:
    raise SystemExit("latest resident harness gate audit missing closed-gate check")
  latest_oracle_bundle = capture_plan.get("latest_oracle_bundle", {})
  if (
      latest_oracle_bundle.get("tool")
      != "tools/intel-qwen36-r0-oracle-bundle-assemble.py"
  ):
    raise SystemExit("latest oracle bundle assembler tool mismatch")
  if latest_oracle_bundle.get("schema_version") != "intel-qwen36-r0-oracle-full-bundle-v0":
    raise SystemExit("latest oracle bundle schema mismatch")
  if latest_oracle_bundle.get("token_topk_rows") != 26:
    raise SystemExit("latest oracle bundle token/top-k row count mismatch")
  if latest_oracle_bundle.get("teacher_forced_distribution_rows") != 26:
    raise SystemExit("latest oracle bundle distribution row count mismatch")
  if latest_oracle_bundle.get("teacher_forced_distribution_positions") != 12744:
    raise SystemExit("latest oracle bundle distribution position count mismatch")
  if latest_oracle_bundle.get("boundary_input_rows") != 524:
    raise SystemExit("latest oracle bundle boundary input row count mismatch")
  if latest_oracle_bundle.get("boundary_output_rows") != 524:
    raise SystemExit("latest oracle bundle boundary output row count mismatch")
  if latest_oracle_bundle.get("prompt_edge_rows") != 2:
    raise SystemExit("latest oracle bundle prompt-edge row count mismatch")
  if latest_oracle_bundle.get("full_acceptance_bundle") is not True:
    raise SystemExit("latest oracle bundle must claim full acceptance")
  if latest_oracle_bundle.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("latest oracle bundle must close R0 oracle gate")
  bundle_path = latest_oracle_bundle.get("path")
  if not isinstance(bundle_path, str) or not bundle_path:
    raise SystemExit("latest oracle bundle path missing")
  bundle_dir = ROOT / bundle_path
  for required_bundle_path in oracle.get("required_bundle_paths", []):
    if not (bundle_dir / required_bundle_path).exists():
      raise SystemExit(f"latest oracle bundle missing required path: {required_bundle_path}")
  bundle_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
  if bundle_manifest.get("status", {}).get("full_acceptance_bundle") is not True:
    raise SystemExit("latest oracle bundle manifest must claim full acceptance")
  bundle_correctness = json.loads((bundle_dir / "correctness.json").read_text(encoding="utf-8"))
  if bundle_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle bundle correctness checks failed")
  if bundle_correctness.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("latest oracle bundle correctness must close oracle gate")
  latest_bundle_validation = capture_plan.get("latest_oracle_bundle_validation", {})
  if (
      latest_bundle_validation.get("tool")
      != "tools/intel-qwen36-r0-oracle-bundle-validate.py"
  ):
    raise SystemExit("latest oracle bundle validation tool mismatch")
  if latest_bundle_validation.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle bundle validation checks must pass")
  if latest_bundle_validation.get("candidate_bundle_count") != 1:
    raise SystemExit("latest oracle bundle validation must scan one candidate")
  if latest_bundle_validation.get("candidate_valid_bundle_count") != 1:
    raise SystemExit("latest oracle bundle validation must find one valid bundle")
  if latest_bundle_validation.get("expected_boundary_record_count") != 524:
    raise SystemExit("latest oracle bundle validation boundary count mismatch")
  if latest_bundle_validation.get("expected_prompt_row_count") != 26:
    raise SystemExit("latest oracle bundle validation prompt row count mismatch")
  if latest_bundle_validation.get("prompt_edge_policy_required") is not True:
    raise SystemExit("latest oracle bundle validation must require prompt-edge policy")
  if latest_bundle_validation.get("prompt_edge_case_ids") != [
      "prefill_shape_256k",
      "sentinel_256k",
  ]:
    raise SystemExit("latest oracle bundle validation prompt-edge cases mismatch")
  if latest_bundle_validation.get("oracle_contract_gate_closed") is not True:
    raise SystemExit("latest oracle bundle validation must record closed oracle contract gate")
  if latest_bundle_validation.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("latest oracle bundle validation must close oracle gate")
  bundle_validation_path = latest_bundle_validation.get("path")
  if not isinstance(bundle_validation_path, str) or not bundle_validation_path:
    raise SystemExit("latest oracle bundle validation path missing")
  bundle_validation_dir = ROOT / bundle_validation_path
  bundle_validation_audit_path = bundle_validation_dir / "audit.json"
  bundle_validation_correctness_path = bundle_validation_dir / "correctness.json"
  if (
      not bundle_validation_audit_path.exists()
      or not bundle_validation_correctness_path.exists()
  ):
    raise SystemExit("latest oracle bundle validation artifact missing")
  bundle_validation_audit = json.loads(
      bundle_validation_audit_path.read_text(encoding="utf-8")
  )
  bundle_gate = bundle_validation_audit.get("oracle_bundle_validation_gate", {})
  if bundle_gate.get("candidate_bundle_count") != 1:
    raise SystemExit("latest oracle bundle validation artifact candidate count mismatch")
  if bundle_gate.get("candidate_valid_bundle_count") != 1:
    raise SystemExit("latest oracle bundle validation artifact valid count mismatch")
  if bundle_gate.get("oracle_contract_gate_closed") is not True:
    raise SystemExit("latest oracle bundle validation artifact must record closed contract gate")
  if bundle_gate.get("r0_oracle_gate_closed") is not True:
    raise SystemExit("latest oracle bundle validation artifact must close gate")
  candidate_status = bundle_validation_audit.get("candidate_bundle_status", [{}])[0]
  if candidate_status.get("path") != bundle_path:
    raise SystemExit("latest oracle bundle validation artifact candidate path mismatch")
  if candidate_status.get("valid_full_oracle_bundle") is not True:
    raise SystemExit("latest oracle bundle validation artifact candidate must be valid")
  bundle_coverage = bundle_validation_audit.get("expected_coverage", {})
  if bundle_coverage.get("expected_boundary_record_count") != 524:
    raise SystemExit("latest oracle bundle validation artifact boundary count mismatch")
  if bundle_coverage.get("expected_prompt_row_count") != 26:
    raise SystemExit("latest oracle bundle validation artifact prompt count mismatch")
  if bundle_coverage.get("prompt_edge_policy_required") is not True:
    raise SystemExit("latest oracle bundle validation artifact must require prompt-edge policy")
  if bundle_coverage.get("prompt_edge_case_ids") != [
      "prefill_shape_256k",
      "sentinel_256k",
  ]:
    raise SystemExit("latest oracle bundle validation artifact prompt-edge cases mismatch")
  if (
      bundle_coverage.get("prompt_edge_policy_path")
      != "output/r0-oracle-256k-prompt-edge-policy-20260626T145727Z"
  ):
    raise SystemExit("latest oracle bundle validation artifact prompt-edge policy path mismatch")
  bundle_validation_correctness = json.loads(
      bundle_validation_correctness_path.read_text(encoding="utf-8")
  )
  if bundle_validation_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest oracle bundle validation artifact checks failed")
  bundle_validation_check_names = {
      check.get("name")
      for check in bundle_validation_correctness.get("checks", [])
      if check.get("pass") is True
  }
  if "prompt_edge_policy_available" not in bundle_validation_check_names:
    raise SystemExit("latest oracle bundle validation missing prompt-edge policy check")
  if "valid_oracle_bundle_required" not in bundle_validation_check_names:
    raise SystemExit("latest oracle bundle validation missing valid bundle required check")

  # ---------------------------------------------------------------------------
  # R1 component-compare status comes from the boundary registry and the
  # generated ladder rollup, not from a per-compare blob copied into the
  # contracts.  Until 2026-06-28 this section was ~9900 lines that pinned every
  # compare's float results inside contracts/*.json (a ~320KB-per-contract
  # blob).  That blob and its golden assertions are replaced by
  # engine/boundaries.json + output/ladder.json (regenerate the ladder with
  # tools/iq36-ladder.py).
  model_contract = parsed["contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"]
  for contract_name, contract_value in (
      ("target", target_contract),
      ("model", model_contract),
  ):
    if "r1_native_correctness" in contract_value:
      raise SystemExit(
          f"{contract_name} contract must not carry the r1_native_correctness "
          "progress blob; per-compare status lives in output/ladder.json"
      )

  registry = load_json("engine/boundaries.json")
  if registry.get("schema_version") != "intel-qwen36-boundaries-v1":
    raise SystemExit("boundary registry schema mismatch")
  registry_infra_targets = registry.get("infra_targets", [])
  if not isinstance(registry_infra_targets, list) or not registry_infra_targets:
    raise SystemExit("boundary registry has no infra targets")
  registry_infra_target_names = []
  for infra_target in registry_infra_targets:
    for field in ("target", "source"):
      if not infra_target.get(field):
        raise SystemExit(f"boundary registry infra target missing field: {field}")
    if not (ROOT / "engine" / infra_target["source"]).exists():
      raise SystemExit(f"boundary registry infra source missing: {infra_target['source']}")
    registry_infra_target_names.append(infra_target["target"])
  if len(registry_infra_target_names) != len(set(registry_infra_target_names)):
    raise SystemExit("boundary registry has duplicate infra targets")
  registry_boundaries = registry.get("boundaries", [])
  if len(registry_boundaries) < 1:
    raise SystemExit("boundary registry has no boundaries")
  registry_ids = []
  for boundary in registry_boundaries:
    for field in ("id", "target", "source", "artifact_prefix"):
      if not boundary.get(field):
        raise SystemExit(f"boundary registry entry missing field: {field}")
    if not (ROOT / "engine" / boundary["source"]).exists():
      raise SystemExit(f"boundary registry source missing: {boundary['source']}")
    registry_ids.append(boundary["id"])
  if len(registry_ids) != len(set(registry_ids)):
    raise SystemExit("boundary registry has duplicate ids")

  cmake_text = (ROOT / "engine" / "CMakeLists.txt").read_text(encoding="utf-8")
  if "boundaries.json" not in cmake_text:
    raise SystemExit(
        "engine/CMakeLists.txt must generate targets from boundaries.json"
    )

  ladder = load_json("output/ladder.json")
  if ladder.get("schema_version") != "intel-qwen36-ladder-v1":
    raise SystemExit("ladder schema mismatch")
  if ladder.get("selector_gate") != "r1_native_gguf_correctness_first_token_loop":
    raise SystemExit("ladder selector gate mismatch")
  if [row.get("id") for row in ladder.get("boundaries", [])] != registry_ids:
    raise SystemExit(
        "ladder boundaries out of sync with registry; run tools/iq36-ladder.py"
    )
  if ladder.get("boundary_count") != len(registry_ids):
    raise SystemExit("ladder boundary_count mismatch")
  for row in ladder["boundaries"]:
    if row.get("status") == "pass" and (
        row.get("r1_native_correctness_gate_closed") is not False
    ):
      raise SystemExit(
          f"ladder boundary {row.get('id')} must keep the R1 gate open"
      )

  route_dirs = sorted((ROOT / "output").glob("r1-native-candidate-route-*"))
  if not route_dirs:
    raise SystemExit("missing R1 native candidate route artifact")
  route_correctness = load_json(
      route_dirs[-1].relative_to(ROOT).as_posix() + "/correctness.json"
  )
  if route_correctness.get("gate") != "r1_native_candidate_route":
    raise SystemExit("latest native candidate route gate mismatch")
  if route_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest native candidate route checks must pass")
  if route_correctness.get("r1_native_correctness_gate_closed") is not False:
    raise SystemExit("latest native candidate route must keep the R1 gate open")

  candidate_dirs = sorted((ROOT / "output").glob("r1-native-candidate-jsonl-*"))
  if not candidate_dirs:
    raise SystemExit("missing R1 native candidate JSONL artifact")
  candidate_dir = None
  candidate_correctness = {}
  for possible_candidate_dir in reversed(candidate_dirs):
    possible_correctness_path = possible_candidate_dir / "correctness.json"
    if not possible_correctness_path.exists():
      continue
    possible_correctness = json.loads(
        possible_correctness_path.read_text(encoding="utf-8")
    )
    if (
        possible_correctness.get("required_checks_passed") is True
        and possible_correctness.get("r1_native_correctness_gate_closed") is True
    ):
      if possible_correctness.get("selected_expert_minimal_outputs") is True:
        continue
      if possible_correctness.get("selected_expert_down_expert_major") is True:
        continue
      if possible_correctness.get("selected_expert_down_q4_pair_dot") is True:
        continue
      if possible_correctness.get("selected_expert_down_q6_pair_dot") is True:
        continue
      if possible_correctness.get("q4_direct_minsum_pair") is True:
        continue
      if possible_correctness.get("q4_block_meta_cache") is True:
        continue
      if possible_correctness.get("small_q4_direct_dot") is True:
        continue
      if possible_correctness.get("matvec_q8_input_reuse") is True:
        continue
      if possible_correctness.get("decode_top1_only") is True:
        continue
      if possible_correctness.get("lm_head_q6_pair_dot") is True:
        continue
      if possible_correctness.get("shared_expert_gate_up_fused") is True:
        continue
      possible_manifest_path = possible_candidate_dir / "manifest.json"
      if not possible_manifest_path.exists():
        continue
      possible_manifest = load_json_path(possible_manifest_path)
      if possible_manifest.get("dense_matvec_threads") != 16:
        continue
      if possible_manifest.get("dense_matvec_min_rows") != 256:
        continue
      candidate_dir = possible_candidate_dir
      candidate_correctness = possible_correctness
      break
  if candidate_dir is None:
    raise SystemExit("missing gate-closing R1 native candidate JSONL artifact")
  if candidate_correctness.get("gate") != "r1_native_candidate_jsonl_generation":
    raise SystemExit("gate-closing native candidate JSONL generation gate mismatch")
  if candidate_correctness.get("required_checks_passed") is not True:
    raise SystemExit("gate-closing native candidate JSONL generation checks must pass")
  if candidate_correctness.get("native_candidate_jsonl_emitted") is not True:
    raise SystemExit("gate-closing native candidate JSONL artifact must emit rows")
  if candidate_correctness.get("r1_native_correctness_gate_closed") is not True:
    raise SystemExit("gate-closing native candidate JSONL artifact must close R1")
  if candidate_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("gate-closing native candidate JSONL must forbid speed claims")
  if candidate_correctness.get("dense_q6_pair_dot") is not True:
    raise SystemExit("latest native candidate JSONL must enable dense Q6 pair dot")
  candidate_rows = load_jsonl(candidate_dir / "candidate.jsonl")
  if len(candidate_rows) != 6:
    raise SystemExit("latest native candidate JSONL must contain six seed rows")
  if any(row.get("native_output_source") != "intel_qwen36_native" for row in candidate_rows):
    raise SystemExit("latest native candidate JSONL must be native sourced")
  candidate_gate = json.loads(
      (candidate_dir / "gate" / "gate.json").read_text(encoding="utf-8")
  )
  candidate_gate_state = candidate_gate.get("r1_native_correctness_gate", {})
  if candidate_gate_state.get("r1_native_correctness_gate_closed") is not True:
    raise SystemExit("latest nested native correctness gate must be closed")
  if candidate_gate_state.get("missing_for_gate") != []:
    raise SystemExit("latest nested native correctness gate must have no missing items")
  if candidate_gate_state.get("oracle_seed_row_count") != 6:
    raise SystemExit("latest nested native correctness gate seed row count mismatch")
  candidate_case_results = candidate_gate.get("case_results", [])
  if len(candidate_case_results) != 6:
    raise SystemExit("latest native candidate gate case count mismatch")
  for case_result in candidate_case_results:
    if case_result.get("candidate_present") is not True:
      raise SystemExit("latest native candidate gate missing a case")
    if case_result.get("workstream_match") is not True:
      raise SystemExit("latest native candidate gate workstream mismatch")
    if case_result.get("native_output_source_ok") is not True:
      raise SystemExit("latest native candidate gate rejected native output source")
    if case_result.get("prompt_utf8_sha256_match") is not True:
      raise SystemExit("latest native candidate gate prompt hash mismatch")
    if case_result.get("prompt_token_ids_match") is not True:
      raise SystemExit("latest native candidate gate prompt tokens mismatch")
    candidate_target_results = case_result.get("target_results", [])
    if len(candidate_target_results) != 2:
      raise SystemExit("latest native candidate gate target count mismatch")
    for target_result in candidate_target_results:
      if target_result.get("generated_token_ids_match") is not True:
        raise SystemExit("latest native candidate gate token replay mismatch")
      if target_result.get("top1_id_match") is not True:
        raise SystemExit("latest native candidate gate top1 mismatch")

  post_r1_dirs = sorted((ROOT / "output").glob("post-r1-resident-timed-*"))
  if not post_r1_dirs:
    raise SystemExit("missing post-R1 resident/timed diagnostic artifact")
  post_r1_dir = None
  post_r1_correctness = {}
  for possible_post_r1_dir in reversed(post_r1_dirs):
    possible_correctness_path = possible_post_r1_dir / "correctness.json"
    if not possible_correctness_path.exists():
      continue
    possible_correctness = json.loads(
        possible_correctness_path.read_text(encoding="utf-8")
    )
    if (
        possible_correctness.get("required_checks_passed") is True
        and possible_correctness.get("r1_native_correctness_gate_closed") is True
    ):
      if possible_correctness.get("shared_expert_gate_up_fused") is True:
        continue
      if possible_correctness.get("selected_expert_down_expert_major") is True:
        continue
      if possible_correctness.get("selected_expert_down_q4_pair_dot") is True:
        continue
      if possible_correctness.get("selected_expert_down_q6_pair_dot") is True:
        continue
      if possible_correctness.get("q4_direct_minsum_pair") is True:
        continue
      if possible_correctness.get("q4_block_meta_cache") is True:
        continue
      if possible_correctness.get("small_q4_direct_dot") is True:
        continue
      if possible_correctness.get("matvec_q8_input_reuse") is True:
        continue
      if possible_correctness.get("decode_top1_only") is True:
        continue
      if possible_correctness.get("lm_head_q6_pair_dot") is True:
        continue
      possible_diagnostic_path = possible_post_r1_dir / "diagnostic.json"
      if not possible_diagnostic_path.exists():
        continue
      possible_diagnostic = load_json_path(possible_diagnostic_path)
      possible_benchmark_metadata = possible_diagnostic.get("benchmark_metadata", {})
      if possible_benchmark_metadata.get("dense_matvec_threads_requested") != 16:
        continue
      if possible_benchmark_metadata.get("dense_matvec_threads") != 16:
        continue
      if possible_benchmark_metadata.get("dense_matvec_min_rows_requested") != 256:
        continue
      if possible_benchmark_metadata.get("dense_matvec_min_rows") != 256:
        continue
      post_r1_dir = possible_post_r1_dir
      post_r1_correctness = possible_correctness
      break
  if post_r1_dir is None:
    raise SystemExit("missing passing post-R1 resident/timed diagnostic artifact")
  if post_r1_correctness.get("gate") != "post_r1_resident_timed_diagnostic":
    raise SystemExit("latest passing post-R1 resident/timed gate mismatch")
  if post_r1_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest passing post-R1 resident/timed checks must pass")
  if post_r1_correctness.get("r1_native_correctness_gate_closed") is not True:
    raise SystemExit("latest passing post-R1 artifact must preserve closed R1 gate")
  if post_r1_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest passing post-R1 artifact must not allow speedup claims")
  if post_r1_correctness.get("dense_q6_pair_dot") is not True:
    raise SystemExit("latest passing post-R1 artifact must enable dense Q6 pair dot")
  post_r1_diagnostic = json.loads(
      (post_r1_dir / "diagnostic.json").read_text(encoding="utf-8")
  )
  post_r1_state = post_r1_diagnostic.get("diagnostic", {})
  if post_r1_state.get("candidate_row_count") != 6:
    raise SystemExit("latest post-R1 diagnostic candidate row count mismatch")
  if post_r1_state.get("timed_case_row_count") != 6:
    raise SystemExit("latest post-R1 diagnostic timed case count mismatch")
  if not isinstance(post_r1_state.get("warmup_runs"), int) or (
      post_r1_state["warmup_runs"] < 1
  ):
    raise SystemExit("latest post-R1 diagnostic must record at least one warmup")
  thread_sweep_dirs = sorted((ROOT / "output").glob("lm-head-topk-thread-sweep-*"))
  if not thread_sweep_dirs:
    raise SystemExit("missing LM-head top-k thread sweep artifact")
  thread_sweep_dir = thread_sweep_dirs[-1]
  thread_sweep_correctness = json.loads(
      (thread_sweep_dir / "correctness.json").read_text(encoding="utf-8")
  )
  if thread_sweep_correctness.get("gate") != "lm_head_topk_thread_sweep":
    raise SystemExit("latest LM-head top-k thread sweep gate mismatch")
  if thread_sweep_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest LM-head top-k thread sweep checks must pass")
  if thread_sweep_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest LM-head top-k thread sweep must forbid speedup claims")
  thread_sweep_diagnostic = json.loads(
      (thread_sweep_dir / "diagnostic.json").read_text(encoding="utf-8")
  )
  if thread_sweep_diagnostic.get("thread_counts") != [1, 2, 4, 8, 16]:
    raise SystemExit("latest LM-head top-k thread sweep thread counts mismatch")
  thread_sweep_state = thread_sweep_diagnostic.get("diagnostic", {})
  for field in (
      "all_candidate_checks_passed",
      "all_candidate_runs_executed",
      "all_lm_head_profiles_present",
      "all_routes_enabled",
      "signatures_match",
  ):
    if thread_sweep_state.get(field) is not True:
      raise SystemExit(f"latest LM-head top-k thread sweep {field} must pass")
  selected_gate_sweep_dirs = sorted((ROOT / "output").glob("selected-gate-q4-thread-sweep-*"))
  if not selected_gate_sweep_dirs:
    raise SystemExit("missing selected gate Q4 thread sweep artifact")
  selected_gate_sweep_dir = selected_gate_sweep_dirs[-1]
  selected_gate_sweep_correctness = json.loads(
      (selected_gate_sweep_dir / "correctness.json").read_text(encoding="utf-8")
  )
  if selected_gate_sweep_correctness.get("gate") != "selected_gate_q4_thread_sweep":
    raise SystemExit("latest selected gate Q4 thread sweep gate mismatch")
  if selected_gate_sweep_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest selected gate Q4 thread sweep checks must pass")
  if selected_gate_sweep_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest selected gate Q4 thread sweep must forbid speedup claims")
  selected_gate_sweep_diagnostic = json.loads(
      (selected_gate_sweep_dir / "diagnostic.json").read_text(encoding="utf-8")
  )
  if selected_gate_sweep_diagnostic.get("thread_counts") != [1, 2, 4, 8, 16, 32]:
    raise SystemExit("latest selected gate Q4 thread sweep thread counts mismatch")
  selected_gate_sweep_state = selected_gate_sweep_diagnostic.get("diagnostic", {})
  for field in (
      "all_candidate_checks_passed",
      "all_candidate_runs_executed",
      "all_q4direct_profiles_present",
      "all_routes_enabled",
      "signatures_match",
  ):
    if selected_gate_sweep_state.get(field) is not True:
      raise SystemExit(f"latest selected gate Q4 thread sweep {field} must pass")
  selected_gate_results = selected_gate_sweep_diagnostic.get("results", [])
  if not isinstance(selected_gate_results, list) or len(selected_gate_results) != 6:
    raise SystemExit("latest selected gate Q4 thread sweep result count mismatch")
  for result in selected_gate_results:
    if result.get("selected_gate_q4_direct_dot_enabled") is not True:
      raise SystemExit("latest selected gate Q4 thread sweep route disabled")
    profile = result.get("q4direct_gate_profile", {})
    if (
        not isinstance(profile, dict)
        or profile.get("profile_row_count") != 40
        or not isinstance(profile.get("total_ns"), int)
        or profile["total_ns"] <= 0
    ):
      raise SystemExit("latest selected gate Q4 thread sweep profile missing")
  benchmark_metadata = post_r1_diagnostic.get("benchmark_metadata", {})
  if benchmark_metadata.get("cache_state") != "single_process_hot_after_internal_warmup":
    raise SystemExit("latest post-R1 diagnostic cache state mismatch")
  if benchmark_metadata.get("resident_cache_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request resident cache")
  if benchmark_metadata.get("resident_tensor_cache_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic resident cache must be enabled")
  if benchmark_metadata.get("matvec_profile_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request matvec profile")
  if benchmark_metadata.get("matvec_profile_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic matvec profile must be enabled")
  if benchmark_metadata.get("prefill_final_logits_only_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request prefill final logits only")
  if benchmark_metadata.get("prefill_final_logits_only_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic prefill final logits only must be enabled")
  if benchmark_metadata.get("full_attention_inplace_history_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request full-attention inplace history")
  if benchmark_metadata.get("full_attention_inplace_history_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic full-attention inplace history must be enabled")
  if benchmark_metadata.get("decode_top1_only_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request rejected decode top1-only route")
  if benchmark_metadata.get("decode_top1_only_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic rejected decode top1-only route must be disabled")
  if benchmark_metadata.get("dense_matvec_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request dense matvec route")
  if benchmark_metadata.get("dense_matvec_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic dense matvec route must be enabled")
  if benchmark_metadata.get("dense_matvec_threads_requested") != 16:
    raise SystemExit("latest post-R1 diagnostic dense requested threads mismatch")
  if benchmark_metadata.get("dense_matvec_threads") != 16:
    raise SystemExit("latest post-R1 diagnostic dense actual threads mismatch")
  if benchmark_metadata.get("dense_matvec_min_rows_requested") != 256:
    raise SystemExit("latest post-R1 diagnostic dense requested min rows mismatch")
  if benchmark_metadata.get("dense_matvec_min_rows") != 256:
    raise SystemExit("latest post-R1 diagnostic dense actual min rows mismatch")
  if benchmark_metadata.get("dense_matvec_payload_cache_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request dense payload cache")
  if benchmark_metadata.get("dense_matvec_payload_cache_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic dense payload cache must be enabled")
  if benchmark_metadata.get("dense_q4_direct_dot_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request dense Q4 direct dot")
  if benchmark_metadata.get("dense_q4_direct_dot_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic dense Q4 direct dot must be enabled")
  if benchmark_metadata.get("dense_q4_pair_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request rejected dense Q4 pair dot")
  if benchmark_metadata.get("dense_q4_pair_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic rejected dense Q4 pair dot must be disabled")
  if benchmark_metadata.get("dense_q6_direct_dot_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request dense Q6 direct dot")
  if benchmark_metadata.get("dense_q6_direct_dot_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic dense Q6 direct dot must be enabled")
  if benchmark_metadata.get("dense_q6_pair_dot_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request dense Q6 pair dot")
  if benchmark_metadata.get("dense_q6_pair_dot_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic dense Q6 pair dot must be enabled")
  if benchmark_metadata.get("q4_direct_minsum_pair_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental Q4 direct min-sum pair")
  if benchmark_metadata.get("q4_direct_minsum_pair_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental Q4 direct min-sum pair must be disabled")
  if benchmark_metadata.get("q4_block_meta_cache_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental Q4 block meta cache")
  if benchmark_metadata.get("q4_block_meta_cache_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental Q4 block meta cache must be disabled")
  if benchmark_metadata.get("small_q4_direct_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental small Q4 direct dot")
  if benchmark_metadata.get("small_q4_direct_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental small Q4 direct dot must be disabled")
  if benchmark_metadata.get("matvec_q8_input_reuse_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental Q8 input reuse")
  if benchmark_metadata.get("matvec_q8_input_reuse_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental Q8 input reuse must be disabled")
  if benchmark_metadata.get("lm_head_top_k_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request LM-head top-k route")
  if benchmark_metadata.get("lm_head_top_k_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic LM-head top-k route must be enabled")
  if benchmark_metadata.get("lm_head_q6_pair_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental LM-head Q6 pair dot")
  if benchmark_metadata.get("lm_head_q6_pair_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental LM-head Q6 pair dot must be disabled")
  if benchmark_metadata.get("lm_head_threads_requested") != 16:
    raise SystemExit("latest post-R1 diagnostic LM-head requested threads mismatch")
  if benchmark_metadata.get("lm_head_threads") != 16:
    raise SystemExit("latest post-R1 diagnostic LM-head actual threads mismatch")
  if benchmark_metadata.get("expert_slice_matvec_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request expert-slice route")
  if benchmark_metadata.get("expert_slice_matvec_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic expert-slice route must be enabled")
  if benchmark_metadata.get("expert_slice_threads_requested") != 16:
    raise SystemExit("latest post-R1 diagnostic expert-slice requested threads mismatch")
  if benchmark_metadata.get("expert_slice_threads") != 16:
    raise SystemExit("latest post-R1 diagnostic expert-slice actual threads mismatch")
  if benchmark_metadata.get("shared_parallel_executor_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request shared parallel executor")
  if benchmark_metadata.get("shared_parallel_executor_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic shared parallel executor must be enabled")
  if benchmark_metadata.get("shared_expert_gate_up_fused_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request rejected shared expert fused route")
  if benchmark_metadata.get("shared_expert_gate_up_fused_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic rejected shared expert fused route must be disabled")
  if benchmark_metadata.get("selected_expert_ffn_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request selected-expert FFN route")
  if benchmark_metadata.get("selected_expert_ffn_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic selected-expert FFN route must be enabled")
  if benchmark_metadata.get("selected_expert_ffn_threads_requested") != 16:
    raise SystemExit("latest post-R1 diagnostic selected-expert FFN requested threads mismatch")
  if benchmark_metadata.get("selected_expert_ffn_threads") != 16:
    raise SystemExit("latest post-R1 diagnostic selected-expert FFN actual threads mismatch")
  if benchmark_metadata.get("selected_expert_minimal_outputs_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request rejected selected-expert minimal outputs")
  if benchmark_metadata.get("selected_expert_minimal_outputs_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic rejected selected-expert minimal outputs must be disabled")
  if benchmark_metadata.get("selected_expert_slice_cache_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request selected expert slice cache")
  if benchmark_metadata.get("selected_expert_slice_cache_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic selected expert slice cache must be enabled")
  if benchmark_metadata.get("selected_expert_down_slice_cache_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request selected expert down slice cache")
  if benchmark_metadata.get("selected_expert_down_slice_cache_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic selected expert down slice cache must be enabled")
  if benchmark_metadata.get("selected_expert_down_expert_major_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental selected down expert-major")
  if benchmark_metadata.get("selected_expert_down_expert_major_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental selected down expert-major must be disabled")
  if benchmark_metadata.get("selected_expert_down_q4_pair_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental selected down Q4 pair dot")
  if benchmark_metadata.get("selected_expert_down_q4_pair_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental selected down Q4 pair dot must be disabled")
  if benchmark_metadata.get("selected_expert_down_q6_pair_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request experimental selected down Q6 pair dot")
  if benchmark_metadata.get("selected_expert_down_q6_pair_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic experimental selected down Q6 pair dot must be disabled")
  if benchmark_metadata.get("selected_gate_q4_direct_dot_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request selected gate Q4 direct dot")
  if benchmark_metadata.get("selected_gate_q4_direct_dot_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic selected gate Q4 direct dot must be enabled")
  if benchmark_metadata.get("selected_gate_q4_pair_dot_requested") is not True:
    raise SystemExit("latest post-R1 diagnostic must request selected gate Q4 pair dot")
  if benchmark_metadata.get("selected_gate_q4_pair_dot_enabled") is not True:
    raise SystemExit("latest post-R1 diagnostic selected gate Q4 pair dot must be enabled")
  if benchmark_metadata.get("selected_gate_q4_pair_sum_dot_requested") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic must not request rejected selected gate Q4 pair-sum dot")
  if benchmark_metadata.get("selected_gate_q4_pair_sum_dot_enabled") not in (False, None):
    raise SystemExit("latest post-R1 diagnostic rejected selected gate Q4 pair-sum dot must be disabled")
  matvec_profile_top = benchmark_metadata.get("matvec_profile_top", [])
  if not isinstance(matvec_profile_top, list) or not matvec_profile_top:
    raise SystemExit("latest post-R1 diagnostic matvec profile missing")
  top_matvec = matvec_profile_top[0]
  if (
      top_matvec.get("op") != "top_k_matvec_tensor"
      or top_matvec.get("tensor_name") != "output.weight"
  ):
    raise SystemExit("latest post-R1 diagnostic top LM-head profile mismatch")
  for field in ("call_count", "total_ns", "average_ns", "row_count"):
    if not isinstance(top_matvec.get(field), int) or top_matvec[field] <= 0:
      raise SystemExit(f"latest post-R1 diagnostic top matvec {field} missing")
  if top_matvec.get("call_count") != 182:
    raise SystemExit("latest post-R1 diagnostic must skip non-final prefill LM-head calls")
  post_r1_candidate_stdout = json.loads(
      (post_r1_dir / "native-candidate-jsonl" / "native-candidate-stdout.json").read_text(
          encoding="utf-8"
      )
  )
  post_r1_full_profile = post_r1_candidate_stdout.get("matvec_profile", [])
  if not isinstance(post_r1_full_profile, list):
    raise SystemExit("latest post-R1 diagnostic full matvec profile missing")
  profile_ops = {
      row.get("op")
      for row in post_r1_full_profile
      if isinstance(row, dict)
  }
  if "selected_expert_ffn_gate_swiglu_q4pair" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing selected-expert Q4 pair gate/swiglu profile")
  if "selected_expert_ffn_gate_up_read" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing selected-expert gate/up read profile")
  if "selected_expert_ffn_gate_swiglu_q4pairsum" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use rejected selected-expert Q4 pair-sum profile")
  if "selected_expert_ffn_down_aggregate" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing selected-expert down aggregate profile")
  if "selected_expert_ffn_down_expert_major" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental selected down expert-major profile")
  if "selected_expert_ffn_down_q4pair" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental selected down Q4 pair profile")
  if "selected_expert_ffn_down_q6pair" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental selected down Q6 pair profile")
  if "top_k_matvec_tensor_q6pair" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental LM-head Q6 pair profile")
  if "matvec_tensor_dense_q4directmeta" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental Q4 direct metadata profile")
  if "matvec_tensor_small_q4direct" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental small Q4 direct profile")
  selected_down_total_ns = sum(
      row.get("total_ns", 0)
      for row in post_r1_full_profile
      if isinstance(row, dict)
      and row.get("op") == "selected_expert_ffn_down_aggregate"
  )
  if (
      not isinstance(selected_down_total_ns, int)
      or selected_down_total_ns <= 0
      or selected_down_total_ns > 13500000000
  ):
    raise SystemExit(
        "latest post-R1 diagnostic selected-expert down direct-dot profile missing"
    )
  if "matvec_tensor_dense" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing dense matvec profile")
  if "matvec_tensor_dense_q4direct" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing dense Q4 direct matvec profile")
  if "matvec_tensor_dense_q4directminpair" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use experimental dense Q4 min-sum pair profile")
  if "matvec_tensor_dense_q4pair" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic must not use rejected dense Q4 pair profile")
  if "matvec_tensor_dense_q6pair" not in profile_ops:
    raise SystemExit("latest post-R1 diagnostic missing dense Q6 pair matvec profile")
  if "matvec_tensor_dense_q6direct" in profile_ops:
    raise SystemExit("latest post-R1 diagnostic should route dense Q6 rows through pair-dot")
  resident_cache_stats = benchmark_metadata.get("resident_tensor_cache_stats", {})
  if not isinstance(resident_cache_stats, dict):
    raise SystemExit("latest post-R1 diagnostic resident cache stats missing")
  for field in (
      "decoded_row_hits",
      "decoded_row_misses",
      "tensor_payload_hits",
      "tensor_payload_misses",
      "tensor_payload_cached_bytes",
      "expert_slice_hits",
      "expert_slice_misses",
      "expert_slice_cached_bytes",
  ):
    if (
        not isinstance(resident_cache_stats.get(field), int)
        or resident_cache_stats[field] <= 0
    ):
      raise SystemExit(f"latest post-R1 diagnostic resident cache {field} missing")
  if resident_cache_stats.get("tensor_payload_cached_bytes", 0) <= 1347000000:
    raise SystemExit(
        "latest post-R1 diagnostic dense/shared-FFN/generic small-weight payload cache bytes missing"
    )
  if resident_cache_stats.get("expert_slice_cached_bytes", 0) <= 12000000000:
    raise SystemExit("latest post-R1 diagnostic selected down slice cache bytes missing")
  if benchmark_metadata.get("model_path") != model.get("gguf_model_path"):
    raise SystemExit("latest post-R1 diagnostic model path mismatch")
  prompt_counts = benchmark_metadata.get("prompt_token_counts", {})
  generated_counts = benchmark_metadata.get("generated_token_counts", {})
  if len(prompt_counts) != 6 or len(generated_counts) != 6:
    raise SystemExit("latest post-R1 diagnostic token count metadata incomplete")
  timed_rows = post_r1_diagnostic.get("timed_case_rows", [])
  if len(timed_rows) != 6:
    raise SystemExit("latest post-R1 diagnostic timed rows mismatch")
  for row in timed_rows:
    timing = row.get("timing_ns", {})
    if not isinstance(timing.get("case_total"), int) or timing["case_total"] <= 0:
      raise SystemExit("latest post-R1 diagnostic missing case timing")
    if (
        not isinstance(timing.get("prompt_prefill"), int)
        or timing["prompt_prefill"] <= 0
    ):
      raise SystemExit("latest post-R1 diagnostic missing prompt timing")
    if not isinstance(timing.get("decode_continuation"), int):
      raise SystemExit("latest post-R1 diagnostic missing decode timing")

  q6_pair_post_r1_dir = None
  q6_pair_post_r1_correctness = {}
  for possible_post_r1_dir in reversed(post_r1_dirs):
    possible_correctness_path = possible_post_r1_dir / "correctness.json"
    if not possible_correctness_path.exists():
      continue
    possible_correctness = json.loads(
        possible_correctness_path.read_text(encoding="utf-8")
    )
    if (
        possible_correctness.get("required_checks_passed") is True
        and possible_correctness.get("r1_native_correctness_gate_closed") is True
        and possible_correctness.get("dense_q6_pair_dot") is True
        and possible_correctness.get("selected_expert_down_expert_major") is not True
        and possible_correctness.get("selected_expert_down_q4_pair_dot") is not True
        and possible_correctness.get("selected_expert_down_q6_pair_dot") is not True
        and possible_correctness.get("lm_head_q6_pair_dot") is not True
        and possible_correctness.get("q4_block_meta_cache") is not True
        and possible_correctness.get("small_q4_direct_dot") is not True
    ):
      possible_diagnostic_path = possible_post_r1_dir / "diagnostic.json"
      if not possible_diagnostic_path.exists():
        continue
      possible_diagnostic = load_json_path(possible_diagnostic_path)
      possible_benchmark_metadata = possible_diagnostic.get("benchmark_metadata", {})
      if possible_benchmark_metadata.get("dense_matvec_threads_requested") != 16:
        continue
      if possible_benchmark_metadata.get("dense_matvec_threads") != 16:
        continue
      if possible_benchmark_metadata.get("dense_matvec_min_rows_requested") != 256:
        continue
      if possible_benchmark_metadata.get("dense_matvec_min_rows") != 256:
        continue
      q6_pair_post_r1_dir = possible_post_r1_dir
      q6_pair_post_r1_correctness = possible_correctness
      break
  if q6_pair_post_r1_dir is None:
    raise SystemExit("missing dense Q6 pair-dot post-R1 diagnostic artifact")
  if q6_pair_post_r1_dir.relative_to(ROOT).as_posix() != (
      "output/post-r1-resident-timed-20260628T054920Z"
  ):
    raise SystemExit("latest dense Q6 pair-dot diagnostic artifact mismatch")
  if q6_pair_post_r1_correctness.get("gate") != "post_r1_resident_timed_diagnostic":
    raise SystemExit("dense Q6 pair-dot post-R1 gate mismatch")
  if q6_pair_post_r1_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("dense Q6 pair-dot diagnostic must forbid speedup claims")
  q6_pair_diagnostic = json.loads(
      (q6_pair_post_r1_dir / "diagnostic.json").read_text(encoding="utf-8")
  )
  q6_pair_state = q6_pair_diagnostic.get("diagnostic", {})
  if q6_pair_state.get("candidate_row_count") != 6:
    raise SystemExit("dense Q6 pair-dot diagnostic candidate row count mismatch")
  if q6_pair_state.get("timed_case_row_count") != 6:
    raise SystemExit("dense Q6 pair-dot diagnostic timed case count mismatch")
  if q6_pair_state.get("warmup_runs") != 1 or q6_pair_state.get("timed_runs") != 1:
    raise SystemExit("dense Q6 pair-dot diagnostic warmup/timed run mismatch")
  q6_pair_benchmark = q6_pair_diagnostic.get("benchmark_metadata", {})
  for field in (
      "prefill_final_logits_only",
      "full_attention_inplace_history",
      "dense_matvec",
      "dense_matvec_payload_cache",
      "dense_q4_direct_dot",
      "dense_q6_direct_dot",
      "dense_q6_pair_dot",
      "lm_head_top_k",
      "expert_slice_matvec",
      "shared_parallel_executor",
      "selected_expert_ffn",
      "selected_expert_slice_cache",
      "selected_expert_down_slice_cache",
      "selected_gate_q4_direct_dot",
      "selected_gate_q4_pair_dot",
  ):
    if q6_pair_benchmark.get(f"{field}_requested") is not True:
      raise SystemExit(f"dense Q6 pair-dot diagnostic did not request {field}")
    if q6_pair_benchmark.get(f"{field}_enabled") is not True:
      raise SystemExit(f"dense Q6 pair-dot diagnostic did not enable {field}")
  q6_pair_stdout = json.loads(
      (
          q6_pair_post_r1_dir
          / "native-candidate-jsonl"
          / "native-candidate-stdout.json"
      ).read_text(encoding="utf-8")
  )
  if q6_pair_stdout.get("dense_q6_pair_dot_enabled") is not True:
    raise SystemExit("dense Q6 pair-dot native stdout flag missing")
  q6_pair_timed_runs = q6_pair_stdout.get("timed_runs", [])
  if (
      not isinstance(q6_pair_timed_runs, list)
      or len(q6_pair_timed_runs) != 1
      or q6_pair_timed_runs[0].get("total_ns") != 51328448188
  ):
    raise SystemExit("dense Q6 pair-dot timed total mismatch")
  q6_pair_profile = q6_pair_stdout.get("matvec_profile", [])
  if not isinstance(q6_pair_profile, list):
    raise SystemExit("dense Q6 pair-dot full matvec profile missing")
  q6_pair_ops = {
      row.get("op")
      for row in q6_pair_profile
      if isinstance(row, dict)
  }
  if "matvec_tensor_dense_q6pair" not in q6_pair_ops:
    raise SystemExit("dense Q6 pair-dot profile op missing")
  if "matvec_tensor_dense_q6direct" in q6_pair_ops:
    raise SystemExit("dense Q6 pair-dot route should replace dense Q6 direct profile rows")
  q6_pair_profile_totals = {
      "dense_q6_pair": sum(
          row.get("total_ns", 0)
          for row in q6_pair_profile
          if isinstance(row, dict)
          and row.get("op") == "matvec_tensor_dense_q6pair"
      ),
      "attn_qkv": sum(
          row.get("total_ns", 0)
          for row in q6_pair_profile
          if isinstance(row, dict)
          and str(row.get("tensor_name", "")).endswith("attn_qkv.weight")
      ),
      "output_topk": sum(
          row.get("total_ns", 0)
          for row in q6_pair_profile
          if isinstance(row, dict)
          and row.get("op") == "top_k_matvec_tensor"
          and row.get("tensor_name") == "output.weight"
      ),
      "selected_down": sum(
          row.get("total_ns", 0)
          for row in q6_pair_profile
          if isinstance(row, dict)
          and row.get("op") == "selected_expert_ffn_down_aggregate"
      ),
  }
  expected_q6_pair_totals = {
      "dense_q6_pair": 9132707322,
      "attn_qkv": 14191688820,
      "output_topk": 6073379838,
      "selected_down": 11747651266,
  }
  if q6_pair_profile_totals != expected_q6_pair_totals:
    raise SystemExit("dense Q6 pair-dot profile totals mismatch")

  context_rollup_dirs = sorted((ROOT / "output").glob("context-ladder-rollup-*"))
  if not context_rollup_dirs:
    raise SystemExit("missing context-ladder rollup artifact")
  expected_context_rollup_artifact_paths = {
      "output/context-ladder-native-diagnostic-20260628T055916Z",
      "output/context-ladder-native-diagnostic-20260628T062127Z",
      "output/context-ladder-native-diagnostic-20260628T091204Z",
      "output/context-ladder-native-diagnostic-20260628T101623Z",
      "output/context-ladder-native-diagnostic-20260628T113254Z",
      "output/context-ladder-native-diagnostic-20260628T154237Z",
  }
  context_rollup_dir = None
  context_rollup_correctness = {}
  context_rollup = {}
  for possible_context_rollup_dir in reversed(context_rollup_dirs):
    context_rollup_correctness_path = possible_context_rollup_dir / "correctness.json"
    context_rollup_path = possible_context_rollup_dir / "rollup.json"
    if not context_rollup_correctness_path.exists() or not context_rollup_path.exists():
      continue
    possible_context_rollup_correctness = json.loads(
        context_rollup_correctness_path.read_text(encoding="utf-8")
    )
    possible_context_rollup = json.loads(
        context_rollup_path.read_text(encoding="utf-8")
    )
    possible_artifacts = possible_context_rollup.get("rollup", {}).get("artifacts", [])
    possible_artifact_paths = {
        artifact.get("artifact")
        for artifact in possible_artifacts
        if isinstance(artifact, dict)
    }
    if (
        possible_context_rollup_correctness.get("required_checks_passed") is True
        and possible_artifact_paths == expected_context_rollup_artifact_paths
    ):
      context_rollup_dir = possible_context_rollup_dir
      context_rollup_correctness = possible_context_rollup_correctness
      context_rollup = possible_context_rollup
      break
  if context_rollup_dir is None:
    raise SystemExit("missing accepted dense Q6 pair context-ladder rollup artifact")
  if context_rollup_correctness.get("schema_version") != "intel-qwen36-context-ladder-rollup-v0":
    raise SystemExit("latest context-ladder rollup schema mismatch")
  if context_rollup_correctness.get("gate") != "context_ladder_rollup":
    raise SystemExit("latest context-ladder rollup gate mismatch")
  if context_rollup_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest context-ladder rollup checks must pass")
  if context_rollup_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest context-ladder rollup must forbid speedup claims")
  rollup_checks = {
      check.get("name"): check
      for check in context_rollup_correctness.get("checks", [])
      if isinstance(check, dict)
  }
  for check_name in (
      "all_artifacts_required_checks_passed",
      "all_artifacts_disable_speedup_claims",
      "all_artifacts_are_cold_no_prefix",
      "all_artifacts_use_case_process_isolation",
      "route_consistent",
      "dense_q6_pair_dot_policy_consistent",
      "max_new_tokens_consistent",
      "resident_cache_policy_consistent",
      "no_duplicate_case_ids",
      "all_rows_have_positive_prefill",
      "prefill_shape_prefill_monotonic",
      "sentinel_retrieval_prefill_monotonic",
  ):
    if rollup_checks.get(check_name, {}).get("pass") is not True:
      raise SystemExit(f"latest context-ladder rollup check failed: {check_name}")
  expected_counts = [1024, 2048, 4096, 8192, 16384]
  expected_prefill = {
      "prefill_shape_prefill_monotonic": [
          202962111367,
          442181798231,
          1120719829575,
          3810905856705,
          14907565332894,
      ],
      "sentinel_retrieval_prefill_monotonic": [
          203502134709,
          442833507009,
          1119225404548,
          3788155089592,
          14914021571734,
      ],
  }
  for check_name, expected_ns in expected_prefill.items():
    check = rollup_checks[check_name]
    if check.get("prompt_token_counts") != expected_counts:
      raise SystemExit(f"latest context-ladder rollup {check_name} prompt counts mismatch")
    if check.get("prompt_prefill_ns") != expected_ns:
      raise SystemExit(f"latest context-ladder rollup {check_name} timing mismatch")
  if context_rollup.get("schema_version") != "intel-qwen36-context-ladder-rollup-v0":
    raise SystemExit("latest context-ladder rollup payload schema mismatch")
  if context_rollup.get("required_checks_passed") is not True:
    raise SystemExit("latest context-ladder rollup payload checks must pass")
  if context_rollup.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest context-ladder rollup payload must forbid speedup claims")
  rollup = context_rollup.get("rollup", {})
  artifacts = rollup.get("artifacts", [])
  if len(artifacts) != 6:
    raise SystemExit("latest context-ladder rollup artifact count mismatch")
  artifact_paths = {artifact.get("artifact") for artifact in artifacts}
  if artifact_paths != expected_context_rollup_artifact_paths:
    raise SystemExit("latest context-ladder rollup source artifacts mismatch")
  series = rollup.get("series", [])
  if len(series) != 10:
    raise SystemExit("latest context-ladder rollup series count mismatch")
  series_by_case = {row.get("case_id"): row for row in series if isinstance(row, dict)}
  expected_cases = {
      "prefill_shape_001k": ("prefill_shape", 1024, 202962111367, 264),
      "prefill_shape_002k": ("prefill_shape", 2048, 442181798231, 264),
      "prefill_shape_004k": ("prefill_shape", 4096, 1120719829575, 264),
      "prefill_shape_008k": ("prefill_shape", 8192, 3810905856705, 264),
      "prefill_shape_016k": ("prefill_shape", 16384, 14907565332894, 264),
      "sentinel_001k": ("sentinel_retrieval", 1024, 203502134709, 271),
      "sentinel_002k": ("sentinel_retrieval", 2048, 442833507009, 271),
      "sentinel_004k": ("sentinel_retrieval", 4096, 1119225404548, 271),
      "sentinel_008k": ("sentinel_retrieval", 8192, 3788155089592, 271),
      "sentinel_016k": ("sentinel_retrieval", 16384, 14914021571734, 271),
  }
  if set(series_by_case) != set(expected_cases):
    raise SystemExit("latest context-ladder rollup cases mismatch")
  for case_id, (kind, prompt_count, prefill_ns, first_token_id) in expected_cases.items():
    row = series_by_case[case_id]
    if row.get("kind") != kind:
      raise SystemExit(f"latest context-ladder rollup {case_id} kind mismatch")
    if row.get("prompt_token_count") != prompt_count:
      raise SystemExit(f"latest context-ladder rollup {case_id} prompt count mismatch")
    if row.get("prompt_prefill_ns") != prefill_ns:
      raise SystemExit(f"latest context-ladder rollup {case_id} prefill timing mismatch")
    if row.get("first_generated_token_id") != first_token_id:
      raise SystemExit(f"latest context-ladder rollup {case_id} first token mismatch")

  long_run_policy_dirs = sorted((ROOT / "output").glob("context-ladder-long-run-policy-*"))
  if not long_run_policy_dirs:
    raise SystemExit("missing context-ladder long-run policy artifact")
  accepted_rollup_rel = context_rollup_dir.relative_to(ROOT).as_posix()
  long_run_policy_dir = None
  long_run_policy_correctness = {}
  long_run_policy_payload = {}
  for possible_long_run_policy_dir in reversed(long_run_policy_dirs):
    possible_correctness_path = possible_long_run_policy_dir / "correctness.json"
    possible_policy_path = possible_long_run_policy_dir / "policy.json"
    if not possible_correctness_path.exists() or not possible_policy_path.exists():
      continue
    possible_correctness = json.loads(
        possible_correctness_path.read_text(encoding="utf-8")
    )
    possible_payload = json.loads(possible_policy_path.read_text(encoding="utf-8"))
    possible_policy = possible_payload.get("policy", {})
    if (
        possible_correctness.get("required_checks_passed") is True
        and possible_payload.get("required_checks_passed") is True
        and possible_policy.get("accepted_rollup") == accepted_rollup_rel
    ):
      long_run_policy_dir = possible_long_run_policy_dir
      long_run_policy_correctness = possible_correctness
      long_run_policy_payload = possible_payload
      break
  if long_run_policy_dir is None:
    raise SystemExit("missing accepted context-ladder long-run policy artifact")
  if (
      long_run_policy_payload.get("schema_version")
      != "intel-qwen36-context-ladder-long-run-policy-v0"
  ):
    raise SystemExit("latest context-ladder long-run policy schema mismatch")
  if long_run_policy_payload.get("required_checks_passed") is not True:
    raise SystemExit("latest context-ladder long-run policy checks must pass")
  if long_run_policy_payload.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest context-ladder long-run policy must forbid speedup claims")
  if (
      long_run_policy_correctness.get("schema_version")
      != "intel-qwen36-context-ladder-long-run-policy-v0"
  ):
    raise SystemExit("latest context-ladder long-run policy correctness schema mismatch")
  if long_run_policy_correctness.get("gate") != "context_ladder_long_run_policy":
    raise SystemExit("latest context-ladder long-run policy gate mismatch")
  if long_run_policy_correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest context-ladder long-run policy correctness failed")
  if long_run_policy_correctness.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest context-ladder long-run policy correctness must forbid claims")
  long_run_policy_checks = {
      check.get("name"): check
      for check in long_run_policy_correctness.get("checks", [])
      if isinstance(check, dict)
  }
  for check_name in (
      "accepted_rollup_checks_passed",
      "accepted_rollup_forbids_speedup_claims",
      "accepted_rollup_route_is_q6_pair",
      "accepted_rollup_is_cold_no_prefix",
      "accepted_rollup_uses_case_process_isolation",
      "accepted_rollup_has_observed_counts_for_both_kinds",
      "next_bucket_is_policy_bucket_only",
      "next_jobs_are_single_case_isolated",
      "next_jobs_keep_q6_pair_route",
      "timeout_has_projection_margin",
      "higher_buckets_deferred_until_policy_bucket_rollup",
  ):
    if long_run_policy_checks.get(check_name, {}).get("pass") is not True:
      raise SystemExit(f"latest context-ladder long-run policy check failed: {check_name}")
  policy = long_run_policy_payload.get("policy", {})
  if policy.get("policy_id") != "q6_pair_post_16k_32k_explicit_long_jobs_v0":
    raise SystemExit("latest context-ladder long-run policy id mismatch")
  if policy.get("decision") != "run_032k_next_as_explicit_isolated_long_jobs_only":
    raise SystemExit("latest context-ladder long-run policy decision mismatch")
  if policy.get("run_policy_gate_closed") is not True:
    raise SystemExit("latest context-ladder long-run policy gate must be closed")
  if policy.get("full_context_ladder_claim_allowed") is not False:
    raise SystemExit("latest context-ladder long-run policy must not allow full ladder claim")
  if policy.get("speedup_claims_allowed") is not False:
    raise SystemExit("latest context-ladder long-run policy payload must forbid speedup claims")
  if policy.get("prefix_cache_enabled") is not False:
    raise SystemExit("latest context-ladder long-run policy must keep prefix cache disabled")
  if policy.get("route") != "post_r1_20260628T054920Z_dense_q6_pair_dot_flags":
    raise SystemExit("latest context-ladder long-run policy route mismatch")
  if policy.get("accepted_rollup") != accepted_rollup_rel:
    raise SystemExit("latest context-ladder long-run policy accepted rollup mismatch")
  if policy.get("deferred_buckets") != [65536, 102400, 131072, 262144]:
    raise SystemExit("latest context-ladder long-run policy deferred buckets mismatch")
  if policy.get("stop_before") != "064k_or_larger_until_032k_rollup_is_accepted":
    raise SystemExit("latest context-ladder long-run policy stop-before mismatch")
  next_bucket = policy.get("next_bucket", {})
  if next_bucket.get("label") != "032k" or next_bucket.get("tokens") != 32768:
    raise SystemExit("latest context-ladder long-run policy next bucket mismatch")
  if next_bucket.get("timeout_s") != 90000:
    raise SystemExit("latest context-ladder long-run policy timeout mismatch")
  expected_jobs = {
      "sentinel_032k": (
          "sentinel_retrieval",
          "python3 tools/intel-qwen36-context-ladder-native-diagnostic.py "
          "--case-id sentinel_032k --dense-q6-pair-dot --timeout-s 90000",
      ),
      "prefill_shape_032k": (
          "prefill_shape",
          "python3 tools/intel-qwen36-context-ladder-native-diagnostic.py "
          "--case-id prefill_shape_032k --dense-q6-pair-dot --timeout-s 90000",
      ),
  }
  jobs = next_bucket.get("jobs", [])
  jobs_by_case = {job.get("case_id"): job for job in jobs if isinstance(job, dict)}
  if set(jobs_by_case) != set(expected_jobs):
    raise SystemExit("latest context-ladder long-run policy job set mismatch")
  for case_id, (kind, command) in expected_jobs.items():
    job = jobs_by_case[case_id]
    if job.get("kind") != kind:
      raise SystemExit(f"latest context-ladder long-run policy {case_id} kind mismatch")
    if job.get("command") != command:
      raise SystemExit(f"latest context-ladder long-run policy {case_id} command mismatch")
    if job.get("process_policy") != "isolated_single_case_target_process":
      raise SystemExit(f"latest context-ladder long-run policy {case_id} isolation mismatch")
  projections = {
      projection.get("case_id"): projection
      for projection in next_bucket.get("projections", [])
      if isinstance(projection, dict)
  }
  expected_projections = {
      "sentinel_032k": ("sentinel_retrieval", 58716719400763),
      "prefill_shape_032k": ("prefill_shape", 58315663653432),
  }
  if set(projections) != set(expected_projections):
    raise SystemExit("latest context-ladder long-run policy projection set mismatch")
  for case_id, (kind, projected_ns) in expected_projections.items():
    projection = projections[case_id]
    if projection.get("kind") != kind:
      raise SystemExit(f"latest context-ladder long-run policy {case_id} projection kind mismatch")
    if projection.get("projectable") is not True:
      raise SystemExit(f"latest context-ladder long-run policy {case_id} must be projectable")
    if projection.get("projected_prefill_ns") != projected_ns:
      raise SystemExit(f"latest context-ladder long-run policy {case_id} projection mismatch")

  resolution_path = latest_resolution.get("path")
  if not isinstance(resolution_path, str) or not resolution_path:
    raise SystemExit("latest denominator/oracle resolution path missing")
  resolution_dir = ROOT / resolution_path
  correctness_path = resolution_dir / "correctness.json"
  resolution_json_path = resolution_dir / "resolution.json"
  if not correctness_path.exists() or not resolution_json_path.exists():
    raise SystemExit("latest denominator/oracle resolution artifact missing")
  correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
  if correctness.get("required_checks_passed") is not True:
    raise SystemExit("latest denominator/oracle resolution artifact checks failed")
  resolution = json.loads(resolution_json_path.read_text(encoding="utf-8"))
  if resolution.get("r0_gate_status", {}).get("r0_closed") is not False:
    raise SystemExit("latest denominator/oracle resolution must keep R0 open")
  if (
      resolution.get("denominator_262144", {}).get("interpretation")
      != "openvino_262144_resource_failure_not_metric"
  ):
    raise SystemExit("latest denominator/oracle resolution interpretation mismatch")
  llama = resolution.get("llama_denominator_262144", {})
  if (
      llama.get("interpretation")
      != "llama_vulkan_262144_timeout_no_metric_cleanup_complete"
  ):
    raise SystemExit("latest denominator/oracle llama interpretation mismatch")
  if llama.get("denominator_metric_available") is not False:
    raise SystemExit("latest denominator/oracle llama row must not claim metric")
  cleanup_path = llama.get("evidence", {}).get("post_timeout_cleanup")
  if not isinstance(cleanup_path, str) or not (ROOT / cleanup_path).exists():
    raise SystemExit("latest denominator/oracle cleanup evidence missing")
  cleanup = json.loads((ROOT / cleanup_path).read_text(encoding="utf-8"))
  if cleanup.get("post_cleanup_status", {}).get("lingering_llama_bench_process") is not False:
    raise SystemExit("latest denominator/oracle cleanup must show no lingering process")
  if resolution.get("oracle_boundary_bundle", {}).get("boundary_count") != len(
      oracle.get("boundary_types", [])
  ):
    raise SystemExit("latest denominator/oracle resolution boundary count mismatch")

  check_doc_discipline()

  print("validate_repo ok")


def check_doc_discipline() -> None:
  """Structural gates that keep the SSOT thin and logs changelog-shaped.

  These exist because distillation that relies on agent discipline alone snaps
  back within hours (STATUS re-bloated to 285 lines and a single day's meta-log
  reached 158KB the same day the docs were distilled). The factory rule is that
  a stop rule must be a harness-enforced gate, not prose. See AGENTS.md.
  """
  active = ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km"

  # 1. STATUS is a current-state board, not a route history. Keep it thin;
  #    rejected-route detail belongs in rejected-routes.json, narration in meta-log/.
  status_path = active / "STATUS.md"
  if not status_path.exists():
    raise SystemExit("STATUS.md missing")
  status_text = status_path.read_text(encoding="utf-8")
  status_lines = status_text.splitlines()
  # 80, not 120: the 120 cap left ~40 lines of headroom that filled with
  # accepted-cut narration within two days. Accepted work now has a machine
  # ledger (accepted-cuts.json); STATUS states the gate + next action only.
  status_max = 80
  if len(status_lines) > status_max:
    raise SystemExit(
        f"STATUS.md is {len(status_lines)} lines (> {status_max}); move "
        "accepted work to accepted-cuts.json, rejected-route detail to "
        "rejected-routes.json, and history to meta-log/"
    )
  # 1a. The line cap is gameable: a single run-on NEXT ACTION paragraph passed it
  #     while enumerating 63 `N/8` per-token results and dozens of dead variants
  #     (2026-07-02). Gate on SUBSTANCE, not lines. STATUS states the gate + next
  #     action; per-token diagnostics and artifact lists live in output/ + meta-log/.
  status_topk = len(re.findall(r"\b\d+/8\b", status_text))
  status_arts = len(re.findall(r"output/[^\s`)\]]+Z", status_text))
  if status_topk > 3 or status_arts > 8:
    raise SystemExit(
        f"STATUS.md is narrating experiments ({status_topk} `N/8` per-token results, "
        f"{status_arts} output/ refs > 8); it is a state board, not a lab notebook. Move "
        "per-token/per-variant detail to meta-log/ + output/ and the JSON ledgers."
    )

  # 1b. current-frontier.md is the Tier-3 POINTER (state + where things live), not
  #     a running notebook. It rode just under the 40KB active-doc cap at 39.9KB
  #     while pasting 100+ per-boundary lines. Enforce pointer-shape: small, few
  #     artifact refs, and zero `N/8` per-token result lines.
  cf_path = active / "current-frontier.md"
  if cf_path.exists():
    cf_text = cf_path.read_text(encoding="utf-8")
    cf_size = cf_path.stat().st_size
    cf_topk = len(re.findall(r"\b\d+/8\b", cf_text))
    cf_arts = len(re.findall(r"output/[^\s`)\]]+Z", cf_text))
    # 5KB/6 refs, not 8KB/12: the old caps admitted ~100 lines of accumulated
    # per-cut narration riding just under the limit. The pointer needs none of
    # it (accepted-cuts.json / frontier.json carry the substance).
    if cf_size > 5 * 1024 or cf_topk > 0 or cf_arts > 6:
      raise SystemExit(
          f"current-frontier.md is a lab notebook, not a Tier-3 pointer "
          f"({cf_size // 1024}KB > 5KB, {cf_arts} output/ refs > 6, {cf_topk} `N/8` result lines); "
          "it must point (state + where things live). Move accepted work to "
          "accepted-cuts.json, per-gate narration to meta-log/ + output/, "
          "archive long narration to doc/frozen/."
      )

  # 1c. The accepted board is a machine ledger, like the rejected board: a new
  #     session must be able to read what is already accepted without scrolling
  #     prose, and STATUS/current-frontier must not regrow the enumeration.
  cuts_path = active / "accepted-cuts.json"
  if not cuts_path.exists():
    raise SystemExit("accepted-cuts.json ledger missing (accepted resident-loop cuts)")
  cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
  if cuts.get("schema_version") != "intel-qwen36-accepted-cuts-v0":
    raise SystemExit("accepted-cuts.json schema_version mismatch")
  if not isinstance(cuts.get("accepted"), list) or not cuts["accepted"]:
    raise SystemExit("accepted-cuts.json must list accepted cuts")

  # 1d. The explore log is stall-census input (frontier-sync counts its lines);
  #     a silently-corrupt line would undercount no-progress runs. Fail loudly.
  explore_log = ROOT / "output/explore-log.jsonl"
  if explore_log.exists():
    for i, line in enumerate(explore_log.read_text(encoding="utf-8").splitlines(), 1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"output/explore-log.jsonl:{i} does not parse: {exc}")
      if not isinstance(row, dict) or not re.fullmatch(r"\d{8}T\d{6}Z", str(row.get("ts", ""))):
        raise SystemExit(f"output/explore-log.jsonl:{i} missing valid ts (stall census input)")

  # 2. The machine-readable rejected-route ledger must exist and parse, so a new
  #    session reads it instead of scrolling STATUS, and does not re-run a closed route.
  ledger_path = active / "rejected-routes.json"
  if not ledger_path.exists():
    raise SystemExit("rejected-routes.json ledger missing")
  ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
  if ledger.get("schema_version") != "intel-qwen36-rejected-routes-v0":
    raise SystemExit("rejected-routes.json schema_version mismatch")
  if not isinstance(ledger.get("rejected"), list) or not ledger["rejected"]:
    raise SystemExit("rejected-routes.json must list rejected routes")

  # 3. Daily notes are a thin changelog. Raw ns profile tables belong in output/
  #    artifacts and the frontier ledger, not hand-copied into the narrative.
  #    Warn (do not fail historical files) above the soft cap.
  soft_cap = 40 * 1024
  hard_cap = 60 * 1024
  meta_log_dir = ROOT / "meta-log"
  if meta_log_dir.is_dir():
    dailies = sorted(meta_log_dir.glob("20*.md"))
    oversized = [
        (p.name, p.stat().st_size) for p in dailies if p.stat().st_size > soft_cap
    ]
    if oversized:
      detail = ", ".join(f"{n} {sz // 1024}KB" for n, sz in oversized)
      print(
          f"WARNING: meta-log over {soft_cap // 1024}KB/day "
          f"(keep changelog thin; ns tables belong in output/): {detail}"
      )
    # The 40KB soft cap only WARNed and was ignored: 2026-06-28 hit 158KB, the fix
    # was gates, 2026-07-01 then hit 176KB. Hard-fail the NEWEST (actively-written)
    # daily above the hard cap so today's bloat is caught live; older files are
    # grandfathered as warn-only (do not rewrite committed history).
    if dailies:
      newest = dailies[-1]
      if newest.stat().st_size > hard_cap:
        raise SystemExit(
            f"meta-log/{newest.name} is {newest.stat().st_size // 1024}KB "
            f"(> {hard_cap // 1024}KB hard cap for the current day); a daily note is a "
            "thin changelog. Reference artifacts by output/ path; do not transcribe "
            "per-token/ns tables into the narrative."
        )

  # 3b. doc/active holds stable conclusions, not a running lab notebook. The
  #     124KB gpu-backend-bringup doc escaped the meta-log gate because that gate
  #     only globbed meta-log/ — bloat simply relocated to the one path the gate
  #     did not watch. Fail oversized active docs (STATUS has its own line gate;
  #     ledgers are JSON). Per-gate narration belongs in meta-log/ + output/.
  active_doc_cap = 40 * 1024
  oversized_active = [
      (p.name, p.stat().st_size)
      for p in sorted(active.glob("*.md"))
      if p.name != "STATUS.md" and p.stat().st_size > active_doc_cap
  ]
  if oversized_active:
    detail = ", ".join(f"{n} {sz // 1024}KB" for n, sz in oversized_active)
    raise SystemExit(
        f"doc/active doc over {active_doc_cap // 1024}KB (stable conclusions, not "
        f"a lab notebook; move per-gate narration to meta-log/ + output/, archive "
        f"long narration to doc/frozen/): {detail}"
    )

  # 4. Routes ledger: pre-registered alternates + direction trigger (ch.3 §3.5).
  #    A route family that accrues >=2 consecutive sub-threshold candidates with
  #    no recorded switch decision is optimize-the-lie micro-tuning; fail so it
  #    cannot continue silently. The escape is a switch_decision: switch routes
  #    (pop active, push the highest-rank parked_route) or record an escalation /
  #    decision covering the family's latest candidate seq.
  routes_path = active / "routes-ledger.json"
  if not routes_path.exists():
    raise SystemExit("routes-ledger.json missing (pre-registered alternates + direction trigger)")
  routes = json.loads(routes_path.read_text(encoding="utf-8"))
  if routes.get("schema_version") != "intel-qwen36-routes-ledger-v0":
    raise SystemExit("routes-ledger.json schema_version mismatch")
  cand_hist = routes.get("candidate_history")
  decisions = routes.get("switch_decisions")
  parked = routes.get("parked_routes")
  if not isinstance(cand_hist, list):
    raise SystemExit("routes-ledger.json must have a candidate_history list")
  if not isinstance(decisions, list):
    raise SystemExit("routes-ledger.json must have a switch_decisions list")
  if not isinstance(parked, list) or not parked:
    raise SystemExit("routes-ledger.json must pre-register parked_routes (2-3 alternates, ch.3 §3.5)")
  fam_count: dict[str, int] = {}
  fam_max_seq: dict[str, int] = {}
  for c in cand_hist:
    if c.get("sub_threshold"):
      fam = c.get("route_family", "?")
      fam_count[fam] = fam_count.get(fam, 0) + 1
      fam_max_seq[fam] = max(fam_max_seq.get(fam, 0), int(c.get("seq", 0)))
  for fam, count in sorted(fam_count.items()):
    if count >= 2:
      covered = any(
          d.get("family") == fam and int(d.get("seq_covered", -1)) >= fam_max_seq[fam]
          for d in decisions
      )
      if not covered:
        raise SystemExit(
            f"routes-ledger direction trigger (ch.3 §3.5): family '{fam}' has "
            f"{count} consecutive sub-threshold candidates (default-off, decode "
            f"gain <3%) with no switch_decision covering seq {fam_max_seq[fam]}. "
            f"Switch routes (pop active, push highest-rank parked_route) or "
            f"record a switch_decision before adding another '{fam}' candidate."
        )

  # 5. Code-volume / build-time is itself a stop trigger (ch.0 §0.3): when tools/
  #    or engine/tests/ grow one-file-per-variant, or a single source file gets
  #    huge, the O(1) structure has been flattened to O(N). Warn (collapse needs
  #    a dedicated on-target-parity session; see parameterized-compare-runner SOP).
  bloat: list[str] = []
  tests_dir = ROOT / "engine/tests"
  if tests_dir.is_dir():
    n_cmp = len(list(tests_dir.glob("*_compare.cpp")))
    if n_cmp > 12:
      bloat.append(f"{n_cmp} engine/tests/*_compare.cpp (> 12): collapse to one boundaries.json-driven runner")
  tools_dir = ROOT / "tools"
  if tools_dir.is_dir():
    n_drv = len(list(tools_dir.glob("*-compare.py")))
    if n_drv > 12:
      bloat.append(f"{n_drv} tools/*-compare.py drivers (> 12): collapse to one parameterized compare runner")
  line_cap = 1500
  for rel in ("engine/src", "engine/tests", "engine/include", "tools"):
    d = ROOT / rel
    if not d.is_dir():
      continue
    for p in sorted(d.rglob("*")):
      if p.is_file() and p.suffix in (".cpp", ".hpp", ".h", ".py"):
        n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        if n > line_cap:
          bloat.append(f"{p.relative_to(ROOT)} is {n} lines (> {line_cap}): split into per-op TUs (ch.2 §2.2)")
  if bloat:
    print("WARNING: code-volume stop trigger (ch.0 §0.3):")
    for b in bloat:
      print(f"  - {b}")

  # 6. Machine Tier-2 (frontier.json) is the layer that lets the changelog stay
  #    thin (ch.3 §3.4). Here it is validated STRUCTURALLY (exists + parses +
  #    schema): the file partly derives from the output/ census, and output/ is
  #    Tier-1 disposable (gitignored), so a byte-exact `--check` would falsely
  #    fail a fresh clone with no output/. Freshness (`frontier-sync --check`) is
  #    a dev-loop discipline, run when output/ is present (see working-discipline
  #    SOP), not a clone-breaking CI gate. The code-volume RATCHET *is* hard (it
  #    counts tools/ files, no output/ dependency — a new per-layer probe fails
  #    the build; point 5 only WARNs and only on *-compare.py, so the GPU
  #    *-probe.py sprawl needs this probe-aware gate). The stall-gate runs for its
  #    reflection output only (soft; non-blocking here).
  def _gate(script: str, *args: str, fail: bool = True) -> int:
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), *args],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
      print(r.stdout.rstrip())
    if fail and r.returncode != 0:
      if r.stderr.strip():
        print(r.stderr.rstrip())
      raise SystemExit(f"{script} gate failed (exit {r.returncode})")
    return r.returncode

  frontier_path = active / "frontier.json"
  if not frontier_path.exists():
    raise SystemExit(
        "frontier.json missing (Tier-2 machine state, ch.3 §3.4); run "
        "tools/intel-qwen36-frontier-sync.py"
    )
  try:
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise SystemExit(f"frontier.json does not parse: {exc}")
  if frontier.get("schema") != "intel-qwen36-frontier-v1":
    raise SystemExit("frontier.json schema mismatch (expected intel-qwen36-frontier-v1)")
  for key in ("active_route", "goal_anchor", "no_progress", "controls"):
    if key not in frontier:
      raise SystemExit(f"frontier.json missing required key '{key}'")
  _gate("intel-qwen36-code-volume-check.py")          # code volume grew -> fail (ratchet)
  # D-controller (ch.3 §3.5): hard stall BLOCKS (exit 1) unless a keyed review is
  # recorded in routes-ledger.json#goal_stall_reviews; soft reflection returns 0
  # (prints only). fail=False here was the last latch that let the tar-pit run.
  _gate("intel-qwen36-stall-gate.py")


if __name__ == "__main__":
  main()
