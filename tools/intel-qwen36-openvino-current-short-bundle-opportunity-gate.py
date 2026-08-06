#!/usr/bin/env python3
"""Admit one current-carrier Q/K plus router-shared source route, with lanes separated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-current-short-bundle-opportunity-gate-v1"
SOURCE_TREE = Path("/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
PATCH = ROOT / "engine/openvino/iq36-current-router-shared-triple.patch"
OLD_PATCH = ROOT / (
    "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
QK_SOURCE = ROOT / "engine/openvino/custom/iq36_qk_rope_layout.cl"
PROFILE = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/metrics.json")
QK_POINT = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-abba-precheck-"
    "20260731Tseq2198-clean/result.json")
QK_FORMAL = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-formal-abba8-"
    "20260731Tseq2202-clean/result.json")
SHARED = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
TRIPLE_128K = ROOT / (
    "output/openvino-exact-attention-triple-cohort-component-"
    "20260724Tseq2146-clean/result.json")
REJECTED = ROOT / f"doc/active/{WS}/rejected-routes.json"
PINNED_SOURCE_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
EXPECTED_FC_SOURCE_SHA256 = (
    "4a32d9c17d84390aef343bd60c992859fc75bc72d2f8ddff3a355c5276ba6020")
EXPECTED_CARRIER_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
CURRENT_BUCKET = 2048
CURRENT_OUTPUT_TOKENS = 130
PREFILL_POINT_FLOOR = 0.995
DECODE_TOTAL_POINT_TARGET = 1.02
FORMAL_PREFILL_LCB_FLOOR = 0.995
FORMAL_DECODE_TOTAL_LCB_TARGET = 1.02


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, text=True, capture_output=True, check=False)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], ROOT).stdout.strip()
  rows = run(["git", "status", "--porcelain"], ROOT).stdout.splitlines()
  try:
    output_rel = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_rel = None
  dirty = [
      row for row in rows
      if output_rel is None or not row[3:].startswith(output_rel)]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def rejected_route(ledger: dict[str, Any], route: str) -> dict[str, Any]:
  matches = [
      row for row in ledger.get("rejected", [])
      if row.get("route") == route]
  if len(matches) != 1:
    return {}
  return matches[0]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      FC_SOURCE, PATCH, OLD_PATCH, PRODUCT_TOOL, QK_SOURCE, PROFILE, QK_POINT,
      QK_FORMAL, SHARED, TRIPLE_128K, REJECTED)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing opportunity inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  start_available = available_memory_bytes()
  if start_available < stop_bytes:
    raise RuntimeError("memory stop tripped before opportunity gate")

  git = git_state(output)
  profile = load_json(PROFILE)
  qk_point = load_json(QK_POINT)
  qk_formal = load_json(QK_FORMAL)
  shared = load_json(SHARED)
  triple = load_json(TRIPLE_128K)
  rejected = load_json(REJECTED)
  shared_rejection = rejected_route(
      rejected, "openvino_router_isolated_shared_triple_v30a")
  long_rejection = rejected_route(
      rejected, "openvino_qk_shared_pr36747_rms_igc2382_32k_v30d")

  source_commit = run(
      ["git", "rev-parse", "HEAD"], SOURCE_TREE).stdout.strip()
  patch_check = run(
      ["git", "apply", "--check", str(PATCH)], SOURCE_TREE)
  patch_reverse = run(
      ["git", "apply", "--reverse", "--check", str(PATCH)], SOURCE_TREE)
  old_patch_check = run(
      ["git", "apply", "--check", str(OLD_PATCH)], SOURCE_TREE)
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")
  patch_text = PATCH.read_text(encoding="utf-8")
  qk_text = QK_SOURCE.read_text(encoding="utf-8")

  qk_phases = qk_formal["phase_inference"]
  qk_prefill_lcb = float(
      qk_phases["prefill_tokens_s"]["lower_confidence_bound_ratio"])
  qk_decode_lcb = float(
      qk_phases["decode_tokens_s"]["lower_confidence_bound_ratio"])
  qk_total_lcb = float(
      qk_phases["total_rate"]["lower_confidence_bound_ratio"])
  control_tail = sum(
      float(qk_point["tails"][name]["p50_ms"])
      for name in ("control-a1", "control-a2")) / 2.0
  qk_tail = sum(
      float(qk_point["tails"][name]["p50_ms"])
      for name in ("qk-b1", "qk-b2")) / 2.0
  qk_tail_saving = control_tail - qk_tail
  shared_saving = float(
      shared["performance"]["incremental_fc_observed_saving_ms"])
  target_bundle_tail = control_tail / DECODE_TOTAL_POINT_TARGET
  required_shared_realization = max(0.0, qk_tail - target_bundle_tail)
  funding_multiple = (
      shared_saving / required_shared_realization
      if required_shared_realization > 0.0 else math.inf)

  triple_context = int(triple["result"]["context_tokens"])
  triple_ucb_saving_per_layer = abs(float(
      triple["performance_inference"]["upper_confidence_bound_ms"]))
  invalid_triple_total = 10 * triple_ucb_saving_per_layer
  invalid_cross_lane_sum = (
      float(shared["performance"]["total_observed_saving_ms"]) +
      invalid_triple_total)
  retained = {
      row["node_type"]: row
      for row in profile["profile_rollup"][
          "ranked_retained_node_types_nonadditive"]}
  current_attention_profile_ms = (
      float(retained["IQ36ExactPhaseDualCohortHotAttentionGQA"][
          "raw_real_time_us_nonadditive"]) / 1000.0)

  current_counts = profile["worker_result_summary"]["executed_type_counts"]
  shared_profile = shared["profile"]["candidate"]
  shared_counts = shared_profile["core_counts"]
  same_lane_shared = (
      shared["source_summary"].get("fixed_cold_capacity") == CURRENT_BUCKET)
  qk_streams = qk_formal.get("runs", {})
  qk_jitter = qk_formal.get("qk_jitter", [])
  end_available = available_memory_bytes()

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("pinned_current_source_is_exact",
            source_commit == PINNED_SOURCE_COMMIT
            and sha256(FC_SOURCE) == EXPECTED_FC_SOURCE_SHA256,
            source_commit=source_commit, source_sha256=sha256(FC_SOURCE)),
      check("seq2204_is_one_safe_current_2k_attribution_row",
            profile.get("required_checks_passed") is True
            and profile.get("bucket") == CURRENT_BUCKET
            and profile.get("output_tokens") == CURRENT_OUTPUT_TOKENS
            and profile.get("candidate_gpu_plugin_sha256") ==
                EXPECTED_CARRIER_PLUGIN_SHA256
            and profile["profile_rollup"].get(
                "profile_time_is_direct_savings_evidence") is False
            and current_counts.get("FullyConnectedCompressed") == 371
            and current_counts.get(
                "IQ36ExactPhaseDualCohortHotAttentionGQA") == 10
            and profile["worker"].get("returncode") == 0
            and profile["worker"].get("oom_observed") is False
            and profile["worker"]["memory_guard"].get("tripped") is False),
      check("stock_half_qk_is_exact_but_formally_standalone_rejected",
            qk_formal.get("verdict") ==
                "reject_stock_half_qk_rope_after_formal_incremental_inference"
            and qk_formal.get("block_count") == 8
            and len(qk_streams) == 8
            and sum(len(block) for block in qk_streams.values()) == 32
            and len(qk_jitter) == 16
            and all(row.get("jitter_pass") is True for row in qk_jitter)
            and qk_formal["correctness"].get(
                "bitwise_checkpoint_count") == 512
            and qk_prefill_lcb < 1.005
            and qk_decode_lcb >= 1.005
            and qk_total_lcb >= 1.005,
            prefill_lcb=qk_prefill_lcb, decode_lcb=qk_decode_lcb,
            total_lcb=qk_total_lcb),
      check("router_shared_component_is_exact_and_same_short_lane",
            shared.get("evidence_checks_passed") is True
            and shared.get("activation_passed") is True
            and shared.get("correctness_passed") is True
            and same_lane_shared
            and shared_counts.get("FullyConnectedCompressed") == 291
            and shared_profile.get("fused_shared_triple_count") == 40
            and shared_profile.get("unfused_router_gate_count") == 40
            and shared_profile.get("unfused_linear_original_count") == 120
            and shared_profile.get("existing_fused_qkv_count") == 10
            and shared.get("worker", {}).get("oom_observed") is False),
      check("shared_reopen_condition_allows_only_this_nonoverlap_bundle",
            "source-bounded non-overlapping bundle" in
                shared_rejection.get("reopen_condition", "")
            and "0.844252" in
                shared_rejection.get("reopen_condition", "")
            and "distinct source-level state/codegen contract" in
                long_rejection.get("reopen_condition", "")),
      check("current_default_off_patch_applies_and_old_patch_does_not",
            patch_check.returncode == 0
            and patch_reverse.returncode != 0
            and old_patch_check.returncode != 0
            and patch_text.count("diff --git ") == 1
            and "fc_horizontal_fusion.cpp" in patch_text
            and "IQ36_ROUTER_SHARED_TRIPLE" in patch_text
            and "fixed_m1024_scope_enabled() ==" in patch_text
            and "router_shared_triple_enabled() ? 3 : 2" in patch_text,
            patch_sha256=sha256(PATCH),
            patch_check_stderr=patch_check.stderr.strip(),
            old_patch_check_stderr=old_patch_check.stderr.strip()),
      check("runtime_switch_is_explicit_candidate_only_and_auditable",
            "--fuse-router-shared-triple" in product_text
            and product_text.count("IQ36_ROUTER_SHARED_TRIPLE") == 3
            and '"fuse_router_shared_triple"' in product_text
            and "exclusive with fixed-FC routes" in product_text),
      check("qk_and_shared_sources_are_mechanically_nonoverlapping",
            "fc_horizontal_fusion.cpp" not in qk_text
            and "iq36_qk_rope_layout" not in patch_text
            and "IQ36_ROUTER_SHARED_TRIPLE" not in qk_text
            and "IQ36QKRopeLayout" not in patch_text),
      check("same_lane_bundle_has_conservative_point_funding",
            qk_tail_saving > 0.0
            and shared_saving > required_shared_realization
            and funding_multiple >= 5.0,
            control_tail_ms=control_tail, qk_tail_ms=qk_tail,
            qk_saving_ms=qk_tail_saving,
            historical_same_lane_shared_increment_ms=shared_saving,
            point_target=DECODE_TOTAL_POINT_TARGET,
            target_bundle_tail_ms=target_bundle_tail,
            required_shared_realization_ms=required_shared_realization,
            historical_funding_multiple=funding_multiple),
      check("cross_context_attention_sum_is_explicitly_forbidden",
            triple_context == 131072
            and triple.get("component_rejected") is True
            and triple.get("graph_integration_admitted") is False
            and invalid_triple_total > current_attention_profile_ms
            and CURRENT_BUCKET != triple_context,
            current_lane_tokens=CURRENT_BUCKET,
            triple_lane_tokens=triple_context,
            invalid_cross_lane_sum_ms=invalid_cross_lane_sum,
            invalid_triple_projection_ms=invalid_triple_total,
            current_2k_attention_profile_ms=current_attention_profile_ms,
            cross_lane_sum_admitted=False),
      check("new_point_and_formal_targets_are_pre_registered",
            PREFILL_POINT_FLOOR == 0.995
            and DECODE_TOTAL_POINT_TARGET == 1.02
            and FORMAL_PREFILL_LCB_FLOOR == 0.995
            and FORMAL_DECODE_TOTAL_LCB_TARGET == 1.02,
            point_targets={
                "prefill_ratio": PREFILL_POINT_FLOOR,
                "decode_ratio": DECODE_TOTAL_POINT_TARGET,
                "total_ratio": DECODE_TOTAL_POINT_TARGET,
                "stable_tail_saving_ms": control_tail - target_bundle_tail,
            },
            formal_lcb_targets={
                "prefill_ratio": FORMAL_PREFILL_LCB_FLOOR,
                "decode_ratio": FORMAL_DECODE_TOTAL_LCB_TARGET,
                "total_ratio": FORMAL_DECODE_TOTAL_LCB_TARGET,
            }),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_stop_never_tripped",
            min(start_available, end_available) >= stop_bytes,
            start_available_bytes=start_available,
            end_available_bytes=end_available,
            stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_build_admitted = required_checks_passed
  verdict = (
      "admit_current_qk_router_shared_source_build"
      if source_build_admitted else "inconclusive")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_build_admitted": source_build_admitted,
      "plugin_build_admitted": source_build_admitted,
      "gpu_worker_admitted": False,
      "model_worker_admitted": False,
      "cross_lane_sum_admitted": False,
      "short_bundle": {
          "lane_tokens": CURRENT_BUCKET,
          "members": [
              "current seq2189 exact-phase carrier",
              "corrected stock-half Q/K layout",
              "router-isolated N=[1,512,512] shared triple",
          ],
          "expected_execution_census": {
              "FullyConnectedCompressed": 291,
              "IQ36QKRopeLayout": 10,
              "IQ36ExactPhaseDualCohortHotAttentionGQA": 10,
              "shared_triples": 40,
              "unfused_router_gates": 40,
          },
          "point_targets": {
              "prefill_ratio": PREFILL_POINT_FLOOR,
              "decode_ratio": DECODE_TOTAL_POINT_TARGET,
              "total_ratio": DECODE_TOTAL_POINT_TARGET,
              "stable_tail_saving_ms": control_tail - target_bundle_tail,
          },
          "formal_lcb_targets": {
              "prefill_ratio": FORMAL_PREFILL_LCB_FLOOR,
              "decode_ratio": FORMAL_DECODE_TOTAL_LCB_TARGET,
              "total_ratio": FORMAL_DECODE_TOTAL_LCB_TARGET,
          },
      },
      "separate_long_lane": {
          "lane_tokens": triple_context,
          "route": "exact triple-cohort attention adaptation",
          "status": "parked_separate_source_route",
          "reason": (
              "fixed 128k component is not additive with the 2k bundle and "
              "still lacks a graph-compatible mutable-state owner"),
      },
      "ranked_directions": [
          {
              "rank": 1,
              "route": "current_qk_router_shared_short_bundle",
              "decision": "admit_source_build_only",
          },
          {
              "rank": 2,
              "route": "graph_compatible_128k_triple_attention",
              "decision": "park_separate_from_short_arithmetic",
          },
          {
              "rank": 3,
              "route": "future_official_igc_sample_tail_dealias",
              "decision": "watch_only_no_source_build",
          },
      ],
      "next_action": {
          "route": "openvino_current_qk_router_shared_source_build",
          "requirements": [
              "apply exactly the default-off durable patch after this gate",
              "build one candidate GPU plugin serially with -j1",
              "use 8-GiB preflight and 4-GiB abort guards",
              "launch no GPU/model worker until build identity passes",
          ],
      },
      "checks": checks,
      "memory": {
          "stop_bytes": stop_bytes,
          "start_available_bytes": start_available,
          "end_available_bytes": end_available,
      },
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Current short-bundle opportunity gate

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

The valid 2k bundle is the current exact-phase carrier plus corrected
stock-half Q/K and the router-isolated `[1,512,512]` shared triple. Q/K and
the FC transformation touch different owners. To reach the registered
`{DECODE_TOTAL_POINT_TARGET:.3f}x` decode/total point, the shared member must
realize only `{required_shared_realization:.6f} ms` beyond Q/K; its retained
same-lane component movement is `{shared_saving:.6f} ms`
(`{funding_multiple:.2f}x` funding). This funds source/build work, not a
speed claim.

The tempting `{invalid_cross_lane_sum:.6f}-ms` sum is invalid: it adds a fixed
128k attention component to a 2k FC/QK lane. The projected 128k triple saving
alone is `{invalid_triple_total:.6f} ms`, already larger than seq2204's entire
2k attention profile (`{current_attention_profile_ms:.6f} ms`). The long
triple route stays separate and admits no graph/plugin work here.

The new source patch is default-off, applies to the current cumulative source,
and keeps the fixed-m1024 route mutually exclusive. If admitted, build exactly
one plugin with `-j1`; retain the 8/4-GiB memory guards and run no model worker
until build identity is proven.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_build_admitted": source_build_admitted,
      "funding_multiple": funding_multiple,
      "cross_lane_sum_admitted": False,
      "gpu_or_model_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
