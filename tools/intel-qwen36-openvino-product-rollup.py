#!/usr/bin/env python3
"""Roll up isolated OpenVINO product-gate artifacts without rerunning workers.

Every input remains an independently isolated ABBA gate.  This rollup accepts
only the frozen bucket-scoped carrier policy: the exact seq2291 affine-Q4
carrier with full-logit gated-exact timing at 2k/4k/8k, and the exact accepted
legacy carrier with compact token-only timing at 16k+.  The compatibility
bridge is limited to those two pre-registered carrier fingerprints; the compact
long timing path bypasses the count25 full-logit fallback changed by seq2291.
The rollup verifies complete case coverage and bound evidence and recomputes
the seven-bucket smoothness ladder.  It cannot turn a partial, unknown-carrier,
or failed case into a product claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-product-rollup-v1"
PRODUCT_SCHEMA = "intel-qwen36-openvino-hot-cold-product-gate-v1"
CONFIRM_SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-"
    "128k-confirmation-gate-v1")
CORE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
EXACT_TIMING_BUCKETS = (2048, 4096, 8192)
COMPACT_TIMING_BUCKETS = (16384, 32768, 65536, 131072)
CUSTOM_BUCKETS = CORE_BUCKETS
PROMPT_SETS = ("prefill_shape", "sentinel", "filler")
SUFFIX_TO_BUCKET = {
    "002k": 2048,
    "004k": 4096,
    "008k": 8192,
    "016k": 16384,
    "032k": 32768,
    "064k": 65536,
    "128k": 131072,
}
MIN_BLOCKS = 8
LEGACY_COMPACT_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
AFFINE_SHORT_PLUGIN_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
EXPECTED_CUSTOM_CONFIG_SHA256 = (
    "bd7a679031bbde2fa2626f2138bf79a5626469ccbc041faadef3b12e811200ad")
EXPECTED_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
AFFINE_SHORT_PROFILE = "seq2291_affine_q4_short"
LEGACY_COMPACT_PROFILE = "accepted_legacy_compact_long"
# These include the complete source census, plugin identity, and every carrier
# switch below.  Adding the affine flag to CARRIER_KEYS deliberately changes
# the historical hash; neither profile is admitted by a partial-field match.
EXPECTED_AFFINE_SHORT_CARRIER_FINGERPRINT = (
    "23f09faa984283129cda8f305e54803688d1d4d2438540e4c9c2f3d35bfa211c")
EXPECTED_LEGACY_COMPACT_CARRIER_FINGERPRINT = (
    "24aeff1e89e2f87647ccfeb3df63afd77e230b9de4f9380fa521824324cc783b")
ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/"
    "acceptance-matrix.json")
CARRIER_KEYS = (
    "alias_linear_state_assign",
    "candidate_dq_realloc_fastpath",
    "candidate_fc_stable_prepare_fastpath",
    "candidate_gpu_plugin_sha256",
    "custom_composition",
    "custom_config_sha256",
    "custom_sources",
    "decode_stock_micro_layers",
    "direct_ssm_state_assign",
    "exact_phase_dual_cohort",
    "exact_history_layers",
    "fuse_fixed_fc",
    "fuse_linear_conv_state",
    "linear_state_alias_scope",
    "lm_head_i8q1",
    "lm_head_i8q1_gated_exact",
    "lm_head_i8q1_gated_exact_affine_q4",
    "lm_head_i8q1_gated_q4",
    "lm_head_i8q1_greedy_local2",
    "lm_head_i8q4",
    "lm_head_token_only_feedback",
    "pack_gdn_state",
    "self_bind_hot_states",
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("artifacts", nargs="+", type=Path)
  parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
  parser.add_argument("--out-dir", type=Path, required=True)
  return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    while chunk := fh.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and value == value


def coefficient_of_variation(values: list[float]) -> float | None:
  if len(values) < 2:
    return None
  mean = statistics.mean(values)
  return statistics.pstdev(values) / mean if mean else None


def case_bucket(case_id: str) -> int:
  suffix = case_id.rsplit("_", 1)[-1]
  if suffix not in SUFFIX_TO_BUCKET:
    raise SystemExit(f"unsupported case id: {case_id}")
  return SUFFIX_TO_BUCKET[suffix]


def expected_case_ids() -> set[str]:
  return {
      f"{prompt_set}_{suffix}"
      for prompt_set in PROMPT_SETS
      for suffix in SUFFIX_TO_BUCKET
  }


def expected_candidate_path(case_id: str) -> str:
  return (
      "hot_cold_custom"
      if case_bucket(case_id) in CUSTOM_BUCKETS else "stock_sdpa")


def carrier_fingerprint(config: dict[str, Any]) -> str:
  payload = {key: config.get(key) for key in CARRIER_KEYS}
  return sha256_bytes(
      json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def expected_carrier_profile(case_id: str) -> str:
  bucket = case_bucket(case_id)
  if bucket in EXACT_TIMING_BUCKETS:
    return AFFINE_SHORT_PROFILE
  if bucket in COMPACT_TIMING_BUCKETS:
    return LEGACY_COMPACT_PROFILE
  raise SystemExit(f"unsupported carrier bucket: {bucket}")


def carrier_profile(
    config: dict[str, Any], case_ids: list[str],
) -> str | None:
  if not case_ids:
    return None
  fingerprint = carrier_fingerprint(config)
  expected_profiles = {expected_carrier_profile(case_id) for case_id in case_ids}
  if expected_profiles == {AFFINE_SHORT_PROFILE} and fingerprint == (
      EXPECTED_AFFINE_SHORT_CARRIER_FINGERPRINT):
    return AFFINE_SHORT_PROFILE
  if expected_profiles == {LEGACY_COMPACT_PROFILE} and fingerprint == (
      EXPECTED_LEGACY_COMPACT_CARRIER_FINGERPRINT):
    return LEGACY_COMPACT_PROFILE
  return None


def model_fingerprint(model: dict[str, Any]) -> str | None:
  if model.get("required_checks_passed") is not True:
    return None
  rows = [
      (row.get("file"), row.get("observed_sha256"), row.get("observed_bytes"))
      for row in model.get("files", [])
  ]
  if not rows:
    return None
  return sha256_bytes(json.dumps(rows, sort_keys=True).encode())


def historical_source_matches(
    path_text: str, expected: str, commit: str,
) -> bool:
  path = ROOT / path_text
  if path_text.startswith("output/"):
    return path.is_file() and sha256_file(path) == expected
  result = subprocess.run(
      ["git", "show", f"{commit}:{path_text}"],
      cwd=ROOT,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      check=False)
  return result.returncode == 0 and sha256_bytes(result.stdout) == expected


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      stdout=subprocess.PIPE, text=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain", "--untracked-files=all"],
      cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True).stdout
  dirty_paths = [
      line[3:] for line in status.splitlines() if len(line) >= 4]
  return {
      "commit": commit,
      "dirty": bool(dirty_paths),
      "dirty_paths": dirty_paths,
  }


def accepted_bucket_timing_policy(
    config: dict[str, Any], profile: str,
) -> bool:
  cases = config.get("cases")
  if not isinstance(cases, list) or not cases:
    return False
  for case in cases:
    case_id = str(case.get("case_id", ""))
    bucket = case_bucket(case_id)
    if case.get("candidate_path") != "hot_cold_custom":
      return False
    provider = case.get("timing_lm_head_provider")
    if profile == AFFINE_SHORT_PROFILE and bucket in EXACT_TIMING_BUCKETS:
      if not (
          provider == "full_logits_gated_exact_affine_q4"
          and case.get("timing_lm_head_i8q1_gated_exact") is True
          and case.get(
              "timing_lm_head_i8q1_gated_exact_affine_q4") is True
          and case.get("timing_lm_head_i8q1_greedy_local2") is False
          and case.get("timing_lm_head_token_only_feedback") is False):
        return False
    elif profile == LEGACY_COMPACT_PROFILE and bucket in COMPACT_TIMING_BUCKETS:
      # Accepted long artifacts predate the explicit per-case fields. Their
      # top-level local2/token-only contract and worker isolation checks are
      # equivalent; new artifacts must record the effective provider.
      if provider is None:
        continue
      if not (
          provider == "compact_local2_token_only"
          and case.get("timing_lm_head_i8q1_gated_exact") is False
          and case.get(
              "timing_lm_head_i8q1_gated_exact_affine_q4", False) is False
          and case.get("timing_lm_head_i8q1_greedy_local2") is True
          and case.get("timing_lm_head_token_only_feedback") is True):
        return False
    else:
      return False
  return True


def accepted_custom_config(
    config: dict[str, Any], case_ids: list[str],
) -> bool:
  profile = carrier_profile(config, case_ids)
  return (
      profile is not None
      and config.get("candidate_gpu_plugin_sha256") == (
          AFFINE_SHORT_PLUGIN_SHA256
          if profile == AFFINE_SHORT_PROFILE
          else LEGACY_COMPACT_PLUGIN_SHA256)
      and config.get("custom_config_sha256") ==
      EXPECTED_CUSTOM_CONFIG_SHA256
      and config.get("custom_composition") == "exact_phase"
      and config.get("exact_phase_dual_cohort") is True
      and config.get("decode_stock_micro_layers") == EXPECTED_LAYERS
      and config.get("exact_history_layers") == EXPECTED_LAYERS
      and config.get("alias_linear_state_assign") is True
      and config.get("linear_state_alias_scope") == "all"
      and config.get("fuse_linear_conv_state") is True
      and config.get("direct_ssm_state_assign") is False
      and config.get("fuse_fixed_fc") is False
      and config.get("pack_gdn_state") is False
      and config.get("self_bind_hot_states") is False
      and config.get("candidate_dq_realloc_fastpath") is True
      and config.get("candidate_fc_stable_prepare_fastpath") is True
      and config.get("lm_head_i8q4") is False
      and config.get("lm_head_i8q1") is True
      and config.get("lm_head_i8q1_gated_exact") is True
      and (
          config.get("lm_head_i8q1_gated_exact_affine_q4") is True
          if profile == AFFINE_SHORT_PROFILE
          else config.get(
              "lm_head_i8q1_gated_exact_affine_q4", False) is False)
      and config.get("lm_head_i8q1_gated_q4") is False
      and config.get("lm_head_i8q1_greedy_local2") is True
      and config.get("lm_head_token_only_feedback") is True
      and accepted_bucket_timing_policy(config, profile)
      and bool(config.get("custom_sources")))


def product_artifact(path: Path, gate: dict[str, Any]) -> dict[str, Any]:
  config = gate.get("config", {})
  correctness = gate.get("correctness", [])
  performance = gate.get("performance", [])
  memory = read_json(path / "memory.json")
  smoothness = read_json(path / "smoothness.json")
  model = read_json(path / "model-identity.json")
  case_ids = [row.get("case_id") for row in performance]
  normalized_case_ids = [str(case_id) for case_id in case_ids]
  profile = carrier_profile(config, normalized_case_ids)
  has_custom_case = any(
      expected_candidate_path(str(case_id)) == "hot_cold_custom"
      for case_id in case_ids)
  checks = [
      gate.get("run_checks_passed") is True,
      gate.get("stopped_reason") is None,
      gate.get("git", {}).get("dirty") is False,
      config.get("output_tokens") == 512,
      config.get("paired_blocks", 0) >= MIN_BLOCKS,
      len(performance) == len(correctness) == len(case_ids),
      all(row.get("paired_block_count", 0) >= MIN_BLOCKS
          for row in performance),
      all(row.get("promotion_rate_pass") is True for row in performance),
      all(row.get("required_checks_passed") is True for row in correctness),
      memory.get("required_checks_passed") is True,
      smoothness.get("required_checks_passed") is True,
      model.get("required_checks_passed") is True,
      not has_custom_case or accepted_custom_config(
          config, normalized_case_ids),
  ]
  return {
      "artifact": relative(path),
      "artifact_gate_sha256": sha256_file(path / "gate.json"),
      "carrier_fingerprint": carrier_fingerprint(config),
      "carrier_profile": profile,
      "case_ids": case_ids,
      "checks_passed": all(checks),
      "correctness": correctness,
      "git": gate.get("git"),
      "jitter_rows": smoothness.get("jitter_rows", []),
      "memory": memory,
      "model_fingerprint": model_fingerprint(model),
      "performance": performance,
      "schema_version": PRODUCT_SCHEMA,
  }


def confirmation_artifact(path: Path, gate: dict[str, Any]) -> dict[str, Any]:
  commit = str(gate.get("git", {}).get("commit", ""))
  sources = gate.get("sources", {})
  correctness_paths = [
      ROOT / source for source in sources
      if source.endswith("/correctness.json")
  ]
  if len(correctness_paths) != 1:
    raise SystemExit(f"{path}: expected one bound correctness.json")
  correctness_doc = read_json(correctness_paths[0])
  correctness = correctness_doc.get("cases", [])
  candidate_results = sorted(path.glob("raw/**/candidate-*/worker-result.json"))
  plugin_shas = {
      read_json(result).get("candidate_gpu_plugin_sha256")
      for result in candidate_results
  }
  candidate_configs = sorted(path.glob("raw/**/candidate-*/worker-config.json"))
  config_rows = [read_json(config) for config in candidate_configs]
  config_pass = bool(config_rows) and all(
      row.get("custom_composition") == "exact_phase"
      and row.get("exact_phase_dual_cohort") is True
      and row.get("decode_stock_micro_layers") == EXPECTED_LAYERS
      and row.get("exact_history_layers") == EXPECTED_LAYERS
      and row.get("alias_linear_state_assign") is True
      and row.get("fuse_linear_conv_state") is True
      and row.get("lm_head_i8q1") is True
      and row.get("lm_head_i8q1_gated_exact") is True
      and row.get(
          "lm_head_i8q1_gated_exact_affine_q4", False) is False
      and row.get("lm_head_i8q1_greedy_local2") is True
      and row.get("lm_head_token_only_feedback") is True
      and row.get("candidate_fc_stable_prepare_fastpath") is True
      for row in config_rows)
  bound_sources_pass = bool(sources) and all(
      historical_source_matches(source, digest, commit)
      for source, digest in sources.items())
  performance = [gate.get("performance", {})]
  memory = gate.get("memory", {})
  smoothness = gate.get("smoothness", {})
  model = gate.get("model_identity", {})
  case_ids = [row.get("case_id") for row in performance]
  checks = [
      gate.get("required_checks_passed") is True,
      gate.get("lane_128k_confirmed") is True,
      gate.get("git", {}).get("dirty") is False,
      gate.get("paired_block_count", 0) >= MIN_BLOCKS,
      case_ids == ["sentinel_128k"],
      performance[0].get("paired_block_count", 0) >= MIN_BLOCKS,
      performance[0].get("promotion_rate_pass") is True,
      len(correctness) == 1,
      correctness[0].get("required_checks_passed") is True,
      correctness_doc.get("required_checks_passed") is True,
      memory.get("required_checks_passed") is True,
      smoothness.get("required_checks_passed") is True,
      model.get("required_checks_passed") is True,
      plugin_shas == {LEGACY_COMPACT_PLUGIN_SHA256},
      sources.get("engine/openvino/custom/iq36_hot_attention_gqa.xml") ==
      EXPECTED_CUSTOM_CONFIG_SHA256,
      config_pass,
      bound_sources_pass,
  ]
  return {
      "artifact": relative(path),
      "artifact_gate_sha256": sha256_file(path / "gate.json"),
      "carrier_fingerprint": None,
      "carrier_profile": LEGACY_COMPACT_PROFILE,
      "case_ids": case_ids,
      "checks_passed": all(checks),
      "correctness": correctness,
      "git": gate.get("git"),
      "jitter_rows": smoothness.get("jitter_rows", []),
      "memory": memory,
      "model_fingerprint": model_fingerprint(model),
      "performance": performance,
      "schema_version": CONFIRM_SCHEMA,
  }


def load_artifact(path: Path) -> dict[str, Any]:
  path = path.resolve()
  gate_path = path / "gate.json"
  if not gate_path.is_file():
    raise SystemExit(f"missing gate.json: {path}")
  gate = read_json(gate_path)
  schema = gate.get("schema_version")
  if schema == PRODUCT_SCHEMA:
    return product_artifact(path, gate)
  if schema == CONFIRM_SCHEMA:
    return confirmation_artifact(path, gate)
  raise SystemExit(f"{path}: unsupported schema {schema!r}")


def smoothness_rollup(
    performance: list[dict[str, Any]],
    jitter_rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
  threshold = float(
      acceptance["smoothness"]["decode_tpot_p95_over_p50_max"])
  jitter_counts = Counter(row.get("case_id") for row in jitter_rows)
  by_bucket: dict[int, list[dict[str, Any]]] = {}
  for row in performance:
    by_bucket.setdefault(case_bucket(str(row["case_id"])), []).append(row)
  floors = acceptance["bootstrap_targets"]
  ladder = []
  for bucket, rows in sorted(by_bucket.items()):
    prefill = statistics.median(
        float(row["absolute_floors"]["prefill_median"]) for row in rows)
    decode = statistics.median(
        float(row["absolute_floors"]["decode_median"]) for row in rows)
    ladder.append({
        "bucket": bucket,
        "decode_normalized": (
            decode / float(floors["decode_tokens_s"][str(bucket)])),
        "decode_tokens_s": decode,
        "prefill_normalized": (
            prefill / float(floors["prefill_tokens_s"][str(bucket)])),
        "prefill_tokens_s": prefill,
    })
  adjacent = []
  for previous, current in zip(ladder, ladder[1:]):
    adjacent.append({
        "decode_normalized_retention": (
            current["decode_normalized"] / previous["decode_normalized"]),
        "from_bucket": previous["bucket"],
        "prefill_normalized_retention": (
            current["prefill_normalized"] / previous["prefill_normalized"]),
        "to_bucket": current["bucket"],
    })
  decode_cv = coefficient_of_variation(
      [row["decode_normalized"] for row in ladder])
  prefill_cv = coefficient_of_variation(
      [row["prefill_normalized"] for row in ladder])
  expected = expected_case_ids()
  checks = [
      {
          "name": "all_cases_have_sixteen_candidate_jitter_rows",
          "pass": set(jitter_counts) == expected and all(
              jitter_counts[case_id] == 16 for case_id in expected),
          "counts": dict(sorted(jitter_counts.items())),
      },
      {
          "name": "decode_tpot_p95_over_p50",
          "pass": bool(jitter_rows) and all(
              row.get("pass") is True
              and finite(row.get("p95_over_p50"))
              and float(row["p95_over_p50"]) <= threshold
              for row in jitter_rows),
          "threshold": threshold,
      },
      {
          "name": "adjacent_decode_normalized_retention",
          "pass": len(ladder) == len(CORE_BUCKETS) and all(
              row["decode_normalized_retention"] >= 0.75
              for row in adjacent),
      },
      {
          "name": "adjacent_prefill_normalized_retention",
          "pass": len(ladder) == len(CORE_BUCKETS) and all(
              row["prefill_normalized_retention"] >= 0.75
              for row in adjacent),
      },
      {
          "name": "decode_target_normalized_cv",
          "pass": finite(decode_cv) and decode_cv <= float(
              acceptance["smoothness"]["target_normalized_score_cv_max"]),
          "value": decode_cv,
      },
      {
          "name": "prefill_target_normalized_cv",
          "pass": finite(prefill_cv) and prefill_cv <= float(
              acceptance["smoothness"][
                  "prefill_target_normalized_score_cv_max"]),
          "value": prefill_cv,
      },
  ]
  return {
      "adjacent": adjacent,
      "checks": checks,
      "jitter_rows": jitter_rows,
      "ladder": ladder,
      "required_checks_passed": all(check["pass"] for check in checks),
  }


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# OpenVINO Product Evidence Rollup",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- product promotion ready: `{str(payload['product_promotion_ready']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- artifacts: `{len(payload['artifacts'])}`",
      f"- cases: `{len(payload['performance'])}`",
      "",
      "| case | path | blocks | prefill LCB | decode LCB | total LCB | absolute |",
      "|---|---|---:|---:|---:|---:|:---:|",
  ]
  for row in payload["performance"]:
    phase = row["phase_inference"]
    lines.append(
        f"| {row['case_id']} | {row['candidate_path']} | "
        f"{row['paired_block_count']} | "
        f"{phase['prefill_tokens_s']['lower_confidence_bound_ratio']:.6f} | "
        f"{phase['decode_tokens_s']['lower_confidence_bound_ratio']:.6f} | "
        f"{phase['total_rate']['lower_confidence_bound_ratio']:.6f} | "
        f"{'pass' if row['absolute_floors']['pass'] else 'fail'} |")
  lines.append("")
  return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
  if args.out_dir.exists():
    raise SystemExit(f"output directory exists: {args.out_dir}")
  artifacts = [load_artifact(path) for path in args.artifacts]
  performance = [
      row for artifact in artifacts for row in artifact["performance"]]
  correctness = [
      row for artifact in artifacts for row in artifact["correctness"]]
  jitter_rows = [
      row for artifact in artifacts for row in artifact["jitter_rows"]]
  memory_rows = [
      row for artifact in artifacts
      for row in artifact["memory"].get("rows", [])]
  case_ids = [str(row.get("case_id")) for row in performance]
  duplicate_cases = sorted(
      case_id for case_id, count in Counter(case_ids).items() if count > 1)
  correctness_by_case = {
      str(row.get("case", {}).get("case_id")): row for row in correctness}
  correctness_case_ids = [
      str(row.get("case", {}).get("case_id")) for row in correctness]
  duplicate_correctness_cases = sorted(
      case_id for case_id, count in Counter(correctness_case_ids).items()
      if count > 1)
  carrier_fingerprints = sorted({
      artifact["carrier_fingerprint"] for artifact in artifacts
      if artifact["carrier_fingerprint"] is not None})
  carrier_profile_by_case = {
      str(case_id): artifact["carrier_profile"]
      for artifact in artifacts for case_id in artifact["case_ids"]}
  expected_carrier_profile_by_case = {
      case_id: expected_carrier_profile(case_id)
      for case_id in expected_case_ids()}
  model_fingerprints = {
      artifact["model_fingerprint"] for artifact in artifacts
      if artifact["model_fingerprint"] is not None}
  acceptance = read_json(args.acceptance)
  smoothness = smoothness_rollup(performance, jitter_rows, acceptance)
  expected = expected_case_ids()
  rollup_git = git_state()
  checks = [
      {
          "name": "rollup_repository_clean",
          "pass": rollup_git["dirty"] is False,
          "git": rollup_git,
      },
      {
          "name": "all_artifact_gates_pass",
          "pass": all(artifact["checks_passed"] for artifact in artifacts),
      },
      {
          "name": "complete_unique_seven_by_three_matrix",
          "pass": set(case_ids) == expected and not duplicate_cases,
          "missing": sorted(expected - set(case_ids)),
          "unexpected": sorted(set(case_ids) - expected),
          "duplicates": duplicate_cases,
      },
      {
          "name": "bucket_selected_paths_exact",
          "pass": all(
              row.get("candidate_path") == expected_candidate_path(
                  str(row.get("case_id")))
              for row in performance),
      },
      {
          "name": "every_case_has_formal_paired_inference",
          "pass": len(performance) == len(expected) and all(
              row.get("paired_block_count", 0) >= MIN_BLOCKS
              and row.get("promotion_rate_pass") is True
              for row in performance),
      },
      {
          "name": "every_case_has_required_correctness",
          "pass": (
              len(correctness) == len(expected)
              and not duplicate_correctness_cases
              and set(correctness_by_case) == expected
              and all(
              row.get("required_checks_passed") is True
              and float(row.get("top1_rate", 0.0)) >= 0.99
              and float(row.get("kld_max", 1.0)) <= 0.005
              for row in correctness_by_case.values())),
          "duplicates": duplicate_correctness_cases,
      },
      {
          "name": "bucket_scoped_carrier_compatibility_exact",
          "pass": (
              carrier_profile_by_case == expected_carrier_profile_by_case
              and set(carrier_fingerprints) == {
                  EXPECTED_AFFINE_SHORT_CARRIER_FINGERPRINT,
                  EXPECTED_LEGACY_COMPACT_CARRIER_FINGERPRINT,
              }),
          "profiles": dict(sorted(carrier_profile_by_case.items())),
          "expected_profiles": dict(sorted(
              expected_carrier_profile_by_case.items())),
          "fingerprints": carrier_fingerprints,
          "bridge": (
              "seq2291 affine-Q4 only for 2k/4k/8k full-logit timing; "
              "accepted legacy plugin only for 16k+ compact token-only timing"),
      },
      {
          "name": "locked_model_fingerprint_consistent",
          "pass": len(model_fingerprints) == 1,
          "fingerprints": sorted(model_fingerprints),
      },
      {
          "name": "all_memory_gates_pass",
          "pass": bool(memory_rows)
          and all(artifact["memory"].get("required_checks_passed") is True
                  for artifact in artifacts)
          and all(row.get("oom_observed") is False for row in memory_rows)
          and all(row.get("memory_guard_tripped") is False
                  for row in memory_rows),
      },
      {
          "name": "complete_context_ladder_smoothness",
          "pass": smoothness["required_checks_passed"],
      },
  ]
  checks_by_name = {check["name"]: check for check in checks}
  passed = all(check["pass"] for check in checks)
  performance.sort(key=lambda row: (
      case_bucket(str(row["case_id"])), str(row["case_id"])))
  correctness.sort(key=lambda row: (
      case_bucket(str(row["case"]["case_id"])),
      str(row["case"]["case_id"])))
  artifact_rows = [{
      key: artifact[key] for key in (
          "artifact", "artifact_gate_sha256", "case_ids", "checks_passed",
          "carrier_fingerprint", "carrier_profile", "git", "schema_version")
  } for artifact in artifacts]
  payload = {
      "artifacts": artifact_rows,
      "checks": checks,
      "correctness": correctness,
      "created_at": iso_now(),
      "git": rollup_git,
      "performance": performance,
      "product_promotion_ready": passed,
      "required_checks_passed": passed,
      "route_label": "product_candidate" if passed else "rejected",
      "schema_version": SCHEMA,
      "speedup_claims_allowed": passed,
      "workstream": WORKSTREAM,
  }
  memory = {
      "checks": [checks_by_name["all_memory_gates_pass"]],
      "required_checks_passed": checks_by_name[
          "all_memory_gates_pass"]["pass"],
      "rows": memory_rows,
  }
  args.out_dir.mkdir(parents=True)
  write_json(args.out_dir / "gate.json", payload)
  write_json(args.out_dir / "correctness.json", {
      "cases": correctness,
      "required_checks_passed": checks_by_name[
          "every_case_has_required_correctness"]["pass"],
  })
  write_json(args.out_dir / "performance.json", {
      "cases": performance,
      "product_promotion_ready": passed,
      "speedup_claims_allowed": passed,
  })
  write_json(args.out_dir / "memory.json", memory)
  write_json(args.out_dir / "smoothness.json", smoothness)
  write_json(args.out_dir / "manifest.json", {
      "artifacts": artifact_rows,
      "acceptance": relative(args.acceptance),
      "acceptance_sha256": sha256_file(args.acceptance),
      "git": rollup_git,
      "schema_version": SCHEMA,
      "tool": relative(Path(__file__)),
      "tool_sha256": sha256_file(Path(__file__)),
      "workstream": WORKSTREAM,
  })
  (args.out_dir / "summary.md").write_text(
      build_summary(payload), encoding="utf-8")
  print(json.dumps({
      "event": "rollup_complete",
      "out_dir": relative(args.out_dir),
      "product_promotion_ready": passed,
      "required_checks_passed": passed,
      "speedup_claims_allowed": passed,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main(parse_args()))
