#!/usr/bin/env python3
"""Bucket-scoped OpenVINO product carrier and LM-head timing policy."""

from __future__ import annotations

from typing import Any


CORE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
EXACT_TIMING_BUCKETS = (2048, 4096, 8192)
COMPACT_TIMING_BUCKETS = (16384, 32768, 65536, 131072)


def candidate_path(args: Any, bucket: int) -> str:
  if args.candidate_policy == "custom":
    return "hot_cold_custom"
  if args.candidate_policy == "stock":
    return "stock_sdpa"
  return "hot_cold_custom" if bucket in CORE_BUCKETS else "stock_sdpa"


def timing_lm_head_policy(args: Any, bucket: int) -> dict[str, Any]:
  selected_path = candidate_path(args, bucket)
  exact_short = (
      args.candidate_policy == "auto" and bucket in EXACT_TIMING_BUCKETS)
  greedy_local2 = bool(
      args.lm_head_i8q1_greedy_local2 and not exact_short)
  device_feedback = bool(
      args.lm_head_device_greedy_feedback and greedy_local2)
  token_only = bool(args.lm_head_token_only_feedback and greedy_local2)
  gated_exact = bool(args.lm_head_i8q1_gated_exact and not greedy_local2)
  affine_q4 = bool(
      gated_exact and
      getattr(args, "lm_head_i8q1_gated_exact_affine_q4", False))
  if selected_path == "stock_sdpa":
    greedy_local2 = device_feedback = token_only = gated_exact = False
    affine_q4 = False
    provider = "stock_host_argmax"
  elif affine_q4:
    provider = "full_logits_gated_exact_affine_q4"
  elif gated_exact:
    provider = "full_logits_gated_exact"
  elif greedy_local2 and token_only:
    provider = "compact_local2_token_only"
  elif greedy_local2 and device_feedback:
    provider = "local2_device_greedy"
  elif greedy_local2:
    provider = "local2_full_logits"
  else:
    provider = "configured_full_logits"
  return {
      "timing_lm_head_provider": provider,
      "timing_lm_head_i8q1_gated_exact": gated_exact,
      "timing_lm_head_i8q1_gated_exact_affine_q4": affine_q4,
      "timing_lm_head_i8q1_greedy_local2": greedy_local2,
      "timing_lm_head_device_greedy_feedback": device_feedback,
      "timing_lm_head_token_only_feedback": token_only,
  }


def timing_provider_isolated(
    result: dict[str, Any],
    *,
    affine_q4_expected: bool,
    gated_exact_expected: bool,
    local2_expected: bool,
    token_only_expected: bool,
) -> bool:
  compiler_cache = result.get("compiler_cache", {})
  selection_rows = result.get(
      "lm_head_i8q1_trace", {}).get("selection_rows", [])
  local2_observed = (
      result.get("lm_head_i8q1_greedy_local2") is local2_expected and
      compiler_cache.get("lm_head_i8q1_greedy_local2_env") ==
          ("1" if local2_expected else None) and
      compiler_cache.get("lm_head_i8q1_token_only_env") ==
          ("1" if token_only_expected else None) and
      (not local2_expected or (
          result.get("lm_head_i8q1_gated_q4") is False and
          compiler_cache.get("lm_head_i8q1_gated_q4_env") is None and
          bool(selection_rows) and
          all(
              row.get("topk") == 2 and
              row.get("correction_rows") == 1940 and
              row.get("token_only") is token_only_expected and
              row.get("compact_rows") ==
                  (2910 if token_only_expected else 0) and
              ((
                  "local_top3_compact" in str(row.get("provider", "")) and
                  "compact_top3_merge_top8" in
                      str(row.get("provider", "")) and
                  "direct_compact_top8_correction" in
                      str(row.get("provider", "")) and
                  "top8_encode_token" in str(row.get("provider", "")))
               if token_only_expected else
               "local_top2" in str(row.get("provider", "")))
              for row in selection_rows))))
  gated_exact_observed = (
      result.get("lm_head_i8q1_gated_exact") is gated_exact_expected and
      compiler_cache.get("lm_head_i8q1_gated_exact_env") ==
          ("1" if gated_exact_expected else None) and
      (not gated_exact_expected or (
          bool(selection_rows) and
          all(
              row.get("topk") == 12 and
              row.get("correction_rows") == 11640 and
              row.get("token_only") is False and
              row.get("compact_rows") == 0 and
              "gated_exact" in str(row.get("provider", ""))
              for row in selection_rows))))
  affine_q4_observed = (
      result.get("lm_head_i8q1_gated_exact_affine_q4") is
          affine_q4_expected and
      compiler_cache.get("lm_head_i8q1_gated_exact_affine_q4_env") ==
          ("1" if affine_q4_expected else None) and
      (not affine_q4_expected or (
          bool(selection_rows) and
          all(
              row.get("adaptive_correction_capacity") == 16812 and
              row.get("correction_passes") == 3 and
              "affine_q4_hidden_group_norms" in
                  str(row.get("provider", "")) and
              "affine_q4_exact_candidates" in
                  str(row.get("provider", ""))
              for row in selection_rows))))
  return local2_observed and gated_exact_observed and affine_q4_observed
