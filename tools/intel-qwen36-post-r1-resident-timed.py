#!/usr/bin/env python3
"""Capture a post-R1 resident/timed diagnostic artifact.

This is not a benchmark promotion tool and never enables speedup claims. It
wraps the R1 native candidate generator after that gate is closed, asks the same
single-process native loop to run optional internal warmup plus timed passes, and
records the benchmark-discipline metadata needed before optimization work.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-post-r1-resident-timed-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
CANDIDATE_TOOL = ROOT / "tools/intel-qwen36-r1-native-candidate-jsonl.py"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=21600)
  parser.add_argument("--max-new-tokens", type=int, default=16)
  parser.add_argument("--warmup-runs", type=int, default=1)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument(
      "--no-resident-cache",
      dest="resident_cache",
      action="store_false",
      help="Disable the native loop's resident tensor cache for baseline diagnostics.",
  )
  parser.set_defaults(resident_cache=True)
  parser.add_argument(
      "--no-profile-matvec",
      dest="profile_matvec",
      action="store_false",
      help="Disable native per-tensor matvec profiling.",
  )
  parser.set_defaults(profile_matvec=True)
  parser.add_argument(
      "--prefill-final-logits-only",
      action="store_true",
      help="Only compute LM-head logits for the final prompt token during prefill.",
  )
  parser.add_argument(
      "--decode-top1-only",
      action="store_true",
      help="Only compute top-1 logits for decode continuation tokens.",
  )
  parser.add_argument(
      "--full-attention-inplace-history",
      action="store_true",
      help="Update full-attention K/V history in place in the native runner.",
  )
  parser.add_argument(
      "--lm-head-top-k",
      action="store_true",
      help="Use the switch-gated fused top-k matvec route for output.weight.",
  )
  parser.add_argument(
      "--lm-head-threads",
      type=int,
      default=1,
      help="Thread count for --lm-head-top-k. Defaults to 1.",
  )
  parser.add_argument(
      "--lm-head-q6-pair-dot",
      action="store_true",
      help="Use paired direct Q6_K row-dot route for LM-head top-k rows.",
  )
  parser.add_argument(
      "--expert-slice-matvec",
      action="store_true",
      help="Use contiguous selected-expert slice reads for expert matvecs.",
  )
  parser.add_argument(
      "--expert-slice-threads",
      type=int,
      default=1,
      help="Thread count for --expert-slice-matvec. Defaults to 1.",
  )
  parser.add_argument(
      "--dense-matvec",
      action="store_true",
      help="Use parallel dense matvec route for large generic matvec_tensor calls.",
  )
  parser.add_argument(
      "--dense-matvec-threads",
      type=int,
      default=1,
      help="Thread count for --dense-matvec. Defaults to 1.",
  )
  parser.add_argument(
      "--dense-matvec-min-rows",
      type=int,
      default=1024,
      help="Minimum tensor row count for --dense-matvec. Defaults to 1024.",
  )
  parser.add_argument(
      "--dense-matvec-payload-cache",
      action="store_true",
      help="Cache selected dense matvec tensor payloads in the resident process.",
  )
  parser.add_argument(
      "--dense-q4-direct-dot",
      action="store_true",
      help="Use the direct Q4_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q4-pair-dot",
      action="store_true",
      help="Use paired direct Q4_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q6-direct-dot",
      action="store_true",
      help="Use the direct Q6_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q6-pair-dot",
      action="store_true",
      help="Use paired direct Q6_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--q4-direct-minsum-pair",
      action="store_true",
      help="Use paired bsums for the Q4_K direct min-sum term.",
  )
  parser.add_argument(
      "--q4-block-meta-cache",
      action="store_true",
      help="Cache decoded Q4_K block metadata for resident tensor payload dots.",
  )
  parser.add_argument(
      "--small-q4-direct-dot",
      action="store_true",
      help="Use direct Q4_K dot for cached non-dense small Q4 matvec rows.",
  )
  parser.add_argument(
      "--matvec-q8-input-reuse",
      action="store_true",
      help="Reuse Q8_K activation blocks across same-input quantized matvec calls.",
  )
  parser.add_argument(
      "--shared-parallel-executor",
      action="store_true",
      help="Use the switch-gated shared worker executor for parallel matvec routes.",
  )
  parser.add_argument(
      "--shared-expert-gate-up-fused",
      action="store_true",
      help="Use the switch-gated fused direct Q4_K route for shared expert gate/up.",
  )
  parser.add_argument(
      "--selected-expert-ffn",
      action="store_true",
      help="Use the switch-gated fused selected-expert FFN route.",
  )
  parser.add_argument(
      "--selected-expert-ffn-threads",
      type=int,
      default=1,
      help="Thread count for --selected-expert-ffn. Defaults to 1.",
  )
  parser.add_argument(
      "--selected-expert-minimal-outputs",
      action="store_true",
      help="Skip diagnostic selected-expert intermediate vectors in the native loop.",
  )
  parser.add_argument(
      "--selected-expert-slice-cache",
      action="store_true",
      help="Cache selected expert tensor slices in the resident process.",
  )
  parser.add_argument(
      "--selected-expert-down-slice-cache",
      action="store_true",
      help="Cache selected expert down tensor slices in the resident process.",
  )
  parser.add_argument(
      "--selected-expert-down-expert-major",
      action="store_true",
      help="Compute selected-expert down rows expert-major, then aggregate in selected order.",
  )
  parser.add_argument(
      "--selected-expert-down-q4-pair-dot",
      action="store_true",
      help="Use a Q4_K row-pair dot route for adjacent selected-expert down rows.",
  )
  parser.add_argument(
      "--selected-expert-down-q6-pair-dot",
      action="store_true",
      help="Use a Q6_K row-pair dot route for adjacent selected-expert down rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-direct-dot",
      action="store_true",
      help="Use the direct Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-pair-dot",
      action="store_true",
      help="Use the paired direct Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-pair-sum-dot",
      action="store_true",
      help="Use the paired block-sum Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--case-id",
      action="append",
      default=[],
      help="Case id to run. Repeatable. Defaults to all six seed cases.",
  )
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def rel(path: Path) -> str:
  return path.resolve().relative_to(ROOT).as_posix()


def collect_host_metadata(host: str, env_script: str, timeout_s: int) -> dict[str, Any]:
  commands = {
      "uname_a": "uname -a",
      "kernel_release": "uname -r",
      "lscpu_summary": (
          "bash -lc "
          + shlex.quote("lscpu | egrep 'Architecture|CPU\\(s\\)|Thread|Core|Socket|Model name'")
      ),
      "gxx_version": (
          "bash -lc "
          + shlex.quote(
              f"source {shlex.quote(env_script)} >/dev/null 2>&1 && "
              "g++ --version | head -n 1"
          )
      ),
      "ldd_version": "bash -lc " + shlex.quote("ldd --version | head -n 1"),
      "env_script_probe": (
          "bash -lc "
          + shlex.quote(
              f"source {shlex.quote(env_script)} >/dev/null 2>&1 && "
              "env | sort | egrep '^(LD_LIBRARY_PATH|PATH|ONEAPI|OCL|ZE|SYCL)=' | head -n 80"
          )
      ),
  }
  return {
      name: iq36_local.run_target(host, command, timeout_s)
      for name, command in commands.items()
  }


def timed_case_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  timed_runs = parsed.get("timed_runs", [])
  if not isinstance(timed_runs, list):
    return rows
  for timed_run in timed_runs:
    if not isinstance(timed_run, dict):
      continue
    run_index = timed_run.get("run_index")
    for case in timed_run.get("cases", []):
      if not isinstance(case, dict):
        continue
      timing = case.get("timing_ns", {})
      if not isinstance(timing, dict):
        timing = {}
      rows.append({
          "case_id": case.get("case_id"),
          "generated_token_count": len(case.get("generated_token_ids", [])),
          "prompt_token_count": case.get("prompt_token_count"),
          "run_index": run_index,
          "timing_ns": timing,
      })
  return rows


def timing_present(parsed: dict[str, Any], expected_timed_runs: int) -> bool:
  if parsed.get("timing_schema_version") != "intel-qwen36-engine-timing-v0":
    return False
  if parsed.get("timed_run_count") != expected_timed_runs:
    return False
  rows = timed_case_rows(parsed)
  if not rows:
    return False
  for row in rows:
    timing = row.get("timing_ns", {})
    if not all(isinstance(timing.get(key), int) and timing.get(key) > 0 for key in (
        "case_total",
        "prompt_prefill",
    )):
      return False
    if not isinstance(timing.get("decode_continuation"), int):
      return False
  return True


def build_summary(payload: dict[str, Any]) -> str:
  diag = payload["diagnostic"]
  metadata = payload["benchmark_metadata"]
  cache_stats = metadata.get("resident_tensor_cache_stats", {})
  top_profile = metadata.get("matvec_profile_top", [])
  top_line = "none"
  if top_profile and isinstance(top_profile[0], dict):
    top = top_profile[0]
    top_line = (
        f"{top.get('op')} {top.get('tensor_name')} "
        f"calls={top.get('call_count')} total_ns={top.get('total_ns')}"
    )
  lines = [
      "# Post-R1 Resident Timed Diagnostic",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- candidate artifact: `{diag['candidate_artifact']}`",
      f"- warmup runs: {diag['warmup_runs']}",
      f"- timed runs: {diag['timed_runs']}",
      f"- case rows timed: {diag['timed_case_row_count']}",
      f"- prefill final logits only: `{str(metadata.get('prefill_final_logits_only_enabled')).lower()}`",
      f"- decode top1 only: `{str(metadata.get('decode_top1_only_enabled')).lower()}`",
      f"- full-attention inplace history: `{str(metadata.get('full_attention_inplace_history_enabled')).lower()}`",
      f"- resident cache enabled: `{str(metadata.get('resident_tensor_cache_enabled')).lower()}`",
      f"- resident payload hits/misses: {cache_stats.get('tensor_payload_hits')} / {cache_stats.get('tensor_payload_misses')}",
      f"- decoded row hits/misses: {cache_stats.get('decoded_row_hits')} / {cache_stats.get('decoded_row_misses')}",
      f"- dense matvec route: `{str(metadata.get('dense_matvec_enabled')).lower()}`",
      f"- dense matvec threads: {metadata.get('dense_matvec_threads')}",
      f"- dense matvec min rows: {metadata.get('dense_matvec_min_rows')}",
      f"- dense matvec payload cache: `{str(metadata.get('dense_matvec_payload_cache_enabled')).lower()}`",
      f"- dense Q4 direct dot: `{str(metadata.get('dense_q4_direct_dot_enabled')).lower()}`",
      f"- dense Q4 pair dot: `{str(metadata.get('dense_q4_pair_dot_enabled')).lower()}`",
      f"- dense Q6 direct dot: `{str(metadata.get('dense_q6_direct_dot_enabled')).lower()}`",
      f"- dense Q6 pair dot: `{str(metadata.get('dense_q6_pair_dot_enabled')).lower()}`",
      f"- Q4 direct min-sum pair: `{str(metadata.get('q4_direct_minsum_pair_enabled')).lower()}`",
      f"- Q4 block meta cache: `{str(metadata.get('q4_block_meta_cache_enabled')).lower()}`",
      f"- small Q4 direct dot: `{str(metadata.get('small_q4_direct_dot_enabled')).lower()}`",
      f"- matvec Q8 input reuse: `{str(metadata.get('matvec_q8_input_reuse_enabled')).lower()}`",
      f"- LM-head top-k route: `{str(metadata.get('lm_head_top_k_enabled')).lower()}`",
      f"- LM-head Q6 pair dot: `{str(metadata.get('lm_head_q6_pair_dot_enabled')).lower()}`",
      f"- LM-head threads: {metadata.get('lm_head_threads')}",
      f"- expert-slice matvec route: `{str(metadata.get('expert_slice_matvec_enabled')).lower()}`",
      f"- expert-slice threads: {metadata.get('expert_slice_threads')}",
      f"- selected-expert FFN route: `{str(metadata.get('selected_expert_ffn_enabled')).lower()}`",
      f"- selected-expert FFN threads: {metadata.get('selected_expert_ffn_threads')}",
      f"- selected-expert minimal outputs: `{str(metadata.get('selected_expert_minimal_outputs_enabled')).lower()}`",
      f"- selected expert slice cache: `{str(metadata.get('selected_expert_slice_cache_enabled')).lower()}`",
      f"- selected expert down slice cache: `{str(metadata.get('selected_expert_down_slice_cache_enabled')).lower()}`",
      f"- selected expert down expert-major: `{str(metadata.get('selected_expert_down_expert_major_enabled')).lower()}`",
      f"- selected expert down Q4 pair dot: `{str(metadata.get('selected_expert_down_q4_pair_dot_enabled')).lower()}`",
      f"- selected expert down Q6 pair dot: `{str(metadata.get('selected_expert_down_q6_pair_dot_enabled')).lower()}`",
      f"- selected gate Q4 direct dot: `{str(metadata.get('selected_gate_q4_direct_dot_enabled')).lower()}`",
      f"- selected gate Q4 pair dot: `{str(metadata.get('selected_gate_q4_pair_dot_enabled')).lower()}`",
      f"- selected gate Q4 pair-sum dot: `{str(metadata.get('selected_gate_q4_pair_sum_dot_enabled')).lower()}`",
      f"- shared parallel executor: `{str(metadata.get('shared_parallel_executor_enabled')).lower()}`",
      f"- shared expert gate/up fused route: `{str(metadata.get('shared_expert_gate_up_fused_enabled')).lower()}`",
      f"- matvec profile enabled: `{str(metadata.get('matvec_profile_enabled')).lower()}`",
      f"- top matvec: `{top_line}`",
      f"- R1 native correctness gate closed: `{str(diag['r1_native_correctness_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "This artifact records timing and benchmark metadata for route diagnosis",
      "only. It is not an accepted speedup or performance matrix.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.max_new_tokens < 1 or args.max_new_tokens > 16:
    raise SystemExit("--max-new-tokens must be 1..16")
  if args.warmup_runs < 0 or args.warmup_runs > 8:
    raise SystemExit("--warmup-runs must be 0..8")
  if args.timed_runs < 1 or args.timed_runs > 8:
    raise SystemExit("--timed-runs must be 1..8")
  if args.lm_head_threads < 1 or args.lm_head_threads > 256:
    raise SystemExit("--lm-head-threads must be 1..256")
  if args.expert_slice_threads < 1 or args.expert_slice_threads > 256:
    raise SystemExit("--expert-slice-threads must be 1..256")
  if args.dense_matvec_threads < 1 or args.dense_matvec_threads > 256:
    raise SystemExit("--dense-matvec-threads must be 1..256")
  if args.dense_matvec_min_rows < 1 or args.dense_matvec_min_rows > 1048576:
    raise SystemExit("--dense-matvec-min-rows must be 1..1048576")
  if args.selected_expert_ffn_threads < 1 or args.selected_expert_ffn_threads > 256:
    raise SystemExit("--selected-expert-ffn-threads must be 1..256")

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/post-r1-resident-timed-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  candidate_out_dir = out_dir / "native-candidate-jsonl"

  host_metadata = collect_host_metadata(args.host, args.env_script, min(args.timeout_s, 120))
  candidate_cmd = [
      "python3",
      str(CANDIDATE_TOOL),
      "--host",
      args.host,
      "--model",
      args.model,
      "--env-script",
      args.env_script,
      "--out-dir",
      str(candidate_out_dir),
      "--timeout-s",
      str(args.timeout_s),
      "--max-new-tokens",
      str(args.max_new_tokens),
      "--warmup-runs",
      str(args.warmup_runs),
      "--timed-runs",
      str(args.timed_runs),
  ]
  if args.resident_cache:
    candidate_cmd.append("--resident-cache")
  if args.profile_matvec:
    candidate_cmd.append("--profile-matvec")
  if args.prefill_final_logits_only:
    candidate_cmd.append("--prefill-final-logits-only")
  if args.decode_top1_only:
    candidate_cmd.append("--decode-top1-only")
  if args.full_attention_inplace_history:
    candidate_cmd.append("--full-attention-inplace-history")
  if args.lm_head_top_k:
    candidate_cmd.append("--lm-head-top-k")
  candidate_cmd += ["--lm-head-threads", str(args.lm_head_threads)]
  if args.lm_head_q6_pair_dot:
    candidate_cmd.append("--lm-head-q6-pair-dot")
  if args.expert_slice_matvec:
    candidate_cmd.append("--expert-slice-matvec")
  candidate_cmd += ["--expert-slice-threads", str(args.expert_slice_threads)]
  if args.dense_matvec:
    candidate_cmd.append("--dense-matvec")
  candidate_cmd += ["--dense-matvec-threads", str(args.dense_matvec_threads)]
  candidate_cmd += ["--dense-matvec-min-rows", str(args.dense_matvec_min_rows)]
  if args.dense_matvec_payload_cache:
    candidate_cmd.append("--dense-matvec-payload-cache")
  if args.dense_q4_direct_dot:
    candidate_cmd.append("--dense-q4-direct-dot")
  if args.dense_q4_pair_dot:
    candidate_cmd.append("--dense-q4-pair-dot")
  if args.dense_q6_direct_dot:
    candidate_cmd.append("--dense-q6-direct-dot")
  if args.dense_q6_pair_dot:
    candidate_cmd.append("--dense-q6-pair-dot")
  if args.q4_direct_minsum_pair:
    candidate_cmd.append("--q4-direct-minsum-pair")
  if args.q4_block_meta_cache:
    candidate_cmd.append("--q4-block-meta-cache")
  if args.small_q4_direct_dot:
    candidate_cmd.append("--small-q4-direct-dot")
  if args.matvec_q8_input_reuse:
    candidate_cmd.append("--matvec-q8-input-reuse")
  if args.shared_parallel_executor:
    candidate_cmd.append("--shared-parallel-executor")
  if args.shared_expert_gate_up_fused:
    candidate_cmd.append("--shared-expert-gate-up-fused")
  if args.selected_expert_ffn:
    candidate_cmd.append("--selected-expert-ffn")
  candidate_cmd += [
      "--selected-expert-ffn-threads",
      str(args.selected_expert_ffn_threads),
  ]
  if args.selected_expert_minimal_outputs:
    candidate_cmd.append("--selected-expert-minimal-outputs")
  if args.selected_expert_slice_cache:
    candidate_cmd.append("--selected-expert-slice-cache")
  if args.selected_expert_down_slice_cache:
    candidate_cmd.append("--selected-expert-down-slice-cache")
  if args.selected_expert_down_expert_major:
    candidate_cmd.append("--selected-expert-down-expert-major")
  if args.selected_expert_down_q4_pair_dot:
    candidate_cmd.append("--selected-expert-down-q4-pair-dot")
  if args.selected_expert_down_q6_pair_dot:
    candidate_cmd.append("--selected-expert-down-q6-pair-dot")
  if args.selected_gate_q4_direct_dot:
    candidate_cmd.append("--selected-gate-q4-direct-dot")
  if args.selected_gate_q4_pair_dot:
    candidate_cmd.append("--selected-gate-q4-pair-dot")
  if args.selected_gate_q4_pair_sum_dot:
    candidate_cmd.append("--selected-gate-q4-pair-sum-dot")
  for case_id in args.case_id:
    candidate_cmd += ["--case-id", case_id]

  candidate_run = iq36_local.run(candidate_cmd, args.timeout_s + 120)
  candidate_stdout = load_json(candidate_out_dir / "native-candidate-stdout.json")
  candidate_payload = load_json(candidate_out_dir / "candidate.json")
  candidate_correctness = load_json(candidate_out_dir / "correctness.json")
  gate_payload = load_json(candidate_out_dir / "gate" / "gate.json")
  candidate_rows = (
      iq36_local.load_jsonl(candidate_out_dir / "candidate.jsonl")
      if (candidate_out_dir / "candidate.jsonl").exists()
      else []
  )

  gate_state = gate_payload.get("r1_native_correctness_gate", {})
  timed_rows = timed_case_rows(candidate_stdout)
  benchmark_metadata = {
      "cache_state": candidate_stdout.get("cache_state"),
      "candidate_command": candidate_cmd,
      "candidate_target_run_command": candidate_payload.get("target_run", {}).get("cmd"),
      "decode_top1_only_enabled": candidate_stdout.get("decode_top1_only_enabled"),
      "decode_top1_only_requested": args.decode_top1_only,
      "full_attention_inplace_history_enabled": candidate_stdout.get(
          "full_attention_inplace_history_enabled"
      ),
      "full_attention_inplace_history_requested": args.full_attention_inplace_history,
      "dense_matvec_enabled": candidate_stdout.get("dense_matvec_enabled"),
      "dense_matvec_min_rows": candidate_stdout.get("dense_matvec_min_rows"),
      "dense_matvec_min_rows_requested": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache_enabled": candidate_stdout.get(
          "dense_matvec_payload_cache_enabled"
      ),
      "dense_matvec_payload_cache_requested": args.dense_matvec_payload_cache,
      "dense_matvec_requested": args.dense_matvec,
      "dense_matvec_threads": candidate_stdout.get("dense_matvec_threads"),
      "dense_matvec_threads_requested": args.dense_matvec_threads,
      "dense_q4_direct_dot_enabled": candidate_stdout.get(
          "dense_q4_direct_dot_enabled"
      ),
      "dense_q4_direct_dot_requested": args.dense_q4_direct_dot,
      "dense_q4_pair_dot_enabled": candidate_stdout.get(
          "dense_q4_pair_dot_enabled"
      ),
      "dense_q4_pair_dot_requested": args.dense_q4_pair_dot,
      "dense_q6_direct_dot_enabled": candidate_stdout.get(
          "dense_q6_direct_dot_enabled"
      ),
      "dense_q6_direct_dot_requested": args.dense_q6_direct_dot,
      "dense_q6_pair_dot_enabled": candidate_stdout.get(
          "dense_q6_pair_dot_enabled"
      ),
      "dense_q6_pair_dot_requested": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair_enabled": candidate_stdout.get(
          "q4_direct_minsum_pair_enabled"
      ),
      "q4_direct_minsum_pair_requested": args.q4_direct_minsum_pair,
      "q4_block_meta_cache_enabled": candidate_stdout.get(
          "q4_block_meta_cache_enabled"
      ),
      "q4_block_meta_cache_requested": args.q4_block_meta_cache,
      "small_q4_direct_dot_enabled": candidate_stdout.get(
          "small_q4_direct_dot_enabled"
      ),
      "small_q4_direct_dot_requested": args.small_q4_direct_dot,
      "generated_token_counts": {
          row.get("case_id"): len(row.get("generated_token_ids", []))
          for row in candidate_stdout.get("cases", [])
          if isinstance(row, dict)
      },
      "expert_slice_matvec_enabled": candidate_stdout.get(
          "expert_slice_matvec_enabled"
      ),
      "expert_slice_matvec_requested": args.expert_slice_matvec,
      "expert_slice_threads": candidate_stdout.get("expert_slice_threads"),
      "expert_slice_threads_requested": args.expert_slice_threads,
      "lm_head_threads": candidate_stdout.get("lm_head_threads"),
      "lm_head_threads_requested": args.lm_head_threads,
      "lm_head_q6_pair_dot_enabled": candidate_stdout.get(
          "lm_head_q6_pair_dot_enabled"
      ),
      "lm_head_q6_pair_dot_requested": args.lm_head_q6_pair_dot,
      "lm_head_top_k_enabled": candidate_stdout.get("lm_head_top_k_enabled"),
      "lm_head_top_k_requested": args.lm_head_top_k,
      "matvec_q8_input_reuse_enabled": candidate_stdout.get(
          "matvec_q8_input_reuse_enabled"
      ),
      "matvec_q8_input_reuse_requested": args.matvec_q8_input_reuse,
      "max_new_tokens": args.max_new_tokens,
      "matvec_profile_enabled": candidate_stdout.get("matvec_profile_enabled"),
      "matvec_profile_requested": args.profile_matvec,
      "matvec_profile_top": candidate_stdout.get("matvec_profile", [])[:20],
      "model_path": args.model,
      "precision": "GGUF Q4_K_M native dequant path",
      "prefill_final_logits_only_enabled": candidate_stdout.get(
          "prefill_final_logits_only_enabled"
      ),
      "prefill_final_logits_only_requested": args.prefill_final_logits_only,
      "prompt_token_counts": {
          row.get("case_id"): row.get("prompt_token_count")
          for row in candidate_stdout.get("cases", [])
          if isinstance(row, dict)
      },
      "resident_cache_requested": args.resident_cache,
      "resident_tensor_cache_enabled": candidate_stdout.get(
          "resident_tensor_cache_enabled"
      ),
      "resident_tensor_cache_stats": candidate_stdout.get(
          "resident_tensor_cache_stats", {}
      ),
      "selected_expert_ffn_enabled": candidate_stdout.get(
          "selected_expert_ffn_enabled"
      ),
      "selected_expert_ffn_requested": args.selected_expert_ffn,
      "selected_expert_ffn_threads": candidate_stdout.get(
          "selected_expert_ffn_threads"
      ),
      "selected_expert_ffn_threads_requested": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs_enabled": candidate_stdout.get(
          "selected_expert_minimal_outputs_enabled"
      ),
      "selected_expert_minimal_outputs_requested": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache_enabled": candidate_stdout.get(
          "selected_expert_slice_cache_enabled"
      ),
      "selected_expert_slice_cache_requested": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache_enabled": candidate_stdout.get(
          "selected_expert_down_slice_cache_enabled"
      ),
      "selected_expert_down_slice_cache_requested": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major_enabled": candidate_stdout.get(
          "selected_expert_down_expert_major_enabled"
      ),
      "selected_expert_down_expert_major_requested": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot_enabled": candidate_stdout.get(
          "selected_expert_down_q4_pair_dot_enabled"
      ),
      "selected_expert_down_q4_pair_dot_requested": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot_enabled": candidate_stdout.get(
          "selected_expert_down_q6_pair_dot_enabled"
      ),
      "selected_expert_down_q6_pair_dot_requested": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot_enabled": candidate_stdout.get(
          "selected_gate_q4_direct_dot_enabled"
      ),
      "selected_gate_q4_direct_dot_requested": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot_enabled": candidate_stdout.get(
          "selected_gate_q4_pair_dot_enabled"
      ),
      "selected_gate_q4_pair_dot_requested": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot_enabled": candidate_stdout.get(
          "selected_gate_q4_pair_sum_dot_enabled"
      ),
      "selected_gate_q4_pair_sum_dot_requested": args.selected_gate_q4_pair_sum_dot,
      "shared_parallel_executor_enabled": candidate_stdout.get(
          "shared_parallel_executor_enabled"
      ),
      "shared_parallel_executor_requested": args.shared_parallel_executor,
      "shared_expert_gate_up_fused_enabled": candidate_stdout.get(
          "shared_expert_gate_up_fused_enabled"
      ),
      "shared_expert_gate_up_fused_requested": args.shared_expert_gate_up_fused,
      "timed_runs": args.timed_runs,
      "warmup_runs": args.warmup_runs,
  }

  diag = {
      "candidate_artifact": rel(candidate_out_dir),
      "candidate_row_count": len(candidate_rows),
      "case_ids": [row.get("case_id") for row in candidate_rows],
      "decode_top1_only": args.decode_top1_only,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "expert_slice_matvec": args.expert_slice_matvec,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "expert_slice_threads": args.expert_slice_threads,
      "host_metadata": host_metadata,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "r1_native_correctness_gate_closed": (
          gate_state.get("r1_native_correctness_gate_closed") is True
      ),
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "timed_case_row_count": len(timed_rows),
      "timed_runs": args.timed_runs,
      "warmup_runs": args.warmup_runs,
  }
  payload = {
      "benchmark_metadata": benchmark_metadata,
      "candidate_correctness": candidate_correctness,
      "candidate_run": candidate_run,
      "created_at": created_at,
      "diagnostic": diag,
      "host": args.host,
      "model_path": args.model,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "timed_case_rows": timed_rows,
      "workstream": WORKSTREAM,
  }

  checks = [
      {"name": "host_metadata_collected", "pass": host_metadata.get("uname_a", {}).get("returncode") == 0},
      {"name": "native_candidate_runner_executed", "pass": candidate_run.get("returncode") == 0},
      {"name": "native_candidate_artifact_present", "pass": candidate_correctness != {}},
      {
          "name": "native_candidate_generation_checks_passed",
          "pass": candidate_correctness.get("required_checks_passed") is True,
      },
      {
          "name": "r1_native_correctness_gate_remains_closed",
          "pass": gate_state.get("r1_native_correctness_gate_closed") is True,
      },
      {
          "name": "timing_schema_present",
          "pass": timing_present(candidate_stdout, args.timed_runs),
      },
      {
          "name": "warmup_metadata_recorded",
          "pass": candidate_stdout.get("warmup_run_count") == args.warmup_runs
          and isinstance(candidate_stdout.get("warmup_runs"), list),
      },
      {
          "name": "benchmark_metadata_recorded",
          "pass": bool(benchmark_metadata["prompt_token_counts"])
          and bool(benchmark_metadata["generated_token_counts"])
          and benchmark_metadata["candidate_target_run_command"] is not None,
      },
      {
          "name": "resident_cache_state_recorded",
          "pass": benchmark_metadata["resident_tensor_cache_enabled"] is args.resident_cache
          and isinstance(benchmark_metadata["resident_tensor_cache_stats"], dict),
      },
      {
          "name": "matvec_profile_recorded",
          "pass": benchmark_metadata["matvec_profile_enabled"] is args.profile_matvec
          and isinstance(benchmark_metadata["matvec_profile_top"], list)
          and (not args.profile_matvec or bool(benchmark_metadata["matvec_profile_top"])),
      },
      {
          "name": "prefill_final_logits_only_recorded",
          "pass": benchmark_metadata["prefill_final_logits_only_enabled"]
          is args.prefill_final_logits_only,
      },
      {
          "name": "decode_top1_only_recorded",
          "pass": benchmark_metadata["decode_top1_only_enabled"]
          is args.decode_top1_only,
      },
      {
          "name": "full_attention_inplace_history_recorded",
          "pass": benchmark_metadata["full_attention_inplace_history_enabled"]
          is args.full_attention_inplace_history,
      },
      {
          "name": "lm_head_route_recorded",
          "pass": benchmark_metadata["lm_head_top_k_enabled"] is args.lm_head_top_k
          and benchmark_metadata["lm_head_threads"] == args.lm_head_threads,
      },
      {
          "name": "lm_head_q6_pair_dot_recorded",
          "pass": benchmark_metadata["lm_head_q6_pair_dot_enabled"]
          is args.lm_head_q6_pair_dot,
      },
      {
          "name": "expert_slice_route_recorded",
          "pass": benchmark_metadata["expert_slice_matvec_enabled"]
          is args.expert_slice_matvec
          and benchmark_metadata["expert_slice_threads"] == args.expert_slice_threads,
      },
      {
          "name": "dense_matvec_route_recorded",
          "pass": benchmark_metadata["dense_matvec_enabled"] is args.dense_matvec
          and benchmark_metadata["dense_matvec_min_rows"] == args.dense_matvec_min_rows
          and benchmark_metadata["dense_matvec_threads"] == args.dense_matvec_threads,
      },
      {
          "name": "dense_matvec_payload_cache_recorded",
          "pass": benchmark_metadata["dense_matvec_payload_cache_enabled"]
          is args.dense_matvec_payload_cache,
      },
      {
          "name": "dense_q4_direct_dot_recorded",
          "pass": benchmark_metadata["dense_q4_direct_dot_enabled"]
          is args.dense_q4_direct_dot,
      },
      {
          "name": "dense_q4_pair_dot_recorded",
          "pass": benchmark_metadata["dense_q4_pair_dot_enabled"]
          is args.dense_q4_pair_dot,
      },
      {
          "name": "dense_q6_direct_dot_recorded",
          "pass": benchmark_metadata["dense_q6_direct_dot_enabled"]
          is args.dense_q6_direct_dot,
      },
      {
          "name": "dense_q6_pair_dot_recorded",
          "pass": benchmark_metadata["dense_q6_pair_dot_enabled"]
          is args.dense_q6_pair_dot,
      },
      {
          "name": "q4_direct_minsum_pair_recorded",
          "pass": benchmark_metadata["q4_direct_minsum_pair_enabled"]
          is args.q4_direct_minsum_pair,
      },
      {
          "name": "q4_block_meta_cache_recorded",
          "pass": benchmark_metadata["q4_block_meta_cache_enabled"]
          is args.q4_block_meta_cache,
      },
      {
          "name": "small_q4_direct_dot_recorded",
          "pass": benchmark_metadata["small_q4_direct_dot_enabled"]
          is args.small_q4_direct_dot,
      },
      {
          "name": "matvec_q8_input_reuse_recorded",
          "pass": benchmark_metadata["matvec_q8_input_reuse_enabled"]
          is args.matvec_q8_input_reuse,
      },
      {
          "name": "shared_parallel_executor_recorded",
          "pass": benchmark_metadata["shared_parallel_executor_enabled"]
          is args.shared_parallel_executor,
      },
      {
          "name": "shared_expert_gate_up_fused_recorded",
          "pass": benchmark_metadata["shared_expert_gate_up_fused_enabled"]
          is args.shared_expert_gate_up_fused,
      },
      {
          "name": "selected_expert_ffn_route_recorded",
          "pass": benchmark_metadata["selected_expert_ffn_enabled"]
          is args.selected_expert_ffn
          and benchmark_metadata["selected_expert_ffn_threads"]
          == args.selected_expert_ffn_threads,
      },
      {
          "name": "selected_expert_minimal_outputs_recorded",
          "pass": benchmark_metadata["selected_expert_minimal_outputs_enabled"]
          is args.selected_expert_minimal_outputs,
      },
      {
          "name": "selected_expert_slice_cache_recorded",
          "pass": benchmark_metadata["selected_expert_slice_cache_enabled"]
          is args.selected_expert_slice_cache,
      },
      {
          "name": "selected_expert_down_slice_cache_recorded",
          "pass": benchmark_metadata["selected_expert_down_slice_cache_enabled"]
          is args.selected_expert_down_slice_cache,
      },
      {
          "name": "selected_expert_down_expert_major_recorded",
          "pass": benchmark_metadata["selected_expert_down_expert_major_enabled"]
          is args.selected_expert_down_expert_major,
      },
      {
          "name": "selected_expert_down_q4_pair_dot_recorded",
          "pass": benchmark_metadata["selected_expert_down_q4_pair_dot_enabled"]
          is args.selected_expert_down_q4_pair_dot,
      },
      {
          "name": "selected_expert_down_q6_pair_dot_recorded",
          "pass": benchmark_metadata["selected_expert_down_q6_pair_dot_enabled"]
          is args.selected_expert_down_q6_pair_dot,
      },
      {
          "name": "selected_gate_q4_direct_dot_recorded",
          "pass": benchmark_metadata["selected_gate_q4_direct_dot_enabled"]
          is args.selected_gate_q4_direct_dot,
      },
      {
          "name": "selected_gate_q4_pair_dot_recorded",
          "pass": benchmark_metadata["selected_gate_q4_pair_dot_enabled"]
          is args.selected_gate_q4_pair_dot,
      },
      {
          "name": "selected_gate_q4_pair_sum_dot_recorded",
          "pass": benchmark_metadata["selected_gate_q4_pair_sum_dot_enabled"]
          is args.selected_gate_q4_pair_sum_dot,
      },
      {"name": "speedup_claims_forbidden", "pass": True},
  ]

  iq36_local.write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "candidate_artifact": rel(candidate_out_dir),
      "decode_top1_only": args.decode_top1_only,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "expert_slice_matvec": args.expert_slice_matvec,
      "expert_slice_threads": args.expert_slice_threads,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "host": args.host,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "max_new_tokens": args.max_new_tokens,
      "model_path": args.model,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "profile_matvec": args.profile_matvec,
      "resident_cache": args.resident_cache,
      "schema_version": SCHEMA_VERSION,
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "speedup_claims_allowed": False,
      "timed_runs": args.timed_runs,
      "tool": "tools/intel-qwen36-post-r1-resident-timed.py",
      "warmup_runs": args.warmup_runs,
      "workstream": WORKSTREAM,
  })
  iq36_local.write_json(out_dir / "diagnostic.json", payload)
  iq36_local.write_json(out_dir / "host-runtime.json", host_metadata)
  iq36_local.write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "post_r1_resident_timed_diagnostic",
      "decode_top1_only": args.decode_top1_only,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "post_r1_resident_timed_artifact_created": True,
      "r1_native_correctness_gate_closed": (
          gate_state.get("r1_native_correctness_gate_closed") is True
      ),
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })

  metrics: list[tuple[str, Any]] = [
      ("warmup_runs", args.warmup_runs),
      ("timed_runs", args.timed_runs),
      ("timed_case_row_count", len(timed_rows)),
      ("candidate_row_count", len(candidate_rows)),
      ("decode_top1_only", args.decode_top1_only),
      (
          "full_attention_inplace_history",
          args.full_attention_inplace_history,
      ),
      ("dense_matvec", args.dense_matvec),
      ("dense_matvec_min_rows", args.dense_matvec_min_rows),
      ("dense_matvec_payload_cache", args.dense_matvec_payload_cache),
      ("dense_matvec_threads", args.dense_matvec_threads),
      ("dense_q4_direct_dot", args.dense_q4_direct_dot),
      ("dense_q4_pair_dot", args.dense_q4_pair_dot),
      ("dense_q6_direct_dot", args.dense_q6_direct_dot),
      ("dense_q6_pair_dot", args.dense_q6_pair_dot),
      ("q4_direct_minsum_pair", args.q4_direct_minsum_pair),
      ("q4_block_meta_cache", args.q4_block_meta_cache),
      ("small_q4_direct_dot", args.small_q4_direct_dot),
      ("matvec_q8_input_reuse", args.matvec_q8_input_reuse),
      ("expert_slice_matvec", args.expert_slice_matvec),
      ("expert_slice_threads", args.expert_slice_threads),
      ("lm_head_q6_pair_dot", args.lm_head_q6_pair_dot),
      ("lm_head_threads", args.lm_head_threads),
      ("lm_head_top_k", args.lm_head_top_k),
      ("profile_matvec", args.profile_matvec),
      ("prefill_final_logits_only", args.prefill_final_logits_only),
      ("resident_cache", args.resident_cache),
      ("selected_expert_ffn", args.selected_expert_ffn),
      ("selected_expert_ffn_threads", args.selected_expert_ffn_threads),
      ("selected_expert_minimal_outputs", args.selected_expert_minimal_outputs),
      ("selected_expert_slice_cache", args.selected_expert_slice_cache),
      ("selected_expert_down_slice_cache", args.selected_expert_down_slice_cache),
      ("selected_expert_down_expert_major", args.selected_expert_down_expert_major),
      ("selected_expert_down_q4_pair_dot", args.selected_expert_down_q4_pair_dot),
      ("selected_expert_down_q6_pair_dot", args.selected_expert_down_q6_pair_dot),
      ("selected_gate_q4_direct_dot", args.selected_gate_q4_direct_dot),
      ("selected_gate_q4_pair_dot", args.selected_gate_q4_pair_dot),
      ("selected_gate_q4_pair_sum_dot", args.selected_gate_q4_pair_sum_dot),
      ("shared_expert_gate_up_fused", args.shared_expert_gate_up_fused),
      ("shared_parallel_executor", args.shared_parallel_executor),
      ("r1_native_correctness_gate_closed", gate_state.get("r1_native_correctness_gate_closed") is True),
  ]
  for run in candidate_stdout.get("timed_runs", []):
    if isinstance(run, dict):
      metrics.append((f"timed_run_{run.get('run_index')}_total_ns", run.get("total_ns")))
  iq36_local.write_metric(out_dir / "metrics.jsonl", "post_r1_resident_timed", metrics)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")

  print(f"post-R1 resident timed diagnostic output: {out_dir}")
  return 0 if all(check["pass"] for check in checks) else 1


if __name__ == "__main__":
  raise SystemExit(main())
