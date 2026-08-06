#!/usr/bin/env python3
"""Run the R2 native 1k-8k x 512 speed-denominator matrix.

This is an evidence collector, not a speedup-claim tool. It runs the accepted
post-R1 native route against materialized prompt-token buckets, captures native
prefill and continuation decode timing, and aligns each row to the acceptance
floor plus the R0 KV-read roofline.

The runner's timing model is important:

  * prompt_prefill covers all prompt tokens and the first generated token's
    final-logit top-k.
  * decode_continuation covers generated token 2..N.

So decode tok/s is computed from generated_token_count - 1, not from the full
generated token count.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r2-native-matrix-v0"
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
)
DEFAULT_KV_PRESSURE = ROOT / "output/r0-kv-read-pressure-20260626T043722Z/budget.json"
DEFAULT_BUCKETS = ("001k", "002k", "004k", "008k")
BUCKET_TOKEN_COUNTS = {
    "001k": 1024,
    "002k": 2048,
    "004k": 4096,
    "008k": 8192,
}
REQUIRED_R2_BUCKETS = tuple(BUCKET_TOKEN_COUNTS.values())


def load_context_tool() -> ModuleType:
  path = ROOT / "tools/intel-qwen36-context-ladder-native-diagnostic.py"
  spec = importlib.util.spec_from_file_location("iq36_context_ladder_native_diagnostic", path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"unable to load context ladder helper: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


CTX = load_context_tool()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=CTX.DEFAULT_HOST)
  parser.add_argument("--model", default=CTX.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=CTX.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=CTX.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--token-id-refs", type=Path, default=CTX.DEFAULT_TOKEN_ID_REFS)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--kv-pressure", type=Path, default=DEFAULT_KV_PRESSURE)
  parser.add_argument("--kv-dtype", default="fp16_or_bf16")
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=7200)
  parser.add_argument("--max-new-tokens", type=int, default=512)
  parser.add_argument("--warmup-runs", type=int, default=0)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument(
      "--case-id",
      action="append",
      default=[],
      help="Case id from the token-id reference JSONL. Repeatable.",
  )
  parser.add_argument(
      "--bucket",
      action="append",
      default=[],
      help=(
          "Run sentinel_<bucket> and prefill_shape_<bucket>, for example 001k. "
          "Repeatable. Defaults to 001k, 002k, 004k, and 008k."
      ),
  )
  parser.add_argument(
      "--combined-process",
      action="store_true",
      help="Run selected cases in one target process instead of one process per case.",
  )
  parser.add_argument(
      "--no-resident-cache",
      dest="resident_cache",
      action="store_false",
      help="Disable resident tensor/slice caches in the native runner.",
  )
  parser.set_defaults(resident_cache=True)
  parser.add_argument(
      "--profile-matvec",
      action="store_true",
      help="Enable per-tensor matvec profiling. Disabled by default for timing.",
  )
  parser.add_argument(
      "--dense-q6-pair-dot",
      dest="dense_q6_pair_dot",
      action="store_true",
      help="Enable the dense Q6 row-pair dot route.",
  )
  parser.add_argument(
      "--no-dense-q6-pair-dot",
      dest="dense_q6_pair_dot",
      action="store_false",
      help="Disable the dense Q6 row-pair dot route.",
  )
  parser.set_defaults(dense_q6_pair_dot=True)
  parser.add_argument(
      "--selected-expert-down-q4-pair-dot",
      action="store_true",
      help="Enable the selected-expert down Q4 row-pair dot route.",
  )
  parser.add_argument(
      "--selected-expert-down-q6-pair-dot",
      action="store_true",
      help="Enable the selected-expert down Q6 row-pair dot route.",
  )
  parser.add_argument(
      "--q4-plane-layout",
      action="store_true",
      help="Enable the R3 q4k_plane_v0 layout route for selected Q4_K lanes.",
  )
  parser.add_argument(
      "--dense-q4-plane-pair-dot",
      action="store_true",
      help="Enable fused q4-plane pair dots for generic dense Q4_K rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-plane-pair-dot",
      action="store_true",
      help="Enable fused q4-plane pair dots for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--allow-partial-selected-gate-q4-plane-route",
      action="store_true",
      help=(
          "Allow selected-gate q4-plane pair timing without the accepted "
          "selected-expert-down Q6 companion route. This is exploratory and "
          "not comparable with the packaged R3 candidate."
      ),
  )
  parser.add_argument(
      "--allow-short-generation",
      action="store_true",
      help="Allow EOS before --max-new-tokens while still writing the artifact.",
  )
  parser.add_argument(
      "--ignore-eos",
      action="store_true",
      help="Force benchmark output to --max-new-tokens even after EOS.",
  )
  return parser.parse_args()


def case_ids_from_args(args: argparse.Namespace) -> list[str]:
  if args.case_id:
    return list(dict.fromkeys(args.case_id))
  buckets = args.bucket or list(DEFAULT_BUCKETS)
  out: list[str] = []
  for bucket in buckets:
    if bucket not in BUCKET_TOKEN_COUNTS:
      raise SystemExit(f"unsupported R2 bucket: {bucket}")
    out.append(f"sentinel_{bucket}")
    out.append(f"prefill_shape_{bucket}")
  return list(dict.fromkeys(out))


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def ns_to_s(value: Any) -> float | None:
  if not isinstance(value, int) or value < 0:
    return None
  return value / 1e9


def ratio(num: float | None, den: float | None) -> float | None:
  if num is None or den is None or den == 0:
    return None
  return num / den


def rounded(value: float | None, digits: int = 6) -> float | None:
  if value is None:
    return None
  return round(value, digits)


def floor_for(acc: dict[str, Any], bucket: int) -> dict[str, Any]:
  fresh = acc.get("same_host_floor", {})
  fresh_prefill = fresh.get("prefill_tokens_s", {}).get(str(bucket))
  fresh_decode = fresh.get("decode_tokens_s", {}).get(str(bucket))
  if fresh_prefill is not None and fresh_decode is not None:
    return {
        "artifact": fresh.get("artifacts", {}).get(str(bucket)),
        "bucket": bucket,
        "basis": fresh.get("basis"),
        "decode_tok_s": fresh_decode,
        "is_bootstrap_placeholder": fresh.get("is_bootstrap_placeholder") is not False,
        "prefill_tok_s": fresh_prefill,
        "refreshed_at": fresh.get("refreshed_at"),
        "same_host_floor_required": acc.get("r0_target_policy", {}).get(
            "same_host_floor_required", True
        ),
        "source": str(DEFAULT_ACCEPTANCE.relative_to(ROOT)),
        "status": fresh.get("status"),
    }
  bootstrap = acc.get("bootstrap_targets", {})
  prefill = bootstrap.get("prefill_tokens_s", {}).get(str(bucket))
  decode = bootstrap.get("decode_tokens_s", {}).get(str(bucket))
  policy = acc.get("r0_target_policy", {})
  return {
      "bucket": bucket,
      "basis": policy.get("bootstrap_target_basis"),
      "decode_tok_s": decode,
      "is_bootstrap_placeholder": True,
      "prefill_tok_s": prefill,
      "refresh_required_before_product_claim": policy.get(
          "refresh_required_before_product_claim", True
      ),
      "same_host_floor_required": policy.get("same_host_floor_required", True),
      "source": str(DEFAULT_ACCEPTANCE.relative_to(ROOT)),
  }


def roofline_for(kv: dict[str, Any], bucket: int, dtype: str) -> dict[str, Any]:
  for row in kv.get("rows", []):
    if row.get("bucket") == bucket and row.get("dtype") == dtype:
      return {
          "bucket": bucket,
          "decode_tok_s_at_qmatvec_max": row.get("ceiling_tok_s_at_qmatvec_max"),
          "decode_tok_s_at_source_stream_max": row.get(
              "ceiling_tok_s_at_source_stream_max"
          ),
          "decode_tok_s_at_target_line": row.get("ceiling_tok_s_at_target_line"),
          "decode_tok_s_at_target_ratio_line": row.get(
              "ceiling_tok_s_at_target_ratio_line"
          ),
          "dtype": dtype,
          "kv_read_gb_per_decode_token": row.get("kv_read_gb_per_decode_token"),
          "source": str(DEFAULT_KV_PRESSURE.relative_to(ROOT)),
      }
  return {
      "bucket": bucket,
      "decode_tok_s_at_qmatvec_max": None,
      "dtype": dtype,
      "source": str(DEFAULT_KV_PRESSURE.relative_to(ROOT)),
  }


def enrich_case_rows(
    parsed_runs: list[dict[str, Any]],
    token_manifest: dict[str, Any],
    acc: dict[str, Any],
    kv: dict[str, Any],
    kv_dtype: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for row in CTX.normalize_case_results(parsed_runs, token_manifest):
    case_id = row.get("case_id")
    prompt_tokens = row.get("prompt_token_count")
    generated_tokens = row.get("generated_token_count")
    prompt_prefill_ns = row.get("prompt_prefill_ns")
    decode_ns = row.get("decode_continuation_ns")
    prefill_s = ns_to_s(prompt_prefill_ns)
    decode_s = ns_to_s(decode_ns)
    continuation_tokens = (
        max(0, int(generated_tokens) - 1) if isinstance(generated_tokens, int) else None
    )
    bucket = (
        int(prompt_tokens)
        if isinstance(prompt_tokens, int) and prompt_tokens in REQUIRED_R2_BUCKETS
        else row.get("target_prompt_tokens")
    )
    if not isinstance(bucket, int):
      bucket = int(prompt_tokens) if isinstance(prompt_tokens, int) else 0
    floor = floor_for(acc, bucket)
    roofline = roofline_for(kv, bucket, kv_dtype)
    prefill_tok_s = (
        prompt_tokens / prefill_s
        if isinstance(prompt_tokens, int) and prompt_tokens > 0 and prefill_s
        else None
    )
    decode_tok_s = (
        continuation_tokens / decode_s
        if isinstance(continuation_tokens, int)
        and continuation_tokens > 0
        and decode_s
        and decode_s > 0
        else None
    )
    total_s = None
    if isinstance(prompt_prefill_ns, int) and isinstance(decode_ns, int):
      total_s = (prompt_prefill_ns + decode_ns) / 1e9
    end_to_end_generated_tok_s = (
        generated_tokens / total_s
        if isinstance(generated_tokens, int) and total_s and total_s > 0
        else None
    )
    rows.append({
        "bucket": bucket,
        "case_id": case_id,
        "decode_continuation_ns": decode_ns,
        "decode_continuation_output_tokens": continuation_tokens,
        "decode_continuation_s": rounded(decode_s, 9),
        "decode_continuation_tok_s": rounded(decode_tok_s, 6),
        "decode_roofline_util": rounded(
            ratio(decode_tok_s, roofline.get("decode_tok_s_at_qmatvec_max")),
            8,
        ),
        "decode_target_bar_util": rounded(
            ratio(decode_tok_s, roofline.get("decode_tok_s_at_target_ratio_line")),
            8,
        ),
        "decode_vs_floor": rounded(ratio(decode_tok_s, floor.get("decode_tok_s")), 8),
        "end_to_end_generated_tok_s": rounded(end_to_end_generated_tok_s, 6),
        "first_generated_token_id": row.get("first_generated_token_id"),
        "floor": floor,
        "generated_token_count": generated_tokens,
        "generated_token_count_matches_requested": generated_tokens == max_new_tokens,
        "kind": row.get("kind"),
        "last_generated_token_id": (
            row.get("generated_token_ids", [None])[-1]
            if row.get("generated_token_ids")
            else None
        ),
        "prefill_s": rounded(prefill_s, 9),
        "prefill_tok_s": rounded(prefill_tok_s, 6),
        "prefill_vs_floor": rounded(ratio(prefill_tok_s, floor.get("prefill_tok_s")), 8),
        "prompt_prefill_ns": prompt_prefill_ns,
        "prompt_token_count": prompt_tokens,
        "requested_output_tokens": max_new_tokens,
        "roofline": roofline,
        "run_index": row.get("run_index"),
        "target_prompt_tokens": row.get("target_prompt_tokens"),
        "timing_ns": row.get("timing_ns"),
        "top_logprob_id_signature": row.get("top_logprob_id_signature"),
    })
  return rows


def build_summary(payload: dict[str, Any]) -> str:
  matrix = payload["matrix"]
  rows = matrix["case_results"]
  lines = [
      "# R2 Native Matrix",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- route: `{matrix['route']}`",
      f"- q4 plane layout: `{str(matrix['q4_plane_layout']).lower()}`",
      f"- dense Q4 plane-pair dot: `{str(matrix['dense_q4_plane_pair_dot']).lower()}`",
      f"- selected gate Q4 plane-pair dot: `{str(matrix['selected_gate_q4_plane_pair_dot']).lower()}`",
      f"- selected expert down Q6 pair dot: `{str(matrix['selected_expert_down_q6_pair_dot']).lower()}`",
      f"- max new tokens requested: `{matrix['max_new_tokens']}`",
      f"- ignore EOS: `{str(matrix['ignore_eos']).lower()}`",
      f"- case process isolation: `{str(matrix['case_process_isolation']).lower()}`",
      f"- floor placeholder: `{str(matrix['floor_is_bootstrap_placeholder']).lower()}`",
      f"- r2 exit gate closed: `{str(payload['r2_exit_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | bucket | kind | gen | prefill tok/s | decode tok/s | vs floor | roofline util |",
      "|---|---:|---|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("bucket")),
            str(row.get("kind")),
            str(row.get("generated_token_count")),
            str(row.get("prefill_tok_s")),
            str(row.get("decode_continuation_tok_s")),
            str(row.get("decode_vs_floor")),
            str(row.get("decode_roofline_util")),
        ])
        + " |"
    )
  lines += [
      "",
      "This artifact is an R2 denominator input. It does not permit speedup claims",
      "until the same-host floor is refreshed and the R2 exit checks close.",
      "",
  ]
  return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> int:
  args = parse_args()
  if args.max_new_tokens < 1 or args.max_new_tokens > 512:
    raise SystemExit("--max-new-tokens must be 1..512")
  if args.warmup_runs != 0:
    raise SystemExit("R2 native matrix must use --warmup-runs 0 for cold no-prefix")
  if args.timed_runs < 1 or args.timed_runs > 8:
    raise SystemExit("--timed-runs must be 1..8")
  if not args.token_id_refs.exists():
    raise SystemExit(f"token-id refs missing: {args.token_id_refs}")
  if not args.acceptance.exists():
    raise SystemExit(f"acceptance matrix missing: {args.acceptance}")
  if not args.kv_pressure.exists():
    raise SystemExit(f"kv pressure budget missing: {args.kv_pressure}")
  if args.dense_q4_plane_pair_dot and not args.q4_plane_layout:
    raise SystemExit("--dense-q4-plane-pair-dot requires --q4-plane-layout")
  if args.selected_gate_q4_plane_pair_dot:
    if not args.q4_plane_layout:
      raise SystemExit("--selected-gate-q4-plane-pair-dot requires --q4-plane-layout")
    if (
        not args.selected_expert_down_q6_pair_dot
        and not args.allow_partial_selected_gate_q4_plane_route
    ):
      raise SystemExit(
          "--selected-gate-q4-plane-pair-dot candidate timing must include "
          "--selected-expert-down-q6-pair-dot; pass "
          "--allow-partial-selected-gate-q4-plane-route for exploratory "
          "non-comparable probes"
      )

  acceptance = read_json(args.acceptance)
  kv_pressure = read_json(args.kv_pressure)
  created_at = CTX.iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r2-native-matrix-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  token_dir = out_dir / "token-input"
  stdout_dir = out_dir / "native-stdout"
  stdout_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/r2-native-matrix-{stamp}"
  remote_token_dir = f"{remote_dir}/tokens"

  case_ids = case_ids_from_args(args)
  token_rows = CTX.selected_rows(args.token_id_refs, case_ids)
  token_manifest = CTX.prepare_token_inputs(token_rows, token_dir)
  mkdir, source_transfers = CTX.source_stage(args.host, remote_dir, args.timeout_s)
  token_transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0 and all(item.get("returncode") == 0 for item in source_transfers):
    token_transfers = CTX.token_stage(
        args.host,
        token_dir,
        remote_token_dir,
        token_manifest,
        args.timeout_s,
    )
  staged = (
      mkdir.get("returncode") == 0
      and all(item.get("returncode") == 0 for item in source_transfers)
      and bool(token_transfers)
      and all(item.get("returncode") == 0 for item in token_transfers)
  )
  build = (
      iq36_local.run_target(
          args.host,
          f"bash -lc {shlex.quote(CTX.build_command(remote_dir, args.env_script))}",
          args.timeout_s,
      )
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )

  route_args = CTX.accepted_route_args(args)
  if args.ignore_eos:
    route_args.append("--ignore-eos")
  run_case_groups = [case_ids] if args.combined_process else [[case_id] for case_id in case_ids]
  target_runs: list[dict[str, Any]] = []
  parsed_runs: list[dict[str, Any]] = []
  parse_errors: list[dict[str, Any]] = []
  if build.get("returncode") == 0:
    for group in run_case_groups:
      run_command = CTX.run_native_command(
          remote_dir,
          remote_token_dir,
          args.model,
          args.max_new_tokens,
          route_args,
          group,
      )
      run_result = iq36_local.run_target(args.host, run_command, args.timeout_s)
      target_runs.append(CTX.stdout_run_record(run_result))
      parsed: dict[str, Any] = {}
      try:
        parsed = CTX.parse_stdout(run_result.get("stdout", ""))
      except json.JSONDecodeError as exc:
        parse_errors.append({"case_ids": group, "error": str(exc)})
      if parsed:
        parsed_runs.append(parsed)
        CTX.write_json(stdout_dir / ("+".join(group) + ".json"), parsed)

  case_results = enrich_case_rows(
      parsed_runs,
      token_manifest,
      acceptance,
      kv_pressure,
      args.kv_dtype,
      args.max_new_tokens,
  )
  observed_buckets = sorted({
      int(row["bucket"])
      for row in case_results
      if isinstance(row.get("bucket"), int)
  })
  expected_case_count = len(case_ids)
  full_r2_bucket_set = observed_buckets == list(REQUIRED_R2_BUCKETS)
  all_full_generation = all(
      row.get("generated_token_count_matches_requested") is True for row in case_results
  ) and len(case_results) == expected_case_count
  route_flags_ok = all(
      parsed.get("prefill_final_logits_only_enabled") is True
      and parsed.get("full_attention_inplace_history_enabled") is True
      and parsed.get("lm_head_top_k_enabled") is True
      and parsed.get("dense_matvec_enabled") is True
      and parsed.get("dense_q4_direct_dot_enabled") is True
      and parsed.get("dense_q6_direct_dot_enabled") is True
      and parsed.get("dense_q6_pair_dot_enabled") is args.dense_q6_pair_dot
      and parsed.get("dense_q4_plane_pair_dot_enabled")
      is args.dense_q4_plane_pair_dot
      and parsed.get("selected_expert_down_q4_pair_dot_enabled")
      is args.selected_expert_down_q4_pair_dot
      and parsed.get("selected_expert_down_q6_pair_dot_enabled")
      is args.selected_expert_down_q6_pair_dot
      and parsed.get("q4_plane_layout_enabled") is args.q4_plane_layout
      and parsed.get("selected_gate_q4_plane_pair_dot_enabled")
      is args.selected_gate_q4_plane_pair_dot
      and parsed.get("selected_expert_ffn_enabled") is True
      and parsed.get("selected_gate_q4_pair_dot_enabled") is True
      for parsed in parsed_runs
  ) and bool(parsed_runs)
  run_checks = [
      {"name": "token_id_refs_present", "pass": args.token_id_refs.exists()},
      {"name": "acceptance_matrix_present", "pass": args.acceptance.exists()},
      {"name": "kv_pressure_budget_present", "pass": args.kv_pressure.exists()},
      {"name": "selected_cases_present", "pass": len(token_rows) == expected_case_count},
      {"name": "token_inputs_prepared", "pass": token_manifest["case_count"] == expected_case_count},
      {"name": "cold_no_prefix_no_warmup", "pass": args.warmup_runs == 0},
      {"name": "case_process_isolation_enabled", "pass": not args.combined_process},
      {
          "name": "dense_q4_plane_pair_route_complete",
          "pass": not args.dense_q4_plane_pair_dot or args.q4_plane_layout,
      },
      {
          "name": "selected_gate_q4_plane_pair_route_complete",
          "pass": (
              not args.selected_gate_q4_plane_pair_dot
              or (
                  args.q4_plane_layout
                  and (
                      args.selected_expert_down_q6_pair_dot
                      or args.allow_partial_selected_gate_q4_plane_route
                  )
              )
          ),
          "partial_route_allowed": args.allow_partial_selected_gate_q4_plane_route,
      },
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(item.get("returncode") == 0 for item in source_transfers),
      },
      {
          "name": "token_inputs_transferred",
          "pass": bool(token_transfers) and all(item.get("returncode") == 0 for item in token_transfers),
      },
      {"name": "target_native_runner_built", "pass": build.get("returncode") == 0},
      {
          "name": "target_native_runner_ran_all_groups",
          "pass": len(target_runs) == len(run_case_groups)
          and all(item.get("returncode") == 0 for item in target_runs),
      },
      {"name": "target_stdout_parsed_all_groups", "pass": len(parsed_runs) == len(run_case_groups)},
      {"name": "case_timing_rows_present", "pass": len(case_results) == expected_case_count},
      {
          "name": "prompt_counts_match_token_refs",
          "pass": all(
              token_manifest["cases"][row["case_id"]]["prompt_token_count"]
              == row.get("prompt_token_count")
              for row in case_results
              if row.get("case_id") in token_manifest["cases"]
          ) and len(case_results) == expected_case_count,
      },
      {"name": "accepted_route_flags_enabled", "pass": route_flags_ok},
      {
          "name": "ignore_eos_policy_recorded",
          "pass": all(parsed.get("ignore_eos_enabled") is args.ignore_eos for parsed in parsed_runs)
          and bool(parsed_runs),
          "ignore_eos": args.ignore_eos,
      },
      {
          "name": "generated_token_count_matches_requested",
          "pass": args.allow_short_generation or all_full_generation,
          "requested_output_tokens": args.max_new_tokens,
      },
      {
          "name": "all_rows_have_positive_prefill",
          "pass": all(
              isinstance(row.get("prompt_prefill_ns"), int)
              and row["prompt_prefill_ns"] > 0
              for row in case_results
          ) and bool(case_results),
      },
      {
          "name": "all_continuation_decode_rows_have_timing",
          "pass": all(
              (
                  row.get("decode_continuation_output_tokens") == 0
                  and isinstance(row.get("decode_continuation_ns"), int)
              )
              or (
                  isinstance(row.get("decode_continuation_output_tokens"), int)
                  and row["decode_continuation_output_tokens"] > 0
                  and isinstance(row.get("decode_continuation_ns"), int)
                  and row["decode_continuation_ns"] > 0
              )
              for row in case_results
          ) and bool(case_results),
      },
      {
          "name": "selected_buckets_have_floor_and_roofline",
          "pass": all(
              row.get("floor", {}).get("decode_tok_s") is not None
              and row.get("roofline", {}).get("decode_tok_s_at_qmatvec_max") is not None
              for row in case_results
          ) and bool(case_results),
      },
  ]
  r2_exit_checks = [
      {"name": "r2_required_1k_8k_buckets_covered", "pass": full_r2_bucket_set},
      {
          "name": "r2_expected_case_count_2_per_bucket",
          "pass": len(case_results) == len(REQUIRED_R2_BUCKETS) * 2,
      },
      {
          "name": "r2_output_tokens_512_all_rows",
          "pass": args.max_new_tokens == 512 and all_full_generation,
      },
      {
          "name": "same_host_floor_refreshed_not_bootstrap",
          "pass": False,
          "status": "pending_refresh",
      },
      {
          "name": "all_rows_report_decode_vs_floor_and_roofline_util",
          "pass": all(
              row.get("decode_vs_floor") is not None
              and row.get("decode_roofline_util") is not None
              for row in case_results
          ) and bool(case_results),
      },
  ]
  required_checks_passed = all(check["pass"] for check in run_checks)
  r2_exit_gate_closed = required_checks_passed and all(check["pass"] for check in r2_exit_checks)
  host_metadata = CTX.collect_host_metadata(args.host, args.env_script, min(args.timeout_s, 120))
  floor_is_bootstrap_placeholder = any(
      row.get("floor", {}).get("is_bootstrap_placeholder") is not False
      for row in case_results
  ) if case_results else True

  matrix = {
      "acceptance_matrix": str(args.acceptance.resolve().relative_to(ROOT)),
      "case_ids": case_ids,
      "case_process_isolation": not args.combined_process,
      "case_results": case_results,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "floor_is_bootstrap_placeholder": floor_is_bootstrap_placeholder,
      "host_metadata": host_metadata,
      "ignore_eos": args.ignore_eos,
      "kv_dtype": args.kv_dtype,
      "kv_pressure_budget": str(args.kv_pressure.resolve().relative_to(ROOT)),
      "max_new_tokens": args.max_new_tokens,
      "observed_buckets": observed_buckets,
      "partial_selected_gate_q4_plane_route_allowed": (
          args.allow_partial_selected_gate_q4_plane_route
      ),
      "prefix_cache_enabled": False,
      "profile_matvec": args.profile_matvec,
      "q4_plane_layout": args.q4_plane_layout,
      "resident_cache": args.resident_cache,
      "route": CTX.route_name(args),
      "route_args": route_args,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "target_runs": target_runs,
      "timed_runs": args.timed_runs,
      "warmup_runs": args.warmup_runs,
  }
  payload = {
      "created_at": created_at,
      "host": args.host,
      "matrix": matrix,
      "model_path": args.model,
      "parse_errors": parse_errors,
      "remote_dir": remote_dir,
      "required_checks_passed": required_checks_passed,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "token_id_refs": str(args.token_id_refs.resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  }

  CTX.write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_ids": case_ids,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_plane_layout": args.q4_plane_layout,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "partial_selected_gate_q4_plane_route_allowed": (
          args.allow_partial_selected_gate_q4_plane_route
      ),
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "host": args.host,
      "ignore_eos": args.ignore_eos,
      "max_new_tokens": args.max_new_tokens,
      "model_path": args.model,
      "remote_dir": remote_dir,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r2-native-matrix.py",
      "workstream": WORKSTREAM,
  })
  CTX.write_json(out_dir / "token-input-manifest.json", token_manifest)
  CTX.write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "source_files": CTX.SOURCE_FILES,
      "source_transfers": source_transfers,
      "token_transfers": token_transfers,
  })
  CTX.write_json(out_dir / "build.json", build)
  CTX.write_json(out_dir / "matrix.json", payload)
  CTX.write_json(out_dir / "correctness.json", {
      "checks": run_checks,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "gate": "r2_native_speed_denominator_matrix",
      "q4_plane_layout": args.q4_plane_layout,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "partial_selected_gate_q4_plane_route_allowed": (
          args.allow_partial_selected_gate_q4_plane_route
      ),
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "r2_exit_checks": r2_exit_checks,
      "r2_exit_gate_closed": r2_exit_gate_closed,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "case-results.jsonl", case_results)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r2_native_matrix",
      [
          ("case_count", len(case_results)),
          ("case_process_isolation", not args.combined_process),
          ("dense_q4_plane_pair_dot", args.dense_q4_plane_pair_dot),
          ("dense_q6_pair_dot", args.dense_q6_pair_dot),
          ("q4_plane_layout", args.q4_plane_layout),
          ("selected_gate_q4_plane_pair_dot", args.selected_gate_q4_plane_pair_dot),
          (
              "partial_selected_gate_q4_plane_route_allowed",
              args.allow_partial_selected_gate_q4_plane_route,
          ),
          ("selected_expert_down_q4_pair_dot", args.selected_expert_down_q4_pair_dot),
          ("selected_expert_down_q6_pair_dot", args.selected_expert_down_q6_pair_dot),
          ("ignore_eos", args.ignore_eos),
          ("max_new_tokens", args.max_new_tokens),
          ("observed_bucket_count", len(observed_buckets)),
          ("required_checks_passed", required_checks_passed),
          ("r2_exit_gate_closed", r2_exit_gate_closed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")

  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
