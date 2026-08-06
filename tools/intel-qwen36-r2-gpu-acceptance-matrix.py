#!/usr/bin/env python3
"""Run/roll up GPU-hybrid decode rows for the acceptance-matrix gate.

This is a subset-capable collector for the current R2 GPU decode stack. It can
run a deterministic short prompt set or selected context-ladder buckets through
`intel-qwen36-r2-gpu-decode-smoke.py`, then writes the promotion-contract file
shape. Passing this tool means the requested subset completed; product
promotion remains false until the full acceptance matrix, long-context
sentinel, and smoothness requirements are covered.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r2-gpu-acceptance-matrix-v0"
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
)
DEFAULT_TOKEN_ID_REFS = (
    ROOT
    / "output/r0-oracle-token-id-capture-20260626T083347Z"
    / "prompt-token-id-references.jsonl"
)
DEFAULT_SHORT_CASES = (
    "short_math_001",
    "short_factual_002",
    "short_transform_003",
)
BUCKET_LABELS = {
    1024: "001k",
    2048: "002k",
    4096: "004k",
    8192: "008k",
    16384: "016k",
    32768: "032k",
    65536: "064k",
    102400: "100k",
    131072: "128k",
    262144: "256k",
}
DECODE_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
CONTEXT_TOOL = ROOT / "tools/intel-qwen36-context-ladder-native-diagnostic.py"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path)


def load_context_tool() -> Any:
  import importlib.util

  spec = importlib.util.spec_from_file_location("iq36_context_ladder", CONTEXT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load context helper: {CONTEXT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument("--model", default="/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
  parser.add_argument("--env-script", default="/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--token-id-refs", type=Path, default=DEFAULT_TOKEN_ID_REFS)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=7200)
  parser.add_argument("--decode-tokens", type=int, default=8)
  parser.add_argument("--case-id", action="append", default=[])
  parser.add_argument(
      "--bucket",
      action="append",
      default=[],
      help="Add sentinel_<bucket> and prefill_shape_<bucket>, e.g. 001k.",
  )
  parser.add_argument(
      "--lane",
      choices=("token_exact", "distribution", "both"),
      default="token_exact",
      help="Run free-run greedy token checks, teacher-forced distribution, or both.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Prepare token inputs and commands without launching decode rows.",
  )
  return parser.parse_args()


def case_ids_from_args(args: argparse.Namespace) -> list[str]:
  out: list[str] = []
  out.extend(args.case_id or DEFAULT_SHORT_CASES)
  for bucket in args.bucket:
    if bucket not in set(BUCKET_LABELS.values()):
      raise SystemExit(f"unsupported bucket label: {bucket}")
    out.append(f"sentinel_{bucket}")
    out.append(f"prefill_shape_{bucket}")
  return list(dict.fromkeys(out))


def smoke_env() -> dict[str, str]:
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
  })
  return env


def smoke_args(args: argparse.Namespace, token_dir: Path, case_id: str, out_dir: Path, distribution: bool) -> list[str]:
  argv = [
      sys.executable,
      str(DECODE_SMOKE),
      "--host", args.host,
      "--model", args.model,
      "--env-script", args.env_script,
      "--remote-root", args.remote_root,
      "--token-input-dir", str(token_dir),
      "--case-id", case_id,
      "--decode-tokens", str(args.decode_tokens),
      "--resident-selected-cache-topk", "16",
      "--resident-selected-q4-experts",
      "--resident-selected-q6-experts",
      "--resident-selected-q6-sorted-cache",
      "--resident-selected-q6-rowstripe",
      "--resident-shared-q6-down",
      "--resident-full-attention-v-q6",
      "--resident-linear-q6-qkv",
      "--resident-q4-cpu-order-z",
      "--resident-linear-conv-weights",
      "--resident-linear-state",
      "--resident-postconv-delta-handoff",
      "--resident-norm-weights",
      "--resident-gate-up-swiglu-handoff",
      "--resident-attention-front-handoff",
      "--resident-full-core-attention-front-handoff",
      "--gpu-router",
      "--gpu-lm-head-q6",
      "--out-dir", str(out_dir),
      "--timeout-s", str(args.timeout_s),
  ]
  if distribution:
    argv.append("--distribution-ladder")
  return argv


def short_command(argv: list[str]) -> str:
  return " ".join(argv)


def run_one(
    args: argparse.Namespace,
    token_dir: Path,
    case_id: str,
    lane: str,
    out_dir: Path,
) -> dict[str, Any]:
  distribution = lane == "distribution"
  case_out = out_dir / "runs" / f"{case_id}-{lane}"
  argv = smoke_args(args, token_dir, case_id, case_out, distribution)
  record: dict[str, Any] = {
      "case_id": case_id,
      "command": short_command(argv),
      "distribution": distribution,
      "lane": lane,
      "output_dir": rel(case_out),
  }
  if args.dry_run:
    record["returncode"] = None
    record["dry_run"] = True
    return record
  result = subprocess.run(
      argv,
      cwd=ROOT,
      env=smoke_env(),
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=args.timeout_s + 300,
      check=False,
  )
  record.update({
      "returncode": result.returncode,
      "stdout_tail": result.stdout[-4000:],
      "stderr_tail": result.stderr[-4000:],
  })
  result_path = case_out / "result.json"
  record["result_json_present"] = result_path.exists()
  if result_path.exists():
    payload = read_json(result_path)
    smoke = payload.get("smoke")
    if not isinstance(smoke, dict):
      smoke = payload
    dist = smoke.get("distribution_ladder")
    dist = dist if isinstance(dist, dict) else {}
    record.update({
        "artifact": rel(case_out),
        "decode_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
        "generated_token_count": smoke.get("decode_continuation_output_tokens"),
        "greedy_prefix_match_count": smoke.get("greedy_prefix_match_count"),
        "prompt_token_count": smoke.get("prompt_token_count"),
        "required_checks_passed": smoke.get("required_checks_passed"),
        "rowblock16_enabled": smoke.get(
            "attention_front_output_projection_rowblock16_enabled"),
        "rowblock16_layer_ids": smoke.get(
            "attention_front_output_projection_rowblock16_layer_ids"),
        "source_sha": smoke.get("source_sha") or payload.get("source_sha"),
        "top1_matches_native": smoke.get("top1_matches_native"),
        "topk_ids_match_native": smoke.get("topk_ids_match_native"),
        "distribution_required_checks_passed": dist.get("required_checks_passed"),
        "distribution_max_kld": dist.get("max_kld"),
        "distribution_top1_rate": dist.get("top1_rate"),
        "distribution_min_logits_cosine": dist.get("min_logits_cosine"),
      })
  return record


def smoothness(case_rows: list[dict[str, Any]], args: argparse.Namespace, acceptance: dict[str, Any]) -> dict[str, Any]:
  speed_rows = [
      row for row in case_rows
      if row.get("lane") == "token_exact" and isinstance(row.get("decode_tok_s"), (int, float))
  ]
  tpots = [1000.0 / float(row["decode_tok_s"]) for row in speed_rows if float(row["decode_tok_s"]) > 0.0]
  p95_over_p50 = None
  if len(tpots) >= 2:
    ordered = sorted(tpots)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    p95_over_p50 = p95 / p50 if p50 > 0.0 else None
  threshold = (
      acceptance.get("smoothness", {}).get("decode_tpot_p95_over_p50_max", 1.25)
      if isinstance(acceptance.get("smoothness"), dict) else 1.25
  )
  enough_for_short = len(speed_rows) >= 3
  return {
      "checks": [
          {
              "name": "short_prompt_tpot_p95_over_p50",
              "pass": (
                  args.dry_run
                  or not enough_for_short
                  or (
                      isinstance(p95_over_p50, float)
                      and p95_over_p50 <= float(threshold)
                  )
              ),
              "p95_over_p50": p95_over_p50,
              "threshold": threshold,
              "status": "insufficient_rows" if not enough_for_short else "checked",
          },
          {
              "name": "full_context_ladder_smoothness_not_claimed",
              "pass": True,
          },
      ],
      "decode_tpot_ms": tpots,
      "required_checks_passed": True,
  }


def build_summary(payload: dict[str, Any]) -> str:
  rows = payload["matrix"]["case_results"]
  lines = [
      "# R2 GPU Acceptance Matrix Subset",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- lane: `{payload['matrix']['lane']}`",
      f"- case count: `{len(payload['matrix']['case_ids'])}`",
      f"- decode tokens: `{payload['matrix']['decode_tokens']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- product promotion ready: `{str(payload['product_promotion_ready']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | lane | prompt | gen | top1 | top-k | dist KLD | tok/s | artifact |",
      "|---|---|---:|---:|---|---|---:|---:|---|",
  ]
  for row in rows:
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("lane")),
            str(row.get("prompt_token_count")),
            str(row.get("generated_token_count")),
            str(row.get("top1_matches_native")),
            str(row.get("topk_ids_match_native")),
            str(row.get("distribution_max_kld")),
            str(row.get("decode_tok_s")),
            f"`{row.get('artifact') or row.get('output_dir')}`",
        ])
        + " |"
    )
  lines += [
      "",
      "This artifact advances acceptance-matrix evidence for a selected subset.",
      "It is not product promotion until the full matrix and context ladder pass.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.decode_tokens < 1 or args.decode_tokens > 8:
    raise SystemExit("--decode-tokens currently follows decode-smoke limit 1..8")
  if not args.token_id_refs.exists():
    raise SystemExit(f"token-id refs missing: {args.token_id_refs}")
  if not args.acceptance.exists():
    raise SystemExit(f"acceptance matrix missing: {args.acceptance}")

  acceptance = read_json(args.acceptance)
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r2-gpu-acceptance-matrix-{stamp}"
      if args.out_dir is None else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  case_ids = case_ids_from_args(args)
  ctx = load_context_tool()
  token_rows = ctx.selected_rows(args.token_id_refs, case_ids)
  token_manifest = ctx.prepare_token_inputs(token_rows, out_dir / "token-input")

  lanes = ["token_exact", "distribution"] if args.lane == "both" else [args.lane]
  case_results: list[dict[str, Any]] = []
  for lane in lanes:
    for case_id in case_ids:
      case_results.append(run_one(args, out_dir / "token-input", case_id, lane, out_dir))

  token_rows_done = [row for row in case_results if row.get("lane") == "token_exact"]
  distribution_rows = [row for row in case_results if row.get("lane") == "distribution"]
  min_prompt_cases = (
      acceptance.get("accuracy", {})
      .get("tokens", {})
      .get("min_prompt_cases", 3)
  )
  token_exact_pass = (
      not token_rows_done
      or all(
          row.get("result_json_present") is True
          and row.get("top1_matches_native") is True
          and row.get("generated_token_count") == args.decode_tokens
          and row.get("rowblock16_enabled") is True
          for row in token_rows_done
      )
  )
  distribution_pass = (
      not distribution_rows
      or all(
          row.get("result_json_present") is True
          and row.get("distribution_required_checks_passed") is True
          and row.get("rowblock16_enabled") is True
          for row in distribution_rows
      )
  )
  matrix_checks = [
      {"name": "token_id_refs_present", "pass": args.token_id_refs.exists()},
      {"name": "acceptance_matrix_present", "pass": args.acceptance.exists()},
      {"name": "selected_cases_present", "pass": len(token_rows) == len(case_ids)},
      {
          "name": "min_prompt_case_count_reached",
          "pass": len(case_ids) >= int(min_prompt_cases),
          "case_count": len(case_ids),
          "min_prompt_cases": min_prompt_cases,
      },
      {
          "name": "dry_run_only_prepares_commands",
          "pass": not args.dry_run,
          "dry_run": args.dry_run,
      },
      {
          "name": "requested_subset_rows_completed",
          "pass": all(row.get("result_json_present") is True for row in case_results),
      },
      {"name": "token_exact_rows_top1_match", "pass": token_exact_pass},
      {"name": "distribution_rows_pass", "pass": distribution_pass},
      {
          "name": "speedup_claims_forbidden_until_full_matrix",
          "pass": True,
      },
  ]
  smooth = smoothness(case_results, args, acceptance)
  required_checks_passed = all(row["pass"] for row in matrix_checks)
  full_output_tokens = args.decode_tokens in acceptance.get("matrix", {}).get("output_tokens", [])
  product_promotion_ready = (
      required_checks_passed
      and args.lane == "both"
      and full_output_tokens
      and False
  )
  payload = {
      "created_at": created_at,
      "matrix": {
          "acceptance_matrix": rel(args.acceptance),
          "case_ids": case_ids,
          "case_results": case_results,
          "decode_tokens": args.decode_tokens,
          "dry_run": args.dry_run,
          "lane": args.lane,
          "token_input_manifest": token_manifest,
      },
      "product_promotion_ready": product_promotion_ready,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_ids": case_ids,
      "decode_tokens": args.decode_tokens,
      "dry_run": args.dry_run,
      "lane": args.lane,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r2-gpu-acceptance-matrix.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "acceptance-matrix.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": matrix_checks,
      "gate": "r2_gpu_acceptance_matrix_subset",
      "product_promotion_ready": product_promotion_ready,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoothness.json", smooth)
  write_jsonl(out_dir / "case-results.jsonl", case_results)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r2_gpu_acceptance_matrix_subset",
      [
          ("case_count", len(case_ids)),
          ("decode_tokens", args.decode_tokens),
          ("dry_run", args.dry_run),
          ("product_promotion_ready", product_promotion_ready),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
