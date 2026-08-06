#!/usr/bin/env python3
"""Run router math/code distribution rows for the residual product source."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-residual-product-"
    "router-distribution-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ324 = (
    ROOT
    / "output/router-full-attention-residual-product-implementation-decode-gate-20260708Tseq324Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-residual-product-router-distribution-gate-20260708Tseq325Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
DECODE_TOKENS = 8
EXPECTED_LAYERS = 9 * DECODE_TOKENS
EXPECTED_VALUES = EXPECTED_LAYERS * 2048
ROWBLOCK16_26MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,"
    "24,25,26,28,29,30,33,34,36,37,38"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _smoke_from_stdout(run: dict[str, Any]) -> dict[str, Any]:
  stdout = str(run.get("stdout") or "").strip()
  if not stdout:
    return {}
  parsed = json.loads(stdout.splitlines()[0])
  return parsed if isinstance(parsed, dict) else {}


def _run_case(args: argparse.Namespace, binary: str, remote_token_dir: str,
              case_id: str) -> dict[str, Any]:
  flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(remote_token_dir),
      "--case-id", case_id,
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", str(DECODE_TOKENS),
      "--lm-head-threads", "16",
      "--shared-q4-runner",
      "--resident-q4-weights",
      "--resident-selected-q4-experts",
      "--resident-selected-q6-experts",
      "--resident-selected-q6-sorted-cache",
      "--resident-selected-q6-rowstripe",
      "--resident-selected-cache-topk", "16",
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
      "--opencl-double-swiglu",
      "--distribution-ladder",
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      f"IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS={ROWBLOCK16_26MASK}",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE=1",
  ]
  remote_script = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([*env, shlex.quote(binary), *flags]),
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _distribution_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "position_count": dist.get("position_count"),
      "top1_rate": dist.get("top1_rate"),
      "top1_match_count": dist.get("top1_match_count"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "min_logits_cosine": dist.get("min_logits_cosine"),
      "required_checks_passed": dist.get("required_checks_passed"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "logits_cosine_pass": dist.get("logits_cosine_pass"),
  }


def _case_summary(case_id: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  return {
      "case_id": case_id,
      "returncode": run.get("returncode"),
      "stdout_bytes": len(str(run.get("stdout") or "")),
      "stderr_bytes": len(str(run.get("stderr") or "")),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "top1_match_count": smoke.get("top1_match_count"),
      "greedy_prefix_match_count": smoke.get("greedy_prefix_match_count"),
      "source_layers": smoke.get(
          "full_attention_residual_product_source_layers"),
      "source_values": smoke.get(
          "full_attention_residual_product_source_values"),
      "source_misses": smoke.get(
          "full_attention_residual_product_source_misses"),
      "source_ready": smoke.get(
          "full_attention_residual_product_source_ready"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "cpu_shadow_layer_input_layers": smoke.get("cpu_shadow_layer_input_layers"),
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": _distribution_summary(smoke),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq324 = _load_json(args.seq324)
  binary = seq324.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq324 binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for case_id in CASES:
      run = _run_case(args, binary, str(token_cache.get("dir")), case_id)
      smoke = _smoke_from_stdout(run)
      runs.append({
          "case_id": case_id,
          "run": {
              "cmd": run.get("cmd"),
              "returncode": run.get("returncode"),
              "stdout_bytes": len(str(run.get("stdout") or "")),
              "stderr_bytes": len(str(run.get("stderr") or "")),
          },
          "summary": _case_summary(case_id, run, smoke),
      })

  def row_counters_ready(row: dict[str, Any]) -> bool:
    summary = row.get("summary", {})
    return (
        row.get("run", {}).get("returncode") in (0, 2)
        and summary.get("source_layers") == EXPECTED_LAYERS
        and summary.get("source_values") == EXPECTED_VALUES
        and summary.get("source_misses") == 0
        and summary.get("source_ready") is True
        and summary.get("cpu_shadow_state_each_token_enabled") is False
        and summary.get("cpu_shadow_layer_input_layers") == 0
        and summary.get("cpu_shadow_attention_output_layers") == 0
        and summary.get("speedup_claims_allowed") is False
    )

  def row_distribution_pass(row: dict[str, Any]) -> bool:
    dist = row.get("summary", {}).get("distribution", {})
    return (
        dist.get("required_checks_passed") is True
        and _num(dist.get("top1_rate")) >= 0.99
        and _num(dist.get("max_kld")) <= 0.005
    )

  preconditions_pass = (
      seq324.get("required_checks_passed") is True
      and seq324.get("selected_next_route")
      == "router_prompt_full_attention_residual_product_router_distribution_gate"
      and _has_candidate(
          routes, 324,
          "accept_full_attention_residual_product_implementation_decode_gate")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_residual_product_router_distribution_gate",
          324)
  )
  rows_emitted = (
      len(runs) == len(CASES)
      and all(row.get("run", {}).get("returncode") in (0, 2) for row in runs)
  )
  counters_ready = rows_emitted and all(row_counters_ready(row) for row in runs)
  distribution_passed = (
      counters_ready and all(row_distribution_pass(row) for row in runs)
  )
  route_control_passed = preconditions_pass and rows_emitted and counters_ready
  checks = [
      {"name": "seq324_selected_router_distribution_gate",
       "pass": preconditions_pass},
      {"name": "router_distribution_rows_emitted", "pass": rows_emitted},
      {"name": "residual_source_counters_ready_for_router_rows",
       "pass": counters_ready,
       "detail": [row["summary"] for row in runs]},
      {"name": "router_distribution_thresholds_pass",
       "pass": distribution_passed,
       "detail": [row["summary"]["distribution"] for row in runs]},
  ]
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq324_decode": _rel(args.seq324),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
          "rowblock16_layers": ROWBLOCK16_26MASK,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "runs": runs,
      "checks": checks,
      "required_checks_passed": route_control_passed,
      "router_distribution_passed": distribution_passed,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_product_router_distribution_gate"
          if distribution_passed else
          "reject_source_only_residual_router_distribution_select_consumer_source"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_product_speed_gate"
          if distribution_passed else
          "router_prompt_full_attention_residual_product_consumer_source_gate"
      ),
      "next_route_reason": (
          "Router distribution passed for math/code; speed promotion still "
          "needs full benchmark evidence."
          if distribution_passed else
          "The product source is resident and counter-clean, but source-only "
          "wiring does not repair router distribution. Add a product consumer "
          "for the residual handles before speed promotion or long-context "
          "expansion."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  lines = [
      "# Router Full-Attention Residual Product Router Distribution Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- router_distribution_passed: `{str(metrics['router_distribution_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
  ]
  for row in metrics["runs"]:
    summary = row["summary"]
    dist = summary["distribution"]
    lines.extend([
        f"## {summary['case_id']}",
        "",
        f"- top1 rate: `{dist.get('top1_rate')}`",
        f"- max KLD: `{dist.get('max_kld')}`",
        f"- distribution required: `{str(dist.get('required_checks_passed')).lower()}`",
        f"- residual counters: `{summary.get('source_layers')}` / `{summary.get('source_values')}` / `{summary.get('source_misses')}`",
        "",
    ])
  lines.extend([
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq324", type=Path, default=DEFAULT_SEQ324)
  parser.add_argument("--token-input-dir", type=Path,
                      default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "router_distribution_passed": metrics["router_distribution_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
