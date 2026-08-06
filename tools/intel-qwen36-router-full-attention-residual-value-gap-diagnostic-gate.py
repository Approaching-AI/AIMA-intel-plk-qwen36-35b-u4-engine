#!/usr/bin/env python3
"""Run the full-attention residual value-gap diagnostic split."""

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
    "intel-qwen36-router-full-attention-residual-value-gap-diagnostic-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ331 = (
    ROOT
    / "output/router-full-attention-residual-product-consumer-distribution-fix-gate-20260708Tseq331Z"
    / "metrics.json"
)
DEFAULT_MATH_RESIDUAL = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-20260708Tseq272Z"
    / "result.json"
)
DEFAULT_CODE_RESIDUAL = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-20260708Tseq317diagZ"
    / "result.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-residual-value-gap-diagnostic-gate-20260708Tseq332Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

CASES = ["router_math_reason_001", "router_code_reason_002"]
MODES = ["ffn_residual_only", "layer_input"]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
FULL_ATTENTION_LAYER_TEXT = ",".join(str(layer) for layer in FULL_ATTENTION_LAYERS)
EXPECTED_LAYER_EVENTS = len(FULL_ATTENTION_LAYERS) * DECODE_TOKENS
EXPECTED_VALUES = EXPECTED_LAYER_EVENTS * HIDDEN_SIZE
ROWBLOCK16_26MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,"
    "24,25,26,28,29,30,33,34,36,37,38"
)
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
RESIDUAL_GAP_COSINE_THRESHOLD = 0.9999


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


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


def _has_switch(routes: dict[str, Any], decision: str,
                seq_covered: int) -> bool:
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


def _result_smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


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
      "max_abs_logit_diff": dist.get("max_abs_logit_diff"),
      "max_mean_abs_logit_diff": dist.get("max_mean_abs_logit_diff"),
      "required_checks_passed": dist.get("required_checks_passed"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "logits_cosine_pass": dist.get("logits_cosine_pass"),
  }


def _legacy_residual_context(path: Path) -> dict[str, Any]:
  smoke = _result_smoke(path)
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution": _distribution_summary(smoke),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "cpu_shadow_ffn_input_each_token_enabled": smoke.get(
          "cpu_shadow_ffn_input_each_token_enabled"),
      "cpu_shadow_ffn_input_layers": smoke.get("cpu_shadow_ffn_input_layers"),
      "cpu_shadow_ffn_input_layer_ids": smoke.get(
          "cpu_shadow_ffn_input_layer_ids"),
      "cpu_shadow_ffn_input_residual_only_enabled": smoke.get(
          "cpu_shadow_ffn_input_residual_only_enabled"),
      "cpu_shadow_layer_input_layers": smoke.get("cpu_shadow_layer_input_layers"),
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
  }


def _dist_pass(dist: dict[str, Any]) -> bool:
  return (
      dist.get("required_checks_passed") is True
      and _num(dist.get("max_kld")) <= KLD_THRESHOLD
      and _num(dist.get("top1_rate")) >= TOP1_THRESHOLD
  )


def _dist_fail(dist: dict[str, Any]) -> bool:
  return (
      dist.get("required_checks_passed") is False
      or _num(dist.get("max_kld")) > KLD_THRESHOLD
      or _num(dist.get("top1_rate"), 1.0) < TOP1_THRESHOLD
  )


def _run_case(args: argparse.Namespace, binary: str, remote_token_dir: str,
              case_id: str, mode: str) -> dict[str, Any]:
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
      "--cpu-shadow-state-each-token",
      "--cpu-shadow-trace-no-state-refresh",
  ]
  if mode == "ffn_residual_only":
    flags.extend([
        "--cpu-shadow-ffn-input-each-token",
        "--cpu-shadow-ffn-input-residual-only",
        "--cpu-shadow-ffn-input-layers",
        FULL_ATTENTION_LAYER_TEXT,
    ])
  elif mode == "layer_input":
    flags.extend([
        "--cpu-shadow-layer-input-layers",
        FULL_ATTENTION_LAYER_TEXT,
    ])
  else:
    raise ValueError(f"unknown mode: {mode}")
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      f"IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS={ROWBLOCK16_26MASK}",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
  ]
  remote_script = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([*env, shlex.quote(binary), *flags]),
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _case_summary(case_id: str, mode: str, run: dict[str, Any],
                  smoke: dict[str, Any]) -> dict[str, Any]:
  return {
      "case_id": case_id,
      "mode": mode,
      "returncode": run.get("returncode"),
      "stdout_bytes": len(str(run.get("stdout") or "")),
      "stderr_bytes": len(str(run.get("stderr") or "")),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "top1_match_count": smoke.get("top1_match_count"),
      "greedy_prefix_match_count": smoke.get("greedy_prefix_match_count"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "cpu_shadow_trace_no_state_refresh_enabled": smoke.get(
          "cpu_shadow_trace_no_state_refresh_enabled"),
      "cpu_shadow_ffn_input_each_token_enabled": smoke.get(
          "cpu_shadow_ffn_input_each_token_enabled"),
      "cpu_shadow_ffn_input_residual_only_enabled": smoke.get(
          "cpu_shadow_ffn_input_residual_only_enabled"),
      "cpu_shadow_ffn_input_layers": smoke.get("cpu_shadow_ffn_input_layers"),
      "cpu_shadow_ffn_input_layer_ids": smoke.get(
          "cpu_shadow_ffn_input_layer_ids"),
      "cpu_shadow_layer_input_layers": smoke.get("cpu_shadow_layer_input_layers"),
      "cpu_shadow_layer_input_layer_ids": smoke.get(
          "cpu_shadow_layer_input_layer_ids"),
      "cpu_shadow_attention_output_layers": smoke.get(
          "cpu_shadow_attention_output_layers"),
      "full_attention_residual_product_source_enabled": smoke.get(
          "full_attention_residual_product_source_enabled"),
      "full_attention_residual_product_consumer_source_enabled": smoke.get(
          "full_attention_residual_product_consumer_source_enabled"),
      "full_attention_residual_product_source_layers": smoke.get(
          "full_attention_residual_product_source_layers"),
      "full_attention_residual_product_consumer_source_layers": smoke.get(
          "full_attention_residual_product_consumer_source_layers"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "distribution": _distribution_summary(smoke),
  }


def _row_emitted(row: dict[str, Any]) -> bool:
  summary = row.get("summary", {})
  return (
      row.get("run", {}).get("returncode") in (0, 2)
      and summary.get("distribution", {}).get("position_count") == DECODE_TOKENS
  )


def _shadow_counters_ready(row: dict[str, Any]) -> bool:
  summary = row.get("summary", {})
  mode = summary.get("mode")
  common = (
      summary.get("cpu_shadow_state_each_token_enabled") is True
      and summary.get("cpu_shadow_trace_no_state_refresh_enabled") is True
      and summary.get("cpu_shadow_attention_output_layers") == 0
      and summary.get("full_attention_residual_product_source_enabled") is False
      and summary.get("full_attention_residual_product_consumer_source_enabled")
          is False
      and summary.get("speedup_claims_allowed") is False
  )
  if mode == "ffn_residual_only":
    return (
        common
        and summary.get("cpu_shadow_ffn_input_each_token_enabled") is True
        and summary.get("cpu_shadow_ffn_input_residual_only_enabled") is True
        and summary.get("cpu_shadow_ffn_input_layers") == EXPECTED_LAYER_EVENTS
        and summary.get("cpu_shadow_ffn_input_layer_ids")
            == FULL_ATTENTION_LAYERS
        and summary.get("cpu_shadow_layer_input_layers") == 0
    )
  if mode == "layer_input":
    return (
        common
        and summary.get("cpu_shadow_ffn_input_each_token_enabled") is False
        and summary.get("cpu_shadow_ffn_input_residual_only_enabled") is False
        and summary.get("cpu_shadow_ffn_input_layers") == 0
        and summary.get("cpu_shadow_layer_input_layers") == EXPECTED_LAYER_EVENTS
        and summary.get("cpu_shadow_layer_input_layer_ids")
            == FULL_ATTENTION_LAYERS
    )
  return False


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq331 = _load_json(args.seq331)
  legacy_math_residual = _legacy_residual_context(args.math_residual)
  legacy_code_residual = _legacy_residual_context(args.code_residual)
  binary = seq331.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq331 binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)

  runs: list[dict[str, Any]] = []
  if token_cache.get("ok") is True:
    for mode in MODES:
      for case_id in CASES:
        run = _run_case(args, binary, str(token_cache.get("dir")),
                        case_id, mode)
        smoke = _smoke_from_stdout(run)
        runs.append({
            "case_id": case_id,
            "mode": mode,
            "run": {
                "cmd": run.get("cmd"),
                "returncode": run.get("returncode"),
                "stdout_bytes": len(str(run.get("stdout") or "")),
                "stderr_bytes": len(str(run.get("stderr") or "")),
            },
            "summary": _case_summary(case_id, mode, run, smoke),
        })

  seq331_runs = seq331.get("runs")
  seq331_runs = seq331_runs if isinstance(seq331_runs, list) else []
  seq331_value_gap = seq331.get("value_gap_observed") is True
  seq331_distribution_failed = all(
      isinstance(row, dict)
      and _dist_fail(row.get("summary", {}).get("distribution", {}))
      for row in seq331_runs
  )
  seq331_min_residual_cosine = _num(
      seq331.get("min_selected_residual_cosine"), 1.0)
  seq331_max_residual_abs_diff = _num(
      seq331.get("max_selected_residual_abs_diff"))
  preconditions_pass = (
      seq331.get("required_checks_passed") is True
      and seq331.get("selected_next_route")
      == "router_prompt_full_attention_residual_value_gap_diagnostic_gate"
      and seq331_value_gap
      and seq331_distribution_failed
      and seq331_min_residual_cosine < RESIDUAL_GAP_COSINE_THRESHOLD
      and seq331_max_residual_abs_diff > 0.0
      and _has_candidate(
          routes, 331,
          "accept_residual_product_consumer_distribution_fix_root_diagnostic")
      and _has_switch(
          routes,
          "select_router_prompt_full_attention_residual_value_gap_diagnostic_gate",
          331)
  )

  rows_emitted = len(runs) == len(MODES) * len(CASES) and all(
      _row_emitted(row) for row in runs)
  shadow_counters_ready = rows_emitted and all(
      _shadow_counters_ready(row) for row in runs)
  residual_rows = [
      row for row in runs
      if row.get("summary", {}).get("mode") == "ffn_residual_only"
  ]
  layer_input_rows = [
      row for row in runs
      if row.get("summary", {}).get("mode") == "layer_input"
  ]
  residual_only_passed = (
      shadow_counters_ready
      and len(residual_rows) == len(CASES)
      and all(_dist_pass(row.get("summary", {}).get("distribution", {}))
              for row in residual_rows)
  )
  layer_input_passed = (
      shadow_counters_ready
      and len(layer_input_rows) == len(CASES)
      and all(_dist_pass(row.get("summary", {}).get("distribution", {}))
              for row in layer_input_rows)
  )
  layer_input_failed = (
      shadow_counters_ready
      and len(layer_input_rows) == len(CASES)
      and any(_dist_fail(row.get("summary", {}).get("distribution", {}))
              for row in layer_input_rows)
  )
  legacy_residual_context_passed = (
      _dist_pass(legacy_math_residual.get("distribution", {}))
      and _dist_pass(legacy_code_residual.get("distribution", {}))
      and legacy_math_residual.get("cpu_shadow_ffn_input_layers")
          == EXPECTED_LAYER_EVENTS
      and legacy_code_residual.get("cpu_shadow_ffn_input_layers")
          == EXPECTED_LAYER_EVENTS
      and legacy_math_residual.get(
          "cpu_shadow_ffn_input_residual_only_enabled") is True
      and legacy_code_residual.get(
          "cpu_shadow_ffn_input_residual_only_enabled") is True
  )
  diagnostic_classification = (
      "layer_input_substitution_sufficient"
      if residual_only_passed and layer_input_passed else
      "layer_input_substitution_insufficient"
      if residual_only_passed and layer_input_failed else
      "residual_only_reproduction_failed"
  )
  selected_next = (
      "router_prompt_full_attention_layer_input_product_source_gate"
      if diagnostic_classification == "layer_input_substitution_sufficient" else
      "router_prompt_full_attention_attention_output_value_gap_diagnostic_gate"
      if diagnostic_classification == "layer_input_substitution_insufficient"
      else "router_prompt_full_attention_residual_value_gap_diagnostic_gate"
  )
  checks = [
      {
          "name": "seq331_selected_value_gap_diagnostic_gate",
          "pass": preconditions_pass,
      },
      {
          "name": "legacy_residual_only_context_still_passes",
          "pass": legacy_residual_context_passed,
          "detail": {
              "math_residual": legacy_math_residual,
              "code_residual": legacy_code_residual,
          },
      },
      {
          "name": "diagnostic_shadow_rows_emitted",
          "pass": rows_emitted,
      },
      {
          "name": "diagnostic_shadow_counters_ready",
          "pass": shadow_counters_ready,
          "detail": [row.get("summary", {}) for row in runs],
      },
      {
          "name": "current_binary_residual_only_passes_math_and_code",
          "pass": residual_only_passed,
          "detail": [
              row.get("summary", {}).get("distribution", {})
              for row in residual_rows
          ],
      },
      {
          "name": "layer_input_split_classified",
          "pass": residual_only_passed and (layer_input_passed or layer_input_failed),
          "detail": [
              row.get("summary", {}).get("distribution", {})
              for row in layer_input_rows
          ],
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq331": _rel(args.seq331),
          "math_residual_context": _rel(args.math_residual),
          "code_residual_context": _rel(args.code_residual),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "binary": binary,
          "rowblock16_layers": ROWBLOCK16_26MASK,
          "full_attention_layers": FULL_ATTENTION_LAYERS,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "seq331": {
          "value_gap_observed": seq331_value_gap,
          "distribution_failed": seq331_distribution_failed,
          "min_selected_residual_cosine": seq331_min_residual_cosine,
          "max_selected_residual_abs_diff": seq331_max_residual_abs_diff,
      },
      "legacy_residual_only_context": {
          "math_residual": legacy_math_residual,
          "code_residual": legacy_code_residual,
      },
      "runs": runs,
      "checks": checks,
      "required_checks_passed": required,
      "current_binary_residual_only_passed": residual_only_passed,
      "layer_input_passed": layer_input_passed,
      "layer_input_failed": layer_input_failed,
      "diagnostic_classification": diagnostic_classification,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_value_gap_split_diagnostic"
          if required else
          "block_full_attention_residual_value_gap_split_diagnostic"
      ),
      "selected_next_route": selected_next if required else
          "router_prompt_full_attention_residual_value_gap_diagnostic_gate",
      "next_route_reason": (
          "Replacing native layer inputs at all nine selected full-attention "
          "layers also passes math/code distribution, so the residual gap is "
          "inherited from the upstream layer-input source; productize or "
          "repair that source before another FFN-residual consumer route."
          if required and diagnostic_classification
          == "layer_input_substitution_sufficient" else
          "Current-binary residual-only replacement passes math/code, but "
          "native layer-input replacement at the same nine full-attention "
          "layers is insufficient. The next split is attention-output versus "
          "residual-add value inside the full-attention block; speed promotion "
          "and long-context expansion remain blocked."
          if required and diagnostic_classification
          == "layer_input_substitution_insufficient" else
          "The value-gap split did not reproduce the residual-only passing "
          "row; keep the residual value-gap diagnostic gate open."
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
      "# Router Full-Attention Residual Value-Gap Diagnostic Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- current residual-only passed: `{str(metrics['current_binary_residual_only_passed']).lower()}`",
      f"- layer-input passed: `{str(metrics['layer_input_passed']).lower()}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
  ]
  for row in metrics["runs"]:
    summary = row["summary"]
    dist = summary["distribution"]
    lines.extend([
        f"## {summary['mode']} / {summary['case_id']}",
        "",
        f"- distribution max KLD: `{dist.get('max_kld')}`",
        f"- distribution top1 rate: `{dist.get('top1_rate')}`",
        f"- distribution required: `{str(dist.get('required_checks_passed')).lower()}`",
        f"- CPU-shadow FFN input layers: `{summary.get('cpu_shadow_ffn_input_layers')}`",
        f"- CPU-shadow layer-input layers: `{summary.get('cpu_shadow_layer_input_layers')}`",
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
  parser.add_argument("--seq331", type=Path, default=DEFAULT_SEQ331)
  parser.add_argument("--math-residual", type=Path,
                      default=DEFAULT_MATH_RESIDUAL)
  parser.add_argument("--code-residual", type=Path,
                      default=DEFAULT_CODE_RESIDUAL)
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
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "layer_input_passed": metrics["layer_input_passed"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "residual_only_passed": metrics["current_binary_residual_only_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
