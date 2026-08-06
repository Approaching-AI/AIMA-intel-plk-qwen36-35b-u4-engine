#!/usr/bin/env python3
"""Probe product-owned full-attention residual source counters on target."""

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
    "implementation-probe-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ322 = (
    ROOT
    / "output/router-full-attention-residual-product-implementation-target-compile-gate-20260708Tseq322Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r1-engine-seed-prompt-input-check-20260627T155328Z/token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-residual-product-implementation-probe-gate-20260708Tseq323Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
HIDDEN_SIZE = 2048
DECODE_TOKENS = 1
EXPECTED_VALUES = len(PRODUCER_LAYERS) * HIDDEN_SIZE * DECODE_TOKENS
SOURCE_ONLY_GUARD = (
    "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE is source-gate only"
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


def _parse_stdout_json(run: dict[str, Any]) -> dict[str, Any]:
  stdout = str(run.get("stdout") or "").strip()
  if not stdout:
    return {}
  first_line = stdout.splitlines()[0]
  parsed = json.loads(first_line)
  return parsed if isinstance(parsed, dict) else {}


def _run_probe(args: argparse.Namespace, binary: str,
               remote_token_dir: str) -> dict[str, Any]:
  run_flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(remote_token_dir),
      "--case-id", "short_math_001",
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
  ]
  run_parts = [
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([
          "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE=1",
          shlex.quote(binary),
          *run_flags,
      ]),
  ]
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(' && '.join(run_parts))}",
      args.timeout_s)


def _smoke_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  keys = [
      "schema_version",
      "case_id",
      "decode_continuation_output_tokens",
      "full_attention_residual_product_source_enabled",
      "full_attention_residual_product_source_layers",
      "full_attention_residual_product_source_values",
      "full_attention_residual_product_source_misses",
      "full_attention_residual_product_source_ready",
      "resident_full_core_attention_front_handoff_enabled",
      "resident_attention_front_handoff_enabled",
      "cpu_shadow_state_each_token_enabled",
      "cpu_shadow_ffn_input_each_token_enabled",
      "cpu_shadow_layer_input_layers",
      "cpu_shadow_attention_output_layers",
      "top1_matches_native",
      "top1_match_count",
      "required_checks_passed",
      "speedup_claims_allowed",
      "gpu_hybrid_decode_tok_s",
  ]
  return {key: smoke.get(key) for key in keys}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq322 = _load_json(args.seq322)
  compile_summary = seq322.get("compile_summary")
  compile_summary = compile_summary if isinstance(compile_summary, dict) else {}
  binary = compile_summary.get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq322 compile binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  probe_run = (
      _run_probe(args, binary, str(token_cache.get("dir")))
      if token_cache.get("ok") is True else
      {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
  )
  smoke = _parse_stdout_json(probe_run)
  run_text = str(probe_run.get("stdout") or "") + str(probe_run.get("stderr") or "")

  checks = [
      {
          "name": "seq322_selected_implementation_probe_gate",
          "pass": (
              seq322.get("required_checks_passed") is True
              and seq322.get("selected_next_route")
              == "router_prompt_full_attention_residual_product_implementation_probe_gate"
              and _has_candidate(
                  routes, 322,
                  "accept_full_attention_residual_product_implementation_target_compile")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_product_implementation_probe_gate",
                  322)
          ),
      },
      {
          "name": "token_inputs_staged",
          "pass": token_cache.get("ok") is True,
          "detail": {
              "cache_hit": token_cache.get("hit"),
              "key": token_cache.get("key"),
              "dir": token_cache.get("dir"),
          },
      },
      {
          "name": "target_binary_emitted_probe_json",
          "pass": (
              smoke.get("schema_version")
              == "intel-qwen36-r2-gpu-decode-smoke-v0"
              and probe_run.get("returncode") in (0, 2)
          ),
          "detail": {
              "returncode": probe_run.get("returncode"),
              "stdout_bytes": len(str(probe_run.get("stdout") or "")),
              "stderr_bytes": len(str(probe_run.get("stderr") or "")),
          },
      },
      {
          "name": "product_residual_source_counters_ready",
          "pass": (
              smoke.get("full_attention_residual_product_source_enabled") is True
              and smoke.get("full_attention_residual_product_source_layers")
              == len(PRODUCER_LAYERS) * DECODE_TOKENS
              and smoke.get("full_attention_residual_product_source_values")
              == EXPECTED_VALUES
              and smoke.get("full_attention_residual_product_source_misses") == 0
              and smoke.get("full_attention_residual_product_source_ready") is True
          ),
          "detail": {
              "expected_layers": len(PRODUCER_LAYERS) * DECODE_TOKENS,
              "expected_values": EXPECTED_VALUES,
              "observed_layers": smoke.get(
                  "full_attention_residual_product_source_layers"),
              "observed_values": smoke.get(
                  "full_attention_residual_product_source_values"),
              "observed_misses": smoke.get(
                  "full_attention_residual_product_source_misses"),
              "observed_ready": smoke.get(
                  "full_attention_residual_product_source_ready"),
          },
      },
      {
          "name": "residual_source_is_cpu_shadow_free",
          "pass": (
              smoke.get("cpu_shadow_state_each_token_enabled") is False
              and smoke.get("cpu_shadow_ffn_input_each_token_enabled") is False
              and smoke.get("cpu_shadow_layer_input_layers") == 0
              and smoke.get("cpu_shadow_attention_output_layers") == 0
          ),
      },
      {
          "name": "probe_keeps_greedy_top1_and_no_speed_claim",
          "pass": (
              smoke.get("top1_matches_native") is True
              and smoke.get("top1_match_count") == DECODE_TOKENS
              and smoke.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "required_checks_passed": smoke.get("required_checks_passed"),
              "gpu_hybrid_decode_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
          },
      },
      {
          "name": "source_only_guard_not_hit",
          "pass": SOURCE_ONLY_GUARD not in run_text,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq322_target_compile": _rel(args.seq322),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "model": args.model,
          "env_script": args.env_script,
          "binary": binary,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "probe_run": {
          "cmd": probe_run.get("cmd"),
          "returncode": probe_run.get("returncode"),
          "stdout_bytes": len(str(probe_run.get("stdout") or "")),
          "stderr": probe_run.get("stderr"),
      },
      "smoke_summary": _smoke_summary(smoke),
      "residual_counter_expectation": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "expected_layers": len(PRODUCER_LAYERS) * DECODE_TOKENS,
          "expected_values": EXPECTED_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "residual_product_source_probe_passed": required,
      "decode_probe_allowed": required,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_product_implementation_probe"
          if required else
          "reject_full_attention_residual_product_implementation_probe"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_product_implementation_decode_gate"
          if required else
          "router_prompt_full_attention_residual_product_implementation_probe_fix_gate"
      ),
      "next_route_reason": (
          "The target binary executes the product-owned residual source, emits "
          "ready counters for all selected full-attention residual source "
          "layers, and uses no CPU-shadow source inputs. The next admissible "
          "unit is an 8-token decode/correctness gate, still without speed "
          "promotion."
          if required else
          "The product-owned residual source probe did not prove ready "
          "counters; fix residual source/runtime wiring before decode or "
          "router distribution."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  summary = metrics["smoke_summary"]
  lines = [
      "# Router Full-Attention Residual Product Implementation Probe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- source layers: `{summary.get('full_attention_residual_product_source_layers')}`",
      f"- source values: `{summary.get('full_attention_residual_product_source_values')}`",
      f"- source misses: `{summary.get('full_attention_residual_product_source_misses')}`",
      f"- source ready: `{summary.get('full_attention_residual_product_source_ready')}`",
      f"- top1 matches native: `{summary.get('top1_matches_native')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a one-token residual-source counter probe. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq322", type=Path, default=DEFAULT_SEQ322)
  parser.add_argument("--token-input-dir", type=Path, default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "residual_product_source_probe_passed": metrics[
          "residual_product_source_probe_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
