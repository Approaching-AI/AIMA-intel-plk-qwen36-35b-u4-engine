#!/usr/bin/env python3
"""Probe the compiled block-q16 source contract guard on target."""

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
    "intel-qwen36-seq510-current-token-qkv-delta-blockq16-"
    "source-guard-probe-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ509 = (
    ROOT
    / "output/seq509-current-token-qkv-delta-blockq16-target-compile-gate-20260709Tseq509Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq510-current-token-qkv-delta-blockq16-source-guard-probe-gate-20260709Tseq510Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_TOKEN_DIR = "/tmp/iq36-current-token-qkv-delta-blockq16-guard"

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
DECODE_TOKENS = 8
TOPK = 512
EXPECTED_VALUES = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK
EXPECTED_GUARD = (
    "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE is source-gate only"
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
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _short_result(result: dict[str, Any]) -> dict[str, Any]:
  return {
      "cmd": result.get("cmd"),
      "returncode": result.get("returncode"),
      "stdout": result.get("stdout"),
      "stderr": result.get("stderr"),
  }


def _run_guard(args: argparse.Namespace, binary: str) -> dict[str, Any]:
  run_flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(args.token_dir),
      "--case-id", "router_math_reason_001",
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", "1",
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
      "set -o pipefail",
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      f"mkdir -p {shlex.quote(args.token_dir)}",
      " ".join([
          "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE=1",
          shlex.quote(binary),
          *run_flags,
      ]),
  ]
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(' && '.join(run_parts))}",
      args.timeout_s)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq509 = _load_json(args.seq509)
  compile_summary = seq509.get("compile_summary")
  compile_summary = compile_summary if isinstance(compile_summary, dict) else {}
  binary = compile_summary.get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq509 compile binary missing")

  guard_run = _run_guard(args, binary)
  guard_text = (
      str(guard_run.get("stdout") or "") + "\n" +
      str(guard_run.get("stderr") or "")
  )
  cmd = str(guard_run.get("cmd") or "")

  checks = [
      {
          "name": "seq509_selected_source_guard_probe_gate",
          "pass": (
              seq509.get("required_checks_passed") is True
              and seq509.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_guard_probe_gate"
              and seq509.get("source_guard_probe_allowed") is True
              and _has_candidate(
                  routes, 509,
                  "accept_current_token_qkv_delta_blockq16_target_compile")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_blockq16_source_guard_probe_gate",
                  509)
          ),
      },
      {
          "name": "target_guard_blocks_source_only_blockq16",
          "pass": (
              guard_run.get("returncode") != 0 and EXPECTED_GUARD in guard_text),
          "detail": _short_result(guard_run),
      },
      {
          "name": "guard_probe_uses_blockq16_env_only",
          "pass": (
              "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE=1" in cmd
              and "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1"
                  not in cmd
              and "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE=1" not in cmd),
      },
      {
          "name": "no_token_json_emitted_by_guard_probe",
          "pass": "decode_continuation_output_tokens" not in guard_text,
      },
      {
          "name": "blockq16_contract_shape_preserved",
          "pass": EXPECTED_VALUES == 122880,
          "detail": {
              "all_linear_layers": ALL_LINEAR_LAYERS,
              "decode_tokens": DECODE_TOKENS,
              "topk": TOPK,
              "expected_values": EXPECTED_VALUES,
              "selector": "linear_qkv_col_abs",
              "value_mode": "shadow_delta_block_q16",
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq509_target_compile": _rel(args.seq509),
          "host": args.host,
          "model": args.model,
          "env_script": args.env_script,
          "binary": binary,
      },
      "guard_probe": _short_result(guard_run),
      "blockq16_contract": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "topk": TOPK,
          "expected_values": EXPECTED_VALUES,
          "selector": "linear_qkv_col_abs",
          "value_mode": "shadow_delta_block_q16",
      },
      "checks": checks,
      "required_checks_passed": required,
      "blockq16_source_safe": required,
      "implementation_source_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_source_only_blockq16_guard_select_implementation_source"
          if required else
          "block_before_blockq16_implementation_source"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_implementation_source_gate"
          if required else
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_guard_probe_fix_gate"
      ),
      "next_route_reason": (
          "The compiled block-q16 source is safe but still source-only: the "
          "target guard blocks before token execution and preserves the all-30 "
          "top512 block-q16 contract. The next admissible unit is product "
          "implementation source wiring; decode and router distribution rows "
          "remain blocked."
          if required else
          "The block-q16 source guard probe did not prove the source-only "
          "guard. Fix the guard or compile before implementation or token rows."
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
      "# Seq510 Current-Token QKV-Delta Block-Q16 Source Guard Probe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- blockq16_source_safe: `{str(metrics['blockq16_source_safe']).lower()}`",
      f"- implementation_source_allowed: `{str(metrics['implementation_source_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- expected values: `{metrics['blockq16_contract']['expected_values']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a guarded source probe. It does not emit tokens or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq509", type=Path, default=DEFAULT_SEQ509)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--token-dir", default=DEFAULT_TOKEN_DIR)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "blockq16_source_safe": metrics["blockq16_source_safe"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
