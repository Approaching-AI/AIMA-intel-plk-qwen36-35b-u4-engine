#!/usr/bin/env python3
"""Run one teacher-forced router-math token through fused exact projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-one-token-probe-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_one_token_probe_gate"
)
SUCCESS_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_router_math_distribution_gate"
)
REJECT_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_route_close_gate"
)
LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
LINEAR_LAYER_CSV = ",".join(str(layer) for layer in LINEAR_LAYERS)
ROWBLOCK16_26MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,"
    "24,25,26,28,29,30,33,34,36,37,38"
)
KLD_MAX = 0.005


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _parse_stdout(run: dict[str, Any]) -> dict[str, Any]:
  for line in str(run.get("stdout") or "").splitlines():
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def _run_probe(args: argparse.Namespace, binary: str,
               remote_token_dir: str) -> dict[str, Any]:
  run_flags = [
      "--model", args.model,
      "--token-dir", remote_token_dir,
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
      "--full-attention-state-diff",
      "--teacher-force-native-tokens",
      "--distribution-ladder",
      "--diagnostic-layer-range", "0:2",
      "--diagnostic-token-limit", "1",
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS="
      + ROWBLOCK16_26MASK,
      "IQ36_INPUT_RMSNORM_SERIAL_REDUCTION_LAYERS=" + LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS=" + LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_ROWBLOCK16_CPUORDER_FINALIZE=1",
      "IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF=1",
  ]
  command = " ".join(
      [*env, shlex.quote(binary),
       *(shlex.quote(value) for value in run_flags)])
  return iq36_local.run_target(
      args.host,
      f"bash -lc {shlex.quote(' && '.join([f'source {shlex.quote(args.env_script)} >/dev/null 2>&1', command]))}",
      args.timeout_s,
  )


def _summary(smoke: dict[str, Any]) -> dict[str, Any]:
  distribution = smoke.get("distribution_ladder")
  distribution = distribution if isinstance(distribution, dict) else {}
  return {
      "schema_version": smoke.get("schema_version"),
      "case_id": smoke.get("case_id"),
      "decode_tokens_per_session": smoke.get("decode_tokens_per_session"),
      "linear_output_projection_rowblock16_cpuorder_finalize": smoke.get(
          "linear_output_projection_rowblock16_cpuorder_finalize"),
      "linear_output_projection_cpu_order_layers": smoke.get(
          "linear_output_projection_cpu_order_layers"),
      "input_rmsnorm_serial_reduction_layers": smoke.get(
          "input_rmsnorm_serial_reduction_layers"),
      "linear_final_device_q8_handoff_enabled": smoke.get(
          "linear_final_device_q8_handoff_enabled"),
      "top1_matches_native": smoke.get("top1_matches_native"),
      "top1_match_count": smoke.get("top1_match_count"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
      "gpu_hybrid_decode_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
      "distribution": {
          "required_checks_passed": distribution.get("required_checks_passed"),
          "position_count": distribution.get("position_count"),
          "top1_pass": distribution.get("top1_pass"),
          "top1_rate": distribution.get("top1_rate"),
          "max_kld": distribution.get("max_kld"),
      },
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  compile_summary = predecessor.get("compile_summary", {})
  binary = str(compile_summary.get("binary", ""))
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  run = (
      _run_probe(args, binary, str(token_cache.get("dir")))
      if token_cache.get("ok") is True and binary else
      {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
  )
  smoke = _parse_stdout(run)
  summary = _summary(smoke)
  distribution = summary["distribution"]
  args.out_dir.mkdir(parents=True, exist_ok=True)
  iq36_local.write_json(args.out_dir / "raw-run.json", run)
  if smoke:
    iq36_local.write_json(args.out_dir / "smoke.json", smoke)

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("one_token_probe_allowed") is True
      and predecessor.get("multi_token_probe_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and compile_summary.get("ok") is True
      and compile_summary.get("key") == args.expected_binary_key
      and _has_candidate(routes, 591, CURRENT_ROUTE)
      and _has_switch(
          routes, 591,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "one_token_probe_gate"))
  execution_complete = (
      run.get("returncode") in (0, 2)
      and summary.get("schema_version")
      == "intel-qwen36-r2-gpu-decode-smoke-v0"
      and summary.get("case_id") == "router_math_reason_001"
      and summary.get("decode_tokens_per_session") == 1)
  selectors_pass = (
      summary.get("linear_output_projection_rowblock16_cpuorder_finalize")
      is True
      and summary.get("linear_output_projection_cpu_order_layers")
      == LINEAR_LAYERS
      and summary.get("input_rmsnorm_serial_reduction_layers")
      == LINEAR_LAYERS
      and summary.get("linear_final_device_q8_handoff_enabled") is True)
  correctness_pass = (
      summary.get("required_checks_passed") is True
      and summary.get("top1_matches_native") is True
      and summary.get("top1_match_count") == 1
      and distribution.get("required_checks_passed") is True
      and distribution.get("position_count") == 1
      and distribution.get("top1_pass") is True
      and distribution.get("top1_rate") == 1
      and isinstance(distribution.get("max_kld"), (int, float))
      and float(distribution["max_kld"]) <= KLD_MAX
      and summary.get("speedup_claims_allowed") is False)
  required = route_selects and execution_complete and selectors_pass and correctness_pass
  measurement_complete = route_selects and execution_complete and selectors_pass
  checks = [
      {"name": "seq591_selected_exactly_one_token_probe",
       "pass": route_selects},
      {"name": "router_token_inputs_are_cached",
       "pass": token_cache.get("ok") is True,
       "detail": {
           "hit": token_cache.get("hit"),
           "key": token_cache.get("key"),
           "dir": token_cache.get("dir"),
       }},
      {"name": "target_binary_emitted_one_router_math_token",
       "pass": execution_complete,
       "detail": {"returncode": run.get("returncode")}},
      {"name": "fused_device_q8_and_serial_norm_cover_all_30_linear_layers",
       "pass": selectors_pass},
      {"name": "one_token_top1_and_kld_contract_pass",
       "pass": correctness_pass, "detail": summary},
  ]
  if required:
    disposition = "accept_fused_exact_projection_one_token"
    selected_next = SUCCESS_NEXT_ROUTE
    reason = (
        "The one-token all-linear parity row activates fused device-Q8 "
        "projection plus serial input-RMSNorm on all 30 linear layers, "
        "preserves greedy top-1, and clears KLD. Run only the 8-token "
        "router-math distribution row next; router-code and speed remain "
        "blocked."
    )
  elif measurement_complete:
    disposition = "reject_fused_exact_projection_one_token"
    selected_next = REJECT_NEXT_ROUTE
    reason = (
        "The fused selectors were active and the one-token evidence completed, "
        "but top-1 or KLD failed. Close the route before multi-token work."
    )
  else:
    disposition = "block_incomplete_fused_exact_projection_one_token"
    selected_next = CURRENT_ROUTE
    reason = (
        "Repair token staging, execution, parsing, or selector activation "
        "without changing the locked kernel before retrying this one row."
    )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "model": args.model,
          "binary": binary,
          "binary_key": compile_summary.get("key"),
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "run": {
          "returncode": run.get("returncode"),
          "stdout_bytes": len(str(run.get("stdout") or "")),
          "stderr_bytes": len(str(run.get("stderr") or "")),
      },
      "smoke_summary": summary,
      "checks": checks,
      "measurement_complete": measurement_complete,
      "required_checks_passed": required,
      "one_token_passed": required,
      "router_math_distribution_allowed": required,
      "router_code_distribution_allowed": False,
      "speed_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next,
      "next_route_reason": reason,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "measurement_complete": metrics["measurement_complete"],
          "one_token_passed": metrics["one_token_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "router_code_distribution_allowed": False,
          "speed_probe_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  summary = metrics["smoke_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection One-Token Probe",
      "",
      f"- measurement_complete: `{str(metrics['measurement_complete']).lower()}`",
      f"- one_token_passed: `{str(metrics['one_token_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- top1: `{summary.get('top1_matches_native')}`",
      f"- max KLD: `{summary.get('distribution', {}).get('max_kld')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is one-token correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=592)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq591-fused-exact-linear-projection-decode-target-compile-gate-20260710Tseq591Z/metrics.json")
  parser.add_argument(
      "--token-input-dir", type=Path,
      default=ROOT / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z/token-input")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq592-fused-exact-linear-projection-one-token-probe-gate-20260710Tseq592Z")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument(
      "--model",
      default="/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
  parser.add_argument(
      "--env-script",
      default="/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--expected-binary-key",
                      default="5553e5fbb1dc5aea9ae2d0fe")
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "measurement_complete": metrics["measurement_complete"],
      "one_token_passed": metrics["one_token_passed"],
      "disposition": metrics["disposition"],
      "smoke_summary": metrics["smoke_summary"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["measurement_complete"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
