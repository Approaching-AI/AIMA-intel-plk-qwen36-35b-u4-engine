#!/usr/bin/env python3
"""Run the 8-token fused exact projection router-math distribution gate."""

from __future__ import annotations

import argparse
import importlib.util
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


ONE_TOKEN_TOOL = (
    ROOT / "tools/intel-qwen36-fused-exact-linear-projection-one-token-probe-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-router-math-distribution-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_router_math_distribution_gate"
)
SUCCESS_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_router_code_distribution_gate"
)
REJECT_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_route_close_gate"
)
DECODE_TOKENS = 8
KLD_MAX = 0.005


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


ONE = _load_module(ONE_TOKEN_TOOL, "iq36_fused_exact_one_token_gate")


def _run_probe(args: argparse.Namespace, binary: str,
               remote_token_dir: str) -> dict[str, Any]:
  run_flags = [
      "--model", args.model,
      "--token-dir", remote_token_dir,
      "--case-id", "router_math_reason_001",
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
      "--full-attention-state-diff",
      "--teacher-force-native-tokens",
      "--distribution-ladder",
      "--diagnostic-layer-range", "0:2",
      "--diagnostic-token-limit", str(DECODE_TOKENS),
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS="
      + ONE.ROWBLOCK16_26MASK,
      "IQ36_INPUT_RMSNORM_SERIAL_REDUCTION_LAYERS=" + ONE.LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS=" + ONE.LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_ROWBLOCK16_CPUORDER_FINALIZE=1",
      "IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF=1",
  ]
  command = " ".join(
      [*env, shlex.quote(binary),
       *(shlex.quote(value) for value in run_flags)])
  shell = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      command,
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(shell)}", args.timeout_s)


def _summary(smoke: dict[str, Any]) -> dict[str, Any]:
  return ONE._summary(smoke)  # noqa: SLF001


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = ONE._load(args.routes)  # noqa: SLF001
  predecessor = ONE._load(args.predecessor)  # noqa: SLF001
  binary = str(predecessor.get("inputs", {}).get("binary", ""))
  binary_key = predecessor.get("inputs", {}).get("binary_key")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  run = (
      _run_probe(args, binary, str(token_cache.get("dir")))
      if token_cache.get("ok") is True and binary else
      {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
  )
  smoke = ONE._parse_stdout(run)  # noqa: SLF001
  summary = _summary(smoke)
  distribution = summary["distribution"]
  args.out_dir.mkdir(parents=True, exist_ok=True)
  iq36_local.write_json(args.out_dir / "raw-run.json", run)
  if smoke:
    iq36_local.write_json(args.out_dir / "smoke.json", smoke)

  route_selects = (
      predecessor.get("measurement_complete") is True
      and predecessor.get("required_checks_passed") is True
      and predecessor.get("one_token_passed") is True
      and predecessor.get("router_math_distribution_allowed") is True
      and predecessor.get("router_code_distribution_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and binary_key == args.expected_binary_key
      and ONE._has_candidate(routes, 592, CURRENT_ROUTE)  # noqa: SLF001
      and ONE._has_switch(  # noqa: SLF001
          routes, 592,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "router_math_distribution_gate"))
  execution_complete = (
      run.get("returncode") in (0, 2)
      and summary.get("schema_version")
      == "intel-qwen36-r2-gpu-decode-smoke-v0"
      and summary.get("case_id") == "router_math_reason_001"
      and summary.get("decode_tokens_per_session") == DECODE_TOKENS)
  selectors_pass = (
      summary.get("linear_output_projection_rowblock16_cpuorder_finalize")
      is True
      and summary.get("linear_output_projection_cpu_order_layers")
      == ONE.LINEAR_LAYERS
      and summary.get("input_rmsnorm_serial_reduction_layers")
      == ONE.LINEAR_LAYERS
      and summary.get("linear_final_device_q8_handoff_enabled") is True)
  correctness_pass = (
      summary.get("required_checks_passed") is True
      and summary.get("top1_matches_native") is True
      and summary.get("top1_match_count") == DECODE_TOKENS
      and distribution.get("required_checks_passed") is True
      and distribution.get("position_count") == DECODE_TOKENS
      and distribution.get("top1_pass") is True
      and distribution.get("top1_rate") == 1
      and isinstance(distribution.get("max_kld"), (int, float))
      and float(distribution["max_kld"]) <= KLD_MAX
      and summary.get("speedup_claims_allowed") is False)
  required = route_selects and execution_complete and selectors_pass and correctness_pass
  measurement_complete = route_selects and execution_complete and selectors_pass
  checks = [
      {"name": "seq592_selected_eight_token_router_math_only",
       "pass": route_selects},
      {"name": "router_token_inputs_are_cached",
       "pass": token_cache.get("ok") is True,
       "detail": {
           "hit": token_cache.get("hit"),
           "key": token_cache.get("key"),
           "dir": token_cache.get("dir"),
       }},
      {"name": "target_binary_emitted_eight_router_math_tokens",
       "pass": execution_complete,
       "detail": {"returncode": run.get("returncode")}},
      {"name": "fused_device_q8_and_serial_norm_remain_all_30_layers",
       "pass": selectors_pass},
      {"name": "router_math_top1_8_of_8_and_kld_pass",
       "pass": correctness_pass, "detail": summary},
  ]
  if required:
    disposition = "accept_fused_exact_projection_router_math_distribution"
    selected_next = SUCCESS_NEXT_ROUTE
    reason = (
        "The 8-token router-math row keeps the exact 30-layer selectors, "
        "passes greedy top-1 and KLD, and earns exactly the paired 8-token "
        "router-code distribution row next. Speed remains blocked."
    )
  elif measurement_complete:
    disposition = "reject_fused_exact_projection_router_math_distribution"
    selected_next = REJECT_NEXT_ROUTE
    reason = (
        "The 8-token router-math evidence completed with the fused path active "
        "but failed top-1 or KLD. Close the route before router-code or speed."
    )
  else:
    disposition = "block_incomplete_fused_exact_router_math_distribution"
    selected_next = CURRENT_ROUTE
    reason = (
        "Repair token staging, execution, parsing, or selector activation "
        "without changing the locked kernel before retrying this row."
    )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": ONE._rel(args.routes),  # noqa: SLF001
          "predecessor": ONE._rel(args.predecessor),  # noqa: SLF001
          "token_input_dir": ONE._rel(args.token_input_dir),  # noqa: SLF001
          "host": args.host,
          "model": args.model,
          "binary": binary,
          "binary_key": binary_key,
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
      "router_math_distribution_passed": required,
      "router_code_distribution_allowed": required,
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
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  summary = metrics["smoke_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Router-Math Distribution",
      "",
      f"- measurement_complete: `{str(metrics['measurement_complete']).lower()}`",
      f"- router_math_distribution_passed: "
      f"`{str(metrics['router_math_distribution_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- top1 count: `{summary.get('top1_match_count')}/8`",
      f"- max KLD: `{summary.get('distribution', {}).get('max_kld')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is router-math correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=593)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq592-fused-exact-linear-projection-one-token-probe-gate-20260710Tseq592Z/metrics.json")
  parser.add_argument(
      "--token-input-dir", type=Path,
      default=ROOT / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z/token-input")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq593-fused-exact-linear-projection-router-math-distribution-gate-20260710Tseq593Z")
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
      "router_math_distribution_passed": metrics[
          "router_math_distribution_passed"],
      "disposition": metrics["disposition"],
      "smoke_summary": metrics["smoke_summary"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": ONE._rel(args.out_dir),  # noqa: SLF001
  }, sort_keys=True))
  return 0 if metrics["measurement_complete"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
