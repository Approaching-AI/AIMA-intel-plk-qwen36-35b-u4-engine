#!/usr/bin/env python3
"""Target-compile the fused exact projection decode source without a token."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SMOKE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-decode-target-compile-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_decode_target_compile_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_one_token_probe_gate"
)
LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


SMOKE = _load_module(SMOKE_SOURCE, "iq36_fused_exact_projection_decode_smoke")


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


def _command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def _markers(source: str) -> list[dict[str, Any]]:
  markers = {
      "generated_selector_defaults_off": (
          "bool linear_output_projection_rowblock16_cpuorder_finalize = false;"
          in source),
      "generated_selector_reads_runtime_env": (
          "IQ36_LINEAR_OUTPUT_PROJECTION_ROWBLOCK16_CPUORDER_FINALIZE"
          in source),
      "generated_path_separates_old_and_fused_cpuorder": (
          "use_separate_cpu_order_output_projection" in source
          and "use_fused_exact_output_projection" in source),
      "generated_device_q8_handoff_receives_selector": (
          "use_fused_exact_output_projection);" in source),
      "generated_host_fallback_rejects_fused_selector": (
          "fused exact output projection requires device-Q8 handoff" in source),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  source_gate = _load(args.source_gate)
  manifest_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(manifest_path)
  generated_markers = _markers(generated_path.read_text(encoding="utf-8"))
  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      f"{args.remote_root}/cache",
      SMOKE.SOURCE_FILES,
      ROOT,
      generated_path,
      "tests/r2_gpu_decode_smoke.cpp",
      lambda remote_dir: SMOKE.build_command(remote_dir, args.env_script),
      "build/r2-gpu-decode-smoke",
      args.timeout_s,
  )
  build = _command(compile_result, "build")
  publish = _command(compile_result, "publish")
  compile_summary = {
      "ok": compile_result.get("ok"),
      "cache_hit": compile_result.get("hit"),
      "key": compile_result.get("key"),
      "binary": compile_result.get("binary"),
      "build_returncode": build.get("returncode"),
      "publish_returncode": publish.get("returncode"),
  }
  source_selects = (
      source_gate.get("required_checks_passed") is True
      and source_gate.get("target_compile_allowed") is True
      and source_gate.get("one_token_probe_allowed") is False
      and source_gate.get("token_row_allowed") is False
      and source_gate.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 590, CURRENT_ROUTE)
      and _has_switch(
          routes, 590,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "decode_target_compile_gate"))
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("fused_exact_linear_output_projection_source") is True
      and manifest.get(
          "linear_output_projection_rowblock16_cpuorder_finalize") is True
      and manifest.get("fused_exact_linear_output_projection_layers")
      == LINEAR_LAYERS
      and manifest.get(
          "fused_exact_linear_output_projection_device_q8_handoff") is True
      and manifest.get(
          "fused_exact_linear_output_projection_host_bridge_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  no_execution = not any(
      (args.out_dir / name).exists()
      for name in ("smoke.json", "tokens.jsonl", "run.json"))
  checks = [
      {"name": "seq590_selected_target_compile_only_gate",
       "pass": source_selects},
      {"name": "generate_only_manifest_locks_exact_no_bridge_path",
       "pass": manifest_passes},
      {"name": "generated_cpp_preserves_fused_selector_contract",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
      {"name": "target_binary_compile_or_fresh_cache_publish_passed",
       "pass": compile_result.get("ok") is True,
       "detail": compile_summary},
      {"name": "target_compile_gate_did_not_launch_a_token",
       "pass": no_execution},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "source_gate": _rel(args.source_gate),
          "generate_only_result": _rel(manifest_path),
          "generated_cpp": _rel(generated_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "compile": compile_result,
      "compile_summary": compile_summary,
      "checks": checks,
      "required_checks_passed": required,
      "one_token_probe_allowed": required,
      "multi_token_probe_allowed": False,
      "router_distribution_allowed": False,
      "token_row_allowed": required,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_fused_exact_projection_decode_target_compile"
          if required else "reject_fused_exact_projection_decode_target_compile"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The exact 30-linear-layer device-Q8 decode source target-compiles "
          "without execution. Run exactly one teacher-forced router-math token "
          "from this binary and require selector coverage, no host bridge, and "
          "greedy top-1 before any multi-token or distribution row."
          if required else
          "Repair target compile, generated markers, or route provenance before "
          "a token."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "compile_summary": metrics["compile_summary"],
          "selected_next_route": metrics["selected_next_route"],
          "one_token_probe_allowed": metrics["one_token_probe_allowed"],
          "multi_token_probe_allowed": False,
          "router_distribution_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  summary = metrics["compile_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Decode Target Compile",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- compile_ok: `{str(summary['ok']).lower()}`",
      f"- cache_hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is target-compile evidence only. No token was run.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=591)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / "output/seq590-fused-exact-linear-projection-decode-source-gate-20260710Tseq590Z/metrics.json")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq590-fused-exact-linear-projection-decode-source-20260710Tseq590Z")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq591-fused-exact-linear-projection-decode-target-compile-gate-20260710Tseq591Z")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument(
      "--env-script",
      default="/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--timeout-s", type=int, default=7200)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "compile_summary": metrics["compile_summary"],
      "one_token_probe_allowed": metrics["one_token_probe_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
