#!/usr/bin/env python3
"""Gate the static sparse head-logit bias distribution diagnostic source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-sparse-head-logit-bias-source-gate-v0"
CURRENT_ROUTE = "router_prompt_distribution_sparse_head_logit_bias_source_gate"
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_sparse_head_logit_bias_target_compile_gate"
)
EXPECTED_SPEC = (
    "22:0.28438186,25:0.39105892,264:0.79164315,"
    "421:-0.05538559,821:0.21105385,71093:0.01916504,"
    "248068:-0.228602415"
)
EXPECTED_BIASES = [
    {"token_id": 22, "bias": 0.28438186},
    {"token_id": 25, "bias": 0.39105892},
    {"token_id": 264, "bias": 0.79164315},
    {"token_id": 421, "bias": -0.05538559},
    {"token_id": 821, "bias": 0.21105385},
    {"token_id": 71093, "bias": 0.01916504},
    {"token_id": 248068, "bias": -0.228602415},
]


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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _markers(source: str, *, wrapper: bool) -> list[dict[str, Any]]:
  markers = {
      "default_empty_bias_table": (
          "std::vector<DecodeSparseLogitBias> sparse_head_logit_bias;"
          in source),
      "runtime_spec_parser": (
          "DecodeParseSparseLogitBias" in source
          and 'std::getenv("IQ36_SPARSE_HEAD_LOGIT_BIAS_SPEC")' in source),
      "compiled_static_contract": (
          "DecodeSparseHeadLogitBiasMatchesContract" in source
          and "{248068, -0.228602415f}" in source),
      "distribution_only_guard": (
          "IQ36_SPARSE_HEAD_LOGIT_BIAS_SPEC is distribution-diagnostic only"
          in source),
      "post_lmhead_application": (
          "DecodeApplySparseHeadLogitBias(args.sparse_head_logit_bias, &gpu_next)"
          in source),
      "topk_recomputed_from_biased_logits": (
          "result->topk = DecodeTopKFromLogits(result->logits);" in source),
      "result_records_bias_table": (
          "sparse_head_logit_bias_enabled" in source
          and "WriteSparseHeadLogitBias(args.sparse_head_logit_bias)" in source),
  }
  if wrapper:
    markers.update({
        "source_only_guard": (
            "IQ36_SPARSE_HEAD_LOGIT_BIAS_SOURCE is source-gate only"
            in source),
        "fixed_seq564_spec_guard": (
            "must match the seq564 static contract" in source),
        "remote_spec_forwarding": (
            '"IQ36_SPARSE_HEAD_LOGIT_BIAS_SPEC="' in source),
        "manifest_records_source_and_mode": (
            '"sparse_head_logit_bias_source"' in source
            and '"sparse_head_logit_bias_mode": "static_token_id"' in source),
    })
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _apply_body(source: str) -> str:
  start = source.find("void DecodeApplySparseHeadLogitBias(")
  end = source.find("DecodeTokenResult DecodeLmHeadTopK(", start)
  if start < 0 or end < 0:
    return ""
  return source[start:end]


def _compile_generated(args: argparse.Namespace) -> dict[str, Any]:
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  command = [
      args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
      _rel(generated), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o"),
  ]
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = compile_dir / "compile.stdout.txt"
  stderr_path = compile_dir / "compile.stderr.txt"
  stdout_path.write_text(completed.stdout, encoding="utf-8")
  stderr_path.write_text(completed.stderr, encoding="utf-8")
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": _rel(stdout_path),
      "stderr": _rel(stderr_path),
  }


def _code_volume() -> dict[str, Any]:
  command = ["python3", "tools/intel-qwen36-code-volume-check.py"]
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout.strip(),
      "stderr": completed.stderr.strip(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  predecessor = _load(args.predecessor)
  routes = _load(args.routes)
  result_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(result_path)
  wrapper_source = args.decode_source.read_text(encoding="utf-8")
  generated_source = generated_path.read_text(encoding="utf-8")
  wrapper_markers = _markers(wrapper_source, wrapper=True)
  generated_markers = _markers(generated_source, wrapper=False)
  apply_body = _apply_body(generated_source)
  forbidden_apply_terms = [
      term for term in ("case_id", "prompt", "position", "native", "step")
      if term in apply_body
  ]
  compile_result = _compile_generated(args)
  code_volume = _code_volume()
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("source_gate_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 564, CURRENT_ROUTE)
      and _has_switch(
          routes, 564,
          "select_router_prompt_distribution_sparse_head_logit_bias_source_gate"))
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("sparse_head_logit_bias_source") is True
      and manifest.get("sparse_head_logit_bias_spec") == EXPECTED_SPEC
      and manifest.get("sparse_head_logit_bias") == EXPECTED_BIASES
      and manifest.get("sparse_head_logit_bias_mode") == "static_token_id"
      and manifest.get("speedup_claims_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  checks = [
      {"name": "seq564_selected_sparse_bias_source_gate",
       "pass": predecessor_selects},
      {"name": "generate_only_manifest_records_exact_static_contract",
       "pass": manifest_passes},
      {"name": "wrapper_is_default_off_guarded_and_exact",
       "pass": all(row["pass"] for row in wrapper_markers),
       "detail": wrapper_markers},
      {"name": "generated_cpp_parses_applies_and_records_bias",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
      {"name": "bias_application_has_no_prompt_case_position_or_oracle_branch",
       "pass": bool(apply_body) and not forbidden_apply_terms,
       "detail": {"forbidden_terms": forbidden_apply_terms}},
      {"name": "generated_cpp_compiles_locally",
       "pass": compile_result["passed"], "detail": compile_result},
      {"name": "code_volume_ceiling_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "routes": _rel(args.routes),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
      },
      "contract": {
          "biases": EXPECTED_BIASES,
          "mode": "static_token_id",
          "distribution_only": True,
          "runtime_native_oracle": False,
          "prompt_case_position_branching": False,
          "default_enabled": False,
      },
      "checks": checks,
      "compile": compile_result,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "token_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_static_sparse_head_logit_bias_source"
          if required else "block_static_sparse_head_logit_bias_source"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The default-empty generated path accepts only the exact seq564 "
          "seven-token table, requires full distribution logits, recomputes "
          "top-k after correction, contains no prompt/case/position/oracle "
          "branch, compiles locally, and preserves the code-volume ceiling. "
          "Target-compile next without a token."
          if required else
          "Fix the static contract, branch isolation, compile, or code-volume "
          "failure before any target work."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Sparse Head-Logit Bias Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- local compile passed: `{str(metrics['compile']['passed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It is not calibration or speed evidence.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=565)
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq564-non-arithmetic-product-source-route-control-gate-20260710Tseq564Z/metrics.json")
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--decode-source", type=Path,
      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq565-sparse-head-logit-bias-source-20260710Tseq565Z")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq565-sparse-head-logit-bias-source-gate-20260710Tseq565Z")
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "target_compile_allowed": metrics["target_compile_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
