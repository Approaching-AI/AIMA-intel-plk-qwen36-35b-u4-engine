#!/usr/bin/env python3
"""Gate the distribution-only GPU-top8 fit observable source."""

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
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-fit-observable-source-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_observable_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_observable_target_compile_gate"
)


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
      "top8_constant": "constexpr int kDecodeFitObservableTopK = 8;" in source,
      "bounded_feature_row": (
          "struct DecodeFitObservableRow" in source
          and "gpu_logit_minus_top1" in source
          and "native_minus_gpu_logit" in source),
      "distribution_step_owns_rows": (
          "std::vector<DecodeFitObservableRow> gpu_top8_fit_observables;"
          in source),
      "single_vocab_pass_top8_selection": (
          "gpu_fit_topk.reserve(kDecodeFitObservableTopK);" in source
          and "InsertDecodeTopK(" in source),
      "exact_eight_row_require": (
          "distribution fit observable top8 is incomplete" in source),
      "records_fit_targets_and_features": (
          '"gpu_top8_fit_observables"' in source
          or '\\"gpu_top8_fit_observables\\"' in source),
  }
  if wrapper:
    markers.update({
        "source_only_guard": (
            "IQ36_STATE_CONDITIONED_HEAD_FIT_OBSERVABLE_SOURCE is source-gate only"
            in source),
        "manifest_records_top8_distribution_only": (
            '"state_conditioned_head_fit_observable_topk": 8' in source
            and '"state_conditioned_head_fit_observable_distribution_only": True'
            in source),
    })
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _feature_struct_body(source: str) -> str:
  start = source.find("struct DecodeFitObservableRow {")
  end = source.find("};", start)
  return source[start:end + 2] if start >= 0 and end >= 0 else ""


def _compile(args: argparse.Namespace) -> dict[str, Any]:
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
      "returncode": completed.returncode,
      "stdout": completed.stdout.strip(),
      "stderr": completed.stderr.strip(),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  result_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(result_path)
  wrapper_source = args.decode_source.read_text(encoding="utf-8")
  generated_source = generated_path.read_text(encoding="utf-8")
  wrapper_markers = _markers(wrapper_source, wrapper=True)
  generated_markers = _markers(generated_source, wrapper=False)
  feature_body = _feature_struct_body(generated_source)
  forbidden_feature_terms = [
      term for term in ("prompt", "case", "position", "native_oracle")
      if term in feature_body
  ]
  compile_result = _compile(args)
  code_volume = _code_volume()
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("fit_observable_source_gate_allowed") is True
      and predecessor.get("decode_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 571, CURRENT_ROUTE)
      and _has_switch(
          routes, 571,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_fit_observable_source_gate"))
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("case_id") == "fresh_arithmetic_01"
      and manifest.get("state_conditioned_head_fit_observable_source") is True
      and manifest.get("state_conditioned_head_fit_observable_topk") == 8
      and manifest.get(
          "state_conditioned_head_fit_observable_distribution_only") is True
      and manifest.get("speedup_claims_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  checks = [
      {"name": "seq571_selected_fit_observable_source_gate",
       "pass": predecessor_selects},
      {"name": "generate_only_manifest_records_top8_source",
       "pass": manifest_passes},
      {"name": "wrapper_source_is_guarded_and_distribution_only",
       "pass": all(row["pass"] for row in wrapper_markers),
       "detail": wrapper_markers},
      {"name": "generated_cpp_records_exact_top8_targets_and_features",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
      {"name": "runtime_feature_row_has_no_prompt_case_or_position_feature",
       "pass": bool(feature_body) and not forbidden_feature_terms,
       "detail": {"forbidden_terms": forbidden_feature_terms}},
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
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
      },
      "observable_contract": {
          "topk": 8,
          "features": [
              "token_id", "rank", "gpu_logit", "gpu_logit_minus_top1"],
          "offline_targets": ["native_logit", "native_minus_gpu_logit"],
          "distribution_only": True,
          "decode_math_changed": False,
          "token_selection_changed": False,
      },
      "checks": checks,
      "compile": compile_result,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "fit_decode_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_distribution_top8_fit_observable_source"
          if required else "reject_distribution_top8_fit_observable_source"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The distribution-only record adds exactly eight GPU-ranked rows with "
          "the locked runtime features and offline native-minus-GPU target, "
          "without changing decode math or top-k selection. Generated C++ "
          "compiles locally and the 97-flag ceiling passes. Target-compile next "
          "without decoding."
          if required else
          "Fix source isolation, row shape, manifest, compile, or code volume "
          "before target use."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} State-Conditioned Top8 Observable Source",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- local compile passed: `{str(metrics['compile']['passed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. No prompt was decoded.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=572)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z/metrics.json")
  parser.add_argument(
      "--decode-source", type=Path,
      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq572-state-conditioned-head-fit-observable-source-20260710Tseq572Z")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq572-state-conditioned-head-fit-observable-source-gate-20260710Tseq572Z")
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
