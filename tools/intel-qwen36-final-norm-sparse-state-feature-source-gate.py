#!/usr/bin/env python3
"""Gate the distribution-only GPU final-norm fit observable source."""

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
SCHEMA_VERSION = "intel-qwen36-final-norm-sparse-state-feature-source-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_source_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_final_norm_sparse_state_feature_target_compile_gate"
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
      "build_selector_enabled": (
          (
              "constexpr bool kDecodeFinalNormFitObservableBuild = false;"
              in source
              and "IQ36_FINAL_NORM_SPARSE_STATE_FIT_OBSERVABLE" in source
          ) if wrapper else (
              "constexpr bool kDecodeFinalNormFitObservableBuild = true;"
              in source
          )),
      "distribution_step_owns_gpu_vector": (
          "std::vector<float> gpu_final_norm_fit_observables;" in source),
      "exact_hidden_size_guard": (
          "distribution GPU final norm fit observable size mismatch" in source
          and "gpu_final_norm.size() == kHiddenSize" in source),
      "copies_existing_gpu_final_norm_only": (
          "step.gpu_final_norm_fit_observables = gpu_final_norm;" in source),
      "writer_records_fit_vector": (
          '"gpu_final_norm_fit_observables"' in source
          or '\\"gpu_final_norm_fit_observables\\"' in source),
      "does_not_emit_native_fit_vector": (
          "native_final_norm_fit_observables" not in source),
  }
  if wrapper:
    markers.update({
        "distribution_only_guard": (
            "IQ36_FINAL_NORM_SPARSE_STATE_FIT_OBSERVABLE is distribution-only"
            in source),
        "manifest_records_2048_gpu_only_contract": (
            '"final_norm_sparse_state_fit_observable_hidden_size": 2048'
            in source
            and '"final_norm_sparse_state_fit_observable_distribution_only": True'
            in source
            and '"final_norm_sparse_state_runtime_full_vector_host_read_allowed": False'
            in source),
    })
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


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
  contract = _load(args.contract)
  result_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(result_path)
  wrapper_source = args.decode_source.read_text(encoding="utf-8")
  generated_source = generated_path.read_text(encoding="utf-8")
  wrapper_markers = _markers(wrapper_source, wrapper=True)
  generated_markers = _markers(generated_source, wrapper=False)
  compile_result = _compile(args)
  code_volume = _code_volume()
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("final_norm_feature_source_allowed") is True
      and predecessor.get("fit_collection_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 580, CURRENT_ROUTE)
      and _has_switch(
          routes, 580,
          "select_router_prompt_distribution_final_norm_sparse_state_"
          "feature_source_gate"))
  feature = contract.get("feature_collection_contract", {})
  contract_passes = (
      contract.get("schema_version")
      == "intel-qwen36-final-norm-sparse-state-feature-contract-v0"
      and feature.get("hidden_size") == 2048
      and feature.get("distribution_only") is True
      and feature.get("runtime_full_vector_host_read_allowed") is False
      and feature.get("runtime_selected_dimension_gather_only") is True)
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("required_checks_passed") is False
      and manifest.get("case_id") == "fresh_arithmetic_01"
      and manifest.get("final_norm_sparse_state_fit_observable") is True
      and manifest.get("final_norm_sparse_state_fit_observable_hidden_size")
      == 2048
      and manifest.get(
          "final_norm_sparse_state_fit_observable_distribution_only") is True
      and manifest.get(
          "final_norm_sparse_state_runtime_full_vector_host_read_allowed")
      is False
      and manifest.get("speedup_claims_allowed") is False
      and not (args.generate_dir / "smoke.json").exists())
  checks = [
      {"name": "seq580_selected_final_norm_source_gate",
       "pass": predecessor_selects},
      {"name": "locked_sparse_state_contract_matches_source_shape",
       "pass": contract_passes},
      {"name": "generate_only_manifest_records_gpu_final_norm_source",
       "pass": manifest_passes},
      {"name": "wrapper_is_distribution_guarded_and_manifested",
       "pass": all(row["pass"] for row in wrapper_markers),
       "detail": wrapper_markers},
      {"name": "generated_cpp_records_exact_gpu_2048_vector",
       "pass": all(row["pass"] for row in generated_markers),
       "detail": generated_markers},
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
          "contract": _rel(args.contract),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
      },
      "observable_contract": {
          "source": "GPU final-normalized hidden state",
          "hidden_size": 2048,
          "distribution_only": True,
          "native_vector_emitted": False,
          "decode_math_changed": False,
          "token_selection_changed": False,
          "runtime_full_vector_host_read_allowed": False,
      },
      "checks": checks,
      "compile": compile_result,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "fit_decode_allowed": False,
      "validation_or_test_allowed": False,
      "runtime_selected_dimension_source_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_distribution_gpu_final_norm_fit_observable_source"
          if required else
          "reject_distribution_gpu_final_norm_fit_observable_source"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The build-gated distribution diagnostic copies exactly the existing "
          "2048-value GPU final-normalized vector, emits no native vector, and "
          "changes neither decode math nor token selection. Generated C++ "
          "compiles locally and code volume remains at 97 flags. Target-compile "
          "next without decoding."
          if required else
          "Fix contract, source isolation, generated markers, compile, or code "
          "volume before target use."),
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
          "observable_contract": metrics["observable_contract"],
          "selected_next_route": metrics["selected_next_route"],
          "fit_decode_allowed": False,
          "validation_or_test_allowed": False,
          "runtime_selected_dimension_source_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} GPU Final-Norm Fit Observable Source",
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
  parser.add_argument("--sequence", type=int, default=581)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq580-state-conditioned-head-observable-route-close-gate-20260710Tseq580Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-final-norm-sparse-state-feature-contract-2026-07-10.json")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq581-final-norm-sparse-state-fit-observable-source-20260710Tseq581Z")
  parser.add_argument("--cxx", default="c++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq581-final-norm-sparse-state-fit-observable-source-gate-20260710Tseq581Z")
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
