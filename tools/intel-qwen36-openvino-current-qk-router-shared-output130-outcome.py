#!/usr/bin/env python3
"""Classify seq2209 and bound a distinct N=1024 shared-pair successor.

This audit never creates a GPU context or loads the model.  It binds the
immutable seq2209 worker, corrects the known horizontal-split census
bookkeeping, compares all 130 saved logit vectors with the accepted carrier,
and decides whether any narrower source-only successor is justified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-"
    "output130-outcome-v1")
SEQ2209 = ROOT / (
    "output/openvino-current-qk-router-shared-output130-correctness-"
    "20260731Tseq2209-clean")
RESULT = SEQ2209 / "result.json"
WORKER_CONFIG = SEQ2209 / "raw/candidate/worker-config.json"
WORKER_RESULT = SEQ2209 / "raw/candidate/worker-result.json"
MANIFEST = SEQ2209 / "manifest.json"
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness")
REFERENCE_CANDIDATE = REFERENCE_ROOT / "candidate/worker-result.json"
REFERENCE_STOCK = REFERENCE_ROOT / "stock/worker-result.json"
QK_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-output512-correctness-"
    "20260731Tseq2200-clean/result.json")
PAIR_COMPONENT = ROOT / (
    "output/openvino-fixed-fc-plugin-phase-provider-"
    "20260718Tseq1429-m1024-optin-manager-t1-t128-t1-cleancommit/"
    "metrics.json")
PAIR_GRAPH = ROOT / (
    "output/openvino-fixed-fc-phase-provider-full-graph-"
    "20260718Tseq1430-m1024-optin-native-manager-t1-cleancommit/"
    "metrics.json")
TRIPLE_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-triple.patch"
EXPECTED_SHA256 = {
    RESULT: "2052aed7022d7dd3d2e56037ee44d0d0465a6508c5fcd08b6d8b9c3317d4cdf8",
    WORKER_CONFIG: "e02bdec6541f1bf6ea574101fed7a14b37ae0c0ffa19a6c1e573ab69bdcc95d3",
    WORKER_RESULT: "81bc4fd6cf8c5fd2d7e2800e51b5e811b9f7240182f2d5c0ceba367ef4a477f5",
    MANIFEST: "803d170b22149dc9ba6a06b912b8529fb51e26c328d8785e33c4e429f8e2a1ba",
    REFERENCE_CANDIDATE: "fa6a4aacdd45251c6818b467477794688754ffc7c5fa744ad9fb22e4961523b3",
    REFERENCE_STOCK: "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215",
    QK_GATE: "ff862015c9cec1aad4fb1c7efa8aa519927417361b480d90d50a95c9292512df",
    PAIR_COMPONENT: "49558a4adde45bb5dfa706d41965b640a5fc30ff1e34dd20cd2d4187313f56fa",
    PAIR_GRAPH: "cfc5da0caf846c21e9098c27e39227d814fe35f6f48812237e258647d50e05d4",
    TRIPLE_PATCH: "ae013a8a610de89d6f8b48971e7238b240db31d2d1d832fce328a6a4290f4420",
}
SEQ2209_COMMIT = "6c5db636aa0c317c3637746da21da902c736c70f"
OUTPUT_TOKENS = 130
KLD_MAX = 0.005
MEMORY_STOP_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True, check=False)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def checkpoints(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
  return {
      int(row["step"]): row
      for row in result.get("distribution_checkpoints", [])
      if isinstance(row, dict) and isinstance(row.get("step"), int)
  }


def checkpoint_path(row: dict[str, Any]) -> Path:
  path = Path(str(row["file"]))
  return path if path.is_absolute() else ROOT / path


def logsumexp(values: np.ndarray) -> float:
  maximum = float(np.max(values))
  return maximum + math.log(float(np.exp(values - maximum).sum()))


def kld(reference: np.ndarray, candidate: np.ndarray) -> float:
  left = reference.astype(np.float64, copy=False)
  right = candidate.astype(np.float64, copy=False)
  left_log = left - logsumexp(left)
  right_log = right - logsumexp(right)
  probability = np.exp(left_log)
  return float(np.sum(probability * (left_log - right_log)))


def horizontal_split_delta(
    baseline: dict[str, int], observed: dict[str, int],
) -> dict[str, int]:
  return {
      key: int(observed.get(key, 0) - baseline.get(key, 0))
      for key in sorted(set(baseline) | set(observed))
      if observed.get(key, 0) != baseline.get(key, 0)
  }


def named_check(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  return next(
      row for row in metrics.get("checks", [])
      if isinstance(row, dict) and row.get("name") == name)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  missing = [display(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit("missing seq2209 outcome inputs: " + ", ".join(missing))

  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  result = load_json(RESULT)
  worker_result = load_json(WORKER_RESULT)
  accepted = load_json(REFERENCE_CANDIDATE)
  stock = load_json(REFERENCE_STOCK)
  qk_gate = load_json(QK_GATE)
  pair_component = load_json(PAIR_COMPONENT)
  pair_graph = load_json(PAIR_GRAPH)
  worker = result.get("worker") or {}
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  status = run(["git", "status", "--porcelain"]).stdout.splitlines()
  ancestor = run(
      ["git", "merge-base", "--is-ancestor", SEQ2209_COMMIT, head])

  new_rows = checkpoints(worker_result)
  accepted_rows = checkpoints(accepted)
  stock_rows = checkpoints(stock)
  distribution_rows: list[dict[str, Any]] = []
  invalid_checkpoint_steps: list[int] = []
  for step in range(OUTPUT_TOKENS):
    new_row = new_rows.get(step)
    accepted_row = accepted_rows.get(step)
    stock_row = stock_rows.get(step)
    if new_row is None or accepted_row is None or stock_row is None:
      invalid_checkpoint_steps.append(step)
      continue
    paths = [
        checkpoint_path(new_row),
        checkpoint_path(accepted_row),
        checkpoint_path(stock_row),
    ]
    if any(not path.is_file() for path in paths):
      invalid_checkpoint_steps.append(step)
      continue
    if any(
        row.get("sha256") != sha256(path)
        for row, path in zip(
            (new_row, accepted_row, stock_row), paths, strict=True)):
      invalid_checkpoint_steps.append(step)
      continue
    new_logits, accepted_logits, stock_logits = (
        np.fromfile(path, dtype=np.float32) for path in paths)
    if not (
        new_logits.shape == accepted_logits.shape == stock_logits.shape
        == (248320,) and
        np.isfinite(new_logits).all() and
        np.isfinite(accepted_logits).all() and
        np.isfinite(stock_logits).all()):
      invalid_checkpoint_steps.append(step)
      continue
    distribution_rows.append({
        "step": step,
        "accepted_to_triple_kld": kld(accepted_logits, new_logits),
        "stock_to_triple_kld": kld(stock_logits, new_logits),
        "stock_to_accepted_kld": kld(stock_logits, accepted_logits),
        "max_abs_vs_accepted": float(
            np.max(np.abs(new_logits - accepted_logits))),
        "mean_abs_vs_accepted": float(
            np.mean(np.abs(new_logits - accepted_logits))),
        "top1_triple": int(np.argmax(new_logits)),
        "top1_accepted": int(np.argmax(accepted_logits)),
        "bitwise_equal": bool(np.array_equal(new_logits, accepted_logits)),
    })

  worst = max(
      distribution_rows,
      key=lambda row: float(row["accepted_to_triple_kld"]))
  top1_mismatches = [
      row for row in distribution_rows
      if row["top1_triple"] != row["top1_accepted"]]
  bitwise_matches = sum(row["bitwise_equal"] for row in distribution_rows)
  accepted_tokens = [
      int(value) for value in accepted["generated_token_ids"][:OUTPUT_TOKENS]]
  triple_tokens = [
      int(value)
      for value in worker_result.get("generated_token_ids", [])[:OUTPUT_TOKENS]]
  token_mismatches = [
      {
          "step": step,
          "accepted": expected,
          "triple": actual,
      }
      for step, (expected, actual) in enumerate(
          zip(accepted_tokens, triple_tokens, strict=False))
      if expected != actual
  ]

  observed_counts = (
      (result.get("execution") or {}).get("executed_type_counts") or {})
  qk_counts = (
      (qk_gate.get("execution") or {}).get("executed_type_counts") or {})
  census_delta = horizontal_split_delta(qk_counts, observed_counts)
  expected_delta = {
      "Crop": 40,
      "FullyConnectedCompressed": -80,
      "Multiply": 40,
      "VariadicSplit": 40,
  }
  pair_decode_check = named_check(
      pair_component, "all_decode_outputs_are_bit_exact_to_stock")
  pair_runtime_check = named_check(
      pair_graph, "minimal_execution_runs_expected_fixed_fc_provider_set")

  checks = [
      check("repository_is_clean_pushed_and_contains_seq2209",
            not status and head == origin_main and ancestor.returncode == 0,
            head=head, origin_main=origin_main, status=status,
            seq2209_is_ancestor=ancestor.returncode == 0),
      check("all_outcome_inputs_have_exact_hashes",
            all(
                observed_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                display(path): digest
                for path, digest in observed_hashes.items()}),
      check("seq2209_worker_completed_once_without_oom",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            guard.get("tripped") is False and
            int(monitor.get("system_available_min_bytes", 0))
                >= MEMORY_STOP_BYTES and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True,
            monitor=monitor),
      check("qk_only_remains_bitwise_exact_output512",
            qk_gate.get("required_checks_passed") is True and
            qk_gate.get("correctness", {}).get(
                "bitwise_checkpoint_count") == 512 and
            qk_gate.get("correctness", {}).get(
                "current_carrier_relative", {}).get("max_kld") == 0.0),
      check("all_40_n1025_triples_activate_with_expected_split_census",
            census_delta == expected_delta and
            observed_counts.get("FullyConnectedCompressed") == 291 and
            (result.get("execution") or {}).get(
                "runtime_profile", {}).get(
                    "fused_shared_triple_count") == 40 and
            (result.get("execution") or {}).get(
                "runtime_profile", {}).get(
                    "unfused_router_gate_count") == 40,
            census_delta=census_delta,
            note=("Crop/Multiply/VariadicSplit +40 is the expected executed "
                  "horizontal-split topology, not a route failure")),
      check("n1025_triple_fails_complete_numeric_boundary",
            not invalid_checkpoint_steps and
            len(distribution_rows) == OUTPUT_TOKENS and
            bitwise_matches == 0 and
            float(worst["accepted_to_triple_kld"]) > KLD_MAX and
            top1_mismatches == [worst] and
            token_mismatches == [
                {"step": 41, "accepted": 1049, "triple": 5141}],
            invalid_checkpoint_steps=invalid_checkpoint_steps,
            bitwise_matches=bitwise_matches,
            worst=worst,
            top1_mismatches=top1_mismatches,
            token_mismatches=token_mismatches),
      check("initial_prefill_passes_but_t1_sequence_accumulates_error",
            distribution_rows[0]["accepted_to_triple_kld"] < KLD_MAX and
            distribution_rows[2]["accepted_to_triple_kld"] > KLD_MAX and
            worst["step"] == 41,
            step0=distribution_rows[0],
            step2=distribution_rows[2],
            worst_step=int(worst["step"])),
      check("n1024_pair_has_distinct_exact_t1_component_evidence",
            pair_component.get("required_checks_passed") is True and
            pair_component.get("rollup", {}).get(
                "decode_compared_rows") == 320 and
            pair_decode_check.get("pass") is True and
            pair_graph.get("required_checks_passed") is True and
            pair_runtime_check.get("pass") is True and
            pair_runtime_check.get("compressed_fc_count") == 331 and
            pair_runtime_check.get("manager_selection_count") == 0,
            pair_decode_rows=pair_component.get(
                "rollup", {}).get("decode_compared_rows"),
            pair_graph_runtime=pair_runtime_check),
      check("no_gpu_compiler_or_model_worker_ran", True,
            gpu_contexts=0, compilers=0, model_workers=0),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_n1025_triple_admit_n1024_stock_pair_source_gate"
      if passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": {
          "head": head,
          "origin_main": origin_main,
          "dirty": bool(status),
          "dirty_paths": status,
      },
      "verdict": verdict,
      "required_checks_passed": passed,
      "triple_route_closed": passed,
      "point_or_formal_performance_admitted": False,
      "n1024_stock_pair_source_gate_admitted": passed,
      "correctness": {
          "row_count": len(distribution_rows),
          "bitwise_matches": bitwise_matches,
          "max_accepted_to_triple_kld": float(
              worst["accepted_to_triple_kld"]),
          "max_kld_step": int(worst["step"]),
          "top1_mismatches": top1_mismatches,
          "token_mismatches": token_mismatches,
          "step0_accepted_to_triple_kld": float(
              distribution_rows[0]["accepted_to_triple_kld"]),
          "step2_accepted_to_triple_kld": float(
              distribution_rows[2]["accepted_to_triple_kld"]),
      },
      "execution": {
          "observed_counts": observed_counts,
          "delta_vs_qk_only": census_delta,
      },
      "next_route": {
          "route": "current_qk_router_shared_n1024_stock_pair",
          "source_contract": {
              "fused_widths": [512, 512],
              "fused_output_width": 1024,
              "shared_scalar_gate_width": 1,
              "router_width": 256,
              "shared_scalar_gate_stays_independent": True,
              "router_stays_independent": True,
              "fixed_fc_manager_enabled": False,
              "expected_fully_connected_compressed": 331,
          },
          "why_distinct": (
              "seq2209's N=1025 triple changes the scalar shared-expert gate "
              "and accumulates T1 error; the admitted successor fuses only "
              "the two N=512 bulk projections, for which prior T1 component "
              "evidence is bit-exact, and does not enable the closed fixed-FC "
              "manager"),
          "requirements": [
              "default-off exact source predicate and mutual exclusion first",
              "bind same-2k arithmetic before one serial build",
              "compile then output130 correctness before any point timing",
              "no unchanged N=1025 triple or fixed-manager product rerun",
          ],
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": metrics["git"],
      "inputs": {
          display(path): digest for path, digest in observed_hashes.items()},
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
  })
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "max_kld": metrics["correctness"][
          "max_accepted_to_triple_kld"],
      "max_kld_step": metrics["correctness"]["max_kld_step"],
      "token_mismatches": token_mismatches,
      "gpu_workers_launched": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
