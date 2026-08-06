#!/usr/bin/env python3
"""Lock the fresh state-conditioned head-correction corpus contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-state-conditioned-head-correction-corpus-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "corpus_contract_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "corpus_token_materialization_gate"
)
EXPECTED_DOMAINS = {
    "arithmetic", "code", "instruction", "factual", "transformation",
    "structured_extraction",
}
EXPECTED_SPLITS = {"fit": 12, "validation": 6, "test": 6}
ACCEPTANCE_CASES = {
    "short_math_001", "short_factual_002", "short_transform_003",
    "router_math_reason_001", "router_code_reason_002",
    "router_instruction_003",
}


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


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
  return _sha256_bytes(value.encode("utf-8"))


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


def _benchmark_prompts(prompt_dir: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in sorted(prompt_dir.glob("*.jsonl")):
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1):
      if not line.strip():
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise TypeError(f"{path}:{line_number} is not a JSON object")
      rows.append(row)
  return rows


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  contract = _load(args.contract)
  prompts = contract.get("prompts")
  prompt_rows = prompts if isinstance(prompts, list) else []
  ids = [row.get("id") for row in prompt_rows if isinstance(row, dict)]
  texts = [row.get("prompt") for row in prompt_rows if isinstance(row, dict)]
  domains = [row.get("domain") for row in prompt_rows if isinstance(row, dict)]
  splits = [row.get("split") for row in prompt_rows if isinstance(row, dict)]
  split_counts = Counter(value for value in splits if isinstance(value, str))
  domain_counts = Counter(value for value in domains if isinstance(value, str))
  domain_split_counts = Counter(
      (row.get("domain"), row.get("split"))
      for row in prompt_rows if isinstance(row, dict))
  benchmark_rows = _benchmark_prompts(args.prompt_dir)
  benchmark_text_hashes = {
      _sha256_text(row["prompt"])
      for row in benchmark_rows if isinstance(row.get("prompt"), str)
  }
  prompt_hashes = [
      _sha256_text(text) for text in texts if isinstance(text, str)
  ]
  route_contract = predecessor.get("corpus_contract")
  model = contract.get("model_contract")
  policy = contract.get("data_policy")
  correctness = contract.get("correctness_contract")
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("corpus_contract_gate_allowed") is True
      and predecessor.get("decode_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 569, CURRENT_ROUTE)
      and _has_switch(
          routes, 569,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_corpus_contract_gate"))
  shape_passes = (
      len(prompt_rows) == 24
      and len(ids) == 24 and len(set(ids)) == 24
      and len(texts) == 24 and len(set(texts)) == 24
      and set(domains) == EXPECTED_DOMAINS
      and split_counts == Counter(EXPECTED_SPLITS)
      and all(domain_counts[domain] == 4 for domain in EXPECTED_DOMAINS)
      and all(domain_split_counts[(domain, "fit")] == 2
              and domain_split_counts[(domain, "validation")] == 1
              and domain_split_counts[(domain, "test")] == 1
              for domain in EXPECTED_DOMAINS))
  text_passes = (
      len(prompt_hashes) == 24
      and len(set(prompt_hashes)) == 24
      and not (set(prompt_hashes) & benchmark_text_hashes)
      and all(isinstance(text, str) and text and text.isascii()
              for text in texts)
      and not (set(ids) & ACCEPTANCE_CASES))
  policy_passes = (
      isinstance(policy, dict)
      and policy.get("split_locked_before_tokenization") is True
      and policy.get("split_locked_before_decode") is True
      and policy.get("acceptance_prompt_text_or_ids_allowed_in_fit") is False
      and policy.get("validation_or_test_results_allowed_to_change_model") is False
      and policy.get("runtime_native_oracle_allowed") is False
      and policy.get("runtime_prompt_case_or_position_features_allowed") is False)
  model_passes = (
      isinstance(model, dict)
      and model.get("maximum_token_ids") == 16
      and model.get("maximum_parameters") == 32
      and model.get("parameters_per_token") == 2
      and model.get("allowed_runtime_features") == [
          "gpu_top8_token_ids", "gpu_top8_logits", "top1_margin"]
      and model.get("full_vocab_host_rescan_in_speed_lane_allowed") is False
      and isinstance(model.get("implementation_added_wall_us_per_token_max"),
                     (int, float))
      and abs(float(model["implementation_added_wall_us_per_token_max"])
              - 20.525576981789345) <= 1e-12)
  correctness_passes = (
      isinstance(correctness, dict)
      and correctness.get("top1_rate_min") == 1.0
      and correctness.get("max_kld") == 0.005
      and correctness.get("per_case_kld_regression_epsilon") == 1e-7
      and correctness.get("validation_and_test_must_both_pass") is True
      and set(correctness.get("acceptance_recheck_after_fresh_test", []))
      == ACCEPTANCE_CASES)
  route_contract_passes = (
      isinstance(route_contract, dict)
      and route_contract.get("new_prompt_count") == 24
      and set(route_contract.get("domains", [])) == EXPECTED_DOMAINS
      and route_contract.get("split") == EXPECTED_SPLITS
      and route_contract.get("maximum_token_ids") == 16
      and route_contract.get("maximum_parameters") == 32
      and route_contract.get("prompt_case_position_features_allowed") is False
      and route_contract.get("runtime_native_oracle_allowed") is False)
  checks = [
      {"name": "seq569_selected_fresh_corpus_contract_gate",
       "pass": predecessor_selects},
      {"name": "corpus_has_balanced_locked_24_prompt_shape",
       "pass": shape_passes},
      {"name": "prompt_ids_and_text_are_fresh_unique_ascii",
       "pass": text_passes},
      {"name": "data_policy_forbids_acceptance_fit_and_holdout_retuning",
       "pass": policy_passes},
      {"name": "top8_model_class_and_floor_budget_are_locked",
       "pass": model_passes},
      {"name": "correctness_contract_is_locked", "pass": correctness_passes},
      {"name": "contract_matches_seq569_route_control",
       "pass": route_contract_passes},
      {"name": "no_token_materialization_or_decode_exists_yet",
       "pass": not args.materialize_dir.exists()},
  ]
  required = all(bool(row["pass"]) for row in checks)
  prompt_manifest = [
      {
          "id": row["id"],
          "domain": row["domain"],
          "split": row["split"],
          "prompt_sha256": _sha256_text(row["prompt"]),
          "prompt_utf8_bytes": len(row["prompt"].encode("utf-8")),
      }
      for row in prompt_rows
      if isinstance(row, dict)
      and all(isinstance(row.get(key), str)
              for key in ("id", "domain", "split", "prompt"))
  ]
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "contract": _rel(args.contract),
          "contract_sha256": _sha256_bytes(args.contract.read_bytes()),
          "benchmark_prompt_dir": _rel(args.prompt_dir),
          "planned_materialize_dir": _rel(args.materialize_dir),
      },
      "corpus": {
          "prompt_count": len(prompt_rows),
          "domain_counts": dict(sorted(domain_counts.items())),
          "split_counts": dict(sorted(split_counts.items())),
          "prompt_manifest": prompt_manifest,
      },
      "checks": checks,
      "required_checks_passed": required,
      "token_materialization_allowed": required,
      "decode_row_allowed": False,
      "model_fit_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_fresh_state_conditioned_head_correction_corpus_contract"
          if required else "reject_fresh_corpus_contract"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The 24 prompts are unique, ASCII, non-overlapping with benchmark "
          "prompts, domain-balanced, and split 12/6/6 before tokenization. "
          "The top8-only 32-parameter ceiling, no-retuning policy, correctness "
          "ruler, and 20.52557698 us/token implementation budget are locked. "
          "Materialize token IDs next without decoding."
          if required else
          "Fix corpus freshness, split balance, policy, model, or budget before "
          "tokenization or decode."),
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
          "corpus": metrics["corpus"],
          "selected_next_route": metrics["selected_next_route"],
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} State-Conditioned Corpus Contract Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- prompt count: `{metrics['corpus']['prompt_count']}`",
      f"- split counts: `{metrics['corpus']['split_counts']}`",
      f"- token_materialization_allowed: `{str(metrics['token_materialization_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a corpus contract. No prompt was tokenized or decoded.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=570)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq569-post-static-bias-route-control-gate-20260710Tseq569Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument(
      "--prompt-dir", type=Path,
      default=ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts")
  parser.add_argument(
      "--materialize-dir", type=Path,
      default=ROOT / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq570-state-conditioned-head-correction-corpus-gate-20260710Tseq570Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "token_materialization_allowed": metrics["token_materialization_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
