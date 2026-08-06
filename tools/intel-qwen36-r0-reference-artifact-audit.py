#!/usr/bin/env python3
"""Audit imported reference artifacts for the intel-qwen36 R0 oracle gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: row must be a JSON object")
      rows.append(value)
  return rows


def check(name: str, passed: bool, **extra: Any) -> dict[str, Any]:
  row = {"name": name, "pass": bool(passed)}
  row.update(extra)
  return row


def audit_token_contract(root: Path) -> dict[str, Any]:
  manifest = load_json(root / "manifest.json")
  correctness = load_json(root / "correctness.json")
  rows = load_jsonl(root / "token-references.jsonl")
  token_ids_match = all(row.get("token_ids_match") is True for row in rows)
  prompt_sets = sorted({row.get("prompt_set") for row in rows})
  checks = [
      check("artifact_present", True),
      check("model_path_matches", manifest.get("model_path") == MODEL_PATH, value=manifest.get("model_path")),
      check("required_checks_passed", correctness.get("required_checks_passed") is True),
      check("six_short_router_cases", len(rows) == 6, value=len(rows)),
      check("prompt_sets_short_router", prompt_sets == ["router-stability", "short"], value=prompt_sets),
      check("token_ids_match", token_ids_match),
  ]
  return {
      "classification": "prompt_token_reference",
      "usable_for": ["prompt bytes", "prompt token ids", "llama.cpp/OpenVINO tokenizer identity"],
      "closes": ["tokenizer/prompt-byte subgate"],
      "does_not_close": ["generation oracle", "teacher-forced distribution oracle", "per-boundary tensor oracle"],
      "checks": checks,
      "passed": all(row["pass"] for row in checks),
  }


def generation_target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  targets = []
  for row in rows:
    for target in row.get("generation_targets", []):
      target_row = dict(target)
      target_row["case_id"] = row.get("case_id")
      target_row["prompt_set"] = row.get("prompt_set")
      targets.append(target_row)
  return targets


def audit_reference_generation_contract(root: Path) -> dict[str, Any]:
  manifest = load_json(root / "manifest.json")
  correctness = load_json(root / "correctness.json")
  rows = load_jsonl(root / "generation-references.jsonl")
  checks_by_name = {row.get("name"): row.get("pass") for row in correctness.get("checks", [])}
  targets = generation_target_rows(rows)
  first_token_ok = all(
      target.get("first_token_text_match") is True
      and target.get("first_token_id_match_when_available") is True
      for target in targets
      if target.get("max_new_tokens") == 1
  )
  short_text_match = checks_by_name.get("short_generation_text_match") is True
  checks = [
      check("artifact_present", True),
      check("model_path_matches", manifest.get("model_path") == MODEL_PATH, value=manifest.get("model_path")),
      check("prompt_cases", len(rows) == 6, value=len(rows)),
      check("first_token_text_and_id_match", first_token_ok),
      check("short_generation_text_match", short_text_match),
      check("required_checks_passed", correctness.get("required_checks_passed") is True),
  ]
  return {
      "classification": "cross_runtime_generation_probe",
      "usable_for": ["first-token llama.cpp/OpenVINO sanity", "OpenVINO denominator boundary"],
      "closes": ["first-token cross-runtime sanity"],
      "does_not_close": ["cross-runtime exact short-generation oracle", "per-boundary tensor oracle"],
      "checks": checks,
      "passed": False,
  }


def audit_llama_generation_oracle(root: Path) -> dict[str, Any]:
  manifest = load_json(root / "manifest.json")
  correctness = load_json(root / "correctness.json")
  rows = load_jsonl(root / "oracle-references.jsonl")
  checks_by_name = {row.get("name"): row.get("pass") for row in correctness.get("checks", [])}
  targets = generation_target_rows(rows)
  token_ids_present = all(target.get("generated_token_ids_present") is True for target in targets)
  top_logprobs_present = all(bool(target.get("top_logprobs")) for target in targets)
  first_targets = [target for target in targets if target.get("max_new_tokens") == 1]
  short_targets = [target for target in targets if target.get("max_new_tokens") == 16]
  first_token_deterministic = all(
      target.get("same_server_token_ids_deterministic") is True for target in first_targets
  )
  short_token_deterministic = all(
      target.get("same_server_token_ids_deterministic") is True for target in short_targets
  )
  checks = [
      check("artifact_present", True),
      check("model_path_matches", manifest.get("model_path") == MODEL_PATH, value=manifest.get("model_path")),
      check("prompt_cases", len(rows) == 6, value=len(rows)),
      check("token_ids_present", token_ids_present),
      check("top_logprobs_present", top_logprobs_present),
      check("first_token_oracle_present", checks_by_name.get("first_token_oracle_present") is True),
      check("short_generation_oracle_present", checks_by_name.get("short_generation_oracle_present") is True),
      check("first_token_ids_deterministic", first_token_deterministic),
      check("short_token_ids_deterministic", short_token_deterministic),
      check("required_checks_passed", correctness.get("required_checks_passed") is True),
  ]
  passed = all(row["pass"] for row in checks)
  return {
      "classification": "llama_cpp_token_topk_oracle_seed",
      "usable_for": ["prompt token ids", "first-token generated ids", "short generated ids", "first-token top-5 logprobs"],
      "closes": ["deterministic token/top-k oracle seed"] if passed else ["partial token/top-k oracle seed"],
      "does_not_close": [
          "teacher-forced distribution ladder",
          "per-boundary reference inputs/outputs",
      ],
      "checks": checks,
      "passed": passed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("artifact_dir", type=Path)
  args = parser.parse_args()

  artifact_dir = args.artifact_dir
  audits = {
      "token_accuracy_contract": audit_token_contract(artifact_dir / "token-accuracy-contract"),
      "reference_generation_contract": audit_reference_generation_contract(
          artifact_dir / "reference-generation-contract"
      ),
      "llama_generation_oracle": audit_llama_generation_oracle(
          artifact_dir / "llama-generation-oracle"
      ),
  }
  result = {
      "schema_version": "0.1",
      "workstream": WORKSTREAM,
      "artifact_dir": str(artifact_dir),
      "audits": audits,
      "r0_oracle_status": {
          "token_reference_seed_available": True,
          "topk_reference_seed_available": audits["llama_generation_oracle"]["passed"],
          "per_boundary_bundle_available": False,
          "r0_oracle_gate_closed": False,
          "next_required_action": "capture per-boundary reference inputs/outputs and teacher-forced distribution references",
      },
  }
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
