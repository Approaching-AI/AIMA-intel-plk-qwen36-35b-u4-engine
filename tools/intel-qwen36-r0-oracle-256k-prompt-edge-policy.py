#!/usr/bin/env python3
"""Accept the exact 262144-token first-token top-k row as a prompt-edge policy case."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
POLICY_ID = "r0_256k_exact_prompt_first_token_topk_edge"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-256k-prompt-edge-policy-v0"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
CONTEXT_LENGTH = 262144
EXPECTED_CASES = ("sentinel_256k", "prefill_shape_256k")


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      value = json.loads(text)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


def latest(pattern: str, filename: str) -> Path:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  if not paths:
    raise SystemExit(f"missing artifact: {pattern}/{filename}")
  return paths[-1]


def topk_attempt_dir_from_contract() -> Path:
  oracle = load_json(ROOT / "oracle/oracle-bundle-contract.json")
  attempt = (
      oracle.get("capture_plan", {})
      .get("latest_oracle_topk_256k_exact_context_attempt", {})
  )
  path = attempt.get("path")
  if not isinstance(path, str) or not path:
    raise SystemExit("oracle contract missing latest 256k exact-context attempt")
  return ROOT / path


def build_summary(payload: dict[str, Any]) -> str:
  policy = payload["policy"]
  lines = [
      "# R0 Oracle 256k Prompt-Edge Policy",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- policy id: `{POLICY_ID}`",
      f"- decision: `{policy['decision']}`",
      f"- prompt-edge policy gate closed: `{str(policy['prompt_edge_policy_gate_closed']).lower()}`",
      f"- top-k logits available for exact 256k row: `{str(policy['topk_logprobs_available']).lower()}`",
      f"- R0 oracle gate closed: `{str(policy['r0_oracle_gate_closed']).lower()}`",
      "",
      "The exact 262144-token prompt is valid prompt/tokenization evidence,",
      "but first-token prediction would require one additional context slot.",
      "Under the locked model context contract, this exact-context top-k row is",
      "therefore policy-resolved as an edge case rather than captured logits.",
      "",
      "Required follow-up gates:",
      "",
  ]
  for item in payload["required_followups"]:
    lines.append(f"- {item}")
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-oracle-256k-prompt-edge-policy-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  model_contract = load_json(model_contract_path)
  model = model_contract.get("model", {})

  materialization_path = latest("r0-oracle-prompt-materialization-*", "materialization.json")
  materialization_jsonl = materialization_path.parent / "materialized-prompts.jsonl"
  materialized_rows = {
      row.get("case_id"): row for row in load_jsonl(materialization_jsonl)
  }

  token_id_capture_path = latest("r0-oracle-token-id-capture-*", "capture.json")
  token_id_jsonl = token_id_capture_path.parent / "prompt-token-id-references.jsonl"
  token_rows = {row.get("case_id"): row for row in load_jsonl(token_id_jsonl)}

  attempt_dir = topk_attempt_dir_from_contract()
  attempt_capture_path = attempt_dir / "capture.json"
  attempt_correctness_path = attempt_dir / "correctness.json"
  attempt_jsonl_path = attempt_dir / "topk-smoke.jsonl"
  attempt_capture = load_json(attempt_capture_path)
  attempt_correctness = load_json(attempt_correctness_path)
  attempt_rows = load_jsonl(attempt_jsonl_path)

  raw_errors: dict[str, dict[str, Any]] = {}
  for case_number, case_id in enumerate(EXPECTED_CASES, start=1):
    raw_path = (
        attempt_dir
        / f"case-{case_number:02d}-{case_id}"
        / "raw"
        / "remote"
        / "completion_response.raw"
    )
    raw_errors[case_id] = load_json(raw_path).get("error", {})

  materialized_ok = all(
      materialized_rows.get(case_id, {}).get("observed_prompt_tokens") == CONTEXT_LENGTH
      and materialized_rows.get(case_id, {}).get("target_prompt_tokens") == CONTEXT_LENGTH
      for case_id in EXPECTED_CASES
  )
  token_ids_ok = all(
      token_rows.get(case_id, {}).get("prompt_token_count") == CONTEXT_LENGTH
      for case_id in EXPECTED_CASES
  )
  attempt_cases = [case.get("case_id") for case in attempt_capture.get("case_results", [])]
  attempt_failed_as_expected = (
      attempt_capture.get("required_checks_passed") is False
      and attempt_correctness.get("required_checks_passed") is False
      and attempt_cases == list(EXPECTED_CASES)
      and all(
          case.get("expected_prompt_token_count") == CONTEXT_LENGTH
          and case.get("prompt_token_count") == CONTEXT_LENGTH
          and case.get("request_status") == 400
          and case.get("result_ok") is False
          and case.get("top_logprob_count") == 0
          for case in attempt_capture.get("case_results", [])
      )
  )
  attempt_rows_ok = (
      [row.get("case_id") for row in attempt_rows] == list(EXPECTED_CASES)
      and all(
          row.get("prompt_token_count") == CONTEXT_LENGTH
          and row.get("request_status") == 400
          and row.get("first_token", {}).get("top_logprobs") == []
          for row in attempt_rows
      )
  )
  raw_errors_ok = all(
      error.get("code") == 400
      and error.get("type") == "exceed_context_size_error"
      and error.get("n_prompt_tokens") == CONTEXT_LENGTH
      and error.get("n_ctx") == CONTEXT_LENGTH
      for error in raw_errors.values()
  )

  checks = [
      {
          "name": "model_context_contract_locked",
          "pass": model.get("context_length") == CONTEXT_LENGTH
          and model.get("gguf_model_path") == MODEL_PATH
          and model.get("gguf_sha256") == MODEL_SHA256,
          "context_length": model.get("context_length"),
      },
      {
          "name": "materialized_256k_prompts_are_exact",
          "pass": materialized_ok,
          "cases": list(EXPECTED_CASES),
      },
      {
          "name": "token_id_capture_has_exact_256k_rows",
          "pass": token_ids_ok,
          "cases": list(EXPECTED_CASES),
      },
      {
          "name": "exact_context_topk_attempt_failed_as_expected",
          "pass": attempt_failed_as_expected,
          "attempt": rel(attempt_dir),
      },
      {
          "name": "exact_context_topk_rows_have_no_logits",
          "pass": attempt_rows_ok,
      },
      {
          "name": "llama_server_reported_context_edge",
          "pass": raw_errors_ok,
          "failure_type": "exceed_context_size_error",
      },
      {
          "name": "policy_does_not_close_oracle_gate",
          "pass": True,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)

  policy = {
      "context_length": CONTEXT_LENGTH,
      "context_safe_max_prompt_tokens_for_first_token_prediction": CONTEXT_LENGTH - 1,
      "decision": "accept_exact_262144_prompt_first_token_topk_as_context_edge_for_r0",
      "exact_prompt_token_count": CONTEXT_LENGTH,
      "forbidden_claims": [
          "captured exact-262144 first-token top-k logits",
          "full-ladder token/top-k oracle coverage without explicit prompt-edge rows or superseding capture",
          "R0 oracle closure from this policy alone",
          "product correctness at 262144 based only on this policy",
      ],
      "prompt_edge_policy_gate_closed": required_checks_passed,
      "r0_oracle_gate_closed": False,
      "scope": {
          "case_ids": list(EXPECTED_CASES),
          "model": MODEL_PATH,
          "model_sha256": MODEL_SHA256,
          "workstream": WORKSTREAM,
      },
      "topk_logprobs_available": False,
      "usable_for": [
          "documenting why exact 262144-token next-token top-k is not captured under n_ctx=262144",
          "preventing repeated exact-context llama.cpp CPU top-k attempts without a changed mechanism",
          "allowing later oracle bundle manifests to encode explicit prompt-edge rows instead of silent missing logits",
      ],
  }
  required_followups = [
      "full-ladder teacher-forced distribution capture",
      "per-boundary reference input/output tensors",
      "full oracle bundle validation with explicit handling for 256k prompt-edge rows",
      "resident harness load(model, oracle_bundle) against the real bundle",
      "superseding capture or ADR before claiming exact 262144 first-token top-k logits",
  ]
  payload = {
      "created_at": created_at,
      "evidence": {
          "exact_context_attempt": rel(attempt_dir),
          "materialization": rel(materialization_path.parent),
          "model_contract": rel(model_contract_path),
          "token_id_capture": rel(token_id_capture_path.parent),
      },
      "policy": policy,
      "policy_id": POLICY_ID,
      "required_followups": required_followups,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "policy_id": POLICY_ID,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-256k-prompt-edge-policy.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "policy.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_256k_prompt_edge_policy",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in (
        ("prompt_edge_policy_gate_closed", policy["prompt_edge_policy_gate_closed"]),
        ("topk_logprobs_available", False),
        ("exact_prompt_token_count", CONTEXT_LENGTH),
        ("context_safe_max_prompt_tokens_for_first_token_prediction", CONTEXT_LENGTH - 1),
        ("r0_oracle_gate_closed", False),
    ):
      handle.write(json.dumps({
          "metric": metric,
          "phase": "r0_oracle_256k_prompt_edge_policy",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle 256k prompt-edge policy output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
