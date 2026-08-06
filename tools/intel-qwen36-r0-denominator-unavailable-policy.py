#!/usr/bin/env python3
"""Accept the 262144 denominator lane as unavailable for R0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
POLICY_ID = "r0_262144_denominator_lane_unavailable"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def latest(pattern: str, filename: str) -> Path:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  if not paths:
    raise SystemExit(f"missing artifact: {pattern}/{filename}")
  return paths[-1]


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def build_summary(payload: dict[str, Any]) -> str:
  policy = payload["policy"]
  lines = [
      "# R0 262144 Denominator Unavailable Policy",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- policy id: `{POLICY_ID}`",
      f"- decision: `{policy['decision']}`",
      f"- denominator metric available: `{str(policy['denominator_metric_available']).lower()}`",
      f"- denominator policy gate closed: `{str(policy['denominator_policy_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(policy['speedup_claims_allowed']).lower()}`",
      f"- R0 closed: `{str(policy['r0_closed']).lower()}`",
      "",
      "This policy closes only the R0 interpretation of the missing 262144",
      "same-host denominator lane. It does not create a throughput denominator,",
      "does not authorize speedup claims, and does not close the oracle or",
      "resident-harness gates.",
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
  out_dir = (ROOT / f"output/r0-denominator-unavailable-policy-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  resolution_path = latest(
      "r0-denominator-oracle-boundary-resolution-*",
      "resolution.json",
  )
  resolution = load_json(resolution_path)
  openvino = resolution.get("denominator_262144", {})
  llama = resolution.get("llama_denominator_262144", {})
  cleanup_path_value = llama.get("evidence", {}).get("post_timeout_cleanup")
  cleanup_path = ROOT / cleanup_path_value if isinstance(cleanup_path_value, str) else None
  cleanup = load_json(cleanup_path) if cleanup_path and cleanup_path.exists() else {}

  checks = [
      {
          "name": "openvino_resource_failure_evidence_present",
          "pass": openvino.get("interpretation") == "openvino_262144_resource_failure_not_metric"
          and openvino.get("denominator_metric_available") is False,
          "interpretation": openvino.get("interpretation"),
      },
      {
          "name": "llama_timeout_evidence_present",
          "pass": llama.get("interpretation")
          == "llama_vulkan_262144_timeout_no_metric_cleanup_complete"
          and llama.get("denominator_metric_available") is False,
          "interpretation": llama.get("interpretation"),
      },
      {
          "name": "llama_cleanup_complete",
          "pass": cleanup.get("post_cleanup_status", {}).get("lingering_llama_bench_process") is False,
          "cleanup_path": rel(cleanup_path),
      },
      {
          "name": "no_denominator_metric_claimed",
          "pass": openvino.get("denominator_metric_available") is False
          and llama.get("denominator_metric_available") is False,
      },
      {
          "name": "r0_still_open",
          "pass": resolution.get("r0_gate_status", {}).get("r0_closed") is False,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)

  policy = {
      "decision": "accept_262144_denominator_lane_unavailable_for_r0",
      "denominator_metric_available": False,
      "denominator_policy_gate_closed": required_checks_passed,
      "forbidden_claims": [
          "262144 speedup versus OpenVINO",
          "262144 speedup versus llama.cpp",
          "full-ladder denominator coverage",
          "product-performance claim at 262144",
      ],
      "r0_closed": False,
      "r0_denominator_gate_status": "closed_by_unavailable_lane_policy"
      if required_checks_passed
      else "open_policy_checks_failed",
      "scope": {
          "bucket": 262144,
          "cache_state": "cold_no_prefix",
          "model": "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf",
          "workstream": WORKSTREAM,
      },
      "speedup_claims_allowed": False,
      "usable_for": [
          "R0 denominator-lane interpretation",
          "documenting denominator unavailable on current target",
          "preventing repeated OpenVINO/llama.cpp 262144 denominator retries without a new mechanism",
      ],
  }
  required_followups = [
      "full oracle bundle capture including full-ladder teacher-forced distributions",
      "per-boundary reference input/output tensors",
      "resident harness load(model, oracle_bundle) against the real bundle",
      "new denominator policy revision before any future 262144 speedup claim",
  ]
  payload = {
      "created_at": created_at,
      "evidence": {
          "llama_cleanup": rel(cleanup_path),
          "llama_denominator_262144_row": llama.get("evidence", {}).get("row_path"),
          "openvino_denominator_262144_row": openvino.get("evidence", {}).get("row_path"),
          "resolution": rel(resolution_path),
      },
      "policy": policy,
      "policy_id": POLICY_ID,
      "required_followups": required_followups,
      "schema_version": "intel-qwen36-r0-denominator-unavailable-policy-v0",
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "policy_id": POLICY_ID,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-denominator-unavailable-policy.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "policy.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_262144_denominator_unavailable_policy",
      "required_checks_passed": required_checks_passed,
      "schema_version": payload["schema_version"],
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("denominator_metric_available", False),
        ("denominator_policy_gate_closed", policy["denominator_policy_gate_closed"]),
        ("speedup_claims_allowed", False),
        ("r0_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_denominator_unavailable_policy",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"denominator unavailable policy output: {out_dir}")


if __name__ == "__main__":
  main()
