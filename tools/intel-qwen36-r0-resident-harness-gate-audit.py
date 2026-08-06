#!/usr/bin/env python3
"""Audit the R0 resident harness load gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-resident-harness-gate-audit-v0"
REQUIRED_BUNDLE_PATHS = [
    "manifest.json",
    "correctness.json",
    "token-topk-references.jsonl",
    "teacher-forced-distribution-references.jsonl",
    "boundary-references/inputs.jsonl",
    "boundary-references/outputs.jsonl",
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def candidate_bundle_paths() -> list[Path]:
  candidates = []
  root = ROOT / "oracle"
  if root.exists():
    for path in root.iterdir():
      if path.is_dir():
        candidates.append(path)
  return sorted(candidates)


def bundle_status(path: Path) -> dict[str, Any]:
  present = []
  missing = []
  for relative in REQUIRED_BUNDLE_PATHS:
    required_path = path / relative
    if required_path.is_file():
      present.append(relative)
    else:
      missing.append(relative)
  return {
      "path": rel(path),
      "present_required_paths": present,
      "missing_required_paths": missing,
      "loadable_by_structure": not missing,
  }


def build_summary(payload: dict[str, Any]) -> str:
  gate = payload["resident_harness_gate"]
  lines = [
      "# R0 Resident Harness Gate Audit",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- locked model path enforced: `{str(gate['locked_model_path_enforced']).lower()}`",
      f"- oracle bundle directory required: `{str(gate['oracle_bundle_directory_required']).lower()}`",
      f"- required bundle files: {len(REQUIRED_BUNDLE_PATHS)}",
      f"- candidate real bundles found: {gate['candidate_real_bundle_count']}",
      f"- resident harness load executed: `{str(gate['resident_harness_load_executed']).lower()}`",
      f"- resident harness load artifact: `{gate['resident_harness_load_artifact']}`",
      f"- resident harness gate closed: `{str(gate['r0_resident_harness_gate_closed']).lower()}`",
      "",
      "This audit verifies the load contract and current bundle availability.",
      "It consumes the latest resident harness load artifact; it does not",
      "claim optimized inference performance.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-resident-harness-gate-audit-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  capture_spec_path = latest("r0-oracle-capture-spec-*", "capture-spec.json")
  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  capture_spec = load_json(capture_spec_path) if capture_spec_path else {}
  load_path = latest("r0-resident-harness-load-*", "load.json")
  load_correctness_path = (
      load_path.parent / "correctness.json" if load_path is not None else None
  )
  load_payload = load_json(load_path) if load_path is not None else {}
  load_correctness = (
      load_json(load_correctness_path)
      if load_correctness_path is not None and load_correctness_path.exists()
      else {}
  )
  candidates = [bundle_status(path) for path in candidate_bundle_paths()]
  loadable = [candidate for candidate in candidates if candidate["loadable_by_structure"]]
  r0_oracle_closed = oracle_contract.get("r0_oracle_gate_closed") is True
  load_gate = load_payload.get("resident_harness_load_gate", {})
  latest_bundle = oracle_contract.get("capture_plan", {}).get("latest_oracle_bundle", {})
  resident_harness_load_executed = (
      load_correctness.get("required_checks_passed") is True
      and load_correctness.get("r0_resident_harness_gate_closed") is True
      and load_gate.get("resident_harness_loaded") is True
      and load_gate.get("oracle_bundle_path") == latest_bundle.get("path")
  )
  gate_closed = bool(loadable) and r0_oracle_closed and resident_harness_load_executed

  payload = {
      "created_at": created_at,
      "evidence": {
          "capture_spec": rel(capture_spec_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
      },
      "required_oracle_bundle_paths": REQUIRED_BUNDLE_PATHS,
      "resident_harness_gate": {
          "candidate_bundle_status": candidates,
          "candidate_real_bundle_count": len(loadable),
          "locked_model_path": model_contract["model"]["gguf_model_path"],
          "locked_model_path_enforced": True,
          "oracle_bundle_directory_required": True,
          "r0_oracle_gate_closed": r0_oracle_closed,
          "resident_harness_loaded": load_gate.get("resident_harness_loaded") is True,
          "resident_harness_load_artifact": rel(load_path),
          "resident_harness_load_executed": resident_harness_load_executed,
          "r0_resident_harness_gate_closed": gate_closed,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "capture_spec_available",
          "pass": capture_spec_path is not None
          and capture_spec.get("schema_version") == "intel-qwen36-r0-oracle-capture-spec-v0",
          "path": rel(capture_spec_path),
      },
      {
          "name": "oracle_gate_closed_with_valid_bundle",
          "pass": r0_oracle_closed is True and bool(loadable),
      },
      {
          "name": "structurally_loadable_real_bundle_found",
          "pass": len(loadable) >= 1,
          "candidate_count": len(candidates),
      },
      {
          "name": "resident_harness_load_artifact_passed",
          "pass": resident_harness_load_executed is True,
          "path": rel(load_path),
      },
      {
          "name": "resident_harness_gate_closed",
          "pass": gate_closed is True,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-resident-harness-gate-audit.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "audit.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_resident_harness_load_gate",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("candidate_real_bundle_count", len(loadable)),
        ("required_oracle_bundle_path_count", len(REQUIRED_BUNDLE_PATHS)),
        ("r0_oracle_gate_closed", r0_oracle_closed),
        ("resident_harness_load_executed", resident_harness_load_executed),
        ("r0_resident_harness_gate_closed", gate_closed),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_resident_harness_gate",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"resident harness gate audit output: {out_dir}")


if __name__ == "__main__":
  main()
