#!/usr/bin/env python3
"""Run the current resident harness load path against the validated oracle bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-resident-harness-load-v0"
DEFAULT_EXECUTABLE = "build/engine/iq36-load-bundle"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--bundle-dir",
      type=Path,
      default=None,
      help="Oracle bundle directory. Defaults to latest_oracle_bundle in the contract.",
  )
  parser.add_argument(
      "--executable",
      type=Path,
      default=Path(DEFAULT_EXECUTABLE),
      help="iq36-load-bundle executable path.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-resident-harness-load-<UTC>.",
  )
  return parser.parse_args()


def resolve_path(path: Path) -> Path:
  return path if path.is_absolute() else ROOT / path


def build_summary(payload: dict[str, Any]) -> str:
  gate = payload["resident_harness_load_gate"]
  lines = [
      "# R0 Resident Harness Load",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- model path: `{gate['model_path']}`",
      f"- oracle bundle: `{gate['oracle_bundle_path']}`",
      f"- executable: `{gate['executable']}`",
      f"- load return code: {gate['returncode']}",
      f"- resident harness loaded: `{str(gate['resident_harness_loaded']).lower()}`",
      f"- R0 resident harness gate closed: `{str(gate['r0_resident_harness_gate_closed']).lower()}`",
      "",
      "This is the current C++ resident harness contract load path. It verifies",
      "the locked model path and validated oracle bundle layout; it is not an",
      "optimized inference engine or a speed claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      (ROOT / f"output/r0-resident-harness-load-{stamp}").resolve()
      if args.out_dir is None
      else resolve_path(args.out_dir).resolve()
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  model_path = model_contract["model"]["gguf_model_path"]

  latest_bundle = oracle_contract.get("capture_plan", {}).get("latest_oracle_bundle", {})
  bundle_value = args.bundle_dir or Path(latest_bundle.get("path", ""))
  if not str(bundle_value):
    raise SystemExit("no oracle bundle path provided and no latest bundle registered")
  bundle_dir = resolve_path(bundle_value).resolve()
  executable = resolve_path(args.executable).resolve()
  if not executable.is_file():
    raise SystemExit(f"load executable missing: {executable}")

  required_bundle_paths = oracle_contract.get("required_bundle_paths", [])
  missing = [
      relative for relative in required_bundle_paths
      if not (bundle_dir / relative).is_file()
  ]
  if missing:
    raise SystemExit(f"bundle missing required files: {', '.join(missing)}")

  command = [str(executable), model_path, str(bundle_dir)]
  started = time.monotonic()
  completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
  )
  elapsed_s = time.monotonic() - started
  loaded = completed.returncode == 0 and "iq36-load-bundle ok" in completed.stdout
  oracle_gate_closed = oracle_contract.get("r0_oracle_gate_closed") is True
  bundle_claims_closed = latest_bundle.get("r0_oracle_gate_closed") is True
  gate_closed = loaded and oracle_gate_closed and bundle_claims_closed

  payload = {
      "created_at": created_at,
      "evidence": {
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
          "oracle_bundle": rel(bundle_dir),
      },
      "resident_harness_load_gate": {
          "bundle_claims_r0_oracle_gate_closed": bundle_claims_closed,
          "command": command,
          "elapsed_s": elapsed_s,
          "executable": rel(executable),
          "model_path": model_path,
          "oracle_bundle_path": rel(bundle_dir),
          "oracle_contract_gate_closed": oracle_gate_closed,
          "required_bundle_paths": required_bundle_paths,
          "returncode": completed.returncode,
          "r0_resident_harness_gate_closed": gate_closed,
          "resident_harness_loaded": loaded,
          "stderr": completed.stderr,
          "stdout": completed.stdout,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "oracle_contract_gate_closed",
          "pass": oracle_gate_closed,
      },
      {
          "name": "latest_oracle_bundle_claims_gate_closed",
          "pass": bundle_claims_closed,
          "path": latest_bundle.get("path"),
      },
      {
          "name": "bundle_required_paths_present",
          "pass": not missing,
          "missing": missing,
      },
      {
          "name": "resident_harness_load_returned_success",
          "pass": loaded,
          "returncode": completed.returncode,
      },
      {
          "name": "no_speed_claim",
          "pass": True,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-resident-harness-load.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "load.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_resident_harness_load",
      "required_checks_passed": all(check["pass"] for check in checks),
      "r0_resident_harness_gate_closed": gate_closed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("elapsed_s", elapsed_s),
        ("returncode", completed.returncode),
        ("resident_harness_loaded", loaded),
        ("r0_resident_harness_gate_closed", gate_closed),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_resident_harness_load",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"resident harness load output: {out_dir}")
  return 0 if all(check["pass"] for check in checks) else 1


if __name__ == "__main__":
  raise SystemExit(main())
