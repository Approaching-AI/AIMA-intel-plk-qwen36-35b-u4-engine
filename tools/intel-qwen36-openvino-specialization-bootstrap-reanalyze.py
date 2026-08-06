#!/usr/bin/env python3
"""Re-evaluate a completed OV0 raw bundle without rerunning the 35B model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools/intel-qwen36-openvino-specialization-bootstrap-gate.py"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
SCHEMA = "intel-qwen36-openvino-specialization-bootstrap-reanalysis-v0"


def load_gate() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_ov0_gate", GATE_PATH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load gate module: {GATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  return parser.parse_args()


def source_run(gate: Any, source: Path, name: str) -> dict[str, Any]:
  directory = source / "raw" / name
  record = gate.load_json(directory / "run.json")
  record["result"] = gate.load_json(directory / "worker-result.json")
  return record


def main() -> int:
  args = parse_args()
  gate = load_gate()
  source = args.source.resolve()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  source_manifest = gate.load_json(source / "manifest.json")
  model_identity = gate.load_json(source / "model-identity.json")
  analysis_git = gate.git_state(out_dir)
  analysis_args = SimpleNamespace(
      acceptance=gate.ACCEPTANCE,
      custom_config=gate.CUSTOM_CONFIG,
      materialization_dir=gate.MATERIALIZATION,
      model_contract=gate.MODEL_CONTRACT,
      sentinel_prompts=gate.SENTINEL_PROMPTS,
      short_prompts=gate.SHORT_PROMPTS,
      smoke=False,
      smoke_bucket=8192,
  )
  cases = gate.build_cases(analysis_args)
  mechanism_run = source_run(gate, source, "mechanism")
  stock_run = source_run(gate, source, "stock")
  candidate_run = source_run(gate, source, "candidate")
  correctness, metrics, details = gate.analyze(
      analysis_args, analysis_git, model_identity, cases, mechanism_run,
      stock_run, candidate_run)
  evidence = {
      "source": str(source),
      "source_manifest_sha256": gate.sha256_file(source / "manifest.json"),
      "worker_git": source_manifest["git"],
  }
  correctness["evidence_source"] = evidence
  details["evidence_source"] = evidence
  gate.write_json(out_dir / "model-identity.json", model_identity)
  host = gate.load_json(source / "host.json")
  host["evidence_source"] = evidence
  gate.write_json(out_dir / "host.json", host)
  gate.write_json(out_dir / "correctness.json", correctness)
  gate.write_json(out_dir / "comparison.json", details)
  gate.write_jsonl(out_dir / "metrics.jsonl", metrics)
  gate.write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "notes": (
          "OV0 correctness/mechanism evidence only; paired product context "
          "smoothness begins after OV1."),
      "route_label": correctness["route_label"],
  })
  (out_dir / "summary.md").write_text(
      gate.summary_markdown(correctness, details), encoding="utf-8")
  gate.write_json(out_dir / "manifest.json", {
      "analysis_git": analysis_git,
      "captured_at": gate.iso_now(),
      "evidence_source": evidence,
      "mode": "full_ov0_reanalysis",
      "model_contract": str(gate.MODEL_CONTRACT),
      "route_label": correctness["route_label"],
      "schema_version": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": gate.WORKSTREAM,
  })
  print(json.dumps({
      "event": "reanalysis_complete",
      "out_dir": str(out_dir),
      "ov0_exit": correctness["ov0_exit"],
      "required_checks_passed": correctness["required_checks_passed"],
      "route_label": correctness["route_label"],
  }, sort_keys=True))
  return 0 if correctness["required_checks_passed"] else 2


if __name__ == "__main__":
  try:
    import numpy  # noqa: F401
  except ModuleNotFoundError:
    os.execv(
        str(OV_PYTHON),
        [str(OV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
  raise SystemExit(main())
