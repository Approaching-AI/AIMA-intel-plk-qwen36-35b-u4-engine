#!/usr/bin/env python3
"""Audit prior same-model performance artifacts for the intel-qwen36 R0 gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL_GGUF = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_OV = "/home/intel/Qwen3.6-35B-A3B-ov"
REQUIRED_BUCKETS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400, 131072, 262144]

INTEL_BOX_OUTPUT = Path("/Users/jiawei-macmini/projects/intel-box/output")
DEFAULT_OPENVINO_DENOMINATOR = (
    INTEL_BOX_OUTPUT / "intel-box-qwen36-openvino-required-r1-20260613T153430Z"
)
DEFAULT_SOURCE_STREAM_ROOF = (
    INTEL_BOX_OUTPUT / "intel-box-qwen36-native-source-stream-roof-20260615T121249Z"
    / "remote-output/source-stream-roof"
)
DEFAULT_FULL_LAYER_BYTE_BUDGET = (
    INTEL_BOX_OUTPUT / "intel-box-qwen36-native-full-layer-byte-budget-20260615T114027Z"
    / "remote-output/full-layer-byte-budget"
)
DEFAULT_QMATVEC_PROBE = (
    INTEL_BOX_OUTPUT / "intel-box-qwen36-native-llama-generation-oracle-cpu-20260615T133419Z"
    / "remote-output/qmatvec-packed-lowbit"
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--openvino-denominator", type=Path, default=DEFAULT_OPENVINO_DENOMINATOR)
  parser.add_argument("--source-stream-roof", type=Path, default=DEFAULT_SOURCE_STREAM_ROOF)
  parser.add_argument("--full-layer-byte-budget", type=Path, default=DEFAULT_FULL_LAYER_BYTE_BUDGET)
  parser.add_argument("--qmatvec-probe", type=Path, default=DEFAULT_QMATVEC_PROBE)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-performance-artifact-audit-<UTC>.",
  )
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def int_bucket(value: Any) -> int | None:
  if isinstance(value, int):
    return value
  if isinstance(value, str):
    lower = value.lower().strip()
    if lower.endswith("k") and lower[:-1].isdigit():
      return int(lower[:-1]) * 1024
    if lower.endswith("k") and lower[:-1] == "100":
      return 102400
    if lower.isdigit():
      return int(lower)
  return None


def audit_openvino_denominator(path: Path) -> dict[str, Any]:
  metrics_path = path / "metrics.jsonl"
  manifest_path = path / "manifest.json"
  metrics = load_jsonl(metrics_path)
  manifest = load_json(manifest_path)
  rows = [row for row in metrics if row.get("phase") == "openvino_denominator"]
  buckets = sorted({bucket for bucket in (int_bucket(row.get("bucket")) for row in rows) if bucket})
  missing = sorted(set(REQUIRED_BUCKETS) - set(buckets))
  prefill_ratios = [float(row["prefill_target_ratio"]) for row in rows if isinstance(row.get("prefill_target_ratio"), (int, float))]
  decode_ratios = [float(row["decode_target_ratio"]) for row in rows if isinstance(row.get("decode_target_ratio"), (int, float))]
  return {
      "artifact_path": str(path),
      "buckets": buckets,
      "captured_at": manifest.get("captured_at"),
      "classification": "same_host_openvino_denominator_seed",
      "coverage": {
          "missing_required_buckets": missing,
          "required_buckets": REQUIRED_BUCKETS,
      },
      "device": manifest.get("config", {}).get("device"),
      "gate_closed": False,
      "max_decode_target_ratio": max(decode_ratios) if decode_ratios else None,
      "max_prefill_target_ratio": max(prefill_ratios) if prefill_ratios else None,
      "min_decode_target_ratio": min(decode_ratios) if decode_ratios else None,
      "min_prefill_target_ratio": min(prefill_ratios) if prefill_ratios else None,
      "model_path": manifest.get("config", {}).get("model_path"),
      "reason_not_closed": "prior diagnostic run lacks the 262144 bucket, has one repeat, and predates the current live target recapture",
      "route_label": manifest.get("route_label"),
      "row_count": len(rows),
      "usable_for": ["same-host denominator seed", "OpenVINO floor shape"],
  }


def audit_source_stream_roof(path: Path) -> dict[str, Any]:
  metrics = load_jsonl(path / "metrics.jsonl")
  correctness = load_json(path / "correctness.json")
  stream_rows = [row for row in metrics if row.get("phase") == "opencl_source_stream"]
  gb_s = [float(row["source_effective_gb_s"]) for row in stream_rows if isinstance(row.get("source_effective_gb_s"), (int, float))]
  ratios = [
      float(row["source_effective_gb_s"]) / float(row["target_gb_s"])
      for row in stream_rows
      if isinstance(row.get("source_effective_gb_s"), (int, float))
      and isinstance(row.get("target_gb_s"), (int, float))
      and float(row["target_gb_s"]) > 0
  ]
  return {
      "artifact_path": str(path),
      "classification": "native_source_stream_roof_seed",
      "gate": correctness.get("gate"),
      "gate_closed": False,
      "max_source_gb_s": max(gb_s) if gb_s else None,
      "max_target_ratio": max(ratios) if ratios else None,
      "min_source_gb_s": min(gb_s) if gb_s else None,
      "min_target_ratio": min(ratios) if ratios else None,
      "reason_not_closed": correctness.get("decision", {}).get("reason"),
      "required_checks_passed": correctness.get("required_checks_passed"),
      "route_label": correctness.get("route_label"),
      "target_gb_s": correctness.get("decision", {}).get("target_gb_s"),
      "usable_for": ["route rejection evidence", "raw low-bit source bandwidth seed"],
  }


def audit_full_layer_byte_budget(path: Path) -> dict[str, Any]:
  metrics = load_jsonl(path / "metrics.jsonl")
  decode_rows = [row for row in metrics if row.get("phase") == "decode_bucket_budget"]
  prefill_rows = [row for row in metrics if row.get("phase") == "prefill_bucket_budget"]
  decode_buckets = sorted({int(row["bucket"]) for row in decode_rows if isinstance(row.get("bucket"), int)})
  missing = sorted(set(REQUIRED_BUCKETS) - set(decode_buckets))
  q6_ratios = [
      float(row["observed_q6_plus_kv_target_ratio"])
      for row in decode_rows
      if isinstance(row.get("observed_q6_plus_kv_target_ratio"), (int, float))
  ]
  type_weighted = [
      float(row["type_weighted_plus_kv_target_ratio"])
      for row in decode_rows
      if isinstance(row.get("type_weighted_plus_kv_target_ratio"), (int, float))
  ]
  return {
      "artifact_path": str(path),
      "classification": "full_layer_byte_budget_seed",
      "decode_buckets": decode_buckets,
      "gate_closed": False,
      "max_q6_plus_kv_target_ratio": max(q6_ratios) if q6_ratios else None,
      "max_type_weighted_plus_kv_target_ratio": max(type_weighted) if type_weighted else None,
      "min_q6_plus_kv_target_ratio": min(q6_ratios) if q6_ratios else None,
      "missing_required_decode_buckets": missing,
      "prefill_budget_rows": len(prefill_rows),
      "reason_not_closed": "byte-budget artifact is route-label rejected and lacks 262144 decode coverage",
      "route_label": "rejected",
      "usable_for": ["decode kill-number seed", "route-selection pressure"],
  }


def audit_qmatvec(path: Path) -> dict[str, Any]:
  metrics = load_jsonl(path / "metrics.jsonl")
  correctness = load_json(path / "correctness.json")
  rows = [row for row in metrics if row.get("phase") == "opencl_qmatvec"]
  gb_s = [float(row["effective_tensor_gb_s"]) for row in rows if isinstance(row.get("effective_tensor_gb_s"), (int, float))]
  checks = correctness.get("checks", [])
  rel_l2_values = [
      float(check["opencl_relative_l2"])
      for check in checks
      if isinstance(check, dict) and isinstance(check.get("opencl_relative_l2"), (int, float))
  ]
  cosine_values = [
      float(check["opencl_cosine"])
      for check in checks
      if isinstance(check, dict) and isinstance(check.get("opencl_cosine"), (int, float))
  ]
  return {
      "artifact_path": str(path),
      "classification": "real_tensor_qmatvec_seed",
      "gate_closed": False,
      "max_effective_tensor_gb_s": max(gb_s) if gb_s else None,
      "max_relative_l2": max(rel_l2_values) if rel_l2_values else None,
      "min_cosine": min(cosine_values) if cosine_values else None,
      "reason_not_closed": "diagnostic two-tensor qmatvec coverage is not a full model-real roofline or accepted route",
      "required_checks_passed": correctness.get("required_checks_passed"),
      "route_label": "diagnostic",
      "usable_for": ["real K-quant unpack numeric seed", "M=1 qmatvec bandwidth seed"],
  }


def build_summary(audit: dict[str, Any]) -> str:
  ov = audit["artifacts"]["openvino_denominator"]
  roof = audit["artifacts"]["source_stream_roof"]
  budget = audit["artifacts"]["full_layer_byte_budget"]
  qmv = audit["artifacts"]["qmatvec"]
  return "\n".join(
      [
          "# R0 performance artifact audit",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- R0 performance gate closed: `{str(audit['r0_performance_gate_closed']).lower()}`",
          f"- OpenVINO denominator rows: {ov['row_count']}, missing buckets: {ov['coverage']['missing_required_buckets']}",
          f"- source-stream roof: {roof['min_source_gb_s']:.2f}-{roof['max_source_gb_s']:.2f} GB/s, route `{roof['route_label']}`",
          f"- full-layer Q6+KV target ratio: {budget['min_q6_plus_kv_target_ratio']:.3f}-{budget['max_q6_plus_kv_target_ratio']:.3f}",
          f"- qmatvec max tensor GB/s: {qmv['max_effective_tensor_gb_s']:.2f}",
          "",
          "These are seed/diagnostic artifacts from the prior same-model workstream.",
          "They do not close R0 because the live target was recaptured later and",
          "the required 262144 denominator/byte-budget coverage is incomplete.",
          "",
      ]
  )


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-performance-artifact-audit-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  artifacts = {
      "openvino_denominator": audit_openvino_denominator(args.openvino_denominator.resolve()),
      "source_stream_roof": audit_source_stream_roof(args.source_stream_roof.resolve()),
      "full_layer_byte_budget": audit_full_layer_byte_budget(args.full_layer_byte_budget.resolve()),
      "qmatvec": audit_qmatvec(args.qmatvec_probe.resolve()),
  }
  audit = {
      "artifacts": artifacts,
      "created_at": created_at,
      "model": {
          "gguf_path": MODEL_GGUF,
          "openvino_path": MODEL_OV,
      },
      "next_required_actions": [
          "rerun same-host denominator on the current target contract including 262144",
          "rerun model-real roofline probes on the current target contract",
          "change route away from raw OpenCL GGUF source-stream qmatvec unless a new feasibility probe clears the bandwidth class",
      ],
      "r0_performance_gate_closed": False,
      "schema_version": "intel-qwen36-r0-performance-artifact-audit-v0",
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "audit.json", audit)
  (out_dir / "summary.md").write_text(build_summary(audit), encoding="utf-8")
  print(f"performance audit output: {out_dir}")


if __name__ == "__main__":
  main()
