#!/usr/bin/env python3
"""Consolidate R0 route feasibility evidence into a route decision artifact."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"


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


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.relative_to(ROOT))


def build_summary(payload: dict[str, Any]) -> str:
  summary = payload["summary"]
  selected = payload["selected_next_artifact"]
  lines = [
      "# R0 Route Feasibility Decision",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- dominant route feasibility hard gate closed: `{str(payload['dominant_route_feasibility_hard_gate_closed']).lower()}`",
      f"- current dominant route result: `{summary['current_dominant_route_result']}`",
      f"- raw source-stream max: {summary['source_stream_max_gb_s']} GB/s",
      f"- raw qmatvec max: {summary['qmatvec_max_gb_s']} GB/s",
      f"- 256K fp16/bf16 KV-only ceiling at source-stream max: {summary['kv_262144_ceiling_tok_s_at_source_stream_max']:.3f} tok/s",
      f"- 256K OpenVINO denominator status: `{summary['openvino_262144_status']}`",
      f"- selected next artifact: `{selected['id']}`",
      f"- R0 closed: `{str(payload['r0_closed']).lower()}`",
      "",
      "The raw OpenCL GGUF source-stream/qmatvec route is hard-rejected for",
      "performance work. R0 remains open because the full oracle bundle and",
      "262144 denominator lane are not closed.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-route-feasibility-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  target_contract_path = ROOT / "contracts/intel-qwen36-target-contract.json"
  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  source_path = latest("r0-source-stream-roof-*", "audit.json")
  qmatvec_path = latest("r0-qmatvec-probe-*", "audit.json")
  kv_path = latest("r0-kv-read-pressure-*", "budget.json")
  denominator_path = latest("r0-openvino-denominator-*", "row.json")

  target_contract = load_json(target_contract_path)
  oracle_contract = load_json(oracle_contract_path)
  source_payload = load_json(source_path) if source_path else {}
  qmatvec_payload = load_json(qmatvec_path) if qmatvec_path else {}
  kv_payload = load_json(kv_path) if kv_path else {}
  denominator_payload = load_json(denominator_path) if denominator_path else {}

  source_audit = source_payload.get("audit", {})
  qmatvec_audit = qmatvec_payload.get("audit", {})
  kv_findings = kv_payload.get("findings", {})
  fp16_256k = kv_findings.get("fp16_or_bf16_262144", {})
  denominator_row = denominator_payload.get("row", {})

  source_max = source_audit.get("max_source_gb_s")
  qmatvec_max = qmatvec_audit.get("max_effective_tensor_gb_s")
  target_line = source_audit.get("target_gb_s", 115.0)
  source_ratio = source_audit.get("max_target_ratio")
  qmatvec_ratio = None
  if finite(qmatvec_max) and finite(target_line) and float(target_line) > 0:
    qmatvec_ratio = float(qmatvec_max) / float(target_line)

  openvino_failure = denominator_row.get("failure_class")
  if openvino_failure is None:
    stderr_path = denominator_row.get("raw", {}).get("stderr")
    if isinstance(stderr_path, str):
      path = Path(stderr_path)
      if path.exists() and "CL_OUT_OF_RESOURCES" in path.read_text(encoding="utf-8"):
        openvino_failure = "openvino_gpu_cl_out_of_resources"
  openvino_status = (
      "resource_failed_at_262144_mt1"
      if openvino_failure == "openvino_gpu_cl_out_of_resources"
      else "not_closed"
  )

  oracle_refs = oracle_contract.get("oracle_bundle", {})
  r0_oracle_closed = bool(oracle_refs.get("reference_artifacts", {}).get("r0_oracle_gate_closed"))
  if not oracle_refs:
    r0_oracle_closed = False

  route_decisions = [
      {
          "id": "raw_opencl_gguf_source_stream_qmatvec",
          "decision": "rejected",
          "evidence": [rel(source_path), rel(qmatvec_path), rel(kv_path)],
          "reason": (
              "Current-target source stream reaches only "
              f"{source_max} GB/s ({source_ratio:.3f}x of {target_line} GB/s) "
              f"and qmatvec reaches only {qmatvec_max} GB/s. "
              "The 262144 fp16/bf16 KV-only ceiling at measured source-stream "
              f"bandwidth is {fp16_256k.get('ceiling_tok_s_at_source_stream_max'):.3f} tok/s."
          ),
          "stop_rule": (
              "Do not reopen local-size, tile-shape, command-submit, or same-body "
              "OpenCL qmatvec variants unless a new real-tensor source-stream or "
              "qmatvec mechanism clears the route line with checksum and numeric evidence."
          ),
      },
      {
          "id": "openvino_262144_denominator_lane",
          "decision": "not_closed_resource_failure",
          "evidence": [rel(denominator_path)],
          "reason": (
              "The 262144 prompt materialized and parsed, but OpenVINO GPU "
              "generation failed with CL_OUT_OF_RESOURCES even at -mt 1."
          ),
          "next_gate": (
              "Decide whether R0 accepts a denominator ladder capped at 131072 "
              "plus explicit 262144 resource-failure evidence, or run another "
              "same-host denominator route for 262144."
          ),
      },
      {
          "id": "native_gguf_correctness_first_token_loop",
          "decision": "actionable_accuracy_only",
          "evidence": [
              "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl",
              "output/r0-oracle-seed-replay-20260626T034841Z/correctness.json",
              "output/r0-teacher-forced-distribution-seed-20260626T035426Z/teacher-forced-distribution-seed.jsonl",
          ],
          "reason": (
              "The staged CPU llama.cpp seed and replay harness can support a "
              "bounded native-output skeleton check, but this is not a performance "
              "route and does not replace the full oracle bundle."
          ),
          "stop_rule": (
              "Do not promote performance work from a token-loop skeleton until "
              "native output passes replay and the full oracle/denominator gates are closed."
          ),
      },
      {
          "id": "level_zero_ocloc_same_body_lowbit",
          "decision": "not_selected_without_new_mechanism",
          "evidence": [
              "prior intel-box route reassessment",
              rel(source_path),
              rel(qmatvec_path),
          ],
          "reason": (
              "Historical same-body Level Zero/OpenCL rows were in the same low "
              "bandwidth class, and current target source-stream/qmatvec evidence "
              "does not justify rerunning same-body compiler/API variants."
          ),
          "next_gate": (
              "Only reopen Level Zero/ocloc if the experiment changes byte traffic "
              "or math structure, not just API or command-list plumbing."
          ),
      },
  ]

  selected_next_artifact = {
      "id": "r0_denominator_resolution_and_oracle_boundary_bundle",
      "reason": (
          "The current dominant performance route is hard-rejected. The highest-SNR "
          "next R0 work is to close the correctness/denominator blockers: full "
          "oracle boundary bundle plus an explicit 262144 denominator interpretation "
          "or alternate same-host denominator."
      ),
      "required_outputs": [
          "full teacher-forced distribution references beyond the short/router seed",
          "per-boundary reference input/output bundle",
          "262144 denominator decision or alternate denominator artifact",
          "resident harness load(model, oracle_bundle) evidence",
      ],
  }

  hard_gate_closed = (
      source_audit.get("route_label") == "rejected"
      and bool(source_audit.get("required_checks_passed"))
      and qmatvec_audit.get("required_checks_passed") is True
      and fp16_256k.get("ceiling_tok_s_at_source_stream_max") is not None
  )

  r0_pending = target_contract.get("r0_refresh", {}).get("pending_items", [])
  r0_closed = False
  payload = {
      "created_at": created_at,
      "dominant_route_feasibility_hard_gate_closed": hard_gate_closed,
      "evidence": {
          "denominator_row": rel(denominator_path),
          "kv_pressure": rel(kv_path),
          "oracle_contract": rel(oracle_contract_path),
          "qmatvec_probe": rel(qmatvec_path),
          "source_stream": rel(source_path),
          "target_contract": rel(target_contract_path),
      },
      "r0_closed": r0_closed,
      "r0_pending_items": r0_pending,
      "route_decisions": route_decisions,
      "schema_version": "intel-qwen36-r0-route-feasibility-v0",
      "selected_next_artifact": selected_next_artifact,
      "summary": {
          "current_dominant_route_result": "raw_opencl_gguf_source_stream_qmatvec_rejected",
          "kv_262144_ceiling_tok_s_at_qmatvec_max": fp16_256k.get("ceiling_tok_s_at_qmatvec_max"),
          "kv_262144_ceiling_tok_s_at_source_stream_max": (
              fp16_256k.get("ceiling_tok_s_at_source_stream_max")
          ),
          "openvino_262144_status": openvino_status,
          "qmatvec_max_gb_s": qmatvec_max,
          "qmatvec_target_ratio": qmatvec_ratio,
          "r0_oracle_gate_closed": r0_oracle_closed,
          "source_stream_max_gb_s": source_max,
          "source_stream_target_ratio": source_ratio,
          "target_line_gb_s": target_line,
      },
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "route_label": "rejected",
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-route-feasibility.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "decision.json", payload)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for decision in route_decisions:
      fh.write(json.dumps({
          "decision": decision["decision"],
          "id": decision["id"],
          "phase": "r0_route_feasibility",
      }, sort_keys=True) + "\n")
  write_json(out_dir / "correctness.json", {
      "checks": [
          {
              "denominator_artifact_found": denominator_path is not None,
              "hard_gate_closed": hard_gate_closed,
              "kv_pressure_found": kv_path is not None,
              "qmatvec_probe_found": qmatvec_path is not None,
              "selected_next_artifact_present": bool(selected_next_artifact.get("id")),
              "source_stream_found": source_path is not None,
          }
      ],
      "gate": "r0_dominant_route_feasibility",
      "required_checks_passed": hard_gate_closed,
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "notes": "route decision only; smoothness remains a product-candidate gate",
      "route_label": "rejected",
  })
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"route feasibility output: {out_dir}")


if __name__ == "__main__":
  main()
