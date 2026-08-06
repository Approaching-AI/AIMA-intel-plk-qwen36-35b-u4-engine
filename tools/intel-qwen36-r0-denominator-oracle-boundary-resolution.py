#!/usr/bin/env python3
"""Record the R0 denominator/oracle-boundary blocker interpretation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
REQUIRED_DENOMINATOR_FIELDS = (
    "prompt_tokens",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
    "decode_tokens_s",
    "prefill_tokens_s",
)


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


def latest_llama_preflight() -> Path | None:
  return latest("r0-llama-denominator-preflight-*", "preflight.json")


def llama_rows() -> list[tuple[Path, dict[str, Any]]]:
  rows = []
  for path in sorted((ROOT / "output").glob("r0-llama-denominator-*/row.json")):
    try:
      rows.append((path, load_json(path)))
    except (OSError, json.JSONDecodeError):
      continue
  return rows


def latest_llama_row_for_bucket(bucket: int) -> tuple[Path | None, dict[str, Any]]:
  selected_path = None
  selected_payload: dict[str, Any] = {}
  for path, payload in llama_rows():
    if payload.get("row", {}).get("bucket") == bucket:
      selected_path = path
      selected_payload = payload
  return selected_path, selected_payload


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def path_from_record(value: Any) -> Path | None:
  if not isinstance(value, str) or not value:
    return None
  path = Path(value)
  if not path.is_absolute():
    path = ROOT / path
  return path


def read_optional_text(path: Path | None) -> str:
  if path is None or not path.exists():
    return ""
  return path.read_text(encoding="utf-8", errors="replace")


def finite_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def classify_denominator(row_payload: dict[str, Any]) -> dict[str, Any]:
  row = row_payload.get("row", {})
  raw = row.get("raw", {})
  stderr_path = path_from_record(raw.get("stderr"))
  stdout_path = path_from_record(raw.get("stdout"))
  stderr = read_optional_text(stderr_path)
  stdout = read_optional_text(stdout_path)

  present_fields = [
      field for field in REQUIRED_DENOMINATOR_FIELDS if finite_number(row.get(field))
  ]
  missing_fields = [
      field for field in REQUIRED_DENOMINATOR_FIELDS if field not in present_fields
  ]
  metric_available = not missing_fields and row.get("parse_status") == "parsed"
  resource_failure = "CL_OUT_OF_RESOURCES" in stderr
  prompt_tokens = row.get("prompt_tokens")
  prompt_parsed = prompt_tokens == 262144 or "Prompt token size: 262144" in stdout

  if metric_available:
    interpretation = "openvino_262144_metric_available"
    decision = "usable_denominator_metric"
  elif resource_failure and prompt_parsed:
    interpretation = "openvino_262144_resource_failure_not_metric"
    decision = "resource_failure_row_only"
  elif row_payload:
    interpretation = "openvino_262144_unresolved_failure"
    decision = "not_usable_denominator_metric"
  else:
    interpretation = "openvino_262144_not_attempted"
    decision = "missing_denominator_attempt"

  return {
      "bucket": 262144,
      "decision": decision,
      "denominator_metric_available": metric_available,
      "evidence": {
          "row_path": None,
          "stderr_path": rel(stderr_path) if stderr_path else None,
          "stdout_path": rel(stdout_path) if stdout_path else None,
      },
      "failure_class": (
          "openvino_gpu_cl_out_of_resources" if resource_failure else None
      ),
      "interpretation": interpretation,
      "missing_metric_fields": missing_fields,
      "parse_status": row.get("parse_status"),
      "prompt_tokens": prompt_tokens,
      "required_metric_fields": list(REQUIRED_DENOMINATOR_FIELDS),
      "r0_denominator_gate_closed": False,
      "resource_failure_after_prompt_parse": resource_failure and prompt_parsed,
      "route_label": row_payload.get("route_label"),
      "usable_for": (
          ["OpenVINO 262144 unavailable-lane evidence"]
          if resource_failure and prompt_parsed
          else []
      ),
      "not_usable_for": [
          "prefill throughput metric",
          "decode throughput metric",
          "speedup denominator",
          "R0 denominator closure",
      ],
  }


def classify_llama_denominator(
    row_path: Path | None,
    row_payload: dict[str, Any],
    preflight_path: Path | None,
) -> dict[str, Any]:
  row = row_payload.get("row", {})
  raw = row.get("raw", {})
  returncode = raw.get("returncode")
  paired_tokens_s = row.get("paired_tokens_s")
  metric_available = (
      row.get("bucket") == 262144
      and finite_number(paired_tokens_s)
      and returncode == 0
  )
  cleanup_path = row_path.parent / "post-timeout-cleanup.json" if row_path else None
  cleanup = load_json(cleanup_path) if cleanup_path and cleanup_path.exists() else {}
  timeout = returncode == 124

  if metric_available:
    interpretation = "llama_vulkan_262144_metric_available"
    decision = "usable_denominator_metric"
  elif timeout and cleanup.get("post_cleanup_status", {}).get("lingering_llama_bench_process") is False:
    interpretation = "llama_vulkan_262144_timeout_no_metric_cleanup_complete"
    decision = "timeout_row_only"
  elif timeout:
    interpretation = "llama_vulkan_262144_timeout_cleanup_unverified"
    decision = "timeout_row_only_cleanup_unverified"
  elif row_payload:
    interpretation = "llama_vulkan_262144_no_metric"
    decision = "not_usable_denominator_metric"
  else:
    interpretation = "llama_vulkan_262144_not_attempted"
    decision = "missing_denominator_attempt"

  smoke_paths = []
  for path, payload in llama_rows():
    bucket = payload.get("row", {}).get("bucket")
    if isinstance(bucket, int) and bucket < 262144:
      smoke_paths.append(rel(path))

  return {
      "bucket": 262144,
      "decision": decision,
      "denominator_metric_available": metric_available,
      "evidence": {
          "latest_preflight": rel(preflight_path),
          "post_timeout_cleanup": rel(cleanup_path) if cleanup else None,
          "row_path": rel(row_path),
          "smoke_rows": smoke_paths,
      },
      "interpretation": interpretation,
      "mode": row.get("llama_bench_mode"),
      "paired_tokens_s": paired_tokens_s,
      "parse_status": row.get("parse_status"),
      "remote_returncode": returncode,
      "r0_denominator_gate_closed": False,
      "route_label": row_payload.get("route_label"),
      "timeout_cleanup_complete": (
          cleanup.get("post_cleanup_status", {}).get("lingering_llama_bench_process") is False
      ),
      "usable_for": (
          ["llama.cpp/Vulkan same-host 262144 denominator metric"]
          if metric_available
          else ["llama.cpp/Vulkan unavailable-lane evidence"]
          if timeout
          else []
      ),
      "not_usable_for": [] if metric_available else [
          "prefill throughput metric",
          "decode throughput metric",
          "paired prompt+gen throughput metric",
          "speedup denominator",
          "R0 denominator closure",
      ],
  }


def classify_oracle(
    oracle_contract: dict[str, Any],
    model_contract: dict[str, Any],
) -> dict[str, Any]:
  required_fields = set(oracle_contract.get("required_bundle_fields", []))
  missing = list(oracle_contract.get("missing_for_r0_close", []))
  seed = oracle_contract.get("available_seed_artifacts", {}).get(
      "deterministic_cpu_llama_cpp_token_topk_seed", {}
  )
  distribution_seed = seed.get("latest_teacher_forced_distribution_seed", {})
  model_boundaries = model_contract.get("boundary_types", [])
  oracle_boundaries = oracle_contract.get("boundary_types", [])
  boundary_types_match = model_boundaries == oracle_boundaries

  boundary_capture_plan = []
  for boundary in oracle_boundaries:
    boundary_capture_plan.append({
        "boundary_type": boundary,
        "reference_input": "missing",
        "reference_output": "missing",
        "required_for_r0": True,
    })

  return {
      "available_subgates": seed.get("available_subgates", []),
      "boundary_capture_plan": boundary_capture_plan,
      "boundary_count": len(oracle_boundaries),
      "boundary_types_match_model_contract": boundary_types_match,
      "full_acceptance_teacher_forced_distribution_available": (
          distribution_seed.get("full_acceptance_bundle") is True
      ),
      "missing_for_r0_close": missing,
      "per_boundary_bundle_available": False,
      "r0_oracle_gate_closed": False,
      "required_bundle_fields": sorted(required_fields),
      "seed_distribution_positions": distribution_seed.get("distribution_positions"),
      "seed_required_checks_passed": distribution_seed.get("required_checks_passed"),
      "usable_for": [
          "short/router token replay seed",
          "short/router first-token top-k sanity",
          "short/router per-position top-5 distribution seed",
      ],
      "not_usable_for": [
          "full acceptance teacher-forced distribution gate",
          "per-boundary tensor promotion",
          "resident harness real oracle load",
          "R0 oracle closure",
      ],
  }


def build_summary(payload: dict[str, Any]) -> str:
  denom = payload["denominator_262144"]
  llama = payload["llama_denominator_262144"]
  oracle = payload["oracle_boundary_bundle"]
  gates = payload["r0_gate_status"]
  lines = [
      "# R0 Denominator/Oracle Boundary Resolution",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- OpenVINO 262144 interpretation: `{denom['interpretation']}`",
      f"- llama.cpp/Vulkan 262144 interpretation: `{llama['interpretation']}`",
      f"- OpenVINO metric available: `{str(denom['denominator_metric_available']).lower()}`",
      f"- llama.cpp/Vulkan metric available: `{str(llama['denominator_metric_available']).lower()}`",
      f"- denominator gate closed: `{str(gates['denominator_gate_closed']).lower()}`",
      f"- oracle gate closed: `{str(gates['oracle_gate_closed']).lower()}`",
      f"- resident harness real-load ready: `{str(gates['resident_harness_real_load_ready']).lower()}`",
      f"- oracle boundary types requiring tensors: {oracle['boundary_count']}",
      f"- oracle seed distribution positions: {oracle['seed_distribution_positions']}",
      "",
      "Decision: the OpenVINO 262144 row and the llama.cpp/Vulkan 262144",
      "timeout are denominator-attempt evidence, but neither is a throughput",
      "metric. The short/router CPU llama.cpp seed remains useful for bounded",
      "replay checks, but it is not the full teacher-forced and per-boundary",
      "oracle bundle.",
      "",
      "Next required outputs:",
      "",
  ]
  for item in payload["next_required_outputs"]:
    lines.append(f"- {item}")
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-denominator-oracle-boundary-resolution-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  denominator_path = latest("r0-openvino-denominator-*", "row.json")
  llama_preflight_path = latest_llama_preflight()
  llama_262144_path, llama_262144_payload = latest_llama_row_for_bucket(262144)
  route_path = latest("r0-route-feasibility-*", "decision.json")
  teacher_forced_path = latest("r0-teacher-forced-distribution-seed-*", "correctness.json")
  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  target_contract_path = ROOT / "contracts/intel-qwen36-target-contract.json"

  denominator_payload = load_json(denominator_path) if denominator_path else {}
  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  target_contract = load_json(target_contract_path)
  route_payload = load_json(route_path) if route_path else {}

  denominator = classify_denominator(denominator_payload)
  if denominator_path:
    denominator["evidence"]["row_path"] = rel(denominator_path)
  llama_denominator = classify_llama_denominator(
      llama_262144_path,
      llama_262144_payload,
      llama_preflight_path,
  )
  oracle = classify_oracle(oracle_contract, model_contract)

  next_required_outputs = [
      "explicit R0 policy accepting the 262144 denominator lane as unavailable, or a different bounded 262144 metric route",
      "teacher-forced top-k/logprob references across the full acceptance ladder",
      "per-boundary reference input tensors for every contract boundary type",
      "per-boundary reference output tensors for every contract boundary type",
      "resident harness load(model, oracle_bundle) evidence after the real bundle exists",
  ]
  checks = [
      {
          "name": "openvino_262144_attempt_found",
          "pass": denominator_path is not None,
          "evidence": rel(denominator_path),
      },
      {
          "name": "openvino_262144_prompt_parsed",
          "pass": denominator.get("prompt_tokens") == 262144,
          "observed": denominator.get("prompt_tokens"),
      },
      {
          "name": "openvino_resource_failure_classified",
          "pass": denominator.get("failure_class") == "openvino_gpu_cl_out_of_resources",
          "failure_class": denominator.get("failure_class"),
      },
      {
          "name": "openvino_resource_failure_not_promoted_to_metric",
          "pass": denominator.get("denominator_metric_available") is False
          and denominator.get("r0_denominator_gate_closed") is False,
      },
      {
          "name": "llama_262144_attempt_found",
          "pass": llama_262144_path is not None,
          "evidence": rel(llama_262144_path),
      },
      {
          "name": "llama_262144_timeout_or_metric_classified",
          "pass": llama_denominator.get("interpretation") in (
              "llama_vulkan_262144_metric_available",
              "llama_vulkan_262144_timeout_no_metric_cleanup_complete",
          ),
          "interpretation": llama_denominator.get("interpretation"),
      },
      {
          "name": "llama_timeout_not_promoted_to_metric",
          "pass": (
              llama_denominator.get("denominator_metric_available") is False
              and llama_denominator.get("r0_denominator_gate_closed") is False
          ) or llama_denominator.get("denominator_metric_available") is True,
      },
      {
          "name": "oracle_missing_fields_identified",
          "pass": all(
              field in oracle.get("missing_for_r0_close", [])
              for field in (
                  "teacher_forced_distribution_references",
                  "per_boundary_reference_inputs",
                  "per_boundary_reference_outputs",
              )
          ),
      },
      {
          "name": "boundary_types_match_model_contract",
          "pass": oracle.get("boundary_types_match_model_contract") is True,
          "boundary_count": oracle.get("boundary_count"),
      },
      {
          "name": "no_false_r0_gate_close",
          "pass": denominator.get("r0_denominator_gate_closed") is False
          and llama_denominator.get("r0_denominator_gate_closed") is False
          and oracle.get("r0_oracle_gate_closed") is False,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)

  payload = {
      "created_at": created_at,
      "denominator_262144": denominator,
      "evidence": {
          "denominator_row": rel(denominator_path),
          "llama_denominator_262144_row": rel(llama_262144_path),
          "llama_denominator_preflight": rel(llama_preflight_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
          "route_feasibility_decision": rel(route_path),
          "target_contract": rel(target_contract_path),
          "teacher_forced_seed_correctness": rel(teacher_forced_path),
      },
      "llama_denominator_262144": llama_denominator,
      "next_required_outputs": next_required_outputs,
      "oracle_boundary_bundle": oracle,
      "r0_gate_status": {
          "denominator_gate_closed": False,
          "dominant_route_feasibility_gate_closed": route_payload.get(
              "dominant_route_feasibility_hard_gate_closed", False
          ),
          "oracle_gate_closed": False,
          "resident_harness_real_load_ready": False,
          "r0_closed": False,
          "target_pending_items": target_contract.get("r0_refresh", {}).get(
              "pending_items", []
          ),
      },
      "schema_version": "intel-qwen36-r0-denominator-oracle-boundary-resolution-v0",
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-denominator-oracle-boundary-resolution.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "resolution.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_denominator_oracle_boundary_resolution",
      "required_checks_passed": required_checks_passed,
      "schema_version": payload["schema_version"],
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for key, value in payload["r0_gate_status"].items():
      fh.write(json.dumps({
          "metric": key,
          "phase": "r0_gate_status",
          "value": value,
      }, sort_keys=True) + "\n")
    fh.write(json.dumps({
        "metric": "openvino_262144_denominator_metric_available",
        "phase": "denominator_resolution",
        "value": denominator["denominator_metric_available"],
      }, sort_keys=True) + "\n")
    fh.write(json.dumps({
        "metric": "llama_262144_denominator_metric_available",
        "phase": "denominator_resolution",
        "value": llama_denominator["denominator_metric_available"],
    }, sort_keys=True) + "\n")
    fh.write(json.dumps({
        "metric": "oracle_boundary_types_missing_tensors",
        "phase": "oracle_boundary_bundle",
        "value": oracle["boundary_count"],
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"denominator/oracle boundary resolution output: {out_dir}")


if __name__ == "__main__":
  main()
