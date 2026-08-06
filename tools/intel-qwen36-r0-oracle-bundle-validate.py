#!/usr/bin/env python3
"""Validate whether a staged oracle bundle is sufficient to close R0."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-bundle-validation-v0"
REQUIRED_BUNDLE_PATHS = [
    "manifest.json",
    "correctness.json",
    "token-topk-references.jsonl",
    "teacher-forced-distribution-references.jsonl",
    "boundary-references/inputs.jsonl",
    "boundary-references/outputs.jsonl",
]
PROMPT_ID_FIELDS = (
    "case_id",
    "prompt_id",
    "source_prompt_case_id",
    "id",
)
TENSOR_INPUT_FIELDS = (
    "reference_input_tensor",
    "reference_input_tensor_data",
    "tensor_data",
)
TENSOR_INPUT_PATH_FIELDS = (
    "reference_input_tensor_path",
    "input_tensor_path",
    "tensor_path",
)
TENSOR_OUTPUT_FIELDS = (
    "reference_output_tensor",
    "reference_output_tensor_data",
    "tensor_data",
)
TENSOR_OUTPUT_PATH_FIELDS = (
    "reference_output_tensor_path",
    "output_tensor_path",
    "tensor_path",
)
PROMPT_EDGE_CAPTURE_STATUSES = {
    "policy_resolved_prompt_edge",
    "prompt_edge_unavailable",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise ValueError(f"{path} must be a JSON object")
  return value


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
  rows: list[dict[str, Any]] = []
  errors: list[str] = []
  if not path.is_file():
    return rows, [f"missing JSONL file: {rel(path)}"]
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        errors.append(f"{rel(path)}:{line_number}: invalid JSONL: {exc}")
        continue
      if not isinstance(row, dict):
        errors.append(f"{rel(path)}:{line_number}: row must be a JSON object")
        continue
      rows.append(row)
  return rows, errors


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def bundle_candidates(explicit: str | None) -> list[Path]:
  if explicit:
    return [(ROOT / explicit).resolve() if not Path(explicit).is_absolute() else Path(explicit)]
  oracle_dir = ROOT / "oracle"
  if not oracle_dir.exists():
    return []
  return sorted(path for path in oracle_dir.iterdir() if path.is_dir())


def field_value(row: dict[str, Any], names: tuple[str, ...]) -> Any:
  for name in names:
    if name in row:
      return row[name]
  return None


def model_value(model: dict[str, Any], *names: str) -> Any:
  for name in names:
    if name in model:
      return model[name]
  return None


def prompt_id(row: dict[str, Any]) -> str | None:
  value = field_value(row, PROMPT_ID_FIELDS)
  return value if isinstance(value, str) and value else None


def normalize_layer(row: dict[str, Any]) -> int | str | None:
  for field in ("layer", "layer_id", "layer_index"):
    if field not in row:
      continue
    value = row[field]
    if isinstance(value, int):
      return value
    if isinstance(value, str):
      if value == "global":
        return value
      if value.isdigit():
        return int(value)
  return None


def expected_boundary_keys(capture_spec: dict[str, Any]) -> set[tuple[str, int | str]]:
  keys: set[tuple[str, int | str]] = set()
  for spec in capture_spec.get("boundary_specs", []):
    boundary = spec.get("boundary_type")
    if not isinstance(boundary, str):
      continue
    for layer in spec.get("capture_layers", []):
      keys.add((boundary, layer))
  return keys


def expected_prompts(capture_spec: dict[str, Any]) -> tuple[set[str], set[int], dict[str, int]]:
  rows = capture_spec.get("capture_ladder", {}).get("prompt_rows", [])
  ids: set[str] = set()
  buckets: set[int] = set()
  bucket_by_id: dict[str, int] = {}
  for row in rows:
    if not isinstance(row, dict):
      continue
    row_id = row.get("id")
    if isinstance(row_id, str) and row_id:
      ids.add(row_id)
      bucket = row.get("bucket")
      if isinstance(bucket, int):
        buckets.add(bucket)
        bucket_by_id[row_id] = bucket
  return ids, buckets, bucket_by_id


def prompt_edge_policy(oracle_contract: dict[str, Any]) -> dict[str, Any]:
  policy_info = (
      oracle_contract.get("capture_plan", {})
      .get("latest_oracle_256k_prompt_edge_policy", {})
  )
  path_value = policy_info.get("path")
  if not isinstance(path_value, str) or not path_value:
    return {
        "available": False,
        "case_ids": set(),
        "policy_id": None,
      }
  policy_path = ROOT / path_value / "policy.json"
  if not policy_path.exists():
    return {
        "available": False,
        "case_ids": set(),
        "policy_id": policy_info.get("policy_id"),
      }
  try:
    policy_json = load_json(policy_path)
  except Exception:  # noqa: BLE001 - validation records unavailable below.
    return {
        "available": False,
        "case_ids": set(),
        "policy_id": policy_info.get("policy_id"),
      }
  policy = policy_json.get("policy", {})
  scope = policy.get("scope", {}) if isinstance(policy, dict) else {}
  case_ids = scope.get("case_ids", [])
  return {
      "available": policy.get("prompt_edge_policy_gate_closed") is True,
      "case_ids": {
          case_id for case_id in case_ids if isinstance(case_id, str) and case_id
      },
      "context_safe_max_prompt_tokens_for_first_token_prediction": policy.get(
          "context_safe_max_prompt_tokens_for_first_token_prediction"
      ),
      "exact_prompt_token_count": policy.get("exact_prompt_token_count"),
      "path": rel(policy_path.parent),
      "policy_id": policy_json.get("policy_id") or policy_info.get("policy_id"),
      "topk_logprobs_available": policy.get("topk_logprobs_available"),
  }


def is_prompt_edge_row(row: dict[str, Any], edge_policy: dict[str, Any]) -> bool:
  if edge_policy.get("available") is not True:
    return False
  row_id = prompt_id(row)
  case_ids = edge_policy.get("case_ids", set())
  if row_id not in case_ids:
    return False
  policy_id = edge_policy.get("policy_id")
  row_policy_id = row.get("policy_id") or row.get("prompt_edge_policy_id")
  if policy_id and row_policy_id != policy_id:
    return False
  if row.get("capture_status") not in PROMPT_EDGE_CAPTURE_STATUSES:
    return False
  return True


def tensor_payload_available(
    bundle_dir: Path,
    row: dict[str, Any],
    inline_fields: tuple[str, ...],
    path_fields: tuple[str, ...],
) -> bool:
  if field_value(row, inline_fields) is not None:
    return True
  for field in path_fields:
    value = row.get(field)
    if not isinstance(value, str) or not value:
      continue
    path = Path(value)
    if not path.is_absolute():
      path = bundle_dir / path
    if path.is_file():
      return True
  return False


def boundary_row_errors(
    bundle_dir: Path,
    row: dict[str, Any],
    tensor_kind: str,
) -> list[str]:
  errors = []
  boundary = row.get("boundary_type")
  if not isinstance(boundary, str) or not boundary:
    errors.append("missing boundary_type")
  if normalize_layer(row) is None:
    errors.append("missing layer/layer_id/layer_index")
  for field in ("shape_metadata", "dtype_metadata", "source_prompt_case_id", "source_token_position"):
    if field not in row:
      errors.append(f"missing {field}")
  if row.get("capture_status") == "policy_not_applicable":
    if not isinstance(row.get("policy_id"), str) or not row.get("policy_id"):
      errors.append("policy_not_applicable row missing policy_id")
    if not isinstance(row.get("policy_reason"), str) or not row.get("policy_reason"):
      errors.append("policy_not_applicable row missing policy_reason")
    return errors
  if tensor_kind == "input":
    if not tensor_payload_available(bundle_dir, row, TENSOR_INPUT_FIELDS, TENSOR_INPUT_PATH_FIELDS):
      errors.append("missing readable reference input tensor payload")
  else:
    if not tensor_payload_available(bundle_dir, row, TENSOR_OUTPUT_FIELDS, TENSOR_OUTPUT_PATH_FIELDS):
      errors.append("missing readable reference output tensor payload")
  return errors


def distribution_summary(
    rows: list[dict[str, Any]],
    expected_prompt_ids: set[str],
    expected_buckets: set[int],
    bucket_by_id: dict[str, int],
    edge_policy: dict[str, Any],
) -> dict[str, Any]:
  seen_ids: set[str] = set()
  seen_buckets: set[int] = set()
  seen_edge_ids: set[str] = set()
  position_count = 0
  row_errors = []
  seed_limitations = []

  for index, row in enumerate(rows, start=1):
    row_id = prompt_id(row)
    is_edge = is_prompt_edge_row(row, edge_policy)
    if row_id:
      seen_ids.add(row_id)
      if row_id in bucket_by_id:
        seen_buckets.add(bucket_by_id[row_id])
      if is_edge:
        seen_edge_ids.add(row_id)
    bucket = row.get("bucket")
    if isinstance(bucket, int):
      seen_buckets.add(bucket)
    if is_edge:
      positions = row.get("distribution_positions")
      if positions not in (None, []):
        row_errors.append(f"row {index}: prompt-edge row must not carry distribution positions")
      if row.get("distribution_available", False) is not False:
        row_errors.append(f"row {index}: prompt-edge row must mark distribution unavailable")
      continue
    positions = row.get("distribution_positions")
    if isinstance(positions, list):
      position_count += len(positions)
      for pos_index, position in enumerate(positions, start=1):
        if not isinstance(position, dict):
          row_errors.append(f"row {index} position {pos_index}: position must be object")
          continue
        if "top_logprobs" not in position:
          row_errors.append(f"row {index} position {pos_index}: missing top_logprobs")
    else:
      position_count += 1
      if "top_logprobs" not in row:
        row_errors.append(f"row {index}: missing top_logprobs")
    limitations = row.get("limitations")
    if isinstance(limitations, dict):
      for name, value in limitations.items():
        if value is False and name in {
            "full_acceptance_context_ladder",
            "full_acceptance_bundle",
        }:
          seed_limitations.append(f"row {index}: {name}=false")
        if value is True and name in {
            "short_router_seed_only",
            "not_a_full_r0_oracle_bundle",
            "not_a_per_boundary_tensor_bundle",
        }:
          seed_limitations.append(f"row {index}: {name}=true")

  missing_ids = sorted(expected_prompt_ids - seen_ids)
  missing_buckets = sorted(expected_buckets - seen_buckets)
  return {
      "distribution_position_count": position_count,
      "prompt_edge_policy": {
          "available": edge_policy.get("available") is True,
          "case_ids": sorted(edge_policy.get("case_ids", set())),
          "path": edge_policy.get("path"),
          "policy_id": edge_policy.get("policy_id"),
      },
      "seen_prompt_edge_ids": sorted(seen_edge_ids),
      "missing_prompt_ids": missing_ids,
      "missing_required_buckets": missing_buckets,
      "row_errors": row_errors[:50],
      "row_error_count": len(row_errors),
      "seed_limitations": seed_limitations[:50],
      "seed_limitation_count": len(seed_limitations),
      "seen_prompt_id_count": len(seen_ids),
      "seen_required_buckets": sorted(seen_buckets & expected_buckets),
      "valid": (
          not missing_ids
          and not missing_buckets
          and not row_errors
          and not seed_limitations
          and position_count > 91
      ),
}


def token_topk_summary(
    rows: list[dict[str, Any]],
    expected_prompt_ids: set[str],
    edge_policy: dict[str, Any],
) -> dict[str, Any]:
  seen_ids = {row_id for row in rows if (row_id := prompt_id(row))}
  seen_edge_ids: set[str] = set()
  row_errors = []
  for index, row in enumerate(rows, start=1):
    is_edge = is_prompt_edge_row(row, edge_policy)
    if is_edge and (row_id := prompt_id(row)):
      seen_edge_ids.add(row_id)
    if prompt_id(row) is None:
      row_errors.append(f"row {index}: missing prompt id")
    if "prompt_token_ids" not in row:
      row_errors.append(f"row {index}: missing prompt_token_ids")
    has_topk = "top_logprobs" in row
    for target in row.get("generation_targets", []):
      if isinstance(target, dict) and "top_logprobs" in target:
        has_topk = True
    if is_edge:
      if row.get("topk_logprobs_available", True) is not False:
        row_errors.append(f"row {index}: prompt-edge row must mark top-k unavailable")
      if has_topk:
        row_errors.append(f"row {index}: prompt-edge row must not carry top_logprobs")
    elif not has_topk:
      row_errors.append(f"row {index}: missing top_logprobs")
  missing_ids = sorted(expected_prompt_ids - seen_ids)
  return {
      "missing_prompt_ids": missing_ids,
      "prompt_edge_policy": {
          "available": edge_policy.get("available") is True,
          "case_ids": sorted(edge_policy.get("case_ids", set())),
          "path": edge_policy.get("path"),
          "policy_id": edge_policy.get("policy_id"),
      },
      "row_errors": row_errors[:50],
      "row_error_count": len(row_errors),
      "seen_prompt_edge_ids": sorted(seen_edge_ids),
      "seen_prompt_id_count": len(seen_ids),
      "valid": not missing_ids and not row_errors,
  }


def boundary_summary(
    bundle_dir: Path,
    rows: list[dict[str, Any]],
    expected_keys: set[tuple[str, int | str]],
    tensor_kind: str,
) -> dict[str, Any]:
  seen_keys: set[tuple[str, int | str]] = set()
  row_errors = []
  for index, row in enumerate(rows, start=1):
    boundary = row.get("boundary_type")
    layer = normalize_layer(row)
    if isinstance(boundary, str) and layer is not None:
      seen_keys.add((boundary, layer))
    errors = boundary_row_errors(bundle_dir, row, tensor_kind)
    if errors:
      row_errors.append(f"row {index}: {', '.join(errors)}")
  missing = sorted(
      ({"boundary_type": boundary, "layer": layer} for boundary, layer in expected_keys - seen_keys),
      key=lambda item: (str(item["boundary_type"]), str(item["layer"])),
  )
  unexpected = sorted(
      ({"boundary_type": boundary, "layer": layer} for boundary, layer in seen_keys - expected_keys),
      key=lambda item: (str(item["boundary_type"]), str(item["layer"])),
  )
  return {
      "missing_record_count": len(missing),
      "missing_records": missing[:80],
      "row_errors": row_errors[:50],
      "row_error_count": len(row_errors),
      "seen_record_count": len(seen_keys),
      "unexpected_record_count": len(unexpected),
      "unexpected_records": unexpected[:50],
      "valid": not missing and not unexpected and not row_errors,
  }


def validate_candidate(
    bundle_dir: Path,
    oracle_contract: dict[str, Any],
    model_contract: dict[str, Any],
    capture_spec: dict[str, Any],
) -> dict[str, Any]:
  status: dict[str, Any] = {
      "path": rel(bundle_dir) if bundle_dir.exists() else str(bundle_dir),
      "exists": bundle_dir.is_dir(),
  }
  missing_paths = []
  present_paths = []
  for relative in REQUIRED_BUNDLE_PATHS:
    path = bundle_dir / relative
    if path.is_file():
      present_paths.append(relative)
    else:
      missing_paths.append(relative)
  status["present_required_paths"] = present_paths
  status["missing_required_paths"] = missing_paths
  if missing_paths:
    status["valid_full_oracle_bundle"] = False
    return status

  errors = []
  manifest = {}
  correctness = {}
  try:
    manifest = load_json(bundle_dir / "manifest.json")
  except Exception as exc:  # noqa: BLE001 - audit needs the parse error text.
    errors.append(f"manifest parse failed: {exc}")
  try:
    correctness = load_json(bundle_dir / "correctness.json")
  except Exception as exc:  # noqa: BLE001
    errors.append(f"correctness parse failed: {exc}")

  expected_model = oracle_contract["model"]
  manifest_model = manifest.get("model", {}) if isinstance(manifest.get("model"), dict) else {}
  model_checks = {
      "batch_size": model_value(manifest_model, "batch_size") == expected_model["batch_size"],
      "path": model_value(manifest_model, "path", "gguf_model_path") == expected_model["path"],
      "sha256": model_value(manifest_model, "sha256", "gguf_sha256") == expected_model["sha256"],
      "workstream": manifest.get("workstream") == WORKSTREAM,
  }
  if not all(model_checks.values()):
    errors.append("manifest workstream/model does not match locked contract")

  manifest_status = manifest.get("status")
  manifest_status_full = (
      manifest_status.get("full_acceptance_bundle") is True
      if isinstance(manifest_status, dict)
      else False
  )
  manifest_gate_claim = (
      manifest.get("r0_oracle_gate_closed") is True
      or manifest.get("full_acceptance_bundle") is True
      or manifest_status_full
  )
  correctness_gate_claim = (
      correctness.get("required_checks_passed") is True
      and (
          correctness.get("r0_oracle_gate_closed") is True
          or correctness.get("full_acceptance_bundle") is True
          or correctness.get("gate") == "r0_oracle_full_bundle"
      )
  )
  if not manifest_gate_claim:
    errors.append("manifest does not claim a full acceptance oracle bundle")
  if not correctness_gate_claim:
    errors.append("correctness.json does not pass the full oracle bundle gate")

  expected_prompt_ids, expected_buckets, bucket_by_id = expected_prompts(capture_spec)
  expected_keys = expected_boundary_keys(capture_spec)
  edge_policy = prompt_edge_policy(oracle_contract)
  token_rows, token_errors = load_jsonl(bundle_dir / "token-topk-references.jsonl")
  distribution_rows, distribution_errors = load_jsonl(
      bundle_dir / "teacher-forced-distribution-references.jsonl"
  )
  input_rows, input_errors = load_jsonl(bundle_dir / "boundary-references/inputs.jsonl")
  output_rows, output_errors = load_jsonl(bundle_dir / "boundary-references/outputs.jsonl")

  token_summary = token_topk_summary(token_rows, expected_prompt_ids, edge_policy)
  distribution = distribution_summary(
      distribution_rows,
      expected_prompt_ids,
      expected_buckets,
      bucket_by_id,
      edge_policy,
  )
  inputs = boundary_summary(bundle_dir, input_rows, expected_keys, "input")
  outputs = boundary_summary(bundle_dir, output_rows, expected_keys, "output")

  parse_errors = token_errors + distribution_errors + input_errors + output_errors
  if parse_errors:
    errors.extend(parse_errors)
  for name, summary in (
      ("token_topk", token_summary),
      ("teacher_forced_distribution", distribution),
      ("boundary_inputs", inputs),
      ("boundary_outputs", outputs),
  ):
    if not summary["valid"]:
      errors.append(f"{name} coverage is incomplete")

  expected_coverage = {
      "boundary_type_count": capture_spec.get("coverage", {}).get("boundary_type_count"),
      "expected_boundary_record_count": len(expected_keys),
      "expected_prompt_row_count": len(expected_prompt_ids),
      "prompt_edge_case_ids": sorted(edge_policy.get("case_ids", set())),
      "prompt_edge_policy_path": edge_policy.get("path"),
      "prompt_edge_policy_required": edge_policy.get("available") is True,
      "required_input_buckets": sorted(expected_buckets),
  }
  status.update({
      "boundary_inputs": inputs,
      "boundary_outputs": outputs,
      "errors": errors[:100],
      "error_count": len(errors),
      "expected_coverage": expected_coverage,
      "manifest_model_checks": model_checks,
      "manifest_r0_gate_claim": manifest_gate_claim,
      "correctness_r0_gate_claim": correctness_gate_claim,
      "teacher_forced_distribution": distribution,
      "token_topk": token_summary,
      "valid_full_oracle_bundle": not errors,
  })
  return status


def build_summary(payload: dict[str, Any]) -> str:
  gate = payload["oracle_bundle_validation_gate"]
  lines = [
      "# R0 Oracle Bundle Validation",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- candidate directories checked: {gate['candidate_bundle_count']}",
      f"- valid full oracle bundles: {gate['candidate_valid_bundle_count']}",
      f"- R0 oracle gate closed by this audit: `{str(gate['r0_oracle_gate_closed']).lower()}`",
      f"- required bundle paths: {len(REQUIRED_BUNDLE_PATHS)}",
      f"- expected boundary records: {payload['expected_coverage']['expected_boundary_record_count']}",
      f"- expected prompt rows: {payload['expected_coverage']['expected_prompt_row_count']}",
      f"- prompt-edge policy required: `{str(payload['expected_coverage']['prompt_edge_policy_required']).lower()}`",
      f"- prompt-edge case ids: `{', '.join(payload['expected_coverage']['prompt_edge_case_ids'])}`",
      "",
      "A valid bundle must cover the full prompt ladder, teacher-forced distributions,",
      "all per-boundary input/output tensor records, and explicit 256k",
      "prompt-edge rows unless a later capture supersedes the policy.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--bundle-dir",
      help="Validate one bundle directory instead of scanning oracle/.",
  )
  parser.add_argument(
      "--require-valid-bundle",
      action="store_true",
      help="Exit non-zero unless at least one full oracle bundle validates.",
  )
  args = parser.parse_args()

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-oracle-bundle-validation-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  capture_spec_path = latest("r0-oracle-capture-spec-*", "capture-spec.json")
  if capture_spec_path is None:
    raise SystemExit("no oracle capture spec artifact found under output/")

  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  capture_spec = load_json(capture_spec_path)
  expected_prompt_ids, expected_buckets, _ = expected_prompts(capture_spec)
  expected_keys = expected_boundary_keys(capture_spec)
  edge_policy = prompt_edge_policy(oracle_contract)
  candidates = bundle_candidates(args.bundle_dir)
  candidate_status = [
      validate_candidate(path, oracle_contract, model_contract, capture_spec)
      for path in candidates
  ]
  valid_candidates = [
      status for status in candidate_status if status.get("valid_full_oracle_bundle") is True
  ]
  oracle_contract_gate_closed = oracle_contract.get("r0_oracle_gate_closed") is True
  gate_closed = bool(valid_candidates) and oracle_contract_gate_closed
  expected_coverage = {
      "boundary_type_count": capture_spec.get("coverage", {}).get("boundary_type_count"),
      "expected_boundary_record_count": len(expected_keys),
      "expected_prompt_row_count": len(expected_prompt_ids),
      "prompt_edge_case_ids": sorted(edge_policy.get("case_ids", set())),
      "prompt_edge_policy_path": edge_policy.get("path"),
      "prompt_edge_policy_required": edge_policy.get("available") is True,
      "required_input_buckets": sorted(expected_buckets),
  }
  payload = {
      "candidate_bundle_status": candidate_status,
      "created_at": created_at,
      "evidence": {
          "capture_spec": rel(capture_spec_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
      },
      "expected_coverage": expected_coverage,
      "oracle_bundle_validation_gate": {
          "candidate_bundle_count": len(candidate_status),
          "candidate_valid_bundle_count": len(valid_candidates),
          "oracle_contract_gate_closed": oracle_contract_gate_closed,
          "r0_oracle_gate_closed": gate_closed,
          "required_bundle_paths": REQUIRED_BUNDLE_PATHS,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "capture_spec_available",
          "pass": capture_spec.get("schema_version") == "intel-qwen36-r0-oracle-capture-spec-v0",
          "path": rel(capture_spec_path),
      },
      {
          "name": "required_bundle_paths_match_contract",
          "pass": oracle_contract.get("required_bundle_paths") == REQUIRED_BUNDLE_PATHS,
      },
      {
          "name": "expected_boundary_records_match_capture_spec",
          "pass": len(expected_keys)
          == (
              capture_spec.get("coverage", {}).get("per_layer_boundary_record_count", 0)
              + capture_spec.get("coverage", {}).get("global_boundary_record_count", 0)
          ),
          "expected_boundary_record_count": len(expected_keys),
      },
      {
          "name": "expected_prompt_rows_match_capture_spec",
          "pass": len(expected_prompt_ids)
          == capture_spec.get("coverage", {}).get("prompt_row_count"),
          "expected_prompt_row_count": len(expected_prompt_ids),
      },
      {
          "name": "prompt_edge_policy_available",
          "pass": edge_policy.get("available") is True
          and edge_policy.get("case_ids") == {"sentinel_256k", "prefill_shape_256k"}
          and edge_policy.get("exact_prompt_token_count") == 262144
          and edge_policy.get(
              "context_safe_max_prompt_tokens_for_first_token_prediction"
          )
          == 262143
          and edge_policy.get("topk_logprobs_available") is False,
          "path": edge_policy.get("path"),
          "case_ids": sorted(edge_policy.get("case_ids", set())),
      },
      {
          "name": "no_false_oracle_gate_close",
          "pass": gate_closed is False or bool(valid_candidates),
          "candidate_valid_bundle_count": len(valid_candidates),
          "r0_oracle_gate_closed": gate_closed,
      },
  ]
  if args.require_valid_bundle:
    checks.append({
        "name": "valid_oracle_bundle_required",
        "pass": bool(valid_candidates),
        "candidate_valid_bundle_count": len(valid_candidates),
    })
  correctness = {
      "checks": checks,
      "gate": "r0_oracle_full_bundle_validation",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-bundle-validate.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "audit.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("candidate_bundle_count", len(candidate_status)),
        ("candidate_valid_bundle_count", len(valid_candidates)),
        ("expected_boundary_record_count", len(expected_keys)),
        ("expected_prompt_row_count", len(expected_prompt_ids)),
        ("r0_oracle_gate_closed", gate_closed),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_oracle_bundle_validation",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle bundle validation output: {out_dir}")
  if not correctness["required_checks_passed"]:
    return 1
  if args.require_valid_bundle and not valid_candidates:
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main())
