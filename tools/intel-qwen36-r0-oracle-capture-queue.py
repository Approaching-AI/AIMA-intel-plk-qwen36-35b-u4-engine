#!/usr/bin/env python3
"""Expand the R0 oracle capture spec into concrete bundle capture tasks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-capture-queue-v0"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def task_layer_id(layer: int | str) -> str:
  if isinstance(layer, int):
    return f"L{layer:02d}"
  return str(layer)


def prompt_rows(capture_spec: dict[str, Any]) -> list[dict[str, Any]]:
  rows = capture_spec.get("capture_ladder", {}).get("prompt_rows", [])
  if not isinstance(rows, list):
    raise SystemExit("capture spec missing capture_ladder.prompt_rows")
  return rows


def required_output_tokens(row: dict[str, Any], capture_spec: dict[str, Any]) -> list[int]:
  max_new_tokens = row.get("max_new_tokens")
  if isinstance(max_new_tokens, int):
    return [max_new_tokens]
  tokens = capture_spec.get("capture_ladder", {}).get("required_output_tokens", [])
  return [value for value in tokens if isinstance(value, int)]


def select_boundary_source(oracle_contract: dict[str, Any]) -> dict[str, Any]:
  seed_info = oracle_contract.get("available_seed_artifacts", {}).get(
      "deterministic_cpu_llama_cpp_token_topk_seed",
      {},
  )
  seed_path_value = (
      seed_info.get("latest_staged_seed", {}).get("token_topk_seed_path")
      if isinstance(seed_info.get("latest_staged_seed"), dict)
      else None
  )
  if not isinstance(seed_path_value, str):
    raise SystemExit("oracle contract missing latest staged token-topk seed path")
  seed_path = ROOT / seed_path_value
  rows = load_jsonl(seed_path)
  if not rows:
    raise SystemExit(f"{seed_path}: empty staged seed")
  selected = next((row for row in rows if row.get("case_id") == "short_math_001"), rows[0])
  prompt_token_count = selected.get("prompt_token_count")
  if not isinstance(prompt_token_count, int) or prompt_token_count <= 0:
    raise SystemExit("selected boundary source missing prompt_token_count")
  return {
      "capture_phase": "prefill_last_prompt_token",
      "case_id": selected["case_id"],
      "prompt_set": selected.get("prompt_set"),
      "prompt_token_count": prompt_token_count,
      "source": rel(seed_path),
      "source_token_position": prompt_token_count - 1,
      "source_token_position_policy": "last prompt token from staged deterministic seed",
  }


def token_topk_task(row: dict[str, Any]) -> dict[str, Any]:
  row_id = row["id"]
  return {
      "bundle_jsonl_path": "token-topk-references.jsonl",
      "capture_status": "missing",
      "case_id": row_id,
      "kind": row.get("kind"),
      "prompt_set": row.get("prompt_set"),
      "required_fields": [
          "case_id",
          "prompt_token_ids",
          "prompt_token_count",
          "prompt_utf8_sha256",
          "top_logprobs",
          "generation_targets",
      ],
      "suite": row.get("suite"),
      "target_prompt_tokens": row.get("target_prompt_tokens"),
      "task_id": f"token_topk:{row_id}",
  }


def distribution_task(row: dict[str, Any], capture_spec: dict[str, Any]) -> dict[str, Any]:
  row_id = row["id"]
  return {
      "bundle_jsonl_path": "teacher-forced-distribution-references.jsonl",
      "capture_status": "missing",
      "case_id": row_id,
      "kind": row.get("kind"),
      "prompt_set": row.get("prompt_set"),
      "required_fields": [
          "case_id",
          "distribution_positions",
          "reference_token_id",
          "reference_token_logprob",
          "top_logprobs",
      ],
      "required_output_token_counts": required_output_tokens(row, capture_spec),
      "suite": row.get("suite"),
      "target_prompt_tokens": row.get("target_prompt_tokens"),
      "task_id": f"teacher_forced_distribution:{row_id}",
  }


def boundary_tasks(
    capture_spec: dict[str, Any],
    source: dict[str, Any],
    tensor_kind: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  jsonl_path = (
      "boundary-references/inputs.jsonl"
      if tensor_kind == "input"
      else "boundary-references/outputs.jsonl"
  )
  tensor_field = (
      "reference_input_tensor_path_or_reference_input_tensor"
      if tensor_kind == "input"
      else "reference_output_tensor_path_or_reference_output_tensor"
  )
  for spec in capture_spec.get("boundary_specs", []):
    boundary = spec["boundary_type"]
    for layer in spec.get("capture_layers", []):
      layer_id = task_layer_id(layer)
      rows.append({
          "boundary_type": boundary,
          "bundle_jsonl_path": jsonl_path,
          "capture_layer": layer,
          "capture_scope": spec.get("capture_scope"),
          "capture_status": "missing",
          "dtype_policy": spec.get("dtype_policy"),
          "required_fields": [
              "boundary_type",
              "layer",
              "shape_metadata",
              "dtype_metadata",
              "source_prompt_case_id",
              "source_token_position",
              tensor_field,
          ],
          "shape_basis": spec.get("shape_basis"),
          "source_prompt_case_id": source["case_id"],
          "source_token_position": source["source_token_position"],
          "source_token_position_policy": source["source_token_position_policy"],
          "task_id": f"boundary_{tensor_kind}:{boundary}:{layer_id}",
          "tensor_kind": tensor_kind,
      })
  return rows


def build_summary(payload: dict[str, Any]) -> str:
  totals = payload["task_totals"]
  lines = [
      "# R0 Oracle Capture Queue",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- token/top-k tasks: {totals['token_topk_tasks']}",
      f"- teacher-forced distribution tasks: {totals['teacher_forced_distribution_tasks']}",
      f"- boundary input tensor tasks: {totals['boundary_input_tasks']}",
      f"- boundary output tensor tasks: {totals['boundary_output_tasks']}",
      f"- total bundle JSONL rows required: {totals['total_bundle_jsonl_rows']}",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact is a queue for capture work. It is not an oracle bundle",
      "and intentionally records every task with `capture_status=missing`.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-oracle-capture-queue-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  capture_spec_path = latest("r0-oracle-capture-spec-*", "capture-spec.json")
  if capture_spec_path is None:
    raise SystemExit("no oracle capture spec found under output/")
  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  capture_spec = load_json(capture_spec_path)
  rows = prompt_rows(capture_spec)
  boundary_source = select_boundary_source(oracle_contract)
  token_tasks = [token_topk_task(row) for row in rows]
  distribution_tasks = [distribution_task(row, capture_spec) for row in rows]
  boundary_input_tasks = boundary_tasks(capture_spec, boundary_source, "input")
  boundary_output_tasks = boundary_tasks(capture_spec, boundary_source, "output")
  total_bundle_jsonl_rows = (
      len(token_tasks)
      + len(distribution_tasks)
      + len(boundary_input_tasks)
      + len(boundary_output_tasks)
  )
  payload = {
      "boundary_capture_source": boundary_source,
      "created_at": created_at,
      "evidence": {
          "capture_spec": rel(capture_spec_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
      },
      "model": oracle_contract["model"],
      "r0_oracle_gate_closed": False,
      "schema_version": SCHEMA_VERSION,
      "task_files": {
          "boundary_input_tasks": "boundary-input-tasks.jsonl",
          "boundary_output_tasks": "boundary-output-tasks.jsonl",
          "teacher_forced_distribution_tasks": "teacher-forced-distribution-tasks.jsonl",
          "token_topk_tasks": "token-topk-tasks.jsonl",
      },
      "task_totals": {
          "boundary_input_tasks": len(boundary_input_tasks),
          "boundary_output_tasks": len(boundary_output_tasks),
          "teacher_forced_distribution_tasks": len(distribution_tasks),
          "token_topk_tasks": len(token_tasks),
          "total_bundle_jsonl_rows": total_bundle_jsonl_rows,
      },
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "capture_spec_available",
          "pass": capture_spec.get("schema_version")
          == "intel-qwen36-r0-oracle-capture-spec-v0",
          "path": rel(capture_spec_path),
      },
      {
          "name": "model_matches_oracle_contract",
          "pass": model_contract["model"]["gguf_model_path"] == oracle_contract["model"]["path"]
          and model_contract["model"]["gguf_sha256"] == oracle_contract["model"]["sha256"]
          and model_contract["model"]["batch_size"] == oracle_contract["model"]["batch_size"],
      },
      {
          "name": "token_topk_tasks_match_prompt_rows",
          "pass": len(token_tasks) == capture_spec.get("coverage", {}).get("prompt_row_count"),
          "count": len(token_tasks),
      },
      {
          "name": "distribution_tasks_match_prompt_rows",
          "pass": len(distribution_tasks)
          == capture_spec.get("coverage", {}).get("prompt_row_count"),
          "count": len(distribution_tasks),
      },
      {
          "name": "boundary_input_tasks_match_capture_spec",
          "pass": len(boundary_input_tasks)
          == (
              capture_spec.get("coverage", {}).get("per_layer_boundary_record_count", 0)
              + capture_spec.get("coverage", {}).get("global_boundary_record_count", 0)
          ),
          "count": len(boundary_input_tasks),
      },
      {
          "name": "boundary_output_tasks_match_capture_spec",
          "pass": len(boundary_output_tasks) == len(boundary_input_tasks),
          "count": len(boundary_output_tasks),
      },
      {
          "name": "boundary_source_prompt_seed_available",
          "pass": boundary_source["case_id"] == "short_math_001"
          and boundary_source["source_token_position"] >= 0,
          "case_id": boundary_source["case_id"],
          "source_token_position": boundary_source["source_token_position"],
      },
      {
          "name": "queue_is_not_a_bundle",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-capture-queue.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture-queue.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_capture_queue",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "token-topk-tasks.jsonl", token_tasks)
  write_jsonl(out_dir / "teacher-forced-distribution-tasks.jsonl", distribution_tasks)
  write_jsonl(out_dir / "boundary-input-tasks.jsonl", boundary_input_tasks)
  write_jsonl(out_dir / "boundary-output-tasks.jsonl", boundary_output_tasks)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in payload["task_totals"].items():
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_oracle_capture_queue",
          "value": value,
      }, sort_keys=True) + "\n")
    fh.write(json.dumps({
        "metric": "r0_oracle_gate_closed",
        "phase": "r0_oracle_capture_queue",
        "value": False,
    }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle capture queue output: {out_dir}")


if __name__ == "__main__":
  main()
