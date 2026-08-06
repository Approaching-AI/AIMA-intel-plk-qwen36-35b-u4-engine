#!/usr/bin/env python3
"""Assemble boundary reference JSONLs from the hybrid-aware capture evidence."""

from __future__ import annotations

import argparse
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-boundary-bundle-fragment-v0"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
SOURCE_CASE_ID = "short_math_001"
SOURCE_TOKEN_POSITION = 15


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-bundle-fragment-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None, base: Path = ROOT) -> str | None:
  if path is None:
    return None
  return os.path.relpath(path.resolve(), base.resolve())


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
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
      value = json.loads(line)
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


def load_prompt_token_id() -> int:
  seed_path = ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
  for row in load_jsonl(seed_path):
    if row.get("case_id") != SOURCE_CASE_ID:
      continue
    token_ids = row.get("prompt_token_ids", [])
    if not isinstance(token_ids, list) or len(token_ids) <= SOURCE_TOKEN_POSITION:
      raise SystemExit("seed token ids missing source token position")
    token_id = token_ids[SOURCE_TOKEN_POSITION]
    if not isinstance(token_id, int):
      raise SystemExit("seed token id must be integer")
    return token_id
  raise SystemExit(f"{seed_path}: missing {SOURCE_CASE_ID}")


def tensor_index(run_dir: Path) -> dict[str, dict[str, Any]]:
  rows = load_jsonl(run_dir / "remote-output" / "tensor-dumps.jsonl")
  index: dict[str, dict[str, Any]] = {}
  for row in rows:
    name = row.get("tensor_name")
    if isinstance(name, str) and name not in index:
      index[name] = row
  return index


def resolve_tensor_name(
    cue: str,
    layer: int | str,
    index: dict[str, dict[str, Any]],
) -> str | None:
  if cue in index:
    return cue
  if isinstance(layer, int):
    candidate = f"{cue}-{layer}"
    if candidate in index:
      return candidate
  return None


def relative_payload_path(
    out_dir: Path,
    run_dir: Path,
    tensor_row: dict[str, Any],
) -> str:
  payload_path = tensor_row.get("payload_path")
  if not isinstance(payload_path, str) or not payload_path:
    raise SystemExit("tensor row missing payload_path")
  source = run_dir / "remote-output" / payload_path
  if not source.exists():
    raise SystemExit(f"missing tensor payload: {source}")
  return rel(source, out_dir) or ""


def metadata_from_tensor(tensor_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  shape = {
      "ne": tensor_row.get("ne"),
      "nb": tensor_row.get("nb"),
      "nbytes": tensor_row.get("nbytes"),
      "tensor_op": tensor_row.get("tensor_op"),
      "tensor_name": tensor_row.get("tensor_name"),
  }
  dtype = {
      "tensor_type": tensor_row.get("tensor_type"),
      "encoding": "raw_ggml_tensor_payload",
  }
  return shape, dtype


def add_f32_payload(left: Path, right: Path, out: Path) -> int:
  left_bytes = left.read_bytes()
  right_bytes = right.read_bytes()
  if len(left_bytes) != len(right_bytes) or len(left_bytes) % 4 != 0:
    raise SystemExit("cannot derive f32 payload from mismatched operands")
  count = len(left_bytes) // 4
  left_values = struct.unpack("<" + "f" * count, left_bytes)
  right_values = struct.unpack("<" + "f" * count, right_bytes)
  summed = [a + b for a, b in zip(left_values, right_values)]
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_bytes(struct.pack("<" + "f" * count, *summed))
  return len(left_bytes)


def make_reference_row(
    task: dict[str, Any],
    coverage: dict[str, Any],
    run_dir: Path,
    out_dir: Path,
    index: dict[str, dict[str, Any]],
    token_id: int,
    tensor_kind: str,
) -> dict[str, Any]:
  boundary_type = task.get("boundary_type")
  layer = task.get("capture_layer")
  base: dict[str, Any] = {
      "boundary_type": boundary_type,
      "capture_status": "captured",
      "dtype_metadata": {},
      "layer": layer,
      "shape_metadata": {},
      "source_prompt_case_id": SOURCE_CASE_ID,
      "source_token_position": SOURCE_TOKEN_POSITION,
      "task_id": task.get("task_id"),
      "tensor_kind": tensor_kind,
  }

  if coverage.get("policy_status") == "not_applicable_linear_attention_layer":
    base.update({
        "capture_status": "policy_not_applicable",
        "dtype_metadata": {"policy": "no tensor payload"},
        "policy_id": "qwen35moe_linear_attention_no_rope",
        "policy_reason": "linear-attention layers do not execute RoPE in qwen35moe.cpp",
        "shape_metadata": {"policy": "not_applicable"},
    })
    return base

  if boundary_type == "embedding" and tensor_kind == "input":
    base.update({
        "capture_status": "captured_inline",
        "dtype_metadata": {"tensor_type": "token_id"},
        "reference_input_tensor": {"token_id": token_id},
        "shape_metadata": {"shape": ["token_id"]},
    })
    return base

  if boundary_type == "sampler" and tensor_kind == "output":
    sampler_path = run_dir / "remote-output" / "sampler-topk.json"
    base.update({
        "capture_status": "captured_sampler_topk",
        "dtype_metadata": {"tensor_type": "sampler_topk_json"},
        "reference_output_tensor_path": rel(sampler_path, out_dir),
        "shape_metadata": {"shape": ["token_id", "top_k_logprobs"]},
    })
    return base

  if coverage.get("policy_status") == "derived_from_residual_add":
    if not isinstance(layer, int):
      raise SystemExit("derived residual row requires integer layer")
    left_name = f"attn_residual-{layer}"
    right_name = f"ffn_out-{layer}"
    left = index[left_name]
    right = index[right_name]
    left_path = run_dir / "remote-output" / left["payload_path"]
    right_path = run_dir / "remote-output" / right["payload_path"]
    out_payload = out_dir / "payloads" / "derived" / f"moe_residual-{layer}.bin"
    nbytes = add_f32_payload(left_path, right_path, out_payload)
    shape, dtype = metadata_from_tensor(left)
    shape.update({
        "derived_from": [left_name, right_name],
        "nbytes": nbytes,
        "tensor_name": f"moe_residual-{layer}",
        "tensor_op": "ADD",
    })
    dtype.update({"derivation": "f32_elementwise_add"})
    base.update({
        "capture_status": "derived_from_captured_tensors",
        "dtype_metadata": dtype,
        "reference_output_tensor_path": rel(out_payload, out_dir),
        "reference_output_tensor_paths": {
            left_name: relative_payload_path(out_dir, run_dir, left),
            right_name: relative_payload_path(out_dir, run_dir, right),
        },
        "shape_metadata": shape,
    })
    return base

  candidate_cues = []
  for field in ("available_cues", "derived_cues", "policy_cues"):
    values = coverage.get(field, [])
    if isinstance(values, list):
      candidate_cues.extend(value for value in values if isinstance(value, str))
  resolved: list[tuple[str, dict[str, Any]]] = []
  for cue in candidate_cues:
    name = resolve_tensor_name(cue, layer, index)
    if name is not None:
      resolved.append((name, index[name]))
  if not resolved:
    raise SystemExit(f"{task.get('task_id')}: no tensor resolved from coverage")

  primary_name, primary_tensor = resolved[0]
  shape, dtype = metadata_from_tensor(primary_tensor)
  all_paths = {
      name: relative_payload_path(out_dir, run_dir, tensor)
      for name, tensor in resolved
  }
  path_field = (
      "reference_input_tensor_path"
      if tensor_kind == "input"
      else "reference_output_tensor_path"
  )
  paths_field = (
      "reference_input_tensor_paths"
      if tensor_kind == "input"
      else "reference_output_tensor_paths"
  )
  base.update({
      "dtype_metadata": dtype,
      path_field: all_paths[primary_name],
      paths_field: all_paths,
      "shape_metadata": shape,
  })
  if coverage.get("policy_status") == "linear_attention_equivalent":
    base["capture_status"] = "captured_linear_attention_equivalent"
    base["policy_id"] = "qwen35moe_hybrid_attention_equivalent_boundary"
  return base


def build_rows(
    tasks: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    run_dir: Path,
    out_dir: Path,
    index: dict[str, dict[str, Any]],
    token_id: int,
    tensor_kind: str,
) -> list[dict[str, Any]]:
  coverage_by_task = {
      row.get("task_id"): row for row in coverage_rows if isinstance(row.get("task_id"), str)
  }
  output = []
  for task in tasks:
    task_id = task.get("task_id")
    coverage = coverage_by_task.get(task_id)
    if coverage is None:
      raise SystemExit(f"missing coverage row for {task_id}")
    if coverage.get("effective_policy_match") is not True:
      raise SystemExit(f"coverage row is not policy-effective: {task_id}")
    output.append(make_reference_row(
        task,
        coverage,
        run_dir,
        out_dir,
        index,
        token_id,
        tensor_kind,
    ))
  return output


def build_summary(payload: dict[str, Any]) -> str:
  counts = payload["fragment_counts"]
  lines = [
      "# R0 Boundary Bundle Fragment",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- input rows: `{counts['input_rows']}`",
      f"- output rows: `{counts['output_rows']}`",
      f"- policy-not-applicable rows: `{counts['policy_not_applicable_rows']}`",
      f"- derived output rows: `{counts['derived_output_rows']}`",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is a boundary-reference fragment only. It does not include token",
      "top-k references or teacher-forced distribution references and is not a",
      "full oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-bundle-fragment-{stamp}"
  out_dir = out_dir.resolve()
  (out_dir / "boundary-references").mkdir(parents=True, exist_ok=True)

  coverage_path = latest("r0-boundary-capture-coverage-*", "coverage.json")
  if coverage_path is None:
    raise SystemExit("no boundary coverage artifact found")
  coverage = load_json(coverage_path)
  if coverage.get("route_status") != "raw_boundary_capture_effectively_covers_queue_with_hybrid_policy":
    raise SystemExit("latest coverage is not policy-effective")
  run_path = ROOT / coverage["evidence"]["boundary_capture_run"] / "capture-run.json"
  run = load_json(run_path)
  run_dir = run_path.parent
  queue_dir = ROOT / coverage["evidence"]["capture_queue"]
  index = tensor_index(run_dir)
  token_id = load_prompt_token_id()

  input_tasks = load_jsonl(queue_dir / "boundary-input-tasks.jsonl")
  output_tasks = load_jsonl(queue_dir / "boundary-output-tasks.jsonl")
  input_coverage = load_jsonl(coverage_path.parent / "input-coverage.jsonl")
  output_coverage = load_jsonl(coverage_path.parent / "output-coverage.jsonl")
  input_rows = build_rows(input_tasks, input_coverage, run_dir, out_dir, index, token_id, "input")
  output_rows = build_rows(output_tasks, output_coverage, run_dir, out_dir, index, token_id, "output")

  write_jsonl(out_dir / "boundary-references" / "inputs.jsonl", input_rows)
  write_jsonl(out_dir / "boundary-references" / "outputs.jsonl", output_rows)

  policy_not_applicable_rows = sum(
      1 for row in input_rows + output_rows
      if row.get("capture_status") == "policy_not_applicable"
  )
  derived_output_rows = sum(
      1 for row in output_rows
      if row.get("capture_status") == "derived_from_captured_tensors"
  )
  fragment_counts = {
      "captured_inline_rows": sum(
          1 for row in input_rows + output_rows
          if row.get("capture_status") == "captured_inline"
      ),
      "derived_output_rows": derived_output_rows,
      "input_rows": len(input_rows),
      "output_rows": len(output_rows),
      "policy_not_applicable_rows": policy_not_applicable_rows,
      "sampler_rows": sum(
          1 for row in input_rows + output_rows
          if row.get("capture_status") == "captured_sampler_topk"
      ),
  }
  route_status = (
      "boundary_reference_fragment_assembled"
      if len(input_rows) == 524
      and len(output_rows) == 524
      and policy_not_applicable_rows == 60
      and derived_output_rows == 40
      else "boundary_reference_fragment_incomplete"
  )
  payload = {
      "created_at": created_at,
      "evidence": {
          "boundary_capture_coverage": rel(coverage_path.parent),
          "boundary_capture_run": rel(run_dir),
          "capture_queue": rel(queue_dir),
      },
      "fragment_counts": fragment_counts,
      "model": {
          "batch_size": 1,
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
      },
      "output_paths": {
          "boundary_inputs": "boundary-references/inputs.jsonl",
          "boundary_outputs": "boundary-references/outputs.jsonl",
      },
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_coverage_policy_effective",
          "pass": coverage.get("route_status")
          == "raw_boundary_capture_effectively_covers_queue_with_hybrid_policy"
          and coverage.get("r0_oracle_gate_closed") is False,
      },
      {
          "name": "input_output_row_counts",
          "pass": len(input_rows) == 524 and len(output_rows) == 524,
      },
      {
          "name": "policy_and_derivation_counts",
          "pass": policy_not_applicable_rows == 60 and derived_output_rows == 40,
      },
      {
          "name": "fragment_does_not_claim_full_bundle",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "full_oracle_bundle": False,
      "gate": "r0_boundary_bundle_fragment",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "model": payload["model"],
      "schema_version": SCHEMA_VERSION,
      "status": {
          "boundary_fragment": True,
          "full_acceptance_bundle": False,
      },
      "tool": "tools/intel-qwen36-r0-boundary-bundle-fragment-assemble.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "fragment.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("input_rows", len(input_rows)),
        ("output_rows", len(output_rows)),
        ("policy_not_applicable_rows", policy_not_applicable_rows),
        ("derived_output_rows", derived_output_rows),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_bundle_fragment",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary bundle fragment output: {out_dir}")
  print(f"route_status={route_status}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
