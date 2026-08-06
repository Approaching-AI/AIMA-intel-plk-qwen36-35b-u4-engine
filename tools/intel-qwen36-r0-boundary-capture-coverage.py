#!/usr/bin/env python3
"""Assess raw boundary capture tensor coverage against the oracle queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-boundary-capture-coverage-v0"
N_LAYERS = 40
FULL_ATTENTION_LAYERS = tuple(range(3, N_LAYERS, 4))
LINEAR_ATTENTION_LAYERS = tuple(
    layer for layer in range(N_LAYERS) if layer not in FULL_ATTENTION_LAYERS
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-capture-coverage-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


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


def mapping_by_boundary(instrumentation_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
  result: dict[str, dict[str, Any]] = {}
  for row in instrumentation_map.get("boundary_mappings", []):
    boundary = row.get("boundary_type")
    if isinstance(boundary, str):
      result[boundary] = row
  return result


def tensor_names_by_layer(rows: list[dict[str, Any]]) -> tuple[set[str], dict[int, set[str]]]:
  names = {
      row.get("tensor_name")
      for row in rows
      if isinstance(row.get("tensor_name"), str)
  }
  by_layer: dict[int, set[str]] = {}
  for name in names:
    if "-" not in name:
      continue
    base, suffix = name.rsplit("-", 1)
    if not suffix.isdigit():
      continue
    by_layer.setdefault(int(suffix), set()).add(base)
  return names, by_layer


def cue_available(
    cue: str,
    layer: int | str,
    names: set[str],
    by_layer: dict[int, set[str]],
    sampler_topk_present: bool,
) -> bool:
  if cue in {"token_id", "inp_tokens"}:
    return True
  if cue in {"sampled token id", "top_k_candidates", "top_k_rows", "llama_get_logits_ith"}:
    return sampler_topk_present
  if layer == "global":
    return cue in names
  if not isinstance(layer, int):
    return False
  if cue in by_layer.get(layer, set()):
    return True
  return f"{cue}-{layer}" in names


def derived_input_cues(
    boundary_type: str,
    layer: int | str,
    names: set[str],
    sampler_topk_present: bool,
) -> list[str]:
  if boundary_type == "embedding":
    return ["prompt_token_id"]
  if boundary_type == "layer_input_rmsnorm" and isinstance(layer, int):
    if layer == 0 and "model.input_embed" in names:
      return ["model.input_embed"]
    previous = f"l_out-{layer - 1}" if f"l_out-{layer - 1}" in names else f"ffn_out-{layer - 1}"
    if previous in names:
      return [previous]
  if boundary_type == "final_norm":
    if "l_out-39" in names:
      return ["l_out-39"]
    if "ffn_out-39" in names:
      return ["ffn_out-39"]
  if boundary_type == "sampler" and sampler_topk_present and "result_output" in names:
    return ["result_output", "sampler-topk.json"]
  return []


def hybrid_policy_cues(
    boundary_type: str,
    layer: int | str,
    tensor_kind: str,
    names: set[str],
) -> tuple[str, list[str]]:
  if not isinstance(layer, int) or layer not in LINEAR_ATTENTION_LAYERS:
    if boundary_type == "moe_residual" and tensor_kind == "output":
      operands = [f"attn_residual-{layer}", f"ffn_out-{layer}"]
      if all(operand in names for operand in operands):
        return "derived_from_residual_add", operands
    return "none", []
  if boundary_type == "rope":
    return "not_applicable_linear_attention_layer", []
  linear_cues: dict[tuple[str, str], list[str]] = {
      ("qkv_projection", "output"): [
          f"linear_attn_qkv_mixed-{layer}",
          f"q_conv-{layer}",
          f"k_conv-{layer}",
          f"v_conv_predelta-{layer}",
      ],
      ("attention", "input"): [
          f"q_conv_predelta-{layer}",
          f"k_conv_predelta-{layer}",
          f"v_conv_predelta-{layer}",
          f"state_predelta-{layer}",
          f"gate-{layer}",
          f"beta_sigmoid-{layer}",
      ],
      ("attention", "output"): [
          f"final_output-{layer}",
      ],
      ("attention_output_projection", "input"): [
          f"final_output-{layer}",
      ],
      ("attention_output_projection", "output"): [
          f"linear_attn_out-{layer}",
      ],
  }
  cues = linear_cues.get((boundary_type, tensor_kind), [])
  if cues and all(cue in names for cue in cues):
    return "linear_attention_equivalent", cues
  if boundary_type == "moe_residual" and tensor_kind == "output":
    operands = [f"attn_residual-{layer}", f"ffn_out-{layer}"]
    if all(operand in names for operand in operands):
      return "derived_from_residual_add", operands
  return "none", []


def task_coverage(
    task: dict[str, Any],
    mapping: dict[str, Any],
    tensor_names: set[str],
    by_layer: dict[int, set[str]],
    sampler_topk_present: bool,
) -> dict[str, Any]:
  boundary_type = task.get("boundary_type")
  layer = task.get("capture_layer")
  tensor_kind = task.get("tensor_kind")
  cues = (
      mapping.get("input_tensor_cues", [])
      if tensor_kind == "input"
      else mapping.get("output_tensor_cues", [])
  )
  cue_matches = [
      cue for cue in cues
      if isinstance(cue, str)
      and cue_available(cue, layer, tensor_names, by_layer, sampler_topk_present)
  ]
  derived = (
      derived_input_cues(boundary_type, layer, tensor_names, sampler_topk_present)
      if tensor_kind == "input"
      else []
  )
  policy_status, policy_cues = hybrid_policy_cues(
      boundary_type,
      layer,
      tensor_kind,
      tensor_names,
  )
  any_match = bool(cue_matches or derived)
  all_cues_match = bool(cues) and len(cue_matches) == len(cues)
  effective_match = any_match or policy_status != "none"
  return {
      "all_cues_matched": all_cues_match,
      "available_cues": cue_matches,
      "boundary_type": boundary_type,
      "capture_layer": layer,
      "direct_or_derived_match": any_match,
      "derived_cues": derived,
      "effective_policy_match": effective_match,
      "missing_cues": [
          cue for cue in cues
          if isinstance(cue, str) and cue not in cue_matches
      ],
      "policy_cues": policy_cues,
      "policy_status": policy_status,
      "task_id": task.get("task_id"),
      "tensor_kind": tensor_kind,
  }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  task_count = len(rows)
  any_count = sum(1 for row in rows if row["direct_or_derived_match"])
  all_count = sum(1 for row in rows if row["all_cues_matched"])
  effective_count = sum(1 for row in rows if row["effective_policy_match"])
  policy_not_applicable_count = sum(
      1 for row in rows
      if row["policy_status"] == "not_applicable_linear_attention_layer"
  )
  policy_equivalent_count = sum(
      1 for row in rows
      if row["policy_status"] == "linear_attention_equivalent"
  )
  policy_derived_count = sum(
      1 for row in rows
      if row["policy_status"] == "derived_from_residual_add"
  )
  missing_any_by_boundary = Counter(
      row["boundary_type"]
      for row in rows
      if not row["direct_or_derived_match"]
  )
  missing_effective_by_boundary = Counter(
      row["boundary_type"]
      for row in rows
      if not row["effective_policy_match"]
  )
  missing_all_by_boundary = Counter(
      row["boundary_type"]
      for row in rows
      if not row["all_cues_matched"]
  )
  return {
      "all_cues_matched_count": all_count,
      "all_cues_matched_complete": all_count == task_count,
      "direct_or_derived_match_count": any_count,
      "direct_or_derived_match_complete": any_count == task_count,
      "effective_policy_match_complete": effective_count == task_count,
      "effective_policy_match_count": effective_count,
      "missing_all_cues_by_boundary": dict(sorted(missing_all_by_boundary.items())),
      "missing_direct_or_derived_by_boundary": dict(sorted(missing_any_by_boundary.items())),
      "missing_effective_policy_by_boundary": dict(sorted(missing_effective_by_boundary.items())),
      "policy_derived_count": policy_derived_count,
      "policy_equivalent_count": policy_equivalent_count,
      "policy_not_applicable_count": policy_not_applicable_count,
      "task_count": task_count,
  }


def build_summary(payload: dict[str, Any]) -> str:
  coverage = payload["coverage"]
  lines = [
      "# R0 Boundary Capture Coverage",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- capture run: `{payload['evidence']['boundary_capture_run']}`",
      f"- tensor rows: `{coverage['raw_capture']['tensor_jsonl_row_count']}`",
      f"- input tasks with any direct/derived tensor: `{coverage['inputs']['direct_or_derived_match_count']}/{coverage['inputs']['task_count']}`",
      f"- output tasks with any direct tensor: `{coverage['outputs']['direct_or_derived_match_count']}/{coverage['outputs']['task_count']}`",
      f"- input tasks effective under hybrid policy: `{coverage['inputs']['effective_policy_match_count']}/{coverage['inputs']['task_count']}`",
      f"- output tasks effective under hybrid policy: `{coverage['outputs']['effective_policy_match_count']}/{coverage['outputs']['task_count']}`",
      f"- inputs all-cues matched: `{coverage['inputs']['all_cues_matched_count']}/{coverage['inputs']['task_count']}`",
      f"- outputs all-cues matched: `{coverage['outputs']['all_cues_matched_count']}/{coverage['outputs']['task_count']}`",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is a coverage preflight only. Missing cue rows require either a",
      "bundle assembler that explicitly derives them from already captured",
      "tensors, or another instrumentation pass that names the missing tensors.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-capture-coverage-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  capture_run_path = latest("r0-boundary-capture-run-*", "capture-run.json")
  if capture_run_path is None:
    raise SystemExit("no latest boundary capture run found")
  capture_run = load_json(capture_run_path)
  tensor_jsonl_path = capture_run_path.parent / "remote-output" / "tensor-dumps.jsonl"
  sampler_topk_path = capture_run_path.parent / "remote-output" / "sampler-topk.json"
  tensor_rows = load_jsonl(tensor_jsonl_path)
  tensor_names, by_layer = tensor_names_by_layer(tensor_rows)

  queue_path = latest("r0-oracle-capture-queue-*", "capture-queue.json")
  if queue_path is None:
    raise SystemExit("no latest oracle capture queue found")
  input_tasks_path = queue_path.parent / "boundary-input-tasks.jsonl"
  output_tasks_path = queue_path.parent / "boundary-output-tasks.jsonl"
  input_tasks = load_jsonl(input_tasks_path)
  output_tasks = load_jsonl(output_tasks_path)

  instrumentation_map_path = latest("r0-llama-instrumentation-map-*", "instrumentation-map.json")
  if instrumentation_map_path is None:
    raise SystemExit("no latest instrumentation map found")
  instrumentation_map = load_json(instrumentation_map_path)
  mappings = mapping_by_boundary(instrumentation_map)

  sampler_topk_present = sampler_topk_path.exists()
  input_coverage = [
      task_coverage(task, mappings.get(task.get("boundary_type"), {}), tensor_names, by_layer, sampler_topk_present)
      for task in input_tasks
  ]
  output_coverage = [
      task_coverage(task, mappings.get(task.get("boundary_type"), {}), tensor_names, by_layer, sampler_topk_present)
      for task in output_tasks
  ]
  input_summary = summarize(input_coverage)
  output_summary = summarize(output_coverage)
  route_status = (
      "raw_boundary_capture_effectively_covers_queue_with_hybrid_policy"
      if input_summary["effective_policy_match_complete"]
      and output_summary["effective_policy_match_complete"]
      else "raw_boundary_capture_has_mapping_gaps"
  )
  payload = {
      "coverage": {
          "inputs": input_summary,
          "outputs": output_summary,
          "raw_capture": {
              "full_attention_layers": list(FULL_ATTENTION_LAYERS),
              "linear_attention_layers": list(LINEAR_ATTENTION_LAYERS),
              "sampler_topk_present": sampler_topk_present,
              "tensor_jsonl_row_count": len(tensor_rows),
              "unique_tensor_name_count": len(tensor_names),
          },
      },
      "created_at": created_at,
      "evidence": {
          "boundary_capture_run": rel(capture_run_path.parent),
          "capture_queue": rel(queue_path.parent),
          "instrumentation_map": rel(instrumentation_map_path.parent),
      },
      "next_required_actions": [
          "record the hybrid attention applicability policy in the oracle contract",
          "build a bundle assembler that emits policy-not-applicable rows for linear-layer RoPE tasks",
          "derive moe_residual outputs from attn_residual + ffn_out or add a forced post_moe materialization hook",
      ],
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_capture_run_succeeded",
          "pass": capture_run.get("route_status") == "boundary_capture_run_succeeded"
          and capture_run.get("capture_analysis", {}).get("tensor_jsonl_row_count") == 1493,
      },
      {
          "name": "queue_tasks_loaded",
          "pass": len(input_tasks) == 524 and len(output_tasks) == 524,
      },
      {
          "name": "coverage_applies_hybrid_policy_without_closing_gate",
          "pass": route_status
          == "raw_boundary_capture_effectively_covers_queue_with_hybrid_policy",
      },
      {
          "name": "coverage_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_boundary_capture_coverage",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-boundary-capture-coverage.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "coverage.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  write_jsonl(out_dir / "input-coverage.jsonl", input_coverage)
  write_jsonl(out_dir / "output-coverage.jsonl", output_coverage)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("input_task_count", input_summary["task_count"]),
        ("input_direct_or_derived_match_count", input_summary["direct_or_derived_match_count"]),
        ("input_all_cues_matched_count", input_summary["all_cues_matched_count"]),
        ("output_task_count", output_summary["task_count"]),
        ("output_direct_or_derived_match_count", output_summary["direct_or_derived_match_count"]),
        ("output_all_cues_matched_count", output_summary["all_cues_matched_count"]),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_capture_coverage",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary capture coverage output: {out_dir}")
  print(f"route_status={route_status}")
  print(
      "input_direct_or_derived_match_count="
      f"{input_summary['direct_or_derived_match_count']}/{input_summary['task_count']}"
  )
  print(
      "output_direct_or_derived_match_count="
      f"{output_summary['direct_or_derived_match_count']}/{output_summary['task_count']}"
  )
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
