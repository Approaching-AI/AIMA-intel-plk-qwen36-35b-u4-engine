#!/usr/bin/env python3
"""Generate the R0 full oracle capture specification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-capture-spec-v0"

LAYER_BOUNDARIES = {
    "layer_input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention",
    "attention_output_projection",
    "post_attention_residual",
    "ffn_rmsnorm",
    "router_topk",
    "selected_expert_gate_up",
    "swiglu",
    "selected_expert_down",
    "shared_expert",
    "moe_residual",
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
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
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


def prompt_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
  base = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km"
  rows = []
  for suite_name, suite in manifest.get("suites", {}).items():
    path = base / suite["path"]
    for row in load_jsonl(path):
      rows.append({
          "bucket": row.get("bucket"),
          "id": row["id"],
          "kind": row["kind"],
          "max_new_tokens": row.get("max_new_tokens"),
          "prompt_set": row.get("prompt_set"),
          "suite": suite_name,
          "target_prompt_tokens": row.get("target_prompt_tokens"),
      })
  return rows


def boundary_shape(boundary: str, model: dict[str, Any]) -> dict[str, Any]:
  hidden = model["hidden_size"]
  heads = model["attention_heads"]
  kv_heads = model["kv_heads"]
  head_dim = model["head_dim"]
  experts = model["experts"]
  active = model["active_experts"]
  intermediate = model["moe_intermediate_size"]
  if boundary == "embedding":
    return {"input": ["token_id"], "output": [hidden]}
  if boundary == "qkv_projection":
    return {
        "input": [hidden],
        "output": {
            "k": [kv_heads, head_dim],
            "q": [heads, head_dim],
            "v": [kv_heads, head_dim],
        },
    }
  if boundary == "rope":
    return {"input": ["q", "k", "position"], "output": ["q_rope", "k_rope"]}
  if boundary == "attention":
    return {"input": ["q", "k_cache", "v_cache"], "output": [hidden]}
  if boundary == "router_topk":
    return {
        "input": [hidden],
        "output": {
            "expert_ids": [active],
            "expert_weights": [active],
            "router_logits": [experts],
        },
    }
  if boundary == "selected_expert_gate_up":
    return {"input": [hidden], "output": [active, 2, intermediate]}
  if boundary == "swiglu":
    return {"input": [active, 2, intermediate], "output": [active, intermediate]}
  if boundary == "selected_expert_down":
    return {"input": [active, intermediate], "output": [hidden]}
  if boundary == "lm_head":
    return {"input": [hidden], "output": ["vocab_logits"]}
  if boundary == "sampler":
    return {"input": ["vocab_logits"], "output": ["token_id", "top_k_logprobs"]}
  return {"input": [hidden], "output": [hidden]}


def boundary_specs(model_contract: dict[str, Any], oracle_contract: dict[str, Any]) -> list[dict[str, Any]]:
  model = model_contract["model"]
  layers = model["layers"]
  specs = []
  for boundary in oracle_contract["boundary_types"]:
    per_layer = boundary in LAYER_BOUNDARIES
    specs.append({
        "boundary_type": boundary,
        "capture_layers": list(range(layers)) if per_layer else ["global"],
        "capture_scope": "per_layer" if per_layer else "global",
        "dtype_policy": "reference_runtime_native_float_or_logits_with_metadata",
        "required_artifacts": [
            "reference_input_tensor",
            "reference_output_tensor",
            "shape_metadata",
            "dtype_metadata",
            "source_prompt_case_id",
            "source_token_position",
        ],
        "shape_basis": boundary_shape(boundary, model),
        "status": "missing",
    })
  return specs


def build_summary(payload: dict[str, Any]) -> str:
  status = payload["r0_oracle_gate_status"]
  lines = [
      "# R0 Oracle Capture Specification",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- boundary types: {payload['coverage']['boundary_type_count']}",
      f"- per-layer boundary records required: {payload['coverage']['per_layer_boundary_record_count']}",
      f"- prompt rows in capture ladder: {payload['coverage']['prompt_row_count']}",
      f"- input buckets: {payload['coverage']['input_buckets']}",
      f"- token/top-k seed available: `{str(status['token_topk_seed_available']).lower()}`",
      f"- short/router distribution seed available: `{str(status['short_router_distribution_seed_available']).lower()}`",
      f"- full acceptance distribution available: `{str(status['full_acceptance_distribution_available']).lower()}`",
      f"- per-boundary tensors available: `{str(status['per_boundary_tensors_available']).lower()}`",
      f"- R0 oracle gate closed: `{str(status['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is the capture contract for the real oracle bundle. It does not",
      "capture tensors by itself and must not be loaded as a real oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-oracle-capture-spec-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  matrix_path = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
  prompt_manifest_path = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompt-suites.json"
  teacher_seed_path = ROOT / "output/r0-teacher-forced-distribution-seed-20260626T035426Z/correctness.json"

  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  matrix = load_json(matrix_path)
  prompt_manifest = load_json(prompt_manifest_path)
  teacher_seed = load_json(teacher_seed_path)
  rows = prompt_rows(prompt_manifest)
  specs = boundary_specs(model_contract, oracle_contract)
  per_layer_records = sum(
      len(spec["capture_layers"])
      for spec in specs
      if spec["capture_scope"] == "per_layer"
  )
  global_records = sum(1 for spec in specs if spec["capture_scope"] == "global")
  missing = [
      "full_acceptance_teacher_forced_distribution_references",
      "per_boundary_reference_inputs",
      "per_boundary_reference_outputs",
  ]
  payload = {
      "boundary_specs": specs,
      "capture_ladder": {
          "prompt_rows": rows,
          "required_input_buckets": matrix["matrix"]["input_buckets"],
          "required_output_tokens": matrix["matrix"]["output_tokens"],
          "tokenizer_count_must_match_target_prompt_tokens": True,
      },
      "coverage": {
          "boundary_type_count": len(specs),
          "global_boundary_record_count": global_records,
          "input_buckets": matrix["matrix"]["input_buckets"],
          "per_layer_boundary_record_count": per_layer_records,
          "prompt_row_count": len(rows),
      },
      "created_at": created_at,
      "evidence": {
          "acceptance_matrix": rel(matrix_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
          "prompt_manifest": rel(prompt_manifest_path),
          "teacher_forced_distribution_seed_correctness": rel(teacher_seed_path),
      },
      "missing_for_r0_close": missing,
      "model": oracle_contract["model"],
      "r0_oracle_gate_status": {
          "full_acceptance_distribution_available": False,
          "per_boundary_tensors_available": False,
          "r0_oracle_gate_closed": False,
          "short_router_distribution_seed_available": (
              teacher_seed.get("required_checks_passed") is True
          ),
          "token_topk_seed_available": True,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "boundary_types_match_contract",
          "pass": [spec["boundary_type"] for spec in specs] == oracle_contract["boundary_types"],
          "count": len(specs),
      },
      {
          "name": "per_layer_boundary_records_present",
          "pass": per_layer_records == len(LAYER_BOUNDARIES) * model_contract["model"]["layers"],
          "count": per_layer_records,
      },
      {
          "name": "prompt_rows_cover_acceptance_buckets",
          "pass": sorted({row.get("bucket") for row in rows if isinstance(row.get("bucket"), int)})
          == matrix["matrix"]["input_buckets"],
      },
      {
          "name": "short_router_distribution_seed_is_limited",
          "pass": teacher_seed.get("required_checks_passed") is True
          and teacher_seed.get("total_distribution_positions") == 91,
      },
      {
          "name": "r0_oracle_gate_remains_open",
          "pass": payload["r0_oracle_gate_status"]["r0_oracle_gate_closed"] is False,
      },
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-capture-spec.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture-spec.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_capture_spec",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "boundary-specs.jsonl").open("w", encoding="utf-8") as fh:
    for spec in specs:
      fh.write(json.dumps(spec, sort_keys=True, ensure_ascii=False) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle capture spec output: {out_dir}")


if __name__ == "__main__":
  main()
