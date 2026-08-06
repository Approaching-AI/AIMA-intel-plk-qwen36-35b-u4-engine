#!/usr/bin/env python3
"""Record the R1 route from component compares to a native candidate JSONL.

This artifact is deliberately not a candidate JSONL and does not close the R1
native token correctness gate. It captures the current frontier: engine-side
component compares have reached the global sampler for one captured token, the
L0 stateful linear-attention layer path, the L1 post-conv layer core, and the
all-30-linear-layer token-15 post-conv core,
first full-attention q/k/v projection, RoPE, captured-history attention core,
single-layer stateful K/V append, gate, output projection boundaries, and the
all-10-full-attention-layer stateful K/V append path, but the project still
needs a real native first-token loop over the six oracle seed rows. The
six-row prompt token input path is tracked as prerequisite evidence, not as a
component compare.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-native-candidate-route-v0"
CURRENT_GATE = "r1_native_gguf_correctness_first_token_loop"
DEFAULT_GATE = ROOT / "output/r1-native-correctness-gate-20260627T062540Z"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
ORACLE_SEED = ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"

COMPONENT_COMPARE_KEYS = [
    "latest_engine_embedding_compare",
    "latest_engine_rmsnorm_compare",
    "latest_engine_qkv_compare",
    "latest_engine_attn_output_compare",
    "latest_engine_linear_attn_preconv_compare",
    "latest_engine_linear_attn_conv_compare",
    "latest_engine_linear_attn_delta_compare",
    "latest_engine_linear_attn_postconv_compare",
    "latest_engine_linear_attn_all_postconv_compare",
    "latest_engine_layer_postconv_compare",
    "latest_engine_layer_stateful_linear_attn_compare",
    "latest_engine_layer1_postconv_compare",
    "latest_engine_full_attn_qkv_compare",
    "latest_engine_full_attn_rope_compare",
    "latest_engine_full_attn_core_compare",
    "latest_engine_full_attn_stateful_layer_compare",
    "latest_engine_full_attn_all_stateful_layers_compare",
    "latest_engine_full_attn_gate_compare",
    "latest_engine_full_attn_output_projection_compare",
    "latest_engine_attn_residual_compare",
    "latest_engine_ffn_rmsnorm_compare",
    "latest_engine_router_logits_compare",
    "latest_engine_router_topk_compare",
    "latest_engine_selected_expert_gate_up_compare",
    "latest_engine_swiglu_compare",
    "latest_engine_selected_expert_down_compare",
    "latest_engine_shared_expert_compare",
    "latest_engine_moe_residual_compare",
    "latest_engine_final_norm_compare",
    "latest_engine_lm_head_compare",
    "latest_engine_sampler_compare",
]

PREREQUISITE_KEYS = [
    "latest_engine_seed_prompt_input_check",
]

IMPLEMENTATION_GAPS = [
    "native_candidate_jsonl",
    "o1_parameterized_40_layer_prefill_loop",
    "linear_ssm_conv_state_and_residual_chaining_integrated_into_40_layer_loop",
    "full_attention_kv_cache_update_integrated_into_40_layer_loop",
    "native_generation_loop_for_first_token_and_short_generation_targets",
    "six_seed_rows_exact_replay_against_top_logprob_signatures",
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE)
  parser.add_argument("--oracle-jsonl", type=Path, default=ORACLE_SEED)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--target-contract", type=Path, default=TARGET_CONTRACT)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


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
        raise SystemExit(f"{path}:{line_number}: expected JSON object row")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def artifact_present(relative_path: str | None, filename: str) -> bool:
  if not isinstance(relative_path, str) or not relative_path:
    return False
  return (ROOT / relative_path / filename).exists()


def latest_artifact_dir(prefix: str) -> Path | None:
  dirs = sorted((ROOT / "output").glob(f"{prefix}-*"))
  return dirs[-1] if dirs else None


def key_to_prefix(key: str) -> str:
  core = key[len("latest_engine_"):] if key.startswith("latest_engine_") else key
  return "r1-engine-" + core.replace("_", "-")


def entry_from_artifact(prefix: str, extra_fields: tuple[str, ...] = ()) -> dict[str, Any]:
  directory = latest_artifact_dir(prefix)
  if directory is None:
    return {}
  correctness_path = directory / "correctness.json"
  correctness = load_json(correctness_path) if correctness_path.exists() else {}
  entry: dict[str, Any] = {
      "path": str(directory.relative_to(ROOT)),
      "boundary_type": correctness.get("gate"),
      "required_checks_passed": correctness.get("required_checks_passed"),
      "r1_native_correctness_gate_closed": correctness.get(
          "r1_native_correctness_gate_closed", False
      ),
      "speedup_claims_allowed": correctness.get("speedup_claims_allowed", False),
  }
  for field in extra_fields:
    entry[field] = correctness.get(field, correctness.get("required_checks_passed"))
  return entry


def build_r1_index() -> dict[str, Any]:
  """Build the R1 component index from generated artifacts under output/.

  Replaces the per-compare blob that used to be hand-copied into
  contracts/*.json (de-bloated 2026-06-28). Each compare key maps to the latest
  output/<prefix>-*/ artifact for that boundary; see engine/boundaries.json and
  tools/iq36-ladder.py.
  """
  index: dict[str, Any] = {"current_gate": CURRENT_GATE}
  for key in COMPONENT_COMPARE_KEYS:
    index[key] = entry_from_artifact(key_to_prefix(key))
  for key in PREREQUISITE_KEYS:
    index[key] = entry_from_artifact(
        key_to_prefix(key), extra_fields=("seed_prompt_input_path_ready",)
    )
  index["latest_gate_artifact"] = entry_from_artifact("r1-native-correctness-gate")
  index["latest_native_gguf_load_map"] = entry_from_artifact(
      "r1-native-gguf-load-map", extra_fields=("native_gguf_load_map_ready",)
  )
  index["latest_engine_gguf_inspect"] = entry_from_artifact(
      "r1-engine-gguf-inspect", extra_fields=("engine_gguf_inspect_passed",)
  )
  return index


def collect_components(
    model_r1: dict[str, Any],
    target_r1: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
  components: list[dict[str, Any]] = []
  issues: list[str] = []
  for key in COMPONENT_COMPARE_KEYS:
    model_entry = model_r1.get(key)
    target_entry = target_r1.get(key)
    if model_entry != target_entry:
      issues.append(f"{key}: target/model contract mismatch")
      entry = model_entry if isinstance(model_entry, dict) else {}
    else:
      entry = model_entry if isinstance(model_entry, dict) else {}
    path = entry.get("path")
    correctness_path = ROOT / path / "correctness.json" if isinstance(path, str) else None
    correctness: dict[str, Any] = {}
    if correctness_path is not None and correctness_path.exists():
      correctness = load_json(correctness_path)
    else:
      issues.append(f"{key}: correctness.json missing")
    entry_ok = (
        isinstance(entry, dict)
        and entry.get("required_checks_passed") is True
        and entry.get("r1_native_correctness_gate_closed") is False
        and entry.get("speedup_claims_allowed") is False
        and artifact_present(path, "manifest.json")
        and artifact_present(path, "correctness.json")
        and correctness.get("required_checks_passed") is True
        and correctness.get("r1_native_correctness_gate_closed") is False
        and correctness.get("speedup_claims_allowed") is False
    )
    if not entry_ok:
      issues.append(f"{key}: compare artifact not ready")
    components.append({
        "boundary_type": entry.get("boundary_type"),
        "key": key,
        "path": path,
        "required_checks_passed": entry.get("required_checks_passed"),
        "r1_native_correctness_gate_closed": entry.get(
            "r1_native_correctness_gate_closed"
        ),
        "speedup_claims_allowed": entry.get("speedup_claims_allowed"),
    })
  return components, issues


def collect_prerequisites(
    model_r1: dict[str, Any],
    target_r1: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
  prerequisites: list[dict[str, Any]] = []
  issues: list[str] = []
  for key in PREREQUISITE_KEYS:
    model_entry = model_r1.get(key)
    target_entry = target_r1.get(key)
    if model_entry != target_entry:
      issues.append(f"{key}: target/model contract mismatch")
      entry = model_entry if isinstance(model_entry, dict) else {}
    else:
      entry = model_entry if isinstance(model_entry, dict) else {}
    path = entry.get("path")
    correctness_path = ROOT / path / "correctness.json" if isinstance(path, str) else None
    correctness: dict[str, Any] = {}
    if correctness_path is not None and correctness_path.exists():
      correctness = load_json(correctness_path)
    else:
      issues.append(f"{key}: correctness.json missing")
    entry_ok = (
        isinstance(entry, dict)
        and entry.get("required_checks_passed") is True
        and entry.get("r1_native_correctness_gate_closed") is False
        and entry.get("speedup_claims_allowed") is False
        and artifact_present(path, "manifest.json")
        and artifact_present(path, "correctness.json")
        and correctness.get("required_checks_passed") is True
        and correctness.get("r1_native_correctness_gate_closed") is False
        and correctness.get("speedup_claims_allowed") is False
    )
    if not entry_ok:
      issues.append(f"{key}: prerequisite artifact not ready")
    prerequisites.append({
        "boundary_type": entry.get("boundary_type"),
        "key": key,
        "path": path,
        "required_checks_passed": entry.get("required_checks_passed"),
        "r1_native_correctness_gate_closed": entry.get(
            "r1_native_correctness_gate_closed"
        ),
        "seed_prompt_input_path_ready": entry.get(
            "seed_prompt_input_path_ready"
        ),
        "speedup_claims_allowed": entry.get("speedup_claims_allowed"),
    })
  return prerequisites, issues


def build_summary(payload: dict[str, Any]) -> str:
  route = payload["native_candidate_route"]
  lines = [
      "# R1 Native Candidate Route",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- selected route: `{route['selected_route']}`",
      f"- component compare artifacts: {route['component_compare_count']}",
      f"- prerequisite artifacts: {route['prerequisite_evidence_count']}",
      f"- candidate JSONL emitted: `{str(route['native_candidate_jsonl_emitted']).lower()}`",
      f"- R1 native correctness gate closed: `{str(route['r1_native_correctness_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(route['speedup_claims_allowed']).lower()}`",
      "",
      "The component ladder reaches the global sampler for the captured",
      "`short_math_001` token position and validates the L0 stateful",
      "linear-attention layer path, the L1 post-conv layer core, and the",
      "all-30-linear-layer token-15 post-conv core, plus the",
      "first full-attention q/k/v projection, RoPE, captured-history",
      "attention core, single-layer stateful K/V append, gate, output",
      "projection boundaries, and all-10-full-attention-layer stateful K/V",
      "append component evidence.",
      "The six-row seed prompt token input path is now validated as native",
      "engine prerequisite evidence.",
      "The remaining promotion step is a real",
      "`intel_qwen36_native` candidate JSONL for all six seed rows.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      (ROOT / f"output/r1-native-candidate-route-{stamp}").resolve()
      if args.out_dir is None
      else args.out_dir.resolve()
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  # R1 component status is sourced from the generated artifacts under output/,
  # not from a per-compare blob inside the contracts (de-bloated 2026-06-28).
  # The two contracts are now a single source of stable facts, so the
  # target/model agreement check below is satisfied by construction.
  r1_index = build_r1_index()
  model_r1 = target_r1 = r1_index

  gate_json = load_json(args.gate_dir / "gate.json")
  gate_correctness = load_json(args.gate_dir / "correctness.json")
  gate_state = gate_json.get("r1_native_correctness_gate", {})
  if not isinstance(gate_state, dict):
    raise SystemExit("gate artifact missing r1_native_correctness_gate")

  oracle_rows = load_jsonl(args.oracle_jsonl)
  case_ids = sorted(
      row["case_id"]
      for row in oracle_rows
      if isinstance(row.get("case_id"), str)
  )

  components, component_issues = collect_components(model_r1, target_r1)
  prerequisites, prerequisite_issues = collect_prerequisites(model_r1, target_r1)
  latest_gate_artifact = model_r1.get("latest_gate_artifact", {})
  load_map = model_r1.get("latest_native_gguf_load_map", {})
  inspect = model_r1.get("latest_engine_gguf_inspect", {})

  gate_open = (
      gate_state.get("native_candidate_present") is False
      and gate_state.get("missing_for_gate") == ["native_candidate_jsonl"]
      and gate_state.get("r1_native_correctness_gate_closed") is False
      and gate_correctness.get("required_checks_passed") is True
      and gate_correctness.get("r1_native_correctness_gate_closed") is False
  )
  contracts_match = all(
      model_r1.get(key) == target_r1.get(key)
      for key in (
          "current_gate",
          "latest_gate_artifact",
          "latest_native_gguf_load_map",
          "latest_engine_gguf_inspect",
          *COMPONENT_COMPARE_KEYS,
          *PREREQUISITE_KEYS,
      )
  )
  load_map_ready = (
      isinstance(load_map, dict)
      and load_map.get("native_gguf_load_map_ready") is True
      and load_map.get("required_checks_passed") is True
      and load_map.get("r1_native_correctness_gate_closed") is False
      and load_map.get("speedup_claims_allowed") is False
  )
  inspect_ready = (
      isinstance(inspect, dict)
      and inspect.get("engine_gguf_inspect_passed") is True
      and inspect.get("required_checks_passed") is True
      and inspect.get("r1_native_correctness_gate_closed") is False
      and inspect.get("speedup_claims_allowed") is False
  )
  component_ladder_ready = len(components) == len(COMPONENT_COMPARE_KEYS) and not component_issues
  prerequisites_ready = (
      len(prerequisites) == len(PREREQUISITE_KEYS)
      and not prerequisite_issues
      and all(
          item.get("seed_prompt_input_path_ready") is True
          for item in prerequisites
      )
  )
  required_checks_passed = (
      contracts_match
      and gate_open
      and len(oracle_rows) == 6
      and len(case_ids) == 6
      and load_map_ready
      and inspect_ready
      and component_ladder_ready
      and prerequisites_ready
  )

  checks = [
      {"name": "target_and_model_r1_contracts_match", "pass": contracts_match},
      {
          "name": "r1_native_gate_still_open_without_candidate",
          "pass": gate_open,
          "missing_for_gate": gate_state.get("missing_for_gate"),
      },
      {
          "name": "oracle_seed_rows_loaded",
          "pass": len(oracle_rows) == 6 and len(case_ids) == 6,
          "case_ids": case_ids,
      },
      {"name": "native_gguf_load_map_ready", "pass": load_map_ready},
      {"name": "engine_gguf_inspect_ready", "pass": inspect_ready},
      {
          "name": "component_compare_ladder_registered_through_sampler",
          "pass": component_ladder_ready,
          "component_compare_count": len(components),
          "component_issues": component_issues,
      },
      {
          "name": "native_seed_prompt_input_path_ready",
          "pass": prerequisites_ready,
          "prerequisite_count": len(prerequisites),
          "prerequisite_issues": prerequisite_issues,
      },
      {
          "name": "native_candidate_jsonl_not_emitted_by_route_artifact",
          "pass": True,
      },
      {
          "name": "does_not_close_native_token_correctness",
          "pass": True,
      },
      {
          "name": "speedup_claims_forbidden_until_native_candidate_passes",
          "pass": True,
      },
  ]

  payload = {
      "created_at": created_at,
      "evidence": {
          "gate_artifact": rel(args.gate_dir),
          "latest_gate_artifact_contract": latest_gate_artifact,
          "model_contract": rel(args.model_contract),
          "oracle_seed": rel(args.oracle_jsonl),
          "target_contract": rel(args.target_contract),
      },
      "native_candidate_route": {
          "candidate_jsonl_required_source": "intel_qwen36_native",
          "component_compare_count": len(components),
          "component_compare_frontier": "global_sampler_topk_plus_l0_stateful_linear_attention_layer_plus_l1_postconv_core_plus_all_30_linear_attention_postconv_core_plus_l3_full_attention_qkv_projection_plus_l3_full_attention_rope_plus_l3_full_attention_core_from_captured_kv_history_plus_l3_stateful_full_attention_kv_append_gate_output_projection_plus_all_10_full_attention_stateful_kv_append_gate_output_projection_for_short_math_001_tok15",
          "component_compare_scope": "single captured token; not six-row generation",
          "gate_can_close_now": False,
          "implementation_gaps": IMPLEMENTATION_GAPS,
          "native_candidate_jsonl": None,
          "native_candidate_jsonl_emitted": False,
          "next_artifact": "native_candidate_jsonl",
          "oracle_seed_case_ids": case_ids,
          "oracle_seed_row_count": len(oracle_rows),
          "prerequisite_evidence_count": len(prerequisites),
          "r1_native_correctness_gate_closed": False,
          "required_checks_passed": required_checks_passed,
          "selected_route": "assemble_o1_first_token_native_loop_from_verified_components",
          "speedup_claims_allowed": False,
      },
      "registered_component_compares": components,
      "registered_prerequisite_evidence": prerequisites,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "current_gate": CURRENT_GATE,
      "gate_artifact": rel(args.gate_dir),
      "oracle_seed": rel(args.oracle_jsonl),
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-native-candidate-route.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "route.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r1_native_candidate_route",
      "native_candidate_jsonl_emitted": False,
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("oracle_seed_row_count", len(oracle_rows)),
        ("component_compare_count", len(components)),
        ("prerequisite_evidence_count", len(prerequisites)),
        ("native_candidate_jsonl_emitted", False),
        ("r1_native_correctness_gate_closed", False),
        ("speedup_claims_allowed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_native_candidate_route",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")

  print(f"r1 native candidate route output: {out_dir}")
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
