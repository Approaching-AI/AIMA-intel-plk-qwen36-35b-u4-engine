#!/usr/bin/env python3
"""Classify bounded early-layer divergence attribution rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-early-divergence-source-gate-v0"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
COSINE_CLOSURE = 0.999999
MAX_LOCAL_MATH_ABS = 2.0e-5


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _failed_tokens(smoke: dict[str, Any]) -> list[int]:
  steps = _dist(smoke).get("steps")
  steps = steps if isinstance(steps, list) else []
  return [
      int(step["token_index"])
      for step in steps
      if isinstance(step, dict)
      and isinstance(step.get("token_index"), int)
      and (_num(step.get("kld")) > KLD_THRESHOLD
           or step.get("top1_matches") is False)
  ]


def _table_layer(smoke: dict[str, Any], table: str, token: int,
                 layer: int) -> dict[str, Any]:
  steps = smoke.get(table)
  steps = steps if isinstance(steps, list) else []
  for step in steps:
    if not isinstance(step, dict) or step.get("token_index") != token:
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if isinstance(row, dict) and row.get("layer") == layer:
        return row
  return {}


def _boundary_step(smoke: dict[str, Any], token: int) -> dict[str, Any]:
  steps = smoke.get("layer_boundary_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  for step in steps:
    if isinstance(step, dict) and step.get("token_index") == token:
      return step
  return {}


def _boundary_layer(smoke: dict[str, Any], token: int,
                    layer: int) -> dict[str, Any]:
  rows = _boundary_step(smoke, token).get("layers")
  rows = rows if isinstance(rows, list) else []
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _row(path: Path, label: str, layers: list[int]) -> dict[str, Any]:
  payload = _load(path)
  smoke = _smoke(payload)
  dist = _dist(smoke)
  tokens = _failed_tokens(smoke)
  entries = []
  for token in tokens:
    layer_rows = []
    for layer in layers:
      boundary = _boundary_layer(smoke, token, layer)
      residual = _table_layer(
          smoke, "residual_source_diff_by_step", token, layer)
      preconv = _table_layer(
          smoke, "linear_preconv_source_diff_by_step", token, layer)
      ffn = _table_layer(smoke, "ffn_live_math_diff_by_step", token, layer)
      projection = _table_layer(
          smoke, "linear_projection_source_diff_by_step", token, layer)
      layer_rows.append({
          "layer": layer,
          "input_cosine": boundary.get("input_cosine"),
          "input_max_abs_diff": boundary.get("input_max_abs_diff"),
          "output_cosine": boundary.get("output_cosine"),
          "output_max_abs_diff": boundary.get("output_max_abs_diff"),
          "attention_output_cosine": residual.get("attention_output_cosine"),
          "attention_output_max_abs_diff": residual.get(
              "attention_output_max_abs_diff"),
          "gpu_ffn_vs_cpu_cosine": ffn.get("gpu_output_vs_cpu_ffn_cosine"),
          "gpu_ffn_vs_cpu_max_abs_diff": ffn.get(
              "gpu_output_vs_cpu_ffn_max_abs_diff"),
          "gpu_attn_norm_vs_cpu_cosine": preconv.get(
              "gpu_attn_norm_vs_cpu_cosine"),
          "gpu_attn_norm_vs_cpu_max_abs_diff": preconv.get(
              "gpu_attn_norm_vs_cpu_max_abs_diff"),
          "gpu_qkv_vs_cpu_cosine": preconv.get("gpu_qkv_vs_cpu_cosine"),
          "gpu_qkv_vs_cpu_max_abs_diff": preconv.get(
              "gpu_qkv_vs_cpu_max_abs_diff"),
          "gpu_z_vs_cpu_cosine": preconv.get("gpu_z_vs_cpu_cosine"),
          "gpu_z_vs_cpu_max_abs_diff": preconv.get(
              "gpu_z_vs_cpu_max_abs_diff"),
          "projection_cpu_from_gpu_input_cosine": projection.get(
              "cpu_projection_from_gpu_input_vs_native_cosine"),
          "projection_cpu_from_gpu_input_max_abs_diff": projection.get(
              "cpu_projection_from_gpu_input_vs_native_max_abs_diff"),
          "projection_gpu_vs_cpu_cosine": projection.get(
              "gpu_output_vs_cpu_projection_from_gpu_input_cosine"),
          "projection_gpu_vs_cpu_max_abs_diff": projection.get(
              "gpu_output_vs_cpu_projection_from_gpu_input_max_abs_diff"),
          "projection_q8_qs_mismatch_count": projection.get(
              "q8_qs_mismatch_count"),
          "projection_q8_bsums_mismatch_count": projection.get(
              "q8_bsums_mismatch_count"),
      })
    boundary_step = _boundary_step(smoke, token)
    entries.append({
        "token_index": token,
        "boundary_layer_count": len(boundary_step.get("layers", [])),
        "first_input_cosine_below_9999": boundary_step.get(
            "first_input_cosine_below_9999"),
        "first_output_cosine_below_9999": boundary_step.get(
            "first_output_cosine_below_9999"),
        "layers": layer_rows,
    })
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "distribution_required_checks_passed": dist.get(
          "required_checks_passed"),
      "failed_tokens": tokens,
      "entries": entries,
  }


def _row_ran(row: dict[str, Any], diagnostic_range: str) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("diagnostic_layer_range") == diagnostic_range
      and row.get("distribution_required_checks_passed") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and row.get("kld_pass") is False
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and bool(row.get("entries"))
      and all(entry.get("boundary_layer_count") == 40
              for entry in row.get("entries", [])))


def _layer_rows(*rows: dict[str, Any]) -> list[dict[str, Any]]:
  return [
      layer
      for row in rows
      for entry in row.get("entries", [])
      for layer in entry.get("layers", [])
  ]


def _max_metric(name: str, *rows: dict[str, Any]) -> float:
  return max((_num(row.get(name)) for row in _layer_rows(*rows)), default=0.0)


def _min_metric(name: str, *rows: dict[str, Any]) -> float:
  return min((_num(row.get(name), 1.0) for row in _layer_rows(*rows)),
             default=1.0)


def _local_math_closes(*rows: dict[str, Any]) -> bool:
  layer_rows = _layer_rows(*rows)
  return bool(layer_rows) and all(
      _num(row.get("gpu_ffn_vs_cpu_cosine"), 1.0) >= COSINE_CLOSURE
      and _num(row.get("gpu_ffn_vs_cpu_max_abs_diff")) <= MAX_LOCAL_MATH_ABS
      and _num(row.get("gpu_attn_norm_vs_cpu_cosine"), 1.0) >= COSINE_CLOSURE
      and _num(row.get("gpu_attn_norm_vs_cpu_max_abs_diff")) <= MAX_LOCAL_MATH_ABS
      and _num(row.get("gpu_qkv_vs_cpu_cosine"), 1.0) >= COSINE_CLOSURE
      and _num(row.get("gpu_qkv_vs_cpu_max_abs_diff")) <= MAX_LOCAL_MATH_ABS
      and _num(row.get("gpu_z_vs_cpu_cosine"), 1.0) >= COSINE_CLOSURE
      and _num(row.get("gpu_z_vs_cpu_max_abs_diff")) <= MAX_LOCAL_MATH_ABS
      for row in layer_rows)


def _first_nonzero_layers(*rows: dict[str, Any]) -> list[int]:
  # The bounded rows include layers in ascending order and global boundary
  # summaries. A 1e-6 absolute threshold separates arithmetic noise from the
  # first amplified boundary without reusing the looser 0.9999 material ruler.
  candidates: list[int] = []
  for row in rows:
    payload = _load(ROOT / row["path"])
    smoke = _smoke(payload)
    for token in row.get("failed_tokens", []):
      step = _boundary_step(smoke, token)
      layers = step.get("layers")
      layers = layers if isinstance(layers, list) else []
      hit = next((
          layer.get("layer") for layer in layers
          if isinstance(layer, dict)
          and _num(layer.get("output_max_abs_diff")) > 1.0e-6
      ), None)
      if isinstance(hit, int):
        candidates.append(hit)
  return sorted(set(candidates))


def _mode_config(mode: str) -> dict[str, Any]:
  if mode == "material":
    return {
        "layers": [4, 5],
        "range": "4:6",
        "current": "router_prompt_distribution_layers4_5_first_material_divergence_source_gate",
        "next": "router_prompt_distribution_layers1_2_earliest_nonzero_source_gate",
        "rejected": "router_prompt_distribution_layers4_5_local_kernel_math_source",
        "disposition": "reject_layers4_5_local_kernel_math_select_layers1_2_earliest_nonzero_source",
    }
  if mode == "earliest":
    return {
        "layers": [1, 2],
        "range": "1:3",
        "current": "router_prompt_distribution_layers1_2_earliest_nonzero_source_gate",
        "next": "router_prompt_distribution_layer0_exact_delta_source_gate",
        "rejected": "router_prompt_distribution_layers1_2_local_kernel_math_source",
        "disposition": "reject_layers1_2_local_kernel_math_select_layer0_exact_delta_source",
    }
  if mode == "root":
    return {
        "layers": [0],
        "range": "0:1",
        "current": "router_prompt_distribution_layer0_exact_delta_source_gate",
        "next": "router_prompt_distribution_layer0_1_precision_island_feasibility_gate",
        "rejected": "router_prompt_distribution_layer0_projection_q8_bridge_source",
        "disposition": "reject_layer0_projection_q8_bridge_select_layer0_1_precision_island_feasibility",
    }
  raise ValueError(f"unsupported mode: {mode}")


def compute(args: argparse.Namespace) -> dict[str, Any]:
  config = _mode_config(args.mode)
  predecessor = _load(args.predecessor)
  math_row = _row(args.math, f"seq{args.sequence}_{args.mode}_math",
                  config["layers"])
  code_row = _row(args.code, f"seq{args.sequence}_{args.mode}_code",
                  config["layers"])
  rows_ran = _row_ran(math_row, config["range"]) and _row_ran(
      code_row, config["range"])
  local_math_closes = _local_math_closes(math_row, code_row)
  first_nonzero = _first_nonzero_layers(math_row, code_row)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("selected_next_route") == config["current"])
  extra_check = True
  if args.mode == "root":
    root_rows = _layer_rows(math_row, code_row)
    extra_check = (
        bool(root_rows)
        and all(_num(row.get("input_max_abs_diff")) == 0.0 for row in root_rows)
        and any(_num(row.get("output_max_abs_diff")) > 0.0 for row in root_rows)
        and all(row.get("projection_q8_qs_mismatch_count") == 0
                and row.get("projection_q8_bsums_mismatch_count") == 0
                for row in root_rows))
  checks = [
      {"name": "predecessor_selected_current_route",
       "pass": predecessor_selects},
      {"name": "bounded_router_rows_ran", "pass": rows_ran},
      {"name": "local_gpu_math_matches_cpu_from_live_input",
       "pass": local_math_closes},
      {"name": "mode_specific_source_evidence", "pass": extra_check},
  ]
  required = all(bool(check["pass"]) for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "mode": args.mode,
      "inputs": {
          "predecessor": _rel(args.predecessor),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "rows": {"math": math_row, "code": code_row},
      "metrics": {
          "first_nonzero_output_layers": first_nonzero,
          "max_input_abs_diff": _max_metric(
              "input_max_abs_diff", math_row, code_row),
          "max_output_abs_diff": _max_metric(
              "output_max_abs_diff", math_row, code_row),
          "min_gpu_ffn_vs_cpu_cosine": _min_metric(
              "gpu_ffn_vs_cpu_cosine", math_row, code_row),
          "max_gpu_ffn_vs_cpu_abs_diff": _max_metric(
              "gpu_ffn_vs_cpu_max_abs_diff", math_row, code_row),
          "max_gpu_attn_norm_vs_cpu_abs_diff": _max_metric(
              "gpu_attn_norm_vs_cpu_max_abs_diff", math_row, code_row),
          "max_gpu_qkv_vs_cpu_abs_diff": _max_metric(
              "gpu_qkv_vs_cpu_max_abs_diff", math_row, code_row),
          "max_projection_gpu_vs_cpu_abs_diff": _max_metric(
              "projection_gpu_vs_cpu_max_abs_diff", math_row, code_row),
      },
      "disposition": (config["disposition"] if required else
                      f"block_seq{args.sequence}_{args.mode}_inconsistent_evidence"),
      "rejected_route": config["rejected"] if required else None,
      "selected_next_route": config["next"] if required else config["current"],
      "next_route_reason": (
          "The bounded rows reproduce CPU FFN/preconv math from the same live "
          "input. The remaining divergence is inherited or sub-ULP state/order "
          "drift, so continue with the selected earlier source/precision route."
          if required else
          "The bounded early-divergence evidence is inconsistent; keep the "
          "current route open and do not launch promotion rows."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  lines = [
      f"# Seq{metrics['sequence']} Early Divergence Source Gate",
      "",
      f"- mode: `{metrics['mode']}`",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- first_nonzero_output_layers: `{metrics['metrics']['first_nonzero_output_layers']}`",
      f"- max_input_abs_diff: `{metrics['metrics']['max_input_abs_diff']}`",
      f"- max_output_abs_diff: `{metrics['metrics']['max_output_abs_diff']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, required=True)
  parser.add_argument("--mode", choices=("material", "earliest", "root"),
                      required=True)
  parser.add_argument("--predecessor", type=Path, required=True)
  parser.add_argument("--math", type=Path, required=True)
  parser.add_argument("--code", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, required=True)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
