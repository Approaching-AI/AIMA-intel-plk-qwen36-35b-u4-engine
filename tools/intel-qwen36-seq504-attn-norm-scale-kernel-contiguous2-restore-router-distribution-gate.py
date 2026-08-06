#!/usr/bin/env python3
"""Run router distribution after restoring accepted contiguous2 RMSNorm scale."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SEQ501_GATE = (
    ROOT
    / "tools/intel-qwen36-seq501-attn-norm-scale-kernel-reduction-order-target-gate.py"
)
SEQ502_GATE = (
    ROOT
    / "tools/intel-qwen36-seq502-attn-norm-serial-scale-cpu-sqrt-target-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq504-attn-norm-scale-kernel-contiguous2-restore-router-distribution-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ502 = (
    ROOT
    / "output/seq502-attn-norm-serial-scale-cpu-sqrt-target-gate-20260709Tseq502Z"
    / "metrics.json"
)
DEFAULT_SEQ503 = (
    ROOT
    / "output/seq503-product-serial-cpu-sqrt-router-distribution-gate-20260709Tseq503Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq504-attn-norm-scale-kernel-contiguous2-restore-router-distribution-gate-20260709Tseq504Z"
)
SOURCE_PATH = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"

KLD_MAX = 0.005
TOP1_MIN = 0.99


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ501 = _load_module(SEQ501_GATE, "iq36_seq501_for_seq504")
SEQ502 = _load_module(SEQ502_GATE, "iq36_seq502_for_seq504")
CASES = SEQ501.SEQ500.CASES
SERIAL_CPU_SQRT_ROUTE = SEQ502.SERIAL_CPU_SQRT_OPEN_ROUTE
SERIAL_DISTRIBUTION_ROUTE = SERIAL_CPU_SQRT_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_unresolved_gate",
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_router_distribution_gap_gate")
ACCEPTED_CONTIGUOUS2_ROUTE = SERIAL_DISTRIBUTION_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_router_distribution_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_accepted_contiguous2_router_distribution_gap_gate")
CONTIGUOUS2_PASS_ROUTE = SERIAL_DISTRIBUTION_ROUTE.replace(
    "_linear_z_source_attn_norm_scale_kernel_serial_cpu_sqrt_router_distribution_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_accepted_contiguous2_router_distribution_pass_gate")


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _block(text: str, marker: str) -> str:
  start = text.find(marker)
  if start < 0:
    return ""
  next_kernel = text.find("__kernel void", start + len(marker))
  return text[start:] if next_kernel < 0 else text[start:next_kernel]


def _contiguous2_shape(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8")
  block = _block(text, "__kernel void rms_norm_hidden_scale_f32")
  return {
      "path": _rel(path),
      "kernel_present": bool(block),
      "partial_array": "__local float partial[256];" in block,
      "chunked_by_local_size": (
          "const uint chunk = (hidden_size + local_size - 1U) / local_size;"
          in block),
      "rsqrt_scale_out": "scale_out[0] = rsqrt(mean_square + epsilon);" in block,
      "cpu_sqrt_scale_out": "scale_out[0] = 1.0f / sqrt(mean_square + epsilon);" in block,
  }


def _shape_ok(shape: dict[str, Any]) -> bool:
  return (
      shape["kernel_present"]
      and shape["partial_array"]
      and shape["chunked_by_local_size"]
      and shape["rsqrt_scale_out"]
      and not shape["cpu_sqrt_scale_out"])


def _dist_row(case_id: str, result_path: Path) -> dict[str, Any]:
  result = _load_json(result_path)
  smoke = result.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "case_id": case_id,
      "result": _rel(result_path),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": _num(dist.get("max_kld")),
      "top1_rate": _num(dist.get("top1_rate")),
      "min_logits_cosine": _num(dist.get("min_logits_cosine")),
      "position_count": int(_num(dist.get("position_count"))),
      "opencl_cpu_sqrt_norm_enabled": smoke.get("opencl_cpu_sqrt_norm_enabled"),
      "source_enabled": smoke.get("full_attention_layer_input_product_source_enabled"),
      "source_ready": smoke.get("full_attention_layer_input_product_source_ready"),
      "source_layers": smoke.get("full_attention_layer_input_product_source_layers"),
      "source_values": smoke.get("full_attention_layer_input_product_source_values"),
      "source_misses": smoke.get("full_attention_layer_input_product_source_misses"),
      "consumer_enabled": smoke.get("full_attention_layer_input_product_consumer_source_enabled"),
      "consumer_ready": smoke.get("full_attention_layer_input_product_consumer_source_ready"),
      "consumer_layers": smoke.get("full_attention_layer_input_product_consumer_source_layers"),
      "consumer_values": smoke.get("full_attention_layer_input_product_consumer_source_values"),
      "consumer_misses": smoke.get("full_attention_layer_input_product_consumer_source_misses"),
      "speedup_claims_allowed": smoke.get("speedup_claims_allowed"),
  }


def _case_pass(row: dict[str, Any]) -> bool:
  return row["max_kld"] <= KLD_MAX and row["top1_rate"] >= TOP1_MIN


def _run_restored_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
  runs = [SEQ501._run_case(args, case_id) for case_id in CASES]
  rows: list[dict[str, Any]] = []
  for run in runs:
    case_id = str(run.get("case_id"))
    result_path = args.out_dir / "cases" / case_id / "result.json"
    row = _dist_row(case_id, result_path) if result_path.exists() else {
        "case_id": case_id,
        "result": _rel(result_path),
        "required_checks_passed": None,
        "distribution_required_checks_passed": None,
        "max_kld": 0.0,
        "top1_rate": 0.0,
        "min_logits_cosine": 0.0,
        "position_count": 0,
        "opencl_cpu_sqrt_norm_enabled": None,
        "source_enabled": None,
        "source_ready": None,
        "source_layers": None,
        "source_values": None,
        "source_misses": None,
        "consumer_enabled": None,
        "consumer_ready": None,
        "consumer_layers": None,
        "consumer_values": None,
        "consumer_misses": None,
        "speedup_claims_allowed": None,
    }
    row["returncode"] = run.get("returncode")
    generated = args.out_dir / "cases" / case_id / "r2_gpu_decode_smoke.cpp"
    row["generated_source_shape"] = (
        _contiguous2_shape(generated) if generated.exists() else None)
    row["distribution_passed"] = _case_pass(row)
    rows.append(row)
  return rows


def _load_seq503_rows(seq503_dir: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for case_id in CASES:
    path = seq503_dir / "cases" / case_id / "result.json"
    if path.exists():
      row = _dist_row(case_id, path)
      row["distribution_passed"] = _case_pass(row)
      rows.append(row)
  return rows


def _max_by_case(rows: list[dict[str, Any]]) -> dict[str, float]:
  return {str(row["case_id"]): _num(row.get("max_kld")) for row in rows}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq502 = _load_json(args.seq502)
  source_shape = _contiguous2_shape(SOURCE_PATH)
  seq503_rows = _load_seq503_rows(args.seq503_dir)
  restored_rows = _run_restored_cases(args)
  preconditions_pass = (
      seq502.get("required_checks_passed") is True
      and seq502.get("selected_next_route") == SERIAL_CPU_SQRT_ROUTE
      and seq502.get("serial_cpu_sqrt_still_open") is True
      and _has_candidate(
          routes, 503, "reject_serial_cpu_sqrt_scale_kernel_product_distribution")
      and _has_switch(routes, f"select_{SERIAL_DISTRIBUTION_ROUTE}", 503)
  )
  rows_emitted = (
      len(restored_rows) == len(CASES)
      and all(row.get("position_count") == SEQ501.SEQ500.DECODE_TOKENS
              for row in restored_rows)
      and all(row.get("opencl_cpu_sqrt_norm_enabled") is not True
              for row in restored_rows))
  generated_shapes_ok = all(
      _shape_ok(row.get("generated_source_shape") or {})
      for row in restored_rows)
  counters_ready = rows_emitted and all(
      row.get("source_enabled") is True
      and row.get("source_ready") is True
      and row.get("source_layers") == SEQ501.SEQ500.EXPECTED_COUNTER_LAYERS
      and row.get("source_values") == SEQ501.SEQ500.EXPECTED_COUNTER_VALUES
      and row.get("source_misses") == 0
      and row.get("consumer_enabled") is True
      and row.get("consumer_ready") is True
      and row.get("consumer_layers") == SEQ501.SEQ500.EXPECTED_COUNTER_LAYERS
      and row.get("consumer_values") == SEQ501.SEQ500.EXPECTED_COUNTER_VALUES
      and row.get("consumer_misses") == 0
      and row.get("speedup_claims_allowed") is False
      for row in restored_rows)
  distribution_classified = rows_emitted and all(
      isinstance(row.get("distribution_passed"), bool) for row in restored_rows)
  restored_distribution_passed = (
      distribution_classified
      and all(row["distribution_passed"] for row in restored_rows))
  serial_by_case = _max_by_case(seq503_rows)
  restored_by_case = _max_by_case(restored_rows)
  deltas = {
      case_id: restored_by_case.get(case_id, 0.0) - serial_by_case.get(case_id, 0.0)
      for case_id in sorted(set(serial_by_case) | set(restored_by_case))
  }
  required = (
      preconditions_pass
      and _shape_ok(source_shape)
      and generated_shapes_ok
      and counters_ready
      and distribution_classified)
  selected = (
      CONTIGUOUS2_PASS_ROUTE if restored_distribution_passed
      else ACCEPTED_CONTIGUOUS2_ROUTE)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq502": _rel(args.seq502),
          "seq503_dir": _rel(args.seq503_dir),
          "token_input_dir": _rel(args.token_input_dir),
      },
      "source_shape": source_shape,
      "seq503_serial_cpu_sqrt_cases": seq503_rows,
      "restored_contiguous2_cases": restored_rows,
      "distribution_thresholds": {
          "max_kld": KLD_MAX,
          "top1_min": TOP1_MIN,
      },
      "comparison": {
          "serial_cpu_sqrt_max_kld_by_case": serial_by_case,
          "restored_contiguous2_max_kld_by_case": restored_by_case,
          "restored_minus_serial_max_kld_by_case": deltas,
      },
      "checks": [
          {"name": "seq503_serial_distribution_gate_selected",
           "pass": preconditions_pass},
          {"name": "product_source_restored_to_accepted_contiguous2",
           "pass": _shape_ok(source_shape), "detail": source_shape},
          {"name": "target_restored_rows_emitted", "pass": rows_emitted},
          {"name": "generated_sources_use_accepted_contiguous2",
           "pass": generated_shapes_ok},
          {"name": "product_source_consumer_counters_ready",
           "pass": counters_ready},
          {"name": "router_distribution_classified",
           "pass": distribution_classified,
           "detail": {"restored_distribution_passed":
                      restored_distribution_passed}},
      ],
      "required_checks_passed": required,
      "diagnostic_classification": (
          "attn_norm_scale_kernel_accepted_contiguous2_router_distribution_pass"
          if required and restored_distribution_passed else
          "attn_norm_scale_kernel_accepted_contiguous2_router_distribution_gap"
          if required else
          "attn_norm_scale_kernel_accepted_contiguous2_router_distribution_unclassified"),
      "restored_distribution_passed": restored_distribution_passed,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_accepted_contiguous2_scale_kernel_router_distribution_pass"
          if required and restored_distribution_passed else
          "accept_accepted_contiguous2_scale_kernel_router_distribution_gap"
          if required else
          "block_accepted_contiguous2_scale_kernel_router_distribution_gate"),
      "selected_next_route": selected if required else SERIAL_DISTRIBUTION_ROUTE,
      "next_route_reason": (
          "Restoring the accepted contiguous2/rsqrt shared scale source clears "
          "router distribution; speed promotion still needs the full acceptance "
          "matrix."
          if required and restored_distribution_passed else
          "Restoring the accepted contiguous2/rsqrt shared scale source avoids "
          "carrying the serial/CPU-sqrt diagnostic as product state, but router "
          "distribution still fails; continue from the accepted-contiguous2 "
          "product baseline."
          if required else
          "Restored contiguous2 distribution evidence is incomplete; keep the "
          "serial distribution gate open.")
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  rows = metrics["restored_contiguous2_cases"]
  lines = [
      "# Seq504 Accepted Contiguous2 Scale Router Distribution Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- restored_distribution_passed: `{str(metrics['restored_distribution_passed']).lower()}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "| case | max KLD | top-1 rate | min cosine | pass |",
      "|---|---:|---:|---:|---|",
  ]
  for row in rows:
    lines.append(
        f"| `{row['case_id']}` | `{row['max_kld']}` | "
        f"`{row['top1_rate']}` | `{row['min_logits_cosine']}` | "
        f"`{str(row['distribution_passed']).lower()}` |")
  lines.extend([
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness/distribution evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq502", type=Path, default=DEFAULT_SEQ502)
  parser.add_argument("--seq503-dir", type=Path, default=DEFAULT_SEQ503)
  parser.add_argument("--token-input-dir", type=Path,
                      default=SEQ501.DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=SEQ501.DEFAULT_HOST)
  parser.add_argument("--model", default=SEQ501.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=SEQ501.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=SEQ501.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=1800)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
