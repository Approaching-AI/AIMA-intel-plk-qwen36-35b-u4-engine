#!/usr/bin/env python3
"""Estimate per-bucket decode KV/read pressure for the locked model."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
ACCEPTANCE = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/acceptance-matrix.json"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
KV_DTYPES = {
    "fp8": 1,
    "fp16_or_bf16": 2,
    "fp32": 4,
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def latest_audit(prefix: str) -> tuple[Path | None, dict[str, Any] | None]:
  paths = sorted((ROOT / "output").glob(f"{prefix}*/audit.json"))
  if not paths:
    return None, None
  path = paths[-1]
  return path, load_json(path)


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def tok_s(gb_s: float | None, gb_per_token: float) -> float | None:
  if gb_s is None or gb_per_token <= 0:
    return None
  return gb_s / gb_per_token


def build_summary(payload: dict[str, Any]) -> str:
  findings = payload["findings"]
  fp16_256k = findings["fp16_or_bf16_262144"]
  longest = findings["longest_bootstrap_bucket"]
  return "\n".join(
      [
          "# R0 KV/Read Pressure",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- full attention layers counted: {payload['model']['full_attention_layers']}",
          f"- KV heads: {payload['model']['kv_heads']}",
          f"- head dim: {payload['model']['head_dim']}",
          f"- fp16/bf16 KV read at 262144: {fp16_256k['kv_read_gb_per_decode_token']:.6f} GB/token",
          f"- 262144 ceiling at source-stream max: {fp16_256k['ceiling_tok_s_at_source_stream_max']:.3f} tok/s",
          f"- 262144 ceiling at qmatvec max: {fp16_256k['ceiling_tok_s_at_qmatvec_max']:.3f} tok/s",
          f"- 262144 ceiling at 115 GB/s line: {fp16_256k['ceiling_tok_s_at_target_line']:.3f} tok/s",
          f"- longest known bootstrap bucket: {longest['bucket']}",
          f"- longest known bootstrap KV bandwidth demand: {longest['observed_bootstrap_kv_gb_s']:.3f} GB/s",
          f"- R0 performance gate closed: `{str(payload['r0_performance_gate_closed']).lower()}`",
          "",
          "This is a model-contract pressure estimate for decode KV reads. It",
          "does not measure end-to-end latency and does not close R0 by itself.",
          "",
      ]
  )


def main() -> None:
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (ROOT / f"output/r0-kv-read-pressure-{stamp}").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  model_contract = load_json(MODEL_CONTRACT)
  acceptance = load_json(ACCEPTANCE)
  target_contract = load_json(TARGET_CONTRACT)
  source_path, source_audit_payload = latest_audit("r0-source-stream-roof-")
  qmatvec_path, qmatvec_audit_payload = latest_audit("r0-qmatvec-probe-")

  model = model_contract["model"]
  matrix = acceptance["matrix"]
  r0_policy = acceptance["r0_target_policy"]
  bootstrap_decode = acceptance.get("bootstrap_targets", {}).get("decode_tokens_s", {})

  full_attention_layers = int(model["full_attention_layers"])
  kv_heads = int(model["kv_heads"])
  head_dim = int(model["head_dim"])
  context_length = int(model["context_length"])
  buckets = [int(bucket) for bucket in matrix["input_buckets"]]

  # Decode for one new token reads K and V for each prior token in each full
  # attention layer. Use the bucket size as a conservative proxy for prior
  # tokens. Linear-attention layers are not counted in this KV term.
  kv_elements_per_context_token_per_full_layer = 2 * kv_heads * head_dim
  kv_elements_per_context_token_all_full_layers = (
      full_attention_layers * kv_elements_per_context_token_per_full_layer
  )

  source_audit = (source_audit_payload or {}).get("audit", {})
  qmatvec_audit = (qmatvec_audit_payload or {}).get("audit", {})
  source_max_gb_s = source_audit.get("max_source_gb_s")
  if not finite(source_max_gb_s):
    source_max_gb_s = None
  qmatvec_max_gb_s = qmatvec_audit.get("max_effective_tensor_gb_s")
  if not finite(qmatvec_max_gb_s):
    qmatvec_max_gb_s = None
  target_line_gb_s = source_audit.get("target_gb_s")
  if not finite(target_line_gb_s):
    target_line_gb_s = 115.0
  target_ratio = float(r0_policy.get("roofline_default_target_ratio", 0.7))
  target_ratio_line_gb_s = float(target_line_gb_s) * target_ratio

  rows: list[dict[str, Any]] = []
  for dtype, scalar_bytes in KV_DTYPES.items():
    kv_bytes_per_context_token = (
        kv_elements_per_context_token_all_full_layers * scalar_bytes
    )
    for bucket in buckets:
      kv_read_bytes = kv_bytes_per_context_token * bucket
      kv_read_gb = kv_read_bytes / 1.0e9
      bootstrap_tok_s = bootstrap_decode.get(str(bucket))
      observed_bootstrap_kv_gb_s = None
      if finite(bootstrap_tok_s):
        observed_bootstrap_kv_gb_s = float(bootstrap_tok_s) * kv_read_gb
      rows.append({
          "bucket": bucket,
          "cache_state": matrix["cache_mode"],
          "ceiling_tok_s_at_qmatvec_max": tok_s(qmatvec_max_gb_s, kv_read_gb),
          "ceiling_tok_s_at_source_stream_max": tok_s(source_max_gb_s, kv_read_gb),
          "ceiling_tok_s_at_target_line": tok_s(float(target_line_gb_s), kv_read_gb),
          "ceiling_tok_s_at_target_ratio_line": tok_s(target_ratio_line_gb_s, kv_read_gb),
          "dtype": dtype,
          "kv_bytes_per_context_token_all_full_layers": kv_bytes_per_context_token,
          "kv_elements_per_context_token_all_full_layers": (
              kv_elements_per_context_token_all_full_layers
          ),
          "kv_read_bytes_per_decode_token": kv_read_bytes,
          "kv_read_gb_per_decode_token": kv_read_gb,
          "kv_read_gib_per_decode_token": kv_read_bytes / (1024.0 ** 3),
          "observed_bootstrap_decode_tok_s": (
              float(bootstrap_tok_s) if finite(bootstrap_tok_s) else None
          ),
          "observed_bootstrap_kv_gb_s": observed_bootstrap_kv_gb_s,
          "phase": "decode_kv_read_pressure",
      })

  fp16_rows = [row for row in rows if row["dtype"] == "fp16_or_bf16"]
  fp16_256k = next(row for row in fp16_rows if row["bucket"] == context_length)
  bootstrap_known = [
      row for row in fp16_rows if finite(row.get("observed_bootstrap_kv_gb_s"))
  ]
  longest_bootstrap = max(bootstrap_known, key=lambda row: row["bucket"])

  findings = {
      "fp16_or_bf16_262144": {
          "ceiling_tok_s_at_qmatvec_max": fp16_256k["ceiling_tok_s_at_qmatvec_max"],
          "ceiling_tok_s_at_source_stream_max": (
              fp16_256k["ceiling_tok_s_at_source_stream_max"]
          ),
          "ceiling_tok_s_at_target_line": fp16_256k["ceiling_tok_s_at_target_line"],
          "ceiling_tok_s_at_target_ratio_line": (
              fp16_256k["ceiling_tok_s_at_target_ratio_line"]
          ),
          "kv_read_gb_per_decode_token": fp16_256k["kv_read_gb_per_decode_token"],
      },
      "longest_bootstrap_bucket": {
          "bucket": longest_bootstrap["bucket"],
          "observed_bootstrap_decode_tok_s": (
              longest_bootstrap["observed_bootstrap_decode_tok_s"]
          ),
          "observed_bootstrap_kv_gb_s": (
              longest_bootstrap["observed_bootstrap_kv_gb_s"]
          ),
      },
      "source_stream_max_gb_s": source_max_gb_s,
      "qmatvec_max_effective_tensor_gb_s": qmatvec_max_gb_s,
      "target_line_gb_s": target_line_gb_s,
      "target_ratio_line_gb_s": target_ratio_line_gb_s,
  }

  payload = {
      "assumptions": {
          "batch_size": model["batch_size"],
          "context_bucket_used_as_prior_tokens": True,
          "formula": (
              "bucket * full_attention_layers * 2(K,V) * kv_heads * head_dim "
              "* kv_scalar_bytes"
          ),
          "linear_attention_layers_kv_not_counted": True,
          "weight_reads_not_counted_in_kv_rows": True,
      },
      "created_at": created_at,
      "evidence": {
          "acceptance_matrix": str(ACCEPTANCE.relative_to(ROOT)),
          "model_contract": str(MODEL_CONTRACT.relative_to(ROOT)),
          "qmatvec_audit": str(qmatvec_path.relative_to(ROOT)) if qmatvec_path else None,
          "source_stream_audit": str(source_path.relative_to(ROOT)) if source_path else None,
          "target_contract": str(TARGET_CONTRACT.relative_to(ROOT)),
      },
      "findings": findings,
      "model": {
          "context_length": context_length,
          "full_attention_layers": full_attention_layers,
          "head_dim": head_dim,
          "kv_heads": kv_heads,
          "linear_attention_layers": model["linear_attention_layers"],
      },
      "r0_performance_gate_closed": False,
      "rows": rows,
      "schema_version": "intel-qwen36-r0-kv-read-pressure-v0",
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-kv-read-pressure.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "budget.json", payload)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True) + "\n")
  write_json(out_dir / "correctness.json", {
      "checks": [
          {
              "all_required_buckets_present": set(buckets) == set(matrix["input_buckets"]),
              "full_attention_layers_positive": full_attention_layers > 0,
              "kv_shape_positive": kv_heads > 0 and head_dim > 0,
              "qmatvec_audit_found": qmatvec_path is not None,
              "rows_positive_and_finite": all(
                  row["kv_read_bytes_per_decode_token"] > 0
                  and finite(row["kv_read_gb_per_decode_token"])
                  for row in rows
              ),
              "source_stream_audit_found": source_path is not None,
          }
      ],
      "gate": "r0_kv_read_pressure_estimate",
      "required_checks_passed": (
          full_attention_layers > 0
          and kv_heads > 0
          and head_dim > 0
          and source_path is not None
          and qmatvec_path is not None
      ),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": True,
      "checks": [
          {
              "dtype": dtype,
              "kv_read_monotonic_with_bucket": all(
                  typed[i]["kv_read_bytes_per_decode_token"]
                  < typed[i + 1]["kv_read_bytes_per_decode_token"]
                  for i in range(len(typed) - 1)
              ),
          }
          for dtype in KV_DTYPES
          for typed in [[row for row in rows if row["dtype"] == dtype]]
      ],
      "notes": "linear KV/read pressure estimate, not end-to-end smoothness",
  })
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"kv/read pressure output: {out_dir}")


if __name__ == "__main__":
  main()
