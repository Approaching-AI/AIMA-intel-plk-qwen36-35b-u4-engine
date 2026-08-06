#!/usr/bin/env python3
"""Correct the packed-token state census and reallocate its wall budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-packed-token-state-budget-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEDULE = (
    ROOT / "output/packed-token-schedule-gate-20260712Tseq732cleanZ/result.json")
LEVEL_ZERO = (
    ROOT / "output/packed-token-level-zero-gate-20260712Tseq733cleanZ/result.json")
ROUTE = (
    ROOT / "output/product-decode-route-gate-20260712Tseq731cleanZ/result.json")

LINEAR_LAYER_COUNT = 30
LINEAR_RECURRENT_STATE_VALUES = 32 * 128 * 128
LINEAR_CONV_STATE_VALUES = (4 - 1) * 8192
F32_BYTES = 4
ACTIVE_WEIGHT_BYTES = 1_975_676_544
KV_HISTORY_READ_BYTES = 20_971_520
KV_APPEND_WRITE_BYTES = 20_480
HOST_BUDGET_MS = 0.100


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/packed-token-state-budget-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  proc = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return proc.stdout.strip() if proc.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty), "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def residuals(row: dict[str, Any]) -> list[float]:
  device = row.get("device_time_samples_us", [])
  wall = row.get("wall_time_samples_us", [])
  if not isinstance(device, list) or not isinstance(wall, list):
    return []
  return [float(w) - float(d) for d, w in zip(device, wall)]


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  required = [SCHEDULE, LEVEL_ZERO, ROUTE]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  schedule = load_json(SCHEDULE)
  level_zero = load_json(LEVEL_ZERO)
  route = load_json(ROUTE)
  repeat = level_zero.get("repeat", {})
  confirm = level_zero.get("confirm", {})
  route_budget = route.get("budget", {})
  old_smoke = schedule.get("smoke", {})

  recurrent_read_bytes = (
      LINEAR_LAYER_COUNT * LINEAR_RECURRENT_STATE_VALUES * F32_BYTES)
  recurrent_write_bytes = recurrent_read_bytes
  conv_read_bytes = LINEAR_LAYER_COUNT * LINEAR_CONV_STATE_VALUES * F32_BYTES
  conv_write_bytes = conv_read_bytes
  resident_state_read_bytes = (
      recurrent_read_bytes + conv_read_bytes + KV_HISTORY_READ_BYTES)
  resident_state_write_bytes = (
      recurrent_write_bytes + conv_write_bytes + KV_APPEND_WRITE_BYTES)
  strict_stream_bytes = (
      ACTIVE_WEIGHT_BYTES + resident_state_read_bytes +
      resident_state_write_bytes)

  product_floor = float(route_budget.get("product_floor_tokens_s", 0.0))
  wall_ms = 1000.0 / product_floor
  kernel_ms = wall_ms - HOST_BUDGET_MS
  strict_target_gb_s = strict_stream_bytes / 1e9 * product_floor
  kernel_window_gb_s = strict_stream_bytes / 1e6 / kernel_ms
  observed_residuals = residuals(repeat) + residuals(confirm)
  observed_host_residual_max_us = (
      max(observed_residuals) if observed_residuals else math.inf)
  host_safety_factor = (
      HOST_BUDGET_MS * 1000.0 / observed_host_residual_max_us
      if observed_host_residual_max_us > 0 else math.inf)

  q4_bytes = int(old_smoke.get("q4_stream_bytes_per_token", 0))
  q6_bytes = int(old_smoke.get("q6_stream_bytes_per_token", 0))
  f32_weight_bytes = int(old_smoke.get("f32_stream_bytes_per_token", 0))
  q4_gb_s = float(route_budget.get("q4_measured_gb_s", 0.0))
  q6_gb_s = float(route_budget.get("q6_measured_gb_s", 0.0))
  state_proxy_gb_s = min(
      float(repeat.get("effective_stream_gb_s", 0.0)),
      float(confirm.get("effective_stream_gb_s", 0.0)))
  q4_ms = q4_bytes / 1e6 / q4_gb_s
  q6_ms = q6_bytes / 1e6 / q6_gb_s
  f32_ms = f32_weight_bytes / 1e6 / strict_target_gb_s
  state_ms = (
      (resident_state_read_bytes + resident_state_write_bytes) /
      1e6 / state_proxy_gb_s)
  carrier_envelope_ms = q4_ms + q6_ms + f32_ms + state_ms
  fused_math_margin_ms = kernel_ms - carrier_envelope_ms

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("seq732_weight_census_anchor",
            schedule.get("required_checks_passed") is True and
            old_smoke.get("active_weight_bytes_per_token") ==
                ACTIVE_WEIGHT_BYTES),
      check("seq733_backend_measurement_anchor",
            level_zero.get("required_checks_passed") is True and
            repeat.get("payload_allocation_bytes") ==
                int(old_smoke.get("strict_stream_bytes_per_token", -1)) and
            confirm.get("payload_allocation_bytes") ==
                int(old_smoke.get("strict_stream_bytes_per_token", -2))),
      check("linear_state_shape_is_locked",
            LINEAR_RECURRENT_STATE_VALUES == 524_288 and
            LINEAR_CONV_STATE_VALUES == 24_576),
      check("state_complete_census_exceeds_seq732",
            strict_stream_bytes == 2_128_395_904 and
            strict_stream_bytes >
                int(old_smoke.get("strict_stream_bytes_per_token", 0)),
            resident_state_read_bytes=resident_state_read_bytes,
            resident_state_write_bytes=resident_state_write_bytes,
            strict_stream_bytes=strict_stream_bytes),
      check("measured_host_boundary_supports_100us_budget",
            finite(observed_host_residual_max_us) and
            observed_host_residual_max_us <= 12.5 and
            host_safety_factor >= 8.0,
            observed_host_residual_max_us=observed_host_residual_max_us,
            host_safety_factor=host_safety_factor),
      check("state_complete_kernel_rate_is_below_real_carriers",
            q4_gb_s >= kernel_window_gb_s and
            q6_gb_s >= kernel_window_gb_s,
            kernel_window_gb_s=kernel_window_gb_s,
            q4_gb_s=q4_gb_s, q6_gb_s=q6_gb_s),
      check("state_aware_carrier_envelope_leaves_fused_math_budget",
            finite(carrier_envelope_ms) and fused_math_margin_ms >= 1.0,
            carrier_envelope_ms=carrier_envelope_ms,
            fused_math_margin_ms=fused_math_margin_ms),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "inputs": {
          "schedule_gate": str(SCHEDULE.relative_to(ROOT)),
          "level_zero_gate": str(LEVEL_ZERO.relative_to(ROOT)),
          "route_gate": str(ROUTE.relative_to(ROOT)),
      },
      "state_census": {
          "linear_layer_count": LINEAR_LAYER_COUNT,
          "linear_recurrent_state_values_per_layer":
              LINEAR_RECURRENT_STATE_VALUES,
          "linear_conv_state_values_per_layer": LINEAR_CONV_STATE_VALUES,
          "recurrent_read_bytes": recurrent_read_bytes,
          "recurrent_write_bytes": recurrent_write_bytes,
          "conv_read_bytes": conv_read_bytes,
          "conv_write_bytes": conv_write_bytes,
          "kv_history_read_bytes": KV_HISTORY_READ_BYTES,
          "kv_append_write_bytes": KV_APPEND_WRITE_BYTES,
          "resident_state_read_bytes_per_token": resident_state_read_bytes,
          "resident_state_write_bytes_per_token": resident_state_write_bytes,
          "active_weight_bytes_per_token": ACTIVE_WEIGHT_BYTES,
          "strict_stream_bytes_per_token": strict_stream_bytes,
      },
      "admission": {
          "product_floor_tokens_s": product_floor,
          "full_token_wall_ms_max": wall_ms,
          "host_boundary_ms_max": HOST_BUDGET_MS,
          "kernel_schedule_ms_max": kernel_ms,
          "strict_stream_bandwidth_gb_s_min": strict_target_gb_s,
          "kernel_window_stream_bandwidth_gb_s_min": kernel_window_gb_s,
          "observed_host_residual_max_us": observed_host_residual_max_us,
          "host_safety_factor": host_safety_factor,
      },
      "carrier_projection": {
          "q4_ms": q4_ms, "q6_ms": q6_ms, "f32_weight_ms": f32_ms,
          "resident_state_ms": state_ms,
          "state_proxy_gb_s": state_proxy_gb_s,
          "carrier_envelope_ms": carrier_envelope_ms,
          "fused_math_margin_ms": fused_math_margin_ms,
      },
      "checks": checks, "required_checks_passed": passed,
      "disposition": (
          "revise_schedule_census_and_backend_admission"
          if passed else "reject_state_complete_schedule_budget"),
      "product_promotion_ready": False, "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False, "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "artifact": str(out),
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "git": state, "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  with (out / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in [
        ("strict_stream_bytes_per_token", strict_stream_bytes),
        ("kernel_window_stream_bandwidth_gb_s_min", kernel_window_gb_s),
        ("observed_host_residual_max_us", observed_host_residual_max_us),
        ("carrier_envelope_ms", carrier_envelope_ms),
        ("fused_math_margin_ms", fused_math_margin_ms),
    ]:
      fh.write(json.dumps({
          "metric": metric, "phase": "state_budget", "value": value,
      }, sort_keys=True) + "\n")
  (out / "summary.md").write_text("\n".join([
      "# Packed token state-complete budget gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- active weight bytes: `{ACTIVE_WEIGHT_BYTES}`",
      f"- resident state read / write bytes: `"
      f"{resident_state_read_bytes} / {resident_state_write_bytes}`",
      f"- corrected strict bytes/token: `{strict_stream_bytes}`",
      f"- wall / kernel / host budget: `"
      f"{wall_ms:.3f} / {kernel_ms:.3f} / {HOST_BUDGET_MS:.3f} ms`",
      f"- kernel-window stream rate: `{kernel_window_gb_s:.3f} GB/s`",
      f"- observed maximum seq733 host residual: `"
      f"{observed_host_residual_max_us:.3f} us`",
      f"- state-aware carrier envelope: `{carrier_envelope_ms:.3f} ms`",
      f"- remaining fused-math margin: `{fused_math_margin_ms:.3f} ms`", "",
      "Seq732 omitted linear recurrent and convolution state traffic. This gate "
      "corrects the source contract and reallocates the unused host reserve; it "
      "does not execute model math or claim product speed.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out), "pass": passed,
      "strict_stream_bytes": strict_stream_bytes,
      "kernel_ms": kernel_ms,
      "kernel_window_gb_s": kernel_window_gb_s,
      "fused_math_margin_ms": fused_math_margin_ms,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
