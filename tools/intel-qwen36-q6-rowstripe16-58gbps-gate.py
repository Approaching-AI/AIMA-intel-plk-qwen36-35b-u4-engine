#!/usr/bin/env python3
"""Prove the fixed real-tensor Q6 rowstripe carrier clears 58 GB/s."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-q6-rowstripe16-58gbps-gate-v0"
PROBE_TOOL = ROOT / "tools/intel-qwen36-gpu-q4x8-qmatvec-probe.py"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_TENSOR = "blk.7.ffn_down_exps.weight"
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
EXPECTED_COLS = 512
EXPECTED_ROWS = 524_288
EXPECTED_BLOCKS_PER_ROW = 2
EXPECTED_RAW_BYTES = 220_200_960
MIN_GB_S = 58.0
MIN_REPEAT = 9


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--repeat", type=int, default=MIN_REPEAT)
  parser.add_argument("--timeout-s", type=int, default=1_200)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.repeat < MIN_REPEAT:
    parser.error(f"--repeat must be at least {MIN_REPEAT}")
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/q6-rowstripe16-58gbps-gate-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "timed_out": True,
    }


def variant(probe: dict[str, Any], name: str) -> dict[str, Any]:
  variants = probe.get("gpu_variants", [])
  if not isinstance(variants, list):
    return {}
  for item in variants:
    if isinstance(item, dict) and item.get("name") == name:
      return item
  return {}


def number(value: Any) -> float | None:
  return float(value) if isinstance(value, (int, float)) else None


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def build_summary(result: dict[str, Any]) -> str:
  row = result["rowstripe_variant"]
  raw = result["raw_variant"]
  return "\n".join([
      "# Exact Q6 rowstripe16 58 GB/s gate",
      "",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      f"- commit: `{result['git']['commit']}`",
      f"- tensor / bytes: `{result['tensor']}` / `{EXPECTED_RAW_BYTES}`",
      f"- raw baseline: `{raw.get('gpu_kernel_min_us')} us` / "
      f"`{raw.get('gpu_effective_packed_gb_s')} GB/s`",
      f"- rowstripe16: `{row.get('gpu_kernel_min_us')} us` / "
      f"`{row.get('gpu_effective_packed_gb_s')} GB/s`",
      f"- promotion gate: `>={MIN_GB_S} GB/s`",
      f"- rowstripe relL2 / cosine: "
      f"`{(row.get('comparison_vs_cpu_packed') or {}).get('rel_l2')}` / "
      f"`{(row.get('comparison_vs_cpu_packed') or {}).get('cosine')}`",
      "",
      "This is a real full-tensor exact-component carrier gate, not a token or",
      "whole-model speed claim.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir
  out_dir.mkdir(parents=True, exist_ok=False)
  child_dir = out_dir / "qmatvec"
  state = git_state()
  command = [
      sys.executable,
      str(PROBE_TOOL),
      "--target", "local",
      "--model", str(args.model),
      "--tensor", args.tensor,
      "--env-script", str(args.env_script),
      "--repeat", str(args.repeat),
      "--timeout-s", str(args.timeout_s),
      "--out-dir", str(child_dir),
  ]
  probe_run = run(command, args.timeout_s)
  probe_path = child_dir / "probe-result.json"
  probe = load_json(probe_path) if probe_path.is_file() else {}
  raw_variant = variant(probe, "q6_raw_row")
  rowstripe_variant = variant(probe, "q6_rowstripe16")
  raw_comparison = raw_variant.get("comparison_vs_cpu_packed", {})
  rowstripe_comparison = rowstripe_variant.get(
      "comparison_vs_cpu_packed", {})
  raw_gb_s = number(raw_variant.get("gpu_effective_packed_gb_s"))
  rowstripe_gb_s = number(
      rowstripe_variant.get("gpu_effective_packed_gb_s"))
  rowstripe_speedup = (
      rowstripe_gb_s / raw_gb_s
      if rowstripe_gb_s is not None and raw_gb_s is not None and raw_gb_s > 0
      else None)

  checks = [
      check("clean_committed_source", state["dirty"] is False, git=state),
      check("probe_process_succeeded",
            probe_run["returncode"] == 0 and
            probe.get("required_checks_passed") is True,
            returncode=probe_run["returncode"]),
      check("locked_model_and_tensor",
            args.model == DEFAULT_MODEL and args.tensor == DEFAULT_TENSOR,
            model=str(args.model), tensor=args.tensor),
      check("locked_real_full_tensor_shape",
            probe.get("tensor_type") == "Q6_K" and
            probe.get("cols") == EXPECTED_COLS and
            probe.get("rows") == EXPECTED_ROWS and
            probe.get("blocks_per_row") == EXPECTED_BLOCKS_PER_ROW and
            probe.get("raw_bytes") == EXPECTED_RAW_BYTES,
            cols=probe.get("cols"), rows=probe.get("rows"),
            blocks_per_row=probe.get("blocks_per_row"),
            raw_bytes=probe.get("raw_bytes")),
      check("arc_b390_selected", "B390" in str(probe.get("device_name", "")),
            device_name=probe.get("device_name")),
      check("paired_repeat_depth", probe.get("repeat") == args.repeat and
            args.repeat >= MIN_REPEAT, repeat=probe.get("repeat")),
      check("raw_q6_reference_passed",
            raw_variant.get("passed") is True and
            isinstance(raw_comparison, dict) and
            raw_comparison.get("passed") is True,
            comparison=raw_comparison),
      check("rowstripe16_all_value_reference_passed",
            rowstripe_variant.get("passed") is True and
            isinstance(rowstripe_comparison, dict) and
            rowstripe_comparison.get("passed") is True,
            comparison=rowstripe_comparison),
      check("rowstripe16_58_gb_s_gate",
            rowstripe_gb_s is not None and rowstripe_gb_s >= MIN_GB_S,
            measured_gb_s=rowstripe_gb_s, required_gb_s=MIN_GB_S),
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  disposition = (
      "accept_exact_q6_rowstripe16_58gbps_carrier"
      if required_checks_passed
      else "reject_exact_q6_rowstripe16_58gbps_carrier")
  result = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": iso_now(),
      "git": state,
      "model": str(args.model),
      "tensor": args.tensor,
      "repeat": args.repeat,
      "required_gb_s": MIN_GB_S,
      "raw_variant": raw_variant,
      "rowstripe_variant": rowstripe_variant,
      "rowstripe_speedup_vs_raw": rowstripe_speedup,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "child_artifact": str(child_dir),
      "speedup_claims_allowed": False,
  }
  write_json(out_dir / "gate.json", result)
  write_json(out_dir / "correctness.json", {
      "schema_version": SCHEMA_VERSION,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": result["created_at"],
      "tool": str(PROBE_TOOL.relative_to(ROOT)),
      "gate_tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out_dir),
      "git": state,
      "model": str(args.model),
      "tensor": args.tensor,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "speedup_claims_allowed": False,
  })
  iq36_local.write_metric(
      out_dir / "metrics.jsonl", "q6_rowstripe16_58gbps_gate", [
          ("raw_q6_min_us", raw_variant.get("gpu_kernel_min_us")),
          ("raw_q6_gb_s", raw_gb_s),
          ("rowstripe16_min_us", rowstripe_variant.get("gpu_kernel_min_us")),
          ("rowstripe16_gb_s", rowstripe_gb_s),
          ("rowstripe16_speedup_vs_raw", rowstripe_speedup),
          ("required_gb_s", MIN_GB_S),
          ("required_checks_passed", required_checks_passed),
      ])
  (out_dir / "summary.md").write_text(
      build_summary(result), encoding="utf-8")
  write_json(out_dir / "raw-probe-run.json", probe_run)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
