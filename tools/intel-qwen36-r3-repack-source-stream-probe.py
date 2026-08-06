#!/usr/bin/env python3
"""R3 offline Q4_K/Q6_K repack and source-stream probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r3-repack-source-stream-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_LANES = ["attn_qkv.weight:Q4_K", "ffn_gate_up_exps.weight:Q4_K"]
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/layout_repack_probe.cpp", "tests/layout_repack_probe.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--lane", action="append", default=None)
  parser.add_argument("--iterations", type=int, default=5)
  parser.add_argument("--max-tensors-per-lane", type=int, default=1)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def source_stage(host: str, remote_dir: str, timeout_s: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  mkdir = iq36_local.run_target(
      host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build")
      ),
      timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(
          iq36_local.copy_to(host, ROOT / local, f"{remote_dir}/{remote}", timeout_s)
      )
  return mkdir, transfers


def build_command(remote_dir: str, env_script: str) -> str:
  return " && ".join([
      f"source {shlex.quote(env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/layout_repack_probe.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-layout-repack-probe')}",
  ])


def probe_command(
    remote_dir: str,
    model: str,
    lanes: list[str],
    iterations: int,
    max_tensors_per_lane: int,
) -> str:
  parts = [
      shlex.quote(remote_dir + "/build/iq36-layout-repack-probe"),
      shlex.quote(model),
      "--iterations",
      shlex.quote(str(iterations)),
      "--max-tensors-per-lane",
      shlex.quote(str(max_tensors_per_lane)),
  ]
  for lane in lanes:
    parts += ["--lane", shlex.quote(lane)]
  return " ".join(parts)


def stream_gb_s(row: dict[str, Any], key: str) -> float:
  value = row.get(key, {})
  if not isinstance(value, dict):
    return 0.0
  metric = value.get("gb_s")
  return float(metric) if isinstance(metric, (int, float)) else 0.0


def summarize_probe(payload: dict[str, Any]) -> str:
  probe = payload.get("probe", {})
  rows = probe.get("rows", [])
  aggregate = probe.get("aggregate", {})
  lines = [
      "# R3 Repack Source Stream Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- selected tensors: `{aggregate.get('selected_tensor_count', 0)}`",
      f"- aggregate repack overhead: `{aggregate.get('repacked_overhead_ratio')}`",
      "",
      "| lane | tensor | layout | raw MB | repacked MB | overhead | repack GB/s | raw stream GB/s | repacked stream GB/s | quant-only GB/s |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  if isinstance(rows, list):
    for row in rows:
      if not isinstance(row, dict):
        continue
      lines.append(
          "| "
          + " | ".join([
              f"`{row.get('selected_by_lane')}`",
              f"`{row.get('name')}`",
              f"`{row.get('layout')}`",
              f"{float(row.get('raw_bytes', 0)) / 1e6:.1f}",
              f"{float(row.get('repacked_bytes', 0)) / 1e6:.1f}",
              f"{float(row.get('repacked_overhead_ratio', 0)):.6f}",
              f"{float(row.get('repack_gb_s', 0)):.3f}",
              f"{stream_gb_s(row, 'raw_stream'):.3f}",
              f"{stream_gb_s(row, 'repacked_stream'):.3f}",
              f"{stream_gb_s(row, 'repacked_quant_only_stream'):.3f}",
          ])
          + " |"
      )
  lines += [
      "",
      "Conclusion: this artifact measures offline layout conversion and source-stream",
      "behavior only. It is not a model throughput or speedup claim. If accepted,",
      "the next R3 step is to wire the plane layout into the CPU dense matvec hot path",
      "for the measured lanes.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.iterations <= 0:
    raise SystemExit("--iterations must be positive")
  if args.max_tensors_per_lane <= 0:
    raise SystemExit("--max-tensors-per-lane must be positive")
  lanes = args.lane or DEFAULT_LANES
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r3-repack-source-stream-probe-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/r3-repack-source-stream-probe-{stamp}"

  mkdir, transfers = source_stage(args.host, remote_dir, args.timeout_s)
  build_cmd = build_command(remote_dir, args.env_script)
  build = (
      iq36_local.run_target(args.host, f"bash -lc {shlex.quote(build_cmd)}", args.timeout_s)
      if mkdir.get("returncode") == 0 and all(item.get("returncode") == 0 for item in transfers)
      else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_cmd = probe_command(
      remote_dir,
      args.model,
      lanes,
      args.iterations,
      args.max_tensors_per_lane,
  )
  target_run = (
      iq36_local.run_target(args.host, run_cmd, args.timeout_s)
      if build.get("returncode") == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  probe: dict[str, Any] = {}
  parse_error = None
  try:
    probe = json.loads(target_run.get("stdout", "") or "{}")
  except json.JSONDecodeError as exc:
    parse_error = str(exc)

  rows = probe.get("rows", []) if isinstance(probe, dict) else []
  if not isinstance(rows, list):
    rows = []
  selected_lanes = {
      row.get("selected_by_lane")
      for row in rows
      if isinstance(row, dict) and isinstance(row.get("selected_by_lane"), str)
  }
  q4_rows = [row for row in rows if isinstance(row, dict) and row.get("type_name") == "Q4_K"]
  stream_rows = [row for row in rows if isinstance(row, dict)]
  checks = [
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers),
      },
      {"name": "target_repack_probe_built", "pass": build.get("returncode") == 0},
      {"name": "target_repack_probe_ran", "pass": target_run.get("returncode") == 0},
      {"name": "target_repack_probe_stdout_parsed", "pass": parse_error is None and bool(probe)},
      {
          "name": "requested_hot_lanes_selected",
          "pass": all(lane in selected_lanes for lane in lanes),
      },
      {
          "name": "q4_repack_overhead_within_plane_v0_bound",
          "pass": bool(q4_rows)
          and all(float(row.get("repacked_overhead_ratio", 0)) <= 1.03 for row in q4_rows),
      },
      {
          "name": "stream_bandwidth_positive",
          "pass": bool(stream_rows)
          and all(
              stream_gb_s(row, "raw_stream") > 0.0
              and stream_gb_s(row, "repacked_stream") > 0.0
              and stream_gb_s(row, "repacked_quant_only_stream") > 0.0
              for row in stream_rows
          ),
      },
      {
          "name": "repack_cost_recorded",
          "pass": bool(stream_rows)
          and all(float(row.get("repack_ns", 0)) > 0.0 for row in stream_rows),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "created_at": created_at,
      "host": args.host,
      "iterations": args.iterations,
      "lanes": lanes,
      "max_tensors_per_lane": args.max_tensors_per_lane,
      "model_path": args.model,
      "parse_error": parse_error,
      "probe": probe,
      "remote_dir": remote_dir,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  stdout = target_run.get("stdout", "") or ""
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "lanes": lanes,
      "model_path": args.model,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r3-repack-source-stream-probe.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "build_command": build_cmd,
      "mkdir": mkdir,
      "run_command": run_cmd,
      "source_files": SOURCE_FILES,
      "source_transfers": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "target-run.json", {
      "cmd": target_run.get("cmd"),
      "returncode": target_run.get("returncode"),
      "stderr": target_run.get("stderr"),
      "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
      "stdout_size_bytes": len(stdout.encode("utf-8")),
  })
  (out_dir / "probe-stdout.json").write_text(stdout, encoding="utf-8")
  write_json(out_dir / "probe.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r3_repack_source_stream_probe",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "rows.jsonl", stream_rows)
  aggregate = probe.get("aggregate", {}) if isinstance(probe, dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r3_repack_source_stream_probe",
      [
          ("selected_tensor_count", aggregate.get("selected_tensor_count")),
          ("raw_bytes", aggregate.get("raw_bytes")),
          ("repacked_bytes", aggregate.get("repacked_bytes")),
          ("repacked_overhead_ratio", aggregate.get("repacked_overhead_ratio")),
          ("raw_stream_gb_s_weighted", aggregate.get("raw_stream_gb_s_weighted")),
          ("repacked_stream_gb_s_weighted", aggregate.get("repacked_stream_gb_s_weighted")),
          ("repacked_quant_only_stream_gb_s_weighted", aggregate.get("repacked_quant_only_stream_gb_s_weighted")),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(summarize_probe(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
