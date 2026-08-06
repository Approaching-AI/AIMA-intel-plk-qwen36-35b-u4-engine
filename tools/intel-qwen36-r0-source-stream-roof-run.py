#!/usr/bin/env python3
"""Run the current-target native GGUF source-stream roof probe."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_PROBE = (
    "/home/intel/intel-box-run/native-source-stream-roof-20260615T121249Z/"
    "build/native-intel-box-qwen36-engine/ibx-source-stream-probe"
)
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-run"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_TENSORS = (
    "blk.0.attn_gate.weight",
    "blk.0.attn_qkv.weight",
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--probe", default=DEFAULT_PROBE)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--iterations", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--target-gb-s", type=float, default=115.0)
  parser.add_argument("--local-size", type=int, default=128)
  parser.add_argument("--preferred-global-work-items", type=int, default=32768)
  parser.add_argument("--max-global-work-items", type=int, default=1048576)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument(
      "--tensor",
      action="append",
      default=None,
      help="Tensor name to stream. May be repeated. Defaults to prior route-gate tensors.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-source-stream-roof-<UTC>.",
  )
  return parser.parse_args()


def run(
    cmd: list[str],
    *,
    timeout_s: int,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_command: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_from(host: str, remote_path: str, local_path: Path, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_from(host, remote_path, local_path, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def read_json(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def parse_metrics(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  rows = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      rows.append(json.loads(line))
  return rows


def source_stream_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [row for row in metrics if row.get("phase") == "opencl_source_stream"]


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_remote_command(args: argparse.Namespace, remote_out: str) -> str:
  tensors = args.tensor if args.tensor else list(DEFAULT_TENSORS)
  pieces = [
      shlex.quote(args.probe),
      "--model",
      shlex.quote(args.model),
      "--out",
      shlex.quote(remote_out),
      "--iterations",
      str(args.iterations),
      "--warmup",
      str(args.warmup),
      "--target-gb-s",
      str(args.target_gb_s),
      "--local-size",
      str(args.local_size),
      "--preferred-global-work-items",
      str(args.preferred_global_work_items),
      "--max-global-work-items",
      str(args.max_global_work_items),
  ]
  for tensor in tensors:
    pieces.extend(["--tensor", shlex.quote(tensor)])
  command = " ".join(pieces)
  if args.env_script:
    return f"source {shlex.quote(args.env_script)} && {command}"
  return command


def build_summary(payload: dict[str, Any]) -> str:
  audit = payload["audit"]
  lines = [
      "# R0 Source-Stream Roof Run",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- route label: `{audit.get('route_label')}`",
      f"- target GB/s: {audit.get('target_gb_s')}",
      f"- min source GB/s: {audit.get('min_source_gb_s')}",
      f"- max source GB/s: {audit.get('max_source_gb_s')}",
      f"- max target ratio: {audit.get('max_target_ratio')}",
      f"- required checks passed: `{str(audit.get('required_checks_passed')).lower()}`",
      "",
      "This is a current-target source-stream roof probe for route feasibility.",
      "It is not a full denominator row and does not close R0 by itself.",
      "",
  ]
  return "\n".join(lines)


def audit_remote_output(remote_output: Path, *, target_gb_s: float) -> dict[str, Any]:
  manifest = read_json(remote_output / "manifest.json")
  correctness = read_json(remote_output / "correctness.json")
  metrics = parse_metrics(remote_output / "metrics.jsonl")
  rows = source_stream_rows(metrics)
  speeds = []
  for row in rows:
    value = row.get("effective_source_gb_s")
    if value is None:
      value = row.get("source_effective_gb_s")
    if finite(value):
      speeds.append(float(value))
  ratios = [speed / target_gb_s for speed in speeds]
  route_label = None
  if correctness:
    route_label = correctness.get("route_label")
  if route_label is None and manifest:
    route_label = manifest.get("route_label")
  required_checks = False
  if correctness:
    required_checks = bool(correctness.get("required_checks_passed"))
  return {
      "artifact_path": str(remote_output),
      "classification": "current_target_native_source_stream_roof",
      "gate": "native_gguf_q6_source_stream_roof_gate",
      "gate_closed": False,
      "max_source_gb_s": max(speeds) if speeds else None,
      "max_target_ratio": max(ratios) if ratios else None,
      "min_source_gb_s": min(speeds) if speeds else None,
      "min_target_ratio": min(ratios) if ratios else None,
      "required_checks_passed": required_checks,
      "route_label": route_label or "missing",
      "source_stream_rows": len(rows),
      "target_gb_s": target_gb_s,
      "usable_for": [
          "current-target route feasibility",
          "raw low-bit source bandwidth roof seed",
      ],
  }


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-source-stream-roof-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  stamp = created_at.replace("-", "").replace(":", "")
  remote_out = f"{args.remote_root.rstrip('/')}/source-stream-roof-{stamp}"

  preflight = run_target(
      args.host,
      (
          f"test -x {shlex.quote(args.probe)} && "
          f"test -f {shlex.quote(args.model)} && "
          f"test -f {shlex.quote(args.env_script)} && "
          "printf 'preflight_ok\\n'"
      ),
      timeout_s=30,
  )
  (raw_dir / "preflight.stdout").write_text(preflight["stdout"], encoding="utf-8")
  (raw_dir / "preflight.stderr").write_text(preflight["stderr"], encoding="utf-8")

  remote_command = build_remote_command(args, remote_out)
  result = {
      "command": remote_command,
      "returncode": 127,
      "stdout": "",
      "stderr": "preflight failed",
      "timed_out": False,
  }
  if preflight["returncode"] == 0:
    bash_command = (
        f"rm -rf {shlex.quote(remote_out)} && "
        f"mkdir -p {shlex.quote(remote_out)} && "
        f"{remote_command}"
    )
    result = run_target(
        args.host,
        f"bash -lc {shlex.quote(bash_command)}",
        timeout_s=args.timeout_s,
    )
  (raw_dir / "source-stream.stdout").write_text(result["stdout"], encoding="utf-8")
  (raw_dir / "source-stream.stderr").write_text(result["stderr"], encoding="utf-8")

  remote_output = out_dir / "remote-output" / "source-stream-roof"
  fetch = {
      "command": [],
      "returncode": 127,
      "stdout": "",
      "stderr": "probe did not run",
      "timed_out": False,
  }
  if preflight["returncode"] == 0:
    fetch = copy_from(args.host, remote_out.rstrip("/") + "/", remote_output, timeout_s=120)
  (raw_dir / "fetch.stdout").write_text(fetch["stdout"], encoding="utf-8")
  (raw_dir / "fetch.stderr").write_text(fetch["stderr"], encoding="utf-8")

  audit = audit_remote_output(remote_output, target_gb_s=args.target_gb_s)
  if result["returncode"] != 0 and audit["route_label"] == "missing":
    audit["route_label"] = "rejected"
    audit["failure_class"] = "probe_execution_failed"

  payload = {
      "audit": audit,
      "created_at": created_at,
      "host": args.host,
      "model": {
          "gguf_path": args.model,
      },
      "probe": {
          "env_script": args.env_script,
          "path": args.probe,
          "remote_command": remote_command,
          "remote_out": remote_out,
      },
      "r0_performance_gate_closed": False,
      "raw": {
          "fetch": fetch,
          "preflight": preflight,
          "run": result,
      },
      "schema_version": "intel-qwen36-r0-source-stream-roof-run-v0",
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "route_label": audit.get("route_label"),
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-source-stream-roof-run.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "audit.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": [
          {
              "fetch_returncode": fetch["returncode"],
              "preflight_returncode": preflight["returncode"],
              "probe_returncode": result["returncode"],
              "remote_metrics_present": bool(source_stream_rows(parse_metrics(remote_output / "metrics.jsonl"))),
              "required_checks_passed": audit.get("required_checks_passed"),
              "route_label": audit.get("route_label"),
          }
      ],
      "gate": "current_target_native_source_stream_roof",
      "required_checks_passed": (
          preflight["returncode"] == 0
          and result["returncode"] == 0
          and fetch["returncode"] == 0
          and bool(audit.get("required_checks_passed"))
      ),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "notes": "source-stream roof probe only",
      "route_label": audit.get("route_label"),
  })
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"source-stream roof output: {out_dir}")
  if result["returncode"] != 0 or fetch["returncode"] != 0:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
