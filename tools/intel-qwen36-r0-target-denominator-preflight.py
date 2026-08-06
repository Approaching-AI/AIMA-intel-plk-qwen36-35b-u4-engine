#!/usr/bin/env python3
"""Read-only preflight for current-target OpenVINO denominator reruns."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
OPENVINO_MODEL = "/home/intel/Qwen3.6-35B-A3B-ov"
OPENVINO_DIR = "/home/intel/ov"
OPENVINO_BENCH = "/home/intel/ov/benchmark_vlm_new.py"
PROMPT_DIR = "/home/intel/ov/prompts"
REQUIRED_DENOMINATOR_BUCKETS = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    102400,
    131072,
    262144,
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-target-denominator-preflight-<UTC>.",
  )
  return parser.parse_args()


def run_target(host: str, command: str, timeout_s: int = 30) -> dict[str, Any]:
  return iq36_local.run_target(host, command, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def parse_path_checks(stdout: str) -> dict[str, bool]:
  checks = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    checks[key.strip()] = value.strip() == "1"
  return checks


def parse_prompt_inventory(stdout: str) -> list[dict[str, Any]]:
  rows = []
  pattern = re.compile(r"^(.+?)\s+([0-9]+)$")
  for line in stdout.splitlines():
    match = pattern.match(line.strip())
    if not match:
      continue
    name = match.group(1)
    size_bytes = int(match.group(2))
    bucket = None
    repeat = None
    bucket_match = re.match(r"prompt_[0-9]+_([0-9]+)Kin_([0-9]+)out_r([0-9]+)\.txt$", name)
    if bucket_match:
      bucket_k = int(bucket_match.group(1))
      bucket = bucket_k * 1024
      repeat = int(bucket_match.group(3))
    rows.append(
        {
            "bucket": bucket,
            "name": name,
            "repeat": repeat,
            "size_bytes": size_bytes,
        }
    )
  return rows


def prompt_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  buckets = sorted({row["bucket"] for row in rows if isinstance(row.get("bucket"), int)})
  by_bucket: dict[str, int] = {}
  for bucket in buckets:
    by_bucket[str(bucket)] = sum(1 for row in rows if row.get("bucket") == bucket)
  missing = sorted(set(REQUIRED_DENOMINATOR_BUCKETS) - set(buckets))
  return {
      "bucket_counts": by_bucket,
      "buckets_present": buckets,
      "missing_required_buckets": missing,
      "required_buckets": REQUIRED_DENOMINATOR_BUCKETS,
      "total_prompt_files": len(rows),
  }


def build_summary(preflight: dict[str, Any]) -> str:
  prompt_summary = preflight["prompt_summary"]
  checks = preflight["path_checks"]
  missing = prompt_summary["missing_required_buckets"]
  return "\n".join(
      [
          "# R0 target denominator preflight",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- host: `{preflight['host']}`",
          f"- OpenVINO model present: `{str(checks.get('ov_model')).lower()}`",
          f"- OpenVINO benchmark present: `{str(checks.get('ov_bench')).lower()}`",
          f"- prompt files: {prompt_summary['total_prompt_files']}",
          f"- buckets present: {prompt_summary['buckets_present']}",
          f"- missing required buckets: {missing}",
          f"- denominator ready: `{str(preflight['denominator_ready']).lower()}`",
          "",
          "This is a read-only preflight. It does not run the denominator",
          "benchmark or close R0.",
          "",
      ]
  )


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-target-denominator-preflight-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  commands = {
      "host_facts": run_target(
          args.host,
          "hostname; date -Is; uname -a",
          timeout_s=20,
      ),
      "path_checks": run_target(
          args.host,
          "printf 'ov_model='; test -d /home/intel/Qwen3.6-35B-A3B-ov && echo 1 || echo 0; "
          "printf 'ov_bench='; test -f /home/intel/ov/benchmark_vlm_new.py && echo 1 || echo 0; "
          "printf 'ov_env='; test -d /home/intel/ov/openvino_env && echo 1 || echo 0; "
          "printf 'prompt_dir='; test -d /home/intel/ov/prompts && echo 1 || echo 0",
          timeout_s=20,
      ),
      "openvino_import": run_target(
          args.host,
          "cd /home/intel/ov && . openvino_env/bin/activate && "
          "python - <<'PY'\n"
          "import openvino_genai as ov\n"
          "print(getattr(ov, '__version__', 'unknown'))\n"
          "PY",
          timeout_s=60,
      ),
      "prompt_inventory": run_target(
          args.host,
          "find /home/intel/ov/prompts -maxdepth 1 -type f -printf '%f %s\\n' 2>/dev/null | sort",
          timeout_s=30,
      ),
      "opencl_devices": run_target(
          args.host,
          "timeout 15 clinfo -l 2>/dev/null || true",
          timeout_s=25,
      ),
      "recent_runs": run_target(
          args.host,
          "ls -dt /home/intel/intel-box-run/* 2>/dev/null | head -n 20",
          timeout_s=20,
      ),
  }
  path_checks = parse_path_checks(commands["path_checks"]["stdout"])
  prompt_rows = parse_prompt_inventory(commands["prompt_inventory"]["stdout"])
  prompt_summary = prompt_bucket_summary(prompt_rows)
  required_paths_ok = all(
      path_checks.get(name) is True
      for name in ("ov_model", "ov_bench", "ov_env", "prompt_dir")
  )
  openvino_import_ok = commands["openvino_import"]["returncode"] == 0
  denominator_ready = (
      required_paths_ok
      and openvino_import_ok
      and not prompt_summary["missing_required_buckets"]
  )
  preflight = {
      "commands": commands,
      "created_at": created_at,
      "denominator_ready": denominator_ready,
      "host": args.host,
      "model": {
          "openvino_path": OPENVINO_MODEL,
          "prompt_dir": PROMPT_DIR,
          "benchmark_script": OPENVINO_BENCH,
      },
      "next_required_actions": [
          "materialize or install a 262144-token OpenVINO prompt on the target",
          "run current-target OpenVINO denominator with 262144 coverage",
          "record raw stdout/stderr and parsed metrics under output/",
      ],
      "openvino_import_ok": openvino_import_ok,
      "path_checks": path_checks,
      "prompt_inventory": prompt_rows,
      "prompt_summary": prompt_summary,
      "r0_denominator_gate_closed": False,
      "schema_version": "intel-qwen36-r0-target-denominator-preflight-v0",
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "preflight.json", preflight)
  (out_dir / "summary.md").write_text(build_summary(preflight), encoding="utf-8")
  print(f"target denominator preflight output: {out_dir}")


if __name__ == "__main__":
  main()
