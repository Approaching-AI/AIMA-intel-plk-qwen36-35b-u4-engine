#!/usr/bin/env python3
"""R3 backend + quantized-layout audit for the memory-access route."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r3-backend-layout-audit-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_REJECT_TABLE = ROOT / "output/bandwidth-roofline-reject-20260628T161001Z/reject-table.json"
DEFAULT_R2_BIND = ROOT / "output/r2-floor-bind-20260629T052941Z/floor-bind.json"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/layout_inventory.cpp", "tests/layout_inventory.cpp"),
]
GPU_RUNTIME_PATTERNS = (
    "CL/cl.h",
    "clCreate",
    "clEnqueue",
    "cl_kernel",
    "vulkan/vulkan.h",
    "vkCreate",
    "sycl/",
    "ze_api.h",
    "zeCommand",
)


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--reject-table", type=Path, default=DEFAULT_REJECT_TABLE)
  parser.add_argument("--r2-bind", type=Path, default=DEFAULT_R2_BIND)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
  return str(path.resolve().relative_to(ROOT))


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
      f"{shlex.quote(remote_dir + '/tests/layout_inventory.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-layout-inventory')}",
  ])


def scan_native_backend_sources() -> dict[str, Any]:
  paths = [
      ROOT / "engine/include/intel_qwen36/gguf_loader.hpp",
      ROOT / "engine/src/gguf_loader.cpp",
      ROOT / "engine/tests/native_candidate_jsonl.cpp",
      ROOT / "engine/tests/layout_inventory.cpp",
  ]
  hits: list[dict[str, Any]] = []
  for path in paths:
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in GPU_RUNTIME_PATTERNS:
      if pattern in text:
        hits.append({"path": rel(path), "pattern": pattern})
  return {
      "gpu_runtime_patterns": list(GPU_RUNTIME_PATTERNS),
      "scanned_files": [rel(path) for path in paths],
      "runtime_hits": hits,
      "native_backend": "cpu_threads",
      "native_backend_confirmed": len(hits) == 0,
  }


def lane_inventory(inventory: dict[str, Any], reject_table: dict[str, Any]) -> list[dict[str, Any]]:
  by_suffix_type: dict[tuple[str, str], dict[str, Any]] = {}
  for tensor in inventory.get("quantized_tensors", []):
    suffix = tensor.get("suffix") or tensor.get("name")
    type_name = tensor.get("type_name")
    if not isinstance(suffix, str) or not isinstance(type_name, str):
      continue
    key = (suffix, type_name)
    row = by_suffix_type.setdefault(
        key,
        {
            "inventory_bytes": 0,
            "inventory_tensor_count": 0,
            "sample_tensors": [],
            "suffix": suffix,
            "type_name": type_name,
        },
    )
    row["inventory_bytes"] += int(tensor.get("nbytes", 0))
    row["inventory_tensor_count"] += 1
    if len(row["sample_tensors"]) < 3:
      row["sample_tensors"].append(tensor.get("name"))

  rows: list[dict[str, Any]] = []
  for lane in reject_table.get("lanes", []):
    suffix = lane.get("lane")
    quant = lane.get("quant")
    if not isinstance(suffix, str) or not isinstance(quant, str):
      continue
    inv = by_suffix_type.get((suffix, quant), {})
    if quant not in ("Q4_K", "Q6_K"):
      continue
    rows.append({
        "bytes_per_call": lane.get("bytes_per_call"),
        "effective_gb_s": lane.get("effective_gb_s"),
        "inventory_bytes": inv.get("inventory_bytes", 0),
        "inventory_tensor_count": inv.get("inventory_tensor_count", 0),
        "lane": suffix,
        "profile_calls": lane.get("calls"),
        "quant": quant,
        "r0_qmatvec_ceiling_gb_s": lane.get("r0_qmatvec_ceiling_gb_s"),
        "recoverable_ns_to_qmatvec_ceiling": lane.get("recoverable_ns_to_qmatvec_ceiling"),
        "sample_tensors": inv.get("sample_tensors", []),
        "util_vs_qmatvec": lane.get("util_vs_qmatvec"),
        "verdict": lane.get("verdict"),
      })
  rows.sort(key=lambda row: int(row.get("recoverable_ns_to_qmatvec_ceiling") or 0), reverse=True)
  return rows


def build_summary(payload: dict[str, Any]) -> str:
  audit = payload["audit"]
  backend = audit["native_backend"]
  inv = audit["layout_inventory"]
  lines = [
      "# R3 Backend + Layout Audit",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- native backend: `{backend['native_backend']}`",
      f"- backend confirmed: `{str(backend['native_backend_confirmed']).lower()}`",
      f"- Q4_K tensors: `{inv['tensor_count_by_type'].get('Q4_K')}`",
      f"- Q6_K tensors: `{inv['tensor_count_by_type'].get('Q6_K')}`",
      f"- R2 denominator gate closed: `{str(audit['r2_denominator_gate_closed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| lane | quant | tensors | inventory MB | eff GB/s | util vs qmatvec | recover ns |",
      "|---|---|---:|---:|---:|---:|---:|",
  ]
  for row in audit["repack_candidate_lanes"][:10]:
    mb = row.get("inventory_bytes", 0) / 1e6
    lines.append(
        "| "
        + " | ".join([
            f"`{row['lane']}`",
            row["quant"],
            str(row.get("inventory_tensor_count")),
            f"{mb:.1f}",
            str(row.get("effective_gb_s")),
            str(row.get("util_vs_qmatvec")),
            str(row.get("recoverable_ns_to_qmatvec_ceiling")),
        ])
        + " |"
    )
  lines += [
      "",
      "Conclusion: the native candidate route is CPU-threaded and below the R0",
      "qmatvec bandwidth ceiling. R3 should start with an offline Q4_K/Q6_K",
      "repack/source-stream prototype, then wire that layout into the dense",
      "matvec hot lanes. This is not a speedup claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r3-backend-layout-audit-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/r3-backend-layout-audit-{stamp}"

  mkdir, transfers = source_stage(args.host, remote_dir, args.timeout_s)
  build_cmd = build_command(remote_dir, args.env_script)
  build = (
      iq36_local.run_target(args.host, f"bash -lc {shlex.quote(build_cmd)}", args.timeout_s)
      if mkdir.get("returncode") == 0 and all(item.get("returncode") == 0 for item in transfers)
      else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-layout-inventory"),
      shlex.quote(args.model),
  ])
  target_run = (
      iq36_local.run_target(args.host, run_command, args.timeout_s)
      if build.get("returncode") == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )
  inventory: dict[str, Any] = {}
  parse_error = None
  try:
    inventory = json.loads(target_run.get("stdout", "") or "{}")
  except json.JSONDecodeError as exc:
    parse_error = str(exc)

  reject_table = read_json(args.reject_table)
  r2_bind = read_json(args.r2_bind)
  backend = scan_native_backend_sources()
  lanes = lane_inventory(inventory, reject_table) if inventory else []
  quantized_bytes = {
      key: value
      for key, value in inventory.get("bytes_by_type", {}).items()
      if key in ("Q4_K", "Q6_K")
  }
  checks = [
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers),
      },
      {"name": "target_layout_inventory_built", "pass": build.get("returncode") == 0},
      {"name": "target_layout_inventory_ran", "pass": target_run.get("returncode") == 0},
      {"name": "target_layout_inventory_stdout_parsed", "pass": parse_error is None and bool(inventory)},
      {
          "name": "native_backend_cpu_threaded_no_gpu_runtime_in_engine",
          "pass": backend["native_backend_confirmed"],
      },
      {
          "name": "r2_denominator_gate_closed",
          "pass": r2_bind.get("r2_denominator_gate_closed") is True,
      },
      {
          "name": "q4_q6_inventory_present",
          "pass": inventory.get("tensor_count_by_type", {}).get("Q4_K", 0) > 0
          and inventory.get("tensor_count_by_type", {}).get("Q6_K", 0) > 0,
      },
      {
          "name": "hot_quant_lanes_mapped_to_inventory",
          "pass": len(lanes) >= 8 and all(row.get("inventory_tensor_count", 0) > 0 for row in lanes[:8]),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "audit": {
          "layout_inventory": inventory,
          "native_backend": backend,
          "quantized_bytes": quantized_bytes,
          "r2_denominator_gate_closed": r2_bind.get("r2_denominator_gate_closed"),
          "repack_candidate_lanes": lanes,
      },
      "created_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r3-backend-layout-audit.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "build_command": build_cmd,
      "mkdir": mkdir,
      "source_files": SOURCE_FILES,
      "source_transfers": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "target-run.json", {
      "cmd": target_run.get("cmd"),
      "returncode": target_run.get("returncode"),
      "stderr": target_run.get("stderr"),
      "stdout_sha256": __import__("hashlib").sha256(
          (target_run.get("stdout", "") or "").encode("utf-8")
      ).hexdigest(),
      "stdout_size_bytes": len((target_run.get("stdout", "") or "").encode("utf-8")),
  })
  (out_dir / "layout-inventory-stdout.json").write_text(
      target_run.get("stdout", "") or "",
      encoding="utf-8",
  )
  write_json(out_dir / "audit.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r3_backend_layout_audit",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "repack-candidate-lanes.jsonl", lanes)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r3_backend_layout_audit",
      [
          ("q4_k_tensor_count", inventory.get("tensor_count_by_type", {}).get("Q4_K")),
          ("q6_k_tensor_count", inventory.get("tensor_count_by_type", {}).get("Q6_K")),
          ("repack_candidate_lane_count", len(lanes)),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
