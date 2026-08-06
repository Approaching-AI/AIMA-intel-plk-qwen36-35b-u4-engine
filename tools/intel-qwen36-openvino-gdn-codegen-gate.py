#!/usr/bin/env python3
"""Audit actual stock/custom GatedDeltaNet OpenCL programs and GPU ISA.

The gate runs the existing all-layer numeric substitution in isolated workers
under the repository's OpenCL audit library.  It captures the exact source and
CL_PROGRAM_BINARIES returned by the live driver, disassembles both binaries
with the installed ocloc, and compares compiler metadata plus instruction
shape.  It is a component/codegen attribution gate, not product timing.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-gdn-codegen-gate-v0"
NUMERIC_GATE = ROOT / "tools/intel-qwen36-openvino-gdn-custom-gate.py"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
OCLOC = Path("/usr/bin/ocloc")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--numeric-gate", type=Path, default=NUMERIC_GATE)
  parser.add_argument("--replace-layers", type=int, default=30)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if not 1 <= args.replace_layers <= 30:
    parser.error("replace-layers must be in 1..30")
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-gdn-codegen-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  rows = []
  for number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"{path}:{number}: expected JSON object")
    rows.append(value)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*items: str) -> str:
    run = subprocess.run(
        ["git", *items], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative_out = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative_out = ""
  dirty = [row for row in dirty if not relative_out or relative_out not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def run_command(
    command: list[str], *, timeout_s: int, cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  try:
    run = subprocess.run(
        command, cwd=cwd, env=environment, check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout_s)
    return {
        "command": command,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
    }
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command,
        "returncode": 124,
        "stdout": str(exc.stdout or ""),
        "stderr": str(exc.stderr or exc),
    }


def build_trace(raw: Path, timeout_s: int) -> dict[str, Any]:
  configure = run_command([
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ], timeout_s=timeout_s)
  build = run_command([
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET,
      "-j8",
  ], timeout_s=timeout_s)
  result = {
      "configure": configure,
      "build": build,
      "library": relative(TRACE_LIBRARY),
      "library_sha256": (
          sha256_file(TRACE_LIBRARY) if TRACE_LIBRARY.is_file() else None),
      "pass": (
          configure["returncode"] == 0 and build["returncode"] == 0 and
          TRACE_LIBRARY.is_file()),
  }
  write_json(raw / "trace-build.json", result)
  return result


def first_program_dump(
    rows: list[dict[str, Any]], kind: str,
) -> dict[str, Any] | None:
  key = "stock_gdn" if kind == "stock" else "custom_gdn"
  return next(
      (row for row in rows
       if row.get("event") == "program_dump" and row.get(key) is True),
      None)


def disassemble(
    binary: Path, destination: Path, timeout_s: int,
) -> dict[str, Any]:
  destination.mkdir(parents=True, exist_ok=False)
  result = run_command([
      str(OCLOC), "disasm", "-file", str(binary),
      "-dump", str(destination),
  ], timeout_s=timeout_s)
  result["binary"] = relative(binary)
  result["destination"] = relative(destination)
  return result


def kernel_asm(directory: Path, needle: str) -> Path | None:
  rows = [
      path for path in directory.glob(".text.*.asm")
      if needle in path.name and "Symbol_Table" not in path.name
  ]
  return sorted(rows)[0] if len(rows) == 1 else None


def kernel_name_from_asm(path: Path) -> str:
  name = path.name
  if name.startswith(".text."):
    name = name[len(".text."):]
  if name.endswith(".asm"):
    name = name[:-len(".asm")]
  return name


def kernel_ze_block(path: Path, kernel_name: str) -> str:
  text = path.read_text(encoding="utf-8")
  match = re.search(
      rf"^  - name:\s+{re.escape(kernel_name)}\s*$"
      r"(.*?)(?=^  - name:|\Z)", text, flags=re.MULTILINE | re.DOTALL)
  return match.group(1) if match else ""


def integer_field(block: str, name: str, default: int = 0) -> int:
  match = re.search(rf"^\s+{re.escape(name)}:\s+(\d+)\s*$", block,
                    flags=re.MULTILINE)
  return int(match.group(1)) if match else default


def per_thread_scratch_bytes(block: str) -> int:
  return sum(int(value) for value in re.findall(
      r"^\s+- type:\s+scratch\s*$"
      r".*?^\s+size:\s+(\d+)\s*$",
      block, flags=re.MULTILINE | re.DOTALL))


def assembly_metrics(path: Path) -> dict[str, Any]:
  counts: collections.Counter[str] = collections.Counter()
  instruction_lines = 0
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("//"):
      continue
    if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*:\s*$", line):
      continue
    match = re.match(r"^\s*(?:\([^)]*\)\s+)?([a-z][a-z0-9_.]*)\s", line)
    if not match:
      continue
    opcode = match.group(1)
    counts[opcode] += 1
    instruction_lines += 1
  return {
      "path": relative(path),
      "bytes": path.stat().st_size,
      "sha256": sha256_file(path),
      "instruction_lines": instruction_lines,
      "opcode_counts": dict(sorted(counts.items())),
      "send_ugm": counts["send.ugm"],
      "branches": sum(counts[name] for name in (
          "break", "call", "goto", "jmpi", "ret", "while")),
      "integer_address_ops": sum(counts[name] for name in (
          "add", "add3", "asr", "macl", "mach", "mad", "mul", "shl")),
  }


def codegen_metrics(
    kind: str, directory: Path, needle: str, binary: Path, source: Path,
) -> dict[str, Any] | None:
  asm = kernel_asm(directory, needle)
  ze_info = directory / ".ze_info"
  if asm is None or not ze_info.is_file():
    return None
  kernel_name = kernel_name_from_asm(asm)
  block = kernel_ze_block(ze_info, kernel_name)
  if not block:
    return None
  source_text = source.read_text(encoding="utf-8", errors="replace")
  reported_spill_mem_size = integer_field(block, "spill_mem_size")
  reported_spill_size = integer_field(block, "spill_size")
  scratch_size = per_thread_scratch_bytes(block)
  return {
      "kind": kind,
      "kernel_name": kernel_name,
      "binary": {
          "path": relative(binary),
          "bytes": binary.stat().st_size,
          "sha256": sha256_file(binary),
      },
      "source": {
          "path": relative(source),
          "bytes": source.stat().st_size,
          "sha256": sha256_file(source),
          "pitch_array_definitions": len(re.findall(
              r"#define INPUT\d+_PITCHES \(long \[\]\)", source_text)),
          "pitch_array_index_uses": len(re.findall(
              r"INPUT\d+_PITCHES\[", source_text)),
      },
      "execution_env": {
          "simd_size": integer_field(block, "simd_size"),
          "grf_count": integer_field(block, "grf_count"),
          "eu_thread_count": integer_field(block, "eu_thread_count"),
          "indirect_stateless_count": integer_field(
              block, "indirect_stateless_count"),
          "private_size": integer_field(block, "private_size"),
          # IGC schema revisions use either spill_mem_size or spill_size and
          # may expose the actual allocation only as a per-thread scratch
          # buffer.  Treat any of the three as compiler spill/scratch memory.
          "spill_mem_size": max(
              reported_spill_mem_size, reported_spill_size, scratch_size),
          "reported_spill_mem_size": reported_spill_mem_size,
          "reported_spill_size": reported_spill_size,
          "scratch_size": scratch_size,
      },
      "assembly": assembly_metrics(asm),
  }


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = git_state(out)
  trace_build = build_trace(raw, args.timeout_s)

  programs = raw / "programs"
  programs.mkdir()
  trace_path = raw / "opencl-trace.jsonl"
  numeric_out = raw / "numeric"
  environment = os.environ.copy()
  environment.update({
      "LD_AUDIT": str(TRACE_LIBRARY.resolve()),
      "IQ36_OPENCL_TRACE_PATH": str(trace_path),
      "IQ36_OPENCL_TRACE_FILTER": "gated_delta_net",
      "IQ36_OPENCL_PROGRAM_DUMP_DIR": str(programs),
  })
  numeric = run_command([
      sys.executable, str(args.numeric_gate.resolve()),
      "--replace-layers", str(args.replace_layers),
      "--out-dir", str(numeric_out),
      "--timeout-s", str(args.timeout_s),
  ], timeout_s=args.timeout_s, environment=environment)
  write_json(raw / "numeric-command.json", {
      **numeric,
      "environment": {
          key: environment[key] for key in (
              "LD_AUDIT", "IQ36_OPENCL_TRACE_PATH",
              "IQ36_OPENCL_TRACE_FILTER", "IQ36_OPENCL_PROGRAM_DUMP_DIR")
      },
  })

  trace_rows = load_jsonl(trace_path)
  stock_dump = first_program_dump(trace_rows, "stock")
  custom_dump = first_program_dump(trace_rows, "custom")
  dump_rows = {"stock": stock_dump, "custom": custom_dump}
  disassembly: dict[str, Any] = {}
  metrics: dict[str, Any] = {}
  for kind, row in dump_rows.items():
    if row is None:
      continue
    binaries = row.get("binaries", [])
    if not binaries:
      continue
    binary = Path(str(binaries[0]["path"]))
    source = Path(str(row["source_path"]))
    if not binary.is_file() or not source.is_file():
      continue
    destination = raw / f"{kind}-disasm"
    disassembly[kind] = disassemble(binary, destination, args.timeout_s)
    needle = "gated_delta_net_ref" if kind == "stock" else "iq36_gated_delta_net"
    parsed = codegen_metrics(kind, destination, needle, binary, source)
    if parsed is not None:
      metrics[kind] = parsed
  write_json(raw / "disassembly.json", disassembly)

  numeric_correctness_path = numeric_out / "correctness.json"
  numeric_correctness = (
      load_json(numeric_correctness_path)
      if numeric_correctness_path.is_file() else {})
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("trace_library_builds", trace_build["pass"],
            trace_build=trace_build),
      check("numeric_all_layer_gate_passes",
            numeric["returncode"] == 0 and
            numeric_correctness.get("required_checks_passed") is True,
            returncode=numeric["returncode"],
            numeric_correctness=relative(numeric_correctness_path)),
      check("stock_program_source_and_binary_captured",
            stock_dump is not None and
            stock_dump.get("source_written") is True and
            stock_dump.get("binaries_written") is True,
            row=stock_dump),
      check("custom_program_source_and_binary_captured",
            custom_dump is not None and
            custom_dump.get("source_written") is True and
            custom_dump.get("binaries_written") is True,
            row=custom_dump),
      check("stock_and_custom_programs_disassemble",
            set(disassembly) == {"stock", "custom"} and
            all(row["returncode"] == 0 for row in disassembly.values()),
            rows=disassembly),
      check("stock_and_custom_gdn_codegen_metadata_parsed",
            set(metrics) == {"stock", "custom"}, rows=metrics),
      check("both_kernels_are_simd16_without_reported_spill",
            set(metrics) == {"stock", "custom"} and
            all(row["execution_env"]["simd_size"] == 16 and
                row["execution_env"]["spill_mem_size"] == 0 and
                row["execution_env"]["private_size"] == 0
                for row in metrics.values()), rows=metrics),
  ]
  required_checks_passed = all(row["pass"] for row in checks)

  comparison = {}
  if set(metrics) == {"stock", "custom"}:
    stock = metrics["stock"]
    custom = metrics["custom"]
    comparison = {
        "grf_delta": (
            custom["execution_env"]["grf_count"] -
            stock["execution_env"]["grf_count"]),
        "eu_thread_delta": (
            custom["execution_env"]["eu_thread_count"] -
            stock["execution_env"]["eu_thread_count"]),
        "indirect_stateless_delta": (
            custom["execution_env"]["indirect_stateless_count"] -
            stock["execution_env"]["indirect_stateless_count"]),
        "asm_bytes_ratio": (
            custom["assembly"]["bytes"] /
            stock["assembly"]["bytes"]),
        "instruction_lines_ratio": (
            custom["assembly"]["instruction_lines"] /
            stock["assembly"]["instruction_lines"]),
        "integer_address_ops_ratio": (
            custom["assembly"]["integer_address_ops"] /
            stock["assembly"]["integer_address_ops"]),
    }

  manifest = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "tool": relative(Path(__file__)),
      "numeric_gate": relative(args.numeric_gate),
      "replace_layers": args.replace_layers,
      "trace_library": relative(TRACE_LIBRARY),
      "ocloc": str(OCLOC),
      "command": sys.argv,
      "required_checks_passed": required_checks_passed,
      "product_speedup_claim": False,
  }
  correctness = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "checks": checks,
      "numeric_gate_correctness": relative(numeric_correctness_path),
      "claim_boundary": "compiler/codegen attribution only",
      "product_speedup_claim": False,
  }
  smoothness = {
      "schema_version": SCHEMA,
      "required_checks_passed": True,
      "applicable": False,
      "reason": "fixed sequence-1024 compiler artifact audit; no context claim",
  }
  write_json(out / "manifest.json", manifest)
  write_json(out / "correctness.json", correctness)
  write_json(out / "smoothness.json", smoothness)
  metric_rows = [
      {"metric_scope": "codegen", **row} for row in metrics.values()
  ]
  if comparison:
    metric_rows.append({"metric_scope": "comparison", **comparison})
  write_jsonl(out / "metrics.jsonl", metric_rows)
  summary = [
      "# OpenVINO GatedDeltaNet codegen gate",
      "",
      f"- required checks: {'PASS' if required_checks_passed else 'FAIL'}",
      f"- commit: `{git['commit']}`; dirty: `{git['dirty']}`",
      f"- replaced layers: `{args.replace_layers}`",
      "- claim boundary: compiler/codegen attribution only; no product speed claim",
  ]
  if set(metrics) == {"stock", "custom"}:
    summary.extend([
        "",
        "| kernel | SIMD | GRF | EU threads | indirect stateless | asm bytes | instructions | integer/address ops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| {kind} | {simd} | {grf} | {threads} | {indirect} | {asm_bytes} | {instructions} | {address} |".format(
                kind=kind,
                simd=metrics[kind]["execution_env"]["simd_size"],
                grf=metrics[kind]["execution_env"]["grf_count"],
                threads=metrics[kind]["execution_env"]["eu_thread_count"],
                indirect=metrics[kind]["execution_env"]["indirect_stateless_count"],
                asm_bytes=metrics[kind]["assembly"]["bytes"],
                instructions=metrics[kind]["assembly"]["instruction_lines"],
                address=metrics[kind]["assembly"]["integer_address_ops"],
            ) for kind in ("stock", "custom")
        ],
        "",
        f"Comparison: `{json.dumps(comparison, sort_keys=True)}`",
    ])
  (out / "summary.md").write_text(
      "\n".join(summary) + "\n", encoding="utf-8")
  print(json.dumps({
      "out_dir": relative(out),
      "required_checks_passed": required_checks_passed,
      "comparison": comparison,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
