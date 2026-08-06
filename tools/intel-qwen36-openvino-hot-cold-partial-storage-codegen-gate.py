#!/usr/bin/env python3
"""Compile-audit cold-only and hot-only attention partial programs.

This is the sole compiler-only successor admitted by seq1276.  It invokes
ocloc serially for the locked K2/V4 hot16k shape, disassembles both offline
device binaries, and enforces the pre-registered resource and source-isolation
stops.  It never creates an OpenCL context or executes a GPU kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-hot-cold-partial-storage-codegen-gate-v0"
ROUTE = "openvino_hot_cold_partial_storage_specialization_codegen_gate"
NEXT_ROUTE = "openvino_integer_dpas_attention_arithmetic_bound"

SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REFLECTION = ROOT / (
    "output/openvino-route-exhaustion-reflection-"
    "20260715Tseq1276-cleanZ/metrics.json")
OCLOC = Path("/usr/bin/ocloc")
DEVICE = "ptl"
BASE_OPTIONS = (
    "-cl-std=CL3.0 -DIQ36_KEY_QUANT_GROUP=2 "
    "-DIQ36_VALUE_QUANT_GROUP=4 -DIQ36_HOT_TOKENS=16384")

MODES = (
    {
        "storage": "cold",
        "define": 1,
        "kernel": "iq36_direct_i8_cold_partial",
        "pointer_arguments": 8,
    },
    {
        "storage": "hot",
        "define": 2,
        "kernel": "iq36_direct_f16_hot_partial",
        "pointer_arguments": 6,
    },
)
PARTIAL_NAMES = {
    "iq36_direct_i8_hotcold_partial",
    "iq36_direct_i8_cold_partial",
    "iq36_direct_f16_hot_partial",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def run_command(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    run = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
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


def git_output(*arguments: str) -> str:
  run = subprocess.run(
      ["git", *arguments], cwd=ROOT, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace")
  if run.returncode != 0:
    raise RuntimeError(
        f"git {' '.join(arguments)} failed: {run.stderr.strip()}")
  return run.stdout


def git_state(output: Path) -> dict[str, Any]:
  rows = git_output("status", "--porcelain").splitlines()
  try:
    relative_output = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative_output = ""
  rows = [
      row for row in rows
      if not relative_output or relative_output not in row]
  return {
      "commit": git_output("rev-parse", "HEAD").strip(),
      "dirty": bool(rows),
      "dirty_paths": rows,
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def kernel_source_block(text: str, name: str) -> str:
  name_at = text.find(f"__kernel void {name}(")
  if name_at < 0:
    return ""
  brace = text.find("{", name_at)
  if brace < 0:
    return ""
  depth = 0
  for index in range(brace, len(text)):
    if text[index] == "{":
      depth += 1
    elif text[index] == "}":
      depth -= 1
      if depth == 0:
        return text[name_at:index + 1]
  return ""


def kernel_ze_block(text: str, name: str) -> str:
  main = text.split("kernels_misc_info:", maxsplit=1)[0]
  match = re.search(
      rf"^  - name:\s+{re.escape(name)}\s*$"
      r"(.*?)(?=^  - name:|\Z)", main,
      flags=re.MULTILINE | re.DOTALL)
  return match.group(1) if match else ""


def integer_field(block: str, name: str, default: int = 0) -> int:
  match = re.search(
      rf"^\s+{re.escape(name)}:\s+(\d+)\s*$", block,
      flags=re.MULTILINE)
  return int(match.group(1)) if match else default


def scratch_bytes(block: str) -> int:
  return sum(int(value) for value in re.findall(
      r"^\s+- type:\s+scratch\s*$"
      r".*?^\s+size:\s+(\d+)\s*$",
      block, flags=re.MULTILINE | re.DOTALL))


def compile_mode(
    mode: dict[str, Any], raw: Path, timeout_s: int, stop_bytes: int,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
  storage = str(mode["storage"])
  mode_root = raw / storage
  compile_dir = mode_root / "compile"
  disasm_dir = mode_root / "disasm"
  compile_dir.mkdir(parents=True)
  options = (
      f"{BASE_OPTIONS} -DIQ36_PARTIAL_STORAGE_CLASS={mode['define']}")
  compile_command = [
      str(OCLOC), "compile", "-file", str(SOURCE), "-device", DEVICE,
      "-output", f"iq36_{storage}_partial", "-out_dir", str(compile_dir),
      "-options", options,
  ]
  sample_memory(f"before-{storage}-compile", stop_bytes, memory)
  compile_run = run_command(compile_command, timeout_s)
  sample_memory(f"after-{storage}-compile", stop_bytes, memory)
  write_json(mode_root / "compile-command.json", compile_run)
  binaries = sorted(compile_dir.glob("*.bin"))
  binary = binaries[0] if len(binaries) == 1 else None

  disasm_run: dict[str, Any] = {
      "command": [], "returncode": 125, "stdout": "",
      "stderr": "compile did not produce exactly one binary"}
  if compile_run["returncode"] == 0 and binary is not None:
    disasm_dir.mkdir()
    disasm_command = [
        str(OCLOC), "disasm", "-file", str(binary),
        "-dump", str(disasm_dir),
    ]
    sample_memory(f"before-{storage}-disasm", stop_bytes, memory)
    disasm_run = run_command(disasm_command, timeout_s)
    sample_memory(f"after-{storage}-disasm", stop_bytes, memory)
  write_json(mode_root / "disasm-command.json", disasm_run)

  ze_info = disasm_dir / ".ze_info"
  asm = disasm_dir / f".text.{mode['kernel']}.asm"
  if (disasm_run["returncode"] != 0 or binary is None or
      not ze_info.is_file() or not asm.is_file()):
    return {
        **mode,
        "options": options,
        "compile_returncode": compile_run["returncode"],
        "disasm_returncode": disasm_run["returncode"],
        "binary_count": len(binaries),
        "parsed": False,
    }

  ze_text = ze_info.read_text(encoding="utf-8", errors="replace")
  block = kernel_ze_block(ze_text, str(mode["kernel"]))
  front = ze_text.split("kernels_misc_info:", maxsplit=1)[0]
  all_kernels = sorted(set(re.findall(
      r"^  - name:\s+(\S+)\s*$", front, flags=re.MULTILINE)))
  build_options_path = disasm_dir / ".misc.buildOptions"
  build_options = (
      build_options_path.read_text(encoding="utf-8", errors="replace").strip()
      if build_options_path.is_file() else "")
  reported_spill = max(
      integer_field(block, "spill_mem_size"),
      integer_field(block, "spill_size"),
      scratch_bytes(block))
  return {
      **mode,
      "options": options,
      "compile_returncode": compile_run["returncode"],
      "disasm_returncode": disasm_run["returncode"],
      "binary_count": len(binaries),
      "binary": {
          "path": display_path(binary),
          "bytes": binary.stat().st_size,
          "sha256": sha256(binary),
      },
      "build_options": build_options,
      "all_kernels": all_kernels,
      "partial_kernels": sorted(PARTIAL_NAMES.intersection(all_kernels)),
      "simd": integer_field(block, "simd_size"),
      "grf": integer_field(block, "grf_count"),
      "slm_bytes": integer_field(block, "slm_size"),
      "eu_threads": integer_field(block, "eu_thread_count"),
      "spill_bytes": reported_spill,
      "private_bytes": integer_field(block, "private_size"),
      "pointer_argument_count": len(re.findall(
          r"^\s+- arg_type:\s+arg_bypointer\s*$", block,
          flags=re.MULTILINE)),
      "required_work_group_size": [int(value) for value in (
          re.search(
              r"required_work_group_size:\s*\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]",
              block).groups()
          if re.search(
              r"required_work_group_size:\s*\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]",
              block) else ())],
      "asm": {
          "path": display_path(asm),
          "bytes": asm.stat().st_size,
          "lines": len(asm.read_text(
              encoding="utf-8", errors="replace").splitlines()),
          "sha256": sha256(asm),
      },
      "parsed": bool(block),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = (SOURCE, STATUS, ROUTES, REFLECTION, OCLOC)
  missing = [display_path(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing codegen inputs: " + ", ".join(missing))

  git = git_state(output)
  reflection = load_json(REFLECTION)
  routes = load_json(ROUTES)
  status_text = STATUS.read_text(encoding="utf-8")
  source_text = SOURCE.read_text(encoding="utf-8")
  baseline_commit = str(reflection.get("git", {}).get("commit", ""))
  baseline_source = git_output(
      "show", f"{baseline_commit}:{display_path(SOURCE)}")
  engine_changes = git_output(
      "diff", "--name-only", baseline_commit, "--", "engine").splitlines()
  source_diff = git_output(
      "diff", "--unified=3", baseline_commit, "--",
      display_path(SOURCE))
  write_json(raw / "source-audit.json", {
      "baseline_commit": baseline_commit,
      "baseline_sha256": hashlib.sha256(
          baseline_source.encode("utf-8")).hexdigest(),
      "current_sha256": sha256(SOURCE),
      "engine_changes": engine_changes,
      "diff": source_diff,
  })

  unchanged_kernels = {
      name: kernel_source_block(source_text, name) ==
          kernel_source_block(baseline_source, name) != ""
      for name in (
          "iq36_direct_i8_update_state",
          "iq36_direct_i8_hotcold_reduce",
          "iq36_direct_i8_reference_score",
          "iq36_direct_i8_reference_apply",
      )
  }
  source_isolation = {
      "only_attention_source_changed_under_engine": engine_changes == [
          display_path(SOURCE)],
      "non_partial_kernels_unchanged": all(unchanged_kernels.values()),
      "default_mixed_entrypoint_preserved": all(fragment in source_text for fragment in (
          "#define IQ36_PARTIAL_STORAGE_CLASS IQ36_PARTIAL_STORAGE_MIXED",
          "#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_i8_hotcold_partial",
      )),
      "specialized_entrypoints_are_distinct": all(fragment in source_text for fragment in (
          "#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_i8_cold_partial",
          "#define IQ36_PARTIAL_KERNEL_NAME iq36_direct_f16_hot_partial",
      )),
      "inactive_state_arguments_are_preprocessed_out": all(
          fragment in source_text for fragment in (
              "#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_HOT\n"
              "    __global const uint* cold_k,",
              "#if IQ36_PARTIAL_STORAGE_CLASS != IQ36_PARTIAL_STORAGE_COLD\n"
              "    __global const uint* hot_k,",
          )),
      "compact_launch_maps_to_original_workspace": all(
          fragment in source_text for fragment in (
              "const uint group = kv_head * IQ36_CHUNK_COUNT + chunk;",
              "const uint chunk = IQ36_COLD_CHUNK_COUNT + hot_chunk;",
          )),
      "locked_shape_constants_preserved": all(fragment in source_text for fragment in (
          "#define IQ36_TOKEN_TILE 16U",
          "#define IQ36_CHUNK_TOKENS 512U",
          "#define IQ36_BLOCKS_PER_CHUNK 32U",
          "__attribute__((reqd_work_group_size(128, 1, 1)))",
          "__attribute__((intel_reqd_sub_group_size(16)))",
      )),
  }

  rows: list[dict[str, Any]] = []
  # Deliberately serial: complete cold compile+disasm before starting hot.
  for mode in MODES:
    rows.append(compile_mode(
        mode, raw, args.timeout_s, stop_bytes, memory))
  sample_memory("after-codegen", stop_bytes, memory)
  write_jsonl(output / "metrics.jsonl", rows)

  contract = dict(reflection.get("codegen_contract", {}))
  active = dict(routes.get("active_route", {}))
  candidate_selected = any(
      row.get("seq") == 1276 and row.get("selected_next_route") == ROUTE
      for row in routes.get("candidate_history", [])
      if isinstance(row, dict))
  switch_selected = any(
      row.get("seq_covered") == 1276 and
      row.get("decision") ==
          "select_compile_only_hot_cold_partial_storage_specialization" and
      row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
      if isinstance(row, dict))
  compile_pass = len(rows) == 2 and all(
      row.get("compile_returncode") == 0 and
      row.get("disasm_returncode") == 0 and row.get("parsed") is True
      for row in rows)
  exact_names = len(rows) == 2 and all(
      row.get("partial_kernels") == [row.get("kernel")] for row in rows)
  exact_arguments = len(rows) == 2 and all(
      row.get("pointer_argument_count") == row.get("pointer_arguments")
      for row in rows)
  exact_options = len(rows) == 2 and all(
      f"-DIQ36_PARTIAL_STORAGE_CLASS={row['define']}" in
          str(row.get("build_options", "")) and
      "-DIQ36_KEY_QUANT_GROUP=2" in str(row.get("build_options", "")) and
      "-DIQ36_VALUE_QUANT_GROUP=4" in str(row.get("build_options", "")) and
      "-DIQ36_HOT_TOKENS=16384" in str(row.get("build_options", ""))
      for row in rows)
  codegen_pass = len(rows) == 2 and all(
      row.get("simd") == int(contract.get("required_simd", -1)) and
      int(row.get("grf", 10**9)) <= int(
          contract.get("maximum_grf_per_specialized_kernel", -1)) and
      int(row.get("slm_bytes", 10**9)) <= int(
          contract.get("maximum_slm_bytes", -1)) and
      row.get("spill_bytes") == 0 and row.get("private_bytes") == 0 and
      int(row.get("asm", {}).get("lines", 10**9)) < int(
          contract.get("each_specialized_asm_lines_must_be_below", -1)) and
      row.get("required_work_group_size") == [128, 1, 1]
      for row in rows)
  no_runtime_evidence = not any(
      (output / name).exists() for name in (
          "run.json", "probe.json", "tokens.jsonl", "worker.time"))

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1276_admits_this_exact_compiler_only_route",
            reflection.get("required_checks_passed") is True and
            reflection.get("selected_next_route") == ROUTE and
            reflection.get("compiler_only_gate_allowed") is True and
            reflection.get("gpu_worker_allowed") is False and
            active.get("id") == ROUTE and candidate_selected and
            switch_selected and "compile-only" in status_text),
      check("source_edit_is_isolated_and_locked",
            all(source_isolation.values()), source_isolation=source_isolation,
            unchanged_kernels=unchanged_kernels,
            engine_changes=engine_changes),
      check("exactly_two_serial_offline_programs_compile_and_disassemble",
            compile_pass, rows=rows),
      check("each_program_contains_only_its_specialized_partial_entrypoint",
            exact_names, rows=rows),
      check("inactive_storage_arguments_are_absent_from_compiler_metadata",
            exact_arguments, rows=rows),
      check("locked_k2_v4_hot16k_compiler_options_are_exact",
            exact_options, rows=rows),
      check("both_specialized_kernels_clear_registered_codegen_stops",
            codegen_pass, contract=contract, rows=rows),
      check("gate_created_no_gpu_or_model_runtime_evidence",
            no_runtime_evidence),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_20_sample_specialized_partial_component"
      if passed else
      "close_partial_storage_specialization_select_integer_dpas_bound")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "required_checks_passed": passed,
      "verdict": verdict,
      "selected_next_route": (
          "openvino_hot_cold_partial_storage_specialized_component"
          if passed else NEXT_ROUTE),
      "compiler_only": True,
      "gpu_kernel_executed": False,
      "model_worker_executed": False,
      "product_speedup_claim": False,
      "source": {
          "path": display_path(SOURCE),
          "sha256": sha256(SOURCE),
          "isolation": source_isolation,
      },
      "contract": contract,
      "programs": rows,
      "checks": checks,
      "memory_samples": memory,
  }
  write_json(output / "metrics.json", result)
  write_json(output / "correctness.json", {
      "schema": SCHEMA,
      "required_checks_passed": passed,
      "checks": checks,
      "claim_boundary": "offline compiler/codegen attribution only",
      "product_speedup_claim": False,
  })
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "created_at": result["created_at"],
      "git": git,
      "tool": display_path(Path(__file__)),
      "source": display_path(SOURCE),
      "reflection": display_path(REFLECTION),
      "ocloc": str(OCLOC),
      "device": DEVICE,
      "command": sys.argv,
      "required_checks_passed": passed,
      "gpu_kernel_executed": False,
  })
  summary = [
      "# OpenVINO cold/hot partial-storage codegen gate",
      "",
      f"Required checks: **{str(passed).lower()}**. Verdict: `{verdict}`.",
      "No OpenCL context, model worker, or GPU kernel was executed.",
      "",
      "| storage | kernel | SIMD | GRF | SLM B | spill B | args | asm lines |",
      "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    summary.append(
        f"| {row['storage']} | {row['kernel']} | {row.get('simd', '-')} | "
        f"{row.get('grf', '-')} | {row.get('slm_bytes', '-')} | "
        f"{row.get('spill_bytes', '-')} | "
        f"{row.get('pointer_argument_count', '-')} | "
        f"{row.get('asm', {}).get('lines', '-')} |")
  summary.extend([
      "",
      ("Passing admits exactly one serial 20-sample component; failing closes "
       "the specialization before GPU execution and selects the integer-DPAS "
       "source bound."),
  ])
  (output / "summary.md").write_text(
      "\n".join(summary) + "\n", encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "required_checks_passed": passed,
      "verdict": verdict,
      "programs": [{
          key: row.get(key) for key in (
              "storage", "simd", "grf", "slm_bytes", "spill_bytes")}
          for row in rows],
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
