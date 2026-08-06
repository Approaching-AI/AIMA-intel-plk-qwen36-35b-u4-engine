#!/usr/bin/env python3
"""Bind live OpenVINO attention dispatches to their exact IGC programs.

Consumes a clean attention phase-profile artifact whose trace records a
program ordinal on every dispatch.  Each captured stock/custom binary is
disassembled with the installed ocloc and audited for SIMD width, GRF/thread
occupancy, private/spill memory, and instruction shape.  This is codegen and
resource attribution only; it is not a product speed claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-attention-codegen-gate-v0"
CODEGEN_HELPER = ROOT / "tools/intel-qwen36-openvino-gdn-codegen-gate.py"


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CG = load_module("iq36_openvino_codegen_helper", CODEGEN_HELPER)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--profile-artifact", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  if args.out_dir is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-attention-codegen-{stamp}"
  return args


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


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def phase_key(marker: str) -> str:
  parts = marker.split("-")
  return next((part for part in parts if part.startswith("phase")), "unknown")


def main() -> int:
  args = parse_args()
  profile = args.profile_artifact.resolve()
  profile_metrics_path = profile / "metrics.json"
  if not profile_metrics_path.is_file():
    raise SystemExit(f"missing profile metrics: {profile_metrics_path}")
  profile_metrics = load_json(profile_metrics_path)

  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = CG.git_state(out)
  codegen_rows: list[dict[str, Any]] = []
  disassembly_rows: list[dict[str, Any]] = []
  phase_requirements: set[tuple[str, str, str]] = set()
  phase_coverage: set[tuple[str, str, str]] = set()

  profile_raw = profile / "raw"
  for lane_dir in sorted(path for path in profile_raw.iterdir()
                         if path.is_dir()):
    lane = lane_dir.name
    for mode in ("stock", "candidate"):
      worker = lane_dir / mode
      trace = load_jsonl(worker / "opencl-trace.jsonl")
      dispatches = [row for row in trace if row.get("event") == "ndrange"]
      for row in dispatches:
        phase_requirements.add((lane, mode, str(row.get("marker", ""))))
      by_ordinal: dict[int, list[dict[str, Any]]] = {}
      for row in dispatches:
        ordinal = row.get("program_ordinal")
        if isinstance(ordinal, int):
          by_ordinal.setdefault(ordinal, []).append(row)

      dump_flag = "stock_sdpa" if mode == "stock" else "custom_sdpa"
      dumps = [row for row in trace
               if row.get("event") == "program_dump" and
               row.get(dump_flag) is True]
      seen_ordinals: set[int] = set()
      for dump in dumps:
        ordinal = int(dump["ordinal"])
        # A plugin may call clBuildProgram more than once on the same program
        # object.  The dump path and final binary are ordinal-owned, so audit
        # that program once rather than treating repeated builds as variants.
        if ordinal in seen_ordinals:
          continue
        seen_ordinals.add(ordinal)
        binaries = list(dump.get("binaries", []))
        if not binaries:
          continue
        binary = Path(str(binaries[0]["path"]))
        source = Path(str(dump["source_path"]))
        destination = raw / lane / mode / f"program{ordinal:03d}"
        disassembly = CG.disassemble(binary, destination, args.timeout_s)
        disassembly_rows.append({
            "lane": lane,
            "mode": mode,
            "ordinal": ordinal,
            **disassembly,
        })
        kernel_needles = (
            ["sdpa_micro"] if mode == "stock" else
            sorted({str(row.get("kernel", ""))
                    for row in by_ordinal.get(ordinal, [])
                    if str(row.get("kernel", "")).startswith("iq36_")}) or
            ["iq36_hot_attention_single_owner",
             "iq36_prefill_attention_tiled"])
        parsed = None
        if disassembly["returncode"] == 0:
          for needle in kernel_needles:
            parsed = CG.codegen_metrics(
                mode, destination, needle, binary, source)
            if parsed is not None:
              break
        if parsed is None:
          continue
        bound = by_ordinal.get(ordinal, [])
        if not bound and mode == "stock":
          # Stock programs are compiled while compile_model has no active
          # phase marker and may be rehydrated through the plugin cache, so
          # their dispatch handle need not retain our source-program ordinal.
          # The captured prefill/generate programs expose distinct full
          # kernel symbols; bind those symbols directly and require coverage
          # of every timed stock dispatch below.
          bound = [row for row in dispatches
                   if row.get("kernel") == parsed["kernel_name"]]
        markers = sorted({str(row.get("marker", "")) for row in bound})
        for marker in markers:
          phase_coverage.add((lane, mode, marker))
        codegen_rows.append({
            "metric_scope": "attention_codegen",
            "lane": lane,
            "mode": mode,
            "program_ordinal": ordinal,
            "build_marker": str(dump.get("marker", "")),
            "dispatch_markers": markers,
            "dispatch_count": len(bound),
            "dispatch_total_ms": (
                sum(int(row.get("duration_ns", 0)) for row in bound) /
                1_000_000.0),
            **parsed,
        })

  write_json(raw / "disassembly.json", disassembly_rows)
  write_jsonl(out / "metrics.jsonl", codegen_rows)
  profile_git = dict(profile_metrics.get("git", {}))
  parsed_phases = {
      (row["lane"], row["mode"], marker)
      for row in codegen_rows for marker in row["dispatch_markers"]
  }
  all_no_private = bool(codegen_rows) and all(
      row["execution_env"]["private_size"] == 0
      for row in codegen_rows)
  decode_rows = [
      row for row in codegen_rows
      if any(phase_key(marker) == "phase1"
             for marker in row["dispatch_markers"])]
  all_decode_no_spill = bool(decode_rows) and all(
      row["execution_env"]["spill_mem_size"] == 0
      for row in decode_rows)
  prefill_spill: dict[str, dict[str, int]] = {}
  for row in codegen_rows:
    if not any(phase_key(marker) == "phase0"
               for marker in row["dispatch_markers"]):
      continue
    lane_rows = prefill_spill.setdefault(row["lane"], {})
    lane_rows[row["mode"]] = max(
        lane_rows.get(row["mode"], 0),
        row["execution_env"]["spill_mem_size"])
  candidate_prefill_spill_not_worse = bool(prefill_spill) and all(
      set(rows) == {"stock", "candidate"} and
      rows["candidate"] <= rows["stock"]
      for rows in prefill_spill.values())
  all_simd16 = bool(codegen_rows) and all(
      row["execution_env"]["simd_size"] == 16 for row in codegen_rows)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("profile_artifact_is_clean_current_commit",
            profile_git.get("dirty") is False and
            profile_git.get("commit") == git["commit"] and
            profile_metrics.get("attribution_checks_passed") is True,
            profile_git=profile_git,
            attribution_checks_passed=profile_metrics.get(
                "attribution_checks_passed")),
      check("every_timed_phase_binds_to_a_captured_program",
            phase_requirements == phase_coverage == parsed_phases,
            required=sorted(phase_requirements),
            covered=sorted(phase_coverage),
            parsed=sorted(parsed_phases)),
      check("all_bound_programs_disassemble_and_parse",
            bool(codegen_rows) and
            len(codegen_rows) == len(disassembly_rows) and
            all(row["returncode"] == 0 for row in disassembly_rows),
            parsed_programs=len(codegen_rows),
            disassembled_programs=len(disassembly_rows)),
      check("all_attention_programs_are_simd16", all_simd16),
      check("all_attention_programs_report_no_private_memory",
            all_no_private),
      check("all_decode_programs_report_no_spill_memory",
            all_decode_no_spill,
            programs=[{
                "lane": row["lane"],
                "mode": row["mode"],
                "program_ordinal": row["program_ordinal"],
                "spill_mem_size": row["execution_env"]["spill_mem_size"],
            } for row in decode_rows]),
      check("candidate_prefill_spill_does_not_exceed_stock",
            candidate_prefill_spill_not_worse,
            rows=prefill_spill),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  correctness = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "checks": checks,
      "claim_boundary": "live attention compiler/codegen attribution only",
      "product_speedup_claim": False,
  }
  manifest = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "tool": relative(Path(__file__)),
      "profile_artifact": relative(profile),
      "ocloc": str(CG.OCLOC),
      "command": sys.argv,
      "required_checks_passed": required_checks_passed,
      "product_speedup_claim": False,
  }
  write_json(out / "correctness.json", correctness)
  write_json(out / "manifest.json", manifest)

  summary = [
      "# OpenVINO attention codegen gate",
      "",
      f"- required checks: {'PASS' if required_checks_passed else 'FAIL'}",
      f"- commit: `{git['commit']}`; dirty: `{git['dirty']}`",
      f"- profile: `{relative(profile)}`",
      "- claim boundary: compiler/codegen attribution only",
      "",
      "| lane | mode | phase | programs | SIMD | GRF | EU threads | spill | private | instructions | dispatch ms |",
      "|---|---|---|---:|---|---|---|---|---|---|---:|",
  ]
  groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
  for row in codegen_rows:
    markers = row["dispatch_markers"] or [row["build_marker"]]
    for phase in sorted({phase_key(marker) for marker in markers}):
      groups.setdefault(
          (row["lane"], row["mode"], phase), []).append(row)
  for (lane, mode, phase), rows in sorted(groups.items()):
    def values(field: str) -> str:
      return "/".join(map(str, sorted({
          row["execution_env"][field] for row in rows})))
    instructions = "/".join(map(str, sorted({
        row["assembly"]["instruction_lines"] for row in rows})))
    summary.append(
        f"| {lane} | {mode} | {phase} | {len(rows)} | "
        f"{values('simd_size')} | {values('grf_count')} | "
        f"{values('eu_thread_count')} | {values('spill_mem_size')} | "
        f"{values('private_size')} | {instructions} | "
        f"{sum(row['dispatch_total_ms'] for row in rows):.6f} |")
  (out / "summary.md").write_text(
      "\n".join(summary) + "\n", encoding="utf-8")
  print(json.dumps({
      "out_dir": relative(out),
      "required_checks_passed": required_checks_passed,
      "programs": len(codegen_rows),
      "phases": len(phase_coverage),
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
