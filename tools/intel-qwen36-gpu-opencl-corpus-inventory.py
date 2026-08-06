#!/usr/bin/env python3
"""Inventory the sibling intel-plk OpenCL kernel corpus for GPU bring-up.

This tool is deliberately static: it does not benchmark, copy kernels, or claim
speedups. It produces a compact manifest of the reusable OpenCL source, kernel
entry points, mode strings, and feature tokens that matter for the Arc B390 GPU
route selected in routes-ledger.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_CORPUS_DIR = Path(
    "/Users/jiawei-macmini/projects/intel-plk-highspeed/"
    "native/intel-plk-qwen36-native/src/gpu_opencl"
)

FEATURE_KEYWORDS = {
    "attention": ["attention", "score", "softmax", "context"],
    "decode_q4": ["q4", "q4_k"],
    "decode_q6": ["q6", "q6_k"],
    "kq8_input": ["kq8", "q8"],
    "moe_down": ["routed_down", "down", "expert"],
    "packed_layout": ["packed", "rowstripe", "split", "tile"],
    "resident": ["resident", "shared"],
    "swiglu_gate_up": ["swiglu", "gate_up", "gate-up"],
    "route_expert2pair": ["expert2pair"],
    "route_dualdot": ["dualdot"],
    "route_weightfold": ["weightfold"],
    "dpas": ["dpas"],
}


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path) -> str:
  return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
  text = read_text(path)
  return {
      "path": str(path),
      "exists": path.exists(),
      "bytes": path.stat().st_size if path.exists() else 0,
      "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
      "sha256": sha256_file(path) if path.exists() else None,
  }


def extract_kernel_names(cpp_text: str) -> list[str]:
  names = sorted(set(re.findall(r"\b__kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", cpp_text)))
  return names


def extract_kernel_source_span(cpp_text: str) -> dict[str, Any]:
  start = cpp_text.find("R\"CLC(")
  end = cpp_text.find(")CLC\"", start + 1) if start >= 0 else -1
  if start < 0 or end < 0:
    return {"present": False}
  source = cpp_text[start:end]
  return {
      "present": True,
      "start_byte": start,
      "end_byte": end,
      "bytes": end - start,
      "lines": source.count("\n"),
  }


def extract_string_literals(text: str) -> list[str]:
  values = []
  for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text):
    value = match.group(1)
    if 2 <= len(value) <= 220:
      values.append(value)
  return values


def is_relevant_mode(value: str) -> bool:
  lower = value.lower()
  if " " in value and "-" not in value and "_" not in value:
    return False
  return any(token in lower for tokens in FEATURE_KEYWORDS.values() for token in tokens)


def extract_relevant_modes(cpp_text: str, header_text: str) -> list[str]:
  values = set()
  for value in extract_string_literals(cpp_text) + extract_string_literals(header_text):
    if is_relevant_mode(value):
      values.add(value)
  return sorted(values)


def classify(values: list[str]) -> dict[str, list[str]]:
  result: dict[str, list[str]] = {}
  for feature, keywords in FEATURE_KEYWORDS.items():
    hits = []
    for value in values:
      lower = value.lower()
      if any(keyword in lower for keyword in keywords):
        hits.append(value)
    result[feature] = hits
  return result


def extract_option_defaults(header_text: str) -> list[dict[str, Any]]:
  options: list[dict[str, Any]] = []
  struct_pattern = re.compile(r"struct\s+(OpenCl[A-Za-z0-9_]*Options)\s*\{(.*?)\};", re.S)
  default_pattern = re.compile(r"std::string\s+([A-Za-z0-9_]*kernel_mode)\s*=\s*\"([^\"]+)\"")
  for struct_name, body in struct_pattern.findall(header_text):
    defaults = [{"field": field, "value": value} for field, value in default_pattern.findall(body)]
    options.append({"struct": struct_name, "kernel_mode_defaults": defaults})
  return options


def write_json(path: Path, data: Any) -> None:
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()

  stamp = utc_stamp()
  output_dir = args.output_dir or Path(f"output/gpu-opencl-corpus-inventory-{stamp}")
  output_dir.mkdir(parents=True, exist_ok=False)

  cpp_path = args.corpus_dir / "matvec_microbench.cpp"
  header_path = args.corpus_dir / "matvec_microbench.hpp"
  cpp_text = read_text(cpp_path) if cpp_path.exists() else ""
  header_text = read_text(header_path) if header_path.exists() else ""

  kernels = extract_kernel_names(cpp_text)
  modes = extract_relevant_modes(cpp_text, header_text)
  feature_modes = classify(modes)
  feature_kernels = classify(kernels)
  option_defaults = extract_option_defaults(header_text)
  checks = [
      {"name": "corpus_dir_exists", "pass": args.corpus_dir.is_dir()},
      {"name": "matvec_microbench_cpp_present", "pass": cpp_path.is_file()},
      {"name": "matvec_microbench_hpp_present", "pass": header_path.is_file()},
      {"name": "opencl_kernel_source_present", "pass": extract_kernel_source_span(cpp_text).get("present", False)},
      {"name": "kernel_entry_points_present", "pass": len(kernels) > 0},
      {"name": "q4_q6_modes_present", "pass": bool(feature_modes["decode_q4"]) and bool(feature_modes["decode_q6"])},
      {"name": "moe_down_modes_present", "pass": bool(feature_modes["moe_down"])},
      {"name": "expert2pair_dualdot_weightfold_present", "pass": bool(feature_modes["route_expert2pair"] and feature_modes["route_dualdot"] and feature_modes["route_weightfold"])},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)

  inventory = {
      "schema_version": "intel-qwen36-gpu-opencl-corpus-inventory-v0",
      "workstream": WORKSTREAM,
      "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
      "corpus_dir": str(args.corpus_dir),
      "sources": {
          "cpp": source_record(cpp_path) if cpp_path.exists() else {"path": str(cpp_path), "exists": False},
          "header": source_record(header_path) if header_path.exists() else {"path": str(header_path), "exists": False},
      },
      "opencl_kernel_source": extract_kernel_source_span(cpp_text),
      "kernel_entry_points": kernels,
      "kernel_entry_point_count": len(kernels),
      "relevant_mode_count": len(modes),
      "relevant_modes": modes,
      "feature_mode_counts": {key: len(value) for key, value in feature_modes.items()},
      "feature_kernel_counts": {key: len(value) for key, value in feature_kernels.items()},
      "feature_modes": feature_modes,
      "feature_kernels": feature_kernels,
      "option_defaults": option_defaults,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "recommendation": "port the OpenCL runtime loader plus a narrow source-stream/repack probe before integrating a decode backend",
  }
  manifest = {
      "schema_version": "intel-qwen36-gpu-opencl-corpus-inventory-v0",
      "workstream": WORKSTREAM,
      "created_at_utc": inventory["created_at_utc"],
      "tool": "tools/intel-qwen36-gpu-opencl-corpus-inventory.py",
      "inventory": str(output_dir / "inventory.json"),
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  summary = [
      "# GPU OpenCL Corpus Inventory",
      "",
      f"- corpus: `{args.corpus_dir}`",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      "- speedup claims allowed: `false`",
      f"- source files: `{cpp_path.name}`, `{header_path.name}`",
      f"- OpenCL kernel entry points: `{len(kernels)}`",
      f"- relevant mode strings: `{len(modes)}`",
      f"- q4 modes: `{len(feature_modes['decode_q4'])}`",
      f"- q6 modes: `{len(feature_modes['decode_q6'])}`",
      f"- MoE/down modes: `{len(feature_modes['moe_down'])}`",
      f"- expert2pair/dualdot/weightfold modes: `{len(feature_modes['route_expert2pair'])}` / `{len(feature_modes['route_dualdot'])}` / `{len(feature_modes['route_weightfold'])}`",
      "",
      "Decision: use this as the GPU bring-up entry map. The next code step is a narrow OpenCL runtime plus source-stream/repack probe, not a full backend port.",
      "",
  ]
  write_json(output_dir / "inventory.json", inventory)
  write_json(output_dir / "manifest.json", manifest)
  (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(output_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
