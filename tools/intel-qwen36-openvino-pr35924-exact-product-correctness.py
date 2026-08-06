#!/usr/bin/env python3
"""Run output130 correctness for the exact-upstream PR35924 candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-pr35924-swish-parity-correctness.py")
EXACT_COMPILE = ROOT / (
    "output/openvino-pr35924-exact-product-compile-"
    "20260801Tseq2241-clean/result.json")
EXACT_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2240d/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")

EXPECTED_COMPILE_SHA256 = (
    "731fdc71a1b7060c3d791d3728b80ef0b92c4e92a1f3c13ea556bec29d4a9f4a")
EXPECTED_PLUGIN_SHA256 = (
    "3aef097cac080702ba5fde47e28504fd3896c0bcc1dc88254303dab2925d048a")
BUILD_COMMIT = "0c326f7fc9a415e39a8ab8e8b24b119be613831b"


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module("iq36_pr35924_exact_correctness_base", BASE_TOOL)
ORIGINAL_LOAD_JSON = BASE.PRODUCT.load_json


def exact_load_json(path: Path) -> dict[str, Any]:
  value = ORIGINAL_LOAD_JSON(path)
  if path.resolve() != EXACT_COMPILE.resolve():
    return value
  return {
      **value,
      "candidate_plugin": value.get("plugin"),
  }


BASE.SCHEMA = "intel-qwen36-openvino-pr35924-exact-product-correctness-v0"
BASE.BUILD_AUDIT = EXACT_COMPILE
BASE.PLUGIN = EXACT_PLUGIN
BASE.EXPECTED_BUILD_AUDIT_SHA256 = EXPECTED_COMPILE_SHA256
BASE.EXPECTED_PLUGIN_SHA256 = EXPECTED_PLUGIN_SHA256
BASE.BUILD_COMMIT = BUILD_COMMIT
BASE.PRODUCT.load_json = exact_load_json
BASE.__file__ = __file__


if __name__ == "__main__":
  raise SystemExit(BASE.main())
