#!/usr/bin/env python3
"""Run the PR35924 compile gate with the exact upstream oneDNN candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL = (
    ROOT / "tools/intel-qwen36-openvino-pr35924-product-compile-gate.py")
EXACT_BUILD = ROOT / (
    "output/openvino-pr35924-exact-onednn-build-"
    "20260801Tseq2240d-clean/result.json")
EXACT_EVENT_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-babb7375-ze-profile-event-pool-chain.patch")
EXACT_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2240d/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")

EXPECTED_BUILD_SHA256 = (
    "f580792352023d53b9047ca2f1ba536eedbe31ee82f686e9aa92c0c5a6fedec7")
EXPECTED_PLUGIN_SHA256 = (
    "3aef097cac080702ba5fde47e28504fd3896c0bcc1dc88254303dab2925d048a")
EXPECTED_EVENT_PATCH_SHA256 = (
    "6263da09724cb09d34667306f30cb62711a76716c0ea4b158e5d0a28c61e277c")
BUILD_COMMIT = "fd726f425b10a279d50d49c3d7a444fb06e82c6c"


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module("iq36_pr35924_exact_compile_base", BASE_TOOL)
ORIGINAL_LOAD_JSON = BASE.PRODUCT.load_json


def exact_load_json(path: Path) -> dict[str, Any]:
  value = ORIGINAL_LOAD_JSON(path)
  if path.resolve() != EXACT_BUILD.resolve():
    return value
  return {
      **value,
      "verdict": {
          "required_checks_passed": value.get("required_checks_passed"),
          "verdict": "admit_pr35924_plugin_for_compile_only_graph_gate",
          "compile_only_graph_gate_admitted":
              value.get("exact_product_compile_admitted"),
          "inference_admitted":
              value.get("product_correctness_admitted"),
      },
  }


BASE.SCHEMA = "intel-qwen36-openvino-pr35924-exact-product-compile-gate-v0"
BASE.BUILD_AUDIT = EXACT_BUILD
BASE.BUILD_METRICS = EXACT_BUILD
BASE.ONEDNN_PATCH = EXACT_EVENT_PATCH
BASE.PLUGIN = EXACT_PLUGIN
BASE.EXPECTED_BUILD_AUDIT_SHA256 = EXPECTED_BUILD_SHA256
BASE.EXPECTED_BUILD_METRICS_SHA256 = EXPECTED_BUILD_SHA256
BASE.EXPECTED_ONEDNN_PATCH_SHA256 = EXPECTED_EVENT_PATCH_SHA256
BASE.EXPECTED_PLUGIN_SHA256 = EXPECTED_PLUGIN_SHA256
BASE.BUILD_AUDIT_COMMIT = BUILD_COMMIT
BASE.PRODUCT.load_json = exact_load_json
BASE.__file__ = __file__


if __name__ == "__main__":
  raise SystemExit(BASE.main())
