#!/usr/bin/env python3
"""Run the one paired correctly-rounded Q/K scale component probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = (
    ROOT / "tools/intel-qwen36-gpu-level-zero-postconv-recurrent-component-"
    "probe-gate.py")


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_module(BASE_PATH, "iq36_level_zero_probe_base")


def main() -> int:
  defaults = [
      "--sequence", "623",
      "--target-sequence", "622",
      "--schema-version",
      "intel-qwen36-gpu-software-correctly-rounded-qk-scale-probe-v0",
      "--current-route",
      "gpu_software_correctly_rounded_qk_scale_primitive_probe_gate",
      "--pass-next-route",
      "gpu_software_correctly_rounded_qk_scale_primitive_integration_contract_gate",
      "--fail-next-route",
      "gpu_software_correctly_rounded_qk_scale_primitive_route_close_gate",
      "--target-decision",
      "select_gpu_software_correctly_rounded_qk_scale_primitive_probe_gate",
      "--target-disposition",
      "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_target_compile",
      "--accept-disposition",
      "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_component",
      "--reject-disposition",
      "reject_level_zero_ocloc_cr_recip_postconv_recurrent_v2_component",
      "--pass-reason",
      "The paired v2 component passes all six exact boundaries and the whole-"
      "shell floor ruler. Audit only a contiguous Level Zero island or external-"
      "memory integration contract next; decode and tokens remain blocked.",
      "--fail-reason",
      "The sole correctly-rounded reciprocal attempt failed exactness, budget, "
      "or runtime compatibility. Close the exact-kernel route under the seq620 "
      "stop condition; do not add another reciprocal, sqrt, sigmoid, flag, "
      "workgroup, or arithmetic-order variant.",
      "--tool-path",
      "tools/intel-qwen36-gpu-software-correctly-rounded-qk-scale-primitive-"
      "probe-gate.py",
      "--target-compile",
      str(ROOT / (
          "output/seq622-gpu-software-correctly-rounded-qk-scale-primitive-"
          "target-compile-gate-20260710Tseq622Z/metrics.json")),
      "--source-gate",
      str(ROOT / (
          "output/seq621-gpu-software-correctly-rounded-qk-scale-primitive-"
          "source-gate-20260710Tseq621Z/metrics.json")),
      "--native-module",
      str(ROOT / (
          "output/seq621-gpu-software-correctly-rounded-qk-scale-primitive-"
          "source-gate-20260710Tseq621Z/generated/iq36_postconv_recurrent.bin")),
      "--out-dir",
      str(ROOT / (
          "output/seq623-gpu-software-correctly-rounded-qk-scale-primitive-"
          "probe-gate-20260710Tseq623Z")),
  ]
  sys.argv[1:1] = defaults
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
