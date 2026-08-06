#!/usr/bin/env python3
"""Target-compile the captured correctly-rounded Q/K scale component harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = (
    ROOT / "tools/intel-qwen36-gpu-level-zero-postconv-recurrent-component-"
    "target-compile-gate.py")


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_module(BASE_PATH, "iq36_level_zero_target_compile_base")


def main() -> int:
  defaults = [
      "--sequence", "622",
      "--source-sequence", "621",
      "--schema-version",
      "intel-qwen36-gpu-software-correctly-rounded-qk-scale-target-compile-v0",
      "--current-route",
      "gpu_software_correctly_rounded_qk_scale_primitive_target_compile_gate",
      "--selected-next-route",
      "gpu_software_correctly_rounded_qk_scale_primitive_probe_gate",
      "--source-decision",
      "select_gpu_software_correctly_rounded_qk_scale_primitive_target_compile_gate",
      "--source-disposition",
      "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_source",
      "--accept-disposition",
      "accept_level_zero_ocloc_cr_recip_postconv_recurrent_v2_target_compile",
      "--repair-disposition",
      "repair_level_zero_ocloc_cr_recip_postconv_recurrent_v2_target_compile",
      "--tool-path",
      "tools/intel-qwen36-gpu-software-correctly-rounded-qk-scale-primitive-"
      "target-compile-gate.py",
      "--source-gate",
      str(ROOT / (
          "output/seq621-gpu-software-correctly-rounded-qk-scale-primitive-"
          "source-gate-20260710Tseq621Z/metrics.json")),
      "--out-dir",
      str(ROOT / (
          "output/seq622-gpu-software-correctly-rounded-qk-scale-primitive-"
          "target-compile-gate-20260710Tseq622Z")),
  ]
  sys.argv[1:1] = defaults
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
