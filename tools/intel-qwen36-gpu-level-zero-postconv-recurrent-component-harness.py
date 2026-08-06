#!/usr/bin/env python3
"""Generate the captured-layer Level Zero component harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = (
    ROOT / "tools/intel-qwen36-gpu-vulkan-postconv-recurrent-component-harness.py"
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_module(BASE_SOURCE, "iq36_vulkan_component_harness_base")

WORKSTREAM = BASE.WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-probe-v0")
DEFAULT_HOST = BASE.DEFAULT_HOST
DEFAULT_MODEL = BASE.DEFAULT_MODEL
DEFAULT_ENV_SCRIPT = BASE.DEFAULT_ENV_SCRIPT
DEFAULT_REMOTE_ROOT = BASE.DEFAULT_REMOTE_ROOT
PAYLOAD_ROOT = BASE.PAYLOAD_ROOT
OPENCL_SOURCE = BASE.OPENCL_SOURCE
PAYLOAD_SPECS = BASE.PAYLOAD_SPECS
SOURCE_FILES = [
    row for row in BASE.SOURCE_FILES
    if "gpu_vulkan_postconv_recurrent" not in row[0]
] + [
    ("engine/include/intel_qwen36/gpu_level_zero_postconv_recurrent.hpp",
     "include/intel_qwen36/gpu_level_zero_postconv_recurrent.hpp"),
    ("engine/src/gpu_level_zero_postconv_recurrent.cpp",
     "src/gpu_level_zero_postconv_recurrent.cpp"),
]


def _level_zero_cpp() -> str:
  source = BASE.HARNESS_CPP
  replacements = [
      (
          '#include "intel_qwen36/gpu_vulkan_postconv_recurrent.hpp"',
          '#include "intel_qwen36/gpu_level_zero_postconv_recurrent.hpp"'),
      (
          "intel-qwen36-gpu-vulkan-postconv-recurrent-component-probe-v0",
          "intel-qwen36-gpu-level-zero-postconv-recurrent-component-probe-v0"),
      ("std::string postconv_spirv;\n  std::string recurrent_spirv;",
       "std::string native_module;"),
      ('  std::string vulkan_device = "PTL";\n', ""),
      ('    else if (key == "--postconv-spv") args.postconv_spirv = value();\n'
       '    else if (key == "--recurrent-spv") args.recurrent_spirv = value();',
       '    else if (key == "--native-module") args.native_module = value();'),
      ('    else if (key == "--vulkan-device") args.vulkan_device = value();\n',
       ""),
      ('  Require(!args.postconv_spirv.empty(), "--postconv-spv is required");\n'
       '  Require(!args.recurrent_spirv.empty(), "--recurrent-spv is required");',
       '  Require(!args.native_module.empty(), "--native-module is required");'),
      ("GpuVulkanPostconvRecurrentInput", "GpuLevelZeroPostconvRecurrentInput"),
      ("GpuVulkanPostconvRecurrentRunner", "GpuLevelZeroPostconvRecurrentRunner"),
      ("vulkan_input", "level_zero_input"),
      ('    iq36::GpuLevelZeroPostconvRecurrentRunner vulkan(\n'
       '        args.postconv_spirv, args.recurrent_spirv, args.vulkan_device);\n'
       '    vulkan.Run(level_zero_input, 1);\n'
       '    const auto candidate = vulkan.Run(level_zero_input, args.samples);',
       '    iq36::GpuLevelZeroPostconvRecurrentRunner level_zero(\n'
       '        args.native_module);\n'
       '    level_zero.Run(level_zero_input, 1);\n'
       '    const auto candidate = level_zero.Run(\n'
       '        level_zero_input, args.samples);'),
      ('"\\\"vulkan_device\\\":\\\"" << vulkan.device_name()',
       '"\\\"level_zero_device\\\":\\\"" << level_zero.device_name()'),
      ("Vulkan component probe error", "Level Zero component probe error"),
  ]
  for old, new in replacements:
    if old not in source:
      raise ValueError(f"base harness marker missing: {old!r}")
    source = source.replace(old, new)
  return source


HARNESS_CPP = _level_zero_cpp()


def generate_cpp(opencl_source: str) -> str:
  literal = 'R"IQ36L0(' + opencl_source + ')IQ36L0"'
  return HARNESS_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", literal)


def payload_manifest(layer: int) -> dict[str, dict[str, object]]:
  return BASE.payload_manifest(layer)
