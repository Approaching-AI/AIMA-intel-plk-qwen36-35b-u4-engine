#!/usr/bin/env python3
"""Run the isolated OpenVINO hot/cold output512 product gate.

The gate keeps one worker process per measured row, uses a fresh NEO cache for
every worker, and executes workers strictly serially.  Each logical prompt is
fed through frozen resident 8192-token chunks in one InferRequest.  A full
promotion run uses the exact seven-bucket, three-prompt-class matrix and at
least eight stock/candidate/candidate/stock (ABBA) blocks per case.  Smaller
runs are bounded diagnostics and cannot enable a speedup claim.

The orchestrator samples process RSS/swap and system available memory for the
entire worker lifetime.  It never launches a second GPU worker until the first
one exits, and completed worker directories can be reused with ``--resume``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import iq36_product_policy as PRODUCT_POLICY


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-hot-cold-product-gate-v1"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/"
    "acceptance-matrix.json")
MATERIALIZATION = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z")
FILLER_DIR = ROOT / "output/r0-openvino-exact-filler-prompts-20260711"
SENTINEL_SPECS = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "long-context-sentinels.jsonl")
CUSTOM_CONFIG = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
CUSTOM_SOURCES = (
    ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl",
    ROOT / "engine/openvino/custom/iq36_adaptive_attention_decode.cl",
    ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl",
    ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl",
    ROOT / "engine/openvino/custom/iq36_prefill_microkernel_shims.cl",
    ROOT / "engine/openvino/custom/iq36_prefill32split_microkernel_shims.cl",
    ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl",
    ROOT / "engine/openvino/custom/iq36_stock_micro_attention_oracle.cl",
    ROOT / "engine/openvino/custom/iq36_prefill_state_update.cl",
    ROOT / "engine/openvino/custom/iq36_linear_conv_swish.cl",
    ROOT / "engine/openvino/custom/iq36_greedy_top1.cl",
    ROOT / "engine/openvino/custom/iq36_greedy_top1_merge.cl",
    ROOT / "engine/openvino/custom/iq36_greedy_token_or_logits.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefix.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefill_wide_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_prefill_small_microkernel_shim.cl",
    ROOT / "engine/openvino/custom/iq36_fixed_fc_multi_output.cl",
    ROOT / "engine/openvino/iq36-fixed-fc-moe-router-compat.patch",
    ROOT / "engine/openvino/iq36-fixed-fc-phase-provider.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q4-phase-provider.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q4-adaptive-correction.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-exact.patch",
    ROOT / "engine/openvino/iq36-lm-head-affine-q4-group128.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-q4.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-gated-q4-compact.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-greedy-local2.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-token-only.patch",
    ROOT / "engine/openvino/iq36-lm-head-i8q1-compact-direct-top8.patch",
    ROOT / "engine/openvino/iq36-onednn-ze-profile-event-pool-chain.patch",
    ROOT / "engine/openvino/iq36-dq-runtime-skip-realloc-fastpath.patch",
    ROOT / "engine/openvino/iq36-fc-q1-reuse-prefill-capacity.patch",
    ROOT / "engine/openvino/iq36-fc-stable-prepare-fastpath.patch",
    ROOT / "engine/openvino/iq36-custom-adaptive-attention-multikernel.patch",
    ROOT / "engine/gpu/opencl/iq36_lm_head_i8q4_integrated.cl",
    ROOT / "tools/intel_qwen36_openvino_fixed_fc.py",
)
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
HOT_COLD_GATE = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
BOOTSTRAP_GATE = (
    ROOT / "tools/intel-qwen36-openvino-specialization-bootstrap-gate.py")
PERF_INFERENCE_MODULE = ROOT / "tools/iq36_perf_inference.py"
ATTENTION_DIAGNOSTICS_MODULE = (
    ROOT / "tools/intel_qwen36_openvino_attention_diagnostics.py")

CORE_BUCKETS = PRODUCT_POLICY.CORE_BUCKETS
PRIORITY_BUCKETS = (32768, 65536, 131072)
CUSTOM_BUCKETS = CORE_BUCKETS
PROMPT_SETS = ("prefill_shape", "sentinel", "filler")
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
LM_HEAD_NAME = "__module.model.lm_head/ov_ext::linear/MatMul"
GREEDY_TOP1_PARTIAL_COUNT = 64
GREEDY_TOP1_VOCABULARY = 248320
LM_HEAD_HIDDEN_SIZE = 2048
FIXED_FC_COHORT_COUNTS = {
    "linear_attention_input": (30, 4),
    "full_attention_qkv": (10, 3),
    "router_shared_input": (40, 4),
    "attention_output": (40, 1),
    "shared_expert_down": (40, 1),
}
FIXED_FC_COHORTS = tuple(FIXED_FC_COHORT_COUNTS)
CASE_SUFFIX = {
    2048: "002k", 4096: "004k", 8192: "008k", 16384: "016k",
    32768: "032k", 65536: "064k", 131072: "128k",
}
CHECKPOINT_STEPS = (0, 1, 7, 36, 44, 63, 255, 511)
KLD_MAX = 0.005
TOP1_MIN = 0.99
COSINE_MIN = 0.999
FROZEN_CHUNK_TOKENS = 8192
MIN_PROMOTION_BLOCKS = 8


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load module from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


HOT = load_module("iq36_hot_cold_gate_product", HOT_COLD_GATE)
GRAPH = HOT.GRAPH
BOOT = load_module("iq36_ov_bootstrap_product", BOOTSTRAP_GATE)
perf_inference = load_module(
    "iq36_perf_inference_product", PERF_INFERENCE_MODULE)
ATTENTION_DIAGNOSTICS = load_module(
    "iq36_attention_diagnostics_product", ATTENTION_DIAGNOSTICS_MODULE)


def device_greedy_custom_classes(ov: Any) -> tuple[type, type]:
  class IQ36GreedyTop1Partials(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(
          0, ov.Type.i64,
          ov.PartialShape([1, 1, 1, GREEDY_TOP1_PARTIAL_COUNT]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GreedyTop1Partials(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  class IQ36GreedyTop1Merge(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(0, ov.Type.i32, ov.PartialShape([1, 1, 1, 1]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GreedyTop1Merge(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36GreedyTop1Partials, IQ36GreedyTop1Merge


def token_or_logits_custom_class(ov: Any) -> type:
  class IQ36GreedyTokenOrLogits(ov.Op):
    def __init__(self, inputs: Any = None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self) -> None:
      self.set_output_type(0, ov.Type.i32, ov.PartialShape([1, 1, 1, 1]))

    def clone_with_new_inputs(self, new_inputs: Any) -> Any:
      return IQ36GreedyTokenOrLogits(new_inputs)

    def visit_attributes(self, visitor: Any) -> bool:
      return True

  return IQ36GreedyTokenOrLogits


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      value = json.loads(text)
      if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def parse_csv_ints(value: str) -> tuple[int, ...]:
  try:
    values = tuple(int(item.strip()) for item in value.split(",")
                   if item.strip())
  except ValueError as exc:
    raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
  if not values or len(set(values)) != len(values):
    raise argparse.ArgumentTypeError("values must be non-empty and unique")
  unknown = sorted(set(values) - set(CORE_BUCKETS))
  if unknown:
    raise argparse.ArgumentTypeError(f"unsupported buckets: {unknown}")
  return values


def parse_prompt_sets(value: str) -> tuple[str, ...]:
  if value.strip() == "all":
    return PROMPT_SETS
  values = tuple(item.strip() for item in value.split(",") if item.strip())
  unknown = sorted(set(values) - set(PROMPT_SETS))
  if not values or unknown or len(set(values)) != len(values):
    raise argparse.ArgumentTypeError(
        f"prompt sets must be unique values from {PROMPT_SETS} or all")
  return values


def parse_fixed_fc_cohorts(value: str) -> tuple[str, ...]:
  values = tuple(item.strip() for item in value.split(",") if item.strip())
  unknown = sorted(set(values) - set(FIXED_FC_COHORTS))
  if not values or unknown or len(set(values)) != len(values):
    raise argparse.ArgumentTypeError(
        f"fixed FC cohorts must be unique values from {FIXED_FC_COHORTS}")
  expected_order = tuple(
      cohort for cohort in FIXED_FC_COHORTS if cohort in set(values))
  if values != expected_order:
    raise argparse.ArgumentTypeError(
        f"fixed FC cohorts must follow model order {FIXED_FC_COHORTS}")
  return values


def parse_target_layers(value: str) -> tuple[int, ...]:
  try:
    values = tuple(int(item.strip()) for item in value.split(",")
                   if item.strip())
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        "target layers must be comma-separated integers") from exc
  if not values or len(set(values)) != len(values):
    raise argparse.ArgumentTypeError(
        "target layers must be non-empty and unique")
  unknown = sorted(set(values) - set(FULL_ATTENTION_LAYERS))
  if unknown:
    raise argparse.ArgumentTypeError(
        f"unsupported full-attention layers: {unknown}")
  expected_order = tuple(
      layer for layer in FULL_ATTENTION_LAYERS if layer in set(values))
  if values != expected_order:
    raise argparse.ArgumentTypeError(
        f"target layers must follow model order {FULL_ATTENTION_LAYERS}")
  return values


def parse_nonnegative_csv_ints(value: str) -> tuple[int, ...]:
  try:
    values = tuple(int(item.strip()) for item in value.split(",")
                   if item.strip())
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        "expected comma-separated non-negative integers") from exc
  if (not values or len(set(values)) != len(values) or
      any(item < 0 for item in values)):
    raise argparse.ArgumentTypeError(
        "values must be non-empty, unique, and non-negative")
  return values


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--buckets", type=parse_csv_ints, default=(32768,))
  parser.add_argument(
      "--prompt-sets", type=parse_prompt_sets, default=("sentinel",),
      help="comma-separated prefill_shape,sentinel,filler or all")
  parser.add_argument("--output-tokens", type=int, default=512)
  parser.add_argument("--paired-blocks", type=int, default=1)
  parser.add_argument(
      "--capture-all-correctness-logits", action="store_true",
      help=("diagnostic-only: persist every teacher-forced logits row, not "
            "only the fixed gate checkpoints"))
  parser.add_argument(
      "--capture-lm-head-hidden", action="store_true",
      help=("diagnostic-only: persist and compare the last-query LM-head "
            "input at every selected correctness checkpoint"))
  parser.add_argument(
      "--capture-attention-layers", type=parse_target_layers, default=(),
      help=("diagnostic-only: expose last-query Q/K/V and attention for this "
            "ordered target-layer subset in correctness workers"))
  parser.add_argument(
      "--capture-attention-steps", type=parse_nonnegative_csv_ints,
      default=(),
      help=("diagnostic-only: output-token steps whose exposed attention "
            "boundaries are persisted"))
  parser.add_argument(
      "--capture-attention-history-layers", type=parse_target_layers,
      default=(),
      help=("diagnostic-only: capture logical stock/custom K/V history for "
            "this exact-history layer subset"))
  parser.add_argument(
      "--capture-attention-history-steps", type=parse_nonnegative_csv_ints,
      default=(),
      help=("diagnostic-only: output-token steps after which selected K/V "
            "VariableState tensors are persisted"))
  parser.add_argument(
      "--candidate-policy", choices=("auto", "custom", "stock"),
      default="auto",
      help="auto selects custom at 32k+ and stock SDPA for short guards")
  parser.add_argument(
      "--target-layers", type=parse_target_layers,
      default=FULL_ATTENTION_LAYERS,
      help=("ordered custom full-attention layer subset; omitted layers "
            "remain stock SDPA"))
  parser.add_argument(
      "--decode-chunk256-layers", type=parse_target_layers, default=(),
      help=("ordered target-layer subset that uses the 256-token decode "
            "reduction tile; default keeps the 512-token carrier"))
  parser.add_argument(
      "--decode-f32-numerator-layers", type=parse_target_layers, default=(),
      help=("diagnostic target-layer subset that keeps the 256-token decode "
            "schedule but accumulates the softmax-value numerator in F32"))
  parser.add_argument(
      "--decode-dual256-layers", type=parse_target_layers, default=(),
      help=("ordered target-layer subset that keeps one 512-token work-group "
            "but performs two independent 256-token numerator reductions"))
  parser.add_argument(
      "--decode-stock256-layers", type=parse_target_layers, default=(),
      help=("diagnostic target-layer subset that exports two F16-rounded "
            "256-token partials per 512-token work-group before final merge"))
  parser.add_argument(
      "--decode-stock-score-layers", type=parse_target_layers, default=(),
      help=("diagnostic target-layer subset that uses the 256-token schedule "
            "and stock lane-strided F32-FMA QK accumulation order"))
  parser.add_argument(
      "--decode-stock-partition-layers", type=parse_target_layers, default=(),
      help=("diagnostic target-layer subset that reproduces the complete "
            "stock 256-key softmax/value/finalization arithmetic"))
  parser.add_argument(
      "--decode-stock-micro-layers", type=parse_target_layers, default=(),
      help=("diagnostic exact-history layer subset that reuses the actual "
            "stock 256-WI Gemmstone SDPA microkernel arithmetic"))
  parser.add_argument(
      "--decode-page-sparse-layers", type=parse_target_layers, default=(),
      help=("exact-phase stock-micro subset that uses the bounded fused "
            "sample16/page512/keep64 long-context route"))
  parser.add_argument(
      "--exact-phase-context-partition4", action="store_true",
      help=("default-off source-bounded route: split exact stock-micro decode "
            "into four chronological context partitions per KV head"))
  parser.add_argument(
      "--exact-phase-dual-cohort-buckets", type=parse_csv_ints, default=(),
      help=("selected custom exact-phase buckets that pipeline generated KQ "
            "and chronological softmax/VS in two on-chip cohorts"))
  parser.add_argument(
      "--prefill-history-capacity", type=int,
      help=("diagnostic unified-carrier ring capacity; must be a power of "
            "two and at least the largest selected bucket"))
  parser.add_argument(
      "--exact-history-layers", type=parse_target_layers, default=(),
      help=("diagnostic target-layer subset whose carrier uses the larger "
            "--exact-history-capacity instead of the base prefill history"))
  parser.add_argument(
      "--exact-history-capacity", type=int,
      help=("fixed carrier capacity for --exact-history-layers; must cover "
            "the largest prompt plus all requested decode tokens"))
  parser.add_argument(
      "--exact-history-capacity-slack-tokens", type=int,
      help=("derive each case's exact-history capacity independently as its "
            "base prefill capacity plus this many tokens; mutually exclusive "
            "with --exact-history-capacity"))
  parser.add_argument(
      "--custom-composition",
      choices=("unified", "phase_branch", "stock_prefill", "exact_phase",
               "direct_i8_fixed", "adaptive_i8_fixed"),
      default="unified",
      help=("unified, experimental nested phase_branch, stock-prefill with "
            "custom decode state owner, single-owner fast prefill plus exact "
            "decode, fixed-state direct_i8_fixed attention, or the "
            "single-owner four-kernel adaptive_i8_fixed route"))
  parser.add_argument(
      "--adaptive-attention-topk", type=int,
      choices=(128, 252, 256, 512, 1024, 2048), default=512,
      help=("global exact-correction budget per query head for "
            "adaptive_i8_fixed; default 512"))
  parser.add_argument(
      "--adaptive-attention-high-topk-layers",
      type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset that overrides the base exact-"
            "correction budget with --adaptive-attention-high-topk"))
  parser.add_argument(
      "--adaptive-attention-high-topk", type=int,
      choices=(128, 252, 256, 512, 1024, 2048), default=256,
      help=("per-query-head correction budget for the high-top-k layer "
            "subset; default 256"))
  parser.add_argument(
      "--adaptive-attention-v16-layers", type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset that uses group16 V scales; "
            "other adaptive layers retain group32 V"))
  parser.add_argument(
      "--adaptive-attention-key-exact-layers",
      type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset that scans retained dense F16 K "
            "while keeping group32 V plus selected exact-V replacement"))
  parser.add_argument(
      "--adaptive-attention-key-residual1-layers",
      type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset with one shared-scale K "
            "fractional bit-plane"))
  parser.add_argument(
      "--adaptive-attention-value-residual1-layers",
      type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset with one shared-scale V "
            "fractional bit-plane"))
  parser.add_argument(
      "--adaptive-attention-packed-kv-layers",
      type=parse_target_layers, default=(),
      help=("adaptive_i8_fixed layer subset with packed group32 signed "
            "low-bit K/V cold state"))
  parser.add_argument(
      "--adaptive-attention-packed-kv-variant",
      choices=("k6v7", "k7v7", "k7v8", "k8v7"),
      help="packed K/V bit-width pair used by the selected layer subset")
  parser.add_argument(
      "--adaptive-attention-exact-layers", type=parse_target_layers,
      default=(),
      help=("adaptive_i8_fixed or direct_i8_fixed layer subset that keeps "
            "the existing exact-phase stock-micro owner while the remaining "
            "selected layers use the requested compressed owner"))
  parser.add_argument(
      "--alias-linear-state-assign", action="store_true",
      help=("candidate plugin: retain each linear-attention "
            "producer buffer selected by --linear-state-alias-scope as the "
            "next request's VariableState"))
  parser.add_argument(
      "--linear-state-alias-scope", choices=("conv", "ssm", "all"),
      default="all",
      help=("state family selected by --alias-linear-state-assign; all is "
            "the clean-carrier product default"))
  parser.add_argument(
      "--fuse-linear-conv-state", action="store_true",
      help=("replace all 30 linear-attention transpose/conv/state/SiLU "
            "boundaries on the custom candidate path"))
  parser.add_argument(
      "--fuse-qk-rope-layout", action="store_true",
      help=("exact-phase dual-cohort candidate only: replace all ten Q/K "
            "transpose, partial-RoPE, and concat producer boundaries"))
  parser.add_argument(
      "--fuse-router-shared-triple", action="store_true",
      help=("candidate plugin only: enable the default-off horizontal fusion "
            "of the locked N=[1,512,512] shared-expert projections while "
            "leaving the N=256 router branch independent"))
  parser.add_argument(
      "--fuse-router-shared-pair", action="store_true",
      help=("candidate plugin only: enable the default-off horizontal fusion "
            "of only the locked N=[512,512] shared-expert bulk projections "
            "while leaving the N=1 scalar gate and N=256 router independent"))
  parser.add_argument(
      "--direct-ssm-state-assign", action="store_true",
      help=("candidate graph: feed all 30 SSM Assigns directly from the "
            "Loop final-state output instead of its concat/slice repack"))
  parser.add_argument(
      "--fuse-fixed-fc", action="store_true",
      help=("replace all 160 fixed-FC groups / 390 projections on the "
            "custom candidate path"))
  parser.add_argument(
      "--fixed-fc-cohorts", type=parse_fixed_fc_cohorts,
      help=("diagnostic subset of fixed-FC cohorts to replace; omitted "
            "means all five cohorts"))
  parser.add_argument(
      "--fixed-fc-manager-direct", action="store_true",
      help=("leave every fixed FC native and exercise the candidate plugin's "
            "oneDNN-T1/row-major-router-T>1 implementation manager"))
  parser.add_argument(
      "--fixed-fc-manager-scope", choices=("m1024",),
      default="m1024",
      help=("T>1 provider scope used by --fixed-fc-manager-direct: the "
            "fused shared-expert gate/up bulk projection"))
  parser.add_argument(
      "--lm-head-i8q4", action="store_true",
      help=("candidate plugin: replace only the decode T=1 LM head with the "
            "locked signed-Q4 row-stripe provider"))
  parser.add_argument(
      "--lm-head-i8q1", action="store_true",
      help=("candidate plugin: replace only the decode T=1 LM head with the "
            "locked binary two-centroid plus local-top12 exact provider"))
  parser.add_argument(
      "--lm-head-i8q1-gated-exact", action="store_true",
      help=("binary LM head only: count final Q1 logits within max-11 and "
            "run the full exact-I8 distribution path when at least 25 rows "
            "qualify"))
  parser.add_argument(
      "--lm-head-i8q1-gated-exact-affine-q4", action="store_true",
      help=("gated-exact binary LM head only: replace the count25 full-I8 "
            "distribution scan with the certified affine-Q4/group128 bound "
            "and exact-candidate path; retain full-I8 on capacity overflow"))
  parser.add_argument(
      "--lm-head-i8q1-gated-q4", action="store_true",
      help=("binary LM head only: count final Q1 logits within max-11 and "
            "run a nested signed-Q4 distribution plus exact max-8 correction "
            "when at least 25 rows qualify"))
  parser.add_argument(
      "--lm-head-i8q1-greedy-local2", action="store_true",
      help=("paired candidate timing workers only: use local-top2 Q1 exact "
            "correction without the distribution fallback; full-logit "
            "correctness workers retain the configured local-top12 fallback"))
  parser.add_argument(
      "--lm-head-device-greedy-feedback", action="store_true",
      help=("paired local2 candidate timing workers only: reduce the last "
            "query logits to one exact I32 token on GPU; correctness and "
            "stock retain their existing output boundaries"))
  parser.add_argument(
      "--lm-head-token-only-feedback", action="store_true",
      help=("paired local2 candidate timing workers only: let the LM-head "
            "provider emit a compact encoded token and use one phase-safe "
            "custom boundary; correctness and stock remain unchanged"))
  parser.add_argument(
      "--pack-gdn-state", action="store_true",
      help=("store custom-candidate GDN recurrent FP32 state as provider-"
            "private [V,K] rows; stock workers remain unchanged"))
  parser.add_argument(
      "--self-bind-hot-states", action="store_true",
      help=("diagnostic: bypass graph-initialized hot ReadValue storage and "
            "assign each request state tensor back to itself; unsafe across "
            "warmup reset on runtimes that retain device contents"))
  parser.add_argument(
      "--prefill-chunk-tokens", type=int, default=FROZEN_CHUNK_TOKENS)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
  parser.add_argument("--materialization-dir", type=Path,
                      default=MATERIALIZATION)
  parser.add_argument("--filler-dir", type=Path, default=FILLER_DIR)
  parser.add_argument("--sentinel-specs", type=Path, default=SENTINEL_SPECS)
  parser.add_argument("--custom-config", type=Path, default=CUSTOM_CONFIG)
  parser.add_argument(
      "--candidate-gpu-plugin", type=Path,
      help=("candidate-only OpenVINO GPU plugin; required whenever the "
            "selected policy uses the custom graph"))
  parser.add_argument(
      "--candidate-fc-stable-prepare-fastpath", action="store_true",
      help=("candidate plugin: skip unchanged dynamic FullyConnected/RMS/"
            "eltwise shape preparation while preserving the normal "
            "transition path"))
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument(
      "--host-time-profiling", type=int, choices=(0, 1, 2), default=0,
      help=("diagnostic GPU-plugin host-time logging: 1 prints enqueue; "
            "2 also splits input, enqueue, wait, and output time"))
  parser.add_argument(
      "--capture-execution-census", action="store_true",
      help=("diagnostic: enable PERF_COUNT on correctness workers and retain "
            "the final inference's executed types and slowest rows; paired "
            "timing workers remain unprofiled"))
  parser.add_argument(
      "--capture-prefill-profiles", action="store_true",
      help=("diagnostic: retain cumulative execution-profile snapshots after "
            "the warmup and every resident prefill chunk; correctness-only"))
  parser.add_argument(
      "--prime-candidate-exact-decode-shape", action="store_true",
      help=("exact-phase candidate only: use the existing one-token warmup at "
            "the case's full prompt position/attention-mask shape, then reset "
            "all request state before measured cold no-prefix prefill"))
  parser.add_argument(
      "--candidate-impls-cache-capacity", type=int,
      help=("exact-prime candidate only: set the release-internal GPU "
            "dynamic-implementation LRU capacity through the isolated worker "
            "environment; zero means unbounded and is diagnostic only"))
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument(
      "--abort-below-available-gib", type=float, default=4.0,
      help=("interrupt the active isolated worker before host available "
            "memory falls below this threshold (default: 4)"))
  parser.add_argument(
      "--worker-transient-scope", action="store_true",
      help=("place each serial worker in a fresh user transient scope so "
            "memory pressure and OOM containment cannot leak between workers "
            "or into the orchestrator; no resource limit is changed"))
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--plan-only", action="store_true")
  parser.add_argument("--no-warmup", action="store_true")
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.output_tokens < 2:
    parser.error("output-tokens must be at least two")
  if bool(args.capture_attention_layers) != bool(args.capture_attention_steps):
    parser.error(
        "capture-attention-layers and capture-attention-steps are required "
        "together")
  if (args.capture_attention_steps and
      max(args.capture_attention_steps) >= args.output_tokens):
    parser.error("capture-attention-steps must be below output-tokens")
  if args.capture_attention_layers and args.paired_blocks:
    parser.error(
        "attention boundary capture is correctness-only and requires "
        "paired-blocks=0")
  if (bool(args.capture_attention_history_layers) !=
      bool(args.capture_attention_history_steps)):
    parser.error(
        "capture-attention-history-layers and "
        "capture-attention-history-steps are required together")
  if (args.capture_attention_history_steps and
      max(args.capture_attention_history_steps) >= args.output_tokens):
    parser.error(
        "capture-attention-history-steps must be below output-tokens")
  if args.capture_attention_history_layers and args.paired_blocks:
    parser.error(
        "attention history capture is correctness-only and requires "
        "paired-blocks=0")
  if args.paired_blocks < 0:
    parser.error("paired-blocks must be non-negative")
  if args.capture_prefill_profiles and args.paired_blocks:
    parser.error(
        "prefill profile capture is correctness-only and requires "
        "paired-blocks=0")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeouts and polling intervals must be positive")
  if args.min_available_gib < 0 or args.abort_below_available_gib < 0:
    parser.error("memory thresholds must be non-negative")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed the preflight threshold")
  if args.worker_transient_scope and not SYSTEMD_RUN.is_file():
    parser.error(f"worker transient scope requires {SYSTEMD_RUN}")
  if args.prefill_chunk_tokens != FROZEN_CHUNK_TOKENS:
    parser.error(
        f"the accepted resident schedule is frozen at {FROZEN_CHUNK_TOKENS}")
  if args.worker_config is None:
    custom_needed = (
        args.candidate_policy == "custom" or
        (args.candidate_policy == "auto" and
         any(bucket in CUSTOM_BUCKETS for bucket in args.buckets)))
    if args.pack_gdn_state and not custom_needed:
      parser.error("pack-gdn-state requires a custom candidate lane")
    if args.prefill_history_capacity is not None:
      if not custom_needed:
        parser.error(
            "prefill-history-capacity requires a custom candidate lane")
      if args.custom_composition not in (
          "unified", "stock_prefill", "exact_phase",
          "adaptive_i8_fixed"):
        parser.error(
            "prefill-history-capacity requires unified or a stock-prefill "
            "composition")
      if args.prefill_history_capacity < max(args.buckets):
        parser.error(
            "prefill-history-capacity must cover the largest bucket")
      if (args.prefill_history_capacity < GRAPH.RING_CAPACITY or
          args.prefill_history_capacity &
              (args.prefill_history_capacity - 1)):
        parser.error(
            "prefill-history-capacity must be a power of two no smaller "
            f"than {GRAPH.RING_CAPACITY}")
    if (args.exact_history_capacity is not None and
        args.exact_history_capacity_slack_tokens is not None):
      parser.error(
          "exact-history-capacity and exact-history-capacity-slack-tokens are "
          "mutually exclusive")
    exact_capacity_configured = (
        args.exact_history_capacity is not None or
        args.exact_history_capacity_slack_tokens is not None)
    if bool(args.exact_history_layers) != exact_capacity_configured:
      parser.error(
          "exact-history-layers and one exact-history capacity mode are "
          "required together")
    if args.exact_history_layers:
      if not custom_needed:
        parser.error("exact-history-layers requires a custom candidate lane")
      if args.custom_composition not in (
          "unified", "stock_prefill", "exact_phase",
          "adaptive_i8_fixed", "direct_i8_fixed"):
        parser.error(
            "exact-history-layers requires unified or a fixed custom "
            "attention composition")
      if not set(args.exact_history_layers).issubset(args.target_layers):
        parser.error("exact-history-layers must be a subset of target-layers")
      if (args.exact_history_capacity_slack_tokens is not None and
          args.exact_history_capacity_slack_tokens < args.output_tokens):
        parser.error(
            "exact-history-capacity-slack-tokens must cover all requested "
            "decode tokens")
      for bucket in args.buckets:
        base_history_capacity = prefill_history_capacity_for_bucket(
            args, bucket)
        exact_history_capacity = exact_history_capacity_for_bucket(
            args, bucket)
        if exact_history_capacity <= base_history_capacity:
          parser.error(
              "each exact-history capacity must be greater than its base "
              "prefill history capacity")
        if exact_history_capacity < bucket + args.output_tokens:
          parser.error(
              "each exact-history capacity must cover its prompt plus all "
              "requested decode tokens")
    if args.alias_linear_state_assign and not custom_needed:
      parser.error(
          "alias-linear-state-assign requires a custom candidate lane")
    if args.direct_ssm_state_assign and not custom_needed:
      parser.error(
          "direct-ssm-state-assign requires a custom candidate lane")
    if args.fuse_fixed_fc and not custom_needed:
      parser.error("fuse-fixed-fc requires a custom candidate lane")
    if args.fuse_router_shared_triple and not custom_needed:
      parser.error(
          "fuse-router-shared-triple requires a custom candidate lane")
    if args.fuse_router_shared_pair and not custom_needed:
      parser.error(
          "fuse-router-shared-pair requires a custom candidate lane")
    if (args.fuse_router_shared_triple and args.fuse_router_shared_pair):
      parser.error(
          "fuse-router-shared-triple and pair are mutually exclusive")
    if ((args.fuse_router_shared_triple or
         args.fuse_router_shared_pair) and
        (args.fixed_fc_manager_direct or args.fuse_fixed_fc)):
      parser.error(
          "router-shared fusion is exclusive with fixed-FC routes")
    if args.fixed_fc_cohorts is not None and not args.fuse_fixed_fc:
      parser.error("fixed-fc-cohorts requires --fuse-fixed-fc")
    if args.fixed_fc_manager_direct and not custom_needed:
      parser.error("fixed-fc-manager-direct requires a custom candidate lane")
    if args.fixed_fc_manager_direct and args.fuse_fixed_fc:
      parser.error(
          "fixed-fc-manager-direct is exclusive with graph fixed-FC fusion")
    if args.candidate_fc_stable_prepare_fastpath and not custom_needed:
      parser.error(
          "candidate-fc-stable-prepare-fastpath requires a custom candidate")
    if args.lm_head_i8q4 and not custom_needed:
      parser.error("lm-head-i8q4 requires a custom candidate lane")
    if args.lm_head_i8q1 and not custom_needed:
      parser.error("lm-head-i8q1 requires a custom candidate lane")
    if args.lm_head_i8q1 and args.lm_head_i8q4:
      parser.error("lm-head-i8q1 and lm-head-i8q4 are mutually exclusive")
    if args.lm_head_i8q1_gated_exact and not args.lm_head_i8q1:
      parser.error("lm-head-i8q1-gated-exact requires --lm-head-i8q1")
    if (args.lm_head_i8q1_gated_exact_affine_q4 and
        not args.lm_head_i8q1_gated_exact):
      parser.error(
          "lm-head-i8q1-gated-exact-affine-q4 requires "
          "--lm-head-i8q1-gated-exact")
    if args.lm_head_i8q1_gated_q4 and not args.lm_head_i8q1:
      parser.error("lm-head-i8q1-gated-q4 requires --lm-head-i8q1")
    if (args.lm_head_i8q1_greedy_local2 and
        not (args.lm_head_i8q1_gated_exact or
             args.lm_head_i8q1_gated_q4)):
      parser.error(
          "lm-head-i8q1-greedy-local2 requires one full-logit fallback")
    if (args.lm_head_device_greedy_feedback and
        not args.lm_head_i8q1_greedy_local2):
      parser.error(
          "lm-head-device-greedy-feedback requires "
          "--lm-head-i8q1-greedy-local2")
    if (args.lm_head_token_only_feedback and
        not args.lm_head_i8q1_greedy_local2):
      parser.error(
          "lm-head-token-only-feedback requires "
          "--lm-head-i8q1-greedy-local2")
    if (args.lm_head_token_only_feedback and
        args.lm_head_device_greedy_feedback):
      parser.error(
          "LM-head device-greedy and token-only feedback are mutually "
          "exclusive")
    if args.lm_head_i8q1_gated_exact and args.lm_head_i8q1_gated_q4:
      parser.error("binary LM-head fallbacks are mutually exclusive")
    if (args.decode_chunk256_layers and
        not set(args.decode_chunk256_layers).issubset(args.target_layers)):
      parser.error(
          "decode-chunk256-layers must be a subset of target-layers")
    if args.decode_chunk256_layers and not custom_needed:
      parser.error("decode-chunk256-layers requires a custom candidate lane")
    if (args.decode_f32_numerator_layers and
        not set(args.decode_f32_numerator_layers).issubset(
            args.target_layers)):
      parser.error(
          "decode-f32-numerator-layers must be a subset of target-layers")
    if args.decode_f32_numerator_layers and not custom_needed:
      parser.error(
          "decode-f32-numerator-layers requires a custom candidate lane")
    if (args.capture_attention_layers and
        not set(args.capture_attention_layers).issubset(args.target_layers)):
      parser.error(
          "capture-attention-layers must be a subset of target-layers")
    if args.capture_attention_layers and not custom_needed:
      parser.error("attention boundary capture requires a custom candidate")
    if (args.capture_attention_history_layers and
        not set(args.capture_attention_history_layers).issubset(
            args.target_layers)):
      parser.error(
          "capture-attention-history-layers must be a subset of "
          "target-layers")
    if args.capture_attention_history_layers and not custom_needed:
      parser.error("attention history capture requires a custom candidate")
    if (args.decode_dual256_layers and
        not set(args.decode_dual256_layers).issubset(args.target_layers)):
      parser.error(
          "decode-dual256-layers must be a subset of target-layers")
    if args.decode_dual256_layers and not custom_needed:
      parser.error("decode-dual256-layers requires a custom candidate lane")
    if (args.decode_stock256_layers and
        not set(args.decode_stock256_layers).issubset(args.target_layers)):
      parser.error(
          "decode-stock256-layers must be a subset of target-layers")
    if args.decode_stock256_layers and not custom_needed:
      parser.error(
          "decode-stock256-layers requires a custom candidate lane")
    if (args.decode_stock_score_layers and
        not set(args.decode_stock_score_layers).issubset(
            args.target_layers)):
      parser.error(
          "decode-stock-score-layers must be a subset of target-layers")
    if args.decode_stock_score_layers and not custom_needed:
      parser.error(
          "decode-stock-score-layers requires a custom candidate lane")
    if (args.decode_stock_partition_layers and
        not set(args.decode_stock_partition_layers).issubset(
            args.target_layers)):
      parser.error(
          "decode-stock-partition-layers must be a subset of target-layers")
    if args.decode_stock_partition_layers and not custom_needed:
      parser.error(
          "decode-stock-partition-layers requires a custom candidate lane")
    if (args.decode_stock_micro_layers and
        not set(args.decode_stock_micro_layers).issubset(args.target_layers)):
      parser.error(
          "decode-stock-micro-layers must be a subset of target-layers")
    if args.decode_stock_micro_layers and not custom_needed:
      parser.error(
          "decode-stock-micro-layers requires a custom candidate lane")
    if (not set(args.decode_page_sparse_layers).issubset(
            args.decode_stock_micro_layers) or
        args.decode_page_sparse_layers and
            args.custom_composition != "exact_phase"):
      parser.error(
          "decode-page-sparse-layers requires an exact-phase stock-micro "
          "subset")
    if args.exact_phase_context_partition4 and (
        args.custom_composition != "exact_phase" or
        not args.decode_stock_micro_layers):
      parser.error(
          "exact-phase-context-partition4 requires exact-phase stock-micro "
          "decode")
    if not set(args.exact_phase_dual_cohort_buckets).issubset(args.buckets):
      parser.error(
          "exact-phase-dual-cohort-buckets must be a subset of --buckets")
    if args.exact_phase_dual_cohort_buckets and (
        args.custom_composition != "exact_phase" or
        not args.decode_stock_micro_layers):
      parser.error(
          "exact-phase-dual-cohort-buckets requires exact-phase stock-micro "
          "decode")
    if any(
        candidate_path(args, bucket) != "hot_cold_custom"
        for bucket in args.exact_phase_dual_cohort_buckets):
      parser.error(
          "exact-phase-dual-cohort-buckets must select custom candidate lanes")
    if args.exact_phase_dual_cohort_buckets and (
        args.exact_phase_context_partition4 or
        args.decode_page_sparse_layers):
      parser.error(
          "exact-phase dual cohort is incompatible with partition/page routes")
    if args.fuse_qk_rope_layout and (
        not custom_needed or args.custom_composition != "exact_phase" or
        tuple(args.target_layers) != FULL_ATTENTION_LAYERS or
        tuple(args.decode_stock_micro_layers) != FULL_ATTENTION_LAYERS or
        set(args.exact_phase_dual_cohort_buckets) != set(args.buckets)):
      parser.error(
          "fuse-qk-rope-layout requires the all-ten exact-phase dual-cohort "
          "custom carrier on every selected bucket")
    if not set(args.decode_stock_micro_layers).issubset(
        args.exact_history_layers):
      parser.error(
          "decode-stock-micro-layers requires exact history for every layer")
    decode_arithmetic_sets = (
        set(args.decode_chunk256_layers),
        set(args.decode_f32_numerator_layers),
        set(args.decode_dual256_layers),
        set(args.decode_stock256_layers),
        set(args.decode_stock_score_layers),
        set(args.decode_stock_partition_layers),
        set(args.decode_stock_micro_layers))
    if any(decode_arithmetic_sets[left] & decode_arithmetic_sets[right]
           for left in range(len(decode_arithmetic_sets))
           for right in range(left + 1, len(decode_arithmetic_sets))):
      parser.error(
          "decode arithmetic layer subsets must be pairwise disjoint")
    if (args.adaptive_attention_exact_layers and
        args.custom_composition not in (
            "adaptive_i8_fixed", "direct_i8_fixed")):
      parser.error(
          "adaptive-attention-exact-layers requires a fixed compressed "
          "attention composition")
    if not set(args.adaptive_attention_exact_layers).issubset(
        args.target_layers):
      parser.error(
          "adaptive-attention-exact-layers must be a subset of "
          "target-layers")
    if (args.adaptive_attention_exact_layers and
        set(args.adaptive_attention_exact_layers) ==
            set(args.target_layers)):
      parser.error(
          "adaptive-attention-exact-layers must leave at least one "
          "compressed layer")
    if args.custom_composition == "adaptive_i8_fixed":
      if set(args.exact_history_layers) != set(args.target_layers):
        parser.error(
            "adaptive_i8_fixed requires exact history on every target layer")
      if not set(args.adaptive_attention_v16_layers).issubset(
          args.target_layers):
        parser.error(
            "adaptive-attention-v16-layers must be a subset of target-layers")
      if not set(args.adaptive_attention_high_topk_layers).issubset(
          args.target_layers):
        parser.error(
            "adaptive-attention-high-topk-layers must be a subset of "
            "target-layers")
      if not set(args.adaptive_attention_key_exact_layers).issubset(
          args.target_layers):
        parser.error(
            "adaptive-attention-key-exact-layers must be a subset of "
            "target-layers")
      residual1_layers = (
          set(args.adaptive_attention_key_residual1_layers) |
          set(args.adaptive_attention_value_residual1_layers))
      packed_kv_layers = set(
          args.adaptive_attention_packed_kv_layers)
      if not residual1_layers.issubset(args.target_layers):
        parser.error(
            "adaptive residual1 layers must be a subset of target-layers")
      if not packed_kv_layers.issubset(args.target_layers):
        parser.error(
            "adaptive packed K/V layers must be a subset of target-layers")
      if bool(packed_kv_layers) != bool(
          args.adaptive_attention_packed_kv_variant):
        parser.error(
            "adaptive packed K/V layers and variant are required together")
      if (set(args.adaptive_attention_exact_layers) &
          set(args.adaptive_attention_v16_layers)):
        parser.error(
            "adaptive exact layers and V16 layers must be disjoint")
      if (set(args.adaptive_attention_exact_layers) &
          set(args.adaptive_attention_high_topk_layers)):
        parser.error(
            "adaptive exact and high-top-k layers must be disjoint")
      if (set(args.adaptive_attention_exact_layers) &
          set(args.adaptive_attention_key_exact_layers)):
        parser.error(
            "adaptive exact and key-exact layers must be disjoint")
      if set(args.adaptive_attention_exact_layers) & residual1_layers:
        parser.error(
            "adaptive exact layers and residual1 layers must be disjoint")
      if (set(args.adaptive_attention_v16_layers) & residual1_layers):
        parser.error("adaptive V16 and residual1 layers must be disjoint")
      if (set(args.adaptive_attention_key_exact_layers) &
          (set(args.adaptive_attention_v16_layers) | residual1_layers)):
        parser.error(
            "adaptive key-exact, V16, and residual1 layers must be disjoint")
      if (packed_kv_layers &
          (set(args.adaptive_attention_v16_layers) |
           set(args.adaptive_attention_key_exact_layers) | residual1_layers |
           set(args.adaptive_attention_exact_layers))):
        parser.error(
            "adaptive packed K/V layers must be disjoint from exact, V16, "
            "key-exact, and residual1 layers")
      if (args.adaptive_attention_high_topk_layers and
          args.adaptive_attention_high_topk ==
              args.adaptive_attention_topk):
        parser.error(
            "adaptive high top-k must differ from the base top-k")
      topk_by_layer = {
          layer: (args.adaptive_attention_high_topk
                  if layer in args.adaptive_attention_high_topk_layers else
                  args.adaptive_attention_topk)
          for layer in args.target_layers
      }
      if any(topk_by_layer[layer] != 512
             for layer in args.adaptive_attention_v16_layers):
        parser.error(
            "adaptive-attention-v16-layers requires top-k 512")
      if any(topk_by_layer[layer] not in (256, 512)
             for layer in residual1_layers):
        parser.error("adaptive residual1 layers require top-k 256 or 512")
      if any(topk_by_layer[layer] != 256
             for layer in args.adaptive_attention_key_exact_layers):
        parser.error("adaptive key-exact layers require top-k 256")
      if any(topk_by_layer[layer] not in (256, 512)
             for layer in packed_kv_layers):
        parser.error("adaptive packed K/V layers require top-k 256 or 512")
      if (args.adaptive_attention_packed_kv_variant != "k7v8" and
          any(topk_by_layer[layer] == 512 for layer in packed_kv_layers)):
        parser.error("only packed K7/V8 currently admits top-k 512")
      if any(decode_arithmetic_sets):
        parser.error(
            "adaptive_i8_fixed is exclusive with legacy decode variants")
    elif (args.adaptive_attention_topk != 512 or
          args.adaptive_attention_high_topk_layers or
          args.adaptive_attention_v16_layers or
          args.adaptive_attention_key_exact_layers or
          args.adaptive_attention_key_residual1_layers or
          args.adaptive_attention_value_residual1_layers or
          args.adaptive_attention_packed_kv_layers or
          args.adaptive_attention_packed_kv_variant or
          (args.adaptive_attention_exact_layers and
           args.custom_composition != "direct_i8_fixed")):
      parser.error(
          "adaptive attention variants require adaptive_i8_fixed")
    if custom_needed and args.candidate_gpu_plugin is None:
      parser.error("custom candidate policy requires --candidate-gpu-plugin")
    if args.prime_candidate_exact_decode_shape:
      if not custom_needed:
        parser.error(
            "prime-candidate-exact-decode-shape requires a custom candidate")
      if args.custom_composition != "exact_phase":
        parser.error(
            "prime-candidate-exact-decode-shape requires exact_phase")
      if args.no_warmup:
        parser.error(
            "prime-candidate-exact-decode-shape requires the product warmup")
      if args.self_bind_hot_states:
        parser.error(
            "prime-candidate-exact-decode-shape requires reset-owned states")
    if args.candidate_impls_cache_capacity is not None:
      if args.candidate_impls_cache_capacity < 0:
        parser.error("candidate-impls-cache-capacity must be non-negative")
      if not args.prime_candidate_exact_decode_shape:
        parser.error(
            "candidate-impls-cache-capacity requires the exact decode prime")
    if (args.capture_attention_history_layers and
        not set(args.capture_attention_history_layers).issubset(
            args.exact_history_layers)):
      parser.error(
          "attention history capture requires every selected layer in "
          "--exact-history-layers")
    if (args.candidate_gpu_plugin is not None and
        not args.candidate_gpu_plugin.is_file()):
      parser.error(
          f"candidate GPU plugin does not exist: {args.candidate_gpu_plugin}")
  if args.out_dir is None and args.worker_config is None and not args.plan_only:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-hot-cold-product-{stamp}"
  return args


candidate_path = PRODUCT_POLICY.candidate_path
timing_lm_head_policy = PRODUCT_POLICY.timing_lm_head_policy


def prefill_history_capacity_for_bucket(
    args: argparse.Namespace, bucket: int) -> int:
  if args.prefill_history_capacity is not None:
    return args.prefill_history_capacity
  if args.custom_composition in ("direct_i8_fixed", "adaptive_i8_fixed"):
    return GRAPH.RING_CAPACITY
  return max(2 * FROZEN_CHUNK_TOKENS, bucket)


def exact_history_capacity_for_bucket(
    args: argparse.Namespace, bucket: int) -> int | None:
  if args.exact_history_capacity is not None:
    return args.exact_history_capacity
  if args.exact_history_capacity_slack_tokens is None:
    return None
  return (
      prefill_history_capacity_for_bucket(args, bucket) +
      args.exact_history_capacity_slack_tokens)


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
  materialized = {
      row["case_id"]: row for row in load_jsonl(
          args.materialization_dir / "materialized-prompts.jsonl")}
  sentinel_specs = {
      row["id"]: row for row in load_jsonl(args.sentinel_specs)}
  cases = []
  for bucket in args.buckets:
    suffix = CASE_SUFFIX[bucket]
    for prompt_set in args.prompt_sets:
      case_id = f"{prompt_set}_{suffix}"
      if prompt_set == "filler":
        path = args.filler_dir / f"filler_{suffix}.txt"
        expected_answer = None
        expected_tokens = bucket
      else:
        source = materialized.get(case_id)
        if source is None:
          raise ValueError(f"materialization missing {case_id}")
        path = ROOT / str(source["materialized_prompt_path"])
        expected_tokens = int(source["observed_prompt_tokens"])
        if expected_tokens != bucket:
          raise ValueError(f"{case_id}: materialized token count mismatch")
        if sha256_file(path) != source["prompt_file_sha256"]:
          raise ValueError(f"{case_id}: materialized prompt digest mismatch")
        expected_answer = (
            sentinel_specs[case_id]["expected_answer"]
            if prompt_set == "sentinel" else None)
      if not path.is_file():
        raise ValueError(f"prompt missing: {path}")
      cases.append({
          "bucket": bucket,
          "candidate_path": candidate_path(args, bucket),
          "capture_all_correctness_logits": (
              args.capture_all_correctness_logits),
          "capture_lm_head_hidden": args.capture_lm_head_hidden,
          "capture_attention_layers": list(args.capture_attention_layers),
          "capture_attention_steps": list(args.capture_attention_steps),
          "capture_attention_history_layers": list(
              args.capture_attention_history_layers),
          "capture_attention_history_steps": list(
              args.capture_attention_history_steps),
          "custom_composition": args.custom_composition,
          "adaptive_attention_topk": args.adaptive_attention_topk,
          "adaptive_attention_high_topk_layers": list(
              args.adaptive_attention_high_topk_layers),
          "adaptive_attention_high_topk": (
              args.adaptive_attention_high_topk),
          "adaptive_attention_v16_layers": list(
              args.adaptive_attention_v16_layers),
          "adaptive_attention_key_exact_layers": list(
              args.adaptive_attention_key_exact_layers),
          "adaptive_attention_key_residual1_layers": list(
              args.adaptive_attention_key_residual1_layers),
          "adaptive_attention_value_residual1_layers": list(
              args.adaptive_attention_value_residual1_layers),
          "adaptive_attention_packed_kv_layers": list(
              args.adaptive_attention_packed_kv_layers),
          "adaptive_attention_packed_kv_variant": (
              args.adaptive_attention_packed_kv_variant),
          "adaptive_attention_exact_layers": list(
              args.adaptive_attention_exact_layers),
          "fuse_fixed_fc": args.fuse_fixed_fc,
          "fixed_fc_cohorts": (
              list(args.fixed_fc_cohorts or FIXED_FC_COHORTS)
              if args.fuse_fixed_fc else []),
          "fixed_fc_manager_direct": args.fixed_fc_manager_direct,
          "fixed_fc_manager_scope": (
              args.fixed_fc_manager_scope
              if args.fixed_fc_manager_direct else "all"),
          "lm_head_i8q4": args.lm_head_i8q4,
          "lm_head_i8q1": args.lm_head_i8q1,
          "lm_head_i8q1_gated_exact": args.lm_head_i8q1_gated_exact,
          "lm_head_i8q1_gated_exact_affine_q4": (
              args.lm_head_i8q1_gated_exact_affine_q4),
          "lm_head_i8q1_gated_q4": args.lm_head_i8q1_gated_q4,
          "lm_head_i8q1_greedy_local2": args.lm_head_i8q1_greedy_local2,
          "lm_head_device_greedy_feedback": (
              args.lm_head_device_greedy_feedback),
          "lm_head_token_only_feedback": (
              args.lm_head_token_only_feedback),
          **timing_lm_head_policy(args, bucket),
          "fuse_linear_conv_state": args.fuse_linear_conv_state,
          "fuse_qk_rope_layout": args.fuse_qk_rope_layout,
          "fuse_router_shared_triple": args.fuse_router_shared_triple,
          "fuse_router_shared_pair": args.fuse_router_shared_pair,
          "direct_ssm_state_assign": args.direct_ssm_state_assign,
          "pack_gdn_state": args.pack_gdn_state,
          "prime_candidate_exact_decode_shape": (
              args.prime_candidate_exact_decode_shape),
          "candidate_impls_cache_capacity": (
              args.candidate_impls_cache_capacity
              if candidate_path(args, bucket) == "hot_cold_custom" else None),
          "candidate_dq_realloc_fastpath": (
              candidate_path(args, bucket) == "hot_cold_custom"),
          "candidate_fc_stable_prepare_fastpath": (
              args.candidate_fc_stable_prepare_fastpath and
              candidate_path(args, bucket) == "hot_cold_custom"),
          "prefill_history_capacity": (
              prefill_history_capacity_for_bucket(args, bucket)),
          "exact_history_layers": list(args.exact_history_layers),
          "exact_history_capacity": exact_history_capacity_for_bucket(
              args, bucket),
          "alias_linear_state_assign": args.alias_linear_state_assign,
          "linear_state_alias_scope": (
              args.linear_state_alias_scope
              if args.alias_linear_state_assign else "none"),
          "case_id": case_id,
          "expected_answer": expected_answer,
          "expected_tokens": expected_tokens,
          "path": str(path.resolve()),
          "prompt_set": prompt_set,
          "sha256": sha256_file(path),
          "target_layers": list(args.target_layers),
          "decode_chunk256_layers": list(args.decode_chunk256_layers),
          "decode_f32_numerator_layers": list(
              args.decode_f32_numerator_layers),
          "decode_dual256_layers": list(args.decode_dual256_layers),
          "decode_stock256_layers": list(args.decode_stock256_layers),
          "decode_stock_score_layers": list(
              args.decode_stock_score_layers),
          "decode_stock_partition_layers": list(
              args.decode_stock_partition_layers),
          "decode_stock_micro_layers": list(
              args.decode_stock_micro_layers),
          "decode_page_sparse_layers": list(
              args.decode_page_sparse_layers),
          "exact_phase_context_partition4": (
              args.exact_phase_context_partition4),
          "exact_phase_dual_cohort": (
              bucket in args.exact_phase_dual_cohort_buckets),
      })
  return cases


def jsonable(value: Any) -> Any:
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, bytes):
    return value.hex()
  if isinstance(value, dict):
    return {str(key): jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple, set)):
    return [jsonable(item) for item in value]
  if hasattr(value, "value"):
    try:
      return jsonable(value.value)
    except Exception:
      pass
  if hasattr(value, "tolist"):
    try:
      return jsonable(value.tolist())
    except Exception:
      pass
  return str(value)


def merge_exact_compressed_source_summaries(
    exact: dict[str, Any], compressed: dict[str, Any],
    target_layers: tuple[int, ...], exact_layers: tuple[int, ...],
    composition: str,
) -> dict[str, Any]:
  """Describe one-owner-per-layer exact/compressed heterogeneous graphs."""
  merged = dict(compressed)
  compressed_layers = tuple(
      layer for layer in target_layers if layer not in set(exact_layers))
  names_by_layer = {
      **dict(zip(exact["target_layers"], exact["target_names"])),
      **dict(zip(compressed["target_layers"], compressed["target_names"])),
  }
  adaptive_layers = (
      compressed_layers if composition == "adaptive_i8_fixed" else ())
  direct_i8_layers = (
      compressed_layers if composition == "direct_i8_fixed" else ())
  merged.update({
      "target_layer": target_layers[0] if len(target_layers) == 1 else None,
      "target_layers": list(target_layers),
      "target_names": [names_by_layer[layer] for layer in target_layers],
      "stock_sdpa_count_before": exact["stock_sdpa_count_before"],
      "adaptive_attention_exact_layers": list(exact_layers),
      "adaptive_attention_layers": list(adaptive_layers),
      "adaptive_topk_by_layer": (
          compressed["adaptive_topk_by_layer"]
          if composition == "adaptive_i8_fixed" else {}),
      "direct_i8_compressed_layers": list(direct_i8_layers),
      "decode_stock_micro_layers": [],
      "exact_phase_decode": False,
      "direct_i8_fixed_layout": True,
      "exact_history_layers": list(target_layers),
      "hot_key_shape": None,
      "hot_key_storage_planes": None,
      "hot_storage": (
          "per-layer heterogeneous exact-phase F16 or compressed direct-I8 "
          "state; one custom state owner per layer"),
      "cold_storage": (
          "exact-phase F16 carrier on exact islands; fixed-capacity block32 "
          "I8 K/V plus exact F16 scales on compressed layers"),
      "removed_stock_states": (
          list(exact["removed_stock_states"]) +
          list(compressed["removed_stock_states"])),
      "custom_states": list(GRAPH.custom_state_names(target_layers)),
  })
  for field in (
      "physical_ring_capacity_by_layer", "physical_hot_capacity_by_layer",
      "hot_key_shape_by_layer", "hot_value_shape_by_layer",
      "key_scale_bytes_by_layer",
      "value_scale_bytes_by_layer"):
    merged[field] = {**exact.get(field, {}), **compressed.get(field, {})}
  return merged


def gpu_memory(core: Any, device: str) -> dict[str, Any]:
  try:
    value = jsonable(core.get_property(device, "GPU_MEMORY_STATISTICS"))
    return value if isinstance(value, dict) else {"value": value}
  except Exception as exc:
    return {"error": repr(exc)}


def memory_total(value: dict[str, Any]) -> int | None:
  numbers = [
      int(item) for item in value.values()
      if isinstance(item, (int, float)) and item >= 0]
  return sum(numbers) if numbers else None


def state_schema(request: Any) -> list[dict[str, Any]]:
  rows = []
  for state in request.query_state():
    try:
      tensor = state.state
      rows.append({
          "bytes": int(tensor.byte_size),
          "element_type": str(tensor.element_type),
          "materialized": True,
          "name": str(state.name),
          "shape": [int(value) for value in tensor.shape],
      })
    except Exception as exc:
      rows.append({
          "materialization_error": repr(exc),
          "materialized": False,
          "name": str(state.name),
      })
  return sorted(rows, key=lambda row: row["name"])


def state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  return {
      "byte_count": sum(int(row.get("bytes", 0)) for row in rows),
      "count": len(rows),
      "materialized_count": sum(row.get("materialized") is True for row in rows),
  }


def top8(logits: Any, np: Any) -> list[dict[str, Any]]:
  indices = np.argpartition(logits, -8)[-8:]
  indices = sorted(
      (int(index) for index in indices),
      key=lambda index: float(logits[index]), reverse=True)
  return [{"id": index, "value": float(logits[index])}
          for index in indices]


def make_inputs(
    token_ids: list[int], start: int, total: int,
    attention_mask: Any, beam_idx: Any, np: Any,
) -> dict[str, Any]:
  ids = np.asarray([token_ids], dtype=np.int64)
  count = len(token_ids)
  positions = np.arange(start, start + count, dtype=np.int64)
  return {
      "attention_mask": attention_mask[:, :total],
      "beam_idx": beam_idx,
      "input": ids,
      "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, count),
  }


def runtime_census(compiled: Any) -> dict[str, Any]:
  rows = []
  for node in compiled.get_runtime_model().get_ordered_ops():
    info = {str(key): jsonable(value) for key, value in node.get_rt_info().items()}
    name = str(node.get_friendly_name())
    layer_type = str(info.get("layerType", ""))
    primitive_type = str(info.get("primitiveType", ""))
    if ("iq36" in name.lower() or "attention" in name.lower() or
        "moe" in name.lower() or "moe" in layer_type.lower() or
        "moe" in primitive_type.lower() or
        layer_type in ("CustomGPUPrimitive", "scaled_dot_product_attention",
                       "condition", "If", "MOE3GemmFusedCompressed")):
      rows.append({
          "layer_type": layer_type,
          "name": name,
          "primitive_type": primitive_type,
      })
  return {
      "attention_rows": rows,
      "custom_like_count": sum(
          "iq36" in row["name"].lower() or
          row["layer_type"] in ("CustomGPUPrimitive", "condition", "If")
          for row in rows),
      "stock_sdpa_like_count": sum(
          row["layer_type"] == "scaled_dot_product_attention"
          for row in rows),
      "hot_attention_custom_count": sum(
          row["layer_type"] in ("CustomGPUPrimitive", "condition", "If") and
          row["name"] in {
              f"iq36_hot_attention_layer{layer}"
              for layer in FULL_ATTENTION_LAYERS}
          for row in rows),
      "linear_conv_custom_count": sum(
          row["layer_type"] == "CustomGPUPrimitive" and
          row["name"] in {
              f"iq36_linear_conv_swish_layer{layer}"
              for layer in GRAPH.LINEAR_ATTENTION_LAYERS}
          for row in rows),
      "fixed_fc_custom_count": sum(
          row["layer_type"] == "CustomGPUPrimitive" and
          row["name"].startswith("iq36_fixed_fc")
          for row in rows),
      "qk_rope_layout_custom_count": sum(
          row["layer_type"] == "CustomGPUPrimitive" and
          row["name"].startswith("iq36_qk_rope_layout_layer")
          for row in rows),
      "moe_3gemm_fused_compressed_count": sum(
          row["layer_type"] == "MOE3GemmFusedCompressed" or
          row["name"].startswith("MOE3GemmFusedCompressed") or
          "moe_3gemm_swiglu_opt" in row["primitive_type"]
          for row in rows),
      "moe_3gemm_fused_compressed_rows": [
          row for row in rows
          if row["layer_type"] == "MOE3GemmFusedCompressed" or
          row["name"].startswith("MOE3GemmFusedCompressed") or
          "moe_3gemm_swiglu_opt" in row["primitive_type"]],
  }


def execution_census(request: Any) -> dict[str, Any]:
  rows = HOT.profile_rows(request, attention_only=False)
  executed = [row for row in rows if row.get("status") == "Status.EXECUTED"]
  counts = Counter(row.get("node_type") for row in executed)
  retained_types = {
      "DynamicQuantize", "FullyConnectedCompressed", "IQ36FixedFC1",
      "IQ36FixedFC3", "IQ36FixedFC4", "IQ36HotAttentionGQA",
      "IQ36DecodeChunk256HotAttentionGQA",
      "IQ36StockMicroOwnerHotAttentionGQA",
      "IQ36ExactPhaseHotAttentionGQA",
      "IQ36ExactPhaseDualCohortHotAttentionGQA",
      "IQ36ExactPhasePageSparseHotAttentionGQA",
      "IQ36ExactPhaseContextPartition4HotAttentionGQA",
      "IQ36StockMicroAttentionOracle",
      "IQ36F32NumeratorChunk256HotAttentionGQA",
      "IQ36Stock256PartialsHotAttentionGQA",
      "IQ36StockScoreChunk256HotAttentionGQA",
      "IQ36StockPartitionChunk256HotAttentionGQA",
      "IQ36LinearConvSwish", "IQ36QKRopeLayout",
      "MOE3GemmFusedCompressed",
      "IndirectSDPA", "ScaledDotProductAttention",
  }
  return {
      "attention_boundary_rows": [
          row for row in executed
          if ".self_attn/" in str(row.get("node_name", "")) and
          row.get("node_type") in {
              "Concat", "IQ36QKRopeLayout", "Multiply", "RoPE",
              "StridedSlice", "Transpose"}],
      "executed_type_counts": {
          str(key): int(value) for key, value in sorted(counts.items())},
      "retained_rows": [
          row for row in executed if row.get("node_type") in retained_types],
      "top_rows": sorted(
          executed, key=lambda row: float(row.get("real_time_us", 0.0)),
          reverse=True)[:64],
  }


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise RuntimeError(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  cfg = load_json(config_path)
  compile_only = bool(cfg.get("compile_only", False))
  instantiate_only = bool(cfg.get("instantiate_only", False))
  mode = str(cfg["mode"])
  selected_path = str(cfg["candidate_path"])
  if compile_only and instantiate_only:
    raise ValueError("compile-only and instantiate-only are mutually exclusive")
  if (compile_only or instantiate_only) and (
      mode != "candidate" or selected_path != "hot_cold_custom"):
    raise ValueError(
        "compile/instantiate-only boundary requires the isolated custom "
        "candidate")
  pack_gdn_state = bool(cfg.get("pack_gdn_state", False))
  fixed_fc_manager_direct = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("fixed_fc_manager_direct", False)))
  fixed_fc_manager_scope = str(cfg.get("fixed_fc_manager_scope", "all"))
  lm_head_i8q4 = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("lm_head_i8q4", False)))
  lm_head_i8q1 = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("lm_head_i8q1", False)))
  lm_head_i8q1_greedy_local2 = (
      lm_head_i8q1 and
      bool(cfg.get("lm_head_i8q1_greedy_local2", False)) and
      str(cfg.get("purpose", "")) == "paired_product_timing")
  lm_head_i8q1_gated_exact = (
      lm_head_i8q1 and not lm_head_i8q1_greedy_local2 and
      bool(cfg.get("lm_head_i8q1_gated_exact", False)))
  lm_head_i8q1_gated_exact_affine_q4 = (
      lm_head_i8q1_gated_exact and
      bool(cfg.get("lm_head_i8q1_gated_exact_affine_q4", False)))
  lm_head_device_greedy_feedback = (
      lm_head_i8q1_greedy_local2 and
      bool(cfg.get("lm_head_device_greedy_feedback", False)))
  lm_head_token_only_feedback = (
      lm_head_i8q1_greedy_local2 and
      bool(cfg.get("lm_head_token_only_feedback", False)))
  lm_head_i8q1_gated_q4 = (
      lm_head_i8q1 and not lm_head_i8q1_greedy_local2 and
      bool(cfg.get("lm_head_i8q1_gated_q4", False)))
  if lm_head_i8q1 and lm_head_i8q4:
    raise ValueError("LM-head I8Q1 and I8Q4 providers are mutually exclusive")
  if lm_head_i8q1_gated_exact and lm_head_i8q1_gated_q4:
    raise ValueError("binary LM-head fallbacks are mutually exclusive")
  if (mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("lm_head_i8q1_gated_exact_affine_q4", False)) and
      not lm_head_i8q1_gated_exact):
    raise ValueError(
        "affine-Q4 fallback requires the binary gated-exact LM head")
  if fixed_fc_manager_scope not in ("all", "m1024"):
    raise ValueError(
        f"invalid fixed-FC manager scope: {fixed_fc_manager_scope}")
  if fixed_fc_manager_direct and bool(cfg.get("fuse_fixed_fc", False)):
    raise ValueError(
        "fixed-FC manager-direct mode requires the native fixed-FC graph")
  os.environ.pop("IQ36_GDN_TRANSPOSED_STATE", None)
  if (pack_gdn_state and mode == "candidate" and
      selected_path == "hot_cold_custom"):
    os.environ["IQ36_GDN_TRANSPOSED_STATE"] = "1"

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai
  model_dir = Path(cfg["model_dir"])
  raw = Path(cfg["raw"])
  device = str(cfg["device"])
  target_layers = tuple(int(layer) for layer in cfg["target_layers"])
  if (not target_layers or len(set(target_layers)) != len(target_layers) or
      any(layer not in FULL_ATTENTION_LAYERS for layer in target_layers)):
    raise ValueError(f"invalid target layers: {target_layers}")
  decode_chunk256_layers = tuple(
      int(layer) for layer in cfg.get("decode_chunk256_layers", []))
  if (len(set(decode_chunk256_layers)) != len(decode_chunk256_layers) or
      not set(decode_chunk256_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid decode chunk256 layers: {decode_chunk256_layers}")
  decode_f32_numerator_layers = tuple(
      int(layer) for layer in cfg.get("decode_f32_numerator_layers", []))
  if (len(set(decode_f32_numerator_layers)) !=
      len(decode_f32_numerator_layers) or
      not set(decode_f32_numerator_layers).issubset(target_layers)):
    raise ValueError(
        "invalid decode F32-numerator layers: "
        f"{decode_f32_numerator_layers}")
  decode_dual256_layers = tuple(
      int(layer) for layer in cfg.get("decode_dual256_layers", []))
  if (len(set(decode_dual256_layers)) != len(decode_dual256_layers) or
      not set(decode_dual256_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid decode dual256 layers: {decode_dual256_layers}")
  decode_stock256_layers = tuple(
      int(layer) for layer in cfg.get("decode_stock256_layers", []))
  if (len(set(decode_stock256_layers)) != len(decode_stock256_layers) or
      not set(decode_stock256_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid decode stock256 layers: {decode_stock256_layers}")
  decode_stock_score_layers = tuple(
      int(layer) for layer in cfg.get("decode_stock_score_layers", []))
  if (len(set(decode_stock_score_layers)) !=
      len(decode_stock_score_layers) or
      not set(decode_stock_score_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid decode stock-score layers: {decode_stock_score_layers}")
  decode_stock_partition_layers = tuple(
      int(layer) for layer in cfg.get("decode_stock_partition_layers", []))
  if (len(set(decode_stock_partition_layers)) !=
      len(decode_stock_partition_layers) or
      not set(decode_stock_partition_layers).issubset(target_layers)):
    raise ValueError(
        "invalid decode stock-partition layers: "
        f"{decode_stock_partition_layers}")
  decode_stock_micro_layers = tuple(
      int(layer) for layer in cfg.get("decode_stock_micro_layers", []))
  if (len(set(decode_stock_micro_layers)) !=
      len(decode_stock_micro_layers) or
      not set(decode_stock_micro_layers).issubset(target_layers)):
    raise ValueError(
        "invalid decode stock-micro layers: "
        f"{decode_stock_micro_layers}")
  decode_page_sparse_layers = tuple(
      int(layer) for layer in cfg.get("decode_page_sparse_layers", []))
  if (len(set(decode_page_sparse_layers)) !=
      len(decode_page_sparse_layers) or
      not set(decode_page_sparse_layers).issubset(
          decode_stock_micro_layers)):
    raise ValueError(
        "invalid decode page-sparse layers: "
        f"{decode_page_sparse_layers}")
  decode_arithmetic_sets = (
      set(decode_chunk256_layers), set(decode_f32_numerator_layers),
      set(decode_dual256_layers), set(decode_stock256_layers),
      set(decode_stock_score_layers), set(decode_stock_partition_layers),
      set(decode_stock_micro_layers))
  if any(decode_arithmetic_sets[left] & decode_arithmetic_sets[right]
         for left in range(len(decode_arithmetic_sets))
         for right in range(left + 1, len(decode_arithmetic_sets))):
    raise ValueError("decode arithmetic layers must be pairwise disjoint")
  custom_composition = str(cfg["custom_composition"])
  if custom_composition not in (
      "unified", "phase_branch", "stock_prefill", "exact_phase",
      "direct_i8_fixed", "adaptive_i8_fixed"):
    raise ValueError(f"invalid custom composition: {custom_composition}")
  adaptive_attention_topk = int(cfg.get("adaptive_attention_topk", 512))
  if adaptive_attention_topk not in (128, 252, 256, 512, 1024, 2048):
    raise ValueError(
        f"invalid adaptive attention top-k: {adaptive_attention_topk}")
  if (custom_composition != "adaptive_i8_fixed" and
      adaptive_attention_topk != 512):
    raise ValueError(
        "non-default adaptive attention top-k requires adaptive composition")
  adaptive_attention_high_topk_layers = tuple(
      int(layer) for layer in cfg.get(
          "adaptive_attention_high_topk_layers", []))
  adaptive_attention_high_topk = int(
      cfg.get("adaptive_attention_high_topk", 256))
  if adaptive_attention_high_topk not in (
      128, 252, 256, 512, 1024, 2048):
    raise ValueError(
        "invalid adaptive attention high top-k: "
        f"{adaptive_attention_high_topk}")
  if (len(set(adaptive_attention_high_topk_layers)) !=
      len(adaptive_attention_high_topk_layers) or
      not set(adaptive_attention_high_topk_layers).issubset(target_layers)):
    raise ValueError(
        "invalid adaptive high-top-k layers: "
        f"{adaptive_attention_high_topk_layers}")
  if adaptive_attention_high_topk_layers and (
      custom_composition != "adaptive_i8_fixed" or
      adaptive_attention_high_topk == adaptive_attention_topk):
    raise ValueError(
        "adaptive high-top-k layers require adaptive composition and a "
        "budget different from the base top-k")
  adaptive_topk_by_layer = {
      layer: (adaptive_attention_high_topk
              if layer in adaptive_attention_high_topk_layers else
              adaptive_attention_topk)
      for layer in target_layers
  }
  adaptive_attention_v16_layers = tuple(
      int(layer) for layer in cfg.get("adaptive_attention_v16_layers", []))
  if (len(set(adaptive_attention_v16_layers)) !=
      len(adaptive_attention_v16_layers) or
      not set(adaptive_attention_v16_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid adaptive V16 layers: {adaptive_attention_v16_layers}")
  if adaptive_attention_v16_layers and (
      custom_composition != "adaptive_i8_fixed" or
      any(adaptive_topk_by_layer[layer] != 512
          for layer in adaptive_attention_v16_layers)):
    raise ValueError(
        "adaptive V16 layers require top-512 adaptive composition")
  adaptive_attention_key_exact_layers = tuple(
      int(layer) for layer in cfg.get(
          "adaptive_attention_key_exact_layers", []))
  if (len(set(adaptive_attention_key_exact_layers)) !=
      len(adaptive_attention_key_exact_layers) or
      not set(adaptive_attention_key_exact_layers).issubset(target_layers)):
    raise ValueError(
        "invalid adaptive key-exact layers: "
        f"{adaptive_attention_key_exact_layers}")
  if adaptive_attention_key_exact_layers and (
      custom_composition != "adaptive_i8_fixed" or
      any(adaptive_topk_by_layer[layer] != 256
          for layer in adaptive_attention_key_exact_layers)):
    raise ValueError(
        "adaptive key-exact layers require top-256 adaptive composition")
  adaptive_attention_key_residual1_layers = tuple(
      int(layer) for layer in cfg.get(
          "adaptive_attention_key_residual1_layers", []))
  adaptive_attention_value_residual1_layers = tuple(
      int(layer) for layer in cfg.get(
          "adaptive_attention_value_residual1_layers", []))
  adaptive_attention_packed_kv_layers = tuple(
      int(layer) for layer in cfg.get(
          "adaptive_attention_packed_kv_layers", []))
  adaptive_attention_packed_kv_variant = cfg.get(
      "adaptive_attention_packed_kv_variant")
  residual1_layers = (
      set(adaptive_attention_key_residual1_layers) |
      set(adaptive_attention_value_residual1_layers))
  if (len(set(adaptive_attention_key_residual1_layers)) !=
          len(adaptive_attention_key_residual1_layers) or
      len(set(adaptive_attention_value_residual1_layers)) !=
          len(adaptive_attention_value_residual1_layers) or
      not residual1_layers.issubset(target_layers)):
    raise ValueError("invalid adaptive residual1 layer subset")
  if residual1_layers and (
      custom_composition != "adaptive_i8_fixed" or
      any(adaptive_topk_by_layer[layer] not in (256, 512)
          for layer in residual1_layers)):
    raise ValueError(
        "adaptive residual1 layers require top-256/512 adaptive composition")
  if set(adaptive_attention_v16_layers) & residual1_layers:
    raise ValueError("adaptive V16 and residual1 layers must be disjoint")
  if (set(adaptive_attention_key_exact_layers) &
      (set(adaptive_attention_v16_layers) | residual1_layers)):
    raise ValueError(
        "adaptive key-exact, V16, and residual1 layers must be disjoint")
  if (len(set(adaptive_attention_packed_kv_layers)) !=
          len(adaptive_attention_packed_kv_layers) or
      not set(adaptive_attention_packed_kv_layers).issubset(target_layers)):
    raise ValueError(
        "invalid adaptive packed K/V layers: "
        f"{adaptive_attention_packed_kv_layers}")
  if adaptive_attention_packed_kv_variant not in (
      None, "k6v7", "k7v7", "k7v8", "k8v7"):
    raise ValueError(
        "invalid adaptive packed K/V variant: "
        f"{adaptive_attention_packed_kv_variant}")
  if bool(adaptive_attention_packed_kv_layers) != bool(
      adaptive_attention_packed_kv_variant):
    raise ValueError(
        "adaptive packed K/V layers and variant are required together")
  if adaptive_attention_packed_kv_layers and (
      custom_composition != "adaptive_i8_fixed" or
      any(adaptive_topk_by_layer[layer] not in (256, 512)
          for layer in adaptive_attention_packed_kv_layers)):
    raise ValueError(
        "adaptive packed K/V layers require top-256/512 adaptive composition")
  if (adaptive_attention_packed_kv_variant != "k7v8" and
      any(adaptive_topk_by_layer[layer] == 512
          for layer in adaptive_attention_packed_kv_layers)):
    raise ValueError("only packed K7/V8 currently admits top-512 correction")
  adaptive_attention_exact_layers = tuple(
      int(layer) for layer in cfg.get("adaptive_attention_exact_layers", []))
  if (len(set(adaptive_attention_exact_layers)) !=
      len(adaptive_attention_exact_layers) or
      not set(adaptive_attention_exact_layers).issubset(target_layers)):
    raise ValueError(
        "invalid adaptive exact layers: "
        f"{adaptive_attention_exact_layers}")
  if adaptive_attention_exact_layers and (
      custom_composition not in ("adaptive_i8_fixed", "direct_i8_fixed") or
      set(adaptive_attention_exact_layers) == set(target_layers)):
    raise ValueError(
        "exact compressed-composition islands require adaptive/direct I8 "
        "and must leave at least one compressed layer")
  if (set(adaptive_attention_exact_layers) &
      set(adaptive_attention_v16_layers)):
    raise ValueError("adaptive exact layers and V16 layers must be disjoint")
  if (set(adaptive_attention_exact_layers) &
      set(adaptive_attention_high_topk_layers)):
    raise ValueError(
        "adaptive exact layers and high-top-k layers must be disjoint")
  if (set(adaptive_attention_exact_layers) &
      set(adaptive_attention_key_exact_layers)):
    raise ValueError(
        "adaptive exact and key-exact layers must be disjoint")
  if set(adaptive_attention_exact_layers) & residual1_layers:
    raise ValueError(
        "adaptive exact layers and residual1 layers must be disjoint")
  if (set(adaptive_attention_packed_kv_layers) &
      (set(adaptive_attention_v16_layers) |
       set(adaptive_attention_key_exact_layers) | residual1_layers |
       set(adaptive_attention_exact_layers))):
    raise ValueError(
        "adaptive packed K/V layers must be disjoint from exact, V16, "
        "key-exact, and residual1 layers")
  exact_phase_context_partition4 = bool(
      cfg.get("exact_phase_context_partition4", False))
  if exact_phase_context_partition4 and (
      custom_composition != "exact_phase" or
      not decode_stock_micro_layers):
    raise ValueError(
        "exact-phase context partition4 requires exact stock-micro decode")
  exact_phase_dual_cohort = bool(
      cfg.get("exact_phase_dual_cohort", False))
  if exact_phase_dual_cohort and (
      custom_composition != "exact_phase" or
      not decode_stock_micro_layers):
    raise ValueError(
        "exact-phase dual cohort requires exact stock-micro decode")
  if exact_phase_dual_cohort and (
      exact_phase_context_partition4 or decode_page_sparse_layers):
    raise ValueError(
        "exact-phase dual cohort is incompatible with partition/page routes")
  fuse_qk_rope_layout = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("fuse_qk_rope_layout", False)))
  fuse_router_shared_triple = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("fuse_router_shared_triple", False)))
  fuse_router_shared_pair = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("fuse_router_shared_pair", False)))
  if fuse_router_shared_triple and fuse_router_shared_pair:
    raise ValueError("router-shared triple and pair are mutually exclusive")
  if ((fuse_router_shared_triple or fuse_router_shared_pair) and
      (fixed_fc_manager_direct or bool(cfg.get("fuse_fixed_fc", False)))):
    raise ValueError("router-shared fusion leaked into a fixed-FC route")
  if fuse_qk_rope_layout and (
      custom_composition != "exact_phase" or not exact_phase_dual_cohort or
      target_layers != FULL_ATTENTION_LAYERS or
      decode_stock_micro_layers != FULL_ATTENTION_LAYERS):
    raise ValueError(
        "Q/K RoPE layout requires the all-ten exact-phase dual-cohort carrier")
  prime_candidate_exact_decode_shape = bool(
      cfg.get("prime_candidate_exact_decode_shape", False))
  if prime_candidate_exact_decode_shape and (
      mode != "candidate" or selected_path != "hot_cold_custom"):
    raise ValueError("exact decode shape prime leaked outside custom candidate")
  if (prime_candidate_exact_decode_shape and
      custom_composition != "exact_phase"):
    raise ValueError("exact decode shape prime requires exact_phase")
  if (prime_candidate_exact_decode_shape and
      not bool(cfg.get("warmup", True))):
    raise ValueError("exact decode shape prime requires warmup")
  candidate_impls_cache_capacity_raw = cfg.get(
      "candidate_impls_cache_capacity")
  candidate_impls_cache_capacity = (
      int(candidate_impls_cache_capacity_raw)
      if candidate_impls_cache_capacity_raw is not None else None)
  if (candidate_impls_cache_capacity is not None and
      (candidate_impls_cache_capacity < 0 or
       mode != "candidate" or selected_path != "hot_cold_custom")):
    raise ValueError(
        "implementation cache capacity leaked outside custom candidate")
  if (candidate_impls_cache_capacity is not None and
      not prime_candidate_exact_decode_shape):
    raise ValueError("implementation cache capacity requires exact prime")
  dq_realloc_fastpath_env = os.environ.get(
      "IQ36_GPU_DQ_REALLOC_FASTPATH")
  dq_realloc_fastpath_env_enabled = (
      dq_realloc_fastpath_env is not None and
      dq_realloc_fastpath_env != "" and
      not dq_realloc_fastpath_env.startswith("0"))
  candidate_dq_realloc_fastpath = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      dq_realloc_fastpath_env_enabled)
  if (dq_realloc_fastpath_env_enabled and
      not candidate_dq_realloc_fastpath):
    raise ValueError(
        "DQ reallocation fast path leaked outside custom candidate")
  fc_stable_prepare_fastpath_env = os.environ.get(
      "IQ36_GPU_FC_STABLE_PREP_FASTPATH")
  fc_stable_prepare_fastpath_env_enabled = (
      fc_stable_prepare_fastpath_env is not None and
      fc_stable_prepare_fastpath_env != "" and
      not fc_stable_prepare_fastpath_env.startswith("0"))
  candidate_fc_stable_prepare_fastpath = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      fc_stable_prepare_fastpath_env_enabled)
  if (fc_stable_prepare_fastpath_env_enabled and
      not candidate_fc_stable_prepare_fastpath):
    raise ValueError(
        "FC stable-prepare fast path leaked outside custom candidate")
  bucket = int(cfg["bucket"])
  prefill_history_capacity = int(cfg.get(
      "prefill_history_capacity",
      GRAPH.RING_CAPACITY if custom_composition in (
          "direct_i8_fixed", "adaptive_i8_fixed") else
      max(2 * FROZEN_CHUNK_TOKENS, bucket)))
  exact_history_layers = tuple(
      int(layer) for layer in cfg.get("exact_history_layers", []))
  exact_history_capacity_raw = cfg.get("exact_history_capacity")
  exact_history_capacity = (
      int(exact_history_capacity_raw)
      if exact_history_capacity_raw is not None else None)
  if (len(set(exact_history_layers)) != len(exact_history_layers) or
      not set(exact_history_layers).issubset(target_layers)):
    raise ValueError(f"invalid exact-history layers: {exact_history_layers}")
  if bool(exact_history_layers) != (exact_history_capacity is not None):
    raise ValueError(
        "exact-history layers and capacity must be present together")
  if not set(decode_stock_micro_layers).issubset(exact_history_layers):
    raise ValueError(
        "decode stock-micro layers require exact history")
  if (exact_history_capacity is not None and
      exact_history_capacity <= prefill_history_capacity):
    raise ValueError(
        "exact-history capacity must exceed prefill-history capacity")
  candidate_gpu_plugin = (
      Path(cfg["candidate_gpu_plugin"])
      if cfg.get("candidate_gpu_plugin") else None)
  diagnostic_gpu_plugin = (
      Path(cfg["diagnostic_gpu_plugin"])
      if cfg.get("diagnostic_gpu_plugin") else None)
  dump_sources_path = (
      Path(cfg["dump_sources_path"])
      if cfg.get("dump_sources_path") else None)
  if candidate_gpu_plugin is not None and diagnostic_gpu_plugin is not None:
    raise ValueError(
        "candidate and diagnostic GPU plugins are mutually exclusive")
  if diagnostic_gpu_plugin is not None and (
      mode != "stock" or selected_path != "stock_sdpa"):
    raise ValueError(
        "diagnostic GPU plugin requires the untouched stock graph")
  if dump_sources_path is not None and diagnostic_gpu_plugin is None:
    raise ValueError("source dumping requires a diagnostic GPU plugin")
  if (pack_gdn_state and mode == "candidate" and
      selected_path == "hot_cold_custom" and
      candidate_gpu_plugin is None):
    raise ValueError("packed GDN state requires the candidate GPU plugin")
  plugin_registry = None
  selected_gpu_plugin = candidate_gpu_plugin or diagnostic_gpu_plugin
  if selected_gpu_plugin is not None and not selected_gpu_plugin.is_file():
    raise FileNotFoundError(selected_gpu_plugin)
  if dump_sources_path is not None:
    dump_sources_path.mkdir(parents=True, exist_ok=True)
    # Debug-only options intentionally cannot be set through the public Core
    # property API; the GPU debug registry resolves them from OV_* at finalize.
    os.environ["OV_GPU_DUMP_SOURCES_PATH"] = str(
        dump_sources_path.resolve())
  else:
    os.environ.pop("OV_GPU_DUMP_SOURCES_PATH", None)
  if candidate_impls_cache_capacity is not None:
    # This release-internal GPU property is intentionally unavailable through
    # Core.set_property(). Each lane runs in a fresh process, so set it only in
    # the isolated custom candidate worker.
    os.environ["OV_GPU_IMPLS_CACHE_CAPACITY"] = str(
        candidate_impls_cache_capacity)
  else:
    os.environ.pop("OV_GPU_IMPLS_CACHE_CAPACITY", None)
  if candidate_gpu_plugin is not None:
    if mode != "candidate" or selected_path != "hot_cold_custom":
      raise ValueError("candidate plugin leaked outside the custom worker")
  if selected_gpu_plugin is not None:
    plugin_registry = raw / (
        "candidate-plugins.xml" if candidate_gpu_plugin is not None else
        "diagnostic-plugins.xml")
    plugin_registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(selected_gpu_plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(plugin_registry))
  else:
    core = ov.Core()
  config_before = str(core.get_property(device, "CONFIG_FILE"))
  source_summary = None
  alias_linear_state_assign = (
      mode == "candidate" and selected_path == "hot_cold_custom" and
      bool(cfg.get("alias_linear_state_assign", False)))
  linear_state_alias_scope = (
      str(cfg.get("linear_state_alias_scope", "all"))
      if alias_linear_state_assign else "none")
  if (alias_linear_state_assign and
      linear_state_alias_scope not in ("conv", "ssm", "all")):
    raise ValueError(
        f"unsupported linear-state alias scope: {linear_state_alias_scope}")
  if mode == "candidate" and selected_path == "hot_cold_custom":
    core.set_property(device, {"CONFIG_FILE": cfg["custom_config"]})
    if adaptive_attention_exact_layers:
      compressed_layers = tuple(
          layer for layer in target_layers
          if layer not in set(adaptive_attention_exact_layers))
      source, exact_summary = GRAPH.make_candidate_model(
          core, model_dir, ov, np, adaptive_attention_exact_layers,
          exact_phase_decode=True,
          initialize_hot_states=not bool(
              cfg.get("self_bind_hot_states", False)),
          fixed_cold_capacity=bucket,
          prefill_history_capacity=prefill_history_capacity,
          exact_history_layers=adaptive_attention_exact_layers,
          exact_history_capacity=exact_history_capacity,
          decode_stock_micro_layers=adaptive_attention_exact_layers)
      compressed_kwargs = {}
      if custom_composition == "adaptive_i8_fixed":
        compressed_kwargs = {
            "adaptive_attention_layers": compressed_layers,
            "adaptive_attention_topk": adaptive_attention_topk,
            "adaptive_attention_high_topk_layers": (
                adaptive_attention_high_topk_layers),
            "adaptive_attention_high_topk": adaptive_attention_high_topk,
            "adaptive_attention_v16_layers": adaptive_attention_v16_layers,
            "adaptive_attention_key_exact_layers": (
                adaptive_attention_key_exact_layers),
            "adaptive_attention_key_residual1_layers": (
                adaptive_attention_key_residual1_layers),
            "adaptive_attention_value_residual1_layers": (
                adaptive_attention_value_residual1_layers),
            "adaptive_attention_packed_kv_layers": (
                adaptive_attention_packed_kv_layers),
            "adaptive_attention_packed_kv_variant": (
                adaptive_attention_packed_kv_variant),
        }
      source, compressed_summary = GRAPH.make_candidate_model(
          core, model_dir, ov, np, compressed_layers,
          direct_i8_fixed_layout=True,
          initialize_hot_states=not bool(
              cfg.get("self_bind_hot_states", False)),
          fixed_cold_capacity=bucket,
          prefill_history_capacity=prefill_history_capacity,
          exact_history_layers=compressed_layers,
          exact_history_capacity=exact_history_capacity,
          fuse_fixed_fc=bool(cfg.get("fuse_fixed_fc", False)),
          fixed_fc_cohorts=(
              tuple(cfg["fixed_fc_cohorts"])
              if cfg.get("fixed_fc_cohorts") else None),
          fuse_linear_conv_state=bool(
              cfg.get("fuse_linear_conv_state", False)),
          direct_ssm_state_assign=bool(
              cfg.get("direct_ssm_state_assign", False)),
          source_model=source,
          **compressed_kwargs)
      source_summary = merge_exact_compressed_source_summaries(
          exact_summary, compressed_summary, target_layers,
          adaptive_attention_exact_layers, custom_composition)
    else:
      source, source_summary = GRAPH.make_candidate_model(
          core, model_dir, ov, np, target_layers,
          phase_branch_prefill=custom_composition == "phase_branch",
          stock_prefill_custom_decode=custom_composition == "stock_prefill",
          exact_phase_decode=custom_composition == "exact_phase",
          exact_phase_context_partition4=exact_phase_context_partition4,
          exact_phase_dual_cohort=exact_phase_dual_cohort,
          direct_i8_fixed_layout=custom_composition in (
              "direct_i8_fixed", "adaptive_i8_fixed"),
          adaptive_attention_layers=(
              target_layers
              if custom_composition == "adaptive_i8_fixed" else ()),
          adaptive_attention_topk=adaptive_attention_topk,
          adaptive_attention_high_topk_layers=(
              adaptive_attention_high_topk_layers),
          adaptive_attention_high_topk=adaptive_attention_high_topk,
          adaptive_attention_v16_layers=adaptive_attention_v16_layers,
          adaptive_attention_key_exact_layers=(
              adaptive_attention_key_exact_layers),
          adaptive_attention_key_residual1_layers=(
              adaptive_attention_key_residual1_layers),
          adaptive_attention_value_residual1_layers=(
              adaptive_attention_value_residual1_layers),
          adaptive_attention_packed_kv_layers=(
              adaptive_attention_packed_kv_layers),
          adaptive_attention_packed_kv_variant=(
              adaptive_attention_packed_kv_variant),
          initialize_hot_states=not bool(
              cfg.get("self_bind_hot_states", False)),
          fixed_cold_capacity=bucket,
          prefill_history_capacity=prefill_history_capacity,
          exact_history_layers=exact_history_layers,
          exact_history_capacity=exact_history_capacity,
          fuse_fixed_fc=bool(cfg.get("fuse_fixed_fc", False)),
          fixed_fc_cohorts=(
              tuple(cfg["fixed_fc_cohorts"])
              if cfg.get("fixed_fc_cohorts") else None),
          fuse_linear_conv_state=bool(
              cfg.get("fuse_linear_conv_state", False)),
          fuse_qk_rope_layout=fuse_qk_rope_layout,
          direct_ssm_state_assign=bool(
              cfg.get("direct_ssm_state_assign", False)),
          decode_chunk256_layers=decode_chunk256_layers,
          decode_f32_numerator_layers=decode_f32_numerator_layers,
          decode_dual256_layers=decode_dual256_layers,
          decode_stock256_layers=decode_stock256_layers,
          decode_stock_score_layers=decode_stock_score_layers,
          decode_stock_partition_layers=decode_stock_partition_layers,
          decode_stock_micro_layers=decode_stock_micro_layers,
          decode_page_sparse_layers=decode_page_sparse_layers)
  elif mode in ("stock", "candidate") and selected_path == "stock_sdpa":
    source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  elif mode == "stock":
    source = core.read_model(str(model_dir / "openvino_language_model.xml"))
  else:
    raise ValueError(f"unsupported worker mode/path: {mode}/{selected_path}")
  config_after = str(core.get_property(device, "CONFIG_FILE"))
  embedding_source = core.read_model(
      str(model_dir / "openvino_text_embeddings_model.xml"))
  embedding_parameter = embedding_source.get_parameters()[0]
  embedding_value = embedding_source.get_results()[0].input_value(0)
  inputs_embeds_parameter = next(
      parameter for parameter in source.get_parameters()
      if "inputs_embeds" in parameter.output(0).get_names())
  inputs_embeds_parameter.output(0).replace(embedding_value)
  source.remove_parameter(inputs_embeds_parameter)
  source.add_parameters([embedding_parameter])
  source.validate_nodes_and_infer_types()
  # Generation consumes only the final query position.  Keeping the full
  # [1,8192,248320] FP32 chunk result alive costs about 7.6 GiB and can turn an
  # otherwise resident long-context row into host-memory pressure.  Apply the
  # same exact graph-side projection to stock and candidate before either the
  # correctness logits result or the timing TopK result crosses to the host.
  original_result = source.get_results()[0]
  logits_output = original_result.input_value(0)
  logits_shape = ov.opset13.shape_of(logits_output, "i64")
  sequence_length = ov.opset13.gather(
      logits_shape, ov.opset13.constant(np.array(1, dtype=np.int64)),
      ov.opset13.constant(np.array(0, dtype=np.int64)))
  last_index = ov.opset13.subtract(
      sequence_length, ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_logits = ov.opset13.gather(
      logits_output, last_index,
      ov.opset13.constant(np.array(1, dtype=np.int64)))
  last_logits.set_friendly_name("iq36_product_last_query_logits")
  capture_lm_head_hidden_matrix = bool(
      cfg.get("capture_lm_head_hidden_matrix", False))
  capture_lm_head_hidden = (
      bool(cfg.get("capture_lm_head_hidden", False)) or
      capture_lm_head_hidden_matrix)
  last_hidden = None
  if capture_lm_head_hidden:
    lm_head = next(
        node for node in source.get_ordered_ops()
        if node.get_friendly_name() == LM_HEAD_NAME)
    last_hidden = ov.opset13.gather(
        lm_head.input_value(0), last_index,
        ov.opset13.constant(np.array(1, dtype=np.int64)))
    last_hidden.set_friendly_name("iq36_product_last_query_lm_head_input")
  source.remove_result(original_result)
  timing_token_output = bool(cfg.get("timing_token_output", False))
  if ((lm_head_device_greedy_feedback or lm_head_token_only_feedback) and
      not timing_token_output):
    raise ValueError("GPU LM-head feedback requires timing-token output")
  projected_results = []
  if timing_token_output:
    if lm_head_token_only_feedback:
      token_class = token_or_logits_custom_class(ov)
      token_input = ov.opset13.reshape(
          last_logits,
          ov.opset13.constant(np.array(
              [1, 1, 1, GREEDY_TOP1_VOCABULARY], dtype=np.int64)),
          False)
      token_input.set_friendly_name(
          "iq36_product_greedy_token_or_logits_input")
      token = token_class([token_input.output(0)])
      token.set_friendly_name("iq36_product_greedy_token_or_logits")
      projected_results.append(ov.opset13.result(token.output(0)))
    elif lm_head_device_greedy_feedback:
      partials_class, merge_class = device_greedy_custom_classes(ov)
      greedy_input = ov.opset13.reshape(
          last_logits,
          ov.opset13.constant(np.array(
              [1, 1, 1, GREEDY_TOP1_VOCABULARY], dtype=np.int64)),
          False)
      greedy_input.set_friendly_name("iq36_product_greedy_top1_input")
      partials = partials_class([greedy_input.output(0)])
      partials.set_friendly_name("iq36_product_greedy_top1_partials")
      merge = merge_class([partials.output(0)])
      merge.set_friendly_name("iq36_product_greedy_top1_merge")
      projected_results.append(ov.opset13.result(merge.output(0)))
    else:
      topk = ov.opset13.topk(
          last_logits, ov.opset13.constant(np.array(1, dtype=np.int64)),
          -1, "max", "none", "i64", name="iq36_product_greedy_top1")
      projected_results.append(ov.opset13.result(topk.output(1)))
  else:
    projected_results.append(ov.opset13.result(last_logits.output(0)))
  if last_hidden is not None:
    projected_results.append(ov.opset13.result(last_hidden.output(0)))
  source.add_results(projected_results)

  capture_attention_layers = tuple(
      int(layer) for layer in cfg.get("capture_attention_layers", []))
  capture_attention_steps = {
      int(step) for step in cfg.get("capture_attention_steps", [])}
  if (len(set(capture_attention_layers)) != len(capture_attention_layers) or
      not set(capture_attention_layers).issubset(target_layers)):
    raise ValueError(
        f"invalid attention capture layers: {capture_attention_layers}")
  if any(step < 0 or step >= int(cfg["output_tokens"])
         for step in capture_attention_steps):
    raise ValueError(
        f"invalid attention capture steps: {sorted(capture_attention_steps)}")
  if bool(capture_attention_layers) != bool(capture_attention_steps):
    raise ValueError("attention capture layers and steps must be paired")
  if capture_attention_layers and timing_token_output:
    raise ValueError("attention boundary capture is correctness-only")
  attention_capture_outputs = []
  if capture_attention_layers:
    query_axis = ov.opset13.constant(np.array(2, dtype=np.int64))
    scalar_axis = ov.opset13.constant(np.array(0, dtype=np.int64))
    custom_attention = (
        mode == "candidate" and selected_path == "hot_cold_custom")
    ordered_ops = source.get_ordered_ops()
    for layer in capture_attention_layers:
      if custom_attention:
        node = next(
            value for value in ordered_ops
            if value.get_friendly_name() ==
               f"iq36_hot_attention_layer{layer}")
        phase_branch_capture = custom_composition == "phase_branch"
        if custom_composition == "stock_prefill":
          attention_value = next(
              value.output(0) for value in ordered_ops
              if value.get_friendly_name() ==
                 f"iq36_hybrid_attention_layer{layer}")
        else:
          attention_value = node.output(0 if phase_branch_capture else 1)
        input_offset = 1 if phase_branch_capture else 0
        debug_values = {
            "attention": attention_value,
            "query": node.input_value(input_offset),
            "key": node.input_value(3 + input_offset),
            "value": node.input_value(4 + input_offset),
        }
        if (custom_composition == "adaptive_i8_fixed" and
            layer not in adaptive_attention_exact_layers):
          # Output0 is the plugin-internal packed four-stage workspace.  It is
          # exposed only when an adaptive attention boundary is already being
          # captured, so ordinary product rows retain the one-result surface.
          # One selected checkpoint can then audit actual candidate heaps,
          # union bits, correction partials, completion, and F32 publication.
          debug_values["workspace"] = node.output(0)
      else:
        node = next(
            value for value in ordered_ops
            if value.get_type_name() == "ScaledDotProductAttention" and
               f"layers.{layer}.self_attn" in value.get_friendly_name())
        debug_values = {
            "attention": node.output(0),
            "query": node.input_value(0),
            "key": node.input_value(1),
            "value": node.input_value(2),
        }
      output_indices = {}
      for role, debug_value in debug_values.items():
        debug_shape = ov.opset13.shape_of(debug_value, "i64")
        debug_tokens = ov.opset13.gather(
            debug_shape, query_axis, scalar_axis)
        debug_last_index = ov.opset13.subtract(
            debug_tokens, ov.opset13.constant(np.array(1, dtype=np.int64)))
        debug_last = ov.opset13.gather(
            debug_value, debug_last_index, query_axis)
        debug_last.set_friendly_name(
            f"iq36_product_{role}_last_query_layer{layer}_{mode}")
        source.add_results([ov.opset13.result(debug_last.output(0))])
        output_indices[role] = len(source.outputs) - 1
      attention_capture_outputs.append({
          "layer": layer,
          "output_indices": output_indices,
      })
  capture_attention_history_layers = tuple(
      int(layer) for layer in cfg.get(
          "capture_attention_history_layers", []))
  capture_attention_history_steps = {
      int(step) for step in cfg.get(
          "capture_attention_history_steps", [])}
  if (len(set(capture_attention_history_layers)) !=
      len(capture_attention_history_layers) or
      not set(capture_attention_history_layers).issubset(target_layers)):
    raise ValueError(
        "invalid attention history capture layers: "
        f"{capture_attention_history_layers}")
  if any(step < 0 or step >= int(cfg["output_tokens"])
         for step in capture_attention_history_steps):
    raise ValueError(
        "invalid attention history capture steps: "
        f"{sorted(capture_attention_history_steps)}")
  if (bool(capture_attention_history_layers) !=
      bool(capture_attention_history_steps)):
    raise ValueError("attention history capture layers and steps must be paired")
  if (capture_attention_history_layers and
      not set(capture_attention_history_layers).issubset(
          exact_history_layers)):
    raise ValueError(
        "attention history capture requires exact-history layers")
  source.validate_nodes_and_infer_types()
  compile_config = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
  }
  if mode == "candidate" and selected_path == "hot_cold_custom":
    # With every stock K/V cache removed, the GPU plugin's generic detector
    # no longer recognizes this as an LLM and would otherwise apply the IR's
    # ACTIVATIONS_SCALE_FACTOR=8 option.  Stock LLM execution explicitly skips
    # that transform, so preserve the same policy in the isolated candidate.
    compile_config["ACTIVATIONS_SCALE_FACTOR"] = 0.0
  if (bool(cfg.get("capture_execution_census", False)) or
      bool(cfg.get("capture_prefill_profiles", False))):
    compile_config["PERF_COUNT"] = True
  compile_started = time.perf_counter_ns()
  language = core.compile_model(source, device, compile_config)
  language_compile_ms = (
      time.perf_counter_ns() - compile_started) / 1_000_000.0
  if compile_only or instantiate_only:
    request = language.create_infer_request() if instantiate_only else None
    result = {
        "alias_linear_state_assign": alias_linear_state_assign,
        "candidate_dq_realloc_fastpath": candidate_dq_realloc_fastpath,
        "candidate_fc_stable_prepare_fastpath": (
            candidate_fc_stable_prepare_fastpath),
        "candidate_gpu_plugin": (
            str(candidate_gpu_plugin.resolve())
            if candidate_gpu_plugin is not None else None),
        "candidate_gpu_plugin_sha256": (
            sha256_file(candidate_gpu_plugin)
            if candidate_gpu_plugin is not None else None),
        "candidate_path": selected_path,
        "case_id": cfg["case_id"],
        "compile_config": compile_config,
        "compile_only": compile_only,
        "instantiate_only": instantiate_only,
        "compiler_cache": {
            "dq_realloc_fastpath_env": dq_realloc_fastpath_env,
            "fc_stable_prepare_fastpath_env": (
                fc_stable_prepare_fastpath_env),
            "lm_head_i8q1_gated_exact_env": os.environ.get(
                "IQ36_LM_HEAD_I8Q1_GATED_EXACT"),
            "lm_head_i8q1_gated_exact_affine_q4_env": os.environ.get(
                "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4"),
            "lm_head_i8q1_gated_q4_env": os.environ.get(
                "IQ36_LM_HEAD_I8Q1_GATED_Q4"),
            "lm_head_i8q1_greedy_local2_env": os.environ.get(
                "IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2"),
            "lm_head_i8q1_token_only_env": os.environ.get(
                "IQ36_LM_HEAD_I8Q1_TOKEN_ONLY"),
            "neo_cache_dir": os.environ.get("NEO_CACHE_DIR"),
            "neo_cache_max_size": os.environ.get("NEO_CACHE_MAX_SIZE"),
            "neo_cache_persistent": os.environ.get("NEO_CACHE_PERSISTENT"),
        },
        "config_after": config_after,
        "config_before": config_before,
        "custom_composition": custom_composition,
        "adaptive_attention_topk": adaptive_attention_topk,
        "adaptive_attention_high_topk_layers": list(
            adaptive_attention_high_topk_layers),
        "adaptive_attention_high_topk": adaptive_attention_high_topk,
        "adaptive_attention_v16_layers": list(
            adaptive_attention_v16_layers),
        "adaptive_attention_key_exact_layers": list(
            adaptive_attention_key_exact_layers),
        "adaptive_attention_key_residual1_layers": list(
            adaptive_attention_key_residual1_layers),
        "adaptive_attention_value_residual1_layers": list(
            adaptive_attention_value_residual1_layers),
        "adaptive_attention_packed_kv_layers": list(
            adaptive_attention_packed_kv_layers),
        "adaptive_attention_packed_kv_variant": (
            adaptive_attention_packed_kv_variant),
        "adaptive_attention_exact_layers": list(
            adaptive_attention_exact_layers),
        "exact_history_capacity": exact_history_capacity,
        "exact_history_layers": list(exact_history_layers),
        "fuse_linear_conv_state": bool(
            cfg.get("fuse_linear_conv_state", False)),
        "language_compile_ms": language_compile_ms,
        "linear_state_alias_scope": linear_state_alias_scope,
        "lm_head_i8q4": lm_head_i8q4,
        "lm_head_i8q1": lm_head_i8q1,
        "lm_head_i8q1_gated_exact": lm_head_i8q1_gated_exact,
        "lm_head_i8q1_gated_exact_affine_q4": (
            lm_head_i8q1_gated_exact_affine_q4),
        "lm_head_i8q1_gated_q4": lm_head_i8q1_gated_q4,
        "lm_head_i8q1_greedy_local2": lm_head_i8q1_greedy_local2,
        "lm_head_device_greedy_feedback": (
            lm_head_device_greedy_feedback),
        "lm_head_token_only_feedback": (
            lm_head_token_only_feedback),
        "mode": mode,
        "model_dir": str(model_dir.resolve()),
        "openvino_genai_version": ov_genai.__version__,
        "openvino_runtime_version": ov.get_version(),
        "prefill_history_capacity": prefill_history_capacity,
        "runtime_census": runtime_census(language),
        "source_summary": source_summary,
        "target_layers": list(target_layers),
        "timing_token_output": timing_token_output,
        "worker_created_infer_request": request is not None,
        "worker_executed_inference": False,
    }
    write_json(Path(cfg["result"]), result)
    print(json.dumps({
        "case_id": cfg["case_id"],
        "event": (
            "worker_instantiate_complete"
            if instantiate_only else "worker_compile_complete"),
        "language_compile_ms": language_compile_ms,
        "mode": mode,
    }, sort_keys=True), flush=True)
    return 0

  tokenizer = ov_genai.Tokenizer(str(model_dir))
  prompt_path = Path(cfg["prompt"])
  prompt = prompt_path.read_text(encoding="utf-8")
  prompt_ids = np.asarray(
      tokenizer.encode(prompt).input_ids.data).reshape(-1).astype(np.int64)
  reference_ids = None
  reference_result_path = cfg.get("reference_result")
  teacher_forced_prompt_path_raw = cfg.get("teacher_forced_prompt")
  if reference_result_path and teacher_forced_prompt_path_raw:
    raise ValueError(
        "reference result and teacher-forced prompt are mutually exclusive")
  teacher_forced_prompt = None
  if reference_result_path:
    reference_payload = load_json(Path(cfg["reference_result"]))
    reference_ids = [int(value) for value in reference_payload["generated_token_ids"]]
    if len(reference_ids) != int(cfg["output_tokens"]):
      raise ValueError("reference token count differs from requested output")
  elif teacher_forced_prompt_path_raw:
    teacher_forced_prompt_path = Path(teacher_forced_prompt_path_raw)
    teacher_forced_prompt_text = teacher_forced_prompt_path.read_text(
        encoding="utf-8")
    teacher_forced_prompt_ids = np.asarray(
        tokenizer.encode(teacher_forced_prompt_text).input_ids.data
    ).reshape(-1).astype(np.int64)
    teacher_forced_prompt_offset = int(
        cfg.get("teacher_forced_prompt_offset", 0))
    teacher_forced_prompt_end = (
        teacher_forced_prompt_offset + int(cfg["output_tokens"]))
    if (teacher_forced_prompt_offset < 0 or
        teacher_forced_prompt_end > len(teacher_forced_prompt_ids)):
      raise ValueError(
          "teacher-forced prompt does not cover requested token range")
    reference_ids = [
        int(value) for value in
        teacher_forced_prompt_ids[
            teacher_forced_prompt_offset:teacher_forced_prompt_end]
    ]
    teacher_forced_prompt = {
        "file": relative(teacher_forced_prompt_path),
        "file_sha256": sha256_file(teacher_forced_prompt_path),
        "full_token_count": int(len(teacher_forced_prompt_ids)),
        "selected_offset": teacher_forced_prompt_offset,
        "selected_token_count": len(reference_ids),
        "selected_token_ids_sha256": hashlib.sha256(
            np.asarray(reference_ids, dtype="<u4").tobytes()).hexdigest(),
    }
  request = language.create_infer_request()
  attention_mask = np.ones(
      (1, len(prompt_ids) + int(cfg["output_tokens"])), dtype=np.int64)
  beam_idx = np.zeros((1,), dtype=np.int32)
  attention_checkpoints = []
  attention_history_checkpoints = []

  def capture_attention_checkpoint(step: int | None, outputs: Any) -> None:
    if step is None or step not in capture_attention_steps:
      return
    for row in attention_capture_outputs:
      layer = int(row["layer"])
      tensors = {}
      for role, output_index in row["output_indices"].items():
        value = np.ascontiguousarray(
            np.asarray(outputs[language.output(int(output_index))]),
            dtype="<f4")
        path = raw / f"step{step:04d}-{role}-layer{layer}-{mode}.f32"
        value.tofile(path)
        tensors[role] = {
            "byte_count": path.stat().st_size,
            "file": relative(path),
            "finite": bool(np.isfinite(value).all()),
            "l2_norm": float(np.linalg.norm(value.astype(np.float64))),
            "sha256": sha256_file(path),
            "shape": list(value.shape),
        }
      attention_checkpoints.append({
          "layer": layer,
          "step": step,
          "tensors": tensors,
      })

  def capture_attention_history_checkpoint(step: int | None) -> None:
    attention_history_checkpoints.extend(
        ATTENTION_DIAGNOSTICS.capture_attention_history_checkpoint(
            step=step, selected_steps=capture_attention_history_steps,
            layers=capture_attention_history_layers, mode=mode,
            selected_path=selected_path, request=request, graph=GRAPH,
            raw=raw, root=ROOT, prompt_tokens=len(prompt_ids)))

  def reset_request() -> list[dict[str, Any]]:
    request.reset_state()
    if (mode == "candidate" and selected_path == "hot_cold_custom" and
        bool(cfg.get("self_bind_hot_states", False))):
      return GRAPH.bind_request_owned_hot_states(
          request, target_layers)
    return []

  def infer_tokens(
      tokens: list[int], start: int, total: int,
      capture_step: int | None = None,
  ) -> tuple[Any, int, float, Any]:
    started = time.perf_counter_ns()
    outputs = request.infer(make_inputs(
        tokens, start, total, attention_mask, beam_idx, np))
    output = np.asarray(outputs[language.output(0)])
    if timing_token_output:
      logits = None
      greedy = int(output.reshape(-1)[-1])
    else:
      logits = np.asarray(output, dtype=np.float32).reshape(-1)
      greedy = int(np.argmax(logits))
    hidden = (
        np.asarray(outputs[language.output(1)], dtype=np.float32).reshape(-1)
        if capture_lm_head_hidden else None)
    capture_attention_checkpoint(capture_step, outputs)
    capture_attention_history_checkpoint(capture_step)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return logits, greedy, elapsed_ms, hidden

  warmup = None
  if bool(cfg.get("warmup", True)):
    reset_request()
    warm_count = min(int(cfg["prefill_chunk_tokens"]), len(prompt_ids))
    warm_logits, warm_top1, warm_prefill_ms, _ = infer_tokens(
        [int(value) for value in prompt_ids[:warm_count]], 0, warm_count)
    warm_decode_start = (
        len(prompt_ids) if prime_candidate_exact_decode_shape else warm_count)
    warm_decode_total = warm_decode_start + 1
    _, _, warm_decode_ms, _ = infer_tokens(
        [warm_top1], warm_decode_start, warm_decode_total)
    warmup = {
        "decode_ms": warm_decode_ms,
        "decode_start": warm_decode_start,
        "decode_total": warm_decode_total,
        "exact_decode_shape_primed": prime_candidate_exact_decode_shape,
        "prefill_ms": warm_prefill_ms,
        "tokens": warm_count,
        "top1": warm_top1,
    }

  capture_prefill_profiles = bool(cfg.get("capture_prefill_profiles", False))
  prefill_profile_baseline = (
      {
          "execution_profile": execution_census(request),
          "inference_count": 2 if warmup is not None else 0,
      } if capture_prefill_profiles else None)

  hot_bindings = reset_request()
  if warmup is not None:
    warmup["measured_request_reset"] = True
    warmup["state_reuse"] = "none_after_reset_state"
  memory_before = gpu_memory(core, device)
  checkpoint_steps = set(int(value) for value in cfg["checkpoint_steps"])
  generated: list[int] = []
  checkpoints = []
  chunk_rows = []
  lm_head_hidden_checkpoints = []
  output_tokens = int(cfg["output_tokens"])
  lm_head_hidden_matrix_path = raw / "lm-head-inputs.f16"
  lm_head_hidden_matrix = (
      np.memmap(
          lm_head_hidden_matrix_path, dtype="<f2", mode="w+",
          shape=(output_tokens, LM_HEAD_HIDDEN_SIZE))
      if capture_lm_head_hidden_matrix else None)
  chunk_tokens = int(cfg["prefill_chunk_tokens"])
  logits = None
  first_top1 = None
  prefill_wall_ms = 0.0
  for chunk_start in range(0, len(prompt_ids), chunk_tokens):
    chunk_end = min(chunk_start + chunk_tokens, len(prompt_ids))
    chunk_logits, chunk_top1, chunk_ms, chunk_hidden = infer_tokens(
        [int(value) for value in prompt_ids[chunk_start:chunk_end]],
        chunk_start, chunk_end,
        capture_step=0 if chunk_end == len(prompt_ids) else None)
    chunk_rows.append({
        "end_exclusive": chunk_end,
        "start": chunk_start,
        "wall_ms": chunk_ms,
        **({
            "execution_profile": execution_census(request),
            "profile_inference_count": (
                (2 if warmup is not None else 0) + len(chunk_rows) + 1),
        } if capture_prefill_profiles else {}),
    })
    prefill_wall_ms += chunk_ms
    if chunk_end == len(prompt_ids):
      logits = chunk_logits
      first_top1 = chunk_top1
      hidden = chunk_hidden
  if first_top1 is None:
    raise RuntimeError("empty prompt")
  generated.append(first_top1)

  def capture_checkpoint(step: int, values: Any) -> None:
    if not bool(cfg.get("capture_logits", False)) or step not in checkpoint_steps:
      return
    if values is None:
      raise RuntimeError("logits capture requested from timing-token graph")
    path = raw / f"step{step:04d}-logits.f32"
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    contiguous.tofile(path)
    checkpoints.append({
        "byte_count": path.stat().st_size,
        "file": relative(path),
        "l2_norm": float(np.linalg.norm(values.astype(np.float64))),
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "step": step,
        "top8": top8(values, np),
    })

  def capture_hidden_checkpoint(step: int, values: Any) -> None:
    if not capture_lm_head_hidden:
      return
    if values is None:
      raise RuntimeError("LM-head hidden capture requested without a value")
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    if contiguous.shape != (LM_HEAD_HIDDEN_SIZE,):
      raise RuntimeError(
          f"unexpected LM-head hidden shape: {contiguous.shape}")
    if not np.array_equal(
        contiguous, contiguous.astype("<f2").astype("<f4")):
      raise RuntimeError("LM-head hidden is not exactly F16-valued")
    if lm_head_hidden_matrix is not None:
      lm_head_hidden_matrix[step] = contiguous.astype("<f2")
    if step not in checkpoint_steps:
      return
    path = raw / f"step{step:04d}-lm-head-input.f32"
    contiguous.tofile(path)
    lm_head_hidden_checkpoints.append({
        "byte_count": path.stat().st_size,
        "file": relative(path),
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "step": step,
    })

  capture_checkpoint(0, logits)
  capture_hidden_checkpoint(0, hidden)
  memory_after_prefill = gpu_memory(core, device)
  decode_wall_ms = []
  for step in range(1, output_tokens):
    fed = (
        int(reference_ids[step - 1]) if reference_ids is not None
        else int(generated[step - 1]))
    start = len(prompt_ids) + step - 1
    total = len(prompt_ids) + step
    logits, greedy, wall_ms, hidden = infer_tokens(
        [fed], start, total, capture_step=step)
    generated.append(greedy)
    decode_wall_ms.append(wall_ms)
    capture_checkpoint(step, logits)
    capture_hidden_checkpoint(step, hidden)
  lm_head_hidden_matrix_descriptor = None
  if lm_head_hidden_matrix is not None:
    lm_head_hidden_matrix.flush()
    lm_head_hidden_matrix_descriptor = {
        "byte_count": lm_head_hidden_matrix_path.stat().st_size,
        "dtype": "float16-little-endian",
        "file": relative(lm_head_hidden_matrix_path),
        "sha256": sha256_file(lm_head_hidden_matrix_path),
        "shape": [output_tokens, LM_HEAD_HIDDEN_SIZE],
    }
  memory_after_decode = gpu_memory(core, device)
  decoded = str(tokenizer.decode(generated, skip_special_tokens=False))
  expected_answer = cfg.get("expected_answer")
  states = state_schema(request)
  executed = (
      execution_census(request)
      if bool(cfg.get("capture_execution_census", False)) else None)
  manager_trace_path = os.environ.get("IQ36_FIXED_FC_MANAGER_TRACE_PATH")
  manager_trace_rows = (
      load_jsonl(Path(manager_trace_path))
      if fixed_fc_manager_direct and manager_trace_path and
      Path(manager_trace_path).is_file() else [])
  manager_trace = {
      "path": relative(Path(manager_trace_path))
      if manager_trace_path else None,
      "selection_rows": [
          row for row in manager_trace_rows
          if row.get("provider") == "iq36_fixed_fc_row_major_u8zp"],
      "metadata_prepack_rows": [
          row for row in manager_trace_rows
          if row.get("stage") == "metadata_prepack"],
  }
  lm_head_trace_path = os.environ.get(
      "IQ36_LM_HEAD_I8Q1_TRACE_PATH" if lm_head_i8q1 else
      "IQ36_LM_HEAD_I8Q4_TRACE_PATH")
  lm_head_trace_rows = (
      load_jsonl(Path(lm_head_trace_path))
      if (lm_head_i8q4 or lm_head_i8q1) and lm_head_trace_path and
      Path(lm_head_trace_path).is_file() else [])
  lm_head_trace = {
      "path": relative(Path(lm_head_trace_path))
      if lm_head_trace_path else None,
      "selection_rows": [
          row for row in lm_head_trace_rows
          if row.get("stage") == "selection"],
      "weight_prepack_rows": [
          row for row in lm_head_trace_rows
          if row.get("stage") == "weight_prepack"],
  }
  decode_total_ms = sum(decode_wall_ms)
  measured_decode_tokens = len(decode_wall_ms)
  result = {
      "candidate_path": selected_path,
      "candidate_gpu_plugin": (
          str(candidate_gpu_plugin.resolve())
          if candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(candidate_gpu_plugin)
          if candidate_gpu_plugin is not None else None),
      "candidate_impls_cache_capacity": candidate_impls_cache_capacity,
      "candidate_dq_realloc_fastpath": candidate_dq_realloc_fastpath,
      "candidate_fc_stable_prepare_fastpath": (
          candidate_fc_stable_prepare_fastpath),
      "case_id": cfg["case_id"],
      "attention_checkpoints": attention_checkpoints,
      "attention_history_checkpoints": attention_history_checkpoints,
      "capture_attention_layers": list(capture_attention_layers),
      "capture_attention_steps": sorted(capture_attention_steps),
      "capture_attention_history_layers": list(
          capture_attention_history_layers),
      "capture_attention_history_steps": sorted(
          capture_attention_history_steps),
      "capture_lm_head_hidden": capture_lm_head_hidden,
      "capture_lm_head_hidden_matrix": capture_lm_head_hidden_matrix,
      "compile_config": compile_config,
      "custom_composition": custom_composition,
      "adaptive_attention_topk": adaptive_attention_topk,
      "adaptive_attention_high_topk_layers": list(
          adaptive_attention_high_topk_layers),
      "adaptive_attention_high_topk": adaptive_attention_high_topk,
      "adaptive_attention_v16_layers": list(
          adaptive_attention_v16_layers),
      "adaptive_attention_key_exact_layers": list(
          adaptive_attention_key_exact_layers),
      "adaptive_attention_key_residual1_layers": list(
          adaptive_attention_key_residual1_layers),
      "adaptive_attention_value_residual1_layers": list(
          adaptive_attention_value_residual1_layers),
      "adaptive_attention_packed_kv_layers": list(
          adaptive_attention_packed_kv_layers),
      "adaptive_attention_packed_kv_variant": (
          adaptive_attention_packed_kv_variant),
      "adaptive_attention_exact_layers": list(
          adaptive_attention_exact_layers),
      "exact_phase_context_partition4": exact_phase_context_partition4,
      "exact_phase_dual_cohort": exact_phase_dual_cohort,
      "decode_chunk256_layers": list(decode_chunk256_layers),
      "decode_f32_numerator_layers": list(
          decode_f32_numerator_layers),
      "decode_dual256_layers": list(decode_dual256_layers),
      "decode_stock256_layers": list(decode_stock256_layers),
      "decode_stock_score_layers": list(decode_stock_score_layers),
      "decode_stock_partition_layers": list(
          decode_stock_partition_layers),
      "decode_stock_micro_layers": list(decode_stock_micro_layers),
      "decode_page_sparse_layers": list(decode_page_sparse_layers),
      "exact_history_layers": list(exact_history_layers),
      "exact_history_capacity": exact_history_capacity,
      "fuse_fixed_fc": (
          mode == "candidate" and selected_path == "hot_cold_custom" and
          bool(cfg.get("fuse_fixed_fc", False))),
      "fuse_qk_rope_layout": fuse_qk_rope_layout,
      "fuse_router_shared_triple": fuse_router_shared_triple,
      "fuse_router_shared_pair": fuse_router_shared_pair,
      "fixed_fc_cohorts": (
          list(cfg.get("fixed_fc_cohorts", []))
          if mode == "candidate" and selected_path == "hot_cold_custom" and
          bool(cfg.get("fuse_fixed_fc", False)) else []),
      "fixed_fc_manager_direct": fixed_fc_manager_direct,
      "fixed_fc_manager_scope": (
          fixed_fc_manager_scope if fixed_fc_manager_direct else "all"),
      "fixed_fc_manager_trace": manager_trace,
      "lm_head_i8q4": lm_head_i8q4,
      "lm_head_i8q1": lm_head_i8q1,
      "lm_head_i8q1_gated_exact": lm_head_i8q1_gated_exact,
      "lm_head_i8q1_gated_exact_affine_q4": (
          lm_head_i8q1_gated_exact_affine_q4),
      "lm_head_i8q1_gated_q4": lm_head_i8q1_gated_q4,
      "lm_head_i8q1_greedy_local2": lm_head_i8q1_greedy_local2,
      "lm_head_device_greedy_feedback": lm_head_device_greedy_feedback,
      "lm_head_token_only_feedback": lm_head_token_only_feedback,
      "lm_head_hidden_checkpoints": lm_head_hidden_checkpoints,
      "lm_head_hidden_matrix": lm_head_hidden_matrix_descriptor,
      "lm_head_i8q4_trace": lm_head_trace if lm_head_i8q4 else {
          "path": None, "selection_rows": [], "weight_prepack_rows": []},
      "lm_head_i8q1_trace": lm_head_trace if lm_head_i8q1 else {
          "path": None, "selection_rows": [], "weight_prepack_rows": []},
      "fuse_linear_conv_state": (
          mode == "candidate" and selected_path == "hot_cold_custom" and
          bool(cfg.get("fuse_linear_conv_state", False))),
      "direct_ssm_state_assign": (
          mode == "candidate" and selected_path == "hot_cold_custom" and
          bool(cfg.get("direct_ssm_state_assign", False))),
      "pack_gdn_state": (
          mode == "candidate" and selected_path == "hot_cold_custom" and
          pack_gdn_state),
      "prime_candidate_exact_decode_shape": (
          prime_candidate_exact_decode_shape),
      "alias_linear_state_assign": alias_linear_state_assign,
      "linear_state_alias_scope": linear_state_alias_scope,
      "compiler_cache": {
          "dq_realloc_fastpath_env": dq_realloc_fastpath_env,
          "fc_stable_prepare_fastpath_env": (
              fc_stable_prepare_fastpath_env),
          "lm_head_i8q1_gated_exact_env": os.environ.get(
              "IQ36_LM_HEAD_I8Q1_GATED_EXACT"),
          "lm_head_i8q1_gated_exact_affine_q4_env": os.environ.get(
              "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4"),
          "lm_head_i8q1_gated_q4_env": os.environ.get(
              "IQ36_LM_HEAD_I8Q1_GATED_Q4"),
          "lm_head_i8q1_greedy_local2_env": os.environ.get(
              "IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2"),
          "lm_head_i8q1_token_only_env": os.environ.get(
              "IQ36_LM_HEAD_I8Q1_TOKEN_ONLY"),
          "gpu_impls_cache_capacity_env": os.environ.get(
              "OV_GPU_IMPLS_CACHE_CAPACITY"),
          "neo_cache_dir": os.environ.get("NEO_CACHE_DIR"),
          "neo_cache_max_size": os.environ.get("NEO_CACHE_MAX_SIZE"),
          "neo_cache_persistent": os.environ.get("NEO_CACHE_PERSISTENT"),
      },
      "config_after": config_after,
      "config_before": config_before,
      "decode_measured_token_count": measured_decode_tokens,
      "decode_tokens_s": (
          measured_decode_tokens / (decode_total_ms / 1000.0)),
      "decode_total_ms": decode_total_ms,
      "decode_wall_ms": decode_wall_ms,
      "decoded_text": decoded,
      "distribution_checkpoints": checkpoints,
      "embedding_compile_ms": 0.0,
      "embedding_execution": "fused_into_language_graph_on_gpu",
      "expected_answer": expected_answer,
      "generated_token_count": len(generated),
      "generated_token_ids": generated,
      "generated_token_ids_sha256": hashlib.sha256(
          np.asarray(generated, dtype="<u4").tobytes()).hexdigest(),
      "gpu_memory": {
          "after_decode": memory_after_decode,
          "after_prefill": memory_after_prefill,
          "before": memory_before,
      },
      "hot_bindings": hot_bindings,
      "hot_state_self_bind_skipped": bool(
          not cfg.get("self_bind_hot_states", False)),
      "input_token_count": int(len(prompt_ids)),
      "input_token_ids_sha256": hashlib.sha256(
          np.asarray(prompt_ids, dtype="<u4").tobytes()).hexdigest(),
      "language_compile_ms": language_compile_ms,
      "logits_projection": "last_query_before_host_output",
      "mode": mode,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": ov.get_version(),
      "output_tokens": output_tokens,
      "prefill_history_capacity": prefill_history_capacity,
      "prefill_chunk_count": len(chunk_rows),
      "prefill_chunk_tokens": chunk_tokens,
      "prefill_chunks": chunk_rows,
      "prefill_profile_baseline": prefill_profile_baseline,
      "prefill_tokens_s": len(prompt_ids) / (prefill_wall_ms / 1000.0),
      "prefill_wall_ms": prefill_wall_ms,
      "prompt_sha256": sha256_file(prompt_path),
      "reference_result": cfg.get("reference_result"),
      "runtime_census": runtime_census(language),
      "execution_census": executed,
      "same_infer_request": True,
      "sentinel_pass": (
          expected_answer in decoded if isinstance(expected_answer, str)
          else None),
      "source_summary": source_summary,
      "state_schema_after": states,
      "state_summary_after": state_summary(states),
      "teacher_forced": reference_ids is not None,
      "teacher_forced_from_stock": reference_result_path is not None,
      "teacher_forced_prompt": teacher_forced_prompt,
      "target_layers": list(target_layers),
      "timing_token_output": timing_token_output,
      "total_wall_ms": prefill_wall_ms + decode_total_ms,
      "tpot_ms": statistics.mean(decode_wall_ms),
      "warmup": warmup,
  }
  write_json(Path(cfg["result"]), result)
  print(json.dumps({
      "case_id": cfg["case_id"],
      "decode_tokens_s": result["decode_tokens_s"],
      "event": "worker_complete",
      "mode": mode,
      "prefill_tokens_s": result["prefill_tokens_s"],
  }, sort_keys=True), flush=True)
  return 0


def proc_meminfo() -> dict[str, int]:
  rows = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.strip().split()
    if fields and fields[0].isdigit():
      rows[key] = int(fields[0]) * 1024
  return rows


def process_memory(pid: int) -> dict[str, int]:
  path = Path(f"/proc/{pid}/status")
  if not path.is_file():
    return {}
  values = {}
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
      key, rest = line.split(":", 1)
      fields = rest.strip().split()
      if fields and fields[0].isdigit():
        values[key] = int(fields[0]) * 1024
  return values


def wait_for_memory(args: argparse.Namespace) -> dict[str, Any]:
  required = int(args.min_available_gib * 1024**3)
  started = time.monotonic()
  while True:
    info = proc_meminfo()
    available = int(info.get("MemAvailable", 0))
    if available >= required:
      return {
          "available_bytes": available,
          "required_bytes": required,
          "waited_seconds": time.monotonic() - started,
      }
    if time.monotonic() - started >= 60.0:
      raise RuntimeError(
          f"available memory {available} remains below preflight {required}")
    time.sleep(2.0)


def stop_process_group(process: subprocess.Popen[Any], first_signal: int) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def worker_scope_unit(worker_dir: Path) -> str:
  digest = hashlib.sha256(
      str(worker_dir.resolve()).encode("utf-8")).hexdigest()[:20]
  return f"iq36-ov-worker-{digest}"


def worker_scope_cgroup_root(worker_dir: Path) -> Path:
  uid = os.getuid()
  return Path(
      f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/"
      f"app.slice/{worker_scope_unit(worker_dir)}.scope")


def run_worker(
    args: argparse.Namespace, worker_dir: Path, config: dict[str, Any],
) -> dict[str, Any]:
  worker_dir.mkdir(parents=True, exist_ok=args.resume)
  cache = worker_dir / "neo-cache"
  cache.mkdir(exist_ok=args.resume)
  result_path = worker_dir / "worker-result.json"
  config_path = worker_dir / "worker-config.json"
  full_config = {
      **config,
      "pack_gdn_state": (
          args.pack_gdn_state and config.get("mode") == "candidate" and
          config.get("candidate_path") == "hot_cold_custom"),
      "prime_candidate_exact_decode_shape": (
          args.prime_candidate_exact_decode_shape and
          config.get("mode") == "candidate" and
          config.get("candidate_path") == "hot_cold_custom"),
      "candidate_impls_cache_capacity": (
          args.candidate_impls_cache_capacity
          if (config.get("mode") == "candidate" and
              config.get("candidate_path") == "hot_cold_custom") else None),
      "candidate_gpu_plugin": (
          str(args.candidate_gpu_plugin.resolve())
          if (args.candidate_gpu_plugin is not None and
              config.get("mode") == "candidate" and
              config.get("candidate_path") == "hot_cold_custom")
          else None),
      "custom_config": str(args.custom_config.resolve()),
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "raw": str(worker_dir.resolve()),
      "result": str(result_path.resolve()),
  }
  if args.resume and result_path.is_file() and config_path.is_file():
    previous = load_json(config_path)
    run_path = worker_dir / "run.json"
    previous_run = load_json(run_path) if run_path.is_file() else {}
    if previous != full_config:
      raise RuntimeError(f"resume config mismatch: {worker_dir}")
    if previous_run.get("returncode") == 0:
      print(json.dumps({
          "event": "worker_reused", "worker": relative(worker_dir)},
          sort_keys=True), flush=True)
      return {**previous_run, "result": load_json(result_path), "reused": True}
  write_json(config_path, full_config)
  preflight = wait_for_memory(args)
  worker_command = [
      str(args.openvino_python), str(Path(__file__).resolve()),
      "--worker-config", str(config_path.resolve()),
  ]
  use_transient_scope = bool(
      getattr(args, "worker_transient_scope", False))
  scope_unit = worker_scope_unit(worker_dir) if use_transient_scope else None
  command = (
      [
          str(SYSTEMD_RUN), "--user", "--scope", "--quiet", "--collect",
          f"--unit={scope_unit}",
      ] + worker_command
      if use_transient_scope else worker_command)
  environment = os.environ.copy()
  environment.pop("OV_GPU_CONFIG_FILE", None)
  environment.pop("IQ36_GDN_TRANSPOSED_STATE", None)
  environment.pop("IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN", None)
  environment.pop("IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE", None)
  environment.pop("IQ36_GPU_DQ_REALLOC_FASTPATH", None)
  environment.pop("IQ36_GPU_FC_STABLE_PREP_FASTPATH", None)
  environment.pop("IQ36_FIXED_FC_MANAGER_TRACE_PATH", None)
  environment.pop("IQ36_FIXED_FC_MANAGER_SCOPE", None)
  environment.pop("IQ36_ROUTER_SHARED_TRIPLE", None)
  environment.pop("IQ36_ROUTER_SHARED_PAIR", None)
  environment.pop("IQ36_LM_HEAD_I8Q4", None)
  environment.pop("IQ36_LM_HEAD_I8Q4_TRACE_PATH", None)
  environment.pop("IQ36_LM_HEAD_I8Q1", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_TRACE_PATH", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_GATED_EXACT", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_GATED_Q4", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2", None)
  environment.pop("IQ36_LM_HEAD_I8Q1_TOKEN_ONLY", None)
  environment.update({
      "NEO_CACHE_DIR": str(cache.resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if int(full_config.get("host_time_profiling", 0)):
    environment["OV_GPU_HOST_TIME_PROFILING"] = str(
        int(full_config["host_time_profiling"]))
  if (bool(full_config.get("candidate_dq_realloc_fastpath", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_GPU_DQ_REALLOC_FASTPATH"] = "1"
  if (bool(full_config.get("candidate_fc_stable_prepare_fastpath", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_GPU_FC_STABLE_PREP_FASTPATH"] = "1"
  if (bool(full_config.get("alias_linear_state_assign", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN"] = "1"
    environment["IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE"] = str(
        full_config.get("linear_state_alias_scope", "all"))
  if (bool(full_config.get("fixed_fc_manager_direct", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_FIXED_FC_MANAGER_TRACE_PATH"] = str(
        (worker_dir / "fixed-fc-manager-trace.jsonl").resolve())
    environment["IQ36_FIXED_FC_MANAGER_SCOPE"] = str(
        full_config.get("fixed_fc_manager_scope", "all"))
  if (bool(full_config.get("fuse_router_shared_triple", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_ROUTER_SHARED_TRIPLE"] = "1"
  if (bool(full_config.get("fuse_router_shared_pair", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_ROUTER_SHARED_PAIR"] = "1"
  if (bool(full_config.get("lm_head_i8q4", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    environment["IQ36_LM_HEAD_I8Q4"] = "1"
    environment["IQ36_LM_HEAD_I8Q4_TRACE_PATH"] = str(
        (worker_dir / "lm-head-i8q4-trace.jsonl").resolve())
  if (bool(full_config.get("lm_head_i8q1", False)) and
      full_config.get("mode") == "candidate" and
      full_config.get("candidate_path") == "hot_cold_custom"):
    greedy_local2 = (
        bool(full_config.get("lm_head_i8q1_greedy_local2", False)) and
        full_config.get("purpose") == "paired_product_timing")
    environment["IQ36_LM_HEAD_I8Q1"] = "1"
    environment["IQ36_LM_HEAD_I8Q1_TRACE_PATH"] = str(
        (worker_dir / "lm-head-i8q1-trace.jsonl").resolve())
    if (bool(full_config.get("lm_head_i8q1_gated_exact", False)) and
        not greedy_local2):
      environment["IQ36_LM_HEAD_I8Q1_GATED_EXACT"] = "1"
      if bool(full_config.get(
          "lm_head_i8q1_gated_exact_affine_q4", False)):
        environment["IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4"] = "1"
    if (bool(full_config.get("lm_head_i8q1_gated_q4", False)) and
        not greedy_local2):
      environment["IQ36_LM_HEAD_I8Q1_GATED_Q4"] = "1"
    if greedy_local2:
      environment["IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2"] = "1"
      if bool(full_config.get("lm_head_token_only_feedback", False)):
        environment["IQ36_LM_HEAD_I8Q1_TOKEN_ONLY"] = "1"
  stdout_path = worker_dir / "worker.stdout"
  stderr_path = worker_dir / "worker.stderr"
  started = time.monotonic()
  monitor = {
      "process_rss_peak_bytes": 0,
      "process_swap_peak_bytes": 0,
      "sample_count": 0,
      "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0,
  }
  timed_out = False
  memory_guard_tripped = False
  abort_below_bytes = int(args.abort_below_available_gib * 1024**3)
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=stdout_handle,
        stderr=stderr_handle, text=True, start_new_session=True)
    while process.poll() is None:
      elapsed = time.monotonic() - started
      if elapsed > args.timeout_s:
        timed_out = True
        stop_process_group(process, signal.SIGTERM)
        break
      system = proc_meminfo()
      proc = process_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_total = int(system.get("SwapTotal", 0))
      swap_free = int(system.get("SwapFree", 0))
      monitor["sample_count"] += 1
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]), int(proc.get("VmRSS", 0)))
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]), int(proc.get("VmSwap", 0)))
      current_min = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if current_min is None else min(int(current_min), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_total - swap_free)
      if abort_below_bytes and available < abort_below_bytes:
        memory_guard_tripped = True
        stop_process_group(process, signal.SIGINT)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  end_memory = proc_meminfo()
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  oom_text = stderr.lower()
  oom_observed = (
      not memory_guard_tripped and
      (returncode in (-9, 137) or "out of memory" in oom_text or
       "cl_out_of_resources" in oom_text))
  record = {
      "command": command,
      "worker_command": worker_command,
      "worker_transient_scope": {
          "enabled": use_transient_scope,
          "unit": scope_unit,
          "cgroup_root": (
              str(worker_scope_cgroup_root(worker_dir))
              if use_transient_scope else None),
          "resource_limits_changed": False,
      },
      "elapsed_seconds": time.monotonic() - started,
      "end_system_memory": end_memory,
      "environment": {key: environment[key] for key in (
          "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT",
          "OV_GPU_HOST_TIME_PROFILING",
          "IQ36_GPU_DQ_REALLOC_FASTPATH",
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN",
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE",
          "IQ36_FIXED_FC_MANAGER_TRACE_PATH",
          "IQ36_FIXED_FC_MANAGER_SCOPE",
          "IQ36_ROUTER_SHARED_TRIPLE",
          "IQ36_ROUTER_SHARED_PAIR",
          "IQ36_LM_HEAD_I8Q4", "IQ36_LM_HEAD_I8Q4_TRACE_PATH",
          "IQ36_LM_HEAD_I8Q1", "IQ36_LM_HEAD_I8Q1_TRACE_PATH",
          "IQ36_LM_HEAD_I8Q1_GATED_EXACT",
          "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4",
          "IQ36_LM_HEAD_I8Q1_GATED_Q4",
          "IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2",
          "IQ36_LM_HEAD_I8Q1_TOKEN_ONLY")
          if key in environment},
      "memory_preflight": preflight,
      "memory_guard": {
          "abort_below_bytes": abort_below_bytes,
          "tripped": memory_guard_tripped,
      },
      "monitor": monitor,
      "oom_observed": oom_observed,
      "returncode": returncode,
      "timed_out": timed_out,
  }
  write_json(worker_dir / "run.json", record)
  record["result"] = load_json(result_path) if result_path.is_file() else {}
  print(json.dumps({
      "elapsed_seconds": record["elapsed_seconds"],
      "event": "worker_finished",
      "memory_guard_tripped": memory_guard_tripped,
      "oom_observed": oom_observed,
      "returncode": returncode,
      "worker": relative(worker_dir),
  }, sort_keys=True), flush=True)
  return record


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def percentile(values: list[float], probability: float) -> float | None:
  if not values:
    return None
  ordered = sorted(float(value) for value in values)
  index = min(len(ordered) - 1, int(round(probability * (len(ordered) - 1))))
  return ordered[index]


def correctness_for_case(
    case: dict[str, Any], stock_run: dict[str, Any], candidate_run: dict[str, Any],
) -> dict[str, Any]:
  stock = stock_run.get("result", {})
  candidate = candidate_run.get("result", {})
  distributions = (
      ATTENTION_DIAGNOSTICS.distribution_rows(stock, candidate, ROOT)
      if stock and candidate else [])
  attention_boundaries = (
      ATTENTION_DIAGNOSTICS.attention_boundary_rows(
          stock, candidate, ROOT, GRAPH)
      if stock and candidate else [])
  attention_histories = (
      ATTENTION_DIAGNOSTICS.attention_history_rows(
          stock, candidate, ROOT, GRAPH)
      if stock and candidate else [])
  lm_head_hidden_rows = (
      ATTENTION_DIAGNOSTICS.lm_head_hidden_rows(stock, candidate, ROOT)
      if stock and candidate else [])
  klds = [
      float(row["kld_stock_to_candidate"]) for row in distributions
      if finite(row.get("kld_stock_to_candidate"))]
  cosines = [
      float(row["cosine"]) for row in distributions
      if finite(row.get("cosine"))]
  top1_rate = (
      sum(row.get("top1_match") is True for row in distributions)
      / len(distributions) if distributions else 0.0)
  custom_expected = case["candidate_path"] == "hot_cold_custom"
  prime_requested = bool(
      case.get("prime_candidate_exact_decode_shape", False))
  prime_expected = prime_requested and custom_expected
  cache_capacity_requested = case.get("candidate_impls_cache_capacity")
  cache_capacity_expected = (
      int(cache_capacity_requested)
      if cache_capacity_requested is not None and custom_expected else None)
  dq_realloc_fastpath_expected = (
      bool(case.get("candidate_dq_realloc_fastpath", False)) and
      custom_expected)
  fc_stable_prepare_fastpath_expected = (
      bool(case.get("candidate_fc_stable_prepare_fastpath", False)) and
      custom_expected)
  stock_warmup = stock.get("warmup") or {}
  candidate_warmup = candidate.get("warmup") or {}
  selected_fixed_cohorts = tuple(case.get("fixed_fc_cohorts", []))
  manager_direct = bool(case.get("fixed_fc_manager_direct", False))
  lm_head_i8q4 = bool(case.get("lm_head_i8q4", False))
  lm_head_i8q1 = bool(case.get("lm_head_i8q1", False))
  lm_head_i8q1_gated_exact = bool(
      case.get("lm_head_i8q1_gated_exact", False))
  lm_head_i8q1_gated_exact_affine_q4 = bool(
      case.get("lm_head_i8q1_gated_exact_affine_q4", False))
  lm_head_i8q1_gated_q4 = bool(
      case.get("lm_head_i8q1_gated_q4", False))
  lm_head_i8q1_greedy_local2 = bool(
      case.get("lm_head_i8q1_greedy_local2", False))
  lm_head_device_greedy_feedback = bool(
      case.get("lm_head_device_greedy_feedback", False))
  lm_head_token_only_feedback = bool(
      case.get("lm_head_token_only_feedback", False))
  lm_head_lowbit = lm_head_i8q4 or lm_head_i8q1
  qk_rope_layout_expected = bool(
      case.get("fuse_qk_rope_layout", False))
  router_shared_triple_expected = bool(
      case.get("fuse_router_shared_triple", False))
  router_shared_pair_expected = bool(
      case.get("fuse_router_shared_pair", False))
  expected_q1_provider = (
      "iq36_lm_head_q8_group256_f16_sums+"
      "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16+"
      "iq36_lm_head_i8_exact_local_top12_correction_f16+"
      "iq36_lm_head_output_topk8_f16+"
      "iq36_lm_head_topk8_merge_f32+"
      "iq36_lm_head_i8_direct_topk8_correction_f16")
  if lm_head_i8q1_gated_exact_affine_q4:
    expected_q1_provider += (
        "+iq36_lm_head_i8q1_gated_exact_reset_f16"
        "+iq36_lm_head_i8q1_gated_exact_collect_f16"
        "+iq36_lm_head_i8q1_affine_q4_hidden_group_norms_f16"
        "+iq36_lm_head_i8q1_affine_q4_bound_select_f16"
        "+iq36_lm_head_i8_affine_q4_exact_candidates_f16"
        "+iq36_lm_head_i8_gated_exact_matvec_f16"
        "+iq36_lm_head_i8q1_gated_exact_output_topk8_f16"
        "+iq36_lm_head_i8q1_gated_exact_topk8_merge_f32"
        "+iq36_lm_head_i8_gated_exact_topk8_correction_f16")
  elif lm_head_i8q1_gated_exact:
    expected_q1_provider += (
        "+iq36_lm_head_i8q1_gated_exact_reset_f16"
        "+iq36_lm_head_i8q1_gated_exact_collect_f16"
        "+iq36_lm_head_i8_gated_exact_matvec_f16"
        "+iq36_lm_head_i8q1_gated_exact_output_topk8_f16"
        "+iq36_lm_head_i8q1_gated_exact_topk8_merge_f32"
        "+iq36_lm_head_i8_gated_exact_topk8_correction_f16")
  elif lm_head_i8q1_gated_q4:
    expected_q1_provider += (
        "+iq36_lm_head_i8q1_gated_q4_compact_gate_f16"
        "+iq36_lm_head_i8q1_gated_q4_matvec_collect_f16"
        "+iq36_lm_head_i8q1_gated_q4_correction_f16")
  expected_q1_packed_bytes = (
      336225280 if lm_head_i8q1_gated_exact_affine_q4 else
      321326080 if lm_head_i8q1_gated_q4 else 66053120)
  expected_q1_codec = (
      "binary_two_centroid_lloyd5+gated_affine_q4_group128"
      if lm_head_i8q1_gated_exact_affine_q4 else
      "binary_two_centroid_lloyd5+compact_gated_signed_q4"
      if lm_head_i8q1_gated_q4 else "binary_two_centroid_lloyd5")
  expected_q1_adaptive_delta = 8 if lm_head_i8q1_gated_q4 else 0
  expected_q1_adaptive_capacity = (
      16812 if lm_head_i8q1_gated_exact_affine_q4 else
      4096 if lm_head_i8q1_gated_q4 else 0)
  expected_q1_correction_passes = (
      3 if (lm_head_i8q1_gated_exact_affine_q4 or
            lm_head_i8q1_gated_q4) else 2)
  expected_q1_gate_scan_rows = (
      11640 if lm_head_i8q1_gated_q4 else None)
  expected_q1_post_q1_dispatches = (
      3 if lm_head_i8q1_gated_q4 else None)
  manager_scope = str(case.get("fixed_fc_manager_scope", "all"))
  manager_widths = {
      "all": (),
      "m1024": (1024,),
  }[manager_scope]
  expected_fixed_groups = sum(
      FIXED_FC_COHORT_COUNTS[cohort][0]
      for cohort in selected_fixed_cohorts)
  expected_fixed_projections = sum(
      FIXED_FC_COHORT_COUNTS[cohort][0] *
      FIXED_FC_COHORT_COUNTS[cohort][1]
      for cohort in selected_fixed_cohorts)
  expected_fixed_custom_counts: dict[str, int] = {}
  for cohort in selected_fixed_cohorts:
    count, arity = FIXED_FC_COHORT_COUNTS[cohort]
    name = f"IQ36FixedFC{arity}"
    expected_fixed_custom_counts[name] = (
        expected_fixed_custom_counts.get(name, 0) + count)
  full_fixed_fc = selected_fixed_cohorts == FIXED_FC_COHORTS
  source = candidate.get("source_summary") or {}
  fixed = source.get("fixed_fc_summary") or {}
  execution = candidate.get("execution_census") or {}
  execution_counts = execution.get("executed_type_counts") or {}
  attention_boundary_rows = execution.get("attention_boundary_rows") or []
  old_qk_boundary_rows = []
  for row in attention_boundary_rows:
    name = str(row.get("node_name", ""))
    layer = next((value for value in FULL_ATTENTION_LAYERS
                  if f"layers.{value}.self_attn/" in name), None)
    if layer is None:
      continue
    q_transpose = ("/aten::transpose/Transpose_2" if layer == 39
                   else "/aten::transpose/Transpose")
    q_concat = ("/aten::cat/Concat_5" if layer == 39
                else "/aten::cat/Concat_1")
    k_concat = ("/aten::cat/Concat_2" if layer == 39
                else "/aten::cat/Concat_3")
    if ((row.get("node_type") == "Transpose" and name.endswith(
            (q_transpose, "/aten::transpose/Transpose_1"))) or
        (row.get("node_type") == "StridedSlice" and name.endswith(
            ("/aten::slice/Slice", "/aten::slice/Slice_3",
             "/aten::slice/Slice_4", "/aten::slice/Slice_7"))) or
        (row.get("node_type") == "RoPE" and name.endswith(
            ("/aten::add/Add", "/aten::add/Add_1"))) or
        (row.get("node_type") == "Concat" and name.endswith(
            (q_concat, k_concat)))):
      old_qk_boundary_rows.append(row)
  output_boundary_rows = [
      row for row in attention_boundary_rows
      if (row.get("node_type") == "Transpose" and
          str(row.get("node_name", "")).endswith(
              "/aten::transpose/Transpose_3")) or
         (row.get("node_type") == "Multiply" and
          str(row.get("node_name", "")).endswith(
              "/aten::mul/Multiply_6"))
  ]
  attention_execution_count = sum(
      int(execution_counts.get(name, 0)) for name in (
          "IQ36HotAttentionGQA", "IQ36DecodeChunk256HotAttentionGQA",
          "IQ36StockMicroOwnerHotAttentionGQA",
          "IQ36ExactPhaseHotAttentionGQA",
          "IQ36ExactPhaseDualCohortHotAttentionGQA",
          "IQ36ExactPhasePageSparseHotAttentionGQA",
          "IQ36ExactPhaseContextPartition4HotAttentionGQA",
          "IQ36F32NumeratorChunk256HotAttentionGQA",
          "IQ36Dual256HotAttentionGQA",
          "IQ36Stock256PartialsHotAttentionGQA",
          "IQ36StockScoreChunk256HotAttentionGQA",
          "IQ36StockPartitionChunk256HotAttentionGQA"))
  manager_trace = candidate.get("fixed_fc_manager_trace") or {}
  manager_selections = manager_trace.get("selection_rows") or []
  manager_prepacks = manager_trace.get("metadata_prepack_rows") or []
  lm_head_trace = candidate.get(
      "lm_head_i8q1_trace" if lm_head_i8q1 else "lm_head_i8q4_trace") or {}
  lm_head_selections = lm_head_trace.get("selection_rows") or []
  lm_head_prepacks = lm_head_trace.get("weight_prepack_rows") or []
  manager_prefill_tokens = min(case["bucket"], FROZEN_CHUNK_TOKENS)
  manager_selection_shapes = {
      (int(row.get("m", 0)), int(row.get("tokens", 0)))
      for row in manager_selections
  }
  manager_prepack_m_counts = Counter(
      int(row.get("m", 0)) for row in manager_prepacks)
  manager_prepack_node_counts = Counter(
      str(row.get("node", "")) for row in manager_prepacks)
  manager_prepack_bytes = sum(
      int(row.get("scale_bytes", 0)) +
      int(row.get("zero_point_bytes", 0))
      for row in manager_prepacks)
  manager_prepack_multiplicity = (
      2 if candidate.get("warmup") is not None else 1)
  q1_trace_profiles = [{
      "name": "configured",
      "provider": expected_q1_provider,
      "packed_bytes": expected_q1_packed_bytes,
      "codec": expected_q1_codec,
      "adaptive_delta": expected_q1_adaptive_delta,
      "adaptive_capacity": expected_q1_adaptive_capacity,
      "correction_passes": expected_q1_correction_passes,
      "gate_scan_rows": expected_q1_gate_scan_rows,
      "post_q1_dispatches": expected_q1_post_q1_dispatches,
  }]
  if lm_head_i8q1_gated_q4:
    # The accepted non-compact provider remains the performance best.  Keep
    # its exact trace schema valid alongside the compact dispatch experiment;
    # both profiles are strict, so this does not weaken provider isolation.
    q1_trace_profiles.append({
        "name": "noncompact_gated_q4",
        "provider": (
            "iq36_lm_head_q8_group256_f16_sums+"
            "iq36_lm_head_i8q1_rowstripe8_matvec_local_top12_f16+"
            "iq36_lm_head_i8_exact_local_top12_correction_f16+"
            "iq36_lm_head_output_topk8_f16+"
            "iq36_lm_head_topk8_merge_f32+"
            "iq36_lm_head_i8_direct_topk8_correction_f16+"
            "iq36_lm_head_i8q1_gated_exact_reset_f16+"
            "iq36_lm_head_i8q1_gated_exact_collect_f16+"
            "iq36_lm_head_i8q1_gated_q4_matvec_f16+"
            "iq36_lm_head_i8q1_gated_q4_collect_f16+"
            "iq36_lm_head_i8q1_gated_q4_correction_f16+"
            "iq36_lm_head_i8_gated_exact_topk8_correction_f16"),
        "packed_bytes": 321326080,
        "codec": "binary_two_centroid_lloyd5+gated_signed_q4",
        "adaptive_delta": 8,
        "adaptive_capacity": 4096,
        "correction_passes": 4,
        "gate_scan_rows": None,
        "post_q1_dispatches": None,
    })

  def q1_trace_matches(profile: dict[str, Any]) -> bool:
    return (
        len(lm_head_selections) == manager_prepack_multiplicity and
        all(row.get("tokens") == 1 and
            row.get("rows") == 248320 and
            row.get("columns") == 2048 and
            row.get("topk") == 12 and
            row.get("correction_rows") == 11640 and
            row.get("provider") == profile["provider"] and
            row.get("direct_correction_topk") == 8 and
            row.get("adaptive_correction_delta") ==
                profile["adaptive_delta"] and
            row.get("adaptive_correction_capacity") ==
                profile["adaptive_capacity"] and
            row.get("correction_passes") ==
                profile["correction_passes"] and
            row.get("gate_scan_rows") == profile["gate_scan_rows"] and
            row.get("post_q1_dispatches") ==
                profile["post_q1_dispatches"]
            for row in lm_head_selections) and
        len(lm_head_prepacks) == manager_prepack_multiplicity and
        all(row.get("packed_bytes") == profile["packed_bytes"] and
            row.get("packed_allocation") == "usm_device" and
            row.get("codec") == profile["codec"] and
            row.get("exact_correction_topk") == 12 and
            row.get("exact_correction_rows") == 11640 and
            row.get("direct_correction_topk") == 8 and
            row.get("adaptive_correction_delta") ==
                profile["adaptive_delta"] and
            row.get("adaptive_correction_capacity") ==
                profile["adaptive_capacity"] and
            row.get("exact_correction_passes") ==
                profile["correction_passes"] and
            row.get("gate_scan_rows") == profile["gate_scan_rows"] and
            row.get("post_q1_dispatches") ==
                profile["post_q1_dispatches"] and
            row.get("correction_reuses_source_weights") is True
            for row in lm_head_prepacks) and
        sum(row.get("process_cache_hit") is False
            for row in lm_head_prepacks) == 1 and
        sum(row.get("process_cache_hit") is True
            for row in lm_head_prepacks) ==
                manager_prepack_multiplicity - 1)

  matched_q1_trace_profiles = [
      str(profile["name"]) for profile in q1_trace_profiles
      if q1_trace_matches(profile)]
  manager_expected_prepack_counts = {
      width: 40 * manager_prepack_multiplicity
      for width in manager_widths
  }
  manager_expected_prepack_bytes = manager_prepack_multiplicity * sum(
      3_932_160 for _ in manager_widths)
  expected_checkpoints = (
      list(range(int(candidate.get("output_tokens", 0))))
      if (case.get("capture_all_correctness_logits") or
          case.get("fuse_fixed_fc") or manager_direct or lm_head_lowbit) else [
          step for step in CHECKPOINT_STEPS
          if step < int(candidate.get("output_tokens", 0))])
  expected_attention_rows = (
      4 * len(case.get("capture_attention_layers", [])) *
      len(case.get("capture_attention_steps", [])))
  expected_attention_history_rows = (
      2 * len(case.get("capture_attention_history_layers", [])) *
      len(case.get("capture_attention_history_steps", [])))
  expected_lm_head_hidden_rows = (
      len(expected_checkpoints)
      if case.get("capture_lm_head_hidden", False) else 0)
  checks = [
      {"name": "stock_worker_passed", "pass": (
          stock_run.get("returncode") == 0 and not stock_run.get("timed_out"))},
      {"name": "candidate_worker_passed", "pass": (
          candidate_run.get("returncode") == 0 and
          not candidate_run.get("timed_out"))},
      {"name": "no_worker_oom", "pass": (
          stock_run.get("oom_observed") is False and
          candidate_run.get("oom_observed") is False)},
      {"name": "memory_guard_not_tripped", "pass": (
          stock_run.get("memory_guard", {}).get("tripped") is False and
          candidate_run.get("memory_guard", {}).get("tripped") is False)},
      {"name": "prompt_counts_exact", "pass": (
          stock.get("input_token_count") == case["expected_tokens"] ==
          candidate.get("input_token_count"))},
      {"name": "prompt_digests_exact", "pass": (
          stock.get("prompt_sha256") == case["sha256"] ==
          candidate.get("prompt_sha256"))},
      {"name": "output_counts_exact", "pass": (
          stock.get("generated_token_count") ==
          candidate.get("generated_token_count") ==
          stock.get("output_tokens") == candidate.get("output_tokens"))},
      {"name": "last_query_logits_projection_exact", "pass": (
          stock.get("logits_projection") ==
          candidate.get("logits_projection") ==
          "last_query_before_host_output")},
      {"name": "candidate_teacher_forced_from_stock", "pass": (
          candidate.get("teacher_forced_from_stock") is True)},
      {"name": "candidate_exact_decode_shape_prime_isolated_and_reset",
       "pass": (
           stock.get("prime_candidate_exact_decode_shape") is False and
           candidate.get("prime_candidate_exact_decode_shape") is
               prime_expected and
           (not stock_warmup or
            stock_warmup.get("exact_decode_shape_primed") is False) and
           ((not candidate_warmup and not prime_expected) or
            (candidate_warmup.get("exact_decode_shape_primed") is
                 prime_expected and
             candidate_warmup.get("measured_request_reset") is True and
             candidate_warmup.get("state_reuse") ==
                 "none_after_reset_state" and
             (not prime_expected or
              (candidate_warmup.get("decode_start") == case["bucket"] and
               candidate_warmup.get("decode_total") == case["bucket"] + 1)))))},
      {"name": "candidate_impls_cache_capacity_isolated", "pass": (
          stock.get("candidate_impls_cache_capacity") is None and
          stock.get("compiler_cache", {}).get(
              "gpu_impls_cache_capacity_env") is None and
          candidate.get("candidate_impls_cache_capacity") ==
              cache_capacity_expected and
          candidate.get("compiler_cache", {}).get(
              "gpu_impls_cache_capacity_env") ==
              (str(cache_capacity_expected)
               if cache_capacity_expected is not None else None))},
      {"name": "candidate_dq_realloc_fastpath_isolated", "pass": (
          stock.get("candidate_dq_realloc_fastpath") is False and
          stock.get("compiler_cache", {}).get(
              "dq_realloc_fastpath_env") is None and
          candidate.get("candidate_dq_realloc_fastpath") is
              dq_realloc_fastpath_expected and
          candidate.get("compiler_cache", {}).get(
              "dq_realloc_fastpath_env") ==
              ("1" if dq_realloc_fastpath_expected else None))},
      {"name": "candidate_fc_stable_prepare_fastpath_isolated", "pass": (
          stock.get("candidate_fc_stable_prepare_fastpath") is False and
          stock.get("compiler_cache", {}).get(
              "fc_stable_prepare_fastpath_env") is None and
          candidate.get("candidate_fc_stable_prepare_fastpath") is
              fc_stable_prepare_fastpath_expected and
          candidate.get("compiler_cache", {}).get(
              "fc_stable_prepare_fastpath_env") ==
              ("1" if fc_stable_prepare_fastpath_expected else None))},
      {"name": "attention_boundary_capture_complete_and_finite", "pass": (
          len(attention_boundaries) == expected_attention_rows and
          all(row.get("finite") is True and row.get("same_shape") is True
              for row in attention_boundaries)),
       "actual_rows": len(attention_boundaries),
       "expected_rows": expected_attention_rows},
      {"name": "attention_history_capture_complete_and_finite", "pass": (
          len(attention_histories) == expected_attention_history_rows and
          all(row.get("finite") is True and
              row.get("same_logical_shape") is True
              for row in attention_histories)),
       "actual_rows": len(attention_histories),
       "expected_rows": expected_attention_history_rows},
      {"name": "lm_head_hidden_capture_complete_and_finite", "pass": (
          len(lm_head_hidden_rows) == expected_lm_head_hidden_rows and
          all(row.get("finite") is True and row.get("same_shape") is True
              for row in lm_head_hidden_rows)),
       "actual_rows": len(lm_head_hidden_rows),
       "expected_rows": expected_lm_head_hidden_rows},
      {"name": "deterministic_tokens_exact", "pass": (
          bool(stock.get("generated_token_ids")) and
          stock.get("generated_token_ids") ==
          candidate.get("generated_token_ids"))},
      {"name": "stock_sentinel_truth", "pass": (
          case["expected_answer"] is None or stock.get("output_tokens") != 512 or
          stock.get("sentinel_pass") is True),
       "waived_for_non_product_output": stock.get("output_tokens") != 512},
      {"name": "candidate_sentinel_truth", "pass": (
          case["expected_answer"] is None or
          candidate.get("output_tokens") != 512 or
          candidate.get("sentinel_pass") is True)},
      {"name": "distribution_rows_complete", "pass": (
          len(distributions) == len(expected_checkpoints))},
      {"name": "distribution_rows_finite", "pass": (
          bool(distributions) and all(row["finite"] for row in distributions))},
      {"name": "teacher_forced_kld", "pass": (
          bool(klds) and max(klds) <= KLD_MAX),
       "max": max(klds) if klds else None, "threshold": KLD_MAX},
      {"name": "teacher_forced_top1_rate", "pass": top1_rate >= TOP1_MIN,
       "rate": top1_rate, "threshold": TOP1_MIN},
      {"name": "logits_cosine_diagnostic", "pass": (
          bool(cosines) and min(cosines) >= COSINE_MIN),
       "min": min(cosines) if cosines else None, "required": False,
       "threshold": COSINE_MIN},
      {"name": "stock_worker_has_no_custom_config", "pass": (
          not stock.get("config_after") and
          not stock.get("candidate_gpu_plugin"))},
      {"name": "candidate_plugin_isolated_to_custom_path", "pass": (
          (custom_expected and bool(candidate.get("candidate_gpu_plugin")) and
           bool(candidate.get("candidate_gpu_plugin_sha256"))) or
          (not custom_expected and not candidate.get("candidate_gpu_plugin")))},
      {"name": "packed_gdn_state_isolated_to_custom_candidate", "pass": (
          stock.get("pack_gdn_state") is False and
          candidate.get("pack_gdn_state") is (
              case["pack_gdn_state"] if custom_expected else False))},
      {"name": "linear_state_alias_isolated_and_scoped", "pass": (
          stock.get("alias_linear_state_assign") is False and
          stock.get("linear_state_alias_scope") == "none" and
          candidate.get("alias_linear_state_assign") is (
              case["alias_linear_state_assign"] if custom_expected else False) and
          candidate.get("linear_state_alias_scope") == (
              case["linear_state_alias_scope"] if custom_expected else "none"))},
      {"name": "qk_rope_layout_isolated_rewritten_and_executed_exact", "pass": (
          bool(stock.get("fuse_qk_rope_layout", False)) is False and
          stock.get("runtime_census", {}).get(
              "qk_rope_layout_custom_count", 0) == 0 and
          stock.get("execution_census", {}).get(
              "executed_type_counts", {}).get("IQ36QKRopeLayout", 0) == 0 and
          bool(candidate.get("fuse_qk_rope_layout", False)) is (
              qk_rope_layout_expected if custom_expected else False) and
          ((not custom_expected and not source) or
           (custom_expected and
            source.get("fuse_qk_rope_layout") is qk_rope_layout_expected and
            source.get("qk_rope_layout_rewrite_count") == (
                len(FULL_ATTENTION_LAYERS)
                if qk_rope_layout_expected else 0) and
            candidate.get("runtime_census", {}).get(
                "qk_rope_layout_custom_count", 0) == (
                    len(FULL_ATTENTION_LAYERS)
                    if qk_rope_layout_expected else 0) and
            execution_counts.get("IQ36QKRopeLayout", 0) == (
                len(FULL_ATTENTION_LAYERS)
                if qk_rope_layout_expected else 0) and
            (not qk_rope_layout_expected or
             (not old_qk_boundary_rows and
              len(output_boundary_rows) ==
                  2 * len(FULL_ATTENTION_LAYERS)))))),
       "old_qk_boundary_count": len(old_qk_boundary_rows),
       "output_boundary_count": len(output_boundary_rows)},
      {"name": "router_shared_triple_flag_isolated_to_custom_candidate",
       "pass": (
           bool(stock.get("fuse_router_shared_triple", False)) is False and
           bool(candidate.get("fuse_router_shared_triple", False)) is (
               router_shared_triple_expected if custom_expected else False))},
      {"name": "router_shared_pair_flag_isolated_to_custom_candidate",
       "pass": (
           bool(stock.get("fuse_router_shared_pair", False)) is False and
           bool(candidate.get("fuse_router_shared_pair", False)) is (
               router_shared_pair_expected if custom_expected else False))},
      {"name": "lm_head_i8q4_isolated_selected_and_device_packed", "pass": (
          stock.get("lm_head_i8q4") is False and
          candidate.get("lm_head_i8q4") is (
              lm_head_i8q4 if custom_expected else False) and
          (not lm_head_i8q4 or
           (custom_expected and
            len(lm_head_selections) == manager_prepack_multiplicity and
            all(row.get("tokens") == 1 and
                row.get("rows") == 248320 and
                row.get("columns") == 2048 and
                row.get("topk") == 8 and
                row.get("provider") ==
                    "iq36_lm_head_q8_group256_f16+"
                    "iq36_lm_head_i8q4_rowstripe8_matvec_topk8_f16+"
                    "iq36_lm_head_topk8_merge_f32+"
                    "iq36_lm_head_i8_exact_topk8_correction_f16+"
                    "iq36_lm_head_output_topk8_f16+"
                    "iq36_lm_head_topk8_merge_f32+"
                    "iq36_lm_head_i8_direct_topk8_correction_f16+"
                    "iq36_lm_head_output_topk8_f16+"
                    "iq36_lm_head_topk8_merge_f32+"
                    "iq36_lm_head_adaptive_correction_reset+"
                    "iq36_lm_head_adaptive_correction_collect_f16+"
                    "iq36_lm_head_i8_adaptive_correction_f16" and
                row.get("direct_correction_topk") == 8 and
                row.get("adaptive_correction_delta") == 11 and
                row.get("adaptive_correction_capacity") == 4096 and
                row.get("correction_passes") == 3
                for row in lm_head_selections) and
            len(lm_head_prepacks) == manager_prepack_multiplicity and
            all(row.get("packed_bytes") == 255272960 and
                row.get("packed_allocation") == "usm_device" and
                row.get("exact_correction_topk") == 8 and
                row.get("direct_correction_topk") == 8 and
                row.get("adaptive_correction_delta") == 11 and
                row.get("adaptive_correction_capacity") == 4096 and
                row.get("exact_correction_passes") == 3 and
                row.get("correction_reuses_source_weights") is True
                for row in lm_head_prepacks)))),
       "selection_count": len(lm_head_selections),
       "weight_prepack_count": len(lm_head_prepacks),
       "expected_multiplicity": manager_prepack_multiplicity},
      {"name": "lm_head_i8q1_isolated_selected_and_device_packed", "pass": (
          stock.get("lm_head_i8q1") is False and
          stock.get("lm_head_i8q1_gated_exact") is False and
          stock.get("lm_head_i8q1_gated_exact_affine_q4") is False and
          stock.get("lm_head_i8q1_gated_q4") is False and
          stock.get("lm_head_i8q1_greedy_local2") is False and
          stock.get("lm_head_device_greedy_feedback") is False and
          stock.get("lm_head_token_only_feedback") is False and
          stock.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_exact_env") is None and
          stock.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_exact_affine_q4_env") is None and
          stock.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_q4_env") is None and
          stock.get("compiler_cache", {}).get(
              "lm_head_i8q1_greedy_local2_env") is None and
          stock.get("compiler_cache", {}).get(
              "lm_head_i8q1_token_only_env") is None and
          candidate.get("lm_head_i8q1") is (
              lm_head_i8q1 if custom_expected else False) and
          candidate.get("lm_head_i8q1_gated_exact") is (
              lm_head_i8q1_gated_exact if custom_expected else False) and
          candidate.get("lm_head_i8q1_gated_exact_affine_q4") is (
              lm_head_i8q1_gated_exact_affine_q4
              if custom_expected else False) and
          candidate.get("lm_head_i8q1_gated_q4") is (
              lm_head_i8q1_gated_q4 if custom_expected else False) and
          candidate.get("lm_head_i8q1_greedy_local2") is False and
          candidate.get("lm_head_device_greedy_feedback") is False and
          candidate.get("lm_head_token_only_feedback") is False and
          candidate.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_exact_env") ==
              ("1" if custom_expected and lm_head_i8q1_gated_exact else None) and
          candidate.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_exact_affine_q4_env") ==
              ("1" if (custom_expected and
                       lm_head_i8q1_gated_exact_affine_q4) else None) and
          candidate.get("compiler_cache", {}).get(
              "lm_head_i8q1_gated_q4_env") ==
              ("1" if custom_expected and lm_head_i8q1_gated_q4 else None) and
          candidate.get("compiler_cache", {}).get(
              "lm_head_i8q1_greedy_local2_env") is None and
          candidate.get("compiler_cache", {}).get(
              "lm_head_i8q1_token_only_env") is None and
          (not lm_head_i8q1 or
           (custom_expected and bool(matched_q1_trace_profiles)))),
       "selection_count": len(lm_head_selections),
       "weight_prepack_count": len(lm_head_prepacks),
       "expected_multiplicity": manager_prepack_multiplicity,
       "matched_trace_profiles": matched_q1_trace_profiles,
       "affine_q4_requested": lm_head_i8q1_gated_exact_affine_q4,
       "timing_greedy_local2_requested": lm_head_i8q1_greedy_local2,
       "timing_device_greedy_feedback_requested": (
           lm_head_device_greedy_feedback),
       "timing_token_only_feedback_requested": (
           lm_head_token_only_feedback)},
      {"name": "candidate_path_matches_policy", "pass": (
          candidate.get("candidate_path") == case["candidate_path"])},
      {"name": "candidate_custom_graph_exact_or_stock_short", "pass": (
          (custom_expected and source.get("target_layers") ==
           case["target_layers"] and
           candidate.get("target_layers") == case["target_layers"] and
           source.get("decode_chunk256_layers") ==
               case["decode_chunk256_layers"] and
           candidate.get("decode_chunk256_layers") ==
               case["decode_chunk256_layers"] and
           source.get("decode_f32_numerator_layers") ==
               case["decode_f32_numerator_layers"] and
           candidate.get("decode_f32_numerator_layers") ==
               case["decode_f32_numerator_layers"] and
           source.get("decode_dual256_layers") ==
               case["decode_dual256_layers"] and
           candidate.get("decode_dual256_layers") ==
               case["decode_dual256_layers"] and
           source.get("decode_stock256_layers") ==
               case["decode_stock256_layers"] and
           candidate.get("decode_stock256_layers") ==
               case["decode_stock256_layers"] and
           source.get("decode_stock_score_layers") ==
               case["decode_stock_score_layers"] and
           candidate.get("decode_stock_score_layers") ==
               case["decode_stock_score_layers"] and
           source.get("decode_stock_partition_layers") ==
               case["decode_stock_partition_layers"] and
           candidate.get("decode_stock_partition_layers") ==
               case["decode_stock_partition_layers"] and
           source.get("decode_stock_micro_layers") ==
               case["decode_stock_micro_layers"] and
           candidate.get("decode_stock_micro_layers") ==
               case["decode_stock_micro_layers"] and
           source.get("decode_page_sparse_layers") ==
               case["decode_page_sparse_layers"] and
           candidate.get("decode_page_sparse_layers") ==
               case["decode_page_sparse_layers"] and
           source.get("exact_phase_context_partition4") is
               case["exact_phase_context_partition4"] and
           candidate.get("exact_phase_context_partition4") is
               case["exact_phase_context_partition4"] and
           source.get("exact_phase_dual_cohort") is
               case["exact_phase_dual_cohort"] and
           candidate.get("exact_phase_dual_cohort") is
               case["exact_phase_dual_cohort"] and
           source.get("exact_history_layers") ==
               case["exact_history_layers"] and
           candidate.get("exact_history_layers") ==
               case["exact_history_layers"] and
           source.get("exact_history_capacity") ==
               case["exact_history_capacity"] and
           candidate.get("exact_history_capacity") ==
               case["exact_history_capacity"] and
           candidate.get("custom_composition") ==
               case["custom_composition"] and
           source.get("phase_branch_prefill") is
               (case["custom_composition"] == "phase_branch") and
           source.get("stock_prefill_custom_decode") is
               (case["custom_composition"] == "stock_prefill") and
           source.get("exact_phase_decode") is
               (case["custom_composition"] == "exact_phase") and
           source.get("direct_i8_fixed_layout") is
               (case["custom_composition"] in (
                   "direct_i8_fixed", "adaptive_i8_fixed")) and
           source.get("adaptive_attention_layers") == (
               [layer for layer in case["target_layers"]
                if layer not in case["adaptive_attention_exact_layers"]]
               if case["custom_composition"] == "adaptive_i8_fixed" else
               []) and
           source.get("adaptive_topk_by_layer") == (
               {str(layer): (
                    case["adaptive_attention_high_topk"]
                    if layer in case[
                        "adaptive_attention_high_topk_layers"] else
                    case["adaptive_attention_topk"])
                for layer in case["target_layers"]
                if layer not in case["adaptive_attention_exact_layers"]}
               if case["custom_composition"] == "adaptive_i8_fixed" else
               {}) and
           source.get("adaptive_attention_high_topk_layers", []) ==
               case["adaptive_attention_high_topk_layers"] and
           source.get("adaptive_attention_high_topk", 256) ==
               case["adaptive_attention_high_topk"] and
           source.get("adaptive_attention_v16_layers") ==
               case["adaptive_attention_v16_layers"] and
           source.get("adaptive_attention_key_exact_layers") ==
               case["adaptive_attention_key_exact_layers"] and
           source.get("adaptive_attention_key_residual1_layers") ==
               case["adaptive_attention_key_residual1_layers"] and
           source.get("adaptive_attention_value_residual1_layers") ==
               case["adaptive_attention_value_residual1_layers"] and
           source.get("adaptive_attention_packed_kv_layers") ==
               case["adaptive_attention_packed_kv_layers"] and
           source.get("adaptive_attention_packed_kv_variant") ==
               case["adaptive_attention_packed_kv_variant"] and
           source.get("adaptive_attention_exact_layers", []) ==
               case["adaptive_attention_exact_layers"] and
           candidate.get("adaptive_attention_topk") ==
               case["adaptive_attention_topk"] and
           candidate.get("adaptive_attention_high_topk_layers") ==
               case["adaptive_attention_high_topk_layers"] and
           candidate.get("adaptive_attention_high_topk") ==
               case["adaptive_attention_high_topk"] and
           candidate.get("adaptive_attention_v16_layers") ==
               case["adaptive_attention_v16_layers"] and
           candidate.get("adaptive_attention_key_exact_layers") ==
               case["adaptive_attention_key_exact_layers"] and
           candidate.get("adaptive_attention_key_residual1_layers") ==
               case["adaptive_attention_key_residual1_layers"] and
           candidate.get("adaptive_attention_value_residual1_layers") ==
               case["adaptive_attention_value_residual1_layers"] and
           candidate.get("adaptive_attention_packed_kv_layers") ==
               case["adaptive_attention_packed_kv_layers"] and
           candidate.get("adaptive_attention_packed_kv_variant") ==
               case["adaptive_attention_packed_kv_variant"] and
           candidate.get("adaptive_attention_exact_layers") ==
               case["adaptive_attention_exact_layers"] and
           candidate.get("alias_linear_state_assign") is
               case["alias_linear_state_assign"] and
           candidate.get("linear_state_alias_scope") ==
               case["linear_state_alias_scope"] and
           source.get("fuse_linear_conv_state") is
               case["fuse_linear_conv_state"] and
           source.get("direct_ssm_state_assign") is
               case["direct_ssm_state_assign"] and
           source.get("ssm_state_assign_rewrite_count") == (
               len(GRAPH.LINEAR_ATTENTION_LAYERS)
               if case["direct_ssm_state_assign"] else 0) and
           source.get("fuse_fixed_fc") is case["fuse_fixed_fc"] and
           candidate.get("fixed_fc_cohorts") ==
               case["fixed_fc_cohorts"] and
           candidate.get("fixed_fc_manager_direct") is manager_direct and
           candidate.get("fixed_fc_manager_scope") == manager_scope and
           (not case["fuse_fixed_fc"] or
            fixed.get("fixed_fc_selected_cohorts") ==
                case["fixed_fc_cohorts"]) and
           source.get("linear_conv_replacement_count") == (
               len(GRAPH.LINEAR_ATTENTION_LAYERS)
               if case["fuse_linear_conv_state"] else 0) and
           source.get("linear_conv_custom_count_after") == (
               len(GRAPH.LINEAR_ATTENTION_LAYERS)
               if case["fuse_linear_conv_state"] else 0) and
           candidate.get("fuse_linear_conv_state") is
               case["fuse_linear_conv_state"] and
           candidate.get("direct_ssm_state_assign") is
               case["direct_ssm_state_assign"] and
           candidate.get("fuse_fixed_fc") is case["fuse_fixed_fc"] and
           source.get("initialize_hot_states") is
               candidate.get("hot_state_self_bind_skipped") and
           source.get("fixed_cold_capacity") == case["bucket"] and
           source.get("prefill_history_capacity") ==
               case["prefill_history_capacity"] and
           candidate.get("prefill_history_capacity") ==
               case["prefill_history_capacity"] and
           source.get("stock_sdpa_count_after") ==
               len(FULL_ATTENTION_LAYERS) - len(case["target_layers"]) and
           candidate.get("runtime_census", {}).get(
               "hot_attention_custom_count") == len(case["target_layers"]) and
           candidate.get("runtime_census", {}).get(
               "stock_sdpa_like_count") == (
                   len(FULL_ATTENTION_LAYERS) -
                   len(case["target_layers"])) and
           candidate.get("runtime_census", {}).get(
               "linear_conv_custom_count") == (
                   len(GRAPH.LINEAR_ATTENTION_LAYERS)
                   if case["fuse_linear_conv_state"] else 0) and
           bool(candidate.get("config_after"))) or
          (not custom_expected and not candidate.get("config_after") and
           candidate.get("fuse_fixed_fc") is False and
           candidate.get("fuse_linear_conv_state") is False and
           candidate.get("source_summary") is None))},
      {"name": "fixed_fc_source_runtime_and_execution_census_exact",
       "pass": (
           not case["fuse_fixed_fc"] or
           (custom_expected and
            fixed.get("fixed_fc_rewrite_count") == expected_fixed_groups and
            fixed.get("fixed_fc_projection_count") ==
                expected_fixed_projections and
            fixed.get("fixed_fc_custom_counts") ==
                expected_fixed_custom_counts and
            fixed.get("fixed_fc_f16_to_f32_restore_count") ==
                expected_fixed_projections and
            fixed.get("fixed_fc_old_matmuls_remaining") == [] and
            candidate.get("runtime_census", {}).get(
                "fixed_fc_custom_count") == expected_fixed_groups and
            all(execution_counts.get(name, 0) == count
                for name, count in expected_fixed_custom_counts.items()) and
            all(execution_counts.get(name, 0) == 0
                for name in ("IQ36FixedFC1", "IQ36FixedFC3", "IQ36FixedFC4")
                if name not in expected_fixed_custom_counts) and
            attention_execution_count == 10 and
            execution_counts.get("IQ36LinearConvSwish") == 30 and
            execution_counts.get("MOE3GemmFusedCompressed") == 40 and
            ((execution_counts.get("DynamicQuantize") == 1 and
              execution_counts.get("FullyConnectedCompressed") == 1)
             if full_fixed_fc else
             (execution_counts.get("DynamicQuantize", 0) > 1 and
              execution_counts.get("FullyConnectedCompressed", 0) > 1)))),
       "source": fixed,
       "runtime_fixed_fc_count": candidate.get(
           "runtime_census", {}).get("fixed_fc_custom_count"),
       "executed_type_counts": execution_counts},
      {"name": "fixed_fc_manager_selection_and_fallthrough_exact",
       "pass": (
           not manager_direct or
           (custom_expected and not case["fuse_fixed_fc"] and
            candidate.get("fixed_fc_cohorts") == [] and
            candidate.get("fixed_fc_manager_direct") is True and
            source.get("fuse_fixed_fc") is False and not fixed and
            candidate.get("runtime_census", {}).get(
                "fixed_fc_custom_count") == 0 and
            all(execution_counts.get(name, 0) == 0
                for name in (
                    "IQ36FixedFC1", "IQ36FixedFC3", "IQ36FixedFC4")) and
            execution_counts.get("DynamicQuantize") == 161 and
            execution_counts.get("FullyConnectedCompressed") == 331 and
            attention_execution_count == 10 and
            execution_counts.get("IQ36LinearConvSwish") == 30 and
            execution_counts.get("MOE3GemmFusedCompressed") == 40 and
            candidate.get("fixed_fc_manager_scope") == manager_scope and
            len(manager_selections) == len(manager_widths) and
            manager_selection_shapes == {
                (width, manager_prefill_tokens)
                for width in manager_widths} and
            len(manager_prepacks) == sum(
                manager_expected_prepack_counts.values()) and
            manager_prepack_m_counts == manager_expected_prepack_counts and
            len(manager_prepack_node_counts) == 40 * len(manager_widths) and
            set(manager_prepack_node_counts.values()) == {
                manager_prepack_multiplicity} and
            manager_prepack_bytes == manager_expected_prepack_bytes)),
       "scope": manager_scope,
       "selection_count": len(manager_selections),
       "selection_shapes": sorted(manager_selection_shapes),
       "metadata_prepack_count": len(manager_prepacks),
       "metadata_prepack_m_counts": dict(manager_prepack_m_counts),
       "metadata_prepack_unique_nodes": len(manager_prepack_node_counts),
       "metadata_prepack_node_multiplicities": sorted(
           set(manager_prepack_node_counts.values())),
       "metadata_prepack_expected_multiplicity": (
           manager_prepack_multiplicity),
       "metadata_prepack_bytes": manager_prepack_bytes,
       "executed_type_counts": execution_counts,
       "note": ("all provider selections are T>1; the warmed product path "
                "records the exact implementation-local metadata multiplicity; "
                "the final T=1 profile retains the native "
                "161-DQ/331-FC census")},
      {"name": "resident_chunk_schedule_exact", "pass": (
          stock.get("prefill_chunk_tokens") ==
          candidate.get("prefill_chunk_tokens") == FROZEN_CHUNK_TOKENS and
          stock.get("prefill_chunk_count") ==
          candidate.get("prefill_chunk_count") ==
          math.ceil(case["bucket"] / FROZEN_CHUNK_TOKENS))},
      {"name": "same_request_for_all_chunks", "pass": (
          stock.get("same_infer_request") is True and
          candidate.get("same_infer_request") is True)},
  ]
  return {
      "case": case,
      "attention_boundary_rows": attention_boundaries,
      "attention_history_rows": attention_histories,
      "checks": checks,
      "distribution_rows": distributions,
      "lm_head_hidden_rows": lm_head_hidden_rows,
      "kld_max": max(klds) if klds else None,
      "required_checks_passed": all(
          check["pass"] for check in checks if check.get("required", True)),
      "top1_rate": top1_rate,
  }


def block_summary(
    block_index: int, runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  stock = [runs["stock-a1"]["result"], runs["stock-a2"]["result"]]
  candidate = [
      runs["candidate-b1"]["result"], runs["candidate-b2"]["result"]]
  phases = {}
  for key in ("prefill_tokens_s", "decode_tokens_s"):
    stock_value = statistics.median(float(row[key]) for row in stock)
    candidate_value = statistics.median(float(row[key]) for row in candidate)
    phases[key] = {
        "candidate": candidate_value,
        "ratio": candidate_value / stock_value,
        "stock": stock_value,
    }
  stock_total_rate = statistics.median(1000.0 / float(row["total_wall_ms"])
                                       for row in stock)
  candidate_total_rate = statistics.median(
      1000.0 / float(row["total_wall_ms"]) for row in candidate)
  phases["total_rate"] = {
      "candidate": candidate_total_rate,
      "ratio": candidate_total_rate / stock_total_rate,
      "stock": stock_total_rate,
  }
  return {"block": block_index, "phases": phases}


def performance_for_case(
    case: dict[str, Any], blocks: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
  target = 1.10 if case["bucket"] in PRIORITY_BUCKETS else 0.98
  phase_rows = {}
  for phase in ("prefill_tokens_s", "decode_tokens_s", "total_rate"):
    candidate_values = [block["phases"][phase]["candidate"] for block in blocks]
    stock_values = [block["phases"][phase]["stock"] for block in blocks]
    phase_rows[phase] = perf_inference.paired_speedup_inference(
        candidate_values, stock_values, target_ratio=target,
        min_blocks=MIN_PROMOTION_BLOCKS)
  floors = acceptance["bootstrap_targets"]
  absolute = {
      "applicable": case["bucket"] in PRIORITY_BUCKETS,
      "decode_floor": float(floors["decode_tokens_s"][str(case["bucket"])]),
      "decode_median": statistics.median(
          block["phases"]["decode_tokens_s"]["candidate"] for block in blocks),
      "prefill_floor": float(floors["prefill_tokens_s"][str(case["bucket"])]),
      "prefill_median": statistics.median(
          block["phases"]["prefill_tokens_s"]["candidate"] for block in blocks),
  }
  absolute["pass"] = (
      not absolute["applicable"] or
      (absolute["decode_median"] >= absolute["decode_floor"] and
       absolute["prefill_median"] >= absolute["prefill_floor"]))
  return {
      "absolute_floors": absolute,
      "blocks": blocks,
      "candidate_path": case["candidate_path"],
      "case_id": case["case_id"],
      "paired_block_count": len(blocks),
      "phase_inference": phase_rows,
      "promotion_rate_pass": (
          absolute["pass"] and all(
              row["rate_pass"] for row in phase_rows.values())),
      "target_ratio": target,
  }


def gpu_memory_growth(result: dict[str, Any]) -> float | None:
  memory = result.get("gpu_memory", {})
  prefill = memory_total(memory.get("after_prefill", {}))
  decode = memory_total(memory.get("after_decode", {}))
  if prefill is None or decode is None or prefill <= 0:
    return None
  return (decode - prefill) / prefill


def memory_rollup(all_runs: list[dict[str, Any]], acceptance: dict[str, Any]) -> dict[str, Any]:
  threshold = float(acceptance["smoothness"]["resident_memory_growth_max"])
  rows = []
  for item in all_runs:
    result = item.get("result", {})
    growth = gpu_memory_growth(result) if result else None
    rows.append({
        "case_id": result.get("case_id"),
        "candidate_path": result.get("candidate_path"),
        "gpu_memory_growth_after_prefill": growth,
        "mode": result.get("mode"),
        "oom_observed": item.get("oom_observed"),
        "memory_guard_tripped": item.get("memory_guard", {}).get("tripped"),
        "process_rss_peak_bytes": item.get("monitor", {}).get(
            "process_rss_peak_bytes"),
        "process_swap_peak_bytes": item.get("monitor", {}).get(
            "process_swap_peak_bytes"),
        "system_available_min_bytes": item.get("monitor", {}).get(
            "system_available_min_bytes"),
        "worker": item.get("worker"),
    })
  candidate_growth = [
      row["gpu_memory_growth_after_prefill"] for row in rows
      if row["mode"] == "candidate" and
      finite(row.get("gpu_memory_growth_after_prefill"))]
  checks = [
      {"name": "no_worker_oom", "pass": (
          bool(rows) and all(row["oom_observed"] is False for row in rows))},
      {"name": "memory_guard_not_tripped", "pass": (
          bool(rows) and all(
              row["memory_guard_tripped"] is False for row in rows))},
      {"name": "candidate_resident_memory_growth", "pass": (
          bool(candidate_growth) and max(candidate_growth) <= threshold),
       "max": max(candidate_growth) if candidate_growth else None,
       "threshold": threshold},
  ]
  return {
      "checks": checks,
      "required_checks_passed": all(check["pass"] for check in checks),
      "rows": rows,
  }


def coefficient_of_variation(values: list[float]) -> float | None:
  if len(values) < 2:
    return None
  mean = statistics.mean(values)
  return statistics.pstdev(values) / mean if mean else None


def smoothness_rollup(
    performance: list[dict[str, Any]], performance_runs: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
  threshold = float(acceptance["smoothness"]["decode_tpot_p95_over_p50_max"])
  jitter_rows = []
  for run in performance_runs:
    result = run.get("result", {})
    if result.get("mode") != "candidate":
      continue
    samples = [float(value) for value in result.get("decode_wall_ms", [])[16:]]
    p50 = percentile(samples, 0.50)
    p95 = percentile(samples, 0.95)
    ratio = p95 / p50 if p50 and p95 else None
    jitter_rows.append({
        "case_id": result.get("case_id"),
        "p50_ms": p50,
        "p95_ms": p95,
        "p95_over_p50": ratio,
        "pass": finite(ratio) and float(ratio) <= threshold,
    })

  by_bucket: dict[int, list[dict[str, Any]]] = {}
  for row in performance:
    bucket = int(row["case_id"].split("_")[-1][:-1]) * 1024
    if row["case_id"].endswith("128k"):
      bucket = 131072
    by_bucket.setdefault(bucket, []).append(row)
  floors = acceptance["bootstrap_targets"]
  ladder = []
  for bucket, rows in sorted(by_bucket.items()):
    prefill = statistics.median(
        row["absolute_floors"]["prefill_median"] for row in rows)
    decode = statistics.median(
        row["absolute_floors"]["decode_median"] for row in rows)
    ladder.append({
        "bucket": bucket,
        "decode_normalized": decode / float(
            floors["decode_tokens_s"][str(bucket)]),
        "decode_tokens_s": decode,
        "prefill_normalized": prefill / float(
            floors["prefill_tokens_s"][str(bucket)]),
        "prefill_tokens_s": prefill,
    })
  adjacent = []
  for previous, current in zip(ladder, ladder[1:]):
    adjacent.append({
        "decode_normalized_retention": (
            current["decode_normalized"] / previous["decode_normalized"]),
        "from_bucket": previous["bucket"],
        "prefill_normalized_retention": (
            current["prefill_normalized"] / previous["prefill_normalized"]),
        "to_bucket": current["bucket"],
    })
  prefill_cv = coefficient_of_variation(
      [row["prefill_normalized"] for row in ladder])
  decode_cv = coefficient_of_variation(
      [row["decode_normalized"] for row in ladder])
  full_ladder = len(ladder) == len(CORE_BUCKETS)
  checks = [
      {"name": "decode_tpot_p95_over_p50", "pass": (
          bool(jitter_rows) and all(row["pass"] for row in jitter_rows)),
       "threshold": threshold},
      {"name": "adjacent_decode_normalized_retention", "pass": (
          not full_ladder or all(
              row["decode_normalized_retention"] >= 0.75 for row in adjacent)),
       "status": "checked" if full_ladder else "insufficient_ladder"},
      {"name": "adjacent_prefill_normalized_retention", "pass": (
          not full_ladder or all(
              row["prefill_normalized_retention"] >= 0.75 for row in adjacent)),
       "status": "checked" if full_ladder else "insufficient_ladder"},
      {"name": "decode_target_normalized_cv", "pass": (
          not full_ladder or
          (finite(decode_cv) and decode_cv <= float(
              acceptance["smoothness"]["target_normalized_score_cv_max"]))),
       "status": "checked" if full_ladder else "insufficient_ladder",
       "value": decode_cv},
      {"name": "prefill_target_normalized_cv", "pass": (
          not full_ladder or
          (finite(prefill_cv) and prefill_cv <= float(
              acceptance["smoothness"]
              ["prefill_target_normalized_score_cv_max"]))),
       "status": "checked" if full_ladder else "insufficient_ladder",
       "value": prefill_cv},
  ]
  return {
      "adjacent": adjacent,
      "checks": checks,
      "jitter_rows": jitter_rows,
      "ladder": ladder,
      "required_checks_passed": all(check["pass"] for check in checks),
  }


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# OpenVINO hot/cold product gate",
      "",
      f"- route label: `{payload['route_label']}`",
      f"- run checks passed: `{str(payload['run_checks_passed']).lower()}`",
      f"- product promotion ready: `{str(payload['product_promotion_ready']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- output tokens: `{payload['config']['output_tokens']}`",
      f"- paired ABBA blocks per case: `{payload['config']['paired_blocks']}`",
      "",
      "| case | path | blocks | prefill LCB | decode LCB | total LCB | absolute |",
      "|---|---|---:|---:|---:|---:|:---:|",
  ]
  for row in payload["performance"]:
    inference = row["phase_inference"]
    lines.append(
        f"| {row['case_id']} | {row['candidate_path']} | "
        f"{row['paired_block_count']} | "
        f"{inference['prefill_tokens_s']['lower_confidence_bound_ratio']:.6f} | "
        f"{inference['decode_tokens_s']['lower_confidence_bound_ratio']:.6f} | "
        f"{inference['total_rate']['lower_confidence_bound_ratio']:.6f} | "
        f"{'pass' if row['absolute_floors']['pass'] else 'fail'} |")
  lines += [
      "",
      "A partial matrix or fewer than eight paired blocks is diagnostic only.",
      "Every worker used a fresh compiler cache and ran after the preceding",
      "worker exited; resident 8k chunks were timed as one logical prompt.",
      "",
  ]
  return "\n".join(lines)


def orchestrator_main(args: argparse.Namespace) -> int:
  cases = build_cases(args)
  plan = {
      "abort_below_available_gib": args.abort_below_available_gib,
      "buckets": list(args.buckets),
      "candidate_gpu_plugin": (
          str(args.candidate_gpu_plugin.resolve())
          if args.candidate_gpu_plugin is not None else None),
      "candidate_gpu_plugin_sha256": (
          sha256_file(args.candidate_gpu_plugin)
          if args.candidate_gpu_plugin is not None else None),
      "candidate_impls_cache_capacity": (
          args.candidate_impls_cache_capacity),
      "candidate_dq_realloc_fastpath": any(
          case["candidate_dq_realloc_fastpath"] for case in cases),
      "candidate_fc_stable_prepare_fastpath": any(
          case["candidate_fc_stable_prepare_fastpath"] for case in cases),
      "candidate_policy": args.candidate_policy,
      "capture_execution_census": (
          args.capture_execution_census or args.fuse_fixed_fc or
          args.fixed_fc_manager_direct or
          args.lm_head_i8q4 or args.lm_head_i8q1 or
          args.fuse_qk_rope_layout or
          args.fuse_router_shared_triple or
          args.fuse_router_shared_pair),
      "capture_prefill_profiles": args.capture_prefill_profiles,
      "capture_all_correctness_logits": (
          args.capture_all_correctness_logits),
      "capture_lm_head_hidden": args.capture_lm_head_hidden,
      "capture_attention_layers": list(args.capture_attention_layers),
      "capture_attention_steps": list(args.capture_attention_steps),
      "capture_attention_history_layers": list(
          args.capture_attention_history_layers),
      "capture_attention_history_steps": list(
          args.capture_attention_history_steps),
      "custom_config": relative(args.custom_config),
      "custom_config_sha256": sha256_file(args.custom_config),
      "custom_composition": args.custom_composition,
      "fuse_qk_rope_layout": args.fuse_qk_rope_layout,
      "fuse_router_shared_triple": args.fuse_router_shared_triple,
      "fuse_router_shared_pair": args.fuse_router_shared_pair,
      "adaptive_attention_topk": args.adaptive_attention_topk,
      "adaptive_attention_high_topk_layers": list(
          args.adaptive_attention_high_topk_layers),
      "adaptive_attention_high_topk": args.adaptive_attention_high_topk,
      "adaptive_attention_v16_layers": list(
          args.adaptive_attention_v16_layers),
      "adaptive_attention_key_exact_layers": list(
          args.adaptive_attention_key_exact_layers),
      "adaptive_attention_key_residual1_layers": list(
          args.adaptive_attention_key_residual1_layers),
      "adaptive_attention_value_residual1_layers": list(
          args.adaptive_attention_value_residual1_layers),
      "adaptive_attention_packed_kv_layers": list(
          args.adaptive_attention_packed_kv_layers),
      "adaptive_attention_packed_kv_variant": (
          args.adaptive_attention_packed_kv_variant),
      "adaptive_attention_exact_layers": list(
          args.adaptive_attention_exact_layers),
      "exact_phase_context_partition4": (
          args.exact_phase_context_partition4),
      "exact_phase_dual_cohort": bool(
          args.exact_phase_dual_cohort_buckets),
      "exact_phase_dual_cohort_buckets": list(
          args.exact_phase_dual_cohort_buckets),
      "custom_sources": [{
          "path": relative(path), "sha256": sha256_file(path),
      } for path in CUSTOM_SOURCES],
      "fuse_linear_conv_state": args.fuse_linear_conv_state,
      "direct_ssm_state_assign": args.direct_ssm_state_assign,
      "fuse_fixed_fc": args.fuse_fixed_fc,
      "fixed_fc_cohorts": (
          list(args.fixed_fc_cohorts or FIXED_FC_COHORTS)
          if args.fuse_fixed_fc else []),
      "fixed_fc_manager_direct": args.fixed_fc_manager_direct,
      "fixed_fc_manager_scope": (
          args.fixed_fc_manager_scope
          if args.fixed_fc_manager_direct else "all"),
      "lm_head_i8q4": args.lm_head_i8q4,
      "lm_head_i8q1": args.lm_head_i8q1,
      "lm_head_i8q1_gated_exact": args.lm_head_i8q1_gated_exact,
      "lm_head_i8q1_gated_exact_affine_q4": (
          args.lm_head_i8q1_gated_exact_affine_q4),
      "lm_head_i8q1_gated_q4": args.lm_head_i8q1_gated_q4,
      "lm_head_i8q1_greedy_local2": args.lm_head_i8q1_greedy_local2,
      "lm_head_device_greedy_feedback": (
          args.lm_head_device_greedy_feedback),
      "lm_head_token_only_feedback": (
          args.lm_head_token_only_feedback),
      "host_time_profiling": args.host_time_profiling,
      "pack_gdn_state": args.pack_gdn_state,
      "prime_candidate_exact_decode_shape": (
          args.prime_candidate_exact_decode_shape),
      "alias_linear_state_assign": args.alias_linear_state_assign,
      "linear_state_alias_scope": (
          args.linear_state_alias_scope
          if args.alias_linear_state_assign else "none"),
      "self_bind_hot_states": args.self_bind_hot_states,
      "cases": cases,
      "correctness_only": args.paired_blocks == 0,
      "output_tokens": args.output_tokens,
      "paired_blocks": args.paired_blocks,
      "prefill_chunk_tokens": args.prefill_chunk_tokens,
      "prefill_history_capacity": args.prefill_history_capacity,
      "exact_history_layers": list(args.exact_history_layers),
      "exact_history_capacity": args.exact_history_capacity,
      "exact_history_capacity_slack_tokens": (
          args.exact_history_capacity_slack_tokens),
      "preflight_min_available_gib": args.min_available_gib,
      "prompt_sets": list(args.prompt_sets),
      "target_layers": list(args.target_layers),
      "decode_chunk256_layers": list(args.decode_chunk256_layers),
      "decode_f32_numerator_layers": list(
          args.decode_f32_numerator_layers),
      "decode_dual256_layers": list(args.decode_dual256_layers),
      "decode_stock256_layers": list(args.decode_stock256_layers),
      "decode_stock_score_layers": list(args.decode_stock_score_layers),
      "decode_stock_partition_layers": list(
          args.decode_stock_partition_layers),
      "decode_stock_micro_layers": list(args.decode_stock_micro_layers),
      "decode_page_sparse_layers": list(args.decode_page_sparse_layers),
      "schedule": "stock,candidate,candidate,stock per paired block",
      "strict_worker_serialization": True,
      "worker_transient_scope": args.worker_transient_scope,
      "worker_scope_resource_limits_changed": False,
      "timing_output_policy": (
          "bucket_scoped_exact_short_compact_long"
          if (args.candidate_policy == "auto" and
              args.lm_head_token_only_feedback) else
          "candidate_token_only_provider_stock_host_argmax"
          if args.lm_head_token_only_feedback else
          "candidate_device_greedy_stock_host_argmax"
          if args.lm_head_device_greedy_feedback
          else "last_query_logits_host_argmax"),
      "timing_provider_by_bucket": {
          str(bucket): timing_lm_head_policy(
              args, bucket)["timing_lm_head_provider"]
          for bucket in args.buckets
      },
      "warmup": not args.no_warmup,
      "worker_count_planned": len(cases) * (2 + 4 * args.paired_blocks),
  }
  if args.plan_only:
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0

  out_dir = args.out_dir.resolve()
  if out_dir.exists() and not args.resume:
    raise SystemExit(f"output directory exists; use --resume: {out_dir}")
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)
  write_json(out_dir / "plan.json", plan)
  git = BOOT.git_state(out_dir)
  model_identity = BOOT.capture_model_identity(
      args.model_dir.resolve(), args.model_contract.resolve())
  write_json(out_dir / "model-identity.json", model_identity)
  write_json(out_dir / "host.json", BOOT.capture_host())

  all_runs = []
  performance_runs = []
  correctness_rows = []
  performance_rows = []
  stopped_reason = None
  for case in cases:
    case_root = raw / case["case_id"]
    common = {
        "bucket": case["bucket"],
        "case_id": case["case_id"],
        "checkpoint_steps": [
            step for step in (
                range(args.output_tokens)
                if (args.capture_all_correctness_logits or
                    args.fuse_fixed_fc or args.fixed_fc_manager_direct or
                    args.lm_head_i8q4 or args.lm_head_i8q1) else
                CHECKPOINT_STEPS) if step < args.output_tokens],
        "capture_execution_census": (
            args.capture_execution_census or args.fuse_fixed_fc or
            args.fixed_fc_manager_direct or
            args.lm_head_i8q4 or args.lm_head_i8q1 or
            args.fuse_qk_rope_layout or
            args.fuse_router_shared_triple or
            args.fuse_router_shared_pair),
        "capture_prefill_profiles": args.capture_prefill_profiles,
        "capture_lm_head_hidden": args.capture_lm_head_hidden,
        "adaptive_attention_topk": case["adaptive_attention_topk"],
        "adaptive_attention_high_topk_layers": case[
            "adaptive_attention_high_topk_layers"],
        "adaptive_attention_high_topk": case[
            "adaptive_attention_high_topk"],
        "adaptive_attention_v16_layers": case[
            "adaptive_attention_v16_layers"],
        "adaptive_attention_key_exact_layers": case[
            "adaptive_attention_key_exact_layers"],
        "adaptive_attention_key_residual1_layers": case[
            "adaptive_attention_key_residual1_layers"],
        "adaptive_attention_value_residual1_layers": case[
            "adaptive_attention_value_residual1_layers"],
        "adaptive_attention_packed_kv_layers": case[
            "adaptive_attention_packed_kv_layers"],
        "adaptive_attention_packed_kv_variant": case[
            "adaptive_attention_packed_kv_variant"],
        "adaptive_attention_exact_layers": case[
            "adaptive_attention_exact_layers"],
        "capture_attention_layers": list(args.capture_attention_layers),
        "capture_attention_steps": list(args.capture_attention_steps),
        "capture_attention_history_layers": list(
            args.capture_attention_history_layers),
        "capture_attention_history_steps": list(
            args.capture_attention_history_steps),
        "expected_answer": case["expected_answer"],
        "custom_composition": args.custom_composition,
        "exact_phase_context_partition4": (
            args.exact_phase_context_partition4),
        "exact_phase_dual_cohort": case["exact_phase_dual_cohort"],
        "fuse_fixed_fc": args.fuse_fixed_fc,
        "fixed_fc_cohorts": plan["fixed_fc_cohorts"],
        "fixed_fc_manager_direct": args.fixed_fc_manager_direct,
        "fixed_fc_manager_scope": plan["fixed_fc_manager_scope"],
        "lm_head_i8q4": args.lm_head_i8q4,
        "lm_head_i8q1": args.lm_head_i8q1,
        "lm_head_i8q1_gated_exact": args.lm_head_i8q1_gated_exact,
        "lm_head_i8q1_gated_exact_affine_q4": (
            args.lm_head_i8q1_gated_exact_affine_q4),
        "lm_head_i8q1_gated_q4": args.lm_head_i8q1_gated_q4,
        "lm_head_i8q1_greedy_local2": args.lm_head_i8q1_greedy_local2,
        "lm_head_device_greedy_feedback": (
            args.lm_head_device_greedy_feedback),
        "lm_head_token_only_feedback": (
            args.lm_head_token_only_feedback),
        "fuse_linear_conv_state": args.fuse_linear_conv_state,
        "fuse_qk_rope_layout": args.fuse_qk_rope_layout,
        "fuse_router_shared_triple": args.fuse_router_shared_triple,
        "fuse_router_shared_pair": args.fuse_router_shared_pair,
        "direct_ssm_state_assign": args.direct_ssm_state_assign,
        "host_time_profiling": args.host_time_profiling,
        "pack_gdn_state": args.pack_gdn_state,
        "prime_candidate_exact_decode_shape": (
            args.prime_candidate_exact_decode_shape),
        "candidate_impls_cache_capacity": (
            args.candidate_impls_cache_capacity),
        "candidate_dq_realloc_fastpath": case[
            "candidate_dq_realloc_fastpath"],
        "candidate_fc_stable_prepare_fastpath": case[
            "candidate_fc_stable_prepare_fastpath"],
        "alias_linear_state_assign": args.alias_linear_state_assign,
        "linear_state_alias_scope": plan["linear_state_alias_scope"],
        "self_bind_hot_states": args.self_bind_hot_states,
        "output_tokens": args.output_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "prefill_history_capacity": case["prefill_history_capacity"],
        "exact_history_layers": list(args.exact_history_layers),
        "exact_history_capacity": case["exact_history_capacity"],
        "prompt": case["path"],
        "warmup": not args.no_warmup,
        "target_layers": list(args.target_layers),
        "decode_chunk256_layers": list(args.decode_chunk256_layers),
        "decode_f32_numerator_layers": list(
            args.decode_f32_numerator_layers),
        "decode_dual256_layers": list(args.decode_dual256_layers),
        "decode_stock256_layers": list(args.decode_stock256_layers),
        "decode_stock_score_layers": list(args.decode_stock_score_layers),
        "decode_stock_partition_layers": list(
            args.decode_stock_partition_layers),
        "decode_stock_micro_layers": list(args.decode_stock_micro_layers),
        "decode_page_sparse_layers": list(args.decode_page_sparse_layers),
    }
    stock_run = run_worker(args, case_root / "correctness/stock", {
        **common,
        "candidate_path": "stock_sdpa",
        "capture_logits": True,
        "mode": "stock",
        "purpose": "correctness_reference",
        "reference_result": None,
        "timing_token_output": False,
    })
    stock_run["worker"] = relative(case_root / "correctness/stock")
    all_runs.append(stock_run)
    if stock_run.get("returncode") != 0:
      stopped_reason = f"{case['case_id']}: stock correctness worker failed"
      break
    stock_result_path = case_root / "correctness/stock/worker-result.json"
    candidate_run = run_worker(args, case_root / "correctness/candidate", {
        **common,
        "candidate_path": case["candidate_path"],
        "capture_logits": True,
        "mode": "candidate",
        "purpose": "teacher_forced_correctness",
        "reference_result": str(stock_result_path.resolve()),
        "timing_token_output": False,
    })
    candidate_run["worker"] = relative(case_root / "correctness/candidate")
    all_runs.append(candidate_run)
    correctness = correctness_for_case(case, stock_run, candidate_run)
    correctness_rows.append(correctness)
    write_json(case_root / "correctness/comparison.json", correctness)
    if not correctness["required_checks_passed"]:
      stopped_reason = f"{case['case_id']}: correctness precheck failed"
      break

    blocks = []
    for block_index in range(args.paired_blocks):
      block_root = case_root / f"block{block_index:02d}"
      block_runs = {}
      schedule = (
          ("stock-a1", "stock", "stock_sdpa"),
          ("candidate-b1", "candidate", case["candidate_path"]),
          ("candidate-b2", "candidate", case["candidate_path"]),
          ("stock-a2", "stock", "stock_sdpa"),
      )
      for label, mode, selected_path in schedule:
        timing_local2 = bool(
            case["timing_lm_head_i8q1_greedy_local2"] and
            mode == "candidate")
        timing_device_feedback = bool(
            case["timing_lm_head_device_greedy_feedback"] and
            mode == "candidate")
        timing_token_only = bool(
            case["timing_lm_head_token_only_feedback"] and
            mode == "candidate")
        timing_affine_q4 = bool(
            case["timing_lm_head_i8q1_gated_exact_affine_q4"] and
            mode == "candidate")
        run = run_worker(args, block_root / label, {
            **common,
            "candidate_path": selected_path,
            # Correctness workers retain PERF_COUNT for the exact execution
            # census.  Profiling serializes parts of the GPU schedule and is
            # therefore forbidden in the paired product-timing lane.
            "capture_execution_census": False,
            "capture_prefill_profiles": False,
            "capture_logits": False,
            "lm_head_i8q1_gated_exact_affine_q4": timing_affine_q4,
            "lm_head_i8q1_greedy_local2": timing_local2,
            "lm_head_device_greedy_feedback": timing_device_feedback,
            "lm_head_token_only_feedback": timing_token_only,
            "mode": mode,
            "purpose": "paired_product_timing",
            "reference_result": str(stock_result_path.resolve()),
            # Keep stock on its unmodified host-argmax boundary.  Candidate
            # timing can use either the retained two-pass GPU reduction or
            # the compact provider plus one phase-safe token boundary.
            "timing_token_output": bool(
                timing_device_feedback or timing_token_only),
        })
        run["worker"] = relative(block_root / label)
        all_runs.append(run)
        performance_runs.append(run)
        block_runs[label] = run
        if run.get("returncode") != 0 or run.get("oom_observed"):
          stopped_reason = f"{case['case_id']} block {block_index} {label} failed"
          break
        result = run.get("result", {})
        local2_expected = timing_local2
        token_only_expected = timing_token_only
        gated_exact_expected = bool(
            case["timing_lm_head_i8q1_gated_exact"] and
            mode == "candidate")
        if not PRODUCT_POLICY.timing_provider_isolated(
            result,
            affine_q4_expected=timing_affine_q4,
            gated_exact_expected=gated_exact_expected,
            local2_expected=local2_expected,
            token_only_expected=token_only_expected):
          stopped_reason = (
              f"{case['case_id']} block {block_index} {label} "
              "timing-provider isolation/trace mismatch")
          break
        device_feedback_expected = timing_device_feedback
        runtime_rows = result.get(
            "runtime_census", {}).get("attention_rows", [])
        greedy_input_indices = [
            index for index, row in enumerate(runtime_rows)
            if row.get("name") == "iq36_product_greedy_top1_input" and
            row.get("layer_type") == "Reshape"]
        # The GPU runtime keeps the partial node's friendly name but renames
        # the terminal custom primitive after its Result consumer.  Prove the
        # two-pass boundary by its exact ordered tail rather than that unstable
        # compiler-generated Result_<id> name.
        device_feedback_rows = (
            runtime_rows[greedy_input_indices[0] + 1:]
            if len(greedy_input_indices) == 1 else [])
        device_feedback_runtime_exact = (
            len(greedy_input_indices) == 1 and
            len(device_feedback_rows) == 2 and
            device_feedback_rows[0].get("name") ==
                "iq36_product_greedy_top1_partials" and
            all(row.get("layer_type") == "CustomGPUPrimitive"
                for row in device_feedback_rows))
        device_feedback_observed = (
            result.get("lm_head_device_greedy_feedback") is
                device_feedback_expected and
            (device_feedback_runtime_exact
             if device_feedback_expected else
             not greedy_input_indices and not device_feedback_rows))
        if not device_feedback_observed:
          stopped_reason = (
              f"{case['case_id']} block {block_index} {label} "
              "device-greedy isolation/runtime mismatch")
          break
        token_input_indices = [
            index for index, row in enumerate(runtime_rows)
            if row.get("name") ==
                "iq36_product_greedy_token_or_logits_input" and
            row.get("layer_type") == "Reshape"]
        token_feedback_rows = (
            runtime_rows[token_input_indices[0] + 1:]
            if len(token_input_indices) == 1 else [])
        token_feedback_runtime_exact = (
            len(token_input_indices) == 1 and
            len(token_feedback_rows) == 1 and
            token_feedback_rows[0].get("layer_type") ==
                "CustomGPUPrimitive")
        token_feedback_observed = (
            result.get("lm_head_token_only_feedback") is
                token_only_expected and
            result.get("timing_token_output") is
                (device_feedback_expected or token_only_expected) and
            (token_feedback_runtime_exact
             if token_only_expected else
             not token_input_indices and not token_feedback_rows))
        if not token_feedback_observed:
          stopped_reason = (
              f"{case['case_id']} block {block_index} {label} "
              "token-only isolation/runtime mismatch")
          break
        if (run.get("result", {}).get("generated_token_ids") !=
            stock_run["result"].get("generated_token_ids")):
          stopped_reason = (
              f"{case['case_id']} block {block_index} {label} token divergence")
          break
      if stopped_reason:
        break
      blocks.append(block_summary(block_index, block_runs))
      write_json(block_root / "block-summary.json", blocks[-1])
    if blocks:
      performance_rows.append(performance_for_case(
          case, blocks, load_json(args.acceptance)))
    if stopped_reason:
      break

  acceptance = load_json(args.acceptance)
  memory = memory_rollup(all_runs, acceptance) if all_runs else {
      "checks": [], "required_checks_passed": False, "rows": []}
  smoothness = smoothness_rollup(
      performance_rows, performance_runs, acceptance) if performance_rows else {
          "applicable": False,
          "checks": [],
          "required_checks_passed": args.paired_blocks == 0,
          "jitter_rows": [],
          "ladder": []}
  repository_clean = not git["dirty"]
  correctness_pass = (
      len(correctness_rows) == len(cases) and
      all(row["required_checks_passed"] for row in correctness_rows))
  planned_blocks_complete = (
      (args.paired_blocks == 0 and not performance_rows) or
      (len(performance_rows) == len(cases) and all(
          row["paired_block_count"] == args.paired_blocks
          for row in performance_rows)))
  run_checks_passed = (
      stopped_reason is None and repository_clean and
      model_identity["required_checks_passed"] and correctness_pass and
      planned_blocks_complete and memory["required_checks_passed"] and
      smoothness["required_checks_passed"])
  full_scope = (
      tuple(args.buckets) == CORE_BUCKETS and
      tuple(args.prompt_sets) == PROMPT_SETS and
      args.output_tokens == 512 and args.paired_blocks >= MIN_PROMOTION_BLOCKS and
      args.candidate_policy == "auto" and not args.no_warmup)
  rate_pass = (
      len(performance_rows) == len(cases) and all(
          row["promotion_rate_pass"] for row in performance_rows))
  product_promotion_ready = run_checks_passed and full_scope and rate_pass
  route_label = (
      "product_candidate" if product_promotion_ready
      else "diagnostic" if run_checks_passed else "rejected")
  payload = {
      "config": plan,
      "correctness": correctness_rows,
      "created_at": iso_now(),
      "full_promotion_scope": full_scope,
      "git": git,
      "performance": performance_rows,
      "product_promotion_ready": product_promotion_ready,
      "route_label": route_label,
      "run_checks_passed": run_checks_passed,
      "schema_version": SCHEMA,
      "speedup_claims_allowed": product_promotion_ready,
      "stopped_reason": stopped_reason,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "candidate_gpu_plugin": plan["candidate_gpu_plugin"],
      "candidate_gpu_plugin_sha256": plan["candidate_gpu_plugin_sha256"],
      "captured_at": payload["created_at"],
      "capture_attention_layers": plan["capture_attention_layers"],
      "capture_attention_steps": plan["capture_attention_steps"],
      "capture_attention_history_layers": plan[
          "capture_attention_history_layers"],
      "capture_attention_history_steps": plan[
          "capture_attention_history_steps"],
      "custom_composition": plan["custom_composition"],
      "exact_phase_context_partition4": plan[
          "exact_phase_context_partition4"],
      "exact_phase_dual_cohort": plan["exact_phase_dual_cohort"],
      "exact_phase_dual_cohort_buckets": plan[
          "exact_phase_dual_cohort_buckets"],
      "decode_chunk256_layers": plan["decode_chunk256_layers"],
      "decode_f32_numerator_layers": plan[
          "decode_f32_numerator_layers"],
      "decode_dual256_layers": plan["decode_dual256_layers"],
      "decode_stock256_layers": plan["decode_stock256_layers"],
      "decode_stock_score_layers": plan["decode_stock_score_layers"],
      "decode_stock_partition_layers": plan[
          "decode_stock_partition_layers"],
      "decode_stock_micro_layers": plan["decode_stock_micro_layers"],
      "decode_page_sparse_layers": plan["decode_page_sparse_layers"],
      "exact_history_layers": plan["exact_history_layers"],
      "exact_history_capacity": plan["exact_history_capacity"],
      "exact_history_capacity_slack_tokens": plan[
          "exact_history_capacity_slack_tokens"],
      "custom_config": plan["custom_config"],
      "custom_config_sha256": plan["custom_config_sha256"],
      "custom_sources": plan["custom_sources"],
      "fuse_fixed_fc": plan["fuse_fixed_fc"],
      "fuse_qk_rope_layout": plan["fuse_qk_rope_layout"],
      "fuse_router_shared_triple": plan["fuse_router_shared_triple"],
      "fuse_router_shared_pair": plan["fuse_router_shared_pair"],
      "fixed_fc_cohorts": plan["fixed_fc_cohorts"],
      "lm_head_i8q4": plan["lm_head_i8q4"],
      "lm_head_i8q1": plan["lm_head_i8q1"],
      "lm_head_i8q1_gated_exact": plan["lm_head_i8q1_gated_exact"],
      "lm_head_i8q1_gated_exact_affine_q4": plan[
          "lm_head_i8q1_gated_exact_affine_q4"],
      "lm_head_i8q1_gated_q4": plan["lm_head_i8q1_gated_q4"],
      "lm_head_i8q1_greedy_local2": plan[
          "lm_head_i8q1_greedy_local2"],
      "lm_head_device_greedy_feedback": plan[
          "lm_head_device_greedy_feedback"],
      "lm_head_token_only_feedback": plan[
          "lm_head_token_only_feedback"],
      "fuse_linear_conv_state": plan["fuse_linear_conv_state"],
      "direct_ssm_state_assign": plan["direct_ssm_state_assign"],
      "pack_gdn_state": plan["pack_gdn_state"],
      "prime_candidate_exact_decode_shape": plan[
          "prime_candidate_exact_decode_shape"],
      "candidate_impls_cache_capacity": plan[
          "candidate_impls_cache_capacity"],
      "candidate_dq_realloc_fastpath": plan[
          "candidate_dq_realloc_fastpath"],
      "candidate_fc_stable_prepare_fastpath": plan[
          "candidate_fc_stable_prepare_fastpath"],
      "prefill_history_capacity": plan["prefill_history_capacity"],
      "alias_linear_state_assign": plan["alias_linear_state_assign"],
      "linear_state_alias_scope": plan["linear_state_alias_scope"],
      "git": git,
      "model_contract": relative(args.model_contract),
      "route_label": route_label,
      "schema_version": SCHEMA,
      "target_layers": plan["target_layers"],
      "tool": relative(Path(__file__)),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "cases": correctness_rows,
      "checks": [
          {"name": "repository_clean_at_gate", "pass": repository_clean,
           "value": git},
          {"name": "locked_model_identity", "pass": (
              model_identity["required_checks_passed"])},
          {"name": "all_case_correctness", "pass": correctness_pass},
      ],
      "gate": "hot_cold_output512_product_correctness",
      "required_checks_passed": (
          repository_clean and model_identity["required_checks_passed"] and
          correctness_pass),
      "route_label": route_label,
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "performance.json", {
      "cases": performance_rows,
      "full_promotion_scope": full_scope,
      "product_promotion_ready": product_promotion_ready,
      "speedup_claims_allowed": product_promotion_ready,
  })
  write_json(out_dir / "memory.json", memory)
  write_json(out_dir / "smoothness.json", smoothness)
  write_json(out_dir / "gate.json", payload)
  metrics = []
  for row in all_runs:
    result = row.get("result", {})
    if not result:
      continue
    metrics.append({
        "cache_state": "cold_no_prefix_same_request_resident_chunks",
        "candidate_path": result.get("candidate_path"),
        "case_id": result.get("case_id"),
        "decode_tokens_s": result.get("decode_tokens_s"),
        "input_tokens": result.get("input_token_count"),
        "mode": result.get("mode"),
        "output_tokens": result.get("generated_token_count"),
        "prefill_tokens_s": result.get("prefill_tokens_s"),
        "route_label": route_label,
        "total_wall_ms": result.get("total_wall_ms"),
        "worker": row.get("worker"),
    })
  write_jsonl(out_dir / "metrics.jsonl", metrics)
  (out_dir / "summary.md").write_text(
      build_summary(payload), encoding="utf-8")
  print(json.dumps({
      "event": "gate_complete",
      "out_dir": relative(out_dir),
      "product_promotion_ready": product_promotion_ready,
      "route_label": route_label,
      "run_checks_passed": run_checks_passed,
      "stopped_reason": stopped_reason,
  }, sort_keys=True), flush=True)
  return 0 if run_checks_passed else 2


if __name__ == "__main__":
  parsed = parse_args()
  if parsed.worker_config is not None:
    raise SystemExit(worker_main(parsed.worker_config))
  raise SystemExit(orchestrator_main(parsed))
