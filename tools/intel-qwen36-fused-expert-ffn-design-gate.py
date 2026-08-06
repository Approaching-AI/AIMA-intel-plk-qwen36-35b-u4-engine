#!/usr/bin/env python3
"""Lock one M8 expert-major fused-FFN design from real layer-27 evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-fused-expert-ffn-design-gate-v0"
DEFAULT_TOPK = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq673cleanZ/raw/"
    "schedule-probes/layer-27.topk.i32")
DEFAULT_GROUPED = (
    ROOT / "output/grouped-prefill-device-schedule-gate-"
    "20260712Tseq749cleanZ/result.json")
DEFAULT_LINEAR = (
    ROOT / "output/linear-prefill-nonstate-feasibility-"
    "20260712Tseq758cleanZ/result.json")
DEFAULT_STATE = (
    ROOT / "output/linear-attention-prefill-state-"
    "20260712Tseq753cleanZ/result.json")
DEFAULT_PROFILE = (
    ROOT / "output/openvino-hidden-prefill-profile-"
    "20260712Tseq751cleanZ/profile.json")
DEFAULT_GATEUP_ZE = (
    ROOT / "output/grouped-prefill-device-schedule-gate-"
    "20260712Tseq749cleanZ/raw/persistent-gateup-disasm/.ze_info")
DEFAULT_DOWN_ZE = (
    ROOT / "output/grouped-prefill-device-schedule-gate-"
    "20260712Tseq749cleanZ/raw/persistent-down-disasm/.ze_info")
DEFAULT_GATEUP_ASM = (
    ROOT / "output/grouped-prefill-device-schedule-gate-"
    "20260712Tseq749cleanZ/raw/persistent-gateup-disasm/"
    ".text.grouped_micro_gemm.asm")
DEFAULT_DOWN_ASM = (
    ROOT / "output/grouped-prefill-device-schedule-gate-"
    "20260712Tseq749cleanZ/raw/persistent-down-disasm/"
    ".text.grouped_micro_gemm.asm")

TOKENS = 1024
TOP_K = 8
EXPERTS = 256
FFN_INPUT = 2048
FFN_INTERMEDIATE = 512
GATEUP_OUTPUT = 1024
FFN_OUTPUT = 2048
TARGET_PREFILL_TOK_S = 2510.0
PRODUCT_RATIO = 1.10
FFN_CAP_US = 6250.0
MATRIX_RATE_TMAC_S = 5.4
M_TILE = 8
GATEUP_N_TILE = 32
DOWN_N_TILE = 64
PERSISTENT_WORKGROUPS = 96
TARGET_GRF_COUNT = 128
TARGET_EU_THREADS = 8
SLM_BYTES = M_TILE * FFN_INTERMEDIATE * 4


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--topk", type=Path, default=DEFAULT_TOPK)
  parser.add_argument("--grouped-result", type=Path, default=DEFAULT_GROUPED)
  parser.add_argument("--linear-result", type=Path, default=DEFAULT_LINEAR)
  parser.add_argument("--state-result", type=Path, default=DEFAULT_STATE)
  parser.add_argument("--openvino-profile", type=Path, default=DEFAULT_PROFILE)
  parser.add_argument("--gateup-ze-info", type=Path, default=DEFAULT_GATEUP_ZE)
  parser.add_argument("--down-ze-info", type=Path, default=DEFAULT_DOWN_ZE)
  parser.add_argument("--gateup-asm", type=Path, default=DEFAULT_GATEUP_ASM)
  parser.add_argument("--down-asm", type=Path, default=DEFAULT_DOWN_ASM)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/fused-expert-ffn-design-gate-{stamp}"
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"expected JSON object: {path}")
  return value


def git_output(*args: str) -> str:
  completed = subprocess.run(
      ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
  return completed.stdout.strip()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def first_number(text: str, name: str) -> int | None:
  match = re.search(rf"\b{name}:\s*(\d+)", text)
  return int(match.group(1)) if match else None


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.topk, args.grouped_result, args.linear_result, args.state_result,
      args.openvino_profile, args.gateup_ze_info, args.down_ze_info,
      args.gateup_asm, args.down_asm,
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  grouped = load_json(args.grouped_result)
  linear = load_json(args.linear_result)
  state = load_json(args.state_result)
  profile = load_json(args.openvino_profile)

  topk_bytes = args.topk.read_bytes()
  topk = list(struct.unpack(f"<{len(topk_bytes) // 4}i", topk_bytes))
  histogram = Counter(topk)
  active_experts = len(histogram)
  assignments = len(topk)
  maximum_group = max(histogram.values(), default=0)

  def padded_rows(tile: int) -> int:
    return sum(math.ceil(count / tile) * tile for count in histogram.values())

  pad8 = padded_rows(8)
  pad16 = padded_rows(16)
  pad32 = padded_rows(32)
  expert_tiles_m8 = pad8 // M_TILE
  gateup_chunks = GATEUP_OUTPUT // GATEUP_N_TILE
  down_chunks = FFN_OUTPUT // DOWN_N_TILE
  gateup_logical_tiles = expert_tiles_m8 * gateup_chunks
  down_logical_tiles = expert_tiles_m8 * down_chunks

  mac_per_row = GATEUP_OUTPUT * FFN_INPUT + FFN_OUTPUT * FFN_INTERMEDIATE
  m8_matrix_macs = pad8 * mac_per_row
  m32_matrix_macs = pad32 * mac_per_row
  matrix_budget_us = m8_matrix_macs / (MATRIX_RATE_TMAC_S * 1.0e6)

  native_confirm = grouped.get("native_runtime_probes", {}).get(
      "native-runtime-confirm", {}).get("result", {})
  first_stages = native_confirm.get("first_stage_us", [])
  gateup_us = float(first_stages[1]) if len(first_stages) == 5 else math.inf
  quantize_us = float(first_stages[2]) if len(first_stages) == 5 else math.inf
  down_us = float(first_stages[3]) if len(first_stages) == 5 else math.inf
  scatter_us = float(first_stages[4]) if len(first_stages) == 5 else math.inf
  router_schedule_us = float(native_confirm.get("device_schedule_us", math.inf))
  current_matrix_us = gateup_us + down_us
  current_padded_rate = m32_matrix_macs / current_matrix_us / 1.0e6
  required_rate_uplift = MATRIX_RATE_TMAC_S / current_padded_rate - 1.0
  fixed_nonmatrix_us = router_schedule_us + scatter_us
  complete_projection_us = matrix_budget_us + fixed_nonmatrix_us
  remaining_epilogue_us = FFN_CAP_US - complete_projection_us

  tile_budget_us = TOKENS / TARGET_PREFILL_TOK_S * 1.0e6
  state_medians = [
      float(row.get("probe", {}).get("state_core_median_us", math.nan))
      for row in state.get("rows", []) if isinstance(row, dict)]
  linear_medians = [
      float(value) for value in linear.get("complete_projection_medians_us", [])]
  linear_charge_us = 30 * (max(state_medians) + max(linear_medians))
  profile_run = profile.get("runs", [{}])[0]
  categories = profile_run.get("category_ms", {})
  other_allocation_us = (
      float(categories.get("linear_attention_conv_reorder", math.nan)) *
      1000 / PRODUCT_RATIO +
      (float(categories.get("full_attention_projection", math.nan)) +
       float(categories.get("full_attention_sdpa", math.nan))) *
      1000 / PRODUCT_RATIO)
  derived_ffn_cap_us = (
      tile_budget_us - linear_charge_us - other_allocation_us) / 40

  gateup_ze = args.gateup_ze_info.read_text(
      encoding="utf-8", errors="replace")
  down_ze = args.down_ze_info.read_text(encoding="utf-8", errors="replace")
  gateup_asm = args.gateup_asm.read_text(encoding="utf-8", errors="replace")
  down_asm = args.down_asm.read_text(encoding="utf-8", errors="replace")
  existing_resources = {
      "gateup_grf": first_number(gateup_ze, "grf_count"),
      "gateup_eu_threads": first_number(gateup_ze, "eu_thread_count"),
      "down_grf": first_number(down_ze, "grf_count"),
      "down_eu_threads": first_number(down_ze, "eu_thread_count"),
      "gateup_static_dpas": len(re.findall(r"\bdpas(?:\.|\b)", gateup_asm)),
      "down_static_dpas": len(re.findall(r"\bdpas(?:\.|\b)", down_asm)),
  }

  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("seq749_zero_readback_grouped_reference_passed",
            grouped.get("required_checks_passed") is True and
            native_confirm.get("device_schedule_host_upload_bytes") == 0 and
            native_confirm.get("device_schedule_host_read_bytes") == 0),
      check("seq758_closed_linear_route_is_input_not_candidate",
            linear.get("evaluation_completed") is True and
            linear.get("required_checks_passed") is False),
      check("real_layer27_histogram_locked",
            assignments == TOKENS * TOP_K and active_experts == 222 and
            maximum_group == 361 and min(histogram) >= 0 and
            max(histogram) < EXPERTS,
            assignments=assignments, active_experts=active_experts,
            maximum_group=maximum_group),
      check("m8_is_single_natural_dpas_padding_cut",
            pad32 == 12896 and pad16 == 10224 and pad8 == 9120 and
            pad8 < pad16 < pad32,
            padded_rows={"m32": pad32, "m16": pad16, "m8": pad8},
            padding_cut_vs_m32=1.0 - pad8 / pad32),
      check("current_generated_kernels_are_dpas_but_occupancy_limited",
            existing_resources["gateup_grf"] == 256 and
            existing_resources["down_grf"] == 256 and
            existing_resources["gateup_eu_threads"] == 4 and
            existing_resources["down_eu_threads"] == 4 and
            existing_resources["gateup_static_dpas"] > 0 and
            existing_resources["down_static_dpas"] > 0,
            resources=existing_resources),
      check("product_derived_ffn_cap_matches_adr0045",
            6252.0 <= derived_ffn_cap_us < 6253.0 and
            FFN_CAP_US <= derived_ffn_cap_us,
            tile_budget_us=tile_budget_us,
            linear_charge_us=linear_charge_us,
            other_allocation_us=other_allocation_us,
            derived_ffn_cap_us=derived_ffn_cap_us,
            registered_ffn_cap_us=FFN_CAP_US),
      check("fixed_m8_matrix_rate_and_complete_projection_clear_cap",
            matrix_budget_us < FFN_CAP_US and
            complete_projection_us < FFN_CAP_US and
            remaining_epilogue_us > 300.0,
            matrix_macs=m8_matrix_macs,
            matrix_rate_tmac_s=MATRIX_RATE_TMAC_S,
            matrix_budget_us=matrix_budget_us,
            router_schedule_us=router_schedule_us,
            scatter_us=scatter_us,
            complete_projection_us=complete_projection_us,
            remaining_epilogue_us=remaining_epilogue_us),
      check("required_rate_uplift_is_bounded_by_m8_resource_change",
            0.09 <= required_rate_uplift <= 0.11 and
            TARGET_GRF_COUNT * 2 == existing_resources["gateup_grf"] and
            TARGET_EU_THREADS == 2 * existing_resources["gateup_eu_threads"],
            current_padded_rate_tmac_s=current_padded_rate,
            required_rate_uplift=required_rate_uplift,
            target_grf_count=TARGET_GRF_COUNT,
            target_eu_threads=TARGET_EU_THREADS),
      check("fused_local_intermediate_fits_slm",
            SLM_BYTES == 16384 and SLM_BYTES <= 65536,
            slm_bytes=SLM_BYTES),
      check("single_fixed_design_has_no_sweep_axis",
            M_TILE == 8 and GATEUP_N_TILE == 32 and DOWN_N_TILE == 64 and
            PERSISTENT_WORKGROUPS == 96,
            expert_tiles=expert_tiles_m8,
            gateup_logical_tiles=gateup_logical_tiles,
            down_logical_tiles=down_logical_tiles,
            persistent_workgroups=PERSISTENT_WORKGROUPS),
  ]
  required = all(bool(item["pass"]) for item in checks)
  disposition = (
      "accept_fixed_m8_expert_major_fused_ffn_source_gate"
      if required else "reject_fused_ffn_design_return_product_reflection")
  selected_next_route = (
      "native_prefill_fused_expert_major_m8_ffn_source_gate"
      if required else "native_prefill_product_route_reflection_gate")
  reason = (
      "The real histogram makes M8 the natural no-sweep DPAS unit: it cuts "
      "padded rows 29.3%, needs only about 10% more padded-MAC rate than the "
      "existing 256-GRF kernels, and leaves over 0.3 ms inside the 6.25 ms "
      "complete cap. Implement one 128-GRF/eight-thread persistent source."
      if required else
      "The fixed M8 design cannot account for the registered product cap; "
      "do not write a fused kernel or sweep tile/resource shapes.")
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "inputs": {"topk": relative(args.topk),
                 "grouped_result": relative(args.grouped_result),
                 "linear_result": relative(args.linear_result),
                 "state_result": relative(args.state_result),
                 "openvino_profile": relative(args.openvino_profile)},
      "histogram": {"active_experts": active_experts,
                    "assignments": assignments,
                    "maximum_group": maximum_group,
                    "counts": {str(key): histogram[key]
                               for key in sorted(histogram)}},
      "design": {"m_tile": M_TILE, "gateup_n_tile": GATEUP_N_TILE,
                 "down_n_tile": DOWN_N_TILE,
                 "persistent_workgroups": PERSISTENT_WORKGROUPS,
                 "target_grf_count": TARGET_GRF_COUNT,
                 "target_eu_threads": TARGET_EU_THREADS,
                 "slm_bytes": SLM_BYTES,
                 "padded_rows": {"m8": pad8, "m16": pad16, "m32": pad32},
                 "matrix_macs": m8_matrix_macs,
                 "matrix_rate_tmac_s": MATRIX_RATE_TMAC_S,
                 "matrix_budget_us": matrix_budget_us,
                 "fixed_nonmatrix_us": fixed_nonmatrix_us,
                 "complete_projection_us": complete_projection_us,
                 "remaining_epilogue_us": remaining_epilogue_us,
                 "current_padded_rate_tmac_s": current_padded_rate,
                 "required_rate_uplift": required_rate_uplift,
                 "gateup_logical_tiles": gateup_logical_tiles,
                 "down_logical_tiles": down_logical_tiles},
      "product_budget": {"tile_budget_us": tile_budget_us,
                         "linear_charge_us": linear_charge_us,
                         "other_allocation_us": other_allocation_us,
                         "derived_ffn_cap_us": derived_ffn_cap_us,
                         "registered_ffn_cap_us": FFN_CAP_US},
      "existing_resources": existing_resources,
      "checks": checks, "required_checks_passed": required,
      "disposition": disposition, "selected_next_route": selected_next_route,
      "next_route_reason": reason,
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "manifest.json").write_text(json.dumps({
      "schema_version": SCHEMA, "created_at": created_at, "commit": commit,
      "git_dirty": bool(dirty), "required_checks_passed": required,
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [item["name"] for item in checks if not item["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Fused expert-major FFN design gate", "",
      f"- required_checks_passed: `{str(required).lower()}`",
      f"- disposition: `{disposition}`",
      f"- real padded rows M32/M16/M8: `{pad32} / {pad16} / {pad8}`",
      f"- M8 matrix MACs / budget: `{m8_matrix_macs} / "
      f"{matrix_budget_us:.3f} us`",
      f"- projected complete / cap: `{complete_projection_us:.3f} / "
      f"{FFN_CAP_US:.3f} us`",
      f"- current padded rate / required uplift: "
      f"`{current_padded_rate:.3f} TMAC/s / {required_rate_uplift:.3%}`",
      f"- failed checks: `{failed}`", "", reason, ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required, "disposition": disposition,
      "matrix_budget_us": matrix_budget_us,
      "complete_projection_us": complete_projection_us,
      "required_rate_uplift": required_rate_uplift,
      "selected_next_route": selected_next_route,
      "out_dir": relative(out)}, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
