#!/usr/bin/env python3
"""Build and run the integrated native candidate JSONL generator.

The generated candidate rows declare ``native_output_source=intel_qwen36_native``
and are fed to the R1 native correctness gate. This tool is allowed to produce
a diagnostic non-closing artifact: when run with fewer than six cases or fewer
than 16 generated tokens, the gate should stay open and report the exact
missing/replay mismatch.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-native-candidate-jsonl-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-native-candidate-jsonl-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_SEED = ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
DEFAULT_TOKEN_INPUT = ROOT / "output/r1-engine-seed-prompt-input-check-20260627T155328Z/token-input"
GATE_TOOL = ROOT / "tools/intel-qwen36-r1-native-correctness-gate.py"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/resident_harness.hpp", "include/intel_qwen36/resident_harness.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/resident_harness.cpp", "src/resident_harness.cpp"),
    ("engine/tests/native_candidate_jsonl.cpp", "tests/native_candidate_jsonl.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-seed", type=Path, default=DEFAULT_ORACLE_SEED)
  parser.add_argument("--token-input-dir", type=Path, default=DEFAULT_TOKEN_INPUT)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--max-new-tokens", type=int, default=1)
  parser.add_argument("--warmup-runs", type=int, default=0)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument(
      "--resident-cache",
      action="store_true",
      help="Enable process-resident tensor/decoded-row cache in the native loop.",
  )
  parser.add_argument(
      "--profile-matvec",
      action="store_true",
      help="Enable per-tensor matvec profiling in the native loop.",
  )
  parser.add_argument(
      "--prefill-final-logits-only",
      action="store_true",
      help="Only compute LM-head logits for the final prompt token during prefill.",
  )
  parser.add_argument(
      "--decode-top1-only",
      action="store_true",
      help="Only compute top-1 logits for decode continuation tokens.",
  )
  parser.add_argument(
      "--full-attention-inplace-history",
      action="store_true",
      help="Update full-attention K/V history in place in the native runner.",
  )
  parser.add_argument(
      "--lm-head-top-k",
      action="store_true",
      help="Use the switch-gated fused top-k matvec route for output.weight.",
  )
  parser.add_argument(
      "--lm-head-threads",
      type=int,
      default=1,
      help="Thread count for --lm-head-top-k. Defaults to 1.",
  )
  parser.add_argument(
      "--lm-head-q6-pair-dot",
      action="store_true",
      help="Use paired direct Q6_K row-dot route for LM-head top-k rows.",
  )
  parser.add_argument(
      "--expert-slice-matvec",
      action="store_true",
      help="Use contiguous selected-expert slice reads for expert matvecs.",
  )
  parser.add_argument(
      "--expert-slice-threads",
      type=int,
      default=1,
      help="Thread count for --expert-slice-matvec. Defaults to 1.",
  )
  parser.add_argument(
      "--dense-matvec",
      action="store_true",
      help="Use parallel dense matvec route for large generic matvec_tensor calls.",
  )
  parser.add_argument(
      "--dense-matvec-threads",
      type=int,
      default=1,
      help="Thread count for --dense-matvec. Defaults to 1.",
  )
  parser.add_argument(
      "--dense-matvec-min-rows",
      type=int,
      default=1024,
      help="Minimum tensor row count for --dense-matvec. Defaults to 1024.",
  )
  parser.add_argument(
      "--dense-matvec-payload-cache",
      action="store_true",
      help="Cache selected dense matvec tensor payloads in the resident process.",
  )
  parser.add_argument(
      "--dense-q4-direct-dot",
      action="store_true",
      help="Use the direct Q4_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q4-pair-dot",
      action="store_true",
      help="Use paired direct Q4_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q6-direct-dot",
      action="store_true",
      help="Use the direct Q6_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--dense-q6-pair-dot",
      action="store_true",
      help="Use paired direct Q6_K row-dot route for dense matvec rows.",
  )
  parser.add_argument(
      "--q4-direct-minsum-pair",
      action="store_true",
      help="Use paired bsums for the Q4_K direct min-sum term.",
  )
  parser.add_argument(
      "--q4-block-meta-cache",
      action="store_true",
      help="Cache decoded Q4_K block metadata for resident tensor payload dots.",
  )
  parser.add_argument(
      "--q4-plane-layout",
      action="store_true",
      help="Enable the R3 q4k_plane_v0 layout route for selected Q4_K lanes.",
  )
  parser.add_argument(
      "--dense-q4-plane-pair-dot",
      action="store_true",
      help="Enable fused q4-plane pair dots for generic dense Q4_K rows.",
  )
  parser.add_argument(
      "--small-q4-direct-dot",
      action="store_true",
      help="Use direct Q4_K dot for cached non-dense small Q4 matvec rows.",
  )
  parser.add_argument(
      "--matvec-q8-input-reuse",
      action="store_true",
      help="Reuse Q8_K activation blocks across same-input quantized matvec calls.",
  )
  parser.add_argument(
      "--shared-parallel-executor",
      action="store_true",
      help="Use the switch-gated shared worker executor for parallel matvec routes.",
  )
  parser.add_argument(
      "--shared-expert-gate-up-fused",
      action="store_true",
      help="Use the switch-gated fused direct Q4_K route for shared expert gate/up.",
  )
  parser.add_argument(
      "--selected-expert-ffn",
      action="store_true",
      help="Use the switch-gated fused selected-expert FFN route.",
  )
  parser.add_argument(
      "--selected-expert-ffn-threads",
      type=int,
      default=1,
      help="Thread count for --selected-expert-ffn. Defaults to 1.",
  )
  parser.add_argument(
      "--selected-expert-minimal-outputs",
      action="store_true",
      help="Skip diagnostic selected-expert intermediate vectors in the native loop.",
  )
  parser.add_argument(
      "--selected-expert-slice-cache",
      action="store_true",
      help="Cache selected expert tensor slices in the resident process.",
  )
  parser.add_argument(
      "--selected-expert-down-slice-cache",
      action="store_true",
      help="Cache selected expert down tensor slices in the resident process.",
  )
  parser.add_argument(
      "--selected-expert-down-expert-major",
      action="store_true",
      help="Compute selected-expert down rows expert-major, then aggregate in selected order.",
  )
  parser.add_argument(
      "--selected-expert-down-q4-pair-dot",
      action="store_true",
      help="Use a Q4_K row-pair dot route for adjacent selected-expert down rows.",
  )
  parser.add_argument(
      "--selected-expert-down-q6-pair-dot",
      action="store_true",
      help="Use a Q6_K row-pair dot route for adjacent selected-expert down rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-direct-dot",
      action="store_true",
      help="Use the direct Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-pair-dot",
      action="store_true",
      help="Use the paired direct Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-pair-sum-dot",
      action="store_true",
      help="Use the paired block-sum Q4_K dot route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--selected-gate-q4-plane-pair-dot",
      action="store_true",
      help="Use the q4-plane fused pair route for selected-expert gate/up rows.",
  )
  parser.add_argument(
      "--case-id",
      action="append",
      default=[],
      help="Case id to run. Repeatable. Defaults to all six seed cases.",
  )
  return parser.parse_args()


def run(cmd: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    return {
        "cmd": cmd,
        "returncode": 124,
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + "\ntimeout",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
    }
  return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
  }


def run_target(host: str, remote_command: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_to(host: str, local_path: Path, remote_path: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_to(host, local_path, remote_path, timeout_s)


def copy_tree_to(host: str, local_dir: Path, remote_dir: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_tree_to(host, local_dir, remote_dir, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected object row")
      rows.append(value)
  return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def build_candidate_rows(
    oracle_rows: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
  parsed_cases = {
      case["case_id"]: case
      for case in parsed.get("cases", [])
      if isinstance(case, dict) and isinstance(case.get("case_id"), str)
  }
  candidate_rows: list[dict[str, Any]] = []
  for oracle in oracle_rows:
    case_id = oracle.get("case_id")
    if case_id not in parsed_cases:
      continue
    native = parsed_cases[case_id]
    generated = native.get("generated_token_ids", [])
    first_signature = native.get("first_token_top_logprob_id_signature", [])
    targets = []
    for target in oracle.get("generation_targets", []):
      name = target.get("target")
      if name == "first_token":
        ids = generated[:1]
      elif name == "short_generation":
        ids = generated
      else:
        ids = []
      targets.append({
          "generated_token_count": len(ids),
          "generated_token_ids": ids,
          "max_new_tokens": target.get("max_new_tokens"),
          "target": name,
          "top_logprob_id_signature": first_signature,
      })
    candidate_rows.append({
        "case_id": case_id,
        "generation_targets": targets,
        "native_output_source": "intel_qwen36_native",
        "prompt": oracle.get("prompt"),
        "prompt_set": oracle.get("prompt_set"),
        "prompt_token_count": oracle.get("prompt_token_count"),
        "prompt_token_ids": oracle.get("prompt_token_ids"),
        "prompt_utf8_sha256": oracle.get("prompt_utf8_sha256"),
        "schema_version": "intel-qwen36-native-candidate-jsonl-v0",
        "workstream": WORKSTREAM,
      })
  return candidate_rows


def build_summary(payload: dict[str, Any], gate_closed: bool) -> str:
  state = payload["native_candidate_jsonl"]
  lines = [
      "# R1 Native Candidate JSONL",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- case ids: `{state['case_ids']}`",
      f"- max new tokens run: {state['max_new_tokens']}",
      f"- prefill final logits only: `{str(state['prefill_final_logits_only']).lower()}`",
      f"- decode top1 only: `{str(state['decode_top1_only']).lower()}`",
      f"- full-attention inplace history: `{str(state['full_attention_inplace_history']).lower()}`",
      f"- dense matvec route: `{str(state['dense_matvec']).lower()}`",
      f"- dense matvec threads: {state['dense_matvec_threads']}",
      f"- dense matvec min rows: {state['dense_matvec_min_rows']}",
      f"- dense matvec payload cache: `{str(state['dense_matvec_payload_cache']).lower()}`",
      f"- dense Q4 direct dot: `{str(state['dense_q4_direct_dot']).lower()}`",
      f"- dense Q4 pair dot: `{str(state['dense_q4_pair_dot']).lower()}`",
      f"- dense Q6 direct dot: `{str(state['dense_q6_direct_dot']).lower()}`",
      f"- dense Q6 pair dot: `{str(state['dense_q6_pair_dot']).lower()}`",
      f"- Q4 direct min-sum pair: `{str(state['q4_direct_minsum_pair']).lower()}`",
      f"- Q4 block meta cache: `{str(state['q4_block_meta_cache']).lower()}`",
      f"- q4-plane layout: `{str(state['q4_plane_layout']).lower()}`",
      f"- dense Q4 plane-pair dot: `{str(state['dense_q4_plane_pair_dot']).lower()}`",
      f"- small Q4 direct dot: `{str(state['small_q4_direct_dot']).lower()}`",
      f"- matvec Q8 input reuse: `{str(state['matvec_q8_input_reuse']).lower()}`",
      f"- LM-head top-k route: `{str(state['lm_head_top_k']).lower()}`",
      f"- LM-head Q6 pair dot: `{str(state['lm_head_q6_pair_dot']).lower()}`",
      f"- LM-head threads: {state['lm_head_threads']}",
      f"- expert-slice matvec route: `{str(state['expert_slice_matvec']).lower()}`",
      f"- expert-slice threads: {state['expert_slice_threads']}",
      f"- shared parallel executor: `{str(state['shared_parallel_executor']).lower()}`",
      f"- shared expert gate/up fused route: `{str(state['shared_expert_gate_up_fused']).lower()}`",
      f"- selected-expert FFN route: `{str(state['selected_expert_ffn']).lower()}`",
      f"- selected-expert FFN threads: {state['selected_expert_ffn_threads']}",
      f"- selected-expert minimal outputs: `{str(state['selected_expert_minimal_outputs']).lower()}`",
      f"- selected expert slice cache: `{str(state['selected_expert_slice_cache']).lower()}`",
      f"- selected expert down slice cache: `{str(state['selected_expert_down_slice_cache']).lower()}`",
      f"- selected expert down expert-major: `{str(state['selected_expert_down_expert_major']).lower()}`",
      f"- selected expert down Q4 pair dot: `{str(state['selected_expert_down_q4_pair_dot']).lower()}`",
      f"- selected expert down Q6 pair dot: `{str(state['selected_expert_down_q6_pair_dot']).lower()}`",
      f"- selected gate Q4 direct dot: `{str(state['selected_gate_q4_direct_dot']).lower()}`",
      f"- selected gate Q4 pair dot: `{str(state['selected_gate_q4_pair_dot']).lower()}`",
      f"- selected gate Q4 pair-sum dot: `{str(state['selected_gate_q4_pair_sum_dot']).lower()}`",
      f"- selected gate Q4 plane-pair dot: `{str(state['selected_gate_q4_plane_pair_dot']).lower()}`",
      f"- candidate rows: {state['candidate_row_count']}",
      f"- gate artifact: `{state['gate_artifact']}`",
      f"- R1 native correctness gate closed: `{str(gate_closed).lower()}`",
      "",
      "This is native candidate evidence. If the gate remains open, use the",
      "nested gate artifact to inspect missing cases or replay mismatches.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.max_new_tokens < 1 or args.max_new_tokens > 16:
    raise SystemExit("--max-new-tokens must be 1..16")
  if args.warmup_runs < 0 or args.warmup_runs > 8:
    raise SystemExit("--warmup-runs must be 0..8")
  if args.timed_runs < 1 or args.timed_runs > 8:
    raise SystemExit("--timed-runs must be 1..8")
  if args.lm_head_threads < 1 or args.lm_head_threads > 256:
    raise SystemExit("--lm-head-threads must be 1..256")
  if args.expert_slice_threads < 1 or args.expert_slice_threads > 256:
    raise SystemExit("--expert-slice-threads must be 1..256")
  if args.dense_matvec_threads < 1 or args.dense_matvec_threads > 256:
    raise SystemExit("--dense-matvec-threads must be 1..256")
  if args.dense_matvec_min_rows < 1 or args.dense_matvec_min_rows > 1048576:
    raise SystemExit("--dense-matvec-min-rows must be 1..1048576")
  if args.selected_expert_ffn_threads < 1 or args.selected_expert_ffn_threads > 256:
    raise SystemExit("--selected-expert-ffn-threads must be 1..256")
  if args.dense_q4_plane_pair_dot and not args.q4_plane_layout:
    raise SystemExit("--dense-q4-plane-pair-dot requires --q4-plane-layout")
  if args.selected_gate_q4_plane_pair_dot and not args.q4_plane_layout:
    raise SystemExit("--selected-gate-q4-plane-pair-dot requires --q4-plane-layout")
  if not args.token_input_dir.exists():
    raise SystemExit(f"token input dir missing: {args.token_input_dir}")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-native-candidate-jsonl-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/native-candidate-jsonl-{stamp}"
  remote_token_dir = f"{remote_dir}/tokens"

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "tokens")
      ),
      args.timeout_s,
  )
  source_transfers: list[dict[str, Any]] = []
  token_transfer: dict[str, Any] = {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  if mkdir.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    if all(item.get("returncode") == 0 for item in source_transfers):
      token_transfer = copy_tree_to(
          args.host,
          args.token_input_dir.resolve(),
          remote_token_dir,
          args.timeout_s,
      )

  staged = (
      mkdir.get("returncode") == 0
      and all(item.get("returncode") == 0 for item in source_transfers)
      and token_transfer.get("returncode") == 0
  )
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/src/resident_harness.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/native_candidate_jsonl.cpp')} "
      "-pthread "
      f"-o {shlex.quote(remote_dir + '/build/iq36-native-candidate-jsonl')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  case_args = " ".join(shlex.quote(case_id) for case_id in args.case_id)
  run_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-native-candidate-jsonl"),
      shlex.quote(args.model),
      shlex.quote(remote_token_dir),
      str(args.max_new_tokens),
      "--warmup-runs",
      str(args.warmup_runs),
      "--timed-runs",
      str(args.timed_runs),
      "--resident-cache" if args.resident_cache else "",
      "--profile-matvec" if args.profile_matvec else "",
      "--prefill-final-logits-only" if args.prefill_final_logits_only else "",
      "--decode-top1-only" if args.decode_top1_only else "",
      "--full-attention-inplace-history"
      if args.full_attention_inplace_history
      else "",
      "--lm-head-top-k" if args.lm_head_top_k else "",
      "--lm-head-threads",
      str(args.lm_head_threads),
      "--lm-head-q6-pair-dot" if args.lm_head_q6_pair_dot else "",
      "--expert-slice-matvec" if args.expert_slice_matvec else "",
      "--expert-slice-threads",
      str(args.expert_slice_threads),
      "--dense-matvec" if args.dense_matvec else "",
      "--dense-matvec-threads",
      str(args.dense_matvec_threads),
      "--dense-matvec-min-rows",
      str(args.dense_matvec_min_rows),
      "--dense-matvec-payload-cache" if args.dense_matvec_payload_cache else "",
      "--dense-q4-direct-dot" if args.dense_q4_direct_dot else "",
      "--dense-q4-pair-dot" if args.dense_q4_pair_dot else "",
      "--dense-q6-direct-dot" if args.dense_q6_direct_dot else "",
      "--dense-q6-pair-dot" if args.dense_q6_pair_dot else "",
      "--q4-direct-minsum-pair" if args.q4_direct_minsum_pair else "",
      "--q4-block-meta-cache" if args.q4_block_meta_cache else "",
      "--q4-plane-layout" if args.q4_plane_layout else "",
      "--dense-q4-plane-pair-dot" if args.dense_q4_plane_pair_dot else "",
      "--small-q4-direct-dot" if args.small_q4_direct_dot else "",
      "--matvec-q8-input-reuse" if args.matvec_q8_input_reuse else "",
      "--shared-parallel-executor" if args.shared_parallel_executor else "",
      "--shared-expert-gate-up-fused" if args.shared_expert_gate_up_fused else "",
      "--selected-expert-ffn" if args.selected_expert_ffn else "",
      "--selected-expert-ffn-threads",
      str(args.selected_expert_ffn_threads),
      "--selected-expert-minimal-outputs" if args.selected_expert_minimal_outputs else "",
      "--selected-expert-slice-cache" if args.selected_expert_slice_cache else "",
      "--selected-expert-down-slice-cache" if args.selected_expert_down_slice_cache else "",
      "--selected-expert-down-expert-major" if args.selected_expert_down_expert_major else "",
      "--selected-expert-down-q4-pair-dot" if args.selected_expert_down_q4_pair_dot else "",
      "--selected-expert-down-q6-pair-dot" if args.selected_expert_down_q6_pair_dot else "",
      "--selected-gate-q4-direct-dot" if args.selected_gate_q4_direct_dot else "",
      "--selected-gate-q4-pair-dot" if args.selected_gate_q4_pair_dot else "",
      "--selected-gate-q4-pair-sum-dot" if args.selected_gate_q4_pair_sum_dot else "",
      "--selected-gate-q4-plane-pair-dot"
      if args.selected_gate_q4_plane_pair_dot
      else "",
      case_args,
  ]).strip()
  target_run = (
      run_target(args.host, run_command, args.timeout_s)
      if build.get("returncode") == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )
  parsed: dict[str, Any] = {}
  parse_error = None
  if target_run.get("stdout"):
    try:
      parsed = json.loads(target_run["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  oracle_rows = load_jsonl(args.oracle_seed)
  candidate_rows = build_candidate_rows(oracle_rows, parsed) if parsed else []
  candidate_jsonl = out_dir / "candidate.jsonl"
  write_jsonl(candidate_jsonl, candidate_rows)

  gate_dir = out_dir / "gate"
  gate = run([
      "python3",
      str(GATE_TOOL),
      "--candidate-jsonl",
      str(candidate_jsonl),
      "--oracle-jsonl",
      str(args.oracle_seed),
      "--out-dir",
      str(gate_dir),
  ], args.timeout_s)
  gate_json = {}
  gate_closed = False
  gate_path = gate_dir / "gate.json"
  if gate_path.exists():
    gate_json = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_closed = (
        gate_json.get("r1_native_correctness_gate", {})
        .get("r1_native_correctness_gate_closed") is True
    )

  state = {
      "candidate_jsonl": str(candidate_jsonl.relative_to(ROOT)),
      "candidate_row_count": len(candidate_rows),
      "case_ids": [row.get("case_id") for row in candidate_rows],
      "decode_top1_only": args.decode_top1_only,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "q4_plane_layout": args.q4_plane_layout,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "expert_slice_matvec": args.expert_slice_matvec,
      "expert_slice_threads": args.expert_slice_threads,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "gate_artifact": str(gate_dir.relative_to(ROOT)),
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "max_new_tokens": args.max_new_tokens,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "profile_matvec": args.profile_matvec,
      "resident_cache": args.resident_cache,
      "requested_case_ids": args.case_id,
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "target_build_returncode": build.get("returncode"),
      "target_run_returncode": target_run.get("returncode"),
      "timed_runs": args.timed_runs,
      "warmup_runs": args.warmup_runs,
  }
  payload = {
      "created_at": created_at,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "decode_top1_only": args.decode_top1_only,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "q4_plane_layout": args.q4_plane_layout,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "host": args.host,
      "expert_slice_matvec": args.expert_slice_matvec,
      "expert_slice_threads": args.expert_slice_threads,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "model_path": args.model,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "native_candidate_jsonl": state,
      "native_candidate_jsonl_emitted": bool(candidate_rows),
      "parse_error": parse_error,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "r1_native_correctness_gate_closed": gate_closed,
      "remote_dir": remote_dir,
      "profile_matvec": args.profile_matvec,
      "resident_cache": args.resident_cache,
      "schema_version": SCHEMA_VERSION,
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "speedup_claims_allowed": False,
      "target_build": build,
      "target_run": target_run,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "decode_top1_only": args.decode_top1_only,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "q4_plane_layout": args.q4_plane_layout,
      "dense_q4_plane_pair_dot": args.dense_q4_plane_pair_dot,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "host": args.host,
      "expert_slice_matvec": args.expert_slice_matvec,
      "expert_slice_threads": args.expert_slice_threads,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "max_new_tokens": args.max_new_tokens,
      "model_path": args.model,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "profile_matvec": args.profile_matvec,
      "remote_dir": remote_dir,
      "resident_cache": args.resident_cache,
      "schema_version": SCHEMA_VERSION,
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "selected_gate_q4_plane_pair_dot": args.selected_gate_q4_plane_pair_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "timed_runs": args.timed_runs,
      "tool": "tools/intel-qwen36-r1-native-candidate-jsonl.py",
      "warmup_runs": args.warmup_runs,
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "decode_top1_only": args.decode_top1_only,
      "dense_matvec": args.dense_matvec,
      "dense_matvec_min_rows": args.dense_matvec_min_rows,
      "dense_matvec_payload_cache": args.dense_matvec_payload_cache,
      "dense_matvec_threads": args.dense_matvec_threads,
      "dense_q4_direct_dot": args.dense_q4_direct_dot,
      "dense_q4_pair_dot": args.dense_q4_pair_dot,
      "dense_q6_direct_dot": args.dense_q6_direct_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "expert_slice_matvec": args.expert_slice_matvec,
      "expert_slice_threads": args.expert_slice_threads,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "lm_head_threads": args.lm_head_threads,
      "lm_head_top_k": args.lm_head_top_k,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "remote_dir": remote_dir,
      "profile_matvec": args.profile_matvec,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "resident_cache": args.resident_cache,
      "remote_token_dir": remote_token_dir,
      "selected_expert_ffn": args.selected_expert_ffn,
      "selected_expert_ffn_threads": args.selected_expert_ffn_threads,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "selected_expert_slice_cache": args.selected_expert_slice_cache,
      "selected_expert_down_slice_cache": args.selected_expert_down_slice_cache,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "selected_gate_q4_direct_dot": args.selected_gate_q4_direct_dot,
      "selected_gate_q4_pair_dot": args.selected_gate_q4_pair_dot,
      "selected_gate_q4_pair_sum_dot": args.selected_gate_q4_pair_sum_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
      "token_transfer": token_transfer,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "native-candidate-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "candidate.json", payload)
  write_json(out_dir / "gate-run.json", gate)
  checks = [
      {"name": "seed_token_inputs_present", "pass": args.token_input_dir.exists()},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {"name": "seed_prompt_token_inputs_transferred", "pass": token_transfer.get("returncode") == 0},
      {"name": "target_native_candidate_generator_built", "pass": build.get("returncode") == 0},
      {"name": "target_native_candidate_generator_ran", "pass": target_run.get("returncode") == 0},
      {"name": "target_native_candidate_generator_output_parsed", "pass": bool(parsed)},
      {"name": "native_candidate_jsonl_emitted", "pass": bool(candidate_rows)},
      {"name": "r1_native_correctness_gate_ran", "pass": gate.get("returncode") == 0 and gate_path.exists()},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r1_native_candidate_jsonl_generation",
      "native_candidate_jsonl_emitted": bool(candidate_rows),
      "r1_native_correctness_gate_closed": gate_closed,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "decode_top1_only": args.decode_top1_only,
      "full_attention_inplace_history": args.full_attention_inplace_history,
      "prefill_final_logits_only": args.prefill_final_logits_only,
      "selected_expert_minimal_outputs": args.selected_expert_minimal_outputs,
      "matvec_q8_input_reuse": args.matvec_q8_input_reuse,
      "lm_head_q6_pair_dot": args.lm_head_q6_pair_dot,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_direct_minsum_pair": args.q4_direct_minsum_pair,
      "q4_block_meta_cache": args.q4_block_meta_cache,
      "small_q4_direct_dot": args.small_q4_direct_dot,
      "selected_expert_down_expert_major": args.selected_expert_down_expert_major,
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "shared_expert_gate_up_fused": args.shared_expert_gate_up_fused,
      "shared_parallel_executor": args.shared_parallel_executor,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("native_candidate_jsonl_emitted", bool(candidate_rows)),
        ("native_candidate_row_count", len(candidate_rows)),
        ("decode_top1_only", args.decode_top1_only),
        ("dense_matvec", args.dense_matvec),
        ("dense_matvec_min_rows", args.dense_matvec_min_rows),
        ("dense_matvec_payload_cache", args.dense_matvec_payload_cache),
        ("dense_matvec_threads", args.dense_matvec_threads),
        ("dense_q4_direct_dot", args.dense_q4_direct_dot),
        ("dense_q4_pair_dot", args.dense_q4_pair_dot),
        ("dense_q6_direct_dot", args.dense_q6_direct_dot),
        ("dense_q6_pair_dot", args.dense_q6_pair_dot),
        ("q4_direct_minsum_pair", args.q4_direct_minsum_pair),
        ("q4_block_meta_cache", args.q4_block_meta_cache),
        ("q4_plane_layout", args.q4_plane_layout),
        ("dense_q4_plane_pair_dot", args.dense_q4_plane_pair_dot),
        ("small_q4_direct_dot", args.small_q4_direct_dot),
        ("expert_slice_matvec", args.expert_slice_matvec),
        ("expert_slice_threads", args.expert_slice_threads),
        (
            "full_attention_inplace_history",
            args.full_attention_inplace_history,
        ),
        ("max_new_tokens", args.max_new_tokens),
        ("lm_head_q6_pair_dot", args.lm_head_q6_pair_dot),
        ("lm_head_threads", args.lm_head_threads),
        ("lm_head_top_k", args.lm_head_top_k),
        ("matvec_q8_input_reuse", args.matvec_q8_input_reuse),
        ("prefill_final_logits_only", args.prefill_final_logits_only),
        ("profile_matvec", args.profile_matvec),
        ("resident_cache", args.resident_cache),
        ("selected_expert_ffn", args.selected_expert_ffn),
        ("selected_expert_ffn_threads", args.selected_expert_ffn_threads),
        ("selected_expert_minimal_outputs", args.selected_expert_minimal_outputs),
        ("selected_expert_slice_cache", args.selected_expert_slice_cache),
        ("selected_expert_down_slice_cache", args.selected_expert_down_slice_cache),
        ("selected_expert_down_expert_major", args.selected_expert_down_expert_major),
        ("selected_expert_down_q4_pair_dot", args.selected_expert_down_q4_pair_dot),
        ("selected_expert_down_q6_pair_dot", args.selected_expert_down_q6_pair_dot),
        ("selected_gate_q4_direct_dot", args.selected_gate_q4_direct_dot),
        ("selected_gate_q4_pair_dot", args.selected_gate_q4_pair_dot),
        ("selected_gate_q4_pair_sum_dot", args.selected_gate_q4_pair_sum_dot),
        ("selected_gate_q4_plane_pair_dot", args.selected_gate_q4_plane_pair_dot),
        ("shared_expert_gate_up_fused", args.shared_expert_gate_up_fused),
        ("shared_parallel_executor", args.shared_parallel_executor),
        ("timed_runs", args.timed_runs),
        ("warmup_runs", args.warmup_runs),
        ("r1_native_correctness_gate_closed", gate_closed),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_native_candidate_jsonl_generation",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload, gate_closed), encoding="utf-8")
  print(f"r1 native candidate JSONL output: {out_dir}")
  return 0 if all(check["pass"] for check in checks) else 1


if __name__ == "__main__":
  raise SystemExit(main())
