#!/usr/bin/env python3
"""Run current native engine route on long-context prompt-token buckets.

This is a diagnostic tool for the cold no-prefix context ladder. It reuses the
integrated native C++ runner, but it does not emit R1 candidate JSONL rows and
does not run the R1 gate. Each case is run in a separate target process by
default so resident in-process caches cannot leak from one bucket to the next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-context-ladder-native-diagnostic-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-native-candidate-jsonl-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_TOKEN_ID_REFS = (
    ROOT
    / "output/r0-oracle-token-id-capture-20260626T083347Z"
    / "prompt-token-id-references.jsonl"
)
DEFAULT_CASE_IDS = ("sentinel_001k", "prefill_shape_001k")
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

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
  parser.add_argument("--token-id-refs", type=Path, default=DEFAULT_TOKEN_ID_REFS)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=21600)
  parser.add_argument("--max-new-tokens", type=int, default=1)
  parser.add_argument("--warmup-runs", type=int, default=0)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument(
      "--case-id",
      action="append",
      default=[],
      help=(
          "Case id from the token-id reference JSONL. Repeatable. Defaults to "
          "sentinel_001k and prefill_shape_001k."
      ),
  )
  parser.add_argument(
      "--bucket",
      action="append",
      default=[],
      help=(
          "Run both sentinel_<bucket> and prefill_shape_<bucket>, for example "
          "001k or 016k. Repeatable. Ignored when --case-id is supplied."
      ),
  )
  parser.add_argument(
      "--combined-process",
      action="store_true",
      help="Run selected cases in one target process instead of one process per case.",
  )
  parser.add_argument(
      "--no-resident-cache",
      dest="resident_cache",
      action="store_false",
      help="Disable resident tensor/slice caches in the native runner.",
  )
  parser.set_defaults(resident_cache=True)
  parser.add_argument(
      "--profile-matvec",
      action="store_true",
      help="Enable per-tensor matvec profiling. Disabled by default for timing.",
  )
  parser.add_argument(
      "--dense-q6-pair-dot",
      action="store_true",
      help="Enable the default-off dense Q6 row-pair dot route.",
  )
  parser.add_argument(
      "--selected-expert-down-q4-pair-dot",
      action="store_true",
      help="Enable the default-off selected-expert down Q4 row-pair dot route.",
  )
  parser.add_argument(
      "--selected-expert-down-q6-pair-dot",
      action="store_true",
      help="Enable the default-off selected-expert down Q6 row-pair dot route.",
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
      "--selected-gate-q4-plane-pair-dot",
      action="store_true",
      help="Enable fused q4-plane pair dots for selected-expert gate/up rows.",
  )
  return parser.parse_args()


def fnv64(data: bytes) -> int:
  value = FNV_OFFSET
  for byte in data:
    value ^= byte
    value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
  return value


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      text = line.strip()
      if not text:
        continue
      row = json.loads(text)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(row)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def case_ids_from_args(args: argparse.Namespace) -> list[str]:
  if args.case_id:
    return list(dict.fromkeys(args.case_id))
  if args.bucket:
    out: list[str] = []
    for bucket in args.bucket:
      out.append(f"sentinel_{bucket}")
      out.append(f"prefill_shape_{bucket}")
    return list(dict.fromkeys(out))
  return list(DEFAULT_CASE_IDS)


def selected_rows(token_refs: Path, case_ids: list[str]) -> list[dict[str, Any]]:
  rows = load_jsonl(token_refs)
  by_case = {row.get("case_id"): row for row in rows}
  missing = [case_id for case_id in case_ids if case_id not in by_case]
  if missing:
    raise SystemExit(f"missing token-id reference cases: {missing}")
  out = [by_case[case_id] for case_id in case_ids]
  for row in out:
    token_ids = row.get("prompt_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
      raise SystemExit(f"{row.get('case_id')}: missing prompt_token_ids")
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
      raise SystemExit(f"{row.get('case_id')}: invalid prompt_token_ids")
    if row.get("prompt_token_count") != len(token_ids):
      raise SystemExit(f"{row.get('case_id')}: prompt token count mismatch")
  return out


def prepare_token_inputs(rows: list[dict[str, Any]], token_dir: Path) -> dict[str, Any]:
  token_dir.mkdir(parents=True, exist_ok=True)
  lines: list[str] = []
  cases: dict[str, dict[str, Any]] = {}
  total_prompt_tokens = 0
  for row in rows:
    case_id = str(row["case_id"])
    token_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
    data = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    token_file = f"{case_id}.tokens.u32"
    token_path = token_dir / token_file
    token_path.write_bytes(data)
    token_fnv = f"{fnv64(data):016x}"
    token_sha256 = sha256_bytes(data)
    total_prompt_tokens += len(token_ids)
    cases[case_id] = {
        "kind": row.get("kind"),
        "last_token_id": token_ids[-1],
        "prompt_set": row.get("prompt_set"),
        "prompt_token_count": len(token_ids),
        "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
        "source_schema_version": row.get("schema_version"),
        "target_prompt_tokens": row.get("target_prompt_tokens"),
        "token_file": token_file,
        "token_file_fnv64": token_fnv,
        "token_file_sha256": token_sha256,
        "token_file_size_bytes": len(data),
      }
    lines.append(
        "\t".join([
            case_id,
            str(len(token_ids)),
            token_fnv,
            str(token_ids[0]),
            str(token_ids[-1]),
            token_file,
        ])
    )
  (token_dir / "cases.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return {
      "case_count": len(rows),
      "cases": cases,
      "cases_tsv_sha256": iq36_local.sha256_file(token_dir / "cases.tsv"),
      "total_prompt_tokens": total_prompt_tokens,
  }


def source_stage(host: str, remote_dir: str, timeout_s: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  mkdir = iq36_local.run_target(
      host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "tokens")
      ),
      timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(
          iq36_local.copy_to(host, ROOT / local, f"{remote_dir}/{remote}", timeout_s)
      )
  return mkdir, transfers


def token_stage(
    host: str,
    token_dir: Path,
    remote_token_dir: str,
    token_manifest: dict[str, Any],
    timeout_s: int,
) -> list[dict[str, Any]]:
  transfers = [
      iq36_local.copy_to(host, token_dir / "cases.tsv", f"{remote_token_dir}/cases.tsv", timeout_s)
  ]
  for item in token_manifest["cases"].values():
    token_file = item["token_file"]
    transfers.append(
        iq36_local.copy_to(host, token_dir / token_file, f"{remote_token_dir}/{token_file}", timeout_s)
    )
  return transfers


def build_command(remote_dir: str, env_script: str) -> str:
  return " && ".join([
      f"source {shlex.quote(env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/src/resident_harness.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/native_candidate_jsonl.cpp')} "
      "-pthread "
      f"-o {shlex.quote(remote_dir + '/build/iq36-native-candidate-jsonl')}",
  ])


def accepted_route_args(args: argparse.Namespace) -> list[str]:
  out = [
      "--warmup-runs",
      str(args.warmup_runs),
      "--timed-runs",
      str(args.timed_runs),
  ]
  if args.resident_cache:
    out.append("--resident-cache")
  if args.profile_matvec:
    out.append("--profile-matvec")
  out += [
      "--prefill-final-logits-only",
      "--full-attention-inplace-history",
      "--lm-head-top-k",
      "--lm-head-threads",
      "16",
      "--expert-slice-matvec",
      "--expert-slice-threads",
      "16",
      "--dense-matvec",
      "--dense-matvec-threads",
      "16",
      "--dense-matvec-min-rows",
      "256",
      "--dense-matvec-payload-cache",
      "--dense-q4-direct-dot",
      "--dense-q6-direct-dot",
      "--shared-parallel-executor",
      "--selected-expert-ffn",
      "--selected-expert-ffn-threads",
      "16",
      "--selected-expert-slice-cache",
      "--selected-expert-down-slice-cache",
      "--selected-gate-q4-direct-dot",
      "--selected-gate-q4-pair-dot",
  ]
  if args.dense_q6_pair_dot:
    out.append("--dense-q6-pair-dot")
  if args.selected_expert_down_q4_pair_dot:
    out.append("--selected-expert-down-q4-pair-dot")
  if args.selected_expert_down_q6_pair_dot:
    out.append("--selected-expert-down-q6-pair-dot")
  if getattr(args, "q4_plane_layout", False):
    out.append("--q4-plane-layout")
  if getattr(args, "dense_q4_plane_pair_dot", False):
    out.append("--dense-q4-plane-pair-dot")
  if getattr(args, "selected_gate_q4_plane_pair_dot", False):
    out.append("--selected-gate-q4-plane-pair-dot")
  return out


def route_name(args: argparse.Namespace) -> str:
  suffix = "_q4_plane_layout" if getattr(args, "q4_plane_layout", False) else ""
  if getattr(args, "dense_q4_plane_pair_dot", False):
    suffix += "_dense_q4_plane_pair_dot"
  if getattr(args, "selected_gate_q4_plane_pair_dot", False):
    suffix += "_selected_gate_q4_plane_pair_dot"
  down_bits = []
  if args.selected_expert_down_q4_pair_dot:
    down_bits.append("q4_pair")
  if args.selected_expert_down_q6_pair_dot:
    down_bits.append("q6_pair")
  down_suffix = "_selected_down_" + "_".join(down_bits) if down_bits else ""
  if args.dense_q6_pair_dot and down_bits:
    return (
        "post_r1_20260628T074112Z_dense_q6_pair_dot"
        f"{down_suffix}_dot_flags{suffix}"
    )
  if args.dense_q6_pair_dot:
    return f"post_r1_20260628T054920Z_dense_q6_pair_dot_flags{suffix}"
  return f"accepted_post_r1_20260628T040743Z_flags{suffix}"


def run_native_command(
    remote_dir: str,
    remote_token_dir: str,
    model: str,
    max_new_tokens: int,
    route_args: list[str],
    case_ids: list[str],
) -> str:
  argv = [
      remote_dir + "/build/iq36-native-candidate-jsonl",
      model,
      remote_token_dir,
      str(max_new_tokens),
  ] + route_args + case_ids
  return " ".join(shlex.quote(item) for item in argv)


def parse_stdout(stdout: str) -> dict[str, Any]:
  if not stdout:
    return {}
  return json.loads(stdout)


def stdout_run_record(run: dict[str, Any]) -> dict[str, Any]:
  stdout = run.get("stdout", "")
  stderr = run.get("stderr", "")
  return {
      "cmd": run.get("cmd"),
      "returncode": run.get("returncode"),
      "stderr_tail": stderr[-4000:] if isinstance(stderr, str) else "",
      "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest()
      if isinstance(stdout, str)
      else "",
      "stdout_size_bytes": len(stdout.encode("utf-8")) if isinstance(stdout, str) else 0,
  }


def timed_rows_from(parsed: dict[str, Any]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for timed_run in parsed.get("timed_runs", []):
    if not isinstance(timed_run, dict):
      continue
    run_index = timed_run.get("run_index")
    for case in timed_run.get("cases", []):
      if not isinstance(case, dict):
        continue
      timing = case.get("timing_ns", {})
      if not isinstance(timing, dict):
        timing = {}
      rows.append({
          "case_id": case.get("case_id"),
          "generated_token_count": len(case.get("generated_token_ids", [])),
          "generated_token_ids": case.get("generated_token_ids", []),
          "prompt_token_count": case.get("prompt_token_count"),
          "run_index": run_index,
          "timing_ns": timing,
          "top_logprob_id_signature": case.get("first_token_top_logprob_id_signature", []),
      })
  if not rows:
    for case in parsed.get("cases", []):
      if not isinstance(case, dict):
        continue
      timing = case.get("timing_ns", {})
      if not isinstance(timing, dict):
        timing = {}
      rows.append({
          "case_id": case.get("case_id"),
          "generated_token_count": len(case.get("generated_token_ids", [])),
          "generated_token_ids": case.get("generated_token_ids", []),
          "prompt_token_count": case.get("prompt_token_count"),
          "run_index": 0,
          "timing_ns": timing,
          "top_logprob_id_signature": case.get("first_token_top_logprob_id_signature", []),
      })
  return rows


def normalize_case_results(
    parsed_by_run: list[dict[str, Any]],
    token_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  case_meta = token_manifest["cases"]
  for parsed in parsed_by_run:
    for row in timed_rows_from(parsed):
      case_id = row.get("case_id")
      timing = row.get("timing_ns", {})
      prompt_tokens = row.get("prompt_token_count")
      prompt_prefill_ns = timing.get("prompt_prefill")
      per_token = None
      if isinstance(prompt_prefill_ns, int) and isinstance(prompt_tokens, int) and prompt_tokens > 0:
        per_token = prompt_prefill_ns / prompt_tokens
      meta = case_meta.get(case_id, {})
      out.append({
          "case_id": case_id,
          "decode_continuation_ns": timing.get("decode_continuation"),
          "first_generated_token_id": (
              row.get("generated_token_ids", [None])[0]
              if row.get("generated_token_ids")
              else None
          ),
          "generated_token_count": row.get("generated_token_count"),
          "kind": meta.get("kind"),
          "prompt_prefill_ns": prompt_prefill_ns,
          "prompt_prefill_ns_per_token": per_token,
          "prompt_token_count": prompt_tokens,
          "run_index": row.get("run_index"),
          "target_prompt_tokens": meta.get("target_prompt_tokens"),
          "timing_ns": timing,
          "top_logprob_id_signature": row.get("top_logprob_id_signature"),
      })
  return out


def smoothness_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  checks: list[dict[str, Any]] = []
  checks.append({
      "name": "all_timing_rows_have_positive_prefill_and_decode",
      "pass": all(
          isinstance(row.get("prompt_prefill_ns"), int)
          and row["prompt_prefill_ns"] > 0
          and isinstance(row.get("decode_continuation_ns"), int)
          and row["decode_continuation_ns"] >= 0
          for row in rows
      ),
  })
  for kind in sorted({row.get("kind") for row in rows if row.get("kind")}):
    group = [
        row for row in rows
        if row.get("kind") == kind and isinstance(row.get("prompt_token_count"), int)
    ]
    group.sort(key=lambda row: int(row["prompt_token_count"]))
    if len(group) < 2:
      checks.append({
          "name": f"{kind}_prefill_monotonic",
          "pass": True,
          "status": "insufficient_buckets",
          "bucket_count": len(group),
      })
      continue
    monotonic = all(
        int(group[i]["prompt_prefill_ns"]) <= int(group[i + 1]["prompt_prefill_ns"])
        for i in range(len(group) - 1)
    )
    checks.append({
        "name": f"{kind}_prefill_monotonic",
        "pass": monotonic,
        "bucket_count": len(group),
        "prompt_token_counts": [row["prompt_token_count"] for row in group],
        "prompt_prefill_ns": [row["prompt_prefill_ns"] for row in group],
    })
  return checks


def collect_host_metadata(host: str, env_script: str, timeout_s: int) -> dict[str, Any]:
  commands = {
      "uname_a": "uname -a",
      "kernel_release": "uname -r",
      "lscpu_summary": (
          "bash -lc "
          + shlex.quote("lscpu | egrep 'Architecture|CPU\\(s\\)|Thread|Core|Socket|Model name'")
      ),
      "gxx_version": (
          "bash -lc "
          + shlex.quote(
              f"source {shlex.quote(env_script)} >/dev/null 2>&1 && "
              "g++ --version | head -n 1"
          )
      ),
  }
  return {
      name: iq36_local.run_target(host, command, timeout_s)
      for name, command in commands.items()
  }


def build_summary(payload: dict[str, Any]) -> str:
  diag = payload["diagnostic"]
  rows = diag["case_results"]
  lines = [
      "# Native Context Ladder Diagnostic",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- case process isolation: `{str(diag['case_process_isolation']).lower()}`",
      f"- prefix cache enabled: `{str(diag['prefix_cache_enabled']).lower()}`",
      f"- max new tokens: {diag['max_new_tokens']}",
      f"- route: `{diag['route']}`",
      f"- q4-plane layout: `{str(diag['q4_plane_layout']).lower()}`",
      f"- dense Q4 plane-pair dot: `{str(diag['dense_q4_plane_pair_dot']).lower()}`",
      f"- dense Q6 pair dot: `{str(diag['dense_q6_pair_dot']).lower()}`",
      f"- selected gate Q4 plane-pair dot: `{str(diag['selected_gate_q4_plane_pair_dot']).lower()}`",
      f"- selected expert down Q4 pair dot: `{str(diag['selected_expert_down_q4_pair_dot']).lower()}`",
      f"- selected expert down Q6 pair dot: `{str(diag['selected_expert_down_q6_pair_dot']).lower()}`",
      f"- case count: {len(rows)}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | kind | prompt tokens | prefill ns | prefill ns/token | decode ns | first token |",
      "|---|---|---:|---:|---:|---:|---:|",
  ]
  for row in rows:
    per_token = row.get("prompt_prefill_ns_per_token")
    per_token_text = f"{per_token:.2f}" if isinstance(per_token, float) else ""
    lines.append(
        "| "
        + " | ".join([
            str(row.get("case_id")),
            str(row.get("kind")),
            str(row.get("prompt_token_count")),
            str(row.get("prompt_prefill_ns")),
            per_token_text,
            str(row.get("decode_continuation_ns")),
            str(row.get("first_generated_token_id")),
        ])
        + " |"
    )
  lines += [
      "",
      "This artifact is context-ladder timing evidence only. It does not close a",
      "promotion benchmark matrix and must not be used as a speedup claim.",
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
  if args.dense_q4_plane_pair_dot and not args.q4_plane_layout:
    raise SystemExit("--dense-q4-plane-pair-dot requires --q4-plane-layout")
  if not args.token_id_refs.exists():
    raise SystemExit(f"token-id refs missing: {args.token_id_refs}")

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/context-ladder-native-diagnostic-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  token_dir = out_dir / "token-input"
  stdout_dir = out_dir / "native-stdout"
  stdout_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/context-ladder-native-diagnostic-{stamp}"
  remote_token_dir = f"{remote_dir}/tokens"

  case_ids = case_ids_from_args(args)
  rows = selected_rows(args.token_id_refs, case_ids)
  token_manifest = prepare_token_inputs(rows, token_dir)
  mkdir, source_transfers = source_stage(args.host, remote_dir, args.timeout_s)
  token_transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0 and all(item.get("returncode") == 0 for item in source_transfers):
    token_transfers = token_stage(
        args.host,
        token_dir,
        remote_token_dir,
        token_manifest,
        args.timeout_s,
    )
  staged = (
      mkdir.get("returncode") == 0
      and all(item.get("returncode") == 0 for item in source_transfers)
      and bool(token_transfers)
      and all(item.get("returncode") == 0 for item in token_transfers)
  )

  build = (
      iq36_local.run_target(
          args.host,
          f"bash -lc {shlex.quote(build_command(remote_dir, args.env_script))}",
          args.timeout_s,
      )
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )

  route_args = accepted_route_args(args)
  run_case_groups = [case_ids] if args.combined_process else [[case_id] for case_id in case_ids]
  target_runs: list[dict[str, Any]] = []
  parsed_runs: list[dict[str, Any]] = []
  parse_errors: list[dict[str, Any]] = []
  if build.get("returncode") == 0:
    for group in run_case_groups:
      run_command = run_native_command(
          remote_dir,
          remote_token_dir,
          args.model,
          args.max_new_tokens,
          route_args,
          group,
      )
      run_result = iq36_local.run_target(args.host, run_command, args.timeout_s)
      target_runs.append(stdout_run_record(run_result))
      parsed: dict[str, Any] = {}
      try:
        parsed = parse_stdout(run_result.get("stdout", ""))
      except json.JSONDecodeError as exc:
        parse_errors.append({"case_ids": group, "error": str(exc)})
      if parsed:
        parsed_runs.append(parsed)
        write_json(stdout_dir / ("+".join(group) + ".json"), parsed)

  case_results = normalize_case_results(parsed_runs, token_manifest)
  smooth_checks = smoothness_checks(case_results)
  host_metadata = collect_host_metadata(args.host, args.env_script, min(args.timeout_s, 120))

  checks = [
      {"name": "token_id_refs_present", "pass": args.token_id_refs.exists()},
      {"name": "selected_cases_present", "pass": len(rows) == len(case_ids)},
      {"name": "token_inputs_prepared", "pass": token_manifest["case_count"] == len(case_ids)},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(item.get("returncode") == 0 for item in source_transfers),
      },
      {
          "name": "token_inputs_transferred",
          "pass": bool(token_transfers) and all(item.get("returncode") == 0 for item in token_transfers),
      },
      {"name": "target_native_runner_built", "pass": build.get("returncode") == 0},
      {
          "name": "target_native_runner_ran_all_groups",
          "pass": len(target_runs) == len(run_case_groups)
          and all(item.get("returncode") == 0 for item in target_runs),
      },
      {"name": "target_stdout_parsed_all_groups", "pass": len(parsed_runs) == len(run_case_groups)},
      {"name": "case_timing_rows_present", "pass": len(case_results) == len(case_ids)},
      {
          "name": "prompt_counts_match_token_refs",
          "pass": all(
              token_manifest["cases"][row["case_id"]]["prompt_token_count"]
              == row.get("prompt_token_count")
              for row in case_results
              if row.get("case_id") in token_manifest["cases"]
          ) and len(case_results) == len(case_ids),
      },
      {
          "name": "accepted_route_flags_enabled",
          "pass": all(
              parsed.get("prefill_final_logits_only_enabled") is True
              and parsed.get("full_attention_inplace_history_enabled") is True
              and parsed.get("lm_head_top_k_enabled") is True
              and parsed.get("dense_matvec_enabled") is True
              and parsed.get("dense_q4_direct_dot_enabled") is True
              and parsed.get("dense_q6_direct_dot_enabled") is True
              and parsed.get("dense_q6_pair_dot_enabled") is args.dense_q6_pair_dot
              and parsed.get("selected_expert_down_q4_pair_dot_enabled")
              is args.selected_expert_down_q4_pair_dot
              and parsed.get("selected_expert_down_q6_pair_dot_enabled")
              is args.selected_expert_down_q6_pair_dot
              and parsed.get("q4_plane_layout_enabled")
              is getattr(args, "q4_plane_layout", False)
              and parsed.get("dense_q4_plane_pair_dot_enabled")
              is getattr(args, "dense_q4_plane_pair_dot", False)
              and parsed.get("selected_gate_q4_plane_pair_dot_enabled")
              is getattr(args, "selected_gate_q4_plane_pair_dot", False)
              and parsed.get("selected_expert_ffn_enabled") is True
              and parsed.get("selected_gate_q4_pair_dot_enabled") is True
              for parsed in parsed_runs
          ) and bool(parsed_runs),
      },
      {
          "name": "dense_q6_pair_dot_recorded",
          "pass": all(
              parsed.get("dense_q6_pair_dot_enabled") is args.dense_q6_pair_dot
              for parsed in parsed_runs
          ) and bool(parsed_runs),
      },
      {
          "name": "selected_gate_q4_plane_pair_dot_recorded",
          "pass": all(
              parsed.get("selected_gate_q4_plane_pair_dot_enabled")
              is getattr(args, "selected_gate_q4_plane_pair_dot", False)
              for parsed in parsed_runs
          ) and bool(parsed_runs),
      },
      {
          "name": "selected_expert_down_q4_pair_dot_recorded",
          "pass": all(
              parsed.get("selected_expert_down_q4_pair_dot_enabled")
              is args.selected_expert_down_q4_pair_dot
              for parsed in parsed_runs
          ) and bool(parsed_runs),
      },
      {
          "name": "selected_expert_down_q6_pair_dot_recorded",
          "pass": all(
              parsed.get("selected_expert_down_q6_pair_dot_enabled")
              is args.selected_expert_down_q6_pair_dot
              for parsed in parsed_runs
          ) and bool(parsed_runs),
      },
  ] + smooth_checks
  required_checks_passed = all(check["pass"] for check in checks)

  diagnostic = {
      "case_ids": case_ids,
      "case_process_isolation": not args.combined_process,
      "case_results": case_results,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "host_metadata": host_metadata,
      "max_new_tokens": args.max_new_tokens,
      "prefix_cache_enabled": False,
      "profile_matvec": args.profile_matvec,
      "q4_plane_layout": getattr(args, "q4_plane_layout", False),
      "dense_q4_plane_pair_dot": getattr(args, "dense_q4_plane_pair_dot", False),
      "resident_cache": args.resident_cache,
      "route": route_name(args),
      "route_args": route_args,
      "selected_gate_q4_plane_pair_dot": getattr(
          args, "selected_gate_q4_plane_pair_dot", False),
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "target_runs": target_runs,
      "warmup_runs": args.warmup_runs,
      "timed_runs": args.timed_runs,
  }
  payload = {
      "created_at": created_at,
      "diagnostic": diagnostic,
      "host": args.host,
      "model_path": args.model,
      "parse_errors": parse_errors,
      "remote_dir": remote_dir,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "token_id_refs": str(args.token_id_refs.resolve().relative_to(ROOT)),
      "token_input_manifest": str((out_dir / "token-input-manifest.json").relative_to(ROOT)),
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_ids": case_ids,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_plane_layout": getattr(args, "q4_plane_layout", False),
      "dense_q4_plane_pair_dot": getattr(args, "dense_q4_plane_pair_dot", False),
      "selected_gate_q4_plane_pair_dot": getattr(
          args, "selected_gate_q4_plane_pair_dot", False),
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "host": args.host,
      "model_path": args.model,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-context-ladder-native-diagnostic.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "token-input-manifest.json", token_manifest)
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
      "token_transfers": token_transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "diagnostic.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "dense_q6_pair_dot": args.dense_q6_pair_dot,
      "q4_plane_layout": getattr(args, "q4_plane_layout", False),
      "dense_q4_plane_pair_dot": getattr(args, "dense_q4_plane_pair_dot", False),
      "selected_gate_q4_plane_pair_dot": getattr(
          args, "selected_gate_q4_plane_pair_dot", False),
      "selected_expert_down_q4_pair_dot": args.selected_expert_down_q4_pair_dot,
      "selected_expert_down_q6_pair_dot": args.selected_expert_down_q6_pair_dot,
      "gate": "context_ladder_native_diagnostic",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "case-results.jsonl", case_results)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "context_ladder_native_diagnostic",
      [
          ("case_count", len(case_results)),
          ("case_process_isolation", not args.combined_process),
          ("dense_q6_pair_dot", args.dense_q6_pair_dot),
          ("q4_plane_layout", getattr(args, "q4_plane_layout", False)),
          ("dense_q4_plane_pair_dot", getattr(args, "dense_q4_plane_pair_dot", False)),
          ("selected_gate_q4_plane_pair_dot", getattr(
              args, "selected_gate_q4_plane_pair_dot", False)),
          ("selected_expert_down_q4_pair_dot", args.selected_expert_down_q4_pair_dot),
          ("selected_expert_down_q6_pair_dot", args.selected_expert_down_q6_pair_dot),
          ("max_new_tokens", args.max_new_tokens),
          ("prefix_cache_enabled", False),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")

  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
