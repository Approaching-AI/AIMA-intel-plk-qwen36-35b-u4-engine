#!/usr/bin/env python3
"""Score the accepted short GPU decode carrier on consensus token cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-native-consensus-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
REFERENCE_ARTIFACT = (
    ROOT / "output/reference-consensus-matrix-"
    "20260712Tseq729-fresh9-cleanZ")
TOKEN_INPUT_DIR = (
    ROOT / "output/seq571-state-conditioned-head-correction-token-input-"
    "20260710Tseq571Z/token-input")
DECODE_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
ACCEPTED_CUTS = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/accepted-cuts.json")
CASES = (
    "fresh_arithmetic_01",   # fit
    "fresh_code_03",         # validation
    "fresh_instruction_04",  # test
)
ROWBLOCK16_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    24, 25, 26, 28, 29, 30, 33, 34, 36, 37, 38,
)
ROUTE_FLAGS = (
    "--resident-selected-q4-experts",
    "--resident-selected-q6-experts",
    "--resident-selected-q6-sorted-cache",
    "--resident-selected-q6-rowstripe",
    "--resident-shared-q6-down",
    "--resident-full-attention-v-q6",
    "--resident-linear-q6-qkv",
    "--resident-q4-cpu-order-z",
    "--resident-linear-conv-weights",
    "--resident-linear-state",
    "--resident-postconv-delta-handoff",
    "--resident-norm-weights",
    "--resident-gate-up-swiglu-handoff",
    "--resident-attention-front-handoff",
    "--resident-full-core-attention-front-handoff",
    "--gpu-router",
    "--gpu-lm-head-q6",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--reference-artifact", type=Path,
                      default=REFERENCE_ARTIFACT)
  parser.add_argument("--token-input-dir", type=Path,
                      default=TOKEN_INPUT_DIR)
  parser.add_argument("--timeout-s", type=int, default=7200)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/native-consensus-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.write_text(
      "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
      encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def route_environment() -> dict[str, str]:
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS":
          ",".join(str(layer) for layer in ROWBLOCK16_LAYERS),
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
  })
  return env


def decode_command(args: argparse.Namespace, case_id: str,
                   out: Path) -> list[str]:
  return [
      sys.executable, str(DECODE_SMOKE),
      "--target", "local",
      "--model", str(args.model.resolve()),
      "--token-input-dir", str(args.token_input_dir.resolve()),
      "--case-id", case_id,
      "--decode-tokens", "8",
      "--resident-selected-cache-topk", "16",
      *ROUTE_FLAGS,
      "--out-dir", str(out),
      "--timeout-s", str(args.timeout_s),
  ]


def run_case(args: argparse.Namespace, case_id: str,
             out: Path, raw: Path) -> dict[str, Any]:
  command = decode_command(args, case_id, out)
  try:
    process = subprocess.run(
        command, cwd=ROOT, env=route_environment(), check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=args.timeout_s + 300)
    timed_out = False
  except subprocess.TimeoutExpired as exc:
    process = subprocess.CompletedProcess(
        command, 124,
        exc.stdout if isinstance(exc.stdout, str) else "",
        exc.stderr if isinstance(exc.stderr, str) else "")
    timed_out = True
  (raw / f"{case_id}.stdout").write_text(
      process.stdout, encoding="utf-8")
  (raw / f"{case_id}.stderr").write_text(
      process.stderr, encoding="utf-8")
  write_json(raw / f"{case_id}.command.json", {
      "command": command,
      "returncode": process.returncode,
      "timed_out": timed_out,
  })
  result_path = out / "result.json"
  result = load_json(result_path) if result_path.is_file() else {}
  smoke = result.get("smoke", {}) if isinstance(result, dict) else {}
  if not isinstance(smoke, dict):
    smoke = {}
  return {
      "command": command,
      "returncode": process.returncode,
      "timed_out": timed_out,
      "result": result,
      "smoke": smoke,
  }


def finite_positive(value: Any) -> bool:
  return (
      isinstance(value, (int, float)) and
      math.isfinite(float(value)) and float(value) > 0.0)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  runs = out / "runs"
  raw.mkdir(parents=True, exist_ok=False)
  runs.mkdir()
  required = [
      args.model, args.reference_artifact / "result.json",
      args.reference_artifact / "correctness.json", args.token_input_dir,
      DECODE_SMOKE, ACCEPTED_CUTS,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  reference = load_json(args.reference_artifact / "result.json")
  reference_rows = {
      str(row.get("case_id")): row for row in reference.get("rows", [])
      if isinstance(row, dict)
  }
  accepted = load_json(ACCEPTED_CUTS).get("accepted", [])
  accepted_ids = {
      str(row.get("id")) for row in accepted if isinstance(row, dict)
  }

  rows = []
  for case_id in CASES:
    reference_row = reference_rows.get(case_id, {})
    token_file = args.token_input_dir / f"{case_id}.tokens.u32"
    case_out = runs / case_id
    execution = run_case(args, case_id, case_out, raw)
    smoke = execution["smoke"]
    reference_ids = [
        int(value) for value in reference_row.get("llama_cpp_token_ids", [])]
    native_ids = [
        int(value) for value in smoke.get("native_reference_token_ids", [])]
    gpu_ids = [
        int(value) for value in smoke.get("gpu_generated_token_ids", [])]
    gpu_inputs = [
        int(value) for value in smoke.get("gpu_input_token_ids", [])]
    token_sha = sha256_file(token_file) if token_file.is_file() else None
    reference_bound = (
        reference_row.get("reference_consensus") is True and
        len(reference_ids) == 9 and
        token_sha == reference_row.get("token_sha256"))
    candidate_exact = (
        reference_bound and
        smoke.get("input_generated_token_id") == reference_ids[0] and
        gpu_inputs == reference_ids[:-1] and
        native_ids == reference_ids[1:] and
        gpu_ids == reference_ids[1:] and
        smoke.get("greedy_prefix_match_count") == 8 and
        smoke.get("top1_match_count") == 8 and
        smoke.get("top1_matches_native") is True)
    route_active = (
        smoke.get(
            "attention_front_output_projection_rowblock16_enabled") is True and
        smoke.get(
            "attention_front_output_projection_rowblock16_layer_ids") ==
            list(ROWBLOCK16_LAYERS) and
        smoke.get("selected_shared_q4_gateup_combined_enabled") is True and
        int(smoke.get("selected_shared_q4_gateup_combined_layers", 0)) > 0 and
        smoke.get("selected_shared_q4_down_combined_enabled") is True and
        int(smoke.get("selected_shared_q4_down_combined_layers", 0)) > 0 and
        smoke.get("selected_shared_q6_down_combined_enabled") is True and
        int(smoke.get("selected_shared_q6_down_combined_layers", 0)) > 0 and
        smoke.get("defer_ffn_down_finish_bundle") is True)
    complete = (
        execution["timed_out"] is False and bool(smoke) and
        smoke.get("case_id") == case_id and
        smoke.get("decode_continuation_output_tokens") == 8 and
        finite_positive(smoke.get("gpu_hybrid_decode_tok_s")) and
        route_active)
    rows.append({
        "case_id": case_id,
        "split": reference_row.get("split"),
        "domain": reference_row.get("domain"),
        "artifact": str(case_out.relative_to(ROOT)),
        "tool_returncode": execution["returncode"],
        "complete": complete,
        "reference_bound": reference_bound,
        "reference_token_ids": reference_ids,
        "native_reference_token_ids": native_ids,
        "gpu_generated_token_ids": gpu_ids,
        "candidate_exact_reference_match": candidate_exact,
        "route_active": route_active,
        "decode_tokens_s": smoke.get("gpu_hybrid_decode_tok_s"),
        "source_sha": execution["result"].get("source_sha"),
        "diagnostic_topk_ids_match_native": smoke.get(
            "topk_ids_match_native"),
        "underlying_required_checks_passed": smoke.get(
            "required_checks_passed"),
    })

  decode_values = [
      float(row["decode_tokens_s"]) for row in rows
      if finite_positive(row.get("decode_tokens_s"))]
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("clean_nine_token_reference_consensus_prerequisite",
            reference.get("required_checks_passed") is True and
            reference.get("git", {}).get("dirty") is False and
            reference.get("config", {}).get("generated_tokens") == 9 and
            all(reference_rows.get(case_id, {}).get(
                "reference_consensus") is True for case_id in CASES),
            reference_artifact=str(args.reference_artifact)),
      check("accepted_rowblock26_route_selected",
            "attention_front_output_projection_rowblock16_26mask" in
                accepted_ids),
      check("fit_validation_test_cases_preregistered",
            [reference_rows.get(case_id, {}).get("split")
             for case_id in CASES] == ["fit", "validation", "test"],
            case_ids=list(CASES)),
      check("all_candidate_rows_complete",
            len(rows) == 3 and all(row["complete"] for row in rows)),
      check("all_candidate_rows_exact_to_both_references",
            len(rows) == 3 and all(
                row["candidate_exact_reference_match"] for row in rows)),
      check("all_candidate_rows_use_fixed_route",
            all(row["route_active"] for row in rows)),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "reference_artifact": str(args.reference_artifact),
      "selected_cases": list(CASES),
      "route": {
          "id": "attention_front_output_projection_rowblock16_26mask",
          "rowblock16_layers": list(ROWBLOCK16_LAYERS),
          "decode_tokens": 8,
          "cpu_prefill_seed_token": True,
      },
      "rows": rows,
      "decode_tokens_s_median": (
          statistics.median(decode_values) if len(decode_values) == 3 else
          None),
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "accept_consensus_exact_decode_carrier_select_product_integration"
          if passed else "reject_consensus_exact_decode_carrier"),
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_jsonl(out / "case-results.jsonl", rows)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out),
      "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  summary = [
      "# Native consensus-token gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- median diagnostic decode: `{result['decode_tokens_s_median']}` tok/s",
      "- product promotion ready: `false`", "",
      "| case | split | exact to both references | decode tok/s |",
      "|---|---|---:|---:|",
  ]
  for row in rows:
    summary.append(
        f"| {row['case_id']} | {row['split']} | "
        f"{str(row['candidate_exact_reference_match']).lower()} | "
        f"{row['decode_tokens_s']} |")
  summary.extend([
      "", "The CPU/llama prefill supplies the seed token. This artifact "
      "accepts the following eight-token native decode carrier only; it is "
      "not a native-prefill or product-speed claim.", "",
  ])
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "pass": passed,
      "median_decode_tokens_s": result["decode_tokens_s_median"],
      "case_matches": {
          row["case_id"]: row["candidate_exact_reference_match"]
          for row in rows
      },
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
