#!/usr/bin/env python3
"""Gate prompt-conditioned state built by the locked native GGUF token loop."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import struct
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-native-sequential-semantic-gate-v0"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
ORACLE_DIR = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
ORACLE_DISTRIBUTIONS = ORACLE_DIR / "teacher-forced-distribution-references.jsonl"
SENTINELS = (
    ROOT
    / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts"
    / "long-context-sentinels.jsonl"
)
SHORT_CASE = "fresh_code_03"
SHORT_TOKENS = (
    ROOT
    / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z"
    / "token-input/fresh_code_03.tokens.u32"
)
SHORT_REFERENCE_RESULT = (
    ROOT
    / "output/packed-token-level-zero-backend-20260713Tseq793-int8-hot8192-tile4-hostucb-cleanZ"
    / "result.json"
)
CORE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
LONG_CASES = tuple(
    f"{prompt_set}_{bucket // 1024:03d}k"
    for bucket in CORE_BUCKETS
    for prompt_set in ("sentinel", "prefill_shape")
)
ALLOWED_CASES = (SHORT_CASE, *LONG_CASES)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument(
      "--case", action="append", dest="cases", choices=ALLOWED_CASES,
      help="Case to run; repeat for multiple cases (default: fresh_code_03).")
  parser.add_argument(
      "--max-reference-tokens", type=int, default=513,
      help="Maximum oracle predictions to validate per long case.")
  parser.add_argument("--timeout-s", type=int, default=10800)
  parser.add_argument("--allow-dirty-development", action="store_true")
  return parser.parse_args()


def run(
    command: list[str], timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  try:
    return subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, env=environment)
  except subprocess.TimeoutExpired as error:
    stdout = error.stdout if isinstance(error.stdout, str) else ""
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    return subprocess.CompletedProcess(
        command, 124, stdout, stderr + f"\ntimeout after {timeout}s\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def read_u32(path: Path) -> list[int]:
  payload = path.read_bytes()
  if not payload or len(payload) % 4:
    raise RuntimeError(f"invalid u32 token file: {path}")
  return list(struct.unpack(f"<{len(payload) // 4}I", payload))


def write_u32(path: Path, values: list[int]) -> None:
  if not values:
    raise RuntimeError(f"refusing to write empty token file: {path}")
  path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def short_case() -> dict[str, Any]:
  source = json.loads(SHORT_REFERENCE_RESULT.read_text())
  row = next(
      item for item in source["rows"] if item.get("case_id") == SHORT_CASE)
  references = [int(row["cpu_topk"][0]["id"]), *map(int, row["cpu_generated_ids"])]
  source_ok = bool(
      source.get("required_checks_passed") is True
      and source.get("distribution_diagnostic_passed") is True
      and row.get("exact_generated_ids") is True)
  return {
      "case_id": SHORT_CASE,
      "kind": "short_mechanism",
      "prompt_tokens": read_u32(SHORT_TOKENS),
      "reference_tokens": references,
      "reference_source": str(SHORT_REFERENCE_RESULT.relative_to(ROOT)),
      "reference_source_checks_passed": source_ok,
      "source_reference_runtime": "exact locked GGUF CPU implementation",
  }


def answer_token_index(response: dict[str, Any], expected: str) -> int | None:
  text = ""
  for index, position in enumerate(response.get("completion_probabilities", []), 1):
    text += str(position.get("token", ""))
    if expected in text:
      return index
  return None


def long_case(
    case_id: str, oracle: dict[str, dict[str, Any]],
    sentinels: dict[str, dict[str, Any]], max_reference_tokens: int,
) -> dict[str, Any]:
  row = oracle[case_id]
  generated = [int(value) for value in row["generated_token_ids"]]
  if len(generated) < 2:
    raise RuntimeError(f"{case_id} has fewer than two oracle tokens")
  references = generated[:max_reference_tokens]
  result: dict[str, Any] = {
      "case_id": case_id,
      "kind": str(row.get("kind") or row.get("prompt_set")),
      "prompt_tokens": [int(value) for value in row["prompt_token_ids"]],
      "reference_tokens": references,
      "oracle_generated_token_count": len(generated),
      "oracle_prompt_token_count": int(row["prompt_token_count"]),
      "oracle_prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
      "reference_source": str(ORACLE_DISTRIBUTIONS.relative_to(ROOT)),
      "reference_source_checks_passed": bool(
          row.get("bundle_row_status")
          == "accepted_teacher_forced_distribution_reference"),
      "source_reference_runtime": row.get("source_reference_runtime"),
      "stopped_before_request_limit": row.get("stopped_before_request_limit"),
  }
  if case_id.startswith("sentinel_"):
    expected = str(sentinels[case_id]["expected_answer"])
    source_artifact = ROOT / str(row["source_artifact"])
    matches = sorted(source_artifact.parent.glob(
        f"case-*-{case_id}/raw/remote/completion_response.json"))
    if len(matches) != 1:
      raise RuntimeError(
          f"expected one raw reference response for {case_id}, found {len(matches)}")
    response_path = matches[0]
    response = json.loads(response_path.read_text())
    response_ids = [
        int(position["id"])
        for position in response.get("completion_probabilities", [])]
    answer_index = answer_token_index(response, expected)
    result["sentinel"] = {
        "answer_completion_token_index": answer_index,
        "answer_within_matched_reference_prefix": bool(
            answer_index is not None and answer_index <= len(references)),
        "expected_answer": expected,
        "reference_content_contains_answer": expected in str(response.get("content", "")),
        "reference_generated_ids_match_bundle": response_ids == generated,
        "reference_response": str(response_path.relative_to(ROOT)),
        "reference_response_sha256": sha256(response_path),
    }
  return result


def materialize_case(case: dict[str, Any], token_dir: Path) -> tuple[Path, Path]:
  prompt_path = token_dir / f"{case['case_id']}.prompt.tokens.u32"
  reference_path = token_dir / f"{case['case_id']}.reference.tokens.u32"
  write_u32(prompt_path, case["prompt_tokens"])
  write_u32(reference_path, case["reference_tokens"])
  return prompt_path, reference_path


def case_checks(case: dict[str, Any], native: dict[str, Any], returncode: int) -> list[dict[str, Any]]:
  expected_prompt_count = len(case["prompt_tokens"])
  expected_reference_count = len(case["reference_tokens"])
  checks = [
      {"name": "reference_source_accepted", "pass": (
          case["reference_source_checks_passed"] is True)},
      {"name": "native_process_passed", "pass": (
          returncode == 0 and native.get("required_checks_passed") is True)},
      {"name": "exact_prompt_count", "pass": (
          native.get("prompt_tokens") == expected_prompt_count)},
      {"name": "exact_reference_prediction_count", "pass": (
          native.get("reference_prediction_count") == expected_reference_count)},
      {"name": "teacher_forced_top1_ids_exact", "pass": (
          native.get("exact_reference_ids") is True
          and native.get("first_divergence_index") is None
          and native.get("reference_ids") == case["reference_tokens"])},
      {"name": "greedy_equivalence_by_induction", "pass": (
          native.get("deterministic_greedy_exact_match_proved_by_induction") is True)},
      {"name": "locked_native_state_representation", "pass": (
          native.get("state_semantics") == "native_sequential_locked_gguf"
          and native.get("full_kv_dtype")
          == "int8_block32_fp16_scale_f32_hot8192")},
      {"name": "product_claims_forbidden", "pass": (
          native.get("speedup_claims_allowed") is False
          and native.get("prefill_product_claim_allowed") is False)},
  ]
  if case["case_id"] == SHORT_CASE:
    ladder = native.get("cpu_distribution_ladder") or {}
    checks.append({"name": "short_full_vocab_cpu_distribution", "pass": (
        native.get("cpu_distribution_check") is True
        and ladder.get("required_checks_passed") is True
        and int(ladder.get("position_count", 0)) == expected_reference_count)})
  sentinel = case.get("sentinel")
  if sentinel is not None:
    checks.append({"name": "sentinel_answer_in_exact_matched_prefix", "pass": (
        sentinel["reference_content_contains_answer"] is True
        and sentinel["reference_generated_ids_match_bundle"] is True
        and sentinel["answer_within_matched_reference_prefix"] is True
        and native.get("exact_reference_ids") is True)})
  return checks


def disposition(cases: list[str], passed: bool) -> str:
  if not passed:
    return "reject_native_sequential_semantic_state"
  case_set = set(cases)
  if case_set == {SHORT_CASE}:
    return "admit_bounded_2k_native_sequential_semantic_audit"
  if case_set == {"sentinel_002k", "prefill_shape_002k"}:
    return "admit_priority_long_context_native_sequential_semantic_audit"
  if set(LONG_CASES).issubset(case_set):
    return "record_complete_prompt_conditioned_semantic_ladder_product_gate_unchanged"
  return "record_bounded_prompt_conditioned_semantic_pass_product_gate_unchanged"


def summary(payload: dict[str, Any]) -> str:
  lines = [
      "# Native sequential GGUF semantic-state gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      "- product prefill / speedup claim: `forbidden`",
      "",
      "| case | prompt | exact predictions | sequential state build tok/s | decode tok/s | sentinel |",
      "|---|---:|---:|---:|---:|:---:|",
  ]
  for row in payload["rows"]:
    native = row["native"]
    sentinel = row.get("sentinel")
    lines.append(
        f"| {row['case_id']} | {len(row['prompt_tokens'])} | "
        f"{native.get('matching_reference_ids', 0)}/{len(row['reference_tokens'])} | "
        f"{float(native.get('sequential_state_build_wall_tokens_s', 0.0)):.3f} | "
        f"{float(native.get('decode_wall_tokens_s', 0.0)):.3f} | "
        f"{'pass' if sentinel and sentinel.get('answer_within_matched_reference_prefix') else ('n/a' if not sentinel else 'fail')} |")
  lines += [
      "",
      "Prompt state is built by replaying locked GGUF tokens through the unchanged",
      "native decode program. That serial construction is correctness evidence only;",
      "it is not product prefill and cannot support a speedup claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.max_reference_tokens < 2:
    raise SystemExit("--max-reference-tokens must be at least 2")
  cases = args.cases or [SHORT_CASE]
  if len(cases) != len(set(cases)):
    raise SystemExit("duplicate --case values are not allowed")
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  generated_dir = out_dir / "generated"
  token_dir = generated_dir / "token-input"
  raw_dir.mkdir(parents=True, exist_ok=False)
  token_dir.mkdir(parents=True, exist_ok=True)
  git = git_state(out_dir)
  created_at = dt.datetime.now(dt.timezone.utc).isoformat()

  compile_command = [
      "ocloc", "compile", "-file",
      str(ROOT / "engine/gpu/opencl/q4x8_matvec.cl"),
      "-device", "0xb080", "-output", "iq36_q4x8_all",
      "-out_dir", str(generated_dir), "-output_no_suffix",
      "--format", "zebin", "-options",
      "-cl-std=CL3.0 -D IQ36_USE_INTEGER_DOT=1", "-q",
  ]
  compile_run = run(compile_command, 300)
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure_run = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target",
      "iq36-packed-token-level-zero-backend-smoke", "-j8",
  ]
  build_run = run(build_command, 600)
  executable = BUILD_DIR / "iq36-packed-token-level-zero-backend-smoke"
  module = generated_dir / "iq36_q4x8_all.bin"
  build_ok = all((
      compile_run.returncode == 0, configure_run.returncode == 0,
      build_run.returncode == 0, executable.is_file(), module.is_file()))
  write_json(raw_dir / "build.json", {
      "compile": {"command": compile_command, "returncode": compile_run.returncode,
                  "stdout": compile_run.stdout, "stderr": compile_run.stderr},
      "configure": {"command": configure_command,
                    "returncode": configure_run.returncode,
                    "stdout": configure_run.stdout, "stderr": configure_run.stderr},
      "build": {"command": build_command, "returncode": build_run.returncode,
                "stdout": build_run.stdout, "stderr": build_run.stderr},
  })

  oracle = {row["case_id"]: row for row in load_jsonl(ORACLE_DISTRIBUTIONS)}
  sentinels = {row["id"]: row for row in load_jsonl(SENTINELS)}
  case_inputs = [
      short_case() if case_id == SHORT_CASE else long_case(
          case_id, oracle, sentinels, args.max_reference_tokens)
      for case_id in cases
  ]
  rows: list[dict[str, Any]] = []
  environment = os.environ.copy()
  environment["IQ36_INT8_BLOCK32_KV_GQA"] = "1"
  if build_ok:
    for case in case_inputs:
      case_id = case["case_id"]
      prompt_path, reference_path = materialize_case(case, token_dir)
      case_environment = environment.copy()
      if case_id == SHORT_CASE:
        case_environment["IQ36_SEQUENTIAL_CPU_DISTRIBUTION_CHECK"] = "1"
      command = [
          str(executable), str(MODEL), str(module), str(prompt_path),
          "--sequential-prompt", str(reference_path),
      ]
      print(
          f"running {case_id}: prompt={len(case['prompt_tokens'])} "
          f"reference={len(case['reference_tokens'])}", flush=True)
      completed = run(command, args.timeout_s, case_environment)
      (raw_dir / f"{case_id}.stdout").write_text(completed.stdout)
      (raw_dir / f"{case_id}.stderr").write_text(completed.stderr)
      write_json(raw_dir / f"{case_id}.command.json", {
          "command": command,
          "environment": {
              "IQ36_INT8_BLOCK32_KV_GQA": "1",
              "IQ36_SEQUENTIAL_CPU_DISTRIBUTION_CHECK": (
                  "1" if case_id == SHORT_CASE else "unset"),
          },
          "returncode": completed.returncode,
      })
      native = parse_last_json(completed.stdout)
      checks = case_checks(case, native, completed.returncode)
      rows.append({
          **case,
          "checks": checks,
          "native": native,
          "required_checks_passed": all(bool(check["pass"]) for check in checks),
          "returncode": completed.returncode,
      })

  repository_check = bool(not git["dirty"] or args.allow_dirty_development)
  checks = [
      {"name": "repository_clean_at_gate", "pass": repository_check,
       "allow_dirty_development": args.allow_dirty_development,
       "dirty_paths": git["dirty_paths"]},
      {"name": "target_module_and_smoke_build", "pass": build_ok},
      {"name": "all_requested_cases_executed", "pass": (
          len(rows) == len(case_inputs))},
      {"name": "all_prompt_conditioned_semantic_rows_pass", "pass": (
          bool(rows) and all(row["required_checks_passed"] for row in rows))},
      {"name": "product_speedup_not_claimed", "pass": (
          bool(rows) and all(
              row["native"].get("speedup_claims_allowed") is False
              and row["native"].get("prefill_product_claim_allowed") is False
              for row in rows))},
  ]
  required = all(bool(check["pass"]) for check in checks)
  payload = {
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition(cases, required),
      "git": git,
      "product_promotion_ready": False,
      "required_checks_passed": required,
      "rows": rows,
      "schema_version": SCHEMA,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": created_at,
      "git": git,
      "required_checks_passed": required,
      "route_label": "diagnostic_pass" if required else "rejected",
      "schema_version": SCHEMA,
      "source_sha256": {
          "backend_smoke": sha256(
              ROOT / "engine/tools/packed_token_level_zero_backend_smoke.cpp"),
          "gate": sha256(Path(__file__)),
      },
      "speedup_claims_allowed": False,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "correctness_applicable": True,
      "prompt_conditioned": True,
      "required_checks_passed": required,
      "rows": [
          {"case_id": row["case_id"], "checks": row["checks"],
           "required_checks_passed": row["required_checks_passed"]}
          for row in rows
      ],
      "speedup_claims_allowed": False,
      "state_semantics": "native_sequential_locked_gguf",
  })
  with (out_dir / "metrics.jsonl").open("w") as handle:
    for row in rows:
      native = row["native"]
      handle.write(json.dumps({
          "case_id": row["case_id"],
          "decode_wall_ms_median": native.get("decode_wall_ms_median"),
          "decode_wall_tokens_s": native.get("decode_wall_tokens_s"),
          "exact_reference_ids": native.get("exact_reference_ids"),
          "prompt_tokens": native.get("prompt_tokens"),
          "reference_prediction_count": native.get("reference_prediction_count"),
          "required_checks_passed": row["required_checks_passed"],
          "sentinel_retrieval_passed": (
              row.get("sentinel", {}).get("answer_within_matched_reference_prefix")),
          "sequential_state_build_wall_tokens_s": (
              native.get("sequential_state_build_wall_tokens_s")),
          "speedup_claims_allowed": False,
      }, sort_keys=True) + "\n")
  decode_tpots = [
      float(row["native"]["decode_wall_ms_median"])
      for row in rows if row["native"].get("decode_wall_ms_median") is not None]
  write_json(out_dir / "smoothness.json", {
      "decode_tpot_median_cv": (
          statistics.pstdev(decode_tpots) / statistics.mean(decode_tpots)
          if len(decode_tpots) >= 2 else None),
      "measured_case_count": len(rows),
      "product_smoothness_applicable": False,
      "reason": "serial token replay is semantic state construction, not product prefill",
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  })
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "disposition": payload["disposition"],
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
