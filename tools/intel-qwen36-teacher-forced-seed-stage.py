#!/usr/bin/env python3
"""Stage per-position top-k distributions from the CPU llama.cpp oracle raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-teacher-forced-distribution-seed-v0"
SEED_SCHEMA_VERSION = "intel-qwen36-oracle-seed-stage-v0"
DEFAULT_TOKEN_TOPK_SEED = (
    ROOT / "output/r0-oracle-seed-stage-20260626T034356Z/token-topk-seed.jsonl"
)
DEFAULT_RAW_DIR = Path(
    "/Users/jiawei-macmini/projects/intel-box/output/"
    "intel-box-qwen36-native-llama-generation-oracle-cpu-20260615T133419Z/"
    "remote-output/llama-generation-oracle/raw"
)
MISSING_FOR_R0_CLOSE = [
    "full_acceptance_teacher_forced_distribution_references",
    "per_boundary_reference_inputs",
    "per_boundary_reference_outputs",
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--token-topk-seed",
      type=Path,
      default=DEFAULT_TOKEN_TOPK_SEED,
      help="Staged token/top-k seed JSONL.",
  )
  parser.add_argument(
      "--raw-dir",
      type=Path,
      default=DEFAULT_RAW_DIR,
      help="llama.cpp raw response directory containing primary/repeat JSON.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-teacher-forced-distribution-seed-<UTC>.",
  )
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    value = json.load(fh)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.write_text(
      "".join(
          json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
          for row in rows
      ),
      encoding="utf-8",
  )


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def is_int_list(value: Any) -> bool:
  return isinstance(value, list) and all(isinstance(item, int) for item in value)


def normalize_top_logprobs(value: Any, case_id: str, position: int) -> list[dict[str, Any]]:
  if not isinstance(value, list) or not value:
    raise SystemExit(f"{case_id}: position {position} missing top_logprobs")
  out: list[dict[str, Any]] = []
  for index, item in enumerate(value):
    if not isinstance(item, dict):
      raise SystemExit(f"{case_id}: position {position} top_logprobs[{index}] must be an object")
    token_id = item.get("id")
    logprob = item.get("logprob")
    if not isinstance(token_id, int) or not isinstance(logprob, (float, int)):
      raise SystemExit(f"{case_id}: position {position} invalid top_logprobs[{index}]")
    out.append(
        {
            "bytes": item.get("bytes") if is_int_list(item.get("bytes")) else [],
            "id": token_id,
            "logprob": float(logprob),
            "token": item.get("token") if isinstance(item.get("token"), str) else "",
        }
    )
  return out


def completion_probabilities(path: Path) -> list[dict[str, Any]]:
  value = load_json(path)
  probabilities = value.get("completion_probabilities")
  if not isinstance(probabilities, list) or not probabilities:
    raise SystemExit(f"{path}: missing completion_probabilities")
  out: list[dict[str, Any]] = []
  for index, item in enumerate(probabilities):
    if not isinstance(item, dict) or not isinstance(item.get("id"), int):
      raise SystemExit(f"{path}: invalid completion_probabilities[{index}]")
    out.append(item)
  return out


def target_by_name(row: dict[str, Any], target_name: str) -> dict[str, Any]:
  targets = row.get("generation_targets")
  if not isinstance(targets, list):
    raise SystemExit(f"{row.get('case_id')}: missing generation_targets")
  for target in targets:
    if isinstance(target, dict) and target.get("target") == target_name:
      return target
  raise SystemExit(f"{row.get('case_id')}: missing {target_name} target")


def response_path(raw_dir: Path, case_id: str, attempt: str) -> Path:
  return raw_dir / f"llama_{case_id}_short_generation_{attempt}_response.json"


def ids_from_probabilities(probabilities: list[dict[str, Any]]) -> list[int]:
  return [int(item["id"]) for item in probabilities]


def top_ids(probability: dict[str, Any], case_id: str, position: int) -> list[int]:
  return [item["id"] for item in normalize_top_logprobs(probability.get("top_logprobs"), case_id, position)]


def normalize_case(row: dict[str, Any], raw_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  case_id = row.get("case_id")
  if not isinstance(case_id, str) or not case_id:
    raise SystemExit("seed row missing case_id")
  if row.get("schema_version") != SEED_SCHEMA_VERSION:
    raise SystemExit(f"{case_id}: unexpected token/top-k seed schema")
  short_target = target_by_name(row, "short_generation")
  expected_ids = short_target.get("generated_token_ids")
  if not is_int_list(expected_ids) or not expected_ids:
    raise SystemExit(f"{case_id}: short_generation missing generated token ids")

  primary_path = response_path(raw_dir, case_id, "primary")
  repeat_path = response_path(raw_dir, case_id, "repeat")
  if not primary_path.exists():
    raise SystemExit(f"missing primary response: {primary_path}")
  if not repeat_path.exists():
    raise SystemExit(f"missing repeat response: {repeat_path}")
  primary = completion_probabilities(primary_path)
  repeat = completion_probabilities(repeat_path)
  primary_ids = ids_from_probabilities(primary)
  repeat_ids = ids_from_probabilities(repeat)

  positions: list[dict[str, Any]] = []
  top_signature_repeat_matches = True
  top1_matches_reference = True
  min_topk = None
  for index, item in enumerate(primary):
    top = normalize_top_logprobs(item.get("top_logprobs"), case_id, index)
    repeat_top = top_ids(repeat[index], case_id, index) if index < len(repeat) else []
    primary_top = [entry["id"] for entry in top]
    top_signature_repeat_matches = top_signature_repeat_matches and primary_top == repeat_top
    top1_matches_reference = top1_matches_reference and bool(primary_top) and primary_top[0] == item["id"]
    min_topk = len(top) if min_topk is None else min(min_topk, len(top))
    positions.append(
        {
            "context_token_count": int(row.get("prompt_token_count", 0)) + index,
            "position": index,
            "reference_token_id": item["id"],
            "reference_token_logprob": float(item.get("logprob")),
            "reference_token_text": item.get("token") if isinstance(item.get("token"), str) else "",
            "top1_id": primary_top[0] if primary_top else None,
            "top_logprob_id_signature": primary_top,
            "top_logprobs": top,
        }
    )

  staged_ids_match = primary_ids == expected_ids
  repeat_ids_match = repeat_ids == primary_ids
  case_status = {
      "case_id": case_id,
      "generated_token_count": len(primary_ids),
      "primary_response_path": str(primary_path),
      "primary_response_sha256": file_sha256(primary_path),
      "repeat_response_path": str(repeat_path),
      "repeat_response_sha256": file_sha256(repeat_path),
      "repeat_token_ids_match_primary": repeat_ids_match,
      "staged_generated_token_ids_match": staged_ids_match,
      "top1_matches_reference": top1_matches_reference,
      "top_logprob_signature_repeat_matches": top_signature_repeat_matches,
      "topk_min": min_topk,
  }
  out = {
      "case_id": case_id,
      "capture_mode": "deterministic_greedy_reference_path_distribution_seed",
      "distribution_positions": positions,
      "generated_token_count": len(primary_ids),
      "generated_token_ids": primary_ids,
      "limitations": {
          "full_acceptance_context_ladder": False,
          "not_a_per_boundary_tensor_bundle": True,
          "short_router_seed_only": True,
      },
      "prompt_set": row.get("prompt_set"),
      "prompt_token_count": row.get("prompt_token_count"),
      "prompt_token_ids": row.get("prompt_token_ids"),
      "prompt_utf8_sha256": row.get("prompt_utf8_sha256"),
      "schema_version": SCHEMA_VERSION,
      "source_reference_runtime": "llama.cpp CPU",
      "suite": row.get("suite"),
      "suite_manifest_name": row.get("suite_manifest_name"),
      "workstream": WORKSTREAM,
  }
  return out, case_status


def check(name: str, passed: bool, **extra: Any) -> dict[str, Any]:
  out = {"name": name, "pass": bool(passed)}
  out.update(extra)
  return out


def build_correctness(rows: list[dict[str, Any]], case_status: list[dict[str, Any]]) -> dict[str, Any]:
  total_positions = sum(int(row.get("generated_token_count", 0)) for row in rows)
  topk_values = sorted({status.get("topk_min") for status in case_status})
  checks = [
      check("seed_cases_loaded", len(rows) >= 6, count=len(rows)),
      check("distribution_positions_recorded", total_positions > 0, count=total_positions),
      check(
          "staged_generated_token_ids_match",
          all(status.get("staged_generated_token_ids_match") is True for status in case_status),
      ),
      check(
          "repeat_token_ids_match_primary",
          all(status.get("repeat_token_ids_match_primary") is True for status in case_status),
      ),
      check(
          "top_logprob_signature_repeat_matches",
          all(status.get("top_logprob_signature_repeat_matches") is True for status in case_status),
      ),
      check(
          "top1_matches_reference",
          all(status.get("top1_matches_reference") is True for status in case_status),
      ),
      check(
          "topk_min_at_least_5",
          all(isinstance(status.get("topk_min"), int) and status["topk_min"] >= 5 for status in case_status),
          observed=topk_values,
      ),
  ]
  return {
      "case_status": case_status,
      "checks": checks,
      "required_checks_passed": all(item["pass"] is True for item in checks),
      "schema_version": SCHEMA_VERSION,
      "total_distribution_positions": total_positions,
      "workstream": WORKSTREAM,
  }


def build_manifest(
    *,
    created_at: str,
    rows: list[dict[str, Any]],
    raw_dir: Path,
    token_topk_seed: Path,
    correctness: dict[str, Any],
) -> dict[str, Any]:
  return {
      "created_at": created_at,
      "model": {
          "batch_size": 1,
          "path": "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf",
          "sha256": "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e",
      },
      "r0_oracle_gate_closed": False,
      "schema_version": SCHEMA_VERSION,
      "seed_scope": {
          "case_count": len(rows),
          "distribution_positions": correctness["total_distribution_positions"],
          "prompt_sets": sorted({row.get("prompt_set") for row in rows if isinstance(row.get("prompt_set"), str)}),
          "top_k": 5,
      },
      "source": {
          "raw_dir": str(raw_dir),
          "token_topk_seed_path": str(token_topk_seed),
          "token_topk_seed_sha256": file_sha256(token_topk_seed),
      },
      "status": {
          "missing_for_r0_close": MISSING_FOR_R0_CLOSE,
          "per_boundary_bundle_available": False,
          "teacher_forced_distribution_seed_available": True,
          "teacher_forced_distribution_seed_is_full_acceptance_bundle": False,
      },
      "workstream": WORKSTREAM,
  }


def build_summary(manifest: dict[str, Any], correctness: dict[str, Any], out_dir: Path) -> str:
  scope = manifest["seed_scope"]
  return "\n".join(
      [
          "# R0 teacher-forced distribution seed",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- cases: {scope['case_count']}",
          f"- distribution positions: {scope['distribution_positions']}",
          f"- prompt sets: {', '.join(scope['prompt_sets'])}",
          f"- required checks passed: `{str(correctness['required_checks_passed']).lower()}`",
          f"- output: `{out_dir}`",
          "- R0 oracle gate closed: `false`",
          "",
          "This artifact captures per-position top-5 logprobs along the",
          "deterministic greedy reference path. It is a short/router distribution",
          "seed, not the full acceptance-ladder or per-boundary tensor bundle.",
          "",
      ]
  )


def main() -> None:
  args = parse_args()
  token_topk_seed = args.token_topk_seed.resolve()
  raw_dir = args.raw_dir.resolve()
  if not token_topk_seed.exists():
    raise SystemExit(f"missing token/top-k seed: {token_topk_seed}")
  if not raw_dir.exists():
    raise SystemExit(f"missing raw response directory: {raw_dir}")
  seed_rows = load_jsonl(token_topk_seed)
  rows: list[dict[str, Any]] = []
  case_status: list[dict[str, Any]] = []
  for row in seed_rows:
    normalized, status = normalize_case(row, raw_dir)
    rows.append(normalized)
    case_status.append(status)

  correctness = build_correctness(rows, case_status)
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-teacher-forced-distribution-seed-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  manifest = build_manifest(
      created_at=created_at,
      rows=rows,
      raw_dir=raw_dir,
      token_topk_seed=token_topk_seed,
      correctness=correctness,
  )
  write_json(out_dir / "manifest.json", manifest)
  write_jsonl(out_dir / "teacher-forced-distribution-seed.jsonl", rows)
  write_json(out_dir / "correctness.json", correctness)
  (out_dir / "summary.md").write_text(
      build_summary(manifest, correctness, out_dir),
      encoding="utf-8",
  )
  print(f"teacher-forced distribution seed required_checks_passed={correctness['required_checks_passed']}")
  print(f"distribution seed output: {out_dir}")
  if correctness["required_checks_passed"] is not True:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
