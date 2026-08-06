#!/usr/bin/env python3
"""Stage the deterministic CPU llama.cpp oracle seed into current bundle shape."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-oracle-seed-stage-v0"
DEFAULT_SEED_DIR = ROOT / "output/r0-cpu-llama-oracle-seed-20260626T033706Z"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
MISSING_FOR_R0_CLOSE = [
    "teacher_forced_distribution_references",
    "per_boundary_reference_inputs",
    "per_boundary_reference_outputs",
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--seed-dir",
      type=Path,
      default=DEFAULT_SEED_DIR,
      help="Imported CPU llama.cpp seed artifact directory.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-oracle-seed-stage-<UTC>.",
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


def prompt_token_ids(row: dict[str, Any]) -> list[int]:
  reference = row.get("prompt_token_reference")
  if not isinstance(reference, dict):
    raise SystemExit(f"{row.get('case_id')}: missing prompt_token_reference")
  token_ids = reference.get("reference_token_ids")
  if not is_int_list(token_ids) or not token_ids:
    raise SystemExit(f"{row.get('case_id')}: missing prompt token ids")
  return token_ids


def normalize_top_logprobs(value: Any, case_id: str, target_name: str) -> list[dict[str, Any]]:
  if not isinstance(value, list) or not value:
    raise SystemExit(f"{case_id}/{target_name}: missing top_logprobs")
  out: list[dict[str, Any]] = []
  for index, item in enumerate(value):
    if not isinstance(item, dict):
      raise SystemExit(f"{case_id}/{target_name}: top_logprobs[{index}] must be an object")
    token_id = item.get("id")
    logprob = item.get("logprob")
    if not isinstance(token_id, int) or not isinstance(logprob, (float, int)):
      raise SystemExit(f"{case_id}/{target_name}: invalid top_logprobs[{index}]")
    out.append(
        {
            "bytes": item.get("bytes") if is_int_list(item.get("bytes")) else [],
            "id": token_id,
            "logprob": float(logprob),
            "token": item.get("token") if isinstance(item.get("token"), str) else "",
        }
    )
  return out


def normalize_target(case_id: str, target: dict[str, Any]) -> dict[str, Any]:
  target_name = target.get("target")
  if not isinstance(target_name, str) or not target_name:
    raise SystemExit(f"{case_id}: generation target missing target name")
  token_ids = target.get("generated_token_ids")
  if not is_int_list(token_ids) or not token_ids:
    raise SystemExit(f"{case_id}/{target_name}: missing generated token ids")
  max_new_tokens = target.get("max_new_tokens")
  if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
    raise SystemExit(f"{case_id}/{target_name}: invalid max_new_tokens")
  return {
      "generated_token_count": len(token_ids),
      "generated_token_ids": token_ids,
      "generated_token_ids_sha256": target.get("generated_token_ids_sha256"),
      "max_new_tokens": max_new_tokens,
      "target": target_name,
      "text_sha256": target.get("text_sha256"),
      "top_logprob_id_signature": target.get("top_logprob_id_signature"),
      "top_logprobs": normalize_top_logprobs(target.get("top_logprobs"), case_id, target_name),
  }


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
  case_id = row.get("case_id")
  if not isinstance(case_id, str) or not case_id:
    raise SystemExit("seed row missing case_id")
  targets = row.get("generation_targets")
  if not isinstance(targets, list) or not targets:
    raise SystemExit(f"{case_id}: missing generation_targets")
  normalized_targets = [normalize_target(case_id, target) for target in targets]
  target_names = {target["target"] for target in normalized_targets}
  for required in ("first_token", "short_generation"):
    if required not in target_names:
      raise SystemExit(f"{case_id}: missing {required} generation target")
  token_ids = prompt_token_ids(row)
  return {
      "case_id": case_id,
      "generation_targets": normalized_targets,
      "limitations": {
          "not_a_full_r0_oracle_bundle": True,
          "per_boundary_reference_inputs": False,
          "per_boundary_reference_outputs": False,
          "teacher_forced_distribution_references": False,
      },
      "prompt": row.get("prompt") if isinstance(row.get("prompt"), str) else "",
      "prompt_set": row.get("prompt_set"),
      "prompt_token_count": len(token_ids),
      "prompt_token_ids": token_ids,
      "prompt_utf8_bytes": row.get("prompt_utf8_bytes"),
      "prompt_utf8_len": row.get("prompt_utf8_len"),
      "prompt_utf8_sha256": row.get("prompt_utf8_sha256"),
      "schema_version": SCHEMA_VERSION,
      "source_reference_runtime": "llama.cpp CPU",
      "suite": row.get("suite"),
      "suite_manifest_name": row.get("suite_manifest_name"),
      "workstream": WORKSTREAM,
  }


def assert_seed_audit(seed_dir: Path) -> dict[str, Any]:
  audit = load_json(seed_dir / "audit.json")
  if audit.get("workstream") != WORKSTREAM:
    raise SystemExit("seed audit workstream mismatch")
  status = audit.get("r0_oracle_status")
  if not isinstance(status, dict):
    raise SystemExit("seed audit missing r0_oracle_status")
  if status.get("token_reference_seed_available") is not True:
    raise SystemExit("seed audit does not provide token reference seed")
  if status.get("topk_reference_seed_available") is not True:
    raise SystemExit("seed audit does not provide top-k reference seed")
  if status.get("r0_oracle_gate_closed") is not False:
    raise SystemExit("seed audit must not close R0 oracle gate")
  return audit


def build_manifest(
    *,
    created_at: str,
    rows: list[dict[str, Any]],
    seed_dir: Path,
    seed_refs: Path,
) -> dict[str, Any]:
  prompt_sets = sorted(
      {row.get("prompt_set") for row in rows if isinstance(row.get("prompt_set"), str)}
  )
  targets = sorted(
      {
          target.get("target")
          for row in rows
          for target in row.get("generation_targets", [])
          if isinstance(target.get("target"), str)
      }
  )
  topk_values = sorted(
      {
          len(target.get("top_logprobs", []))
          for row in rows
          for target in row.get("generation_targets", [])
      }
  )
  return {
      "created_at": created_at,
      "model": {
          "batch_size": 1,
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
      },
      "r0_oracle_gate_closed": False,
      "schema_version": SCHEMA_VERSION,
      "seed_scope": {
          "case_count": len(rows),
          "generation_targets": targets,
          "prompt_sets": prompt_sets,
          "top_logprobs_k_values": topk_values,
      },
      "source": {
          "audit_path": str(seed_dir / "audit.json"),
          "oracle_references_path": str(seed_refs),
          "oracle_references_sha256": file_sha256(seed_refs),
          "seed_dir": str(seed_dir),
      },
      "status": {
          "missing_for_r0_close": MISSING_FOR_R0_CLOSE,
          "per_boundary_bundle_available": False,
          "teacher_forced_distribution_references_available": False,
          "token_reference_seed_available": True,
          "topk_reference_seed_available": True,
      },
      "workstream": WORKSTREAM,
  }


def build_status(manifest: dict[str, Any]) -> dict[str, Any]:
  return {
      "available_subgates": [
          "prompt_token_ids",
          "first_token_top_k_logprobs",
          "short_greedy_generated_token_ids",
      ],
      "gate": "r0_oracle",
      "missing_subgates": MISSING_FOR_R0_CLOSE,
      "r0_oracle_gate_closed": False,
      "schema_version": manifest["schema_version"],
      "workstream": WORKSTREAM,
  }


def build_summary(manifest: dict[str, Any], out_dir: Path) -> str:
  scope = manifest["seed_scope"]
  missing = ", ".join(manifest["status"]["missing_for_r0_close"])
  return "\n".join(
      [
          "# R0 oracle seed stage",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- cases: {scope['case_count']}",
          f"- prompt sets: {', '.join(scope['prompt_sets'])}",
          f"- generation targets: {', '.join(scope['generation_targets'])}",
          f"- output: `{out_dir}`",
          "- R0 oracle gate closed: `false`",
          f"- still missing: {missing}",
          "",
          "This artifact is a deterministic CPU llama.cpp token/top-k seed only.",
          "It is not a complete teacher-forced or per-boundary oracle bundle.",
          "",
      ]
  )


def main() -> None:
  args = parse_args()
  seed_dir = args.seed_dir.resolve()
  seed_refs = seed_dir / "llama-generation-oracle/oracle-references.jsonl"
  if not seed_refs.exists():
    raise SystemExit(f"missing seed oracle references: {seed_refs}")
  assert_seed_audit(seed_dir)
  rows = [normalize_row(row) for row in load_jsonl(seed_refs)]
  case_ids = [row["case_id"] for row in rows]
  if len(case_ids) != len(set(case_ids)):
    raise SystemExit("duplicate seed case ids")
  if len(rows) < 6:
    raise SystemExit("expected at least six short/router seed cases")

  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "").replace("T", "T").replace("Z", "Z")
    out_dir = ROOT / f"output/r0-oracle-seed-stage-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  manifest = build_manifest(
      created_at=created_at,
      rows=rows,
      seed_dir=seed_dir,
      seed_refs=seed_refs,
  )
  write_json(out_dir / "manifest.json", manifest)
  write_jsonl(out_dir / "token-topk-seed.jsonl", rows)
  write_json(out_dir / "r0-gate-status.json", build_status(manifest))
  (out_dir / "summary.md").write_text(build_summary(manifest, out_dir), encoding="utf-8")
  print(f"staged oracle seed: {out_dir}")


if __name__ == "__main__":
  main()
