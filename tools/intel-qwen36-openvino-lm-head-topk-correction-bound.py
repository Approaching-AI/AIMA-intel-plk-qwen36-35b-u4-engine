#!/usr/bin/env python3
"""Screen a dynamic exact-row correction pass for the signed-Q4 LM head.

The first pass is the integrated signed-Q4 provider.  This offline gate ranks
its logits, replaces only the candidate top-K rows with logits from an exact
I8 correction source, and measures the resulting full-vocabulary
distribution against the stock reference.  When the correction source is the
stock reference itself the result is an explicit optimistic numeric bound;
an integrated route still has to recompute the selected rows from the
candidate's original I8 weights and pass the product gate.

No model or GPU worker is launched.  The gate consumes existing full-logit
artifacts so a failed codec can be rejected without another long-context run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-openvino-lm-head-topk-correction-bound-v0"
VOCAB_ROWS = 248_320
HIDDEN_SIZE = 2_048
Q4_TOTAL_BYTES = 255_777_792
CONSERVATIVE_BANDWIDTH_GBPS = 108.0
LM_HEAD_PROFILE_MS = 5.911
KILL_NUMBER_MS = 2.525586
DEFAULT_CANDIDATE = REPO / (
    "output/openvino-lm-head-i8q4-full-graph-20260718Tseq1464-clean-"
    "32k-o64/raw/sentinel_032k/correctness/candidate")
DEFAULT_REFERENCE = REPO / (
    "output/openvino-lm-head-i8q4-full-graph-20260718Tseq1464-clean-"
    "32k-o64/raw/sentinel_032k/correctness/stock")
DEFAULT_TOPK = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                2048, 4096, 8192)
ADMISSION_KLD = (0.005, 0.001, 0.0005, 0.0001)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
  parser.add_argument("--correction-dir", type=Path,
                      default=DEFAULT_REFERENCE)
  parser.add_argument("--reference-dir", type=Path,
                      default=DEFAULT_REFERENCE)
  parser.add_argument("--topk", type=int, nargs="+", default=DEFAULT_TOPK)
  parser.add_argument("--decode-only", action=argparse.BooleanOptionalAction,
                      default=True)
  args = parser.parse_args()
  if any(value < 0 or value >= VOCAB_ROWS for value in args.topk):
    parser.error(f"top-K values must be in [0, {VOCAB_ROWS})")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO))
  except ValueError:
    return str(path.resolve())


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    output_relative = str(output.resolve().relative_to(REPO))
  except ValueError:
    output_relative = ""
  rows = [row for row in rows
          if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(rows), "status": rows}


def logits_files(directory: Path) -> dict[int, Path]:
  result: dict[int, Path] = {}
  for path in directory.glob("step*-logits.f32"):
    middle = path.name.removeprefix("step").removesuffix("-logits.f32")
    if middle.isdigit():
      result[int(middle)] = path
  return result


def load_logits(path: Path) -> np.ndarray:
  values = np.fromfile(path, dtype="<f4")
  if values.shape != (VOCAB_ROWS,):
    raise ValueError(f"unexpected logits shape {values.shape}: {path}")
  if not np.isfinite(values).all():
    raise ValueError(f"non-finite logits: {path}")
  return values


def distribution_metrics(
    reference: np.ndarray, candidate: np.ndarray,
) -> dict[str, Any]:
  ref = reference.astype(np.float64, copy=False)
  cand = candidate.astype(np.float64, copy=False)
  ref_probability = np.exp(ref - float(np.max(ref)))
  cand_probability = np.exp(cand - float(np.max(cand)))
  ref_probability /= float(ref_probability.sum())
  cand_probability /= float(cand_probability.sum())
  epsilon = np.finfo(np.float64).tiny
  ref_top1 = int(np.argmax(ref))
  candidate_top1 = int(np.argmax(cand))
  return {
      "kld": float(np.sum(ref_probability * (
          np.log(np.maximum(ref_probability, epsilon)) -
          np.log(np.maximum(cand_probability, epsilon))))),
      "max_abs": float(np.max(np.abs(cand - ref))),
      "reference_top1": ref_top1,
      "candidate_top1": candidate_top1,
      "top1_match": ref_top1 == candidate_top1,
  }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  worst = max(rows, key=lambda row: float(row["kld"]))
  return {
      "phase_count": len(rows),
      "max_kld": float(worst["kld"]),
      "worst_phase": int(worst["phase"]),
      "top1_matches": sum(bool(row["top1_match"]) for row in rows),
      "minimum_selected_reference_probability_mass": min(
          float(row["selected_reference_probability_mass"]) for row in rows),
  }


def traffic_bound(topk: int) -> dict[str, Any]:
  # The base accounting already includes one output write.  A correction pass
  # additionally reads the half logits for selection, gathers K original I8
  # rows and scales, reads/writes K row values, and carries K uint indices.
  extra_bytes = (
      VOCAB_ROWS * 2 + topk * (HIDDEN_SIZE + 2 + 2 + 4))
  total_bytes = Q4_TOTAL_BYTES + extra_bytes
  floor_ms = total_bytes / (CONSERVATIVE_BANDWIDTH_GBPS * 1e9) * 1000.0
  ceiling_ms = LM_HEAD_PROFILE_MS - floor_ms
  return {
      "q4_base_total_bytes": Q4_TOTAL_BYTES,
      "correction_extra_bytes": extra_bytes,
      "total_bytes": total_bytes,
      "effective_bits_per_weight": 4.0 + 8.0 * extra_bytes /
          (VOCAB_ROWS * HIDDEN_SIZE),
      "bandwidth_floor_ms_at_108_gbps": floor_ms,
      "savings_ceiling_ms_vs_5_911_ms_profile_row": ceiling_ms,
      "ceiling_clears_2_525586_ms_kill_number": ceiling_ms > KILL_NUMBER_MS,
      "selection_compute_and_launch_overhead_included": False,
  }


def write_summary(output: Path, metrics: dict[str, Any]) -> None:
  lines = [
      "# LM-head dynamic top-K exact correction bound",
      "",
      f"- verdict: `{metrics['verdict']}`",
      f"- phases: `{metrics['phase_indices']}`",
      f"- correction source equals reference: "
      f"`{str(metrics['correction_source_is_reference']).lower()}`",
      "- GPU workers launched: `0`",
      "",
      "| K | max KLD | top-1 | min selected probability mass | "
      "effective bits | bandwidth ceiling clears gap |",
      "|---:|---:|---:|---:|---:|:---:|",
  ]
  for key, row in metrics["topk_rows"].items():
    summary = row["summary"]
    traffic = row["traffic_bound"]
    lines.append(
        f"| {key} | {summary['max_kld']:.9g} | "
        f"{summary['top1_matches']}/{summary['phase_count']} | "
        f"{summary['minimum_selected_reference_probability_mass']:.9g} | "
        f"{traffic['effective_bits_per_weight']:.6f} | "
        f"{traffic['ceiling_clears_2_525586_ms_kill_number']} |")
  lines.extend([
      "",
      "The bandwidth row excludes selection reduction, synchronization, and "
      "kernel-launch cost. When correction equals the stock reference this is "
      "an optimistic numeric bound, not product correctness or a speedup claim.",
      "",
  ])
  (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  candidate_dir = args.candidate_dir.resolve()
  correction_dir = args.correction_dir.resolve()
  reference_dir = args.reference_dir.resolve()
  if output.exists():
    raise SystemExit(f"output already exists: {output}")
  for path in (candidate_dir, correction_dir, reference_dir):
    if not path.is_dir():
      raise SystemExit(f"missing logits directory: {path}")
  output.mkdir(parents=True)

  sources = {
      "candidate": logits_files(candidate_dir),
      "correction": logits_files(correction_dir),
      "reference": logits_files(reference_dir),
  }
  phases = sorted(set.intersection(*(set(rows) for rows in sources.values())))
  if args.decode_only:
    phases = [phase for phase in phases if phase > 0]
  if not phases:
    raise SystemExit("no common logits phases")
  topk_values = sorted(set(args.topk))
  max_topk = max(topk_values)
  topk_rows: dict[str, dict[str, Any]] = {
      str(value): {"phases": [], "traffic_bound": traffic_bound(value)}
      for value in topk_values}
  input_hashes: dict[str, dict[str, str]] = {
      name: {} for name in sources}

  for phase in phases:
    candidate = load_logits(sources["candidate"][phase])
    correction = load_logits(sources["correction"][phase])
    reference = load_logits(sources["reference"][phase])
    if max_topk:
      selected = np.argpartition(candidate, -max_topk)[-max_topk:]
      selected = selected[np.argsort(candidate[selected])[::-1]]
    else:
      selected = np.empty(0, dtype=np.int64)
    shifted = reference.astype(np.float64) - float(np.max(reference))
    reference_probability = np.exp(shifted)
    reference_probability /= float(reference_probability.sum())
    for topk in topk_values:
      corrected = candidate.copy()
      indices = selected[:topk]
      corrected[indices] = correction[indices]
      row = distribution_metrics(reference, corrected)
      row.update({
          "phase": phase,
          "selected_reference_probability_mass": float(
              reference_probability[indices].sum()),
          "reference_top1_selected": bool(
              row["reference_top1"] in indices) if topk else False,
      })
      topk_rows[str(topk)]["phases"].append(row)
    for name, files in sources.items():
      input_hashes[name][str(phase)] = sha256(files[phase])

  for row in topk_rows.values():
    row["summary"] = summarize(row["phases"])
  admissions: dict[str, Any] = {}
  for threshold in ADMISSION_KLD:
    eligible = [
        value for value in topk_values
        if topk_rows[str(value)]["summary"]["max_kld"] <= threshold and
        topk_rows[str(value)]["summary"]["top1_matches"] == len(phases) and
        topk_rows[str(value)]["traffic_bound"][
            "ceiling_clears_2_525586_ms_kill_number"]]
    admissions[str(threshold)] = min(eligible) if eligible else None
  selected_topk = admissions[str(ADMISSION_KLD[0])]
  correction_is_reference = correction_dir == reference_dir
  verdict = (
      "fund_dynamic_topk_exact_row_component_bound"
      if selected_topk is not None else
      "reject_dynamic_topk_exact_row_numeric_bound")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "verdict": verdict,
      "git": git_state(output),
      "gpu_workers_launched": 0,
      "decode_only": args.decode_only,
      "phase_indices": phases,
      "correction_source_is_reference": correction_is_reference,
      "bound_kind": (
          "optimistic_exact-stock-row-correction"
          if correction_is_reference else "candidate-I8-row-correction"),
      "selected_topk_at_product_kld_limit": selected_topk,
      "minimum_topk_by_kld_limit": admissions,
      "topk_rows": topk_rows,
      "inputs": {
          "candidate_dir": display_path(candidate_dir),
          "correction_dir": display_path(correction_dir),
          "reference_dir": display_path(reference_dir),
          "sha256_by_phase": input_hashes,
      },
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  write_summary(output, metrics)
  print(json.dumps({
      "verdict": verdict,
      "phase_count": len(phases),
      "selected_topk": selected_topk,
      "minimum_topk_by_kld_limit": admissions,
      "output": display_path(output),
  }, sort_keys=True))
  return 0 if selected_topk is not None else 2


if __name__ == "__main__":
  raise SystemExit(main())
