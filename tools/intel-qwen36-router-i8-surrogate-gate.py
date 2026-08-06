#!/usr/bin/env python3
"""Gate an I8 router surrogate plus exact-F32 candidate refinement."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-router-i8-surrogate-gate-v0"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_VECTOR_GLOB = str(ROOT / "output/**/attn_post_norm-*.bin")
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/"
    "acceptance-matrix.json")
DEFAULT_HEAD_GATE = (
    ROOT / "output/lm-head-q4-surrogate-gate-20260711Tseq627Z/result.json")
DEFAULT_Q4_GATEUP = (
    ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/"
    "probe-result.json")
DEFAULT_Q4_DOWN = (
    ROOT / "output/gpu-q6-qmatvec-ffn-down-full-20260702T233000Z/"
    "probe-result.json")
CPP_SOURCE = ROOT / "engine/tools/router_i8_surrogate_gate.cpp"
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
LAYER_COUNT = 40
HIDDEN_SIZE = 2048
EXPERT_COUNT = 256
EXACT_ROUTER_BYTES = 83_886_080
KV_BYTES_PER_CONTEXT_TOKEN = 20_480
OTHER_CARRIER_GB_S = 115.0
ROUTER_TOKEN_BUDGET_MS = 0.45
SCHEDULE_RESERVE_MS = 0.35
Q6_PROMOTION_KILL_GB_S = 96.0


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--vector-glob", default=DEFAULT_VECTOR_GLOB)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--head-gate", type=Path, default=DEFAULT_HEAD_GATE)
  parser.add_argument("--q4-gateup", type=Path, default=DEFAULT_Q4_GATEUP)
  parser.add_argument("--q4-down", type=Path, default=DEFAULT_Q4_DOWN)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        timeout=timeout_s)
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }
  return {
      "command": command,
      "returncode": process.returncode,
      "stderr": process.stderr,
      "stdout": process.stdout,
      "timed_out": False,
  }


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def corpus_priority(path: Path) -> tuple[int, str]:
  text = str(path)
  canonical = "r0-boundary-capture-run-20260627T054024Z" in text
  return (0 if canonical else 1, text)


def collect_corpus(pattern: str) -> list[dict[str, Any]]:
  expression = re.compile(r"attn_post_norm-(\d+)__tok\d+__ord\d+\.bin$")
  candidates: list[tuple[Path, int, str]] = []
  for raw_path in glob.glob(pattern, recursive=True):
    path = Path(raw_path)
    match = expression.search(path.name)
    if not match or path.stat().st_size != HIDDEN_SIZE * 4:
      continue
    layer = int(match.group(1))
    if not 0 <= layer < LAYER_COUNT:
      continue
    candidates.append((path, layer, sha256_file(path)))
  if not candidates:
    raise SystemExit(f"no router-input vectors match: {pattern}")

  by_hash: dict[str, tuple[Path, int]] = {}
  for path, layer, digest in sorted(
      candidates, key=lambda row: corpus_priority(row[0])):
    previous = by_hash.get(digest)
    if previous is not None and previous[1] != layer:
      raise SystemExit(f"same router vector hash assigned to layers {previous[1]} and {layer}")
    by_hash.setdefault(digest, (path, layer))

  rows = []
  for digest, (path, layer) in by_hash.items():
    oracle_matches = sorted(path.parent.glob(f"ffn_moe_logits-{layer}__tok*.bin"))
    oracle = next(
        (item for item in oracle_matches if item.stat().st_size == EXPERT_COUNT * 4),
        None)
    rows.append({
        "layer": layer,
        "oracle_logits_path": relative(oracle) if oracle else "-",
        "sha256": digest,
        "vector_path": relative(path),
    })
  rows.sort(key=lambda row: (row["layer"], row["sha256"]))
  return rows


def write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
  lines = ["# layer\tvector_path\toracle_logits_path\tsha256"]
  for row in rows:
    lines.append(
        f"{row['layer']}\t{row['vector_path']}\t"
        f"{row['oracle_logits_path']}\t{row['sha256']}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_candidate(probe: dict[str, Any]) -> dict[str, Any] | None:
  candidates = []
  row_count = int(probe.get("corpus_row_count", 0))
  for scheme in probe.get("schemes", []):
    if not isinstance(scheme, dict):
      continue
    resident = int(scheme.get("resident_bytes_per_layer", 0))
    for cap_text, metrics in scheme.get("caps", {}).items():
      if not isinstance(metrics, dict):
        continue
      cap = int(cap_text)
      if (
          cap <= 64
          and int(metrics.get("top8_match_count", 0)) == row_count
          and float(metrics.get("maximum_normalized_weight_abs_diff", math.inf))
          <= 1e-12
      ):
        candidate_bytes = resident + cap * HIDDEN_SIZE * 4
        candidates.append({
            "candidate_bytes_per_layer": candidate_bytes,
            "cap": cap,
            "resident_bytes_per_layer": resident,
            "scheme": scheme.get("name"),
        })
  return min(
      candidates,
      key=lambda row: (row["candidate_bytes_per_layer"], row["cap"]),
      default=None)


def build_budget(
    selected: dict[str, Any], head: dict[str, Any], acceptance: dict[str, Any],
    q4_carrier: float) -> dict[str, Any]:
  inventory = head["inventory"]
  quant_types = inventory["quant_types"]
  strict_active_bytes = int(inventory["strict_active_bytes"])
  exact_head_bytes = int(head["traffic_budget"]["exact_head_bytes"])
  candidate_head_bytes = int(head["traffic_budget"]["candidate_head_bytes"])
  surrogate_head_bytes = int(head["traffic_budget"]["surrogate_head_bytes"])
  router_candidate_bytes = (
      int(selected["candidate_bytes_per_layer"]) * LAYER_COUNT)
  exact_q6_row_bytes = int(head["traffic_budget"]["exact_q6_row_bytes"])
  q4_bytes = int(quant_types["Q4_K"]) + surrogate_head_bytes
  q6_bytes = (
      int(quant_types["Q6_K"]) - exact_head_bytes
      + int(selected["cap"]) * exact_q6_row_bytes)
  active_candidate_bytes = (
      strict_active_bytes - exact_head_bytes + candidate_head_bytes
      - EXACT_ROUTER_BYTES + router_candidate_bytes)
  other_bytes = active_candidate_bytes - q4_bytes - q6_bytes - router_candidate_bytes
  if min(q4_bytes, q6_bytes, router_candidate_bytes, other_bytes) < 0:
    raise SystemExit("mixed-carrier byte decomposition is invalid")

  target_rows = []
  targets = acceptance["bootstrap_targets"]["decode_tokens_s"]
  for bucket in acceptance["matrix"]["input_buckets"]:
    target = float(targets[str(bucket)])
    budget_ms = 1000.0 / target
    kv_bytes = int(bucket) * KV_BYTES_PER_CONTEXT_TOKEN
    q4_ms = q4_bytes / 1e6 / q4_carrier
    other_ms = (other_bytes + kv_bytes) / 1e6 / OTHER_CARRIER_GB_S
    fixed_ms = (
        q4_ms + other_ms + ROUTER_TOKEN_BUDGET_MS + SCHEDULE_RESERVE_MS)
    q6_budget_ms = budget_ms - fixed_ms
    required_q6 = math.inf if q6_budget_ms <= 0 else q6_bytes / 1e6 / q6_budget_ms
    uniform_bytes = active_candidate_bytes + kv_bytes
    target_rows.append({
        "bucket": bucket,
        "budget_ms": budget_ms,
        "decode_target_tokens_s": target,
        "fixed_non_q6_ms": fixed_ms,
        "q6_budget_ms": q6_budget_ms,
        "required_q6_carrier_gb_s": required_q6,
        "uniform_required_carrier_gb_s": uniform_bytes / 1e9 * target,
    })
  q6_worst = max(target_rows, key=lambda row: row["required_q6_carrier_gb_s"])
  uniform_worst = max(
      target_rows, key=lambda row: row["uniform_required_carrier_gb_s"])
  return {
      "active_candidate_bytes": active_candidate_bytes,
      "exact_router_bytes": EXACT_ROUTER_BYTES,
      "other_bytes": other_bytes,
      "q4_bytes": q4_bytes,
      "q4_carrier_gb_s": q4_carrier,
      "q6_bytes": q6_bytes,
      "q6_promotion_kill_gb_s": Q6_PROMOTION_KILL_GB_S,
      "required_q6_carrier_bucket": q6_worst["bucket"],
      "required_q6_carrier_gb_s": q6_worst["required_q6_carrier_gb_s"],
      "router_candidate_bytes": router_candidate_bytes,
      "router_token_budget_ms": ROUTER_TOKEN_BUDGET_MS,
      "router_traffic_reduction": (
          (EXACT_ROUTER_BYTES - router_candidate_bytes) / EXACT_ROUTER_BYTES),
      "schedule_reserve_ms": SCHEDULE_RESERVE_MS,
      "target_rows": target_rows,
      "uniform_required_carrier_bucket": uniform_worst["bucket"],
      "uniform_required_carrier_gb_s": uniform_worst[
          "uniform_required_carrier_gb_s"],
  }


def build_summary(result: dict[str, Any]) -> str:
  selected = result["selected_candidate"]
  budget = result["traffic_budget"]
  return "\n".join([
      "# I8 surrogate + exact-F32 router component gate",
      "",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- distinct real router inputs: `{result['corpus_row_count']}` across "
      f"`{result['covered_layer_count']}` layers",
      f"- oracle-paired exact top-8: `{result['oracle_top8_match_count']}/"
      f"{result['oracle_row_count']}`",
      f"- selected scheme / candidate cap: `{selected['scheme']}` / "
      f"`{selected['cap']}`",
      f"- router traffic: `{budget['exact_router_bytes']}` -> "
      f"`{budget['router_candidate_bytes']}` bytes/token",
      f"- router traffic reduction: `{budget['router_traffic_reduction']:.4%}`",
      f"- maximum mixed-carrier Q6 requirement: "
      f"`{budget['required_q6_carrier_gb_s']:.3f} GB/s` at "
      f"`{budget['required_q6_carrier_bucket']}` tokens",
      f"- Q6 promotion kill-number: `>={budget['q6_promotion_kill_gb_s']:.1f} GB/s`",
      f"- whole-token router budget: "
      f"`<={budget['router_token_budget_ms']:.3f} ms`",
      "",
      "The candidate refinement recomputes every surrogate candidate with the",
      "original F32 row before selecting top-8. Full recall therefore preserves",
      "the exact expert IDs and normalized weights; this is not a learned or",
      "post-hoc correction. It is still a CPU component/traffic gate, not a GPU",
      "kernel or native token speed claim. The next bounded gate must meet both",
      "the router timing budget and the Q6 carrier kill-number on Arc B390.",
      "",
  ])


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (args.out_dir or (
      ROOT / f"output/router-i8-surrogate-gate-{stamp}")).resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)

  for required in (
      args.model, args.acceptance, args.head_gate, args.q4_gateup,
      args.q4_down, args.cxx, CPP_SOURCE, GGUF_SOURCE):
    if not required.is_file():
      raise SystemExit(f"required input missing: {required}")

  corpus = collect_corpus(args.vector_glob)
  corpus_path = raw_dir / "corpus.tsv"
  write_corpus(corpus_path, corpus)
  write_json(raw_dir / "corpus.json", corpus)

  binary = raw_dir / "router-i8-surrogate-gate"
  build = run([
      str(args.cxx), "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
      "-I", str(ROOT / "engine/include"), str(GGUF_SOURCE),
      str(CPP_SOURCE), "-o", str(binary),
  ], args.timeout_s)
  (raw_dir / "build.stdout").write_text(build["stdout"], encoding="utf-8")
  (raw_dir / "build.stderr").write_text(build["stderr"], encoding="utf-8")
  probe_run = (
      run([str(binary), str(args.model), str(corpus_path)], args.timeout_s)
      if build["returncode"] == 0
      else {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
            "timed_out": False}
  )
  (raw_dir / "probe.stdout").write_text(
      probe_run["stdout"], encoding="utf-8")
  (raw_dir / "probe.stderr").write_text(
      probe_run["stderr"], encoding="utf-8")
  probe = json.loads(probe_run["stdout"]) if probe_run["returncode"] == 0 else {}
  selected = select_candidate(probe)

  head = load_json(args.head_gate)
  acceptance = load_json(args.acceptance)
  q4_carriers = [
      float(load_json(args.q4_gateup)["gpu_effective_packed_gb_s"]),
      float(load_json(args.q4_down)["gpu_effective_packed_gb_s"]),
  ]
  budget = (
      build_budget(selected, head, acceptance, min(q4_carriers))
      if selected is not None else {})
  covered_layers = int(probe.get("covered_layer_count", 0))
  oracle_rows = int(probe.get("oracle_row_count", 0))
  oracle_matches = int(probe.get("oracle_top8_match_count", 0))
  checks = [
      {"name": "build_passed", "pass": build["returncode"] == 0},
      {"name": "probe_passed", "pass": probe_run["returncode"] == 0},
      {"name": "component_schema", "pass": probe.get("schema_version") ==
       "intel-qwen36-router-i8-surrogate-component-v0"},
      {"name": "all_40_layers_covered", "pass": covered_layers == LAYER_COUNT},
      {"name": "real_vector_corpus_at_least_64", "pass": len(corpus) >= 64},
      {"name": "oracle_pairs_at_least_40", "pass": oracle_rows >= LAYER_COUNT},
      {"name": "exact_top8_matches_all_oracles", "pass": oracle_matches == oracle_rows},
      {"name": "exact_oracle_logit_tolerance", "pass":
       float(probe.get("oracle_max_abs_logit_diff", math.inf)) <= 2e-5},
      {"name": "candidate_cap_at_most_64", "pass":
       selected is not None and int(selected["cap"]) <= 64},
      {"name": "router_candidate_traffic_at_most_32mb", "pass":
       bool(budget) and int(budget["router_candidate_bytes"]) <= 32_000_000},
      {"name": "head_component_gate_passed", "pass":
       head.get("required_checks_passed") is True},
      {"name": "uniform_demand_below_q4_carrier", "pass":
       bool(budget) and budget["uniform_required_carrier_gb_s"] <= min(q4_carriers)},
      {"name": "q6_kill_number_bounded", "pass":
       bool(budget) and budget["required_q6_carrier_gb_s"]
       <= Q6_PROMOTION_KILL_GB_S},
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  result = {
      "checks": checks,
      "corpus_row_count": len(corpus),
      "covered_layer_count": covered_layers,
      "created_at": created_at,
      "git": git_state(),
      "model": {"path": str(args.model), "size_bytes": args.model.stat().st_size},
      "oracle_max_abs_logit_diff": probe.get("oracle_max_abs_logit_diff"),
      "oracle_row_count": oracle_rows,
      "oracle_top8_match_count": oracle_matches,
      "probe": probe,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "selected_candidate": selected,
      "speedup_claims_allowed": False,
      "traffic_budget": budget,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "oracle_max_abs_logit_diff": result["oracle_max_abs_logit_diff"],
      "oracle_top8_match_count": oracle_matches,
      "required_checks_passed": required_checks_passed,
      "selected_candidate": selected,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "standalone router component gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = {
      "required_checks_passed": required_checks_passed,
      "corpus_row_count": len(corpus),
      "covered_layer_count": covered_layers,
      "selected_candidate_cap": selected["cap"] if selected else None,
      "router_candidate_bytes": budget.get("router_candidate_bytes"),
      "required_q6_carrier_gb_s": budget.get("required_q6_carrier_gb_s"),
  }
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in metrics.items():
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out_dir / "summary.md").write_text(
      build_summary(result) if selected else "# Router surrogate gate\n\nNo candidate passed.\n",
      encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
      "selected_candidate": selected,
  }))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
