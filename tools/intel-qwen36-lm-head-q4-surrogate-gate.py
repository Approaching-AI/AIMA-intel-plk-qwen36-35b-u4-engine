#!/usr/bin/env python3
"""Gate a Q4 surrogate plus exact-Q6 candidate-refine LM-head route."""

from __future__ import annotations

import argparse
import array
import glob
import hashlib
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-lm-head-q4-surrogate-gate-v0"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_SURROGATE = (
    ROOT / "output/lm-head-q4-surrogate-asset-20260711/"
    "qwen36-output-head-q4-surrogate.gguf")
DEFAULT_SMOKE_GLOB = str(
    ROOT / "output/seq583-final-norm-sparse-state-fit-collection-fresh_*"
    "-20260710Tseq583Z/smoke.json")
DEFAULT_TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/"
    "acceptance-matrix.json")
DEFAULT_Q4_GATEUP = (
    ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-20260702T225500Z/"
    "probe-result.json")
DEFAULT_Q4_DOWN = (
    ROOT / "output/gpu-q6-qmatvec-ffn-down-full-20260702T233000Z/"
    "probe-result.json")
CPP_SOURCE = ROOT / "engine/tools/lm_head_q4_surrogate_gate.cpp"
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
HIDDEN_SIZE = 2048
VOCAB_SIZE = 248320
ACTIVE_EXPERTS = 8
EXPERTS = 256
KV_BYTES_PER_CONTEXT_TOKEN = 20480
EXPECTED_ACTIVE_BYTES = 1_975_676_544


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--surrogate", type=Path, default=DEFAULT_SURROGATE)
  parser.add_argument("--smoke-glob", default=DEFAULT_SMOKE_GLOB)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_TENSOR_INDEX)
  parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
  parser.add_argument("--q4-gateup", type=Path, default=DEFAULT_Q4_GATEUP)
  parser.add_argument("--q4-down", type=Path, default=DEFAULT_Q4_DOWN)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=3600)
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
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def collect_vectors(pattern: str) -> tuple[list[float], list[dict[str, Any]]]:
  values: list[float] = []
  metadata: list[dict[str, Any]] = []
  paths = [Path(path) for path in sorted(glob.glob(pattern))]
  if not paths:
    raise SystemExit(f"no smoke files match: {pattern}")
  for path in paths:
    smoke = load_json(path)
    if smoke.get("sparse_head_logit_bias_enabled") is not False:
      raise SystemExit(f"source smoke has sparse head bias enabled: {path}")
    ladder = smoke.get("distribution_ladder", {})
    steps = ladder.get("steps", []) if isinstance(ladder, dict) else []
    if len(steps) != 8:
      raise SystemExit(f"{path}: expected eight distribution steps")
    for step in steps:
      vector = step.get("gpu_final_norm_fit_observables")
      if (
          not isinstance(vector, list)
          or len(vector) != HIDDEN_SIZE
          or not all(isinstance(item, (int, float)) and math.isfinite(item)
                     for item in vector)
      ):
        raise SystemExit(f"{path}: invalid final-norm vector")
      values.extend(float(item) for item in vector)
      metadata.append({
          "case_id": smoke.get("case_id"),
          "recorded_gpu_top1_id": step.get("gpu_top1_id"),
          "source_required_checks_passed": smoke.get("required_checks_passed"),
          "source": str(path.relative_to(ROOT)),
          "token_index": step.get("token_index"),
          "token_position": step.get("token_position"),
      })
  return values, metadata


def write_vectors(path: Path, values: list[float]) -> None:
  payload = array.array("f", values)
  if payload.itemsize != 4:
    raise SystemExit("host float array is not 32-bit")
  if os.sys.byteorder != "little":
    payload.byteswap()
  with path.open("wb") as handle:
    payload.tofile(handle)


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        timeout=timeout_s)
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": (error.stderr or "") if isinstance(error.stderr, str) else "",
        "stdout": (error.stdout or "") if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }
  return {
      "command": command,
      "returncode": process.returncode,
      "stderr": process.stderr,
      "stdout": process.stdout,
      "timed_out": False,
  }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if line:
        value = json.loads(line)
        if not isinstance(value, dict):
          raise SystemExit(f"{path}: expected object rows")
        rows.append(value)
  return rows


def active_byte_inventory(path: Path) -> dict[str, Any]:
  components: dict[str, int] = {}
  quant_types: dict[str, int] = {}
  for row in load_jsonl(path):
    name = str(row.get("name", ""))
    suffix = str(row.get("suffix", ""))
    nbytes = int(row.get("nbytes", 0))
    if name == "token_embd.weight":
      active = nbytes // VOCAB_SIZE
      component = "embedding_lookup"
    elif suffix in ("ffn_gate_up_exps.weight", "ffn_down_exps.weight"):
      active = nbytes * ACTIVE_EXPERTS // EXPERTS
      component = "selected_experts"
    else:
      active = nbytes
      if name == "output.weight":
        component = "lm_head"
      elif suffix == "ffn_gate_inp.weight":
        component = "router"
      elif "shexp" in suffix:
        component = "shared_expert"
      elif suffix.startswith("attn_") and suffix not in (
          "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"):
        component = "attention_projection"
      elif suffix.startswith("ssm_"):
        component = "linear_attention"
      else:
        component = "norm_and_small_state"
    components[component] = components.get(component, 0) + active
    quant = str(row.get("ggml_type_name", "unknown"))
    quant_types[quant] = quant_types.get(quant, 0) + active
  return {
      "components": dict(sorted(components.items())),
      "quant_types": dict(sorted(quant_types.items())),
      "strict_active_bytes": sum(components.values()),
  }


def percentile(values: list[int], fraction: float) -> int:
  ordered = sorted(values)
  index = max(0, math.ceil(fraction * len(ordered)) - 1)
  return ordered[index]


def build_summary(result: dict[str, Any]) -> str:
  selected = result["selected_candidate_cap"]
  traffic = result["traffic_budget"]
  correctness = result["correctness"]
  return "\n".join([
      "# Q4 surrogate + exact-Q6 LM-head component gate",
      "",
      f"- required checks passed: `{str(result['required_checks_passed']).lower()}`",
      f"- final-norm vectors: `{result['vector_count']}` across "
      f"`{result['source_case_count']}` fresh cases",
      f"- exact-head top-1 source matches: "
      f"`{correctness['recorded_top1_match_count']}/{result['vector_count']}`",
      f"- selected candidate cap: `{selected}`",
      f"- selected-cap recall: `{correctness['recall_by_cap'].get(str(selected))}`",
      f"- selected-cap max KLD: "
      f"`{correctness['max_hybrid_kld_by_cap'].get(str(selected))}`",
      f"- LM-head traffic: `{traffic['exact_head_bytes']}` -> "
      f"`{traffic['candidate_head_bytes']}` bytes/token",
      f"- strict-byte saving: `{traffic['head_bytes_saved']}` bytes/token",
      f"- maximum target demand after the cut: "
      f"`{traffic['max_required_carrier_gb_s']:.6f} GB/s`",
      f"- conservative measured Q4 carrier: "
      f"`{traffic['conservative_q4_carrier_gb_s']:.6f} GB/s`",
      "",
      "This is a component and traffic-feasibility result, not a native token",
      "or product speed claim. The replacement route still needs an exact",
      "router candidate-refine gate, a >=95 GB/s split-Q6 DPAS carrier, and a",
      "whole-layer teacher-forced distribution pass before token promotion.",
      "",
  ])


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (args.out_dir or (
      ROOT / f"output/lm-head-q4-surrogate-gate-{stamp}")).resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  for required in (
      args.model, args.surrogate, args.tensor_index, args.acceptance,
      args.q4_gateup, args.q4_down, args.cxx, CPP_SOURCE, GGUF_SOURCE):
    if not required.is_file():
      raise SystemExit(f"required input missing: {required}")

  values, vector_metadata = collect_vectors(args.smoke_glob)
  vectors_path = raw_dir / "final-norm-vectors.f32"
  write_vectors(vectors_path, values)
  write_json(raw_dir / "vector-metadata.json", vector_metadata)

  binary = raw_dir / "lm-head-q4-surrogate-gate"
  build = run([
      str(args.cxx), "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
      "-I", str(ROOT / "engine/include"), str(GGUF_SOURCE),
      str(CPP_SOURCE), "-o", str(binary),
  ], args.timeout_s)
  (raw_dir / "build.stdout").write_text(build["stdout"], encoding="utf-8")
  (raw_dir / "build.stderr").write_text(build["stderr"], encoding="utf-8")

  probe = (
      run([
          str(binary), str(args.model), str(args.surrogate), str(vectors_path),
          str(len(vector_metadata)),
      ], args.timeout_s)
      if build["returncode"] == 0
      else {"command": [], "returncode": 1, "stdout": "", "stderr": "build failed",
            "timed_out": False}
  )
  (raw_dir / "probe.stdout").write_text(probe["stdout"], encoding="utf-8")
  (raw_dir / "probe.stderr").write_text(probe["stderr"], encoding="utf-8")
  parsed = json.loads(probe["stdout"]) if probe["returncode"] == 0 else {}

  caps = [int(value) for value in parsed.get("caps", [])]
  ranks = [int(value) for value in parsed.get("ranks", [])]
  exact_top1 = [int(value) for value in parsed.get("exact_top1", [])]
  hybrid_kld = parsed.get("hybrid_kld", [])
  recall_by_cap = {
      str(cap): (sum(rank <= cap for rank in ranks) / len(ranks) if ranks else 0.0)
      for cap in caps
  }
  max_hybrid_kld_by_cap = {
      str(cap): max(float(value) for value in hybrid_kld[index])
      for index, cap in enumerate(caps)
      if index < len(hybrid_kld) and hybrid_kld[index]
  }
  selected_cap = next((
      cap for cap in caps
      if recall_by_cap.get(str(cap)) == 1.0
      and max_hybrid_kld_by_cap.get(str(cap), math.inf) <= 0.005
  ), None)
  recorded_top1 = [row["recorded_gpu_top1_id"] for row in vector_metadata]
  recorded_matches = sum(
      lhs == rhs for lhs, rhs in zip(exact_top1, recorded_top1, strict=False))

  inventory = active_byte_inventory(args.tensor_index)
  acceptance = load_json(args.acceptance)
  exact_head_bytes = int(parsed.get("exact_head_bytes", 0))
  surrogate_head_bytes = int(parsed.get("surrogate_head_bytes", 0))
  exact_row_bytes = exact_head_bytes // VOCAB_SIZE if exact_head_bytes else 0
  candidate_head_bytes = (
      surrogate_head_bytes + selected_cap * exact_row_bytes
      if selected_cap is not None else exact_head_bytes)
  head_bytes_saved = exact_head_bytes - candidate_head_bytes
  q4_gateup = load_json(args.q4_gateup)
  q4_down = load_json(args.q4_down)
  carriers = [
      float(q4_gateup["gpu_effective_packed_gb_s"]),
      float(q4_down["gpu_effective_packed_gb_s"]),
  ]
  conservative_carrier = min(carriers)
  target_rows = []
  targets = acceptance["bootstrap_targets"]["decode_tokens_s"]
  for bucket in acceptance["matrix"]["input_buckets"]:
    target = float(targets[str(bucket)])
    bytes_per_token = (
        inventory["strict_active_bytes"] - exact_head_bytes
        + candidate_head_bytes + bucket * KV_BYTES_PER_CONTEXT_TOKEN)
    required = bytes_per_token / 1.0e9 * target
    target_rows.append({
        "bucket": bucket,
        "bytes_per_token": bytes_per_token,
        "decode_target_tokens_s": target,
        "required_carrier_gb_s": required,
    })
  max_target_row = max(target_rows, key=lambda row: row["required_carrier_gb_s"])

  checks = [
      {"name": "build_passed", "pass": build["returncode"] == 0},
      {"name": "probe_passed", "pass": probe["returncode"] == 0},
      {"name": "component_schema", "pass": parsed.get("schema_version") ==
       "intel-qwen36-lm-head-q4-surrogate-component-v0"},
      {"name": "fresh_vector_matrix_96", "pass": len(vector_metadata) >= 96},
      {"name": "fresh_case_count_12", "pass": len({row["case_id"] for row in vector_metadata}) >= 12},
      {"name": "exact_head_q6_bytes", "pass": exact_head_bytes == 417_177_600},
      {"name": "surrogate_head_q4_bytes", "pass": surrogate_head_bytes == 286_064_640},
      {"name": "exact_top1_matches_recorded_source", "pass": recorded_matches == len(vector_metadata)},
      {"name": "candidate_cap_at_most_4096", "pass": selected_cap is not None and selected_cap <= 4096},
      {"name": "selected_cap_full_recall", "pass": selected_cap is not None and recall_by_cap.get(str(selected_cap)) == 1.0},
      {"name": "selected_cap_distribution_kld", "pass": selected_cap is not None and max_hybrid_kld_by_cap.get(str(selected_cap), math.inf) <= 0.005},
      {"name": "strict_active_byte_inventory", "pass": inventory["strict_active_bytes"] == EXPECTED_ACTIVE_BYTES},
      {"name": "strict_demand_below_measured_q4_carrier", "pass": max_target_row["required_carrier_gb_s"] <= conservative_carrier},
  ]
  required_checks_passed = all(row["pass"] for row in checks)

  result = {
      "checks": checks,
      "correctness": {
          "max_hybrid_kld_by_cap": max_hybrid_kld_by_cap,
          "max_rank": max(ranks) if ranks else None,
          "p50_rank": percentile(ranks, 0.50) if ranks else None,
          "p90_rank": percentile(ranks, 0.90) if ranks else None,
          "p99_rank": percentile(ranks, 0.99) if ranks else None,
          "recall_by_cap": recall_by_cap,
          "recorded_top1_match_count": recorded_matches,
          "surrogate_top1_match_count": sum(
              lhs == rhs for lhs, rhs in zip(
                  exact_top1, parsed.get("surrogate_top1", []), strict=False)),
          "surrogate_kld_max": max(parsed.get("surrogate_kld", [math.inf])),
      },
      "created_at": created_at,
      "git": git_state(),
      "inventory": inventory,
      "model": {
          "path": str(args.model),
          "size_bytes": args.model.stat().st_size,
      },
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "selected_candidate_cap": selected_cap,
      "source_case_count": len({row["case_id"] for row in vector_metadata}),
      "source_smoke_glob": args.smoke_glob,
      "speedup_claims_allowed": False,
      "surrogate": {
          "path": str(args.surrogate),
          "sha256": sha256_file(args.surrogate),
          "size_bytes": args.surrogate.stat().st_size,
      },
      "timing_diagnostic": {
          "exact_cpu_ms": parsed.get("exact_ms"),
          "surrogate_cpu_ms": parsed.get("surrogate_ms"),
      },
      "traffic_budget": {
          "candidate_head_bytes": candidate_head_bytes,
          "conservative_q4_carrier_gb_s": conservative_carrier,
          "exact_head_bytes": exact_head_bytes,
          "exact_q6_row_bytes": exact_row_bytes,
          "head_bytes_saved": head_bytes_saved,
          "head_traffic_reduction": (
              head_bytes_saved / exact_head_bytes if exact_head_bytes else 0.0),
          "max_required_carrier_bucket": max_target_row["bucket"],
          "max_required_carrier_gb_s": max_target_row["required_carrier_gb_s"],
          "measured_q4_carriers_gb_s": carriers,
          "surrogate_head_bytes": surrogate_head_bytes,
          "target_rows": target_rows,
      },
      "vector_count": len(vector_metadata),
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "correctness": result["correctness"],
      "required_checks_passed": required_checks_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "standalone LM-head component gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    metrics = {
        "required_checks_passed": required_checks_passed,
        "vector_count": len(vector_metadata),
        "selected_candidate_cap": selected_cap,
        "max_rank": result["correctness"]["max_rank"],
        "selected_cap_max_kld": (
            max_hybrid_kld_by_cap.get(str(selected_cap))
            if selected_cap is not None else None),
        "head_bytes_saved": head_bytes_saved,
        "max_required_carrier_gb_s": max_target_row["required_carrier_gb_s"],
    }
    for metric, value in metrics.items():
      handle.write(json.dumps({"metric": metric, "value": value}) + "\n")
  (out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
      "selected_candidate_cap": selected_cap,
  }))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
