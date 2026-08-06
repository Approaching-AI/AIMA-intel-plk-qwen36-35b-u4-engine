#!/usr/bin/env python3
"""Audit fresh upstream opportunities after the isolated IGC 2.38.2 probe.

This is a source-only routing gate.  It refreshes official OpenVINO and IGC
metadata, matches the new changes against the exact accepted runtime census,
and tests the most favorable non-overlapping RMS + measured-IGC bundle against
the residual left by seq1233's optimistic FC component.  It invokes no
compiler, GPU context, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-post-igc-opportunity-bound-v0"

SEQ1233 = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
SEQ1239 = ROOT / (
    "output/openvino-decode-elementwise-residual-bound-"
    "20260715Tseq1239-cleanZ/metrics.json")
SEQ1240 = ROOT / (
    "output/openvino-accepted-carrier-profile-refresh-"
    "20260715Tseq1240-2k-warm17-cleanZ/metrics.json")
SEQ1299_WORKER = ROOT / (
    "output/openvino-igc2382-component-"
    "20260717Tseq1299-control-igc2344-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
SEQ1301 = ROOT / (
    "output/openvino-igc2382-component-gate-"
    "20260717Tseq1301-cleanZ/metrics.json")

ATTENTION_SOURCES = (
    ROOT / "engine/openvino/custom/iq36_hot_attention_single_owner.cl",
    ROOT / "engine/openvino/custom/iq36_hot_attention_tiled_helpers.cl",
    ROOT / "engine/openvino/custom/iq36_prefill_attention_tiled.cl",
)

OPENVINO_PULLS = {
    "rms_mvn_generalized": 36747,
    "igpu_device_subbuffer_zero_copy": 36645,
    "kv_broadcast_elimination": 36775,
    "rms_without_gamma_fusion": 36865,
    "gathermatmul_decompression_consumer": 36936,
    "sdpa_micro_prefetch_bounds": 36808,
}
EXPECTED_TITLES = {
    36747: "[GPU]Performance optimizing to RMS and MVN",
    36645: "[GPU] Enable zero-copy subbuffers for usm_device on iGPUs",
    36775: "apply GPU transformation to avoid KV cache broadcast",
    36865: "[GPU] Enable RMS fusion to match RMS without gamma",
    36936: "[CPU][GPU] Recognize GatherMatmul as a decompression-multiply consumer",
    36808: "[GPU] correct SDPA micro prefetch bounds",
}

IGC_LATEST_RELEASE = (
    "https://api.github.com/repos/intel/intel-graphics-compiler/releases/latest")
IGC_V239_TAG = (
    "https://api.github.com/repos/intel/intel-graphics-compiler/git/ref/"
    "tags/v2.39.0")
IGC_MASTER = (
    "https://api.github.com/repos/intel/intel-graphics-compiler/commits/master")
IGC_COMMITS = {
    "atomic_loop_runtime_unroll":
        "d2993be0905aec01c0a01a61944ca8655b37af9f",
    "post_ra_spill_cleanup":
        "68ef7642f5b2114bc6465b517ae3c9976d570150",
    "more_movi_revert":
        "e31774ffc72ade92ce6695f2575c10fae2558f24",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.network_timeout_s <= 0.0 or args.memory_stop_gib <= 0.0:
    parser.error("timeouts and memory stop must be positive")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      text=True, capture_output=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      text=True, capture_output=True).stdout.splitlines()
  allowed = {"tools/intel-qwen36-openvino-post-igc-opportunity-bound.py"}
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  dirty = []
  for row in rows:
    path = row[3:]
    if relative and path.startswith(relative):
      continue
    if path in allowed:
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def fetch(
    url: str, destination: Path, timeout_s: float,
    accept: str = "application/vnd.github+json",
) -> bytes:
  request = urllib.request.Request(
      url, headers={"Accept": accept,
                    "User-Agent": "intel-qwen36-opportunity-bound"})
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_json(url: str, destination: Path, timeout_s: float) -> dict[str, Any]:
  value = json.loads(fetch(url, destination, timeout_s))
  if not isinstance(value, dict):
    raise TypeError(f"expected object from {url}")
  return value


def pull_summary(payload: dict[str, Any], patch: bytes) -> dict[str, Any]:
  return {
      "number": payload.get("number"),
      "title": payload.get("title"),
      "state": payload.get("state"),
      "draft": payload.get("draft"),
      "merged_at": payload.get("merged_at"),
      "updated_at": payload.get("updated_at"),
      "head_sha": payload.get("head", {}).get("sha"),
      "base_sha": payload.get("base", {}).get("sha"),
      "html_url": payload.get("html_url"),
      "changed_files": payload.get("changed_files"),
      "additions": payload.get("additions"),
      "deletions": payload.get("deletions"),
      "body_sha256": sha256_bytes(str(payload.get("body", "")).encode()),
      "patch_sha256": sha256_bytes(patch),
      "patch_bytes": len(patch),
  }


def runtime_census(worker: dict[str, Any]) -> dict[str, Any]:
  executed = [
      row for row in worker["full_profile"]
      if row.get("status") == "Status.EXECUTED"
  ]
  counts = Counter(str(row.get("node_type")) for row in executed)
  rms = [row for row in executed if row.get("node_type") == "RMS"]
  return {
      "executed_counts": dict(sorted(counts.items())),
      "rms_count": len(rms),
      "rms_exec_types": dict(sorted(Counter(
          str(row.get("exec_type")) for row in rms).items())),
      "custom_attention_count": counts["IQ36HotAttentionGQA"],
      "stock_attention_counts": {
          name: counts[name]
          for name in ("GroupQueryAttention", "ScaledDotProductAttention",
                       "PagedAttention")
      },
      "gathermatmul_count": counts["GatherMatmul"],
      "broadcast_count": counts["Broadcast"],
  }


def source_atomic_inventory() -> dict[str, Any]:
  needles = ("atomic_", "atomic(", "atomicraw")
  rows = {}
  total = 0
  for path in ATTENTION_SOURCES:
    matches = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1):
      if any(needle in line for needle in needles):
        matches.append({"line": number, "text": line.strip()})
    rows[display(path)] = matches
    total += len(matches)
  return {"total_matches": total, "files": rows}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")

  required = (
      SEQ1233, SEQ1239, SEQ1240, SEQ1299_WORKER, SEQ1301,
      *ATTENTION_SOURCES)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing opportunity-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  seq1233 = load_json(SEQ1233)
  seq1239 = load_json(SEQ1239)
  seq1240 = load_json(SEQ1240)
  worker = load_json(SEQ1299_WORKER)
  seq1301 = load_json(SEQ1301)

  pulls: dict[str, dict[str, Any]] = {}
  raw_files: list[Path] = []
  for name, number in OPENVINO_PULLS.items():
    json_path = raw / f"openvino-pr{number}.json"
    patch_path = raw / f"openvino-pr{number}.patch"
    payload = fetch_json(
        f"https://api.github.com/repos/openvinotoolkit/openvino/pulls/{number}",
        json_path, args.network_timeout_s)
    patch = fetch(
        f"https://github.com/openvinotoolkit/openvino/pull/{number}.patch",
        patch_path, args.network_timeout_s, "application/vnd.github.patch")
    pulls[name] = pull_summary(payload, patch)
    pulls[name]["body"] = str(payload.get("body", ""))
    pulls[name]["patch_text"] = patch.decode("utf-8", errors="replace")
    raw_files.extend((json_path, patch_path))

  latest_release_path = raw / "igc-latest-release.json"
  tag_path = raw / "igc-v2.39.0-tag.json"
  master_path = raw / "igc-master.json"
  latest_release = fetch_json(
      IGC_LATEST_RELEASE, latest_release_path, args.network_timeout_s)
  v239_tag = fetch_json(IGC_V239_TAG, tag_path, args.network_timeout_s)
  igc_master = fetch_json(IGC_MASTER, master_path, args.network_timeout_s)
  raw_files.extend((latest_release_path, tag_path, master_path))

  igc_commits = {}
  for name, commit in IGC_COMMITS.items():
    path = raw / f"igc-{commit}.json"
    payload = fetch_json(
        "https://api.github.com/repos/intel/intel-graphics-compiler/commits/"
        + commit, path, args.network_timeout_s)
    igc_commits[name] = {
        "sha": payload.get("sha"),
        "html_url": payload.get("html_url"),
        "message": payload.get("commit", {}).get("message"),
        "date": payload.get("commit", {}).get("committer", {}).get("date"),
        "files": [row.get("filename") for row in payload.get("files", [])],
    }
    raw_files.append(path)

  census = runtime_census(worker)
  rms_rows = seq1239["locked_ir_rms_audit"]["rows"]
  rms_gamma_exact = (
      len(rms_rows) == 131
      and all(row["direct_reduction_chain"].get("apply->affine") is True
              for row in rms_rows))
  compile_config = worker["compile_config"]
  compiled_model_cache_enabled = any(
      str(key).upper() in {"CACHE_DIR", "CACHE_MODE"}
      for key in compile_config)

  buckets = seq1240["route_selection"][
      "registered_event_buckets_ms_per_token"]
  kill_number_ms = float(
      seq1240["route_selection"]["kill_number_ms_per_token"])
  fixed_fc_saving_ms = float(seq1233["aggregate"]["optimistic_saving_ms"])
  residual_after_fixed_fc_ms = kill_number_ms - fixed_fc_saving_ms
  rms_complete_ceiling_ms = float(buckets["rms"])
  igc_observed_median_ms = float(
      seq1301["performance"]["observed_median_saving_ms"])
  igc_observed_mean_ms = (
      float(seq1301["performance"]["control_mean_ms"])
      - float(seq1301["performance"]["candidate_mean_ms"]))
  favorable_rms_igc_union_ms = rms_complete_ceiling_ms + igc_observed_median_ms
  union_shortfall_ms = residual_after_fixed_fc_ms - favorable_rms_igc_union_ms

  atomic_inventory = source_atomic_inventory()
  decode_codegen = seq1301["codegen"]["igc2382"]["decode"]
  prefill_codegen = seq1301["codegen"]["igc2382"]["prefill"]

  rms_patch = pulls["rms_mvn_generalized"]["patch_text"]
  zero_copy_body = pulls["igpu_device_subbuffer_zero_copy"]["body"]
  kv_body = pulls["kv_broadcast_elimination"]["body"]
  kv_patch = pulls["kv_broadcast_elimination"]["patch_text"]
  no_gamma_body = pulls["rms_without_gamma_fusion"]["body"]
  gather_body = pulls["gathermatmul_decompression_consumer"]["body"]
  sdpa_body = pulls["sdpa_micro_prefetch_bounds"]["body"]

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("registered_bound_inputs_are_exact_and_closed",
            seq1233.get("required_checks_passed") is True
            and seq1233.get("verdict") == "reject_before_graph_integration"
            and seq1239.get("required_checks_passed") is True
            and seq1301.get("required_checks_passed") is True
            and seq1301.get("route_accepted") is False),
      check("official_openvino_pr_identities_refresh_exactly",
            all(pulls[name]["number"] == number
                    and pulls[name]["title"] == EXPECTED_TITLES[number]
                    and pulls[name]["head_sha"]
                    and pulls[name]["patch_bytes"] > 0
                    for name, number in OPENVINO_PULLS.items())),
      check("new_rms_kernel_has_exact_live_locked_consumer",
            census["rms_count"] == 131
            and census["rms_exec_types"] == {"rms_gpu_bfyx_opt__f16": 131}
            and "rms_gpu_bfyx_opt.cl" in rms_patch
            and "On PTL" in rms_patch
            and "Qwen3-Omni" in rms_patch
            and "target_items_per_wi = 8" in rms_patch,
            census=census["rms_exec_types"]),
      check("rms_plus_igc_favorable_union_still_misses_fixed_fc_residual",
            favorable_rms_igc_union_ms < residual_after_fixed_fc_ms,
            residual_after_fixed_fc_ms=residual_after_fixed_fc_ms,
            rms_complete_ceiling_ms=rms_complete_ceiling_ms,
            igc_observed_median_ms=igc_observed_median_ms,
            igc_observed_mean_ms=igc_observed_mean_ms,
            favorable_union_ms=favorable_rms_igc_union_ms,
            shortfall_ms=union_shortfall_ms,
            note=("the IGC median is granted despite no confidence claim and "
                  "the entire RMS bucket is granted as removable")),
      check("zero_copy_subbuffer_change_is_compile_cache_only_here",
            "during cache load" in zero_copy_body
            and "duplicate allocation/copy overhead" in zero_copy_body
            and compiled_model_cache_enabled is False,
            compile_config=compile_config,
            neo_kernel_cache=worker.get("compiler_cache")),
      check("stock_attention_successors_have_no_candidate_attention_consumer",
            census["custom_attention_count"] == 10
            and all(value == 0
                    for value in census["stock_attention_counts"].values())
            and "broadcast" in kv_body.lower()
            and "kvcache_gqa_broadcast_elimination" in kv_patch
            and "sdpa_micro" in pulls["sdpa_micro_prefetch_bounds"]["patch_text"]
            and "prefetch" in sdpa_body.lower(),
            census=census),
      check("no_gamma_rms_and_gathermatmul_have_zero_locked_match",
            rms_gamma_exact
            and "without a learnable gamma" in no_gamma_body
            and census["gathermatmul_count"] == 0
            and "GatherMatmulCompressed" in gather_body),
      check("igc_v2382_remains_latest_release_and_v239_is_source_only",
            latest_release.get("tag_name") == "v2.38.2"
            and v239_tag.get("ref") == "refs/tags/v2.39.0"
            and igc_master.get("sha")
                == "68ef7642f5b2114bc6465b517ae3c9976d570150"
            and all(row["sha"] == IGC_COMMITS[name]
                    for name, row in igc_commits.items()),
            latest_release=latest_release.get("html_url"),
            master_sha=igc_master.get("sha")),
      check("igc_master_changes_are_watch_items_not_current_decode_proof",
            decode_codegen["execution_env"]["spill_size"] == 0
            and prefill_codegen["execution_env"]["spill_size"] > 0
            and atomic_inventory["total_matches"] > 0
            and "atomic" in igc_commits[
                "atomic_loop_runtime_unroll"]["message"].lower()
            and "spill" in igc_commits[
                "post_ra_spill_cleanup"]["message"].lower()
            and "revert" in igc_commits[
                "more_movi_revert"]["message"].lower(),
            decode_spill_bytes=decode_codegen["execution_env"]["spill_size"],
            prefill_spill_bytes=prefill_codegen["execution_env"]["spill_size"],
            source_atomic_matches=atomic_inventory["total_matches"]),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_compiles=0,
            model_workers=0, long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  build_admitted = False
  verdict = (
      "retain_rms_and_igc_release_watch_no_build"
      if required_checks_passed else "inconclusive")
  available_end = available_memory_bytes()

  sanitized_pulls = {}
  for name, row in pulls.items():
    sanitized_pulls[name] = {
        key: value for key, value in row.items()
        if key not in {"body", "patch_text"}
    }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": False,
      "compiler_build_admitted": build_admitted,
      "plugin_build_admitted": build_admitted,
      "gpu_component_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "official_openvino": sanitized_pulls,
      "official_igc": {
          "latest_release": {
              "tag": latest_release.get("tag_name"),
              "published_at": latest_release.get("published_at"),
              "html_url": latest_release.get("html_url"),
          },
          "v2_39_tag": {
              "ref": v239_tag.get("ref"),
              "object": v239_tag.get("object"),
          },
          "master": {
              "sha": igc_master.get("sha"),
              "html_url": igc_master.get("html_url"),
          },
          "watch_commits": igc_commits,
      },
      "locked_runtime": {
          "census": census,
          "rms_all_have_gamma_affine": rms_gamma_exact,
          "compile_config": compile_config,
          "compiled_model_cache_enabled": compiled_model_cache_enabled,
          "attention_source_atomics": atomic_inventory,
      },
      "budget": {
          "current_kill_number_ms": kill_number_ms,
          "seq1233_optimistic_fixed_fc_saving_ms": fixed_fc_saving_ms,
          "residual_after_fixed_fc_ms": residual_after_fixed_fc_ms,
          "complete_registered_rms_bucket_ms": rms_complete_ceiling_ms,
          "seq1301_igc_short_median_point_ms": igc_observed_median_ms,
          "seq1301_igc_short_mean_point_ms": igc_observed_mean_ms,
          "favorable_rms_plus_igc_union_ms": favorable_rms_igc_union_ms,
          "favorable_union_shortfall_ms": union_shortfall_ms,
          "interpretation": (
              "even granting the unconfirmed IGC median as real and deleting "
              "the full RMS bucket, the non-overlapping union remains below "
              "the residual required to fund seq1233 graph integration"),
      },
      "checks": checks,
      "directions": [
          {
              "rank": 1,
              "route": "openvino_pr36747_rms_generalized_bundle",
              "status": "live_consumer_but_below_bundle_cut",
              "next_trigger": (
                  "revisit only after another independent measured cut makes "
                  "the conservative union clear the residual; do not build alone"),
          },
          {
              "rank": 2,
              "route": "official_igc_post_2382_release_watch",
              "status": "source_tag_and_master_only_no_supported_package",
              "next_trigger": (
                  "an official package after the current more-movi revert, "
                  "then exact offline attention codegen before any model worker"),
          },
          {
              "rank": 3,
              "route": "openvino_pr36645_compiled_cache_memory",
              "status": "operational_only_no_steady_decode_cut",
              "next_trigger": (
                  "adopt only if a compiled-model cache becomes part of the "
                  "resident harness and compile-memory/OOM evidence justifies it"),
          },
      ],
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "official_snapshots": {
          display(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
          for path in raw_files
      },
      "compilers": 0,
      "gpu_contexts": 0,
      "model_compiles": 0,
      "model_workers": 0,
      "long_workers": 0,
  })
  report = "\n".join((
      "# Post-IGC upstream opportunity bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`. No build or worker is admitted.",
      "",
      "The strongest fresh live source is OpenVINO PR 36747: all 131 locked "
      "RMS nodes already execute `rms_gpu_bfyx_opt__f16`, and the patch "
      "explicitly adds PTL/Qwen workgroup, vector-IO, single-subgroup, and "
      "register-cache paths. It remains a bundle ingredient, not a standalone "
      "route.",
      "",
      f"Seq1233 leaves `{residual_after_fixed_fc_ms:.6f} ms` after its maximally "
      f"optimistic FC component. Even deleting the complete `{rms_complete_ceiling_ms:.3f}-ms` "
      f"RMS bucket and granting seq1301's unconfirmed `{igc_observed_median_ms:.7f}-ms` "
      f"median point yields only `{favorable_rms_igc_union_ms:.7f} ms`, short by "
      f"`{union_shortfall_ms:.7f} ms`; the IGC mean point is "
      f"`{igc_observed_mean_ms:.7f} ms`. A plugin/model probe is not funded.",
      "",
      "The iGPU usm_device zero-copy change is confined to compiled-model "
      "cache deserialization; the accepted harness sets no OpenVINO CACHE_DIR. "
      "The KV-broadcast and SDPA changes have no stock-attention candidate "
      "consumer, while no-gamma RMS and GatherMatmul have zero locked matches.",
      "",
      "IGC v2.38.2 remains the latest official release. v2.39.0 exists as a "
      "source tag, and master adds atomic-loop unrolling plus prefill-relevant "
      "spill cleanup, but also reverts the more-MOVI path. Keep an official "
      "package watch; do exact offline codegen first when a supported release "
      "appears. No source build is justified now.",
      "",
      f"Available memory stayed `{available_start} -> {available_end}` bytes; "
      "no compiler, GPU context, model compile, model worker, OOM, or restart "
      "occurred.",
      "",
  ))
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "rms_igc_favorable_union_ms": favorable_rms_igc_union_ms,
      "residual_ms": residual_after_fixed_fc_ms,
      "shortfall_ms": union_shortfall_ms,
      "latest_igc_release": latest_release.get("tag_name"),
      "igc_master": igc_master.get("sha"),
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
