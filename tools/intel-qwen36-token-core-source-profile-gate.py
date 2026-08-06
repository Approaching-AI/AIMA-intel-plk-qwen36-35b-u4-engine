#!/usr/bin/env python3
"""Gate token-core source-profile rows and route the next setup cut.

This is route-control evidence only. It consumes artifact-free explore rows
from the RunGpuHybridDecodeToken source-profile path and rejects the generic
token setup cache if it does not beat the frontier. It does not claim speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-token-core-source-profile-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_EXPLORE = ROOT / "output/explore-log.jsonl"
DEFAULT_SOURCE_LABEL = "token-core-dispatch-gap-source-profile-seq99"
DEFAULT_CACHE_LABEL = "token-setup-cache-seq100"
DEFAULT_DEFAULT_OFF_LABEL = "token-source-profile-default-off-seq101"
DEFAULT_OUT_DIR = ROOT / "output/token-core-source-profile-gate-20260707Tseq99-101Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _find_explore(path: Path, label: str) -> dict[str, Any]:
  found: dict[str, Any] | None = None
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("label") == label:
      found = row
  if found is None:
    raise SystemExit(f"explore label not found: {label}")
  return found


def _profile(row: dict[str, Any], key: str) -> dict[str, Any]:
  value = row.get(key)
  return value if isinstance(value, dict) else {}


def _largest_bucket(profile: dict[str, Any]) -> tuple[str, float]:
  buckets = {
      key: _num(value)
      for key, value in profile.items()
      if key != "profiled" and isinstance(value, (int, float))
  }
  if not buckets:
    return "", 0.0
  key = max(buckets, key=buckets.get)
  return key, buckets[key]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  current_best_tps = _num(goal_anchor.get("current_best_tps"))
  floor_tps = _num(goal_anchor.get("same_host_vulkan_floor_tps"))
  noise_rel = _num(noise.get("rel"))

  source = _find_explore(args.explore_log, args.source_label)
  cache = _find_explore(args.explore_log, args.cache_label)
  default_off = _find_explore(args.explore_log, args.default_off_label)

  token_core = _profile(source, "token_core_wall_profile_ns")
  dispatch = _profile(source, "dispatch_gap_source_profile_ns")
  largest_dispatch_key, largest_dispatch_ns = _largest_bucket(dispatch)
  token_core_unprofiled_ns = _num(source.get("token_core_unprofiled_ns"))
  source_tokens = int(_num(source.get("decode_tokens")))
  cache_tps = _num(cache.get("tps"))
  default_off_tps = _num(default_off.get("tps"))
  cache_rel_vs_best = (
      (cache_tps - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )
  default_off_rel_vs_best = (
      (default_off_tps - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )

  checks = [
      {
          "name": "source_profile_row_preserved_top1",
          "pass": (
              source.get("label") == args.source_label
              and source.get("top1_matches_native") is True
              and source.get("token_core_source_profile") is True
              and source_tokens == 8
          ),
      },
      {
          "name": "source_profile_accounts_for_prior_unprofiled_bucket",
          "pass": (
              token_core_unprofiled_ns <= 50_000
              and _num(token_core.get("profiled")) > 500_000
              and _num(dispatch.get("profiled")) > 3_000_000
          ),
          "detail": {
              "token_core_unprofiled_ns": token_core_unprofiled_ns,
              "token_core_profiled_ns": _num(token_core.get("profiled")),
              "dispatch_gap_source_profiled_ns": _num(dispatch.get("profiled")),
          },
      },
      {
          "name": "largest_dispatch_bucket_is_linear_setup",
          "pass": largest_dispatch_key == "linear_setup"
          and largest_dispatch_ns > _num(dispatch.get("ffn_tail_setup")),
          "detail": {
              "largest_bucket": largest_dispatch_key,
              "largest_bucket_ns": largest_dispatch_ns,
              "dispatch_gap_source_profile_ns": dispatch,
          },
      },
      {
          "name": "generic_token_setup_cache_rejected_as_speed_cut",
          "pass": (
              cache.get("label") == args.cache_label
              and cache.get("top1_matches_native") is True
              and cache.get("token_setup_cache") is True
              and cache_tps < current_best_tps
              and cache_tps < floor_tps
          ),
          "detail": {
              "cache_tps": cache_tps,
              "current_best_tps": current_best_tps,
              "floor_tps": floor_tps,
              "relative_vs_best": cache_rel_vs_best,
              "noise_rel": noise_rel,
          },
      },
      {
          "name": "default_off_profile_scaffold_is_not_a_speed_candidate",
          "pass": (
              default_off.get("label") == args.default_off_label
              and default_off.get("top1_matches_native") is True
              and default_off_tps < current_best_tps
          ),
          "detail": {
              "default_off_tps": default_off_tps,
              "relative_vs_best": default_off_rel_vs_best,
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_source_profile_reject_generic_setup_cache"
          if required else "token_core_source_profile_gate_failed"
      ),
      "selected_next_route": (
          "linear_setup_specialized_hoist_source_gate"
          if required else "manual_review_token_core_source_profile"
      ),
      "next_action": (
          "Do not repeat the generic unordered-map token setup cache. The "
          "source-profile path accounts for the prior token-core unprofiled "
          "bucket and points to fixed per-layer setup hoisting, led by "
          "linear setup, then FFN-tail setup and full-attention setup. The "
          "next cut must specialize/hoist fixed metadata and row vectors out "
          "of the decode token hot path without per-token string/map lookup."
          if required else "Fix failed gate checks before changing route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "explore_log": _rel(args.explore_log),
          "source_label": args.source_label,
          "cache_label": args.cache_label,
          "default_off_label": args.default_off_label,
      },
      "frontier": {
          "current_best_tps": current_best_tps,
          "floor_tps": floor_tps,
          "noise_rel": noise_rel,
      },
      "source_profile": {
          "ts": source.get("ts"),
          "label": source.get("label"),
          "source_sha": source.get("source_sha"),
          "tps": source.get("tps"),
          "top1_matches_native": source.get("top1_matches_native"),
          "decode_ns": source.get("decode_ns"),
          "token_core_unprofiled_ns": source.get("token_core_unprofiled_ns"),
          "token_core_wall_profile_ns": token_core,
          "dispatch_gap_source_profile_ns": dispatch,
          "largest_dispatch_bucket": largest_dispatch_key,
          "largest_dispatch_bucket_ns": largest_dispatch_ns,
      },
      "generic_cache_explore": {
          "ts": cache.get("ts"),
          "label": cache.get("label"),
          "source_sha": cache.get("source_sha"),
          "tps": cache_tps,
          "top1_matches_native": cache.get("top1_matches_native"),
          "token_setup_cache": cache.get("token_setup_cache"),
          "relative_vs_best": cache_rel_vs_best,
      },
      "default_off_explore": {
          "ts": default_off.get("ts"),
          "label": default_off.get("label"),
          "source_sha": default_off.get("source_sha"),
          "tps": default_off_tps,
          "top1_matches_native": default_off.get("top1_matches_native"),
          "relative_vs_best": default_off_rel_vs_best,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-token-core-source-profile-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  source = payload["source_profile"]
  dispatch = source["dispatch_gap_source_profile_ns"]
  cache = payload["generic_cache_explore"]
  lines = [
      "# Token-Core Source Profile Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- source-profile tps: `{_num(source['tps']):.8f}`",
      f"- token-core unprofiled after profile: `{source['token_core_unprofiled_ns']}` ns",
      f"- dispatch source profiled: `{dispatch.get('profiled')}` ns",
      f"- largest dispatch bucket: `{source['largest_dispatch_bucket']}` = `{source['largest_dispatch_bucket_ns']}` ns",
      f"- linear setup: `{dispatch.get('linear_setup')}` ns",
      f"- FFN-tail setup: `{dispatch.get('ffn_tail_setup')}` ns",
      f"- full-attention setup+core prep: `{_num(dispatch.get('full_attention_setup')) + _num(dispatch.get('full_attention_core_prep'))}` ns",
      f"- generic setup-cache tps: `{cache['tps']:.8f}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE)
  parser.add_argument("--source-label", default=DEFAULT_SOURCE_LABEL)
  parser.add_argument("--cache-label", default=DEFAULT_CACHE_LABEL)
  parser.add_argument("--default-off-label", default=DEFAULT_DEFAULT_OFF_LABEL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
