#!/usr/bin/env python3
"""Audit default-off resident queue drain-site profile source wiring.

This is source/generate-only evidence. It verifies that seq155's selected
resident/full-GPU loop profile route now has source attribution for selected
down wait, FFN-tail drain, and attention-front drain sites before any
token-emitting profile row is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-queue-drain-site-profile-source-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ155 = (
    ROOT
    / "output/resident-full-gpu-decode-loop-profile-gate-20260708Tseq155Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-queue-drain-site-profile-generate-only-20260708Tseq156Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-queue-drain-site-profile-source-gate-20260708Tseq156Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _line_of(text: str, pattern: str, *, regex: bool = True) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.S | re.M)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present(text: str, label: str, pattern: str, *,
             regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _absent(text: str, label: str, pattern: str, *,
            regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


def _all_absent(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("absent") is True for row in rows)


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "floor_gap_ms_per_token": max(
          0.0,
          _num(per_token.get("wall"))
          - _num(verdict.get("floor_budget_ms_per_token")),
      ),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _source_checks(text: str, *, generated: bool) -> dict[str, Any]:
  present = [
      _present(
          text,
          "env_gate_present",
          "IQ36_RESIDENT_QUEUE_DRAIN_SITE_PROFILE",
          regex=False,
      ),
      _present(
          text,
          "python_or_cpp_flag_present",
          "resident_queue_drain_site_profile",
          regex=False,
      ),
      _present(
          text,
          "selected_down_wait_drain_counter_present",
          "selected_down_wait_drain_site_wall_ns",
          regex=False,
      ),
      _present(
          text,
          "ffn_tail_drain_counter_present",
          "ffn_tail_drain_site_wall_ns",
          regex=False,
      ),
      _present(
          text,
          "attention_front_drain_counter_present",
          "attention_front_drain_site_wall_ns",
          regex=False,
      ),
      _present(
          text,
          "json_profile_object_present",
          "resident_queue_drain_site_profile_ns",
          regex=False,
      ),
  ]
  if generated:
    present.extend([
        _present(
            text,
            "generated_global_flag_present",
            "bool g_decode_resident_queue_drain_site_profile = false;",
            regex=False,
        ),
        _present(
            text,
            "generated_env_parse_present",
            r"g_decode_resident_queue_drain_site_profile\s*=\s*"
            r"std::getenv\(\"IQ36_RESIDENT_QUEUE_DRAIN_SITE_PROFILE\"\)",
        ),
        _present(
            text,
            "selected_down_wait_attributed_to_selected_timing_wait",
            r"selected_down_wait_drain_site_wall_ns\s*\+=\s*"
            r"selected_gpu\.timing\.down_kernel_wait_wall_ns",
        ),
        _present(
            text,
            "ffn_tail_drain_uses_tail_wall_minus_kernel",
            r"ffn_tail_drain_site_wall_ns\s*\+=.*?"
            r"tail_wall_ns > tail_kernel_ns",
        ),
        _present(
            text,
            "attention_front_drain_uses_wall_minus_kernel",
            r"attention_front_drain_site_wall_ns\s*\+=.*?"
            r"attention_front_wall_ns > attention_front_kernel_ns",
        ),
        _present(
            text,
            "drain_profile_writer_present",
            r"void WriteResidentQueueDrainSiteProfile\(const DecodeStats& stats\)",
        ),
        _present(
            text,
            "drain_profile_stdout_enabled_flag_present",
            "resident_queue_drain_site_profile_enabled",
            regex=False,
        ),
    ])
  else:
    present.extend([
        _present(
            text,
            "python_env_propagates_to_remote_run",
            r"env_parts[\s\S]*?IQ36_RESIDENT_QUEUE_DRAIN_SITE_PROFILE",
        ),
        _present(
            text,
            "python_manifest_records_flag",
            '"resident_queue_drain_site_profile"',
            regex=False,
        ),
        _present(
            text,
            "python_explore_profile_keeps_drain_object",
            '"resident_queue_drain_site_profile_ns"',
            regex=False,
        ),
    ])
  absent = [
      _absent(
          text,
          "no_cli_speed_flag_added",
          "--resident-queue-drain-site-profile",
          regex=False,
      ),
      _absent(
          text,
          "no_speedup_claim_enabled",
          "speedup_claims_allowed\":true",
          regex=False,
      ),
  ]
  return {
      "present_checks": present,
      "absent_checks": absent,
      "present": _all_present(present),
      "absent": _all_absent(absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq155 = _load_json(args.seq155)
  decode_source = args.decode_source.read_text(encoding="utf-8")
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = cpp_path.read_text(encoding="utf-8")
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_checks(decode_source, generated=False)
  generated = _source_checks(generated_cpp, generated=True)
  rejected_names = _rejected_names(rejected)
  manifest_checks = {
      "generate_only": result.get("generate_only") is True,
      "resident_queue_drain_site_profile": (
          result.get("resident_queue_drain_site_profile") is True),
      "decode_tokens_eight": result.get("decode_tokens") == 8,
      "active_frontier_stack_flags_present": (
          result.get("shared_q4_runner") is True
          and result.get("resident_q4_weights") is True
          and result.get("resident_selected_q4_experts") is True
          and result.get("resident_selected_q6_experts") is True
          and result.get("resident_selected_q6_rowstripe") is True
          and result.get("resident_shared_q6_down") is True
          and result.get("resident_full_attention_v_q6") is True
          and result.get("resident_linear_q6_qkv") is True
          and result.get("resident_linear_state") is True
          and result.get("resident_norm_weights") is True
          and result.get("resident_full_core_attention_front_handoff") is True
          and result.get("gpu_router") is True
          and result.get("gpu_lm_head_q6") is True
      ),
      "no_smoke_json": not smoke_path.exists(),
  }

  checks = [
      {
          "name": "seq155_selected_drain_site_source_gate",
          "pass": (
              seq155.get("required_checks_passed") is True
              and seq155.get("selected_next_route")
              == "resident_queue_drain_site_profile_source_gate"
              and _has_switch(
                  routes,
                  "select_resident_queue_drain_site_profile_source_gate",
                  155,
              )
          ),
          "detail": {
              "seq155_disposition": seq155.get("disposition"),
              "seq155_selected_next_route": seq155.get("selected_next_route"),
          },
      },
      {
          "name": "coarse_profile_bucket_is_closed_before_token_row",
          "pass": "current_resident_full_gpu_decode_loop_coarse_profile_bucket"
          in rejected_names,
      },
      {
          "name": "source_drain_site_profile_markers_present",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_drain_site_profile_markers_present",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq155_profile_gate": _rel(args.seq155),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": frontier_state,
      "source": source,
      "generated": generated,
      "generate_manifest_checks": manifest_checks,
      "checks": checks,
      "required_checks_passed": required,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_resident_queue_drain_site_profile_source_wiring"
          if required else
          "reject_resident_queue_drain_site_profile_source_wiring"
      ),
      "selected_next_route": (
          "resident_queue_drain_site_profile_target_compile_gate"
          if required else
          "resident_queue_drain_site_profile_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off source attribution now records selected-down wait, "
          "FFN-tail drain, and attention-front drain buckets separately. The "
          "next admissible unit is target compile only; the first token row "
          "after compile is an artifact-free profile explore."
          if required else
          "The drain-site source profile contract is incomplete; fix source "
          "and generate-only evidence before compile or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  summary = [
      "# Resident Queue Drain-Site Profile Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(metrics['target_compile_required_before_decode']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq155", type=Path, default=DEFAULT_SEQ155)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
