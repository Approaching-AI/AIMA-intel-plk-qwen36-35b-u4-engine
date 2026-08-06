#!/usr/bin/env python3
"""Run and classify the paired exact-preprojection component probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


PROBE_SOURCE = (
    ROOT / "tools/intel-qwen36-all-linear-preprojection-parity-component-harness.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-probe-gate-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_probe_gate"
)
SUCCESS_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_decode_source_gate"
)
REJECT_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_route_close_gate"
)
FINAL_CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_final_probe_gate"
)
FINAL_REJECT_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_final_route_close_gate"
)
ADDED_US_MAX = 6.841858993929781
EXACT_COMPARISONS = (
    "exact_qkv_vs_cpu",
    "exact_conv_output_vs_cpu",
    "exact_conv_state_vs_cpu",
    "exact_attention_vs_cpu",
    "exact_final_vs_cpu",
    "exact_recurrent_state_vs_cpu",
    "exact_projection_vs_cpu",
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


PROBE = _load_module(PROBE_SOURCE, "iq36_exact_preprojection_probe_gate_source")


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _nested(obj: dict[str, Any], *keys: str) -> Any:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return current


def _num(obj: dict[str, Any], *keys: str) -> float | None:
  value = _nested(obj, *keys)
  return float(value) if isinstance(value, (int, float)) else None


def _shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def _parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def _comparison_exact(probe: dict[str, Any], name: str) -> bool:
  return (
      _num(probe, "comparisons", name, "max_abs_diff") == 0.0
      and _num(probe, "comparisons", name, "rmse") == 0.0
      and _num(probe, "comparisons", name, "mismatch_count") == 0.0
      and _nested(probe, "comparisons", name, "same_size") is True
      and _nested(probe, "comparisons", name, "finite") is True)


def _run_row(probe: dict[str, Any] | None, label: str) -> dict[str, Any]:
  probe = probe if isinstance(probe, dict) else {}
  candidate_us = _num(probe, "timings", "exact_changed_shell_min_us")
  baseline_us = _num(probe, "timings", "current_changed_shell_min_us")
  added_us = _num(probe, "timings", "candidate_added_min_us")
  exactness = {
      name: _comparison_exact(probe, name) for name in EXACT_COMPARISONS
  }
  exact = (
      all(exactness.values())
      and _nested(probe, "checks", "capture_oracles_bit_exact") is True
      and _nested(probe, "checks", "exact_component_bit_exact") is True)
  budget = (
      candidate_us is not None and candidate_us > 0.0
      and baseline_us is not None and baseline_us > 0.0
      and added_us is not None and added_us <= ADDED_US_MAX)
  return {
      "label": label,
      "schema_version": probe.get("schema_version"),
      "device_name": probe.get("device_name"),
      "layer": probe.get("layer"),
      "samples": probe.get("samples"),
      "probe_required_checks_passed": probe.get("required_checks_passed"),
      "exact_comparisons": exactness,
      "current_changed_shell_min_us": baseline_us,
      "candidate_changed_shell_min_us": candidate_us,
      "candidate_added_min_us": added_us,
      "candidate_added_us_max": ADDED_US_MAX,
      "exactness_passed": exact,
      "budget_passed": budget,
      "passed": exact and budget,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  compile_summary = predecessor.get("compile_summary", {})
  binary = str(compile_summary.get("binary", ""))
  payloads = PROBE.resolve_payloads(args.layer)
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/"
      f"seq{args.sequence}-exact-preprojection-component-probe"
  )
  remote_payload_dir = f"{remote_dir}/oracle"
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + shlex.quote(remote_payload_dir),
      args.timeout_s,
  )
  transfers: dict[str, dict[str, Any]] = {}
  if setup.get("returncode") == 0:
    for name, payload in payloads.items():
      transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )
  stage_ok = (
      setup.get("returncode") == 0
      and len(transfers) == len(payloads)
      and all(row.get("returncode") == 0 for row in transfers.values()))
  run_argv = [
      binary,
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--samples", str(args.samples),
      "--device-substring", args.device_substring,
  ]
  run_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      _shell_join(run_argv),
  ])
  run_results: list[dict[str, Any]] = []
  probes: list[dict[str, Any] | None] = []
  if stage_ok and binary:
    for _ in ("repeat", "confirm"):
      result = iq36_local.run_target(args.host, run_command, args.timeout_s)
      run_results.append(result)
      probes.append(_parse_probe_stdout(str(result.get("stdout", ""))))
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(
      raw_dir / "repeat-run.json", run_results[0] if run_results else {})
  iq36_local.write_json(
      raw_dir / "confirm-run.json",
      run_results[1] if len(run_results) > 1 else {})
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)
  if probes and probes[0] is not None:
    iq36_local.write_json(args.out_dir / "repeat-probe.json", probes[0])
  if len(probes) > 1 and probes[1] is not None:
    iq36_local.write_json(args.out_dir / "confirm-probe.json", probes[1])

  rows = [
      _run_row(probes[index] if index < len(probes) else None, label)
      for index, label in enumerate(("repeat", "confirm"))
  ]
  current_route = FINAL_CURRENT_ROUTE if args.final else CURRENT_ROUTE
  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("component_probe_allowed") is True
      and predecessor.get("component_repeat_and_confirm_required") is True
      and predecessor.get("decode_integration_allowed") is False
      and predecessor.get("token_row_allowed") is False
      and predecessor.get("selected_next_route") == current_route
      and compile_summary.get("ok") is True
      and compile_summary.get("key") == args.expected_binary_key
      and compile_summary.get("binary_sha256") == args.expected_binary_sha256
      and _has_candidate(routes, 603 if args.final else 599, current_route)
      and _has_switch(
          routes, 603 if args.final else 599,
          "select_router_prompt_distribution_all_linear_preprojection_"
          + ("parity_component_final_probe_gate"
             if args.final else "parity_component_probe_gate")))
  execution_complete = (
      stage_ok
      and len(run_results) == 2
      and all(result.get("returncode") in (0, 2) for result in run_results)
      and len(probes) == 2
      and all(isinstance(probe, dict) for probe in probes)
      and all(
          row.get("schema_version") == PROBE.SCHEMA_VERSION for row in rows)
      and all("Arc(TM) B390" in str(row.get("device_name")) for row in rows)
      and all(row.get("layer") == args.layer for row in rows)
      and all(row.get("samples") == args.samples for row in rows))
  component_passes = execution_complete and all(row["passed"] for row in rows)
  cleanup_passes = cleanup.get("returncode") == 0
  required = route_selects and component_passes and cleanup_passes
  measurement_complete = route_selects and execution_complete and cleanup_passes
  checks = [
      {"name": (
           "seq603_selected_one_final_paired_component_probe"
           if args.final else "seq599_selected_one_paired_component_probe"),
       "pass": route_selects},
      {"name": "captured_layer0_payloads_staged_once",
       "pass": stage_ok,
       "detail": {
           name: {
               "path": payload["path"],
               "sha256": payload["sha256"],
               "size_bytes": payload["size_bytes"],
           } for name, payload in payloads.items()
       }},
      {"name": "repeat_and_confirm_completed_on_arc_b390",
       "pass": execution_complete},
      {"name": "both_rows_are_bit_exact_and_within_whole_shell_budget",
       "pass": component_passes, "detail": rows},
      {"name": "remote_probe_payloads_cleaned", "pass": cleanup_passes},
  ]
  if required:
    disposition = (
        "accept_repaired_cpuorder_preprojection_bundle_v1_component"
        if args.final else "accept_cpuorder_preprojection_bundle_v1_component")
    selected_next = SUCCESS_NEXT_ROUTE
    reason = (
        "Both representative-layer rows are bit-exact at every locked "
        "boundary and satisfy the +6.841858994 us changed-shell ruler. Add "
        "only a default-off decode selector for the locked all-linear bundle; "
        "compile gates still precede any token.")
  elif measurement_complete:
    disposition = (
        "close_repaired_cpuorder_preprojection_bundle_v1_component"
        if args.final else "reject_cpuorder_preprojection_bundle_v1_component")
    selected_next = FINAL_REJECT_NEXT_ROUTE if args.final else REJECT_NEXT_ROUTE
    reason = (
        "The paired component evidence is complete, but exactness or the "
        "+6.841858994 us changed-shell ruler failed. Close the whole bundle "
        "before decode and preserve the component diagnostics."
        + (" The one authorized repair is exhausted; no further arithmetic "
           "repair or component rerun is allowed." if args.final else ""))
  else:
    disposition = "block_incomplete_exact_preprojection_component_probe"
    selected_next = CURRENT_ROUTE
    reason = (
        "The paired component evidence is incomplete. Repair staging, target "
        "execution, parsing, or cleanup without changing the locked bundle.")
  slim_payloads = {
      name: {key: value for key, value in payload.items()
             if key != "local_path"}
      for name, payload in payloads.items()
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "host": args.host,
          "model": args.model,
          "binary": binary,
          "binary_key": compile_summary.get("key"),
          "binary_sha256": compile_summary.get("binary_sha256"),
          "payloads": slim_payloads,
          "layer": args.layer,
          "samples": args.samples,
      },
      "rows": rows,
      "checks": checks,
      "measurement_complete": measurement_complete,
      "required_checks_passed": required,
      "component_passed": required,
      "decode_source_allowed": required,
      "decode_probe_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next,
      "next_route_reason": reason,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "measurement_complete": metrics["measurement_complete"],
          "component_passed": metrics["component_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "decode_probe_allowed": False,
          "token_row_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact Preprojection Component Probe",
      "",
      f"- measurement_complete: `{str(metrics['measurement_complete']).lower()}`",
      f"- component_passed: `{str(metrics['component_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
  ]
  for row in metrics["rows"]:
    lines.append(
        f"- {row['label']}: current/candidate/added changed shell "
        f"`{row['current_changed_shell_min_us']}` / "
        f"`{row['candidate_changed_shell_min_us']}` / "
        f"`{row['candidate_added_min_us']}` us; exact "
        f"`{str(row['exactness_passed']).lower()}`; passed "
        f"`{str(row['passed']).lower()}`")
  lines += [
      "",
      metrics["next_route_reason"],
      "",
      "This is paired component evidence only. No decode or token was run.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=600)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq599-all-linear-preprojection-parity-component-target-"
          "compile-gate-20260710Tseq599Z/metrics.json"))
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq600-all-linear-preprojection-parity-component-probe-"
          "gate-20260710Tseq600Z"))
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=PROBE.DEFAULT_HOST)
  parser.add_argument("--model", default=PROBE.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=PROBE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=PROBE.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--layer", type=int, default=0)
  parser.add_argument("--samples", type=int, default=11)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--expected-binary-key",
                      default="80a3f0b198d6ba69a64febec")
  parser.add_argument(
      "--expected-binary-sha256",
      default="ef1b648d49a5465faef7b20f25293cbe7d75b96bd6df45d13558062ad843932c")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--final", action="store_true")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "measurement_complete": metrics["measurement_complete"],
      "component_passed": metrics["component_passed"],
      "disposition": metrics["disposition"],
      "rows": metrics["rows"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["measurement_complete"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
