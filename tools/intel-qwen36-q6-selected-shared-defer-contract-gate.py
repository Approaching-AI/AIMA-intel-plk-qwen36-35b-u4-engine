#!/usr/bin/env python3
"""Gate Q6 selected+shared deferred-finish carrier plumbing.

This is source-contract evidence, not runtime speed evidence. It proves the
combined selected+shared Q6 down path can keep both outputs resident and, only
under the existing finish-bundle env gate, stage Q8 host uploads and skip the
down-kernel clFinish. Seq53 already showed finish deferral alone only moves the
drain into FFN-tail, so this gate is admissible only as carrier plumbing after
the seq61 resident-input tail primitive.
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
SCHEMA_VERSION = "intel-qwen36-q6-selected-shared-defer-contract-gate-v0"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_SEQ61 = (
    ROOT
    / "output/ffn-tail-resident-input-contract-gate-20260706Tseq61Z/metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/q6-selected-shared-defer-contract-gate-20260706Tseq62Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(text: str, pattern: str, label: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  return {
      "label": label,
      "present": match is not None,
      "line": text.count("\n", 0, match.start()) + 1 if match else None,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  source = args.gpu_source.read_text(encoding="utf-8")
  seq61 = _load_json(args.seq61_metrics)
  checks = [
      _check(
          source,
          r"RunResidentRawQ6KExpert8PlusShared\([^)]*"
          r"bool readback_selected_output = true,\s*"
          r"bool readback_shared_output = true\)",
          "public_q6_selected_shared_defaults_to_readback",
      ),
      _check(
          source,
          r"RunResidentRawQ6KExpert8PlusShared\([^)]*\)\s*\{.*?"
          r"const bool defer_finish =\s*!readback_selected_output && "
          r"!readback_shared_output &&\s*DeferFfnDownFinishBundle\(\)",
          "q6_defer_requires_both_outputs_resident_and_env_gate",
      ),
      _check(
          source,
          r"if \(defer_finish\) \{\s*"
          r"pending_host_uploads_\.reserve\(pending_host_uploads_\.size\(\) \+ 4U\)",
          "q6_defer_reserves_four_staged_q8_uploads",
      ),
      _check(
          source,
          r"const void\* selected_q8_qs_data =\s*defer_finish\s*\?"
          r"\s*StagePendingHostUpload\(selected_q8\.qs\.data\(\)",
          "q6_defer_stages_selected_q8_qs",
      ),
      _check(
          source,
          r"const void\* selected_q8_d_data =\s*defer_finish\s*\?"
          r"\s*StagePendingHostUpload\(selected_q8\.d\.data\(\)",
          "q6_defer_stages_selected_q8_d",
      ),
      _check(
          source,
          r"const void\* shared_q8_qs_data =\s*defer_finish\s*\?"
          r"\s*StagePendingHostUpload\(shared_q8\.qs\.data\(\)",
          "q6_defer_stages_shared_q8_qs",
      ),
      _check(
          source,
          r"const void\* shared_q8_d_data =\s*defer_finish\s*\?"
          r"\s*StagePendingHostUpload\(shared_q8\.d\.data\(\)",
          "q6_defer_stages_shared_q8_d",
      ),
      _check(
          source,
          r"RunQ6KExpert8RowstripePlusSharedKernel\([^;]*"
          r"repeat, defer_finish\)",
          "q6_defer_flag_reaches_kernel_runner",
      ),
      _check(
          source,
          r"RunQ6KExpert8RowstripePlusSharedKernel\([^)]*"
          r"bool defer_finish = false\)",
          "q6_kernel_runner_defaults_to_finish",
      ),
      _check(
          source,
          r"clEnqueueNDRangeKernel\([^;]*"
          r"q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw"
          r"[^;]*\);\s*if \(!defer_finish\) \{\s*"
          r"Check\(api_\.clFinish\(queue_\)",
          "q6_kernel_finish_guarded_by_defer_flag",
      ),
      _check(
          source,
          r"if \(!defer_finish\) \{\s*"
          r"ClearPendingHostUploadsAfterQueueDrain\(\);\s*\}",
          "q6_pending_uploads_clear_only_after_queue_drain",
      ),
  ]
  all_checks_pass = all(check["present"] for check in checks)
  seq61_ready = bool(
      seq61.get("derived", {}).get("primitive_ready_for_carrier_wiring")
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "gpu_source": {
              "path": _display_path(args.gpu_source),
              "sha256": _sha256(args.gpu_source),
          },
          "seq61_metrics": {
              "path": _display_path(args.seq61_metrics),
              "sha256": _sha256(args.seq61_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "seq61_resident_tail_ready": seq61_ready,
          "default_behavior_preserved": all_checks_pass,
          "primitive_ready_for_carrier_wiring": all_checks_pass and seq61_ready,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Q6 selected+shared down can now defer its finish only when both "
              "outputs remain resident and the existing finish-bundle gate is "
              "active; staged Q8 uploads stay alive until a later queue drain."
          ),
          "next_route": (
              "Do not use this as a standalone finish-deferral probe. Pair it "
              "with resident tail/residual ownership or down-to-tail fusion so "
              "the seq53 drain shift is removed rather than relocated."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "tool": "tools/intel-qwen36-q6-selected-shared-defer-contract-gate.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  d = result["derived"]
  lines = [
      "# Q6 Selected+Shared Defer Contract Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- seq61 resident tail ready: `{str(d['seq61_resident_tail_ready']).lower()}`",
      f"- default behavior preserved: `{str(d['default_behavior_preserved']).lower()}`",
      f"- primitive ready for carrier wiring: `{str(d['primitive_ready_for_carrier_wiring']).lower()}`",
      "",
      "## Verdict",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gpu-source", type=Path, default=DEFAULT_GPU_SOURCE)
  parser.add_argument("--seq61-metrics", type=Path, default=DEFAULT_SEQ61)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  write_outputs(result, args.out_dir)
  derived = result["derived"]
  print("Q6 selected+shared defer contract gate")
  print(f"all_contract_checks_pass={derived['all_contract_checks_pass']}")
  print(f"default_behavior_preserved={derived['default_behavior_preserved']}")
  print(
      "primitive_ready_for_carrier_wiring="
      f"{derived['primitive_ready_for_carrier_wiring']}"
  )
  print(f"artifact={_display_path(args.out_dir)}")
  if not derived["primitive_ready_for_carrier_wiring"]:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
