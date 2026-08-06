#!/usr/bin/env python3
"""Gate the direct selected/shared Q6 down-to-tail fusion wiring surface.

Seq66 closed hidden-row serial fusion and seq68 rejected atomic reduction after
the Q6 down outputs are already materialized. The only remaining in-family
route has to contribute from the Q6 down work-items directly into the final
tail output. This gate records whether the current source has that contract, and
if not, the exact missing pieces.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
OUT_DIR = ROOT / "output/direct-down-tail-wiring-gate-20260706Tseq69Z"


def require_contains(text: str, needle: str) -> bool:
    return needle in text


def main() -> int:
    engine_cl = (ROOT / "engine/gpu/opencl/q4x8_matvec.cl").read_text(
        encoding="utf-8"
    )
    engine_hpp = (
        ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
    ).read_text(encoding="utf-8")
    engine_cpp = (ROOT / "engine/src/gpu_q4x8_matvec.cpp").read_text(
        encoding="utf-8"
    )
    decode = (ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py").read_text(
        encoding="utf-8"
    )

    direct_kernel_name = (
        "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_tail_atomic_raw"
    )
    direct_runner_name = "RunResidentRawQ6KExpert8PlusSharedToFfnTailAtomic"
    selected_shell_marker = "SelectedFfnRun RunGpuSelectedFfnShell("

    current_q6_materializes_outputs = (
        "q6k_selected_down_matvec_rowstripe_expert8_plus_shared_raw" in engine_cl
        and "selected_out[flat_row] = sum;" in engine_cl
        and "shared_out[row] = sum;" in engine_cl
    )
    post_down_atomic_present = (
        "RunFfnTailAtomicFromDownHandlesResidentInputs" in engine_hpp
        and "IQ36_FFN_TAIL_ATOMIC_REDUCTION" in decode
    )
    direct_kernel_present = direct_kernel_name in engine_cl
    direct_runner_present = (
        direct_runner_name in engine_hpp and direct_runner_name in engine_cpp
    )

    selected_shell_start = decode.find(selected_shell_marker)
    selected_shell_window = (
        decode[selected_shell_start : selected_shell_start + 2200]
        if selected_shell_start >= 0
        else ""
    )
    selected_shell_has_direct_inputs = all(
        token in selected_shell_window
        for token in (
            "shared_input_gate_tensor",
            "ffn_input_handle",
            "attention_residual_handle",
        )
    )
    selected_shell_calls_direct_runner = direct_runner_name in decode
    direct_route_has_env_gate = "IQ36_SELECTED_SHARED_Q6_DOWN_TAIL_DIRECT" in decode
    decode_path_bypasses_materialized_shared_tail = all(
        token in decode
        for token in (
            "direct_down_tail_fused",
            "selected_gpu.direct_layer_output_handle",
            "return std::move(selected_gpu.direct_layer_output)",
        )
    )

    missing = []
    if not direct_kernel_present:
        missing.append(
            "opencl_direct_q6_down_tail_kernel_contributing_from_rows_per_expert_times_9"
        )
    if not direct_runner_present:
        missing.append("engine_runner_for_direct_q6_down_tail_atomic_output")
    if not selected_shell_has_direct_inputs:
        missing.append(
            "selected_shell_signature_inputs_for_shared_gate_and_residual_handles"
        )
    if not selected_shell_calls_direct_runner:
        missing.append("decode_path_calling_direct_runner_before_down_outputs")
    if not direct_route_has_env_gate:
        missing.append("explicit_env_gate_for_direct_q6_down_tail")
    if not decode_path_bypasses_materialized_shared_tail:
        missing.append("decode_path_bypassing_shared_shell_and_tail_after_direct_output")

    passed = not missing
    verdict = (
        "direct_down_tail_contract_present"
        if passed
        else "direct_down_tail_contract_missing"
    )

    metrics = {
        "schema_version": "intel-qwen36-direct-down-tail-wiring-gate-v0",
        "workstream": WORKSTREAM,
        "direct_down_tail_required_checks_passed": passed,
        "verdict": verdict,
        "current_q6_materializes_outputs": current_q6_materializes_outputs,
        "post_down_atomic_present_rejected_seq68": post_down_atomic_present,
        "direct_kernel_present": direct_kernel_present,
        "direct_runner_present": direct_runner_present,
        "selected_shell_has_direct_inputs": selected_shell_has_direct_inputs,
        "selected_shell_calls_direct_runner": selected_shell_calls_direct_runner,
        "direct_route_has_env_gate": direct_route_has_env_gate,
        "decode_path_bypasses_materialized_shared_tail": (
            decode_path_bypasses_materialized_shared_tail
        ),
        "missing_contracts": missing,
        "required_edits": [
            (
                "Add an OpenCL Q6 selected+shared rowstripe kernel that keeps "
                "rows_per_expert*9 work-items and atomically contributes "
                "selected_weighted/shared_gated values into residual-initialized "
                "layer_output bits, without writing selected_out/shared_out."
            ),
            (
                "Add a runner that computes the shared input gate from resident "
                "ffn_norm, initializes the residual output, launches the direct "
                "Q6 contribution kernel, and returns a layer_output handle."
            ),
            (
                "Extend the selected FFN shell or add a sibling path so it can "
                "see shared_input_gate_tensor, ffn_input_handle, and "
                "attention_residual_handle before current selected/shared Q6 "
                "down materialization."
            ),
            (
                "Keep IQ36_FFN_TAIL_ATOMIC_REDUCTION closed for speed work; it "
                "is post-down reduction over materialized buffers and is the "
                "seq68 rejected class."
            ),
        ],
        "speedup_claims_allowed": False,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = [
        "# Direct Down-Tail Wiring Gate",
        "",
        f"- workstream: `{WORKSTREAM}`",
        f"- required checks passed: `{str(passed).lower()}`",
        f"- verdict: `{verdict}`",
        f"- missing contracts: `{len(missing)}`",
        "",
        "This is a route-contract gate only, not speed evidence.",
        "",
    ]
    for item in missing:
        summary.append(f"- missing: `{item}`")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
