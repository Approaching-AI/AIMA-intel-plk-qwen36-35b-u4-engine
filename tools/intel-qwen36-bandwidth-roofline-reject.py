#!/usr/bin/env python3
"""Bandwidth-roofline reject table for batch=1 decode matvec lanes.

Why this tool exists
--------------------
Batch=1 decode reads every weight tensor once per token and does an M=1 GEMV
over it. Arithmetic intensity is ~1 MAC per loaded weight byte, so each lane is
memory-bound: its floor is `weight_bytes / achievable_bandwidth`, not the dot
algorithm. The factory rule (methodology ch.2 trick #1) is to derive that floor
once and let it *reject a whole class of experiments* instead of sweeping
compute-side dot variants (direct / pair / row-pair / min-sum ...).

This tool turns that rule into a one-command judgment:

  measured effective bandwidth (GB/s) = weight_bytes / average_ns
                                      (1 byte/ns == 1 GB/s exactly)

and compares each lane against two R0-measured ceilings on the SAME host/model:
  - qmatvec_achieved: bandwidth an actual R0 qmatvec kernel already reached
  - source_stream:    raw byte-stream readback ceiling (pure move, no compute)

If a lane runs below the qmatvec_achieved ceiling, the win is in the
memory-access path, and compute-side dot variants cannot reach it. The table
prints, per lane, the headroom and the verdict, plus the aggregate ns that a
bandwidth-only fix could recover (the real optimization budget).

Inputs are existing artifacts; nothing runs on the target.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import defaultdict

# GGUF bytes-per-weight, verified against the R0 source-stream probe:
#   attn_gate Q4_K 4096x2048 -> 4718592 bytes  == 4096*2048*0.5625
#   attn_qkv  Q6_K 8192x2048 -> 13762560 bytes == 8192*2048*0.8203125
BYTES_PER_WEIGHT = {
    "Q4_K": 0.5625,       # 144 bytes / 256 weights
    "Q6_K": 0.8203125,    # 210 bytes / 256 weights
    "Q8_0": 1.0625,       # 34  bytes / 32  weights
    "F32": 4.0,
    "F16": 2.0,
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SOURCE_STREAM = os.path.join(
    REPO, "output", "r0-source-stream-roof-20260626T042729Z", "audit.json"
)
DEFAULT_QMATVEC = os.path.join(
    REPO, "output", "r0-qmatvec-probe-20260626T043218Z", "audit.json"
)
DEFAULT_PROFILE = os.path.join(
    REPO,
    "output",
    "post-r1-resident-timed-20260628T054920Z",
    "native-candidate-jsonl",
    "native-candidate-stdout.json",
)


def quant_of(op: str, tensor_name: str) -> str | None:
    """Best-effort quant class from the profile op name and tensor name."""
    op = op or ""
    name = tensor_name or ""
    if "q6" in op:
        return "Q6_K"
    if "q4" in op:
        return "Q4_K"
    # output.weight (lm_head) in Q4_K_M GGUF is Q6_K.
    if name.endswith("output.weight"):
        return "Q6_K"
    # router gate is a small F32 projection in this architecture.
    if "ffn_gate_inp" in name:
        return "F32"
    return None


def lane_of(tensor_name: str) -> str:
    """Collapse blk.<N>.<lane> -> <lane> so per-layer rows aggregate."""
    parts = tensor_name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return ".".join(parts[2:])
    return tensor_name


def parse_source_stream(path: str) -> dict:
    """Return {'Q4_K': gb_s, 'Q6_K': gb_s, 'target_line': gb_s} from R0 audit."""
    d = json.load(open(path))
    out = {"target_line": d["audit"].get("target_gb_s", 115.0)}
    # Parse the per-tensor probe stdout lines for type=... source_gb_s=...
    stdout = d["raw"]["run"]["stdout"]
    for line in stdout.splitlines():
        if "source_gb_s=" not in line:
            continue
        toks = {}
        for kv in line.replace("- ", "").split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                toks[k] = v
        qt = toks.get("type")
        try:
            gb = float(toks.get("source_gb_s"))
        except (TypeError, ValueError):
            continue
        if qt:
            out[qt] = gb
    return out


def parse_qmatvec(path: str) -> dict:
    """Return {'Q4_K': gb_s, 'Q6_K': gb_s} of R0 achieved qmatvec opencl_gb_s."""
    d = json.load(open(path))
    out = {}
    stdout = d["raw"]["run"]["stdout"]
    for line in stdout.splitlines():
        if "opencl_gb_s=" not in line:
            continue
        toks = {}
        for kv in line.replace("- ", "").split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                toks[k] = v
        qt = toks.get("type")
        try:
            gb = float(toks.get("opencl_gb_s"))
        except (TypeError, ValueError):
            continue
        if qt:
            out[qt] = gb
    return out


def load_profile(path: str) -> list:
    d = json.load(open(path))
    if "matvec_profile" in d:
        return d["matvec_profile"]
    # diagnostic.json fallback
    bm = d.get("benchmark_metadata", {})
    if "matvec_profile_top" in bm:
        return bm["matvec_profile_top"]
    raise SystemExit(f"no matvec_profile in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-stream", default=DEFAULT_SOURCE_STREAM)
    ap.add_argument("--qmatvec", default=DEFAULT_QMATVEC)
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--out-dir", default=None,
                    help="output dir; default output/bandwidth-roofline-reject-<UTC>/")
    ap.add_argument(
        "--reject-util",
        type=float,
        default=0.9,
        help="if effective_bw/qmatvec_achieved < this, compute-side dot variants "
        "are rejected for that lane (default 0.9)",
    )
    args = ap.parse_args()

    src = parse_source_stream(args.source_stream)
    qm = parse_qmatvec(args.qmatvec)
    profile = load_profile(args.profile)

    # Aggregate per lane across layers.
    lanes: dict[str, dict] = defaultdict(
        lambda: {
            "total_ns": 0,
            "byte_seconds": 0.0,  # sum(bytes*calls) -> total bytes streamed
            "bytes_per_call": None,
            "quant": None,
            "calls": 0,
            "tensors": 0,
        }
    )

    skipped = []
    for row in profile:
        op = row.get("op", "")
        name = row.get("tensor_name", "")
        cc = row.get("call_count", 0) or 0
        avg = row.get("average_ns", 0) or 0
        rc = row.get("row_count", 0) or 0
        ivc = row.get("input_value_count", 0) or 0
        total_ns = row.get("total_ns", 0) or 0
        if cc <= 0 or avg <= 0 or rc <= 0 or ivc <= 0:
            continue
        quant = quant_of(op, name)
        if quant is None or quant not in BYTES_PER_WEIGHT:
            skipped.append((name, op))
            continue
        out_rows = rc / cc
        in_cols = ivc / cc
        elements = out_rows * in_cols
        bytes_per_call = elements * BYTES_PER_WEIGHT[quant]

        lane = lane_of(name)
        L = lanes[lane]
        L["total_ns"] += total_ns
        L["byte_seconds"] += bytes_per_call * cc
        L["bytes_per_call"] = bytes_per_call
        L["quant"] = quant
        L["calls"] += cc
        L["tensors"] += 1

    # Build per-lane verdicts.
    rows = []
    grand_total_ns = 0
    recoverable_to_qmatvec = 0.0
    recoverable_to_source = 0.0
    for lane, L in lanes.items():
        quant = L["quant"]
        total_ns = L["total_ns"]
        total_bytes = L["byte_seconds"]
        grand_total_ns += total_ns
        # effective bandwidth: total bytes streamed / total ns  (GB/s)
        eff = total_bytes / total_ns if total_ns else 0.0
        qm_ceil = qm.get(quant)
        src_ceil = src.get(quant)
        # ns a bandwidth-only fix could recover (cap improvement at the ceiling).
        ns_at_qm = (total_bytes / qm_ceil) if qm_ceil else None
        ns_at_src = (total_bytes / src_ceil) if src_ceil else None
        rec_qm = max(0.0, total_ns - ns_at_qm) if ns_at_qm else 0.0
        rec_src = max(0.0, total_ns - ns_at_src) if ns_at_src else 0.0
        recoverable_to_qmatvec += rec_qm
        recoverable_to_source += rec_src

        util_vs_qm = eff / qm_ceil if qm_ceil else None
        util_vs_src = eff / src_ceil if src_ceil else None
        if util_vs_qm is not None and util_vs_qm < args.reject_util:
            verdict = "compute-side dot variants REJECTED: below R0 qmatvec kernel; win is in memory-access path"
        elif util_vs_src is not None and util_vs_src < 0.7:
            verdict = "memory-bound headroom remains in bandwidth utilization, not the dot algorithm"
        else:
            verdict = "near bandwidth ceiling; lane is largely saturated"

        rows.append(
            {
                "lane": lane,
                "quant": quant,
                "tensors": L["tensors"],
                "calls": L["calls"],
                "bytes_per_call": round(L["bytes_per_call"]),
                "total_bytes_streamed": round(total_bytes),
                "total_ns": total_ns,
                "effective_gb_s": round(eff, 3),
                "r0_qmatvec_ceiling_gb_s": qm_ceil,
                "r0_source_stream_ceiling_gb_s": src_ceil,
                "util_vs_qmatvec": round(util_vs_qm, 3) if util_vs_qm else None,
                "util_vs_source_stream": round(util_vs_src, 3) if util_vs_src else None,
                "recoverable_ns_to_qmatvec_ceiling": round(rec_qm),
                "recoverable_ns_to_source_ceiling": round(rec_src),
                "verdict": verdict,
            }
        )

    rows.sort(key=lambda r: r["total_ns"], reverse=True)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(
        REPO, "output", f"bandwidth-roofline-reject-{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "schema_version": "intel-qwen36-bandwidth-roofline-reject-v0",
        "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
        "created_at": ts,
        "inputs": {
            "source_stream_audit": os.path.relpath(args.source_stream, REPO),
            "qmatvec_audit": os.path.relpath(args.qmatvec, REPO),
            "profile": os.path.relpath(args.profile, REPO),
        },
        "ceilings_gb_s": {"source_stream": src, "qmatvec_achieved": qm},
        "bytes_per_weight": BYTES_PER_WEIGHT,
        "reject_util_threshold": args.reject_util,
        "covered_total_ns": grand_total_ns,
        "recoverable_ns_to_qmatvec_ceiling": round(recoverable_to_qmatvec),
        "recoverable_ns_to_source_ceiling": round(recoverable_to_source),
        "recoverable_fraction_to_qmatvec": round(recoverable_to_qmatvec / grand_total_ns, 3) if grand_total_ns else None,
        "lanes": rows,
        "skipped_unknown_quant": sorted({lane_of(n) for n, _ in skipped}),
        "speedup_claims_allowed": False,
        "note": "Diagnostic roofline analysis. A floor is not a speed claim. "
        "Recoverable ns assume only bandwidth changes; real matvec cannot exceed "
        "source-stream and rarely reaches it.",
    }
    with open(os.path.join(out_dir, "reject-table.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Markdown table for the reference doc.
    md = []
    md.append("# Bandwidth-roofline reject table (batch=1 decode matvec)\n")
    md.append(f"- profile: `{report['inputs']['profile']}`")
    md.append(f"- R0 qmatvec achieved ceiling: " + ", ".join(f"{k} {v:.1f} GB/s" for k, v in qm.items()))
    md.append(f"- R0 source-stream ceiling: " + ", ".join(f"{k} {v:.1f} GB/s" for k, v in src.items() if k != "target_line"))
    md.append("")
    md.append(f"Covered matvec ns: `{grand_total_ns:,}`. "
              f"Bandwidth-only recoverable to the R0 qmatvec kernel ceiling: "
              f"`{round(recoverable_to_qmatvec):,}` ns "
              f"({report['recoverable_fraction_to_qmatvec']:.0%} of covered).\n")
    md.append("| lane | quant | calls | MB/call | eff GB/s | vs qmatvec | vs source | recover→qmatvec ns | verdict |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|---|")
    for r in rows:
        md.append(
            f"| `{r['lane']}` | {r['quant']} | {r['calls']} | "
            f"{r['bytes_per_call']/1e6:.2f} | {r['effective_gb_s']:.1f} | "
            f"{(str(int(r['util_vs_qmatvec']*100))+'%') if r['util_vs_qmatvec'] is not None else '-'} | "
            f"{(str(int(r['util_vs_source_stream']*100))+'%') if r['util_vs_source_stream'] is not None else '-'} | "
            f"{r['recoverable_ns_to_qmatvec_ceiling']:,} | {r['verdict']} |"
        )
    md.append("")
    with open(os.path.join(out_dir, "reject-table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"wrote {out_dir}/reject-table.json")
    print(f"wrote {out_dir}/reject-table.md")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
