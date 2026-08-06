#!/usr/bin/env python3
"""R2 speed denominator: put native timing on the floor + roofline judges.

The post-R1 perf campaign measured only self-relative ns ("faster than my last
version"). The factory build order requires R2 to provide a *speed denominator*
so every measurement answers the two questions that matter:

  floor   = same-host reference engine tok/s (llama.cpp / OpenVINO bootstrap)
  ceiling = measured roofline tok/s per bucket (R0 kv-read-pressure)
  bar     = roofline_default_target_ratio * ceiling (default 70%)

This tool converts a native post-R1 timed diagnostic into prefill/decode tok/s
and reports, per case, the ratio to floor and the utilization of the roofline.

IMPORTANT: the current post-R1 cases use short prompts (13-39 tokens), so this
is a *sub-1k smoke alignment*, compared against the 1k bucket as the most
optimistic reference (smaller context decodes faster, so a native value below
the 1k floor is a genuine lower-bound gap). The real R2 denominator must run the
full 1k-8k x 512-output matrix on the target; see
doc/sop/intel-qwen36-35b-a3b-gguf-q4km/r2-speed-denominator.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DIAGNOSTIC = os.path.join(
    REPO, "output", "post-r1-resident-timed-20260628T054920Z", "diagnostic.json"
)
DEFAULT_ACCEPTANCE = os.path.join(
    REPO, "benchmarks", "intel-qwen36-35b-a3b-gguf-q4km", "acceptance-matrix.json"
)
DEFAULT_KV_PRESSURE = os.path.join(
    REPO, "output", "r0-kv-read-pressure-20260626T043722Z", "budget.json"
)


def ns_to_s(ns: float) -> float:
    return ns / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostic", default=DEFAULT_DIAGNOSTIC)
    ap.add_argument("--acceptance", default=DEFAULT_ACCEPTANCE)
    ap.add_argument("--kv-pressure", default=DEFAULT_KV_PRESSURE)
    ap.add_argument("--reference-bucket", type=int, default=1024,
                    help="bucket whose floor/ceiling to compare against (default 1024)")
    ap.add_argument("--kv-dtype", default="fp16_or_bf16",
                    help="kv dtype row to read ceilings from (default fp16_or_bf16)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    diag = json.load(open(args.diagnostic))
    acc = json.load(open(args.acceptance))
    kv = json.load(open(args.kv_pressure))

    bucket = args.reference_bucket
    # Floor: same-host reference engine bootstrap tok/s for the reference bucket.
    floor_prefill = acc["bootstrap_targets"]["prefill_tokens_s"].get(str(bucket))
    floor_decode = acc["bootstrap_targets"]["decode_tokens_s"].get(str(bucket))
    floor_basis = acc["r0_target_policy"].get("bootstrap_target_basis")
    target_ratio = acc["r0_target_policy"].get("roofline_default_target_ratio", 0.7)

    # Ceiling: measured roofline decode tok/s for the reference bucket + dtype.
    ceil_decode = None
    for row in kv.get("rows", []):
        if row.get("bucket") == bucket and row.get("dtype") == args.kv_dtype:
            ceil_decode = row.get("ceiling_tok_s_at_qmatvec_max")
            break

    # Native per-case prefill/decode tok/s.
    rows = []
    decode_tps_values = []
    for case in diag.get("timed_case_rows", []):
        cid = case["case_id"]
        t = case["timing_ns"]
        ptoks = case["prompt_token_count"]
        gtoks = case["generated_token_count"]
        prefill_s = ns_to_s(t["prompt_prefill"])
        decode_s = ns_to_s(t["decode_continuation"])
        prefill_tps = ptoks / prefill_s if prefill_s > 0 else None
        decode_tps = gtoks / decode_s if decode_s > 0 else None
        if decode_tps:
            decode_tps_values.append(decode_tps)
        rows.append({
            "case_id": cid,
            "prompt_tokens": ptoks,
            "generated_tokens": gtoks,
            "prefill_tok_s": round(prefill_tps, 3) if prefill_tps else None,
            "decode_tok_s": round(decode_tps, 3) if decode_tps else None,
            "decode_vs_floor": round(decode_tps / floor_decode, 4) if (decode_tps and floor_decode) else None,
            "decode_roofline_util": round(decode_tps / ceil_decode, 5) if (decode_tps and ceil_decode) else None,
        })

    mean_decode = sum(decode_tps_values) / len(decode_tps_values) if decode_tps_values else None
    bar_decode = target_ratio * ceil_decode if ceil_decode else None

    verdict = []
    if mean_decode and floor_decode:
        verdict.append(
            f"native decode mean {mean_decode:.1f} tok/s is {mean_decode/floor_decode:.3f}x the "
            f"{bucket} floor ({floor_decode} tok/s): ~{floor_decode/mean_decode:.0f}x slower than the same-host reference"
        )
    if mean_decode and ceil_decode:
        verdict.append(
            f"native decode is {mean_decode/ceil_decode:.4f} of the {bucket} roofline ceiling "
            f"({ceil_decode:.0f} tok/s); the {int(target_ratio*100)}% bar is {bar_decode:.0f} tok/s"
        )
    verdict.append(
        "sub-1k smoke alignment only; not the R2 matrix. Run the full 1k-8k x 512 "
        "matrix on the target and refresh the same-host floor before any product claim."
    )

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(REPO, "output", f"r2-speed-denominator-{ts}")
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "schema_version": "intel-qwen36-r2-speed-denominator-v0",
        "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
        "created_at": ts,
        "alignment": "sub_1k_smoke",
        "reference_bucket": bucket,
        "kv_dtype": args.kv_dtype,
        "floor": {
            "source": os.path.relpath(args.acceptance, REPO),
            "basis": floor_basis,
            "prefill_tok_s": floor_prefill,
            "decode_tok_s": floor_decode,
            "is_bootstrap_placeholder": True,
        },
        "roofline_ceiling": {
            "source": os.path.relpath(args.kv_pressure, REPO),
            "decode_tok_s_at_qmatvec_max": ceil_decode,
            "target_ratio": target_ratio,
            "bar_decode_tok_s": round(bar_decode, 2) if bar_decode else None,
        },
        "native": {
            "source": os.path.relpath(args.diagnostic, REPO),
            "cases": rows,
            "decode_tok_s_mean": round(mean_decode, 3) if mean_decode else None,
        },
        "verdict": verdict,
        "speedup_claims_allowed": False,
    }
    with open(os.path.join(out_dir, "speed-denominator.json"), "w") as f:
        json.dump(report, f, indent=2)

    md = ["# R2 speed denominator (sub-1k smoke alignment)\n"]
    md.append(f"- native: `{report['native']['source']}`")
    md.append(f"- floor (bucket {bucket}, **bootstrap placeholder**): decode {floor_decode} tok/s — {floor_basis}")
    md.append(f"- roofline ceiling (bucket {bucket}, {args.kv_dtype}): decode {ceil_decode:.0f} tok/s; {int(target_ratio*100)}% bar {bar_decode:.0f} tok/s\n")
    md.append("| case | prompt | gen | decode tok/s | vs floor | roofline util |")
    md.append("|---|--:|--:|--:|--:|--:|")
    for r in rows:
        md.append(
            f"| `{r['case_id']}` | {r['prompt_tokens']} | {r['generated_tokens']} | "
            f"{r['decode_tok_s']} | {r['decode_vs_floor']} | {r['decode_roofline_util']} |"
        )
    md.append("")
    md.append("## Verdict")
    for v in verdict:
        md.append(f"- {v}")
    md.append("")
    with open(os.path.join(out_dir, "speed-denominator.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"wrote {out_dir}/speed-denominator.json")
    print(f"wrote {out_dir}/speed-denominator.md")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
