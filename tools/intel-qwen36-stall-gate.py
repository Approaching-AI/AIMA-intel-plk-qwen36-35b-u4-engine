#!/usr/bin/env python3
"""D-controller stall gate — meta-engine-factory ch.3 §3.5.

The forensic record is blunt: across amd395-highspeed, amd395-native, and the
sibling ports, the search died the *same* way — the agent could not abandon a
dead route (highspeed) or convert off a tar-pit boundary (native), and each time
someone hand-wrote a "do not repeat" list and opened the next micro-variant
anyway. The conclusion: a stop rule must be a **harness-enforced gate, not prose
the agent writes for itself**.

This repo's live risk is the *highspeed* shape, not the native one: structural
gates can keep advancing while the active OpenVINO-derived product goal stays
flat. The retired Vulkan bring-up floor remains historical context only.
The single-counter "has the frontier moved?" never alarms on this, because the
*wrong* frontier is moving. This gate reads the GOAL-axis counter from
frontier.json and forces the reflection.

Triggers (ch.3 §3.5):
  ① soft reflection (~30 goal-flat runs): "am I still on the right path?"
     Non-blocking by default (print the review); --strict makes it block.
  ② hard stall (~50 goal-flat runs): ALWAYS blocks. structural_axis_advancing no
     longer waves the build through — that auto-pass was the exact hole that let a
     60+ run L38/L39 correctness tar-pit continue while the (then-broken) counter
     read 0. It now only selects the REMEDY message: structural still advancing =
     highspeed shape (assemble the decode loop, within-route); structural also
     frozen = native shape (genuine tar-pit, check the ruler then switch). Either
     way the hard stall blocks until a keyed review is recorded in
     routes-ledger.json (goal_stall_reviews[{best_ts,...}]) or the goal moves.
  ③ glide-slope (direction, not motion): micro-improvements keep resetting the
     counters, but if the trailing improvement rate cannot reach the floor
     within the horizon, the route is asymptoting BELOW the goal — reflect now.
     Soft (prints; --strict blocks). The legacy scalar counter uses the
     empirical same-config effect band in frontier.json; this is only a stall
     heuristic. Product/component promotion uses paired confidence bounds.
  ④ budget kill-number (ch.2 §2.1 #1): when goal_budget shows the overhead-only
     ceiling is below the floor, more overhead-cut candidates are sub-threshold
     BY ARITHMETIC — the review must pick kernel-side work (bandwidth / layout /
     fusion; see routes-ledger parked_routes) instead of another micro-cut.

Run it before launching a new probe (session-start / pre-commit / runner). A
non-zero exit means: stop, do the review, convert or switch.

Usage:
  python3 tools/intel-qwen36-stall-gate.py
  python3 tools/intel-qwen36-stall-gate.py --strict   # soft also blocks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONTIER = REPO / "doc" / "active" / "intel-qwen36-35b-a3b-gguf-q4km" / "frontier.json"

REVIEW = """\
Went-off-the-rails review (ch.3 §3.5) — diagnose which layer is wrong, then take
the other branch at that point. Do NOT just run the same boundary again.

  • ROUTE wrong   -> switch kernel/algorithm, OR stop searching and go back to
                     ch.2 understanding (profile / root-cause / read the reference).
  • RULER wrong   -> the tolerance is measuring the wrong thing. LIVE RISK HERE:
                     the z-correction lane chased ~1e-6 CPU float accumulation
                     order, but the invariant is cosine>=0.999 / KLD<0.005 / top-1
                     >=0.99, explicitly NOT bit-exact. If cosine>=0.999 already
                     holds, PASS the boundary; run an FP64 sensitivity check
                     before polishing accumulation order (routes-ledger invariant).
  • SCOPE wrong   -> change architecture / shrink scope / isolate the boundary
                     and let other lanes proceed.

Which correction applies is printed above from the live signals:
  • highspeed shape (structural still advancing, goal flat) -> WITHIN-route: stop
    adding per-boundary gates and ASSEMBLE THE resident/full-GPU DECODE LOOP so the
    goal metric moves. The engine layer is already O(1) — run one parameterized GPU
    layer + the loop, do not re-verify every layer index.
  • native shape (structural ALSO frozen) -> genuine tar-pit: check the RULER
    first (§1.5, NOT exact top-k), then pop a pre-registered alternate.

Pre-registered alternates to pop from (only if GPU itself is wrong):
  doc/active/intel-qwen36-35b-a3b-gguf-q4km/routes-ledger.json#parked_routes
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat soft-reflection breach as blocking too")
    ap.add_argument("--hard-stall-threshold", type=int, default=None,
                    help="override hard threshold (for testing the gate)")
    args = ap.parse_args()

    if not FRONTIER.exists():
        print("stall-gate: frontier.json missing; run frontier-sync first", file=sys.stderr)
        return 2

    state = json.loads(FRONTIER.read_text(encoding="utf-8"))
    np = state["no_progress"]
    # Legacy empirical-effect-band counter; fall back to the raw counter for
    # frontier files generated before the heuristic existed.
    since = int(np.get("runs_since_significant_improvement",
                       np.get("runs_since_goal_improved", 0)))
    since_raw = int(np.get("runs_since_goal_improved", 0))
    noise = np.get("noise") or {}
    soft_t = int(np["soft_reflection_threshold"])
    hard_t = args.hard_stall_threshold if args.hard_stall_threshold is not None \
        else int(np["hard_stall_threshold"])
    structural_advancing = bool(np.get("structural_axis_advancing", False))
    review_recorded = bool(np.get("review_recorded_for_current_best", False))
    ga = state.get("goal_anchor", {})
    bp = state.get("bringup_progress", {})
    glide = np.get("glide_slope") or {}
    budget = state.get("goal_budget") or {}
    budget_verdict = budget.get("verdict") or {}
    active_floor = ga.get(
        "active_product_decode_floor_tps",
        ga.get("same_host_vulkan_floor_tps"),
    )

    print(f"goal anchor ({ga.get('metric')}): best {ga.get('current_best_tps')} tok/s "
          f"(floor {active_floor}, cpu denom "
          f"{ga.get('cpu_native_denominator_tps')})")
    print(f"structural: deepest layer closed {bp.get('deepest_layer_closed')}, "
          f"{bp.get('gpu_probe_runs')} GPU probe runs (advancing={structural_advancing})")
    print(f"no-progress runs since SIGNIFICANT goal improvement: {since} "
          f"(raw {since_raw}; legacy effect band {100 * float(noise.get('rel') or 0):.2f}%; "
          f"soft {soft_t} / hard {hard_t})")
    if glide.get("trailing_rate_tps_per_run") is not None:
        proj = glide.get("projected_runs_to_floor")
        print(f"glide-slope: {glide['trailing_rate_tps_per_run']:+.4f} tok/s per run "
              f"(trailing {glide.get('window_runs')}) -> floor in "
              f"{proj if proj is not None else 'inf'} runs "
              f"(horizon {glide.get('horizon_runs')})")
    if budget_verdict.get("can_reach_floor_without_kernel_work") is False:
        kernel_cut = float(
            budget_verdict.get("min_kernel_time_cut_pct_needed") or 0.0)
        kernel_cut_pct = kernel_cut * 100.0 if kernel_cut <= 1.0 else kernel_cut
        print(f"budget kill-number: overhead-only ceiling "
              f"{budget_verdict.get('overhead_only_ceiling_tok_s')} tok/s < floor; "
              f"kernel device time must shrink >= "
              f"{kernel_cut_pct:.2f}% — more "
              f"overhead-cut micro-candidates are sub-threshold by arithmetic")

    hard = since >= hard_t
    soft = since >= soft_t
    glide_breached = bool(glide.get("breached", False))

    if hard:
        # A hard stall ALWAYS blocks (ch.3 §3.5: the stop rule is a harness gate,
        # not prose the agent writes for itself). structural_advancing only picks
        # the remedy; it no longer waves the build through.
        if review_recorded:
            print("\n* GOAL HARD-STALL — but a keyed review is recorded in "
                  "routes-ledger.json (goal_stall_reviews). Proceeding on the "
                  "recorded route; a new best that stalls again needs a fresh "
                  "review. *\n")
            print(REVIEW)
            return 0
        if structural_advancing:
            print("\n** GOAL HARD-STALL (highspeed shape: structural axis still "
                  "advancing, goal axis flat). BLOCKING. **")
            print("  Fix is WITHIN-route: STOP adding per-boundary gates and "
                  "ASSEMBLE the resident/full-GPU decode loop so the goal moves.")
        else:
            print("\n** GOAL HARD-STALL (native shape: structural axis ALSO frozen "
                  "= genuine tar-pit). BLOCKING. No new probe on this boundary. **")
            print("  Check the RULER first (§1.5: cosine>=0.999 / KLD<0.005 / "
                  "top-1>=0.99, NOT exact top-k), then switch route.")
        print("  To clear: record a keyed review in routes-ledger.json "
              "goal_stall_reviews ([{\"best_ts\": <current best ts>, ...}]), or "
              "move the goal metric.\n")
        print(REVIEW)
        return 1
    if soft:
        print("\n* SOFT REFLECTION breached — am I still on the right path? *")
        print(f"  ({hard_t - since} runs from the hard counter.)\n")
        print(REVIEW)
        return 1 if args.strict else 0
    if glide_breached:
        print("\n* GLIDE-SLOPE breached — the counters keep resetting on micro-"
              "improvements, but at the trailing rate the floor is beyond the "
              "horizon. Direction, not motion (ch.3 §3.5 trigger ①). *")
        if budget_verdict.get("can_reach_floor_without_kernel_work") is False:
            print("  The budget kill-number says WHY: the overhead-only ceiling is "
                  "below the floor. Stop opening overhead-cut candidates; pick the "
                  "kernel-side attack (routes-ledger parked_routes: offline repack "
                  "layout / MoE-down fusion / occupancy) or record why not.\n")
        print(REVIEW)
        return 1 if args.strict else 0

    print("within budget — proceed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
