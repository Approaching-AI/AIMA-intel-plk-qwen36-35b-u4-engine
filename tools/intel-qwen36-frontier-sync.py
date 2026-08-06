#!/usr/bin/env python3
"""Derive machine-readable frontier state (Tier-2) from the route ledgers + output census.

meta-engine-factory ch.3 §3.4 three-tier memory: Tier-2 is the *machine* layer
that holds state / no-progress / next-step so the single hand-written pointer
(`current-frontier.md`, Tier-3) and the changelog (`meta-log/`) stay thin.

This repo shipped Tier-2 only as hand-authored JSON (`routes-ledger.json`,
`rejected-routes.json`) that a validator reads; nothing derived state from the
actual run census, so the no-progress signal lived in 800-2987 line handwritten
logs and a 124KB doc/active lab notebook. Same root cause the gold-standard
siblings (a800-kimi, amd395-qwen36-0626) fixed structurally: add the machine
layer, do not keep hand-trimming prose.

What this does (pure stdlib, local, no model, no remote):
  1. Read the active route from routes-ledger.json (the authoritative attack board).
  2. Census output/ for the active GPU bring-up: how many probe runs, how deep the
     per-layer/per-boundary verification has reached (the *structural* axis), and
     whether ANY end-to-end decode tok/s has been measured (the *goal* axis).
  3. Compute the no-progress counter on the GOAL axis (methodology §1.1/§3.3
     failure-mode ④: judge progress on the goal metric, not a proxy). The GPU
     bring-up's structural axis (deepest boundary closed) advances while the goal
     diagnostic metric (legacy short decode tok/s; the product goal is now the
     separate 2k-128k/output-512 matrix)
     has never moved —
     that is exactly the highspeed failure shape ("frontier moves, route dead for
     the goal"), and this counter makes the machine say it.
  4. Write doc/active/<ws>/frontier.json.

Usage:
  python3 tools/intel-qwen36-frontier-sync.py
  python3 tools/intel-qwen36-frontier-sync.py --check   # CI mode: nonzero if stale
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq36_budget  # noqa: E402  (tools/ sibling module)
import iq36_local  # noqa: E402  (tools/ sibling module)

REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = REPO / "doc" / "active" / WS
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
OUTPUT = REPO / "output"
EXPLORE_LOG = OUTPUT / "explore-log.jsonl"
OUT = ACTIVE / "frontier.json"
ACCEPTANCE = REPO / "benchmarks" / WS / "acceptance-matrix.json"

SOFT_REFLECTION_THRESHOLD = 30  # ch.3 §3.5 trigger ①: "still on the right path?"
HARD_STALL_THRESHOLD = 50       # ch.3 §3.5 trigger ②: forbid same-axis retries

# Legacy effect-size heuristic for the scalar diagnostic frontier. Product and
# component promotion use the confidence-bound policy in the acceptance matrix;
# this empirical same-config spread only prevents old single-row micro-deltas
# from resetting the stall counter.

# Glide-slope (ch.3 §3.5 trigger ① asks about DIRECTION, not motion): even while
# micro-improvements keep resetting the counter, project the trailing improvement
# rate forward; if the floor is not reachable within the horizon, reflect now.
GLIDE_WINDOW_RUNS = 100   # trailing token-emitting runs used to estimate the rate
GLIDE_MIN_RUNS = 30       # do not judge a fresh route on fewer runs than this
GLIDE_HORIZON_RUNS = 200  # "reachable" means within this many further runs

# Diagnostic scalar anchor for the legacy short decode search. ADR 0051 makes
# the seven-row 2k-128k/output-512 matrix the product goal; these short rows
# remain useful for stall/budget history but cannot satisfy product promotion.
GOAL_METRIC = "diagnostic short/1k cold no-prefix decode tok/s (TPOT), conc=1"
SAME_HOST_VULKAN_FLOOR_TPS = 19.5   # interim bring-up floor
DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS = 49.8  # retired 1k product floor
DIAGNOSTIC_SHORT_DECODE_DEVICE_CAP_MS = 19.980
DIAGNOSTIC_SHORT_DECODE_WALL_CAP_MS = 20.080
CORE_PRODUCT_INPUT_BUCKETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
CORE_PRODUCT_OUTPUT_TOKENS = 512
CPU_NATIVE_DENOMINATOR_TPS = 4.2    # current CPU engine = oracle/denominator

# The long dir name carries the authoritative logical timeline (ch.3 §3.4 Tier-1
# "name is the index"); filesystem mtime is unreliable here (output/ is
# disposable/gitignored and gets copied). Order the goal-axis counter on this.
TS_RE = re.compile(r"(\d{8}T\d{6}Z)")
SPEED_OR_DISTRIBUTION_RE = re.compile(
    r"^(?P<family>.+?)-(?P<lane>speed|distribution)-\d{8}T\d{6}Z$"
)


def name_ts(name: str):
    m = TS_RE.search(name)
    return m.group(1) if m else None


def run_ts(run: dict) -> str | None:
    ts = run.get("ts")
    if isinstance(ts, str) and TS_RE.fullmatch(ts):
        return ts
    artifact = run.get("artifact")
    return name_ts(Path(artifact).name) if isinstance(artifact, str) else None


def iso_ts(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", value
    )
    if match is None:
        return None
    return "%s%s%sT%s%s%sZ" % match.groups()


def rejected_artifacts(rejected: dict) -> set[str]:
    artifacts: set[str] = set()
    for route in rejected.get("rejected", []) or []:
        evidence = route.get("evidence")
        if not isinstance(evidence, str):
            continue
        for item in evidence.split(","):
            item = item.strip().rstrip("/")
            if item.startswith("output/"):
                artifacts.add(item)
    return artifacts


def artifact_family(artifact: str) -> str | None:
    """Return the route family for paired speed/distribution artifacts."""
    name = Path(artifact).name
    m = SPEED_OR_DISTRIBUTION_RE.match(name)
    return m.group("family") if m else None


# Config identity (result.json flag set minus volatile keys) is shared with the
# runners via iq36_local so explore-log lines and dir artifacts hash alike.
config_sha = iq36_local.config_sha


def load_explore_log() -> list[dict]:
    """Tier-1 explore rounds append one JSONL line instead of a full artifact dir
    (ch.2 §2.2 collapse-ritual: exploration is lightweight, promotion writes the
    full bundle). They still count as token-emitting runs for the stall counters."""
    if not EXPLORE_LOG.is_file():
        return []
    rows: list[dict] = []
    for line in EXPLORE_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and TS_RE.fullmatch(str(row.get("ts", ""))):
            rows.append(row)
    return rows


NOISE_PAIR_WINDOW_S = 30 * 60  # same-config runs farther apart likely straddle a source edit


def _ts_seconds(ts: str) -> int:
    # YYYYMMDDTHHMMSSZ -> comparable seconds (calendar-approximate is fine for windowing)
    return (
        int(ts[0:4]) * 31_536_000 + int(ts[4:6]) * 2_678_400 + int(ts[6:8]) * 86_400
        + int(ts[9:11]) * 3_600 + int(ts[11:13]) * 60 + int(ts[13:15])
    )


def noise_floor(speed_rows: list[dict]) -> dict:
    """Estimate run-to-run relative spread from same-config repeat pairs.

    Family-name identity is NOT config identity (the same family name gets
    reused across different flag sets, producing fake 90% 'spreads'); group by
    the result.json config hash instead. The flag set alone still under-
    identifies: the engine SOURCE changes between same-flag runs, so only pair
    runs stamped within NOISE_PAIR_WINDOW_S of each other (true repeat/confirm
    runs land minutes apart). Manifests that carry `source_sha` are already
    disambiguated by the config hash itself.
    """
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in speed_rows:
        ts = row.get("ts")
        if not ts:
            continue
        groups.setdefault(row["config_sha"], []).append((_ts_seconds(ts), row["tps"]))
    spreads: list[float] = []
    for vals in groups.values():
        if len(vals) < 2:
            continue
        vals.sort()
        for (t0, a), (t1, b) in zip(vals, vals[1:]):
            if t1 - t0 > NOISE_PAIR_WINDOW_S:
                continue
            hi, lo = max(a, b), min(a, b)
            if hi > 0:
                spreads.append((hi - lo) / hi)
    spreads.sort()
    p50 = spreads[len(spreads) // 2] if spreads else None
    p90 = spreads[int(len(spreads) * 0.9)] if spreads else None
    # Median, not p90: legacy manifests carry no source fingerprint, so the pair
    # census mixes true repeats (<=~1%) with same-flag runs that straddle a
    # source edit (up to ~12%). The median sits in the true-repeat mass and is
    # robust to that tail; manifests with `source_sha` disambiguate at the
    # config-hash level and clean the census going forward.
    rel = p50 if p50 is not None else 0.0
    return {
        "rel": round(rel, 5),
        "same_config_pairs": len(spreads),
        "p50_same_config_spread": round(p50, 5) if p50 is not None else None,
        "p90_same_config_spread": round(p90, 5) if p90 is not None else None,
        "promotion_gate": False,
        "role": "legacy_scalar_stall_heuristic",
        "definition": (
            "median relative spread across same-config-hash repeat pairs stamped "
            "<=%d min apart, with no fixed floor or cap (median, because "
            "legacy same-flag pairs can straddle source edits; source_sha in new "
            "manifests removes that ambiguity). This is a legacy stall heuristic, "
            "not a promotion gate; promotion uses one-sided confidence bounds."
            % (NOISE_PAIR_WINDOW_S // 60)
        ),
    }


def significant_chain(decode_runs: list[dict], noise_rel: float) -> list[dict]:
    """Legacy effect-band improvement events, in timeline order.

    This preserves the old scalar frontier's stall history. New promotion
    decisions are made by paired confidence bounds, not this heuristic.
    """
    events: list[dict] = []
    baseline: float | None = None
    for run in sorted(
        decode_runs,
        key=lambda r: (
            run_ts(r) or "",
            str(r["artifact"]),
        ),
    ):
        tps = run["tps"]
        if baseline is None or tps > baseline * (1.0 + noise_rel):
            ts = run_ts(run)
            events.append({"ts": ts, "tps": round(tps, 6), "artifact": run["artifact"]})
            baseline = tps
    return events


def glide_slope(census_ts: list[str], decode_runs: list[dict], best_tps: float | None,
                floor: float) -> dict:
    """Direction check: at the trailing improvement rate, is the floor reachable
    within the horizon? Soft signal only — the hard stop stays with the counter."""
    out: dict = {
        "window_runs": GLIDE_WINDOW_RUNS,
        "horizon_runs": GLIDE_HORIZON_RUNS,
        "trailing_rate_tps_per_run": None,
        "projected_runs_to_floor": None,
        "breached": False,
        "definition": (
            "best-tps gain over the trailing %d token-emitting runs, projected "
            "forward; breached when the %.1f floor is > %d further runs away at "
            "that rate (soft reflection, ch.3 §3.5 trigger ① 'direction, not "
            "motion')." % (GLIDE_WINDOW_RUNS, floor, GLIDE_HORIZON_RUNS)
        ),
    }
    if best_tps is None or best_tps >= floor:
        return out
    ordered = sorted(t for t in census_ts if t)
    if len(ordered) < GLIDE_MIN_RUNS:
        return out
    window = ordered[-GLIDE_WINDOW_RUNS:]
    window_start = window[0]
    prior = [r["tps"] for r in decode_runs if (run_ts(r) or "") <= window_start]
    if not prior:
        return out
    best_then = max(prior)
    rate = (best_tps - best_then) / len(window)
    out["trailing_rate_tps_per_run"] = round(rate, 6)
    if rate > 0:
        projected = (floor - best_tps) / rate
        out["projected_runs_to_floor"] = int(projected)
        out["breached"] = projected > GLIDE_HORIZON_RUNS
    # rate <= 0 with best < floor is the flat case the no-progress counter owns.
    return out


def distribution_ladder_summary(smoke: dict) -> dict | None:
    dist = smoke.get("distribution_ladder") if isinstance(smoke, dict) else None
    if not isinstance(dist, dict):
        return None
    if dist.get("required_checks_passed") is not True:
        return None
    if dist.get("kld_pass") is False or dist.get("top1_pass") is False:
        return None
    return {
        "max_kld": dist.get("max_kld"),
        "top1_rate": dist.get("top1_rate"),
        "min_logits_cosine": dist.get("min_logits_cosine"),
        "position_count": dist.get("position_count"),
    }


def invalid_kernel_profile_row(smoke: dict, payload: dict | None = None) -> bool:
    """Rows that intentionally disable OpenCL event profiling lack kernel budget.

    The wall tok/s is still useful diagnostic evidence, but the benchmark
    discipline requires a valid kernel-busy floor for the accepted frontier and
    budget. Such rows may only set the speed best when paired with a valid
    profiling row from the same source.
    """
    payload = payload if isinstance(payload, dict) else {}
    return (
        smoke.get("opencl_no_queue_profiling") is True
        or smoke.get("skip_opencl_event_profile_readback") is True
        or payload.get("opencl_no_queue_profiling") is True
        or payload.get("skip_opencl_event_profile_readback") is True
    )


def load_r2_results(rejected: dict) -> list[tuple[str, dict, dict]]:
    rejected_output = rejected_artifacts(rejected)
    rows: list[tuple[str, dict, dict]] = []
    for path in OUTPUT.glob("r2-gpu-*/result.json"):
        artifact = str(path.parent.relative_to(REPO))
        if artifact.rstrip("/") in rejected_output:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        smoke = payload.get("smoke", {})
        if not isinstance(smoke, dict):
            continue
        rows.append((artifact, payload, smoke))
    return rows


def load_real_backend_results() -> list[dict]:
    """Load complete real 40-layer Level Zero rows onto the goal axis.

    These gates use a compact result schema rather than the legacy R2 smoke
    schema. They are the authoritative decode rows once all real layers execute
    and the three-case consensus predicate passes.
    """
    rows: list[dict] = []
    for path in OUTPUT.glob("packed-token-level-zero-real-backend-gate-*/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        tps = payload.get("decode_tokens_s_median")
        device_ms = payload.get("device_ms_median")
        wall_ms = payload.get("wall_ms_median")
        case_rows = payload.get("rows")
        if not all(isinstance(v, (int, float)) for v in (tps, device_ms, wall_ms)):
            continue
        if payload.get("correctness_checks_passed") is not True:
            continue
        if not isinstance(case_rows, list) or len(case_rows) < 3:
            continue
        if any(row.get("exact_generated_ids") is not True for row in case_rows):
            continue
        artifact = str(path.parent.relative_to(REPO))
        git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        distribution = payload.get("distribution_diagnostic")
        distribution_ladder = (
            distribution.get("distribution_ladder")
            if isinstance(distribution, dict)
            and isinstance(distribution.get("distribution_ladder"), dict)
            else {}
        )
        distribution_passed = (
            payload.get("distribution_diagnostic_passed") is True
            and distribution_ladder.get("required_checks_passed") is True
        )
        rows.append({
            "artifact": artifact,
            "ts": iso_ts(payload.get("created_at")),
            "tps": float(tps),
            "case_id": "fit_validation_test_consensus",
            "prompt_token_count": None,
            "top1_matches_native": True,
            "topk_ids_match_native": None,
            "correctness_mode": (
                "three_case_eight_token_consensus_and_teacher_forced_distribution"
                if distribution_passed
                else "three_case_eight_token_consensus"
            ),
            "correctness_artifact": artifact,
            "distribution_max_kld": distribution_ladder.get("max_kld"),
            "distribution_top1_rate": distribution_ladder.get("top1_rate"),
            "distribution_min_logits_cosine": distribution_ladder.get(
                "min_logits_cosine"
            ),
            "kernel_profiles_valid": True,
            "kernel_profile_pair": None,
            "diagnostic_class": "full_real_40_layer_level_zero_decode",
            "device_ms_median": float(device_ms),
            "wall_ms_median": float(wall_ms),
            "host_submit_ms_max": max(
                float(row.get("host_submit_ms_max", 0.0)) for row in case_rows
            ),
            "kernel_count": max(int(row.get("kernel_count", 0)) for row in case_rows),
            "source_sha": str(git.get("commit") or ""),
        })
    return rows


def census_output(rejected: dict) -> dict:
    """Mechanical census of output/ for the active GPU bring-up route.

    Names are the index (ch.3 §3.4 Tier-1). We do not parse run internals; the
    dir-name layer index and the absence of any `*decode*tok*`/`*tps*` GPU run is
    enough to drive the goal-axis counter.
    """
    if not OUTPUT.is_dir():
        return {
            "gpu_probe_runs": 0,
            "deepest_layer_closed": 0,
            "goal_decode_runs": 0,
            "goal_decode_run_names": [],
            "best_goal_decode": None,
        }
    gpu_dirs = [p.name for p in OUTPUT.glob("gpu-*") if p.is_dir()]
    layer_re = re.compile(r"layer-?(\d+)")
    deepest = 0
    gpu_layer_ts = []  # (deepest layer in this dir name, dir timestamp) per gpu-* dir
    for name in gpu_dirs:
        d = 0
        for m in layer_re.finditer(name):
            d = max(d, int(m.group(1)))
        deepest = max(deepest, d)
        gpu_layer_ts.append((d, name_ts(name)))
    # Every token-emitting decode/correctness smoke lands as an r2-gpu-* dir (pass
    # OR fail). This is the population the goal-axis no-progress counter watches:
    # the L38/L39 tar-pit is 60+ FAILING r2-gpu-* runs, invisible to a counter
    # that only looked at passing decode_runs.
    r2_gpu_ts = [t for t in (name_ts(p.name) for p in OUTPUT.glob("r2-gpu-*") if p.is_dir()) if t]
    explore_rows = load_explore_log()
    explore_ts = [str(r["ts"]) for r in explore_rows]
    explore_tps = [r["tps"] for r in explore_rows if isinstance(r.get("tps"), (int, float))]
    explore_profile_rows = [
        r for r in explore_rows
        if (
            isinstance(r.get("profile_smoke"), dict)
            and r.get("opencl_no_queue_profiling") is not True
            and r.get("skip_opencl_event_profile_readback") is not True
        )
    ]
    latest_profile_explore = (
        max(explore_profile_rows, key=lambda r: str(r.get("ts", "")))
        if explore_profile_rows else None
    )
    profile_explore_by_source = {}
    for row in explore_profile_rows:
        source_sha = str(row.get("source_sha") or "")
        if not source_sha:
            continue
        prev = profile_explore_by_source.get(source_sha)
        if prev is None or str(row.get("ts", "")) > str(prev.get("ts", "")):
            profile_explore_by_source[source_sha] = row
    r2_results = load_r2_results(rejected)
    real_backend_rows = load_real_backend_results()
    # Every run with a measured tok/s (regardless of correctness pairing) feeds
    # the same-config noise estimate; explore JSONL rows carry config_sha inline.
    speed_rows = [
        {
            "ts": name_ts(Path(artifact).name),
            "tps": float(smoke["gpu_hybrid_decode_tok_s"]),
            "config_sha": config_sha(payload),
        }
        for artifact, payload, smoke in r2_results
        if isinstance(smoke.get("gpu_hybrid_decode_tok_s"), (int, float))
    ] + [
        {"ts": str(r["ts"]), "tps": float(r["tps"]), "config_sha": str(r.get("config_sha", ""))}
        for r in explore_rows
        if isinstance(r.get("tps"), (int, float)) and r.get("config_sha")
    ] + [
        {
            "ts": row.get("ts"),
            "tps": row["tps"],
            "config_sha": "real-backend:%s" % row.get("source_sha", ""),
        }
        for row in real_backend_rows
        if row.get("ts")
    ]
    distribution_by_family = {}
    for artifact, _payload, smoke in r2_results:
        summary = distribution_ladder_summary(smoke)
        family = artifact_family(artifact)
        if summary is None or family is None:
            continue
        row = {
            "artifact": artifact,
            "ts": name_ts(Path(artifact).name),
            **summary,
        }
        prev = distribution_by_family.get(family)
        if prev is None or (row["ts"] or "") > (prev.get("ts") or ""):
            distribution_by_family[family] = row

    decode_runs = []
    for artifact, payload, smoke in r2_results:
        tps = smoke.get("gpu_hybrid_decode_tok_s")
        if not isinstance(tps, (int, float)):
            continue
        if smoke.get("gpu_live_selected_cache_prewarm_enabled") is True:
            continue
        kernel_profiles_valid = not invalid_kernel_profile_row(smoke, payload)
        profile_pair = None
        if not kernel_profiles_valid:
            paired_profile = profile_explore_by_source.get(str(payload.get("source_sha") or ""))
            if isinstance(paired_profile, dict):
                profile_pair = {
                    "kind": "latest_valid_explore_profile",
                    "source": "output/explore-log.jsonl#%s" % paired_profile.get("ts"),
                    "label": paired_profile.get("label"),
                    "source_sha": paired_profile.get("source_sha"),
                }
            if profile_pair is None:
                continue
        if smoke.get("top1_matches_native") is not True:
            continue

        exact_topk = (
            payload.get("required_checks_passed") is True
            and smoke.get("required_checks_passed") is True
            and smoke.get("topk_ids_match_native") is True
        )
        self_distribution = distribution_ladder_summary(smoke)
        paired_distribution = None
        family = artifact_family(artifact)
        if family is not None:
            candidate = distribution_by_family.get(family)
            artifact_ts = name_ts(Path(artifact).name)
            candidate_ts = candidate.get("ts") if isinstance(candidate, dict) else None
            if candidate and (not artifact_ts or not candidate_ts or candidate_ts >= artifact_ts):
                paired_distribution = candidate

        correctness_mode = None
        correctness_artifact = None
        dist_summary = None
        if self_distribution is not None:
            correctness_mode = "teacher_forced_distribution_ladder"
            correctness_artifact = artifact
            dist_summary = self_distribution
        elif paired_distribution is not None:
            correctness_mode = "paired_teacher_forced_distribution_ladder"
            correctness_artifact = paired_distribution.get("artifact")
            dist_summary = paired_distribution
        elif exact_topk:
            correctness_mode = "legacy_exact_topk"
            correctness_artifact = artifact
        else:
            continue

        decode_runs.append({
            "artifact": artifact,
            "resident_api": smoke.get("resident_api"),
            "tps": float(tps),
            "case_id": smoke.get("case_id"),
            "prompt_token_count": smoke.get("prompt_token_count"),
            "top1_matches_native": smoke.get("top1_matches_native"),
            "topk_ids_match_native": smoke.get("topk_ids_match_native"),
            "correctness_mode": correctness_mode,
            "correctness_artifact": correctness_artifact,
            "distribution_max_kld": (
                dist_summary.get("max_kld") if isinstance(dist_summary, dict) else None
            ),
            "distribution_top1_rate": (
                dist_summary.get("top1_rate") if isinstance(dist_summary, dict) else None
            ),
            "distribution_min_logits_cosine": (
                dist_summary.get("min_logits_cosine") if isinstance(dist_summary, dict) else None
            ),
            "kernel_profiles_valid": kernel_profiles_valid,
            "kernel_profile_pair": profile_pair,
            "diagnostic_class": "cpu_prefill_gpu_hybrid_decode",
        })

    decode_runs.extend(real_backend_rows)

    # A goal-axis run is a token-emitting GPU decode/throughput lane (tok/s),
    # not a captured-boundary teacher-forced gate. The first accepted one is the
    # R2 hybrid smoke: CPU prefill/router/LM-head remain, so it is diagnostic and
    # cannot be promoted as a speedup.
    goal_re = re.compile(r"gpu.*(decode|tok-?s|tps|throughput|prompt-token)", re.I)
    goal_runs = [n for n in gpu_dirs if goal_re.search(n)]
    best_goal = max(decode_runs, key=lambda row: row["tps"]) if decode_runs else None
    return {
        "gpu_probe_runs": len(gpu_dirs),
        "deepest_layer_closed": deepest,
        "gpu_layer_ts": gpu_layer_ts,
        "r2_gpu_ts": r2_gpu_ts,
        "real_backend_ts": [row["ts"] for row in real_backend_rows if row.get("ts")],
        "explore_ts": explore_ts,
        "explore_runs": len(explore_rows),
        "best_explore_tps_unverified": round(max(explore_tps), 6) if explore_tps else None,
        "latest_profile_explore": latest_profile_explore,
        "profile_explore_by_source": profile_explore_by_source,
        "speed_rows": speed_rows,
        "decode_runs": decode_runs,
        "goal_decode_runs": len(goal_runs) + len(decode_runs),
        "goal_decode_run_names": goal_runs + [row["artifact"] for row in decode_runs],
        "best_goal_decode": best_goal,
    }


def build_state() -> dict:
    routes = json.loads(ROUTES.read_text(encoding="utf-8"))
    rejected = json.loads(REJECTED.read_text(encoding="utf-8"))
    cs = census_output(rejected)

    active = routes.get("active_route", {})
    active_id = str(active.get("id") or "")
    openvino_product_active = (
        active.get("family") == "openvino_gpu_specialization"
        or active_id.startswith("openvino_")
    )
    cand_hist = routes.get("candidate_history", []) or []
    switches = routes.get("switch_decisions", []) or []
    parked = routes.get("parked_routes", []) or []

    # Goal axis: token-emitting GPU decode smoke resets the "never measured"
    # condition, but it is still diagnostic until resident/full-GPU decode and
    # the benchmark discipline close.
    goal_best = cs.get("best_goal_decode")
    goal_best_tps = goal_best.get("tps") if isinstance(goal_best, dict) else None
    best_ts = run_ts(goal_best) if isinstance(goal_best, dict) else None
    if isinstance(goal_best, dict):
        if goal_best.get("correctness_mode") == (
            "three_case_eight_token_consensus_and_teacher_forced_distribution"
        ):
            correctness_note = (
                "fit/validation/test eight-token consensus and teacher-forced "
                "distribution passed (max KLD %s, top-1 rate %s)"
                % (
                    goal_best.get("distribution_max_kld"),
                    goal_best.get("distribution_top1_rate"),
                )
            )
        elif goal_best.get("correctness_mode") == "three_case_eight_token_consensus":
            correctness_note = "fit/validation/test eight-token consensus passed"
        elif goal_best.get("correctness_mode") in {
            "teacher_forced_distribution_ladder",
            "paired_teacher_forced_distribution_ladder",
        }:
            correctness_note = (
                "teacher-forced distribution ladder passed"
                " (max KLD %s, top-1 rate %s)"
                % (
                    goal_best.get("distribution_max_kld"),
                    goal_best.get("distribution_top1_rate"),
                )
            )
        elif goal_best.get("correctness_mode") == "legacy_exact_topk":
            correctness_note = "legacy exact top-k matched native"
        else:
            correctness_note = "correctness mode recorded in frontier"
    else:
        correctness_note = None

    # No-progress on the GOAL axis (ch.3 §3.5): how many token-emitting GPU runs
    # (pass OR fail) landed since the best decode tok/s was last improved.
    # The prior formula was `gpu_probe_runs if goal_decode_runs == 0 else 0` — it
    # latched to 0 the instant ANY decode run existed, so a 60+ run L38/L39
    # correctness tar-pit read as "0 stall" and the gate could never fire again.
    # Count r2-gpu-* dirs + explore-log rounds stamped after the current best;
    # fall back to the never-measured case (no decode lane yet) = all probe runs.
    census_all_ts = cs["r2_gpu_ts"] + cs["explore_ts"] + cs["real_backend_ts"]
    if best_ts:
        runs_since_goal = sum(1 for t in census_all_ts if t > best_ts)
    else:
        runs_since_goal = cs["gpu_probe_runs"]

    # Legacy scalar-frontier progress heuristic. It is deliberately separate
    # from the confidence-bound product/component promotion rule.
    noise = noise_floor(cs["speed_rows"])
    sig_events = significant_chain(cs["decode_runs"], noise["rel"])
    last_sig_ts = sig_events[-1]["ts"] if sig_events else None
    if last_sig_ts:
        runs_since_significant = sum(1 for t in census_all_ts if t > last_sig_ts)
    else:
        runs_since_significant = runs_since_goal
    soft = runs_since_significant >= SOFT_REFLECTION_THRESHOLD
    hard = runs_since_significant >= HARD_STALL_THRESHOLD

    latest_explore_profile_budget = None
    def explore_profile_budget(row: dict | None) -> dict | None:
        if not isinstance(row, dict):
            return None
        smoke = dict(row.get("profile_smoke") or {})
        smoke["gpu_hybrid_decode_tok_s"] = row.get("tps")
        smoke["top1_matches_native"] = row.get("top1_matches_native")
        smoke["required_checks_passed"] = row.get("required_checks_passed")
        payload = {
            "decode_tokens": row.get("decode_tokens"),
            "smoke": smoke,
        }
        budget_full = iq36_budget.compute_budget_from_payload(
            payload,
            "output/explore-log.jsonl#%s" % row.get("ts"),
            DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS,
        )
        if budget_full is None:
            return None
        return {
            "label": row.get("label"),
            "ts": row.get("ts"),
            "source_sha": row.get("source_sha"),
            "tps": row.get("tps"),
            "top1_matches_native": row.get("top1_matches_native"),
            "note": (
                "artifact-free explore profile; cannot set best, but can "
                "refresh direction/budget attribution between promotions"
            ),
            "source_artifact": budget_full["source_artifact"],
            "per_token_ms": budget_full["per_token_ms"],
            "verdict": budget_full["verdict"],
            "top_stage_walls_ms_per_token": (
                budget_full.get("stage_walls_ms_per_token", [])[:6]
            ),
            "stage_kernel_gap_estimates_ms_per_token": (
                budget_full.get("stage_kernel_gap_estimates_ms_per_token", [])[:8]
            ),
            "substage_gap_estimates_ms_per_token": (
                budget_full.get("substage_gap_estimates_ms_per_token", [])[:12]
            ),
        }

    latest_profile_explore = cs.get("latest_profile_explore")
    latest_explore_profile_budget = explore_profile_budget(latest_profile_explore)

    # Direction (glide-slope) + budget kill-number for the active route.
    glide = glide_slope(census_all_ts, cs["decode_runs"], goal_best_tps,
                        DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS)
    goal_budget = None
    if isinstance(goal_best, dict) and goal_best.get("artifact"):
        if goal_best.get("diagnostic_class") == "full_real_40_layer_level_zero_decode":
            device_ms = float(goal_best["device_ms_median"])
            wall_ms = float(goal_best["wall_ms_median"])
            goal_budget = {
                "source_artifact": goal_best["artifact"],
                "per_token_ms": {
                    "wall": round(wall_ms, 3),
                    "gpu_kernel_busy_floor": round(device_ms, 3),
                    "non_kernel_overhead": round(wall_ms - device_ms, 3),
                    "host_submit_max": round(float(goal_best["host_submit_ms_max"]), 6),
                },
                "verdict": {
                    "floor_tps": DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS,
                    "wall_cap_ms_per_token": DIAGNOSTIC_SHORT_DECODE_WALL_CAP_MS,
                    "device_cap_ms_per_token": DIAGNOSTIC_SHORT_DECODE_DEVICE_CAP_MS,
                    "overhead_only_ceiling_tok_s": round(1e3 / device_ms, 3),
                    "can_reach_floor_without_kernel_work": (
                        device_ms <= DIAGNOSTIC_SHORT_DECODE_DEVICE_CAP_MS
                    ),
                    "min_kernel_time_cut_ms_needed": round(
                        max(0.0, device_ms - DIAGNOSTIC_SHORT_DECODE_DEVICE_CAP_MS), 3
                    ),
                    "min_kernel_time_cut_pct_needed": round(
                        max(0.0, device_ms - DIAGNOSTIC_SHORT_DECODE_DEVICE_CAP_MS)
                        / device_ms,
                        4,
                    ),
                },
                "kernel_count": goal_best.get("kernel_count"),
                "regenerate_via": "tools/intel-qwen36-frontier-sync.py",
            }
            budget_full = None
        else:
            budget_full = iq36_budget.compute_budget(
                REPO / goal_best["artifact"], DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS)
        if (
            goal_budget is None
            and
            budget_full is not None
            and budget_full.get("per_token_ms", {}).get("gpu_kernel_busy_floor")
            is not None
        ):
            goal_budget = {
                "source_artifact": goal_best["artifact"],
                "per_token_ms": budget_full["per_token_ms"],
                "verdict": budget_full["verdict"],
                "top_stage_walls_ms_per_token": budget_full["stage_walls_ms_per_token"][:6],
                "stage_kernel_gap_estimates_ms_per_token": (
                    budget_full.get("stage_kernel_gap_estimates_ms_per_token", [])[:8]
                ),
                "substage_gap_estimates_ms_per_token": (
                    budget_full.get("substage_gap_estimates_ms_per_token", [])[:12]
                ),
                "regenerate_via": "tools/iq36_budget.py <speed-artifact>",
            }
        elif goal_budget is None and goal_best.get("kernel_profile_pair"):
            pair_info = goal_best.get("kernel_profile_pair") or {}
            pair_row = (cs.get("profile_explore_by_source") or {}).get(
                str(pair_info.get("source_sha") or "")
            )
            paired_profile_budget = explore_profile_budget(pair_row)
            try:
                speed_payload = json.loads(
                    (REPO / goal_best["artifact"] / "result.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                speed_payload = {}
            speed_smoke = speed_payload.get("smoke", {})
            tokens = (
                speed_smoke.get("decode_continuation_output_tokens")
                or speed_payload.get("decode_tokens")
            )
            wall_ns = speed_smoke.get("gpu_hybrid_decode_ns")
            kernel_ms = (
                (paired_profile_budget or {}).get("per_token_ms", {})
                .get("gpu_kernel_busy_floor")
            )
            if (
                isinstance(tokens, (int, float))
                and tokens > 0
                and isinstance(wall_ns, (int, float))
                and isinstance(kernel_ms, (int, float))
            ):
                wall_ms = round(float(wall_ns) / float(tokens) / 1e6, 3)
                overhead_ms = round(wall_ms - float(kernel_ms), 3)
                floor_budget_ms = 1e3 / DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS
                goal_budget = {
                    "source_artifact": goal_best["artifact"],
                    "kernel_profile_source": paired_profile_budget.get(
                        "source_artifact"
                    ),
                    "profile_pairing_note": (
                        "speed wall from profiling-disabled row; kernel-busy "
                        "floor and stage gaps from paired valid profile row"
                    ),
                    "per_token_ms": {
                        "wall": wall_ms,
                        "gpu_kernel_busy_floor": kernel_ms,
                        "non_kernel_overhead": overhead_ms,
                    },
                    "verdict": {
                        "floor_tps": DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS,
                        "floor_budget_ms_per_token": round(floor_budget_ms, 3),
                        "overhead_only_ceiling_tok_s": round(1e3 / float(kernel_ms), 3),
                        "can_reach_floor_without_kernel_work": (
                            float(kernel_ms) <= floor_budget_ms
                        ),
                        "min_kernel_time_cut_pct_needed": round(
                            max(0.0, (float(kernel_ms) - floor_budget_ms)
                                / float(kernel_ms)),
                            4,
                        ),
                    },
                        "top_stage_walls_ms_per_token": (
                        paired_profile_budget.get(
                            "top_stage_walls_ms_per_token", []
                        )
                    ),
                    "stage_kernel_gap_estimates_ms_per_token": (
                        paired_profile_budget.get(
                            "stage_kernel_gap_estimates_ms_per_token", []
                        )
                    ),
                    "substage_gap_estimates_ms_per_token": (
                        paired_profile_budget.get(
                            "substage_gap_estimates_ms_per_token", []
                        )
                    ),
                    "regenerate_via": (
                        "tools/iq36_budget.py <speed-artifact> plus paired valid profile"
                    ),
                }

    # ADR 0070 retired the legacy short scalar as the product ruler. Once the
    # active route is OpenVINO specialization, derive the live 32k wall
    # kill-number from the newest clean exact candidate instead of leaving
    # frontier.json pointed at the old packed-token Level Zero diagnostic.
    if openvino_product_active:
        product_rows: list[tuple[str, dict, dict, Path]] = []
        product_artifacts = {
            *OUTPUT.glob("openvino-hot-cold-product-*"),
            *OUTPUT.glob("openvino-linear-state-alias-validation-*"),
        }
        for artifact in product_artifacts:
            manifest_path = artifact / "manifest.json"
            correctness_path = artifact / "correctness.json"
            if not manifest_path.is_file() or not correctness_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                correctness = json.loads(
                    correctness_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("git", {}).get("dirty") is not False:
                continue
            if manifest.get("alias_linear_state_assign") is not True:
                continue
            rows = correctness.get("cases", []) or []
            if (
                correctness.get("required_checks_passed") is not True
                or not rows
                or not all(r.get("required_checks_passed") for r in rows)
            ):
                continue
            worker_path = next(
                artifact.glob("raw/*/correctness/candidate/worker-result.json"),
                None,
            )
            if worker_path is None:
                continue
            try:
                worker = json.loads(worker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if worker.get("input_token_count") != 32768:
                continue
            if len(worker.get("decode_wall_ms") or []) <= 16:
                continue
            captured = str(manifest.get("captured_at") or artifact.name)
            product_rows.append((captured, manifest, worker, artifact))

        if product_rows:
            _, manifest, worker, artifact = max(product_rows, key=lambda row: row[0])
            acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
            decode_floor = float(
                acceptance["bootstrap_targets"]["decode_tokens_s"]["32768"]
            )
            absolute_cap_ms = 1e3 / decode_floor
            candidate_walls = [float(v) for v in worker["decode_wall_ms"][16:]]
            candidate_median_ms = statistics.median(candidate_walls)

            stock_path = next(
                artifact.glob("raw/*/correctness/stock/worker-result.json"), None
            )
            stock_median_ms = None
            if stock_path is not None:
                stock = json.loads(stock_path.read_text(encoding="utf-8"))
                stock_walls = [
                    float(v) for v in (stock.get("decode_wall_ms") or [])[16:]
                ]
                if stock_walls:
                    stock_median_ms = statistics.median(stock_walls)
            ratio_cap_ms = (
                stock_median_ms / 1.10 if stock_median_ms is not None else None
            )
            effective_cap_ms = min(
                absolute_cap_ms,
                ratio_cap_ms if ratio_cap_ms is not None else absolute_cap_ms,
            )
            remaining_ms = max(0.0, candidate_median_ms - effective_cap_ms)
            goal_budget = {
                "source_artifact": str(artifact.relative_to(REPO)),
                "lane": {
                    "input_tokens": 32768,
                    "measured_output_tokens": worker.get("output_tokens"),
                    "required_product_output_tokens": CORE_PRODUCT_OUTPUT_TOKENS,
                    "diagnostic_only": (
                        worker.get("output_tokens") != CORE_PRODUCT_OUTPUT_TOKENS
                    ),
                    "warmup_decode_walls_excluded": 16,
                },
                "per_token_ms": {
                    "candidate_stable_wall_median": round(candidate_median_ms, 6),
                    "stock_stable_wall_median": (
                        round(stock_median_ms, 6)
                        if stock_median_ms is not None else None
                    ),
                    "absolute_cap": round(absolute_cap_ms, 6),
                    "same_run_1p10_cap": (
                        round(ratio_cap_ms, 6) if ratio_cap_ms is not None else None
                    ),
                    "effective_cap": round(effective_cap_ms, 6),
                    "remaining_cut": round(remaining_ms, 6),
                },
                "verdict": {
                    "floor_tps": decode_floor,
                    "current_stable_tps": round(1e3 / candidate_median_ms, 6),
                    "wall_cap_ms_per_token": round(effective_cap_ms, 6),
                    "can_reach_floor_without_more_wall_cut": remaining_ms == 0.0,
                    "min_complete_wall_cut_ms_needed": round(remaining_ms, 6),
                    "min_complete_wall_cut_pct_needed": round(
                        remaining_ms / candidate_median_ms, 6
                    ),
                },
                "candidate_git_commit": manifest.get("git", {}).get("commit"),
                "note": (
                    "current OpenVINO 32k decode kill-number from a clean exact "
                    "diagnostic row; output512 and paired ABBA remain required "
                    "for product promotion"
                ),
                "regenerate_via": "tools/intel-qwen36-frontier-sync.py",
            }

    # A route can register a matched standalone wall worker after its complete
    # correctness row has been admitted. This keeps the machine-layer budget
    # tied to raw worker samples even when the exploratory wall row intentionally
    # omits the full product-gate bundle. The hand-authored route ledger selects
    # the evidence; every number below is re-derived here.
    route_budget_source = active.get("goal_budget_source")
    if openvino_product_active and isinstance(route_budget_source, dict):
        candidate_path = REPO / str(route_budget_source["candidate_worker"])
        stock_path = REPO / str(route_budget_source["stock_worker"])
        correctness_artifact = REPO / str(
            route_budget_source["correctness_artifact"]
        )
        if not candidate_path.is_file() or not stock_path.is_file():
            raise ValueError("active-route goal-budget worker evidence is missing")
        if not correctness_artifact.is_dir():
            raise ValueError("active-route goal-budget correctness evidence is missing")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        stock = json.loads(stock_path.read_text(encoding="utf-8"))
        input_tokens = int(candidate.get("input_token_count") or 0)
        if input_tokens not in (32768, 65536, 131072):
            raise ValueError(
                "active-route goal-budget candidate is not a priority bucket"
            )
        if int(stock.get("input_token_count") or 0) != input_tokens:
            raise ValueError(
                "active-route goal-budget candidate/stock buckets differ"
            )
        candidate_all_walls = [
            float(value) for value in (candidate.get("decode_wall_ms") or [])
        ]
        stock_all_walls = [
            float(value) for value in (stock.get("decode_wall_ms") or [])
        ]
        candidate_intervals = int(
            candidate.get("decode_measured_token_count") or
            len(candidate_all_walls))
        stock_intervals = int(
            stock.get("decode_measured_token_count") or len(stock_all_walls))
        if (
            not candidate_all_walls or not stock_all_walls or
            candidate_intervals <= 0 or stock_intervals <= 0
        ):
            raise ValueError("active-route goal-budget wall samples are incomplete")
        candidate_total_ms = float(
            candidate.get("decode_total_ms") or sum(candidate_all_walls))
        stock_total_ms = float(
            stock.get("decode_total_ms") or sum(stock_all_walls))
        candidate_complete_ms = candidate_total_ms / candidate_intervals
        stock_complete_ms = stock_total_ms / stock_intervals
        candidate_tail_ms = statistics.median(candidate_all_walls[16:])
        stock_tail_ms = statistics.median(stock_all_walls[16:])
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        decode_floor = float(
            acceptance["bootstrap_targets"]["decode_tokens_s"][
                str(input_tokens)
            ]
        )
        absolute_cap_ms = 1e3 / decode_floor
        ratio_cap_ms = stock_complete_ms / 1.10
        effective_cap_ms = min(absolute_cap_ms, ratio_cap_ms)
        remaining_ms = max(0.0, candidate_complete_ms - effective_cap_ms)
        goal_budget = {
            "source_artifact": str(candidate_path.parent.relative_to(REPO)),
            "correctness_artifact": str(correctness_artifact.relative_to(REPO)),
            "lane": {
                "input_tokens": input_tokens,
                "measured_output_tokens": candidate.get("output_tokens"),
                "measured_decode_intervals": candidate_intervals,
                "required_product_output_tokens": CORE_PRODUCT_OUTPUT_TOKENS,
                "diagnostic_only": (
                    candidate.get("output_tokens") != CORE_PRODUCT_OUTPUT_TOKENS
                ),
                "decode_walls_excluded_from_goal": 0,
                "stable_tail_diagnostic_excludes": 16,
            },
            "per_token_ms": {
                "candidate_complete_wall": round(candidate_complete_ms, 6),
                "stock_complete_wall": round(stock_complete_ms, 6),
                "candidate_stable_tail_median": round(candidate_tail_ms, 6),
                "stock_stable_tail_median": round(stock_tail_ms, 6),
                "absolute_cap": round(absolute_cap_ms, 6),
                "same_run_1p10_cap": round(ratio_cap_ms, 6),
                "effective_cap": round(effective_cap_ms, 6),
                "remaining_cut": round(remaining_ms, 6),
                "remaining_total_cut": round(
                    remaining_ms * candidate_intervals, 6
                ),
            },
            "verdict": {
                "floor_tps": decode_floor,
                "current_complete_tps": round(1e3 / candidate_complete_ms, 6),
                "wall_cap_ms_per_token": round(effective_cap_ms, 6),
                "can_reach_floor_without_more_wall_cut": remaining_ms == 0.0,
                "min_complete_wall_cut_ms_needed": round(remaining_ms, 6),
                "min_complete_wall_cut_pct_needed": round(
                    remaining_ms / candidate_complete_ms, 6
                ),
            },
            "candidate_git_commit": route_budget_source.get(
                "candidate_git_commit"
            ),
            "note": route_budget_source.get("note"),
            "regenerate_via": "tools/intel-qwen36-frontier-sync.py",
        }

    # Structural axis "advancing" must mean the deepest boundary closed a NEW layer
    # SINCE the goal last improved — not merely ">0" (true forever once layer 1
    # closes, so it wrongly painted the native-shape freeze as a benign highspeed
    # within-route conversion and let the stall-gate wave the build through).
    # Pure function of the census + best_ts so `--check` stays idempotent.
    if best_ts:
        deepest_before = max(
            (lyr for lyr, t in cs["gpu_layer_ts"] if (t is None or t <= best_ts)),
            default=0,
        )
        structural_advancing = cs["deepest_layer_closed"] > deepest_before
    else:
        structural_advancing = cs["deepest_layer_closed"] > 0

    # ch.3 §3.5 / working-discipline: a hard stall clears only via a RECORDED,
    # gated review keyed to the exact stall point ("re-defining the finish line
    # must be a recorded, gated decision"). A review in routes-ledger covers the
    # current stall iff its best_ts matches; a new best that stalls again needs a
    # fresh review. This is what the stall-gate checks before blocking.
    reviews = routes.get("goal_stall_reviews", []) or []
    review_recorded = bool(best_ts) and any(
        isinstance(r, dict) and r.get("best_ts") == best_ts for r in reviews
    )

    return {
        "schema": "intel-qwen36-frontier-v1",
        "workstream": WS,
        "stage": (
            "OpenVINO U4 specialization %s; no promoted performance candidate. "
            "R0 closed." % active.get("status")
            if openvino_product_active else
            "Long-context native CPU AVX2/F16C FP16-KV GQA component gate; no "
            "promoted runtime. R0 closed."
            if active.get("family") == "long_context_cpu_vector_gqa_decode" else
            "Long-context semantic-state import correctness gate; no promoted "
            "runtime. R0 closed."
            if active.get("family") == "long_context_semantic_state_provisioning" else
            "Long-context block32-INT8-KV GQA confidence-bound component gate; no "
            "promoted runtime. R0 closed."
            if active.get("family") == "long_context_compressed_kv_gqa_decode" else
            "Long-context product paged-GQA offline-provider source rate "
            "gate; no promoted runtime. R0 closed."
            if active.get("family") == "long_context_paged_gqa_provider_codegen" else
            "Long-context optimized-SDPA offline-provider native component "
            "gate; no promoted runtime. R0 closed."
            if active.get("family") == "long_context_sdpa_provider_codegen" else
            "Long-context fused-GQA FP16-KV decode component gate; no "
            "promoted runtime. R0 closed."
            if active.get("family") == "long_context_full_attention_decode" else
            "Owner contract decision after bounded architecture closure; no "
            "promoted runtime. R0 closed."
            if active.get("family") == "owner_contract_decision" else
            "Measured 1.10x resident native integration gate; no promoted "
            "runtime. R0 closed."
            if active.get("family") == "measured_1p10_dual_phase_native" else
            "GPU+NPU exact product-component feasibility; no promoted runtime. "
            "R0 closed."
        ),
        "generated_by": "tools/intel-qwen36-frontier-sync.py",
        "derived_from": [
            "doc/active/%s/routes-ledger.json" % WS,
            "doc/active/%s/rejected-routes.json" % WS,
            "output/ census (gpu-* + r2-gpu-* dirs + explore-log.jsonl)",
        ],
        "note": (
            "Tier-2 machine state (ch.3 §3.4). Do NOT hand-edit; regenerate via "
            "frontier-sync. Tier-3 prose pointer is current-frontier.md; raw is "
            "output/. routes-ledger.json/rejected-routes.json remain the "
            "hand-authored attack/closed boards; this file derives the no-progress "
            "signal they could not."
        ),
        "active_route": {
            "id": active.get("id"),
            "family": active.get("family"),
            "backend": active.get("backend"),
            "status": active.get("status"),
        },
        "goal_anchor": {
            "metric": GOAL_METRIC,
            "same_host_vulkan_floor_tps": SAME_HOST_VULKAN_FLOOR_TPS,
            "diagnostic_short_decode_floor_tps": DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS,
            "core_product_matrix": {
                "input_buckets": CORE_PRODUCT_INPUT_BUCKETS,
                "output_tokens": CORE_PRODUCT_OUTPUT_TOKENS,
                "minimum_openvino_speedup_ratio": 1.1,
                "minimum_openvino_speedup_ratio_applies_to_buckets": [
                    32768, 65536, 131072
                ],
                "regression_guard_minimum_ratio": 0.98,
                "regression_guard_input_buckets": [2048, 4096, 8192, 16384],
                "both_prefill_and_decode_required": True,
                "route_selection_priority_input_buckets": [32768, 65536, 131072],
                "short_diagnostic_can_satisfy_product": False,
                "exact_bucket_specialization_allowed": True,
                "short_candidate_may_retain_stock_sdpa": True,
                "component_no_slower_each_short_bucket_required": False,
                "performance_contract": (
                    "doc/adr/0075-make-long-context-win-the-product-target.md"
                ),
            },
            "product_runtime": {
                "candidate": "OpenVINO GPU plus custom OpenCL operations",
                "correctness_reference": "isolated stock OpenVINO U4",
                "performance_denominator": "isolated same-run stock OpenVINO",
                "legacy_gguf_native_role": "diagnostic_only",
                "contract": "doc/adr/0070-adopt-openvino-u4-specialization-runtime.md",
            },
            "cpu_native_denominator_tps": CPU_NATIVE_DENOMINATOR_TPS,
            "current_best_tps": goal_best_tps,
            "best_artifact": goal_best.get("artifact") if isinstance(goal_best, dict) else None,
            "best_diagnostic_class": (
                goal_best.get("diagnostic_class") if isinstance(goal_best, dict) else None
            ),
            "best_correctness_mode": (
                goal_best.get("correctness_mode") if isinstance(goal_best, dict) else None
            ),
            "best_correctness_artifact": (
                goal_best.get("correctness_artifact") if isinstance(goal_best, dict) else None
            ),
            "best_distribution_max_kld": (
                goal_best.get("distribution_max_kld") if isinstance(goal_best, dict) else None
            ),
            "best_distribution_top1_rate": (
                goal_best.get("distribution_top1_rate") if isinstance(goal_best, dict) else None
            ),
            "legacy_bringup_history_note": (
                (
                    "The current best is a full-real 40-layer Level Zero "
                    "backend. Its clean short 1k decode speed, three-case "
                    "eight-token consensus, and teacher-forced distribution "
                    "slice pass. ADR 0070 retains it only as legacy diagnostic "
                    "evidence. OV0 proves an OpenVINO custom-op mechanism, "
                    "seq802 locks the real GatedDeltaNet boundary oracle, and "
                    "seq803/804 prove exact one/all-layer GDN substitution. "
                    "Seq836 proves all-ten hot/cold full-attention semantics, "
                    "but its scalar carrier is much slower than stock. A "
                    "bounded tiled split improved diagnostic prefill but is "
                    "rejected because fanned-out input mutation left hot K/V "
                    "state zero and failed decode KLD. The active carrier must "
                    "have one graph-owned state owner per layer; no optimized "
                    "OpenVINO performance candidate exists yet."
                )
                if isinstance(goal_best, dict)
                and goal_best.get("diagnostic_class") ==
                    "full_real_40_layer_level_zero_decode"
                else (
                (
                    "Current checked GPU-hybrid decode-lane best still has CPU "
                    "prefill and remaining CPU-side decode work; %s. It %s "
                    "the interim %.1f tok/s same-host Vulkan decode floor but "
                    "remains a diagnostic short-context lane, not an "
                    "OpenVINO-derived product acceptance target. The unmasked rowblock16 "
                    "tree is product-rejected by the first short-prompt "
                    "acceptance subset; the 26-layer rowblock16 mask is the "
                    "current short bring-up floor-clear speed shape, but router math/code "
                    "distribution now blocks promotion. The router-math/code "
                    "block is isolated to double-SwiGLU plus prior "
                    "full-attention FFN residual-input correction: entry-only "
                    "qkv-weighted layer-input deltas pass, full-attention "
                    "residual-only shadow FFN input passes, norm-only fails, "
                    "and seed/lagged predictors fail. "
                    "Rejected explanations include rowblock, LM-head, "
                    "selected/shared combined toggles, deferred FFN-down "
                    "finish, QK-local CPU-shape recurrence, recurrent-only "
                    "state refresh, global OpenCL no-FMA, resident "
                    "linear-state host-sync/readback, the carrier loop, and "
                    "q4 CPU-order full-attention output projection. Seq318 "
                    "accepted the source-only full-attention residual product "
                    "scaffold, seq319 target-compiled it, seq320 proved "
                    "the guard blocks before token execution, and seq321 "
                    "accepted product-owned residual source wiring. Seq322 "
                    "target-compiled the implementation, and seq323 passed "
                    "the one-token counter probe. Seq324 passed the 8-token "
                    "decode gate; seq325 rejected source-only router "
                    "distribution despite clean counters; seq326..329 "
                    "accepted product-consumer source/compile/probe/decode, "
                    "seq330 rejected its router distribution, seq331 "
                    "measured the selected residual value gap, seq332 "
                    "isolated that gap upstream of selected full-attention "
                    "layer input, seq333 accepted a source-only layer-input "
                    "product scaffold, seq334 target-compiled it on Arc "
                    "B390, seq335 proved the source-only guard blocks before "
                    "tokens, seq336 accepted product-owned source wiring, "
                    "seq338 exposed the missing handle-retention path, "
                    "seq339..342 fixed/probed/decoded it with clean "
                    "counters, seq343 rejected source-only router "
                    "distribution, seq344..347 productized the consumer, "
                    "seq348 rejected its router distribution, seq349 "
                    "observed live selected layer-input value drift, and "
                    "seq350..526 split the root to nested producer linear "
                    "input-source, source FFN-input, attention-output, then "
                    "source linear input-source gap to source FFN-input and "
                    "layer-input drift, preceding linear output drift, and "
                    "preceding linear input/source, producer FFN-input, and "
                    "producer live linear input/source and FFN/attention-output/input-source/layer-output drift to preceding linear input-source, FFN input, linear input/source, source FFN input, source attention output, source linear delta/z, source linear input/source, source FFN-input, source layer input, preceding linear output, delta/z, live input, value source, producer FFN input, producer delta/z, producer live input, producer input source, source FFN input, source attention output, source linear delta/z, source linear input, source linear input source, source FFN input, source layer input, preceding linear output, preceding linear delta/z, preceding linear input, preceding linear input source, producer FFN input, producer linear delta/z, producer linear input, producer linear input source, source FFN input, source attention output, source linear final-mix delta/z, source linear input, source linear input source, source FFN-input, source layer-input, previous layer-output, previous FFN-delta, previous FFN-norm, previous attention-output projection math, Q8 projection bridge, Q8 input sensitivity, coupled linear delta/z, live linear input, previous FFN delta, selected/shared FFN fan-out, selected SwiGLU, gate-up input sensitivity, FFN-norm input sensitivity, attention-output source, projection-Q8 input sensitivity, coupled linear delta/z, live linear input, previous FFN delta source, selected/shared FFN fan-out material drift, selected gate-up input sensitivity, FFN-norm input sensitivity, attention-output source, projection-Q8 input sensitivity, linear-z sensitivity, z-source attribution, GPU attention-norm math attribution, shared RMSNorm scale-kernel attribution, CPU-sqrt shared-scale probe coverage, shared scale-kernel reduction-order attribution, serial-scale unresolved attribution, post-scale router distribution failure, accepted-contiguous2 product-baseline distribution failure, reentry to the known selected layer-input source-value gap, route-control to the current-token qkv-delta recursion-break design gate, selection of the all-30 current-token qkv-column block-q16 source contract, and source/generate-only acceptance, target compile, and source-only guard probe of that contract. "
                    "Seq511 product-wired the block-q16 source with the resident F32 additive overlay runner, counters, and local generated+engine compile. "
                    "Seq512 target-compiled it, seq513/514 proved all-30 counter coverage with zero misses, seq515 rejected router distribution with math/code KLD `0.01629065168` / `0.1067056391`, seq516 classified the product overlay as a zero-delta no-op, seq517 closed that value-source route pending a new non-shadow proof, seq518 selected all-linear state product-source feasibility from route-switch evidence, seq519 rejected direct all-linear state refresh as a product source, seq520 closed the known conv-history/upstream product-source board pending a new non-shadow source class, seq521 kept the acceptance-matrix KLD ruler while selecting a bounded FP64/numerical sensitivity gate, seq522 rejected the FP64 precision pack after it regressed router math/code, seq523 rejected global affine logit calibration after full-vocab anatomy showed math/code max affine KLD `0.02981610327` / `0.01605870478`, seq524 rejected tail-mass attribution after failed-step KLD concentrated in high-probability head-token pairs, seq525 rejected LM-head projection placement after final-norm CPU projection explained the failed-pair logit gaps, and seq526 rejected output RMS scale drift after raw final-residual value deltas dominated the final-norm dimension deltas. "
                    "Next: "
                    "router_prompt_distribution_final_residual_delta_dimension_source_gate."
                    % (
                        correctness_note,
                        "clears" if goal_best_tps >= SAME_HOST_VULKAN_FLOOR_TPS
                        else "is below",
                        SAME_HOST_VULKAN_FLOOR_TPS,
                    )
                )
                if isinstance(goal_best, dict)
                else (
                    "No GPU decode-lane run exists yet: every bring-up gate is a "
                    "captured single-token teacher-forced boundary, explicitly "
                    "'does not prove decode/token/throughput'. The first goal "
                    "(reach/beat %.1f tok/s) needs an assembled decode loop "
                    "(R2), not more per-boundary gates." % SAME_HOST_VULKAN_FLOOR_TPS
                )
                )
            ),
            "note": (
                "The prior 19.5 tok/s Vulkan bring-up floor is retired. The "
                "49.8 tok/s 1k floor and current best are retained only as "
                "diagnostic short-context history. ADR 0051 keeps the seven-row "
                "2k-128k/output-512 matrix; ADR 0070 changes the candidate to an "
                "OpenVINO GPU specialization against isolated stock OpenVINO. "
                "STATUS and the active route ledger hold its current gate."
            ),
        },
        "bringup_progress": {
            "axis": "structural — deepest per-layer/boundary teacher-forced gate closed",
            "deepest_layer_closed": cs["deepest_layer_closed"],
            "gpu_probe_runs": cs["gpu_probe_runs"],
            "advancing": structural_advancing,
            "interpretation": (
                "The short decode bring-up floor is closed but is outside the "
                "core product matrix. Correctness evidence can close a gate "
                "without moving the seven-row long-context product axis."
            ),
        },
        "no_progress": {
            "axis": "diagnostic legacy short-decode anchor; not product completion",
            "definition": (
                "token-emitting GPU runs (r2-gpu-* dirs + explore-log rounds, "
                "pass or fail) stamped after the last SIGNIFICANT legacy "
                "effect-band improvement of the goal metric"
            ),
            "runs_since_goal_improved": runs_since_goal,
            "runs_since_significant_improvement": runs_since_significant,
            "noise": noise,
            "last_significant_improvement": sig_events[-1] if sig_events else None,
            "soft_reflection_threshold": SOFT_REFLECTION_THRESHOLD,
            "hard_stall_threshold": HARD_STALL_THRESHOLD,
            "soft_reflection_breached": soft,
            "hard_stall_breached": hard,
            "glide_slope": glide,
            "structural_axis_advancing": structural_advancing,
            "review_recorded_for_current_best": review_recorded,
        },
        "goal_budget": goal_budget,
        "goal_history": {
            "significant_improvements": len(sig_events),
            "recent": sig_events[-12:],
            "definition": (
                "legacy empirical-effect-band improvement chain of the goal "
                "metric (correctness-checked decode rows only); product and "
                "component promotion use confidence bounds"
            ),
        },
        "ledger": {
            "candidates": len(cand_hist),
            "sub_threshold_candidates": sum(1 for c in cand_hist if c.get("sub_threshold")),
            "switch_decisions": len(switches),
            "rejected_routes": len(rejected.get("rejected", []) or []),
            "rejected_classes": len(rejected.get("rejected_classes", []) or []),
            "parked_alternates": len(parked),
        },
        "pre_registered_alternatives": "doc/active/%s/routes-ledger.json#parked_routes" % WS,
        "controls": {
            "stall_gate": "tools/intel-qwen36-stall-gate.py",
            "code_volume_gate": "tools/intel-qwen36-code-volume-check.py",
            "doc_discipline_gate": "tools/validate_repo.py (check_doc_discipline)",
            "decode_budget": "tools/iq36_budget.py (kill-number, ch.2 §2.1 #1)",
        },
        "explore": {
            "log": "output/explore-log.jsonl",
            "rounds": cs["explore_runs"],
            "best_explore_tps_unverified": cs["best_explore_tps_unverified"],
            "latest_profile_budget": latest_explore_profile_budget,
            "note": (
                "explore rounds are artifact-free (one JSONL line, ch.2 §2.2); "
                "they count toward the stall counters but can never set the best "
                "— promote a promising config by re-running without --explore"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="CI mode: exit non-zero if frontier.json is stale vs the ledgers/census")
    args = ap.parse_args()

    state = build_state()
    rendered = json.dumps(state, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != rendered:
            print("frontier.json is stale; run frontier-sync to regenerate", file=sys.stderr)
            return 1
        print("frontier.json is in sync with the ledgers + output census")
        return 0

    OUT.write_text(rendered, encoding="utf-8")
    np = state["no_progress"]
    bp = state["bringup_progress"]
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  active route: {state['active_route']['id']} ({state['active_route']['backend']})")
    print(f"  goal anchor ({GOAL_METRIC}): best {state['goal_anchor']['current_best_tps']} tok/s "
          f"(diagnostic short floor {DIAGNOSTIC_SHORT_DECODE_FLOOR_TPS}, prior Vulkan floor "
          f"{SAME_HOST_VULKAN_FLOOR_TPS}, cpu denom {CPU_NATIVE_DENOMINATOR_TPS})")
    print(f"  structural: deepest layer closed {bp['deepest_layer_closed']}, "
          f"{bp['gpu_probe_runs']} GPU probe runs")
    print(f"  no-progress runs since SIGNIFICANT goal improvement: "
          f"{np['runs_since_significant_improvement']} (raw since any best: "
          f"{np['runs_since_goal_improved']}; legacy effect band {np['noise']['rel']*100:.2f}%; "
          f"soft {np['soft_reflection_threshold']} / hard {np['hard_stall_threshold']})")
    glide = np["glide_slope"]
    if glide["trailing_rate_tps_per_run"] is not None:
        proj = glide["projected_runs_to_floor"]
        print(f"  glide-slope: {glide['trailing_rate_tps_per_run']:+.4f} tok/s per run over "
              f"last {glide['window_runs']} runs -> floor in "
              f"{proj if proj is not None else 'inf'} runs"
              f"{'  ** BEYOND HORIZON — reflect on direction **' if glide['breached'] else ''}")
    budget = state.get("goal_budget")
    if budget:
        v = budget["verdict"]
        if v.get("can_reach_floor_without_kernel_work") is False:
            print(f"  budget kill-number: overhead-only ceiling "
                  f"{v['overhead_only_ceiling_tok_s']} tok/s < floor — kernel-side "
                  f"work required (kernels must shrink >= "
                  f"{v['min_kernel_time_cut_pct_needed'] * 100:.2f}%)")
    if np["hard_stall_breached"]:
        print("  ** GOAL hard-stall breached — run stall-gate; assemble the R2 decode loop **")
    elif np["soft_reflection_breached"]:
        print("  * GOAL soft reflection breached — still on the right path? (see stall-gate) *")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
