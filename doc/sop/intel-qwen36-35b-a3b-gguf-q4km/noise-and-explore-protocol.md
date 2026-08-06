# Noise + explore-round protocol

Workstream: `intel-qwen36-35b-a3b-gguf-q4km`. Created 2026-07-03 after the
methodology audit found (a) accept/reject decisions being made on 0.2–0.3%
deltas while same-config repeats spread up to ~1%, and (b) every exploration
round paying a full artifact dir (~136KB–1.7MB) plus prose in three documents.

## Performance inference

Performance admission answers the target question directly. A fixed
repeat/confirm median-spread cutoff is not a promotion gate.

- Product comparisons run interleaved `OpenVINO -> native -> native ->
  OpenVINO` blocks on the same host and power state. Use at least eight paired
  blocks per bucket and phase. Promotion requires the one-sided 95% lower
  confidence bound of the paired native/OpenVINO median throughput ratio to
  clear the required ratio (`1.10x` for the hard target).
- Latency components use at least twenty timed samples. Promotion requires the
  one-sided 95% upper confidence bound of median latency to remain below the
  registered component cap.
- `tools/iq36_perf_inference.py` is the canonical deterministic implementation:
  20,000 percentile-bootstrap resamples with a recorded fixed seed.

Config identity remains result flags plus `source_sha` (engine sources and
generated TU). Same flags across a source edit are not the same config.

## Dispersion and environment health

Repeat/confirm spread and robust CV (`1.4826 * MAD / median`) remain telemetry:

- robust CV `<=1%`: normal;
- `1%-2%`: collect more samples;
- `>2%`: inspect power, thermal, frequency, driver scheduling, and background
  load before rerunning.

These labels do not override a confidence-bound performance result. A run is
invalidated only by recorded environment evidence, not by one dispersion
number. `frontier.json#no_progress.noise` is retained as a legacy search/stall
diagnostic for old scalar rows; it is not a product or component promotion
gate.

## Decision rules

1. **A confidence interval that crosses the decision boundary is
   inconclusive.** It gets no solo performance claim. Add samples or bundle
   related micro-cuts; do not select the favorable median.
2. **The legacy scalar stall counter is only a search heuristic.** It may use
   the empirical same-config spread stored in `frontier.json`; it cannot accept
   or reject a performance row.
3. **A new product best requires** paired interleaved blocks whose lower 95%
   confidence bound clears the target, plus the §1.5 correctness row.
4. **Before opening another overhead micro-cut**, check
   `frontier.json#goal_budget`: while `can_reach_floor_without_kernel_work` is
   false, overhead-only candidates are sub-threshold by arithmetic
   (`tools/iq36_budget.py <speed-artifact>` recomputes).

## Explore rounds (artifact-free exploration, ch.2 §2.2)

- Run `tools/intel-qwen36-r2-gpu-decode-smoke.py --explore --label <variant>`.
  The round appends one line to `output/explore-log.jsonl` (config_sha,
  source_sha, tps, checks) and writes NO artifact dir.
- Explore rounds count toward the stall census but can never set the best.
- To promote: re-run the same config WITHOUT `--explore` (full bundle), collect
  the paired inference blocks, and add the §1.5 correctness evidence per rule 3.
- Correctness-evidence lanes (`--distribution-ladder`, `--stream-token-jsonl`)
  always write full artifacts; `--explore` refuses them.
- The remote build/token cache is on by default: a flags-only round skips
  scp + g++ entirely (`--no-remote-cache` restores the legacy path).

## Recording

- Per-run records live in the machine layer (explore log, output/ dirs,
  ledgers). `meta-log/` records per-session decisions and conclusions only.
- Accepted cuts go to `accepted-cuts.json`; closed routes/classes to
  `rejected-routes.json`; neither is narrated in STATUS/current-frontier.
