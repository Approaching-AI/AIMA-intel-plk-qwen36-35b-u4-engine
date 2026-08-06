# ADR 0062: Replace the fixed noise veto and reopen INT8 GQA

Date: 2026-07-13

## Status

Accepted by owner. This supersedes the measurement-rule portion of ADRs
0053-0061. It does not change the model, hardware, correctness, component caps,
or the `1.10x` product target.

## Context

The previous component gates jointly required repeat and confirm medians to
clear the rate cap and differ by at most `0.5%`. That last condition does not
test the proposition the gate needs to establish: it ignores sample count and
within-lane distributions, and it can reject a component with material rate
margin because two medians straddle an arbitrary cutoff.

Seq778 exposed the defect. Its two seven-sample medians are `2.459061` and
`2.471978 ms` against a `2.825 ms` cap, while their spread is `0.522537%`.
The source passed cosine (`0.999999971822`) and relative L2
(`0.000324519358`). It was rejected solely because the median spread exceeded
`0.5%` by `0.022537` percentage points.

## Decision

Performance promotion now uses one-sided 95% confidence bounds:

- product throughput uses at least eight interleaved ABBA paired blocks and
  passes only when the lower confidence bound of native/OpenVINO throughput
  clears the required ratio;
- component latency uses at least twenty samples and passes only when the
  upper confidence bound of latency clears its registered cap;
- the canonical implementation is a deterministic 20,000-resample percentile
  bootstrap of the median in `tools/iq36_perf_inference.py`.

Repeat/confirm median spread is retained only as telemetry. Robust CV
(`1.4826 * MAD / median`) classifies `<=1%` as normal, `1%-2%` as a request for
more samples, and `>2%` as a request to inspect environment telemetry. None is
a performance veto by itself.

The owner-directed contract change reopens only the block32-INT8 GQA route.
It was the fastest and most accurate compressed-KV candidate and was closed
solely by the superseded rule. E4M3 remains closed because it has the same
traffic class while being slower and less accurate than INT8. INT6, scalar,
XMX, provider, CPU, NPU, and prefill closures remain rate or accuracy closures.

## Consequences

- Re-run the unchanged INT8 component with twenty samples under the new
  inference gate. A confidence-bound pass admits 32k/64k/128k packed-backend
  integration guards; it is not a token-loop or product speedup claim.
- Historical artifacts and ADRs keep their original recorded outcomes. New
  decisions cite this ADR rather than rewriting old evidence.
- No target has been lowered: every product bucket and phase still requires
  `max(absolute floor, 1.10x same-run OpenVINO)` and all correctness gates.
