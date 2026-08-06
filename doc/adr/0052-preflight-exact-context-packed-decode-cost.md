# ADR 0052: Preflight exact-context packed decode cost

Date: 2026-07-13

## Status

Accepted as an evidence-only gate. No product row or speedup is admitted.

## Context

Seq769 closes the seven-bucket OpenVINO denominator. The current native decode
frontier, seq743/744, measures the real 40-layer packed Level Zero schedule but
feeds only a short token history. Its comparison to the old 1k target cannot
establish the cost of reading and applying 2k-128k full-attention KV history.

Running 512 native tokens blindly is also the wrong first unit: the current
full-attention apply kernel is serial in context length for each output value.
At the largest buckets, one exact-context token can derive the kill-number
before committing hours to a row that is arithmetically unable to approach the
target.

## Decision

Extend the existing packed Level Zero smoke with one performance-only mode and
run it at exact contexts `2k/4k/8k/16k/32k/64k/128k`:

- reserve device state for the full 512-token continuation;
- execute one measured token per bucket after one warmup;
- use the real 677-kernel, 40-layer backend and real model weights;
- zero-initialize recurrent/KV state and label semantic correctness not
  applicable;
- record wall/device time, resident state/weight bytes, target ratio, and the
  constant-cost 512-token projection;
- prohibit speedup and product claims regardless of timing.

This is a cost preflight, not a substitute for exact prompt state, 512 measured
tokens, prefill, sentinel correctness, or smoothness. If any priority bucket
(`32k/64k/128k`) is below `0.80x` of its decode target, do not launch its full
512-token row. Profile the worst priority bucket and select a materially
different context-attention algorithm. If every priority bucket is at least
`0.80x`, run the full output512 ladder next.

## Consequences

- Seq743/744 remain diagnostic short-history evidence only.
- The new mode may not alter the accepted default smoke path.
- A passing mechanism gate proves only that the kill-number is measured cleanly.
- Long-context semantic state still requires a real native prefill route or an
  independently validated state import before product promotion.
