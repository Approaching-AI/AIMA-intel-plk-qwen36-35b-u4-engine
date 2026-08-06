# output

Promotion artifacts and diagnostic run outputs belong here.

Each promoted output directory must include:

- `manifest.json`
- `metrics.jsonl`
- `correctness.json`
- `smoothness.json` when context-ladder behavior is claimed
- `summary.md`

Bulky raw payloads should be added intentionally.

Small target-denominator, terminal-promotion, or route-disposition bundles may
be tracked intentionally when an authoritative contract or status board links
to them. Raw captures and the wider experiment corpus remain local by default.
