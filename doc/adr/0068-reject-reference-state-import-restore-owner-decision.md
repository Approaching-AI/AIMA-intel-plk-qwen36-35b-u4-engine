# ADR 0068: Reject reference-state import and restore owner decision

Date: 2026-07-13

## Status

Accepted. The project goal and performance target remain active; the next gate
requires an owner contract decision or an independently verified new
capability.

## Context

ADR 0067 admitted one bounded audit of OpenVINO state import before any
semantic long-context run. Commit `148b6f7` adds a diagnostic-only
`--import-state` mode and a capture gate. Clean seq798 uses the locked
24-token `fresh_code_03` prompt, prefills 23 tokens, and then compares one
teacher-forced step.

The fixed mappings are derived from the graph and native source, not selected
from a transform sweep:

- conv `[1,8192,4]` drops the oldest time slot and becomes channel-major
  `[8192,3]`;
- recurrent `[1,32,key,value]` transposes to native
  `[32,value,key]`;
- full-attention K/V `[1,2,context,256]` transposes from head-major to
  token-major `[context,2,256]`.

The clean audit captures exactly 30 conv, 30 recurrent, ten K, and ten V
states. All 80 imported/native comparisons are finite, so the failure is not a
missing tensor, shape error, file truncation, or non-finite value. The state
families nevertheless show incompatible accumulated semantics:

| family | min / median cosine | max relative L2 |
|---|---:|---:|
| conv | `0.367252 / 0.488022` | `1.127563` |
| recurrent | `-0.000112 / 0.030468` | `1.422352` |
| K | `0.979677 / 0.984301` | `0.201869` |
| V | `0.964764 / 0.978045` | `0.264737` |

The terminal teacher-forced distribution triangle is:

| comparison | KLD | logits cosine | top-1 |
|---|---:|---:|:---:|
| CPU GGUF vs OpenVINO | `0.0140285` | `0.965815` | pass |
| CPU GGUF vs imported native | `13.2402172` | `-0.217252` | fail |
| OpenVINO vs imported native | `13.3032512` | `-0.245050` | fail |

The reference itself is already outside the locked `KLD <= 0.005` ruler, and
the imported native step is not close to either side. Its `19.841 ms` wall row
is diagnostic only: a numerically rejected state cannot support a speed or
product claim.

## Decision

Close `openvino_reference_state_import_layout_numeric_audit_v23`. Do not run a
state-axis/channel transform sweep, favorable-prompt rerun, or imported-state
`32k/64k/128k` sentinel. The current OpenVINO U4 state is not a correctness
oracle for the locked GGUF Q4_K_M runtime.

Preserve hot8192 as accepted zero-state decode-capacity evidence only. Restore
the owner-contract gate from ADRs 0048, 0050, and 0061 because:

- no semantic core long-context product row is now admissible;
- native prefill still has no complete source-derived route below the product
  cap; and
- the installed-capability audit found no unclaimed compiler or hardware
  capability that reopens those closures.

A successor requires either an owner-recorded named change to hardware,
model/precision, correctness, batch size, final runtime dependency, product
matrix, or OpenVINO speedup ratio, or a newly verified capability with a
complete bound below the relevant cap before implementation.

This is route exhaustion, not project completion, target reduction, or a
speedup claim.

## Consequences

- The semantic-state import route and its long-context sentinel successor are
  closed without re-litigating layouts.
- Seq793-795 remain valid zero-state correctness/capacity evidence, but cannot
  establish prompt attention or product acceptance.
- The next source edit is blocked on a named owner decision or genuinely new
  bounded capability; another current-source kernel would repeat a terminal
  route.

Evidence:

- `output/reference-state-import-20260713Tseq798cleanZ/`
- `doc/adr/0067-audit-reference-state-import-before-long-context.md`
- `doc/adr/0048-record-measured-1p10-prefill-route-exhaustion.md`
- `doc/adr/0050-close-prefill-only-gpu-npu-restore-owner-decision.md`
- `doc/adr/0061-record-long-context-native-route-exhaustion.md`

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
