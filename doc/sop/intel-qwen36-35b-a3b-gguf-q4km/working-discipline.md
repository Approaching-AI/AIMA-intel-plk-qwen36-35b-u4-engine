# SOP — working discipline: thin docs + collapse ritual

Snapshot: 2026-07-02

This SOP makes two factory disciplines enforceable instead of aspirational:
the three-tier memory (methodology ch.3 §3.4) and the collapse ritual
(methodology ch.2 §2.2). Both failed the same way on 2026-06-28: the docs were
distilled in the morning and re-bloated by night (STATUS back to 285 lines,
meta-log 158KB), because the stop rule was prose, not a gate. The fix is
harness-enforced gates plus a short habit list.

## Three-tier memory

| Tier | Where | Rule |
|---|---|---|
| 1 — disposable raw | `output/<name>-<UTC>/` | Name is the index (long, greppable). Contents deletable any time. **Exploration does not land here** — only promotion candidates do. |
| 2 — machine frontier | `frontier.json` (derived by `tools/intel-qwen36-frontier-sync.py`), `routes-ledger.json`, `rejected-routes.json`, profile JSON in `output/` | `frontier.json` holds the **goal-axis no-progress counter + stall flags** derived from the ledgers + `output/` census — the signal the hand-authored boards could not produce. ns numbers live here, not in prose. Do NOT hand-edit `frontier.json`; regenerate it. |
| 3 — single pointer | `current-frontier.md` + `STATUS.md` + `meta-log/` | `current-frontier.md` = first-read Tier-3 pointer. STATUS = current state only. meta-log = thin changelog (story), not a data dump. |

## STATUS rules

- Overwrite in place. It answers only "where are we now + next action".
- Hard cap: **≤ 120 lines** (`validate_repo.py` fails above this).
- Never put rejected-route detail, per-tensor ns tables, or append-only history
  in STATUS. Rejected routes → `rejected-routes.json`. History → `meta-log/`.
- Do not copy STATUS state into goals, plan, contracts, or READMEs.

## rejected-routes.json rules

- Every closed/rejected route gets one entry: `route`, `class`, `reason`,
  `evidence`. Group recurring rejections under `rejected_classes` so a whole
  class (e.g. compute-side dot variants, thread-count sweeps) is closed at once.
- `validate_repo.py` fails if the ledger is missing, mis-schema'd, or empty.
- Before running any candidate, check it is not in the ledger or a rejected
  class. Re-opening a class requires a new rationale recorded here.

## Routes ledger + direction trigger (ch.3 §3.5)

`routes-ledger.json` holds the **active route**, the **pre-registered parked
alternates** (2-3, ranked), and a **candidate_history**. It is the "where is the
attack going" board, the way `rejected-routes.json` is the "what is closed" board.

- Every R3+ candidate gets one `candidate_history` entry: `seq`, `route_family`,
  `disposition`, and `sub_threshold` (true when it lands default-off AND decode
  gain <3% — it does not move the dominant gap; see bandwidth-roofline-reject).
- **Direction trigger (harness-enforced):** if one `route_family` accrues ≥2
  consecutive `sub_threshold` candidates, `validate_repo.py` fails until a
  `switch_decision` covering the family's latest `seq` is recorded. This is the
  ch.3 §3.5 soft reflection ("am I still on the right path?") turned into a gate,
  so micro-tuning on a low-ceiling lane cannot continue silently.
- **The escape is to switch, not to argue:** pop the active route, push the
  highest-rank `parked_route`. Escalate to the user only when the next step is a
  genuine user call (target/backend choice, ch.1) — and record that as the
  switch_decision.

## Goal-axis stall controller (ch.3 §3.5 D — harness-enforced)

The direction trigger above watches R3+ *candidates*. The stall controller watches
the *goal metric* directly, catching the highspeed/native death the candidate
ledger cannot see (a correctness tar-pit logs no candidates).

- `frontier-sync.py` computes `runs_since_goal_improved` = token-emitting GPU runs
  (`r2-gpu-*`, pass **or** fail) stamped after the current best decode tok/s. It
  resets only when the goal actually improves. (The original formula was
  `gpu_probe_runs if goal_decode_runs == 0 else 0` — it latched to 0 the instant
  any decode run existed, so a 60+ run L38/L39 correctness tar-pit read as "0
  stall" and the gate could never fire. Fixed 2026-07-02.)
- `structural_axis_advancing` = the deepest boundary closed a **new** layer since
  the goal last improved (not merely ">0", which is true forever once layer 1
  closes).
- `stall-gate.py`: soft (≥30) prints a non-blocking reflection; **hard (≥50)
  BLOCKS** (`validate_repo.py` runs it with `fail=True`). `structural_advancing`
  only selects the remedy message (highspeed = assemble the loop; native = check
  the ruler then switch), it never waves the build through.
- **The only escape from a hard block is a recorded, keyed review** in
  `routes-ledger.json#goal_stall_reviews`: `[{"best_ts": <current best ts>, ...}]`.
  It clears the gate for that exact stall point; a new best that stalls again needs
  a fresh review. This is "re-defining the finish line must be a recorded, gated
  decision" made mechanical — not prose the agent writes for itself.
- **Ruler reminder** (the native death in a new costume): the promotion bar is
  §1.5 (cosine≥0.999 / KLD<0.005 / top-1≥0.99), **NOT** exact-top-k over free-run
  tokens. Greedy decode depends only on top-1; greedy top-1 exact across the
  teacher-forced set IS the deterministic-token pass. Check the ruler before
  sinking runs into a boundary.

## meta-log rules

- A daily note is a **changelog**: what was closed today, what was learned, the
  next step. Past tense, narrative, short.
- **Do not hand-copy ns profile tables, frontier dumps, or full artifact path
  lists into the note.** Those live in `output/` (Tier 1) and the ledger (Tier 2).
  Reference an artifact by path; do not transcribe its contents.
- Soft cap: **40KB/day** (`validate_repo.py` warns above this). The 2026-06-28
  campaign hit 158KB by transcribing 700+ ns numbers — that is the anti-pattern.

## Collapse ritual (exploration is cheap and leaves no trace)

- The hot inner loop is: change one kernel/op → rebuild that one TU → verify
  against the resident oracle in seconds → record the result in the frontier
  (in memory / one JSON), **not** a new `output/` tree.
- A single logical experiment must **not** spawn 3 output dirs + 6 doc files.
  That churn (intel-box's main waste) is what the resident harness exists to remove.
- Only a **promotion candidate** walks the full artifact contract and lands in
  `output/`. Everything before promotion stays ephemeral.
- **Do not write a new python/cpp file per kernel variant.** Reuse one
  parameterized compare runner driven by `engine/boundaries.json` and CLI flags.
  See `doc/sop/intel-qwen36-35b-a3b-gguf-q4km/parameterized-compare-runner.md`.
- Code-volume / build-time is a stop trigger (ch.0 §0.3): if `tools/`, `engine/tests/`,
  or `output/` start growing one-file-per-variant, stop and generalize.

## Enforced gates (`tools/validate_repo.py` → `check_doc_discipline`)

| Gate | Action |
|---|---|
| STATUS.md > 120 lines | **fail** |
| STATUS.md > 3 `N/8` per-token results or > 12 `output/` refs | **fail** (substance gate — the line cap let a 63-fraction run-on paragraph pass) |
| current-frontier.md > 8KB, any `N/8` line, or > 12 `output/` refs | **fail** (pointer-shape gate — it rode just under 40KB at 39.9KB as a lab notebook) |
| rejected-routes.json missing / mis-schema / empty | **fail** |
| routes-ledger.json missing / mis-schema / no parked_routes | **fail** |
| route family with ≥2 consecutive sub-threshold candidates, no switch_decision | **fail** (direction trigger) |
| **goal-axis HARD stall (`runs_since_goal_improved` ≥ 50) with no keyed `goal_stall_reviews` entry** | **fail** (`stall-gate.py`, now blocking; soft ≥ 30 = non-blocking reflection) |
| **any `doc/active/*.md` (except STATUS) > 40KB** | **fail** (was the escape hatch the 124KB GPU doc used) |
| **`frontier.json` missing / stale vs ledgers+census** | **fail** (`frontier-sync.py --check`) |
| **`tools/*-probe.py` / `*-compare.py` / `engine/tests/*_compare.cpp`, or the decode-smoke runner's `add_argument` count, grew past frozen ceiling** | **fail** (`code-volume-check.py` ratchet) |
| **newest `meta-log/20*.md` > 60KB** | **fail** (current-day hard cap; older files grandfathered) |
| any `meta-log/20*.md` > 40KB, or any source > 1500 lines | **warn** |

Run `python3 tools/validate_repo.py` before committing docs (it now invokes
`frontier-sync --check`, the code-volume ratchet, and the stall-gate). After a
route/candidate change, regenerate machine state first:
`python3 tools/intel-qwen36-frontier-sync.py`.
