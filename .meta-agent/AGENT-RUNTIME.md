# Meta Agent Runtime Guide

This file is the fast path for agents running in a project that integrates `meta-agent`.

Read this file first. Only open `meta-agent/doc/methodology.md` when you need rationale, examples, or help with an edge case.

If you were launched by an automation driver, also follow the driver prompt. In that mode, the driver input is part of the task context, and you must write the expected session result file before ending the session.

## Session Start

Preferred helper:

```bash
bash meta-agent/scripts/session-start.sh
```

Useful variants:

```bash
bash meta-agent/scripts/session-start.sh --claim-first
bash meta-agent/scripts/session-start.sh --claim <pending-file-or-basename>
```

The underlying SOP is:

1. Check the latest `meta-log/`, the files directly referenced by the current task, and any explicit driver input.
2. If there is already a clear task entry point, continue from there.
3. If there is no clear entry point, wait for user instructions.

Notes:

- Some automation setups still provide a queued file under names like `handoff/`; treat it as one possible task entry point, not as the core workflow.

## Session End

Run this flow when the human explicitly asks to end the session.

Preferred helper:

```bash
bash meta-agent/scripts/session-end.sh
```

Useful variants:

```bash
bash meta-agent/scripts/session-end.sh --append-daily --operator <name>
```

These helpers only prepare files and deterministic updates. The agent still decides what summary to write, what the next step is, and when to commit or push.

1. Append a summary to today's daily notes.
2. Include `operator: <name>` at the start of the session entry. If unknown, ask.
3. Record what was done, important conclusions, open problems, and the next step.
4. Commit and push the repository so notes, docs, and code are persisted.
5. Decide whether another agent session can continue the remaining work.
6. If yes, leave a clear next-step entry in `meta-log/` or the active goal/task file.
7. Only create a separate handoff file when a specific automation flow truly needs one.
8. If the work is complete or requires human input next, say so in daily notes.

If you are in autonomous mode, run this same flow before writing the final `session-result.json`. Use the operator value provided by the driver when present, and treat any driver-managed handoff directory as implementation detail rather than the main abstraction.

If the automation setup provides both `questions/` and `answers/` directories:

- ask humans by writing files under `questions/`
- consume human replies from `answers/`
- use the same basename for a question and its answer
- check for a matching answer before asking the same question again or declaring a hard block

The session-result file is the final driver-facing acknowledgement after this flow is complete. It is not a substitute for the flow.

## When To Read More

Open `meta-agent/doc/methodology.md` only if you need one of these:

- the reasoning behind the workflow
- examples of prompts or continuity patterns
- clarification for an unusual case not covered here
