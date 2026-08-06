# Claude Guidance

Use `AGENTS.md` as the source of truth for this repository.

Read and follow the runtime guide in `.meta-agent/AGENT-RUNTIME.md`.

Periodically check whether submodules have updates:

```bash
git submodule update --remote meta-agent
bash meta-agent/scripts/sync-runtime.sh
```

If there are changes, commit the update.
