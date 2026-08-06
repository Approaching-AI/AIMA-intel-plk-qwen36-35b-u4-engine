# Public source boundary

This repository starts with a clean Apache-2.0 public snapshot. Earlier private
Git history is not carried into the public repository so deleted credentials,
host identifiers, local paths, or transient artifacts cannot reappear through
history.

The public source snapshot intentionally excludes:

- model weights and all model-license material;
- ignored raw experiment directories and transient build/cache output;
- release binaries, which are checksum-addressed GitHub Release assets;
- an internal development-methodology submodule that is not a runtime, build,
  test, or deployment dependency.

Selected correctness, performance, source-rebuild, identity, dependency, and
HTTP smoke results are distributed in the versioned release evidence archive.
The repository retains the machine-readable contracts and acceptance matrices
needed to interpret them.

`tools/validate_repo.py` is retained as historical experiment-orchestration
provenance and expects the complete local raw-output census. It is not the
public source-distribution test entry point. The README's service fast suite is
the source-only test entry point; real OpenVINO acceptance uses the release
evidence and exact external runtime/model prerequisites.

The public `meta-agent` submodule is retained for contributor workflow only.
The engine and HTTP service do not import or link it.
