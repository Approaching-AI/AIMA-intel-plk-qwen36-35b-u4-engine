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

## Published release

The canonical Apache-2.0 `v0.1.0` release is published at
<https://github.com/Approaching-AI/AIMA-intel-plk-qwen36-35b-u4-engine/releases/tag/v0.1.0>.
It binds annotated source tag commit
`f4707fd1af6a87390fc29c104acd5ce6a145c261` to the native runtime, offline
CPython 3.12 wheelhouse, service wheel, evidence archive, release manifest, and
top-level checksums. All six assets were downloaded again through anonymous
public URLs and passed the published `SHA256SUMS`.
