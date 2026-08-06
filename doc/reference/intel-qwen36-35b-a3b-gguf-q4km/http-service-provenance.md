# HTTP service runtime provenance

Snapshot: 2026-08-06

This is the release inventory for the resident HTTP carrier. Runtime assets
are deliberately external to the Python package; their identity is verified
before a worker becomes ready.

## Source and runtime

| component | bound identity |
|---|---|
| OpenVINO source | upstream commit `90214e5be052438cec5617ed3ea7e37df1538f68` |
| oneDNN GPU source | submodule commit `20db47e2d3c4df1b66e93bed2e97d30da175512d` |
| promoted plugin embedded build | `2026.2.0-106-90214e5be05` |
| OpenVINO Runtime | `2026.2.0-21902-90214e5be05-releases/2026/2` |
| OpenVINO GenAI | `2026.2.0.0-3121-adf73e80e66` |
| OpenVINO Tokenizers | `2026.2.0.0-681-f43dbd55981` |
| Python distribution metadata | `openvino==2026.2.0rc2`, `openvino-genai==2026.2.0.0rc2`, `openvino-tokenizers==2026.2.0.0rc2` |
| Python ABI | CPython `3.12`, Linux `x86_64` |
| OpenVINO upstream license | Apache License 2.0 in the bound source checkout |
| target | Intel PTL Arc B390 GPU, device `0xb080`, batch size 1 |
| custom CONFIG_FILE SHA-256 | `bd7a679031bbde2fa2626f2138bf79a5626469ccbc041faadef3b12e811200ad` |

The runtime bundle contains a consolidated, binary-safe source postimage for
the promoted short plugin plus a checksum-locked three-file delta for the
promoted long plugin; historical route patches under `engine/openvino/` are
not treated as a composable release stack. The graph constructor used by the
service is `tools/intel_qwen36_openvino_hot_cold_attention.py`.

## Promoted plugins

| service profile | prompt buckets | required plugin SHA-256 |
|---|---|---|
| `short_full` | 2048, 4096, 8192 | `b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12` |
| `long_compact` | 16384, 32768, 65536, 131072 | `01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269` |
| `long_full` | 16384, 32768, 65536, 131072 | `01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269` |

The short plugin is the seq2291 affine-Q4 full-logit carrier. The long plugin
is the accepted seq2119 carrier; `long_compact` exposes its promoted token-only
greedy path, while `long_full` is used when sampling, penalties, or log
probabilities require full logits. Each profile runs in its own process so
plugin registry, environment properties, graph state, and compile cache cannot
leak across profiles.

The formal evidence for accepting this exact two-profile bridge is the seq2300
rollup recorded in the active status board and accepted-cuts ledger. Changing a
plugin, OpenVINO build, CONFIG_FILE, driver, model, or graph constructor creates
a new unvalidated carrier; updating a hard-coded hash is not a substitute for
the full correctness/performance gate.

### Exact plugin rebuilds

`tools/intel-qwen36-package-runtime-assets.py` fails closed on the exact
OpenVINO/oneDNN commits, both 50-file source postimages, and consolidated patch
hashes. Its bundle includes the base source state, the three-file long delta,
build helper, and standalone recipe. An independent clean build reproduced the
51,325,408-byte short plugin byte-for-byte at SHA-256
`b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12`.
The result is
`output/http-openvino-source-rebuild-20260806/result.json`; detailed steps are
in `openvino-plugin-rebuild.md`.

The long postimage reproduces the historical seq2119 source state rather than
combining its LM-head with later short-route fusion changes. Its LM-head object
matches the accepted archive byte-for-byte, and the full 51,296,736-byte
plugin rebuild matches SHA-256
`01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269`.
The result is
`output/http-openvino-long-source-rebuild-20260806/result.json`.

The plugin's embedded build `106` and the Python Runtime build `21902` are
separate artifact identities. Both are intentional and independently checked.
With the plugin's historical RPATH inhibited, the loader resolves
`libopenvino.so.2620` and `libtbb.so.12` from the offline-installed OpenVINO
wheel and `libOpenCL.so.1` from the target OS.

## Locked model

The model is external and is never bundled into the repository or Python
wheel. Its canonical per-file digests are in
`contracts/qwen36-35b-a3b-openvino-u4-model-contract.json`. The two dominant
weight files are:

| file | SHA-256 |
|---|---|
| `openvino_language_model.bin` | `46140b595760e891d9626c5bfaffc2c998cce176d0de7f6c290af5ae1f2393a4` |
| `openvino_text_embeddings_model.bin` | `21b75aed439e3c5a19daedff1c3d564e91a972061f29c100285f97bceb264bf0` |

### Upstream identity and conversion evidence

The upstream model identity is no longer unknown. The official source is
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), whose
official license file is Apache License 2.0. Local evidence independently
agrees:

- `config.json` identifies `Qwen3_5MoeForConditionalGeneration` and
  `qwen3_5_moe`;
- the legacy diagnostic GGUF metadata names `Qwen3.6 35B A3B`, records
  `apache-2.0`, and links to the official Qwen license;
- local `chat_template.jinja` has SHA-256
  `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`,
  matching the examined official Qwen revision.

The local language IR also embeds conversion metadata: PyTorch input,
OpenVINO `2026.2.0-21869-ddeaa6272ce`, NNCF
`3.2.0.dev0+0d246c632dirty`, Optimum Intel
`1.27.0.dev0+fb74525`, Optimum `2.1.0.dev0`, Transformers `5.2.0`, and
INT4 asymmetric compression at ratio `1.0`, group size `64`, with INT8
symmetric backup. Its runtime options record activation scale `8.0` and FP16
KV cache.

The official
[`OpenVINO/Qwen3.6-35B-A3B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov)
artifact documents the same source model and compression recipe. Several local
tokenizer/detokenizer files are byte-identical to that publication, but the
dominant local language and embedding BINs are not. Therefore this record does
not claim that the locked local IR is a byte-for-byte copy of that repository.
No examined public revision matches the dominant local files. The exact source
revision and export invocation that produced the locked IR remain unrecorded,
and cannot be inferred from the artifact metadata alone. The comparison,
irrecoverable fields, and external-model release contract are recorded in
`locked-model-provenance-boundary.md`.

This evidence resolves the technical upstream identity and stated upstream
license. For the public Apache-2.0 source repository, the owner selected the
explicit external-artifact prerequisite: no model bytes are included in the
repository, Python wheel, or native runtime bundle, and the repository makes
no model-redistribution claim. Operators provide the exact locked IR
separately, subject to its own terms, and the service verifies every digest
before readiness. The machine-readable selection is in
`contracts/qwen36-openai-http-publication-policy.json`.

## Release publication

The repository license is Apache-2.0 and the public source-repository model
policy is the external exact-hash prerequisite. Neither grants model
redistribution rights. The remaining publication work is mechanical: publish
the source commit and the exact runtime assets described below, then verify the
public repository and release checksums.

The accepted plugin and exact OpenVINO Python binaries are also not tracked.
The repository provides
`tools/intel-qwen36-package-runtime-assets.py`, which verifies the two promoted
plugin identities and packages them with graph helpers, all referenced custom
OpenCL sources, OpenVINO license/third-party notices, and a per-file checksum
manifest. The 42-file RC7 bundle additionally carries both exact source
postimages, both plugin rebuild recipes, the project Apache-2.0 license, the
selected publication policy, and the locked-model provenance boundary. The
tool also fails on
unresolved ELF dependencies and records a path-scrubbed dynamic dependency
inventory. A second tool,
`tools/intel-qwen36-package-python-runtime.py`, verifies installed distribution
`RECORD` hashes and exact runtime build strings before producing a deterministic
14-wheel offline wheelhouse plus constraints, hash-required installation input,
and checksums. The wheelhouse includes a pinned `pip` bootstrap so a fresh venv
does not retain a vulnerable installer. A fresh venv installed from that
wheelhouse passes `pip check`. A strict zero-finding audit covers all ten
index-resolvable distributions; the three non-index OpenVINO builds and local
service wheel are instead covered by exact wheel hashes, installed `RECORD`
verification, source provenance, service tests, and the release scan. The
complete real-service and OpenAI SDK smokes pass. These create deployable
local/native bundles without model bytes.

Both plugin source recipes are now standalone and bit-identical. For a public
release, the owner must still publish the accepted plugin and exact Python
runtime binaries with the generated notices/manifests (or independently
reconstruct and fully revalidate equivalent source builds). The repository
license and source-repository model policy are already selected.
