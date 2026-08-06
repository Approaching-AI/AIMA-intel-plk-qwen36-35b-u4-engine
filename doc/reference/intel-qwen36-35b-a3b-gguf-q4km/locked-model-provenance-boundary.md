# Locked model provenance and reproducibility boundary

Snapshot: 2026-08-06

The HTTP service is specialized for one externally supplied OpenVINO IR. The
repository and Python wheel do not contain model bytes. Runtime readiness
requires the exact 12-file, 19,705,459,812-byte identity recorded in
`contracts/qwen36-35b-a3b-openvino-u4-model-contract.json`; the aggregate
fingerprint is
`eb05132e47fe0fd1dc42fa3082e7241696ed1449dec246a3cc14bef4af21d7ec`.

This document separates facts recoverable from the locked artifact from
source-to-IR history that is not recoverable. It is a provenance boundary, not
a model redistribution authorization.

## Recoverable identity

The upstream architecture and model identity are evidenced as
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The
locked `config.json` identifies `Qwen3_5MoeForConditionalGeneration` and
`qwen3_5_moe`; the local chat template also matches the examined official
source revision.

The language-model IR embeds this conversion metadata:

| field | embedded value |
|---|---|
| input framework | PyTorch |
| OpenVINO | `2026.2.0-21869-ddeaa6272ce` |
| NNCF | `3.2.0.dev0+0d246c632dirty` |
| Optimum Intel | `1.27.0.dev0+fb74525` |
| Optimum | `2.1.0.dev0` |
| Transformers | `5.2.0` |
| PyTorch | `2.11.0+cpu` |
| weight compression | INT4 asymmetric, ratio `1.0`, group size `64` |
| backup mode | INT8 symmetric |
| activation scale | `8.0` |
| KV-cache precision | FP16 |

The official
[`OpenVINO/Qwen3.6-35B-A3B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov)
publication names the same source model and semantic compression recipe, but
it is not the byte source for the locked local IR. The dominant files differ
from both examined public weight generations:

| artifact | language BIN bytes / SHA-256 | text-embedding BIN bytes / SHA-256 |
|---|---|---|
| locked product IR | `18,646,205,498` / `46140b595760...93a4` | `1,017,118,724` / `21b75aed439e...bf0` |
| public pre-refresh weights | `22,099,454,802` / `984e6609f03d...4c9` | `509,056,004` / `b9957db96d6a...b51` |
| public refreshed weights | `18,646,558,274` / `44b8b72c7009...a62` | `509,056,004` / `b9957db96d6a...b51` |

A local acquisition record dated 2026-06-04 shows that this machine received
an already converted directory. That record is intentionally sanitized and
does not preserve a source-model commit, source-weight digests, or export
command.

## Irrecoverable history

The IR does not encode enough information to reconstruct its exact export. In
particular, it does not preserve the source safetensor hashes or revision, the
complete converter invocation and environment, any calibration inputs, or the
uncommitted NNCF changes denoted by the embedded `dirty` version. Multiple
source/export histories can produce the same visible metadata.

Consequently, no public model revision can honestly be declared a
byte-identical source, and the metadata above is not an exact rebuild recipe.
Recovering the missing history requires an owner-supplied acquisition/export
record; it cannot be derived from the local IR by further inspection.

## Public-release contract

The technically reproducible service boundary is therefore:

1. code, custom OpenCL sources, both GPU-plugin source recipes, Python wheels,
   deployment material, manifests, and third-party notices are release assets;
2. the model is an external immutable input accepted only when every locked
   file digest matches the model contract;
3. the release makes no claim that the exact model IR can be regenerated from
   the currently public OpenVINO model repository;
4. replacing or regenerating the IR creates a new carrier and requires the
   complete correctness and performance gate.

The repository owner selected the following policy for the public
Apache-2.0 source repository on 2026-08-06:

- release the service source with the exact external-artifact prerequisite
  and no model redistribution claim.

This means the repository, Python wheel, and native runtime bundle contain no
model bytes. Operators acquire the model independently, and production
readiness accepts it only when all 12 file digests match the locked model
contract. The repository license does not grant permission to redistribute the
model. The selection is recorded in
`contracts/qwen36-openai-http-publication-policy.json`.

The other policies remain future alternatives that would define a new release
action and require explicit authorization:

- publish or authorize access to the exact locked IR with its notices and
  hashes;
- recover the exact export record and validate a reconstructed artifact; or
- replace the locked IR and rerun the complete product gate.

This record closes the technical investigation and the source-repository model
policy. It does not authorize model redistribution.
