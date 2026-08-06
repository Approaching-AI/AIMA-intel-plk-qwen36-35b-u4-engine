# ADR 0027: Close affine JIT and select external zero-point correction

Date: 2026-07-11

## Status

Accepted as a compiler-route rejection and one bounded algebraic successor.
No teacher-forced, token, context-ladder, or product promotion is implied.

## Context

Clean seq680 applies ADR 0026's K32-aligned affine-U8 descriptor to the pinned
oneDNN generator. Primitive/JIT construction faults with `SIGSEGV` in both
isolated processes before any binary or output exists. K32 alignment therefore
does not repair the generator's U8 zero-point capability.

The affine operation has an exact zero-point-free decomposition. For each U8
weight code `q` and zero point `z`:

```text
(q - z) * scale = (q - 128) * scale + (128 - z) * scale
```

The first term is a supported S8-by-S8 grouped core. The second term is one
coefficient per output/K32 multiplied by the corresponding source K32 integer
sum. It does not require a second dense weight matrix.

Evidence:

- `output/onednn-grouped-q6-affine-u8-per32-gate-20260711Tseq680cleanZ/`
- `output/onednn-grouped-q6-s8-per32-gate-20260711Tseq677cleanZ/`

## Decision

Close direct oneDNN U8 zero-point codegen. Select exactly one external affine
correction implementation on layer 39:

1. store affine U8 codes recentered by 128 as S8 and run the existing
   K32-scaled grouped core;
2. compute each grouped source row's 16 K32 integer sums once;
3. add the compact `(128 - z) * scale * source_scale * group_sum` correction
   to the F32 destination in one native OpenCL kernel.

The combined main core, group-sum kernel, and correction kernel must remain
`<=4316.404 us` in primary and confirm and pass all `16,777,216` outputs at
finite, cosine `>=0.999`, relative L2 `<=0.002`.

This is the exact algebra of the one fixed min/max affine codec. It does not
authorize clipping, calibration, zero-point, group-size, workgroup, tile, or
JIT-strategy variants, and it adds no runtime oneDNN/OpenVINO dependency.

## Consequences

- Direct per-group zero points in the pinned generator are closed for both K16
  and K32.
- The external correction is compact/structured, unlike the closed dense S4
  residual: 16 coefficients per output row and 16 source sums per assignment.
- A passing offline-composed gate still requires binary extraction and a
  pure-OpenCL dependency/mapping proof.

## Follow-Up

- Run the paired external-zero-point affine gate once.
- If it passes, extract the S8 core and integrate the two native correction
  kernels. If it fails, record codec accuracy and correction timing separately
  before changing representation family.

Current gate state remains in
`doc/active/intel-qwen36-35b-a3b-gguf-q4km/STATUS.md`.
