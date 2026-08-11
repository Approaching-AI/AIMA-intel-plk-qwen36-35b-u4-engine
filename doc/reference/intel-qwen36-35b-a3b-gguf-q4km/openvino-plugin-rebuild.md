# Released and successor OpenVINO GPU plugin rebuild

Snapshot: 2026-08-12

This recipe reconstructs the exact OpenVINO/oneDNN working-tree postimages used
by the IQ36 GPU plugins and builds either profile with the recorded toolchain
and CMake contract. The short profile is unchanged. The current source tree's
long profile is the `v0.1.1` release candidate that fixes the arbitrary-length
buffer-layout regression; the published `v0.1.0` runtime asset retains its own
frozen historical recipe. Every source file, patch, commit, tool version, and
final plugin hash is checked before the result can be reported as exact.

This is a source-transparency and identity recipe. It does not make an
arbitrary rebuilt plugin an accepted runtime: any output whose SHA-256 differs
from the promoted value is rejected and must pass the complete product
correctness/performance gate before deployment.

## Current source-tree identities

| input | required identity |
|---|---|
| OpenVINO | `90214e5be052438cec5617ed3ea7e37df1538f68` |
| oneDNN GPU submodule | `20db47e2d3c4df1b66e93bed2e97d30da175512d` |
| short source-state SHA-256 | `0776ca91cd9359a200f1e9a51afaeca83c2e9a9c5952dc2e552839ef12085743` |
| long source-state SHA-256 | `77153ecf9ed7fde3ae32efc78f08b0b0afeb9fe8802d83e34885ca9a292bd067` |
| OpenVINO patch SHA-256 | `e722ba5f225273c8090c8610cd3fb80d2ffb0d2472d5e88511c4ad18ed046e9f` |
| oneDNN patch SHA-256 | `090a385c0fc4f4384d34f79ee187bce7b8933df36ae03dc735f963f5ba716fd9` |
| short-to-long source delta SHA-256 | `017f5eb4925c993db3e084ffa1bd420a44ea8dc66db8765a9ffa47ad3722b2ed` |
| plugin CI build identity | `2026.2.0-106-90214e5be05` |
| expected short plugin SHA-256 | `b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12` |
| expected long plugin SHA-256 | `c0515a401f579620c2fb440031e87e848ceaefab572715d4ace2b76ff2956121` |

The published `v0.1.0` long-profile identities remain frozen for artifact
verification only: source state `b947c32eede6...17a`, short-to-long delta
`a38003733c79...f3e`, LM-head source 105,805 bytes at SHA-256
`8373143a711e...e9`, and plugin SHA-256 `01c04ced415a...269`. Those values do
not validate or substitute for the fixed candidate.

The current candidate runtime asset bundle contains `source-state.json`, both consolidated
binary-safe base patches, the three-file long-profile delta, and
`build-openvino-plugin.py`. The short state records 48 modified or untracked
OpenVINO files and two oneDNN files. The long state records 47 such OpenVINO
rows because one delta target returns to its pinned clean base version; the
helper still verifies the same 50-file profile inventory. The long postimage
selects the historical seq2119 LM-head and predates two later short-route
fusion changes, so it differs in
`iq36_lm_head_i8q4.cpp`, `fc_horizontal_fusion.cpp`, and
`transformations_pipeline.cpp`. The helper independently verifies every
status, byte size, SHA-256, and the selected aggregate source-state
fingerprint.

The plugin's embedded CI build identity is `106`. This is intentionally
different from the separately built OpenVINO Python Runtime identity
`21902-90214e5be05`; the service verifies both identities in their respective
artifacts.

## Build host contract

The identity build uses Linux x86-64 with glibc 2.39 and these exact visible
tool versions:

| tool | version |
|---|---|
| GCC/G++ | conda-forge `14.3.0-19` |
| GNU binutils | `2.45.1` |
| CMake | `4.3.3` |
| Ninja | `1.13.2` |
| Python | `3.12.13` |
| system pkg-config | `1.8.1` |
| Level Zero development package | `1.29.0` |

The conda-forge package builds used on the bound host include
`gcc-14.3.0-h0dff253_19`, `gxx-14.3.0-he448592_7`,
`binutils-2.45.1-default_h4852527_102`, `cmake-4.3.3-hc85cc9f_0`,
`ninja-1.13.2-h171cf75_0`, `level-zero-1.29.0-hb700be7_0`, and
`level-zero-devel-1.29.0-hb700be7_0`. The helper fails closed when the
executable identities or Level Zero version drift.

## Reconstruct and build

Create a clean checkout. The helper can use any checkout location, but it
creates a deterministic logical layout below `WORK_ROOT` so OpenVINO's
directory-name compile definition and relative prefix maps match the promoted
build.

```bash
mkdir -p "$WORK_ROOT/source"
git clone https://github.com/openvinotoolkit/openvino.git \
  "$WORK_ROOT/source/openvino-90214e5be05"
git -C "$WORK_ROOT/source/openvino-90214e5be05" checkout \
  90214e5be052438cec5617ed3ea7e37df1538f68
git -C "$WORK_ROOT/source/openvino-90214e5be05" submodule update \
  --init --recursive
```

Then run the bundled helper. `RUNTIME_BUNDLE` is the directory containing
`manifest.json`; `TOOLCHAIN_PREFIX` is the exact conda environment prefix.

```bash
python "$RUNTIME_BUNDLE/source/build-openvino-plugin.py" \
  --source "$WORK_ROOT/source/openvino-90214e5be05" \
  --bundle "$RUNTIME_BUNDLE" \
  --profile short \
  --work-root "$WORK_ROOT" \
  --toolchain-prefix "$TOOLCHAIN_PREFIX" \
  --parallel 1 \
  --result "$WORK_ROOT/rebuild-result.json"
```

The command deliberately mutates only the clean source checkout by applying
the two verified patches. It refuses an unrelated dirty tree. `--prepare-only`
performs patch reconstruction and postimage verification without compiling;
`--resume` continues an interrupted, already configured build directory.

The result is written to:

```text
$WORK_ROOT/output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/
libopenvino_intel_gpu_plugin.so
```

Success requires `profile: short` and `bit_identical_to_promoted: true`. A
different hash exits nonzero and explicitly marks the output as unaccepted.

To build the long profile, use a separate clean checkout/work root and replace
`--profile short` with `--profile long`. The helper first reconstructs and
verifies the short base postimage, applies the checksum-locked three-file
delta, and requires the long source-state fingerprint before compilation. A
promoted short postimage can also be advanced in place to the long postimage;
the reverse direction intentionally requires a fresh checkout.

## Bound-host verification

The recipe was exercised from independent clean checkout/build directories on
2026-08-06. All 50 changed source files reproduced their selected recorded
postimage, and the 1,882 generated build commands matched the promoted build
after normalizing checkout/output prefixes, an output-only linker
dependency-file flag, and unused non-Release flag defaults. The IQ36 custom
objects, changed oneDNN object, and final plugins were byte-identical.

The final result is recorded in
`output/http-openvino-source-rebuild-20260806/result.json`: 51,325,408 bytes,
SHA-256
`b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12`,
with `bit_identical_to_promoted: true`.

The independently reconstructed long LM-head source is 105,805 bytes at
SHA-256
`8373143a711ee75ff8eb913a1e04b89a270d5b419a5196b863911626de8e45e9`.
Its compiled object is byte-identical to the object extracted from the
accepted seq2119 graph archive at SHA-256
`16dfdc03f2be76a99efcd67ed84f1330bbf629133b6c14e8e67765d62272d34b`.
That historical full long rebuild result is recorded in
`output/http-openvino-long-source-rebuild-20260806/result.json`: 51,296,736
bytes, SHA-256
`01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269`,
with `bit_identical_to_promoted: true`.

The fixed candidate LM-head source is 106,546 bytes at SHA-256
`81be0135a12f6d94b87d5ef3ad9e72bf2dca243f98e4ab9c376a51b3a28d51a4`.
An independent 2026-08-12 reconstruction verified source state
`77153ecf9ed7...067` and produced the 51,296,736-byte candidate plugin at
SHA-256 `c0515a401f57...121` bit-for-bit. The result record SHA-256 is
`790ddb239d38da0fa9508be003558d86179832d68a4946bb5c9367875d9122be`.
This establishes source reproducibility, not product promotion.

## Scope and deployment note

The promoted binary records the historical build RPATH, so the identity build
reproduces that value. Deployment does not rely on the historical directory:
the exact OpenVINO Python wheel supplies `libopenvino.so.2620` and
`libtbb.so.12`, while the target OS supplies `libOpenCL.so.1`. These runtime
identities remain part of the deployment preflight. A loader check with all
embedded RPATH entries inhibited resolved those three libraries from the
offline-installed wheel and target OS.

The bundled base snapshot and long delta are the exact source recipes for the
selected short plugin and candidate long plugin. The exact OpenVINO, GenAI,
and Tokenizers Python wheels remain separately checksummed runtime artifacts.
The fixed long fingerprint must still pass the complete successor product and
publication gates; rebuilding it exactly does not inherit seq2300 acceptance.
