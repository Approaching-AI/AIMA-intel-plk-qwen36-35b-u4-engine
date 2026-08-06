# ADR 0056: Close stateful SDPA and select product paged GQA

Date: 2026-07-13

## Status

Accepted. The stateful optimized-SDPA provider source is rejected; one exact
product-pipeline paged-GQA source gate is selected. Product promotion remains
open.

## Context

ADR 0055 selected exact query-one provider capture followed by native event
timing. Commit `1a5f9a5` adds an OpenCL runtime-audit recorder that captures
the real program, kernel, ranges, arguments, USM allocation metadata, shape
metadata, and device event duration without linking the target runtime to
OpenVINO.

Clean seq775 corrects an important ambiguity in the provider label. All ten
primitives report `ocl::sdpa::opt__f16`, but the actual query-one program is
the DPAS kernel
`sdpa_micro__generate_5781906426501558618__sa`, not one of the scalar
`sdpa_opt__*` programs inferred from cache names alone. The exact 128k dispatch
is fixed at global `[16,256,1]`, local `[16,16,1]`, with unique two-head K/V
allocations of about 128 MiB each and scalar source length `131072`.

Device events are terminal against the registered cap:

- exact 131072: `3.685-3.985 ms`, median `3.704 ms`, ten-layer sum
  `37.423 ms`;
- 131073: median `3.725 ms`, ten-layer sum `37.256 ms`;
- 131074: median `3.764 ms`, ten-layer sum `37.703 ms`;
- cap: `2.825 ms/layer`, or `28.250 ms` across ten full-attention layers.

The exact median misses by `31.1%`; even its fastest layer misses by `30.5%`.
Replaying the identical binary without OpenVINO cannot reduce its device event
below that binary's measured execution time, so a native correctness replay
would not change the route verdict.

The audited product denominator uses a materially different continuous-
batching graph and dispatches `paged_attention_opt__single_token*` kernels.
A 2k mechanism trace found one GQA single-token kernel plus a finalization
kernel per full-attention layer. This is not a rate result for 128k, but it is
a distinct cache layout, argument contract, program family, and work
distribution worth one exact source gate.

## Decision

Close stateful `sdpa_micro__generate` and every property, shape, or replay
variant of that binary. Select one fixed product-pipeline paged-GQA gate:

1. Use the same pinned model/provider and scheduler configuration as the
   accepted denominator: raw `prefill_shape_128k`, prefix caching and chat
   templates disabled, `max_num_batched_tokens=sys.maxsize`, four generated
   tokens, and ignored EOS.
2. Capture only `paged_attention_opt__single_token*` dispatches. Require three
   decode iterations, ten main kernels and ten finalization kernels per
   iteration, successful disassembly, exact program-to-dispatch mapping,
   global/local ranges, all arguments, allocation sizes/offsets, and device
   event timing.
3. Charge main plus finalization across all ten full-attention layers. The
   final two iterations are repeat/confirm and must each be
   `<=28.250 ms/token`; their paired spread must be `<=0.5%`.
4. A rate pass admits one native-only replay using the captured paged layout.
   That replay must perform zero timed host transfer and pass finite output,
   cosine `>=0.999`, relative L2 `<=0.002`, both timing rows, and the same
   noise gate before backend integration.

A selection, capture, mapping, rate, replay, numeric, or noise failure closes
the product paged-attention source without a cache precision, block size,
partition, kernel-property, tile, subgroup, or workgroup sweep. A pass remains
component evidence; it does not admit output-512 or product speed.

## Consequences

- Do not native-replay or integrate seq775's stateful SDPA binary.
- PERF_COUNT primitive aggregates remain dispatch labels, not timing evidence;
  OpenCL event durations are the rate ruler.
- OpenVINO/GenAI are offline source providers only. The promoted runtime must
  map neither dependency.

Evidence:

- `output/openvino-sdpa-provider-capture-20260713Tseq775trace-cleanZ/`
- `output/r0-openvino-denominator-matrix-20260713Tseq769-raw-core-rest-cleanZ/`
