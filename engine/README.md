# engine

Minimal native engine skeleton shaped by `meta-engine-factory`.

The implementation must stay O(1) in layer count:

- one parameterized layer implementation
- one prefill/decode loop
- a small set of ops
- a resident harness adapter for hot boundary validation

Do not add one source file per model layer.

## Build

```bash
cmake -S engine -B build/engine
cmake --build build/engine
ctest --test-dir build/engine --output-on-failure
```

