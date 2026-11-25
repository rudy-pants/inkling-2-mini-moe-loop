# moe-loop

Loop-aware MoE expert tracing for Inkling 2 Mini on Strix Halo. See [PLAN.md](PLAN.md) for the roadmap and QA plan.

## Build

Requires CMake ≥ 3.20 and Ninja (`uv tool install cmake ninja`) and g++ ≥ 13. Links against the prebuilt
Unsloth llama.cpp in `../llama-rocm`, using headers from the matching source tree in `../llama.cpp-src`
(merged commit `189f431bf`, from the `llama.cpp-source-commit-189f431bf*.tar.gz` release asset).

```sh
cmake -S . -B build -G Ninja            # add -DMOE_MODEL_TESTS=ON to register GPU/model tests
cmake --build build
ctest --test-dir build -LE model        # unit tests (no GPU needed)
ctest --test-dir build -L model         # model tests (needs -DMOE_MODEL_TESTS=ON)
```

## Tools

| Tool | Status | Purpose |
|---|---|---|
| `moe-smoke` | done | load model, check metadata + loop layout, greedy-generate (`--info-only` to skip generation) |
| `moe-trace` | todo | record the router's expert choices per token and slot |
| `moe-analyze` | todo | overlap statistics + cache simulator |
| `moe-permute` | todo | GGUF rewriter that reorders experts |
