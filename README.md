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

| Tool | Purpose |
|---|---|
| `moe-smoke` | load model, check metadata + loop layout, greedy-generate (`--info-only`, `--step-times`) |
| `moe-trace` | record the router's expert choices for every token and graph slot ([format](docs/trace-format.md)) |
| `moe-dump` | print a trace as CSV (`--perm` maps a permuted model's ids back) |
| `moe-analyze` | popularity, loop/token overlap, category specialization, cache simulation, contiguity |
| `moe-permute` | `plan` expert orderings from traces, `apply` them to a GGUF, `verify` the result |

Results: [RESULTS.md](RESULTS.md). Reproduce: [scripts/reproduce.sh](scripts/reproduce.sh). Plan and QA: [PLAN.md](PLAN.md).

## Quick start

```sh
./build/moe-trace -o traces/t.mtrace --prompts-dir prompts/rendered -n 384     # ~10 min for the 38-prompt suite
./build/moe-analyze traces/t.mtrace -o results/t
./build/moe-permute plan -t traces/t.mtrace --method clustered -o perms/clustered.txt
./build/moe-permute apply -i ../models/.../model.gguf -p perms/clustered.txt -o /path/new.gguf
```
