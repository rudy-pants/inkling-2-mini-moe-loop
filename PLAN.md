# moe-loop — loop-aware expert tracing for Inkling 2 Mini

A C++ toolset that records which experts the router picks in every pass of Inkling 2 Mini's looped layers, simulates
how those weight reads would behave in the Strix Halo caches, and rewrites the GGUF so that experts used together are
stored next to each other.

Scope: **(1)** trace, **(2)** cache simulator, **(4)** expert permutation. Speculative-verify union analysis (3) is
out of scope for now; the trace format should still allow it later.

## Reference facts

| | |
|---|---|
| Source | `../llama.cpp-src` (merged commit `189f431bf`, same as the shipped `../llama-rocm` build) |
| Link against | `../llama-rocm/libllama.so`, `libggml*.so` (no rebuild of llama.cpp) |
| Router hook | `cb_eval` → tensor `ffn_moe_topk-<il>`, I32 `[n_expert_used=8, n_tokens]` (`src/llama-graph.cpp:2512`) |
| Slot `il` → pass | 0–5 prelude, 6–17 pass 0, 24–35 pass 1, 36–47 pass 2, 18–23 coda (`src/models/inkling2-loop.cpp`) |
| Execution order | 0–17, 24–35, 36–47, 18–23 |
| Slot → physical | `il < 24 ? il : 6 + (il - 24) % 12` |
| Experts | 128, top-8, ~4.48 MB per expert per layer (gate_up 2.99 MB + down 1.49 MB, NVFP4) |
| Mixed quant | 20 layers NVFP4 (with `.scale` tensors), 4 layers Q8_0 (no `.scale`) |
| Caches | L2 2 MB, Infinity Cache (MALL) 32 MB, 128 B lines |

### Implementation notes (learned in Phase 0)
- `llama_decode` / `llama_batch_get_one` are deprecated in this build; use `llama_batch_ext_*` + `llama_process`
- Backends are runtime plugins: call `ggml_backend_load_all_from_path(<llama-rocm>)` before loading the model
- `llama_model_n_layer()` returns 48 (graph slots); `llama_model_n_layer_weights()` returns 24 (physical blocks)
- Upstream bug #32778 (garbage output for prompts over ~96 tokens) does not reproduce here: a 225-token prompt
  gives the same normal text in `moe-smoke` and `llama-completion`
- **Timing pitfall**: `llama_process()` only queues GPU work; it is finished when the logits are read
  (`llama_sampler_sample`). Take timestamps after sampling. An early `moe-smoke` timed from `llama_process()` and
  wrongly showed 20.9 vs 30.9 tok/s for long vs short prompts; with correct timing both are ~32 tok/s
- The first two decode steps after any prefill are slower (~44 ms, ~35 ms vs ~30.2 ms steady): exclude them or
  warm up before measuring
- Real decode cost of context (llama-bench tg64, -fa on): 32.9 tok/s at depth 0, 32.0 at 225, 31.0 at 1024,
  30.0 at 4096 (−9%)
  - Attention passes: physical blocks 5, 11, 17, 23 are global, the rest use the 1024-token sliding window → 8 global
    + 40 sliding-window passes per token. KV is F16, 4 KB per context position per pass
  - On gfx1151 the relative-position bias is built as a separate tensor (F32 gather → F16 mask) and then fed to
    ordinary vector flash attention; the fused-bias path (unslothai#413) is available on HIP but did not fire;
    #415 is NVIDIA-only (Sol review, 2026-11-24)
  - Step 2 warmup is HIP graph capture (needs 2 identical calls, `ggml-cuda.cu` ~line 6003); step 1 is the graph
    rebuild after the prefill→decode shape change
- **Cache simulator (Phase 2)**: add KV reads (≈ 4 KB × visible positions per pass) to the access stream, not just
  weights; they reach ~300 MB/token at depth 4096
- **rocprofv3**: a quick run only produced HIP API timing; hardware counters (FETCH_SIZE, MALL) still need to be
  shown to work on gfx1151 before Phase 3 relies on them

---

## TODO

### Phase 0 — Scaffold
- [x] `CMakeLists.txt`: C++20, include `llama.cpp-src/include` + `ggml/include`, import the prebuilt `.so`s, set
      RPATH to `../llama-rocm`
- [x] Minimal `moe-smoke` binary: load model, tokenize, decode 1 token, print the top-1 token (proves headers and ABI match)
- [x] `tests/` with a header-only test framework (doctest), wired into `ctest`
- [x] `git init`, `.gitignore` (build dirs, traces, *.gguf)

### Phase 1 — `moe-trace` ✅
- [x] `cb_eval` callback on `ffn_moe_topk-<il>` (+ `ffn_moe_probs-<il>` for weights and the CPU check), strided
      `ggml_backend_tensor_get` copy (the top-k tensor is a non-contiguous view of the argsort)
- [x] Parse `il`; `phys`, `pass`, execution order from `moe::LoopLayout`; micro-batch-safe token bookkeeping per slot
- [x] Binary trace format v1, documented in `docs/trace-format.md` (weights stored as f32, not fp16)
- [x] Driver: `-p`, `--prompt-file`, `--prompts-dir`, `-n N` (0 = prefill only), `--temp/--seed` (default = server
      sampling), `--no-trace` for the interference check; JSON sidecar with names, categories, tokens, text, timings
- [x] Prefill/decode phase and position in every record
- [x] `moe-dump` (CSV, `--perm` to map ids of a permuted model back)
- [x] Prompt suite: 38 prompts / 10 categories (`prompts/suite.jsonl`), rendered via the model's chat template
      (`scripts/render_prompts.py`); traced 16,882 tokens (14,213 decode)

### Phase 2 — `moe-analyze` ✅
- [x] Trace reader shared by all tools
- [x] Popularity (counts, entropy, effective experts, top-8 share, Gini), loop overlap (+ weighted, histogram),
      token-lag overlap 1…16, distinct experts per window of W tokens, per-category Jensen-Shannon divergence,
      random baselines
- [x] Cache simulator: object-level, exact for fully associative LRU on whole-object reads (Mattson stack distances,
      all capacities in one pass), size-aware offline Belady bound, reuse-distance histogram, KV chunks included.
      **Not done:** set-associative model (fully associative is the optimistic case, so it bounds LRU from above)
- [x] JSON + CSV output (`results/<trace>/`)

### Phase 3 — `moe-permute` ✅
- [x] Co-activation ordering: greedy chains from every start + 2-opt (max-weight Hamiltonian path); identity and
      random controls
- [x] GGUF rewriter, by-name tensor classification, byte-level chunk moves, never re-quantizes
- [x] Permutation stored in a **sidecar text file** (`perms/*.txt`) instead of GGUF metadata: keeps the header
      untouched, so an identity rewrite is byte-identical to the original (deliberate change from the plan)
- [x] Benchmark original / random / clustered (identity = original, proven byte-identical)
- [~] `rocprofv3`: FETCH_SIZE works on gfx1151 but reports ~52% of real bytes (calibrated on the 469 MB output head);
      no MALL counters exposed; per-kernel original-vs-clustered comparison skipped since bytes read are identical
      by construction and speed showed no difference

### Phase 4 — Write-up ✅
- [x] `RESULTS.md`
- [x] Part (3) note: 4 consecutive tokens need 19.1 of 32 expert slabs per slot (window-union statistic)

---

## QA (results 2026-11-24; kernel 7.0.0-42, ROCm 7.2.3, platform profile `balanced`, llama.cpp `189f431bf`)

### Q1. Build / ABI ✅
- [x] `moe-smoke` greedy output identical to the shipped `llama-completion`
- [x] Metadata check refuses mismatching models

### Q2. Tracer correctness ✅ (one item partial)
- [x] Completeness: exactly 48 records per token (810,336 for 16,882 tokens), 0 incomplete batches
- [x] Validity: 0 invalid or duplicate ids
- [x] CPU cross-check: top-8 recomputed from the captured `ffn_moe_probs` + `exp_probs_b` for **every** record:
      810,335 / 810,336 agree, the miss is an exact tie. (Recomputing from `ffn_inp` was replaced by this: the router
      input passes through the depth mix and loop control first, which the probs capture already includes.)
- [~] vs `llama-eval-callback`: prelude slots 0–5 identical; overall 87% of token-slots identical, 12% order swaps,
      2% set differences, starting at the first looped layer — numerical drift because eval-callback splits the graph
      at every op and disables fusions; not an indexing problem (an index bug would also break the prelude)
- [x] No interference: greedy output identical with tracer on/off (4 prompts × 128 tokens, incl. a 955-token prompt)
- [x] Overhead: ~10% decode (29 vs 32 tok/s)
- [x] Slot mapping unit tests

### Q3. Trace format ✅
- [x] Round-trip (with and without weights), wrong magic / version / truncated / too short rejected
- [x] Model fingerprint in the header; analyzer warns on mismatch

### Q4. Analyzer correctness ✅
- [x] Synthetic: identical routing → overlap 8 and 2/3 cold-cache hit rate on looped slots; disjoint → 0;
      uniform random → 0.5 (±5%)
- [x] Hand-computed stack distances / LRU, Belady > LRU on a cyclic pattern, monotonic LRU, Belady ≥ LRU (unit sizes)
- [x] Hardware: simulated 4.0 GB per decode token vs FETCH_SIZE ×(1/0.52 calibration) ≈ 3.9 GB

### Q5. Permutation correctness ✅ (with one criterion replaced, see below)
- [x] Refuses to overwrite; removes partial output on failure (unit-tested)
- [x] By-name selection; `attn_q_clip` (shape trap) untouched (unit-tested on a synthetic GGUF and on the real model)
- [x] NVFP4 + Q8_0 layers, optional `.scale`: 136 tensors permuted (24/24/48/40), 608 copied
- [x] Byte-level, no re-quantization
- [x] Identity rewrite of the real model: `cmp` byte-identical
- [x] `moe-permute verify`: every chunk of the random and clustered models checked against the original
- [x] **Replaced criterion.** "Greedy output token-identical" does not hold and cannot: the ROCm kernels round
      differently depending on the expert's storage position (router weights already differ by ~1e-5 at layer 3 of
      token 0), and greedy decoding amplifies that in 25–76 tokens. New criterion: the reordered model must be as close
      to the original as a harmless change of the original itself. Result vs original logits (20 × 512 tokens):
      control `-ub 128` KLD 0.00297 / same-top 97.92%; clustered 0.00293 / 97.82%; random 0.00291 / 97.82%
- [x] Perplexity: 4.1650 (orig) / 4.1643 (control) / 4.1677 (clustered) / 4.1678 (random), ± 0.091; PPL ratios
      within ±0.0016 of 1 (4 significant digits not met, the control differs at the 4th digit too)
- [x] Re-trace mapped back: 6.93 / 8 experts shared with the original (raw, unmapped: 0.49 = chance)

### Q6. Benchmark hygiene ✅
- [x] No other GPU users (vLLM not running); kernel/ROCm/profile recorded above
- [x] Same build and flags, `-fa on`, no speculative decoding
- [x] 2 rounds in opposite order × 5 reps; differences < 1% and within round-to-round spread → no effect
- [x] Random-permutation control included; identity = original (byte-identical)

### Acceptance
- Phase 1 ✅ (Q1–Q3; 14,213-token decode trace over 10 categories)
- Phase 2 ✅
- Phase 3 ✅ with the documented exception: the bitwise greedy-identity criterion was replaced by the KLD-vs-control
  criterion above
