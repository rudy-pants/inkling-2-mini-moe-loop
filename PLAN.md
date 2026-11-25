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

### Phase 1 — `moe-trace`
- [ ] `cb_eval` callback: on `ask=true` say yes only for names starting with `ffn_moe_topk-` (and optionally
      `ffn_moe_weights_norm-`); on `ask=false` copy with `ggml_backend_tensor_get` (the tensor lives on the GPU)
- [ ] Parse `il` from the name; derive `phys`, `pass` (−1 for prelude/coda), `exec_order`
- [ ] Binary trace format v1: file header (magic, version, model hash, n_expert, k, slot map) plus one record per
      (token, slot) with 8 expert ids (+ optional 8 fp16 weights). Document it in `docs/trace-format.md`
- [ ] Driver modes:
  - [ ] `--prompt-file` prefill only (records each prompt token separately, batch > 1)
  - [ ] `--generate N` autoregressive decode, batch = 1, sampler settings matching `start-inkling.sh`
  - [ ] `--prompts-dir` batch of prompts (code, prose, chat, math) to vary the workload
- [ ] Mark records as prefill or decode, and record the token position
- [ ] `moe-dump`: print a trace as CSV for quick checks

### Phase 2 — `moe-analyze`
- [ ] Trace reader library (shared with `moe-dump`), checks magic and version
- [ ] Stats:
  - [ ] expert popularity per physical layer (histogram, entropy, top-k share)
  - [ ] **loop overlap**: for each phys layer 6–17, |top8(pass i) ∩ top8(pass j)| for (0,1), (1,2), (0,2); weighted variant
  - [ ] token-to-token overlap per slot (same slot, token t vs t+1), and per lag 1…16
  - [ ] co-activation matrix per physical layer (128×128); this is the input to phase 3
  - [ ] baseline: expected overlap if routing were uniformly random (8·8/128 = 0.5 experts), to compare everything against
- [ ] **Cache simulator**:
  - [ ] Byte-accurate weight access stream in execution order: per slot, attention + router weights, then 8 expert
        slices (offsets taken from the real GGUF tensor layout)
  - [ ] LRU and set-associative models at 128 B line size; sizes 2 MB, 32 MB, plus a sweep (8 MB … 1 GB) for a
        hit-rate-vs-size curve
  - [ ] Reuse-distance histogram (bytes between reuses of the same expert slice)
  - [ ] Oracle bound: Belady/OPT hit rate at 32 MB (best any layout or policy could do)
- [ ] Output JSON + CSV; plotting script optional

### Phase 3 — `moe-permute` (part 4)
- [ ] Clustering: per physical layer, order experts by co-activation (greedy chain / spectral ordering) so that
      experts often picked together sit next to each other; also emit "identity" and "random" orderings as controls
- [ ] GGUF rewriter: for each layer apply permutation P consistently to
      `ffn_gate_inp.weight` (ne1), `exp_probs_b.bias`, `ffn_gate_up_exps.weight` / `ffn_down_exps.weight` (ne2),
      and `ffn_*_exps.scale` when present. Copy every other tensor and all KV metadata unchanged
- [ ] Write the permutation into GGUF metadata (`moe_loop.perm.<layer>`) so traces can be mapped back to the original ids
- [ ] Benchmark: `llama-bench` tg128/pp512 for original vs identity-rewrite vs random vs clustered, ≥5 reps each
- [ ] `rocprofv3` counters on the MoE kernel (FETCH_SIZE, TLB misses if exposed, MALL counters if gfx1151 exposes
      them) for the original vs the clustered model

### Phase 4 — Write-up
- [ ] Results doc: overlap numbers, simulator curves, benchmark table, conclusion on MALL and contiguity
- [ ] Note what would be needed for part (3), the speculative-verify union

---

## QA

### Q1. Build / ABI
- [x] `moe-smoke`'s greedy output (temp 0) for a fixed prompt matches the shipped `llama-cli` with the same
      settings. A mismatch means the headers and the `.so` disagree
- [x] Assert on start-up that `llama_model_meta_val_str("general.architecture") == "inkling2_loop"`, and that
      expert_count, expert_used_count, loop.start/end/count match the expected values; refuse to run otherwise

### Q2. Tracer correctness
- [ ] **Completeness**: each decoded token yields exactly 48 slot records; prefill of N tokens yields N×48
- [ ] **Validity**: every id is in [0,128), the 8 ids in a record are distinct
- [ ] **Independent cross-check**: for a short prompt, compute top-8 on the CPU from the dumped router input
      (`ffn_inp` → norm → `ffn_gate_inp` → sigmoid + `exp_probs_b`) and compare with the traced ids
      (≥ 99% exact; any differences should be near-ties)
- [ ] Compare against the shipped `llama-eval-callback` on the same prompt: same ids for `ffn_moe_topk-*`
- [ ] **No interference**: generated text with the tracer on = text with it off (greedy, fixed seed). The callback
      must not change results, only slow things down
- [ ] Callback overhead measured and reported (tok/s with and without the tracer); tracing speed itself isn't critical
- [x] Slot→phys/pass mapping unit-tested against the table above, including the execution-order edge cases
      (slot 17 → slot 24, slot 47 → slot 18)

### Q3. Trace format
- [ ] Round-trip test: write → read → identical records, on a synthetic trace
- [ ] Reader rejects wrong magic, wrong version and truncated files with a clear error
- [ ] Trace header stores the model file's hash; the analyzer warns when given a mismatched GGUF for layout offsets

### Q4. Analyzer correctness
- [ ] Unit tests with synthetic traces whose answers are known:
  - identical routing every pass → loop overlap = 8; for one token from a cold, unlimited cache, the expert hit
    rate on looped slots = 2/3 (pass 0 misses, passes 1 and 2 hit)
  - disjoint routing → overlap 0
  - uniform random routing → mean overlap ≈ 0.5 (within statistical tolerance)
- [ ] Cache simulator tested against hand-computed cases: tiny cache, short access stream, LRU vs OPT known answers
- [ ] Monotonic check: LRU hit rate must not decrease as cache size grows
- [ ] Sanity check against hardware: total simulated bytes per token ≈ 3.5 GB (±10%), and rocprof FETCH_SIZE per token
      lands in the same ballpark

### Q5. Permutation correctness (most important; this changes a model file)
- [ ] **Never overwrite** the original GGUF; output goes to a new file, refuse if the target path exists
- [ ] Touch only expert-indexed tensors; **`attn_q_clip.weight` also has a 128 dim (head dim), but it is not an
      expert tensor and must not be permuted**. Select tensors by name pattern, never by "has a dim of 128"
- [ ] Handle both NVFP4 and Q8_0 expert layers; treat `.scale` as optional
- [ ] Permute at the byte level along ne2 (whole expert slabs). Never re-quantize
- [ ] Identity permutation → output file is byte-identical to the input (except added metadata) → proves the
      rewriter is lossless
- [ ] Clustered permutation → greedy output token-for-token identical to the original over ≥ 3 prompts × 256 tokens;
      logits within 1e-3 (summing the 8 experts in a different order can change the last bits)
- [ ] Perplexity on a fixed text (`llama-perplexity`, ~20 chunks) matches the original to ≥ 4 significant digits
- [ ] Re-trace the permuted model; after mapping ids back through the stored permutation, the trace matches the
      original model's trace

### Q6. Benchmark hygiene
- [ ] Stop the vLLM server and other GPU users before benchmarking (they compete for memory bandwidth); record
      `rocm-smi` state and power profile
- [ ] Same build, flags, context size and `-fa on` for every variant; spec decoding off for `llama-bench`
- [ ] ≥ 5 repetitions, report mean ± stddev; a result counts only if the difference is larger than 2σ
- [ ] Always include the identity-rewrite and random-permutation controls; only a clustered-vs-random gap shows that
      layout matters
- [ ] Record kernel versions, ROCm version and commit hash next to each result

### Acceptance criteria
- **Phase 1 done** when Q1–Q3 pass and a ≥ 10k-token decode trace across ≥ 4 prompt types exists
- **Phase 2 done** when Q4 passes and loop overlap, reuse distance and hit-rate-vs-size results are produced
- **Phase 3 done** when Q5 passes completely (no exceptions) and the Q6 benchmark table exists, whichever way the
  result goes
