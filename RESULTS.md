# Results: loop-aware expert tracing on Inkling 2 Mini (Strix Halo, 2026-11-24)

Model `Inkling-2-Mini-UD-Q4_K_XL.gguf` (arch `inkling2_loop`), Unsloth llama.cpp `189f431bf` (ROCm 7.2.3, gfx1151),
Linux 7.0.0-42, platform profile `balanced`, no other GPU users.
Full analyzer output: [docs/suite-report.txt](docs/suite-report.txt). Commands: [scripts/reproduce.sh](scripts/reproduce.sh).

## What the model's experts look like

| | |
|---|---|
| Experts | 128 per layer, top-8, no shared expert, 24 physical MoE layers |
| Router | `logits = W·x` (F32 `ffn_gate_inp`), sigmoid, + `exp_probs_b` bias **for selection only** (bias-based load balancing); weights = selected sigmoid probs normalized to 1, × `expert_weights_scale` 8.0 |
| Expert FFN | fused gate+up `[2304 → 2×1152]`, SiTU activation, down `[1152 → 2304]` |
| Storage | blocks 0–19 NVFP4 with per-expert `.scale` (4.27 MB per expert), blocks 20–23 Q8_0 without `.scale` |
| Loop | blocks 6–17 run 3×, the router of each looped block runs 3× per token with the same weights |
| Reads per decode token | 4.0 GB: dense layer weights 1.70 GB, expert slabs 1.76 GB, output head 0.47 GB, KV 0.07 GB (short contexts) |

## Workload (Phase 1)

38 prompts in 10 categories ([prompts/suite.jsonl](prompts/suite.jsonl)), chat prompts rendered through the model's own
template, plus raw continuations: code (Python, C++, Rust, SQL, bash, JS, C), math, 8 languages (zh, ja, ko, ar, ru,
hi, es/de/fr, raw zh), creative, factual/legal/medical, structured data (JSON Schema, CSV, YAML), reasoning with
thinking on, casual chat/role-play, odd inputs (base64, hexdump, repetition, emoji) and a 955-token document.
384 tokens generated per prompt at the server's sampling (temp 1.0, seed 1234).

**16,882 tokens (14,213 decode), 810,336 router decisions.** Tracer self-checks: exactly 48 records per token;
810,335 of 810,336 selections reproduced on the CPU from the captured router probabilities + bias (the one miss is an
exact score tie).

## Findings (Phase 2)

### 1. Routing is broad: every layer uses all 128 experts
Effective experts (2^entropy) are 88–116 of 128 in every layer except block 17; the 8 most popular experts take 12–28% of selections
(uniform = 6.25%). Exception: **block 17**, the last looped block, is the most concentrated (66 effective experts,
top-8 = 37%). There is no small hot set that could be pinned in a cache.

### 2. The loop reuses experts heavily (the main loop-aware result)
Same token, same physical block, different passes (random routing would share 0.5 of 8):

| passes | shared experts (decode) | weighted overlap | tokens sharing ≥ 4 |
|---|---|---|---|
| 0 → 1 | **4.89 / 8** | 0.59 | 82% |
| 1 → 2 | **4.53 / 8** | 0.54 | 74% |
| 0 → 2 | 3.72 / 8 | 0.44 | 57% |

Overlap rises through the core (block 6: 4.2 → blocks 9–14: ~5.2) and collapses at block 17 (3.4 / 1.5 / 2.1),
which acts as the hand-over to the next pass. So ~60% of the expert bytes a looped block reads in pass 1 or 2 are
experts it already read for the same token one pass earlier.

### 3. Consecutive tokens also share experts
Same slot, token t vs t−lag: 3.50 shared at lag 1, 2.84 at lag 4, 2.46 at lag 16 (random 0.5). A window of 4
consecutive tokens needs 19.1 distinct experts per slot instead of 32; 8 tokens need 28.1 instead of 64. This matters
for batched verification (speculative decoding with the DSpark drafter), not for single-token decode.

### 4. Experts barely specialize by topic
Jensen-Shannon divergence of each category's expert distribution vs. all prompts is only 0.08–0.16 bits per layer, and
every category touches 117–127 experts per layer. Multilingual prompts are the *least* distinct (0.08). Topic changes
which experts are slightly preferred, not which experts are needed.

### 5. The 32 MB Infinity Cache cannot capture weight reuse
Fully associative cache over 14,213 decode tokens, whole-object reads (expert slabs, layer weights, 256-position KV
chunks):

| capacity | LRU, all bytes | LRU, expert bytes | offline Belady, all | Belady, experts |
|---|---|---|---|---|
| 2 MB (L2) | 0% | 0% | 0.1% | 0.2% |
| **32 MB (MALL)** | **0%** | **0%** | **2.3%** | **5.3%** |
| 256 MB | 0% | 0% | 19% | 5% |
| 1 GB | 35% | 28% | — | — |
| 4 GB | 89% | 76% | — | — |

The loop reuse in finding 2 sits ~0.9 GB apart (12 layers × ~73 MB), so LRU gets nothing until 1 GB, where the loop
reuse of dense weights (53%) and experts (28%) appears. Even a clairvoyant policy saves at most 2.3% of traffic in
32 MB. Hardware check: `rocprofv3 FETCH_SIZE` reports ~2.0 GB per decode step, but it also reports 245 MB for the
469 MB output-head read, which no cache can hold; scaled by that factor the hardware traffic is ~3.9 GB per token,
matching the simulator's 4.0 GB.

## Expert reordering (Phase 3)

Per layer, experts were reordered so that frequently co-selected experts are adjacent (max-weight Hamiltonian path on
the co-selection matrix: greedy chains + 2-opt). On held-out prompts this cuts the number of separate memory ranges
per top-8 selection from **7.55 to 5.75** (random order: 7.56).

**Correctness** (QA Q5): identity rewrite byte-identical to the original; random and clustered rewrites verified
chunk-by-chunk; 136 tensors permuted (24 router weights, 24 biases, 48 expert weights, 40 scales), 608 copied.
Greedy text is *not* token-identical: the ROCm kernels round differently depending on where an expert is stored, and
greedy decoding amplifies the ~1e-5 differences within 25–76 tokens. Against the original's logits, the reordered
models are as close as the original model run with a different micro-batch size:

| vs original (20 × 512 tokens) | mean KLD | same top token | PPL ratio |
|---|---|---|---|
| original, `-ub 128` (control) | 0.00297 | 97.92% | 0.9981 ± 0.0015 |
| clustered | 0.00293 | 97.82% | 1.0008 ± 0.0016 |
| random | 0.00291 | 97.82% | 1.0013 ± 0.0015 |

**Speed** (llama-bench, 2 rounds in opposite order × 5 reps):

| | original | random | clustered |
|---|---|---|---|
| tg128 (tok/s) | 32.52 | 32.49 | 32.41 |
| pp512 (tok/s) | 563.5 | 560.1 | 563.5 |

No measurable effect: each expert slab is already a 1.5–3 MB contiguous read, so merging neighbours saves almost
nothing, and the reordering cannot change *how many* bytes are read.

## Conclusions

1. The loop is a real source of reuse: ~60% of a looped block's experts in passes 1–2 repeat from the previous pass.
2. That reuse is ~0.9 GB apart in the access stream, 28× the Infinity Cache. Layout changes cannot fix a capacity
   problem; the MALL is irrelevant for weights on this model. Hit rates only start at ~1 GB.
3. Expert reordering is safe (numerically equivalent) but does not change speed.
4. Where the measured reuse *could* pay off: (a) batched verification with the DSpark drafter, where 4 tokens need
   only 19 of 32 expert slabs per slot (finding 3); (b) setups that stream experts from host memory or disk
   (offloading), where a ~1 GB expert cache would serve ~28% of expert reads from the loop reuse alone. On Strix Halo
   all weights already live in the same unified memory, so (b) does not apply here.
