# moe-loop trace format, version 1

Written by `moe-trace`, read by `moe-dump`, `moe-analyze` and `moe-permute plan`. Little-endian, no padding.
Implementation: `include/moe/trace.hpp`, `src/trace.cpp`.

## Header (56 bytes)

| offset | type | field | notes |
|---|---|---|---|
| 0 | char[8] | magic | `M O E T R C \0 \1` |
| 8 | u32 | version | `1` |
| 12 | u32 | flags | bit 0 = records carry router weights |
| 16 | u32 | n_expert | ≤ 256 (expert ids are stored as u8) |
| 20 | u32 | k | experts per token, ≤ 16 |
| 24 | u32 | n_phys | physical blocks (`block_count`) |
| 28 | u32 | core_start | `loop.start` |
| 32 | u32 | core_end | `loop.end` |
| 36 | u32 | n_passes | `loop.count` |
| 40 | u64 | model_size | GGUF file size in bytes |
| 48 | u64 | model_hash | FNV-1a 64 of the first 16 MiB of the GGUF |

The slot layout follows from the four loop fields (`include/moe/layout.hpp`):
slots `[0, n_phys)` are prelude, core pass 0 and coda; slots `[n_phys, n_slots)` are core passes 1…P−1.

## Records (16 + k [+ 4k] bytes each, `k = 8` → 24 or 56 bytes)

| offset | type | field | notes |
|---|---|---|---|
| 0 | u32 | seq | prompt index; names and categories are in the `<trace>.json` sidecar |
| 4 | u32 | pos | token position in the sequence |
| 8 | i32 | token | input token id at that position |
| 12 | u8 | slot | graph slot `il` (0…47 for Inkling 2 Mini) |
| 13 | u8 | phase | 0 = prefill, 1 = decode |
| 14 | u16 | reserved | 0 |
| 16 | u8[k] | ids | selected experts in the router's top-k order (highest score first) |
| 16+k | f32[k] | weights | only if flag bit 0: the chosen experts' sigmoid probabilities normalized to sum 1 (before `expert_weights_scale`) |

Records appear in the order the callback saw them (per micro-batch, slots in execution order). Every token has
exactly one record per slot; `moe-analyze` refuses a trace where a (token, slot) pair is missing or duplicated.

The reader rejects a wrong magic, an unknown version, an invalid header, and a file whose body is not a whole
number of records (truncation).

## Sidecar `<trace>.json`

Per sequence: name, category (from the `<category>__<name>.txt` prompt file name), prompt length, generated
tokens and text, prefill time and decode speed; plus sampling settings and tracer self-check counters
(`set_mismatch` = records where the CPU recomputation of top-k from `ffn_moe_probs` + `exp_probs_b` disagreed).
