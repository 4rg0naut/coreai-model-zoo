# Gemma 4 in Apple's own authoring branches — what differs from the zoo port, and the traps they encode

*(Read 2026-09-15. Apple main `7359dbc` (2026-09-14) has NO Gemma 4; the two branches below are
Apple engineers' personal forks and may never merge as-is. Design lessons here are durable; the
status line is not — re-check `models/` on apple/coreai-models main before acting on it.)*

## Sources

- **carinapeng `carina/gemma4-26b-vlm-export`** (PR #137, closed 2026-08-03 "runtime support to
  follow"): macOS `gemma4_text.py` (E2B/E4B + `gemma4_unified` 12B/31B + 26B-A4B MoE), iOS
  `gemma4_text.py`, `vlm/gemma4.py`, `BENCHMARKS.md` (M5 Max, single-pass llm-runner).
  Apple's 2026-09-14 reply on issue #83 says the internal file is now `models/macos/gemma4.py` with
  staged transformer shards — newer than this branch.
- **srjoglekar246 `gemma4-ios-port`** (Sachin Joglekar, commits `sachin_joglekar@apple.com`,
  2026-06-10..21; GitHub profile still says Microsoft — trust the commit email): iOS static path,
  E2B/E4B text only, numbers on an M4 Max ANE, **no iPhone run recorded**, needs ~580 Swift lines in
  `CoreAIStaticShapeEngine` + a PLE loader + `LanguageConfig` fields.

## Design differences vs the zoo (`models/gemma4-*`, `conversion/export_gemma4_*`)

| topic | Apple branches | zoo |
|---|---|---|
| KV cache | **two caches**: sliding preallocated to exactly `sliding_window` (512) and shift-left on overflow (`KVCache.update_and_fetch_windowed`, static seq dim); full-attention cache grows. 4 states `slidingKeyCache/slidingValueCache/keyCache/valueCache` | one unified pair, sliding layers zero-padded 256→512, window mask over the growing linear cache (~60 KB/token) |
| engine support | **Apple main already accepts 2–4 states**: `StateHandlerFactory` (#132 2026-08-03, #202 2026-08-31) classifies dynamic-dim → growing `kvCache`, static + name contains `cache`/`kv` → `slidingCache`, else `fixed`; `CoreAIPipelinedEngine` takes 2–4 states | zoo fork carries the extra-states patch; the "exactly 2 states" note in `pipelined-engine.md` predates #132 |
| prefill | macOS export keeps `seq_ids` dynamic up to `min(max_ctx−2, sliding_window)` → chunked prefill in ≤512-token chunks; main also has a prefill graph + `ChunkedPrefill.swift` (#204/#211/#240/#249) | S=1 static graph, prompt = pipelined S=1 steps; `export_gemma4_pf_pipelined.py` (static 64-chunk multifunction) exists but is not in the HF ship |
| PLE | externalized int8 sidecar `<name>_ple.safetensors`, **one scale for the whole table**, rows gathered by the runtime (Swift `PerLayerEmbeddings` mmap) into an int8 graph input, dequant in-graph | int8 **per-row** scale, provider mode (per-token mmap rows) or `--tbl` in-graph gather |
| quant | uint4 asymmetric per-block-32 `qformulation: minval` on the plain `-it` checkpoint; SDPA/RoPE/RMSNorm/router excluded; no quality gate in the branch | QAT q4_0 checkpoints + absmax symmetric int4 (`--lin-sym`), fp32-oracle gates |
| MoE | 26B-A4B via stock `SwitchGLU` + `MoERouter` (dense MLP + MoE summed) | no 26B-A4B port (community `visible-cx` bundle is self-declared unqualified) |
| VLM | `embed.aimodel` + `vision.aimodel` + `main(inputs_embeds)`, host scatter-merge like Qwen3-VL; static pooling matmul in `Gemma4StaticVisionEncoder` | extension-id splice, `image_embeds` static input; PLE rows for image positions use the **pad** row — issue #83 reports Apple's staged path used `image_token_id` instead (wrong vs HF) |
| iOS/ANE | Joglekar: sliding **ring** of depth `S = round_up(window + max_q − 1, max_q)` (chunk writes never wrap), flat global cache + `BlockedSDPA` flash in 8192-key blocks, ctx ladder 256..131072 (`extend_{ctx}_{q}`, q ∈ {8,16,64}), runtime right-sizes and re-lays-out the global cache per bucket; RoPE cos/sin as runner inputs | 6-chunk host-cache at ctx 64 (`ios/gemma4_ane_chunks.py`); RoPE cos/sin inputs for a different reason (MPSGraph constant-folding SIGSEGV) |

Fused `qk_norm` (q_norm + k_norm as one per-head RMSNorm, as in `gemma3_text.py`) is used by both Apple
modules and not by the zoo; untested lever (dispatch count), A/B before claiming anything.

## Traps worth keeping (from the branches' own comments and code)

- `coreai_torch` lowers `aten.sym_min` but **not `aten.sym_max`**: write `max(x,0)` as `-min(-x,0)`.
- A dynamic `valid_len` may be used as a *value* (mask comparison) but not to *size* a tensor: narrowing
  a cache to `min(capacity, offset+q)` raises spurious `torch.export` `ConstraintViolationError`s.
- On the static/ANE path a position index above fp16 range (65504) does not fit a fp16 input and an
  int32 position input is not supported by the streaming compile → pass RoPE cos/sin rows instead.
- Plain flash accumulators overflow fp16 past ~15k attended keys on the accelerator: carry the output
  normalized and the denominator block-scaled (`p = exp(s−m)/block_size`).
- Zero-fill an unwritten KV tail: garbage in unwritten positions overflows `q·k` to Inf and the additive
  −40000 mask cannot suppress the resulting NaN.
- A prefill-only function must return a rank-preserving `[:1,…]` slice of the cache, not an integer
  index: the squeeze lowers to a rank-1 `(ctx,)` tensor that exceeds the per-dimension limit at 131k.
- Above ctx 65536 a `[C,W]` transpose of the value cache is hoisted to full width; form `(v @ pᵀ)ᵀ`.

## What this changed for the zoo (decisions live in `~/code/coreai/NEWMODELS_STATE.md`)

Gemma 4 on the stock static/ANE engine waits for Apple main (PLE needs their loader; two internal
variants unconverged). The stock static path itself runs on the release iPhone for plain dense models —
see `coreai-beta-mpsgraph-kvwrite-bug.md` (2026-09-15 status).
