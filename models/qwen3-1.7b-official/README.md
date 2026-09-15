# Qwen3-1.7B (Apple's official recipe) — Neural Engine bundle

[🤗 mlboydaisuke/qwen3-1.7b-CoreAI-official](https://huggingface.co/mlboydaisuke/qwen3-1.7b-CoreAI-official) · Apache-2.0 · source [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

This is **not a zoo port**: the bundles are Apple's own `coreai.llm.export` recipes, unmodified (the
`official/` line — [`../../official/README.md`](../../official/README.md)). This directory exists because
the repo gained a **Neural Engine bundle** on 2026-09-15 that the zoo gated on the phone, and a gated
bundle needs a recipe entry and a transcript like any other. The measurements behind it are in
[`../../knowledge/ane-vs-gpu-iphone-2026-09.md`](../../knowledge/ane-vs-gpu-iphone-2026-09.md).

## Bundles

| subtree | what | size | status |
|---|---|---:|---|
| `ios-gpu/` | dynamic int4 linear, pre-AOT'd h18p `.aimodelc` (2026-06-18, coreai-build 3600.67.5.8.1, beta toolchain) | 1.0 GB | **does not load on iOS 27.0 GA with the current Apple main engine** (`invalidState("Failed to find an extend function with the max context length of 40960")`, 2026-09-15) — needs a re-export; the same `4bit` preset exported with today's toolchain (dynamic IR, 934 MB) loads and decodes |
| `ios/` (staged, not uploaded yet) | **iOS static IR**, Apple's registry iOS preset for `qwen3-1.7b` = `models/qwen3/qwen3_1_7b_6bit.yaml` (k-means **6-bit**, per-grouped-channel group 8; embeddings int8; `--max-context-length 4096`) | 1.3 GB | export 2026-09-15, coreai-torch 0.4.2 |
| `ios-ane-h18p/` (staged, not uploaded yet) | **the same IR, AOT-compiled for the Neural Engine** (`xcrun coreai-build compile --platform iOS --preferred-compute neural-engine --architecture h18p`), 31/31 ANE regions | 1.7 GB | **device gate PASS 3/3** (below) |

Apple's iOS *default* (`4bit_weight_palettized_group32`) was exported and gated too, and **fails**: raw-text
prompts are token-exact, but the chat-templated turn (the only prompt carrying `<|im_start|>` / `<|im_end|>` /
`<think>`) collapses to `<|endoftext|>` from step 0 —
[`gate-qwen3-1.7b-ane-4bit-FAIL.json`](gate-qwen3-1.7b-ane-4bit-FAIL.json). Apple's registry answer for
this size is 6-bit, and 6-bit is clean.

## Device gate (iPhone 17 Pro, iOS 27.0 24A437, `ondevice/_ane_gate`)

fp32 `transformers` oracle, teacher-forced single-step sweep + free-run greedy incl. the stop, floor 0.1
(rules: [`../../knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md) §2026-09-15). Prompts: alphabet
list (24 steps, min margin 0.99), no-think chat "Say hello." (12 steps incl. EOS, min 0.73, EOS 1.00),
396-id counting prompt (16 steps, min 0.94). The poisoned fixture went RED first (only its poisoned
prompt fails).

| bundle | verdict | transcript |
|---|---|---|
| `ios-ane-h18p/` 6-bit (Apple's preset) | **PASS 3/3** — natural 24/24, chat 12/12 incl. the stop, long 16/16; **cold load 795 s** (first launch builds the ANE programs), warm 0.15 s, footprint 2.1 GB | [`gate-qwen3-1.7b-ane-6bit-device.json`](gate-qwen3-1.7b-ane-6bit-device.json) |
| 4-bit g32 (Apple's iOS default) | **FAIL** — chat TF 3/12, free-run 20× `<|endoftext|>`; natural and long exact | [`gate-qwen3-1.7b-ane-4bit-FAIL.json`](gate-qwen3-1.7b-ane-4bit-FAIL.json) |

## Speed (same-day interleaved A/B, p128 g256 n5, one app with all arms)

Decode / prefill tok/s, round 1 / round 2 (A-B-C-A-B-C, 20 s between launches, 24A437, USB) —
[`bench-iphone-ane-vs-gpu-2026-09-15.json`](bench-iphone-ane-vs-gpu-2026-09-15.json):

| arm (weight bits) | round 1 | round 2 | load |
|---|---|---|---|
| ANE 4-bit g32 static AOT (4) — **fails the gate**, speed only | 34.4 / 1146 | 55.5 / 2211 | 0.28 s warm |
| GPU int4 per-block-32 dynamic, the same `4bit` preset exported 2026-09-15 (4) | **63.4** / 1162 | **60.9** / 1160 | 3.9 s cold spec / 0.97 s |
| published `ios-gpu/` (June-beta AOT) | did not load | did not load | see Bundles |

At equal byte width the GPU decodes faster on this model (the opposite sign of MiniCPM5-1B's 8-bit pair).
The 6-bit ANE bundle that passes the gate was **not** speed-measured in this session.

## Rebuild

```
# in a coreai-models checkout (zoo-0.4 = Apple main + overlay, coreai-torch 0.4.2; the 1.7b registry entry is upstream-only, so pass the yaml):
uv run coreai.llm.export Qwen/Qwen3-1.7B --platform iOS --experimental --compute-precision float16 --max-context-length 4096 \
    --compression-config models/qwen3/qwen3_1_7b_6bit.yaml --output-dir exports/qwen3_1_7b_ios_6bit_apple
xcrun coreai-build compile exports/qwen3_1_7b_ios_6bit_apple/qwen3_1_7b_6bit_static/qwen3_1_7b_6bit_static.aimodel \
    --output exports/qwen3_1_7b_ios_6bit_apple/aot_h18p_ane --platform iOS --preferred-compute neural-engine --architecture h18p
find exports/qwen3_1_7b_ios_6bit_apple/aot_h18p_ane -name '*ANE_region*' | wc -l    # 31; 0 = silent GPU fallback
```
The export warns that no metadata is registered for the id; `author`/`license`/`description` were patched
into both `metadata.json` files by hand (Qwen Team / Apache-2.0 / Apple's registry description).
