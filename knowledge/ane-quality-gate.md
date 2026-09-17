# The quality gate for a Neural Engine bundle — token match is not enough (2026-09-17)

**Status:** rule of the repo since 2026-09-17. Tooling in [`apps/AneGate/`](../apps/AneGate/README.md); the case that set it is
MiniCPM5-2B in [`ane-vs-gpu-iphone-2026-09.md`](ane-vs-gpu-iphone-2026-09.md) §9a.

## The rule

A static (`coreai.llm.export --platform iOS`, AOT h18p) bundle ships only after **three** checks, in this order — the first
two on a Mac, before the export that goes to the phone:

1. **Recipe simulation on the Mac.** Apply the palettization yaml (or the int8 linear recipe) to the HF fp32 weights with
   coreai-opt's own k-means (`apps/AneGate/simulate_recipe.py`), judge it teacher-forced on the fixture, then run the same
   200 GSM8K questions (`apps/AneGate/bench/ref_gsm8k_hf.py --recipe …`, scored by `score_gsm8k.py` with the
   litertlm-convert extraction). A recipe that loses task accuracy here is not exported. The simulation reproduced the phone
   within 0.5 pt on MiniCPM5-2B (6-bit 86.0 % vs 86.5 % on the phone; 4-bit 65.5 % on the phone).
2. **Token gate on the phone** (`AneGateRunner`): fp32 oracle fixture, teacher-forced single-step sweep + free-run greedy
   incl. the stop, 3 short prompts **and a 109-token free-form answer** (`make_fixture.py --chat-n 128`, knife-edge steps
   excluded). The poisoned fixture must go RED first. This catches broken numerics; it does **not** measure task accuracy —
   a 4-bit bundle passed the 3-prompt version and scored 20 points below the model on GSM8K.
3. **Task gate on the phone** (`TaskEval`, `AB_TASK=gsm8k_200`) — confirmation of step 1 with the real bundle and engine;
   optional once step 1 and 2 agree, mandatory for a new engine path or a new chip.

Speed numbers go next to the gate, never alone: same-day interleaved A/B (`_abn_f.sh`) and, for a nominal value, a
thermal-recovered 60 s run (`_sustain_f.sh <tag> <A,B> 60 60`) — a phone right after a long ANE program build sits at
`thermalState = serious` and reads half.

## What the tooling assumes

- Apple main `7359dbc` `StaticShapeEngine` unchanged (`forcedContinuation` + `includeLogits`), one app per bundle set,
  `AG_MODEL` / `AB_MODEL` pick the arm. Launches without `--console`; the app mirrors its log into the container and the Mac
  pulls it (`bench/_launch_f.sh`).
- The GSM8K protocol is litertlm-convert's (first 200 test questions, 0-shot CoT asking for `#### <number>`, greedy,
  no-think, numeric normalization), so a number here is comparable to the LiteRT cards.
- Count `*ANE_region*` after every AOT: `coreai-build` exits 0 with 0 regions when the ANE compiler rejects a graph
  (4-bit group-8 palettization does).

## Numbers that set the bar (MiniCPM5-2B, iPhone 17 Pro, 2026-09-17)

| arm | GSM8K 200 |
|---|---:|
| fp32 checkpoint (Mac) | 172 |
| 6-bit g8 ANE bundle, phone | 173 |
| 6-bit recipe simulated on the Mac | 172 |
| int8 per-block-32 recipe simulated | 171 |
| 4-bit g32 ANE bundle, phone (the 2026-09-15 ship) | 131 |

Record: `models/minicpm5-2b/gsm8k-200-2026-09-17.json`. The 1B pair on the same questions (Mac): fp32 123, the shipped
8-bit ANE recipe 124, the shipped int8 recipe 129.
