# Neural Engine vs GPU on the iPhone 17 Pro (2026-09) — what a same-day A/B does and does not isolate

> **Status (2026-09-16 13:00):** §9 added — Apple's 6-bit, mixed 4/8 and 4-bit-g8 recipes at 2B gated on the ANE with a
> 109-token free-form prompt in the fixture, an fp32 simulation of each recipe, the first-load cache experiments. Earlier: measured — the 1B sustained curves (v1 protocol), the Qwen3-1.7B gate at 4- and 6-bit, the
> Qwen3-1.7B 3-arm A/B, and the 2B 4-bit equal-byte pair (in an unlogged, degraded phone state — §4). **Not obtained** — the 2B
> 8-bit pair (the ANE 8-bit 2B never returns its first generation) and the thermal-recovered (v2) sustained runs. The phone was
> updated to 24A437 mid-session (every specialization cache went cold) and left twice. Sections say which.
> Every number is from the run it names; none of this is a headline yet.

The 2026-09-15 same-day A/B on MiniCPM5 ([`minicpm5-1b.md`](minicpm5-1b.md) §2026-09-15) left two
claims tangled: *ANE 8-bit beats GPU int8 by +17 % decode at equal byte width (1B)* and *ANE 4-bit beats
GPU int8 by 2.4× (2B) — with half the weight bytes*. This note attacks the premises one variable at a
time: equal byte width at 2B, a third model family on Apple's own builder, and what happens after the
first minute (sustained decode, thermal state, battery). It also records the instrument, so the
protocol can be re-run.

## 1. Protocol (all arms)

- **Phone**: iPhone 17 Pro (iPhone18,1), iOS 27.0 — **24A435 for the morning A/B and the 1B sustained run, 24A437 for
  everything from 17:04 on** (the phone updated mid-session; nothing is compared across the two builds) — wired USB (so **charging** the whole time),
  screen on (the bench app sets `isIdleTimerDisabled`), one app at a time under the
  `com.coreai.pipelinedbench` id.
- **App**: `ondevice/_ane_gate/bench/` (AppleBenchRunnerGA) — Apple's `llm-benchmark` method on the
  unmodified Apple main 7359dbc package: splitmix64 synthetic prompt, greedy, 1 warmup + n timed trials,
  decode tok/s excludes the first emitted token. Both/all arms are embedded in ONE app and picked per
  launch by `AB_MODEL`; each launch is a fresh process. ANE arms load through `EngineFactory` →
  `StaticShapeEngine` (chunked-static graphs, AOT `.h18p.aimodelc`); GPU arms through the pipelined
  engine (dynamic `.aimodel`, specialized on first load, or a pre-AOT'd `.aimodelc` for the zoo's
  `ios-gpu/`).
- **Speed A/B**: p128 g256 n5, order A-B-A-B (N arms: A-B-C-A-B-C), 20 s idle between launches
  (`_ab.sh`, `_abn.sh`). Reported as the two rounds' values, never one number.
- **Sustained**: `_sustain.sh` → `Sustain.swift`: per arm, release the engine, idle ≥3 min **and until
  `ProcessInfo.thermalState` is back at `fair` or below (cap 10 min; protocol v2 — v1 was a fixed 3 min and
  let the second arm start at `serious`)**, load, warmup, then repeat the p128 g256 trial for 10 minutes.
  Every trial logs elapsed s, prompt/decode tok/s, generated tokens, battery %, battery state and thermal
  state, to the console and to `Documents/sustain/<tag>.log` in the shared container (survives a USB
  unplug).
- **Weight bytes** are decided by the recipe, not the file size: 8-bit k-means g32 (ANE) vs int8
  per-block-32 (GPU) = 1 byte/weight; 4-bit k-means g32 (ANE) vs int4 per-block-32 linear (GPU) =
  0.5 byte/weight. The embedding table is int8 on every iOS static export; the GPU int8/int4 dynamic
  exports quantize it with the body.

## 2. Arms

| arm | model | export | engine / compute | weight bits | on disk |
|---|---|---|---|---|---|
| `minicpm5_1b_ane_pal8` | MiniCPM5-1B | stock iOS static, 8-bit k-means g32 (`conversion/minicpm5_pal8_g32.yaml`), AOT h18p, 31/31 ANE regions | StaticShapeEngine / ANE | 8 | 1.3 GB |
| `minicpm5_1b_gpu_int8` | MiniCPM5-1B | shipped `int8/` (int8 per-block-32, dynamic) | pipelined / GPU | 8 | 1.1 GB |
| `minicpm5_2b_ane` | MiniCPM5-2B | stock iOS static, 4-bit k-means g32 (Apple's iOS default), AOT h18p, 31/31 | StaticShapeEngine / ANE | 4 | 1.4 GB |
| `minicpm5_2b_gpu_int8` | MiniCPM5-2B | shipped `int8/` (int8 per-block-32, dynamic) | pipelined / GPU | 8 | 2.6 GB |
| `minicpm5_2b_gpu_int4lin` | MiniCPM5-2B | macOS `--compression 4bit` preset (int4 per-block-32 symmetric linear, dynamic) — **speed arm only, not gated** | pipelined / GPU | 4 | 1.3 GB |
| `minicpm5_2b_ane_pal8` | MiniCPM5-2B | stock iOS static, 8-bit k-means g32, AOT h18p, 31/31 (resources.bin 2.16 GB) | StaticShapeEngine / ANE | 8 | 2.9 GB |
| `qwen3_1_7b_ane` | Qwen3-1.7B | stock iOS static `4bit_weight_palettized_group32`, `--max-context-length 4096`, AOT h18p, 31/31 | StaticShapeEngine / ANE | 4 | 1.1 GB |
| `qwen3_1_7b_ane_6bit` | Qwen3-1.7B | Apple's registry iOS preset for `qwen3-1.7b` = `models/qwen3/qwen3_1_7b_6bit.yaml` (6-bit k-means g8), AOT h18p, 31/31 | StaticShapeEngine / ANE | 6 | 1.7 GB |
| `qwen3_1_7b_zoo_iosgpu` | Qwen3-1.7B | zoo `mlboydaisuke/qwen3-1.7b-CoreAI-official` `ios-gpu/` — dynamic int4 linear, pre-AOT'd h18p `.aimodelc` (coreai-build 3600.67.5.8.1, 2026-06-18, beta toolchain), bytes identical to HF | pipelined / GPU | 4 | 1.0 GB |
| `qwen3_1_7b_gpu_int4lin` | Qwen3-1.7B | the same `4bit` preset re-exported today (dynamic IR, coreai-torch 0.4.2) — same-toolchain control for the zoo bundle | pipelined / GPU | 4 | 0.93 GB |

All exports: coreai-models checkout `coreai-models-rebase` f7a75ec (zoo-0.4, coreai-torch 0.4.2),
Xcode 27.0 RC `coreai-build` 3600.83.1, `--experimental --compute-precision float16`. MiniCPM5 needs the
zoo overlay's one-line `llama → mistral` remap; Qwen3 is Apple's builder untouched. Count
`*ANE_region*` after every AOT — 0 is the silent GPU fallback.

**A premise correction on the way**: the rollout plan read Apple's `qwen3-1.7b` iOS preset as
"compression none = fp16". The preset's `compression` field is `none` but it carries
`compression_config = models/qwen3/qwen3_1_7b_6bit.yaml` (upstream PR #209; the zoo-0.4 checkout
predates it and has no 1.7b entry at all). Apple's own iOS answer for this size is 6-bit, not 4-bit —
which is why the 6-bit arm exists here as the fallback if 4-bit fails the gate.

## 3. Sustained decode, 10 minutes (protocol v1 run, 1B)

Each arm: p128 g256 trials back to back for 600 s after a warmup. Order ANE → GPU. Wired = charging, so
battery stayed at 95 % for both arms and says nothing (see §6).

| arm | trials | tokens | decode mean | first 3 → last 3 trials | min / max | prefill mean | thermal |
|---|---|---|---|---|---|---|---|
| 1B ANE 8-bit (static, AOT) | 133 | 34,048 | 58.9 | **75.8 → 52.5** | 49.1 / 77.7 | 2655 (3900 at start → 2350) | `fair` at start, `serious` from 72 s on |
| 1B GPU int8 (dynamic, pipelined) — **started at `serious`** (inherited the ANE arm's heat: a fixed 3 min idle did not clear it) | 78 | 19,968 | 34.6 | 62.8 → 33.8 | 30.4 / 66.8 | 1082 | `serious` throughout |

Trajectories (decode tok/s at elapsed s, thermal in brackets):

- ANE: 3 s 76.2 (fair) · 61 s 75.5 (fair) · 119 s 55.6 (serious) · 181 s 58.1 · 240 s 55.6 · 302 s 53.8 ·
  361 s 49.1 · 421 s 52.2 · 481 s 52.2 · 542 s 52.1 · 601 s 52.2. Prefill 3600–4000 → 2350 over the same
  step. The drop is one step at the `fair → serious` transition, then a plateau at ~52 = **−31 %** from the
  first minute (the same-day A/B's 76.8 is the first-minute number).
- GPU (hot start): 4 s 61.4 · 28 s 42.2 · 58 s 30.9 · then 31–34 for the rest; prefill 2269 → ~1000.
  The 67 tok/s of the same-day A/B was measured cold-ish (fresh launch after 20 s idle); under a sustained
  `serious` state the GPU int8 1B runs at about half of it.

What this run isolates and what it does not: **the ANE plateau under sustained load is established
(one arm, one model)**; the GPU row is *not* the GPU's sustained curve from a comparable start, because
it began throttled. The v2 protocol (idle until the thermal state returns to `fair`, cap 10 min) is implemented
(`Sustain.swift`) but **was not run** — the phone left before the queue reached it. "ANE does not drop over 5 trials" (the earlier observation) was a
one-minute window; at ten minutes it drops like everything else on this phone once the thermal state
goes `serious`. Charging over USB with the screen on is itself a heater (memory
`reference_iphone_screenlock_gpu_cap`), so an unplugged run may plateau higher — that is the
battery protocol's job (§6).

## 4. Equal-byte A/B at 2B (2026-09-16 02:5x–03:4x, 24A437, USB)

| pair (p128 g256 n5, A-B-A-B) | ANE (round 1 / 2) | GPU (round 1 / 2) | read |
|---|---|---|---|
| (i) 4-bit: ANE 4-bit k-means g32 (the shipped `ios-ane-h18p/`) vs GPU int4 per-block-32 linear dynamic (speed arm, not gated) | 19.7 / 23.6 decode, 622 / 1033 prefill (load 13.0 s / 0.37 s) | 17.5 / 17.0 decode, 337 / 382 prefill (load 2.6 s / 1.9 s) | ANE ahead by 13–39 % at equal bytes — but **both arms are far below their 24A435 levels** (this ANE bundle did 55.4 / 51.5 the morning before; GPU int8 does 22.8), and the phone's battery/thermal state was not logged in this mode. Same phone state for both arms; absolute values not comparable across the OS update. Record: `models/minicpm5-2b/bench-iphone-equal-byte-4bit-2026-09-16.json` |
| (ii) 8-bit: ANE 8-bit k-means g32 (2.9 GB, resources.bin 2.16 GB) vs GPU int8 per-block-32 dynamic (the shipped `int8/`) | **loads (warm 0.2–0.3 s, footprint ~2.1 GB) but its first generation never returns** — 25 min cap hit in both rounds, the same on 24A437 the evening before after a 325 s cold load, and a third time (2026-09-16 10:1x) with a streaming run: a 24-token prompt produced **no first token in 4 min** — not slow, dead | 24.3 decode / 938 prefill (round 1, cold specialization 13.8 s); round 2 not run | No 8-bit pair: the 2B at 8 bits is past what the ANE path executes here, so the 2B ships on the ANE at 4 bits only — and 4 bits is not fp32-faithful on long answers (§4b). An int8-aligned 2B comparison therefore does not exist on this path; the
untested middle is a 6-bit (≈2.2 GB) or mixed 4/8-bit export in Apple's own style. This is a new failure shape — not the load-time `NSPOSIXErrorDomain 2` wall, an inference that does not come back |

So the morning's "2.4× at 2B" still cannot be split into "ANE" and "half the bytes": the equal-byte 4-bit
pair exists only in a degraded phone state, and the 8-bit pair does not exist. What the 4-bit pair does say,
within its own run, is that the GPU's int4 linear dynamic path is no faster than its int8 path on this
model (17 vs ~22–24), so the byte saving on the GPU side does not buy decode speed the way it does on the ANE.

### 4b. Fidelity on a longer chat prompt (2026-09-16) — the gate's three prompts were not enough

Recording the demo (§8) put one more prompt through both 2B bundles — "Explain in a short paragraph why the sky is
blue.", no-think template, greedy, identical prompt ids (24) on both arms — and an fp32 `transformers` greedy run
of the same ids (109 tokens to EOS) says which bundle is the model:

| bundle | first divergence from fp32 | oracle margin there | what the answer became |
|---|---|---|---|
| ANE 4-bit k-means g32 (`ios-ane-h18p/`) | **step 0** (`The` → `To`) | **1.0** | a different paragraph, physically wrong ("shorter wavelengths travel faster") |
| GPU int8 per-block-32 (`int8/`) | step 19 (` interacts` → ` is`) | 0.094 (below the 0.1 floor = a near-tie) | the fp32 Rayleigh explanation, reworded |

Record: `models/minicpm5-2b/fidelity-probe-sky-2026-09-16.json`. Read: the device gate (alphabet 24, "capital of
France" 8, counting 16 — 48 teacher-forced steps) passed the 4-bit 2B, and a 109-token free-form answer breaks it
at its very first token with the oracle certain. So the "2.4× / 55 tok/s" bundle is fast **and not fp32-faithful on
this kind of prompt**, while the int8 GPU bundle is (to the floor). Two consequences: the gate needs a longer
free-form chat prompt in its fixture (cheap: it is one more oracle rollout), and any speed table that puts the 4-bit
ANE bundle next to the int8 GPU bundle must say the two do not answer the same way. The 1B pair on the same prompt (fp32 1B oracle, 87 tokens): **both bundles are faithful to the floor** — the
ANE 8-bit bundle matches 21 tokens, the GPU int8 bundle 6 tokens, each diverging only on a 0.051 near-tie; decode
73.2 vs 62.3 tok/s on that run (`models/minicpm5-1b/fidelity-probe-sky-2026-09-16.json`). So the equal-byte 1B pair
is the comparison that is clean on both axes — bytes and fidelity; the 2B 4-bit pair is not.

## 5. Third family: Qwen3-1.7B on Apple's own builder — gated (24A437)

Gate = `_ane_gate` (fp32 `transformers` oracle, teacher-forced sweep + free-run incl. the stop). Prompts:
alphabet list 24 steps (min margin 0.99), chat "Say hello." 12 steps incl. EOS (min 0.730, EOS 1.000),
396-id counting prompt 16 steps (min 0.94). The poisoned fixture (natural[5] swapped for the oracle's
2nd-best) must FAIL first.

| ANE arm | red (must fail) | clean | load |
|---|---|---|---|
| 4-bit k-means g32 (`4bit_weight_palettized_group32`), 1.1 GB | FAIL as required (poison caught; **chat also fails**) | **FAIL** — natural 24/24 and long 16/16 exact, but the chat turn collapses: TF 3/12 (step 0 expects `Hello` 9707, gets `<\|endoftext\|>` 151643; steps 1, 2, 3, 5, 7, 9, 10, 11 also wrong at oracle margins 0.73–1.0), free-run emits 151643 twenty times | 5.0 s cold / 0.08 s warm, footprint 1.7 GB |
| **6-bit k-means g8 — Apple's registry iOS preset for `qwen3-1.7b`** (`models/qwen3/qwen3_1_7b_6bit.yaml`), 1.7 GB | FAIL as required (only the poisoned prompt) | **PASS 3/3** — natural 24/24, chat 12/12 incl. the stop, long 16/16 | **795 s cold** (first launch, ANE program build), 0.15 s warm, footprint 2.1 GB |

Read: the 4-bit failure is not the MiniCPM5-1B pattern (a fluent alternative answer); it is a collapse
confined to the one prompt that carries the chat template's special tokens (`<|im_start|>`, `<|im_end|>`,
the empty `<think>` block — ids ≥ 151643). Raw-text prompts are token-exact at 4-bit. Apple's own answer
for this size is 6 bits, and at 6 bits the same template is clean — so **Qwen3-1.7B ships on the ANE at
Apple's 6-bit preset, not at the 4-bit default**. (The isolation arm "4-bit + fp16 embeddings" could not be
run on the ANE: `--disable-embedding-quantization-ios` keeps the table in float32 and `coreai-build`
then emits **0 ANE regions** — the whole graph falls back to the GPU. Which also means the morning's
MiniCPM5-1B "fp16 embeddings reproduce the flips" arm should be re-read with its region count in hand.)
The zoo card's June-beta claim that the 1.7B static ANE export "loads but fails to invoke" no longer
holds on 27.0 GA: it loads, invokes and passes the gate.

Speed — same-day 3-arm A/B (p128 g256 n5, A-B-C-A-B-C, 24A437, USB), decode / prefill tok/s, round 1 / round 2:

| arm (weight bits) | round 1 | round 2 | load |
|---|---|---|---|
| ANE 4-bit k-means g32, static AOT (4) — fails the gate, speed only | 34.4 / 1146 | 55.5 / 2211 | 0.28 s / 0.24 s (warm; the cold ANE build of this bundle took minutes after the OS update) |
| GPU int4 per-block-32 linear, dynamic IR exported today (4) | **63.4** / 1162 | **60.9** / 1160 | 3.9 s cold specialization / 0.97 s |
| zoo `ios-gpu/` (June-beta AOT'd dynamic int4) | did not load | did not load | `invalidState("Failed to find an extend function with the max context length of 40960")` |

Read: **at equal byte width (4 vs 4 bits) the GPU decodes faster than the ANE on Qwen3-1.7B** (61–63 vs
34–56), the opposite sign of MiniCPM5-1B's 8-bit pair (+17 % for the ANE). The ANE's round 1 is depressed
(it ran ~40 min after a 13-minute cold ANE program build on a sibling bundle; thermal state was not logged
in this mode), so 55.5 is the steadier ANE figure — still below the GPU. The 6-bit ANE arm that actually
passes the gate was not speed-measured (the session stopped on the owner's clock). The June-beta `ios-gpu/`
bundle in the zoo repo does not load on 27.0 GA with the current engine; the same preset re-exported
today does. Record: `models/qwen3-1.7b-official/bench-iphone-ane-vs-gpu-2026-09-15.json`.


## 6. Power (battery-% method) — needs the cable out

`UIDevice.batteryLevel` / `batteryState` are logged per trial; wired, the state is `charging` and the
level did not move over 34k generated tokens (95 → 95). `devicectl device info details` exposes neither
battery nor free disk. The instrument therefore works, but the measurement needs one physical action:
launch the sequence, unplug, come back after ~26 min (2 arms), replug, pull
`Documents/sustain/<tag>.log`; tokens generated ÷ battery-% drop is the metric (its name is the
owner's to choose). The Xcode 27 RC `xctrace` has a **Power Profiler** template; attaching Instruments
loads the phone, so it is a separate experiment, not a substitute.

## 7. Traps met on the way

- **An iOS build update invalidates every specialization cache** (`Library/Caches/coreai-cache/<OS build>/`): after
  24A435 → 24A437 each ANE bundle rebuilt its programs on first load — 2B 8-bit 325 s, Qwen3-1.7B 6-bit 795 s, 2B 4-bit
  > 15 min — and the old build's ~22 GB of cache became dead weight (free space 38 → 18.5 GB). "Load 0.2 s" is a
  warm-cache number; budget minutes for the first load after any OS update, and stub the old cache.
- **`devicectl … launch --console` can be refused for minutes** (`CoreDeviceError 10002`, and without `--console` the
  reason shows: `Busy ("Application failed preflight checks")`) after a multi-GB install over Wi-Fi, or after an app
  was terminated mid-ANE-load. A `--start-stopped` launch bypasses that check, so it is not a readiness probe. What
  worked: launch **without** `--console`, have the app mirror its log lines to `Documents/sustain/<tag>.log`, and pull
  the file with `devicectl device copy from` (`_launch_f.sh`, `AB_LOG` / `AG_LOG`).
- **`--disable-embedding-quantization-ios` is not an ANE arm**: the float32 table makes `coreai-build` emit 0 ANE
  regions (whole graph on the GPU). Isolate embeddings on the ANE some other way.
- **`CoreDeviceError 10002 (Invalid argument)` on `process launch` right after `install app`**: the
  launch dies before the app prints anything; the pollers waited out their 900 s cap twice. Fix: sleep
  ~10 s after an install, retry a failed launch, and make every poller break on `ERROR:` (done in
  `_run.sh`, `_ab.sh`, `_abn.sh`, `_sustain.sh`, `_probe.sh`).
- **`_install.sh`'s "device BUSY" guard matched another Claude session** whose prompt quoted the
  `devicectl device process launch` command line; the guard now anchors on the executable.
- **`make_fixture.py`'s chat prompt is capped at 16 steps** — a prompt whose EOS lands at step 20 yields
  a chat fixture without the stop. Pick a prompt that stops within 16 (`explore_chat.py` prints the length).
- **Derived data in another session's scratchpad gets deleted under you** (the parent session's cleanup
  removed a freshly built 4 GB app); the build scripts now default to `/private/tmp/ane_gate_dd/<name>`.
  Two `_build.sh` in the same directory at once clobber each other's `project.yml` — one at a time.
- **Editing a running zsh script** shifts its read offset (`command not found: onfiguration` at the end).
- The shared container's specialization cache was 22.4 GB before this run; free space is not readable
  from `devicectl`, so the bench app now prints `free_gb=` on launch.

## 8. Demo recording and reproduce

```
# exports (coreai-models-rebase checkout; the driver is ondevice/_ane_gate/../scratchpad in this session,
# the commands are the stock CLI):
uv run coreai.llm.export openbmb/MiniCPM5-2B --experimental --compute-precision float16 --compression 4bit --output-dir …   # GPU int4 linear
uv run coreai.llm.export openbmb/MiniCPM5-2B --platform iOS --experimental --compute-precision float16 --max-context-length 4096 \
    --compression-config conversion/minicpm5_pal8_g32.yaml --output-dir …                                             # ANE 8-bit
uv run coreai.llm.export Qwen/Qwen3-1.7B --platform iOS --experimental --compute-precision float16 --max-context-length 4096 \
    --compression 4bit_weight_palettized_group32 --output-dir …                                                        # ANE 4-bit
xcrun coreai-build compile <x.aimodel> --output <dir> --platform iOS --preferred-compute neural-engine --architecture h18p
find <x.h18p.aimodelc> -name '*ANE_region*' | wc -l        # must not be 0
# device (ondevice/_ane_gate/bench):
BUNDLE=<devbundle A> BUNDLE2=<devbundle B> [BUNDLE3=…] ./_build.sh && ../_install.sh "$(cat _app_path.txt)"
./_ab.sh <tag> <A> <B> 2 128 256 5                   # same-day interleaved speed
./_sustain.sh <tag> <A,B> 600 180 128 256            # 10 min per arm, thermal-recovered start, battery/thermal per trial
```

## 9. The 2B on the ANE with Apple's other recipes (2026-09-16, S5) — none is fp32-faithful, and why

§4b left the 2B ANE bundle as "fast but not fp32-faithful": the shipped 4-bit g32 export diverges from fp32 at step 0
of a 109-token answer. This section tries Apple's other iOS recipes at 2B, with the gate strengthened first.

**Gate.** `make_fixture.py --chat-n 128` adds a fourth prompt, `sky` ("Explain in a short paragraph why the sky is
blue.", no-think, greedy, 109 oracle tokens incl. the stop, ids identical to the §4b probe). Its six sub-floor steps
(oracle margin < 0.1) are excluded from the teacher-forced sweep and end the free-run judgement (`knife_ok`); the
poisoned fixture (natural[5]) went RED first on the same app. The old three prompts are byte-identical to 2026-09-15.

**Arms** (all `coreai.llm.export --platform iOS --max-context-length 4096`, coreai-models-rebase f7a75ec / coreai-torch
0.4.2, AOT h18p neural-engine; 8-bit layers for the mixed shapes chosen by a per-layer 4-bit sensitivity scan — layer 7 alone
palettized at 4-bit flips 40 of the 103 margin-clear sky steps, layer 1 27, the last layer 41 11, every other layer ≤ 8 —
so the nine = 1, 6, 7, 8, 10, 11, 16, 40, 41, Apple's 21 % ratio):

| recipe | weights on disk | ANE regions | cold program build | gate (natural / chat / long / sky) |
|---|---|---|---|---|
| 4-bit g8 (preset `4bit_weight_palettized_group8`) | 1.3 GB IR | **0** — `ANECCompileOffline() failed … CompilationFailure` on the first graph (`extend_1024_16`), reproduced alone; `coreai-build` exits 0 and puts the whole model on the GPU | — | not an ANE arm |
| mixed 4-bit **g8** + 9 layers 8-bit per-tensor (the `qwen3_0_6b_mixed_4bit_8bit.yaml` shape) | 1.5 GB IR | **0** — same failure | — | not an ANE arm |
| mixed 4-bit **g32** + 9 layers 8-bit per-tensor (the `qwen3_4b_mixed_4bit_8bit.yaml` shape) | 1.6 GB | 31/31 | 82 s | 24/24 / **flip at 5** (` Paris` → ` **`, margin 0.345) / 16/16 / **flip at 9** (24 of 109 TF steps) — FAIL |
| **6-bit g8** (the `qwen3_1_7b_6bit.yaml` preset shape) | **2.5 GB, resources.bin 1.70 GB** | 31/31 | **1406 s** (23 min), warm 0.2 s, footprint 2.6 GB | 24/24 / 8/8 incl. the stop / 16/16 / **33 exact, flip at 33** (margin 0.168; 5 margin-clear TF flips of 109) — FAIL, the closest |
| 8-bit g32 (§4, 2.9 GB, resources.bin 2.16 GB) | | 31/31 | | loads, first generation never returns |

Transcripts: `models/minicpm5-2b/gate-minicpm5-2b-ane-6bit-device.json`, `…-mixed48g32-FAIL.json`. So the boundary on this phone is
between **1.70 GB of weights (runs) and 2.16 GB (never returns)**, not "≈ 2 GB of bundle": the 6-bit 2B is a 2.5 GB
bundle that works.

**Recipe or chip?** `simulate_recipe.py` applies each yaml to the HF fp32 weights with coreai-opt's own k-means (the
exporter's palettizer — a home-made Lloyd k-means was 1.5× worse on these heavy-tailed tensors and useless) and judges the
fixture teacher-forced, the device rule. It reproduces the device: 4-bit g32 → sky flips at step 0; the mixed arm → chat 5,
sky 9, 25 flips (device 24); 6-bit → first flip at 33 (device: 33 too, plus three more — the phone runs fp16 with int8
embeddings, the simulation fp32). Then it answers what no device run can cheaply: **no palettization below 8 bits reaches
fp32 to the floor on this answer** — 6-bit g8, 6-bit + the 3 outlier layers at 8-bit, 6-bit + 9 layers at 8-bit, and 6-bit + the five layers that flip a sky step
when palettized alone at 6-bit (5, 7, 9, 16, 22 — a 6-bit sensitivity scan) all flip step 33 (0.168) and 91 (0.247); the
nine 8-bit layers do fix the chat-turn flip at step 5. The residual is cumulative across the 6-bit layers, not a few outliers. 8-bit g32 k-means (108/109 + one
knife-edge) and the shipped GPU recipe, int8 per-block-32 linear (109/109), are clean. The step-5 flip of the chat turn
(` Paris` vs ` **`, fp32 margin 0.345) is a razor edge every 4–6-bit recipe lands on; the shipped 4-bit bundle keeps it on
the device by a 0.28 fp16 logit gap. Records: `ondevice/_ane_gate/fixtures/minicpm5_2b/{layer_sensitivity_4bit_g8,
recipe_simulation_fp32,recipe_simulation_fp32_b}.json`.

**Speed of the arm that runs** (same-day, thermal-recovered: engine released, idle until `thermalState ≤ fair`, then
60 s of p128 g256 trials — `models/minicpm5-2b/bench-iphone-ane-6bit-vs-gpu-2026-09-16.json`): 6-bit ANE **38.5 tok/s**
decode, flat over the minute at `fair` (prefill 1679); GPU int8 (shipped) 23.8 mean, 26.0 → 21.6 as the state reached
`serious` (prefill 948). The plain A-B-A-B right after the 23-minute program build read 23.9 / 21.4 vs 13.5 / 12.5 —
the phone was at `serious` throughout, which is why the A/B mode now needs the thermal state next to every number.
The 4-bit ANE bundle measured 54.8 / 55.7 in fresh containers the same morning (§9b), so at 2B: 4-bit ≈ 55 (unfaithful),
6-bit ≈ 38 (nearly), GPU int8 ≈ 24 (faithful).

**Consequence for the zoo card.** The owner set the bar at "runs on the ANE, answers do not break, faster than the GPU"
rather than token-exact, so the **6-bit bundle replaced the 4-bit one as `ios-ane-h18p/`** (2026-09-16; `ios-static/` is
its IR). The card says what it is: exact on the three short prompts, a 0.17-margin divergence at token 33 of the long
answer, 38.5 tok/s, a 23-minute first load. A token-exact 2B ANE bundle would need an 8-bit recipe the ANE path can
execute below ~2 GB, which this model does not fit; the task-accuracy gate (§9a) says the 6-bit bundle matches the checkpoint and the 4-bit one did not.

### 9a. Task accuracy: the token gate was not the quality gate (2026-09-17)

The owner asked the right question — why gate on token match at all — and the answer is a 20-point hole. GSM8K test, first
200 questions in the litertlm-convert order, 0-shot CoT asking for `#### <number>`, chat template no-think, greedy, 640 new
tokens, the litertlm-convert extraction and normalization (`ondevice/_ane_gate/bench/{TaskEval.swift,_task_f.sh,
score_gsm8k.py,ref_gsm8k_hf.py}`; record `models/minicpm5-2b/gsm8k-200-2026-09-17.json`):

| arm | correct / 200 |
|---|---:|
| fp32 checkpoint (bf16 on MPS, Mac) | 172 (86.0 %) |
| **6-bit g8 ANE bundle, on the phone** | **173 (86.5 %)** |
| 6-bit g8 recipe applied to the fp32 weights (Mac, coreai-opt's k-means) | 172 (86.0 %) |
| int8 per-block-32 recipe (the shipped `int8/`) applied to the fp32 weights (Mac) | 171 (85.5 %) |
| **4-bit g32 ANE bundle (the 2026-09-15 `ios-ane-h18p/`), on the phone** | **131 (65.5 %)** |

Three things follow. The 4-bit bundle that passed the three-prompt token gate loses 20 points of task accuracy — the
token gate's 48 steps never saw a multi-step answer, and the "fast but not fp32-faithful" caveat understated it. The 6-bit
bundle costs nothing (173 vs 172; 200 questions resolve ±3.5 pt), which is why it replaced the 4-bit one on HF. And the Mac
simulation of a recipe (coreai-opt's own k-means on the HF fp32 weights, then the same prompts) reproduces the phone within
0.5 pt, so the quality gate no longer needs the phone: `ref_gsm8k_hf.py --recipe <yaml|preset>` runs before any export, the
device run confirms. The int8 GPU bundle on the phone stopped at 30 questions on a Metal `GPU Timeout` (S=1 prefill via
`COREAI_CHUNK_THRESHOLD=1` at ~10 s per prompt, the phone at `serious`) — 23/30 correct there, the fp32 reference 22/30 on
the same 30; the Mac number stands in for the rest. The 1B pair on the same 200 (Mac): fp32 123, the shipped 8-bit ANE recipe
124, the shipped int8 recipe 129 — no loss on either 1B bundle.

### 9b. First load: what the cold build costs and where the cache lives

Measured in a **fresh app container** (`APP_ID=com.coreai.gemmaplebench`, provisioned but not otherwise used; uninstall =
wipe), bench app n=1, the shipped 4-bit g32 bundle:

| launch | engine load |
|---|---|
| 4096-context bundle (30 graphs), fresh container | **69.0 s** |
| same container, second launch | 0.25 s |
| 1024-context bundle (18 graphs, `--max-context-length 1024`, 19/19 regions), fresh container, first time on this phone | **44.7 s** (−35 %; decode unchanged at 54.9) |
| 1024-context bundle, a second fresh container | **7.0 s** |
| 4096-context bundle, fresh container **with the first container's `Library/Caches/coreai-cache` pushed back** (`devicectl device copy from` → uninstall → install → `copy to`, plain, 1.4 GB, 52 entries) | **0.062 s** |

So the specialization cache is two-level — a system-side part survives the container (45 → 7 s) and the per-container part
(`Library/Caches/coreai-cache/<OS build>/<bundle id>/<hash>/<hash>/model.aimodelx/…`) is what makes a launch warm — and it
**can be pre-seeded**: copy it into the app's container and the first launch is warm, provided the OS build, the bundle id
and the two hashes match. It is as large as the bundle (the ANE programs carry the weights again), so shipping it doubles
the download. The 6-bit bundle's 23-minute build and S1's 795 s (Qwen3-1.7B 6-bit) say the build time grows much faster
than the weight bytes for 6-bit LUTs; S1's 225 s / ">15 min" for this same 4-bit bundle did not reproduce (69 s today).

Traps met: `devicectl device info apps` truncated with `head` made an installed app look uninstalled — `uninstall app`
wipes its container (the Gemma PLE bench app of 2026-08 was lost this way; it embedded its model, so only the app and its
cache); a `tail -F` monitor on a log that `_launch_f.sh` re-pulls every 10 s re-emits the whole file; `coreai-build` says
nothing when the ANE compiler rejects a graph — count `*ANE_region*` every time.

