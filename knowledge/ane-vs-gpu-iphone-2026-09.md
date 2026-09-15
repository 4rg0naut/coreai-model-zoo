# Neural Engine vs GPU on the iPhone 17 Pro (2026-09) — what a same-day A/B does and does not isolate

> **Status (2026-09-15, evening):** measured — the 1B sustained curves (v1 protocol), the Qwen3-1.7B gate at 4- and 6-bit,
> and the Qwen3-1.7B 3-arm A/B. **Not obtained** — the 2B equal-byte pairs and the thermal-recovered (v2) sustained runs:
> the phone was updated to 24A437 mid-session (every specialization cache went cold), then left. Sections say which.
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

## 4. Equal-byte A/B at 2B — not obtained

Both pairs were exported and built into A/B apps (`minicpm5_2b_ane` vs `minicpm5_2b_gpu_int4lin`, 4 vs 4 bits;
`minicpm5_2b_ane_pal8` vs `minicpm5_2b_gpu_int8`, 8 vs 8 bits) but no interleaved round completed on
24A437: after the OS update every ANE bundle rebuilds its programs on first load (the 2B 4-bit did not finish
within 15 min, the 2B 8-bit took 325 s), and the phone was disconnected before a warm round could run. The
one number that exists is a single trial of the GPU int4 linear 2B — 13.7 tok/s decode, below the int8
bundle's 22.8 — too little to put in a table. What stands for the 2B is the morning's shipped-vs-shipped
pair (ANE 4-bit 55.4 / 51.5 vs GPU int8 22.8 / 20.6, half the weight bytes on the ANE side), so **the
"2.4×" is still not separated into "ANE" and "4-bit"**. The 2B 8-bit ANE bundle does load and run on the
phone (2.9 GB on disk, resources.bin 2.16 GB) — the load wall did not bite.

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

## 8. Reproduce

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
