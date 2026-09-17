#!/bin/zsh
# S1 device queue (2026-09-15, copy of the session scratchpad device_queue2.sh), resumed after the 14:09 disconnect: Qwen gate (retry) → Qwen A/B → 2B eq4/eq8 A/B → sustained v2 → restore.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd
Q=$B/_queue2.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
inst() { $G/_install.sh "$1" 2>&1 | tail -1 | tee -a $Q; [ ${pipestatus[1]} -eq 0 ] || { step "ABORT install failed: $1"; exit 1; }; sleep 12; }   # post-install launch race (error index: 10002)
gate() { # gate <tag> <fixture-stem>; retries a failed launch (10002) twice
  for a in 1 2 3 4 5; do cd $G && ./_run.sh $1 $2 > /dev/null 2>&1; if grep -q "GATE_SUMMARY" $G/_device_$1.log; then break; fi; grep -m1 "ERROR" $G/_device_$1.log | tee -a $Q; sleep 30; done
  grep -E "GATE_SUMMARY|FATAL" $G/_device_$1.log | tee -a $Q; }
: > $Q
step "queue2 start"
# 1. Qwen3-1.7B gate (the 13:37/13:52 launches died with CoreDevice 10002 before the app printed anything)
if [ -z "$SKIP_GATE" ]; then
# gate app already installed at 13:37 (device info apps: AneGateRunner) — no reinstall
sleep 5
gate qwen_red qwen3_1_7b/fixture_red
gate qwen_clean qwen3_1_7b/fixture
step "qwen gate done"
fi
# 5. sustained with thermal-recovered starts (24A437, phone on Wi-Fi = unplugged → battery delta is live): 1B pair GPU→ANE, then 2B pair GPU→ANE
inst $DD/bench_1b/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
./_sustain.sh 1b_v2 minicpm5_1b_gpu_int8,minicpm5_1b_ane_pal8 600 180 128 256 > /dev/null 2>&1
grep -E "SUSTAIN_START|SUSTAIN_STATS|idle_end|FATAL|ERROR" $B/_device_sustain_1b_v2.log | cut -c1-400 | tee -a $Q
step "sustain 1b v2 done"
inst $DD/bench/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
./_sustain.sh 2b_v2 minicpm5_2b_gpu_int8,minicpm5_2b_ane 600 180 128 256 > /dev/null 2>&1
grep -E "SUSTAIN_START|SUSTAIN_STATS|idle_end|FATAL|ERROR" $B/_device_sustain_2b_v2.log | cut -c1-400 | tee -a $Q
step "sustain 2b v2 done"
# 2. Qwen3-1.7B 3-arm A/B (prints free_gb on first launch)
inst $DD/bench_qwen/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
cd $B && ./_abn.sh qwen qwen3_1_7b_ane,qwen3_1_7b_zoo_iosgpu,qwen3_1_7b_gpu_int4lin 2 128 256 5 20 > /dev/null 2>&1
grep -h -E "free_gb" $B/_device_abn_qwen_r1_*.log | head -1 | tee -a $Q
grep -E "^r[0-9]" $B/_abn_qwen.log | tee -a $Q
step "qwen ab done"
# 3. 2B equal-byte (i): ANE 4-bit pal vs GPU int4 linear dynamic
inst $DD/bench_2b_eq4/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
./_ab.sh eq4 minicpm5_2b_ane minicpm5_2b_gpu_int4lin 2 128 256 5 > /dev/null 2>&1
grep -E "^r[0-9]" $B/_ab_eq4.log | tee -a $Q
step "eq4 ab done"
# 4. 2B equal-byte (ii): ANE 8-bit pal vs GPU int8 dynamic
inst $DD/bench_2b_eq8/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
./_ab.sh eq8 minicpm5_2b_ane_pal8 minicpm5_2b_gpu_int8 2 128 256 5 > /dev/null 2>&1
grep -E "^r[0-9]" $B/_ab_eq8.log | tee -a $Q
step "eq8 ab done"
# 6. restore PipelinedBench and count the model container
inst ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app
N=$(xcrun devicectl device info files --device A6F3E849-1947-5202-9AD1-9C881CA58EEF --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "restored PipelinedBench; Documents/models: $N"
step "QUEUE END"
