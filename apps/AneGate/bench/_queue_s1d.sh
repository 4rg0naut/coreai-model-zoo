#!/bin/zsh
# S1 shortened queue (17:5x, USB back): eq8 → Qwen 3-arm → 6-bit gate → eq4 → sustained 5-min arms (1B, 2B) → restore.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd
Q=$B/_queue5.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
inst() { $G/_install.sh "$1" 2>&1 | tail -2 | tee -a $Q; [ ${pipestatus[1]} -eq 0 ] || { step "ABORT install failed: $1"; exit 1; }; sleep 3; }
gate() { for a in 1 2 3; do $G/_run.sh $1 $2 > /dev/null 2>&1; grep -q "GATE_SUMMARY" $G/_device_$1.log && break; sleep 10; done; grep -E "GATE_SUMMARY|FATAL" $G/_device_$1.log | tee -a $Q; }
: > $Q
step "queue5 start"
cd $B && $B/_ab.sh eq8 minicpm5_2b_ane_pal8 minicpm5_2b_gpu_int8 2 128 256 5 > /dev/null 2>&1
grep -E "^r[0-9]" $B/_ab_eq8.log | tee -a $Q; step "eq8 ab done"
inst $DD/bench_qwen/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_abn.sh qwen qwen3_1_7b_ane,qwen3_1_7b_zoo_iosgpu,qwen3_1_7b_gpu_int4lin 2 128 256 5 20 > /dev/null 2>&1
grep -E "^r[0-9]" $B/_abn_qwen.log | tee -a $Q; step "qwen ab done"
inst $DD/gate_6bit/Build/Products/Release-iphoneos/AneGateRunner.app
gate qwen6_red qwen3_1_7b/fixture_red; gate qwen6_clean qwen3_1_7b/fixture; step "qwen 6bit gate done"
inst $DD/bench_2b_eq4/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_ab.sh eq4 minicpm5_2b_ane minicpm5_2b_gpu_int4lin 2 128 256 5 > /dev/null 2>&1
grep -E "^r[0-9]" $B/_ab_eq4.log | tee -a $Q; step "eq4 ab done"
inst $DD/bench_1b/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_sustain.sh 1b_v2 minicpm5_1b_gpu_int8,minicpm5_1b_ane_pal8 300 120 128 256 > /dev/null 2>&1
grep -E "SUSTAIN_START|SUSTAIN_STATS|idle_end|FATAL" $B/_device_sustain_1b_v2.log | cut -c1-400 | tee -a $Q; step "sustain 1b v2 done"
inst $DD/bench/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_sustain.sh 2b_v2 minicpm5_2b_gpu_int8,minicpm5_2b_ane 300 120 128 256 > /dev/null 2>&1
grep -E "SUSTAIN_START|SUSTAIN_STATS|idle_end|FATAL" $B/_device_sustain_2b_v2.log | cut -c1-400 | tee -a $Q; step "sustain 2b v2 done"
inst ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app
N=$(xcrun devicectl device info files --device A6F3E849-1947-5202-9AD1-9C881CA58EEF --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "restored PipelinedBench; Documents/models: $N"; step "QUEUE5 END"
