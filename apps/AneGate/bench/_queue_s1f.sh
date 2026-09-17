#!/bin/zsh
# S1 queue 7 (file-logged, no-console launches). Waits for each app's build marker, installs, runs.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd; Q=$B/_queue7.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
inst() { until [ -f $DD/$1/.built_f ]; do sleep 10; done; $G/_install.sh "$DD/$1/Build/Products/Release-iphoneos/$2" 2>&1 | tail -1 | tee -a $Q; [ ${pipestatus[1]} -eq 0 ] || { step "ABORT install failed: $1"; exit 1; }; sleep 3; }
: > $Q; step "queue7 start"
inst gate_6bit AneGateRunner.app
$G/_run_f.sh qwen6_red qwen3_1_7b/fixture_red | tee -a $Q; $G/_run_f.sh qwen6_clean qwen3_1_7b/fixture | tee -a $Q; step "qwen 6bit gate done"
inst bench_qwen AppleBenchRunnerGA.app
$B/_abn_f.sh qwen qwen3_1_7b_ane,qwen3_1_7b_zoo_iosgpu,qwen3_1_7b_gpu_int4lin 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_qwen.log | tee -a $Q; step "qwen ab done"
inst bench_2b_eq4 AppleBenchRunnerGA.app
$B/_abn_f.sh eq4 minicpm5_2b_ane,minicpm5_2b_gpu_int4lin 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_eq4.log | tee -a $Q; step "eq4 ab done"
inst bench_2b_eq8 AppleBenchRunnerGA.app
$B/_abn_f.sh eq8 minicpm5_2b_ane_pal8,minicpm5_2b_gpu_int8 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_eq8.log | tee -a $Q; step "eq8 ab done"
inst bench AppleBenchRunnerGA.app
$B/_sustain_f.sh 2b_v2 minicpm5_2b_gpu_int8,minicpm5_2b_ane 300 120 128 256 | tee -a $Q; step "sustain 2b v2 done"
inst bench_1b AppleBenchRunnerGA.app
$B/_sustain_f.sh 1b_v2 minicpm5_1b_gpu_int8,minicpm5_1b_ane_pal8 300 120 128 256 | tee -a $Q; step "sustain 1b v2 done"
$G/_install.sh ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1 | tail -1 | tee -a $Q
N=$(xcrun devicectl device info files --device A6F3E849-1947-5202-9AD1-9C881CA58EEF --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "restored PipelinedBench; Documents/models: $N"; step "QUEUE7 END"
