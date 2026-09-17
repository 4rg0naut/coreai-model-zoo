#!/bin/zsh
# S1 queue part 3 (after _queue_s1b.sh): re-run the eq4 A/B that hit the Busy window, then restore PipelinedBench.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd
Q=$B/_queue4.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
inst() { $G/_install.sh "$1" 2>&1 | tail -2 | tee -a $Q; [ ${pipestatus[1]} -eq 0 ] || { step "ABORT install failed: $1"; exit 1; }; sleep 5; }
: > $Q
step "queue4 start"
inst $DD/bench_2b_eq4/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
cd $B && $B/_ab.sh eq4 minicpm5_2b_ane minicpm5_2b_gpu_int4lin 2 128 256 5 > /dev/null 2>&1
grep -h -E "free_gb" $B/_device_ab_eq4_r1_*.log | head -1 | tee -a $Q
grep -E "^r[0-9]" $B/_ab_eq4.log | tee -a $Q
step "eq4 ab done"
inst ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app
N=$(xcrun devicectl device info files --device A6F3E849-1947-5202-9AD1-9C881CA58EEF --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "restored PipelinedBench; Documents/models: $N"
step "QUEUE4 END"
