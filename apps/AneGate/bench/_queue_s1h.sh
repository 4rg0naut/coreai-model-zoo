#!/bin/zsh
# S1 queue 9 (2026-09-16 02:4x): MiniCPM5-2B equal-byte A/Bs (eq4, eq8) → restore PipelinedBench (plain install, no probe launch).
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
U=A6F3E849-1947-5202-9AD1-9C881CA58EEF; G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd; Q=$B/_queue9.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
inst() { $G/_install.sh "$1" 2>&1 | tail -2 | tee -a $Q; [ ${pipestatus[1]} -eq 0 ] || { step "ABORT install failed: $1"; exit 1; }; sleep 3; }
: > $Q; step "queue9 start"
inst $DD/bench_2b_eq4/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_abn_f.sh eq4 minicpm5_2b_ane,minicpm5_2b_gpu_int4lin 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_eq4.log | tee -a $Q; step "eq4 ab done"
inst $DD/bench_2b_eq8/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app
$B/_abn_f.sh eq8 minicpm5_2b_ane_pal8,minicpm5_2b_gpu_int8 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_eq8.log | tee -a $Q; step "eq8 ab done"
APP=~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app
for a in 1 2 3; do OUT=$(xcrun devicectl device install app --device $U "$APP" 2>&1); echo "$OUT" | grep -q installationURL && { echo "restored: $(echo "$OUT" | grep installationURL | cut -c1-120)" | tee -a $Q; break; }; sleep 15; done
N=$(xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "Documents/models: $N"; step "QUEUE9 END"
