#!/bin/zsh
# S1 queue 8 (after queue7): Qwen3-1.7B 6-bit ANE (gate PASS) vs zoo ios-gpu A/B, then restore PipelinedBench.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; DD=/private/tmp/ane_gate_dd; Q=$B/_queue8.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
until grep -q "QUEUE7 END" $B/_queue7.log 2>/dev/null; do sleep 15; done
: > $Q; step "queue8 start"
$G/_install.sh $DD/bench_qwen6/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app 2>&1 | tail -1 | tee -a $Q
$B/_abn_f.sh qwen6 qwen3_1_7b_ane_6bit,qwen3_1_7b_zoo_iosgpu 2 128 256 5 20 > /dev/null 2>&1; grep -E "^r[0-9]" $B/_abn_qwen6.log | tee -a $Q; step "qwen6 ab done"
$G/_install.sh ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1 | tail -1 | tee -a $Q
N=$(xcrun devicectl device info files --device A6F3E849-1947-5202-9AD1-9C881CA58EEF --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "restored PipelinedBench; Documents/models: $N"; step "QUEUE8 END"
