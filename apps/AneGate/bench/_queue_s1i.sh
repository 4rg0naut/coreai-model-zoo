#!/bin/zsh
# After queue9's eq8: stop queue9 before its restore, probe the phone state (battery/bstate/thermal) with a 40 s
# sustained-mode launch on the still-installed eq8 app, then restore PipelinedBench (plain install).
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
U=A6F3E849-1947-5202-9AD1-9C881CA58EEF; G=~/code/coreai/ondevice/_ane_gate; B=$G/bench; Q=$B/_queue10.log
step() { echo "STEP $(date +%H:%M:%S) $*" | tee -a $Q; }
until grep -q "eq8 ab done" $B/_queue9.log 2>/dev/null; do sleep 5; done
pkill -f "_queue_s1h.sh"; sleep 2
for p in $(ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install)" | awk '{print $1}'); do kill $p 2>/dev/null; done
: > $Q; step "queue10 start (probe)"
$B/_sustain_f.sh probe minicpm5_2b_gpu_int8 40 0 128 256 | tee -a $Q
grep -E "phase=(load|warmup|trial)" $B/_device_sustain_probe.log | head -3 | cut -c1-260 | tee -a $Q
step "probe done"
APP=~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app
for a in 1 2 3; do OUT=$(xcrun devicectl device install app --device $U "$APP" 2>&1); echo "$OUT" | grep -q installationURL && { echo "restored: $(echo "$OUT" | grep installationURL | cut -c1-120)" | tee -a $Q; break; }; sleep 15; done
N=$(xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | head -1)
step "Documents/models: $N"; step "QUEUE10 END"
