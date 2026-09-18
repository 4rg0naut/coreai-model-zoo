#!/bin/zsh
# S7: attach to a gate run already launched on the phone: pull Documents/sustain/<devlog> every 10 s into _device_<tag>.log
# until GATE_SUMMARY|FATAL (or ERROR), cap <iterations> x 10 s. Prints the 'engine loaded' line as soon as it appears.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP_ID=com.coreai.pipelinedbench
TAG=${1:?tag}; DEVLOG=${2:?device log name}; CAP=${3:-540}
cd ~/code/coreai/ondevice/_ane_gate; LOCAL=_device_$TAG.log; seen=0
for i in $(seq 1 $CAP); do
  sleep 10
  rm -f $LOCAL.pull; xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --source Documents/sustain/$DEVLOG --destination $LOCAL.pull >/dev/null 2>&1
  [ -f $LOCAL.pull ] && cp $LOCAL.pull $LOCAL
  if [ $seen -eq 0 ] && grep -q "engine loaded" $LOCAL 2>/dev/null; then seen=1; echo "[$(date '+%H:%M:%S')] $TAG: $(grep -m1 'engine loaded' $LOCAL)"; fi
  grep -qE "GATE_SUMMARY|FATAL" $LOCAL 2>/dev/null && break
done
grep -qE "GATE_SUMMARY|FATAL" $LOCAL || echo "ERROR: cap reached without GATE_SUMMARY ($TAG)" >> $LOCAL
echo "[$(date '+%H:%M:%S')] [$TAG] $(grep -E 'GATE_SUMMARY|FATAL|ERROR' $LOCAL | head -2)"
