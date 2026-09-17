#!/bin/zsh
# Sustained-generation + battery/thermal run on the installed bench app (Sustain.swift):
#   ./_sustain.sh <tag> <bundleA[,bundleB,...]> [duration_s=600] [idle_s=180] [P=128] [G=256]
# One launch runs the whole sequence (idle → load → warmup → trials until duration, per arm).
# Console lines land in _sustain_<tag>.log; the app also appends to Documents/sustain/<tag>.log in
# the shared container, so an unplugged run can be pulled afterwards with
#   xcrun devicectl device copy from --device $UDID --domain-type appDataContainer \
#     --domain-identifier com.coreai.pipelinedbench --source Documents/sustain/<tag>.log --destination .
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
# Launch with retry: over the Wi-Fi transport a launch can be refused for minutes (CoreDeviceError 10002 /
# "Busy: Application failed preflight checks") after an install or a terminate; retry every 15 s, up to 3 min.
launch_retry() {   # launch_retry <log> <devicectl args after "launch --device $UDID"...>; sets CPID
  local L=$1; shift
  for t in $(seq 1 12); do   # 3 min: wired, Busy clears within a minute once the app is gone
    : > $L
    xcrun devicectl device process launch --device $UDID "$@" > $L 2>&1 &
    CPID=$!
    for i in {1..12}; do grep -qE "Launched application|ERROR" $L && break; sleep 1; done
    grep -q "Launched application" $L && return 0
    kill $CPID 2>/dev/null; sleep 15
  done
  return 1
}
TAG=$1; SEQ=$2; DUR=${3:-600}; IDLE=${4:-180}; P=${5:-128}; G=${6:-256}
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=$DIR/_sustain_${TAG}.log
: > $OUT
ARMS=$(echo $SEQ | tr ',' '\n' | wc -l | tr -d ' ')
CAP=$(( (DUR + 600 + 240) * ARMS / 2 + 60 ))   # poll iterations of 2 s; idle can stretch to AB_IDLE_MAX_S=600 (thermal recovery)
echo "SUSTAIN $TAG seq=$SEQ duration=$DUR idle=$IDLE p=$P g=$G arms=$ARMS start $(date)" | tee -a $OUT
launch_retry $DIR/_device_sustain_${TAG}.log --console --terminate-existing --environment-variables "{\"AB_TAG\":\"$TAG\",\"AB_SEQUENCE\":\"$SEQ\",\"AB_DURATION_S\":\"$DUR\",\"AB_IDLE_S\":\"$IDLE\",\"AB_P\":\"$P\",\"AB_G\":\"$G\"}" com.coreai.pipelinedbench
for i in $(seq 1 $CAP); do grep -qE "SUSTAIN_DONE|FATAL|terminated|ERROR:" $DIR/_device_sustain_${TAG}.log && break; sleep 2; done
sleep 2; kill $CPID 2>/dev/null
grep -E "^SUSTAIN|FATAL|terminated" $DIR/_device_sustain_${TAG}.log | tee -a $OUT
echo "SUSTAIN $TAG end $(date)" | tee -a $OUT
