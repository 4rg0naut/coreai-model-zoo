#!/bin/zsh
# N-arm same-day interleaved A/B on one installed bench app that embeds all arms (S1 variant of _ab.sh):
#   ./_abn.sh <tag> <armA,armB[,armC...]> [rounds=2] [P=128] [G=256] [N=5] [gap_s=20]
# Each launch is a fresh process (AB_MODEL picks the bundle); order A B C A B C …; STATS lines are
# collected into _abn_<tag>.log. A GPU dynamic bundle pays its cold specialization on its first launch.
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
TAG=$1; ARMS=(${(s:,:)2}); ROUNDS=${3:-2}; P=${4:-128}; G=${5:-256}; N=${6:-5}; GAP=${7:-20}
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=$DIR/_abn_${TAG}.log
: > $OUT
echo "ABN $TAG arms=${(j:,:)ARMS} rounds=$ROUNDS p=$P g=$G n=$N gap=$GAP start $(date)" | tee -a $OUT
for r in $(seq 1 $ROUNDS); do
  for M in $ARMS; do
    LOG=$DIR/_device_abn_${TAG}_r${r}_${M}.log
    : > $LOG
    launch_retry $LOG --console --terminate-existing --environment-variables "{\"AB_MODEL\":\"$M\",\"AB_P\":\"$P\",\"AB_G\":\"$G\",\"AB_N\":\"$N\"}" com.coreai.pipelinedbench
    for i in {1..450}; do grep -qE "STATS|FATAL|terminated|ERROR:" $LOG && break; sleep 2; done
    sleep 2; kill $CPID 2>/dev/null
    echo "r$r $M: $(grep -E 'engine loaded' $LOG | head -1) | $(grep -E 'STATS|FATAL' $LOG | head -1)" | tee -a $OUT
    grep -E "^trial" $LOG | sed "s/^/    r$r $M /" | tee -a $OUT
    sleep $GAP
  done
done
echo "ABN $TAG end $(date)" | tee -a $OUT
