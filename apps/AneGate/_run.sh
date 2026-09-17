#!/bin/zsh
# Launch AneGateRunner (installed under the borrowed com.coreai.pipelinedbench id) with the
# console attached, poll for GATE_SUMMARY/FATAL, kill the local devicectl.
#   ./_run.sh <tag> <fixture-stem>   (stem is relative to fixtures/, without .json)
# e.g. ./_run.sh red qwen3_0_6b/fixture_red ; ./_run.sh clean qwen3_0_6b/fixture
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
TAG=${1:-run}
FIX=${2:?fixture stem, e.g. qwen3_0_6b/fixture}
EXTRA=$3
DIR=$(cd "$(dirname "$0")" && pwd)
LOG=$DIR/_device_${TAG}.log
: > $LOG
launch_retry $LOG --console --terminate-existing --environment-variables "{\"AG_FIXTURE\":\"$FIX\"$EXTRA}" com.coreai.pipelinedbench
for i in {1..450}; do  # cap 900 s; a cold run pays specialization, never kill mid-flight
  grep -qE "GATE_SUMMARY|FATAL|terminated|ERROR:" $LOG && break
  sleep 2
done
sleep 2
kill $CPID 2>/dev/null
echo "[$TAG] results:"
grep -E "model dir|fixture|engine loaded|^GATE|FR .* (PASS|FAIL)|FATAL|terminated|KNIFE|FAIL" $LOG
echo "[$TAG] DONE (full log: $LOG)"
