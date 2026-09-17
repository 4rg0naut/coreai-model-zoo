#!/bin/zsh
# GA gate probe: launch the bench app with console attached, poll for STATS/FATAL, kill the local devicectl.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
TAG=${1:-run}
EXTRA=$2
DIR=$(cd "$(dirname "$0")" && pwd)
LOG=$DIR/_device_${TAG}.log
: > $LOG
xcrun devicectl device process launch --device $UDID --console --terminate-existing \
  --environment-variables "{\"AB_NOOP\":\"1\"$EXTRA}" \
  com.coreai.pipelinedbench > $LOG 2>&1 &
CPID=$!
for i in {1..450}; do  # cap 900 s; first run pays cold specialization, never kill mid-flight
  grep -qE "STATS|FATAL|terminated|ERROR:" $LOG && break
  sleep 2
done
sleep 2
kill $CPID 2>/dev/null
echo "[$TAG] results:"
grep -E "model dir|engine loaded|trial|STATS|FATAL|terminated" $LOG
echo "[$TAG] DONE (full log: $LOG)"
