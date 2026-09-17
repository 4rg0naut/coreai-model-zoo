#!/bin/zsh
# No-console launch + file pull (S1). ./_launch_f.sh <tag> <done-regex> <cap-iterations(10 s each)> <env-json-without-braces> <local-log>
# The app appends to Documents/sustain/<tag>.log (AB_LOG / AG_LOG); we pull it every 10 s until <done-regex> matches.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
APP_ID=${APP_ID:-com.coreai.pipelinedbench}   # S5: another id = another app container (cold-cache tests)
TAG=$1; DONE=$2; CAP=$3; ENVJ=$4; LOCAL=$5
: > $LOCAL
for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
  OUT=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "{\"AB_LOG\":\"$TAG\",\"AG_LOG\":\"$TAG\",$ENVJ}" $APP_ID 2>&1)
  echo "$OUT" | grep -q "Launched application" && break
  echo "launch retry $t: $(echo "$OUT" | grep -m1 -E 'ERROR|Busy' | cut -c1-100)" >> $LOCAL.launch; sleep 15
done
echo "$OUT" | grep -q "Launched application" || { echo "ERROR: launch never accepted ($TAG)" | tee -a $LOCAL; exit 1; }
for i in $(seq 1 $CAP); do
  sleep 10
  rm -f $LOCAL.pull; xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --source Documents/sustain/$TAG.log --destination $LOCAL.pull >/dev/null 2>&1
  [ -f $LOCAL.pull ] && cp $LOCAL.pull $LOCAL
  grep -qE "$DONE" $LOCAL && break
done
grep -qE "$DONE" $LOCAL || echo "ERROR: cap reached without $DONE ($TAG)" >> $LOCAL
