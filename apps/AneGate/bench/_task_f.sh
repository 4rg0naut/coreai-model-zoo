#!/bin/zsh
# Task-accuracy run, no-console launch (S5). ./_task_f.sh <tag> <arm> [task=gsm8k_200] [max_tokens=640] [extra-env-json]
# Polls Documents/sustain/<tag>.log for TASK_DONE (cap 100 min), then pulls <tag>_answers.log next to it.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; APP_ID=${APP_ID:-com.coreai.pipelinedbench}
TAG=$1; ARM=$2; TASK=${3:-gsm8k_200}; MAXT=${4:-640}; EXTRA=$5
DIR=$(cd "$(dirname "$0")" && pwd); LOG=$DIR/_device_task_${TAG}.log
T=task_${TAG}_$(date +%H%M%S)
$DIR/_launch_f.sh $T "TASK_DONE|FATAL" 600 "\"AB_TASK\":\"$TASK\",\"AB_MODEL\":\"$ARM\",\"AB_MAX_TOKENS\":\"$MAXT\"${EXTRA:+,$EXTRA}" $LOG
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --source Documents/sustain/${T}_answers.log --destination $DIR/_device_task_${TAG}_answers.log >/dev/null 2>&1
echo "[$TAG] $(grep -E 'TASK_START|TASK_DONE|FATAL|ERROR' $LOG | tail -2 | cut -c1-200) | answers: $(grep -c '^{' $DIR/_device_task_${TAG}_answers.log 2>/dev/null)"
