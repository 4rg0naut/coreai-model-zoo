#!/bin/zsh
# Sustained run, no-console launch (S1). ./_sustain_f.sh <tag> <A[,B]> [duration=600] [idle=180] [P=128] [G=256]
TAG=$1; SEQ=$2; DUR=${3:-600}; IDLE=${4:-180}; P=${5:-128}; G=${6:-256}
DIR=$(cd "$(dirname "$0")" && pwd); LOG=$DIR/_device_sustain_${TAG}.log
ARMS=$(echo $SEQ | tr ',' '\n' | wc -l | tr -d ' '); CAP=$(( (DUR + 600 + 240) * ARMS / 10 + 30 ))
$DIR/_launch_f.sh sustain_${TAG}_$(date +%H%M%S) "SUSTAIN_DONE|FATAL" $CAP "\"AB_TAG\":\"$TAG\",\"AB_SEQUENCE\":\"$SEQ\",\"AB_DURATION_S\":\"$DUR\",\"AB_IDLE_S\":\"$IDLE\",\"AB_P\":\"$P\",\"AB_G\":\"$G\"" $LOG
grep -E "SUSTAIN_START|SUSTAIN_STATS|idle_end|FATAL|ERROR" $LOG | cut -c1-400
