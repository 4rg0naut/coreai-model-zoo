#!/bin/zsh
# Gate run, no-console launch (S1). ./_run_f.sh <tag> <fixture-stem> [embedded bundle dir name = AG_MODEL (S5)]
# Poll cap 30 min: a cold ANE program build after an OS update can pass 15 min (S1 §4).
TAG=${1:-run}; FIX=${2:?fixture stem}; M=$3; DIR=$(cd "$(dirname "$0")" && pwd); LOG=$DIR/_device_${TAG}.log
ENVJ="\"AG_FIXTURE\":\"$FIX\""; [ -n "$M" ] && ENVJ="$ENVJ,\"AG_MODEL\":\"$M\""
$DIR/bench/_launch_f.sh gate_${TAG}_$(date +%H%M%S) "GATE_SUMMARY|FATAL" ${AG_CAP:-180} "$ENVJ" $LOG   # AG_CAP = poll iterations (10 s); S7: mixed-fp16 bundles build their ANE programs for >30 min
echo "[$TAG] $(grep -E 'GATE_SUMMARY|FATAL|ERROR' $LOG | head -2)"
