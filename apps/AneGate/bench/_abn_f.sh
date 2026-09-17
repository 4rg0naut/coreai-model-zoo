#!/bin/zsh
# N-arm same-day interleaved A/B, no-console launches (S1). ./_abn_f.sh <tag> <A,B[,C]> [rounds=2] [P=128] [G=256] [N=5] [gap=20]
TAG=$1; ARMS=(${(s:,:)2}); ROUNDS=${3:-2}; P=${4:-128}; G=${5:-256}; N=${6:-5}; GAP=${7:-20}
DIR=$(cd "$(dirname "$0")" && pwd); OUT=$DIR/_abn_${TAG}.log; : > $OUT
echo "ABN $TAG arms=${(j:,:)ARMS} rounds=$ROUNDS p=$P g=$G n=$N gap=$GAP start $(date)" | tee -a $OUT
for r in $(seq 1 $ROUNDS); do for M in $ARMS; do
  LOG=$DIR/_device_abn_${TAG}_r${r}_${M}.log; T=abn_${TAG}_r${r}_${M}_$(date +%H%M%S)
  $DIR/_launch_f.sh $T "STATS|FATAL" 150 "\"AB_MODEL\":\"$M\",\"AB_P\":\"$P\",\"AB_G\":\"$G\",\"AB_N\":\"$N\"" $LOG
  echo "r$r $M: $(grep -E 'engine loaded' $LOG | head -1) | $(grep -E 'STATS|FATAL|ERROR' $LOG | head -1)" | tee -a $OUT
  grep -E "^trial" $LOG | sed "s/^/    r$r $M /" | tee -a $OUT
  sleep $GAP
done; done
echo "ABN $TAG end $(date)" | tee -a $OUT
