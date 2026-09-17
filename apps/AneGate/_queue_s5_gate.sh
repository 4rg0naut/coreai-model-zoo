#!/bin/zsh
# S5 gate queue (2026-09-16): one gate app embeds the three MiniCPM5-2B ANE arms (AG_MODEL picks one per launch).
# Order = most likely to work first, the ~2.2 GB boundary probe last (a hang there ends the queue; its cap is 30 min).
# The RED fixture goes first on the first arm (pays that arm's cold ANE program build), then the clean fixture.
#   ./_queue_s5_gate.sh [app path]        (log: _queue_s5_gate.log)
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR
APP=${1:-$(cat _app_path.txt)}; OUT=$DIR/_queue_s5_gate.log
echo "S5 gate queue start $(date) app=$APP" | tee -a $OUT
./_install.sh "$APP" 2>&1 | tee -a $OUT; [ ${pipestatus[1]} -eq 0 ] || { echo "install failed" | tee -a $OUT; exit 2; }
sleep 10
# arms: mixed48g32 (4-bit g32 + 9 layers 8-bit, 1.6 GB) first, pal6g8 (6-bit g8, 2.5 GB, the boundary) last.
# (pal4g8 and the g8-base mixed compile to 0 ANE regions — not ANE arms, not gated.)
for job in "s5_red_mixed48g32 minicpm5_2b/fixture_red minicpm5_2b_ane_mixed48g32" \
           "s5_clean_mixed48g32 minicpm5_2b/fixture minicpm5_2b_ane_mixed48g32" \
           "s5_clean_pal6g8 minicpm5_2b/fixture minicpm5_2b_ane_pal6g8"; do
  set -- ${=job}
  echo "== $1 ($3) start $(date +%H:%M:%S)" | tee -a $OUT
  ./_run_f.sh $1 $2 $3 2>&1 | tee -a $OUT
  L=_device_$1.log
  echo "   $(grep -E 'engine loaded' $L | head -1) | $(grep -E '^GATE_SUMMARY|FATAL|ERROR:' $L | head -1 | cut -c1-200)" | tee -a $OUT
  grep -E "^GATE prompt=" $L | sed 's/^/   /' | tee -a $OUT
  if grep -qE "ERROR: cap reached" $L; then echo "   cap reached on $3 — stopping the queue (do not terminate a loading app)" | tee -a $OUT; break; fi
  sleep 15
done
echo "S5 gate queue end $(date)" | tee -a $OUT
