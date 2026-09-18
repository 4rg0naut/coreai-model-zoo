#!/bin/zsh
# S7 device queue (2026-09-18): gate the five mixed-fp16 LFM2.5-1.2B ANE bundles on the iPhone 17 Pro one after
# another (two gate apps: cand = C + A, mixed = attn / conv / attn+conv), and compare every device log with the
# Mac fp16 twin in the background (the metric is |gap diff| on jointly-correct TF steps, all-8-bit baseline
# natural 1.44 / chat 1.50 / sky 0.98). Takes the device hold first and KEEPS it (trials/sustain may follow).
# Logs: _s7_queue.log (this), _device_<tag>.log (gate), _s7_fx_<tag>.{log,json} (twin comparison).
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
PHONE_UDID=00008150-0018713A0207801C
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$DIR/../..
PY=$HOME/code/coreai/coreai-models-rebase/.venv/bin/python
SD=/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt
START_AT=${START_AT:-1}
cd $DIR
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
live() {
  xcrun devicectl list devices 2>/dev/null | grep "$PHONE_UDID" | grep -qE '[[:space:]]available[[:space:]]' || echo "   (list: not 'available' right now — checking the container anyway)"
  local out
  out=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents 2>&1)
  echo "$out" | grep -q ERROR && { echo "   (files: $(echo "$out" | grep -m1 ERROR | cut -c1-120))"; return 1; }
  return 0
}
twin() {  # Mac fp16 twin (all-8-bit k-means weights, cached) vs the device log of <tag>; runs in the background
  local tag=$1
  $PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --load-palettized $SD \
      --device-log _device_$tag.log --out _s7_fx_$tag.json > _s7_fx_$tag.log 2>&1
  say "twin vs $tag: $(grep -E '^\[(natural|chat|long|sky)\] tf' _s7_fx_$tag.log | tr '\n' ' ')"
  grep -B1 -E 'device vs this run' _s7_fx_$tag.log | grep -E 'device vs' | sed "s/^/   [$tag] /"
}
gate() {  # gate <tag> <embedded bundle dir>
  local tag=$1 bundle=$2
  say "gate $tag ($bundle)"
  ./_run_f.sh $tag lfm25_1_2b/fixture $bundle
  grep -E '^GATE ' _device_$tag.log | sed 's/^/   /'
  grep -E 'engine loaded' _device_$tag.log | sed 's/^/   /'
  twin $tag &
}

live || { say "ERROR device not live"; exit 1; }
if [ ! -f $ROOT/ondevice/.device_hold ]; then
  echo "S7 LFM2.5 ANE mixed-fp16 gates (5 bundles, ~25 min) since $(date '+%m-%d %H:%M'), session ane-s7" > $ROOT/ondevice/.device_hold
  say "hold taken"
else
  say "hold already present: $(cat $ROOT/ondevice/.device_hold)"
fi

if [ $START_AT -le 1 ]; then
  say "1. cand gate app install (C + A, 3.1 GB)"
  ./_install.sh "$(cat _app_path_s6cand.txt)" || { say "ERROR cand install"; exit 2; }
  sleep 10
  gate candC lfm25_1_2b_ane_pal8_mlpfp16_L15679_attn_conv
  gate candA lfm25_1_2b_ane_pal8_mlpfp16_L17
fi
if [ $START_AT -le 2 ]; then
  say "2. mixed gate app install (attn / conv / attn+conv, 4.4 GB)"
  ./_install.sh "$(cat _app_path_s6mixed.txt)" || { say "ERROR mixed install"; exit 3; }
  sleep 10
  gate mixA lfm25_1_2b_ane_pal8_attnfp16
  gate mixC lfm25_1_2b_ane_pal8_convfp16
  gate mixAC lfm25_1_2b_ane_pal8_attn_conv_fp16
fi
say "device gates done; waiting for the twin comparisons"
wait
say "S7 queue done (hold KEPT: $(cat $ROOT/ondevice/.device_hold))"
