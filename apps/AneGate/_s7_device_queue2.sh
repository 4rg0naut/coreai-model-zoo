#!/bin/zsh
# S7 device queue, part 2 (after candA finished at 18:26 and candC's first cold build was lost at 28 min):
# mixed app (attn / conv / attn+conv fp16) with a 90-min poll cap per gate, then candC again with the 90-min cap.
# Every device log is compared with BOTH Mac twins in the background: the all-8-bit twin (baseline metric,
# _s7_fx_<tag>) and the matched twin with the same fp16 layers (_s7_fxm_<tag>). Hold is KEPT at the end.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$DIR/../..
PY=$HOME/code/coreai/coreai-models-rebase/.venv/bin/python
SD=/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt
export AG_CAP=540
cd $DIR
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
twin() {  # <tag> <arm>: all-8-bit twin and matched twin vs the device log, sequential, in the background
  local tag=$1 arm=$2
  $PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --load-palettized $SD \
      --device-log _device_$tag.log --out _s7_fx_$tag.json > _s7_fx_$tag.log 2>&1
  say "8-bit twin vs $tag: $(grep -E 'device vs this run' _s7_fx_$tag.log | sed -E 's/.*jointly-correct steps: //' | tr '\n' ';')"
  ./_s7_twin_cmp.sh $tag $arm > /dev/null 2>&1
  say "matched twin vs $tag: $(grep -E 'device vs this run' _s7_fxm_$tag.log | sed -E 's/.*jointly-correct steps: //' | tr '\n' ';')"
}
gate() {  # <tag> <embedded bundle dir> <arm>
  local tag=$1 bundle=$2 arm=$3
  say "gate $tag ($bundle, cap $((AG_CAP/6)) min)"
  ./_run_f.sh $tag lfm25_1_2b/fixture $bundle
  grep -E '^GATE |engine loaded' _device_$tag.log | sed 's/^/   /'
  twin $tag $arm &
}
twin candA mlpfp16_L17 &
say "2. mixed gate app install (attn / conv / attn+conv, 4.4 GB)"
./_install.sh "$(cat _app_path_s6mixed.txt)" || { say "ERROR mixed install"; exit 3; }
sleep 10
gate mixA  lfm25_1_2b_ane_pal8_attnfp16       attnfp16
gate mixC  lfm25_1_2b_ane_pal8_convfp16       convfp16
gate mixAC lfm25_1_2b_ane_pal8_attn_conv_fp16 attn_conv_fp16
say "3. cand gate app install again (C + A, 3.1 GB) for candC with the 90-min cap"
./_install.sh "$(cat _app_path_s6cand.txt)" || { say "ERROR cand install"; exit 2; }
sleep 10
gate candC lfm25_1_2b_ane_pal8_mlpfp16_L15679_attn_conv mlpfp16_L15679_attn_conv
say "device gates done; waiting for the twin comparisons"
wait
say "S7 queue2 done (hold KEPT: $(cat $ROOT/ondevice/.device_hold))"
