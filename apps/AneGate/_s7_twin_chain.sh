#!/bin/zsh
# S7: build the MATCHED Mac fp16 twin of each mixed-fp16 candidate (same layers kept fp16, the rest 8-bit k-means g32),
# cache its state_dict, and judge it against the fp32 fixture (must stay 125/125). 5 arms in parallel on the Mac.
cd ~/code/coreai/ondevice/_ane_gate
PY=$HOME/code/coreai/coreai-models-rebase/.venv/bin/python; YAML=../../coreai-models-community/conversion
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for arm in mlpfp16_L15679_attn_conv mlpfp16_L17 attnfp16 convfp16 attn_conv_fp16; do
  say "twin $arm"
  $PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --palettize $YAML/lfm25_pal8_g32_$arm.yaml \
     --save-palettized /private/tmp/ane_gate_dd/s7_${arm}_fp16_sd.pt --out _s7_twin_$arm.json > _s7_twin_$arm.log 2>&1 &
done
wait
for arm in mlpfp16_L15679_attn_conv mlpfp16_L17 attnfp16 convfp16 attn_conv_fp16; do
  say "$arm: $(grep -E 'palettized [0-9]+ Conv2d' _s7_twin_$arm.log) | $(grep -E '^\[(natural|chat|long|sky)\] tf' _s7_twin_$arm.log | sed -E 's/ \([0-9]+ s\) fails=.*//' | tr '\n' ' ') $(grep VERDICT _s7_twin_$arm.log)"
done
