#!/bin/zsh
# after the L10/mixed chain: export the two MLP-selective candidates and build a 2-bundle gate app
cd ~/code/coreai/ondevice/_ane_gate
YAML=../../coreai-models-community/conversion; DEVB=~/code/coreai/coreai-models/exports/apple_bench/devbundles
say() { echo "[$(date '+%H:%M:%S')] $*"; }
while kill -0 ${WAIT_PID:-0} 2>/dev/null; do sleep 10; done
for arm in mlpfp16_L17 mlpfp16_L15679_attn_conv; do
  say "export $arm"
  python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_1_2b_ios_pal8_g32_$arm --compression-config $YAML/lfm25_pal8_g32_$arm.yaml --max-context-length 4096 --devbundle lfm25_1_2b_ane_pal8_$arm 2>&1 | grep -E 'export |AOT |devbundle|FAILED' | cut -c1-200
done
say "cand gate app (2 bundles)"
BUNDLE=$DEVB/lfm25_1_2b_ane_pal8_mlpfp16_L17 BUNDLE2=$DEVB/lfm25_1_2b_ane_pal8_mlpfp16_L15679_attn_conv DD=/private/tmp/ane_gate_dd/gate_s6cand ./_build.sh 2>&1 | tail -2
cp _app_path.txt _app_path_s6cand.txt
say "cand chain done"
