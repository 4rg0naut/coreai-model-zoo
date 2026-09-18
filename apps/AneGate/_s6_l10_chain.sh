#!/bin/zsh
# S6 decisive-diagnosis prep (Mac only): a 10-layer truncation of LFM2.5 exported twice — fp16 dense (no LUT) and 8-bit
# k-means — each with a Mac-twin rollout fixture (fp16 module, int8 table, same weights). On the phone, "device vs its
# own twin" for both tells whether the O(1) logit noise comes from the palettized (LUT) execution path.
# Then the three full-size mixed recipes (attention fp16 / conv fp16 / both) are exported as candidate fixes.
cd ~/code/coreai/ondevice/_ane_gate
PY=~/code/coreai/coreai-models-rebase/.venv/bin/python
YAML=../../coreai-models-community/conversion
DEVB=~/code/coreai/coreai-models/exports/apple_bench/devbundles
export HF_HUB_OFFLINE=1
say() { echo "[$(date '+%H:%M:%S')] $*"; }
say "1. L10 exports"
python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_L10_fp16 --compression none --num-layers 10 --max-context-length 512 --devbundle lfm25_L10_fp16 2>&1 | grep -E 'export |AOT |devbundle|FAILED' | cut -c1-200
python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_L10_pal8 --compression-config $YAML/lfm25_pal8_g32.yaml --num-layers 10 --max-context-length 512 --devbundle lfm25_L10_pal8 2>&1 | grep -E 'export |AOT |devbundle|FAILED' | cut -c1-200
say "2. L10 twin fixtures"
$PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --num-layers 10 --max-ctx 512 --rollout-fixture fixtures/lfm25_1_2b/twin_L10_fp16.json 2>&1 | grep -E '^\[|wrote|Error' | cut -c1-200
$PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --num-layers 10 --max-ctx 512 --palettize $YAML/lfm25_pal8_g32.yaml --save-palettized /private/tmp/ane_gate_dd/s6_L10_pal8_sd.pt --rollout-fixture fixtures/lfm25_1_2b/twin_L10_pal8.json 2>&1 | grep -E '^\[|wrote|palettized|Error' | cut -c1-200
say "3. L10 gate app (2 bundles)"
BUNDLE=$DEVB/lfm25_L10_fp16 BUNDLE2=$DEVB/lfm25_L10_pal8 DD=/private/tmp/ane_gate_dd/gate_s6l10 ./_build.sh 2>&1 | tail -2
cp _app_path.txt _app_path_s6l10.txt
say "4. full-size mixed recipes"
for arm in attnfp16 convfp16 attn_conv_fp16; do
  python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_1_2b_ios_pal8_g32_$arm --compression-config $YAML/lfm25_pal8_g32_$arm.yaml --max-context-length 4096 --devbundle lfm25_1_2b_ane_pal8_$arm 2>&1 | grep -E 'export |AOT |devbundle|FAILED' | cut -c1-200
done
say "5. mixed gate app (3 bundles)"
BUNDLE=$DEVB/lfm25_1_2b_ane_pal8_attnfp16 BUNDLE2=$DEVB/lfm25_1_2b_ane_pal8_convfp16 BUNDLE3=$DEVB/lfm25_1_2b_ane_pal8_attn_conv_fp16 DD=/private/tmp/ane_gate_dd/gate_s6mixed ./_build.sh 2>&1 | tail -2
cp _app_path.txt _app_path_s6mixed.txt
say "chain done"
