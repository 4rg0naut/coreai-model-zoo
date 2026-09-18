#!/bin/zsh
# S6 diag chain (Mac side, no device): after the proxy-vs-device comparison (which caches the palettized fp16 weights),
# (1) write Mac-twin rollout fixtures for three graph variants (prod, nohist, convzero), (2) export+AOT the two diag
# variants (full 16 layers, ctx 4096, 8-bit), (3) build ONE gate app embedding prod + nohist + convzero bundles.
# On the phone (after the queue releases it): AG_MODEL=<arm> with the matching twin fixture -> the device is judged
# against its own Mac twin under the same graph variant = localizes where the ANE deviates.
cd ~/code/coreai/ondevice/_ane_gate
PY=~/code/coreai/coreai-models-rebase/.venv/bin/python
SD=/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt
YAML=../../coreai-models-community/conversion/lfm25_pal8_g32.yaml
DEVB=~/code/coreai/coreai-models/exports/apple_bench/devbundles
export HF_HUB_OFFLINE=1
say() { echo "[$(date '+%H:%M:%S')] $*"; }
while kill -0 ${WAIT_PID:-0} 2>/dev/null; do sleep 10; done
[ -f $SD ] || { say "ERROR no cached palettized state dict $SD"; exit 1; }
say "1. twin rollout fixtures"
for arm in prod nohist convzero; do
  diag=""; [ $arm != prod ] && diag=$arm
  LFM2_IOS_DIAG=$diag $PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --load-palettized $SD \
    --rollout-fixture fixtures/lfm25_1_2b/twin_$arm.json 2>&1 | grep -E '^\[|wrote|VERDICT|Error|error' | cut -c1-220
done
say "2. diag exports"
for arm in nohist convzero; do
  LFM2_IOS_DIAG=$arm python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_1_2b_ios_diag_$arm --compression-config $YAML \
    --max-context-length 4096 --devbundle lfm25_diag_$arm 2>&1 | grep -E 'export |AOT |devbundle|FAILED' | cut -c1-200
done
say "3. gate app with 3 bundles"
BUNDLE=$DEVB/lfm25_1_2b_ane_pal8 BUNDLE2=$DEVB/lfm25_diag_nohist BUNDLE3=$DEVB/lfm25_diag_convzero DD=/private/tmp/ane_gate_dd/gate_s6diag ./_build.sh 2>&1 | tail -3
cp _app_path.txt _app_path_s6diag.txt
say "chain done"
