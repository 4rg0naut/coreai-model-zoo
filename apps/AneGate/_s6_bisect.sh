#!/bin/zsh
# S6 ANE lowering bisect: one variable per arm (3 layers, ctx 256, 8-bit g32), prints ANE regions per arm.
cd ~/code/coreai/ondevice/_ane_gate
YAML=../../coreai-models-community/conversion/lfm25_pal8_g32.yaml
for arm in "$@"; do
  name=${arm//,/_}; [ -z "$name" ] && name=base
  echo "=== arm '$arm' $(date +%H:%M:%S)"
  LFM2_IOS_DIAG="$arm" python3 _ane_export_s6.py --hf-id LiquidAI/LFM2.5-1.2B-Instruct --out lfm25_probe_L3_$name \
    --compression-config $YAML --num-layers 3 --max-context-length 256 2>&1 | grep -v 'Loading weights\|Redirects\|Skipping import' | tail -6
done
echo "=== bisect done $(date +%H:%M:%S)"
