#!/bin/zsh
# per-layer sensitivity to int8-coarse MLP activations (Mac twin), 4 at a time
cd ~/code/coreai/ondevice/_ane_gate
SD=/private/tmp/ane_gate_dd/s6_pal8_fp16_sd.pt; PY=~/code/coreai/coreai-models-rebase/.venv/bin/python; export HF_HUB_OFFLINE=1
for batch in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15"; do
  for L in ${=batch}; do
    ANE_EMU_SUBSET=mlp ANE_EMU_LAYERS=$L $PY _s6_orig_builder.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --load-palettized $SD --ane-emu a8:tensor --device-log _device_s6clean.log --out _s6_fx_a8mlp_L$L.json > _s6_fx_a8mlp_L$L.log 2>&1 &
  done
  wait
done
echo "layer scan done $(date +%H:%M:%S)"
