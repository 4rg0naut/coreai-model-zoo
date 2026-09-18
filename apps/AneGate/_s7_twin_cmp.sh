#!/bin/zsh
# S7: compare a device gate log with its MATCHED Mac fp16 twin (same layers fp16, rest 8-bit; cache from _s7_twin_chain.sh).
# ./_s7_twin_cmp.sh <tag> <arm>   e.g. ./_s7_twin_cmp.sh candC mlpfp16_L15679_attn_conv  -> _s7_fxm_<tag>.{log,json}
cd ~/code/coreai/ondevice/_ane_gate
TAG=${1:?tag}; ARM=${2:?arm}; PY=$HOME/code/coreai/coreai-models-rebase/.venv/bin/python
$PY s6_lfm2_check.py --fixture fixtures/lfm25_1_2b/fixture.json --dtype float16 --load-palettized /private/tmp/ane_gate_dd/s7_${ARM}_fp16_sd.pt \
   --device-log _device_$TAG.log --out _s7_fxm_$TAG.json > _s7_fxm_$TAG.log 2>&1
echo "[$TAG vs matched twin $ARM]"
grep -E 'device vs this run|^\[(natural|chat|long|sky)\] tf' _s7_fxm_$TAG.log | sed -E 's/ fails=\[.*//'
