#!/bin/zsh
# S6: GSM8K 200 on the Mac for LFM2.5-1.2B — HF bf16 reference, then the ANE recipe simulated
# (8-bit g32 k-means on every Linear except the tied head; embedding table int8 per-tensor as the iOS graph has it).
cd ~/code/coreai/ondevice/_ane_gate/bench
PY=~/code/coreai/coreai-models-rebase/.venv/bin/python
HF=LiquidAI/LFM2.5-1.2B-Instruct
export HF_HUB_OFFLINE=1
echo "=== bf16 reference $(date +%H:%M:%S)"
$PY ref_gsm8k_hf.py --hf-id $HF --task tasks/gsm8k_200.jsonl --out _s6_ref_hf_bf16_mps_answers.log 2>&1 | grep -v 'Loading weights\|Redirects\|Skipping' | tail -3
echo "=== pal8 g32 + int8 embedding simulation $(date +%H:%M:%S)"
$PY ref_gsm8k_hf.py --hf-id $HF --task tasks/gsm8k_200.jsonl --out _s6_sim_pal8g32_emb8_bf16_mps_answers.log \
    --recipe ../../../coreai-models-community/conversion/lfm25_pal8_g32.yaml --embed-int8-per-tensor 2>&1 | grep -v 'Loading weights\|Redirects\|Skipping' | tail -3
echo "=== scoring $(date +%H:%M:%S)"
$PY score_gsm8k.py --arm hf_bf16=_s6_ref_hf_bf16_mps_answers.log --arm sim_pal8g32_emb8=_s6_sim_pal8g32_emb8_bf16_mps_answers.log --json _s6_gsm8k_scores.json
echo "=== done $(date +%H:%M:%S)"
