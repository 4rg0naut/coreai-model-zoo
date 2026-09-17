#!/bin/zsh
# S5 resume after the phone left mid-queue (2026-09-16 14:36): wait for the phone, pull the 6-bit answers, finish the
# 6-bit remainder, run the int8 arm, restore PipelinedBench, score everything.   ./_resume_s5_task.sh   (log: _queue_s5_task.log)
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; ID=com.coreai.pipelinedbench
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR; OUT=$DIR/_queue_s5_task.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
say "S5 resume: waiting for the phone"
for i in $(seq 1 1440); do xcrun devicectl list devices 2>/dev/null | grep -E "iPhone 17 Pro.*physical" | grep -q "available" && break; sleep 60; done
xcrun devicectl list devices 2>/dev/null | grep -E "iPhone 17 Pro.*physical" | grep -q "available" || { say "phone never came back (24 h)"; exit 2; }
say "phone is back"; sleep 20
# 1. pull what the 6-bit run wrote on the phone
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $ID --source Documents/sustain/task_6bit_133936_answers.log --destination _device_task_6bit_answers.log >/dev/null 2>&1
N6=$(grep -c '^{' _device_task_6bit_answers.log 2>/dev/null); say "6-bit answers on the phone: $N6"
if [ "${N6:-0}" -lt 200 ]; then
  ./_task_f.sh 6bit_rest minicpm5_2b_ane_pal6g8 gsm8k_200 640 "\"AB_TASK_START\":\"$N6\"" 2>&1 | tee -a $OUT
  cat _device_task_6bit_rest_answers.log >> _device_task_6bit_answers.log 2>/dev/null
  say "6-bit answers after the remainder: $(grep -c '^{' _device_task_6bit_answers.log)"
fi
sleep 20
# 2. the GPU int8 arm (S=1 prefill so the dynamic bundle keeps one prefill shape)
./_task_f.sh int8 minicpm5_2b_gpu_int8 gsm8k_200 640 "\"COREAI_CHUNK_THRESHOLD\":\"1\"" 2>&1 | tee -a $OUT
# 3. PipelinedBench back (Thursday 05:30 dashboard job)
say "restoring PipelinedBench"
for t in 1 2 3; do
  R=$(xcrun devicectl device install app --device $UDID ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
  echo "$R" | grep -q installationURL && { echo "$R" | grep installationURL | tee -a $OUT; break; }
  say "install attempt $t failed: $(echo "$R" | grep -m1 -i error | cut -c1-120)"; sleep 30
done
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $ID --subdirectory Documents/models 2>&1 | head -1 | tee -a $OUT
# 4. score
python3 score_gsm8k.py --arm 4bit_ane=_device_task_4bit_answers.log --arm 6bit_ane=_device_task_6bit_answers.log --arm int8_gpu=_device_task_int8_answers.log \
  --arm hf_bf16=_ref_hf_bf16_mps_answers.log --arm sim6bit_mac=_sim6bit_bf16_mps_answers.log --json _gsm8k_s5_scores.json 2>&1 | tee -a $OUT
say "S5 resume end"
