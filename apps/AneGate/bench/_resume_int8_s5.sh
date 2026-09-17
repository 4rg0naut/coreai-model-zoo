#!/bin/zsh
# S5: the last device arm (GPU int8, 200 GSM8K questions), then PipelinedBench back and the final scores/record.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; ID=com.coreai.pipelinedbench
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR; OUT=$DIR/_queue_s5_task.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
say "S5 int8 arm: install task app"
../_install.sh /private/tmp/ane_gate_dd/bench_task/Build/Products/Release-iphoneos/AppleBenchRunnerGA.app 2>&1 | tee -a $OUT; [ ${pipestatus[1]} -eq 0 ] || { say "install failed"; exit 2; }
sleep 10
./_task_f.sh int8 minicpm5_2b_gpu_int8 gsm8k_200 640 "\"COREAI_CHUNK_THRESHOLD\":\"1\"" 2>&1 | tee -a $OUT
say "restoring PipelinedBench"
for t in 1 2 3; do
  R=$(xcrun devicectl device install app --device $UDID ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
  echo "$R" | grep -q installationURL && { echo "$R" | grep installationURL | tee -a $OUT; break; }
  say "install attempt $t failed: $(echo "$R" | grep -m1 -i error | cut -c1-120)"; sleep 30
done
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $ID --subdirectory Documents/models 2>&1 | head -1 | tee -a $OUT
python3 score_gsm8k.py --arm 4bit_ane=_device_task_4bit_answers.log --arm 6bit_ane=_device_task_6bit_answers.log --arm int8_gpu=_device_task_int8_answers.log \
  --arm hf_bf16=_ref_hf_bf16_mps_answers.log --arm sim6bit_mac=_sim6bit_bf16_mps_answers.log --json _gsm8k_s5_scores.json 2>&1 | tee -a $OUT
say "S5 int8 arm end — phone free"
