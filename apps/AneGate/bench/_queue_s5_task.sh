#!/bin/zsh
# S5 GSM8K task queue: install the task app, run the three arms, restore PipelinedBench.
#   ./_queue_s5_task.sh <app path> [n=200]     (log: _queue_s5_task.log)
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR
APP=${1:?app path}; N=${2:-200}; OUT=$DIR/_queue_s5_task.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
say "S5 task queue start app=$APP n=$N"
../_install.sh "$APP" 2>&1 | tee -a $OUT; [ ${pipestatus[1]} -eq 0 ] || { say "install failed"; exit 2; }
sleep 10
# ANE arms first (their program caches are warm in this container); the GPU int8 bundle last with the S=1 prefill
# (COREAI_CHUNK_THRESHOLD=1) so the dynamic bundle does not re-specialize its prefill for every prompt length.
./_task_f.sh 4bit minicpm5_2b_ane gsm8k_200 640 "\"AB_TASK_N\":\"$N\"" 2>&1 | tee -a $OUT
sleep 20
./_task_f.sh 6bit minicpm5_2b_ane_pal6g8 gsm8k_200 640 "\"AB_TASK_N\":\"$N\"" 2>&1 | tee -a $OUT
sleep 20
./_task_f.sh int8 minicpm5_2b_gpu_int8 gsm8k_200 640 "\"AB_TASK_N\":\"$N\",\"COREAI_CHUNK_THRESHOLD\":\"1\"" 2>&1 | tee -a $OUT
say "restoring PipelinedBench"
xcrun devicectl device install app --device $UDID ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1 | grep -E "installationURL|rror" | tee -a $OUT
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>&1 | head -1 | tee -a $OUT
say "S5 task queue end"
