#!/bin/zsh
# S6 device queue (2026-09-17): waits for the iPhone 17 Pro, then runs the whole device lane unattended:
#   1. gate app (embedded lfm25_1_2b_ane_pal8): red fixture must FAIL, clean fixture (4 prompts; first load = cold ANE build)
#   2. bench app: GSM8K 200 (task mode, explicit LFM2.5 chat template), 2-round p128 g256 trials, 60 s sustain (thermal)
#   3. restore PipelinedBench (same bundle id) and run the shipped GPU int8hu bundle there (same-day GPU arm), twice
#   4. release the device hold
# Logs: _s6_queue.log (this), _device_*.log (per run). Re-runnable: START_AT=<step> skips earlier steps.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$DIR/../..
DEVB=lfm25_1_2b_ane_pal8
GPU_DIR=lfm2_5_1_2b_instruct_decode_int8hu_block32_sym
FMT='"AB_PROMPT_FORMAT":"<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"'
START_AT=${START_AT:-1}
cd $DIR
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 0. wait for the phone (USB): cap 12 h, poll 30 s. `device info details` answers from the CoreDevice record even when
# the phone is away (S6: install then dies with error 4016), so require the list State to be exactly "available" AND a
# live container listing to succeed.
PHONE_UDID=00008150-0018713A0207801C
live() {
  # `list devices` flips between "available (paired)" and other states while the phone is fine (10:26 probe: 1 of 3);
  # the live container listing is the reliable signal — the list state is only logged.
  xcrun devicectl list devices 2>/dev/null | grep "$PHONE_UDID" | grep -qE '[[:space:]]available[[:space:]]' || echo "   (list: not 'available' right now — checking the container anyway)"
  local out
  out=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents 2>&1)
  echo "$out" | grep -q ERROR && { echo "   (files: $(echo "$out" | grep -m1 ERROR | cut -c1-120))"; return 1; }
  return 0
}
say "waiting for $UDID (list State = available + live container listing)"
T_END=$(( $(date +%s) + 12*3600 ))
until live; do
  [ $(date +%s) -ge $T_END ] && { say "ERROR device never came back (12 h cap)"; exit 1; }
  sleep 30
done
say "device available"
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | grep -c . | sed 's/^/PipelinedBench Documents\/models entries before: /'
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | grep -q "$GPU_DIR" && say "GPU bundle $GPU_DIR present" || say "WARNING GPU bundle $GPU_DIR NOT in PipelinedBench container"

if [ $START_AT -le 1 ]; then
  say "1. gate app install"
  ./_install.sh "$(cat _app_path.txt)" || { say "ERROR gate install"; exit 2; }
  sleep 10
  say "1a. red fixture (must FAIL)"
  ./_run_f.sh s6red lfm25_1_2b/fixture_red
  grep -E 'GATE_SUMMARY' _device_s6red.log | grep -q 'VERDICT=FAIL' && say "red is RED (ok)" || say "WARNING red did not fail — gate cannot go red"
  say "1b. clean fixture"
  ./_run_f.sh s6clean lfm25_1_2b/fixture
  grep -E 'GATE|TF .* FAIL|FR ' _device_s6clean.log | tail -12
fi

if [ $START_AT -le 2 ]; then
  say "2. bench app install"
  ./_install.sh "$(cat bench/_app_path.txt)" || { say "ERROR bench install"; exit 3; }
  sleep 10
  say "2a. GSM8K 200 on the ANE bundle"
  (cd bench && ./_task_f.sh s6ane $DEVB gsm8k_200 640 "$FMT")
  say "2b. 2-round trials p128 g256 n5"
  (cd bench && ./_abn_f.sh s6ane $DEVB 2 128 256 5 20) | grep -E '^r[0-9]|ABN'
  say "2c. sustain 60 s (idle until fair)"
  (cd bench && ./_sustain_f.sh s6ane $DEVB 60 60)
fi

if [ $START_AT -le 3 ]; then
  say "3. restore PipelinedBench"
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID $ROOT/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
    echo "$OUT" | grep -q installationURL && { say "PipelinedBench restored"; break; }
    say "restore attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error)"; sleep 15
  done
  xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | grep -c . | sed 's/^/PipelinedBench Documents\/models entries after: /'
  say "3a. GPU arm on PipelinedBench (shipped int8hu), r1 + r2"
  sleep 60
  (cd $ROOT && ondevice/_pipelined_device_probe.sh s6gpu_r1 ",\"PB_MODEL\":\"$GPU_DIR\",\"PB_P\":\"128\",\"PB_G\":\"256\",\"PB_N\":\"5\"") | grep -E 'STATS|TRIAL|NUMERICS|ERROR|DONE'
  sleep 30
  (cd $ROOT && ondevice/_pipelined_device_probe.sh s6gpu_r2 ",\"PB_MODEL\":\"$GPU_DIR\",\"PB_P\":\"128\",\"PB_G\":\"256\",\"PB_N\":\"5\"") | grep -E 'STATS|TRIAL|NUMERICS|ERROR|DONE'
fi

rm -f $ROOT/ondevice/.device_hold
say "device released (hold removed)"
