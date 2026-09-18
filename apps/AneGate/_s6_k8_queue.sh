#!/bin/zsh
# S6 fixed-bundle queue: wait for the k8 export (pid in EXPORT_PID), build the gate app (k8 only) and the bench app (k8),
# wait until the main queue releases the device (hold file gone), take the hold, then gate red/clean -> GSM8K 200 ->
# trials -> sustain -> restore PipelinedBench -> release.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$DIR/../..
DEVB=lfm25_1_2b_ane_pal8_k8
FMT='"AB_PROMPT_FORMAT":"<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"'
cd $DIR
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while kill -0 ${EXPORT_PID:-0} 2>/dev/null; do sleep 10; done
grep -q 'ANE_regions=31' $ROOT/coreai-models/exports/lfm25_1_2b_ios_pal8_g32_k8/_export.log || { say "ERROR k8 export/AOT not 31 regions"; tail -3 $ROOT/coreai-models/exports/lfm25_1_2b_ios_pal8_g32_k8/_export.log; exit 1; }
say "k8 bundle ready; building gate app"
BUNDLE=$ROOT/coreai-models/exports/apple_bench/devbundles/$DEVB DD=/private/tmp/ane_gate_dd/gate_s6k8 ./_build.sh 2>&1 | tail -2
cp _app_path.txt _app_path_s6k8.txt
say "building bench app"
(cd bench && BUNDLE=$ROOT/coreai-models/exports/apple_bench/devbundles/$DEVB DD=/private/tmp/ane_gate_dd/bench_s6k8 ./_build.sh 2>&1 | tail -2; cp _app_path.txt _app_path_s6k8.txt)
say "waiting for the device hold to be released by the main queue"
while [ -f $ROOT/ondevice/.device_hold ]; do sleep 20; done
echo "S6 LFM2.5 ANE port k8 (gate + GSM8K + A/B) since $(date '+%m-%d %H:%M'), session ane-s6" > $ROOT/ondevice/.device_hold
say "hold taken; 1. gate app install"
./_install.sh "$(cat _app_path_s6k8.txt)" || { say "ERROR gate install"; exit 2; }
sleep 10
say "1a. red"; ./_run_f.sh s6k8red lfm25_1_2b/fixture_red
grep -E 'GATE_SUMMARY' _device_s6k8red.log | grep -q 'VERDICT=FAIL' && say "red is RED (ok)" || say "WARNING red did not fail"
say "1b. clean"; ./_run_f.sh s6k8clean lfm25_1_2b/fixture
grep -E 'GATE|TF .* FAIL' _device_s6k8clean.log | tail -12
say "2. bench app install"
./_install.sh "$(cat bench/_app_path_s6k8.txt)" || { say "ERROR bench install"; exit 3; }
sleep 10
say "2a. GSM8K 200"; (cd bench && ./_task_f.sh s6k8 $DEVB gsm8k_200 640 "$FMT")
say "2b. trials"; (cd bench && ./_abn_f.sh s6k8 $DEVB 2 128 256 5 20) | grep -E '^r[0-9]|ABN'
say "2c. sustain 60 s"; (cd bench && ./_sustain_f.sh s6k8 $DEVB 60 60)
say "3. restore PipelinedBench"
for attempt in 1 2 3; do
  OUT=$(xcrun devicectl device install app --device $UDID $ROOT/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
  echo "$OUT" | grep -q installationURL && { say "PipelinedBench restored"; break; }
  say "restore attempt $attempt failed"; sleep 15
done
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | grep -c . | sed 's/^/PipelinedBench models entries: /'
rm -f $ROOT/ondevice/.device_hold
say "device released (hold removed)"
