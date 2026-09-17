#!/bin/zsh
# Record the ANE demo from the Mac (S1, 2026-09-16): QuickTime mirrors the USB iPhone, the bench app runs
# AB_DEMO=1 (one real question answered by the Neural Engine bundle, then the GPU bundle, text streamed with a
# live tok/s), the recording is exported to ~/Desktop as .mov + 1x H.264 MP4.
#   ./_record_demo.sh [test]     # "test" = launch the demo once with the console attached, no recording
#   EXTRA_ENV=',"AB_WARMUP":"1"' ./_record_demo.sh   # hidden warmup before the visible run (warm numbers, as in the bench)
# Prereqs: demo app installed under com.coreai.pipelinedbench (bench/ build with AB_DEMO support, DD bench);
# QuickTime's last capture source is the iPhone (first time: pick it in the recording window by hand).
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; BID=com.coreai.pipelinedbench
DIR=$(cd "$(dirname "$0")" && pwd); TS=$(date +%Y%m%d_%H%M%S)
ENVJ='{"AB_DEMO":"1","AB_SEQUENCE":"minicpm5_2b_ane,minicpm5_2b_gpu_int8","AB_LABEL_minicpm5_2b_ane":"MiniCPM5-2B · Neural Engine","AB_LABEL_minicpm5_2b_gpu_int8":"MiniCPM5-2B · GPU","AB_QUESTION":"Explain in a short paragraph why the sky is blue.","AB_MAX_TOKENS":"200"'${EXTRA_ENV}'}'
if [ "$1" = "test" ]; then
  LOG=$DIR/_device_demo_test.log; : > $LOG
  xcrun devicectl device process launch --device $UDID --console --terminate-existing --environment-variables "$ENVJ" $BID > $LOG 2>&1 &
  CPID=$!; for i in {1..150}; do grep -qE "DEMO_DONE|FATAL|ERROR" $LOG && break; sleep 2; done; sleep 2; kill $CPID 2>/dev/null
  grep -E "DEMO|FATAL|ERROR" $LOG | cut -c1-300; exit 0
fi
RAW=$HOME/Desktop/minicpm5-ane-demo-$TS.mov; MP4=$HOME/Desktop/minicpm5-ane-demo-$TS.mp4; LOG=$DIR/_device_demo_$TS.log
osascript <<'AS'
tell application "QuickTime Player"
    activate
    new movie recording
end tell
delay 3
tell application "System Events" to tell process "QuickTime Player"
    set s to size of window "Movie Recording"
    if (item 2 of s) < (item 1 of s) then error "QuickTime source is not the iPhone (landscape preview)"
end tell
AS
osascript -e 'tell application "QuickTime Player" to start document "Movie Recording"'
echo "recording… $(date +%H:%M:%S)"
: > $LOG
xcrun devicectl device process launch --device $UDID --console --terminate-existing --environment-variables "$ENVJ" $BID > $LOG 2>&1 &
LP=$!
for i in {1..150}; do grep -qE "DEMO_DONE|FATAL|failed to launch|ERROR" $LOG && break; sleep 2; done
sleep 4
osascript -e 'tell application "QuickTime Player" to stop document "Movie Recording"'
kill $LP 2>/dev/null || true
sleep 2
osascript <<AS
tell application "QuickTime Player"
    set d to document 1
    export d in POSIX file "$RAW" using settings preset "1080p"
    delay 2
    close d saving no
end tell
AS
for i in {1..90}; do [[ -s "$RAW" ]] && break; sleep 2; done
grep -E "DEMO|FATAL|ERROR" $LOG | cut -c1-200
echo "raw: $RAW ($(du -h "$RAW" | cut -f1))"
ffmpeg -v error -y -i "$RAW" -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$MP4"
echo "mp4: $MP4 ($(du -h "$MP4" | cut -f1)) $(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$MP4" | cut -c1-6)s"
