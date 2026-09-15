#!/bin/zsh
# Record the demo from the Mac: QuickTime mirrors the USB-connected iPhone, the app runs its
# presets by itself (AGENT_SELFTEST=1 AGENT_RECORD=1 leaves pauses for the human parts), the
# recording is exported to ~/Desktop and a 3x-speed MP4 is cut for X.
#   ./record-demo.sh            # phone in airplane mode first; QuickTime's last source must be the iPhone
set -e
cd "$(dirname "$0")"
UDID=${UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}
BID=com.daisukemajima.llmbench014
TS=$(date +%Y%m%d_%H%M%S)
RAW=$HOME/Desktop/coreaiagent-demo-$TS.mov
FAST=$HOME/Desktop/coreaiagent-demo-$TS-3x.mp4
LOG=_record_$TS.log

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
xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
  --environment-variables '{"AGENT_SELFTEST":"1","AGENT_RECORD":"1"}' "$BID" > "$LOG" 2>&1 &
LP=$!
for i in {1..170}; do grep -qE "\[selftest\] (DONE|ERROR)|failed to launch" "$LOG" && break; sleep 5; done
sleep 3
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
for i in {1..60}; do [[ -s "$RAW" ]] && break; sleep 2; done
grep -E "selftest\] (>|tool|result|<|turn|pause|ERROR|DONE)|failed" "$LOG" | cut -c1-160
echo "raw: $RAW ($(du -h "$RAW" | cut -f1))"
ffmpeg -v error -y -i "$RAW" -an -filter:v "setpts=PTS/3" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$FAST"
echo "3x: $FAST ($(du -h "$FAST" | cut -f1)) $(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$FAST" | cut -c1-6)s"
