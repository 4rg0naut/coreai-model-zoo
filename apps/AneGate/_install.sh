#!/bin/zsh
# Install the app in _app_path.txt on the phone and FAIL loudly unless devicectl reports an installationURL.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd)
APP=${1:-$(cat $DIR/_app_path.txt)}
APP_ID=${APP_ID:-com.coreai.pipelinedbench}   # S5: the id the app was built with (APP_ID= at build time)
# Match a real devicectl process only (a claude session whose prompt quotes the command line would false-positive).
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device process launch" >/dev/null; }
for w in 1 2 3 4 5 6; do busy || break; sleep 5; done   # a just-finished run's devicectl may still be exiting
busy && { echo "device BUSY (another launch holds it)"; exit 2; }
for attempt in 1 2 3; do
  OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1)
  if echo "$OUT" | grep -q "installationURL"; then
    echo "installed: $(echo "$OUT" | grep installationURL)"
    # Over a Wi-Fi (localNetwork) transport iOS keeps the app "Busy (failed preflight checks)" for minutes after
    # a multi-GB install (launches die with CoreDeviceError 10002). A --start-stopped launch bypasses that check,
    # so probe with a NORMAL launch carrying a harmless env: the bench app prints "FATAL no embedded model bundle"
    # and idles, the gate app prints "FATAL fixture ... not in app resources" and idles; the next real launch
    # uses --terminate-existing. Up to 10 min.
    for i in $(seq 1 40); do
      P=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables '{"AB_MODEL":"__probe__","AG_FIXTURE":"__probe__"}' $APP_ID 2>&1)
      if echo "$P" | grep -q "Launched application"; then echo "launch-ready after $((i*15)) s"; exit 0; fi
      sleep 15
    done
    echo "installed but never launch-ready (Busy) within 10 min"; exit 3
  fi
  echo "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i "error")"; sleep 10
done
exit 1
