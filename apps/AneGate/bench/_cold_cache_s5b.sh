#!/bin/zsh
# S5 follow-up: is the ANE program cache per app container or system-wide? The 4096 arm loaded in 69 s in a
# FRESH container although S1 had built it on this OS build in another container (S1's own cold numbers were
# 225 s / >15 min). Discriminator: the c1024 arm has never been on this phone — its first load (main script)
# is a true cold; this script loads it once more in a SECOND fresh container (uninstall/install, no cache push).
#   true-cold ≫ second-fresh  → a system-level cache exists (container-fresh ≠ cold)
#   true-cold ≈ second-fresh  → the cache is per container and the 2B 4-bit cold build is ~1 min
#   ./_cold_cache_s5b.sh <same bench app>
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
export APP_ID=com.coreai.gemmaplebench
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR
APP=${1:?bench app path}; OUT=$DIR/_cold_cache_s5.log
ENVB='"AB_P":"128","AB_G":"256","AB_N":"1"'
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
loaded() { grep -E "engine loaded" $1 | head -1; }
wipe() { say "uninstall $APP_ID (wipes the container)"; xcrun devicectl device uninstall app --device $UDID $APP_ID 2>&1 | tail -1 | tee -a $OUT; sleep 5; }
inst() { ../_install.sh "$APP" 2>&1 | tee -a $OUT; [ ${pipestatus[1]} -eq 0 ] || { say "install failed"; exit 2; }; sleep 10; }
say "S5b: c1024 in a second fresh container"
wipe; inst
./_launch_f.sh s5cold_1024_fresh2 "STATS|FATAL" 240 "\"AB_MODEL\":\"minicpm5_2b_ane_c1024\",$ENVB" _device_s5cold_1024_fresh2.log
say "4a 1024 second fresh container: $(loaded _device_s5cold_1024_fresh2.log) | $(grep -E 'STATS|FATAL|ERROR' _device_s5cold_1024_fresh2.log | head -1 | cut -c1-120)"
wipe
say "S5b end"
