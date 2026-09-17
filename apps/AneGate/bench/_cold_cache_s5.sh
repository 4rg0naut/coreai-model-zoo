#!/bin/zsh
# S5 first-load experiments in a FRESH app container (APP_ID=com.coreai.gemmaplebench: provisioned with the
# increased-memory entitlement, not installed on the phone, so install = empty container, uninstall = wipe).
#   4a  cold ANE program build of the 4-bit 2B: max-context 4096 (30 graphs, the shipped arm) vs 1024 (18 graphs),
#       each measured in a fresh container (uninstall between).
#   4b  after the 4096 cold load, pull Library/Caches/coreai-cache, wipe (uninstall), reinstall, push the cache
#       back (plain copy, never --remove-existing-content), launch the same arm: warm (~0.3 s) or cold again?
# Each launch is the bench app in A/B mode with n=1 (p128 g256) so the log carries `engine loaded in … s`.
#   ./_cold_cache_s5.sh <bench app path built with APP_ID=com.coreai.gemmaplebench, bundles minicpm5_2b_ane + minicpm5_2b_ane_c1024>
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
export APP_ID=com.coreai.gemmaplebench
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR
APP=${1:?bench app path}; OUT=$DIR/_cold_cache_s5.log; CACHE=/private/tmp/ane_gate_dd/s5cache
ENVB='"AB_P":"128","AB_G":"256","AB_N":"1"'
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
loaded() { grep -E "engine loaded" $1 | head -1; }
files() { xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --subdirectory Library/Caches/coreai-cache 2>&1; }
wipe() { say "uninstall $APP_ID (wipes the container)"; xcrun devicectl device uninstall app --device $UDID $APP_ID 2>&1 | tail -1 | tee -a $OUT; sleep 5; }
inst() { ../_install.sh "$APP" 2>&1 | tee -a $OUT; [ ${pipestatus[1]} -eq 0 ] || { say "install failed"; exit 2; }; sleep 10; }

say "S5 cold-cache start app=$APP"
# --- 4a-1: 4096 arm, fresh container
wipe   # no-op if not installed
inst
./_launch_f.sh s5cold_4096 "STATS|FATAL" 240 "\"AB_MODEL\":\"minicpm5_2b_ane\",$ENVB" _device_s5cold_4096.log
say "4a 4096 fresh: $(loaded _device_s5cold_4096.log) | $(grep -E 'STATS|FATAL|ERROR' _device_s5cold_4096.log | head -1 | cut -c1-120)"
files > _s5cold_cache_4096.txt; say "cache entries after 4096 cold: $(head -1 _s5cold_cache_4096.txt)"
# second launch in the same container = the warm reference
./_launch_f.sh s5warm_4096_same "STATS|FATAL" 60 "\"AB_MODEL\":\"minicpm5_2b_ane\",$ENVB" _device_s5warm_4096_same.log
say "4096 same container 2nd launch: $(loaded _device_s5warm_4096_same.log)"
# --- 4b: pull the cache
rm -rf $CACHE; mkdir -p $CACHE
say "pull Library/Caches/coreai-cache → $CACHE"
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --source Library/Caches/coreai-cache --destination $CACHE 2>&1 | tail -2 | tee -a $OUT
du -sh $CACHE | tee -a $OUT; find $CACHE -type f | wc -l | tee -a $OUT
# --- 4a-2: 1024 arm, fresh container
wipe; inst
./_launch_f.sh s5cold_1024 "STATS|FATAL" 240 "\"AB_MODEL\":\"minicpm5_2b_ane_c1024\",$ENVB" _device_s5cold_1024.log
say "4a 1024 fresh: $(loaded _device_s5cold_1024.log) | $(grep -E 'STATS|FATAL|ERROR' _device_s5cold_1024.log | head -1 | cut -c1-120)"
files > _s5cold_cache_1024.txt
# --- 4b: fresh container + pushed cache, 4096 arm
wipe; inst
say "push $CACHE → Library/Caches/ (plain copy, no --remove-existing-content)"
SRC=$(ls -d $CACHE/coreai-cache 2>/dev/null || echo $CACHE)
xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $APP_ID --source $SRC --destination Library/Caches/coreai-cache 2>&1 | tail -2 | tee -a $OUT
files > _s5cold_cache_pushed.txt; say "cache entries after push: $(head -1 _s5cold_cache_pushed.txt) (before wipe: $(head -1 _s5cold_cache_4096.txt))"
./_launch_f.sh s5pushed_4096 "STATS|FATAL" 240 "\"AB_MODEL\":\"minicpm5_2b_ane\",$ENVB" _device_s5pushed_4096.log
say "4b 4096 with pushed cache: $(loaded _device_s5pushed_4096.log) | $(grep -E 'STATS|FATAL|ERROR' _device_s5pushed_4096.log | head -1 | cut -c1-120)"
wipe
say "S5 cold-cache end"
