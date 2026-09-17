#!/bin/zsh
# Build AneGateRunner for the iPhone with the RC toolchain, embedding one bundle dir (or up to three, S5):
#   BUNDLE=<abs or repo-relative devbundle dir> [BUNDLE2=… BUNDLE3=…] [APP_ID=com.coreai.pipelinedbench] ./_build.sh
#   (several bundles: pick one per launch with AG_MODEL=<dir name>; APP_ID other than the default = a fresh app
#   container on the phone, e.g. com.coreai.gemmaplebench — provisioned, not installed — for cold-cache tests)
# (the dir holds metadata.json whose assets.main names the .h18p.aimodelc, plus tokenizer/).
# Derived data lives in the session scratchpad (big); log + .app path are recorded here.
set -e
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
DIR=$(cd "$(dirname "$0")" && pwd)
DD=${DD:-/private/tmp/ane_gate_dd/gate}   # derived data: outside any session scratchpad; per-bundle builds pass DD=/private/tmp/ane_gate_dd/<name>
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF
BUNDLE=${BUNDLE:?set BUNDLE=<devbundle dir>}
cd "$DIR"
[ -f "$BUNDLE/metadata.json" ] || { echo "no metadata.json in $BUNDLE"; exit 1; }
[ -d vendor/coreai-models-main-7359dbc/swift ] || {
  mkdir -p vendor/coreai-models-main-7359dbc
  git -C "$DIR/../../coreai-models" archive 7359dbc | tar -x -C vendor/coreai-models-main-7359dbc
}
APP_ID=${APP_ID:-com.coreai.pipelinedbench}
BLOCK2=""; BLOCK3=""
if [ -n "$BUNDLE2" ]; then
  [ -f "$BUNDLE2/metadata.json" ] || { echo "no metadata.json in $BUNDLE2"; exit 1; }
  BLOCK2="      - path: $BUNDLE2\n        type: folder\n        buildPhase: resources\n"
fi
if [ -n "$BUNDLE3" ]; then
  [ -f "$BUNDLE3/metadata.json" ] || { echo "no metadata.json in $BUNDLE3"; exit 1; }
  BLOCK3="      - path: $BUNDLE3\n        type: folder\n        buildPhase: resources\n"
fi
sed -e "s|__BUNDLE__|$BUNDLE|" -e "s|__BUNDLE2_BLOCK__|$BLOCK2|" -e "s|__BUNDLE3_BLOCK__|$BLOCK3|" -e "s|__APP_ID__|$APP_ID|" project.yml.in > project.yml
echo "bundle: $BUNDLE" > _built_bundle.txt
echo "bundle2: ${BUNDLE2:-none}" >> _built_bundle.txt
echo "bundle3: ${BUNDLE3:-none}" >> _built_bundle.txt
echo "app_id: $APP_ID" >> _built_bundle.txt
xcodegen generate > _xcodegen.log 2>&1
xcodebuild -project AneGateRunner.xcodeproj -scheme AneGateRunner -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath "$DD" build > _xcodebuild.log 2>&1 || { tail -40 _xcodebuild.log; exit 1; }
grep -E "BUILD (SUCCEEDED|FAILED)" _xcodebuild.log
echo "$DD/Build/Products/Release-iphoneos/AneGateRunner.app" > _app_path.txt
du -sh "$(cat _app_path.txt)"
ls "$(cat _app_path.txt)" | grep -v -E "^(_CodeSignature|Frameworks|Info.plist|PkgInfo|AneGateRunner|embedded.mobileprovision|fixtures)$"
