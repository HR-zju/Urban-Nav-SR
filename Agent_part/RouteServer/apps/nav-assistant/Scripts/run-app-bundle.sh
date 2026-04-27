#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="NavAssistantApp"
APP_DIR="$PROJECT_DIR/.build/$APP_NAME.app"
INFO_PLIST="$PROJECT_DIR/Sources/NavAssistantApp/Info.plist"
ICON_SOURCE_DIR="$PROJECT_DIR/Sources/NavAssistantApp/Assets.xcassets/AppIcon.appiconset"

swift build --package-path "$PROJECT_DIR"

BUILD_DIR="$(swift build --package-path "$PROJECT_DIR" --show-bin-path)"
BUILD_BIN="$BUILD_DIR/$APP_NAME"
RESOURCE_BUNDLE="$BUILD_DIR/${APP_NAME}_${APP_NAME}.bundle"
RESOURCE_BUNDLE_NAME="$(basename "$RESOURCE_BUNDLE")"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/${APP_NAME}.iconwork.XXXXXX")"
ICONSET_DIR="$TMP_ROOT/AppIcon.iconset"

trap 'rm -rf "$TMP_ROOT"' EXIT

if [[ ! -x "$BUILD_BIN" ]]; then
  echo "Missing app binary at $BUILD_BIN" >&2
  exit 1
fi

if [[ ! -d "$RESOURCE_BUNDLE" ]]; then
  echo "Missing resource bundle at $RESOURCE_BUNDLE" >&2
  exit 1
fi

if [[ ! -d "$ICON_SOURCE_DIR" ]]; then
  echo "Missing app icon source at $ICON_SOURCE_DIR" >&2
  exit 1
fi

# Stop the existing app before replacing the bundle on disk.
if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
  echo "Stopping existing $APP_NAME instance(s)..."
  pkill -x "$APP_NAME" || true
  sleep 1
fi

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"
cp "$BUILD_BIN" "$MACOS_DIR/$APP_NAME"
cp "$INFO_PLIST" "$CONTENTS_DIR/Info.plist"

ditto "$RESOURCE_BUNDLE" "$RESOURCES_DIR/$RESOURCE_BUNDLE_NAME"

mkdir -p "$ICONSET_DIR"
cp "$ICON_SOURCE_DIR"/icon_*.png "$ICONSET_DIR/"
iconutil --convert icns "$ICONSET_DIR" --output "$RESOURCES_DIR/AppIcon.icns"

/usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$CONTENTS_DIR/Info.plist" >/dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Delete :CFBundleIconFile" "$CONTENTS_DIR/Info.plist" >/dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$CONTENTS_DIR/Info.plist"

codesign --force --deep --sign - "$APP_DIR" >/dev/null
touch "$APP_DIR"

open "$APP_DIR"
