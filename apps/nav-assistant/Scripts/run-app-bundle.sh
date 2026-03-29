#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="NavAssistantApp"
BUILD_BIN="$PROJECT_DIR/.build/debug/$APP_NAME"
APP_DIR="$PROJECT_DIR/.build/$APP_NAME.app"
INFO_PLIST="$PROJECT_DIR/Sources/NavAssistantApp/Info.plist"

swift build --package-path "$PROJECT_DIR"

mkdir -p "$APP_DIR/Contents/MacOS"
cp "$BUILD_BIN" "$APP_DIR/Contents/MacOS/$APP_NAME"
cp "$INFO_PLIST" "$APP_DIR/Contents/Info.plist"

# Launch as a normal app bundle so macOS TCC can show microphone / speech
# permission prompts correctly. The app now resolves bridge executables using
# explicit fallback paths instead of relying on the shell environment.
if pgrep -x "$APP_NAME" >/dev/null 2>&1; then
  echo "Stopping existing $APP_NAME instance(s)..."
  pkill -x "$APP_NAME" || true
  sleep 1
fi

open "$APP_DIR"
