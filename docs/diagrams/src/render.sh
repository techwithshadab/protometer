#!/usr/bin/env sh
# Render every diagram source to its committed PNG at 2x. Requires Google Chrome.
# Usage: sh docs/diagrams/src/render.sh
set -eu
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
render() { # name width height
  "$CHROME" --headless --disable-gpu --hide-scrollbars --virtual-time-budget=12000 \
    --screenshot="../$1.png" --window-size="$2,$3" --force-device-scale-factor=2 \
    "file://$PWD/$1.html" 2>/dev/null
  echo "rendered ../$1.png ($2x$3 @2x)"
}
render architecture-overview 1440 744
render pipeline-strip        1440 216
render architecture-concerns 1440 600
render model-selection       1440 430
