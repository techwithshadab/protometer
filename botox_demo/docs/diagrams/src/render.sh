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
render architecture-overview 1440 788
render system-topology       1440 720
render ingest-pipeline       1440 430
render query-pipeline        1440 460
render graphrag-retrieval    1440 640
render protection-boundary   1440 410
render egress-guards         1440 720
render model-tiering         1440 520
render observability         1440 640
