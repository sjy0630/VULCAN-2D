#!/usr/bin/env bash
# Build the macOS desktop app (.dmg) with the Python engine bundled as a sidecar.
#
# IMPORTANT: builds into a NON-iCloud-synced directory. This project lives on an
# iCloud-synced Desktop, whose file provider constantly re-adds FinderInfo /
# provenance xattrs — which make codesign fail ("resource fork, Finder
# information, or similar detritus not allowed"). /tmp is not synced, so signing
# works there. The finished .dmg is copied back into ./release/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${VULCAN_BUILD_DIR:-/tmp/vulcan-release}"

echo "== [1/3] python engine sidecar =="
bash scripts/build_engine.sh

echo "== [2/3] UI (vite) =="
npm run build

echo "== [3/3] electron-builder (output → $OUT, ad-hoc signed) =="
rm -rf "$OUT"
env -u ELECTRON_RUN_AS_NODE npx electron-builder --mac -c.directories.output="$OUT"

mkdir -p release
cp "$OUT"/*.dmg release/ 2>/dev/null || true
echo
echo "done — dmg in ./release/:"
ls -1 release/*.dmg 2>/dev/null || echo "  (no .dmg produced — check the log above)"
