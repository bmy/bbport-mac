#!/usr/bin/env bash
# tools/macos/run.sh: runs the macOS build through the normal run.sh pipeline.
#   BB_GAME_DIR=~/Games/shadPS4/CUSA03173-109 bash tools/macos/run.sh
# First-light defaults (override by setting them): 30 FPS (no frame-rate patch), no upscaler.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
DEPS=${BB_DEPS:-$PWD/deps-x86_64}
# shellcheck disable=SC1091
source "$DEPS/env.sh"
[[ -x out/bb-probe ]] || { echo "STOP: out/bb-probe missing; run tools/macos/build.sh" >&2; exit 1; }
: "${BB_GAME_DIR:?set BB_GAME_DIR to the merged 1.09 game folder}"
export BB_PREBUILT=1 BB_PROBE="$PWD/out/bb-probe"
export BB_FPS=${BB_FPS:-30} BB_UPSCALER=${BB_UPSCALER:-off}
mkdir -p out
# Keep a copy of everything the port prints for diagnosis.
exec > >(tee out/last-run.log) 2>&1
echo "macOS run: $(sw_vers -productVersion), game $BB_GAME_DIR, fps $BB_FPS, upscaler $BB_UPSCALER"
exec bash run.sh "$@"
