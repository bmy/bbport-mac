#!/usr/bin/env bash
# tools/macos/run.sh: runs the macOS build through the normal run.sh pipeline.
#   BB_GAME_DIR=~/Games/shadPS4/CUSA03173-109 bash tools/macos/run.sh
# First-light defaults (override by setting them): 30 FPS (no frame-rate patch). The upscaler is
# the one saved in bbport.ini (overlay menu), off until one is chosen.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
DEPS=${BB_DEPS:-$PWD/deps-x86_64}
# shellcheck disable=SC1091
source "$DEPS/env.sh"
[[ -f ${VK_DRIVER_FILES:-} ]] || { echo "STOP: KosmicKrisp missing (${VK_DRIVER_FILES:-unset}); run tools/macos/setup_deps.sh" >&2; exit 1; }
[[ -x out/bb-probe ]] || { echo "STOP: out/bb-probe missing; run tools/macos/build.sh" >&2; exit 1; }
: "${BB_GAME_DIR:?set BB_GAME_DIR to the merged 1.09 game folder}"
export BB_PREBUILT=1 BB_PROBE="$PWD/out/bb-probe"
export BB_FPS=${BB_FPS:-30}
mkdir -p out
# FPS counter on by default, as the Linux launcher does; the overlay menu (F1, ` or §) toggles
# it and saves the choice.
ini=${BB_CONFIG:-bbport.ini}
grep -q '^show_fps=' "$ini" 2>/dev/null || echo "show_fps=1" >> "$ini"
# Upscaler: bbport.ini, which the overlay menu saves; BB_UPSCALER=off|fsr3|taa|metalfx overrides
# it for one run. Off until chosen: FSR 3.1, the port's own default, is unproven on KosmicKrisp.
grep -q '^upscaler=' "$ini" 2>/dev/null || echo "upscaler=off" >> "$ini"
# Character motion vectors, as the Linux launcher defaults: without them temporal upscalers
# (FSR 3.1, MetalFX, TAA) smear animated characters. BB_OBJECT_MOTION=0 turns them off.
grep -q '^object_motion=' "$ini" 2>/dev/null || echo "object_motion=1" >> "$ini"
# Keep a copy of everything the port prints for diagnosis.
# Keep the previous run's log: a quick relaunch would otherwise overwrite the one to report.
[[ -f out/last-run.log ]] && mv -f out/last-run.log out/previous-run.log
exec > >(tee out/last-run.log) 2>&1
echo "macOS run: $(sw_vers -productVersion), game $BB_GAME_DIR, fps $BB_FPS, upscaler ${BB_UPSCALER:-from $ini}"
exec bash run.sh "$@"
