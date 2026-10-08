#!/usr/bin/env bash
# tools/macos/run.sh: runs the macOS build through the normal run.sh pipeline.
#   BB_GAME_DIR=~/Games/shadPS4/CUSA03173-109 bash tools/macos/run.sh
# First-light defaults (override by setting them): 30 FPS (no frame-rate patch). The upscaler is
# the one saved in bbport.ini (overlay menu), off until one is chosen.
#
# Two layouts:
# - a checkout: deps-x86_64/ (tools/macos/setup_deps.sh) and out/bb-probe (tools/macos/build.sh);
#   bbport.ini, out/ (generated files, logs) and user/ (saves) live in the checkout.
# - the release app (tools/macos/package.sh): this script inside bbport.app/Contents/Resources/
#   bbport, next to bin/ and lib/ (marker file .packaged). Nothing is written into the app:
#   bbport.ini, out/ and user/ go to BB_DATA_DIR, by default ~/Library/Application Support/bbport.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
if [[ -f .packaged ]]; then
    export VK_DRIVER_FILES="$PWD/lib/kosmickrisp/kosmickrisp_mesa_icd.json"
    export BB_PROBE="$PWD/bin/bb-probe"
    # The app's own Vulkan loader, by path: DYLD_LIBRARY_PATH would not survive bash starting.
    export BB_VULKAN_LIBRARY="$PWD/lib/libvulkan.1.dylib" SDL_VULKAN_LIBRARY="$PWD/lib/libvulkan.1.dylib"
    export BB_DATA_DIR=${BB_DATA_DIR:-$HOME/Library/Application Support/bbport}
    # Python must not write __pycache__ into the signed app.
    export PYTHONDONTWRITEBYTECODE=1
    data=$BB_DATA_DIR
    [[ -f $VK_DRIVER_FILES ]] || { echo "STOP: KosmicKrisp missing from the app ($VK_DRIVER_FILES)" >&2; exit 1; }
    [[ -x $BB_PROBE ]] || { echo "STOP: $BB_PROBE missing from the app" >&2; exit 1; }
else
    DEPS=${BB_DEPS:-$PWD/deps-x86_64}
    # shellcheck disable=SC1091
    source "$DEPS/env.sh"
    [[ -f ${VK_DRIVER_FILES:-} ]] || { echo "STOP: KosmicKrisp missing (${VK_DRIVER_FILES:-unset}); run tools/macos/setup_deps.sh" >&2; exit 1; }
    [[ -x out/bb-probe ]] || { echo "STOP: out/bb-probe missing; run tools/macos/build.sh" >&2; exit 1; }
    export BB_PROBE="$PWD/out/bb-probe"
    data=${BB_DATA_DIR:-.}
fi
: "${BB_GAME_DIR:?set BB_GAME_DIR to the merged 1.09 game folder}"
export BB_PREBUILT=1
export BB_FPS=${BB_FPS:-30}
mkdir -p "$data/out"
# FPS counter on by default, as the Linux launcher does; the overlay menu (F1, ` or §) toggles
# it and saves the choice.
ini=${BB_CONFIG:-$data/bbport.ini}
export BB_CONFIG=$ini
grep -q '^show_fps=' "$ini" 2>/dev/null || echo "show_fps=1" >> "$ini"
# Upscaler: bbport.ini, which the overlay menu saves; BB_UPSCALER=off|fsr3|taa|metalfx overrides
# it for one run. Off until chosen: FSR 3.1, the port's own default, is unproven on KosmicKrisp.
grep -q '^upscaler=' "$ini" 2>/dev/null || echo "upscaler=off" >> "$ini"
# Character motion vectors, as the Linux launcher defaults: without them temporal upscalers
# (FSR 3.1, MetalFX, TAA) smear animated characters. BB_OBJECT_MOTION=0 turns them off.
grep -q '^object_motion=' "$ini" 2>/dev/null || echo "object_motion=1" >> "$ini"
# Keep a copy of everything the port prints for diagnosis.
# Keep the previous run's log: a quick relaunch would otherwise overwrite the one to report.
log=$data/out/last-run.log
[[ -f $log ]] && mv -f "$log" "$data/out/previous-run.log"
exec > >(tee "$log") 2>&1
version=$(cat .version 2>/dev/null || git log -1 --format='%h' 2>/dev/null || echo unknown)
echo "macOS run: $(sw_vers -productVersion), bbport $version, game $BB_GAME_DIR, fps $BB_FPS, upscaler ${BB_UPSCALER:-from $ini}"
exec bash run.sh "$@"
