#!/usr/bin/env bash
# tools/macos/build_launcher.sh: builds the native launcher (launcher-macos/, SwiftUI) into
# out/bbport.app. The app is native arm64; it only starts tools/macos/run.sh, which runs the
# x86-64 game under Rosetta 2. Needs Xcode or the Command Line Tools (swift).
#   bash tools/macos/build_launcher.sh
#   open out/bbport.app
# The repository path is embedded in the app (Info.plist BBRepoPath; Settings can override it).
# The icon comes from the game's sce_sys/icon0.png: BB_GAME_DIR, else the folder chosen in the app.
# Release builds (tools/macos/package.sh): BB_RELEASE=<version> embeds no repository path and no
# game icon (the game's artwork is not ours to distribute); BB_APP sets where the app goes.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
repo=$PWD
pkg=launcher-macos
app=${BB_APP:-out/bbport.app}
work=out/launcher-build
release=${BB_RELEASE:-}

[[ $(uname -s) == Darwin ]] || { echo "STOP: run this on the Mac" >&2; exit 1; }
command -v swift >/dev/null || { echo "STOP: swift missing: xcode-select --install" >&2; exit 1; }

echo "=== swift build (release, $(uname -m))"
swift --version 2>&1 | head -1
# Plain `swift build` output goes to the terminal: send all of it back if this step fails.
if ! swift build --package-path "$pkg" -c release; then
    echo "STOP: swift build failed. Send the whole output above." >&2
    exit 1
fi
bin=$(swift build --package-path "$pkg" -c release --show-bin-path)/BBLauncher
[[ -x $bin ]] || { echo "STOP: $bin missing after the build" >&2; exit 1; }

echo "=== $app"
rm -rf "$app" "$work"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$work"
cp "$bin" "$app/Contents/MacOS/bbport"
version=$(git rev-list --count HEAD 2>/dev/null || echo 1)
# macOS wants numbers here: the upstream part (0.4 of 0.4-v1); the app shows the full version.
short_version=${release%%-*}
short_version=${short_version:-0.1}
# KosmicKrisp needs macOS 26; a checkout build keeps the old minimum so its own checks explain.
min_system=14.0
[[ -n $release ]] && min_system=26.0
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>bbport</string>
    <key>CFBundleExecutable</key>
    <string>bbport</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.bbport.mac</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>bbport</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$short_version</string>
    <key>CFBundleVersion</key>
    <string>$version</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.games</string>
    <key>LSMinimumSystemVersion</key>
    <string>$min_system</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF
# plutil escapes the path for XML.
[[ -n $release ]] || plutil -insert BBRepoPath -string "$repo" "$app/Contents/Info.plist"

make_icon() {
    local source=$1 iconset=$work/AppIcon.iconset size
    mkdir -p "$iconset"
    for size in 16 32 128 256 512; do
        sips -z "$size" "$size" "$source" --out "$iconset/icon_${size}x${size}.png" >/dev/null || return 1
        if (( size * 2 <= 512 )); then
            sips -z $((size * 2)) $((size * 2)) "$source" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null || return 1
        fi
    done
    iconutil -c icns "$iconset" -o "$app/Contents/Resources/AppIcon.icns"
}
game=${BB_GAME_DIR:-$(defaults read io.github.bbport.mac gameFolder 2>/dev/null || true)}
icon=${game:+$game/sce_sys/icon0.png}
if [[ -n $release ]]; then
    echo "release build: no icon from the game (the app shows the game's icon in the Dock anyway)"
elif [[ -n $icon && -f $icon ]]; then
    if make_icon "$icon"; then
        plutil -insert CFBundleIconFile -string AppIcon "$app/Contents/Info.plist"
        echo "icon: $icon"
    else
        echo "note: no icon made from $icon (the app shows the game's icon in the Dock anyway)"
    fi
else
    echo "note: no icon yet: choose the game folder in the app, then run this script again"
fi
plutil -lint "$app/Contents/Info.plist" >/dev/null

# Ad-hoc signature: Apple Silicon refuses unsigned arm64 code. Release builds are signed again
# by tools/macos/package.sh once the game engine is inside.
codesign --force --sign - --deep "$app"
[[ -z $release ]] || exit 0
echo
echo "Built $app ($(lipo -archs "$app/Contents/MacOS/bbport"))."
echo "Open it with:  open $app"
echo "or drag out/bbport.app to /Applications (it keeps using this checkout: $repo)."
