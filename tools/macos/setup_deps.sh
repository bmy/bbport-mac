#!/usr/bin/env bash
# tools/macos/setup_deps.sh: one-time toolchain for the macOS (x86-64 / Rosetta 2) build.
#
# The game's code runs inside our process, so every library linked into bb-probe and
# libbbgpu must be x86-64. Build tools only run at build time and stay native (arm64).
#   native (Apple Silicon Homebrew): cmake ninja pkgconf glslang spirv-tools python3
#   x86-64, built here into $PREFIX:   SDL3 fmt xxHash Vulkan-Headers Vulkan-Loader FFmpeg
#   x86-64, prebuilt (universal):      MoltenVK (KhronosGroup release)
# Intel Homebrew is avoided: it is Tier 3 (no new bottles) since 2026.
#
# Usage: bash tools/macos/setup_deps.sh [prefix]   (default: <repo>/deps-x86_64)
# Re-running skips what is already installed. Log: <prefix>/setup.log
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
REPO=$PWD
PREFIX=${1:-$REPO/deps-x86_64}
SRC=$PREFIX/src
mkdir -p "$PREFIX" "$SRC"
exec > >(tee -a "$PREFIX/setup.log") 2>&1
JOBS=$(sysctl -n hw.ncpu)
CURRENT_STEP=start
step() { CURRENT_STEP=$*; printf '\n=== %s\n' "$*"; }
trap 'printf "\nSTOP: failed during: %s (line %s). Full output above and in %s\n" "$CURRENT_STEP" "$LINENO" "$PREFIX/setup.log" >&2' ERR
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

step "Checks"
[[ $(uname -s) == Darwin ]] || die "run this on the Mac"
arch -x86_64 /usr/bin/true 2>/dev/null || die "Rosetta 2 missing: softwareupdate --install-rosetta --agree-to-license"
xcrun --find clang >/dev/null 2>&1 || die "Command Line Tools missing: xcode-select --install"
BREW=$(command -v brew || true)
[[ -z $BREW && -x /opt/homebrew/bin/brew ]] && BREW=/opt/homebrew/bin/brew
[[ -n $BREW ]] || die "Apple Silicon Homebrew missing: see https://brew.sh"
[[ $("$BREW" --prefix) == /opt/homebrew ]] || echo "note: brew prefix is $("$BREW" --prefix), expected /opt/homebrew"

step "Native build tools (Homebrew)"
"$BREW" install cmake ninja pkgconf glslang spirv-tools python@3.13 nasm
export PATH="$("$BREW" --prefix)/bin:$PATH"

# x86-64 everywhere below. pkg-config must only see $PREFIX, never Homebrew's arm64 .pc files.
export MACOSX_DEPLOYMENT_TARGET=14.0
X86_CMAKE=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64
           -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64
           -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"
           -DCMAKE_FIND_ROOT_PATH="$PREFIX" -DBUILD_SHARED_LIBS=ON)
export PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
unset PKG_CONFIG_PATH

latest_tag() {  # latest_tag <git url> <regex>: highest version tag matching the regex
    git ls-remote --tags --refs "$1" | sed 's|.*refs/tags/||' | grep -E "$2" | sort -V | tail -1
}
fetch() {  # fetch <name> <git url> <tag>
    local dir=$SRC/$1
    if [[ ! -d $dir/.git ]] || [[ $(git -C "$dir" describe --tags 2>/dev/null) != "$3" ]]; then
        rm -rf "$dir"
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$3" "$2" "$dir"
    fi
}
cmake_build() {  # cmake_build <name> [cmake args...]
    local name=$1; shift
    quiet "$name configure" cmake -S "$SRC/$name" -B "$SRC/$name/build-x86_64" "${X86_CMAKE[@]}" "$@"
    quiet "$name build" cmake --build "$SRC/$name/build-x86_64" -j "$JOBS"
    quiet "$name install" cmake --install "$SRC/$name/build-x86_64"
}
have() { [[ -e $PREFIX/$1 ]]; }
quiet() {  # quiet <label> <command...>: run with output in a log; print its tail on failure
    local label=$1; shift
    local log="$PREFIX/logs/${label// /-}.log"
    mkdir -p "$PREFIX/logs"
    if ! "$@" >"$log" 2>&1; then
        printf 'STOP: %s failed; last lines of %s:\n' "$label" "$log" >&2
        tail -30 "$log" >&2
        return 1
    fi
}

step "Vulkan-Headers + Vulkan-Loader"
VK_TAG=$(latest_tag https://github.com/KhronosGroup/Vulkan-Loader.git '^v1\.4\.[0-9]+$')
echo "tag: $VK_TAG"
if ! have lib/libvulkan.1.dylib; then
    fetch Vulkan-Headers https://github.com/KhronosGroup/Vulkan-Headers.git "$VK_TAG"
    cmake_build Vulkan-Headers
    fetch Vulkan-Loader https://github.com/KhronosGroup/Vulkan-Loader.git "$VK_TAG"
    cmake_build Vulkan-Loader -DVULKAN_HEADERS_INSTALL_DIR="$PREFIX" -DBUILD_TESTS=OFF
fi

step "MoltenVK (prebuilt universal release)"
if ! have lib/libMoltenVK.dylib; then
    url=$(curl -fsSL https://api.github.com/repos/KhronosGroup/MoltenVK/releases/latest |
          python3 -c 'import json,sys; a=json.load(sys.stdin)["assets"]; print(next(x["browser_download_url"] for x in a if x["name"]=="MoltenVK-macos.tar"))')
    echo "from: $url"
    curl -fL "$url" -o "$SRC/MoltenVK-macos.tar"
    rm -rf "$SRC/MoltenVK-macos" && mkdir -p "$SRC/MoltenVK-macos"
    tar -xf "$SRC/MoltenVK-macos.tar" -C "$SRC/MoltenVK-macos"
    dylib=$(find "$SRC/MoltenVK-macos" -path '*dynamic*' -name libMoltenVK.dylib | head -1)
    icd=$(find "$SRC/MoltenVK-macos" -name MoltenVK_icd.json | head -1)
    [[ -n $dylib && -n $icd ]] || die "MoltenVK archive layout changed (no libMoltenVK.dylib / MoltenVK_icd.json)"
    mvk_archs=$(lipo -archs "$dylib")
    [[ $mvk_archs == *x86_64* ]] || die "MoltenVK dylib has no x86_64 slice: $mvk_archs"
    mkdir -p "$PREFIX/lib" "$PREFIX/share/vulkan/icd.d"
    cp "$dylib" "$PREFIX/lib/"
    python3 - "$icd" "$PREFIX" <<'PY'
import json, sys
icd, prefix = sys.argv[1], sys.argv[2]
data = json.load(open(icd))
data["ICD"]["library_path"] = f"{prefix}/lib/libMoltenVK.dylib"
json.dump(data, open(f"{prefix}/share/vulkan/icd.d/MoltenVK_icd.json", "w"), indent=2)
PY
fi

step "SDL3"
SDL_TAG=$(latest_tag https://github.com/libsdl-org/SDL.git '^release-3\.[0-9]+\.[0-9]+$')
echo "tag: $SDL_TAG"
if ! have lib/libSDL3.dylib; then
    fetch SDL https://github.com/libsdl-org/SDL.git "$SDL_TAG"
    cmake_build SDL -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF
fi

step "fmt"
FMT_TAG=$(latest_tag https://github.com/fmtlib/fmt.git '^12\.[0-9]+\.[0-9]+$')
echo "tag: $FMT_TAG"
if ! have lib/libfmt.dylib; then
    fetch fmt https://github.com/fmtlib/fmt.git "$FMT_TAG"
    cmake_build fmt -DFMT_TEST=OFF -DFMT_DOC=OFF
fi

step "xxHash"
XXH_TAG=$(latest_tag https://github.com/Cyan4973/xxHash.git '^v0\.8\.[0-9]+$')
echo "tag: $XXH_TAG"
if ! have lib/libxxhash.dylib; then
    fetch xxHash https://github.com/Cyan4973/xxHash.git "$XXH_TAG"
    quiet "xxHash configure" cmake -S "$SRC/xxHash/build/cmake" -B "$SRC/xxHash/build-x86_64" "${X86_CMAKE[@]}" \
          -DXXHASH_BUILD_XXHSUM=OFF
    quiet "xxHash build" cmake --build "$SRC/xxHash/build-x86_64" -j "$JOBS"
    quiet "xxHash install" cmake --install "$SRC/xxHash/build-x86_64"
fi

step "FFmpeg (decoders the game's movies need; x86-64)"
FF_TAG=$(latest_tag https://git.ffmpeg.org/ffmpeg.git '^n[0-9]+\.[0-9]+(\.[0-9]+)?$')
echo "tag: $FF_TAG"
if ! have lib/libavformat.dylib; then
    fetch ffmpeg https://git.ffmpeg.org/ffmpeg.git "$FF_TAG"
    cd "$SRC/ffmpeg"
    quiet "ffmpeg configure" arch -x86_64 ./configure --prefix="$PREFIX" --arch=x86_64 --cc="clang -arch x86_64" \
        --enable-shared --disable-static --disable-programs --disable-doc --disable-network \
        --disable-everything --enable-avformat --enable-avcodec --enable-swscale --enable-swresample \
        --enable-demuxer=mov,h264,hevc,aac,mpegts --enable-parser=h264,hevc,aac \
        --enable-decoder=h264,hevc,aac --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb --enable-protocol=file
    quiet "ffmpeg build" arch -x86_64 make -j "$JOBS"
    quiet "ffmpeg install" make install
    cd "$REPO"
fi

step "Verify: every library must contain x86_64"
bad=0
for lib in "$PREFIX"/lib/*.dylib; do
    [[ -L $lib ]] && continue
    archs=$(lipo -archs "$lib")
    printf '  %-40s %s\n' "$(basename "$lib")" "$archs"
    [[ $archs == *x86_64* ]] || bad=1
done
(( bad == 0 )) || die "a library lacks x86_64 (see above)"

cat > "$PREFIX/env.sh" <<EOF
# source this before building or running the macOS port
export BB_DEPS="$PREFIX"
export PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
unset PKG_CONFIG_PATH
export VK_DRIVER_FILES="$PREFIX/share/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$PREFIX/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export MACOSX_DEPLOYMENT_TARGET=14.0
EOF
step "Done. Next: source \"$PREFIX/env.sh\""
