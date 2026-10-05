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
"$BREW" install cmake ninja pkgconf glslang spirv-tools python@3.13 nasm \
    boost magic_enum robin-map   # header-only: architecture does not matter
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
    # Configure natively and cross-compile to x86-64 (the standard universal-build recipe).
    if ! quiet "ffmpeg configure" ./configure --prefix="$PREFIX" \
        --enable-cross-compile --target-os=darwin --arch=x86_64 --cc=clang \
        --extra-cflags="-arch x86_64 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
        --extra-ldflags="-arch x86_64 -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
        --x86asmexe=nasm \
        --enable-shared --disable-static --disable-programs --disable-doc --disable-network \
        --disable-everything --enable-avformat --enable-avcodec --enable-swscale --enable-swresample \
        --enable-demuxer=mov,h264,hevc,aac,mpegts --enable-parser=h264,hevc,aac \
        --enable-decoder=h264,hevc,aac --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb --enable-protocol=file; then
        echo "--- last lines of ffbuild/config.log:" >&2
        tail -25 ffbuild/config.log >&2
        false
    fi
    quiet "ffmpeg build" make -j "$JOBS"
    quiet "ffmpeg install" make install
    cd "$REPO"
fi

step "VulkanMemoryAllocator (header-only)"
VMA_TAG=$(latest_tag https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git '^v3\.[0-9]+\.[0-9]+$')
echo "tag: $VMA_TAG"
if ! have include/vk_mem_alloc.h; then
    fetch VulkanMemoryAllocator https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git "$VMA_TAG"
    cmake_build VulkanMemoryAllocator -DVMA_BUILD_DOCUMENTATION=OFF -DVMA_BUILD_SAMPLES=OFF
fi

step "xbyak (header-only; revision upstream shadPS4 pins)"
XBYAK_REV=44a72f369268f7d552650891b296693e91db86bb
if ! have include/xbyak/xbyak.h; then
    rm -rf "$SRC/xbyak" && mkdir -p "$SRC/xbyak"
    git -C "$SRC/xbyak" init -q
    git -C "$SRC/xbyak" fetch -q --depth 1 https://github.com/herumi/xbyak.git "$XBYAK_REV"
    git -C "$SRC/xbyak" -c advice.detachedHead=false checkout -q FETCH_HEAD
    mkdir -p "$PREFIX/include"
    cp -R "$SRC/xbyak/xbyak" "$PREFIX/include/"
fi

step "miniz (static, x86-64)"
MZ_TAG=$(latest_tag https://github.com/richgel999/miniz.git '^3\.[0-9]+\.[0-9]+$')
echo "tag: $MZ_TAG"
if ! have lib/libminiz.a; then
    fetch miniz https://github.com/richgel999/miniz.git "$MZ_TAG"
    cmake_build miniz -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_EXAMPLES=OFF -DBUILD_FUZZERS=OFF -DBUILD_TESTS=OFF
fi

step "Zydis (static, x86-64)"
ZY_TAG=$(latest_tag https://github.com/zyantific/zydis.git '^v4\.[0-9]+\.[0-9]+$')
echo "tag: $ZY_TAG"
# Zydis' installed config looks for Zycore as its own package, so Zycore is built and installed
# first from the revision Zydis ships (its submodule), then Zydis.
if ! have lib/libZydis.a || ! have lib/cmake/zycore/zycore-config.cmake; then
    rm -rf "$SRC/zydis" "$PREFIX/lib/libZydis.a" "$PREFIX/lib/cmake/zydis"
    git -c advice.detachedHead=false clone -q --depth 1 --recurse-submodules --shallow-submodules \
        --branch "$ZY_TAG" https://github.com/zyantific/zydis.git "$SRC/zydis"
    quiet "zycore configure" cmake -S "$SRC/zydis/dependencies/zycore" -B "$SRC/zydis/build-zycore" \
        "${X86_CMAKE[@]}" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DZYCORE_BUILD_SHARED_LIB=OFF -DZYCORE_BUILD_EXAMPLES=OFF -DZYCORE_BUILD_TESTS=OFF
    quiet "zycore build" cmake --build "$SRC/zydis/build-zycore" -j "$JOBS"
    quiet "zycore install" cmake --install "$SRC/zydis/build-zycore"
    # Zydis itself compiles against its bundled Zycore (building against the installed one drops
    # Zycore's include path); the installed Zycore above satisfies Zydis' config for consumers.
    cmake_build zydis -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DZYDIS_BUILD_SHARED_LIB=OFF -DZYDIS_BUILD_TOOLS=OFF -DZYDIS_BUILD_EXAMPLES=OFF \
        -DZYDIS_BUILD_DOXYGEN=OFF -DZYDIS_BUILD_MAN=OFF -DZYDIS_BUILD_TESTS=OFF
fi

step "KosmicKrisp (Mesa Vulkan-on-Metal driver, x86-64; macOS 26+)"
# The Vulkan driver upstream shadPS4 bundles on macOS; the vendored renderer already carries its
# driver-specific workarounds. Built with shadPS4's wrapper (meson, cross-compiled to x86-64),
# pinned to the revision upstream used at the vendored shadPS4 commit. MoltenVK stays as the
# fallback (BB_VK_DRIVER=moltenvk). BB_SKIP_KOSMICKRISP=1 skips this step.
KK_REV=${KK_REV:-3af112680499cc5eaf519007404a7366679e1c7f}
macos_major=$(sw_vers -productVersion | cut -d. -f1)
if [[ -n ${BB_SKIP_KOSMICKRISP:-} ]]; then
    echo "skipped (BB_SKIP_KOSMICKRISP)"
elif (( macos_major < 26 )); then
    echo "skipped: needs macOS 26 or later (this Mac: $(sw_vers -productVersion)); MoltenVK will be used"
elif ! have lib/kosmickrisp/libvulkan_kosmickrisp.dylib; then
    "$BREW" install meson llvm spirv-llvm-translator
    if [[ ! -x $PREFIX/pyenv/bin/python3 ]]; then
        "$("$BREW" --prefix python@3.13)/bin/python3.13" -m venv "$PREFIX/pyenv"
    fi
    quiet "kosmickrisp python modules" "$PREFIX/pyenv/bin/pip" install mako packaging pyyaml
    kk=$SRC/mesa-kosmickrisp
    if [[ $(git -C "$kk" rev-parse HEAD 2>/dev/null) != "$KK_REV" ]]; then
        rm -rf "$kk" && mkdir -p "$kk"
        git -C "$kk" init -q
        git -C "$kk" remote add origin https://github.com/shadexternals/mesa-kosmickrisp.git
        git -C "$kk" fetch -q --depth 1 origin "$KK_REV"
        git -C "$kk" -c advice.detachedHead=false checkout -q FETCH_HEAD
        echo "fetching Mesa (large, a few minutes)"
        git -C "$kk" submodule update -q --init --depth 1
    fi
    # Mesa first builds native (arm64) helper tools against Homebrew packages (LLVM,
    # SPIRV-LLVM-Translator), so this step sees Homebrew's pkg-config files, unlike the x86-64
    # libraries above; the x86-64 driver itself only links the SDK (cross file, --prefer-static).
    # Mesa's build scripts run the first python3 on PATH: the venv with mako/pyyaml.
    kk_env=(env -u PKG_CONFIG_LIBDIR
            PKG_CONFIG_PATH="$("$BREW" --prefix)/lib/pkgconfig:$("$BREW" --prefix spirv-llvm-translator)/lib/pkgconfig"
            PATH="$PREFIX/pyenv/bin:$PATH")
    rm -rf "$kk/build-x86_64"   # meson does not re-run a failed setup in place
    quiet "kosmickrisp configure" "${kk_env[@]}" cmake -S "$kk" -B "$kk/build-x86_64" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
    echo "building Mesa/KosmicKrisp (10-20 minutes)"
    quiet "kosmickrisp build" "${kk_env[@]}" cmake --build "$kk/build-x86_64" -j "$JOBS"
    mkdir -p "$PREFIX/lib/kosmickrisp"
    cp "$kk/build-x86_64/outputs/libvulkan_kosmickrisp.dylib" "$kk/build-x86_64/outputs/kosmickrisp_mesa_icd.json" \
        "$PREFIX/lib/kosmickrisp/"
fi

step "Verify: every library must contain x86_64"
bad=0
for lib in "$PREFIX"/lib/*.dylib "$PREFIX"/lib/*.a "$PREFIX"/lib/kosmickrisp/*.dylib; do
    [[ -e $lib ]] || continue
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
# Vulkan driver: KosmicKrisp when built (as upstream shadPS4 on macOS), else MoltenVK.
# BB_VK_DRIVER=moltenvk|kosmickrisp picks one explicitly.
case "\${BB_VK_DRIVER:-auto}" in
    moltenvk) export VK_DRIVER_FILES="$PREFIX/share/vulkan/icd.d/MoltenVK_icd.json" ;;
    *) if [[ -f "$PREFIX/lib/kosmickrisp/kosmickrisp_mesa_icd.json" && "\${BB_VK_DRIVER:-auto}" != moltenvk ]]; then
           export VK_DRIVER_FILES="$PREFIX/lib/kosmickrisp/kosmickrisp_mesa_icd.json"
       else
           export VK_DRIVER_FILES="$PREFIX/share/vulkan/icd.d/MoltenVK_icd.json"
       fi ;;
esac
export DYLD_LIBRARY_PATH="$PREFIX/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export MACOSX_DEPLOYMENT_TARGET=14.0
EOF
step "Done. Next: source \"$PREFIX/env.sh\""
