#!/usr/bin/env bash
# tools/macos/setup_deps.sh: one-time toolchain for the macOS (x86-64 / Rosetta 2) build.
# BB_ARCH=arm64 builds the same libraries natively into deps-arm64 instead, for the native GPU
# process under development (docs/macos-native-gpu.md); the x86-64 build does not need it.
#
# The game's code runs inside our process, so every library linked into bb-probe and
# libbbgpu must be x86-64. Build tools only run at build time and stay native (arm64).
#   native (Apple Silicon Homebrew): cmake ninja pkgconf glslang spirv-tools python3
#   x86-64, built here into $PREFIX:   SDL3 fmt xxHash Vulkan-Headers Vulkan-Loader FFmpeg
#                                      KosmicKrisp (Mesa's Vulkan-on-Metal driver)
# Intel Homebrew is avoided: it is Tier 3 (no new bottles) since 2026.
#
# Usage: bash tools/macos/setup_deps.sh [prefix]   (default: <repo>/deps-x86_64, or deps-arm64)
# Re-running skips what is already installed. Log: <prefix>/setup.log
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
REPO=$PWD
ARCH=${BB_ARCH:-x86_64}
[[ $ARCH == x86_64 || $ARCH == arm64 ]] || { echo "STOP: BB_ARCH must be x86_64 or arm64" >&2; exit 1; }
PREFIX=${1:-$REPO/deps-$ARCH}
SRC=$PREFIX/src
mkdir -p "$PREFIX" "$SRC"
# update.sh's record of what this prefix was built from: no longer true from here on.
rm -f "$PREFIX/.bb-recipe"
exec > >(tee -a "$PREFIX/setup.log") 2>&1
JOBS=$(sysctl -n hw.ncpu)
CURRENT_STEP=start
step() { CURRENT_STEP=$*; printf '\n=== %s\n' "$*"; }
trap 'printf "\nSTOP: failed during: %s (line %s). Full output above and in %s\n" "$CURRENT_STEP" "$LINENO" "$PREFIX/setup.log" >&2' ERR
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

step "Checks"
[[ $(uname -s) == Darwin ]] || die "run this on the Mac"
# KosmicKrisp, the only Vulkan driver that runs bbport on the Mac, needs macOS 26.
(( $(sw_vers -productVersion | cut -d. -f1) >= 26 )) || die "macOS 26 or later required (this Mac: $(sw_vers -productVersion))"
arch -x86_64 /usr/bin/true 2>/dev/null || die "Rosetta 2 missing: softwareupdate --install-rosetta --agree-to-license"
xcrun --find clang >/dev/null 2>&1 || die "Command Line Tools missing: xcode-select --install"
BREW=$(command -v brew || true)
[[ -z $BREW && -x /opt/homebrew/bin/brew ]] && BREW=/opt/homebrew/bin/brew
[[ -n $BREW ]] || die "Apple Silicon Homebrew missing: see https://brew.sh"
[[ $("$BREW" --prefix) == /opt/homebrew ]] || echo "note: brew prefix is $("$BREW" --prefix), expected /opt/homebrew"

step "Native build tools (Homebrew)"
# ccache: compiles of a file the build has seen before (another branch, an earlier commit) come
# from its cache (tools/macos/build.sh).
"$BREW" install cmake ninja pkgconf glslang spirv-tools python@3.13 nasm ccache \
    boost magic_enum robin-map   # header-only: architecture does not matter
export PATH="$("$BREW" --prefix)/bin:$PATH"

# $ARCH everywhere below. pkg-config must only see $PREFIX, never Homebrew's .pc files.
export MACOSX_DEPLOYMENT_TARGET=14.0
X86_CMAKE=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=$ARCH
           -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=$ARCH
           -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX"
           -DCMAKE_FIND_ROOT_PATH="$PREFIX" -DBUILD_SHARED_LIBS=ON)
export PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
unset PKG_CONFIG_PATH

# Every source is pinned to a tag and the commit it pointed to when it was pinned. A tag that
# has since moved stops the build instead of building whatever it points to now. To update a
# dependency, change both, and the commit with:
#   git ls-remote https://github.com/<owner>/<repo>.git 'refs/tags/<tag>^{}' 'refs/tags/<tag>'
# (the line ending in ^{} is the commit, when there is one). After an update, do a security
# review of the change (docs/security-review.md).
verify_rev() {  # verify_rev <dir> <commit>
    local actual
    actual=$(git -C "$1" rev-parse HEAD)
    [[ $actual == "$2" ]] || die "$1: expected commit $2, got $actual (the tag moved?)"
}
# Network steps are tried 3 times (a server's hiccup must not end a long build); what they fetch
# is still checked against the pinned commit.
retry() {  # retry <command...>
    local try
    for try in 1 2 3; do
        "$@" && return 0
        echo "attempt $try failed: $*" >&2
        (( try < 3 )) && sleep $(( try * 15 ))
    done
    return 1
}
clone_into() {  # clone_into <dir> <tag> <git url>: a fresh shallow clone
    rm -rf "$1"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$2" "$3" "$1"
}
fetch() {  # fetch <name> <git url> <tag> <commit> [mirror url]: the mirror if the first fails
    local dir=$SRC/$1 url ok=
    if [[ ! -d $dir/.git ]] || [[ $(git -C "$dir" rev-parse HEAD 2>/dev/null) != "$4" ]]; then
        for url in "$2" ${5:+"$5"}; do
            retry clone_into "$dir" "$3" "$url" && { ok=1; break; }
        done
        [[ -n $ok ]] || die "$1: could not clone $2${5:+ or $5}"
    fi
    verify_rev "$dir" "$4"
}
cmake_build() {  # cmake_build <name> [cmake args...]
    local name=$1; shift
    quiet "$name configure" cmake -S "$SRC/$name" -B "$SRC/$name/build-$ARCH" "${X86_CMAKE[@]}" "$@"
    quiet "$name build" cmake --build "$SRC/$name/build-$ARCH" -j "$JOBS"
    quiet "$name install" cmake --install "$SRC/$name/build-$ARCH"
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
VK_TAG=v1.4.365
VK_HEADERS_REV=c46850864f4661461b0f6cb9922c058ffea4915e
VK_LOADER_REV=f866657ff687a36d767cd7783bab800e31ebafaa
echo "tag: $VK_TAG"
if ! have lib/libvulkan.1.dylib; then
    fetch Vulkan-Headers https://github.com/KhronosGroup/Vulkan-Headers.git "$VK_TAG" "$VK_HEADERS_REV"
    cmake_build Vulkan-Headers
    fetch Vulkan-Loader https://github.com/KhronosGroup/Vulkan-Loader.git "$VK_TAG" "$VK_LOADER_REV"
    cmake_build Vulkan-Loader -DVULKAN_HEADERS_INSTALL_DIR="$PREFIX" -DBUILD_TESTS=OFF
fi

# MoltenVK was the driver here once; it can't run bbport (no sparse buffers). Remove what an
# older setup left behind.
rm -rf "$PREFIX/lib/libMoltenVK.dylib" "$PREFIX/share/vulkan/icd.d/MoltenVK_icd.json" \
       "$SRC/MoltenVK-macos" "$SRC/MoltenVK-macos.tar"

step "SDL3"
SDL_TAG=release-3.4.18 SDL_REV=829a65d769d935c4852f8159e964312c0957260a
echo "tag: $SDL_TAG"
if ! have lib/libSDL3.dylib; then
    fetch SDL https://github.com/libsdl-org/SDL.git "$SDL_TAG" "$SDL_REV"
    cmake_build SDL -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF
fi

step "fmt"
FMT_TAG=12.2.0 FMT_REV=1be298e1bd68957e4cd352e1f676f00e07dcfb57
echo "tag: $FMT_TAG"
if ! have lib/libfmt.dylib; then
    fetch fmt https://github.com/fmtlib/fmt.git "$FMT_TAG" "$FMT_REV"
    cmake_build fmt -DFMT_TEST=OFF -DFMT_DOC=OFF
fi

step "xxHash"
XXH_TAG=v0.8.4 XXH_REV=c87183a77d67f7d37e3d2d1b7eaac5e7c695e4f0
echo "tag: $XXH_TAG"
if ! have lib/libxxhash.dylib; then
    fetch xxHash https://github.com/Cyan4973/xxHash.git "$XXH_TAG" "$XXH_REV"
    quiet "xxHash configure" cmake -S "$SRC/xxHash/build/cmake" -B "$SRC/xxHash/build-$ARCH" "${X86_CMAKE[@]}" \
          -DXXHASH_BUILD_XXHSUM=OFF
    quiet "xxHash build" cmake --build "$SRC/xxHash/build-$ARCH" -j "$JOBS"
    quiet "xxHash install" cmake --install "$SRC/xxHash/build-$ARCH"
fi

step "FFmpeg (decoders the game's movies need; $ARCH)"
FF_TAG=n9.0.2 FF_REV=946fcce07b6dcd0331c8cc609192aeff5e1924f8
echo "tag: $FF_TAG"
if ! have lib/libavformat.dylib; then
    fetch ffmpeg https://git.ffmpeg.org/ffmpeg.git "$FF_TAG" "$FF_REV" https://github.com/FFmpeg/FFmpeg.git
    cd "$SRC/ffmpeg"
    # Configure natively and cross-compile to x86-64 (the standard universal-build recipe);
    # arm64 is a plain native build.
    if [[ $ARCH == x86_64 ]]; then
        ff_target=(--enable-cross-compile --target-os=darwin --arch=x86_64 --x86asmexe=nasm)
    else
        ff_target=(--target-os=darwin --arch=arm64)
    fi
    if ! quiet "ffmpeg configure" ./configure --prefix="$PREFIX" \
        "${ff_target[@]}" --cc=clang \
        --extra-cflags="-arch $ARCH -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
        --extra-ldflags="-arch $ARCH -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET" \
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
VMA_TAG=v3.4.0 VMA_REV=3aa921224c154a0d2c43912bc88e1c42ce1f7607
echo "tag: $VMA_TAG"
if ! have include/vk_mem_alloc.h; then
    fetch VulkanMemoryAllocator https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git "$VMA_TAG" "$VMA_REV"
    cmake_build VulkanMemoryAllocator -DVMA_BUILD_DOCUMENTATION=OFF -DVMA_BUILD_SAMPLES=OFF
fi

if [[ $ARCH == arm64 ]]; then
    # SPIR-V -> MSL for the Metal command path (docs/MACOS_METAL_RENDERER.md; the native GPU
    # process only): static, the MSL back end and what it needs.
    step "SPIRV-Cross (static, arm64)"
    SPVC_TAG=vulkan-sdk-1.4.363.0 SPVC_REV=f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7
    echo "tag: $SPVC_TAG"
    # The stamp names the pin: the GPU build takes SPIRV-Cross only from a prefix that has it
    # (gpu/CMakeLists.txt), and a new pin rebuilds it.
    spvc_stamp=$PREFIX/share/spirv_cross_msl/.bbport-rev
    if ! have lib/libspirv-cross-msl.a || [[ $(cat "$spvc_stamp" 2>/dev/null) != "$SPVC_TAG $SPVC_REV" ]]; then
        rm -f "$spvc_stamp"
        fetch SPIRV-Cross https://github.com/KhronosGroup/SPIRV-Cross.git "$SPVC_TAG" "$SPVC_REV"
        cmake_build SPIRV-Cross -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_STATIC=ON \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DSPIRV_CROSS_CLI=OFF \
            -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_ENABLE_HLSL=OFF \
            -DSPIRV_CROSS_ENABLE_CPP=OFF -DSPIRV_CROSS_ENABLE_REFLECT=OFF \
            -DSPIRV_CROSS_ENABLE_C_API=OFF -DSPIRV_CROSS_ENABLE_UTIL=OFF
        # Apache-2.0 (headers MIT and Khronos): its licences go with the binaries it is linked
        # into (package.sh).
        mkdir -p "$PREFIX/share/licenses/SPIRV-Cross"
        cp "$SRC/SPIRV-Cross/LICENSE" "$PREFIX/share/licenses/SPIRV-Cross/LICENSE.txt"
        cp "$SRC/SPIRV-Cross"/LICENSES/* "$PREFIX/share/licenses/SPIRV-Cross/"
        echo "$SPVC_TAG $SPVC_REV" > "$spvc_stamp"
    fi
fi

step "xbyak (header-only; revision upstream shadPS4 pins)"
XBYAK_REV=44a72f369268f7d552650891b296693e91db86bb
if ! have include/xbyak/xbyak.h; then
    rm -rf "$SRC/xbyak" && mkdir -p "$SRC/xbyak"
    git -C "$SRC/xbyak" init -q
    retry git -C "$SRC/xbyak" fetch -q --depth 1 https://github.com/herumi/xbyak.git "$XBYAK_REV"
    git -C "$SRC/xbyak" -c advice.detachedHead=false checkout -q FETCH_HEAD
    verify_rev "$SRC/xbyak" "$XBYAK_REV"
    mkdir -p "$PREFIX/include"
    cp -R "$SRC/xbyak/xbyak" "$PREFIX/include/"
fi

step "miniz (static, $ARCH)"
MZ_TAG=3.1.2 MZ_REV=77d0dce8627735138c51770d1799a1ef48f2117d
echo "tag: $MZ_TAG"
if ! have lib/libminiz.a; then
    fetch miniz https://github.com/richgel999/miniz.git "$MZ_TAG" "$MZ_REV"
    cmake_build miniz -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_EXAMPLES=OFF -DBUILD_FUZZERS=OFF -DBUILD_TESTS=OFF
fi

step "Zydis (static, $ARCH)"
ZY_TAG=v4.1.1 ZY_REV=a2278f1d254e492f6a6b39f6cb5d1f5d515659dc
echo "tag: $ZY_TAG"
# Zydis' installed config looks for Zycore as its own package, so Zycore is built and installed
# first from the revision Zydis ships (its submodule), then Zydis.
if ! have lib/libZydis.a || ! have lib/cmake/zycore/zycore-config.cmake; then
    rm -rf "$SRC/zydis" "$PREFIX/lib/libZydis.a" "$PREFIX/lib/cmake/zydis"
    zydis_clone() {
        rm -rf "$SRC/zydis"
        git -c advice.detachedHead=false clone -q --depth 1 --recurse-submodules --shallow-submodules \
            --branch "$ZY_TAG" https://github.com/zyantific/zydis.git "$SRC/zydis"
    }
    retry zydis_clone
    verify_rev "$SRC/zydis" "$ZY_REV"
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

step "KosmicKrisp (Mesa Vulkan-on-Metal driver, $ARCH)"
# The Vulkan driver upstream shadPS4 bundles on macOS; the vendored renderer already carries its
# driver-specific workarounds. Built with shadPS4's wrapper (meson, cross-compiled to x86-64).
# squidbus's revision of 2026-10-16, the one mainline shadPS4 bundles: Mesa rebased again
# (b23137f), a sparse buffer fix, and geometry shaders (emulated with compute; af7dd0a);
# BB_GEOMETRY_SHADERS=0 hides them. The previous revision (a82cadb, whose Mesa 8b06c1f upstream
# no longer has on a branch) is kept on this repository's vendor/kosmickrisp-a82cadb and
# vendor/mesa-8b06c1f branches for a rollback.
KK_REV=${KK_REV:-52d87b35f3a4fc8aff5acb4daaa548c8ff7870f1}
# This port's Mesa patches (tools/macos/kosmickrisp-patches): a change rebuilds the driver.
KK_PATCHES=$(cat "$REPO"/tools/macos/kosmickrisp-patches/*.patch 2>/dev/null | shasum | cut -c1-16)
kk_stamp=$PREFIX/lib/kosmickrisp/.bbport-patches
# The stamp holds the revision and the patches, so a change to either rebuilds the driver.
KK_STAMP="$KK_REV $KK_PATCHES"
if ! have lib/kosmickrisp/libvulkan_kosmickrisp.dylib || [[ $(cat "$kk_stamp" 2>/dev/null) != "$KK_STAMP" ]]; then
    # A failed rebuild must not leave the previous driver in place looking current.
    rm -f "$kk_stamp" "$PREFIX/lib/kosmickrisp/libvulkan_kosmickrisp.dylib" \
        "$PREFIX/lib/kosmickrisp/kosmickrisp_mesa_icd.json"
    "$BREW" install meson llvm spirv-llvm-translator
    if [[ ! -x $PREFIX/pyenv/bin/python3 ]]; then
        "$("$BREW" --prefix python@3.13)/bin/python3.13" -m venv "$PREFIX/pyenv"
    fi
    # Mesa's build scripts need these; pinned like the sources above.
    quiet "kosmickrisp python modules" "$PREFIX/pyenv/bin/pip" install --only-binary :all: \
        mako==1.3.10 markupsafe==3.0.2 packaging==25.0 pyyaml==6.0.2
    kk=$SRC/mesa-kosmickrisp
    if [[ $(git -C "$kk" rev-parse HEAD 2>/dev/null) != "$KK_REV" ]]; then
        rm -rf "$kk" && mkdir -p "$kk"
        git -C "$kk" init -q
        git -C "$kk" remote add origin https://github.com/shadexternals/mesa-kosmickrisp.git
        retry git -C "$kk" fetch -q --depth 1 origin "$KK_REV"
        git -C "$kk" -c advice.detachedHead=false checkout -q FETCH_HEAD
        verify_rev "$kk" "$KK_REV"
        echo "fetching Mesa"
        retry git -C "$kk" submodule update -q --init --depth 1
    fi
    # Our patches on top of the pinned Mesa, applied in order to a clean checkout: a changed or
    # added patch must not meet an earlier version of the series (checking each patch on its own
    # breaks when a later patch edits the same lines as an earlier one).
    # (The submodule is checked out at its pinned commit first, in case an earlier run stopped
    # before it was.)
    retry git -C "$kk" submodule update -q --init --force --depth 1
    git -C "$kk/externals/mesa" reset -q --hard
    git -C "$kk/externals/mesa" clean -q -fd
    for patch in "$REPO"/tools/macos/kosmickrisp-patches/*.patch; do
        [[ -f $patch ]] || continue
        git -C "$kk/externals/mesa" apply "$patch"
        echo "applied $(basename "$patch")"
    done
    # Mesa first builds native (arm64) helper tools against Homebrew packages (LLVM,
    # SPIRV-LLVM-Translator), so this step sees Homebrew's pkg-config files, unlike the x86-64
    # libraries above; the x86-64 driver itself only links the SDK (cross file, --prefer-static).
    # Mesa's build scripts run the first python3 on PATH: the venv with mako/pyyaml.
    kk_env=(env -u PKG_CONFIG_LIBDIR
            PKG_CONFIG_PATH="$("$BREW" --prefix)/lib/pkgconfig:$("$BREW" --prefix spirv-llvm-translator)/lib/pkgconfig"
            PATH="$PREFIX/pyenv/bin:$PATH")
    rm -rf "$kk/build-$ARCH"   # meson does not re-run a failed setup in place
    quiet "kosmickrisp configure" "${kk_env[@]}" cmake -S "$kk" -B "$kk/build-$ARCH" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=$ARCH -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
    echo "building Mesa/KosmicKrisp"
    quiet "kosmickrisp build" "${kk_env[@]}" cmake --build "$kk/build-$ARCH" -j "$JOBS"
    mkdir -p "$PREFIX/lib/kosmickrisp"
    cp "$kk/build-$ARCH/outputs/libvulkan_kosmickrisp.dylib" "$kk/build-$ARCH/outputs/kosmickrisp_mesa_icd.json" \
        "$PREFIX/lib/kosmickrisp/"
    echo "$KK_STAMP" > "$kk_stamp"
fi

step "Verify: every library must contain $ARCH"
bad=0
for lib in "$PREFIX"/lib/*.dylib "$PREFIX"/lib/*.a "$PREFIX"/lib/kosmickrisp/*.dylib; do
    [[ -e $lib ]] || continue
    [[ -L $lib ]] && continue
    archs=$(lipo -archs "$lib")
    printf '  %-40s %s\n' "$(basename "$lib")" "$archs"
    [[ $archs == *$ARCH* ]] || bad=1
done
(( bad == 0 )) || die "a library lacks $ARCH (see above)"

q=$(printf '%q' "$PREFIX")   # quoted for the shell that sources env.sh
cat > "$PREFIX/env.sh" <<EOF
# source this before building or running the macOS port
export BB_DEPS=$q
export PKG_CONFIG_LIBDIR=$q/lib/pkgconfig:$q/share/pkgconfig
unset PKG_CONFIG_PATH
# Vulkan driver: KosmicKrisp (Mesa on Metal), as upstream shadPS4 bundles on macOS.
export VK_DRIVER_FILES=$q/lib/kosmickrisp/kosmickrisp_mesa_icd.json
export DYLD_LIBRARY_PATH=$q/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}
export MACOSX_DEPLOYMENT_TARGET=14.0
EOF
step "Done. Next: source \"$PREFIX/env.sh\""
