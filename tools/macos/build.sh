#!/usr/bin/env bash
# tools/macos/build.sh: builds the port for macOS as x86-64 (runs under Rosetta 2).
# Needs tools/macos/setup_deps.sh first. Output: out/bb-probe and out/gpu/libbbgpu.dylib.
#   bash tools/macos/build.sh            # loader + GPU library
#   BB_SKIP_GPU=1 bash tools/macos/build.sh   # loader only (GPU library already built)
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
DEPS=${BB_DEPS:-$PWD/deps-x86_64}
[[ -f $DEPS/env.sh ]] || { echo "STOP: $DEPS/env.sh missing; run tools/macos/setup_deps.sh" >&2; exit 1; }
# shellcheck disable=SC1091
source "$DEPS/env.sh"
BREW_PREFIX=$( (command -v brew >/dev/null && brew --prefix) || echo /opt/homebrew)
export PATH="$BREW_PREFIX/bin:$PATH"
JOBS=$(sysctl -n hw.ncpu)
mkdir -p out
# update.sh's record of what out/ was built from: no longer true from here on. The native GPU
# process's failure mark is set again below if it still does not build.
rm -f out/.bb-built out/.bb-native-failed
# ccache (from setup_deps.sh): files compiled before, on any branch, come from its cache in
# ~/Library/Caches/ccache, at most 10 GB (oldest out first). BB_CCACHE=0: off.
# The compiler is checked by its version output: Apple's /usr/bin/clang is a stub that a
# Command Line Tools update leaves as it is. Clear the cache (ccache -C) after building code you
# do not trust: what it compiled could be served to later builds.
LAUNCHER=()
if [[ ${BB_CCACHE:-1} != 0 ]] && ccache=$(command -v ccache); then
    export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-10G} CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK:-'%compiler% -v'}
    LAUNCHER=(-DCMAKE_C_COMPILER_LAUNCHER="$ccache" -DCMAKE_CXX_COMPILER_LAUNCHER="$ccache")
else
    LAUNCHER=(-UCMAKE_C_COMPILER_LAUNCHER -UCMAKE_CXX_COMPILER_LAUNCHER)
fi
ARCH=(-arch x86_64 -mmacosx-version-min="$MACOSX_DEPLOYMENT_TARGET")

# Submodules and this port's FSR-Vulkan changes (as build.sh does on Linux).
if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ||
      ! -f third_party/LibAtrac9/C/src/libatrac9.h ]]; then
    git submodule update --init --recursive
fi
for patch in gpu/patches/fsr-vulkan/*.patch; do
    if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
    fi
done

if [[ -z ${BB_SKIP_GPU:-} ]]; then
    echo "=== GPU library (x86-64)"
    # Our x86-64 prefix first; Homebrew only supplies header-only packages (Boost, magic_enum,
    # robin-map, VMA). LTO/PGO stay off until the port runs. BB_NET=OFF: no online module (0.51's
    # gpu/bbnet) on the Mac yet; the runtime plays offline without it.
    # No OpenSSL (0.7's trophy names): the app bundles only libraries this build makes, and
    # CMake would find Homebrew's arm64 libcrypto (trophy unlocks are then shown by number).
    cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBB_NET=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
        -DCMAKE_PREFIX_PATH="$DEPS;$BREW_PREFIX" -DBB_LTO=OFF -DBB_PGO=off "${LAUNCHER[@]}" \
        > out/gpu-configure.log 2>&1 ||
        { tail -40 out/gpu-configure.log >&2; echo "STOP: GPU library configure failed (out/gpu-configure.log)" >&2; exit 1; }
    # -k 0: keep compiling after failures, so one run reports every file's errors.
    if ! ninja -C out/gpu -k 0 -j "$JOBS" bbgpu > out/gpu-build.log 2>&1; then
        # Each distinct error once, with the line after it (usually the template/include context).
        grep -A1 -E ': (fatal )?error:' out/gpu-build.log | grep -v '^--$' | awk '!seen[$0]++' > out/gpu-errors.txt
        head -80 out/gpu-errors.txt >&2
        echo "STOP: GPU library build failed: $(grep -cE ': (fatal )?error:' out/gpu-errors.txt) distinct errors" \
             "(all in out/gpu-errors.txt, full log out/gpu-build.log)" >&2
        exit 1
    fi
fi

echo "=== LibAtrac9 (x86-64)"
if [[ ! -f out/libatrac9.a || -n $(find third_party/LibAtrac9/C/src -newer out/libatrac9.a -name '*.c') ]]; then
    rm -rf out/atrac9 && mkdir -p out/atrac9
    for source in third_party/LibAtrac9/C/src/*.c; do
        clang "${ARCH[@]}" -std=c99 -O2 -g -w -c "$source" -o "out/atrac9/$(basename "${source%.c}").o"
    done
    ar rcs out/libatrac9.a out/atrac9/*.o
fi

echo "=== Loader (x86-64)"
read -r -a pkg_cflags <<< "$(pkg-config --cflags vulkan sdl3)"
read -r -a pkg_libs <<< "$(pkg-config --libs vulkan sdl3)"
gpu_lib=out/gpu/libbbgpu.dylib
[[ -f $gpu_lib ]] || { echo "STOP: $gpu_lib missing" >&2; exit 1; }
if ! clang "${ARCH[@]}" -std=c11 -O2 -g -Wall -Wextra -pthread "${pkg_cflags[@]}" -I. -Isrc \
        src/probe.c src/runtime*.c src/low_heap.c src/vulkan_smoke.c out/libatrac9.a -lm \
        -Lout/gpu -lbbgpu -Wl,-rpath,@loader_path/gpu -Wl,-export_dynamic \
        "${pkg_libs[@]}" -o out/bb-probe 2> out/loader-build.log; then
    grep -E 'error' out/loader-build.log | head -40 >&2
    echo "STOP: loader build failed (full log: out/loader-build.log)" >&2
    exit 1
fi
echo "loader warnings: $(grep -c 'warning:' out/loader-build.log || true) (out/loader-build.log)"
clang "${ARCH[@]}" -std=c11 -O2 -Wall -Wextra tools/gpu_capabilities.c "${pkg_libs[@]}" "${pkg_cflags[@]}" \
    -o out/bb-gpu-capabilities
echo "Built: $(lipo -archs out/bb-probe) out/bb-probe, $(lipo -archs "$gpu_lib") $gpu_lib"
# The native GPU process (docs/macos-native-gpu.md, used by default once built), once its arm64
# libraries exist (BB_ARCH=arm64 bash tools/macos/setup_deps.sh). A failure there does not stop
# this build: the game still runs as before.
if [[ -z ${BB_SKIP_GPU:-} && ! -f deps-arm64/env.sh ]]; then
    # Without its libraries an older bb-gpu would not run, or not match the game built here.
    rm -f out/gpu-arm64/bb-gpu
elif [[ -z ${BB_SKIP_GPU:-} ]]; then
    echo "=== Native GPU process (arm64)"
    # out/.bb-native-failed: update.sh then builds again next time (it records nothing).
    # A failed build leaves no bb-gpu: an older one would not match the game built here.
    if ! bash tools/macos/build_gpu_native.sh; then
        touch out/.bb-native-failed
        rm -f out/gpu-arm64/bb-gpu
        echo "warning: native GPU process not built (out/gpu-arm64-errors.txt)" >&2
    fi
fi
