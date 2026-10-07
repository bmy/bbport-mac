#!/usr/bin/env bash
# tools/macos/build_gpu_native.sh: builds the native GPU process for Apple Silicon (arm64):
# out/gpu-arm64/bb-gpu and its libbbgpu.dylib (docs/macos-native-gpu.md). The game runs with it
# when BB_NATIVE_GPU=1 is set; the x86-64 build is unaffected.
#   BB_ARCH=arm64 bash tools/macos/setup_deps.sh     # once: native libraries into deps-arm64
#   bash tools/macos/build_gpu_native.sh             # out/gpu-arm64/bb-gpu
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
DEPS=${BB_DEPS_ARM64:-$PWD/deps-arm64}
[[ -f $DEPS/env.sh ]] || { echo "STOP: $DEPS/env.sh missing; run: BB_ARCH=arm64 bash tools/macos/setup_deps.sh" >&2; exit 1; }
# shellcheck disable=SC1091
source "$DEPS/env.sh"
BREW_PREFIX=$( (command -v brew >/dev/null && brew --prefix) || echo /opt/homebrew)
export PATH="$BREW_PREFIX/bin:$PATH"
JOBS=$(sysctl -n hw.ncpu)
mkdir -p out

if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ]]; then
    git submodule update --init --recursive
fi
for patch in gpu/patches/fsr-vulkan/*.patch; do
    if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
    fi
done

echo "=== GPU library (arm64)"
cmake -S gpu -B out/gpu-arm64 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_PROCESSOR=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
    -DCMAKE_PREFIX_PATH="$DEPS;$BREW_PREFIX" -DBB_LTO=OFF -DBB_PGO=off > out/gpu-arm64-configure.log 2>&1 ||
    { tail -40 out/gpu-arm64-configure.log >&2; echo "STOP: configure failed (out/gpu-arm64-configure.log)" >&2; exit 1; }
if ! ninja -C out/gpu-arm64 -k 0 -j "$JOBS" bbgpu bb-gpu > out/gpu-arm64-build.log 2>&1; then
    grep -A1 -E ': (fatal )?error:' out/gpu-arm64-build.log | grep -v '^--$' | awk '!seen[$0]++' > out/gpu-arm64-errors.txt
    head -80 out/gpu-arm64-errors.txt >&2
    echo "STOP: arm64 GPU library build failed: $(grep -cE ': (fatal )?error:' out/gpu-arm64-errors.txt) distinct errors" \
         "(all in out/gpu-arm64-errors.txt, full log out/gpu-arm64-build.log)" >&2
    exit 1
fi
# The library takes the runtime's functions from the executable at load time (flat namespace):
# one missing there would stop bb-gpu before main.
missing=$(comm -23 \
    <(nm -m out/gpu-arm64/libbbgpu.dylib | sed -n 's/.* \(_[^ ]*\) (dynamically looked up).*/\1/p' | sort -u) \
    <(nm -gU out/gpu-arm64/bb-gpu | awk '{print $3}' | sort -u))
if [[ -n $missing ]]; then
    echo "STOP: bb-gpu does not provide what libbbgpu imports from it:" $missing >&2
    exit 1
fi
echo "Built: out/gpu-arm64/bb-gpu ($(lipo -archs out/gpu-arm64/bb-gpu)) and libbbgpu.dylib"
