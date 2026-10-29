#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
# Builds the Windows package from Linux: dist/bbport-windows/ and dist/bbport-windows.zip.
#   Bloodborne.exe            the launcher (launcher/windows: WPF in the system's Fluent theme, .NET
#                             9 inside, the settings of launcher/settings_schema.json); --play starts
#                             the game at once
#   bbport-run.exe            the game's runner (launcher/bbport_launcher_win.py --play, frozen by
#                             PyInstaller under Wine: players need no Python; it runs run.py)
#   bin/bb-probe.exe          the port (gpu/ and src/, cross-built with llvm-mingw against an MSYS2
#                             CLANG64 sysroot), bin/bb-gpu-capabilities.exe, and the DLLs they load
#   scripts/ patches/         the preparation scripts run.py runs inside Bloodborne.exe
#   README-Windows.txt
# Needs (out/win, prepared once; WIN overrides): llvm-mingw-*/ (toolchain), sysroot/clang64 (MSYS2
# packages, packaging/windows/README.md lists them), toolchain.cmake, pyprefix/ (a Wine prefix with
# Windows Python 3.12 + tkinter + PyInstaller at C:\Python312) and GE-Proton (PROTON overrides).
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
root=$PWD
win=${WIN:-$root/out/win}
proton=${PROTON:-$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-6-x86_64}
# PGO (Clang): WIN_PGO=generate builds an instrumented bb-probe.exe (run it with
# LLVM_PROFILE_FILE=...\bbport-%p.profraw, merge into out/win/pgo/bbport.profdata with
# llvm-profdata); use (the default when that file exists) builds with the profile; off.
pgo=${WIN_PGO:-}
[[ -z $pgo ]] && { [[ -f $win/pgo/bbport.profdata ]] && pgo=use || pgo=off; }
# WIN_LTO=1: link-time optimization (Clang ThinLTO through CMake IPO); its own build directory.
lto=${WIN_LTO:-0}
build=$win/gpu-package
[[ $pgo != off ]] && build=$win/gpu-package-pgo-$pgo
[[ $lto == 1 ]] && build=$build-lto
echo "Windows build: PGO $pgo, LTO $lto"
stage=$win/stage/bbport-windows

# 1. The port and its tool.
nix-shell -p cmake ninja pkg-config glslang python3 spirv-tools protobuf --run "
    set -e
    [[ -f '$build/build.ninja' ]] || cmake -S gpu -B '$build' -G Ninja -DCMAKE_TOOLCHAIN_FILE='$win/toolchain.cmake' \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBB_LTO=$( [[ $lto == 1 ]] && echo ON || echo OFF ) -DBB_PGO=$pgo -DBB_PGO_DIR='$win/pgo' > '$win/package-cmake.log'
    ninja -C '$build' bb-probe \$(grep -q 'build bbnet20.dll' '$build/build.ninja' && echo bbnet20) > '$win/package-build.log' 2>&1 || { tail -30 '$win/package-build.log'; exit 1; }"
cc=$(echo "$win"/llvm-mingw-*/bin/x86_64-w64-mingw32-clang)
"$cc" -std=c11 -O2 -Wall -Wextra -I"$win/sysroot/clang64/include" tools/gpu_capabilities.c \
    -L"$win/sysroot/clang64/lib" -lvulkan-1 -lSDL3 -o "$build/bb-gpu-capabilities.exe"

# 2. The launcher, frozen by the Windows Python in the Wine prefix (the project directory is
# visible there as Z:).
hidden=()
for module in argparse base64 bisect collections contextlib glob hashlib json math re shutil struct \
              tempfile zlib io runpy xml.etree.ElementTree ctypes.wintypes socket urllib.request \
              urllib.error http.server socketserver secrets uuid traceback datetime threading; do
    hidden+=(--hidden-import "$module")
done
z() { printf 'Z:%s' "${1//\//\\}"; }
icon=()
[[ -f packaging/bbport.ico ]] && icon=(--icon "$(z "$root/packaging/bbport.ico")")
rm -rf "$win/pyi-dist" "$win/pyi-work"
WINEPREFIX=$win/pyprefix WINEDEBUG=-all MANGOHUD=0 steam-run "$proton/files/bin/wine" 'C:\Python312\python.exe' \
    -m PyInstaller --noconfirm --clean --log-level WARN --console --name bbport-run \
    "${icon[@]}" --distpath "$(z "$win/pyi-dist")" \
    --workpath "$(z "$win/pyi-work")" --specpath "$(z "$win/pyi-work")" \
    --paths "$(z "$root")" --paths "$(z "$root/scripts")" --hidden-import run "${hidden[@]}" \
    "$(z "$root/launcher/bbport_launcher_win.py")" 2>&1 | grep -v MANGOHUD || true
[[ -f $win/pyi-dist/bbport-run/bbport-run.exe ]] || { echo 'PyInstaller made no bbport-run.exe' >&2; exit 1; }
# The window: WPF, cross-built (EnableWindowsTargeting), one self-contained file.
rm -rf "$win/wpf-dist"
nix-shell -p dotnet-sdk_9 --run "DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 dotnet publish launcher/windows \
    -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true \
    -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true \
    -p:DebugType=none -o '$win/wpf-dist'" > "$win/wpf-build.log" 2>&1 || { tail -20 "$win/wpf-build.log"; exit 1; }

# 3. The package.
rm -rf -- "$stage"
mkdir -p "$stage/bin" "$stage/launcher"
cp -r "$win/pyi-dist/bbport-run/." "$stage/"
cp "$win/wpf-dist/Bloodborne.exe" "$stage/"
strip=$(echo "$win"/llvm-mingw-*/bin/llvm-strip)
for exe in bb-probe.exe bb-gpu-capabilities.exe; do
    "$strip" --strip-debug -o "$stage/bin/$exe" "$build/$exe"
done
# The online module in its own folder with the libraries only it needs (protobuf, abseil, OpenSSL,
# miniupnpc): online\bbnet20.dll, replaceable as a whole (a newer BBSC build).
if [[ -f $build/bbnet20.dll ]]; then
    mkdir -p "$stage/online"
    "$strip" --strip-debug -o "$stage/online/bbnet20.dll" "$build/bbnet20.dll"
fi
# The DLLs they need, from the sysroot first, then llvm-mingw (Windows' own are left out): the
# MSYS2 DLLs import _Unwind_Resume from libc++.dll, which only MSYS2's libc++ exports (llvm-mingw's
# leaves it to libunwind.dll): with llvm-mingw's a real Windows 10 stopped at start, 0xC0000139.
objdump=$(echo "$win"/llvm-mingw-*/bin/llvm-objdump)
search=("$win/sysroot/clang64/bin" "$(echo "$win"/llvm-mingw-*/x86_64-w64-mingw32/bin)")
queue=("$stage/bin/bb-probe.exe" "$stage/bin/bb-gpu-capabilities.exe")
declare -A seen
while ((${#queue[@]})); do
    file=${queue[0]}; queue=("${queue[@]:1}")
    for dll in $("$objdump" -p "$file" | sed -n 's/^ *DLL Name: //p'); do
        key=${dll,,}
        [[ -n ${seen[$key]:-} ]] && continue
        seen[$key]=1
        for dir in "${search[@]}"; do
            found=$(find "$dir" -maxdepth 1 -iname "$dll" -print -quit)
            if [[ -n $found ]]; then cp "$found" "$stage/bin/"; queue+=("$found"); break; fi
        done
    done
done
# vulkan-1.dll is the driver's (the system's loader, installed with the GPU driver): never ours.
rm -f "$stage/bin/vulkan-1.dll"
# The online module's own libraries next to it (the ones bin\ has are loaded already; the
# module is loaded with its folder as the search path).
if [[ -f $stage/online/bbnet20.dll ]]; then
    queue=("$stage/online/bbnet20.dll")
    while ((${#queue[@]})); do
        file=${queue[0]}; queue=("${queue[@]:1}")
        for dll in $("$objdump" -p "$file" | sed -n 's/^ *DLL Name: //p'); do
            key=${dll,,}
            [[ -n ${seen[$key]:-} || $key == bb-probe.exe || $key == vulkan-1.dll ]] && continue
            seen[$key]=1
            for dir in "${search[@]}"; do
                found=$(find "$dir" -maxdepth 1 -iname "$dll" -print -quit)
                if [[ -n $found ]]; then cp "$found" "$stage/online/"; queue+=("$found"); break; fi
            done
        done
    done
fi
mkdir -p "$stage/scripts"
cp scripts/*.py "$stage/scripts/"
# BBSC's game server (run.py starts it when the server's own does not answer).
mkdir -p "$stage/scripts/bbsc_gameserver"
cp scripts/bbsc_gameserver/*.py scripts/bbsc_gameserver/*.xml scripts/bbsc_gameserver/*.txt \
   scripts/bbsc_gameserver/LICENSE* scripts/bbsc_gameserver/README.md "$stage/scripts/bbsc_gameserver/"
cp -r patches "$stage/"
cp run.py LICENSE NOTICE VERSION "$stage/"
# Shader recipes (docs/shader_recipes.md): no game code in it, only what each compile read.
[[ -f shader_recipes.bin ]] && cp shader_recipes.bin "$stage/"
# Image shapes (gpu/textures/image_spares.h): sizes and formats of the game's textures.
[[ -f image_shapes.txt ]] && cp image_shapes.txt "$stage/"
# FSR 4 v07 (MIT, built by Q2RTX from AMD's source; tools/fetch_fsr4_assets.sh), with the faster
# post passes (opt/) when tools/fsr4_optimize.sh made them: the AppImage brings it too.
[[ -d fsr4_shaders ]] && cp -r fsr4_shaders "$stage/"
cp launcher/bloodborne.ico launcher/bloodborne.png "$stage/launcher/" 2>/dev/null || true
cp packaging/windows/README-Windows.txt "$stage/"
# DLSS (NVIDIA RTX), optional: DLSS_SDK_ROOT=<github.com/NVIDIA/DLSS checkout with lib/Windows_x86_64>.
# The bridge (gpu/dlss_bridge) is built with llvm-mingw like the rest: NVIDIA's nvsdk_ngx_s.lib is
# MSVC code, the few pieces of Microsoft's C++ runtime it needs are in msvc_runtime_shim.cpp, and
# its default-library directives (LIBCMT, libcpmt, OLDNAMES) are met by empty archives.
if [[ -n ${DLSS_SDK_ROOT:-} && -f $DLSS_SDK_ROOT/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib ]]; then
    stubs=$win/dlss-stubs
    mkdir -p "$stubs"
    for lib in LIBCMT OLDNAMES libcpmt; do
        "$(echo "$win"/llvm-mingw-*/bin/llvm-ar)" rc "$stubs/lib$lib.a"
    done
    "$(echo "$win"/llvm-mingw-*/bin/x86_64-w64-mingw32-clang++)" -std=c++20 -O2 -shared -static \
        -DNOMINMAX -DWIN32_LEAN_AND_MEAN -I"$DLSS_SDK_ROOT/include" -I"$win/sysroot/clang64/include" \
        gpu/dlss_bridge/bbport_dlss_bridge.cpp gpu/dlss_bridge/msvc_runtime_shim.cpp \
        "$DLSS_SDK_ROOT/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib" -L"$stubs" -o "$stage/bin/bbport_dlss.dll"
    "$strip" --strip-all "$stage/bin/bbport_dlss.dll"
    cp "$DLSS_SDK_ROOT"/lib/Windows_x86_64/rel/nvngx_dlss.dll "$stage/bin/"
    [[ -f $DLSS_SDK_ROOT/lib/Windows_x86_64/rel/nvngx_dlssg.dll ]] && cp "$DLSS_SDK_ROOT"/lib/Windows_x86_64/rel/nvngx_dlssg.dll "$stage/bin/"
    cp "$DLSS_SDK_ROOT/LICENSE.txt" "$stage/NVIDIA-DLSS-LICENSE.txt"
    echo "DLSS: bin/bbport_dlss.dll, nvngx_dlss.dll$( [[ -f $stage/bin/nvngx_dlssg.dll ]] && echo ', nvngx_dlssg.dll')"
fi
# FSR 4.1.1 / 4.1.2 from the player's own AMD DLL (tools/fsr4cap/build_assets_win.py; the launcher
# runs bbport-run.exe --build-fsr411): the recorder and the translation tools, no Proton needed.
#   fsr4cap.exe, amdxc64.dll    ours (tools/fsr4cap; amdxc64.c is MIT)
#   d3d12.dll, d3d12core.dll    vkd3d-proton (LGPL-2.1), from GE-Proton: D3D12 over Vulkan, so that
#                               every GPU records what Proton records
#   dxgi.dll                    DXVK (zlib), from GE-Proton
#   dxil-spirv.exe              HansKristian-Work/dxil-spirv (MIT), out/win/fsr4cap-tools
#   spirv-as.exe, spirv-dis.exe SPIRV-Tools (Apache-2.0), from the MSYS2 sysroot
fsrtools=$win/fsr4cap-tools/build
if [[ -x $fsrtools/dxil-spirv.exe && -f $proton/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12.dll ]]; then
    fb=$stage/tools/fsr4cap/bin
    mkdir -p "$fb"
    cp tools/fsr4cap/*.py "$stage/tools/fsr4cap/"
    cc=$(echo "$win"/llvm-mingw-*/bin/x86_64-w64-mingw32-clang)
    "$cc" -std=c11 -O1 -Wall -Itools/fsr4cap/ffx/api/include -Itools/fsr4cap/ffx/upscalers/include \
        tools/fsr4cap/fsr4cap.c tools/fsr4cap/capture.c tools/fsr4cap/rootsig.c \
        -o "$fb/fsr4cap.exe" -ld3d12 -ldxguid -static
    "$cc" -std=c11 -O2 -Wall -shared tools/fsr4cap/amdxc64.c -o "$fb/amdxc64.dll" -static
    cp "$proton"/files/lib/wine/vkd3d-proton/x86_64-windows/{d3d12,d3d12core}.dll "$fb/"
    cp "$proton/files/lib/wine/dxvk/x86_64-windows/dxgi.dll" "$fb/"
    cp "$fsrtools/dxil-spirv.exe" "$fsrtools/libdxil-spirv-c-shared.dll" "$fb/"
    cp "$win/sysroot/clang64/bin/spirv-as.exe" "$win/sysroot/clang64/bin/spirv-dis.exe" "$fb/"
    # Their DLLs (the same search as bb-probe's above).
    queue=("$fb"/*.exe "$fb/libdxil-spirv-c-shared.dll")
    unset seen; declare -A seen
    while ((${#queue[@]})); do
        file=${queue[0]}; queue=("${queue[@]:1}")
        for dll in $("$objdump" -p "$file" | sed -n 's/^ *DLL Name: //p'); do
            key=${dll,,}
            [[ -n ${seen[$key]:-} || -f $fb/$dll ]] && continue
            seen[$key]=1
            for dir in "$fsrtools" "${search[@]}"; do
                found=$(find "$dir" -maxdepth 1 -iname "$dll" -print -quit)
                if [[ -n $found ]]; then cp "$found" "$fb/"; queue+=("$found"); break; fi
            done
        done
    done
    rm -f "$fb/vulkan-1.dll"
    echo "FSR 4.1.1 recording tools: tools/fsr4cap/bin ($(ls "$fb" | wc -l) files)"
else
    echo "FSR 4.1.1 recording tools left out: no $fsrtools/dxil-spirv.exe or no vkd3d-proton in $proton" >&2
fi
mkdir -p dist
rm -rf dist/bbport-windows && cp -r "$stage" dist/bbport-windows
(cd dist && rm -f bbport-windows.zip && nix-shell -p zip --run 'zip -qr bbport-windows.zip bbport-windows')
echo "Built dist/bbport-windows/ and dist/bbport-windows.zip ($(du -sh dist/bbport-windows.zip | cut -f1))"
