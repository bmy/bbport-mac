#!/usr/bin/env bash
# tools/macos/package.sh: the release download. Puts the launcher, the game engine (bb-probe, the
# GPU library, run.sh and its Python scripts), the x86-64 libraries they load and the KosmicKrisp
# Vulkan driver into one self-contained bbport.app, and zips it with a short read-me. On the
# native branches also the native GPU process (bin/gpu-arm64: bb-gpu and its GPU library) with
# its own arm64 libraries and driver (deps-arm64/lib), which the game then uses by default:
#   bash tools/macos/setup_deps.sh && bash tools/macos/build.sh
#   bash tools/macos/package.sh 0.4-v1           # dist/bbport-mac-0.4-v1.zip (+ .sha256)
# Versions: <upstream bbport version>-v<our build for it>: 0.4-v1, 0.4-v2, 0.4.1-v1, 0.5-v1.
# The app needs no checkout, Homebrew or build tools to run; only Rosetta 2 and Python 3 (the
# Command Line Tools' python3, or Homebrew's). It writes nothing into itself: settings, generated
# files, logs and saves go to ~/Library/Application Support/bbport (tools/macos/run.sh, .packaged).
# No game files and no game artwork are included.
# GitHub Actions runs this for every release tag (.github/workflows/release.yml).
set -Eeuo pipefail # -E: the ERR trap below also reports failures inside functions
cd -- "$(dirname -- "$0")/../.."
repo=$PWD
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
trap 'printf "STOP: failed at line %s of tools/macos/package.sh\n" "$LINENO" >&2' ERR
[[ $(uname -s) == Darwin ]] || die "run this on a Mac"

version=${1:-${BB_VERSION:-}}
if [[ -z $version ]]; then
    version=$(git describe --tags --always 2>/dev/null || echo dev)
fi
[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "version '$version': a letter or digit, then letters, digits, dot, dash or underscore"
DEPS=${BB_DEPS:-$repo/deps-x86_64}
kk=$DEPS/lib/kosmickrisp
for file in out/bb-probe out/bb-gpu-capabilities out/gpu/libbbgpu.dylib \
            "$kk/libvulkan_kosmickrisp.dylib" "$kk/kosmickrisp_mesa_icd.json"; do
    [[ -f $file ]] || die "$file missing: run tools/macos/setup_deps.sh and tools/macos/build.sh first"
done
# The native GPU process: a branch that builds it ships it (the game's default where it is).
native=
if [[ -f tools/macos/build_gpu_native.sh ]]; then
    native=yes
    DEPS_ARM64=${BB_DEPS_ARM64:-$repo/deps-arm64}
    kk_arm64=$DEPS_ARM64/lib/kosmickrisp
    for file in out/gpu-arm64/bb-gpu out/gpu-arm64/libbbgpu.dylib "$DEPS_ARM64/lib/libvulkan.1.dylib" \
                "$kk_arm64/libvulkan_kosmickrisp.dylib" "$kk_arm64/kosmickrisp_mesa_icd.json"; do
        [[ -f $file ]] || die "$file missing: run BB_ARCH=arm64 tools/macos/setup_deps.sh and tools/macos/build_gpu_native.sh first"
    done
fi
command -v python3 >/dev/null || die "python3 missing"

name=bbport-mac-$version
stage=$repo/out/package
top=$stage/$name
app=$top/bbport.app
rm -rf "$stage"
mkdir -p "$top"

echo "=== Launcher"
BB_APP="$app" BB_RELEASE="$version" bash tools/macos/build_launcher.sh

echo "=== Game engine"
engine=$app/Contents/Resources/bbport
mkdir -p "$engine/bin/gpu" "$engine/lib/kosmickrisp" "$engine/scripts" "$engine/tools/macos"
cp run.sh LICENSE "$engine/"
cp tools/macos/run.sh "$engine/tools/macos/"
cp scripts/*.py "$engine/scripts/"
cp -R patches "$engine/"
cp out/bb-probe out/bb-gpu-capabilities "$engine/bin/"
cp out/gpu/libbbgpu.dylib "$engine/bin/gpu/"
cp "$kk/libvulkan_kosmickrisp.dylib" "$engine/lib/kosmickrisp/"
# The driver manifest names the library next to it (a relative path is relative to the manifest).
manifest() {  # manifest <source json> <target json>
    python3 - "$1" "$2" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
manifest["ICD"]["library_path"] = "./libvulkan_kosmickrisp.dylib"
with open(sys.argv[2], "w") as out:
    json.dump(manifest, out, indent=4)
    out.write("\n")
PY
}
manifest "$kk/kosmickrisp_mesa_icd.json" "$engine/lib/kosmickrisp/kosmickrisp_mesa_icd.json"
# The native GPU process, laid out as in a checkout around bb-probe: bin/gpu-arm64/bb-gpu, and
# its libraries in deps-arm64/lib (remote_front.cpp looks there, and sets the Vulkan loader and
# driver from it for bb-gpu).
engine_arm64=$engine/deps-arm64/lib
if [[ -n $native ]]; then
    mkdir -p "$engine/bin/gpu-arm64" "$engine_arm64/kosmickrisp"
    cp out/gpu-arm64/bb-gpu out/gpu-arm64/libbbgpu.dylib "$engine/bin/gpu-arm64/"
    cp -L "$DEPS_ARM64/lib/libvulkan.1.dylib" "$engine_arm64/"
    cp "$kk_arm64/libvulkan_kosmickrisp.dylib" "$engine_arm64/kosmickrisp/"
    manifest "$kk_arm64/kosmickrisp_mesa_icd.json" "$engine_arm64/kosmickrisp/kosmickrisp_mesa_icd.json"
    # Licences of what is linked statically into the native GPU core (SPIRV-Cross, Apache-2.0,
    # on the Metal command path's branches).
    if compgen -G "$DEPS_ARM64/share/licenses/*" > /dev/null; then
        mkdir -p "$engine/licenses"
        cp "$DEPS_ARM64"/share/licenses/* "$engine/licenses/"
    fi
fi
touch "$engine/.packaged"
echo "$version" > "$engine/.version"
chmod -R u+w "$engine"

echo "=== Libraries"
# Every library outside macOS itself is copied into lib/ and loaded through @rpath, so the app
# runs without deps-x86_64/ and without DYLD_LIBRARY_PATH (which macOS drops when bash starts).
loads() {  # the libraries a Mach-O file loads
    otool -l "$1" | awk '/cmd LC_(LOAD|LOAD_WEAK|REEXPORT|LOAD_UPWARD|LAZY_LOAD)_DYLIB$/ { getline; getline; print $2 }'
}
rpaths() {
    otool -l "$1" | awk '/cmd LC_RPATH$/ { getline; getline; print $2 }'
}
find_library() {  # find_library <load command path> <search dir>...: the file to copy, or nothing
    local ref=$1 base dir
    shift
    if [[ $ref == /* ]]; then
        [[ -f $ref ]] && { printf '%s\n' "$ref"; return; }
    fi
    base=${ref##*/}
    for dir in "$@"; do
        [[ -f $dir/$base ]] && { printf '%s\n' "$dir/$base"; return; }
    done
    return 0
}
# bundle <lib dir> <GPU library dir> <search dirs, ':'-separated> <files...>: copies what the
# files load (and what that loads) from the search dirs into the app, loaded through @rpath.
bundle() {
    local libdir=$1 gpudir=$2 file ref base target source_file dir done_files=" "
    local -a search queue
    IFS=: read -r -a search <<< "$3"
    shift 3
    queue=("$@")
    while (( ${#queue[@]} )); do
        file=${queue[0]}
        queue=("${queue[@]:1}")
        [[ $done_files == *" $file "* ]] && continue
        done_files+="$file "
        while IFS= read -r ref; do
            case $ref in
                /usr/lib/*|/System/*) continue ;;
            esac
            base=${ref##*/}
            if [[ $base == libbbgpu.dylib ]]; then
                target=$gpudir/libbbgpu.dylib
            elif [[ $base == libvulkan_kosmickrisp.dylib ]]; then
                target=$libdir/kosmickrisp/$base
            else
                target=$libdir/$base
            fi
            if [[ ! -f $target ]]; then
                source_file=$(find_library "$ref" "${search[@]}")
                [[ -n $source_file ]] || die "$(basename "$file") loads $ref, which was not found in ${search[*]}"
                # Only what this build made goes in: not, say, a Homebrew library a dependency
                # happened to link (unpinned, and outside the review of setup_deps.sh).
                local under=
                for dir in "${search[@]}"; do
                    [[ $source_file == "$dir"/* ]] && under=yes
                done
                [[ -n $under ]] || die "$(basename "$file") loads $ref from outside this build's libraries ($source_file)"
                cp -L "$source_file" "$target"
                chmod u+w "$target"
                echo "  $base  (${source_file/#$HOME/~})"
                queue+=("$target")
            fi
            if [[ $ref != "@rpath/$base" ]]; then
                # (stderr: the "will invalidate the code signature" notice; signed again below)
                install_name_tool -change "$ref" "@rpath/$base" "$file" 2>/dev/null
            fi
        done < <(loads "$file")
    done
}
bundle "$engine/lib" "$engine/bin/gpu" "$DEPS/lib:$repo/out/gpu:$kk" \
    "$engine/bin/bb-probe" "$engine/bin/bb-gpu-capabilities" "$engine/bin/gpu/libbbgpu.dylib" \
    "$engine/lib/kosmickrisp/libvulkan_kosmickrisp.dylib"
if [[ -n $native ]]; then
    echo "  (arm64, for the native GPU process)"
    bundle "$engine_arm64" "$engine/bin/gpu-arm64" "$DEPS_ARM64/lib:$repo/out/gpu-arm64:$kk_arm64" \
        "$engine/bin/gpu-arm64/bb-gpu" "$engine/bin/gpu-arm64/libbbgpu.dylib" \
        "$engine_arm64/libvulkan.1.dylib" "$engine_arm64/kosmickrisp/libvulkan_kosmickrisp.dylib"
fi

# Install names and search paths. Absolute search paths (the build machine's deps-x86_64) go.
fix_rpaths() {  # fix_rpaths <file> <rpath to add>...
    local file=$1 old
    shift
    while IFS= read -r old; do
        if [[ $old == /* ]]; then
            install_name_tool -delete_rpath "$old" "$file" 2>/dev/null
        fi
    done < <(rpaths "$file")
    for new in "$@"; do
        if ! rpaths "$file" | grep -qxF "$new"; then
            install_name_tool -add_rpath "$new" "$file" 2>/dev/null
        fi
    done
}
for lib in "$engine"/lib/*.dylib "$engine"/lib/kosmickrisp/*.dylib "$engine"/bin/gpu/*.dylib \
           "$engine_arm64"/*.dylib "$engine_arm64"/kosmickrisp/*.dylib "$engine"/bin/gpu-arm64/*.dylib; do
    if [[ -f $lib ]]; then
        install_name_tool -id "@rpath/$(basename "$lib")" "$lib" 2>/dev/null
    fi
done
fix_rpaths "$engine/bin/bb-probe" @loader_path/gpu @loader_path/../lib
fix_rpaths "$engine/bin/bb-gpu-capabilities" @loader_path/../lib
fix_rpaths "$engine/bin/gpu/libbbgpu.dylib" @loader_path/../../lib
for lib in "$engine"/lib/*.dylib; do
    if [[ -f $lib ]]; then
        fix_rpaths "$lib" @loader_path
    fi
done
fix_rpaths "$engine/lib/kosmickrisp/libvulkan_kosmickrisp.dylib" @loader_path/..
if [[ -n $native ]]; then
    fix_rpaths "$engine/bin/gpu-arm64/bb-gpu" @loader_path @loader_path/../../deps-arm64/lib
    fix_rpaths "$engine/bin/gpu-arm64/libbbgpu.dylib" @loader_path/../../deps-arm64/lib
    for lib in "$engine_arm64"/*.dylib; do
        if [[ -f $lib ]]; then
            fix_rpaths "$lib" @loader_path
        fi
    done
    fix_rpaths "$engine_arm64/kosmickrisp/libvulkan_kosmickrisp.dylib" @loader_path/..
fi

echo "=== Check"
# tools/macos/run.sh names the Vulkan loader by path (BB_VULKAN_LIBRARY).
[[ -f $engine/lib/libvulkan.1.dylib ]] || die "lib/libvulkan.1.dylib missing from the app: $(ls "$engine/lib")"
# Every load resolves inside the app or to macOS. The engine is x86-64, but for the native GPU
# process and its libraries (arm64), which load only each other.
bad=0
machos=()
while IFS= read -r -d '' file; do
    if file -b "$file" | grep -q 'Mach-O'; then
        machos+=("$file")
    fi
done < <(find "$engine" -type f -print0)
for file in "${machos[@]}"; do
    archs=$(lipo -archs "$file")
    case $file in
        "$engine"/bin/gpu-arm64/*|"$engine"/deps-arm64/*)
            arch=arm64 dirs=("$engine_arm64" "$engine/bin/gpu-arm64" "$engine_arm64/kosmickrisp") ;;
        *)
            arch=x86_64 dirs=("$engine/lib" "$engine/bin/gpu" "$engine/lib/kosmickrisp") ;;
    esac
    if [[ " $archs " != *" $arch "* ]]; then
        echo "  not $arch: ${file#"$app"/} ($archs)"
        bad=1
    fi
    while IFS= read -r ref; do
        case $ref in
            /usr/lib/*|/System/*) ;;
            @rpath/*)
                base=${ref#@rpath/}
                if [[ ! -f ${dirs[0]}/$base && ! -f ${dirs[1]}/$base && ! -f ${dirs[2]}/$base ]]; then
                    echo "  ${file#"$app"/}: $ref is not in the app"
                    bad=1
                fi ;;
            *) echo "  ${file#"$app"/}: loads $ref from outside the app"; bad=1 ;;
        esac
    done < <(loads "$file")
    # Search paths stay inside the app: relative to the file, never absolute.
    while IFS= read -r path; do
        case $path in
            @loader_path*|@executable_path*) ;;
            *) echo "  ${file#"$app"/}: searches $path"; bad=1 ;;
        esac
    done < <(rpaths "$file")
done
(( bad == 0 )) || die "the app is not self-contained (above)"
echo "  ${#machos[@]} engine files, every library inside the app or macOS"

echo "=== Signature (ad hoc)"
# install_name_tool invalidated the signatures. Inner code first, then the app.
for file in "${machos[@]}"; do
    codesign --force --sign - "$file"
done
codesign --force --sign - "$app"
codesign --verify "$app"

echo "=== Smoke test"
# Started with an empty environment, as from the app: dyld stops bb-probe before its usage line
# if any library doesn't resolve inside the app.
run_bare() {  # run_bare <seconds> <command...>: no inherited environment, killed after <seconds>
    local seconds=$1
    shift
    env -i HOME="$HOME" PATH=/usr/bin:/bin VK_DRIVER_FILES="$engine/lib/kosmickrisp/kosmickrisp_mesa_icd.json" \
        BB_VULKAN_LIBRARY="$engine/lib/libvulkan.1.dylib" perl -e 'alarm shift; exec @ARGV' "$seconds" "$@" 2>&1
}
probe_output=$(run_bare 30 "$engine/bin/bb-probe" || true)
printf '%s\n' "$probe_output" | head -5
[[ $probe_output == *Usage:* ]] || die "bb-probe does not start from the app (above)"
if [[ -n $native ]]; then
    # bb-gpu says the game starts it once dyld has loaded everything it needs.
    gpu_output=$(run_bare 30 "$engine/bin/gpu-arm64/bb-gpu" || true)
    printf '%s\n' "$gpu_output" | head -3
    [[ $gpu_output == *"the game process starts this"* ]] || die "bb-gpu does not start from the app (above)"
fi
# The Vulkan loader finds KosmicKrisp through the app's manifest. Informational: a build machine
# may have no GPU the driver accepts.
vulkan_output=$(run_bare 60 "$engine/bin/bb-probe" --vulkan-only || true)
printf '%s\n' "$vulkan_output" | tail -8
report=$stage/report.txt
{
    echo "bbport $version, $(sw_vers -productVersion), $(uname -m)"
    echo
    echo "bb-probe without arguments:"
    printf '%s\n' "$probe_output" | head -5
    echo
    echo "bb-probe --vulkan-only:"
    printf '%s\n' "$vulkan_output" | tail -12
    echo
    echo "Files:"
    (cd "$app/Contents" && find . -type f -print0 | xargs -0 ls -l | awk '{ printf "%10d  %s\n", $5, $9 }' | sort -k2)
} > "$report"

echo "=== Archive"
cp docs/release/READ-ME-FIRST.txt "$top/Read Me First.txt"
cp LICENSE "$top/LICENSE.txt"
mkdir -p dist
zip=$repo/dist/$name.zip
rm -f "$zip" "$zip.sha256"
# ditto keeps the app's symlinks, permissions and signature intact (zip -r doesn't always).
ditto -c -k --sequesterRsrc --keepParent "$top" "$zip"
(cd dist && shasum -a 256 "$name.zip" > "$name.zip.sha256")
echo
echo "Built dist/$name.zip ($(du -h "$zip" | cut -f1)):"
echo "  $name/bbport.app, Read Me First.txt, LICENSE.txt"
cat "dist/$name.zip.sha256"
