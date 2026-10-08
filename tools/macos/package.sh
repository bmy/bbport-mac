#!/usr/bin/env bash
# tools/macos/package.sh: the release download. Puts the launcher, the game engine (bb-probe, the
# GPU library, run.sh and its Python scripts), the x86-64 libraries they load and the KosmicKrisp
# Vulkan driver into one self-contained bbport.app, and zips it with a short read-me:
#   bash tools/macos/setup_deps.sh && bash tools/macos/build.sh
#   bash tools/macos/package.sh 0.4-mac.1        # dist/bbport-0.4-mac.1.zip (+ .sha256)
# The app needs no checkout, Homebrew or build tools to run; only Rosetta 2 and Python 3 (the
# Command Line Tools' python3, or Homebrew's). It writes nothing into itself: settings, generated
# files, logs and saves go to ~/Library/Application Support/bbport (tools/macos/run.sh, .packaged).
# No game files and no game artwork are included.
# GitHub Actions runs this for every release tag (.github/workflows/release.yml).
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
repo=$PWD
die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
trap 'printf "STOP: failed at line %s of tools/macos/package.sh\n" "$LINENO" >&2' ERR
[[ $(uname -s) == Darwin ]] || die "run this on a Mac"

version=${1:-${BB_VERSION:-}}
if [[ -z $version ]]; then
    version=$(git describe --tags --always 2>/dev/null || echo dev)
fi
[[ $version =~ ^[A-Za-z0-9._-]+$ ]] || die "version '$version': letters, digits, dot, dash or underscore only"
DEPS=${BB_DEPS:-$repo/deps-x86_64}
kk=$DEPS/lib/kosmickrisp
for file in out/bb-probe out/bb-gpu-capabilities out/gpu/libbbgpu.dylib \
            "$kk/libvulkan_kosmickrisp.dylib" "$kk/kosmickrisp_mesa_icd.json"; do
    [[ -f $file ]] || die "$file missing: run tools/macos/setup_deps.sh and tools/macos/build.sh first"
done
command -v python3 >/dev/null || die "python3 missing"

name=bbport-$version
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
python3 - "$kk/kosmickrisp_mesa_icd.json" "$engine/lib/kosmickrisp/kosmickrisp_mesa_icd.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
manifest["ICD"]["library_path"] = "./libvulkan_kosmickrisp.dylib"
with open(sys.argv[2], "w") as out:
    json.dump(manifest, out, indent=4)
    out.write("\n")
PY
touch "$engine/.packaged"
echo "$version" > "$engine/.version"
chmod -R u+w "$engine"

echo "=== Libraries"
# Every library outside macOS itself is copied into lib/ and loaded through @rpath, so the app
# runs without deps-x86_64/ and without DYLD_LIBRARY_PATH (which macOS drops when bash starts).
loads() {  # the libraries a Mach-O file loads
    otool -l "$1" | awk '/cmd LC_(LOAD|LOAD_WEAK|REEXPORT)_DYLIB$/ { getline; getline; print $2 }'
}
rpaths() {
    otool -l "$1" | awk '/cmd LC_RPATH$/ { getline; getline; print $2 }'
}
search=("$DEPS/lib" "$repo/out/gpu" "$kk")
find_library() {  # find_library <load command path>: the file to copy, or nothing
    local ref=$1 base dir
    if [[ $ref == /* ]]; then
        [[ -f $ref ]] && { printf '%s\n' "$ref"; return; }
    fi
    base=${ref##*/}
    for dir in "${search[@]}"; do
        [[ -f $dir/$base ]] && { printf '%s\n' "$dir/$base"; return; }
    done
    return 0
}
queue=("$engine/bin/bb-probe" "$engine/bin/bb-gpu-capabilities" "$engine/bin/gpu/libbbgpu.dylib"
       "$engine/lib/kosmickrisp/libvulkan_kosmickrisp.dylib")
done_files=" "
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
            target=$engine/bin/gpu/libbbgpu.dylib
        elif [[ $base == libvulkan_kosmickrisp.dylib ]]; then
            target=$engine/lib/kosmickrisp/$base
        else
            target=$engine/lib/$base
        fi
        if [[ ! -f $target ]]; then
            source_file=$(find_library "$ref")
            [[ -n $source_file ]] || die "$(basename "$file") loads $ref, which was not found in ${search[*]}"
            cp -L "$source_file" "$target"
            chmod u+w "$target"
            echo "  $base  ($source_file)"
            queue+=("$target")
        fi
        if [[ $ref != "@rpath/$base" ]]; then
            # (stderr: the "will invalidate the code signature" notice; signed again below)
            install_name_tool -change "$ref" "@rpath/$base" "$file" 2>/dev/null
        fi
    done < <(loads "$file")
done

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
for lib in "$engine"/lib/*.dylib "$engine"/lib/kosmickrisp/*.dylib "$engine"/bin/gpu/*.dylib; do
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

echo "=== Check"
# tools/macos/run.sh names the Vulkan loader by path (BB_VULKAN_LIBRARY).
[[ -f $engine/lib/libvulkan.1.dylib ]] || die "lib/libvulkan.1.dylib missing from the app: $(ls "$engine/lib")"
# Every load resolves inside the app or to macOS, and the engine is x86-64 throughout.
bad=0
machos=()
while IFS= read -r -d '' file; do
    if file -b "$file" | grep -q 'Mach-O'; then
        machos+=("$file")
    fi
done < <(find "$engine" -type f -print0)
for file in "${machos[@]}"; do
    archs=$(lipo -archs "$file")
    if [[ $archs != *x86_64* ]]; then
        echo "  not x86-64: ${file#"$app"/} ($archs)"
        bad=1
    fi
    while IFS= read -r ref; do
        case $ref in
            /usr/lib/*|/System/*) ;;
            @rpath/*)
                base=${ref#@rpath/}
                if [[ ! -f $engine/lib/$base && ! -f $engine/bin/gpu/$base && ! -f $engine/lib/kosmickrisp/$base ]]; then
                    echo "  ${file#"$app"/}: $ref is not in the app"
                    bad=1
                fi ;;
            *) echo "  ${file#"$app"/}: loads $ref from outside the app"; bad=1 ;;
        esac
    done < <(loads "$file")
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
