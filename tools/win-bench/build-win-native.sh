# Native Windows build of bb-probe.exe in MSYS2 CLANG64 (recipe from the 930MX laptop session).
# Run: set MSYSTEM=CLANG64, CHERE_INVOKING=1; C:\msys64\usr\bin\bash.exe -l E:/bloodborne/bbport/out/win-native/build.sh
set -e
cd /e/bloodborne/bbport
b=out/win-native
if [[ ! -f $b/build.ninja ]]; then
  cmake -S gpu -B $b -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBB_LTO=OFF -DBB_PGO=off -DBB_NET=OFF \
    -DCMAKE_CXX_FLAGS="-I$PWD/$b/deps/xbyak-7.10" > $b/cmake.log 2>&1 || { tail -40 $b/cmake.log; exit 1; }
fi
start=$(date +%s)
ninja -C $b bb-probe > $b/build.log 2>&1 || { grep -E 'error|FAILED' $b/build.log | head -40; tail -20 $b/build.log; exit 1; }
echo "build ok in $(( $(date +%s) - start )) s"
ls -la $b/bb-probe.exe
