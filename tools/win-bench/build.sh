set -e
cd /e/bloodborne/bbport/out/win-native/bench
clang -O2 vp_bench.c -o vp_bench.exe -lvulkan-1
./vp_bench.exe 0
./vp_bench.exe 15
