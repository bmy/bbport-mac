#!/usr/bin/env bash
# tools/bench.sh NAME [ENV=VALUE...]: starts the game through tools/restart.sh (title: cross,
# cross; waits for the level), stands still there for ${HOLD:-60} s, then closes it. The log is
# out/bench_NAME_<time>.log, the per-frame CSV out/bench_NAME_<time>.frames.csv.
set -u
cd -- "$(dirname -- "$0")/.."
name=${1:?usage: tools/bench.sh NAME [ENV=VALUE...]}
shift
base=out/bench_${name}_$(date +%m%d_%H%M%S)
env BB_GAME_DIR="${BB_GAME_DIR:-$PWD/../game_files/CUSA03173}" "$@" BB_FRAME_LOG="$PWD/$base.frames.csv" \
    tools/restart.sh > /dev/null
sleep "${HOLD:-60}"
pkill -x bb-probe; sleep 3; pkill -9 -x bb-probe
cp out/session.log "$base.log"
echo "$base.log"
