#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
# tools/restart.sh [run.sh args]: restarts the game with frame stats, BB_PAD_FILE and
# BB_TOGGLE_FILE (out/pad, out/toggles) and enters the level (title: cross; the port's own
# skip of the online/offline choice is on: one cross is Continue; BB_SKIP_NETWORK_CHOICE=0 shows
# the choice, and then two presses).
# Only a game these tools started is restarted (out/tools_game.pid): with another one running (the
# user's) nothing happens and it exits with 1; FORCE_RESTART=1 closes that one too.
cd -- "$(dirname -- "$0")/.."
T=tools
# Saves: a copy of user/ (out/test_user), so that test runs (autosaves, items, walks) never move
# the player's own character; RESET_TEST_USER=1 copies it afresh. BB_USER_DIR overrides.
if [[ -z ${BB_USER_DIR:-} ]]; then
    if [[ ${RESET_TEST_USER:-0} == 1 || ! -d out/test_user ]]; then
        rm -rf out/test_user && cp -a user out/test_user
    fi
    export BB_USER_DIR=$PWD/out/test_user
fi
export BB_GAME_DIR=${BB_GAME_DIR:-$PWD/../game_files/CUSA03173}
own=$(cat out/tools_game.pid 2>/dev/null)
for pid in $(pgrep -x bb-probe); do
    if [[ $pid != "$own" ]]; then
        if [[ ${FORCE_RESTART:-0} != 1 ]]; then
            echo "restart.sh: a game these tools did not start is running (pid $pid): left alone" >&2
            exit 1
        fi
        kill "$pid"; sleep 2; kill -9 "$pid" 2>/dev/null
    fi
done
$T/stop_own.sh; sleep 1
: > out/pad; echo ${BASE_MASK:-0} > out/toggles
# SDL drops gamepad presses while the window lacks focus: test runs press the virtual pad unattended.
export SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=${SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS:-1}
# BB_FRAME_STATS=0: measured as players run the game (no statistics; its timers are in the draw
# path), with only the present rate (BB_FPS_PRINT) and a fixed wait for the level.
stats=${BB_FRAME_STATS:-1}
BB_SKIP_NETWORK_CHOICE=${BB_SKIP_NETWORK_CHOICE:-1} BB_PAD_FILE=$PWD/out/pad BB_PRESENT_DUMP_TRIGGER=$PWD/out/present_trigger BB_FINAL_DUMP_TRIGGER=$PWD/out/final_trigger BB_DUMP_DIR=$PWD/out/present BB_FRAME_STATS=$stats BB_FPS_PRINT=$((1 - stats)) BB_TOGGLE_FILE=$PWD/out/toggles BB_FPS_LIMIT=${BB_FPS_LIMIT:-0} \
    ${CPUS:+taskset -c $CPUS} setsid stdbuf -oL -eL bash run.sh "$@" > out/session.log 2>&1 < /dev/null &
# The game's process (run.sh starts it after preparing the files): the only bb-probe now.
for i in $(seq 1 120); do
    sleep 0.5
    pid=$(pgrep -n -x bb-probe) && { echo "$pid" > out/tools_game.pid; break; }
done
# Title menu: the pad is opened and frame stats report a light scene for a while.
for i in $(seq 1 180); do
    sleep 1
    [[ $(grep -c '^Frame stats\|^Present rate' out/session.log) -ge 3 ]] && grep -q 'pad opened' out/session.log && break
done
sleep ${TITLE_WAIT:-6}
$T/press.sh cross
if [[ ${BB_SKIP_NETWORK_CHOICE:-1} == 0 ]]; then sleep 6; $T/press.sh cross; fi
for i in $(seq 1 90); do
    sleep 2
    [[ $stats == 0 ]] && { sleep 28; break; }
    last=$(grep '^Frame stats' out/session.log | tail -1 | sed -E 's/.* ([0-9]+) draws\/frame.*/\1/')
    [[ -n $last && $last -gt 600 ]] && break
done
sleep 10
grep '^Frame stats\|^Present rate' out/session.log | tail -1
