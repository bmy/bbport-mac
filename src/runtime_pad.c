// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* libScePad on SDL3 gamepads and the keyboard. SDL events are pumped by the window thread
 * (gpu/shim/window.cpp); here state is only sampled.
 *
 * Keyboard layout (Dark Souls III's on PC, with the player's changes; also with a gamepad
 * connected: both drive the game):
 *   WASD left stick, IJKL right stick (the camera), Space the dodge (a short Circle press: never a
 *   run), Left Shift the run (Circle held), Shift + Space the jump (Circle let go and pressed again
 *   while running), F and Enter Cross
 *   (interact, confirm), Q Triangle (blood vial), E L1 (transform the weapon), Tab and R Square
 *   (the quick item), C R3 (lock-on), Left Ctrl L3, Left Alt walk (held), Escape Options (the menu; Circle, the way back, while a game menu is open), G left touchpad
 *   (gestures), Backspace right touchpad; the d-pad on 1 (left: the left hand's weapon), 2 (right:
 *   the right hand's), 3 (up), 4 (down) and the arrow keys.
 *   Mouse (once the game window is clicked; F1 or leaving the window releases it): look (the
 *   right stick), Left R1 (attack), Shift+Left R2, Right L2 (the gun), Middle R3, side buttons R2
 *   (X1) and L1 (X2), wheel the d-pad's up/down (quick items).
 *   A binding may be a chord, "Left Shift+Mouse Left": while it is held, the second key's plain
 *   bindings do not count (Shift+Left is R2, not R2 and R1).
 *
 * Stick neutral: SDL exposes no way to read a pad's calibration and some clones report a
 * biased neutral (a Switch-style pad was seen returning both sticks at a constant ~ +/-16380).
 * The neutral of each axis is taken from a quiet window after the pad opens (SDL returns zero
 * until the first report arrives, so those samples are skipped) and subtracted when all four axes
 * are biased; otherwise (a genuine pad, perhaps opened with a stick held) the neutral is 0.
 *
 * Travel: such a clone also uses only part of SDL's -32768..32767 span (its neutral sits in the
 * middle of one half), so reading the axis as -128..127 would reach only half deflection and a
 * full push would never run. Each axis is instead scaled by its own travel: a biased axis starts
 * from its neutral's magnitude and is refined by the largest push seen per direction, so both
 * directions reach full deflection even when the two travels differ by a few percent (that
 * difference is what makes one direction run and the other only walk). A genuine pad keeps the
 * plain -32768..32767 -> -128..127 read. The stick is then converted as shadPS4 does, with an
 * inner/outer dead zone (BB_PAD_DEADZONE, default 5; BB_PAD_DEADZONE_OUTER, default 127) mapping
 * the axis' travel up to full deflection. BB_PAD_CENTER=lx,ly,rx,ry overrides the neutral,
 * BB_PAD_CENTER_CAL=0 disables the measurement. */
#define _GNU_SOURCE
#include "runtime.h"
#include "platform.h"
#include "../gpu/shim/bb_env.h"
#include "gpu/bbgpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <SDL3/SDL.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define ERR_INVALID_ARG ((int32_t)0x80920001)
#define ERR_INVALID_HANDLE ((int32_t)0x80920003)
#define ERR_ALREADY_OPENED ((int32_t)0x80920004)
#define ERR_NOT_INITIALIZED ((int32_t)0x80920005)
#define PAD_HANDLE 1

enum {
    BTN_L3=0x2, BTN_R3=0x4, BTN_OPTIONS=0x8, BTN_UP=0x10, BTN_RIGHT=0x20, BTN_DOWN=0x40, BTN_LEFT=0x80,
    BTN_L2=0x100, BTN_R2=0x200, BTN_L1=0x400, BTN_R1=0x800, BTN_TRIANGLE=0x1000, BTN_CIRCLE=0x2000,
    BTN_CROSS=0x4000, BTN_SQUARE=0x8000, BTN_TOUCHPAD=0x100000,
};
typedef struct { uint16_t x, y; uint8_t id, reserve[3]; } PadTouch;
typedef struct {
    uint32_t buttons;
    uint8_t left_x, left_y, right_x, right_y;
    uint8_t l2, r2, analog_padding[2];
    float orientation[4], acceleration[3], angular_velocity[3];
    uint8_t touch_count, touch_reserve[3];
    uint32_t touch_held_time;
    PadTouch touches[2];
    uint8_t connected, pad0[3];
    uint64_t timestamp;
    uint8_t extension[16];
    uint8_t connected_count, reserve[2], unique_length, unique[12];
} PadData;
typedef struct {
    float pixel_density; uint16_t resolution_x, resolution_y;
    uint8_t dead_zone_left, dead_zone_right, connection_type, connected_count;
    uint8_t connected, pad[3];
    int32_t device_class;
    uint8_t reserve[8];
} ControllerInfo;
_Static_assert(sizeof(PadData)==120,"OrbisPadData layout");
_Static_assert(sizeof(PadTouch)==8,"OrbisPadTouch layout");
_Static_assert(__builtin_offsetof(PadData,touches)==60,"OrbisPadData touch offset");
_Static_assert(__builtin_offsetof(PadData,timestamp)==80,"OrbisPadData timestamp offset");
_Static_assert(sizeof(ControllerInfo)==28,"OrbisPadControllerInformation layout");

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static int initialized, opened, sdl_ready;
static SDL_Gamepad *gamepad;
static size_t reads;
static uint8_t connected_count;

/* The pad by topic, in src/runtime_pad/ (one translation unit with this file). */
#include "runtime_pad/calibration.c"
#include "runtime_pad/touchpad.c"
#include "runtime_pad/gamepad.c"
#include "runtime_pad/bindings.c"
#include "runtime_pad/prompts.c"
#include "runtime_pad/motion.c"
#include "runtime_pad/sample.c"

static ABI int32_t pad_init(void) { pthread_mutex_lock(&lock); initialized=1; pthread_mutex_unlock(&lock); return 0; }
static ABI int32_t pad_open(int32_t user, int32_t type, int32_t index, const void *param) {
    (void)param;
    if (!initialized) return ERR_NOT_INITIALIZED;
    if (user!=1 || (type!=0 && type!=2) || index) { /* only user 1's standard / special port */
        printf("Runtime: scePadOpen(user %d, type %d, index %d) refused\n",user,type,index);
        return ERR_INVALID_ARG;
    }
    pthread_mutex_lock(&lock);
    int already=opened; opened=1;
    pthread_mutex_unlock(&lock);
    if (already) return ERR_ALREADY_OPENED;
    puts("Runtime: pad opened for user 1 (SDL gamepad or keyboard)");
    return PAD_HANDLE;
}
static ABI int32_t pad_close(int32_t handle) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    opened=0; return 0;
}
/* Reverse engineering: hardware watchpoints on the game's memory (runtime_rewatch.c). */
void runtime_re_watch_tick(const void *pad_buffer, uint64_t reads);

static ABI int32_t pad_read_state(int32_t handle, PadData *data) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    sample(data); ++reads;
    pthread_mutex_unlock(&lock);
    static int trace=-1;
    if (trace<0) { const char *v=bb_env("BB_PAD_TRACE"); trace=v && *v=='2' ? 2 : v && *v=='1'; }
    if (trace==2) {
        /* Every change of the sticks with the read's time (camera taps, #325; BB_CAMLOG=1 shows
         * what the camera did). */
        static uint8_t last[4]={128,128,128,128};
        const uint8_t now[4]={data->left_x,data->left_y,data->right_x,data->right_y};
        if (memcmp(now,last,4)) {
            printf("Pad trace %llu %u %u %u %u\n",(unsigned long long)data->timestamp,now[0],now[1],now[2],now[3]);
            memcpy(last,now,4);
        }
    }
    if (trace) {
        static uint32_t last_buttons; static uint64_t last_us;
        if (data->buttons!=last_buttons || data->timestamp-last_us>5000000) {
            printf("Pad: read %zu buttons 0x%x sticks %u %u %u %u\n",reads,data->buttons,data->left_x,
                   data->left_y,data->right_x,data->right_y);
            last_buttons=data->buttons; last_us=data->timestamp;
        }
    }
    runtime_re_watch_tick(data,(uint64_t)reads);
    return 0;
}
/* Buffered read: the port samples once per call, so one entry is returned. */
static ABI int32_t pad_read(int32_t handle, PadData *data, int32_t count) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data || count<1 || count>64) return ERR_INVALID_ARG;
    pad_read_state(handle,data);
    return 1;
}
static ABI int32_t pad_info(int32_t handle, ControllerInfo *info) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!info) return ERR_INVALID_ARG;
    memset(info,0,sizeof(*info));
    info->pixel_density=44.86f; info->resolution_x=1920; info->resolution_y=943;
    info->dead_zone_left=info->dead_zone_right=2;
    info->connection_type=0; info->connected=1; info->device_class=0;
    pthread_mutex_lock(&lock);
    current_gamepad();
    info->connected_count=connected_count ? connected_count : 1;
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_vibration(int32_t handle, const uint8_t *param) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!param) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    SDL_Gamepad *g=current_gamepad();
    if (g) SDL_RumbleGamepad(g,(uint16_t)(param[0]*257),(uint16_t)(param[1]*257),1000);
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_ok_handle(int32_t handle) { return handle==PAD_HANDLE && opened ? 0 : ERR_INVALID_HANDLE; }
static ABI int32_t pad_ok_handle_flag(int32_t handle, uint8_t flag) { (void)flag; return pad_ok_handle(handle); }

static const RuntimeExport exports[]={
    {"scePadInit",pad_init}, {"scePadOpen",pad_open}, {"scePadClose",pad_close},
    {"scePadReadState",pad_read_state}, {"scePadRead",pad_read},
    {"scePadGetControllerInformation",pad_info}, {"scePadSetVibration",pad_vibration},
    {"scePadResetOrientation",pad_ok_handle},
    {"scePadSetAngularVelocityDeadbandState",pad_ok_handle_flag}, {"scePadSetTiltCorrectionState",pad_ok_handle_flag},
    {"scePadSetMotionSensorState",pad_ok_handle_flag},
};
uintptr_t runtime_pad_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
void runtime_pad_report(void) { printf("Runtime: pad reads=%zu, gamepad=%s\n",reads,gamepad ? SDL_GetGamepadName(gamepad) : "none"); }
