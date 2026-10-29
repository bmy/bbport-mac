// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_pad.c part: which SDL gamepad drives the game (BB_GAMEPAD), the test virtual pad; the guest libc heap report.
 * #included by runtime_pad.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */

/* BB_GAMEPAD (the launcher's controller choice): its SDL GUID, or part of its name. Issue #15:
 * wheels and other controllers connected for good came first. */
#include "../../gpu/shim/bb_env.h"
static const char *preferred_gamepad(void) {
    static const char *want; static int read;
    if (!read) { want=bb_env("BB_GAMEPAD"); if (want && !*want) want=NULL; read=1; }
    return want;
}
static int is_preferred(SDL_JoystickID id, const char *want) {
    char guid[33];
    SDL_GUIDToString(SDL_GetGamepadGUIDForID(id),guid,sizeof guid);
    const char *name=SDL_GetGamepadNameForID(id);
    return !strcasecmp(guid,want) || (name && strcasestr(name,want));
}
/* BB_TEST_VPAD=1 (tests): a virtual gamepad without a touch surface (an Xbox-like pad) whose
 * buttons the pad file presses ("vpad:back"), so scripted runs go through the real gamepad path
 * (bindings, Back as the touchpad). BB_GAMEPAD=bbport picks it. */
static SDL_Joystick *test_vpad;
static void test_vpad_attach(void) {
    static int done;
    if (done) return;
    done=1;
    const char *env=bb_env("BB_TEST_VPAD");
    if (!env || env[0]!='1') return;
    SDL_VirtualJoystickDesc desc;
    SDL_INIT_INTERFACE(&desc);
    desc.type=SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes=SDL_GAMEPAD_AXIS_COUNT;
    desc.nbuttons=SDL_GAMEPAD_BUTTON_MISC1;
    desc.name="bbport test pad";
    const SDL_JoystickID id=SDL_AttachVirtualJoystick(&desc);
    test_vpad=id ? SDL_OpenJoystick(id) : NULL;
    printf("Runtime: test gamepad %s\n",test_vpad ? "attached" : SDL_GetError());
}
/* The chosen gamepad, else the first while it is not connected (checked again every second, it
 * is taken as soon as it connects); called under lock. */
static SDL_Gamepad *current_gamepad(void) {
    static int on_preferred; static uint64_t last_scan;
    if (!sdl_ready) sdl_ready = SDL_WasInit(SDL_INIT_GAMEPAD) ? 1 : SDL_InitSubSystem(SDL_INIT_GAMEPAD) ? 1 : -1;
    if (sdl_ready<0) return NULL;
    test_vpad_attach();
    if (gamepad && !SDL_GamepadConnected(gamepad)) { SDL_CloseGamepad(gamepad); gamepad=NULL; }
    const char *want=preferred_gamepad();
    const uint64_t now=now_us();
    if (!gamepad || (want && !on_preferred && now-last_scan>1000000)) {
        last_scan=now;
        int count=0, pick=-1;
        SDL_JoystickID *ids=SDL_GetGamepads(&count);
        for (int i=0; want && ids && i<count && pick<0; ++i) if (is_preferred(ids[i],want)) pick=i;
        if (pick<0 && !gamepad && ids && count>0) pick=0;
        if (pick>=0 && (!gamepad || SDL_GetGamepadID(gamepad)!=ids[pick])) {
            if (gamepad) SDL_CloseGamepad(gamepad);
            gamepad=SDL_OpenGamepad(ids[pick]);
            cal_reset(); /* a new pad has its own neutral */
            on_preferred=want && gamepad && is_preferred(ids[pick],want);
            if (gamepad) {
                ++connected_count;
                printf("Runtime: gamepad connected: %s%s\n",SDL_GetGamepadName(gamepad),
                       !want ? "" : on_preferred ? " (the chosen one)" : " (the chosen one is not connected)");
            }
        }
        SDL_free(ids);
    }
    return gamepad;
}
/* bbport (frame stats): the game's libc heap, read every 5 s from a game thread (it reads the pad)
 * with libc.prx's malloc_stats (export stub at libc.prx+0x1f8c0; malloc_stats_fast returns 1 in this libc, the module at image +0x56e0000):
 * in use now and at most, and what it took from the system. A heap that keeps growing is a leak. */
typedef struct { uint16_t size, version; uint32_t reserved; uint64_t max_system, system, max_in_use, in_use; } MallocManagedSize;
static void report_guest_heap(void) {
    static int enabled=-1; static uint64_t last;
    if (enabled<0) enabled=bb_env("BB_FRAME_STATS")!=NULL;
    const uint64_t now=now_us();
    if (!enabled || now-last<5000000) return;
    last=now;
    /* Bloodborne's libc.prx (another game's libc has its exports elsewhere). */
    if (!runtime_libc_base) return;
    const uint8_t *stub=(const uint8_t *)(runtime_libc_base+0x1f8c0);
    if (runtime_libc_base!=0x800000000ull+0x56e0000 || stub[0]!=0xff || stub[1]!=0x25) return;
    ABI int (*stats)(MallocManagedSize *)=(ABI int (*)(MallocManagedSize *))(uintptr_t)stub;
    MallocManagedSize m={.size=sizeof(m),.version=1};
    const int result=stats(&m);
    if (result!=0) { static int told; if (!told++) printf("Guest heap: malloc_stats returned %#x\n",(unsigned)result); return; }
    {
        printf("Guest heap: %.1f MB in use (most %.1f), %.1f MB from the system (most %.1f)\n",
               m.in_use/1048576.0,m.max_in_use/1048576.0,m.system/1048576.0,m.max_system/1048576.0);
    }
}
