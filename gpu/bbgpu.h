// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* C interface of the GPU library (gpu/): the command processor and Vulkan renderer,
 * GnmDriver, VideoOut and kernel event queues, adapted to the native loader. */
#ifndef BBGPU_H
#define BBGPU_H
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const char *title;          /* window title */
    const char *serial;         /* CUSA id, names the pipeline cache */
    const char *user_dir;       /* pipeline cache/logs directory */
    uint32_t sdk_version;       /* from the eboot's procparam */
    uint32_t psf_attributes;    /* param.sfo ATTRIBUTE */
    int32_t width, height;      /* initial window size */
} BbGpuConfig;
/* Registers kernel event queues (needed with or without graphics). */
void bbgpu_register_kernel(void);
/* Creates window, Vulkan device, presenter and GPU command processor. */
int bbgpu_init(const BbGpuConfig *config);
/* Function for an imported NID ("NID#lib#mod"), or 0 when the GPU library does not provide it. */
uintptr_t bbgpu_resolve(const char *scoped_nid);
/* Called first by the loader's SIGSEGV handler: 1 when a GPU page-tracking fault was handled. */
int bbgpu_handle_fault(void *ucontext, void *address);
/* BB_WRITE_LOG=1: prints the logged GPU-side writes to guest memory near the fault. */
void bbgpu_dump_guest_writes(void *ucontext);
/* Keyboard text entry through the game window (IME dialog). begin returns 0 when
 * no window exists; poll returns 0 typing, 1 confirmed, 2 cancelled (UTF-8 text). */
int bbgpu_text_input_begin(const char *initial_utf8, const char *prompt_utf8);
int bbgpu_text_input_poll(char *out_utf8, uint64_t size);
/* The keyboard (SDL scancodes, true while held) of the game window, or NULL without one. */
const bool *bbgpu_keyboard_state(void);
/* Sets latch[scancode] to 1 for keys pressed in the game window since the last call (short taps,
 * runtime_pad.c); only with the native GPU process, whose window is in another process. */
void bbgpu_keyboard_take_latched(volatile uint8_t *latch, int count);
/* 1 while the in-game settings menu is open: the game's pad input is held neutral. */
int bbgpu_overlay_captures_input(void);
/* Mouse look (runtime_pad.c): the window takes the mouse when it is clicked (relative mode) and
 * lets it go on F1, losing focus or the menu opening. take: motion (pixels) and wheel notches
 * since the last call, reset; captured: 1 while the window holds it; enable: mouse_look=0 in
 * bbport.ini keeps the window from taking it. */
void bbgpu_mouse_take(double *dx, double *dy, int *wheel_up, int *wheel_down);
int bbgpu_mouse_captured(void);
/* bbport: the mouse buttons held (SDL_MouseButtonFlags), 0 while the window doesn't hold the
 * mouse; through the GPU library like the keyboard, for the macOS native GPU process. */
uint32_t bbgpu_mouse_buttons(void);
/* Asks the window to take the mouse (relative mode, as a click in the game does); the window
 * thread does it. */
void bbgpu_mouse_capture(void);
void bbgpu_mouse_look_enable(int enabled);
/* A module's look at the window's events first (an SDL_Event and the SDL_Window, on the window
 * thread): nonzero takes the event (the port and the game do not see it). Up to 4. */
typedef int (*BbGpuEventFilter)(const void *sdl_event, void *sdl_window);
void bbgpu_add_event_filter(BbGpuEventFilter filter);
/* A module's menu pointer: nonzero while the game shows a menu the mouse points into. The window
 * then lets the mouse go and shows the cursor, and takes the mouse back after (as it had it).
 * Up to 4; pointer_wanted: any of them says so. */
/* A video (AvPlayer) playing now; bbgpu_video_skip stops it (1 when one was stopped). */
int bbgpu_video_active(void);
/* The game shows its world (not a loading screen): the clock of pad recordings runs. */
int bbgpu_world_active(void);
/* Microseconds the world has been shown (the pad recording clock). */
unsigned long long bbgpu_world_us(void);
int bbgpu_video_skip(void);
typedef int (*BbGpuPointerQuery)(void);
void bbgpu_add_pointer_query(BbGpuPointerQuery query);
int bbgpu_pointer_wanted(void);
/* A module's changes to the pad state the game reads (runtime_pad.c, after the host's input and
 * the scripted one): the buttons (PS4 bits); the sticks (lx ly rx ry) as they are. Up to 4. */
typedef void (*BbGpuPadFilter)(uint32_t *buttons, const uint8_t *sticks);
void bbgpu_add_pad_filter(BbGpuPadFilter filter);
void bbgpu_filter_pad(uint32_t *buttons, const uint8_t *sticks);
/* The shown frame's width / height (the display buffer's; the window letterboxes it). */
float bbgpu_frame_aspect(void);
/* A line of text over the game, near the bottom, until called with NULL (any thread). */
void bbgpu_overlay_banner(const char *utf8);
/* Trophies (gpu/shim/bbport_trophies.cpp): setup takes the game's sce_sys folder and the user
 * folder (once); unlock keeps the trophy and announces it over the frame the first time, and
 * returns the platinum unlocked along with it, else -1. */
void bbgpu_trophy_setup(const char *sce_sys_host_dir, const char *user_dir);
int bbgpu_trophy_unlock(int id);
/* Native mouse look: turns the game's camera (radians, applied at its next update); 0 when the
 * camera hook is not in (another game version, BB_NATIVE_MOUSE=0) or the camera does not take the
 * turns now (lock-on, menus): drive the right stick then. */
int bbgpu_camera_look(float yaw, float pitch);
/* The pipeline caches written by an earlier build, translated again for this one; called before
 * the runtime reserves the guest's memory (render/shader_rebuild.h). */
int bbgpu_rebuild_shader_caches(void);
/* One cache directory (RebuildCaches' child where there is no fork: bb-probe --rebuild-shader-cache). */
int bbgpu_rebuild_shader_cache(const char *dir);
int bbgpu_predict_shaders_test(const char *cache_dir, const char *pack, const char *game_dir,
                               const char *out_dir, int jobs);
int bbgpu_merge_shader_metas(const char *cache_dir, const char *pack, const char *game_dir,
                             const char *meta_dir, const char *out_pack, const char *work_dir, int jobs);
int bbgpu_meta_reconstruct_test(const char *cache_dir, const char *pack, const char *game_dir);
int bbgpu_merge_shader_packs(const char *cache_dir, const char *pack_a, const char *pack_b,
                             const char *game_dir, const char *out_pack, const char *work_dir, int jobs);
/* Shader recipes (render/shader_rebuild.h): bb-probe --export-shader-recipes CACHE_DIR GAME_DIR OUT
 * and --import-shaders CACHE_DIR PACK GAME_DIR MOTION_PARAMS MOTION_POSITIONS (run by the GPU
 * library itself, before the guest's memory exists in that process). */
int bbgpu_export_shader_recipes(const char *cache_dir, const char *game_dir, const char *out_path);
int bbgpu_import_shader_recipes(const char *cache_dir, const char *pack, const char *game_dir,
                                unsigned long long motion_params, unsigned long long motion_positions);
/* 1 while the free camera (the community patch) is on: the mouse drives it through the right stick. */
int bbgpu_free_camera(void);
/* The game's current camera state (its projection parameters at +0x50), 0 until known. */
uint64_t bbgpu_camera_state(void);
/* Patches the loaded image before the game runs (image still writable): libGnm entry hooks. */
void bbgpu_patch_image(unsigned char *image, uint64_t size);
/* bbport (native GPU process): moves a page-aligned, mapped range of the game's
 * memory into memory the GPU process shares, keeping its contents; prot is PROT_*. 0 on success
 * and when there is no GPU process. */
int bbgpu_share_range(void *address, uint64_t size, int prot);
/* Number of symbols registered by the vendored libraries (diagnostics). */
unsigned bbgpu_symbol_count(void);
#ifdef __APPLE__
/* macOS: Cocoa windows exist only on the process's main thread, so the loader runs the game
 * on a secondary thread and hands the main thread to the GPU library here; it runs the
 * window thread's work (window, events) when bbgpu_init asks for it. Never returns. */
void bbgpu_main_thread_loop(void);
#endif
#ifdef __cplusplus
}
#endif
#endif
