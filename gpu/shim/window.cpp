// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: SDL3 window for the Vulkan swapchain (X11, Wayland, or Metal on macOS).
#include <array>
#include "shim/bb_env.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bbgpu.h"
#include <SDL3/SDL.h>
#ifdef __APPLE__
#include <SDL3/SDL_metal.h>
#endif
#include "common/assert.h"
#include "common/logging/log.h"
#include "sdl_window.h"
#include "bbport_cursor.h"
#include "bbport_frame_rate.h"
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include "remote/bb_remote.h"


namespace {
std::array<std::atomic<BbGpuEventFilter>, 4> event_filters{};

/// runtime_pad.c: a Steam Deck's or Steam Controller's own controls are the gamepad, which Steam's
/// desktop layout also sends as keys and the mouse. Weak: tools linking the GPU library alone.
extern "C" int runtime_pad_steam_only(void) __attribute__((weak));

/// A keyboard or mouse event the game must not take (runtime_pad_steam_only): the gamepad's own
/// presses again. The port's menu (the overlay) keeps them, the trackpad as its pointer.
bool SteamPadEcho(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_TEXT_INPUT:
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_WHEEL:
        return runtime_pad_steam_only && runtime_pad_steam_only() && !BbOverlay::MenuOpen();
    default:
        return false;
    }
}

bool FilteredByModule(const SDL_Event* event, SDL_Window* window) {
    for (const auto& slot : event_filters) {
        const BbGpuEventFilter filter = slot.load(std::memory_order_acquire);
        if (filter && filter(event, window)) {
            return true;
        }
    }
    return false;
}
} // namespace

extern "C" void bbgpu_add_event_filter(BbGpuEventFilter filter) {
    for (auto& slot : event_filters) {
        BbGpuEventFilter expected = nullptr;
        if (slot.compare_exchange_strong(expected, filter)) {
            return;
        }
    }
}

namespace {
std::array<std::atomic<BbGpuPointerQuery>, 4> pointer_queries{};
std::array<std::atomic<BbGpuPadFilter>, 4> pad_filters{};

template <typename T>
void AddTo(std::array<std::atomic<T>, 4>& slots, T value) {
    for (auto& slot : slots) {
        T expected = nullptr;
        if (slot.compare_exchange_strong(expected, value)) {
            return;
        }
    }
}
} // namespace

extern "C" void bbgpu_add_pointer_query(BbGpuPointerQuery query) {
    AddTo(pointer_queries, query);
}

extern "C" int bbgpu_pointer_wanted(void) {
    for (const auto& slot : pointer_queries) {
        const BbGpuPointerQuery query = slot.load(std::memory_order_acquire);
        if (query && query()) {
            return 1;
        }
    }
    return 0;
}

extern "C" void bbgpu_add_pad_filter(BbGpuPadFilter filter) {
    AddTo(pad_filters, filter);
}

extern "C" void bbgpu_filter_pad(uint32_t* buttons, const uint8_t* sticks) {
    for (const auto& slot : pad_filters) {
        if (const BbGpuPadFilter filter = slot.load(std::memory_order_acquire)) {
            filter(buttons, sticks);
        }
    }
}

namespace Frontend {

namespace {

// Issue #69: the monitor the window (and fullscreen) goes to. BB_DISPLAY: its number in SDL's
// order (1, 2, ...; bb-gpu-capabilities --displays lists them) or a part of its name; without it
// SDL's primary display. The monitors are logged so a report says which one was taken.
SDL_DisplayID ChooseDisplay() {
    const SDL_DisplayID primary = SDL_GetPrimaryDisplay();
    const char* wanted = bb_env("BB_DISPLAY");
    int count = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&count);
    SDL_DisplayID chosen = 0;
    if (wanted && *wanted) {
        char* end = nullptr;
        const long number = std::strtol(wanted, &end, 10);
        if (end && *end == '\0') {
            if (number >= 1 && number <= count) {
                chosen = ids[number - 1];
            }
        } else {
            for (int i = 0; i < count && !chosen; ++i) {
                const char* name = SDL_GetDisplayName(ids[i]);
                if (name && strcasestr(name, wanted)) {
                    chosen = ids[i];
                }
            }
        }
    }
    const SDL_DisplayID display = chosen ? chosen : primary;
    for (int i = 0; i < count; ++i) {
        const char* name = SDL_GetDisplayName(ids[i]);
        const SDL_DisplayMode* mode = SDL_GetDesktopDisplayMode(ids[i]);
        std::printf("Display %d: %s %dx%d%s%s\n", i + 1, name ? name : "?", mode ? mode->w : 0,
                    mode ? mode->h : 0, ids[i] == primary ? " (primary)" : "",
                    ids[i] == display ? " <- the game's (BB_DISPLAY)" : "");
    }
    if (wanted && *wanted && !chosen) {
        std::printf("Display: BB_DISPLAY=%s matches none, the primary one is used\n", wanted);
    }
    SDL_free(ids);
    return display;
}

// The monitor the window is on, in pixels: the output's aspect ratio "Auto" follows it.
void TellMonitor(SDL_Window* window) {
    if (const SDL_DisplayMode* mode = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window))) {
        const float density = mode->pixel_density > 0.0f ? mode->pixel_density : 1.0f;
        const int width = int(mode->w * density + 0.5f), height = int(mode->h * density + 0.5f);
        BbSettings::SetMonitor(width, height);
        if (BbRemote::BackActive()) {
            BbRemote::Back::Monitor(width, height); // the game's System menu, in the game process
        }
    }
}

} // namespace

WindowSDL::WindowSDL(s32 width_, s32 height_, const char* title) : width{width_}, height{height_} {
    // Wayland with display scaling (KDE at 200%, #328/#234): the window in the monitor's pixels,
    // as SDL_VIDEO_WAYLAND_SCALE_TO_DISPLAY=1 does; text was soft, drawn at the logical size and
    // scaled up by the compositor. The variable, when set, wins.
    SDL_SetHintWithPriority(SDL_HINT_VIDEO_WAYLAND_SCALE_TO_DISPLAY, "1", SDL_HINT_DEFAULT);
    // Gamepads are sampled by runtime_pad.c; their events are pumped here with the window's.
    // DualShock 4 / DualSense in their full reports (over Bluetooth the touchpad comes only so),
    // and the gamepad kept while another window has the focus (a second monitor, an overlay).
    // From the SrShadowy fork; the variables, when set, win.
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ENHANCED_REPORTS, "1", SDL_HINT_DEFAULT);
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1", SDL_HINT_DEFAULT);
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        UNREACHABLE_MSG("Failed to initialize SDL video: {}", SDL_GetError());
    }
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
    const SDL_DisplayID display = ChooseDisplay();
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display));
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display));
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width_);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height_);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);
    const char* fullscreen = bb_env("BB_FULLSCREEN");
    // Issue #291: under gamescope (the Steam Deck's Game Mode) a window is shown letterboxed on
    // all four sides; there it is always fullscreen.
    const char* desktop = std::getenv("XDG_CURRENT_DESKTOP");
    const bool gamescope = std::getenv("GAMESCOPE_WAYLAND_DISPLAY") != nullptr ||
                           (desktop && std::strcmp(desktop, "gamescope") == 0);
    if (gamescope && !(fullscreen && fullscreen[0] == '1')) {
        std::printf("Display: gamescope: fullscreen\n");
    }
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN,
                           gamescope || (fullscreen && fullscreen[0] == '1'));
    // Issue #234: the swapchain at the display's pixels on a scaled desktop (Wayland 200%), not at
    // the logical size stretched by the compositor. Sizes below are read in pixels already.
#ifdef __APPLE__
    // bbport (macOS): Retina too (a CAMetalLayer sized in points drew at half the panel's pixels,
    // stretched 2x). BB_RETINA=0 sizes it in points (diagnostics).
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN,
                           !bb_env_off("BB_RETINA"));
#else
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true);
#endif
    base_title = title;
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    ASSERT_MSG(window, "Failed to create window: {}", SDL_GetError());
    // The cursor over the game: the port's own (BB_CURSOR=0: the system's).
    if (SDL_Cursor* cursor = BbCursor::Create()) {
        SDL_SetCursor(cursor);
    }

    const char* driver = SDL_GetCurrentVideoDriver();
    const SDL_PropertiesID wp = SDL_GetWindowProperties(window);
    if (driver && !std::strcmp(driver, "windows")) {
        window_info.type = WindowSystemType::Windows;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    } else if (driver && !std::strcmp(driver, "x11")) {
        window_info.type = WindowSystemType::X11;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        window_info.render_surface = reinterpret_cast<void*>(SDL_GetNumberProperty(wp, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    } else if (driver && !std::strcmp(driver, "wayland")) {
        window_info.type = WindowSystemType::Wayland;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
#ifdef __APPLE__
    } else if (driver && !std::strcmp(driver, "cocoa")) {
        // MoltenVK presents through a CAMetalLayer (as upstream shadPS4 on macOS).
        window_info.type = WindowSystemType::Metal;
        window_info.render_surface = SDL_Metal_GetLayer(SDL_Metal_CreateView(window));
#endif
    } else {
        UNREACHABLE_MSG("Unsupported SDL video driver {}", driver ? driver : "(none)");
    }
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    width = w;
    height = h;
    LOG_INFO(Frontend, "Window {}x{} on {}", w, h, driver);
#ifdef __APPLE__
    std::printf("Window: %dx%d pixels (pixel density %.2f)\n", w, h,
                SDL_GetWindowPixelDensity(window));
#endif
    TellMonitor(window);
}

WindowSDL::~WindowSDL() {
    SDL_DestroyWindow(window);
}

void WindowSDL::BeginTextInput(const std::string& initial, const std::string& prompt) {
    std::scoped_lock lock{text_mutex};
    text = initial;
    text_prompt = prompt;
    text_state = 0;
    text_requested = true;
}

int WindowSDL::PollTextInput(std::string& out) {
    std::scoped_lock lock{text_mutex};
    out = text;
    return text_state;
}

void WindowSDL::TakeMouse(double& dx, double& dy, int& up, int& down) {
    std::scoped_lock lock{mouse_mutex};
    dx = mouse_dx;
    dy = mouse_dy;
    up = wheel_up;
    down = wheel_down;
    mouse_dx = mouse_dy = 0;
    wheel_up = wheel_down = 0;
}

// Mouse look: relative mode while the window holds the mouse (runtime_pad.c turns its motion into
// the right stick). Motion from before is dropped, so taking the mouse does not turn the camera.
void WindowSDL::CaptureMouse(bool capture) {
    if (capture == mouse_captured.load(std::memory_order_relaxed)) {
        return;
    }
    if (!SDL_SetWindowRelativeMouseMode(window, capture) && capture) {
        std::printf("Mouse look: relative mouse mode failed: %s\n", SDL_GetError());
        return;
    }
    {
        std::scoped_lock lock{mouse_mutex};
        mouse_dx = mouse_dy = 0;
        wheel_up = wheel_down = 0;
    }
    mouse_captured.store(capture, std::memory_order_relaxed);
    if (capture) {
        capture_settle_ms = SDL_GetTicks() + 100;
    }
    static bool told;
    if (capture && !told) {
        told = true;
#ifdef __APPLE__
        // bbport (macOS): F1, ` and § open the overlay menu first, which lets the mouse go too.
        std::printf("Mouse look: on (F1, ` or \u00a7 opens the menu and releases the mouse, a click "
                    "takes it again)\n");
#else
        std::printf("Mouse look: on (F1 releases the mouse, a click takes it again)\n");
#endif
    }
}

void WindowSDL::UpdateTextTitle() {
    const std::string title = text_active ? base_title + " \u2014 " + text_prompt + ": " + text + "_  (Enter = OK, Esc = cancel)"
                                          : base_title;
    SDL_SetWindowTitle(window, title.c_str());
    BbOverlay::SetTextPrompt(text_active, text_prompt, text);
}

#ifdef __APPLE__
void WindowSDL::AppendTypedKey(int scancode, unsigned mod) {
    if (mod & (SDL_KMOD_CTRL | SDL_KMOD_GUI)) {
        return;
    }
    const SDL_Keycode key =
        SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(scancode), static_cast<SDL_Keymod>(mod), false);
    if ((key & SDLK_SCANCODE_MASK) || key < 0x20 || key == 0x7F || (key >= 0x80 && key < 0xA0) ||
        key > 0xFFFF) {
        return; // not a printable character (arrows, function keys, controls)
    }
    // UTF-8 encode one code point (<= U+FFFF).
    if (key < 0x80) {
        text += static_cast<char>(key);
    } else if (key < 0x800) {
        text += static_cast<char>(0xC0 | (key >> 6));
        text += static_cast<char>(0x80 | (key & 0x3F));
    } else {
        text += static_cast<char>(0xE0 | (key >> 12));
        text += static_cast<char>(0x80 | ((key >> 6) & 0x3F));
        text += static_cast<char>(0x80 | (key & 0x3F));
    }
}
#endif

// Mouse: the window holds the pointer (confined, and relative for looking) whenever it has focus, in
// the game's menus and its pointer menus too (the mouse must not leave the window mid-game). F1 lets
// it go (to use another window), a click or F1 again takes it back.
static bool mouse_user_released = false;
static bool mouse_grabbed = false;

bool WindowSDL::PollEvents() {
    {
        std::scoped_lock lock{text_mutex};
        if (text_requested) { // SDL text input must be toggled from the window thread
            text_requested = false;
            text_active = true;
            SDL_StartTextInput(window);
            UpdateTextTitle();
        }
    }
    if (!text_active) {
        BbOverlay::UpdateTextInput(window);
    }
    // A game module's menu wants the mouse as a pointer (bbgpu_add_pointer_query).
    const bool pointer = bbgpu_pointer_wanted() != 0;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (SteamPadEcho(event) || FilteredByModule(&event, window)) {
            continue;
        }
        if (event.type == SDL_EVENT_MOUSE_MOTION) {
            last_mouse_motion_ms = SDL_GetTicks();
            // Just taken: the pointer's jump into relative mode is not a turn of the camera.
            if (mouse_captured.load(std::memory_order_relaxed) && SDL_GetTicks() >= capture_settle_ms) {
                std::scoped_lock lock{mouse_mutex};
                mouse_dx += event.motion.xrel;
                mouse_dy += event.motion.yrel;
            }
        }
        if (text_active && (event.type == SDL_EVENT_TEXT_INPUT || event.type == SDL_EVENT_KEY_DOWN)) {
            std::scoped_lock lock{text_mutex};
            if (event.type == SDL_EVENT_TEXT_INPUT) {
#ifndef __APPLE__
                text += event.text.text;
#endif
            } else if (event.key.key == SDLK_BACKSPACE && !text.empty()) {
                size_t cut = text.size() - 1; // drop one UTF-8 code point
                while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
                text.erase(cut);
            } else if (event.key.key == SDLK_RETURN || event.key.key == SDLK_KP_ENTER || event.key.key == SDLK_ESCAPE) {
                text_state = event.key.key == SDLK_ESCAPE ? 2 : 1;
                text_active = false;
                SDL_StopTextInput(window);
            }
#ifdef __APPLE__
            else if (event.type == SDL_EVENT_KEY_DOWN) {
                // macOS: SDL's text-input events (Cocoa's text input system) did not arrive in
                // the game window, while key events do, so the name is built from key presses
                // with the current layout and Shift/Option state. Shortcuts are left alone.
                AppendTypedKey(static_cast<int>(event.key.scancode), static_cast<unsigned>(event.key.mod));
            }
#endif
            UpdateTextTitle();
            continue;
        }
        if (BbOverlay::HandleEvent(event)) {
            continue;
        }
        // Mouse look: a click in the game takes the mouse (that click is not passed on as a
        // button: runtime_pad.c reads the buttons only while it is held); F1 and leaving the
        // window let it go.
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && mouse_user_released) {
            mouse_user_released = false; // the click takes the window back (it is not a game button)
            continue;
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && !mouse_captured.load(std::memory_order_relaxed) &&
            mouse_look.load(std::memory_order_relaxed) && !BbOverlay::MenuOpen() && !pointer) {
            CaptureMouse(true);
            continue;
        }
        // F3: the frame rate limit 30 on and off (the cinematic of Laurence hangs above 30 FPS).
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F3) {
            BbFrameRate::SetLimit(BbFrameRate::Limit() == 30 ? 0 : 30);
            continue;
        }
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F1 &&
            (mouse_captured.load(std::memory_order_relaxed) || mouse_grabbed || mouse_user_released)) {
            mouse_user_released = !mouse_user_released;
            if (mouse_user_released) {
                CaptureMouse(false);
            }
            continue;
        }
        if (event.type == SDL_EVENT_MOUSE_WHEEL && mouse_captured.load(std::memory_order_relaxed)) {
            const float notches = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event.wheel.y : event.wheel.y;
            // bbport: whole notches only. A trackpad or Magic Mouse sends many small fractions
            // (and momentum after a swipe); each counted as a notch cycled several items at once.
            static float wheel_sum = 0.0f;
            wheel_sum += notches;
            std::scoped_lock lock{mouse_mutex};
            for (; wheel_sum >= 1.0f; wheel_sum -= 1.0f) {
                ++wheel_up;
            }
            for (; wheel_sum <= -1.0f; wheel_sum += 1.0f) {
                ++wheel_down;
            }
        }
        if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            CaptureMouse(false);
        }
        switch (event.type) {
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED: {
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            width = w;
            height = h;
            break;
        }
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
            TellMonitor(window);
            break;
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            // In the log: a session that ends here was closed, not lost to a crash.
            if (is_open) {
                std::printf("Window: closed (%s): the game ends\n",
                            event.type == SDL_EVENT_QUIT ? "quit" : "close button");
            }
            is_open = false;
            break;
        default:
            break;
        }
    }
    // A request from the game's side (runtime_pad.c: the spray can taken): as a click would.
    if (capture_requested.exchange(false, std::memory_order_relaxed) && !BbOverlay::MenuOpen() &&
        !text_active) {
        CaptureMouse(true);
    }
    // The menu, the name dialog and mouse_look=0 want the cursor.
    if (mouse_captured.load(std::memory_order_relaxed) &&
        (BbOverlay::MenuOpen() || text_active || !mouse_look.load(std::memory_order_relaxed))) {
        CaptureMouse(false);
    }
    // The game's pointer menus: the mouse goes, and comes back when they close.
    if (pointer && mouse_captured.load(std::memory_order_relaxed)) {
        CaptureMouse(false);
        released_for_pointer = true;
        // Always the same place: the window's middle (not wherever the camera mouse was left).
        int ww = 0, wh = 0;
        SDL_GetWindowSize(window, &ww, &wh);
        SDL_WarpMouseInWindow(window, float(ww) * 0.5f, float(wh) * 0.5f);
        last_mouse_motion_ms = SDL_GetTicks();
    } else if (!pointer && released_for_pointer) {
        released_for_pointer = false;
        if (mouse_look.load(std::memory_order_relaxed) && !BbOverlay::MenuOpen() && !text_active &&
            (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS)) {
            CaptureMouse(true);
        }
    }
    {
        const bool focused = (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) != 0;
#ifdef __APPLE__
        // bbport (macOS): F1 opens the settings menu here (Mac keyboards have no Insert key), so
        // it can't also release the mouse: the open menu lets it go instead (` or § as well).
        const bool hold = focused && !mouse_user_released && !BbOverlay::MenuOpen();
#else
        const bool hold = focused && !mouse_user_released;
#endif
        if (hold != mouse_grabbed) {
            mouse_grabbed = hold;
            SDL_SetWindowMouseGrab(window, hold);
        }
        // Taken for looking without a click once the window has focus (retried slowly: a
        // compositor may refuse it).
        static u64 last_try_ms = 0;
        if (hold && !mouse_captured.load(std::memory_order_relaxed) &&
            mouse_look.load(std::memory_order_relaxed) && !BbOverlay::MenuOpen() && !text_active &&
            !pointer && !released_for_pointer && SDL_GetTicks() - last_try_ms > 500) {
            last_try_ms = SDL_GetTicks();
            CaptureMouse(true);
        }
    }
    pointer_shown = pointer;
    UpdateCursor();
    return is_open;
}

// Issue #3: the OS cursor over the game. Hidden in fullscreen, and in a window after 3 s without
// moving the mouse; always shown while the settings menu or a game's pointer menu is open.
void WindowSDL::UpdateCursor() {
    const bool fullscreen = (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
    // In a game menu (pointer_shown) it also goes after 1.5 s without moving, so that the tail of a
    // closing menu does not leave it hanging over the game; any motion brings it back.
    const u64 idle = SDL_GetTicks() - last_mouse_motion_ms;
    const bool hide = !BbOverlay::MenuOpen() &&
                      (mouse_captured.load(std::memory_order_relaxed) ||
                       (pointer_shown ? idle > 1500 : (fullscreen || idle > 3000)));
    if (hide != cursor_hidden) {
        cursor_hidden = hide;
        hide ? SDL_HideCursor() : SDL_ShowCursor();
    }
}

} // namespace Frontend
