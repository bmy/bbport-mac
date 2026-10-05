// bbport: SDL3 window for the Vulkan swapchain (X11, Wayland, or Metal on macOS).
#include <cstdlib>
#include <cstring>
#include <SDL3/SDL.h>
#ifdef __APPLE__
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <SDL3/SDL_metal.h>
#endif
#include "common/assert.h"
#include "common/logging/log.h"
#include "sdl_window.h"
#include "bbport_overlay.h"

#ifdef __APPLE__
namespace {
// Refresh rate of the display the window is on, 0 when unknown. Only the main thread may ask
// AppKit which screen the window is on, so the window thread keeps it here for VideoOut.
std::atomic<u32> g_display_hz{0};
} // namespace

// The display the window is on (Linux: the primary display, read once in bbgpu.cpp). A
// MacBook Pro's ProMotion panel reports 120 Hz; an external display its own rate.
u32 BbDisplayRefreshHz() {
    const u32 hz = g_display_hz.load(std::memory_order_relaxed);
    return hz ? std::max<u32>(hz, 60) : 60;
}
#endif

namespace Frontend {

WindowSDL::WindowSDL(s32 width_, s32 height_, const char* title) : width{width_}, height{height_} {
    // Gamepads are sampled by runtime_pad.c; their events are pumped here with the window's.
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        UNREACHABLE_MSG("Failed to initialize SDL video: {}", SDL_GetError());
    }
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width_);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height_);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);
    const char* fullscreen = std::getenv("BB_FULLSCREEN");
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN, fullscreen && fullscreen[0] == '1');
    base_title = title;
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    ASSERT_MSG(window, "Failed to create window: {}", SDL_GetError());

    const char* driver = SDL_GetCurrentVideoDriver();
    const SDL_PropertiesID wp = SDL_GetWindowProperties(window);
    if (driver && !std::strcmp(driver, "x11")) {
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
    UpdateDisplayRefresh();
#endif
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

void WindowSDL::UpdateTextTitle() {
    const std::string title = text_active ? base_title + " \u2014 " + text_prompt + ": " + text + "_  (Enter = OK, Esc = cancel)"
                                          : base_title;
    SDL_SetWindowTitle(window, title.c_str());
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

void WindowSDL::UpdateDisplayRefresh() {
    const SDL_DisplayID display = SDL_GetDisplayForWindow(window);
    const SDL_DisplayMode* mode = display ? SDL_GetCurrentDisplayMode(display) : nullptr;
    const u32 hz = mode && mode->refresh_rate > 0 ? u32(mode->refresh_rate + 0.5f) : 0;
    static bool logged = false;
    if (g_display_hz.exchange(hz) == hz && logged) {
        return;
    }
    logged = true;
    const char* name = display ? SDL_GetDisplayName(display) : nullptr;
    std::printf("Window: display \"%s\", %.2f Hz%s\n", name ? name : "?",
                mode ? double(mode->refresh_rate) : 0.0, hz ? "" : " (unknown: 60 assumed)");
}
#endif

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
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
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
        switch (event.type) {
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED: {
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            width = w;
            height = h;
            break;
        }
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            is_open = false;
            break;
#ifdef __APPLE__
        // Moved to another display, or its refresh rate was changed in System Settings.
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        case SDL_EVENT_DISPLAY_DESKTOP_MODE_CHANGED:
            UpdateDisplayRefresh();
            break;
#endif
        default:
            break;
        }
    }
    return is_open;
}

} // namespace Frontend
