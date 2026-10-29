// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne (1.09): the mouse as a pointer in the game's Scaleform menus.
//
// Reverse engineering (addresses are offsets in the 1.09 executable, image at 0x800000000; the
// facts first published in a fork of bloodborne_pc, checked against the executable here):
//  - 0x1acff30 updates one list component each frame it is shown (rdi the component, rsi its
//    input: the first byte is nonzero while the list has the input focus). It is reached through
//    the list class's vtable and five direct calls; one hook at its entry sees them all.
//  - The component: +0x30 its listener (vtable +0x10: cursor moved, with a pointer to the index),
//    +0x70 item count, +0x74 cursor, +0x78 cursor override (>= 0 wins), +0x7c items per line,
//    +0x80 lines, +0x84 scroll within a line, +0x24c line scroll, +0x258 the item provider
//    (vtable +0x18 makes the Scaleform value of item (cell | line << 32)), +0x2b0 wraps around,
//    +0xc8/+0xd0 the type and data of its root value.
//  - 0x1acf6b0 sets the cursor (component, index; 1 when it scrolled), 0x1ad0dd0 scrolls by
//    (lines, cells), 0x1acf0e0 hides every item's cursor.
//  - Scaleform values are 0x80 bytes: +0x30 the value, +0x40 the object interface (vtable +0x18
//    releases), +0x48 the type (& 0x40: managed; & 0x8f == 10: a display object), +0x50 its data
//    (+0x88: the native display object). 0x1adb220 finds a child by path ("Cursor" at 0x4934044,
//    "Widgets/Slider/LeftArrow", ...); a value's own vtable +8 shows or hides it.
//  - Display objects: +0x38 parent, +0x6b flags (0x40 visible), vtable +0x1f0 local bounds
//    (twips, x1 y1 x2 y2); 0x2040 their world matrix (2x4 floats, twips). The stage is 1920x1080
//    pixels, 20 twips each.
//  - 0x5580e38 the menu flow (+0xb11 zero: the title menu; +0xa88: a menu is open; +0xa78: input
//    blocked), 0x5562878 another input block (+0x2e4), 0x5128948 the region (5: Japan, circle
//    confirms), 0x557fc60 the menu sound request (+8 its label: 0x49344e4 the cursor's move).
#include "bloodborne_menu_pointer.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <SDL3/SDL.h>

#include "bbgpu.h"
#include "bbport_guest_code_hooks.h"
#include "bbport_overlay.h"
#include "bbport_toggles.h"
#include "common/types.h"

namespace BbMenuPointer {
namespace {

using Addr = std::uintptr_t;

constexpr Addr ImageBase = 0x800000000ull;
constexpr Addr CodeEnd = 0x50d96dc;

constexpr Addr ListUpdate = 0x1acff30;
constexpr std::uint8_t ListUpdateEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp...
constexpr Addr SetCursorFn = 0x1acf6b0, ScrollByFn = 0x1ad0dd0, HideCursorsFn = 0x1acf0e0;
constexpr Addr FindChildFn = 0x1adb220, WorldMatrixFn = 0x2040;
constexpr Addr CursorName = 0x4934044;
constexpr Addr MenuFlow = 0x5580e38, InputBlock = 0x5562878, Region = 0x5128948;
constexpr Addr MenuSound = 0x557fc60, CursorMoveSound = 0x49344e4;

enum : std::uint32_t {
    Up = 0x10, Right = 0x20, Down = 0x40, Left = 0x80, Circle = 0x2000, Cross = 0x4000,
};

/// List component fields.
enum : std::size_t {
    Listener = 0x30, Count = 0x70, Cursor = 0x74, CursorOverride = 0x78, PerLine = 0x7c,
    Lines = 0x80, LineScroll = 0x84, RootType = 0xc8, RootData = 0xd0, Scroll = 0x24c,
    Provider = 0x258, Wraps = 0x2b0,
};

Addr At(Addr offset) {
    return ImageBase + offset;
}

/// Copies guest memory; false where nothing is mapped (the runtime's fault recovery).
bool Read(Addr address, void* out, std::size_t size) {
    if (address < 0x10000) {
        return false;
    }
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        runtime_fault_recover = nullptr;
        return false;
    }
    runtime_fault_recover = &recover;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    runtime_fault_recover = nullptr;
    return true;
}

template <typename T>
T Get(Addr address, T fallback = {}) {
    T value = fallback;
    return Read(address, &value, sizeof(value)) ? value : fallback;
}

template <typename T>
T Field(const std::uint8_t* object, std::size_t offset) {
    T value{};
    std::memcpy(&value, object + offset, sizeof(value));
    return value;
}

/// A function of the game (its usual prologue), from a vtable or a field.
bool IsGameFunction(Addr address) {
    if (address < At(0) || address >= At(CodeEnd)) {
        return false;
    }
    std::uint8_t prologue[4] = {};
    return Read(address, prologue, sizeof(prologue)) && prologue[0] == 0x55 &&
           prologue[1] == 0x48 && prologue[2] == 0x89 && prologue[3] == 0xe5;
}

Addr VirtualFunction(Addr object, std::size_t slot_offset) {
    const Addr vtable = Get<Addr>(object);
    const Addr function = vtable ? Get<Addr>(vtable + slot_offset) : 0;
    return IsGameFunction(function) ? function : 0;
}

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// A Scaleform value the game fills in (released when it goes out of scope).
class Value {
public:
    Value() = default;
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    ~Value() {
        if (!(Field<std::uint32_t>(bytes, 0x48) & 0x40)) {
            return;
        }
        const Addr object_interface = Field<Addr>(bytes, 0x40);
        if (const Addr release = object_interface ? VirtualFunction(object_interface, 0x18) : 0) {
            using Fn = void(PS4_SYSV_ABI*)(void*, void*, Addr);
            reinterpret_cast<Fn>(release)(reinterpret_cast<void*>(object_interface), bytes + 0x30,
                                          Field<Addr>(bytes, 0x50));
        }
    }
    void* Raw() {
        return bytes;
    }
    /// The native display object, or 0 (not a display object).
    Addr DisplayObject() const {
        if ((Field<std::uint32_t>(bytes, 0x48) & 0x8f) != 10) {
            return 0;
        }
        const Addr data = Field<Addr>(bytes, 0x50);
        return data ? Get<Addr>(data + 0x88) : 0;
    }
    /// Its child at `path` ("Cursor", "Widgets/Slider/LeftArrow").
    void Child(const char* path, Value& out) {
        using Fn = void(PS4_SYSV_ABI*)(void*, const void*, const char*, int);
        reinterpret_cast<Fn>(At(FindChildFn))(out.Raw(), bytes, path, 0);
    }
    void SetVisible(bool visible) {
        const Addr vtable = Field<Addr>(bytes, 0);
        const Addr set = vtable ? Get<Addr>(vtable + 8) : 0;
        if (DisplayObject() && IsGameFunction(set)) {
            using Fn = void(PS4_SYSV_ABI*)(void*, int);
            reinterpret_cast<Fn>(set)(bytes, visible ? 1 : 0);
        }
    }

private:
    alignas(16) std::uint8_t bytes[0x80]{};
};

struct Rect {
    float x1, y1, x2, y2;
    float CenterX() const {
        return (x1 + x2) * 0.5f;
    }
};

/// Shown: it and every parent visible.
bool Shown(Addr object) {
    for (int depth = 0; object && depth < 32; ++depth) {
        if (!(Get<std::uint8_t>(object + 0x6b) & 0x40)) {
            return false;
        }
        object = Get<Addr>(object + 0x38);
    }
    return true;
}

/// Whether stage point (x, y) is in the display object (its bounds grown by `grow` pixels);
/// `stage` gets its bounds on the stage.
bool Contains(Addr object, float x, float y, Rect* stage, float grow = 0.0f) {
    const Addr get_bounds = VirtualFunction(object, 0x1f0);
    if (!get_bounds) {
        return false;
    }
    alignas(16) const float identity[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    alignas(16) float local[4] = {};
    using BoundsFn = void(PS4_SYSV_ABI*)(float*, void*, const float*);
    reinterpret_cast<BoundsFn>(get_bounds)(local, reinterpret_cast<void*>(object), identity);
    if (!(local[2] > local[0]) || !(local[3] > local[1])) {
        return false;
    }
    alignas(16) float m[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    using MatrixFn = void(PS4_SYSV_ABI*)(void*, float*);
    reinterpret_cast<MatrixFn>(At(WorldMatrixFn))(reinterpret_cast<void*>(object), m);
    constexpr float Twips = 20.0f;
    if (stage) {
        *stage = {1e9f, 1e9f, -1e9f, -1e9f};
        for (const float lx : {local[0], local[2]}) {
            for (const float ly : {local[1], local[3]}) {
                const float sx = (m[0] * lx + m[1] * ly + m[3]) / Twips;
                const float sy = (m[4] * lx + m[5] * ly + m[7]) / Twips;
                *stage = {std::min(stage->x1, sx), std::min(stage->y1, sy),
                          std::max(stage->x2, sx), std::max(stage->y2, sy)};
            }
        }
    }
    // The point in the object's own space (the inverse of the matrix's 2x2 part).
    const float det = m[0] * m[5] - m[1] * m[4];
    if (std::fabs(det) < 1e-12f) {
        return false;
    }
    const float px = x * Twips - m[3], py = y * Twips - m[7];
    const float lx = (m[5] * px - m[1] * py) / det;
    const float ly = (m[0] * py - m[4] * px) / det;
    const float g = grow * Twips / std::sqrt(std::fabs(det));
    return lx >= local[0] - g && lx <= local[2] + g && ly >= local[1] - g && ly <= local[3] + g;
}

/// A list component of the game.
class List {
public:
    explicit List(std::uint8_t* component) : c{component} {}

    int CursorIndex() const {
        const int forced = Field<std::int32_t>(c, CursorOverride);
        return forced >= 0 ? forced : Field<std::int32_t>(c, Cursor);
    }
    /// A one-line spinner (one item shown of several: left and right change it).
    bool IsSpinner() const {
        return Field<std::int32_t>(c, Lines) == 1 && Field<std::int32_t>(c, PerLine) == 1 &&
               Field<std::int32_t>(c, Count) > 1;
    }
    /// Its root display object (spinners).
    Addr Root() const {
        if ((Field<std::uint32_t>(c, RootType) & 0x8f) != 10) {
            return 0;
        }
        const Addr data = Field<Addr>(c, RootData);
        return data ? Get<Addr>(data + 0x88) : 0;
    }

    /// The shown item under stage point (x, y): its index, or -1; `slot` gets its cell and line.
    int ItemAt(float x, float y, std::uint64_t& slot) {
        const int count = Field<std::int32_t>(c, Count);
        const int per_line = Field<std::int32_t>(c, PerLine);
        const int lines = Field<std::int32_t>(c, Lines);
        if (count <= 0 || per_line <= 0 || lines <= 0 || per_line > 64 || lines > 64 ||
            !MakerReady()) {
            return -1;
        }
        const int stride = lines != 1 ? per_line : std::max(per_line, count);
        const int line_scroll = Field<std::int32_t>(c, Scroll);
        const int cell_scroll = Field<std::int32_t>(c, LineScroll);
        const bool wraps = Field<std::uint8_t>(c, Wraps) != 0;
        int best = -1;
        float best_distance = 0.0f;
        for (int line = 0; line < lines; ++line) {
            for (int cell = 0; cell < per_line; ++cell) {
                int index = (line_scroll + line) * stride + cell_scroll + cell;
                if (wraps) {
                    index %= count;
                }
                if (index < 0 || index >= count) {
                    continue;
                }
                const std::uint64_t at = std::uint32_t(cell) | (std::uint64_t(std::uint32_t(line)) << 32);
                Value item;
                MakeItem(at, item);
                const Addr object = item.DisplayObject();
                if (!object || !Shown(object)) {
                    continue;
                }
                Rect bounds{};
                if (!ItemContains(item, object, x, y, bounds)) {
                    continue;
                }
                const float dx = bounds.CenterX() - x, dy = (bounds.y1 + bounds.y2) * 0.5f - y;
                if (best < 0 || dx * dx + dy * dy < best_distance) {
                    best = index;
                    best_distance = dx * dx + dy * dy;
                    slot = at;
                }
            }
        }
        return best;
    }

    /// The button a click at (x, y) on item `slot` presses: a slider's arrow or a combo box's
    /// side, else confirm.
    std::uint32_t ClickButton(std::uint64_t slot, float x, float y, std::uint32_t confirm) {
        if (!MakerReady()) {
            return confirm;
        }
        Value item;
        MakeItem(slot, item);
        if (!item.DisplayObject()) {
            return confirm;
        }
        const auto on = [&](const char* path, float grow, Rect* bounds) {
            Value part;
            item.Child(path, part);
            const Addr object = part.DisplayObject();
            return object && Shown(object) && Contains(object, x, y, bounds, grow);
        };
        Rect bounds{};
        if (on("Widgets/Slider/LeftArrow", 14.0f, &bounds)) {
            return Left;
        }
        if (on("Widgets/Slider/RightArrow", 14.0f, &bounds)) {
            return Right;
        }
        if (on("Widgets/ComboBox", 0.0f, &bounds)) {
            return x < bounds.CenterX() ? Left : Right;
        }
        return confirm;
    }

    /// Moves the game's cursor to `index` (item `slot`) as its own input would: the scroll, the
    /// cursor shown, the listener told, the sound. False: it did not move there.
    bool MoveTo(int index, std::uint64_t slot) {
        const int before = CursorIndex();
        if (index == before) {
            return true;
        }
        const int line_scroll = Field<std::int32_t>(c, LineScroll);
        const int scroll = Field<std::int32_t>(c, Scroll);
        using SetCursor = std::int32_t(PS4_SYSV_ABI*)(std::uint8_t*, int);
        const int scrolled = reinterpret_cast<SetCursor>(At(SetCursorFn))(c, index);
        const int d_line = line_scroll - Field<std::int32_t>(c, LineScroll);
        const int d_scroll = scroll - Field<std::int32_t>(c, Scroll);
        if (scrolled == 1 && (d_line || d_scroll)) {
            // The list scrolled to it: back where the pointer is, with the cursor on that item.
            using ScrollBy = void(PS4_SYSV_ABI*)(std::uint8_t*, int, int);
            using HideCursors = void(PS4_SYSV_ABI*)(std::uint8_t*);
            reinterpret_cast<ScrollBy>(At(ScrollByFn))(c, d_line, d_scroll);
            reinterpret_cast<HideCursors>(At(HideCursorsFn))(c);
            if (MakerReady()) {
                Value item, cursor;
                MakeItem(slot, item);
                item.Child(reinterpret_cast<const char*>(At(CursorName)), cursor);
                cursor.SetVisible(true);
            }
        }
        const int now = CursorIndex();
        if (now != before) {
            CursorMoved(now);
        }
        return now == index;
    }

private:
    bool MakerReady() {
        provider = Field<Addr>(c, Provider);
        make_item = provider ? VirtualFunction(provider, 0x18) : 0;
        return make_item != 0;
    }
    void MakeItem(std::uint64_t slot, Value& out) {
        using Fn = void(PS4_SYSV_ABI*)(void*, void*, const std::uint64_t*);
        reinterpret_cast<Fn>(make_item)(out.Raw(), reinterpret_cast<void*>(provider), &slot);
    }
    /// The item's own cursor area if it has one, else the item; its combo box and slider too.
    bool ItemContains(Value& item, Addr object, float x, float y, Rect& bounds) {
        Value cursor;
        item.Child(reinterpret_cast<const char*>(At(CursorName)), cursor);
        const Addr area = cursor.DisplayObject();
        if (Contains(area ? area : object, x, y, &bounds)) {
            return true;
        }
        for (const char* path : {"Widgets/ComboBox", "Widgets/Slider"}) {
            Value widget;
            item.Child(path, widget);
            const Addr part = widget.DisplayObject();
            if (part && Shown(part) && Contains(part, x, y, &bounds)) {
                return true;
            }
        }
        return false;
    }
    void CursorMoved(int index) {
        const Addr listener = Field<Addr>(c, Listener);
        if (const Addr moved = listener ? VirtualFunction(listener, 0x10) : 0) {
            std::int32_t at = index;
            using Fn = void(PS4_SYSV_ABI*)(void*, std::int32_t*);
            reinterpret_cast<Fn>(moved)(reinterpret_cast<void*>(listener), &at);
        }
        // The cursor's sound, unless one is already queued.
        auto* sound = reinterpret_cast<std::int32_t*>(At(MenuSound));
        if (sound[1] <= 0) {
            sound[0] = 0;
            sound[1] = 0;
            *reinterpret_cast<Addr*>(At(MenuSound) + 8) = At(CursorMoveSound);
        }
    }

    std::uint8_t* c;
    Addr provider = 0, make_item = 0;
};

// The pointer (window thread) and its presses (pad thread), for the game's thread.
std::atomic<float> pointer_x{0.0f}, pointer_y{0.0f};
std::atomic<bool> pointer_in{false};
std::atomic<bool> click{false};            ///< a left click not yet used
std::atomic<std::int64_t> click_ms{0};
std::atomic<bool> left_held{false};
std::atomic<std::uint32_t> presses{0};     ///< buttons to press once (pad thread)
std::atomic<std::uint32_t> repeat_press{0};///< an arrow held under the pointer
std::atomic<bool> pad_steering{false};     ///< the pad's directions are in use: hover waits
std::atomic<std::int64_t> list_focus_ms{0};

constexpr std::int64_t ClickLifeMs = 250;

bool InputBlocked() {
    const Addr flow = Get<Addr>(At(MenuFlow));
    if (flow && Get<std::uint64_t>(flow + 0xa78)) {
        return true;
    }
    const Addr block = Get<Addr>(At(InputBlock));
    return block && Get<std::uint32_t>(block + 0x2e4);
}

bool InMenu() {
    const Addr flow = Get<Addr>(At(MenuFlow));
    const bool title = flow && !Get<std::uint8_t>(flow + 0xb11, 1);
    const bool menu = flow && Get<Addr>(flow + 0xa88);
    // The menu object going away is the menu closing: the list-focus tail (400 ms, for pop-up
    // lists that have no menu object) would keep the pointer ~0.4 s over the game after it.
    static std::atomic<bool> had_menu{false};
    if (had_menu.exchange(menu, std::memory_order_relaxed) && !menu) {
        list_focus_ms.store(0, std::memory_order_relaxed);
    }
    return title || menu || NowMs() - list_focus_ms.load(std::memory_order_relaxed) < 400;
}

std::uint32_t ConfirmButton() {
    return Get<std::int32_t>(At(Region), -1) == 5 ? Circle : Cross;
}

std::uint32_t BackButton() {
    return ConfirmButton() == Circle ? Cross : Circle;
}

bool TakeClick() {
    if (!click.load(std::memory_order_relaxed)) {
        return false;
    }
    if (NowMs() - click_ms.load(std::memory_order_relaxed) > ClickLifeMs) {
        click.store(false, std::memory_order_relaxed);
        return false;
    }
    return true;
}

/// Where the pointer was over each list the last time it was seen: hovering acts on a move only,
/// so a pointer resting somewhere does not fight the keyboard or the pad.
struct Seen {
    const std::uint8_t* list = nullptr;
    float x = 0.0f, y = 0.0f;
    std::int64_t ms = 0;
};
Seen seen[32];
unsigned seen_next = 0;

/// Whether the pointer moved over `list` since it was last seen (a list not seen for a while
/// counts as not moved: it just opened under the pointer).
bool PointerMoved(const std::uint8_t* list, float x, float y) {
    const std::int64_t now = NowMs();
    for (Seen& s : seen) {
        if (s.list == list) {
            const bool moved = now - s.ms <= 250 && (s.x != x || s.y != y);
            s = {list, x, y, now};
            return moved;
        }
    }
    seen[seen_next++ % 32] = {list, x, y, now};
    return false;
}

/// The game's list update (0x1acff30) is about to run for `component`.
void PS4_SYSV_ABI OnListUpdate(std::uint64_t, const Core::GuestRegisterSnapshot* registers) {
    auto* component = reinterpret_cast<std::uint8_t*>(registers->rdi);
    const auto* input = reinterpret_cast<const std::uint8_t*>(registers->rsi);
    if (!component || !input || !*input || InputBlocked()) {
        return;
    }
    list_focus_ms.store(NowMs(), std::memory_order_relaxed);
    if (BbOverlay::CapturesInput() || !pointer_in.load(std::memory_order_relaxed)) {
        return;
    }
    const float x = pointer_x.load(std::memory_order_relaxed);
    const float y = pointer_y.load(std::memory_order_relaxed);
    const bool moved = PointerMoved(component, x, y);
    const bool clicked = TakeClick();
    List list{component};

    if (list.IsSpinner()) {
        // One value shown: a click on its left half goes left, on the right half right.
        Rect bounds{};
        const Addr root = list.Root();
        if (clicked && root && Shown(root) && Contains(root, x, y, &bounds) &&
            click.exchange(false, std::memory_order_relaxed)) {
            presses.fetch_or(x < bounds.CenterX() ? Left : Right, std::memory_order_relaxed);
        }
        return;
    }
    if ((!moved && !clicked) || pad_steering.load(std::memory_order_relaxed)) {
        return;
    }
    std::uint64_t slot = 0;
    const int index = list.ItemAt(x, y, slot);
    if (index < 0) {
        return;
    }
    if (clicked) {
        click.store(false, std::memory_order_relaxed);
    }
    static const bool trace = bb_env("BB_MENU_POINTER_FILE") != nullptr;
    if (trace) {
        std::printf("Menu pointer: list %p item %d (cursor %d)%s\n", static_cast<void*>(component),
                    index, list.CursorIndex(), clicked ? " click" : "");
    }
    if (list.MoveTo(index, slot) && clicked) {
        const std::uint32_t button = list.ClickButton(slot, x, y, ConfirmButton());
        presses.fetch_or(button, std::memory_order_relaxed);
        // A held click on a slider's arrow keeps moving it.
        if (button == Left || button == Right) {
            repeat_press.store(button, std::memory_order_relaxed);
        }
    }
}

/// Window coordinates to the stage's (1920x1080, centred in the shown frame, scaled by height).
void SetPointer(SDL_Window* window, float wx, float wy) {
    int w = 0, h = 0;
    SDL_GetWindowSize(window, &w, &h);
    if (w <= 0 || h <= 0) {
        return;
    }
    const float aspect = std::max(bbgpu_frame_aspect(), 0.1f);
    const float frame_h = std::min(float(h), float(w) / aspect);
    const float scale = frame_h / 1080.0f;
    pointer_x.store((wx - w * 0.5f) / scale + 960.0f, std::memory_order_relaxed);
    pointer_y.store((wy - h * 0.5f) / scale + 540.0f, std::memory_order_relaxed);
    pointer_in.store(true, std::memory_order_relaxed);
}

int PointerWanted() {
    // BB_MENU_POINTER_LOG=1: each of the signals InMenu is made of, with the time of every change.
    static const bool log = bb_env_on("BB_MENU_POINTER_LOG");
    if (log) {
        static int last = -1;
        const Addr flow = Get<Addr>(At(MenuFlow));
        const int title = flow && !Get<std::uint8_t>(flow + 0xb11, 1);
        const int object = flow && Get<Addr>(flow + 0xa88) ? 1 : 0;
        const int focus = NowMs() - list_focus_ms.load(std::memory_order_relaxed) < 400;
        const int state = title | object << 1 | focus << 2;
        if (state != last) {
            last = state;
            std::printf("Menu pointer signals at %lld ms: title %d, menu object %d, list focus %d\n",
                        static_cast<long long>(NowMs()), title, object, focus);
        }
    }
    const bool menu = InMenu();
    static bool was = false;
    if (menu != was) {
        was = menu;
        std::printf("Menu pointer: %s\n", menu ? "a game menu is open (pointer)" : "menus closed");
    }
    return menu ? 1 : 0;
}

/// The window's events (window thread): the pointer, and in a game menu its clicks and wheel.
int OnEvent(const void* sdl_event, void* sdl_window) {
    const auto& event = *static_cast<const SDL_Event*>(sdl_event);
    auto* window = static_cast<SDL_Window*>(sdl_window);
    switch (event.type) {
    case SDL_EVENT_MOUSE_MOTION:
        if (!bbgpu_mouse_captured()) {
            SetPointer(window, event.motion.x, event.motion.y);
        }
        return 0;
    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        pointer_in.store(false, std::memory_order_relaxed);
        return 0;
    default:
        break;
    }
    const bool ours = !bbgpu_mouse_captured() && !BbOverlay::MenuOpen() && InMenu();
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
        left_held.store(false, std::memory_order_relaxed);
        repeat_press.store(0, std::memory_order_relaxed);
        return ours ? 1 : 0;
    }
    if (!ours) {
        return 0;
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        SetPointer(window, event.button.x, event.button.y);
        if (event.button.button == SDL_BUTTON_LEFT) {
            left_held.store(true, std::memory_order_relaxed);
            click_ms.store(NowMs(), std::memory_order_relaxed);
            click.store(true, std::memory_order_relaxed);
        } else if (event.button.button == SDL_BUTTON_RIGHT) {
            presses.fetch_or(BackButton(), std::memory_order_relaxed);
        }
        return 1;
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        return 1;
    }
    if (event.type == SDL_EVENT_MOUSE_WHEEL) {
        const float notches =
            event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -event.wheel.y : event.wheel.y;
        if (notches != 0.0f) {
            presses.fetch_or(notches > 0.0f ? Up : Down, std::memory_order_relaxed);
        }
        return 1;
    }
    return 0;
}

/// BB_MENU_POINTER_FILE=<file> (tests): "x y [click|right|up|down]" in stage pixels, applied
/// when the file changes, as the mouse would.
void ReadTestFile() {
    static const char* path = bb_env("BB_MENU_POINTER_FILE");
    static std::int64_t next_ms = 0;
    static char last[128] = {};
    const std::int64_t now = NowMs();
    if (!path || now < next_ms) {
        return;
    }
    next_ms = now + 200;
    char text[128] = {};
    if (FILE* f = std::fopen(path, "r")) {
        const std::size_t n = std::fread(text, 1, sizeof(text) - 1, f);
        text[n] = 0;
        std::fclose(f);
    }
    if (!text[0] || std::strcmp(text, last) == 0) {
        return;
    }
    std::memcpy(last, text, sizeof(last));
    float x = 0.0f, y = 0.0f;
    char action[16] = {};
    if (std::sscanf(text, "%f %f %15s", &x, &y, action) < 2) {
        return;
    }
    pointer_x.store(x, std::memory_order_relaxed);
    pointer_y.store(y, std::memory_order_relaxed);
    pointer_in.store(true, std::memory_order_relaxed);
    if (!std::strcmp(action, "click")) {
        click_ms.store(now, std::memory_order_relaxed);
        click.store(true, std::memory_order_relaxed);
    } else if (!std::strcmp(action, "right")) {
        presses.fetch_or(BackButton(), std::memory_order_relaxed);
    } else if (!std::strcmp(action, "up") || !std::strcmp(action, "down")) {
        presses.fetch_or(action[0] == 'u' ? Up : Down, std::memory_order_relaxed);
    }
    std::printf("Menu pointer: test %.0f %.0f %s (in a menu: %d)\n", x, y, action, InMenu() ? 1 : 0);
}

/// The pad state the game reads (pad thread): the pointer's presses, each held long enough for
/// the game to see it, with a gap after.
void OnPad(std::uint32_t* buttons, const std::uint8_t* sticks) {
    const auto off_centre = [](std::uint8_t v) { return v < 64 || v > 192; };
    pad_steering.store((*buttons & (Up | Right | Down | Left)) || off_centre(sticks[0]) ||
                           off_centre(sticks[1]),
                       std::memory_order_relaxed);
    ReadTestFile();
    static std::uint32_t holding = 0;
    static std::int64_t until_ms = 0, repeat_ms = 0;
    const std::int64_t now = NowMs();
    if (!InMenu() || BbOverlay::CapturesInput()) {
        presses.store(0, std::memory_order_relaxed);
        repeat_press.store(0, std::memory_order_relaxed);
        holding = 0;
        return;
    }
    constexpr std::int64_t PressMs = 50, GapMs = 40, RepeatDelayMs = 400, RepeatMs = 120;
    if (holding && now >= until_ms) {
        holding = 0;
        until_ms = now + GapMs;
    }
    if (!holding && now >= until_ms) {
        if (const std::uint32_t next = presses.exchange(0, std::memory_order_relaxed)) {
            holding = next;
            until_ms = now + PressMs;
            repeat_ms = now + RepeatDelayMs;
        } else if (const std::uint32_t arrow = repeat_press.load(std::memory_order_relaxed);
                   arrow && left_held.load(std::memory_order_relaxed) && now >= repeat_ms) {
            holding = arrow;
            until_ms = now + PressMs;
            repeat_ms = now + RepeatMs;
        }
    }
    *buttons |= holding;
}

} // namespace

void Install() {
    // BB_MENU_POINTER=0: off.
    if (bb_env_off("BB_MENU_POINTER")) {
        return;
    }
    if (!Core::InstallGuestCodeHook(reinterpret_cast<void*>(At(ListUpdate)), ListUpdateEntry, 0,
                                    OnListUpdate)) {
        std::printf("Menu pointer: the game's list update differs (another build): off\n");
        return;
    }
    bbgpu_add_event_filter(OnEvent);
    bbgpu_add_pointer_query(PointerWanted);
    bbgpu_add_pad_filter(OnPad);
    std::printf("Menu pointer: on (the mouse points and clicks in the game's menus)\n");
}

} // namespace BbMenuPointer
