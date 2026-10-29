// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: the camera and the HUD for any aspect ratio (camera parameters, HUD markers and
// groups, the viewport), the free camera probe and the native mouse look. Split out of
// gpu/shim/bbport_guest_hooks.cpp (2026-10-26).

#include "common/types.h"
#include "shim/bb_env.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "bbport_guest_code.h"
#include "bbport_heap_sites.h"
#include "bbport_toggles.h"
#include "bbport_fx_init_clear.h"
#include "bbport_effect_flags.h"
#include "platform/gnm/gnm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include "bbport_engine_fixes.h"
#include "bbport_guest_code_hooks.h"
#include <cmath>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/uio.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include <unistd.h>
#include <utility>
#include "games/bloodborne/hooks_internal.h"
#include "remote/bb_remote.h"

namespace BbGuestHooks {
namespace {

/// The free camera (the community patch "Restore Debug Camera", Lance McDonald's: hold Cross and
/// press L3). Its code (0x1de3329..) keeps the mode in the camera manager (r12): byte +0xc4 is 1
/// while the free camera is on (+0x150 of r13: the player is locked in the first mode, free in
/// the second). The probe copies r12 out at the check that runs every update of that code
/// (0x1de344c: cmp byte [r12 + 0xc4], 1); the mouse then drives the free camera's right stick
/// instead of the follow camera's own turn. Only installed where the patch is in place.
alignas(8) std::atomic<u64> free_camera_manager{0};

/// Aspect ratio on the fly. The camera manager (0x1436b..) copies the active camera's projection
/// parameters into the current camera every frame (+0x50 vertical field of view, +0x54 aspect
/// ratio, +0x58 near, +0x5c far: four plain copies, and a blend of two cameras, one of them per
/// update); the projection (0x1d16830), the culling and the HUD's markers read them there. After the
/// copy the output's aspect ratio replaces the game's 1.777 when one is set (SetDisplayAspect): a
/// wider output sees more to the sides at the same height; a taller one keeps the width and sees
/// more above and below (the field of view widened: 2 atan(tan(fov / 2) * 16:9 / aspect), x87).
/// Nothing of the game's own code is changed but the jump.
alignas(8) std::atomic<std::uint32_t> camera_aspect_bits{0}; ///< float bits; 0: the game's
alignas(8) std::atomic<u64> camera_state{0};                   ///< the current camera (rbx there)
struct alignas(8) FovScale {
    std::atomic<std::uint32_t> bits{0}; ///< float: the tangent's factor; 0: the game's field of view
    float half = 0.5f;
};
FovScale camera_fov_scale;

struct alignas(16) MarkerScale {
    float game[4];                     ///< the game's constant
    std::atomic<std::uint32_t> now[4]; ///< what the detour multiplies by
};
std::array<MarkerScale, MarkerSites.size()> marker_scales{};
std::atomic<int> marker_sites{0};
std::atomic<float> marker_kx{1.0f}, marker_ky{1.0f}; ///< SetDisplayAspect's, kept for a later install

void UpdateMarkerScales() {
    const float kx = marker_kx.load(std::memory_order_relaxed), ky = marker_ky.load(std::memory_order_relaxed);
    for (MarkerScale& m : marker_scales) {
        const float now[4] = {m.game[0] * kx, m.game[1] * ky, m.game[2], m.game[3]};
        for (int c = 0; c < 4; ++c) {
            std::uint32_t v;
            std::memcpy(&v, &now[c], 4);
            m.now[c].store(v, std::memory_order_relaxed);
        }
    }
}

/// `vmulps disp(%rip), %xmm0, %xmm0; vaddps disp(%rip), %xmm0, %xmm0` reading `scale`, `bias`.
bool MarkerCodeAt(u64 offset, u64 scale, u64 bias) {
    const auto* p = reinterpret_cast<const u8*>(ImageBase + offset);
    const auto reads = [&](int at, u8 opcode, u64 target) {
        std::int32_t disp;
        std::memcpy(&disp, p + at + 4, 4);
        return p[at] == 0xc5 && p[at + 1] == 0xf8 && p[at + 2] == opcode && p[at + 3] == 0x05 &&
               offset + at + 8 + disp == target;
    };
    return reads(0, 0x59, scale) && reads(8, 0x58, bias);
}

/// The HUD anchored to the output's edges. The game lays its HUD (menu/fe.gfx, Scaleform) out for
/// 1920x1080: on a wider output the port draws that screen in the middle, and its edges would stay
/// there. The HUD's groups are moved instead, as the game moves its own clips: through the
/// movie's members (GFx::Value, ObjectInterface::SetMember "x"/"y") from the thread the HUD lives
/// on. The bars, items and gauges (PlayerHUD) go to the left edge, blood echoes and insight
/// (PlayerHUD.Soul, .HeroPoint) and the autosave icon to the right; on a taller output the top
/// ones go up and the boss bar down. BB_HUD_ANCHOR=0 keeps them in the 16:9 middle.
/// GFx::Value as this build has it (the game's own copies: 0x1adb830, its path lookup 0x1f68aa5).
/// A managed value is linked into its ObjectInterface's list (AddRef 0x2f15b0, Release 0x2f1640).
struct GfxValue {
    u64 links[2] = {~u64(0), ~u64(0)};
    void* object_interface = nullptr;
    std::uint32_t type = 0, pad = 0;
    union {
        double number;
        void* data;
    } value{};
    u64 aux = 0;
};
static_assert(sizeof(GfxValue) == 0x30);
constexpr std::uint32_t GfxManaged = 0x40, GfxNumber = 5, GfxDisplayObject = 10;
/// ObjectInterface's methods in Scaleform 4's order: AddRef, Release, HasMember, GetMember
/// (the slot the game's path lookup calls), SetMember.
constexpr std::size_t SlotAddRef = 0x10, SlotRelease = 0x18, SlotGetMember = 0x28, SlotSetMember = 0x30;
using GfxRefFn = void (PS4_SYSV_ABI*)(void* oi, GfxValue* value, void* data);
using GfxGetFn = bool (PS4_SYSV_ABI*)(void* oi, void* data, const char* name, GfxValue* out, bool isdobj);
using GfxSetFn = bool (PS4_SYSV_ABI*)(void* oi, void* data, const char* name, const GfxValue* value, bool isdobj);
template <typename F>
F GfxMethod(void* oi, std::size_t slot) {
    return *reinterpret_cast<F*>(*reinterpret_cast<u8**>(oi) + slot);
}
bool GfxIsObject(const GfxValue& v) {
    return v.object_interface && (v.type & 0x8c) == 8;
}
void GfxRelease(GfxValue& v) {
    if (v.object_interface && (v.type & GfxManaged)) {
        GfxMethod<GfxRefFn>(v.object_interface, SlotRelease)(v.object_interface, &v, v.value.data);
    }
    v = GfxValue{};
}
/// `object`'s member `name` into `out` (released by the caller), false if none.
bool GfxGet(const GfxValue& object, const char* name, GfxValue& out) {
    out = GfxValue{};
    if (!GfxIsObject(object)) return false;
    const bool isdobj = (object.type & 0x8f) == GfxDisplayObject;
    return GfxMethod<GfxGetFn>(object.object_interface, SlotGetMember)(object.object_interface,
                                                                       object.value.data, name, &out, isdobj);
}
bool GfxGetNumber(const GfxValue& object, const char* name, double& number) {
    GfxValue v;
    const bool ok = GfxGet(object, name, v) && (v.type & 0x8f) == GfxNumber;
    if (ok) number = v.value.number;
    GfxRelease(v);
    return ok;
}
bool GfxSetNumber(const GfxValue& object, const char* name, double number) {
    if (!GfxIsObject(object)) return false;
    GfxValue v;
    v.type = GfxNumber;
    v.value.number = number;
    const bool isdobj = (object.type & 0x8f) == GfxDisplayObject;
    return GfxMethod<GfxSetFn>(object.object_interface, SlotSetMember)(object.object_interface,
                                                                       object.value.data, name, &v, isdobj);
}

/// The game's in-game screen (vtable 0x539c988) updates once a frame (0x1bf9e10: rdi the screen,
/// xmm0 the frame time) and makes its HUD there on first use: the HUD (a value wrapper, its
/// movie's root GFx::Value at +0x30) at screen + 0xa80. The groups are moved from that update,
/// through values got and released within it: nothing of a movie is kept past its frame (the HUD
/// movie is unloaded with areas, and its ObjectInterface with it).
constexpr u64 HudFrameSite = 0x1bf9e10;
constexpr unsigned char HudFrameCode[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp; mov rbp, rsp; push r15
constexpr u64 ScreenHudOffset = 0xa80;
constexpr u64 HudWrapperVtable = 0x539a070; ///< the value wrapper's (0x539a060 + 0x10)
std::atomic<float> hud_dx{0.0f}, hud_dy{0.0f}; ///< the shift for an edge, in the HUD's pixels

/// A HUD group: its parent (0 root, 1 PlayerHUD), name and how it follows the edges (x by dx,
/// y by dy), and where the game put it.
struct HudItem {
    int parent;
    const char* name;
    float fx, fy;
    double base_x = 0, base_y = 0;
    double set_x = 0, set_y = 0; ///< what the port set last
    bool known = false;
};
std::array<HudItem, 5> hud_items = {{
    {0, "PlayerHUD", -1.0f, -1.0f},
    {1, "Soul", 2.0f, 0.0f}, // in PlayerHUD: twice, for its move to the left
    {1, "HeroPoint", 2.0f, 0.0f},
    {0, "AutoSaveIcon", 1.0f, -1.0f},
    {0, "BossTop", 0.0f, 1.0f},
}};

/// Each frame, from the game's screen update: the HUD groups where the output's edges are.
void PS4_SYSV_ABI HudFrame(u64 screen) {
    const u64 hud = screen ? *reinterpret_cast<const u64*>(screen + ScreenHudOffset) : 0;
    if (!hud) {
        return;
    }
    if (*reinterpret_cast<const u64*>(hud) != ImageBase + HudWrapperVtable) {
        static bool told = false;
        if (!told) {
            told = true;
            std::printf("HUD: unexpected object at the screen's HUD: not anchored\n");
        }
        return;
    }
    const GfxValue& root = *reinterpret_cast<const GfxValue*>(hud + 0x30);
    static u64 last_hud = 0;
    static unsigned frames = 0;
    static float applied_dx = 0.0f, applied_dy = 0.0f;
    if (hud != last_hud) {
        last_hud = hud; // a new HUD (another area): where the game put its groups is read again
        for (HudItem& item : hud_items) item.known = false;
        static bool told = false;
        if (!told) {
            told = true;
            std::printf("HUD: anchored to the output's edges%s\n", hud_anchor_on ? "" : " off (BB_HUD_ANCHOR=0)");
        }
    }
    const float dx = hud_anchor_on ? hud_dx.load(std::memory_order_relaxed) : 0.0f;
    const float dy = hud_anchor_on ? hud_dy.load(std::memory_order_relaxed) : 0.0f;
    bool all_known = true;
    for (const HudItem& item : hud_items) all_known = all_known && item.known;
    // Rechecked every half second: the game may put a group back itself.
    if (dx == applied_dx && dy == applied_dy && all_known && ++frames % 32 != 0) {
        return;
    }
    if (!GfxIsObject(root)) {
        return;
    }
    GfxValue player;
    GfxGet(root, "PlayerHUD", player);
    for (HudItem& item : hud_items) {
        GfxValue clip;
        if (!GfxGet(item.parent ? player : root, item.name, clip) || !GfxIsObject(clip)) {
            GfxRelease(clip);
            continue;
        }
        double x = 0, y = 0;
        if (GfxGetNumber(clip, "x", x) && GfxGetNumber(clip, "y", y)) {
            // Positions are in twips (1/20 pixel): a read back is that close.
            const auto moved = [](double a, double b) { return std::abs(a - b) > 0.06; };
            if (!item.known || moved(x, item.set_x) || moved(y, item.set_y)) {
                item.base_x = x; // first seen, or moved by the game since
                item.base_y = y;
                item.known = true;
            }
            const double nx = item.base_x + item.fx * dx, ny = item.base_y + item.fy * dy;
            if (moved(nx, x)) GfxSetNumber(clip, "x", nx);
            if (moved(ny, y)) GfxSetNumber(clip, "y", ny);
            double check_x = nx, check_y = ny;
            GfxGetNumber(clip, "x", check_x);
            GfxGetNumber(clip, "y", check_y);
            item.set_x = check_x;
            item.set_y = check_y;
            if (moved(check_x, nx) || moved(check_y, ny)) {
                static bool told = false;
                if (!told) {
                    told = true;
                    std::printf("HUD: %s did not take its place (%.1f, %.1f instead of %.1f, %.1f)\n",
                                item.name, check_x, check_y, nx, ny);
                }
            }
        }
        GfxRelease(clip);
    }
    GfxRelease(player);
    applied_dx = dx;
    applied_dy = dy;
}

/// The game's UI (Scaleform movies) for the output's aspect ratio. The menu system sets each movie's
/// viewport every frame (0x1f59100 -> MovieImpl::SetViewport 0x1f510, a GFx::Viewport on its
/// stack: buffer 1920x1080, the movie's view mode ShowAll, centred). Given the output's pixel
/// aspect ratio (Viewport::AspectRatio: (output width / height) / 16:9), a movie shows its 16:9
/// screen in the middle of a wider (taller) view, squeezed into the 1920x1080 frame, which the
/// output stretches back (UiComposition::Scale): what lies past the game's screen (the HUD's
/// groups at the edges) is drawn, not culled. BB_RE_VIEWPORT=1 logs the calls.
constexpr u64 SetViewportSite = 0x1f510;
std::atomic<float> gfx_pixel_aspect{1.0f};
void PS4_SYSV_ABI ViewportAdjust(u64 movie, u64 viewport, u64 caller) {
    auto* v = reinterpret_cast<std::int32_t*>(viewport);
    static const bool trace = bb_env_on("BB_RE_VIEWPORT");
    if (trace) {
        static std::atomic<int> calls{0};
        float scale, aspect;
        std::memcpy(&scale, v + 11, 4);
        std::memcpy(&aspect, v + 12, 4);
        if (++calls <= 24) {
            std::printf("Viewport: movie %#llx from %#llx: buffer %dx%d at %d,%d %dx%d scissor %d,%d %dx%d "
                        "flags %#x scale %.3f aspect %.3f; mode %d align %d\n",
                        (unsigned long long)movie, (unsigned long long)(caller - ImageBase), v[0], v[1], v[2],
                        v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], double(scale), double(aspect),
                        *reinterpret_cast<const int*>(movie + 0xe8), *reinterpret_cast<const int*>(movie + 0xec));
        }
    }
    // The game's full-screen movies (1920x1080 in a 1920x1080 frame) only.
    if (v[0] == 1920 && v[1] == 1080 && v[2] == 0 && v[3] == 0 && v[4] == 1920 && v[5] == 1080) {
        const float aspect = gfx_pixel_aspect.load(std::memory_order_relaxed);
        std::memcpy(v + 12, &aspect, 4);
    }
}

/// Native mouse look. The follow camera's update (0x143ac60, rdi the camera) derives its angles
/// from where the camera is around the player each frame, pitch at +0x140 and yaw at +0x144; in
/// its normal mode 0x143cde0..0x143cdf0 makes xmm1 = camera stick * turn speed * frame time, the
/// increment 0x143cdf0 adds to (pitch, yaw) before the game wraps the yaw and clamps the pitch,
/// and the camera is then placed from them. The detour adds the mouse's turn since the last update
/// (radians, taken and cleared atomically) to that increment: the mouse turns the camera as far as
/// it moved, with none of the stick's dead zone, response curve, acceleration or top speed.
/// It also scales the stick's own turn by the menu's camera speeds (#208).
constexpr u64 LookSite = 0x143cde0;
constexpr unsigned char LookSiteCode[16] = {0xc5, 0xa8, 0x59, 0xc9,                         // vmulps xmm1, xmm10, xmm1
                                            0xc5, 0xf8, 0x28, 0x95, 0x10, 0xfc, 0xff, 0xff, // vmovaps xmm2, [rbp - 0x3f0]
                                            0xc5, 0xe8, 0x59, 0xc9};                        // vmulps xmm1, xmm2, xmm1
alignas(8) std::atomic<u64> look_delta{0}; ///< (pitch, yaw) as two floats, taken by the detour

std::atomic<long long> look_since_ms{0}; ///< when look_delta last went from empty to not

} // namespace

bool InstallFreeCameraProbe() {
    constexpr u64 offset = 0x1de344c;
    static constexpr std::array<u8, 9> code = {0x41, 0x80, 0xbc, 0x24, 0xc4, 0x00, 0x00, 0x00, 0x01};
    const u64 address = ImageBase + offset;
    if (std::memcmp(reinterpret_cast<const void*>(address), code.data(), code.size())) {
        return false; // the patch is off (or other code there)
    }
    Emitter e;
    e.Bytes({0x9c, 0x50});                         // pushfq; push rax
    e.Bytes({0x48, 0xb8});                         // movabs rax, &free_camera_manager
    e.U64(reinterpret_cast<u64>(&free_camera_manager));
    e.Bytes({0x4c, 0x89, 0x20});                   // mov [rax], r12
    e.Bytes({0x58, 0x9d});                         // pop rax; popfq
    for (const u8 b : code) {
        e.Bytes({b});                              // the check itself
    }
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + code.size());
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

int InstallCameraAspectHooks() {
    int installed = 0;
    for (const auto& site : CameraParamStores) {
        const u64 address = ImageBase + site.offset;
        if (std::memcmp(reinterpret_cast<const void*>(address), site.code.data(), site.code.size())) {
            continue;
        }
        Emitter e;
        for (const u8 b : site.code) {
            e.Bytes({b});
        }
        e.Bytes({0x9c, 0x50});                         // pushfq; push rax
        e.Bytes({0x48, 0xb8});                         // movabs rax, &camera_state
        e.U64(reinterpret_cast<u64>(&camera_state));
        e.Bytes({0x48, 0x89, 0x18});                   // mov [rax], rbx
        e.Bytes({0x48, 0xb8});                         // movabs rax, &camera_aspect_bits
        e.U64(reinterpret_cast<u64>(&camera_aspect_bits));
        e.Bytes({0x8b, 0x00});                         // mov eax, [rax]
        e.Bytes({0x85, 0xc0, 0x74, 0x03});             // test eax, eax; je over the store
        e.Bytes({0x89, 0x43, 0x54});                   // mov [rbx + 0x54], eax
        e.Bytes({0x48, 0xb8});                         // movabs rax, &camera_fov_scale
        e.U64(reinterpret_cast<u64>(&camera_fov_scale));
        e.Bytes({0x83, 0x38, 0x00, 0x74, 0x15});       // cmp dword [rax], 0; je over the x87 part
        e.Bytes({0xd9, 0x43, 0x50});                   // fld dword [rbx + 0x50]: fov
        e.Bytes({0xd8, 0x48, 0x04});                   // fmul dword [rax + 4]: fov / 2
        e.Bytes({0xd9, 0xf2, 0xdd, 0xd8});             // fptan; fstp st0 (its 1.0)
        e.Bytes({0xd8, 0x08});                         // fmul dword [rax]: tan * factor
        e.Bytes({0xd9, 0xe8, 0xd9, 0xf3});             // fld1; fpatan: atan(tan * factor)
        e.Bytes({0xd8, 0xc0});                         // fadd st0, st0
        e.Bytes({0xd9, 0x5b, 0x50});                   // fstp dword [rbx + 0x50]
        e.Bytes({0x58, 0x9d});                         // pop rax; popfq
        e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
        e.U64(address + site.code.size());
        const u64 stub = PlaceStub(e);
        std::uint32_t rel = 0;
        if (!stub || !Rel32To(address + 5, stub, rel)) {
            continue;
        }
        WriteJumpOverTrap(address, 0xe9, rel);
        ++installed;
    }
    return installed;
}

int InstallMarkerHooks() {
    int installed = 0;
    for (std::size_t i = 0; i < MarkerSites.size(); ++i) {
        const MarkerSite& site = MarkerSites[i];
        if (!MarkerCodeAt(site.offset, site.scale, site.bias)) {
            continue;
        }
        MarkerScale& m = marker_scales[i];
        std::memcpy(m.game, reinterpret_cast<const void*>(ImageBase + site.scale), sizeof(m.game));
        UpdateMarkerScales();
        const u64 address = ImageBase + site.offset;
        Emitter e;
        e.Bytes({0x50});                               // push rax
        e.Bytes({0x48, 0xb8});                         // movabs rax, &now
        e.U64(reinterpret_cast<u64>(&m.now));
        e.Bytes({0xc5, 0xf8, 0x59, 0x00});             // vmulps xmm0, xmm0, [rax]
        e.Bytes({0x48, 0xb8});                         // movabs rax, the game's bias
        e.U64(ImageBase + site.bias);
        e.Bytes({0xc5, 0xf8, 0x58, 0x00});             // vaddps xmm0, xmm0, [rax]
        e.Bytes({0x58});                               // pop rax
        e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
        e.U64(address + 16);
        const u64 stub = PlaceStub(e);
        std::uint32_t rel = 0;
        if (!stub || !Rel32To(address + 5, stub, rel)) {
            continue;
        }
        // The rest of the two instructions: traps (the jump returns past them).
        const long page_size = sysconf(_SC_PAGESIZE);
        void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
        mprotect(page, (address + 16) - reinterpret_cast<u64>(page), PROT_READ | PROT_WRITE | PROT_EXEC);
        auto* code = reinterpret_cast<u8*>(address);
        code[0] = 0xcc;
        std::memset(code + 5, 0xcc, 11);
        mprotect(page, (address + 16) - reinterpret_cast<u64>(page), Core::GuestCodeProtection());
        WriteJumpOverTrap(address, 0xe9, rel);
        ++installed;
    }
    marker_sites.store(installed, std::memory_order_relaxed);
    return installed;
}

bool InstallViewportHook() {
    const u64 address = ImageBase + SetViewportSite;
    if (std::memcmp(reinterpret_cast<const void*>(address), HudFrameCode, sizeof(HudFrameCode))) {
        return false;
    }
    Emitter e;
    e.SaveVolatile();
    e.SaveXmm();
    e.Bytes({0x49, 0x8b, 0x54, 0x24, 0x50}); // mov rdx, [r12 + 80]: the return address
    e.CallAbs(reinterpret_cast<const void*>(&ViewportAdjust));
    e.RestoreXmm();
    e.RestoreVolatile();
    for (const unsigned char b : HudFrameCode) e.Bytes({b});
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
    e.U64(address + sizeof(HudFrameCode));
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) return false;
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    mprotect(page, (address + 6) - reinterpret_cast<u64>(page), PROT_READ | PROT_WRITE | PROT_EXEC);
    reinterpret_cast<u8*>(address)[0] = 0xcc;
    reinterpret_cast<u8*>(address)[5] = 0xcc;
    mprotect(page, (address + 6) - reinterpret_cast<u64>(page), Core::GuestCodeProtection());
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

/// The detour at the screen update's entry: the registers (xmm0: its frame time) kept around
/// the call, then its first instructions and a jump back.
bool InstallHudHook() {
    const u64 address = ImageBase + HudFrameSite;
    if (std::memcmp(reinterpret_cast<const void*>(address), HudFrameCode, sizeof(HudFrameCode))) {
        return false;
    }
    Emitter e;
    e.SaveVolatile(); // rdi (the screen) is kept as it is
    e.SaveXmm();
    e.CallAbs(reinterpret_cast<const void*>(&HudFrame));
    e.RestoreXmm();
    e.RestoreVolatile();
    for (const unsigned char b : HudFrameCode) e.Bytes({b});
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + sizeof(HudFrameCode));
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) return false;
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    mprotect(page, (address + 6) - reinterpret_cast<u64>(page), PROT_READ | PROT_WRITE | PROT_EXEC);
    reinterpret_cast<u8*>(address)[0] = 0xcc;
    reinterpret_cast<u8*>(address)[5] = 0xcc;
    mprotect(page, (address + 6) - reinterpret_cast<u64>(page), Core::GuestCodeProtection());
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallLookHook() {
    const u64 address = ImageBase + LookSite;
    if (std::memcmp(reinterpret_cast<const void*>(address), LookSiteCode, sizeof(LookSiteCode))) {
        return false;
    }
    Emitter e;
    for (const u8 b : LookSiteCode) {
        e.Bytes({b}); // the three instructions the jump covers (no RIP-relative operand)
    }
    e.Bytes({0x50, 0x51});                         // push rax; push rcx
    // The stick's turn times the menu's camera speeds (#208): (vertical, horizontal, 1, 1).
    e.Bytes({0x48, 0xb8});                         // movabs rax, &camera_speed
    e.U64(reinterpret_cast<u64>(&BbSettings::Get().camera_speed[0]));
    e.Bytes({0xc5, 0xf0, 0x59, 0x08});             // vmulps xmm1, xmm1, [rax]
    e.Bytes({0x48, 0xb8});                         // movabs rax, &look_delta
    e.U64(reinterpret_cast<u64>(&look_delta));
    e.Bytes({0x31, 0xc9});                         // xor ecx, ecx
    e.Bytes({0x48, 0x87, 0x08});                   // xchg [rax], rcx: taken and cleared
    // A turn counts as camera input, as the stick does (0x143b8cc): else the update pulls the
    // pitch back to its rest angle (+0x150) at once, and the rest angle follows the new pitch.
    e.Bytes({0x48, 0x85, 0xc9});                   // test rcx, rcx
    e.Bytes({0x74, 0x1d});                         // je past the three stores (29 bytes)
    e.Bytes({0xc7, 0x85, 0x3c, 0xfc, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00}); // mov dword [rbp - 0x3c4], 1
    e.Bytes({0x41, 0xc6, 0x85, 0x5c, 0x01, 0x00, 0x00, 0x01});             // mov byte [r13 + 0x15c], 1
    e.Bytes({0x41, 0xc7, 0x85, 0x54, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}); // mov dword [r13 + 0x154], 0
    e.Bytes({0xc4, 0xe1, 0xf9, 0x6e, 0xd1});       // vmovq xmm2, rcx (xmm2 is dead here)
    e.Bytes({0xc5, 0xf0, 0x58, 0xca});             // vaddps xmm1, xmm1, xmm2
    e.Bytes({0x59, 0x58});                         // pop rcx; pop rax
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + sizeof(LookSiteCode));
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool FreeCameraActive() {
    const u64 manager = free_camera_manager.load(std::memory_order_relaxed);
    return manager != 0 && *reinterpret_cast<const volatile u8*>(manager + 0xc4) == 1;
}
std::uint64_t CameraState() {
    return camera_state.load(std::memory_order_relaxed);
}

void SetDisplayAspect(float aspect) {
    // bbport (macOS native GPU process): the upscaler is bb-gpu's, the camera and HUD hooks the
    // game process's (MsgDisplayAspect); called on a change only.
    if (BbRemote::BackActive()) {
        BbRemote::Back::DisplayAspect(aspect);
        return;
    }
    constexpr float game = 16.0f / 9.0f;
    const bool other = aspect > 0.5f && aspect < 8.0f && std::abs(aspect - game) > 0.001f;
    std::uint32_t bits = 0, fov = 0;
    float kx = 1.0f, ky = 1.0f; // the markers' scale: the UI is as high (wide) as the output
    if (other) {
        std::memcpy(&bits, &aspect, 4);
        if (aspect < game) {
            ky = game / aspect;
            std::memcpy(&fov, &ky, 4);
        } else {
            kx = aspect / game;
        }
    }
    camera_aspect_bits.store(bits, std::memory_order_relaxed);
    camera_fov_scale.bits.store(fov, std::memory_order_relaxed);
    // The HUD's groups: as far as the output's edge is from the 16:9 middle (the HUD's pixels).
    hud_dx.store(960.0f * (kx - 1.0f), std::memory_order_relaxed);
    hud_dy.store(540.0f * (ky - 1.0f), std::memory_order_relaxed);
    if (const char* env = bb_env("BB_HUD_DX")) { // diagnostics: another shift
        hud_dx.store(std::strtof(env, nullptr), std::memory_order_relaxed);
    }
    gfx_pixel_aspect.store(other ? aspect / game : 1.0f, std::memory_order_relaxed);
    marker_kx.store(kx, std::memory_order_relaxed);
    marker_ky.store(ky, std::memory_order_relaxed);
    UpdateMarkerScales();
    std::printf("Display: aspect ratio %.3f (%s): camera %s, HUD markers at %d places\n", double(aspect),
                other ? (aspect > game ? "wider than 16:9" : "taller than 16:9") : "the game's",
                other ? (aspect > game ? "sees more to the sides" : "sees more above and below")
                      : "as the game has it",
                marker_sites.load(std::memory_order_relaxed));
}

bool CameraLookAvailable() {
    return look_installed.load(std::memory_order_acquire);
}

bool CameraLookAdd(float pitch, float yaw) {
    if (!CameraLookAvailable()) {
        return false;
    }
    // Whether the camera takes the turns: a turn it has not taken for 100 ms (lock-on, menus,
    // cutscenes: its normal update does not run) means not, until it takes one again. Meanwhile
    // the caller drives the stick (lock-on switches targets with the stick) and the turn waiting
    // here is replaced rather than applied all at once later.
    static std::atomic<bool> active{true}, stored{false};
    const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    u64 old = look_delta.load(std::memory_order_acquire);
    if (stored.load(std::memory_order_relaxed)) {
        if (old == 0) {
            active.store(true, std::memory_order_relaxed);
        } else if (now - look_since_ms.load(std::memory_order_relaxed) > 100) {
            active.store(false, std::memory_order_relaxed);
        }
    }
    const bool taking = active.load(std::memory_order_relaxed);
    if (pitch == 0.0f && yaw == 0.0f) {
        return taking;
    }
    for (;;) {
        float p = 0.0f, y = 0.0f;
        if (old != 0 && taking) {
            const std::uint32_t lo = std::uint32_t(old), hi = std::uint32_t(old >> 32);
            std::memcpy(&p, &lo, 4);
            std::memcpy(&y, &hi, 4);
        }
        p += pitch;
        y += yaw;
        std::uint32_t lo, hi;
        std::memcpy(&lo, &p, 4);
        std::memcpy(&hi, &y, 4);
        const u64 next = u64(lo) | (u64(hi) << 32);
        if (look_delta.compare_exchange_weak(old, next, std::memory_order_acq_rel)) {
            if (old == 0 || !taking) {
                look_since_ms.store(now, std::memory_order_relaxed);
            }
            stored.store(true, std::memory_order_relaxed);
            return taking;
        }
    }
}

} // namespace BbGuestHooks
