// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne (update 1.09): what the translator knows about its code. Addresses are those of the
// 1.09 executable (game_check.py makes sure that is the one running); every retail release runs
// the same one.
#include "game_profile.h"
#include "bbport_guest_hooks.h"
#include "bloodborne_kit_restore.h"
#include "bloodborne_menu_pointer.h"
#include "remote/bb_remote.h"
#include <cstdio>

namespace Game {
namespace {
constexpr std::uint64_t ImageBase = 0x800000000ull;

/// The retail releases (scripts/game_check.py SUPPORTED_TITLES): Bloodborne in the US, EU, UK, JP
/// and Asia, then the editions with The Old Hunters (US, EU, JP, Asia). Without its serial here a
/// region ran on the generic paths alone, and the new memory model crashed on entering the world
/// (issue #119, the US release: a null read in the allocator, 0x2085xxx). Its bookkeeping is in
/// GPU-mapped pages; DynamicWriters below keeps them in place from the first write (the hooks
/// alone do not matter for it: BB_GUEST_HOOKS=0 runs; BB_GAME_PROFILE=none crashes).
constexpr const char* Serials[] = {"CUSA00900", "CUSA00207", "CUSA00208", "CUSA00299", "CUSA01363",
                                   "CUSA03179", "CUSA03173", "CUSA03014", "CUSA03023"};

/// Reverse engineering: 0x2ab7350 computes vertices on the CPU (cloth) into a ~128 MiB ring, each
/// block written now and then; 0x20858a0 is the game's allocator, its bookkeeping in GPU-mapped
/// pages.
constexpr CodeRange DynamicWriters[] = {
    {ImageBase + 0x2ab7350, ImageBase + 0x2aba310},
    {ImageBase + 0x20858a0, ImageBase + 0x2085e20},
};

void InstallHooks(const CoreServices& services) {
    // The resource loaders' copies, the GPU memory range allocator and the heap (bbport_guest_hooks).
    BbGuestHooks::Install(services.fresh_range, services.dead_range, services.live_range);
    // The mouse as a pointer in the game's menus (bloodborne_menu_pointer). bbport (macOS native
    // GPU process): its window part would be bb-gpu's and its game hook this process's; not split
    // yet, so the keyboard and gamepad drive the menus there.
    if (BbRemote::FrontActive()) {
        std::printf("Menu pointer: not with the native GPU process yet\n");
    } else {
        BbMenuPointer::Install();
    }
    // BBSC's kit back to the vanilla bells when BBSC is off (bloodborne_kit_restore).
    BbKitRestore::Install();
}
} // namespace

extern const Profile BloodborneProfile = {
    .name = "Bloodborne",
    .serials = Serials,
    .install_hooks = InstallHooks,
    .dynamic_writers = DynamicWriters,
    // The game's buffer-copy compute shader (not the export-stage copy shader).
    .buffer_copy_shader = 0xfefebf9f,
    // The camera's projection stores and the HUD's markers (BbGuestHooks::SetDisplayAspect).
    .camera_aspect = true,
    // No GPU binding seen in them (2026-10-23, BB_HOST_ONLY tests: no arena binding, no page table
    // entry without a source; 1256 MiB less GPU memory, ~0.6 GB less in all): 0 the memory system's
    // own 116 MiB, 1 MAIN, 2 GFX, 6-9 and 11 small heaps, 12 HAVOK. Used by the GPU: 3 SystemShared,
    // 4 GraphicsPrivate, 5 GraphicsPrivateB, 10 (8 MiB, its type 3).
    .host_only_allocations = (1ull << 0) | (1ull << 1) | (1ull << 2) | (1ull << 6) | (1ull << 7) |
                             (1ull << 8) | (1ull << 9) | (1ull << 11) | (1ull << 12),
};

} // namespace Game
