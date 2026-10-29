// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne (1.09): BBSC's hunter's kit undone when BBSC is off. With BBSC on, its kit (Hunt
// Bell, Kin Bell, Bloodlust Bell, Duel Bell, Parting Blank) replaces the vanilla bells and the
// Silencing Blank, which it takes out of the character's items and storage box; that is in the
// save. Without BBSC the kit items do nothing and the vanilla ones are gone, so a player who tries
// BBSC and turns it off again could no longer play online the normal way. Here, in every run
// without BBSC (offline too: the online module is not loaded then), the kit is taken out of a
// loaded character and the vanilla items go back: what BBSC wrote in its journal when it took
// them, else one vanilla item for each kit item. The save is backed up first.
#pragma once

#include <cstdint>

namespace BbKitRestore {

/// Hooks the game's tick (the one BBSC uses, free without it). Nothing happens with BBSC on
/// (BB_ONLINE=1 BB_BBSC=1), with BB_KIT_RESTORE=0, or when the game's bytes differ.
void Install();

} // namespace BbKitRestore

// The journal BBSC keeps of what it takes out (bloodborne_re.cpp, TickBbscDuelBellGive), one file
// per character save: <user>/bbsc/kit-journal/<user id>-<slot>.txt.
extern "C" {
/// A new kit: whatever an earlier journal of this character said is stale.
void bbport_kit_journal_reset(void);
/// BBSC took `count` of goods `goods` out of the carried items (storage 0) or the storage box.
void bbport_kit_journal_item(std::uint32_t goods, std::int32_t count, int storage);
/// BBSC set event flag `flag` (6622: the Dream messenger's Beckoning Bell counts as given).
void bbport_kit_journal_flag(std::uint32_t flag);
}
