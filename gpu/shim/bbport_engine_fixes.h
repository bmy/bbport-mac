// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: fixes to the game's own code (play time at high frame rates, hit collector cap, save
// character id guard). BB_FIX_PLAY_TIME / BB_FIX_HIT_COLLECTOR / BB_FIX_CHARA_ID=0 turn one off.
#pragma once

namespace BbEngineFixes {
/// Installs the hooks; once, after the game's image is loaded.
void Install();
}
