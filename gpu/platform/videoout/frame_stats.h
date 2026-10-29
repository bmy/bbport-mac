// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the frame statistics report (BB_FRAME_STATS), taken at each flip.
#pragma once

namespace BbVideoOut {
bool FrameStatsEnabled();
void OnFlipStats();
} // namespace BbVideoOut
