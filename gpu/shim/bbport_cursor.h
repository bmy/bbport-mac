// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a worn silver lancet cursor with a hooked guard, wire-bound grip and dark garnet.
// Rendered from geometry at 1x and 2x for high-DPI displays, with no external image files.
// BB_CURSOR=0 leaves the system's cursor.
#pragma once

struct SDL_Cursor;

namespace BbCursor {

/// A new cursor (SDL_DestroyCursor to free it), or nullptr when it is off or cannot be made.
SDL_Cursor* Create();

} // namespace BbCursor
