// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport upscale: FSR 3.1 frame generation passes built in this tree (ffx_shaders.cpp).
#pragma once

namespace FfxShaders {
/// Before the frame generation context is made: its pipelines take the passes built here.
void Install();
} // namespace FfxShaders
