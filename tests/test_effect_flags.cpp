// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_effect_flags.h"

#include <array>
#include <cassert>
#include <cstdio>

int main() {
    std::array<std::uint8_t, 0x29a0> context{};
    const auto put = [&](std::size_t offset, std::uint64_t value) {
        std::memcpy(context.data() + offset, &value, sizeof(value));
    };
    // A menu/auxiliary context must stay off despite the user's enabled preference.
    assert(!BbEffectFlags::SsaoEnabled(context, true));
    put(0x10d0 + 0x68, 0x1040000000ull);
    assert(!BbEffectFlags::SsaoEnabled(context, true));
    put(0x48, 8);
    assert(BbEffectFlags::SsaoEnabled(context, true));
    assert(!BbEffectFlags::SsaoEnabled(context, false));
    // Follow both selector flags exactly; do not use a nonselected target as fallback.
    context[0x10d0 + 0x1b8] = 1;
    assert(BbEffectFlags::SsaoEnabled(context, true));
    context[0x10d0 + 0x189] = 1;
    assert(!BbEffectFlags::SsaoEnabled(context, true));
    put(0x10d0 + 0xc0, 0x1041000000ull);
    assert(BbEffectFlags::SsaoEnabled(context, true));
    // Removing/reinitializing a target must disable the pass again.
    put(0x10d0 + 0xc0, 0);
    assert(!BbEffectFlags::SsaoEnabled(context, true));
    assert(!BbEffectFlags::SsaoEnabled(std::span(context).first(0x1288), true));
    std::puts("effect-flags: SSAO context capability and target selection OK");
}
