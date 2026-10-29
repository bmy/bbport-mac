// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>
#include <cstring>
#include <span>

namespace BbEffectFlags {

// Bloodborne 1.09: 0x22c4cb3 clears SSAO when the context's capability bit 3
// (+0x48) is absent. 0x22a3efb uses the target at +0x10d0 via 0xe669a0,
// then dereferences its result without a null check at 0x22a3f07.
// A live preference must not re-enable this pass on a context without its target.
inline bool SsaoEnabled(std::span<const std::uint8_t> context, bool requested) {
    constexpr std::size_t target = 0x10d0;
    if (!requested || context.size() < target + 0x1b9) {
        return false;
    }
    std::uint64_t capabilities = 0;
    std::memcpy(&capabilities, context.data() + 0x48, sizeof(capabilities));
    if (!(capabilities & 8)) {
        return false;
    }
    // The same selector as guest 0xe669a0. A target may not yet be initialized,
    // even when the context is configured to support SSAO.
    const bool alternate = context[target + 0x1b8] && context[target + 0x189];
    std::uint64_t resource = 0;
    std::memcpy(&resource, context.data() + target + (alternate ? 0xc0 : 0x68), sizeof(resource));
    return resource != 0;
}

} // namespace BbEffectFlags
