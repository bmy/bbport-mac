// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: GCN program to IR program (translation, SSA, the passes,
// structurization), ready for the SPIR-V backend.
#pragma once

#include <span>

#include "common/object_pool.h"
#include "shader/ir/basic_block.h"
#include "shader/ir/program.h"

namespace Shader {

struct Profile;
struct RuntimeInfo;

/// Instruction and block storage, reused from one compilation to the next.
struct Pools {
    static constexpr u32 InstPoolSize = 8192;
    static constexpr u32 BlockPoolSize = 32;

    Common::ObjectPool<IR::Inst> inst_pool{InstPoolSize};
    Common::ObjectPool<IR::Block> block_pool{BlockPoolSize};

    void ReleaseContents() {
        block_pool.ReleaseContents();
        inst_pool.ReleaseContents();
    }
};

[[nodiscard]] IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools,
                                           Info& info, RuntimeInfo& runtime_info,
                                           const Profile& profile);

} // namespace Shader
