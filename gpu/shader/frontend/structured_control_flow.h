// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: structurization of the CFG into an abstract syntax list.
#pragma once

#include "common/object_pool.h"
#include "shader/frontend/cfg.h"
#include "shader/ir/abstract_syntax_list.h"
#include "shader/ir/basic_block.h"

namespace Shader {
struct Pools;
}

namespace Shader::Gcn {

/// The structured program over the CFG's IR blocks; blocks it adds (merges) get flow blocks
/// from `flow_blocks`.
[[nodiscard]] IR::AbstractSyntaxList BuildASL(Pools& pools, CFG& cfg,
                                              Common::ObjectPool<Block>& flow_blocks);

} // namespace Shader::Gcn
