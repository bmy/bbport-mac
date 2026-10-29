// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: the blocks reachable from `first_block` in post order (each
// block's po_index set to its position).
#pragma once

#include "shader/ir/basic_block.h"

namespace Shader::IR {

BlockList PostOrder(Block* first_block);

} // namespace Shader::IR
