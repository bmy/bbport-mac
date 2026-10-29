// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: a program being translated: the decoded GCN instructions, the
// blocks (and their post order), the structured form, and the shader info it fills.
#pragma once

#include <string>
#include <vector>

#include "gcn/isa.h"
#include "shader/ir/abstract_syntax_list.h"
#include "shader/ir/basic_block.h"
#include "shader/info.h"

namespace Shader::IR {

struct Program {
    explicit Program(Info& info_) : info{info_} {}

    AbstractSyntaxList syntax_list;
    BlockList blocks;
    BlockList post_order_blocks;
    std::vector<GcnIsa::GcnInst> ins_list;
    u32 next_reg_index{};
    Info& info;
};

/// BB_DUMP_SHADERS: writes the program (blocks and structure) to the shader dump directory,
/// `type` naming the stage of the translation.
void DumpProgram(const Program& program, const Info& info, const std::string& type = "");

} // namespace Shader::IR
