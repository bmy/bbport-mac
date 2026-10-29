// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/program.h"

#include <filesystem>
#include <map>

#include <fmt/format.h>

#include "common/io_file.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"

namespace Shader::IR {

void DumpProgram(const Program& program, const Info& info, const std::string& type) {
    using namespace Common::FS;
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }
    const auto dir = GetUserPath(PathType::ShaderDir) / "dumps";
    std::filesystem::create_directories(dir);
    const auto name = [&](const char* what) {
        return dir / fmt::format("{}_{:#018x}.{}{}.txt", info.hw_stage, info.pgm_hash, type, what);
    };
    std::map<const Block*, size_t> block_to_index;
    for (const Block* block : program.blocks) {
        block_to_index.emplace(block, block_to_index.size());
    }
    std::map<const Inst*, size_t> inst_to_index;
    size_t inst_index = block_to_index.size();
    const IOFile ir_file{name("irprogram"), FileAccessMode::Create, FileType::TextFile};
    for (const Block* block : program.blocks) {
        ir_file.WriteString(DumpBlock(*block, block_to_index, inst_to_index, inst_index) + '\n');
    }
    const IOFile asl_file{name("asl"), FileAccessMode::Create, FileType::TextFile};
    for (const auto& node : program.syntax_list) {
        asl_file.WriteString(DumpASLNode(node, block_to_index, inst_to_index) + '\n');
    }
}

} // namespace Shader::IR
