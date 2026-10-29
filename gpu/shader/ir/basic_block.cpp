// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/basic_block.h"

#include <algorithm>

#include <fmt/format.h>

namespace Shader::IR {

Block::Block(Common::ObjectPool<Inst>& inst_pool_) : inst_pool{&inst_pool_} {}

Block::~Block() = default;

void Block::AppendNewInst(Opcode op, std::initializer_list<Value> args) {
    PrependNewInst(end(), op, args);
}

Block::iterator Block::PrependNewInst(iterator insertion_point, const Inst& base_inst) {
    Inst* const inst = inst_pool->Create(base_inst);
    inst->SetParent(this);
    return instructions.insert(insertion_point, *inst);
}

Block::iterator Block::PrependNewInst(iterator insertion_point, Opcode op,
                                      std::initializer_list<Value> args, u32 flags) {
    Inst* const inst = inst_pool->Create(op, flags);
    inst->SetParent(this);
    const auto it = instructions.insert(insertion_point, *inst);
    ASSERT_MSG(inst->NumArgs() == args.size(), "{} arguments for {}", args.size(), op);
    size_t index = 0;
    for (const Value& arg : args) {
        inst->SetArg(index++, arg);
    }
    return it;
}

void Block::AddBranch(Block* block) {
    ASSERT_MSG(!std::ranges::contains(imm_successors, block), "successor added twice");
    ASSERT_MSG(!std::ranges::contains(block->imm_predecessors, this), "predecessor added twice");
    imm_successors.push_back(block);
    block->imm_predecessors.push_back(this);
}

namespace {

std::string BlockName(const std::map<const Block*, size_t>& block_to_index, const Block* block) {
    const auto it = block_to_index.find(block);
    return it != block_to_index.end() ? fmt::format("${}", it->second)
                                      : fmt::format("$?{:x}", reinterpret_cast<uintptr_t>(block));
}

size_t InstNumber(std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index,
                  const Inst* inst) {
    const auto [it, inserted] = inst_to_index.emplace(inst, inst_index + 1);
    inst_index += inserted;
    return it->second;
}

std::string ArgText(std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index,
                    const Value& arg) {
    if (arg.IsEmpty()) {
        return "<null>";
    }
    if (!arg.IsImmediate()) {
        return fmt::format("%{}", InstNumber(inst_to_index, inst_index, arg.Inst()));
    }
    switch (arg.Type()) {
    case Type::U1:
        return arg.U1() ? "#true" : "#false";
    case Type::U8:
        return fmt::format("#{}", arg.U8());
    case Type::U16:
        return fmt::format("#{}", arg.U16());
    case Type::U32:
        return fmt::format("#{}", arg.U32());
    case Type::U64:
        return fmt::format("#{}", arg.U64());
    case Type::F32:
        return fmt::format("#{}", arg.F32());
    case Type::ScalarReg:
        return fmt::format("{}", arg.ScalarReg());
    case Type::VectorReg:
        return fmt::format("{}", arg.VectorReg());
    case Type::VirtualReg:
        return fmt::format("{}", arg.VirtualReg());
    case Type::Attribute:
        return fmt::format("{}", arg.Attribute());
    case Type::Patch:
        return fmt::format("{}", arg.Patch());
    default:
        return "<immediate>";
    }
}

} // namespace

std::string DumpBlock(const Block& block) {
    size_t inst_index = 0;
    std::map<const Inst*, size_t> inst_to_index;
    return DumpBlock(block, {}, inst_to_index, inst_index);
}

std::string DumpBlock(const Block& block, const std::map<const Block*, size_t>& block_to_index,
                      std::map<const Inst*, size_t>& inst_to_index, size_t& inst_index) {
    std::string text = "Block " + BlockName(block_to_index, &block) + " predecessors [";
    for (size_t i = 0; i < block.ImmPredecessors().size(); ++i) {
        text += (i ? ", " : "") + BlockName(block_to_index, block.ImmPredecessors()[i]);
    }
    text += "]\n";
    for (const Inst& inst : block) {
        const Opcode op = inst.GetOpcode();
        if (TypeOf(op) != Type::Void) {
            text += fmt::format("%{:<5} = {}", InstNumber(inst_to_index, inst_index, &inst), op);
        } else {
            text += fmt::format("         {}", op);
        }
        for (size_t i = 0; i < inst.NumArgs(); ++i) {
            const Value arg = inst.Arg(i);
            text += (i ? ", " : " ") + ArgText(inst_to_index, inst_index, arg);
            if (op == Opcode::Phi) {
                text += " from " + BlockName(block_to_index, inst.PhiBlock(i));
            } else if (!AreTypesCompatible(arg.Type(), ArgTypeOf(op, i))) {
                text += fmt::format(" <type error: {} != {}>", arg.Type(), ArgTypeOf(op, i));
            }
        }
        if (TypeOf(op) != Type::Void) {
            text += fmt::format(" (uses: {})", inst.UseCount());
        }
        text += '\n';
    }
    return text;
}

} // namespace Shader::IR
