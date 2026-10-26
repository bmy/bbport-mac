// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): robust buffer access for the Metal command path; see mtl_spirv_robust.h.
//
// Word-level rewrite. For an access chain `%p = OpAccessChain %ptr %var %member %index ...` into a
// checked buffer it emits, before the chain:
//   %len  = OpArrayLength %uint %var <member>
//   %in   = OpULessThan %bool %index %len
//   %safe = OpSelect %index_type %in %index %null
// and the chain takes %safe. Then, for the uses of %p:
//   load:          %v' = OpLoad %T %p; %v = OpSelect %T %in %v' %null_T
//   store, atomic: the block is split at the instruction,
//                    OpSelectionMerge %after None; OpBranchConditional %in %do %after
//                  %do: <the instruction (an atomic's result renamed %r')>; OpBranch %after
//                  %after: (an atomic: %r = OpPhi %T %r' %do %null_T <block before>) the rest
// Phis that named the split block as a predecessor name its last part instead.

#include "metal/mtl_spirv_robust.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#define SPV_ENABLE_UTILITY_CODE
#include <spirv_cross/spirv.hpp>

namespace Metal {

namespace {

using Words = std::vector<u32>;

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

u32 Word0(u32 count, spv::Op op) {
    return (count << 16) | u32(op);
}

bool IsTerminator(spv::Op op) {
    switch (op) {
    case spv::OpBranch:
    case spv::OpBranchConditional:
    case spv::OpSwitch:
    case spv::OpReturn:
    case spv::OpReturnValue:
    case spv::OpKill:
    case spv::OpUnreachable:
    case spv::OpTerminateInvocation:
    case spv::OpIgnoreIntersectionKHR:
    case spv::OpTerminateRayKHR:
    case spv::OpEmitMeshTasksEXT:
        return true;
    default:
        return false;
    }
}

bool IsAtomic(spv::Op op) {
    switch (op) {
    case spv::OpAtomicLoad:
    case spv::OpAtomicStore:
    case spv::OpAtomicExchange:
    case spv::OpAtomicCompareExchange:
    case spv::OpAtomicCompareExchangeWeak:
    case spv::OpAtomicIIncrement:
    case spv::OpAtomicIDecrement:
    case spv::OpAtomicIAdd:
    case spv::OpAtomicISub:
    case spv::OpAtomicSMin:
    case spv::OpAtomicUMin:
    case spv::OpAtomicSMax:
    case spv::OpAtomicUMax:
    case spv::OpAtomicAnd:
    case spv::OpAtomicOr:
    case spv::OpAtomicXor:
    case spv::OpAtomicFlagTestAndSet:
    case spv::OpAtomicFlagClear:
    case spv::OpAtomicFMinEXT:
    case spv::OpAtomicFMaxEXT:
    case spv::OpAtomicFAddEXT:
        return true;
    default:
        return false;
    }
}

/// The first operand word that is not an id (literals follow), or the instruction's length.
/// Only for the id scan below: the words before it are ids (after the result type and id).
size_t IdOperandEnd(spv::Op op, size_t count) {
    switch (op) {
    case spv::OpCompositeExtract:
        return std::min<size_t>(count, 4); // type, id, composite, then literal indices
    case spv::OpCompositeInsert:
        return std::min<size_t>(count, 5); // type, id, object, composite, indices
    case spv::OpVectorShuffle:
        return std::min<size_t>(count, 5); // type, id, vector 1, vector 2, components
    case spv::OpExtInst:
        return count; // set and instruction number are words 3 and 4: skipped below
    case spv::OpLoad:
        return std::min<size_t>(count, 4); // type, id, pointer, memory operands
    case spv::OpStore:
        return std::min<size_t>(count, 3); // pointer, object, memory operands
    case spv::OpSwitch:
        return std::min<size_t>(count, 3); // selector, default, literal / label pairs
    case spv::OpSelectionMerge:
    case spv::OpLoopMerge:
    case spv::OpLine:
    case spv::OpNoLine:
    case spv::OpLabel:
        return 0;
    default:
        return count;
    }
}

class Rewriter {
public:
    explicit Rewriter(std::span<const u32> in, u32 skip_set) : spirv{in}, skip_set{skip_set} {}

    RobustRewrite Run() {
        RobustRewrite out;
        try {
            Scan();
            if (targets.empty()) {
                out.ok = true;
                out.spirv.assign(spirv.begin(), spirv.end());
                return out;
            }
            Words functions;
            for (size_t f = 0; f < function_starts.size(); ++f) {
                const size_t end =
                    f + 1 < function_starts.size() ? function_starts[f + 1] : spirv.size();
                RewriteFunction(function_starts[f], end, functions);
            }
            out.spirv.assign(spirv.begin(), spirv.begin() + globals_end);
            out.spirv.insert(out.spirv.end(), decls.begin(), decls.end());
            out.spirv.insert(out.spirv.end(), functions.begin(), functions.end());
            out.spirv[3] = bound;
            out.checked = checked;
            out.ok = true;
        } catch (const Failure& e) {
            out = {};
            out.error = e.what();
        }
        return out;
    }

private:
    struct Type {
        spv::Op op{};
        u32 width{}, signedness{};
        u32 storage{}, pointee{};
        Words members;
    };
    struct Target {
        u32 member{}; ///< the runtime array's member index
    };

    std::span<const u32> spirv;
    u32 skip_set;
    u32 bound{};
    size_t globals_end{};
    std::vector<size_t> function_starts;

    std::unordered_map<u32, Type> types;
    std::unordered_map<u32, u32> constants;  ///< 32-bit integer constants
    std::unordered_map<u32, u32> type_of;    ///< result id -> result type
    std::unordered_map<u32, u32> set_of;     ///< variable -> descriptor set
    std::unordered_set<u32> buffer_blocks;   ///< struct types decorated BufferBlock
    std::unordered_map<u64, u32> member_offsets; ///< (struct << 32 | member) -> Offset
    std::unordered_map<u32, Target> targets; ///< checked buffer variables
    u32 uint_type{}, bool_type{};

    Words decls; ///< new types and constants, before the first function
    std::unordered_map<u32, u32> nulls;
    u32 checked{};

    u32 NewId() {
        return bound++;
    }

    u32 UintType() {
        if (!uint_type) {
            uint_type = NewId();
            decls.insert(decls.end(), {Word0(4, spv::OpTypeInt), uint_type, 32, 0});
        }
        return uint_type;
    }
    u32 BoolType() {
        if (!bool_type) {
            bool_type = NewId();
            decls.insert(decls.end(), {Word0(2, spv::OpTypeBool), bool_type});
        }
        return bool_type;
    }
    u32 Null(u32 type) {
        if (const auto it = nulls.find(type); it != nulls.end()) {
            return it->second;
        }
        const u32 id = NewId();
        decls.insert(decls.end(), {Word0(3, spv::OpConstantNull), type, id});
        nulls.emplace(type, id);
        return id;
    }

    void Scan() {
        if (spirv.size() < 5 || spirv[0] != spv::MagicNumber) {
            throw Failure("not a SPIR-V module");
        }
        bound = spirv[3];
        globals_end = spirv.size();
        std::vector<std::pair<u32, u32>> variables; // id, pointer type
        for (size_t pos = 5; pos < spirv.size();) {
            const u32 count = spirv[pos] >> 16;
            const auto op = spv::Op(spirv[pos] & 0xffff);
            if (count == 0 || pos + count > spirv.size()) {
                throw Failure("malformed instruction");
            }
            const u32* w = &spirv[pos];
            bool has_result = false, has_type = false;
            spv::HasResultAndType(op, &has_result, &has_type);
            if (has_result && has_type && count >= 3) {
                type_of[w[2]] = w[1];
            }
            switch (op) {
            case spv::OpFunction:
                if (function_starts.empty()) {
                    globals_end = pos;
                }
                function_starts.push_back(pos);
                break;
            case spv::OpCapability:
                if (count >= 2 && (w[1] == spv::CapabilityVariablePointers ||
                                   w[1] == spv::CapabilityVariablePointersStorageBuffer)) {
                    // A pointer phi could name an access chain the scan has not reached yet.
                    throw Failure("variable pointers");
                }
                break;
            case spv::OpDecorate:
                if (count >= 4 && w[2] == spv::DecorationDescriptorSet) {
                    set_of[w[1]] = w[3];
                } else if (count >= 3 && w[2] == spv::DecorationBufferBlock) {
                    buffer_blocks.insert(w[1]);
                }
                break;
            case spv::OpMemberDecorate:
                if (count >= 5 && w[3] == spv::DecorationOffset) {
                    member_offsets[(u64(w[1]) << 32) | w[2]] = w[4];
                }
                break;
            case spv::OpTypeInt:
                types[w[1]] = {.op = op, .width = w[2], .signedness = w[3]};
                if (w[2] == 32 && w[3] == 0) {
                    uint_type = w[1];
                }
                break;
            case spv::OpTypeBool:
                types[w[1]] = {.op = op};
                bool_type = w[1];
                break;
            case spv::OpTypePointer:
                types[w[1]] = {.op = op, .storage = w[2], .pointee = w[3]};
                break;
            case spv::OpTypeStruct:
                types[w[1]] = {.op = op, .members = Words(w + 2, w + count)};
                break;
            case spv::OpTypeRuntimeArray:
                types[w[1]] = {.op = op};
                break;
            case spv::OpConstant:
                if (count == 4) {
                    constants[w[2]] = w[3];
                }
                break;
            case spv::OpVariable:
                if (function_starts.empty() && count >= 4 &&
                    (w[3] == spv::StorageClassStorageBuffer ||
                     w[3] == spv::StorageClassUniform)) {
                    variables.emplace_back(w[2], w[1]);
                }
                break;
            default:
                break;
            }
            pos += count;
        }
        for (const auto& [id, pointer_type] : variables) {
            const auto set = set_of.find(id);
            if (set != set_of.end() && set->second == skip_set) {
                continue;
            }
            const Type& pointer = types[pointer_type];
            const auto block = types.find(pointer.pointee);
            if (pointer.storage == spv::StorageClassUniform &&
                (block == types.end() || !buffer_blocks.contains(block->first))) {
                continue; // a uniform buffer: read-only, the null stand-in is its largest size
            }
            // A storage buffer (also an old-style Uniform BufferBlock) of another shape would be
            // left unchecked, and a write through it could reach the shared zero buffer.
            if (block == types.end() || block->second.op != spv::OpTypeStruct ||
                block->second.members.empty() || pointer.storage != spv::StorageClassStorageBuffer) {
                throw Failure("a storage buffer that is not a block in StorageBuffer");
            }
            const u32 member = u32(block->second.members.size() - 1);
            const auto array = types.find(block->second.members.back());
            if (array == types.end() || array->second.op != spv::OpTypeRuntimeArray) {
                throw Failure("a storage buffer without a runtime array");
            }
            // SPIRV-Cross' length is (size - offset) / stride: it wraps for a size below the offset.
            const auto offset = member_offsets.find((u64(block->first) << 32) | member);
            if (offset == member_offsets.end() || offset->second != 0) {
                throw Failure("a storage buffer's runtime array not at offset 0");
            }
            targets[id] = {member};
        }
    }

    void RewriteFunction(size_t begin, size_t end, Words& out) {
        // Which blocks hold a loop merge (their stores and atomics cannot be split off).
        std::unordered_set<u32> loop_headers;
        {
            u32 label = 0;
            for (size_t pos = begin; pos < end; pos += spirv[pos] >> 16) {
                const auto op = spv::Op(spirv[pos] & 0xffff);
                if (op == spv::OpLabel) {
                    label = spirv[pos + 1];
                } else if (op == spv::OpLoopMerge) {
                    loop_headers.insert(label);
                }
            }
        }

        std::unordered_map<u32, u32> in_bounds;  ///< checked element pointer -> its condition
        std::unordered_map<u32, u32> last_label; ///< original block -> label of its last part
        std::vector<size_t> phis;                ///< the function's own OpPhi (offsets in out)
        u32 original = 0, label = 0;
        bool loop_header = false;

        const auto references = [&](const u32* w, u32 count, spv::Op op, size_t from) {
            const size_t ids_end = IdOperandEnd(op, count);
            for (size_t i = from; i < ids_end; ++i) {
                if (op == spv::OpExtInst && (i == 3 || i == 4)) {
                    continue;
                }
                if (targets.contains(w[i]) || in_bounds.contains(w[i])) {
                    return true;
                }
            }
            return false;
        };
        const auto split = [&](u32 condition, u32& do_label, u32& after_label) {
            if (loop_header) {
                throw Failure("a store or atomic on a checked buffer in a loop header");
            }
            do_label = NewId();
            after_label = NewId();
            out.insert(out.end(), {Word0(3, spv::OpSelectionMerge), after_label, 0});
            out.insert(out.end(),
                       {Word0(4, spv::OpBranchConditional), condition, do_label, after_label});
            out.insert(out.end(), {Word0(2, spv::OpLabel), do_label});
        };

        for (size_t pos = begin; pos < end;) {
            const u32 count = spirv[pos] >> 16;
            const auto op = spv::Op(spirv[pos] & 0xffff);
            const u32* w = &spirv[pos];
            pos += count;

            if (op == spv::OpLabel) {
                original = label = w[1];
                loop_header = loop_headers.contains(original);
                out.insert(out.end(), w, w + count);
                continue;
            }
            if (IsTerminator(op)) {
                last_label[original] = label;
            }
            if (op == spv::OpPhi) {
                phis.push_back(out.size());
            }

            if ((op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain) && count >= 4 &&
                targets.contains(w[3])) {
                if (count < 6) {
                    throw Failure("a checked buffer's array taken as a whole");
                }
                const Target& target = targets[w[3]];
                const auto member = constants.find(w[4]);
                if (member == constants.end() || member->second != target.member) {
                    throw Failure("an access chain into a checked buffer's other member");
                }
                const u32 index = w[5];
                const u32 index_type = type_of.contains(index) ? type_of[index] : 0;
                const auto t = types.find(index_type);
                if (t == types.end() || t->second.op != spv::OpTypeInt || t->second.width != 32) {
                    throw Failure("a checked buffer's index that is not a 32-bit integer");
                }
                for (u32 i = 6; i < count; ++i) {
                    if (targets.contains(w[i]) || in_bounds.contains(w[i])) {
                        throw Failure("a checked pointer used as an index");
                    }
                }
                const u32 length = NewId(), condition = NewId(), safe = NewId();
                out.insert(out.end(),
                           {Word0(5, spv::OpArrayLength), UintType(), length, w[3], target.member});
                out.insert(out.end(),
                           {Word0(5, spv::OpULessThan), BoolType(), condition, index, length});
                out.insert(out.end(), {Word0(6, spv::OpSelect), index_type, safe, condition,
                                       index, Null(index_type)});
                const size_t at = out.size();
                out.insert(out.end(), w, w + count);
                out[at + 5] = safe;
                in_bounds[w[2]] = condition;
                ++checked;
                continue;
            }
            if (op == spv::OpArrayLength && count == 5 && targets.contains(w[3])) {
                out.insert(out.end(), w, w + count);
                continue;
            }
            if ((op == spv::OpLoad || op == spv::OpAtomicLoad) && count >= 4 &&
                in_bounds.contains(w[3]) && !references(w, count, op, 4)) {
                const u32 type = w[1], result = w[2], loaded = NewId();
                const size_t at = out.size();
                out.insert(out.end(), w, w + count);
                out[at + 2] = loaded;
                out.insert(out.end(), {Word0(6, spv::OpSelect), type, result, in_bounds[w[3]],
                                       loaded, Null(type)});
                continue;
            }
            if ((op == spv::OpStore || op == spv::OpAtomicStore) && count >= 3 &&
                in_bounds.contains(w[1])) {
                if (references(w, count, op, 2)) {
                    throw Failure("a checked pointer stored");
                }
                u32 do_label = 0, after_label = 0;
                split(in_bounds[w[1]], do_label, after_label);
                out.insert(out.end(), w, w + count);
                out.insert(out.end(), {Word0(2, spv::OpBranch), after_label});
                out.insert(out.end(), {Word0(2, spv::OpLabel), after_label});
                label = after_label;
                continue;
            }
            if (IsAtomic(op) && count >= 4 && in_bounds.contains(w[3])) {
                if (references(w, count, op, 4)) {
                    throw Failure("a checked pointer as an atomic's operand");
                }
                const u32 type = w[1], result = w[2], value = NewId();
                const u32 before = label;
                u32 do_label = 0, after_label = 0;
                split(in_bounds[w[3]], do_label, after_label);
                const size_t at = out.size();
                out.insert(out.end(), w, w + count);
                out[at + 2] = value;
                out.insert(out.end(), {Word0(2, spv::OpBranch), after_label});
                out.insert(out.end(), {Word0(2, spv::OpLabel), after_label});
                out.insert(out.end(), {Word0(7, spv::OpPhi), type, result, value, do_label,
                                       Null(type), before});
                label = after_label;
                continue;
            }
            if (op != spv::OpFunction && op != spv::OpFunctionParameter &&
                op != spv::OpFunctionEnd) {
                bool has_result = false, has_type = false;
                spv::HasResultAndType(op, &has_result, &has_type);
                const size_t from = size_t(1) + has_result + has_type;
                if (references(w, count, op, from)) {
                    throw Failure(std::string("a checked buffer or element pointer used by ") +
                                  spv::OpToString(op));
                }
            }
            out.insert(out.end(), w, w + count);
        }

        // The phis that named a split block as a predecessor: its last part comes before them.
        for (const size_t at : phis) {
            const u32 count = out[at] >> 16;
            for (u32 i = 4; i < count; i += 2) {
                const auto it = last_label.find(out[at + i]);
                if (it != last_label.end()) {
                    out[at + i] = it->second;
                }
            }
        }
    }
};

} // namespace

RobustRewrite RewriteRobustBuffers(std::span<const u32> spirv, u32 skip_set) {
    return Rewriter{spirv, skip_set}.Run();
}

} // namespace Metal
