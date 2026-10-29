// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: what the resource passes share: the sharp references found for each
// resource user, and the classes of instructions that take a buffer or image.
#pragma once

#include <array>

#include <boost/container/small_vector.hpp>

#include "shader/ir/opcodes.h"
#include "shader/ir/value.h"

namespace Shader {
enum class SharpFetchPostOp : u8;
}

namespace Shader::Optimization {

/// The operand of a sharp's post operation: the dword-1 mask or the LOD producer.
union PostOpData {
    u32 dw1_mask;
    IR::Value lod_prod{};
};

/// The dwords one sharp was read from.
struct SharpReference {
    std::array<IR::Value, 8> dwords{};
    u32 num_dwords{};
    SharpFetchPostOp post_op{};
    PostOpData post_op_data;
};

/// An instruction using resources, with its sharps (image and sampler, or one buffer).
struct ResourceDiscovery {
    IR::Inst* user{};
    std::array<SharpReference, 2> sharps;
};
using ResourceDiscoveryList = boost::container::small_vector<ResourceDiscovery, 32>;

inline bool IsBufferAtomic(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case BufferAtomicIAdd32:
    case BufferAtomicIAdd64:
    case BufferAtomicISub32:
    case BufferAtomicSMin32:
    case BufferAtomicSMin64:
    case BufferAtomicUMin32:
    case BufferAtomicUMin64:
    case BufferAtomicFMin32:
    case BufferAtomicSMax32:
    case BufferAtomicSMax64:
    case BufferAtomicUMax32:
    case BufferAtomicUMax64:
    case BufferAtomicFMax32:
    case BufferAtomicInc32:
    case BufferAtomicDec32:
    case BufferAtomicAnd32:
    case BufferAtomicOr32:
    case BufferAtomicXor32:
    case BufferAtomicSwap32:
    case BufferAtomicCmpSwap32:
    case BufferAtomicFCmpSwap32:
        return true;
    default:
        return false;
    }
}

/// Writes the buffer (stores and atomics).
inline bool IsBufferStore(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case StoreBufferU8:
    case StoreBufferU16:
    case StoreBufferU32:
    case StoreBufferU32x2:
    case StoreBufferU32x3:
    case StoreBufferU32x4:
    case StoreBufferU64:
    case StoreBufferF32:
    case StoreBufferF32x2:
    case StoreBufferF32x3:
    case StoreBufferF32x4:
    case StoreBufferFormatF32:
        return true;
    default:
        return IsBufferAtomic(inst);
    }
}

/// GDS accesses (append/consume counters and LDS operations flagged GDS).
inline bool IsDataRingInstruction(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case DataAppend:
    case DataConsume:
        return true;
    case LoadSharedU16:
    case LoadSharedU32:
    case LoadSharedU64:
    case WriteSharedU16:
    case WriteSharedU32:
    case WriteSharedU64:
    case SharedAtomicIAdd32:
    case SharedAtomicIAdd64:
    case SharedAtomicUMin32:
    case SharedAtomicUMin64:
    case SharedAtomicSMin32:
    case SharedAtomicSMin64:
    case SharedAtomicUMax32:
    case SharedAtomicUMax64:
    case SharedAtomicSMax32:
    case SharedAtomicSMax64:
    case SharedAtomicAnd32:
    case SharedAtomicAnd64:
    case SharedAtomicOr32:
    case SharedAtomicOr64:
    case SharedAtomicXor32:
    case SharedAtomicXor64:
    case SharedAtomicISub32:
    case SharedAtomicISub64:
    case SharedAtomicInc32:
    case SharedAtomicInc64:
    case SharedAtomicDec32:
    case SharedAtomicDec64:
    case SharedAtomicCmpSwap32:
    case SharedAtomicCmpSwap64:
        return inst.Flags<bool>(); // GDS
    default:
        return false;
    }
}

inline bool IsBufferInstruction(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case LoadBufferU8:
    case LoadBufferU16:
    case LoadBufferU32:
    case LoadBufferU32x2:
    case LoadBufferU32x3:
    case LoadBufferU32x4:
    case LoadBufferU64:
    case LoadBufferF32:
    case LoadBufferF32x2:
    case LoadBufferF32x3:
    case LoadBufferF32x4:
    case LoadBufferFormatF32:
    case ReadConstBuffer:
        return true;
    default:
        return IsBufferStore(inst);
    }
}

inline bool IsImageAtomicInstruction(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case ImageAtomicIAdd32:
    case ImageAtomicSMin32:
    case ImageAtomicUMin32:
    case ImageAtomicSMax32:
    case ImageAtomicUMax32:
    case ImageAtomicFMax32:
    case ImageAtomicFMin32:
    case ImageAtomicInc32:
    case ImageAtomicDec32:
    case ImageAtomicAnd32:
    case ImageAtomicOr32:
    case ImageAtomicXor32:
    case ImageAtomicExchange32:
    case ImageAtomicCmpSwap32:
        return true;
    default:
        return false;
    }
}

inline bool IsImageInstruction(const IR::Inst& inst) {
    using enum IR::Opcode;
    switch (inst.GetOpcode()) {
    case ImageRead:
    case ImageWrite:
    case ImageQueryDimensions:
    case ImageQueryLod:
    case ImageSampleRaw:
        return true;
    default:
        return IsImageAtomicInstruction(inst);
    }
}

} // namespace Shader::Optimization
