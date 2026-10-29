// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Fills the shader's Info from its instructions: attributes and user data it reads or writes,
// the shared memory types, and the device features it needs (fp16/fp64, subgroup operations,
// image and buffer atomics, gathers, queries, discard, DMA).
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"
#include "shader/profile.h"
#include "shader/resource.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;

void VisitPatch(Info& info, const IR::Inst& inst, bool store) {
    const IR::Patch patch = inst.Arg(0).Patch();
    if (store && patch <= IR::Patch::TessellationLodBottom) {
        info.stores_tess_level_outer = true;
    } else if (store && patch <= IR::Patch::TessellationLodInteriorV) {
        info.stores_tess_level_inner = true;
    } else {
        info.uses_patches |= 1U << IR::GenericPatchIndex(patch);
    }
}

void VisitReadConst(Info& info, const IR::Inst& inst) {
    if (!info.has_readconst) {
        info.buffers.push_back({
            .used_types = IR::Type::U32,
            .buffer_type = BufferType::Flatbuf,
        });
        info.has_readconst = true;
    }
    // Flags 0: the address is not a known constant buffer, read through DMA.
    if (inst.Flags<u32>() == 0) {
        info.readconst_types |= Info::ReadConstType::Immediate;
        info.readconst_types |= Info::ReadConstType::Dynamic;
        info.uses_dma = true;
    }
}

void Visit(Info& info, const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case Opcode::GetAttribute:
    case Opcode::GetAttributeU1:
    case Opcode::GetAttributeU32:
        info.loads.Set(inst.Arg(0).Attribute(), inst.Arg(1).U32());
        break;
    case Opcode::SetAttribute:
        info.stores.Set(inst.Arg(0).Attribute(), inst.Arg(2).U32());
        break;
    case Opcode::GetUserData:
        info.ud_mask.Set(inst.Arg(0).ScalarReg());
        break;
    case Opcode::SetPatch:
        VisitPatch(info, inst, true);
        break;
    case Opcode::GetPatch:
        VisitPatch(info, inst, false);
        break;

    case Opcode::LoadSharedU16:
    case Opcode::WriteSharedU16:
        info.shared_types |= IR::Type::U16;
        break;
    case Opcode::LoadSharedU32:
    case Opcode::WriteSharedU32:
    case Opcode::SharedAtomicIAdd32:
    case Opcode::SharedAtomicISub32:
    case Opcode::SharedAtomicSMin32:
    case Opcode::SharedAtomicUMin32:
    case Opcode::SharedAtomicSMax32:
    case Opcode::SharedAtomicUMax32:
    case Opcode::SharedAtomicInc32:
    case Opcode::SharedAtomicDec32:
    case Opcode::SharedAtomicAnd32:
    case Opcode::SharedAtomicOr32:
    case Opcode::SharedAtomicXor32:
    case Opcode::SharedAtomicCmpSwap32:
        info.shared_types |= IR::Type::U32;
        break;
    case Opcode::SharedAtomicIAdd64:
    case Opcode::SharedAtomicISub64:
    case Opcode::SharedAtomicSMin64:
    case Opcode::SharedAtomicUMin64:
    case Opcode::SharedAtomicSMax64:
    case Opcode::SharedAtomicUMax64:
    case Opcode::SharedAtomicInc64:
    case Opcode::SharedAtomicDec64:
    case Opcode::SharedAtomicAnd64:
    case Opcode::SharedAtomicOr64:
    case Opcode::SharedAtomicXor64:
    case Opcode::SharedAtomicCmpSwap64:
        info.uses_shared_int64_atomics = true;
        info.shared_types |= IR::Type::U64;
        break;
    case Opcode::LoadSharedU64:
    case Opcode::WriteSharedU64:
        info.shared_types |= IR::Type::U64;
        break;

    case Opcode::ConvertF16F32:
    case Opcode::ConvertF32F16:
    case Opcode::BitCastU16F16:
    case Opcode::BitCastF16U16:
        info.uses_fp16 = true;
        break;
    case Opcode::PackDouble2x32:
    case Opcode::UnpackDouble2x32:
        info.uses_fp64 = true;
        break;
    case Opcode::PackUfloat10_11_11:
        info.uses_pack_10_11_11 = true;
        break;
    case Opcode::UnpackUfloat10_11_11:
        info.uses_unpack_10_11_11 = true;
        break;
    case Opcode::BitwiseXor32:
        info.has_bitwise_xor = true;
        break;

    case Opcode::QuadBroadcast:
        info.uses_group_quad = true;
        break;
    case Opcode::Shuffle:
    case Opcode::ShuffleXor:
        info.uses_group_shuffle = true;
        break;
    case Opcode::ReadLane:
    case Opcode::ReadFirstLane:
    case Opcode::WriteLane:
        info.uses_group_ballot = true;
        break;
    case Opcode::DataAppend:
    case Opcode::DataConsume:
    case Opcode::Ballot:
    case Opcode::InverseBallot:
    case Opcode::BallotFindLsb:
        info.uses_group_ballot = true;
        info.uses_lane_id = true;
        break;
    case Opcode::LaneId:
        info.uses_lane_id = true;
        break;
    case Opcode::Memtime:
        info.uses_shader_clock = true;
        break;
    case Opcode::Discard:
    case Opcode::DiscardCond:
        info.has_discard = true;
        break;

    case Opcode::ImageWrite:
        info.has_storage_images = true;
        break;
    case Opcode::ImageGather:
    case Opcode::ImageGatherDref:
        info.has_image_gather = true;
        break;
    case Opcode::ImageQueryDimensions:
    case Opcode::ImageQueryLod:
        info.has_image_query = true;
        break;
    case Opcode::ImageAtomicFMax32:
    case Opcode::ImageAtomicFMin32:
        info.uses_image_atomic_float_min_max = true;
        break;
    case Opcode::BufferAtomicFMax32:
    case Opcode::BufferAtomicFMin32:
        info.uses_buffer_atomic_float_min_max = true;
        break;
    case Opcode::BufferAtomicIAdd64:
    case Opcode::BufferAtomicSMax64:
    case Opcode::BufferAtomicSMin64:
    case Opcode::BufferAtomicUMax64:
    case Opcode::BufferAtomicUMin64:
        info.uses_buffer_int64_atomics = true;
        break;
    case Opcode::ReadConst:
        VisitReadConst(info, inst);
        break;
    default:
        break;
    }
}

} // namespace

void CollectShaderInfoPass(IR::Program& program, const Profile& profile) {
    Info& info = program.info;
    for (IR::Block* const block : program.post_order_blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            Visit(info, inst);
        }
    }

    if (!EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        info.uses_dma = false;
        info.readconst_types = Info::ReadConstType::None;
    }
    if (info.uses_dma) {
        info.buffers.push_back({
            .used_types = IR::Type::U64,
            .buffer_type = BufferType::BdaPagetable,
            .is_written = true,
        });
        info.buffers.push_back({
            .used_types = IR::Type::U32,
            .buffer_type = BufferType::FaultBuffer,
            .is_written = true,
        });
        LOG_WARNING(Render, "Enabling DMA for shader {:#x}", info.pgm_hash);
    }

    // bbport BB_LAYER_MEMORY: buffers over nearly all memory read the page table. Every shader with
    // guest buffers gets it (a later permutation may have a paged buffer where the first one had
    // none, and the program's resource list comes from the first).
    if (profile.paged_buffers) {
        bool guest_buffers = false;
        for (const auto& buffer : info.buffers) {
            if (buffer.buffer_type == BufferType::Guest) {
                guest_buffers = true;
                // Any guest buffer may be paged in a later permutation (size, or lazy guest
                // memory: a range across imports): the module's physical addressing for all of
                // them. Taken from the first permutation only, a later paged one lacked it.
                info.uses_paged_buffers = true;
            }
        }
        if (guest_buffers && !info.uses_dma) {
            info.buffers.push_back({
                .used_types = IR::Type::U64,
                .buffer_type = BufferType::BdaPagetable,
            });
        }
    }
}

} // namespace Shader::Optimization
