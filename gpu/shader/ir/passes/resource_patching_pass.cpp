// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Turns the discovered resource uses into bindings: each distinct sharp (with its post
// operation) becomes a buffer, image or sampler resource of the shader, and the instructions
// take the binding index instead of the sharp. Then their operands are rewritten for the host:
// buffer addresses computed from index/offsets/stride/swizzling (in elements of the access), image
// coordinates taken from the GCN address registers (offsets, bias, compare value, derivatives,
// LOD, cube face fix-up, (un)normalization), and the formats' number conversions applied. GDS
// accesses (append/consume counters, GDS atomics) go to a GDS storage buffer.
#include <bit>
#include <limits>
#include <utility>

#include "common/assert.h"
#include "common/logging/log.h"
#include "gcn/sharps.h"
#include "shader/ir/basic_block.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/operand_helper.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/passes/resource_pass.h"
#include "shader/ir/program.h"
#include "shader/ir/reinterpret.h"
#include "shader/info.h"
#include "shader/profile.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;
using ::Gcn::DataFormat;
using ::Gcn::ImageType;

IR::Type BufferDataType(const IR::Inst& inst, const Profile& profile) {
    switch (inst.GetOpcode()) {
    case Opcode::LoadBufferU8:
    case Opcode::StoreBufferU8:
        return IR::Type::U8;
    case Opcode::LoadBufferU16:
    case Opcode::StoreBufferU16:
        return IR::Type::U16;
    case Opcode::LoadBufferU64:
    case Opcode::StoreBufferU64:
    case Opcode::BufferAtomicIAdd64:
    case Opcode::BufferAtomicSMax64:
    case Opcode::BufferAtomicSMin64:
    case Opcode::BufferAtomicUMax64:
    case Opcode::BufferAtomicUMin64:
        return IR::Type::U64;
    case Opcode::BufferAtomicFMax32:
    case Opcode::BufferAtomicFMin32:
        return profile.supports_buffer_fp32_atomic_min_max ? IR::Type::F32 : IR::Type::U32;
    case Opcode::LoadBufferFormatF32:
    case Opcode::StoreBufferFormatF32:
        // Lowered later to any of these.
        return IR::Type::U32 | IR::Type::F32 | IR::Type::U16 | IR::Type::U8;
    default:
        return IR::Type::U32;
    }
}

/// log2 of the access's element size: buffer addresses are in elements.
u32 BufferAddressShift(const IR::Inst& inst, DataFormat data_format) {
    switch (inst.GetOpcode()) {
    case Opcode::LoadBufferU8:
    case Opcode::StoreBufferU8:
    case Opcode::ReadConstBuffer: // (already in dwords)
        return 0;
    case Opcode::LoadBufferU16:
    case Opcode::StoreBufferU16:
        return 1;
    case Opcode::LoadBufferU64:
    case Opcode::StoreBufferU64:
    case Opcode::BufferAtomicIAdd64:
    case Opcode::BufferAtomicSMax64:
    case Opcode::BufferAtomicSMin64:
    case Opcode::BufferAtomicUMax64:
    case Opcode::BufferAtomicUMin64:
        return 3;
    case Opcode::LoadBufferFormatF32:
    case Opcode::StoreBufferFormatF32:
        switch (data_format) {
        case DataFormat::Format8_8:
        case DataFormat::Format16:
            return 1;
        case DataFormat::Format8_8_8_8:
        case DataFormat::Format16_16:
        case DataFormat::Format10_11_11:
        case DataFormat::Format2_10_10_10:
        case DataFormat::Format16_16_16_16:
        case DataFormat::Format32:
        case DataFormat::Format32_32:
        case DataFormat::Format32_32_32:
        case DataFormat::Format32_32_32_32:
            return 2;
        default:
            return 0;
        }
    default:
        return 2;
    }
}

/// The shader's resource lists; equal descriptions share an entry (usage flags merged).
class Descriptors {
public:
    explicit Descriptors(Info& info)
        : buffers{info.buffers}, images{info.images}, samplers{info.samplers},
          fmasks{info.fmasks} {}

    u32 Add(const BufferResource& desc) {
        const u32 index = Find(buffers, desc, [&](const BufferResource& other) {
            return desc.sharp_fetch == other.sharp_fetch && desc.post_op == other.post_op &&
                   desc.post_op_dw1_mask == other.post_op_dw1_mask &&
                   desc.buffer_type == other.buffer_type;
        });
        BufferResource& buffer = buffers[index];
        buffer.used_types |= desc.used_types;
        buffer.is_written |= desc.is_written;
        buffer.is_formatted |= desc.is_formatted;
        return index;
    }

    u32 Add(const ImageResource& desc) {
        const u32 index = Find(images, desc, [&](const ImageResource& other) {
            return desc.sharp_fetch == other.sharp_fetch && desc.is_array == other.is_array &&
                   desc.mip_fallback_mode == other.mip_fallback_mode &&
                   desc.constant_mip_index == other.constant_mip_index &&
                   desc.post_op == other.post_op;
        });
        ImageResource& image = images[index];
        image.is_atomic |= desc.is_atomic;
        image.is_written |= desc.is_written;
        image.needs_native |= desc.needs_native;
        return index;
    }

    u32 Add(const SamplerResource& desc) {
        return Find(samplers, desc, [&](const SamplerResource& other) {
            return desc.sharp_fetch == other.sharp_fetch && desc.post_op == other.post_op &&
                   desc.post_op_tsharp_dw3_off == other.post_op_tsharp_dw3_off &&
                   desc.is_depth == other.is_depth;
        });
    }

    u32 Add(const FMaskResource& desc) {
        return Find(fmasks, desc,
                    [&](const FMaskResource& other) { return desc.sharp_idx == other.sharp_idx; });
    }

private:
    template <typename List, typename Desc, typename Equal>
    static u32 Find(List& list, const Desc& desc, Equal&& equal) {
        const auto it = std::ranges::find_if(list, equal);
        if (it != list.end()) {
            return static_cast<u32>(std::distance(list.begin(), it));
        }
        list.push_back(desc);
        return static_cast<u32>(list.size()) - 1;
    }

    BufferResourceList& buffers;
    ImageResourceList& images;
    SamplerResourceList& samplers;
    FMaskResourceList& fmasks;
};

/// Where a sharp dword is read from: a user data register, or its flat buffer slot.
SharpLocation LocationOf(const IR::Inst* source) {
    SharpLocation location;
    switch (source->GetOpcode()) {
    case Opcode::GetUserData:
        return static_cast<SharpLocation>(source->Arg(0).ScalarReg());
    case Opcode::ReadConstBuffer:
        location = source->Flags<IR::BufferInstInfo>().flatbuf_off_dw;
        break;
    default:
        location = source->Flags<SharpLocation>();
        break;
    }
    if (location == 0) {
        LOG_WARNING(Render_Recompiler, "Sharp source was not flatenned");
        return UNKNOWN_LOCATION;
    }
    return location;
}

template <typename T>
SharpFetch<T> FetchOf(const SharpReference& sharp) {
    SharpFetch<T> fetch{};
    for (u32 i = 0; i < sharp.num_dwords; i++) {
        const IR::Value dword = sharp.dwords[i];
        if (dword.IsImmediate()) {
            fetch.immediates[i] = dword.U32();
        } else {
            fetch.offsets[i] = LocationOf(dword.Inst());
            fetch.load_mask |= 1 << i;
        }
    }
    return fetch;
}

void BindBuffer(const ResourceDiscovery& resource, Descriptors& descriptors,
                const Profile& profile) {
    IR::Inst& inst = *resource.user;
    const SharpReference& vsharp = resource.sharps[0];
    const u32 binding = descriptors.Add(BufferResource{
        .sharp_fetch = FetchOf<::Gcn::Buffer>(vsharp),
        .used_types = BufferDataType(inst, profile),
        .buffer_type = BufferType::Guest,
        .is_written = IsBufferStore(inst),
        .is_formatted = inst.GetOpcode() == Opcode::LoadBufferFormatF32 ||
                        inst.GetOpcode() == Opcode::StoreBufferFormatF32,
        .post_op = vsharp.post_op,
        .post_op_dw1_mask = vsharp.post_op_data.dw1_mask,
    });
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    inst.SetArg(0, ir.Imm32(binding));
}

/// Sampling with normalized coordinates and no offsets works on a host view of any format;
/// the rest needs the native format.
bool IsNormalizedSample(const IR::Inst& inst, IR::TextureInstInfo flags) {
    switch (inst.GetOpcode()) {
    case Opcode::ImageSampleRaw:
    case Opcode::ImageSampleImplicitLod:
    case Opcode::ImageSampleExplicitLod:
    case Opcode::ImageSampleDrefImplicitLod:
    case Opcode::ImageSampleDrefExplicitLod:
    case Opcode::ImageGather:
    case Opcode::ImageGatherDref:
    case Opcode::ImageGradient:
    case Opcode::ImageQueryLod:
        return !flags.has_offset && !flags.is_unnormalized;
    default:
        return false;
    }
}

/// The LOD operand of an image_load/store_mip in its address vector.
IR::Value MipOperand(const IR::Inst& inst, ImageType view_type) {
    const IR::Inst* const body = inst.Arg(1).Inst();
    switch (view_type) {
    case ImageType::Color1D: // x, [lod]
        return body->Arg(1);
    case ImageType::Color1DArray: // x, slice, [lod]
    case ImageType::Color2D:      // x, y, [lod]
        return body->Arg(2);
    case ImageType::Color2DArray: // x, y, slice, [lod]
    case ImageType::Color3D:      // x, y, z, [lod]
        return body->Arg(3);
    default:
        UNREACHABLE_MSG("Invalid image type {}", view_type);
    }
}

/// FMask reads are answered with constants: the identity sample mapping, one level, the sharp's
/// size (tracked for specialization).
void ReplaceFmaskAccess(IR::Inst& inst, const ::Gcn::Image& image, const SharpReference& tsharp,
                        Descriptors& descriptors) {
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    switch (inst.GetOpcode()) {
    case Opcode::ImageRead:
    case Opcode::ImageSampleRaw: {
        const IR::F32 lo = ir.BitCast<IR::F32>(ir.Imm32(0x76543210));
        const IR::F32 hi = ir.BitCast<IR::F32>(ir.Imm32(0xfedcba98));
        inst.ReplaceUsesWith(ir.CompositeConstruct(lo, hi));
        break;
    }
    case Opcode::ImageQueryLod:
        inst.ReplaceUsesWith(ir.Imm32(1));
        break;
    case Opcode::ImageQueryDimensions:
        inst.ReplaceUsesWith(ir.CompositeConstruct(ir.Imm32(static_cast<u32>(image.width)),
                                                   ir.Imm32(static_cast<u32>(image.height)),
                                                   ir.Imm32(1), ir.Imm32(1)));
        descriptors.Add(FMaskResource{.sharp_idx = LocationOf(tsharp.dwords[0].Inst())});
        break;
    default:
        UNREACHABLE_MSG("Can't patch fmask instruction {}", inst.GetOpcode());
    }
}

void BindImage(const ResourceDiscovery& resource, const Info& info, Descriptors& descriptors,
               const Profile& profile) {
    IR::Inst& inst = *resource.user;
    const SharpReference& tsharp = resource.sharps[0];
    const auto flags = inst.Flags<IR::TextureInstInfo>();
    const bool is_atomic = IsImageAtomicInstruction(inst);
    const bool is_written = inst.GetOpcode() == Opcode::ImageWrite || is_atomic;
    ImageResource desc{
        .sharp_fetch = FetchOf<::Gcn::Image>(tsharp),
        .is_depth = bool(flags.is_depth),
        .is_atomic = is_atomic,
        .is_array = bool(flags.is_array),
        .is_written = is_written,
        .is_r128 = bool(flags.is_r128),
        .needs_native = !IsNormalizedSample(inst, flags),
        .post_op = tsharp.post_op,
    };
    const auto image = desc.GetSharp(info);
    ASSERT(image.GetType() != ImageType::Invalid);

    // Storage access to one mip level without host support: a constant level binds that level
    // alone, a dynamic one binds the levels as an array. (Reads with a LOD are fetches.)
    if (flags.has_lod && is_written && !profile.supports_image_load_store_lod) {
        const IR::Value mip = MipOperand(inst, image.GetViewType(desc.is_array));
        if (mip.IsImmediate()) {
            desc.mip_fallback_mode = MipStorageFallbackMode::ConstantIndex;
            desc.constant_mip_index = mip.U32();
        } else {
            desc.mip_fallback_mode = MipStorageFallbackMode::DynamicIndex;
        }
    }
    if (::Gcn::IsFmask(image.GetDataFmt())) {
        ASSERT_MSG(!is_written, "FMask storage instructions are not supported");
        ReplaceFmaskAccess(inst, image, tsharp, descriptors);
        return;
    }

    const u32 image_binding = descriptors.Add(desc);
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    if (inst.GetOpcode() != Opcode::ImageSampleRaw) {
        inst.SetArg(0, ir.Imm32(image_binding));
        return;
    }
    const SharpReference& ssharp = resource.sharps[1];
    const IR::Value& lod_source = ssharp.post_op_data.lod_prod;
    const u32 sampler_binding = descriptors.Add(SamplerResource{
        .sharp_fetch = FetchOf<::Gcn::Sampler>(ssharp),
        .post_op = ssharp.post_op,
        .post_op_tsharp_dw3_off =
            lod_source.IsEmpty() ? UNKNOWN_LOCATION : LocationOf(lod_source.Inst()),
        .is_depth = bool(flags.is_depth), // the compare opcodes
    });
    inst.SetArg(0, ir.Imm32(image_binding | sampler_binding << 16));
}

/// The operand of `value` = v_mbcnt_(lo|hi)(exec half, operand), where the exec half comes from
/// Ballot(exec) (any ballot for the high half); empty otherwise.
IR::Value MbcntOperand(const IR::Value& value, bool is_hi, const IR::U1& exec) {
    if (value.IsImmediate()) {
        return {};
    }
    const IR::Inst* const mbcnt = value.Inst();
    if (mbcnt->GetOpcode() != Opcode::MaskedBitCount32 || mbcnt->Arg(0).IsImmediate() ||
        mbcnt->Arg(2).U1() != is_hi) {
        return {};
    }
    const IR::Inst* const half = mbcnt->Arg(0).Inst();
    if (half->GetOpcode() != Opcode::CompositeExtractU32x2 || half->Arg(0).IsImmediate() ||
        half->Arg(1).U32() != is_hi) {
        return {};
    }
    const IR::Inst* const unpack = half->Arg(0).Inst();
    if (unpack->GetOpcode() != Opcode::UnpackUint2x32 || unpack->Arg(0).IsImmediate()) {
        return {};
    }
    const IR::Inst* const ballot = unpack->Arg(0).Inst();
    if (ballot->GetOpcode() != Opcode::Ballot || (ballot->Arg(0) != exec && !is_hi)) {
        return {};
    }
    return mbcnt->Arg(1);
}

template <typename Pred>
IR::Use UniqueUser(const IR::Inst* inst, Pred&& pred) {
    IR::Use picked{};
    for (const IR::Use& use : inst->Uses()) {
        if (pred(use.user)) {
            ASSERT(!picked.user);
            picked = use;
        }
    }
    ASSERT(picked.user);
    return picked;
}

bool IsAddOrSub(const IR::Inst* inst) {
    return inst->GetOpcode() == Opcode::IAdd32 || inst->GetOpcode() == Opcode::ISub32;
}

/// ds_append/ds_consume are per wave on GCN, with the lane's index added by v_mbcnt; on the host
/// the counter atomic runs per invocation, so the lane offset goes away. Patterns:
///   ds_append vX; v_mbcnt_hi vY, exec_hi, vX; v_mbcnt_lo idx, exec_lo, vY
///   v_mbcnt_hi vY, exec_hi, 0; v_mbcnt_lo vZ, exec_lo, vY; ds_append vX; v_add idx, vX, vZ
///   ds_consume vX; (mbcnt vZ as above); v_sub vK, vX, vZ; v_add idx, -1, vK (or v_subrev 1)
///   ds_consume vX; (mbcnt vZ as above); v_add vK, -1, vX; v_sub idx, vK, vZ
void RemoveAppendLaneOffset(IR::Inst& vx) {
    const IR::U1 exec{vx.Arg(1)};
    const auto uses = vx.Uses();
    for (IR::Use use : uses) {
        const auto [user, operand] = use;
        if (user->GetOpcode() == Opcode::MaskedBitCount32) {
            ASSERT(vx.GetOpcode() == Opcode::DataAppend);
            IR::Inst* const vy = user;
            const IR::Value hi_operand = MbcntOperand(IR::Value{vy}, true, exec);
            ASSERT(!hi_operand.IsEmpty() && hi_operand == IR::Value{&vx});
            const auto [idx, idx_operand] = UniqueUser(vy, [](const IR::Inst* inst) {
                return inst->GetOpcode() == Opcode::MaskedBitCount32;
            });
            const IR::Value lo_operand = MbcntOperand(IR::Value{idx}, false, exec);
            ASSERT(!lo_operand.IsEmpty() && lo_operand == IR::Value{vy});
            idx->ReplaceUsesWithAndRemove(IR::Value{&vx});
            continue;
        }
        if (!IsAddOrSub(user)) {
            continue;
        }
        const IR::Value other = user->Arg(1 - operand);
        if (other.IsImmediate()) {
            ASSERT_MSG(other.U32() == 1u || other.U32() == std::numeric_limits<u32>::max() &&
                                                vx.GetOpcode() == Opcode::DataConsume,
                       "Unexpected constant offset {} to DataConsume result", other.U32());
            use = UniqueUser(user, IsAddOrSub);
        }
        const IR::Value vz = user->Arg(1 - operand);
        const IR::Value vy = MbcntOperand(vz, false, exec);
        ASSERT(!vy.IsEmpty() && !vy.IsImmediate());
        const IR::Value zero = MbcntOperand(vy, true, exec);
        ASSERT(!zero.IsEmpty() && zero.IsImmediate() && zero.U32() == 0u);
        vz.Inst()->ReplaceUsesWithAndRemove(IR::Value{u32{0u}});
    }
}

void LowerGdsAccess(IR::Inst& inst, Info& info, Descriptors& descriptors) {
    const u32 binding = descriptors.Add(BufferResource{
        .used_types = IR::Type::U32,
        .buffer_type = BufferType::GdsBuffer,
        .is_written = true,
    });
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    if (inst.GetOpcode() == Opcode::DataAppend || inst.GetOpcode() == Opcode::DataConsume) {
        RemoveAppendLaneOffset(inst);
        const IR::Value counter =
            inst.GetOpcode() == Opcode::DataAppend
                ? ir.BufferAtomicIAdd(ir.Imm32(binding), inst.Arg(0), ir.Imm32(1u), {})
                : ir.BufferAtomicISub(ir.Imm32(binding), inst.Arg(0), ir.Imm32(1u), {});
        inst.ReplaceUsesWithAndRemove(counter);
        return;
    }

    // LDS operations flagged GDS: storage buffer accesses at the byte offset in elements.
    BufferResource& buffer = info.buffers[binding];
    const IR::U32 offset{inst.Arg(0)};
    const IR::U32 words = ir.ShiftRightLogical(offset, ir.Imm32(1));
    const IR::U32 dwords = ir.ShiftRightLogical(offset, ir.Imm32(2));
    const IR::U32 qwords = ir.ShiftRightLogical(offset, ir.Imm32(3));
    const IR::U32 handle = ir.Imm32(binding);
    switch (const Opcode op = inst.GetOpcode()) {
    case Opcode::SharedAtomicIAdd32:
        inst.ReplaceUsesWith(ir.BufferAtomicIAdd(handle, dwords, inst.Arg(1), {}));
        break;
    case Opcode::SharedAtomicIAdd64:
        inst.ReplaceUsesWith(ir.BufferAtomicIAdd(handle, qwords, IR::U64{inst.Arg(1)}, {}));
        break;
    case Opcode::SharedAtomicISub32:
        inst.ReplaceUsesWith(ir.BufferAtomicISub(handle, dwords, inst.Arg(1), {}));
        break;
    case Opcode::SharedAtomicSMin32:
    case Opcode::SharedAtomicUMin32:
        inst.ReplaceUsesWith(ir.BufferAtomicIMin(handle, dwords, inst.Arg(1),
                                                 op == Opcode::SharedAtomicSMin32, {}));
        break;
    case Opcode::SharedAtomicSMax32:
    case Opcode::SharedAtomicUMax32:
        inst.ReplaceUsesWith(ir.BufferAtomicIMax(handle, dwords, inst.Arg(1),
                                                 op == Opcode::SharedAtomicSMax32, {}));
        break;
    case Opcode::SharedAtomicInc32:
        inst.ReplaceUsesWith(ir.BufferAtomicInc(handle, dwords, {}));
        break;
    case Opcode::SharedAtomicDec32:
        inst.ReplaceUsesWith(ir.BufferAtomicDec(handle, dwords, {}));
        break;
    case Opcode::SharedAtomicAnd32:
        inst.ReplaceUsesWith(ir.BufferAtomicAnd(handle, dwords, inst.Arg(1), {}));
        break;
    case Opcode::SharedAtomicOr32:
        inst.ReplaceUsesWith(ir.BufferAtomicOr(handle, dwords, inst.Arg(1), {}));
        break;
    case Opcode::SharedAtomicXor32:
        inst.ReplaceUsesWith(ir.BufferAtomicXor(handle, dwords, inst.Arg(1), {}));
        break;
    case Opcode::SharedAtomicCmpSwap32: // (address, value, compare)
        inst.ReplaceUsesWith(
            ir.BufferAtomicCmpSwap(handle, dwords, inst.Arg(1), inst.Arg(2), {}));
        break;
    case Opcode::LoadSharedU16:
        inst.ReplaceUsesWith(ir.LoadBufferU16(handle, words, {}));
        buffer.used_types |= IR::Type::U16;
        break;
    case Opcode::LoadSharedU32:
        inst.ReplaceUsesWith(ir.LoadBufferU32(1, handle, dwords, {}));
        break;
    case Opcode::LoadSharedU64:
        inst.ReplaceUsesWith(ir.LoadBufferU64(handle, qwords, {}));
        buffer.used_types |= IR::Type::U64;
        break;
    case Opcode::WriteSharedU16:
        ir.StoreBufferU16(handle, words, IR::U16{inst.Arg(1)}, {});
        inst.Invalidate();
        buffer.used_types |= IR::Type::U16;
        break;
    case Opcode::WriteSharedU32:
        ir.StoreBufferU32(1, handle, dwords, inst.Arg(1), {});
        inst.Invalidate();
        break;
    case Opcode::WriteSharedU64:
        ir.StoreBufferU64(handle, qwords, IR::U64{inst.Arg(1)}, {});
        inst.Invalidate();
        buffer.used_types |= IR::Type::U64;
        break;
    default:
        UNREACHABLE_MSG("Unexpected opcode {}", op);
    }
}

/// The element address of a buffer access:
///   index = (idxen ? vgpr index : 0) + (add_tid ? lane : 0)
///   offset = inst_offset + soffset + (offen ? vgpr offset : 0)
///   address = index * stride + offset, or with swizzling
///     (index / index_stride * stride + offset / element_size * element_size) * index_stride
///     + index % index_stride * element_size + offset % element_size
/// in bytes, then shifted to elements.
IR::U32 BufferAddress(IR::IREmitter& ir, const IR::Inst& inst, const Info& info,
                      const ::Gcn::Buffer& buffer, u32 stride) {
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const u32 inst_offset = flags.inst_offset.Value();
    const bool typed = flags.inst_data_fmt != DataFormat::FormatInvalid;
    const DataFormat data_format =
        typed ? ::Gcn::RemapDataFormat(flags.inst_data_fmt.Value()) : buffer.GetDataFmt();
    const u32 shift = BufferAddressShift(inst, data_format);
    const u32 mask = (1 << shift) - 1;
    const IR::U32 soffset = IR::GetBufferSOffsetArg(&inst);

    // index * stride + constant, both multiples of the element size: shift the constants.
    if (flags.index_enable && !flags.voffset_enable && soffset.IsImmediate() &&
        !buffer.swizzle_enable && !buffer.add_tid_enable && (stride & mask) == 0) {
        const u32 total_offset = soffset.U32() + inst_offset;
        if ((total_offset & mask) == 0) {
            const IR::U32 index = IR::GetBufferIndexArg(&inst);
            return ir.IAdd(ir.IMul(index, ir.Imm32(stride >> shift)),
                           ir.Imm32(total_offset >> shift));
        }
    }

    IR::U32 index = ir.Imm32(0U);
    if (flags.index_enable) {
        const IR::U32 vgpr_index = IR::GetBufferIndexArg(&inst);
        index = ir.IAdd(index, vgpr_index);
    }
    if (buffer.add_tid_enable) {
        ASSERT_MSG(info.sw_stage == SwStage::Compute,
                   "Thread ID buffer addressing is not supported outside of compute.");
        const IR::U32 lane{ir.LaneId()};
        index = ir.IAdd(index, lane);
    }
    IR::U32 offset = ir.Imm32(inst_offset);
    offset = ir.IAdd(offset, soffset);
    if (flags.voffset_enable) {
        const IR::U32 voffset = IR::GetBufferVOffsetArg(&inst);
        offset = ir.IAdd(offset, voffset);
    }
    const IR::U32 const_stride = ir.Imm32(stride);
    IR::U32 address;
    if (buffer.swizzle_enable) {
        const IR::U32 index_stride = ir.Imm32(buffer.GetIndexStride());
        const IR::U32 element_size = ir.Imm32(buffer.GetElementSize());
        const IR::U32 index_msb{ir.IDiv(index, index_stride)};
        const IR::U32 index_lsb{ir.IMod(index, index_stride)};
        const IR::U32 offset_msb{ir.IDiv(offset, element_size)};
        const IR::U32 offset_lsb{ir.IMod(offset, element_size)};
        const IR::U32 address_msb = ir.IMul(
            ir.IAdd(ir.IMul(index_msb, const_stride), ir.IMul(offset_msb, element_size)),
            index_stride);
        const IR::U32 address_lsb = ir.IAdd(ir.IMul(index_lsb, element_size), offset_lsb);
        address = ir.IAdd(address_msb, address_lsb);
    } else {
        address = ir.IAdd(ir.IMul(index, const_stride), offset);
    }
    if (shift != 0) {
        address = ir.ShiftRightLogical(address, ir.Imm32(shift));
    }
    return address;
}

void PatchBufferAddress(IR::Inst& inst, const Info& info) {
    // Constant buffer reads have their dword address from the translator.
    if (inst.GetOpcode() == Opcode::ReadConstBuffer) {
        return;
    }
    const auto buffer = info.buffers[inst.Arg(0).U32()].GetSharp(info);
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    inst.SetArg(IR::LoadBufferArgs::Address, BufferAddress(ir, inst, info, buffer, buffer.stride));
}

/// Cube coordinates from v_cubema/sc/tc are in [1, 2] with the face as the slice (and the cube
/// array index * 8 added): to [0, 1], and face + 6 * array index.
IR::Value CubeCoords(IR::IREmitter& ir, const ::Gcn::Image& image, const IR::Value& x,
                     const IR::Value& y, const IR::Value& face) {
    if (!image.IsCube()) {
        return ir.CompositeConstruct(x, y, face);
    }
    const auto fixed_x = ir.FPSub(IR::F32{x}, ir.Imm32(1.f));
    const auto fixed_y = ir.FPSub(IR::F32{y}, ir.Imm32(1.f));
    const auto fixed_face =
        ir.FPFma(ir.FPFloor(ir.FPDiv(IR::F32{face}, ir.Imm32(8.f))), ir.Imm32(-2.f), IR::F32{face});
    return ir.CompositeConstruct(fixed_x, fixed_y, fixed_face);
}

/// The GCN address registers of an image_sample, in order (8.2.4 "VGPR Usage" in the Sea
/// Islands ISA): [offsets] [bias] [compare] [derivatives] coordinates [lod | clamp].
class AddressRegs {
public:
    explicit AddressRegs(const IR::Inst& inst)
        : bodies{inst.Arg(2).Inst(), inst.Arg(3).Inst(), inst.Arg(4).Inst()},
          last{inst.Arg(5)} {}

    IR::F32 At(u32 index) const {
        if (index < 12) {
            return IR::F32{bodies[index / 4]->Arg(index % 4)};
        }
        ASSERT(index == 12);
        return last;
    }
    IR::F32 Next() {
        return At(next++);
    }
    /// Skips `count` registers; the first of them.
    u32 Take(u32 count) {
        next += count;
        return next - count;
    }

private:
    std::array<const IR::Inst*, 3> bodies;
    IR::F32 last;
    u32 next{};
};

void PatchImageSample(IR::Inst& inst, const Info& info, const ImageResource& image_res,
                      const ::Gcn::Image& image) {
    const IR::Value handle = inst.Arg(0);
    const auto sampler = info.samplers[(handle.U32() >> 16) & 0xFFFF].GetSharp(info);
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::TextureInstInfo>();
    const ImageType view_type = image.GetViewType(image_res.is_array);
    AddressRegs regs{inst};

    // Texel offsets: six-bit signed X [5:0], Y [13:8], Z [21:16].
    const IR::Value offset = [&] -> IR::Value {
        if (!flags.has_offset) {
            return IR::U32{};
        }
        IR::Value packed = regs.Next();
        if (const IR::Inst* const cast = packed.TryInst()) {
            ASSERT(cast->GetOpcode() == Opcode::BitCastF32U32);
            packed = cast->Arg(0);
        }
        const auto read = [&](u32 bit) -> IR::U32 {
            if (packed.IsImmediate()) {
                const u32 imm =
                    packed.Type() == IR::Type::F32 ? std::bit_cast<u32>(packed.F32()) : packed.U32();
                const u16 field = (imm >> bit) & 0x3F;
                return ir.Imm32(s32(field << 26) >> 26);
            }
            return ir.BitFieldExtract(IR::U32{packed}, ir.Imm32(bit), ir.Imm32(6), true);
        };
        switch (view_type) {
        case ImageType::Color1D:
        case ImageType::Color1DArray:
            return read(0);
        case ImageType::Color2D:
        case ImageType::Color2DMsaa:
        case ImageType::Color2DArray:
            return ir.CompositeConstruct(read(0), read(8));
        case ImageType::Color3D:
            return ir.CompositeConstruct(read(0), read(8), read(16));
        default:
            UNREACHABLE();
        }
    }();
    const IR::F32 bias = flags.has_bias ? regs.Next() : IR::F32{};
    const IR::F32 dref = flags.is_depth ? regs.Next() : IR::F32{};
    const auto [ddx, ddy] = [&] -> std::pair<IR::Value, IR::Value> {
        if (!flags.has_derivatives) {
            return {};
        }
        switch (view_type) {
        case ImageType::Color1D:
        case ImageType::Color1DArray: { // du/dx, du/dy
            const u32 r = regs.Take(2);
            return {regs.At(r), regs.At(r + 1)};
        }
        case ImageType::Color2D:
        case ImageType::Color2DMsaa:
        case ImageType::Color2DArray: { // (du/dx, dv/dx), (du/dy, dv/dy)
            const u32 r = regs.Take(4);
            return {ir.CompositeConstruct(regs.At(r), regs.At(r + 1)),
                    ir.CompositeConstruct(regs.At(r + 2), regs.At(r + 3))};
        }
        case ImageType::Color3D: { // (du/dx, dv/dx, dw/dx), (du/dy, dv/dy, dw/dy)
            const u32 r = regs.Take(6);
            return {ir.CompositeConstruct(regs.At(r), regs.At(r + 1), regs.At(r + 2)),
                    ir.CompositeConstruct(regs.At(r + 3), regs.At(r + 4), regs.At(r + 5))};
        }
        default:
            UNREACHABLE();
        }
    }();

    // MSAA images are read by texel: normalized coordinates are scaled by the size. Other images
    // are sampled: unnormalized coordinates are divided by it.
    const bool is_msaa =
        view_type == ImageType::Color2DMsaa || view_type == ImageType::Color2DMsaaArray;
    const bool unnormalized = sampler.force_unnormalized || flags.is_unnormalized;
    const bool needs_size = is_msaa != unnormalized;
    const auto size =
        needs_size ? ir.ImageQueryDimension(handle, ir.Imm32(0u), ir.Imm1(false), flags)
                   : IR::Value{};
    const auto coord = [&](u32 reg, u32 dim) -> IR::Value {
        const auto value = regs.At(reg);
        if (is_msaa) {
            if (unnormalized) {
                return ir.ConvertFToU(32, value);
            }
            const auto extent = ir.ConvertUToF(32, 32, IR::U32{ir.CompositeExtract(size, dim)});
            return ir.ConvertFToU(32, ir.FPMul(value, extent));
        }
        if (unnormalized) {
            const auto extent = ir.ConvertUToF(32, 32, IR::U32{ir.CompositeExtract(size, dim)});
            return ir.FPDiv(value, extent);
        }
        return value;
    };
    const IR::Value coords = [&] -> IR::Value {
        switch (view_type) {
        case ImageType::Color1D: { // x
            const u32 r = regs.Take(1);
            return coord(r, 0);
        }
        case ImageType::Color1DArray: // x, slice
        case ImageType::Color2D:      // x, y
        case ImageType::Color2DMsaa: { // x, y
            const u32 r = regs.Take(2);
            return ir.CompositeConstruct(coord(r, 0), coord(r + 1, 1));
        }
        case ImageType::Color2DArray: { // x, y, slice (or face, for cubes)
            const u32 r = regs.Take(3);
            return CubeCoords(ir, image, coord(r, 0), coord(r + 1, 1), regs.At(r + 2));
        }
        case ImageType::Color3D: { // x, y, z
            const u32 r = regs.Take(3);
            return ir.CompositeConstruct(coord(r, 0), coord(r + 1, 1), coord(r + 2, 2));
        }
        default:
            UNREACHABLE();
        }
    }();

    ASSERT(!flags.has_lod || !flags.has_lod_clamp);
    const bool explicit_lod = flags.has_lod || flags.force_level0;
    const IR::F32 lod = flags.has_lod        ? regs.Next()
                        : flags.force_level0 ? ir.Imm32(0.0f)
                                             : IR::F32{};
    const IR::F32 lod_clamp = flags.has_lod_clamp ? regs.Next() : IR::F32{};

    const IR::Value texel = [&] -> IR::Value {
        if (is_msaa) {
            return ir.ImageRead(handle, coords, {}, ir.Imm32(0U), flags);
        }
        if (flags.is_gather) {
            return flags.is_depth ? ir.ImageGatherDref(handle, coords, offset, dref, flags)
                                  : ir.ImageGather(handle, coords, offset, flags);
        }
        if (flags.has_derivatives) {
            return ir.ImageGradient(handle, coords, ddx, ddy, offset, lod_clamp, flags);
        }
        if (flags.is_depth) {
            return explicit_lod
                       ? ir.ImageSampleDrefExplicitLod(handle, coords, dref, lod, offset, flags)
                       : ir.ImageSampleDrefImplicitLod(handle, coords, dref, bias, offset, flags);
        }
        return explicit_lod ? ir.ImageSampleExplicitLod(handle, coords, lod, offset, flags)
                            : ir.ImageSampleImplicitLod(handle, coords, bias, offset, flags);
    }();
    IR::Value converted = ApplyReadNumberConversionVec4(ir, texel, image.GetNumberConversion());
    if (sampler.force_degamma && image.GetNumberFmt() != ::Gcn::NumberFormat::Srgb) {
        converted = ApplyForceDegamma(ir, texel);
    }
    inst.ReplaceUsesWith(converted);
}

/// Image loads, stores and atomics: the coordinate vector by view type; the mip or sample
/// operand; storage formats swizzled and converted in the shader.
void PatchImageArgs(IR::Inst& inst, const Info& info) {
    if (inst.GetOpcode() == Opcode::ImageQueryDimensions) {
        return;
    }
    const IR::Value handle = inst.Arg(0);
    const ImageResource& image_res = info.images[handle.U32() & 0xFFFF];
    const auto image = image_res.GetSharp(info);
    if (inst.GetOpcode() == Opcode::ImageSampleRaw) {
        PatchImageSample(inst, info, image_res, image);
        return;
    }
    IR::IREmitter ir{*inst.GetParent(), IR::Block::InstructionList::s_iterator_to(inst)};
    auto flags = inst.Flags<IR::TextureInstInfo>();
    const ImageType view_type = image.GetViewType(image_res.is_array);
    const IR::Inst* const body = inst.Arg(1).Inst();
    const auto [coords, extra] = [&] -> std::pair<IR::Value, IR::Value> {
        switch (view_type) {
        case ImageType::Color1D: // x, [lod]
            return {body->Arg(0), body->Arg(1)};
        case ImageType::Color1DArray: // x, slice, [lod]
        case ImageType::Color2D:      // x, y, [lod]
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1)), body->Arg(2)};
        case ImageType::Color2DMsaa: { // x, y, (sample, in its own operand)
            // A _mip access to an MSAA image: only level 0 exists, the sample follows it.
            bool skip_lod = false;
            if (flags.has_lod) {
                const IR::Value mip = body->Arg(2);
                if (!mip.IsImmediate() || mip.U32() != 0) {
                    UNREACHABLE_MSG(
                        "Encountered a _mip instruction with MSAA image, and mipid is non-zero");
                }
                LOG_WARNING(Render_Recompiler, "Encountered a _mip instruction with MSAA image, "
                                               "and mipid is 0, skipping LoD");
                flags.has_lod.Assign(false);
                skip_lod = true;
            }
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1)),
                    body->Arg(skip_lod ? 3 : 2)};
        }
        case ImageType::Color2DArray:     // x, y, slice, [lod]
        case ImageType::Color2DMsaaArray: // x, y, slice, (sample)
        case ImageType::Color3D:          // x, y, z, [lod]
            return {ir.CompositeConstruct(body->Arg(0), body->Arg(1), body->Arg(2)),
                    body->Arg(3)};
        default:
            UNREACHABLE_MSG("Unknown image type {}", view_type);
        }
    }();
    const bool has_ms =
        view_type == ImageType::Color2DMsaa || view_type == ImageType::Color2DMsaaArray;
    ASSERT(!flags.has_lod || !has_ms);
    // A level bound alone (mip fallback) needs no LOD.
    const auto lod =
        flags.has_lod && image_res.mip_fallback_mode != MipStorageFallbackMode::ConstantIndex
            ? IR::U32{extra}
            : IR::U32{};
    const auto sample = has_ms ? IR::U32{extra} : IR::U32{};
    const bool is_storage = image_res.is_written;

    if (inst.GetOpcode() == Opcode::ImageRead) {
        auto texel = ir.ImageRead(handle, coords, lod, sample, flags);
        if (is_storage) {
            texel = ApplySwizzle(ir, texel, image.DstSelect());
        }
        inst.ReplaceUsesWith(ApplyReadNumberConversionVec4(ir, texel, image.GetNumberConversion()));
        return;
    }
    inst.SetArg(1, coords);
    if (inst.GetOpcode() == Opcode::ImageWrite) {
        inst.SetArg(2, lod);
        inst.SetArg(3, sample);
        IR::Value texel = inst.Arg(4);
        if (is_storage) {
            texel = ApplySwizzle(ir, texel, image.DstSelect().Inverse());
        }
        inst.SetArg(4, ApplyWriteNumberConversionVec4(ir, texel, image.GetNumberConversion()));
    }
}

} // namespace

void ResourcePatchingPass(Info& info, const ResourceDiscoveryList& resources,
                          const Profile& profile) {
    // First all bindings (the operand patching reads the resource lists), then the operands.
    Descriptors descriptors{info};
    for (const ResourceDiscovery& resource : resources) {
        const IR::Inst& inst = *resource.user;
        if (IsBufferInstruction(inst)) {
            BindBuffer(resource, descriptors, profile);
        } else if (IsImageInstruction(inst)) {
            BindImage(resource, info, descriptors, profile);
        }
    }
    for (const ResourceDiscovery& resource : resources) {
        IR::Inst& inst = *resource.user;
        if (IsBufferInstruction(inst)) {
            PatchBufferAddress(inst, info);
        } else if (IsImageInstruction(inst)) {
            PatchImageArgs(inst, info);
        } else if (IsDataRingInstruction(inst)) {
            LowerGdsAccess(inst, info, descriptors);
        }
    }
}

} // namespace Shader::Optimization
