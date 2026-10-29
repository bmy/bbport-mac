// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Typed buffer accesses (buffer_load_format, tbuffer_store and the like) become raw loads and
// stores with the format's unpacking or packing done in the shader: the format comes from the
// instruction when it is typed, otherwise from the buffer's sharp.
#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "gcn/sharps.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"
#include "shader/ir/reinterpret.h"
#include "shader/info.h"

namespace Shader::Optimization {

namespace {

using ::Gcn::DataFormat;
using ::Gcn::NumberFormat;

struct FormatInfo {
    DataFormat data_format;
    NumberFormat num_format;
    ::Gcn::CompMapping swizzle;
    ::Gcn::NumberConversion num_conversion;
    u32 num_components;
};

void AssertRawWords(NumberFormat num_format) {
    ASSERT(num_format == NumberFormat::Uint || num_format == NumberFormat::Sint ||
           num_format == NumberFormat::Float);
}

/// The components of a 32-bit word holding a packed format.
IR::Value UnpackWord(IR::IREmitter& ir, const FormatInfo& format, const IR::U32& word) {
    switch (format.data_format) {
    case DataFormat::Format8:
    case DataFormat::Format8_8:
    case DataFormat::Format8_8_8_8:
        return ir.Unpack4x8(format.num_format, word);
    case DataFormat::Format16:
    case DataFormat::Format16_16:
        return ir.Unpack2x16(format.num_format, word);
    case DataFormat::Format10_11_11:
        return ir.Unpack10_11_11(format.num_format, word);
    case DataFormat::Format2_10_10_10:
        return ir.Unpack2_10_10_10(format.num_format, word);
    default:
        UNREACHABLE();
    }
}

/// A packed format's components in one 32-bit word.
IR::U32 PackWord(IR::IREmitter& ir, const FormatInfo& format, const IR::Value& value) {
    switch (format.data_format) {
    case DataFormat::Format8:
    case DataFormat::Format8_8:
    case DataFormat::Format8_8_8_8:
        return ir.Pack4x8(format.num_format, value);
    case DataFormat::Format16:
    case DataFormat::Format16_16:
        return ir.Pack2x16(format.num_format, value);
    case DataFormat::Format10_11_11:
        return ir.Pack10_11_11(format.num_format, value);
    case DataFormat::Format2_10_10_10:
        return ir.Pack2_10_10_10(format.num_format, value);
    default:
        UNREACHABLE();
    }
}

/// The format's components as read from memory (a scalar for one component).
IR::Value LoadComponents(IR::IREmitter& ir, const IR::Value& handle, const IR::U32& address,
                         IR::BufferInstInfo info, const FormatInfo& format) {
    switch (format.data_format) {
    case DataFormat::FormatInvalid:
        return ir.Imm32(0.f);
    case DataFormat::Format8: {
        const auto raw = ir.UConvert(32, ir.LoadBufferU8(handle, address, info));
        const auto unpacked = UnpackWord(ir, format, raw);
        return ir.CompositeExtract(unpacked, 0);
    }
    case DataFormat::Format8_8:
    case DataFormat::Format16: {
        const auto raw = ir.UConvert(32, ir.LoadBufferU16(handle, address, info));
        const auto unpacked = UnpackWord(ir, format, raw);
        if (format.data_format == DataFormat::Format16) {
            return ir.CompositeExtract(unpacked, 0);
        }
        return ir.CompositeConstruct(ir.CompositeExtract(unpacked, 0),
                                     ir.CompositeExtract(unpacked, 1));
    }
    case DataFormat::Format8_8_8_8:
    case DataFormat::Format16_16:
    case DataFormat::Format10_11_11:
    case DataFormat::Format2_10_10_10:
        return UnpackWord(ir, format, IR::U32{ir.LoadBufferU32(1, handle, address, info)});
    case DataFormat::Format16_16_16_16: {
        const auto raw = ir.LoadBufferU32(2, handle, address, info);
        return ir.CompositeConstruct(
            ir.Unpack2x16(format.num_format, IR::U32{ir.CompositeExtract(raw, 0)}),
            ir.Unpack2x16(format.num_format, IR::U32{ir.CompositeExtract(raw, 1)}));
    }
    case DataFormat::Format32:
    case DataFormat::Format32_32:
    case DataFormat::Format32_32_32:
    case DataFormat::Format32_32_32_32:
        AssertRawWords(format.num_format);
        return ir.LoadBufferF32(format.num_components, handle, address, info);
    default:
        UNREACHABLE_MSG("Unsupported buffer data format: {}", format.data_format);
    }
}

/// A load: the components converted, padded to a vec4 with zeros and swizzled.
IR::Value LoadBufferFormat(IR::IREmitter& ir, const IR::Value& handle, const IR::U32& address,
                           IR::BufferInstInfo info, const FormatInfo& format) {
    const IR::Value loaded = LoadComponents(ir, handle, address, info, format);
    boost::container::static_vector<IR::Value, 4> components;
    for (u32 i = 0; i < 4; i++) {
        if (i >= format.num_components) {
            components.push_back(ir.Imm32(0.f));
            continue;
        }
        const auto component =
            IR::F32{format.num_components == 1 ? loaded : ir.CompositeExtract(loaded, i)};
        components.push_back(ApplyReadNumberConversion(ir, component, format.num_conversion));
    }
    return ApplySwizzle(ir, ir.CompositeConstruct(components), format.swizzle);
}

/// A store: the vec4 unswizzled, cut to the format's components, converted and packed.
void StoreBufferFormat(IR::IREmitter& ir, const IR::Value& handle, const IR::U32& address,
                       const IR::Value& value, IR::BufferInstInfo info,
                       const FormatInfo& format) {
    const auto swizzled = ApplySwizzle(ir, value, format.swizzle.Inverse());
    boost::container::static_vector<IR::Value, 4> components;
    for (u32 i = 0; i < format.num_components; i++) {
        const auto component = IR::F32{ir.CompositeExtract(swizzled, i)};
        components.push_back(ApplyWriteNumberConversion(ir, component, format.num_conversion));
    }
    const auto data = components.size() == 1 ? components[0] : ir.CompositeConstruct(components);

    switch (format.data_format) {
    case DataFormat::FormatInvalid:
        break;
    case DataFormat::Format8: {
        const auto packed = PackWord(
            ir, format, ir.CompositeConstruct(data, ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f)));
        ir.StoreBufferU8(handle, address, ir.UConvert(8, packed), info);
        break;
    }
    case DataFormat::Format8_8: {
        const auto packed = PackWord(ir, format,
                                     ir.CompositeConstruct(ir.CompositeExtract(data, 0),
                                                           ir.CompositeExtract(data, 1),
                                                           ir.Imm32(0.f), ir.Imm32(0.f)));
        ir.StoreBufferU16(handle, address, ir.UConvert(16, packed), info);
        break;
    }
    case DataFormat::Format16: {
        const auto packed = PackWord(ir, format, ir.CompositeConstruct(data, ir.Imm32(0.f)));
        ir.StoreBufferU16(handle, address, ir.UConvert(16, packed), info);
        break;
    }
    case DataFormat::Format8_8_8_8:
    case DataFormat::Format16_16:
    case DataFormat::Format10_11_11:
    case DataFormat::Format2_10_10_10: {
        const auto packed = PackWord(ir, format, data);
        ir.StoreBufferU32(1, handle, address, packed, info);
        break;
    }
    case DataFormat::Format16_16_16_16: {
        const auto packed = ir.CompositeConstruct(
            ir.Pack2x16(format.num_format, ir.CompositeConstruct(ir.CompositeExtract(data, 0),
                                                                 ir.CompositeExtract(data, 1))),
            ir.Pack2x16(format.num_format, ir.CompositeConstruct(ir.CompositeExtract(data, 2),
                                                                 ir.CompositeExtract(data, 3))));
        ir.StoreBufferU32(2, handle, address, packed, info);
        break;
    }
    case DataFormat::Format32:
    case DataFormat::Format32_32:
    case DataFormat::Format32_32_32:
    case DataFormat::Format32_32_32_32:
        AssertRawWords(format.num_format);
        ir.StoreBufferF32(format.num_components, handle, address, data, info);
        break;
    default:
        UNREACHABLE_MSG("Unsupported buffer data format: {}", format.data_format);
    }
}

FormatInfo AccessFormat(IR::BufferInstInfo flags, const ::Gcn::Buffer& buffer) {
    if (flags.inst_data_fmt == DataFormat::FormatInvalid) {
        const DataFormat data_format = buffer.GetDataFmt();
        return {
            .data_format = data_format,
            .num_format = buffer.GetNumberFmt(),
            .swizzle = buffer.DstSelect(),
            .num_conversion = buffer.GetNumberConversion(),
            .num_components = ::Gcn::NumComponents(data_format),
        };
    }
    // A typed instruction: its own format.
    const DataFormat data_format = ::Gcn::RemapDataFormat(flags.inst_data_fmt.Value());
    return {
        .data_format = data_format,
        .num_format = ::Gcn::RemapNumberFormat(flags.inst_num_fmt.Value(), data_format),
        .swizzle = ::Gcn::RemapSwizzle(flags.inst_data_fmt.Value(), ::Gcn::IdentityMapping),
        .num_conversion =
            ::Gcn::MapNumberConversion(flags.inst_num_fmt.Value(), flags.inst_data_fmt.Value()),
        .num_components = ::Gcn::NumComponents(data_format),
    };
}

void LowerBufferFormatInst(IR::Block& block, IR::Inst& inst, const Info& info) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const auto& desc = info.buffers[inst.Arg(0).U32()];
    const FormatInfo format = AccessFormat(flags, desc.GetSharp(info));
    const bool is_load = inst.GetOpcode() == IR::Opcode::LoadBufferFormatF32;

    if (format.data_format == DataFormat::FormatInvalid) {
        LOG_WARNING(Render_Recompiler, "Skipping lowering for null buffer sharp");
        if (is_load) {
            inst.ReplaceUsesWithAndRemove(
                ir.CompositeConstruct(ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f)));
        } else {
            inst.Invalidate();
        }
        return;
    }
    if (is_load) {
        inst.ReplaceUsesWithAndRemove(
            LoadBufferFormat(ir, inst.Arg(0), IR::U32{inst.Arg(1)}, flags, format));
    } else {
        StoreBufferFormat(ir, inst.Arg(0), IR::U32{inst.Arg(1)}, inst.Arg(2), flags, format);
        inst.Invalidate();
    }
}

} // namespace

void LowerBufferFormatToRaw(IR::Program& program) {
    const Info& info = program.info;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const IR::Opcode op = inst.GetOpcode();
            if (op == IR::Opcode::LoadBufferFormatF32 || op == IR::Opcode::StoreBufferFormatF32) {
                LowerBufferFormatInst(*block, inst, info);
            }
        }
    }
}

} // namespace Shader::Optimization
