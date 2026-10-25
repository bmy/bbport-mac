// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/shader_module.h"

#include "common/assert.h"
#include "metal/vk_metal_shadow.h" // bbport (macOS): no-ops elsewhere

namespace Vulkan {

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    const auto [result, module] = device.createShaderModule({
        .codeSize = code.size_bytes(),
        .pCode = code.data(),
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "shader module: {}", vk::to_string(result));
    if (MetalShadow::Enabled()) { // bbport (macOS): docs/MACOS_METAL_RENDERER.md, stage 2
        MetalShadow::RegisterModule(module, code);
    }
    return module;
}

} // namespace Vulkan
