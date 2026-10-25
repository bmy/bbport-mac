// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal device the Metal command path builds its pipelines on
// (docs/MACOS_METAL_RENDERER.md). The interface follows Huax-1839/bloodborne-macos's
// Metal::Device (GPL-2.0-or-later), taken from Wooloomooloo2/bloodborne_mac_port at 78b59f5.
// Commands go to KosmicKrisp's own Metal 4 queue (VK_EXT_metal_objects, patches 0007 and
// 0008), which this class does not hold.

#pragma once

#include <memory>
#include <string>

namespace Metal {

class Device {
public:
    /// `mtl_device`: the id<MTLDevice> to use (KosmicKrisp's); null: the system default device.
    explicit Device(void* mtl_device = nullptr);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    struct Impl;
    Impl& GetImpl() const {
        return *impl;
    }

    const std::string& Name() const {
        return name;
    }

private:
    std::unique_ptr<Impl> impl;
    std::string name;
};

} // namespace Metal
