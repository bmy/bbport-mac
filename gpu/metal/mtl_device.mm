// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): mtl_device.h. Built with ARC.

#include "common/assert.h"
#include "metal/mtl_device_impl.h"

namespace Metal {

Device::Device(void* mtl_device) : impl{std::make_unique<Impl>()} {
    @autoreleasepool {
        impl->device =
            mtl_device ? (__bridge id<MTLDevice>)mtl_device : MTLCreateSystemDefaultDevice();
        ASSERT_MSG(impl->device, "No Metal device");
        name = impl->device.name.UTF8String;
    }
}

Device::~Device() = default;

} // namespace Metal
