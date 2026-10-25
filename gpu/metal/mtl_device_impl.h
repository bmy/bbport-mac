// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifndef __OBJC__
#error "mtl_device_impl.h is for Objective-C++ (.mm) files only"
#endif

#import <Metal/Metal.h>

#include "metal/mtl_device.h"

namespace Metal {

struct Device::Impl {
    id<MTLDevice> device;
};

} // namespace Metal
