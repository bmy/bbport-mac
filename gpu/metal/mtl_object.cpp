// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: from Wooloomooloo2/bloodborne_mac_port at ff64835 (GPL-2.0-or-later),
// gpu/shadps4/video_core/renderer_metal/, its include paths adapted.
// bbport: Metal::Object reference counting through CoreFoundation (plain C++).

#include <CoreFoundation/CoreFoundation.h>
#include "metal/mtl_object.h"

namespace Metal {

Object Object::Retain(void* object) {
    if (object) {
        CFRetain(object);
    }
    return Adopt(object);
}

void Object::Reset() {
    if (ptr) {
        CFRelease(ptr);
        ptr = nullptr;
    }
}

} // namespace Metal
