/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#include "DarwinVirtGpu.h"

DarwinVirtGpuDevice::DarwinVirtGpuDevice(enum VirtGpuCapset capset, int32_t descriptor)
    : VirtGpuDevice(capset) {}

DarwinVirtGpuDevice::~DarwinVirtGpuDevice() {}

struct VirtGpuCaps DarwinVirtGpuDevice::getCaps(void) { return mCaps; }

int64_t DarwinVirtGpuDevice::getDeviceHandle(void) { return mDeviceHandle; }

VirtGpuResourcePtr DarwinVirtGpuDevice::createResource(uint32_t width, uint32_t height,
                                                       uint32_t stride, uint32_t size,
                                                       uint32_t virglFormat, uint32_t target,
                                                       uint32_t bind) {
    return nullptr;  // stub constant
}

VirtGpuResourcePtr DarwinVirtGpuDevice::createBlob(const struct VirtGpuCreateBlob& blobCreate) {
    return nullptr;  // stub constant
}

VirtGpuResourcePtr DarwinVirtGpuDevice::importBlob(const struct VirtGpuExternalHandle& handle) {
    return nullptr;  // stub constant
}

int DarwinVirtGpuDevice::execBuffer(struct VirtGpuExecBuffer& execbuffer,
                                    const VirtGpuResource* blob) {
    return 0;  // stub constant
}

VirtGpuDevice* osCreateVirtGpuDevice(enum VirtGpuCapset capset, int32_t descriptor) {
    return nullptr;  // stub constant
}
