/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#include "DarwinVirtGpu.h"

DarwinVirtGpuResource::DarwinVirtGpuResource(int64_t deviceHandle, uint32_t blobHandle,
                                             uint32_t resourceHandle, uint64_t size)
    : mBlobHandle(blobHandle), mResourceHandle(resourceHandle), mSize(size) {}

DarwinVirtGpuResource::~DarwinVirtGpuResource() {}

void DarwinVirtGpuResource::intoRaw() {
    mBlobHandle = INVALID_DESCRIPTOR;
    mResourceHandle = INVALID_DESCRIPTOR;
}

uint32_t DarwinVirtGpuResource::getBlobHandle() const { return mBlobHandle; }

uint32_t DarwinVirtGpuResource::getResourceHandle() const { return mResourceHandle; }

uint64_t DarwinVirtGpuResource::getSize() const { return mSize; }

VirtGpuResourceMappingPtr DarwinVirtGpuResource::createMapping() {
    return nullptr;  // stub constant
}

int DarwinVirtGpuResource::exportBlob(struct VirtGpuExternalHandle& handle) {
    return 0;  // stub constant
}

int DarwinVirtGpuResource::wait() {
    return 0;  // stub constant
}

int DarwinVirtGpuResource::transferToHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    return 0;  // stub constant
}

int DarwinVirtGpuResource::transferFromHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    return 0;  // stub constant
}
