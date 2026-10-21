/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#include "DarwinVirtGpu.h"

DarwinVirtGpuResourceMapping::DarwinVirtGpuResourceMapping(VirtGpuResourcePtr blob, uint8_t* ptr,
                                                           uint64_t size)
    : mBlob(blob), mPtr(ptr) {}

DarwinVirtGpuResourceMapping::~DarwinVirtGpuResourceMapping(void) {}

uint8_t* DarwinVirtGpuResourceMapping::asRawPtr(void) { return mPtr; }
