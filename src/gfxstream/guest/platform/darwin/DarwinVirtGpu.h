/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "VirtGpu.h"

class DarwinVirtGpuResource : public std::enable_shared_from_this<DarwinVirtGpuResource>,
                              public VirtGpuResource {
   public:
    DarwinVirtGpuResource(int64_t deviceHandle, uint32_t blobHandle, uint32_t resourceHandle,
                          uint64_t size);
    ~DarwinVirtGpuResource();

    void intoRaw() override;
    uint32_t getResourceHandle() const override;
    uint32_t getBlobHandle() const override;
    uint64_t getSize() const override;
    int wait() override;

    VirtGpuResourceMappingPtr createMapping(void) override;
    int exportBlob(struct VirtGpuExternalHandle& handle) override;

    int transferFromHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) override;
    int transferToHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) override;

   private:
    uint32_t mBlobHandle;
    uint32_t mResourceHandle;
    uint64_t mSize;
};

class DarwinVirtGpuResourceMapping : public VirtGpuResourceMapping {
   public:
    DarwinVirtGpuResourceMapping(VirtGpuResourcePtr blob, uint8_t* ptr, uint64_t size);
    ~DarwinVirtGpuResourceMapping(void);

    uint8_t* asRawPtr(void) override;

   private:
    VirtGpuResourcePtr mBlob;
    uint8_t* mPtr;
};

class DarwinVirtGpuDevice : public VirtGpuDevice {
   public:
    DarwinVirtGpuDevice(enum VirtGpuCapset capset, int32_t descriptor = -1);
    ~DarwinVirtGpuDevice() override;

    int64_t getDeviceHandle(void) override;

    struct VirtGpuCaps getCaps(void) override;

    VirtGpuResourcePtr createBlob(const struct VirtGpuCreateBlob& blobCreate) override;
    VirtGpuResourcePtr createResource(uint32_t width, uint32_t height, uint32_t stride,
                                      uint32_t size, uint32_t virglFormat, uint32_t target,
                                      uint32_t bind) override;

    VirtGpuResourcePtr importBlob(const struct VirtGpuExternalHandle& handle) override;
    int execBuffer(struct VirtGpuExecBuffer& execbuffer, const VirtGpuResource* blob) override;

   private:
    int64_t mDeviceHandle;
    struct VirtGpuCaps mCaps;
};
