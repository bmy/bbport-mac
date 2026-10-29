// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport tests: a VMA-allocated Vulkan image owned by the test (device-local, freed with it).
#pragma once

#include <utility>
#include <vk_mem_alloc.h>
#include "vulkan/vk_common.h"

class TestImage {
public:
    TestImage() = default;
    TestImage(vk::Device device_, VmaAllocator allocator_) : device{device_}, allocator{allocator_} {}
    TestImage(TestImage&& other) noexcept { *this = std::move(other); }
    TestImage& operator=(TestImage&& other) noexcept {
        Release();
        device = other.device;
        allocator = std::exchange(other.allocator, nullptr);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, nullptr);
        return *this;
    }
    ~TestImage() {
        Release();
    }

    void Create(const vk::ImageCreateInfo& info) {
        const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        const VkImageCreateInfo create = info;
        vmaCreateImage(allocator, &create, &alloc, &image, &allocation, nullptr);
    }

    explicit operator vk::Image() const {
        return image;
    }

private:
    void Release() {
        if (image != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator, image, allocation);
            image = VK_NULL_HANDLE;
        }
    }

    vk::Device device;
    VmaAllocator allocator = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};
