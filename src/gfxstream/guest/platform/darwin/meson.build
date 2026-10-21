# Copyright 2026 Mesa3D authors
# SPDX-License-Identifier: MIT

files_libplatform_virtgpu_darwin = files(
    'DarwinSync.cpp',
    'DarwinVirtGpuBlob.cpp',
    'DarwinVirtGpuBlobMapping.cpp',
    'DarwinVirtGpuDevice.cpp',
)

libplatform_virtgpu_darwin = static_library(
    'platform_virtgpu_darwin',
    files_libplatform_virtgpu_darwin,
    cpp_args: gfxstream_guest_args,
    include_directories: [inc_platform_virtgpu, inc_src, inc_include],
)
