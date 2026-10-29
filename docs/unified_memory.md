# Unified memory (integrated GPUs, handhelds, Apple silicon)

On a GPU that shares the system's memory there is no "VRAM" to copy the game's data into: the
port reads the game's memory where the game wrote it. This is what the Steam Deck runs today, and
the mode a port to another unified-memory system (ARM handhelds, Macs through MoltenVK) should
keep.

## What changes on unified memory

`Vulkan::Instance::IsIntegrated()` decides it: the device type (`eIntegratedGpu`), or
`BB_UNIFIED_MEMORY=1` / `0` to override it (a driver that reports another type, a port to a
system where the type is wrong).

- **No VRAM mirrors of the game's data** (`GarlicInVram` is off): vertex, index, constant and
  storage buffers are the game's own memory, bound in place through the layer's memory module
  (`gpu/layer`). No promotions, no copies back, no mirror write traps.
- **No buffer write traps in normal play**: with no second copy, a CPU write needs no
  invalidation. Steam Deck, standing in the world (2026-10-23): `0 write faults/s`,
  `0 mirrors of 0 MiB`, `host copies 0/frame`, `write traps hit 0`.
- **Texture sources are read in place**: an upload detiles straight from the game's memory
  (`ObtainBufferForImage` is the in-place chunk, the detiler runs on the GPU); the CPU copies
  nothing.
- **Memory budget**: all heaps count, at least 8 GiB of the system are left to the system, and
  the texture collector follows the driver's current budget (`texture_cache_memory.cpp`).

What stays: images. The game's textures are tiled (PS4 tiling) and render targets are written by
the GPU in its own layouts, so each texture still has a `VkImage` that the GPU fills by detiling,
and CPU writes under an image are trapped (`BB_IMAGE_TRAPS`) so that it is uploaded again. These
traps fire on texture streaming, not every frame.

## What the guest memory needs from the driver

The game's memory must be GPU-visible at the game's own addresses. One of:

| Way | Where | Needs |
| --- | --- | --- |
| dma-buf chunks mapped at the game's addresses | Linux (Mesa: RADV, ANV, Turnip, Panfrost/PanVK) | `VK_EXT_external_memory_dma_buf`, export of host-visible memory |
| host memory imported by the GPU | Windows (AMD, Intel, NVIDIA), Linux fallback | `VK_EXT_external_memory_host`, `minImportedHostPointerAlignment` ≤ 64 KiB |

A third way, Vulkan memory mapped at the game's addresses (`VK_EXT_map_memory_placed`), is not
implemented; it would suit a system that has neither.

Either way needs `bufferDeviceAddress` too (paged bindings read through a page table) and 64-bit integer and
storage-buffer support in shaders.

MoltenVK: Metal's `newBufferWithBytesNoCopy` is what `VK_EXT_external_memory_host` maps to;
check that the MoltenVK build exposes the extension and that the import alignment is the 16 KiB
page of Apple silicon (the game's memory is placed in 64 KiB units, so either works).

## Checking it

The log at start says which way was taken (`Guest memory: ...`), and every 10 s with frame
stats on: `Layer memory: 0 MiB valid in VRAM, 0 mirrors` and `0 write faults/s` in the frame
stats. Mirrors or write faults there mean the device was not taken for unified memory:
set `BB_UNIFIED_MEMORY=1`.
