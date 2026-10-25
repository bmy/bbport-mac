# A Metal command path for the macOS port

Branch `macos-native-0.6-metal`. Status (2026-10-25): **stage 1, groundwork**.

## Why

On the same Mac (M5 Max) the other Mac port, Wooloomooloo2/bloodborne_mac_port 0.5.7, holds
120 FPS (its cap) at native 2560x1600 with TAA, where this port is GPU-bound at about 50-57 FPS
upscaling 1708x1068 to 2560x1600 with MetalFX: roughly four times the GPU work per frame. That
port encodes Metal itself. This one goes through Vulkan and KosmicKrisp, which upstream 0.5 and
later need (sparse buffers, which MoltenVK lacks). KosmicKrisp is young: every encoder ends with a
full queue barrier, tessellation and rect lists are emulated with compute passes, and each frame
is cut into many render passes.

## The constraint: upstream stays mergeable

That port's 0.5 rewrote the renderer's files in place (about 200 files since its Metal work
began). Here upstream's renderer stays as it is, and a **command layer** under it replaces
KosmicKrisp's command encoding, the shape that port's own design took first
(its docs/DESIGN_METAL_BACKEND.md, "M2 ... a command layer"):

- The renderer records every GPU command through vulkan.hpp's dispatcher. The layer replaces the
  command-buffer and submission functions there (about 45). Command buffers it owns are encoded
  in Metal by it; everything else still goes to KosmicKrisp.
- Resources stay KosmicKrisp's: images, buffers, memory, samplers. Their Metal objects come from
  VK_EXT_metal_objects (this port's patches 0007 and 0008); buffers bind by GPU address.
- Our Metal 4 command buffers go to KosmicKrisp's own queue, in submission order.
- Metal 4 tracks no hazards (KosmicKrisp's resources are untracked): the layer turns the renderer's
  own Vulkan barriers into Metal 4 barriers, where KosmicKrisp puts a full barrier after every
  encoder.
- Pipelines: SPIR-V to MSL through SPIRV-Cross with a direct slot binding model, rect lists as
  triangles, tessellation and geometry stages as compute kernels: the compiler and pipeline
  builder of Huax-1839/bloodborne-macos as adapted by Wooloomooloo2/bloodborne_mac_port (both
  GPL-2.0-or-later), with credits kept.
- A switch (to come with stage 3): KosmicKrisp's encoding stays the default and the fallback until
  the Metal path wins on speed and on the picture.

Upstream merges then touch the layer only where upstream changes the commands it records.

## Stages

1. **Groundwork (now).** Patch 0008: KosmicKrisp exports the device, a memory object's MTLBuffer,
   and the MTLTexture of an image, image view or texel buffer view (0007 exported the queue). An
   interop test, `BB_METAL_INTEROP_TEST=1`: at the renderer's start, 32 rounds of a Vulkan clear,
   a Metal 4 copy and render-pass clear on KosmicKrisp's queue, and a Vulkan readback; the log
   says PASS or where an ordering failed.
2. **Shaders.** Every pipeline the renderer builds is also built as a Metal pipeline state from
   the same key and SPIR-V (a shadow build); failures logged. Nothing drawn through it yet.
3. **First pixels.** The layer forwards and counts; then the presenter's command buffer in Metal;
   then the whole frame. Checked against KosmicKrisp's picture.
4. **Speed.** Ordering of guest memory without full barriers, merged render passes, rect lists as
   triangles, tessellation in runs; A/B measured at each step.

## Stage 1 results

- Patch 0008 and the interop test: written, compile-checked; waiting for the test run.
