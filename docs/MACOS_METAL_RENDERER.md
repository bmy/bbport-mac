# A Metal command path for the macOS port

Branch `macos-native-0.6-metal`. Status (2026-10-25): stage 1 passed; **stage 2, the shadow build**.

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

## Where the code comes from

Wooloomooloo2/bloodborne_mac_port at **78b59f5** (2026-10-16 04:42), the last commit with the
command layer, before that port removed it and rewrote its renderer the same day: about 11,000
lines in `gpu/shadps4/video_core/renderer_metal/` (the layer `mtl_vk_layer.mm`, frame encoding
`mtl_frame.mm`, shader translation `mtl_shader.mm` and `mtl_spirv_*`, pipeline states
`mtl_pso_cache.mm` and `vk_to_mtl.mm`, geometry and tessellation `mtl_geometry.mm`,
`mtl_tessellation.mm`, the shadow build `mtl_shadow.mm`).

At that commit its Metal path ran level with MoltenVK (96.4 vs 94.5 FPS, 1080p native, M4 Max).
Its later speed came from what followed: its own ordering of untracked guest buffers, a pipeline
archive, fewer passes, deferred copies. Most of that can live in the layer too; whatever needs the
renderer itself is a decision for stage 4.

Differences to work through here: that layer ran on MoltenVK, whose resources Metal tracks, with
the classic Metal API on MoltenVK's queue. KosmicKrisp's queue is a Metal 4 queue and its resources
are untracked: the encoders become Metal 4 ones (argument tables, residency sets, barriers from
the renderer's own Vulkan barriers). Its tree is upstream's older layout (`gpu/shadps4/...`); this
one is upstream 0.6's (`gpu/render`, `gpu/textures`, `gpu/buffers`, ...), with the same
`GraphicsPipelineKey` and shader recompiler.

## Stages

1. **Groundwork (done).** Patch 0008: KosmicKrisp exports the device, a memory object's MTLBuffer,
   and the MTLTexture of an image, image view or texel buffer view (0007 exported the queue). An
   interop test, `BB_METAL_INTEROP_TEST=1`: at the renderer's start, 32 rounds of a Vulkan clear
   (2048x2048), a Metal 4 copy and render-pass clear on KosmicKrisp's queue, and a Vulkan
   readback; the log says PASS or where an ordering failed. KosmicKrisp's own end-of-encoder
   barrier already orders its work before ours, so a PASS tests our producer barrier (Metal before
   Vulkan) and queue order; `BB_METAL_INTEROP_TEST=2` runs without the render pass's barriers as a
   negative control. Patch 0008 also has two chain entries of this port's own (`gpu/metal/kk_exports.h`):
   an image view's texture for one use (sampled, storage, render with its level and slice, input)
   and a VkBuffer's own MTLBuffer and offset (sparse buffers have their own; a buffer on heap
   memory has its own sub-buffer of the heap, aliasing the memory's). They need the validation
   layers off.
2. **Shaders (now).** Every pipeline the renderer builds is also built as a Metal pipeline state
   from the same key and SPIR-V (a shadow build); failures logged. Nothing drawn through it yet.
3. **First pixels.** The layer forwards and counts (3a, `BB_METAL_TRACE=1`); then the
   presenter's command buffer in Metal; then the whole frame. Checked against KosmicKrisp's
   picture.
4. **Speed.** Ordering of guest memory without full barriers, merged render passes, rect lists as
   triangles, tessellation in runs; A/B measured at each step.

## Stage 1 results

- 2026-10-25, M5 Max, macOS 27.0.1: `Metal interop test: PASS` (32 rounds, about a second at the
  renderer's start).

## Stage 2: the shadow build

`BB_METAL_SHADOW=1` (native GPU process only). Each game pipeline the renderer creates is built a
second time as a Metal pipeline state, on the same pipeline-creation thread, right after the
Vulkan one, preloaded ones from the pipeline cache included; the log counts the results (`Metal
shadow: N pipelines: ...` at the first and every 50th) and names the first failures. The builds
run on two worker threads of their own, off the pipeline-creation path (they stalled area loads
for seconds when they ran on it); a pipeline whose Metal build is not done yet draws through
KosmicKrisp.
Off, it costs one branch per pipeline and shader module.

- Files, from 78b59f5's `renderer_metal/` with includes and names adapted to 0.6
  (`AmdGpu` -> `Gcn`, `LiverpoolToVK` -> `GcnToVk`): `gpu/metal/mtl_shader.*` and
  `mtl_spirv_{rectlist,pervertex,geometry}.*` (SPIR-V to MSL; the rect list, per-vertex and
  geometry rewrites), `mtl_pso_cache.*` and `vk_to_mtl.*` (pipeline states), `mtl_tessellation.*`,
  `mtl_geometry.*`, `mtl_shadow.*`, `vk_metal_shadow.h` (the hooks), `mtl_device.*`.
- Left out: its MoltenVK layer (`mtl_vk_layer.mm`), frame encoding (`mtl_frame.mm`) and the
  shadow build of the layer's own (host pass) pipelines, which came from that layer. Stage 3
  brings a layer of its own for KosmicKrisp.
- Changed: blend states are the Vulkan pipeline's own (our `BlendAttachment`, GCN's differences
  emulated) instead of a second copy of that logic; vertex inputs come from the fetch shader, step
  rates from the static vertex input (preloaded pipelines have no runtime infos); a vertex format
  the key leaves undefined is read as 32-bit floats, as the Vulkan pipeline does; translations are
  kept by SPIR-V content, not by module handle (handles come back after a module is destroyed),
  and a destroyed pipeline's handle is forgotten; the Metal device is the system default (the same
  object as KosmicKrisp's on Apple silicon; stage 3 takes the exported one). Vertex formats come
  from the key, which has them without dynamic vertex input only (true of KosmicKrisp): with
  dynamic vertex input the shadow build skips graphics pipelines. The Metal binary archive is
  written every 200 pipelines.
- Hooks in upstream files: `vulkan/shader_module.cpp` (`CompileSPV`) and
  `render/pipeline_store.cpp` (cached SPIR-V): the module's SPIR-V; `render/pipeline.cpp` (the
  compute and graphics constructors, after the Vulkan create; `~Pipeline`), each a guarded block
  that is a no-op without `BB_METAL_BACKEND`.
- SPIRV-Cross `vulkan-sdk-1.4.363.0` (f11ba9f0), static, MSL and GLSL back ends only, from
  `setup_deps.sh` with `BB_ARCH=arm64`, with a stamp naming the pin. CMake takes it only from a
  prefix with that stamp (never Homebrew's), builds the Metal files and defines
  `BB_METAL_BACKEND` only on arm64 macOS with it found; every other build is unchanged. The pin is
  in the MSL cache's header, so a new pin translates again.
- Licence: SPIRV-Cross is Apache-2.0, linked statically into the native GPU core. Apache-2.0 goes
  with GPL-3.0 (this code is GPL-2.0-or-later), not with GPL-2.0-only, so a binary with it is
  distributed under GPL-3.0; `package.sh` ships its licences (`licenses/SPIRV-Cross`), not yet
  the GPL-3.0 text. To settle before a release from this branch.
- MSL and pipeline caches: the user cache directory, `metal-shadow/` (not per game: the MSL
  files are named by SPIR-V hash).

## Stage 3a: the layer, counting

`BB_METAL_TRACE=1` (any platform): `gpu/metal/mtl_layer.cpp` installs itself over about 90 of
vulkan.hpp's default dispatcher's functions right after the device's dispatcher init
(`vulkan/instance.cpp`), forwards every call unchanged and counts them. Every 5 s, from a
submission, the log has `Metal layer: per second N commands, N submissions of N command buffers
(N commands each); calls per second: ...`. Lookups of `vkCmd*` through `vkGet*ProcAddr` (FSR 3.1,
the overlay, VMA, DLSS) get the layer's entry for what it covers (where the driver has the
function) and are logged once each, so those calls are counted too.

What it is for: the command mix the Metal encoder has to cover first, and which code records
around the dispatcher. Known so far: `upscale/fsr411` (FSR 4.1.1) and the FSR 4 v07 provider
call the loader's `vkCmd*` directly, so their command buffers can never be encoded by the layer;
those must stay KosmicKrisp's (the layer will own a command buffer only when everything recorded
into it comes through it).

Notes for 3b (the layer encoding): KosmicKrisp commits at vkQueueSubmit, waits included, so a
submission with our command buffers in it is split: KosmicKrisp first gets the waits (no command
buffers), then ours are committed to its queue, then KosmicKrisp gets the rest with the signals
and the fence. Queue order then carries the semaphores both ways.

## Stage 3b: the frame encoder (`BB_METAL_FRAME=1`)

`gpu/metal/mtl_frame.{h,mm}`, driven by the layer: every primary command buffer begun through
the dispatcher is owned (secondaries stay KosmicKrisp's); its commands go to the frame encoder,
which keeps the Vulkan state as recorded and encodes what it can in Metal 4 on KosmicKrisp's
queue. Frame mode also turns on the shadow build (its draws need the Metal pipelines) and needs
the validation layers off.

- **Segments.** An owned command buffer becomes a sequence of segments: Metal 4 command buffers
  (one allocator per recording, argument tables for the vertex, fragment and compute stages,
  scratch memory for push constants and buffer sizes in a residency set added to the queue) and
  KosmicKrisp command buffers of the encoder's own (a command pool per recording thread, recycled
  by a timeline semaphore). At submission (`vkQueueSubmit`, `vkQueueSubmit2`) a batch with owned
  command buffers goes to the queue as: the waits (a submission without command buffers), the
  segments in order (Metal ones committed, KosmicKrisp ones submitted), the signals and the fence.
  KosmicKrisp commits at submission, so queue order carries the semaphores both ways.
- **What Metal encodes.** Render passes (`vkCmdBeginRendering`; reopened, loading, after anything
  that has to come between draws), draws and indexed and indirect draws with a Metal pipeline
  from the shadow build, dispatches (direct and indirect), push descriptors (buffers by GPU
  address, image views by their sampled or storage texture, samplers made from their create
  infos), push constants, vertex and index buffers, the dynamic state. Every encoder ends with a
  full producer barrier, as KosmicKrisp's do; a pipeline barrier is an encoder barrier in a
  compute encoder and ends a render encoder.
- **What goes to KosmicKrisp (the fallback).** Each draw or dispatch Metal does not take, with the
  reason counted: no Metal pipeline (host passes, FSR 3.1's own pipelines), tessellation and
  geometry shaders (Metal 4 has no tessellation stage), topologies Metal lacks, bound descriptor
  sets, a descriptor without a Metal object, attachment or sample formats other than the
  pipeline's, rasterizer discard, the depth bounds test, a color write mask other than the
  pipeline's, a GPU draw count, 8-bit indices, render passes with resolves, layers or view masks.
  Its segment gets the whole recorded state when it opens (pipelines, descriptors, push
  constants, vertex and index buffers, dynamic state, the render pass loading), and the state
  commands as they come while it stays open. Transfers, clears, blits, query and timestamp
  commands always go there; while an occlusion query is active everything does (a query begins
  and ends in one command buffer, and in one render pass instance when begun in one).
  Debug labels are dropped.
- **Log:** every 5 s, `Metal frame: per second N draws and N dispatches in Metal, N draws and N
  dispatches and N other commands to KosmicKrisp (N segments, N switches); ...; to KosmicKrisp
  because: <the top reasons>`; once each, what it cannot do at all.
- **Not yet:** a library that records with the loader's own functions (FSR 4.1.1, FSR 4 v07)
  cannot be used in frame mode (its commands would reach KosmicKrisp's copy of the command
  buffer, which is never submitted: choose MetalFX or FSR 3.1). Occlusion queries and
  tessellated draws keep their passes on KosmicKrisp.

## Stage 4a: transfers in Metal

In frame mode, copies (buffer, buffer to image, image to buffer, image to image), fills, updates
and image clears are encoded in Metal (the compute encoder's copies; clears as render passes of
their own) instead of switching to a KosmicKrisp segment for each. They go to KosmicKrisp still
when: an occlusion query is active; the image is not one made with `vkCreateImage` (swapchain
images) or its Metal pixel format is not exactly its Vulkan format's; the aspect or format has no
Metal copy (a size table as Wooloomooloo2's); an image copy between different formats or sample
counts, or of depth/stencil; a clear of a texture without render target usage or not 2D; a fill
whose four bytes differ. `BB_METAL_TRANSFERS=0` sends them all to KosmicKrisp (for bisecting a
broken picture); a comma-separated list of kinds (`copy`, `fill`, `update`, `b2i`, `i2b`,
`image`, `clear`) keeps only those in Metal, and `-kind` entries send just those to KosmicKrisp
(the log names the kinds in Metal). Known bug (2026-10-26): with transfers in Metal, the distant
cathedral behind the Grand Cathedral shows rainbow (corrupt compressed) textures. Blits and
resolves still go to KosmicKrisp.

## Stage 4b: robust buffer access

The renderer relies on `robustBufferAccess2` and `nullDescriptor`: a storage buffer read past its
bound range gives zero, a write there is dropped, and a null buffer descriptor is an empty
buffer. KosmicKrisp provides that; a Metal shader reads raw addresses. So every module is
rewritten before SPIRV-Cross (`gpu/metal/mtl_spirv_robust.{h,cpp}`): each element access of the
recompiler's buffers (`struct { T data[]; }`) is checked against `OpArrayLength`, which
SPIRV-Cross reads from its buffer of buffer sizes, filled from the bound descriptor's range (0 for
a null descriptor or an unbound slot). Loads give zero out of range; stores and atomics run only
in range (their block split into a selection); the index is clamped to 0 as well, so nothing is
addressed outside the buffer (a load still reads the first element when the bound range is
smaller than one). A shape the rewrite does not know (a storage buffer of another shape, a
store in a loop header, a buffer pointer used otherwise, variable pointers) fails the shader's
Metal build rather than leaving an access unchecked; such a shader then never draws in Metal. With it, draws and dispatches with a null storage buffer descriptor (the commonest
fallback reason, about 30,000 to 45,000 draws a second) are encoded in Metal. MSL cache version 5.
Checked on 100 of the game's cached modules: all rewrite, pass `spirv-val` and convert.

## Results so far (M5 Max, 2560x1600, MetalFX native AA)

- Stage 2 (2026-10-25): about 650 pipelines in a session; all but 1 render pipeline (a fragment
  input its vertex shader does not write) and 5 tessellation pipelines (patch constant sizes)
  build in Metal.
- Stage 3a: in play, about 400,000 commands a second: 50,000 draws (800 a frame), 9,000 render
  passes (150 a frame), 13,000 barriers, 8,000 buffer copies, 1,200 command buffers.
- Stage 3b, first run: no GPU errors; only about 2% of draws in Metal, the rest to KosmicKrisp
  with 7,000 switches a second, so slower than KosmicKrisp alone. The reasons, per second: the
  depth bounds test (about 20,000: Metal 4 has one, now used), a descriptor without a Metal object
  (about 25,000: now logged by kind), no Metal pipeline (3,000), clip space z from -w (2,500),
  tessellation, strips without primitive restart. Long stalls when new areas load: the Metal
  builds on the pipeline-creation path (now on worker threads).
