# A Metal command path for the macOS port

Branches `macos-native-0.6-metal`, then `macos-native-0.6-mtl` (2026-10-26). Status: **stage 5, classic Metal on a queue of our own**.

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

## Stage 4c: no precise maths outside the position stages

SPIRV-Cross translates SPIR-V `NoContraction` (the recompiler's exact float ops) into calls to
`[[clang::optnone]]` helpers (`spvFAdd`, `spvFMul`, ...) that the Metal compiler may not optimise:
10,933 calls in 100 of the game's modules. KosmicKrisp compiles the same SPIR-V without them,
which is why `BB_SPIRV_PRECISE=vertex` made no difference there. The Metal path now drops
`NoContraction` from pixel and compute shaders before translating (the vertex, tessellation and
geometry stages keep it, so every pass computes the same positions); Wooloomooloo2's port measured
the same change on its Metal path at 19.6 -> 12.1 ms a frame. `BB_METAL_PRECISE=all` keeps it
everywhere. The MSL of each mode has its own cache file (`_np`). Checked on the 100 modules: all
pass `spirv-val` after the strip, and the translation has no helper calls left. Not exact any more:
a pixel shader writing its own depth, and positions a compute shader computes; the other port
ships the same split.

## Stage 5: classic Metal on a queue of our own (the backend at the Vulkan-call boundary)

Decided 2026-10-26: the Metal backend sits under the renderer at the Vulkan calls it makes (the
layer), not behind a new interface in the renderer; upstream's renderer stays as it is. Metal 4
has no tessellator, so the frame encoder (`BB_METAL_FRAME=1`) now encodes classic Metal on a
`MTLCommandQueue` of its own instead of Metal 4 on KosmicKrisp's queue:

- **Residency.** KosmicKrisp patch 0009 exports the device's residency set (every allocation it
  makes, committed at each submission); our queue has it, and ours (scratch, the stand-ins).
- **Binding.** Directly on the encoders (`setVertexBuffer:offset:atIndex:` and the like, push
  constants and buffer sizes as inline bytes); command buffers retain nothing
  (`commandBufferWithUnretainedReferences`): KosmicKrisp's objects live as their Vulkan objects,
  ours until the segment's command buffers complete.
- **Order on our queue.** Every encoder waits for one fence when it begins and updates it when it
  ends, so each runs after the one before it in commit order, as KosmicKrisp puts a full barrier
  after each of its encoders. Compute encoders are concurrent, with a memory barrier at each
  Vulkan barrier; copies, fills and updates go to blit encoders.
- **Between the queues.** Two KosmicKrisp timeline semaphores and their shared events: every
  KosmicKrisp submission signals `to_metal`, and waits for `to_kk` when our command buffers ran
  since the last one; a run of our command buffers first waits for `to_metal` (a small command
  buffer that passes the fence on) when KosmicKrisp ran since, and its last command buffer
  signals `to_kk` after the fence. Each `vkQueueSubmit` ends with a KosmicKrisp submission
  after our work, so the renderer's fences, semaphores and `vkQueueWaitIdle` cover it.
- **Tessellation in Metal.** A draw with the game's tessellation (direct, one instance) runs its
  vertex and control shaders as compute kernels into private scratch memory, then
  `drawPatches` with the evaluation and fragment shaders in the pass (reopened, loading): the
  pipelines of stage 2 (`mtl_tessellation.h`). Indirect and instanced ones still go to
  KosmicKrisp. The report counts them (`tessellated draws in Metal`).
- **Geometry-shader draws in Metal** (direct, one instance): per chunk of primitives the export
  and geometry shaders as compute kernels writing vertex records and strip indices into private
  scratch, then indexed strip draws of the generated pass-through vertex function
  (`mtl_geometry.h`). Indirect and instanced ones still go to KosmicKrisp.
- **Clip space z from -w in Metal:** pipelines whose key has the guest's [-w, w] clip space get
  a vertex function (or evaluation function) that converts it at its end (SPIRV-Cross'
  `fixup_clipspace`: z = (z + w) / 2; its own MSL cache files, `_cf`). Before, these draws (about
  1,000 a second at the Byrgenwerth lamp) went to KosmicKrisp.
- **Strips without primitive restart in Metal:** Metal always restarts at the all-ones index,
  which only matters for a draw that uses vertex 65535 (or 2^32 - 1);
  `BB_METAL_STRIP_RESTART=0` sends them to KosmicKrisp as before.
- **The renderer's own pipelines in Metal** (presentation, copies, upscaling, motion vectors,
  the HLE shaders: about 2,100 draws and dispatches a second at the lamp had "no Metal
  pipeline"): the layer now sees every shader module and pipeline the renderer creates
  (`vkCreateShaderModule`, `vkCreateGraphicsPipelines`, `vkCreateComputePipelines`,
  `vkDestroyPipeline`); those made outside a `GameBuildScope` (the game's pipelines are made
  inside one, `render/pipeline.cpp`, and come with their keys as before) are built in Metal from
  their Vulkan create infos at creation: specialization constants fixed into the SPIR-V, the
  dynamic rendering formats, blend states, vertex input, topology, and the state they fix rather
  than take from the dynamic state (cull mode, front face, depth and stencil tests, depth bias),
  which the frame encoder applies for their draws. As Wooloomooloo2's MoltenVK layer at 78b59f5.
  Only vertex and fragment stages; render pass objects are not handled (the renderer has none).
- **Depth/stencil format variants:** Metal wants a pipeline's depth and stencil attachment
  formats to be the pass's exactly; Vulkan ignores an attachment the pipeline does not use (a
  pipeline without stencil drawing into a depth-stencil pass, say). About 1,300 draws a second
  at the lamp fell back for this. A game pipeline (not tessellated, no geometry shader, not a
  generic one) is now built again for the pass's formats on first use, on the shadow build's
  workers (`MetalShadow::FormatVariant`; the draw falls back until it is ready); an aspect its
  key has none of is declared and never tested or written. `BB_METAL_FORMAT_VARIANTS=0` turns
  them off.
- **Descriptor sets in Metal:** the game's pipelines with more resources than push
  descriptors take bind descriptor sets (`render/pipeline.cpp`); their draws fell back (about
  80 a second at the lamp). The layer now sees `vkAllocateDescriptorSets` (a set starts empty)
  and `vkUpdateDescriptorSets` (the writes, decoded as push writes are, kept as a snapshot per
  set); a bind takes the sets' snapshots, and the draws read them as they read pushed ones.
  Sets written by copies or with dynamic offsets still fall back. `BB_METAL_DESCRIPTOR_SETS=0`
  turns it off.
- **Color clears of swizzled formats in Metal:** a format stored swizzled (the BGRA twins,
  4-4-4-4) is cleared with the clear color's channels moved to where they are stored.
- **Switches for bisecting:** `BB_METAL_TESS=0`, `BB_METAL_GEOMETRY=0`,
  `BB_METAL_FORMAT_VARIANTS=0` and `BB_METAL_DESCRIPTOR_SETS=0` keep those draws on KosmicKrisp (with `BB_METAL_PRECISE=all` and
  `BB_METAL_TRANSFERS`).
- **Self-test:** `BB_METAL_FRAME_TEST=1` (with `BB_METAL_FRAME=1`) runs 32 rounds of fills on a
  16 MB buffer at the renderer's start, alternating between KosmicKrisp (fills of differing
  bytes) and our queue (fills of one byte value, the copy), in one submission and split in two,
  and reads them back: `Metal frame test: PASS`, or which write landed out of order, or that the
  GPU did not finish (a wait between the queues never signalled).
- **Ordering by the renderer's barriers (`BB_METAL_ORDER=barriers`; the default is `chain`):**
  each encoder updates a fence of its own; encoders wait for the fences of the encoders before
  the last barrier (transitively everything earlier), not for the encoder just before them, so
  the encoders between two barriers may overlap on the GPU as Vulkan allows. A reopened render
  pass still waits for its last part (and a tessellated draw's kernels in between); a compute
  encoder takes a barrier as a memory barrier while it is the only encoder since the last one.
  Each Metal segment starts after the chain and passes it on at its end (one small blit encoder
  that waits for all of its encoders), so segments, KosmicKrisp's work and submissions stay in
  order as before. It relies on the renderer's barriers being complete: KosmicKrisp runs every
  encoder after the one before, so a missing barrier never showed until now.
- Not yet: batched tessellation, a KosmicKrisp-free frame. The classic device and encoder of
  Wooloomooloo2's native renderer are in the tree (`mtl_device`, `mtl_encoder`, `mtl_order`);
  the frame encoder's barrier ordering above is its own, simpler (whole barriers, no per-resource
  tracking).

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
