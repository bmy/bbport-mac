# Native Apple Silicon GPU process (macOS): design

Status: first working version (branch `macos-native`): `BB_NATIVE_GPU=1` runs the GPU side in a
native arm64 process. The in-process x86-64 path stays the default until the native one is faster
and stable.

## Why

On the Mac everything runs as x86-64 under Rosetta 2: the game's own code (which has to), but also
bbport's GPU emulation (libbbgpu: command processor, caches, shader recompiler, Vulkan renderer), the
Vulkan driver (KosmicKrisp) and even Apple's Metal framework and GPU driver, whose x86-64 slices are
translated too. Measurements on the Great Bridge at 1440p Quality (2026-10-07):

- About 1,300 draws a frame cost 8–10 µs each on the GPU thread (~12 ms of the 16.7 ms frame).
- Removing bbport's own waits (host copies, earlier fences) did not raise the frame rate: each time the
  wait moved to the recording threads or to the GPU. The limit is how fast commands become Metal
  work: libbbgpu, KosmicKrisp and Metal's user-space driver, all translated.
- Letting the GPU thread run two frames ahead doubled the per-draw cost (contention), so there is no
  spare CPU-side headroom to buy with more parallelism.
- At 1440p Native AA the GPU itself becomes the limit (up to 41% of the time waiting for it).

A native arm64 build of that whole stack removes the translation cost (typically 20–40% for code like
this; more for the Metal driver's tight loops) and lets MetalFX run without Rosetta's limits. The
game's code stays x86-64 either way.

The `CPU by thread` line in the frame stats (added for this) shows how the CPU time splits between
the port's threads, the game's and the system's: the part that can move native.

## Constraint: one process cannot mix the two

macOS does not let x86-64 and arm64 code share a process. So the GPU side moves to a second process,
`bb-gpu` (arm64), started by `bb-probe` (x86-64, the game, as now). They share the guest's memory and
talk through shared-memory queues.

```
 bb-probe (x86-64, Rosetta)                    bb-gpu (arm64, native)
 ┌──────────────────────────────┐              ┌──────────────────────────────────┐
 │ game code, runtime (src/*.c) │  guest       │ Liverpool (PM4), rasterizer,     │
 │ HLE: Gnm command builders,   │  memory      │ buffer/texture caches, shader    │
 │ VideoOut, event queues,      │◄────────────►│ recompiler, presenter, overlay   │
 │ AvPlayer, audio              │  (one shm,   │ KosmicKrisp (native) → Metal      │
 │ remote GPU shim (forwarding) │  same addrs) │ SDL window, keyboard, gamepads   │
 └──────────────┬───────────────┘              └───────────────┬──────────────────┘
                └──── command/reply rings + wakeups (shared memory) ────┘
```

### Why this split line

- **The runtime↔GPU interface is already narrow.** `gpu/shim/bbgpu.cpp` imports a dozen C functions
  from the runtime (memory region queries, protection, write-backing, clocks, map/unmap/invalidate
  hooks) and exports the HLE symbol table and `bbgpu_*` entry points.
- **The HLE libraries mostly build PM4 packets in guest memory**, which needs no GPU. Only a few calls
  reach the GPU core: `SubmitGfx`, `SubmitAsc`, `SubmitDone`, `IsGpuIdle`, the compute queue table, and
  the VideoOut driver (open/close, register buffers, flip, buffer attributes, VO port). Interrupts
  (end of pipe, flip, vblank) come back as events for the guest's event queues.
- **Guest memory is already shareable.** On macOS the runtime backs direct and flexible memory with
  one POSIX shared memory object (`runtime_memory.c`, `pool()`), mapped `MAP_SHARED` at the guest
  address, with the PS4's 16 KiB page size, the same as arm64 macOS. `bb-gpu` maps the same object at
  the same addresses (it reserves the guest range at startup), so every guest pointer in libbbgpu
  stays valid unchanged. Only executable guest memory is private, and the GPU never reads code.
- **Upstream merges stay manageable.** The split lives in new files (a remote shim on each side);
  shadPS4/bbport code changes are limited to portability (below) and a few hooks.

## Interfaces

Frontend → backend (bb-probe → bb-gpu):

| Message | Today | Notes |
|---|---|---|
| Init (config, shm fd) | `bbgpu_init` | fd passed at spawn |
| Map / Unmap / Invalidate | runtime hooks | Map carries the physical offset and protection so bb-gpu mirrors it |
| Asset write / CPU write | note-write and cpu-write hooks | asynchronous |
| SubmitGfx / SubmitAsc / SubmitDone / MapComputeQueue | `Liverpool::*` from Gnm HLE | asynchronous; the command buffers are read from shared memory |
| VideoOut Open/Close/RegisterBuffers/SubmitFlip/ChangeBufferAttribute | `VideoOutDriver` | the port state becomes a shared struct |
| Write fault (address) | `bbgpu_handle_fault` | synchronous: bb-gpu marks the page and asks to unprotect |

Backend → frontend:

| Message | Today | Notes |
|---|---|---|
| Protect (range, read, write) | `runtime_memory_gpu_protect` | synchronous; the frontend applies it (Rosetta emulates 4 KiB protection, only it can) |
| Interrupt (EOP, compute, flip, vblank) | `Platform::IrqC` → event queues | asynchronous, triggers guest events |
| Pad and keyboard state | `runtime_pad.c` reads SDL | shared struct; the window and its events belong to bb-gpu |

Labels, readbacks and fences are written by bb-gpu straight into shared guest memory: its mapping has
no write protection, the same thing the "write backing" path does today. Those writes need release
ordering on arm64 (below).

Waits use `os_sync_wait_on_address` with `OS_SYNC_WAIT_ON_ADDRESS_SHARED` on the shared rings
(macOS 14.4+), so neither side spins.

## Risks

1. **Memory ordering.** Under Rosetta, x86 TSO ordering holds for every store; native arm64 does not.
   libbbgpu is multi-threaded C++ that has only ever run on x86, so a race TSO hides can appear.
   Mitigation: ThreadSanitizer runs of the backend, release stores/fences where bb-gpu publishes data
   to the guest (labels after readbacks, flip status), acquire loads where it reads guest flags.
2. **Write faults cross processes.** About 1,500–2,000 a second today; each becomes a round trip
   (~10–20 µs). Acceptable, and the fault windows already batch them; a shared dirty bitmap can remove
   the round trip later.
3. **Latency of synchronous calls** (protect, idle checks, `SubmitDone` ordering). Most traffic is
   asynchronous; synchronous calls are rare per frame.
4. **Window ownership.** The window, keyboard and gamepads move to bb-gpu; bb-probe gets the pad state
   through shared memory. Audio stays in bb-probe.
5. **x86-only code in libbbgpu**: TSC reads and pause instructions (portable helpers), x86 register
   access in signal handlers (frontend only), and the shader recompiler's SRT walker, which JITs x86
   code with Xbyak and needs an arm64 path (an interpreter of the same walk is enough: it runs once per
   draw with extended user data).
6. **Guest memory outside the pool.** Shader resource tables can sit in the game image's own
   data (found by the SRT check, 2026-10-07), which the loader maps privately, not from the
   shared pool. bb-gpu needs those segments too: the loader has to place the image's writable
   data in shared memory (or bb-gpu mirrors it), else the walker reads zeros.
7. **Descriptor passing.** `shm_open` sets close-on-exec, so the pool's descriptor has to be
   cleared of it (or duplicated into the child with posix_spawn file actions) for bb-gpu to
   inherit it; the Mac test caught this.
8. **Upstream churn.** Keep the split in new files; portability fixes are candidates to send upstream.

## Phases

1. **Measure** (done): `CPU by thread` in the frame stats.
2. **libbbgpu builds for arm64** (done): portable CPU helpers, SRT walker interpreter, native
   dependencies (`deps-arm64`: SDL3, Vulkan loader, KosmicKrisp, fmt, xxHash…), a `bb-gpu`
   executable that links the core. Checked here with an arm64 syntax pass of every source.
3. **Shared memory and channel** (done): bb-probe spawns bb-gpu, passes the shm fd; map mirroring; rings and
   wakeups; a smoke test that submits nothing but proves both sides see the same memory.
4. **Remote shim** (first version): forward Gnm submits, VideoOut and interrupts; first frame on
   screen.
5. **Window, input, overlay, settings** in bb-gpu; pad state to bb-probe.
6. **Correctness and speed**: TSan, release/acquire at the guest boundary, thread priorities (QoS),
   MetalFX native, compare against the in-process path in the same spots.
7. **Default** once faster and stable; `BB_NATIVE_GPU=0` keeps the in-process path.

## Status (2026-10-07)

**Phase 2 (libbbgpu builds for arm64).** An arm64 syntax pass over the GPU library's sources
(macOS 26.1 SDK, `--target=arm64-apple-macosx14`) compiles 150 of 152 cleanly; the other two only
miss things the build provides (miniz's header, the font path define), and AvPlayer's two FFmpeg
files were not checked here. What changed:

- `shim/bbport_cpu.h`: `Ticks()`, `Pause()`, `TicksFromUs()` instead of `__rdtsc`, pause builtins
  and `<x86intrin.h>` (x86-64 behaviour unchanged).
- Signal-context helpers read pc/sp/fp on arm64.
- The SRT walker: x86 code via Xbyak on x86-64, bytecode for the same walk elsewhere
  (`Shader::RunSrtWalker`). `BB_SRT_CHECK=1` on the x86-64 build runs both on every walk and logs
  any difference; with `BB_PIPELINE_CACHE=0` every shader gets checked. This verifies the bytecode
  in the real game before any native build exists. Result on 0.4 (2026-10-07, shader cache off,
  several areas): 8,388,608 walks, 0 differences.
- A non-x86 build keeps its own pipeline cache folder (the cache stores the walkers' host code).
- `BB_ARCH=arm64 bash tools/macos/setup_deps.sh` builds the native libraries into `deps-arm64`
  (including KosmicKrisp), and `bash tools/macos/build_gpu_native.sh` builds
  `out/gpu-arm64/libbbgpu.dylib`. Neither has run on a Mac yet.

**Phase 3 (shared memory and channel).** `gpu/shim/remote/`:

- `bb_channel.h`: the two-way channel (ordered messages, synchronous calls that may nest).
- `bb_memory_mirror.h`: the guest range reserved and the shared pool mirrored at the same addresses.
- `bb_protocol.h`: the first messages (Hello, Map/Unmap, WriteFault, Protect, Shutdown).

Tests, passing on Linux; on the Mac they test the real pairing (a universal build run under
`arch -x86_64` starts the GPU side as native arm64):

```
c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_channel.cpp -o /tmp/channel-test && /tmp/channel-test
clang++ -std=c++20 -O2 -arch x86_64 -arch arm64 -Igpu/shim tests/test_remote_memory.cpp -o /tmp/memory-test && arch -x86_64 /tmp/memory-test
```

Result on the M5 Max (macOS 27.0.1, 2026-10-07): both pass. The memory test ran with the game side
as x86-64 under Rosetta and the GPU side as a native arm64 process (16 KiB pages): the guest range
was free to reserve in the arm64 process, the pool mapped at the same addresses in both, aliases
and remaps stayed coherent, and a write fault went over the channel and back with a nested Protect
call. Note for the runtime: `shm_open` sets close-on-exec, so the pool's descriptor must have it
cleared (or be passed explicitly) before bb-gpu is spawned.

**Phase 4 (remote shim), first version (2026-10-08).** `BB_NATIVE_GPU=1` runs the game this way:

- bb-probe creates the guest memory pool with a host span after flexible memory
  (`runtime_memory_host_pool`), maps the control block (`remote/bb_control.h`: the channel, the
  VideoOut buffer labels, flags bb-gpu publishes) at `0xfc00000000`, just above the guest range, in
  both processes, and starts `out/gpu-arm64/bb-gpu` with the pool's descriptor. bb-gpu's
  environment gets the arm64 libraries and KosmicKrisp (`deps-arm64`, or `BB_GPU_DEPS`;
  `BB_GPU_PROCESS` overrides the executable).
- Memory: the runtime reports every change to its mapping table (`runtime_memory_set_mirror_hook`;
  queued under its lock, sent in order after it by the thread that made them, before the call
  returns to the guest). bb-gpu maps
  the same pool pages at the same addresses and keeps a copy of the table (`remote/bb_vma_table.h`),
  from which it answers the GPU library's `runtime_memory_*` queries itself
  (`gpu/bb_gpu/bb_gpu_main.cpp`). `tests/test_remote_table.cpp` checks that copy against
  `runtime_memory.c` over random maps, unmaps, protections, releases and reservations. The game's
  data segments move into the pool's host span at load (`bbgpu_share_range`, probe.c): the GPU
  reads resource tables there.
- Page tracking: a write fault in the game process goes to bb-gpu (`MsgWriteFault`). Every page
  protection bb-gpu decides is a call on a second channel (`ControlBlock::protect`) that a game
  process thread of its own applies, in the order they were decided; that thread takes nothing but
  the runtime's lock. A fault from a thread that holds that lock itself (inside the runtime) gets
  its protections back in the reply instead and applies them. Invalidations, file-read notes and
  decoded movie frames are messages.
- GnmDriver stays in the game process; submissions, `SubmitDone`, the idle check and compute queues
  go through a small facade in `gnmdriver.cpp` (host-memory init sequences travel inline). Frames
  retired, the submission lock and every interrupt come back (`IrqController::forward`).
- VideoOut: bb-gpu runs the driver and presents; the game process's driver keeps a copy of the port
  (buffer slots and flip status follow bb-gpu's, with a version so a late copy never replaces a
  newer one) and triggers the guest's flip and vblank events when bb-gpu reports them. EOP flips are
  registered in bb-gpu, whose command processor raises the flip interrupt in stream order.
- Gamepads stay in the game process (`runtime_pad.c`), whose main thread now keeps SDL's gamepad
  events going; the window, the settings menu and keyboard shortcuts are bb-gpu's. The game's
  keyboard is bb-gpu's window's: its window loop copies SDL's key state into the control block,
  and `bbgpu_keyboard_state` reads it there.
- Lifetime: bb-gpu ends when the game process exits or restarts (kqueue on its parent); the game
  process ends when bb-gpu does (its window closed, or a crash, reported with the signal).

Not yet: the game threads' wait statistics in the frame stats (bb-gpu
prints its own threads), ThreadSanitizer runs and the release/acquire audit (phase 6). The first
native run compiles every shader again: the arm64 build keeps its own pipeline cache and KosmicKrisp
its own shader cache.

Tests, on Linux and on the Mac:

```
cc -c -std=c11 -D_GNU_SOURCE -w -Isrc -I. src/runtime_memory.c -o /tmp/rm.o
c++ -std=c++20 -O1 -Igpu/shim tests/test_remote_table.cpp /tmp/rm.o -lpthread -o /tmp/table-test && /tmp/table-test
```
