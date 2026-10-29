# Native Apple Silicon GPU process (macOS): design

Status: the default on the native branches (`macos-native-0.6`, 2026-10-21) wherever bb-gpu is
built (`tools/macos/build_gpu_native.sh`, which needs `deps-arm64`): the GPU side runs in a native
arm64 process. `BB_NATIVE_GPU=0`, or the app's "In-process GPU (Rosetta)", keeps the in-process
x86-64 path; a checkout without bb-gpu uses that path too. The release app ships bb-gpu
(`Contents/Resources/bbport/bin/gpu-arm64`) with its own arm64 libraries and driver
(`deps-arm64/lib` there, `tools/macos/package.sh`). Known gap: the in-game menu pointer. MetalFX works here and not in-process: under Rosetta
its Neural Engine part fails to set up and its work hangs for about 4 s at a time until macOS
stops it.

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
 │ game code, runtime (src/*.c) │  guest       │ command processor, rasterizer,   │
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
| SubmitGfx / SubmitAsc / SubmitDone / MapComputeQueue | `BbCp::CommandProcessor` from Gnm (`gpu/platform/gnm`) | asynchronous; the command buffers are read from shared memory |
| VideoOut Open/Close/RegisterBuffers/SubmitFlip/ChangeBufferAttribute | `BbVideoOut::Display` | the port state becomes a shared struct |
| Write fault (address) | `bbgpu_handle_fault` | synchronous: bb-gpu marks the page and asks to unprotect |

Backend → frontend:

| Message | Today | Notes |
|---|---|---|
| Protect (range, read, write) | `runtime_memory_gpu_protect` | synchronous; the frontend applies it (Rosetta emulates 4 KiB protection, only it can): its protection thread, or during a write fault the faulting thread |
| Write trap (range, reason, on) | `runtime_memory_trap` (bbport 0.5) | as Protect (a `ProtectArgs` with `ProtectTrap`); the game process keeps the reasons that count, bb-gpu a copy for `runtime_memory_trap_reasons`, cleared on Map/Unmap |
| Interrupt (EOP, compute, flip, vblank) | `BbIrq::Raise` → event queues | asynchronous, triggers guest events |
| Pad and keyboard state | `runtime_pad.c` reads SDL | shared struct; the window and its events belong to bb-gpu |
| Button prompts (bbport 0.6) | `runtime_pad_prompt*`, `runtime_file_host_path` | the prompt state is a shared struct the game process updates once a frame; a prompt's label and the game's icon file are calls (rare: when the prompts change), answered by the game process's reader thread |

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
7. **Default** (done, 2026-10-21): as fast as in-process with FSR 3.1 after the mapping-table fix,
   and the only path where MetalFX runs; `BB_NATIVE_GPU=0` keeps the in-process path.

## Status (2026-10-08)

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
  the runtime's lock. The protections bb-gpu's handlers decide for a write fault are asked of the
  faulting thread instead (protocol 11: `Channel::Reply::Ask`, over the call's reply slot), which
  applies each while it waits, before the handler goes on, as its own handler does in one process,
  with the runtime's lock held there or not: bb-gpu makes no protection-thread call for a fault's
  own protections. (Protocol 10 sent them back in the reply when the thread held that lock:
  applied after it, they could land after newer ones the protection thread applied meanwhile, and
  those past the reply's 160 were calls that thread could not serve while the faulting thread held
  the lock.) A fault's handlers can still wait for other bb-gpu threads (owner locks), and those
  for the protection thread, as in one process, where they wait for the runtime's lock directly.
  Faults from a thread that holds the runtime's lock come on a channel of their own
  (protocol 12: `ControlBlock::locked`, bb-gpu's `FaultThread`): on the main reader such a fault
  could wait behind another one whose faulting thread is applying a protection and so waits for
  that lock, a cycle one process does not have (each fault there runs on its own thread). What is
  left is one process's own: a handler that needs an owner lock (`layer_trap_mutex`) a GPU thread
  holds while its protection waits for the lock-holding thread. Invalidations, file-read notes
  and decoded movie frames are messages.
- Gnm stays in the game process; submissions, `SubmitDone`, the idle check and compute queues
  go through a small facade in `gnm_driver.cpp` (`Gpu::`; host-memory init sequences travel
  inline). Frames retired, the submission lock and every interrupt come back (`BbIrq::forward`).
- VideoOut: bb-gpu runs the display and presents; the game process's display keeps a copy of the port
  (buffer slots and flip status follow bb-gpu's, with a version so a late copy never replaces a
  newer one) and triggers the guest's flip and vblank events when bb-gpu reports them. EOP flips are
  registered in bb-gpu, whose command processor raises the flip interrupt in stream order.
- Gamepads stay in the game process (`runtime_pad.c`), whose main thread now keeps SDL's gamepad
  events going; the window, the settings menu and keyboard shortcuts are bb-gpu's. The game's
  keyboard is bb-gpu's window's: its window loop copies SDL's key state into the control block,
  and `bbgpu_keyboard_state` reads it there.
- Lifetime: bb-gpu ends when the game process exits or restarts (kqueue on its parent); the game
  process ends when bb-gpu does (its window closed, or a crash, reported with the signal).

- Frame stats: bb-gpu prints them; after each block it asks the game process
  (`MsgGameStats`) for the parts that live there, which it prints itself: its threads' CPU time
  (`Game process: CPU by thread ...`; bb-gpu's line is `GPU process: CPU by thread ...`), the
  guest's waits, event-queue waits and the Gnm call counts.
- Diagnostics switches: bb-gpu reads `BB_TOGGLES` at start and watches `BB_TOGGLE_FILE` like the
  game process, so live A/B changes reach the GPU code in both.

The first native run compiles every shader again: the arm64 build keeps its own pipeline cache and
KosmicKrisp its own shader cache.

**bbport 0.6 (2026-10-15).** Upstream rewrote the GPU side (`gpu/shadps4` became `gpu/cp`,
`gpu/render`, `gpu/platform/...`); the split was carried over onto the new code:

- `gpu/platform/gnm/gnm_driver.cpp`: the `Gpu::` facade (submissions, idle check, compute queues);
  `RegisterLib` creates the command processor and presenter only in-process or in bb-gpu.
- `gpu/platform/videoout/{display,videoout}.cpp`: the forwarding and the port's copy as before;
  `Port::buffer_labels` points into the control block, and every store to a label is a release
  store (`StoreLabel`), the new early buffer release (`BB_FLIP_AHEAD`) included. bb-gpu's message
  handlers call the library through `BbVideoOut::Remote`.
- `gpu/platform/interrupts.cpp`: `BbIrq::forward`, set in bb-gpu.
- The SRT walker (`gpu/shader/ir/passes/flatten_extended_userdata_pass.cpp`): the x86 builder's
  walk again as bytecode on arm64 (same traversal, same slots), run by `RunSrtWalker`; the cache
  validates bytecode it loads. The 0.5 check against the x86 walker (`BB_SRT_CHECK`) did not come
  along.
- 0.6's button prompts read the pad and the game's files from bb-gpu (table above); the pad's
  mouse capture request is a counter in the control block, the Toy Pad's lights (settings menu)
  three words there.
- bb-gpu checks what it is sent before using it (inline submission lengths, compute queue rings
  and read pointers in guest memory, queue numbers, display buffer counts); the bytecode walker
  reads guest memory only, and a mapping removed under a walk ends that walk (no crash).
- CI builds bb-gpu on the native branches (the arm64 libraries are cached like the x86-64 ones).

**bbport 0.6 final (2026-10-20).** What 0.6 added across the split:

- Effects switched live: the settings menu (bb-gpu) sends its effects (`MsgGameEffects`) when
  they change; the game process switches the game's code (`BbGuestHooks::ApplyGameEffects`).
  The game's own System menu runs in the game process as before (`MsgSettingsChanged`).
- The live frame rate limit (F3, the settings menu): `MsgFrameLimit` to the game process, whose
  frame pacer's hooks apply it.
- Short key taps (`runtime_pad.c` key_latch): bb-gpu's window latches key presses in the control
  block (`SharedState::key_latch`); the game process's pad takes them.
- The FX initialization clear (the save load crash fix) submits its commands inline with
  `force_copy`: bb-gpu's command processor copies them at submit, as in one process.
- The game's main thread for those hooks is the loader's thread (`BbThreads::IsGuestMainThread`),
  recorded by the game process's `bbgpu_init`.
- The render resource cache's extra direct memory is mapped by the game process and mirrored as
  any mapping; bb-gpu has a stub for `runtime_extra_direct_map`.
- 0.6 installs the hooks in the game's code with either memory model, from the buffer cache's
  constructor, which runs in bb-gpu here: the game process installs them in `bbgpu_init`
  instead, and bb-gpu skips them. The output's aspect ratio for the camera and HUD hooks comes
  from bb-gpu's upscaler (`MsgDisplayAspect`).
- Lookups in bb-gpu's copy of the mapping table (`VmaTable::Covered`, `Region`) remember each
  thread's last mapping for one generation of the table. The bytecode SRT walker checks every
  guest read there; with four draw-preparation threads and the command thread, a third of their
  time went to waiting on the table's lock (a `sample` of bb-gpu, 2026-10-21: 35-42 FPS where the
  game process with the GPU library in it holds 60). `tests/test_remote_table_cache.cpp`.
- Not split yet: the mouse as a pointer in the game's menus (`BbMenuPointer`: its game hook
  belongs here, its window part in bb-gpu). With the native GPU process the keyboard and
  gamepad drive the game's menus.

**bbport 0.7 (2026-10-29).** 0.7 has one memory model: the GPU uses the game's memory in place.
How it crosses the split:

- `BbGuestMemory::Install` runs in bb-gpu (the rasterizer is there). Its
  `runtime_memory_set_guest_chunk_*` calls land in bb-gpu (`bb_gpu_main.cpp`), which tells the
  game process (`MsgGuestMemory`) after `MsgInit`. The game process's runtime then gets a
  chunk allocator that calls bb-gpu (`MsgChunkAlloc`, `MsgChunkFree` on `ControlBlock::protect`,
  game process -> bb-gpu, served by a bb-gpu thread that never waits for the game process: the
  runtime may hold its lock). bb-gpu's GPU imports its own view of the whole pool
  (`runtime_memory_backing_pointer`) and answers `CHUNK_IMPORTED`, so the game's mappings stay
  the pool's and the mirror stays right. Dma-buf chunks and private regions (memory one
  process alone has) are refused at the start.
- Lazy imports (`BB_GUEST_LAZY=1`): `runtime_memory_note_pinned` is a call to the game process
  (`MsgNotePinned`), the pool descriptor bb-gpu's own.
- Heaps growing on demand: off with the native GPU process for now (`kernel_api.c`: no mirror
  hook for a commit; the log says "Heap grow: off"), so the GraphicsPrivate heap is imported
  whole at the start. `MsgGpuRemap` is there for it, but nothing in the runtime queues a remap
  yet: enabling growth needs a new `MIRROR_REMAP` (3: `remote_front.cpp` handles a case 3) queued next to the commit and decommit of an
  imported chunk (`heap_commit.c`). The guest hooks' fresh, dead and live ranges go to bb-gpu's
  buffer cache (`MsgFreshRange`, `MsgDeadRange`, `MsgLiveRange`; without growth, dead and live
  do not fire). `MsgFreshRange` is a send by default: in one process the GPU's old data there
  stops being GPU-modified before the game writes the new block; across the split a readback in
  bb-gpu could still land after those writes (about 175 a second). `BB_REMOTE_FRESH_SYNC=1`
  makes it a call on a channel of its own (`ControlBlock::fresh`) that returns once bb-gpu's
  `FreshThread` has run `NoteFreshRange`, after the main reader has handled what the game process
  sent before it (protocol 12: `FreshArgs::after`, `Channel::WaitHandled`: its maps, remaps, dead
  and live ranges). That thread waits for the main reader and for `NoteFreshRange`'s locks; the
  main reader never waits for it or for the hook's thread (game code), nor do those locks'
  holders. Off until measured.
- The BDA page table: on macOS its window has two halves (`gpu/shader/profile.h`), the low region
  and 64 GiB of the guest's user mappings from 0x7000000000: bb-gpu's own 24 GiB, placed by ASLR
  before `main`, can sit in that range, and the game's mappings go around it.
- Write faults carry the game's trap reasons for the page; reasons bb-gpu holds none of (a trap
  that crossed a remap) are lifted, and the fault runs through the same handlers as in one process
  (`Core::Signals`: the layer's VRAM copy traps, then page tracking). A fault only stale reasons
  held is retried. Protocol 12: every trap change gets a sequence number under bb-gpu's copy's
  lock (`remote/bb_trap_copy.h`), the game records the newest it applied per page
  (`runtime_memory_trap_seq`), and a lift carries the number current when it was checked: the
  game drops it on a page a newer change reached, so an owner's trap set meanwhile stays,
  whichever arrives first. (Protocol 11 held that lock across the faulting thread's apply, which
  can wait for the runtime's lock.) The lock is never held across a wait.
- The pad recording clock (`bbgpu_world_*`) and the Steam pad flag are in `SharedState`; the
  camera speeds and light grid setting go to the game process (`MsgGameSettings`), trophy
  pop-ups to bb-gpu (`MsgShowTrophy`).

**Phase 6, started (2026-10-08).**

- ThreadSanitizer (GCC 13, Linux) on the channel, memory and table tests and on
  `tests/test_remote_concurrent.cpp`, where four guest threads map, unmap and protect overlapping
  memory at once and the GPU process's table must then answer as the runtime does: no reports, and
  the table matches every time (six runs).
- Ordering at the process boundary, reviewed: the channel publishes a message with a release store
  of `head` after its payload and the reader takes it with an acquire load; the wait/wake handshake
  is seq_cst on both words (no lost wake-up); reply slots are published with seq_cst stores
  too (protocol 11; a release store could pass the sleepers check on x86), and the waiter's
  check after counting itself in follows a seq_cst fence; EOP and EOS labels
  are single release stores (`StoreLabel`). The VideoOut buffer label resets were plain stores and
  are now release stores (the guest polls them from the other process).
- `BB_GPU_QOS=1` (experiment, both branches): the command processor, recording, draw preparation,
  present and remote-channel threads ask for the user-interactive QoS class.

Still to do: a side-by-side comparison with the in-process path in the same spots, and the GPU
waits on the Great Bridge (`BB_WAIT_TRACE=1` shows where they come from).

Tests, on Linux and on the Mac:

```
cc -c -std=c11 -D_GNU_SOURCE -w -Isrc -I. src/runtime_memory.c -o /tmp/rm.o
c++ -std=c++20 -O1 -Igpu/shim tests/test_remote_table.cpp /tmp/rm.o -lpthread -o /tmp/table-test && /tmp/table-test
c++ -std=c++20 -O1 -Igpu/shim tests/test_remote_concurrent.cpp /tmp/rm.o -lpthread -o /tmp/concurrent-test && /tmp/concurrent-test
c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_protect_order.cpp /tmp/rm.o -o /tmp/order-test && /tmp/order-test
```

With ThreadSanitizer (Linux, GCC; `setarch -R` where ASLR and TSan disagree):

```
gcc -c -std=c11 -D_GNU_SOURCE -w -O1 -g -fsanitize=thread -Isrc -I. src/runtime_memory.c -o /tmp/rm-tsan.o
g++ -std=c++20 -O1 -g -fsanitize=thread -Igpu/shim tests/test_remote_concurrent.cpp /tmp/rm-tsan.o -lpthread -o /tmp/concurrent-tsan && setarch -R /tmp/concurrent-tsan
g++ -std=c++20 -O1 -g -fsanitize=thread -pthread -Igpu/shim tests/test_remote_channel.cpp -o /tmp/channel-tsan && setarch -R /tmp/channel-tsan
g++ -std=c++20 -O1 -g -fsanitize=thread -pthread -Igpu/shim tests/test_remote_protect_order.cpp /tmp/rm-tsan.o -o /tmp/order-tsan && setarch -R /tmp/order-tsan 2
```

`tests/test_remote_protect_order.cpp` models the write traps' owners, handlers, the protection
thread and faulting threads in two processes over the real channel, bb-gpu's real trap copy and
the game's real trap table: no trap may end armed in bb-gpu and missing in the game. Its
`collect` and `unlocked-lift` arguments run protocol 10's ways and count the traps they lose
(some per run with `collect`, about one in 10-20 rounds with `unlocked-lift`). It then has a
lock-holding thread fault while another fault's protection waits for that lock: both must
finish; `one-reader` serves both on one reader, as protocol 11 did, and reports the deadlock. Under
ThreadSanitizer it also shows `runtime_memory.c`'s own trap table race (upstream's: a plain byte
store under `trap_mutex`, read without it by `runtime_memory_trap_reasons`), which only this test
exercises; nothing else is reported.
