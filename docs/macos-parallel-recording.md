# Parallel Vulkan recording on macOS ("Option 2")

Design only, 2026-10-05, based on `macos-port` b46134b. `rv/` is
`gpu/shadps4/video_core/renderer_vulkan/`; "KK" is Mesa `src/kosmickrisp/` at dc41592a, the
revision the port builds.

## Status (2026-10-05)

Phases 1 and 2 are implemented on `macos-mtrec`, untested on a Mac.

| Setting | Default | Effect |
|---|---|---|
| `BB_COPIES_OFF_RECORDER=0/1` | 1 on macOS, 0 on Linux | Phase 1: small guest copies batched for the copy threads (`Scheduler::QueueHostCopy`), EOP fences through `BbCopy::AfterCopies` directly |
| `BB_VK_RECORD_WORKERS=N` (1–8) | 2 on macOS with KosmicKrisp, else 1 | Phase 2: recording workers (`bb:VkRec0`…) |
| `BB_VK_SEGMENTS=0/1` | 1 when N > 1 | Segments; `BB_VK_RECORD_WORKERS=1 BB_VK_SEGMENTS=1` is the cut-only test mode |
| `BB_VK_SEGMENT_UNITS` | 250 | Cut threshold (a draw or dispatch is 1 unit, a render pass 8) |
| Toggle bit 1<<59 `RecorderHostCopies` | clear | Set: copies and fences through the recorder again, and no segments |
| Toggle bit 1<<60 `ParallelRecording` | clear | Set: one recorder, one command buffer per submission |

Both toggle bits and the modes take effect at the next submission (`LatchRecordingMode`), when
nothing of the stream is pending. Segments require copies off the recorder. `BB_FRAME_STATS=1`
prints a `Recording:` line every 5 s with the mode, copies, batches and fences via the copy
threads, busy % of the single recorder and of each worker, segments, cuts and skipped cuts per
frame, the cost of beginning and ending a segment, and the submit wait. `CutPoint` is called in
`DrawRecord`, `DrawIndirectRecord`, `DispatchRecord` and `DispatchIndirectRecord`; debug label
depth comes from `Rasterizer::ScopeMarkerBegin/End`.

## 1. Summary

- The single recorder `bb:VkRecorder` is saturated (93% busy). Removing waits elsewhere recovers
  at most its idle share, about 7%. 60 FPS needs its 19–31 ms of driver work per frame spread
  across threads.
- **Recommendation:** N recorder workers, each recording a contiguous segment of the existing
  command stream into its own primary `VkCommandBuffer`; all buffers go out in order in the
  existing single `vkQueueSubmit`. Cuts happen only where a draw starts a new render pass anyway,
  or at a dispatch, so KK creates no extra Metal encoders. The only redundant work is 15–25
  dynamic-state setters per segment.
- KK permits this: each `VkCommandBuffer` owns its Metal 4 allocators, uploader and state.
  Secondary command buffers do not help: KK replays them serially on the primary's thread.
- **Prerequisite (Phase 1):** take guest "host copies" and EOP fence signals out of the recorded
  stream. They are the only order-dependent side effects in recorded closures, and the reason
  DrawRec waits for the recorder today.
- **Expected:** in 40 FPS areas, 2 workers ≈ 50–60 FPS; with 3, DrawRec becomes the limit at
  about 60. This assumes Apple's Metal driver encodes in parallel under Rosetta: unverified, the
  main risk, and the first thing Phase 2 measures.
- macOS/KK only by default; Linux keeps one recorder and today's code path.

## 2. Current architecture

| Thread | Role |
|---|---|
| Stage A, GPU command thread (`Liverpool::Process`) | PM4 decode, pipeline selection. Drains stage B before non-pipelined packets (`PipelinedOpcode`, liverpool.cpp:194). `Rasterizer::Flush` once per guest frame on `sceGnmSubmitDone` (liverpool.cpp:162). Flips (`Presenter::PrepareFrame` via `SendCommand`). |
| Stage B, `bb:DrawRec` (`DrawPipe::Run`, vk_draw_pipe.h:159) | `Rasterizer::RunDrawPacket` (rv/vk_rasterizer.cpp:693): `DrawRecord` (:942), `DispatchRecord` (:1278) and ordered tasks (`RunInOrder`, :679: EOP, EOS, WriteData, DmaData, flip IRQ). Main caller of `Scheduler::Record`. |
| `bb:VkRecorder` (`Scheduler::RecorderThread`, rv/vk_scheduler.cpp:331) | Executes `RecordChunk`s into `current_cmdbuf`. |
| Copy threads (`BbCopy`) | Large guest-to-staging copies. |

Draw-preparation workers (vk_draw_prep.h) never record. `draw_scheduler` (vk_presenter.cpp:130)
is the only scheduler with a recorder thread.

**Recording.**
- `Record(func)` (vk_scheduler.h:733) placement-news a closure into a 128 KiB `RecordChunk`
  (h:582); `RecordData`/`ReserveRecordData` (h:760–784) copy variable data into the same chunk.
- `KickRecording` (cpp:286), after each draw or dispatch, queues chunks once they hold 32 KiB
  (or when forced). The recorder spins 200 µs before sleeping.
- **Direct mode:** `CommandBuffer()` (h:717) calls `SyncRecording()` (cpp:321: queue empty and
  recorder idle); the caller then records on its own thread until the next `KickRecording`.
  Callers: flip passes, `BlitHelper` (draws inside a pass opened through `Record`),
  `FaultManager`, the upscaler, debug markers.
- `BeginRendering`/`EndRendering` (cpp:100/179) track `is_rendering`/`render_state` on the
  producer; `BeginRendering` returns early when the same state is open.

**Submission.** `SubmitExecution` (cpp:448): `NextTick`; `on_submit` (barrier flush, sparse
binds); `EndRendering`; `SyncRecording`; `WaitHostCopies`; `end()`; one `vkQueueSubmit` of one
buffer signalling the timeline `work_semaphore` at the tick; then `AllocateWorkerCommandBuffers`
(cpp:427) takes the next one-time-submit buffer from `CommandPool` (vk_resource_pool.cpp, reused
once its tick has passed) and invalidates dynamic state. About one submission per frame, plus the
flip and synchronous downloads (`DownloadImageMemory(sync)` calls `Finish`).

**Host copies and fences.**
- `BufferCache::SmallGuestCopy` (buffer_cache.cpp:119) puts small guest-to-staging copies into
  the stream with `RecordHostCopy` (h:802): a closure that copies, then publishes
  `host_copies_done = seq`.
- `WaitHostCopies` (cpp:273) spins until `done ≥ host_copies_issued` (the recorder has executed
  everything up to the last copy), then waits for the copy threads.
- EOP on stage B uses `SignalAfterHostCopies` (cpp:241): a stream closure hands the signal to
  `BbCopy::AfterCopies`. `WaitDeferredSignals` (cpp:258) waits for all such signals.
- `RunEventWriteEos`/`RunWriteData` (liverpool.cpp:330/317) call both waits on stage B. This is
  where DrawRec waits for the recorder's backlog; stage A then waits for DrawRec in
  `DrawPipe::Drain` (vk_draw_pipe.h:105).

## 3. State dependencies and cut points

**Re-recorded for every draw:** the pipeline (`bindPipeline` is inside the draw closure,
vk_rasterizer.cpp:1114), vertex and index buffers (`EmitVertexBuffers` :1655, `EmitIndexBuffer`
:1829), push constants and push descriptors (`Pipeline::BindResources`,
vk_pipeline_common.cpp:26). Dispatches likewise (:1311–1321).

**Carried on the producer:** dynamic state (`DynamicState` dirty flags, h:106; `CommitWith`,
h:185, emits only changed values) and the open render pass.

**Independent of the command buffer:**
- Barriers and image layouts are tracked on the producer and recorded as plain barriers. Their
  scopes follow submission order, which spans all buffers of one submit.
- No Vulkan queries: guest occlusion `EventWrite` is CPU-side; profiler timestamps are
  self-contained.
- Debug labels are off by default; a cut must not split an open label region (track depth).

**Cut rule.** Cuts happen only at an explicit `Scheduler::CutPoint()`, never implicitly, so
helpers, the presenter and direct mode are unaffected. The rasterizer calls it:
- in `DrawRecord`/`DrawIndirectRecord` right after `state = BeginRendering(pipeline)`
  (:988, :1195), only if `scheduler.WillBeginRendering(state)` (the draw ends the pass anyway).
  No vertex, index, push or dynamic-state command of that draw has been recorded yet;
- in `DispatchRecord`/`DispatchIndirectRecord` after `scheduler.EndRendering()` (:1311, :1365),
  before `BindResources`.

It cuts only outside direct mode, with no label open, once the segment holds enough work. It then
calls `EndRendering()` (recording `endRendering` in the old segment, exactly where
`scheduler.BeginRendering` would have), closes the old segment, opens the next one on the next
worker, and calls `dynamic_state.Invalidate()`. Work recorded before the cut point stays in the
old segment: `BindResources` uploads and barriers, `upscaler->OnDraw`, `ObjectMotion::Attach`
(in `BeginRenderingFull`); `ObjectMotion::PrepareDraw`, after it, records nothing. The rest of
the window up to `scheduler.BeginRendering` must be audited for non-state commands when
implementing.

**Cost of re-establishing state** (estimates, to be measured):
- Dynamic state: after `Invalidate`, `CommitWith` emits 15–25 setters (stencil and depth bias
  only when enabled). On KK these only store CPU state, and KK re-flushes all state into every
  new render encoder anyway (`kk_cmd_buffer_dirty_all_gfx` in `cs_start_render`,
  kk_cmd_buffer.c:279): about 25 µs per segment at most.
- Fixed per segment: `vkBeginCommandBuffer` (KK resets three MTL4 command allocators,
  kk_cmd_buffer.c:186), `vkEndCommandBuffer`, one extra Metal commit at submit. Estimate
  50–150 µs; at 12 segments per frame, 0.6–2 ms spread over the workers, i.e. 3–8% of the
  recorder work.
- Mid-pass cuts (end, then reopen with `LOAD`) cost an extra encoder plus a GPU tile store and
  reload of every attachment; avoided.

**Segment size.** Weight a pass as about 8 draws (Begin/End rendering: 33% over 185 passes;
draws: 32% over 1,400). A frame is about 2,900 units; cutting at about 250 (about 2 ms) gives
about 12 segments per frame. Tunable: `BB_VK_SEGMENT_UNITS`.

## 4. What KosmicKrisp does

**VkCommandBuffer to Metal.** Each render pass gets a new `MTL4CommandBuffer` from the device,
begun on the buffer's graphics `MTL4CommandAllocator`, with one render encoder (`cs_start_render`,
:279). Copies and dispatches go to compute encoders, each in its own Metal command buffer
(`pre_gfx`/`post_gfx`, `cs_get_compute` :347). Every encoder ends with
`barrierAfterStages:ALL beforeQueueStages:ALL` (the sync-compute barrier, `kk_stop_encoder` :367)
and is appended to `submit_cmd_bufs` in encode order (`cs_end` :390). Hence BeginRendering 21%
and EndRendering 12%.

**Thread safety.**
- Each `kk_cmd_buffer` owns its three allocators, argument table, upload buffers
  (`kk_pool_alloc` :709) and descriptor/push/dynamic state.
- The `kk_cmd_pool` free list of upload buffers is unlocked (kk_cmd_pool.c): one `VkCommandPool`
  per worker, as Vulkan requires anyway.
- Shared device state during recording: the residency set has a mutex (kk_device.c:428–469) and
  is touched only when a new 128 KiB upload buffer is allocated; the sampler heap has a mutex and
  changes only at `vkCreateSampler`; the geometry heap (`kk_heap`, kk_cmd_draw.c:1051) is
  initialised once, then each command buffer resets its bump pointer on the GPU timeline, safe
  because the queue barriers serialise all Metal buffers (each guest submission already uses its
  own `VkCommandBuffer`). Dynamic depth-stencil states come from `newDepthStencilState`, a device
  call.
- Apple, `beginCommandBuffer(allocator:)`: "Command allocators only service a single command
  buffer at a time. If you need to issue multiple calls to this method simultaneously, for
  example, in a multi-threaded command encoding scenario, create multiple instances … and use one
  for each call." KK's per-buffer allocators satisfy this. Whether the AGX user-space driver
  locks internally (allocation, residency) under Rosetta is unknown.

**Secondary command buffers.** KK's trampolines (util/kk_dispatch_cmd_gen.py) only enqueue
secondary commands into a `vk_cmd_queue`; `vkCmdExecuteCommands` is Mesa's
`vk_common_CmdExecuteCommands` (src/vulkan/runtime/vk_command_buffer.c:157), which replays them
on the primary on the calling thread. A primary begun without `ONE_TIME_SUBMIT` also copies every
command into that queue, so worker buffers must keep `eOneTimeSubmit`.

**Submit.** `kk_queue_submit` (kk_queue.c:259): one `waitForEvent` per wait semaphore; per
`VkCommandBuffer` one `commit:count:options:` with a new `MTL4CommitOptions` and feedback block
(`kk_queue_commit` :95); one `signalEvent` per signal semaphore (timelines are
`MTLSharedEvent`s). N buffers cost N commits and no extra event operations. The GPU runs commits
in order, and the per-encoder queue barriers order work across commits exactly as within one.

## 5. Ordering and synchronisation with N recorders

- **Submission.** Each submission has an ordered segment list; workers end their own segments.
  `SubmitExecution` cuts the last segment, waits for all workers and submits all buffers in one
  `vkQueueSubmit` with the same waits, signals and fence. One tick per submission, signalled after
  the last buffer: `DeferOperation`, `Wait`, readbacks (`Finish`) and presenter flushes keep their
  meaning.
- **Allocation.** The worker begins its segment's buffer from its own `CommandPool` when the
  segment's first chunk arrives (the pool needs external synchronisation, so the producer must not
  touch it). A tick read during `SubmitExecution` may stamp one tick late; that only delays reuse.
- **Direct mode.** `CommandBuffer()` waits only for the current segment's worker and returns that
  segment's buffer (beginning it if needed); other workers keep running. `BlitHelper` keeps
  drawing inside the open pass.
- **Host copies.** With several workers the `host_copies_done = seq` stores would arrive out of
  order, and copies would run whenever their segment's worker got to them. Phase 1 removes them
  from the stream. Small copies go to `BbCopy::QueueCopy` (the existing path behind toggle bit
  524288). `SignalAfterHostCopies` records nothing: it calls `FlushBatch`, increments `issued`,
  then `AfterCopies(signal; done++)`. `AfterCopies` runs callbacks in registration order, so
  fences stay ordered; `WaitDeferredSignals` before GPU idle (liverpool.cpp:179) is unchanged.
  EOS and WriteData then wait only for the copy threads, and every recorded closure is a pure
  Vulkan call that any worker may run.
- **Why this is safe.** The EOP signal is already early today: it fires once earlier copies are
  done, not when the GPU finishes. The recorder reaching that point adds nothing the guest can
  observe, assuming the GPU reads guest data only through copies, which the current behaviour
  already relies on. When the mode is switched at run time, first drain in-stream copies with
  the existing counter.

## 6. Alternatives

At 40 FPS (25 ms): recorder 23.3 ms of work; DrawRec and stage A about 10 ms each (41% busy). At
30 FPS: recorder 31 ms, DrawRec 13.7 ms. GPU time per frame is assumed below 16.7 ms (unchecked).

| Option | Ceiling (Amdahl) | Realistic | Effort | Risks |
|---|---|---|---|---|
| (d) Decouple waits (Phase 1) | Recorder-bound: 25 → 23.3 ms, ≤ +7% | +3–7% | Small, ~100 lines | Fence order (heap corruption, 2026-10-01); a wakeup per copy batch |
| (c) Asynchronous submit / cross-frame pipelining | Same ≤ +7%, not additive with (d) | +0–5% alone; after (a) hides the submission tail | Medium | Waits on unsubmitted ticks; presenter semaphores |
| (b1) Copies in their own buffer | Copies 9%: ≤ +10% | Needs copies hoisted above earlier draws: not order-safe | Medium | Correctness |
| (b2) Secondary buffer per pass | 0 on KK (serial replay) | Negative | — | — |
| (a) N workers, contiguous segments | N=2: 23.3×1.06/2 ≈ 12.4 ms; with DrawRec 10 ms plus tails ≈ 13–15 ms (65–75 FPS). N=3: DrawRec-bound | 40 FPS areas: N=2 ≈ 50–60, N=3 ≈ 60. 30 FPS areas: N=2 ≈ 45–55, N=3 ≈ 55–60 | Large: 500–700 lines in the scheduler, 4 hooks | AGX/Rosetta may serialise encoding; direct-mode corners |
| (e) KK patch: one MTL4 buffer per `VkCommandBuffer`, lighter end barrier | Targets the 33% in Begin/End | Unknown; multiplies with (a) | Medium, out of tree | KK uses separate buffers to order compute around render encoders |

**Interaction with `macos-passes`:** fewer passes shrink the parallelisable work by a few percent;
about 150 natural breaks per frame still allow 12 segments. Its `on_rendering_end` callback (the
held uploads) runs inside `EndRendering`, so `CutPoint` records them in the old segment, as
intended. No conflict; merge order does not matter.

## 7. Plan

**Phase 0, measure (no code):** `BB_GPU_PROFILE=1` for GPU ms per frame (60 FPS needs under
16.7 ms); `BB_FRAME_STATS=1` for submissions per frame; A/B toggle bit 524288 with `tools/ab.sh`
(moves only the copies: part of (d)).

**Phase 1, decouple (d).** Runtime toggle bit `RecorderHostCopies` (set: today's behaviour);
copies off the recorder by default on macOS, on it on Linux. Small copies go to the copy threads
and fences go through `AfterCopies` (§5). Success: `host_copies_wait_ns` and the DrawRec wait
share fall, FPS rises a few percent, and a 20-minute soak passes (the heap-corruption check).

**Phase 2, smallest proof of (a).** `BB_VK_RECORD_WORKERS=N`, default 1 everywhere at first;
ignored with MoltenVK, which encodes Metal at submit time.
- `Worker { jthread, CommandPool, chunk queue, cv, busy }`, `Segment { cmdbuf, worker, done }`;
  `RecordChunk` gets a segment id; one shared, mutex-protected free-chunk list.
- `KickRecording` routes chunks to the current segment's worker; `CutPoint` at the 4 hooks (§3).
  `SyncRecording` waits for all workers, `CommandBuffer()` for one (§5); `SubmitExecution`
  submits the segment list.
- Workers get the QoS of `bb:VkRecorder`, not the helpers' `UTILITY`; workers after the first spin
  less.
- With N>1, assert that the stream holds no host copies or signals, and `!is_rendering`,
  `!direct_mode` and label depth 0 at every cut.
- Runtime toggle bit `ParallelRecording` switches between N and 1 at the next submission, so
  `tools/ab.sh` works in one run.

Measure N=2 first. Proof: the two recorders' busy shares add up to more than 100% and FPS rises.

**Phase 3, extend:** tune N and `BB_VK_SEGMENT_UNITS`; asynchronous submit (c), where a submit
stage issues `vkQueueSubmit` once a submission's segments are recorded, so stage A does not wait
for the slowest worker; N=2 or 3 by default on macOS with KK; then consider (e).

**Instrumentation** (a `Recorders:` line in `BB_FRAME_STATS`): per worker busy %, spin % and
segments; mean segment µs and begin+end µs; segments per submission; tail wait in
`SubmitExecution`; `vkQueueSubmit` µs; cuts skipped and why (direct mode, label, too small); the
existing `direct_recordings`.

**Tests.**
1. Cut-only mode (`BB_VK_RECORD_WORKERS=1`, `BB_VK_SEGMENTS=1`): segments and multi-buffer
   submits on one thread, separating state re-establishment bugs from threading bugs. Then N=2
   and N=3.
2. On a Linux machine (e.g. the Bazzite desktop), force N=3 with `VK_LAYER_KHRONOS_validation`
   and synchronisation validation. The design is driver-independent, and the layers report
   missing state or split label regions.
3. On the Mac, `MTL_DEBUG_LAYER=1` (Metal API validation), if it works for x86-64 processes.
4. Image comparison: a static view with the upscaler off (the default) keeps frames
   deterministic. Flip the toggle, capture the window with `screencapture -l` each time, compare
   with `compare -metric AE -fuzz 1%`. `BB_CAPTURE_TRIGGER` captures must list identical passes
   and draw counts.
5. A 20-minute soak at the level in each mode; on a crash, `BB_WRITE_LOG=2`.

## 8. Assumptions and open questions

- The 93% busy and 59% wait figures come from one profile; DrawRec's busy share sets the ceiling
  for (a).
- Segment and setter costs are estimates; Phase 2 measures them.
- GPU time per frame is unknown; KK serialises the GPU with a barrier at the end of every encoder.
- Metal/AGX scaling across threads under Rosetta decides the outcome; Phase 2 with N=2 answers it
  cheaply before further work.
