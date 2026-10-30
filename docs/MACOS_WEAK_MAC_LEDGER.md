# Weak-Mac branch: what changed and how to undo it

Branch `macos-native-0.6-weak`, on top of `macos-native-0.6` at `a69f56af` (0.6-v1-pre1).
Nothing here is on `macos-native-0.6` yet. Every change below is either switchable at run
time (the Mac app's **Extra variables** field) or isolated in its own commit, so it can be
turned off for a test or reverted for good.

To turn something off for one run, put its switch in Extra variables. To drop a change for
good, revert its commit on the branch (`git revert <commit>`); review-fix commits go with the
change they fix.

## MetalFX on the driver's queue (branch `macos-native-0.6-mfxq`, 2026-10-25, a test)

Not on `macos-native-0.6-weak`; a branch of its own on top of it.

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `65a68331` | KosmicKrisp patch 0007: `vkExportMetalObjectsEXT` returns a queue's Metal 4 queue | none needed (unused unless asked for) | A driver change: the driver is rebuilt. Revert removes the patch file. |
| `6f0a2895`, `c05d9adb`, `6b0e306b` | MetalFX queue mode: MetalFX committed to KosmicKrisp's own queue in its order (no second queue, no event waits) | it is off unless `BB_METALFX_EVENTS=3` | No stall cutoff in this mode; a MetalFX GPU error there may lose the device. After any MetalFX failure, or without Metal 4 MetalFX, the worker hand-over takes over for the session. |

## Diagnostics (2026-10-25)

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `901d520a`, review fixes after it | Tessellated and rect/quad-list draws counted (frame stats, stall reports); each pipeline with the game's tessellation logged once; "Upscaler: now X (N s after the first frame)" | none (log lines only) | `BB_SKIP_TESS_DRAWS=1` is a test: those draws are not drawn (missing from the picture). Off unless set. |

## Batch 3 (2026-10-24): MetalFX stalls, command thread

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `27c5fd0c` | MetalFX stall cutoff: MetalFX work not done 1 s after it was handed to Metal is given up, the waiting frames go on, FSR 3.1 takes over | `BB_METALFX_STALL_MS=0` | Worker mode (the default) only. `BB_METALFX_STALL_MS=<ms>` sets another cutoff. Test hook: `BB_METALFX_TEST_STALL=<N>` makes the 600th frame of each of MetalFX's first N runs stall for 10 s (off unless set). |
| `10770847` | MetalFX tried again automatically after a run-time failure (30 s, 2 min, 10 min; a fourth failure keeps FSR 3.1 for the session); bbport.ini keeps MetalFX while FSR 3.1 stands in | `BB_METALFX_AUTO_RETRY=0` | Choosing any upscaler in a menu (FSR 3.1 too) ends the retries. Revert together with `27c5fd0c` or after it. For exactly the old behaviour, set both `BB_METALFX_STALL_MS=0` and `BB_METALFX_AUTO_RETRY=0`. |
| `a96c4670` | Command thread sleeps between checks in long queue waits (spins 100 us, then 50 us / 200 us / 1 ms sleeps) | `BB_CP_WAIT_SPIN_US=-1` | arm64 macOS only (after `95407257`); elsewhere it spins as upstream. |
| `95407257` | Review fixes for `a96c4670` (a wait that moved starts afresh, one sleep per pass, a submission ends a sleep early) | `BB_CP_WAIT_SPIN_US=-1` | Revert with `a96c4670`. |
| `a5f6b796` | After a run-time failure, the next MetalFX run gets a new scaler and command queue (macOS ignored the old queue after its GPU timeout, run D) | `BB_METALFX_AUTO_RETRY=0` (no retries, so nothing to rebuild) | Revert with `27c5fd0c` and `10770847`. |
| `b2cf1188` | Review fixes for `27c5fd0c` and `10770847` (choosing FSR 3.1 itself ends the retries, setting order, race after a retry, destructor, 15 s for a new scaler's first frame) | as those two | Revert with them. |

## Batch 2 (2026-10-24): hangs from slow compiles, compression follow-ups

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `8914c72d` | A draw stops waiting for a slow pipeline compile after 250 ms (need-now pipelines); compiles of 250 ms or more are logged | `BB_SYNC_PIPELINE_WAIT_MS=-1` | macOS default 250; `-1` waits as before. The log lines are macOS only in 0.7. |
| `ad76880e` | Texture compression: images upgraded once start with full usage next time | `BB_TEXTURE_COMPRESSION=0` | Part of compression. |
| `0e3a395a` | Review fixes for `8914c72d` and `ad76880e` (need-now compiles go first in the queue, bounded waits) | as those two | |

## Batch 1 (2026-10-23/24): weak-Mac CPU and memory

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `f142d9c0` | Extra sharpening copy in half float instead of 32-bit float | none | Revert only. Halves that copy's memory traffic. macOS only in 0.7. |
| `08814af9` | ThinLTO for the native GPU library | build: `BB_LTO=OFF` | A build setting, not a run-time one. |
| `a5a5e588` | Memory budget of the whole Mac, memory-pressure notifications, thrash guard, footprint log line; small-RAM defaults (16 GB or less) dropped in 0.7 | `BB_MEM_BUDGET=metal` (Metal's limit only), `BB_MEM_PRESSURE=0`, `BB_GC_THRASH_GUARD=0` | Each switch turns off one part. `BB_MEM_RESERVE_MB=<MiB>` changes what macOS is left. The small-RAM defaults (`BB_SMALL_VRAM=0`, `BB_PREUPLOAD_FORCE=1` to undo) are gone in 0.7: guest memory is only in place there, so there are no VRAM copies of guest memory to leave and no background upload to tone down. |
| `9cb9d2bd` | Draw pipe: stage A spins 50 us, then sleeps, in long drains | `BB_DRAIN_SPIN_US=-1` | macOS only. |
| `0d35d3ba` | Per-draw costs of switches that are off (cheaper sampler counter, dynamic-state checks, texture-helper hook, breadcrumbs) | none | Same behaviour by design; revert only. |
| `ea05dba0` | `BB_FRAME_STATS` read the same way everywhere (1, y or t) | none | Diagnostics. macOS only in 0.7 (Linux and Windows keep upstream's readers). |
| `b5271258` | Log lines: performance/efficiency core counts; the game's main thread named | none | Diagnostics. |
| `a556cc9a` | Review fixes for batch 1 | as the changes they fix | |

## Texture compression (2026-10-23)

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| `4e891814` | Step 1: depth images and transfer destinations compressible (with KosmicKrisp patch 0006) | `BB_TEXTURE_COMPRESSION=0` | The switch turns off both the renderer's part and patch 0006's. |
| `edc04e88` | Step 2: compressible color images, upgraded on demand | `BB_TEXTURE_COMPRESSION=0` | |
| `d5859642`, `2e3da4af` | Review fixes | `BB_TEXTURE_COMPRESSION=0` | |
| `102b8683` | Upgrades keep GPU-optimized contents; feedback loops per mip range | `BB_TEXTURE_COMPRESSION=0` | |

## Tessellation heap (2026-10-26)

| Commit | Change | Off for a run | Notes |
|---|---|---|---|
| (KosmicKrisp patch 0010) | The driver's heap for emulated tessellation, geometry shaders and unrolled draws: 1 GiB instead of 128 MiB (512 MiB on Macs with less than 16 GiB; 128 MiB if the larger one cannot be had); a tessellated draw that does not fit draws nothing instead of a runaway draw that hangs the GPU | `BB_KK_POLY_HEAP_MB=128` (the old size; the overflow checks stay); 64 to 2048 | Costs the heap's size in memory for as long as the game runs, once anything tessellates. |

## Everything off at once

Every run-time switch above, as one Extra variables entry:

```
BB_METALFX_STALL_MS=0 BB_METALFX_AUTO_RETRY=0 BB_CP_WAIT_SPIN_US=-1 BB_SYNC_PIPELINE_WAIT_MS=-1 BB_TEXTURE_COMPRESSION=0 BB_MEM_BUDGET=metal BB_MEM_PRESSURE=0 BB_GC_THRASH_GUARD=0 BB_DRAIN_SPIN_US=-1
```

That leaves the revert-only commits (`f142d9c0`, `0d35d3ba`, `ea05dba0`, `b5271258`) and
ThinLTO. To be back exactly where 0.6-v1-pre1 is, use the branch `macos-native-0.6`.

## Not done yet (proposed, waiting for approval)

- MetalFX on KosmicKrisp's own queue (the proper fix for the stall); the next batch.
- Recording threads adaptive to load (one, a second only when the first is busy).
- The rest of the weak-Mac list: real game speed down to 20 FPS (changes game behaviour), a
  weak-Mac preset, rect/quad-list tessellation removal, hidden pass breaks, cheaper precise
  maths, the sceGnmDingDong freeze fix, storage-first images with full usage.
