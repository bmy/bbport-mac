// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): interop test for the Metal command path (docs/MACOS_METAL_RENDERER.md, stage 1).
// BB_METAL_INTEROP_TEST=1: once, when the renderer starts, Metal work is committed to
// KosmicKrisp's own queue between Vulkan submissions, on objects the driver exported (a 2048x2048
// image, 32 rounds); the log says whether each side saw the other's writes (PASS) or where it did
// not (FAIL). KosmicKrisp ends its own encoders with a full barrier, so Vulkan-before-Metal holds
// whatever our barriers do; Metal-before-Vulkan rests on ours. BB_METAL_INTEROP_TEST=2: a
// negative control without the render pass's barriers (a FAIL there shows the test can catch
// disorder; a PASS, only that the GPU ran it in order anyway). Skipped with the validation layers
// on (this port's export entries need them off).

#pragma once

namespace Vulkan {
class Instance;
}

namespace BbMetal {

/// Runs the test if BB_METAL_INTEROP_TEST=1 (macOS with KosmicKrisp only). The caller holds no
/// queue lock; the test takes Scheduler::submit_mutex while it uses the queue.
void RunInteropTestIfAsked(const Vulkan::Instance& instance);

} // namespace BbMetal
