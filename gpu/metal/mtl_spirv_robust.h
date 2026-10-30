// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): robust buffer access for the Metal command path (docs/MACOS_METAL_RENDERER.md,
// stage 4). The renderer relies on robustBufferAccess2 and nullDescriptor: a read past the end of
// a storage buffer gives zero, a write there is dropped, and a null buffer descriptor reads as an
// empty buffer. KosmicKrisp provides that; Metal does not (a shader reads raw addresses), so the
// SPIR-V is rewritten before SPIRV-Cross sees it.

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/types.h"

namespace Metal {

/// Result of RewriteRobustBuffers.
struct RobustRewrite {
    bool ok{};              ///< false: `error` says why; `spirv` is empty
    std::vector<u32> spirv; ///< the rewritten module (the input as it was when nothing changed)
    std::string error;
    u32 checked{};          ///< element accesses given a bounds check
};

/// Every element access of a StorageBuffer block whose last member is a runtime array (the
/// recompiler's buffers: `struct { T data[]; }`) is checked against the array's length
/// (OpArrayLength, which SPIRV-Cross reads from its buffer of buffer sizes, filled from the bound
/// descriptor's range):
///   - the element index becomes `in_bounds ? index : 0`, so no access leaves the buffer's first
///     element (a null descriptor binds a zeroed stand-in);
///   - a load (also OpAtomicLoad) gives `in_bounds ? value : 0`;
///   - a store and every other atomic run only when in bounds (their block is split into a
///     selection; an atomic's result is 0 otherwise).
/// Variables in descriptor set `skip_set` (buffers the Metal path binds itself, without a size)
/// are left alone, as are uniform buffers (read-only; a null one binds a 64 KiB zeroed stand-in).
/// Fails, rather than leaving an access unchecked, for shapes it does not know: any other storage
/// buffer (a fixed-size block, an array of buffers, a Uniform BufferBlock, a runtime array not at
/// offset 0), a checked buffer used other than through such access chains and OpArrayLength, an
/// element pointer used other than by loads, stores and atomics, an index that is not a 32-bit
/// integer, variable pointers, and a store or atomic in a loop header block.
/// Known limit: a load is not split, so with a bound range smaller than one element it still
/// reads that first element (up to its stride past the range) before the result is replaced.
/// bbport: `fixed_unchecked`: a fixed-size storage block (no runtime array last) is left
/// unchecked instead of failing (the renderer's own shaders: CompileVariant::host).
RobustRewrite RewriteRobustBuffers(std::span<const u32> spirv, u32 skip_set,
                                   bool fixed_unchecked = false);

} // namespace Metal
