// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "buffers/memory_model.h"
#include "shim/bb_env.h"

#include <cstdio>
#include <cstdlib>

#include "bbport_toggles.h"

namespace VideoCore {

void GuestMemoryUnusable(const char* why) {
#ifdef __APPLE__
    // bbport (macOS): host memory import of the guest pool (a shared memory object) only.
    std::printf("STOP: the GPU driver cannot use the game's memory in place: %s. On macOS the port "
                "needs KosmicKrisp's VK_EXT_external_memory_host; please report the Mac and macOS "
                "version with this log.\n",
                why);
    std::fflush(stdout);
    std::exit(23);
#endif
    std::printf("STOP: the GPU driver cannot use the game's memory in place: %s. The port needs "
                "dma-buf export of system memory or VK_EXT_external_memory_host (a current Mesa, "
                "NVIDIA or Windows driver); please report the GPU and driver with this log.\n",
                why);
    std::fflush(stdout);
    std::exit(23);
}

bool WriteVerify() {
    static const bool on = bb_env_on("BB_WRITE_VERIFY");
    return on;
}

bool ConstantsInPlace() {
    static const bool on = bb_env_on("BB_CONSTANTS_IN_PLACE");
    return on != BbToggle::Experiment(1);
}

} // namespace VideoCore
