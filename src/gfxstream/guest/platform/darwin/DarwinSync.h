/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "Sync.h"

namespace gfxstream {

class DarwinSyncHelper : public SyncHelper {
   public:
    DarwinSyncHelper();

    int wait(int syncFd, int timeoutMilliseconds) override;

    void debugPrint(int syncFd) override;

    int dup(int syncFd) override;

    int close(int syncFd) override;
};

}  // namespace gfxstream
