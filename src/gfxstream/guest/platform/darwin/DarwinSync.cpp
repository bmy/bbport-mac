/*
 * Copyright 2026 Mesa3D authors
 * SPDX-License-Identifier: MIT
 */

#include "DarwinSync.h"

namespace gfxstream {

DarwinSyncHelper::DarwinSyncHelper() {}

int DarwinSyncHelper::wait(int syncFd, int timeoutMilliseconds) {
    return -1;  // stub constant
}

void DarwinSyncHelper::debugPrint(int syncFd) {}

int DarwinSyncHelper::dup(int syncFd) {
    return -1;  // stub constant
}

int DarwinSyncHelper::close(int syncFd) {
    return -1;  // stub constant
}

SyncHelper* osCreateSyncHelper() {
    return nullptr;  // stub constant
}

}  // namespace gfxstream
