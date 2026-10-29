// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: see log.h.
#include <cstdio>
#include "shim/bb_env.h"
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/logging/log.h"

namespace Common::Log {
namespace {

constexpr const char* LevelNames[] = {"trace", "debug", "info", "warning", "error", "critical"};

Level Threshold() {
    static const Level threshold = [] {
        const char* env = bb_env("BB_GPU_LOG");
        if (env) {
            for (int i = 0; i < 6; ++i) {
                if (std::strcmp(env, LevelNames[i]) == 0) {
                    return Level(i);
                }
            }
        }
        return Level::warn;
    }();
    return threshold;
}

} // namespace

bool ShouldLog(Level level) {
    return level >= Threshold();
}

void Write(std::string_view log_class, Level level, const char* file, int line,
           std::string&& message) {
    static std::mutex mutex;
    const char* slash = std::strrchr(file, '/');
    const int index = int(level) >= 0 && int(level) < 6 ? int(level) : 5;
    std::scoped_lock lock{mutex};
    std::fprintf(stderr, "gpu %s (%.*s): %s [%s:%d]\n", LevelNames[index], int(log_class.size()),
                 log_class.data(), message.c_str(), slash ? slash + 1 : file, line);
}

} // namespace Common::Log
