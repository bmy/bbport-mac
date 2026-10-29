// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: one complete line of output in a single write. The game's stdout and stderr are
// unbuffered and go to one pipe, which the launchers read line by line; printf writes each piece
// of a line on its own, so the lines of two threads could mix ("Co-op: gpu Infowarning: ...").
// A line written here stays whole (up to a few KiB).
#pragma once

#include <cerrno>
#include <cstdio>
#include <string_view>

#include <unistd.h>

namespace Common::Log {

inline void WriteLine(std::FILE* stream, std::string_view line) {
    std::fflush(stream);
    const int fd = fileno(stream);
    while (!line.empty()) {
        const ssize_t written = ::write(fd, line.data(), line.size());
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return;
        }
        line.remove_prefix(static_cast<std::size_t>(written));
    }
}

} // namespace Common::Log
