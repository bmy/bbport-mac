// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the GPU library's log: one line per message on stderr,
// "gpu <level> (<class>): <message> [<file>:<line>]". BB_GPU_LOG=trace|debug|info|warning|error|
// critical sets the threshold (default warning). Formatting happens only for messages that pass.
#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include "common/logging/classes.h"
#include "common/logging/formatter.h"
// Callers have used these through the log header.
#include "common/path_util.h"
#include "common/thread.h"

namespace Common::Log {

enum class Level : int { trace, debug, info, warn, err, critical, off };

[[nodiscard]] bool ShouldLog(Level level);
void Write(std::string_view log_class, Level level, const char* file, int line,
           std::string&& message);
inline void Flush() {}

} // namespace Common::Log

// The level's old spelling (vendored callers pick a level at run time).
namespace spdlog {
using level = Common::Log::Level;
}

#define LOG_GENERIC(log_class, log_level, fmt_str, ...)                                            \
    do {                                                                                           \
        if (::Common::Log::ShouldLog(log_level)) {                                                 \
            ::Common::Log::Write(log_class, log_level, __FILE__, __LINE__,                         \
                                 fmt::format(fmt_str __VA_OPT__(, ) __VA_ARGS__));                 \
        }                                                                                          \
    } while (false)

#define LOG_TRACE(cls, ...)                                                                        \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::trace, __VA_ARGS__)
#define LOG_DEBUG(cls, ...)                                                                        \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::debug, __VA_ARGS__)
#define LOG_INFO(cls, ...)                                                                         \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::info, __VA_ARGS__)
#define LOG_WARNING(cls, ...)                                                                      \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::warn, __VA_ARGS__)
#define LOG_ERROR(cls, ...)                                                                        \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::err, __VA_ARGS__)
#define LOG_CRITICAL(cls, ...)                                                                     \
    LOG_GENERIC(::Common::Log::Class::cls, ::Common::Log::Level::critical, __VA_ARGS__)
