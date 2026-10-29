// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <cerrno>
#include <cstring>

#include "common/error.h"

namespace Common {

std::string NativeErrorToString(int e) {
    char buffer[256];
#ifdef __APPLE__
    // bbport (macOS): XSI strerror_r, which fills `buffer`.
    if (strerror_r(e, buffer, sizeof(buffer)) != 0) {
        return "error " + std::to_string(e);
    }
    return std::string{buffer};
#else
    // GNU strerror_r: returns the message, which may or may not be in `buffer`.
    return std::string{strerror_r(e, buffer, sizeof(buffer))};
#endif
}

std::string GetLastErrorMsg() {
    return NativeErrorToString(errno);
}

} // namespace Common
