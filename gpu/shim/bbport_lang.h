// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: language of the in-game overlay. BB_UI_LANGUAGE=en or ru picks it; otherwise English
// on macOS and Russian elsewhere (the original UI). Decided once per process.

#pragma once

#include <cstdlib>
#include <cstring>

namespace BbLang {

inline bool English() {
    static const bool english = [] {
        if (const char* value = std::getenv("BB_UI_LANGUAGE")) {
            if (std::strncmp(value, "en", 2) == 0) return true;
            if (std::strncmp(value, "ru", 2) == 0) return false;
        }
#ifdef __APPLE__
        return true;
#else
        return false;
#endif
    }();
    return english;
}

/// The Russian or English text; both must take the same printf arguments.
inline const char* Tr(const char* ru, const char* en) {
    return English() ? en : ru;
}

} // namespace BbLang
