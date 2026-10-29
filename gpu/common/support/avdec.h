// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: libavutil's av_err2str builds a compound literal, which C++ does not have.
#pragma once

#ifdef av_err2str
#undef av_err2str
#include <string>
inline std::string av_err2string(int error) {
    char text[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(text, AV_ERROR_MAX_STRING_SIZE, error);
}
#define av_err2str(error) av_err2string(error).c_str()
#endif
