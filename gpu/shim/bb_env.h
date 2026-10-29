// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the port's environment variables (BB_*), read through here only. Every name is listed,
// with what it does, in bb_env_vars.def next to this file (docs/ENVIRONMENT.md is made from it by
// scripts/env_vars.py; tests/test_env_vars.py fails on a name read here and missing there, or listed
// there and read nowhere). C and C++.
#ifndef BB_ENV_H
#define BB_ENV_H

#include <stdlib.h>

/// The variable's text, or NULL when it is not set.
static inline const char* bb_env(const char* name) {
    return getenv(name);
}

/// Set and starting with '1' (BB_X=1).
static inline int bb_env_on(const char* name) {
    const char* value = getenv(name);
    return value && value[0] == '1';
}

/// Set and starting with '0' (BB_X=0: a feature on by default switched off).
static inline int bb_env_off(const char* name) {
    const char* value = getenv(name);
    return value && value[0] == '0';
}

/// The variable as a number (decimal, or hex after 0x), or `fallback` when it is not set or empty.
static inline long long bb_env_int(const char* name, long long fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    const int hex = value[0] == '0' && (value[1] == 'x' || value[1] == 'X');
    return strtoll(value, NULL, hex ? 16 : 10);
}

/// The variable as a decimal fraction, or `fallback` when it is not set or empty.
static inline double bb_env_float(const char* name, double fallback) {
    const char* value = getenv(name);
    return value && value[0] ? strtod(value, NULL) : fallback;
}

#endif
