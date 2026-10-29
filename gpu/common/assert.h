// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: checks that stop the port (exit code 23) with a message in the GPU log.
#pragma once

#include "common/logging/log.h"

/// Stops the port; noreturn in practice (assert.cpp).
void assert_fail_impl();
/// Set in the shader recipe import process on Windows (no fork there): a failed check throws
/// std::runtime_error instead, and the translation it was in is given up alone.
extern bool g_bb_assert_throws;
[[noreturn]] void unreachable_impl();

#define ASSERT(cond)                                                                               \
    do {                                                                                           \
        if (!(cond)) [[unlikely]] {                                                                \
            LOG_CRITICAL(Debug, "Assertion Failed! ({})", #cond);                                  \
            assert_fail_impl();                                                                    \
        }                                                                                          \
    } while (false)

#define ASSERT_MSG(cond, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) [[unlikely]] {                                                                \
            LOG_CRITICAL(Debug, "Assertion Failed!\n" __VA_ARGS__);                                \
            assert_fail_impl();                                                                    \
        }                                                                                          \
    } while (false)

#define UNREACHABLE()                                                                              \
    do {                                                                                           \
        LOG_CRITICAL(Debug, "Unreachable code!");                                                  \
        unreachable_impl();                                                                        \
    } while (false)

#define UNREACHABLE_MSG(...)                                                                       \
    do {                                                                                           \
        LOG_CRITICAL(Debug, "Unreachable code!\n" __VA_ARGS__);                                    \
        unreachable_impl();                                                                        \
    } while (false)

// Checks that cost something on hot paths: compiled out (as in release builds before).
#define DEBUG_ASSERT(cond)                                                                         \
    do {                                                                                           \
    } while (false)
#define DEBUG_ASSERT_MSG(cond, ...)                                                                \
    do {                                                                                           \
    } while (false)

#define UNIMPLEMENTED() ASSERT_MSG(false, "Unimplemented code!")
#define UNIMPLEMENTED_MSG(...) ASSERT_MSG(false, __VA_ARGS__)
#define UNIMPLEMENTED_IF(cond) ASSERT_MSG(!(cond), "Unimplemented code!")
#define UNIMPLEMENTED_IF_MSG(cond, ...) ASSERT_MSG(!(cond), __VA_ARGS__)

/// ASSERT, then `otherwise` if execution continued past a failed check.
#define ASSERT_OR_EXECUTE(cond, otherwise)                                                         \
    do {                                                                                           \
        ASSERT(cond);                                                                              \
        if (!(cond)) [[unlikely]] {                                                                \
            otherwise                                                                              \
        }                                                                                          \
    } while (false)

#define ASSERT_OR_EXECUTE_MSG(cond, otherwise, ...)                                                \
    do {                                                                                           \
        ASSERT_MSG(cond, __VA_ARGS__);                                                             \
        if (!(cond)) [[unlikely]] {                                                                \
            otherwise                                                                              \
        }                                                                                          \
    } while (false)
