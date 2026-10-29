// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the host architecture (the port runs on x86-64).
#pragma once

#if defined(__x86_64__)
#define ARCH_X86_64 1
#elif defined(__aarch64__)
#define ARCH_ARM64 1
#endif
