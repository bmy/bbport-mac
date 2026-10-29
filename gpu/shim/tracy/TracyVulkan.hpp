// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: profiler disabled.
#pragma once
#include "Tracy.hpp"
#define TracyVkCtx void*
#define TracyVkCollect(ctx, cmdbuf)
#define TracyVkNamedZoneC(ctx, var, cmd, name, color, active)
#define TracyVkContextName(ctx, name, size)
#define TracyVkContextHostCalibrated(...) nullptr
#define TracyVkDestroy(ctx)
namespace tracy { struct VkCtxScope { template <class... A> VkCtxScope(A&&...) {} }; }
