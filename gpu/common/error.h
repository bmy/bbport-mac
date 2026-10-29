// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: system error texts.
#pragma once

#include <string>

namespace Common {

/// The text of errno as it is now (call right after the failing call).
[[nodiscard]] std::string GetLastErrorMsg();
/// The text of error number `e`.
[[nodiscard]] std::string NativeErrorToString(int e);

} // namespace Common
