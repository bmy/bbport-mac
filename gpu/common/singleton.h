// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: one lazily created instance of a type.
#pragma once

#include <memory>

namespace Common {

template <class T>
class Singleton {
public:
    static T* Instance() {
        // Never destroyed: threads of the game may still use it while the process exits.
        static T* const instance = new T();
        return instance;
    }

protected:
    Singleton() = default;
    ~Singleton() = default;
};

} // namespace Common
