// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include "shim/bb_env.h"
#include <cstdlib>
#include <mutex>

#include <limits.h>

#include "common/logging/log.h"
#include "common/path_util.h"

namespace Common::FS {
namespace fs = std::filesystem;
namespace {

constexpr std::size_t NumTypes = std::size_t(PathType::CustomModulesDir) + 1;

/// Sub-folder of the user folder for each type ("" = the user folder itself).
constexpr std::array<const char*, NumTypes> SubFolders = {
    "",          "log",       "screenshots",   "shader",         "temp",
    "data",      "sys_modules", "download",    "captures",       "cheats",
    "patches",   "game_data", "custom_trophy", "custom_configs", "cache",
    "fonts",     "trophy",    "home",          "custom_modules",
};

struct Paths {
    std::mutex mutex;
    std::array<fs::path, NumTypes> paths;
    std::array<bool, NumTypes> ready{};

    Paths() {
        const char* env = bb_env("BB_GPU_USER_DIR");
        const fs::path user = env && *env ? fs::path{env} : fs::current_path() / "user";
        for (std::size_t i = 0; i < NumTypes; ++i) {
            paths[i] = *SubFolders[i] ? user / SubFolders[i] : user;
        }
    }
};

Paths& Instance() {
    static Paths paths;
    return paths;
}

} // namespace

bool ValidatePath(const fs::path& path) {
    if (path.empty()) {
        LOG_ERROR(Common_Filesystem, "empty path");
        return false;
    }
    if (path.native().size() >= PATH_MAX) {
        LOG_ERROR(Common_Filesystem, "path too long: {}", path.string());
        return false;
    }
    return true;
}

std::string PathToUTF8String(const fs::path& path) {
    const auto text = path.u8string();
    return std::string{text.begin(), text.end()};
}

const fs::path& GetUserPath(PathType type) {
    auto& state = Instance();
    const auto index = std::size_t(type);
    std::scoped_lock lock{state.mutex};
    if (!state.ready[index]) {
        std::error_code ec;
        fs::create_directories(state.paths[index], ec);
        state.ready[index] = true;
    }
    return state.paths[index];
}

std::string GetUserPathString(PathType type) {
    return PathToUTF8String(GetUserPath(type));
}

void SetUserPath(PathType type, const fs::path& path) {
    if (!fs::is_directory(path)) {
        LOG_ERROR(Common_Filesystem, "not a directory: {}", path.string());
        return;
    }
    auto& state = Instance();
    std::scoped_lock lock{state.mutex};
    state.paths[std::size_t(type)] = path;
    state.ready[std::size_t(type)] = true;
}

std::optional<fs::path> FindGameByID(const fs::path& dir, const std::string& game_id,
                                     int max_depth) {
    if (max_depth < 0) {
        return std::nullopt;
    }
    const auto eboot_in = [](const fs::path& root) -> std::optional<fs::path> {
        std::error_code ec;
        if (fs::is_directory(root, ec) && fs::exists(root / "sce_sys" / "param.sfo", ec) &&
            fs::exists(root / "eboot.bin", ec)) {
            return root / "eboot.bin";
        }
        return std::nullopt;
    };
    if (dir.filename() == game_id) {
        if (auto found = eboot_in(dir)) {
            return found;
        }
    }
    if (auto found = eboot_in(dir / game_id)) {
        return found;
    }
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::error_code entry_ec;
        if (entry.is_directory(entry_ec)) {
            if (auto found = FindGameByID(entry.path(), game_id, max_depth - 1)) {
                return found;
            }
        }
    }
    return std::nullopt;
}

} // namespace Common::FS
