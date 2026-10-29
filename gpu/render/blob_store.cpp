// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/blob_store.h"

#include <cstring>
#include <fmt/format.h>

#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"

namespace BbCache {

namespace {

const char* Extension(BlobType type) {
    switch (type) {
    case BlobType::ShaderMeta:
        return ".meta";
    case BlobType::ShaderBinary:
        return ".spv";
    case BlobType::PipelineKey:
        return ".key";
    case BlobType::ShaderProfile:
        return ".bin";
    case BlobType::ShaderInput:
        return ".in";
    }
    return ".blob";
}

} // namespace

BlobStore& BlobStore::Instance() {
    static BlobStore store;
    return store;
}

void BlobStore::Open() {
    if (opened) {
        return;
    }
    // bbport: the cache holds host code (the SRT walkers: x86 code, bytecode elsewhere): another
    // architecture keeps its own (the macOS native GPU process, docs/macos-native-gpu.md).
#if defined(__x86_64__)
    const std::string name{Common::ElfInfo::Instance().GameSerial()};
#else
    const std::string name = std::string{Common::ElfInfo::Instance().GameSerial()} + "-arm64";
#endif
    dir = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / name;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    writer = std::jthread([this](std::stop_token stop) { Writer(stop); });
    opened = true;
}

void BlobStore::Close() {
    if (!opened) {
        return;
    }
    writer.request_stop();
    cv.notify_all();
    writer.join();
    opened = false;
    LOG_INFO(Render, "Pipeline cache written");
}

std::filesystem::path BlobStore::PathOf(BlobType type, const std::string& name) const {
    return dir / (name + Extension(type));
}

bool BlobStore::Save(BlobType type, const std::string& name, std::vector<u8>&& data) {
    return opened && Queue(PathOf(type, name), {}, &data);
}

bool BlobStore::Save(BlobType type, const std::string& name, std::vector<u32>&& data) {
    return opened && Queue(PathOf(type, name),
                           {reinterpret_cast<const u8*>(data.data()), data.size() * sizeof(u32)},
                           nullptr);
}

bool BlobStore::Queue(std::filesystem::path path, std::span<const u8> bytes,
                      std::vector<u8>* owned) {
    Write write{std::move(path), owned ? std::move(*owned)
                                       : std::vector<u8>(bytes.begin(), bytes.end())};
    {
        std::scoped_lock lk{mutex};
        queue.push_back(std::move(write));
    }
    cv.notify_one();
    return true;
}

void BlobStore::Writer(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:PipeCacheIO");
    std::unique_lock lk{mutex};
    for (;;) {
        cv.wait(lk, [&] { return !queue.empty() || stop.stop_requested(); });
        if (queue.empty()) {
            return; // stopping with nothing left to write
        }
        Write write = std::move(queue.front());
        queue.pop_front();
        lk.unlock();
        WriteFile(write);
        lk.lock();
    }
}

void BlobStore::WriteFile(const Write& write) {
    using namespace Common::FS;
    auto temp = write.path;
    temp += ".tmp";
    bool written = false;
    {
        const IOFile file{temp, FileAccessMode::Create};
        written = file.IsOpen() && file.Write(write.bytes) == write.bytes.size();
    }
    std::error_code ec;
    if (written) {
        std::filesystem::rename(temp, write.path, ec);
    }
    if (!written || ec) {
        LOG_ERROR(Render, "Failed to write {}", write.path.string());
        std::filesystem::remove(temp, ec);
    }
}

template <typename T>
void BlobStore::LoadInto(BlobType type, const std::string& name, std::vector<T>& data) {
    if (!opened) {
        return;
    }
    using namespace Common::FS;
    const IOFile file{PathOf(type, name), FileAccessMode::Read};
    if (!file.IsOpen()) {
        data.clear();
        return;
    }
    data.resize(file.GetSize() / sizeof(T));
    file.Read(data);
}

void BlobStore::Load(BlobType type, const std::string& name, std::vector<u8>& data) {
    LoadInto(type, name, data);
}

void BlobStore::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    LoadInto(type, name, data);
}

void BlobStore::ForEachBlob(BlobType type,
                            const std::function<void(std::vector<u8>&& data)>& func) {
    const std::string ext = Extension(type);
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{dir, ec}) {
        if (entry.path().extension() != ext) {
            continue;
        }
        using namespace Common::FS;
        const IOFile file{entry.path(), FileAccessMode::Read};
        if (file.IsOpen()) {
            std::vector<u8> data(file.GetSize());
            file.Read(data);
            func(std::move(data));
        }
    }
}

void BlobStore::Clear() {
    if (!opened) {
        return;
    }
    std::error_code ec;
    u64 removed = 0, kept = 0;
    // The old build's metadata and pipeline keys are kept in previous/: the next shader recipe
    // import rebuilds what it can of them for this build (ImportRecipes, MergeMetas). A cache of
    // another version used to be lost whole (2026-10-27: 1531 files of a player's own cache, the
    // pipelines the recipes lacked compiled again while they played).
    const auto previous = dir / "previous";
    std::filesystem::remove_all(previous, ec);
    std::filesystem::create_directories(previous, ec);
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".in") {
            continue; // the inputs translate again for this build (shader_rebuild.h)
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        if (ext == ".meta" || ext == ".key") {
            std::filesystem::rename(entry.path(), previous / entry.path().filename(), ec);
            if (!ec) {
                ++kept;
                continue;
            }
        }
        if (std::filesystem::remove(entry.path(), ec)) {
            ++removed;
        }
    }
    if (kept == 0) {
        std::filesystem::remove_all(previous, ec);
    }
    // Shader recipes translated for the old profile (shader_rebuild.h): translated again.
    std::filesystem::remove_all(dir / "import", ec);
    std::filesystem::remove_all(dir / "import.tmp", ec);
    LOG_WARNING(Render, "Pipeline cache cleared ({} files): it is rebuilt for this build{}", removed,
                kept ? fmt::format(" ({} metadata and pipeline files kept in previous/ for the "
                                   "shader recipe import)",
                                   kept)
                     : std::string{});
}

} // namespace BbCache
