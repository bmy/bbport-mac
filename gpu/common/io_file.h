// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a file opened for the life of the object (stdio underneath): pipeline cache entries,
// shader dumps and patches.
#pragma once

#include <cstdio>
#include <filesystem>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <sys/stat.h>

#include "common/types.h"

namespace Common::FS {

enum class FileAccessMode {
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Append = 1 << 2,
    ReadAppend = Read | Append,
    Create = 1 << 3, ///< write, emptying the file or making a new one
};

enum class FileType {
    BinaryFile,
    TextFile,
};

class IOFile final {
public:
    IOFile() = default;
    IOFile(const std::filesystem::path& path, FileAccessMode mode,
           FileType type = FileType::BinaryFile) {
        Open(path, mode, type);
    }
    ~IOFile() {
        Close();
    }
    IOFile(const IOFile&) = delete;
    IOFile& operator=(const IOFile&) = delete;
    IOFile(IOFile&& other) noexcept : file{std::exchange(other.file, nullptr)} {}
    IOFile& operator=(IOFile&& other) noexcept {
        if (this != &other) {
            Close();
            file = std::exchange(other.file, nullptr);
        }
        return *this;
    }

    void Open(const std::filesystem::path& path, FileAccessMode mode,
              FileType type = FileType::BinaryFile) {
        Close();
#ifdef _WIN32
        const char* narrow = ModeString(mode, type);
        wchar_t wide[8] = {};
        for (int i = 0; i < 7 && narrow[i]; ++i) {
            wide[i] = wchar_t(narrow[i]);
        }
        file = _wfopen(path.c_str(), wide);
#else
        file = std::fopen(path.c_str(), ModeString(mode, type));
#endif
    }
    void Close() {
        if (file) {
            std::fclose(file);
            file = nullptr;
        }
    }
    [[nodiscard]] bool IsOpen() const {
        return file != nullptr;
    }

    [[nodiscard]] u64 GetSize() const {
        struct stat info {};
        return file && fstat(fileno(file), &info) == 0 ? u64(info.st_size) : 0;
    }

    /// Fills a contiguous container (count of elements read) or one object (1 or 0).
    template <typename T>
    std::size_t Read(T& data) const {
        if constexpr (requires { std::span{data}; }) {
            const auto span = std::span{data};
            return ReadRaw<typename decltype(span)::value_type>(span.data(), span.size());
        } else {
            static_assert(std::is_trivially_copyable_v<T>);
            return ReadRaw<T>(&data, 1);
        }
    }

    /// Writes a contiguous container (count of elements written) or one object (1 or 0).
    template <typename T>
    std::size_t Write(const T& data) const {
        if constexpr (requires { std::span{data}; }) {
            const auto span = std::span{data};
            return WriteRaw<typename decltype(span)::element_type>(span.data(), span.size());
        } else {
            static_assert(std::is_trivially_copyable_v<T>);
            return WriteRaw<T>(&data, 1);
        }
    }

    template <typename T>
    std::size_t WriteSpan(std::span<const T> data) const {
        return WriteRaw<T>(data.data(), data.size());
    }
    std::size_t WriteString(std::string_view text) const {
        return WriteRaw<char>(text.data(), text.size());
    }

    /// `count` objects of T.
    template <typename T>
    std::size_t ReadRaw(void* data, std::size_t count) const {
        return file ? std::fread(data, sizeof(T), count, file) : 0;
    }
    template <typename T>
    std::size_t WriteRaw(const void* data, std::size_t count) const {
        if (!file) {
            return 0;
        }
        const std::size_t written = std::fwrite(data, sizeof(T), count, file);
        return std::fflush(file) == 0 ? written : 0;
    }

private:
    static const char* ModeString(FileAccessMode mode, FileType type) {
        const bool text = type == FileType::TextFile;
        switch (mode) {
        case FileAccessMode::Read:
            return text ? "r" : "rb";
        case FileAccessMode::Write:
        case FileAccessMode::Create:
            return text ? "w" : "wb";
        case FileAccessMode::ReadWrite:
            return text ? "r+" : "r+b";
        case FileAccessMode::Append:
            return text ? "a" : "ab";
        case FileAccessMode::ReadAppend:
            return text ? "a+" : "a+b";
        }
        return "rb";
    }

    std::FILE* file = nullptr;
};

} // namespace Common::FS
