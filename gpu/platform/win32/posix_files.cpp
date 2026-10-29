// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the file calls of bbport_posix_c.h (the runtime's) with Linux semantics.
#include <windows.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <bcrypt.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>

namespace {

// The C declarations (bbport_posix_c.h is C only).
struct BbStat {
    uint64_t st_dev, st_ino;
    uint32_t st_mode, st_nlink, st_uid, st_gid;
    uint64_t st_rdev;
    int64_t st_size, st_blksize, st_blocks;
    struct timespec st_atim, st_mtim, st_ctim;
};
constexpr int OpenDirectory = 0x200000; // O_DIRECTORY there
constexpr int AtCwd = -100;
constexpr int AtRemoveDir = 0x200;

std::wstring Wide(const std::string& utf8) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    std::wstring out(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, out.data(), n);
    return out;
}

std::string Narrow(const wchar_t* wide) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    std::string out(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), n, nullptr, nullptr);
    return out;
}

void SetErrnoFromWin32() {
    switch (GetLastError()) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_NAME:
        errno = ENOENT;
        break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
        errno = EACCES;
        break;
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS:
        errno = EEXIST;
        break;
    case ERROR_DIR_NOT_EMPTY:
        errno = ENOTEMPTY;
        break;
    case ERROR_DISK_FULL:
        errno = ENOSPC;
        break;
    default:
        errno = EIO;
        break;
    }
}

// Directory fds: no Windows file descriptor opens a directory; these stand for one by its path.
constexpr int DirectoryFdBase = 0x40000000;
std::mutex directories_mutex;
std::unordered_map<int, std::string> directories;
int next_directory = DirectoryFdBase;

bool IsDirectoryFd(int fd) {
    return fd >= DirectoryFdBase;
}

std::string DirectoryPath(int fd) {
    std::scoped_lock lock{directories_mutex};
    const auto it = directories.find(fd);
    return it == directories.end() ? std::string{} : it->second;
}

int NewDirectoryFd(const std::string& path) {
    std::scoped_lock lock{directories_mutex};
    const int fd = next_directory++;
    directories.emplace(fd, path);
    return fd;
}

/// `path` relative to directory fd `dir` (AT_FDCWD: the working directory).
std::string At(int dir, const char* path) {
    if (dir == AtCwd || !path || path[0] == '/' || path[0] == '\\' ||
        (path[0] && path[1] == ':')) {
        return path ? path : "";
    }
    const std::string base = DirectoryPath(dir);
    return base.empty() ? std::string{path} : base + "/" + path;
}

timespec FromFileTime(FILETIME t) {
    // 100 ns units since 1601.
    const uint64_t ticks = (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    const uint64_t unix_ticks = ticks - 116444736000000000ull;
    return timespec{time_t(unix_ticks / 10000000ull), long((unix_ticks % 10000000ull) * 100)};
}

void FillStat(const BY_HANDLE_FILE_INFORMATION& info, BbStat* out) {
    std::memset(out, 0, sizeof(*out));
    const bool directory = info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
    out->st_dev = info.dwVolumeSerialNumber;
    out->st_ino = (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
    out->st_mode = directory ? (_S_IFDIR | 0755)
                             : (_S_IFREG | ((info.dwFileAttributes & FILE_ATTRIBUTE_READONLY) ? 0444 : 0644));
    out->st_nlink = info.nNumberOfLinks;
    out->st_size = directory ? 0 : int64_t((uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow);
    out->st_blksize = 4096;
    out->st_blocks = (out->st_size + 511) / 512;
    out->st_atim = FromFileTime(info.ftLastAccessTime);
    out->st_mtim = FromFileTime(info.ftLastWriteTime);
    out->st_ctim = FromFileTime(info.ftCreationTime);
}

int StatPath(const std::string& path, BbStat* out) {
    const HANDLE file = CreateFileW(Wide(path).c_str(), FILE_READ_ATTRIBUTES,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        SetErrnoFromWin32();
        return -1;
    }
    BY_HANDLE_FILE_INFORMATION info{};
    const BOOL ok = GetFileInformationByHandle(file, &info);
    CloseHandle(file);
    if (!ok) {
        SetErrnoFromWin32();
        return -1;
    }
    FillStat(info, out);
    return 0;
}

int OpenPath(const std::string& path, int flags, int mode) {
    if (flags & OpenDirectory) {
        const DWORD attributes = GetFileAttributesW(Wide(path).c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES) {
            SetErrnoFromWin32();
            return -1;
        }
        if (!(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            errno = ENOTDIR;
            return -1;
        }
        return NewDirectoryFd(path);
    }
    // A directory opened without O_DIRECTORY (stat-like uses) works the same way.
    const DWORD attributes = GetFileAttributesW(Wide(path).c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) &&
        !(flags & (_O_WRONLY | _O_RDWR))) {
        return NewDirectoryFd(path);
    }
    int fd = -1;
    const errno_t error = _wsopen_s(&fd, Wide(path).c_str(), (flags & ~OpenDirectory) | _O_BINARY |
                                                                 _O_NOINHERIT,
                                    _SH_DENYNO, mode ? (_S_IREAD | _S_IWRITE) : _S_IREAD | _S_IWRITE);
    if (error) {
        errno = error;
        return -1;
    }
    return fd;
}

} // namespace

struct bb_DIR {
    HANDLE find = INVALID_HANDLE_VALUE;
    std::string path;
    bool first = true;
    bool done = false;
    WIN32_FIND_DATAW data{};
    struct {
        uint64_t d_ino;
        unsigned char d_type;
        char d_name[1024];
    } entry{};
    int fd = -1;
};

extern "C" {

int bb_stat(const char* path, BbStat* out) {
    return StatPath(path, out);
}

int bb_fstat(int fd, BbStat* out) {
    if (IsDirectoryFd(fd)) {
        return StatPath(DirectoryPath(fd), out);
    }
    const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    BY_HANDLE_FILE_INFORMATION info{};
    if (file == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(file, &info)) {
        errno = EBADF;
        return -1;
    }
    FillStat(info, out);
    return 0;
}

int bb_fstatat(int dir, const char* path, BbStat* out, int) {
    return StatPath(At(dir, path), out);
}

int bb_open(const char* path, int flags, ...) {
    int mode = 0;
    if (flags & _O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    return OpenPath(path, flags, mode);
}

int bb_openat(int dir, const char* path, int flags, ...) {
    int mode = 0;
    if (flags & _O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    return OpenPath(At(dir, path), flags, mode);
}

int bb_close(int fd) {
    if (IsDirectoryFd(fd)) {
        std::scoped_lock lock{directories_mutex};
        return directories.erase(fd) ? 0 : (errno = EBADF, -1);
    }
    return _close(fd);
}

ssize_t bb_pread(int fd, void* buffer, size_t size, int64_t offset) {
    const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    OVERLAPPED at{};
    at.Offset = DWORD(offset);
    at.OffsetHigh = DWORD(uint64_t(offset) >> 32);
    DWORD done = 0;
    if (!ReadFile(file, buffer, DWORD(size > 0x7fffffff ? 0x7fffffff : size), &done, &at)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        SetErrnoFromWin32();
        return -1;
    }
    return ssize_t(done);
}

ssize_t bb_pwrite(int fd, const void* buffer, size_t size, int64_t offset) {
    const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    OVERLAPPED at{};
    at.Offset = DWORD(offset);
    at.OffsetHigh = DWORD(uint64_t(offset) >> 32);
    DWORD done = 0;
    if (!WriteFile(file, buffer, DWORD(size > 0x7fffffff ? 0x7fffffff : size), &done, &at)) {
        SetErrnoFromWin32();
        return -1;
    }
    return ssize_t(done);
}

int bb_fsync(int fd) {
    if (IsDirectoryFd(fd)) return 0;
    return FlushFileBuffers(reinterpret_cast<HANDLE>(_get_osfhandle(fd))) ? 0 : (errno = EIO, -1);
}

int bb_ftruncate(int fd, int64_t size) {
    const errno_t error = _chsize_s(fd, size);
    return error ? (errno = error, -1) : 0;
}

int bb_rename(const char* from, const char* to) {
    if (!MoveFileExW(Wide(from).c_str(), Wide(to).c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        SetErrnoFromWin32();
        return -1;
    }
    return 0;
}

int bb_mkdir(const char* path, int) {
    if (!CreateDirectoryW(Wide(path).c_str(), nullptr)) {
        SetErrnoFromWin32();
        return -1;
    }
    return 0;
}

int bb_mkdirat(int dir, const char* path, int mode) {
    return bb_mkdir(At(dir, path).c_str(), mode);
}

int bb_unlinkat(int dir, const char* path, int flags) {
    const std::wstring wide = Wide(At(dir, path));
    if (!((flags & AtRemoveDir) ? RemoveDirectoryW(wide.c_str()) : DeleteFileW(wide.c_str()))) {
        SetErrnoFromWin32();
        return -1;
    }
    return 0;
}

int bb_renameat(int from_dir, const char* from, int to_dir, const char* to) {
    return bb_rename(At(from_dir, from).c_str(), At(to_dir, to).c_str());
}

ssize_t bb_readlink(const char* path, char* out, size_t size) {
    // The only link the runtime reads: its own executable.
    if (std::strcmp(path, "/proc/self/exe") != 0) {
        errno = EINVAL;
        return -1;
    }
    wchar_t wide[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, wide, DWORD(std::size(wide)));
    if (!n) {
        errno = EIO;
        return -1;
    }
    std::string utf8 = Narrow(wide);
    for (char& c : utf8) {
        if (c == '\\') c = '/';
    }
    const size_t copied = utf8.size() < size ? utf8.size() : size;
    std::memcpy(out, utf8.data(), copied);
    return ssize_t(copied);
}

bb_DIR* bb_opendir(const char* path) {
    auto* dir = new bb_DIR;
    dir->path = path;
    dir->find = FindFirstFileExW(Wide(dir->path + "/*").c_str(), FindExInfoBasic, &dir->data,
                                 FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
    if (dir->find == INVALID_HANDLE_VALUE) {
        SetErrnoFromWin32();
        delete dir;
        return nullptr;
    }
    return dir;
}

bb_DIR* bb_fdopendir(int fd) {
    if (!IsDirectoryFd(fd)) {
        errno = ENOTDIR;
        return nullptr;
    }
    bb_DIR* dir = bb_opendir(DirectoryPath(fd).c_str());
    if (dir) {
        dir->fd = fd; // closedir closes it, as on Linux
    }
    return dir;
}

void* bb_readdir(bb_DIR* dir) {
    if (dir->done) return nullptr;
    if (!dir->first && !FindNextFileW(dir->find, &dir->data)) {
        dir->done = true;
        return nullptr;
    }
    dir->first = false;
    const std::string name = Narrow(dir->data.cFileName);
    std::snprintf(dir->entry.d_name, sizeof(dir->entry.d_name), "%s", name.c_str());
    dir->entry.d_type = (dir->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 4 : 8;
    dir->entry.d_ino = 0;
    return &dir->entry;
}

int bb_closedir(bb_DIR* dir) {
    if (!dir) return -1;
    FindClose(dir->find);
    if (dir->fd >= 0) bb_close(dir->fd);
    delete dir;
    return 0;
}

int bb_dirfd(bb_DIR* dir) {
    if (dir->fd < 0) dir->fd = NewDirectoryFd(dir->path);
    return dir->fd;
}

void bb_rewinddir(bb_DIR* dir) {
    FindClose(dir->find);
    dir->find = FindFirstFileExW(Wide(dir->path + "/*").c_str(), FindExInfoBasic, &dir->data,
                                 FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
    dir->first = true;
    dir->done = dir->find == INVALID_HANDLE_VALUE;
}

int memfd_create(const char* name, unsigned) {
    wchar_t directory[MAX_PATH], path[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory) || !GetTempFileNameW(directory, L"bb", 0, path)) {
        errno = EIO;
        return -1;
    }
    (void)name;
    const HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                                    nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        SetErrnoFromWin32();
        return -1;
    }
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(file), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        CloseHandle(file);
    }
    return fd;
}

void bb_thread_stack(uintptr_t* low, uintptr_t* high) {
    ULONG_PTR l = 0, h = 0;
    GetCurrentThreadStackLimits(&l, &h);
    *low = l;
    *high = h;
}

long bb_gmtoff(const struct tm* local) {
    struct tm copy = *local;
    const time_t t = _mkgmtime(&copy);
    copy = *local;
    const time_t l = mktime(&copy);
    return long(t - l);
}

ssize_t getrandom(void* buffer, size_t size, unsigned) {
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(buffer), ULONG(size),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0
               ? ssize_t(size)
               : -1;
}

} // extern "C"
