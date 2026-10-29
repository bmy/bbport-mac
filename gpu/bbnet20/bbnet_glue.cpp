// SPDX-License-Identifier: GPL-2.0-or-later
// bbport co-op: what shadPS4 0.20.0's network and NP libraries need from the rest of
// shadPS4, implemented on top of the bbport runtime, and their registration.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "common/elf_info.h"
#include "common/logging/line.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/debugger.h"
#include "core/bbport_bbsc_kit.h" // Teefpyre (SrShadowy fork)
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/libs.h"
#include "core/libraries/system/systemservice.h"
#include "core/libraries/system/userservice.h"
#include "core/user_manager.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/shadnet_notifications_layer.h"

#include "core/libraries/network/http.h"
#include "core/libraries/network/http2.h"
#include "core/libraries/net/net.h"
#include "core/libraries/network/netctl.h"
#include "core/libraries/network/ssl2.h"
#include "core/libraries/np/np_auth.h"
#include "core/libraries/np/np_common.h"
#include "core/libraries/np/np_manager.h"
#include "core/libraries/np/np_matching2/np_matching2.h"
#include "core/libraries/np/np_partner.h"
#include "core/libraries/np/np_party.h"
#include "core/libraries/np/np_score/np_score.h"
#include "core/libraries/np/np_signaling/np_signaling.h"
#include "core/libraries/np/np_tus/np_tus.h"
#include "core/libraries/np/np_utility/np_utility.h"
#include "core/libraries/np/np_web_api/np_web_api.h"
#include "core/libraries/np/np_web_api2/np_web_api2.h"
#include <algorithm>
#include <sys/mman.h>
#include <SDL3/SDL_events.h>

#include "bbgpu.h"
#include "common/memory_patcher.h"
#include "common/singleton.h"
#include "core/bbsc_cinematic.h"
#include "core/bbsc_persistence.h"
#include "core/bbsc_relay_client.h"
#include "core/bloodborne_re.h"
#include "core/cpu_patches.h"
#include "core/file_format/psf.h"
#include "games/game_profile.h"
#include "thd/thd.h"

#include "bbnet20/bbnet.h"

namespace {
/// The runtime's side (bbnet_init).
BbNetHost g_host{};
} // namespace

// The Hunter's Dream backend reads game structures (the player's position for the live map):
// only from mapped guest memory (bbport).
namespace BbEngine {
bool host_read_safe(const void* p, void* out, std::size_t n) {
    if (!g_host.memory_is_mapped(reinterpret_cast<uintptr_t>(p), n)) {
        return false;
    }
    std::memcpy(out, p, n);
    return true;
}
} // namespace BbEngine

// ---- Kernel ---------------------------------------------------------------------------------

namespace Libraries::Kernel {

struct BbNetMutex {
    std::recursive_mutex mutex;
    bool recursive{};
    int depth{};
};
struct BbNetMutexAttr {
    PthreadMutexType type{PthreadMutexType::ErrorCheck};
};
struct BbNetCond {
    std::condition_variable_any cond;
};
struct BbNetCondAttr {};
struct BbNetThreadAttr {
    u64 stack{};
};

s32 ErrnoToSceKernelError(s32 e) {
    return e > 0 && e <= 0xffff ? s32(0x80020000u | u32(e)) : e;
}

s32 PS4_SYSV_ABI sceKernelGetSystemSwVersion(SwVersionStruct* ret) {
    if (!ret) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    std::memset(ret->text_representation, 0, sizeof(ret->text_representation));
    std::snprintf(ret->text_representation, sizeof(ret->text_representation), "%s", "11.000.000");
    ret->hex_representation = 0x11000001;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelClockGettime(u32 clock_id, OrbisKernelTimespec* tp) {
    if (!tp) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    s64 ns;
    if (clock_id == ORBIS_CLOCK_REALTIME || clock_id == ORBIS_CLOCK_REALTIME_PRECISE ||
        clock_id == ORBIS_CLOCK_REALTIME_FAST || clock_id == ORBIS_CLOCK_SECOND) {
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
    } else {
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count();
    }
    tp->tv_sec = ns / 1000000000;
    tp->tv_nsec = ns % 1000000000;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMapNamedFlexibleMemory(void** addr_in_out, u64 len, s32, s32,
                                                 const char*) {
    if (!addr_in_out || !len) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    void* memory = std::calloc(1, len);
    if (!memory) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }
    *addr_in_out = memory;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMunmap(void* addr, u64) {
    std::free(addr);
    return ORBIS_OK;
}

int posix_pthread_mutexattr_init(PthreadMutexAttrT* attr) {
    if (!attr) return 22;
    *attr = new BbNetMutexAttr;
    return 0;
}
int posix_pthread_mutexattr_settype(PthreadMutexAttrT* attr, PthreadMutexType type) {
    if (!attr || !*attr) return 22;
    (*attr)->type = type;
    return 0;
}
int posix_pthread_mutexattr_destroy(PthreadMutexAttrT* attr) {
    if (!attr) return 22;
    delete *attr;
    *attr = nullptr;
    return 0;
}
int scePthreadMutexInit(PthreadMutexT* mutex, const PthreadMutexAttrT* attr, const char*) {
    if (!mutex) return 22;
    *mutex = new BbNetMutex;
    (*mutex)->recursive = attr && *attr && (*attr)->type == PthreadMutexType::Recursive;
    return 0;
}
int posix_pthread_mutex_destroy(PthreadMutexT* mutex) {
    if (!mutex || !*mutex) return 22;
    delete *mutex;
    *mutex = nullptr;
    return 0;
}
int posix_pthread_mutex_lock(PthreadMutexT* mutex) {
    if (!mutex || !*mutex) return 22;
    (*mutex)->mutex.lock();
    if (++(*mutex)->depth > 1 && !(*mutex)->recursive) {
        --(*mutex)->depth;
        (*mutex)->mutex.unlock();
        return 11; // EDEADLK
    }
    return 0;
}
int posix_pthread_mutex_trylock(PthreadMutexT* mutex) {
    if (!mutex || !*mutex) return 22;
    if (!(*mutex)->mutex.try_lock()) return 16; // EBUSY
    if (++(*mutex)->depth > 1 && !(*mutex)->recursive) {
        --(*mutex)->depth;
        (*mutex)->mutex.unlock();
        return 16;
    }
    return 0;
}
int posix_pthread_mutex_unlock(PthreadMutexT* mutex) {
    if (!mutex || !*mutex) return 22;
    --(*mutex)->depth;
    (*mutex)->mutex.unlock();
    return 0;
}

int posix_pthread_condattr_init(PthreadCondAttrT* attr) {
    if (!attr) return 22;
    *attr = new BbNetCondAttr;
    return 0;
}
int posix_pthread_condattr_destroy(PthreadCondAttrT* attr) {
    if (!attr) return 22;
    delete *attr;
    *attr = nullptr;
    return 0;
}
int scePthreadCondInit(PthreadCondT* cond, const PthreadCondAttrT*, const char*) {
    if (!cond) return 22;
    *cond = new BbNetCond;
    return 0;
}
int posix_pthread_cond_destroy(PthreadCondT* cond) {
    if (!cond || !*cond) return 22;
    delete *cond;
    *cond = nullptr;
    return 0;
}
int posix_pthread_cond_signal(PthreadCondT* cond) {
    if (!cond || !*cond) return 22;
    (*cond)->cond.notify_one();
    return 0;
}
namespace {
// The caller holds the mutex once; the wait releases and retakes it like pthread_cond_wait.
struct HeldMutex {
    BbNetMutex* mutex;
    void lock() {
        mutex->mutex.lock();
        ++mutex->depth;
    }
    void unlock() {
        --mutex->depth;
        mutex->mutex.unlock();
    }
};
} // namespace
int posix_pthread_cond_wait(PthreadCondT* cond, PthreadMutexT* mutex) {
    if (!cond || !*cond || !mutex || !*mutex) return 22;
    HeldMutex held{*mutex};
    (*cond)->cond.wait(held);
    return 0;
}
int posix_pthread_cond_reltimedwait_np(PthreadCondT* cond, PthreadMutexT* mutex, u64 usec) {
    if (!cond || !*cond || !mutex || !*mutex) return 22;
    HeldMutex held{*mutex};
    const auto status = (*cond)->cond.wait_for(held, std::chrono::microseconds(usec));
    return status == std::cv_status::timeout ? 60 : 0; // ETIMEDOUT
}

int posix_pthread_attr_init(PthreadAttrT* attr) {
    if (!attr) return 22;
    *attr = new BbNetThreadAttr;
    return 0;
}
int posix_pthread_attr_destroy(PthreadAttrT* attr) {
    if (!attr) return 22;
    delete *attr;
    *attr = nullptr;
    return 0;
}
int posix_pthread_attr_setstacksize(PthreadAttrT* attr, size_t stack_size) {
    if (!attr || !*attr) return 22;
    (*attr)->stack = stack_size;
    return 0;
}
int posix_pthread_attr_setinheritsched(PthreadAttrT*, int) {
    return 0;
}
int posix_pthread_attr_setschedpolicy(PthreadAttrT*, SchedPolicy) {
    return 0;
}
int posix_pthread_attr_setschedparam(PthreadAttrT*, const SchedParam*) {
    return 0;
}
int scePthreadAttrSetaffinity(PthreadAttrT*, u64) {
    return 0;
}
int posix_pthread_create_name_np(PthreadT* thread, const PthreadAttrT* attr,
                                 PthreadEntryFunc start, void* arg, const char* name) {
    if (!thread || !start) return 22;
    const u64 stack = attr && *attr ? (*attr)->stack : 0;
    return g_host.thread_spawn(thread, start, arg, stack, name ? name : "np");
}
int posix_pthread_join(PthreadT thread, void** ret) {
    return g_host.thread_join(thread, ret);
}
void posix_pthread_exit(void* status) {
    g_host.thread_exit(status);
}

} // namespace Libraries::Kernel

// ---- User and system service, debugger ----------------------------------------------------------

namespace Libraries::UserService {
s32 PS4_SYSV_ABI sceUserServiceGetInitialUser(int* user_id) {
    if (!user_id) {
        return s32(0x80960009); // SCE_USER_SERVICE_ERROR_INVALID_ARGUMENT
    }
    *user_id = UserManager::RuntimeUserId;
    return ORBIS_OK;
}
} // namespace Libraries::UserService

namespace Libraries::SystemService {
void PushSystemServiceEvent(const OrbisSystemServiceEvent& event) {
    std::printf("Co-op: system service event %#x not delivered (invitations are not ported)\n",
                unsigned(event.event_type));
}
} // namespace Libraries::SystemService

namespace Core::Debugger {
s32 GetCurrentPid() {
#ifdef _WIN32
    return s32(GetCurrentProcessId());
#else
    return s32(getpid());
#endif
}
} // namespace Core::Debugger

// ---- Notifications ------------------------------------------------------------------------------
// The console and the log get every one; an Info notice (the login failed, the server is down or
// refuses this client) also shows for a while on the port's overlay line, over the game.

namespace ImGui::ShadNetNotify {
namespace {
std::mutex history_mutex;
std::vector<HistoryEntry> history;
std::size_t unread{};
// Bumped by every user of the overlay line: a notice clears it only while it is still its own.
std::atomic<u64> overlay_line_generation{0};
constexpr auto NoticeShown = std::chrono::seconds(15);
} // namespace
void OverlayLineTaken() {
    ++overlay_line_generation;
}
KindDisplay DisplayOf(Kind kind) {
    switch (kind) {
    case Kind::FriendRequest:
        return {"Friend request", {0.9f, 0.8f, 0.3f, 1.0f}};
    case Kind::FriendNew:
        return {"Friend", {0.4f, 0.9f, 0.4f, 1.0f}};
    case Kind::FriendLost:
        return {"Friend", {0.9f, 0.4f, 0.4f, 1.0f}};
    case Kind::Online:
        return {"Online", {0.4f, 0.8f, 0.9f, 1.0f}};
    case Kind::Warning:
        return {"Warning", {0.95f, 0.6f, 0.2f, 1.0f}};
    default:
        return {"Info", {0.8f, 0.8f, 0.8f, 1.0f}};
    }
}
void Push(Kind kind, std::string text) {
    // One write of the whole line: the shadNet threads log at the same moment.
    Common::Log::WriteLine(stdout,
                           std::string("Co-op: ") + DisplayOf(kind).tag + ": " + text + "\n");
    if (kind == Kind::Info) {
        const u64 mine = ++overlay_line_generation;
        bbgpu_overlay_banner(text.c_str());
        std::thread([mine] {
            std::this_thread::sleep_for(NoticeShown);
            if (overlay_line_generation.load() == mine) {
                bbgpu_overlay_banner(nullptr);
            }
        }).detach();
    }
    std::scoped_lock lock{history_mutex};
    history.push_back({kind, std::move(text), {}});
    ++unread;
}
std::vector<HistoryEntry> GetHistory() {
    std::scoped_lock lock{history_mutex};
    return history;
}
void ClearHistory() {
    std::scoped_lock lock{history_mutex};
    history.clear();
    unread = 0;
}
std::size_t UnreadCount() {
    std::scoped_lock lock{history_mutex};
    return unread;
}
void MarkRead() {
    std::scoped_lock lock{history_mutex};
    unread = 0;
}
void Register() {}
void Unregister() {}
} // namespace ImGui::ShadNetNotify

namespace ImGui::InvitationPrompt {
void Push(s32, std::string invitation_id, std::string, std::string from_npid) {
    std::printf("Co-op: invitation %s from %s (invitations are not supported yet)\n",
                invitation_id.c_str(), from_npid.c_str());
}
void Dismiss(const std::string&) {}
void Register() {}
void Unregister() {}
} // namespace ImGui::InvitationPrompt

// ---- Game identity -----------------------------------------------------------------------------

// ElfInfo is filled in by shadPS4's Core::Emulator; this port has no emulator class, so the
// same name is used here for that one job (bbport).
namespace Core {
class Emulator {
public:
    static void SetGame(const char* app0, const char* serial, const char* title,
                        const char* app_ver) {
        auto& info = Common::ElfInfo::Instance();
        info.game_serial = serial ? serial : "";
        info.title = title ? title : "";
        info.app_ver = app_ver ? app_ver : "";
        info.game_folder = app0 ? app0 : "";
        info.initialized = true;
        info.npCommIds = ReadNpCommIds(info.game_folder / "sce_sys" / "npbind.dat");
    }

private:
    // npbind.dat: a 0x80-byte header, then tagged records (big-endian u16 tag and length);
    // tag 0x10 holds an NP communication id such as NPWR05818_00.
    static std::vector<std::string> ReadNpCommIds(const std::filesystem::path& path) {
        std::vector<std::string> ids;
        std::ifstream file{path, std::ios::binary};
        const std::vector<u8> data{std::istreambuf_iterator<char>(file), {}};
        for (size_t at = 0x80; at + 4 <= data.size();) {
            const u16 tag = u16(data[at] << 8 | data[at + 1]);
            const u16 size = u16(data[at + 2] << 8 | data[at + 3]);
            at += 4;
            if (at + size > data.size()) {
                break;
            }
            if (tag == 0x10 && size) {
                std::string id(reinterpret_cast<const char*>(&data[at]), size);
                id.resize(std::strlen(id.c_str()));
                ids.push_back(id);
            }
            at += size;
        }
        return ids;
    }
};
} // namespace Core

// ---- The module's C interface (bbnet.h) -------------------------------------------------------

// The module's own symbol table (libbbgpu keeps another one: this library's symbols are its own,
// -Bsymbolic).
namespace {
struct Symbol {
    std::string nid;
    u64 address;
};
std::vector<Symbol> g_symbols;
} // namespace

namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char* nid, const char*, const char*, SymbolType,
                                u64 address) {
    g_symbols.push_back({nid, address});
}
} // namespace Core::Loader

#ifdef _WIN32
#define BBNET_EXPORT extern "C" __declspec(dllexport)
#else
#define BBNET_EXPORT extern "C" __attribute__((visibility("default")))
#endif

BBNET_EXPORT uintptr_t bbnet_resolve(const char* scoped_nid) {
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_symbols) {
        if (symbol.nid.size() == length && !std::memcmp(symbol.nid.data(), scoped_nid, length)) {
            return uintptr_t(symbol.address);
        }
    }
    return 0;
}

BBNET_EXPORT void bbnet_image_mapped(uint8_t* image, uint64_t size) {
    // The Hunter's Dream backend reads the game's code before the patches change it.
    BbThd::SetImage(image, size);
}

BBNET_EXPORT void bbnet_image_ready(uint64_t base, uint64_t size) {
    MemoryPatcher::g_eboot_address = base;
    MemoryPatcher::g_eboot_image_size = size;
    // BBSC (Bloodborne seamless co-op): BB_BBSC=1 (the launcher's
    // co-op switch). Its game code checks for CUSA03173 01.09; every retail Bloodborne runs the
    // same 1.09 executable (gpu/games/bloodborne.cpp), and each of its sites is byte-checked.
    const char* bbsc = std::getenv("BB_BBSC");
    if (!bbsc || bbsc[0] != '1') {
        return;
    }
    const Game::Profile* profile = Game::Active();
    if (!profile || std::string_view{profile->name} != "Bloodborne") {
        std::printf("BBSC: not Bloodborne: seamless co-op off\n");
        return;
    }
    MemoryPatcher::g_game_serial = "CUSA03173";
    Common::Singleton<PSF>::Instance()->app_ver = "01.09"; // [extra: Teefpyre]
    extra_bbsc_setup_save_dir(); // [extra: Teefpyre]
    // BBSC writes its byte patches straight into the game's code, and turns some on and off
    // while the game runs (session teardown, the host's lamp): its code pages stay writable
    // (shadPS4 maps them so; the port maps them read and execute).
    unsigned writable = 0;
#ifdef _WIN32
    // Windows: the image's executable regions from VirtualQuery, made writable through the
    // runtime's mprotect (it keeps its own record of the guest's protections).
    for (unsigned long long at = base; at < base + size;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) || !info.RegionSize) {
            break;
        }
        const unsigned long long start = std::max<unsigned long long>(
            reinterpret_cast<unsigned long long>(info.BaseAddress), base);
        const unsigned long long end = std::min<unsigned long long>(
            reinterpret_cast<unsigned long long>(info.BaseAddress) + info.RegionSize, base + size);
        constexpr DWORD Executable =
            PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        if (info.State == MEM_COMMIT && (info.Protect & Executable) && start < end) {
            writable += mprotect(reinterpret_cast<void*>(start), end - start,
                                 PROT_READ | PROT_WRITE | PROT_EXEC) == 0;
            std::printf("BBSC: code %#llx-%#llx writable\n", start, end);
        }
        at = reinterpret_cast<unsigned long long>(info.BaseAddress) + info.RegionSize;
    }
#endif
    if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), maps)) {
            unsigned long long start = 0, end = 0;
            char perms[5] = {};
            if (std::sscanf(line, "%llx-%llx %4s", &start, &end, perms) != 3 || perms[2] != 'x') {
                continue;
            }
            start = std::max<unsigned long long>(start, base);
            end = std::min<unsigned long long>(end, base + size);
            if (start < end) {
                writable += mprotect(reinterpret_cast<void*>(start), end - start,
                                     PROT_READ | PROT_WRITE | PROT_EXEC) == 0;
                std::printf("BBSC: code %#llx-%#llx writable\n", start, end);
            }
        }
        std::fclose(maps);
    }
    std::printf("BBSC: seamless co-op hooks for the image at %#llx (%u code mappings writable)\n",
                static_cast<unsigned long long>(base), writable);
    if (const char* probe = std::getenv("BB_BBSC_PROBE")) {
        const unsigned long long at = std::strtoull(probe, nullptr, 0);
        if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
            char line[512];
            while (std::fgets(line, sizeof(line), maps)) {
                unsigned long long a = 0, b = 0;
                if (std::sscanf(line, "%llx-%llx", &a, &b) == 2 && at >= a && at < b) {
                    std::printf("BBSC probe: %s", line);
                }
            }
            std::fclose(maps);
        }
        std::fflush(stdout);
        std::printf("BBSC probe: byte %#x\n", unsigned(*reinterpret_cast<volatile u8*>(at)));
    }
    // The cinematic camera (F1, keyboard and mouse): BBSC sees the window's events first.
    if (Core::Bbsc::SwitchEnabled("SHADPS4_BBSC_CINEMATIC")) {
        bbgpu_add_event_filter([](const void* event, void* window) -> int {
            return Core::Bbsc::CinematicHandleEvent(*static_cast<const SDL_Event*>(event),
                                                    static_cast<SDL_Window*>(window));
        });
    }
    Core::Bloodborne::InstallSeamlessCoopPatches();
    Core::Bbsc::InstallPersistentSessionPatches();
    Core::Bloodborne::InstallReverseEngineeringTrace();
    Core::Bbsc::StartRelayClient(MemoryPatcher::g_game_serial);
}

namespace Libraries::Np::NpManager {
void InitializeRuntime();
}

BBNET_EXPORT void bbnet_start(void) {
    // The online session starts with the game (not at registration: the account and the
    // game's SDK version are known only now).
    if (BbThd::Enabled()) {
        BbThd::Initialize();
        return;
    }
    Libraries::Np::NpManager::InitializeRuntime();
}

BBNET_EXPORT int bbnet_init(const BbNetHost* host) {
    if (!host || host->abi != BBNET_ABI_VERSION || !host->thread_spawn || !host->thread_join ||
        !host->thread_exit || !host->memory_is_mapped) {
        std::printf("Online: the runtime and the online module are from different versions\n");
        return -1;
    }
    g_host = *host;
    Core::Emulator::SetGame(host->app0, host->serial, host->title, host->app_ver);
    // What BBSC reads of the game (param.sfo, the serial: shadPS4's PSF and memory patcher).
    {
        auto* psf = Common::Singleton<PSF>::Instance();
        psf->app_ver = host->app_ver ? host->app_ver : "";
        psf->title_id = host->serial ? host->serial : "";
        psf->title = host->title ? host->title : "";
        MemoryPatcher::g_game_serial = host->serial ? host->serial : "";
    }
    // BBSC patches the game's code, some of it while the game runs: from now on the port's own
    // code patches leave it writable (bbnet_image_ready).
    if (const char* bbsc = std::getenv("BB_BBSC"); bbsc && bbsc[0] == '1') {
        Core::KeepGuestCodeWritable();
    }
    const auto& ids = Common::ElfInfo::Instance().GetNpCommIds();
    std::string line = std::string("Co-op: ") + (host->serial ? host->serial : "?") +
                       ", NP communication id " +
                       (ids.empty() ? std::string("missing") : ids.front());
    if (EmulatorSettings.IsConnectedToNetwork()) {
        line += "; online play on"; // a launcher can look for this
    }
    Common::Log::WriteLine(stdout, line + "\n"); // one write (line.h)
    Core::Loader::SymbolsResolver resolver;
    Core::Loader::SymbolsResolver* sym = &resolver;
    if (BbThd::Enabled()) {
        // The Hunter's Dream: its NP and sceNet imports; the game's HTTP stays sceHttp's.
        BbThd::Register(sym);
        Libraries::Http::RegisterLib(sym);
        Libraries::Http2::RegisterLib(sym);
        Libraries::Ssl2::RegisterLib(sym);
    } else {
        Libraries::Net::RegisterLib(sym);
        Libraries::NetCtl::RegisterLib(sym);
        Libraries::Http::RegisterLib(sym);
        Libraries::Http2::RegisterLib(sym);
        // libSceSsl: not registered, as in shadPS4 (its loading is behind a title workaround).
        Libraries::Ssl2::RegisterLib(sym);
        Libraries::Np::NpAuth::RegisterLib(sym);
        Libraries::Np::NpCommon::RegisterLib(sym);
        Libraries::Np::NpManager::RegisterLib(sym);
        Libraries::Np::NpMatching2::RegisterLib(sym);
        Libraries::Np::NpPartner::RegisterLib(sym);
        Libraries::Np::NpParty::RegisterLib(sym);
        Libraries::Np::NpScore::RegisterLib(sym);
        Libraries::Np::NpSignaling::RegisterLib(sym);
        Libraries::Np::NpTus::RegisterLib(sym);
        Libraries::Np::NpUtility::RegisterLib(sym);
        Libraries::Np::NpWebApi::RegisterLib(sym);
        Libraries::Np::NpWebApi2::RegisterLib(sym);
    }
    std::printf("Online: %zu network and PSN functions (%s)\n", g_symbols.size(),
                BbThd::Enabled() ? "The Hunter's Dream" : "shadNet");
    return 0;
}

// The Hunter's Dream backend starts its dispatcher as a guest thread (thd_session.cpp).
extern "C" int32_t runtime_thread_spawn(void** out, void* PS4_SYSV_ABI (*entry)(void*),
                                        void* argument, uint64_t stack, const char* name) {
    return g_host.thread_spawn(out, entry, argument, stack, name);
}
