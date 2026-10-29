// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne (1.09): BBSC's hunter's kit undone when BBSC is off (bloodborne_kit_restore.h).
//
// Reverse engineering: BBSC's (gpu/bbnet20/bbsc/core/bloodborne_re.cpp, "The BBSC hunter's kit"),
// addresses are offsets in the 1.09 executable, image at 0x800000000:
//  - 0x1308dc0 IsValidSos.Update: called every frame in the game; BBSC's game tick.
//  - [0x553e878] + 0x60: the local player character (+0x410 readable while it exists); its HP at
//    [[chr + 0x3b0] + 0x20] + 0xf8. [0x5556678]: the global state, +0x08 nonzero while loading.
//  - The carried items: [[0x553b130] + 0x08] + 0x328. The storage box: [game data + 0x5b0], the
//    game data from the player's vtable +0x1c8 (chr).
//  - An inventory: hash [+0x68][code % 0x7e1] (s16, < 0: none) into entries [+0x78] (8 bytes:
//    u32 code, u32 slot in bits 0-11 / next + 1 in bits 12-23); slot entries (16 bytes: +0 item
//    key, +8 count) at [+0x58] + slot * 16 below [+0x24], else [+0x48] + (slot - [+0x24]) * 16.
//    Goods have the code 0x40000000 | id and the key 0x80000000 | 3 << 28 | id.
//  - 0x14d9a80 (inventory, &code, count): how many fit. 0x14da0a0 (inventory, &key, count): adds.
//    0x14d94a0 (inventory, slot, delta, &over) changes a count; at 0 left 0x14dab20 (inventory,
//    slot, 1) frees the slot.
//  - Event flags: 0x1399bf0 (0, flag) reads one, 0x132e5a0 (0, flag, value) sets it; both need
//    SprjEventFlagMan ([0x553b100]) to exist.
#include "bloodborne_kit_restore.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "bbgpu.h"
#include "bbport_guest_code_hooks.h"
#include "bbport_toggles.h"
#include "common/types.h"

extern "C" const char* runtime_file_user_dir(void);
extern "C" int runtime_savedata_active_slot(char* dir, size_t size);

namespace BbKitRestore {
namespace {

using Addr = std::uintptr_t;

constexpr Addr ImageBase = 0x800000000ull;
constexpr Addr CodeEnd = 0x50d96dc;

constexpr Addr GameTick = 0x1308dc0;
constexpr std::uint8_t GameTickEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp...
constexpr Addr LocalState = 0x553e878, GlobalState = 0x5556678, CarriedHolder = 0x553b130;
constexpr Addr EventFlagMan = 0x553b100;
constexpr Addr FitsFn = 0x14d9a80, AddFn = 0x14da0a0, ChangeFn = 0x14d94a0, FreeFn = 0x14dab20;
constexpr Addr GetFlagFn = 0x1399bf0, SetFlagFn = 0x132e5a0;

/// BBSC's kit (BbscKit) and the vanilla item each stands for (0: none, the Duel Bell).
struct KitItem {
    std::uint32_t id, vanilla;
};
constexpr KitItem Kit[] = {
    {4016, 200},  // Hunt Bell -> Beckoning Bell
    {4121, 205},  // Kin Bell -> Small Resonant Bell
    {11201, 225}, // Bloodlust Bell -> Sinister Resonant Bell
    {4313, 0},    // Duel Bell
    {11251, 111}, // Parting Blank -> Silencing Blank
};

/// A bell's ringing form (BBSC takes those too) is given back as the bell.
std::uint32_t BaseGoods(std::uint32_t id) {
    return id == 201 || id == 206 || id == 226 ? id - 1 : id;
}

/// The bells are single items: given only when the character has none.
bool IsBell(std::uint32_t id) {
    return id == 200 || id == 205 || id == 225;
}

Addr At(Addr offset) {
    return ImageBase + offset;
}

/// Copies guest memory; false where nothing is mapped (the runtime's fault recovery).
bool Read(Addr address, void* out, std::size_t size) {
    if (address < 0x10000) {
        return false;
    }
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        runtime_fault_recover = nullptr;
        return false;
    }
    runtime_fault_recover = &recover;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    runtime_fault_recover = nullptr;
    return true;
}

template <typename T>
T Get(Addr address, T fallback = {}) {
    T value = fallback;
    return Read(address, &value, sizeof(value)) ? value : fallback;
}

bool Readable(Addr address) {
    std::uint8_t byte;
    return Read(address, &byte, 1);
}

// ---- The character and its inventories ----------------------------------------------------------

Addr Player() {
    const Addr state = Get<Addr>(At(LocalState));
    const Addr chr = state ? Get<Addr>(state + 0x60) : 0;
    return chr && Readable(chr + 0x410) ? chr : 0;
}

/// In the game, alive and not loading: the inventories are the character's.
bool Playing(Addr chr) {
    const Addr modules = Get<Addr>(chr + 0x3b0);
    const Addr data = modules ? Get<Addr>(modules + 0x20) : 0;
    const Addr global = Get<Addr>(At(GlobalState));
    return data && Get<std::int32_t>(data + 0xf8) > 0 && global && Readable(global + 0x1598) &&
           Get<std::uint8_t>(global + 0x08) == 0;
}

Addr Carried() {
    const Addr holder = Get<Addr>(At(CarriedHolder));
    const Addr data = holder ? Get<Addr>(holder + 0x08) : 0;
    return data && Readable(data + 0x328 + 0x90) ? data + 0x328 : 0;
}

Addr Storage(Addr chr) {
    const Addr vtable = Get<Addr>(chr);
    const Addr getter = vtable ? Get<Addr>(vtable + 0x1c8) : 0;
    if (getter < At(0) || getter >= At(CodeEnd)) {
        return 0;
    }
    using GameData = Addr PS4_SYSV_ABI (*)(Addr chr);
    const Addr data = reinterpret_cast<GameData>(getter)(chr);
    const Addr inventory = data ? Get<Addr>(data + 0x5b0) : 0;
    return inventory && Readable(inventory + 0x90) ? inventory : 0;
}

/// The slot entry (+0 key, +8 count) holding goods `id`, or 0.
Addr SlotEntry(Addr inventory, std::uint32_t id, std::uint32_t* slot_out = nullptr) {
    const std::uint32_t code = 0x40000000u | id;
    const Addr hashes = Get<Addr>(inventory + 0x68);
    const Addr entries = Get<Addr>(inventory + 0x78);
    if (!hashes || !entries) {
        return 0;
    }
    std::int64_t index = Get<std::int16_t>(hashes + (code % 0x7e1) * 2, -1);
    for (int guard = 0; index >= 0 && guard < 4096; ++guard) {
        const Addr entry = entries + static_cast<Addr>(index) * 8;
        const std::uint32_t info = Get<std::uint32_t>(entry + 4);
        if (Get<std::uint32_t>(entry) == code) {
            const std::uint32_t slot = info & 0xfff;
            const std::uint32_t split = Get<std::uint32_t>(inventory + 0x24);
            const Addr at = slot < split ? Get<Addr>(inventory + 0x58) + slot * 16ull
                                         : Get<Addr>(inventory + 0x48) + (slot - split) * 16ull;
            if (Get<std::uint32_t>(at) == 0) {
                return 0;
            }
            if (slot_out) {
                *slot_out = slot;
            }
            return at;
        }
        const std::uint32_t next = (((info >> 12) & 0xfff) + 0xffff) & 0xffff; // next + 1; 0: end
        if (next & 0x8000) {
            return 0;
        }
        index = next;
    }
    return 0;
}

std::int32_t Count(Addr inventory, std::uint32_t id) {
    const Addr entry = inventory ? SlotEntry(inventory, id) : 0;
    return entry ? std::max(Get<std::int32_t>(entry + 8), 1) : 0;
}

/// Adds `count` of goods `id`, one at a time as BBSC does; how many went in.
std::int32_t Add(Addr inventory, std::uint32_t id, std::int32_t count) {
    using Fits = std::int32_t PS4_SYSV_ABI (*)(Addr inventory, const std::uint32_t* code,
                                               std::int32_t count);
    using AddItem = std::int32_t PS4_SYSV_ABI (*)(Addr inventory, const std::uint32_t* key,
                                                  std::int32_t count);
    alignas(16) static std::uint32_t code = 0;
    alignas(16) static std::uint32_t key = 0;
    std::int32_t added = 0;
    for (std::int32_t i = 0; i < count; ++i) {
        code = 0x40000000u | id;
        key = 0x80000000u | (3u << 28) | id;
        const std::int32_t before = Count(inventory, id);
        if (reinterpret_cast<Fits>(At(FitsFn))(inventory, &code, 1) <= 0) {
            break;
        }
        reinterpret_cast<AddItem>(At(AddFn))(inventory, &key, 1);
        if (Count(inventory, id) <= before) {
            break;
        }
        ++added;
    }
    return added;
}

/// Takes every one of goods `id` out; true when none is left.
bool Remove(Addr inventory, std::uint32_t id) {
    std::uint32_t slot = 0;
    const Addr entry = SlotEntry(inventory, id, &slot);
    if (!entry) {
        return true;
    }
    using Change = std::int32_t PS4_SYSV_ABI (*)(Addr inventory, std::uint32_t slot,
                                                 std::int32_t delta, std::int32_t* over);
    using Free = void PS4_SYSV_ABI (*)(Addr inventory, std::uint32_t slot, std::int32_t one);
    alignas(16) static std::int32_t over = 0;
    over = 0;
    const std::int32_t count = std::max(Get<std::int32_t>(entry + 8), 1);
    const std::int32_t left = reinterpret_cast<Change>(At(ChangeFn))(inventory, slot, -count, &over);
    if (over == 0 && left == 0) {
        reinterpret_cast<Free>(At(FreeFn))(inventory, slot, 1);
    }
    return SlotEntry(inventory, id) == 0;
}

bool EventFlag(std::uint32_t flag) {
    if (Get<Addr>(At(EventFlagMan)) == 0) {
        return false;
    }
    using GetFlag = bool PS4_SYSV_ABI (*)(std::uint64_t unused, std::uint32_t flag);
    return reinterpret_cast<GetFlag>(At(GetFlagFn))(0, flag);
}

void ClearEventFlag(std::uint32_t flag) {
    if (Get<Addr>(At(EventFlagMan)) == 0) {
        return;
    }
    using SetFlag = std::uint64_t PS4_SYSV_ABI (*)(std::uint64_t unused, std::uint32_t flag,
                                                   std::uint8_t value);
    reinterpret_cast<SetFlag>(At(SetFlagFn))(0, flag, 0);
}

// ---- The journal ---------------------------------------------------------------------------------

std::mutex journal_mutex;

/// The character save the game uses, and its journal; false before the game opened one.
bool ActiveSave(std::filesystem::path* save_dir, std::filesystem::path* journal) {
    char dir[1024];
    const int slot = runtime_savedata_active_slot(dir, sizeof(dir));
    if (slot < 0 || !dir[0]) {
        return false;
    }
    // <user>/savedata/<user id>/CUSA00207/SPRJ0005: the user id names the journal with the slot.
    const std::filesystem::path folder{dir};
    std::string user = folder.parent_path().parent_path().filename().string();
    if (user.empty()) {
        user = "save";
    }
    if (save_dir) {
        *save_dir = folder;
    }
    if (journal) {
        *journal = std::filesystem::path{runtime_file_user_dir()} / "bbsc" / "kit-journal" /
                   (user + "-" + std::to_string(slot) + ".txt");
    }
    return true;
}

bool AppendJournal(const std::string& line) {
    std::filesystem::path save_dir, journal;
    if (!ActiveSave(&save_dir, &journal)) {
        std::printf("BBSC kit journal: no character save open yet - not recorded: %s\n", line.c_str());
        return false;
    }
    std::scoped_lock lock{journal_mutex};
    std::error_code ec;
    std::filesystem::create_directories(journal.parent_path(), ec);
    const bool fresh = !std::filesystem::exists(journal, ec);
    std::ofstream out{journal, std::ios::app};
    if (fresh) {
        out << "# What BBSC took out of the character in " << save_dir.string()
            << " (bbport: given back when BBSC is off)\n";
    }
    out << line << '\n';
    out.flush();
    if (!out) {
        std::printf("BBSC kit journal: could not write %s\n", journal.string().c_str());
        return false;
    }
    return true;
}

/// What goes back: goods, count, into the storage box.
struct Give {
    std::uint32_t id;
    std::int32_t count;
    bool storage;
};

/// The journal's items (bells once, the rest summed per place) and flags; false without one.
bool LoadJournal(const std::filesystem::path& journal, std::vector<Give>& gives,
                 std::vector<std::uint32_t>& flags) {
    std::ifstream in{journal};
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream words{line};
        std::string kind;
        words >> kind;
        if (kind == "item") {
            std::uint32_t id = 0;
            std::int32_t count = 0;
            std::string place;
            if (!(words >> id >> count >> place) || count <= 0) {
                continue;
            }
            id = BaseGoods(id);
            const bool storage = place == "storage";
            const auto same = std::ranges::find_if(gives, [&](const Give& g) {
                return g.id == id && (IsBell(id) || g.storage == storage);
            });
            if (same == gives.end()) {
                gives.push_back({id, IsBell(id) ? 1 : count, storage});
            } else if (!IsBell(id)) {
                same->count += count;
            }
        } else if (kind == "flag") {
            std::uint32_t flag = 0;
            if (words >> flag) {
                flags.push_back(flag);
            }
        }
    }
    return true;
}

bool BackupSave(const std::filesystem::path& save_dir) {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    const fs::path target = fs::path{runtime_file_user_dir()} / "save-backups" /
                            ("before-kit-restore-" + std::to_string(stamp)) / save_dir.filename();
    std::error_code ec;
    fs::create_directories(target.parent_path(), ec);
    fs::copy(save_dir, target, fs::copy_options::recursive, ec);
    if (ec) {
        std::printf("BBSC kit restore: could not back up %s (%s) - nothing changed\n",
                    save_dir.string().c_str(), ec.message().c_str());
        return false;
    }
    std::printf("BBSC kit restore: save backed up %s -> %s\n", save_dir.string().c_str(),
                target.string().c_str());
    return true;
}

// ---- The restore ---------------------------------------------------------------------------------

Addr checked_player = 0;
std::chrono::steady_clock::time_point next_check{}, banner_until{};
bool banner_shown = false;

void ShowBanner(const char* text) {
    bbgpu_overlay_banner(text);
    banner_shown = true;
    banner_until = std::chrono::steady_clock::now() + std::chrono::seconds(8);
}

/// Once per loaded character: kit out, vanilla items in (given first: an interruption leaves an
/// extra item, never a missing one).
void Restore(Addr chr) {
    const Addr carried = Carried();
    const Addr storage = Storage(chr);
    if (!carried || !storage) {
        return; // not ready yet: tried again
    }
    checked_player = chr;
    std::vector<std::uint32_t> present;
    for (const KitItem& item : Kit) {
        if (Count(carried, item.id) || Count(storage, item.id)) {
            present.push_back(item.id);
        }
    }
    if (present.empty()) {
        return;
    }
    std::filesystem::path save_dir, journal;
    if (!ActiveSave(&save_dir, &journal)) {
        checked_player = 0; // the save is opened right after the load: tried again
        return;
    }
    std::vector<Give> gives;
    std::vector<std::uint32_t> flags;
    const bool from_journal = LoadJournal(journal, gives, flags);
    if (!from_journal) {
        // A kit given before the journal existed: one vanilla item for each kit item.
        for (const KitItem& item : Kit) {
            if (item.vanilla && std::ranges::find(present, item.id) != present.end()) {
                gives.push_back({item.vanilla, 1, false});
            }
        }
    }
    std::printf("BBSC kit restore: BBSC is off and %s has %zu kit item(s); giving back %zu vanilla "
                "item(s) from %s\n",
                save_dir.string().c_str(), present.size(), gives.size(),
                from_journal ? journal.string().c_str() : "the kit (no journal)");
    if (!BackupSave(save_dir)) {
        return;
    }
    bool complete = true;
    for (const Give& give : gives) {
        if (IsBell(give.id) && (Count(carried, give.id) || Count(storage, give.id))) {
            std::printf("BBSC kit restore: goods %u already there\n", give.id);
            continue;
        }
        const Addr first = give.storage ? storage : carried;
        const Addr second = give.storage ? carried : storage;
        std::int32_t added = Add(first, give.id, give.count);
        if (added < give.count) {
            added += Add(second, give.id, give.count - added);
        }
        std::printf("BBSC kit restore: goods %u x%d given back (%d went in)\n", give.id, give.count,
                    added);
        complete = complete && added == give.count;
    }
    if (!complete) {
        // The kit stays: nothing is lost, and the next load tries again (the bells are not given
        // twice).
        std::printf("BBSC kit restore: not everything fit - the kit stays for now\n");
        ShowBanner("BBSC is off: some original items did not fit - make room and load again");
        return;
    }
    for (const std::uint32_t id : present) {
        const bool from_carried = Remove(carried, id);
        const bool gone = Remove(storage, id) && from_carried;
        std::printf("BBSC kit restore: kit goods %u taken out (%s)\n", id,
                    gone ? "gone" : "STILL THERE");
    }
    for (const std::uint32_t flag : flags) {
        if (EventFlag(flag)) {
            ClearEventFlag(flag);
            std::printf("BBSC kit restore: event flag %u cleared (BBSC had set it)\n", flag);
        }
    }
    if (from_journal) {
        std::error_code ec;
        const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
        std::filesystem::path done = journal;
        done += ".restored-" + std::to_string(stamp);
        std::filesystem::rename(journal, done, ec);
    }
    ShowBanner("BBSC is off: your original bells are back in your items");
}

void PS4_SYSV_ABI OnGameTick(std::uint64_t, const Core::GuestRegisterSnapshot*) {
    const auto now = std::chrono::steady_clock::now();
    if (banner_shown && now >= banner_until) {
        bbgpu_overlay_banner(nullptr);
        banner_shown = false;
    }
    if (now < next_check) {
        return;
    }
    next_check = now + std::chrono::seconds(2);
    const Addr chr = Player();
    if (!chr || chr == checked_player || !Playing(chr)) {
        return;
    }
    Restore(chr);
}

} // namespace

void Install() {
    if (bb_env_off("BB_KIT_RESTORE")) {
        return;
    }
    if (bb_env_on("BB_ONLINE") && bb_env_on("BB_BBSC")) {
        return; // BBSC runs and hooks the same tick
    }
    if (!Core::InstallGuestCodeHook(reinterpret_cast<void*>(At(GameTick)), GameTickEntry, 0,
                                    OnGameTick)) {
        std::printf("BBSC kit restore: the game's tick differs (another build): off\n");
        return;
    }
    std::printf("BBSC kit restore: on (BBSC is off: its kit goes back to the vanilla bells)\n");
}

} // namespace BbKitRestore

extern "C" void bbport_kit_journal_reset(void) {
    std::filesystem::path journal;
    if (!BbKitRestore::ActiveSave(nullptr, &journal)) {
        return;
    }
    std::error_code ec;
    if (std::filesystem::remove(journal, ec)) {
        std::printf("BBSC kit journal: a new kit - the old journal %s is dropped\n", journal.string().c_str());
    }
}

extern "C" void bbport_kit_journal_item(std::uint32_t goods, std::int32_t count, int storage) {
    BbKitRestore::AppendJournal("item " + std::to_string(goods) + ' ' + std::to_string(count) + ' ' +
                                (storage ? "storage" : "carried"));
}

extern "C" void bbport_kit_journal_flag(std::uint32_t flag) {
    BbKitRestore::AppendJournal("flag " + std::to_string(flag));
}
