// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

// Returns $BB_GPU_USER_DIR or $BB_USER_DIR when set and non-empty, else nullptr.
extern "C" const char* extra_bbsc_save_root(void);

// BBSC save directory automatic resolution across title IDs (CUSA00207, CUSA03173, etc.)
// [Teefpyre patch: dynamic save folder detection]
extern "C" void extra_bbsc_setup_save_dir(void);

// Hunter's Kit fix: the bell's param rows (goods + SpEffect) are scannable once the game
// data is mapped, but upstream's cap of 36 attempts (3 min) could expire if the user stayed
// at the title screen. This worker calls prepare every 2 s until ready turns true.
extern "C" void extra_bbsc_bell_scan_forever(std::atomic<bool>* ready, void (*prepare)(void));

// Safely reads memory even on DMABUF or protected pages, preventing SIGSEGV/SIGBUS.
extern "C" bool extra_safe_read(void* dst, const void* src, size_t size);

// Verifies if a memory mapping should be scanned (skips read-only and host .so libraries).
extern "C" bool extra_bbsc_check_mapping(const char* param, const char* line);

// Memory scan for BBSC param tables and symbols [Teefpyre patch / Linux BBSC Fix]:
// Handles low mappings (< GuestMin) via /proc/self/maps and guest mappings (>= GuestMin)
// via runtime_memory_read_backing.
extern "C" void extra_bbsc_find_in_memory(
    const std::uint8_t* needle_ptr, std::size_t needle_len,
    std::uint64_t lo, std::uint64_t hi, std::size_t max_hits,
    std::vector<std::uint64_t>* hits);

// Memory access callbacks used by BBSC parameter operations
struct BbscParamMemory {
    bool (*readable)(std::uint64_t addr, std::size_t size);
    std::uint64_t (*read_u64)(std::uint64_t addr, std::size_t offset);
    std::int32_t (*read_s32)(std::uint64_t addr, std::size_t offset);
    std::uint16_t (*read_u16)(std::uint64_t addr, std::size_t offset);
    std::uint32_t (*read_u32)(std::uint64_t addr, std::size_t offset);
    bool (*match_name)(std::uint64_t table_addr, const char* name, std::size_t name_len);
};

// Returns the SoloParamRepository pointer if initialized and valid, else 0.
// Checks image_base + 0x05540340 and [[image_base + 0x05537F58] + 0x68].
// [Teefpyre patch: direct SoloParamRepository lookup without memory scanning]
extern "C" std::uint64_t extra_bbsc_get_solo_param_repo(
    std::uint64_t image_base, const BbscParamMemory* mem);

// Finds param tables directly from SoloParamRepository without memory scanning.
extern "C" void extra_bbsc_find_param_tables_direct(
    std::uint64_t repo, const char* type, std::size_t type_len, std::uint16_t min_rows, std::uint32_t first_id,
    const BbscParamMemory* mem, std::vector<std::uint64_t>* out_tables);