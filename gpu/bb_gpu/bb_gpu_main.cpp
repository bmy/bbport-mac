// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: bb-gpu, the native GPU process (docs/macos-native-gpu.md). The GPU library imports
// the runtime's GPU interface from the executable that loads it (bb-probe in-process); here this
// executable provides it from bb-gpu's own view of the guest's memory (remote_back.cpp).
// What the library imports (build_gpu_native.sh checks it) is answered here locally, from
// bb-gpu's own state, or by the game process over the channel (remote_back.cpp); functions only
// the game's code hooks call (they run in the game process) are inert.
#include <csetjmp>
#include <cstddef>
#include <cstdint>

#include "remote/bb_remote.h"

namespace Back = BbRemote::Back;

extern "C" {
// Diagnostics switches (BB_TOGGLES), as runtime_memory.c.
uint64_t runtime_disabled_optimizations;
// Experiment bits: the second number in BB_TOGGLE_FILE (remote_back.cpp watches it).
uint64_t runtime_experiment_bits;
// Speculative guest reads (draw preparation): a fault there jumps back (remote_back.cpp).
__thread sigjmp_buf* runtime_fault_recover;
volatile int runtime_restarting;

int32_t* runtime_errno(void) {
    static thread_local int32_t value;
    return &value;
}
int runtime_file_translate(const char*, char*, size_t) {
    return -1; // the movie player runs in the game process
}
uint64_t runtime_heap_growths(void) {
    return 0;
}
void runtime_sleep_stats(uint64_t* calls, uint64_t* ns) {
    *calls = 0;
    *ns = 0;
}
void runtime_wait_report(double) {}
void runtime_thread_attach_host(const char*) {}
void runtime_restart(void) {
    Back::RequestRestart();
}

uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size) {
    return Back::MemoryClamp(address, size);
}
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped) {
    return Back::MemoryRegion(address, start, end, mapped);
}
const uint64_t* runtime_memory_generation(void) {
    return Back::MemoryGeneration();
}
int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end) {
    return Back::MemoryVmaInfo(address, prot, type, end);
}
int runtime_memory_direct_phys(uintptr_t address, uint64_t* phys, uintptr_t* end) {
    return Back::MemoryDirectPhys(address, phys, end);
}
int runtime_memory_is_mapped(uintptr_t address, uint64_t size) {
    return Back::MemoryIsMapped(address, size);
}
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size) {
    return Back::MemoryWriteBacking(address, data, size);
}
void runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size) {
    Back::MemoryReadBacking(address, data, size);
}
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write) {
    Back::MemoryGpuProtect(address, size, read, write);
}
void runtime_memory_trap(uintptr_t address, uint64_t size, unsigned reason, int on) {
    Back::MemoryTrap(address, size, reason, on);
}
unsigned runtime_memory_trap_reasons(uintptr_t address) {
    return Back::MemoryTrapReasons(address);
}
// bbport 0.7's memory model: BbGuestMemory::Install sets its chunk allocator here; the game
// process's runtime calls it (MsgChunkAlloc), and its chunks import bb-gpu's own view of the
// pool. Private regions are refused there (the game's private memory is not in the pool).
void* runtime_memory_backing_pointer(uint64_t phys) {
    return Back::BackingPointer(phys);
}
int runtime_memory_pool_fd(void) {
    return Back::PoolFd(); // the same pool, this process's descriptor
}
void runtime_memory_set_guest_chunk_allocator(int (*alloc)(uint64_t phys, uint64_t size)) {
    Back::SetChunkAllocator(alloc);
}
void runtime_memory_set_guest_chunk_free(void (*free_chunk)(uint64_t phys, uint64_t size)) {
    Back::SetChunkFree(free_chunk);
}
void runtime_memory_set_guest_chunk_private(int (*import)(uint64_t phys, uint64_t size,
                                                          void* pointer)) {
    Back::SetChunkPrivate(import);
}
void runtime_memory_set_guest_chunk_whole(int whole) {
    Back::SetChunkWhole(whole);
}
void runtime_memory_set_guest_lazy(int lazy) {
    Back::SetGuestLazy(lazy);
}
void runtime_memory_note_pinned(uint64_t phys, uint64_t size) {
    Back::NotePinned(phys, size); // the game process's runtime keeps pinned pages on release
}
// Only Windows gives pages back there; the game process's runtime discards its own.
void runtime_memory_discard_direct(uint64_t, uint64_t) {}
// The game's direct allocations are placed by the game process (its bbgpu_init sets this there).
void runtime_memory_set_host_only_allocations(uint64_t) {}
void runtime_memory_trap_stats(uint64_t* wait_ns, uint64_t* apply_ns, uint64_t* calls,
                               uint64_t* bytes) {
    Back::TrapStats(wait_ns, apply_ns, calls, bytes);
}
// The heap growing on demand (hooks_heap.cpp) is a hook in the game's code: the game process's.
void runtime_memory_set_grow(const uint64_t*, uint64_t) {}
int runtime_memory_deferred(uint64_t* phys, uint64_t* end) {
    *phys = *end = 0;
    return 0;
}
int runtime_memory_commit(uint64_t, uint64_t) {
    return -1;
}
int runtime_memory_decommit(uint64_t, uint64_t) {
    return -1;
}
void runtime_memory_commit_ahead(uint64_t, uint64_t) {}
// Memory near the game's code (bbport_map_near.h): for its hooks, the game process's.
void* runtime_low_map(size_t, int) {
    return nullptr;
}
// Guest functions by name (the PM4 self test): the game's code runs in the game process.
uintptr_t runtime_memory_resolve(const char*) {
    return 0;
}
void runtime_memory_set_gpu_hooks(Back::GpuRange map, Back::GpuRange unmap,
                                  Back::GpuRange invalidate) {
    Back::SetGpuHooks(map, unmap, invalidate);
}
void runtime_memory_set_note_write_hook(Back::GpuRange note) {
    Back::SetNoteWriteHook(note);
}
void runtime_memory_set_cpu_write_hook(Back::GpuRange hook) {
    Back::SetCpuWriteHook(hook);
}
// BB_GUEST_IN_PLACE's write watch (the PC memory model, off on macOS).
void runtime_memory_set_write_watch(uintptr_t, uint64_t, int) {}
// The game process's side (remote_front.cpp), linked into the same library: never called here.
int runtime_memory_host_pool(uint64_t, int*, uint64_t*, uint64_t*) {
    return -1;
}
void runtime_memory_set_mirror_hook(void (*)(int, uintptr_t, uint64_t, uint64_t, int, int, int,
                                             int)) {}
int runtime_memory_lock_held(void) {
    return 0;
}
int runtime_memory_exclude(uintptr_t, uintptr_t) {
    return -1;
}

// The guest code hooks' notes: the hooks run in the game process.
void runtime_memory_note_write(uintptr_t, uint64_t) {}
void runtime_re_watch(uint64_t) {}
// bbport 0.6: the render resource cache's backing is allocated by the game's code hooks, which run
// in the game process (its mapping reaches this process as any other).
int runtime_extra_direct_map(uint64_t, uint64_t, void** out) {
    if (out) {
        *out = nullptr;
    }
    return -1;
}
// The button prompts (bbport 0.6): the pad and the game's files are the game process's.
int runtime_pad_prompt(const char* name, int prefer_mouse, char* label, int size) {
    return Back::PadPrompt(name, prefer_mouse, label, size);
}
unsigned runtime_pad_prompt_generation(void) {
    return Back::PadPromptGeneration();
}
int runtime_pad_steam_only(void) {
    return Back::PadSteamOnly(); // window.cpp: keys and mouse a Steam pad echoes
}
int runtime_pad_prompt_device(void) {
    return Back::PadPromptDevice();
}
int runtime_pad_mouse_look(void) {
    return Back::PadMouseLook();
}
int runtime_file_host_path(const char* guest, char* out, size_t size) {
    return Back::FileHostPath(guest, out, size);
}
const char* runtime_file_user_dir(void) {
    return Back::UserDir();
}
// The save the game uses (the BBSC kit's restore, a hook in the game's code): the game process's.
int runtime_savedata_active_slot(char* dir, size_t size) {
    if (dir && size) {
        dir[0] = 0;
    }
    return -1;
}

uint64_t runtime_process_time_us(void) {
    return Back::ProcessTimeUs();
}
uint64_t runtime_process_time_counter(void) {
    return Back::ProcessTimeCounter();
}
uint64_t runtime_tsc_frequency(void) {
    return Back::TscFrequency();
}
}

int main(int argc, char** argv) {
    return Back::Main(argc, argv);
}
