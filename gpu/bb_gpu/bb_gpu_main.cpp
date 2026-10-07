// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: bb-gpu, the native GPU process (docs/macos-native-gpu.md). The GPU library imports
// the runtime's GPU interface from the executable that loads it (bb-probe in-process); here this
// executable provides it from bb-gpu's own view of the guest's memory (remote_back.cpp).
#include <csetjmp>
#include <cstddef>
#include <cstdint>

#include "remote/bb_remote.h"

namespace Back = BbRemote::Back;

extern "C" {
// Diagnostics switches (BB_TOGGLES), as runtime_memory.c.
uint64_t runtime_disabled_optimizations;
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
