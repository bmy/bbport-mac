// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the native GPU process split (docs/macos-native-gpu.md), as the rest of the GPU library
// sees it. With the native GPU process (macOS, where bb-gpu is built; BB_NATIVE_GPU=0: off) the
// game process (bb-probe, x86-64) keeps the HLE libraries
// (Gnm, VideoOut, event queues) and forwards what reaches the GPU core to bb-gpu (arm64), which
// runs that core: the command processor, the caches, the shader recompiler, the presenter and the
// window.
// In one process (BB_NATIVE_GPU=0, Linux, a build without bb-gpu) both flags stay false and none
// of this runs.
#pragma once

#include <cstdint>
#include <span>

#include "../../bbgpu.h"
#include "common/types.h"

namespace BbVideoOut {
struct Port;
struct BufferAttribute;
struct FlipStatus;
struct VblankStatus;
} // namespace BbVideoOut

namespace BbRemote {

inline bool front_active = false; ///< this is the game process and the GPU core is in bb-gpu
inline bool back_active = false;  ///< this is bb-gpu

inline bool FrontActive() {
    return front_active;
}
inline bool BackActive() {
    return back_active;
}

/// The game process should start bb-gpu: on macOS where it is built, unless BB_NATIVE_GPU=0;
/// BB_NATIVE_GPU=1 anywhere.
bool Requested();

namespace Front {
/// bb-gpu exists where this process would start it (BB_GPU_PROCESS, else next to bb-probe).
bool Built();
/// Starts bb-gpu and initializes the GPU core there. False (reason printed) on failure.
bool Init(const BbGpuConfig& config);

/// False when host words were too many to send inline (dropped).
bool SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy = false);
void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);
void SubmitDone(u64 frame);
bool IsGpuIdle();

struct AscQueue {
    u64 map_addr;
    u32 ring_size_dw;
};
/// Slot index of the new compute queue, or -1.
s32 MapComputeQueue(u64 ring_base, u32* read_ptr, u32 ring_size_dw, u32 pipe_id);
void UnmapComputeQueue(u32 index);
AscQueue GetAscQueue(u32 index);

/// The game process's VideoOut port (the display's main_port): kept in step with bb-gpu's.
void SetVideoOutPort(BbVideoOut::Port* port);
u64* VideoOutLabels();
int VoOpen();
void VoClose(s32 handle);
int VoRegisterBuffers(s32 start, void* const* addresses, s32 count,
                      const BbVideoOut::BufferAttribute* attribute);
int VoUnregisterBuffers(s32 attribute_index);
int VoChangeBufferAttribute(s32 attribute_index,
                            const BbVideoOut::BufferAttribute* attribute);
bool VoSubmitFlip(s32 index, s64 flip_arg);
s32 VoSubmitEopFlip(s32 handle, u32 buf_id, u32 mode, s64 flip_arg);
void VoSetFlipRate(s32 handle, s32 rate);
void VoSetHdr(s32 handle, bool hdr);
bool VoIsHdrSupported();
void VoSetGamma(float gamma);

/// The loader's SIGSEGV/SIGBUS handler: 1 when the fault was GPU page tracking (handled).
int HandleFault(void* context, void* address);
/// CPU wrote guest memory outside page tracking (decoded video frames).
void InvalidateMemory(u64 address, u64 size);
/// Moves [address, address + size) (page-aligned, mapped) into shared memory with protection
/// `prot` (PROT_*), so bb-gpu sees it at the same address. 0 on success.
int ShareRange(void* address, u64 size, int prot);
/// The game's hooks' GPU ranges (Game::CoreServices), for bb-gpu's buffer cache.
void FreshRange(u64 address, u64 size);
void DeadRange(u64 address, u64 size);
void LiveRange(u64 address, u64 size);
/// bb-gpu's pad recording clock (bbgpu_world_active, bbgpu_world_us).
int WorldActive();
unsigned long long WorldUs();
/// A trophy's pop-up (BbOverlay::ShowTrophy): the overlay is bb-gpu's.
void ShowTrophy(char grade, const char* name, const char* icon, float delay);

int OverlayCapturesInput();
/// bb-gpu's window keyboard (bbgpu_keyboard_state).
const bool* KeyboardState();
/// Keys pressed in bb-gpu's window since the last call: latch[scancode] set to 1 (count entries).
void TakeKeyLatch(volatile std::uint8_t* latch, int count);
/// bb-gpu's window mouse (bbgpu_mouse_*).
void MouseTake(double* dx, double* dy, int* wheel_up, int* wheel_down);
int MouseCaptured();
u32 MouseButtons();
void MouseLookEnable(int enabled);
void MouseCapture();
int TextInputBegin(const char* initial, const char* prompt);
int TextInputPoll(char* out, u64 size);
} // namespace Front

namespace Back {
/// bb-gpu's entry (bb_gpu_main.cpp): never returns.
int Main(int argc, char** argv);

// Calls from the GPU core that the game process must hear of.
void FramesRetired(u64 frames);
void ReleaseSubmissionLock();
void Irq(u32 irq);
/// Called with the port's status_mutex held, after flip_status changed; `flip_event`: a flip
/// completed (the game process triggers its flip events with `flip_arg`).
void VoFlipStatus(const BbVideoOut::FlipStatus& status, bool flip_event, s64 flip_arg);
/// Called with vblank_mutex held after a vblank: `status` after it, `event_count` the events'
/// count.
void VoVblank(const BbVideoOut::VblankStatus& status, u64 event_count);
u64* VideoOutLabels();
/// From the window loop: state the game process reads directly (settings menu open...).
void PublishWindowState();
void RequestRestart();
/// The settings menu's effects (bit e: effects[e]) and frame rate limit, for the game's code.
void GameEffects(u64 mask);
void FrameLimit(int fps);
/// The output's aspect ratio for the game's camera and HUD hooks (BbGuestHooks::SetDisplayAspect).
void DisplayAspect(float aspect);
/// The window's monitor in pixels, for the game's System menu ("Auto" follows it).
void Monitor(int width, int height);
/// After the frame stats: the game process prints its threads' CPU time and the guest's waits.
void GameStats(double window_s, double frames);
// The game process's pad and files for the button prompts (runtime_pad_prompt*,
// runtime_pad_mouse_look, runtime_file_host_path there).
int PadPrompt(const char* name, int prefer_mouse, char* label, int size);
unsigned PadPromptGeneration();
int PadPromptDevice();
int PadMouseLook();
int FileHostPath(const char* guest, char* out, u64 size);
int PadSteamOnly();
/// The game's user folder (InitArgs::user_dir).
const char* UserDir();
/// The bytecode SRT walker's check (Shader::srt_guest_readable): guest memory only.
bool GuestReadable(u64 address, u64 size);

// The runtime's GPU interface (runtime_memory_*, clocks), served from bb-gpu's own view.
u64 MemoryClamp(uintptr_t address, u64 size);
int MemoryRegion(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
const u64* MemoryGeneration();
int MemoryVmaInfo(uintptr_t address, int* prot, int* type, uintptr_t* end);
int MemoryDirectPhys(uintptr_t address, u64* phys, uintptr_t* end);
int MemoryIsMapped(uintptr_t address, u64 size);
int MemoryWriteBacking(uintptr_t address, const void* data, u64 size);
void MemoryReadBacking(uintptr_t address, void* data, u64 size);
void MemoryGpuProtect(uintptr_t address, u64 size, int read, int write);
void MemoryTrap(uintptr_t address, u64 size, unsigned reason, int on);
unsigned MemoryTrapReasons(uintptr_t address);
using GpuRange = void (*)(uintptr_t address, u64 size);
void SetGpuHooks(GpuRange map, GpuRange unmap, GpuRange invalidate);
void SetNoteWriteHook(GpuRange hook);
void SetCpuWriteHook(GpuRange hook);
// The memory model's chunk allocator (bbport 0.7): BbGuestMemory::Install sets it here and the
// game process's runtime calls it over the channel (MsgChunkAlloc); its chunks are imports of this
// process's own view of the pool.
using ChunkAlloc = int (*)(u64 phys, u64 size);
using ChunkFree = void (*)(u64 phys, u64 size);
using ChunkImport = int (*)(u64 phys, u64 size, void* pointer);
void SetChunkAllocator(ChunkAlloc alloc);
void SetChunkFree(ChunkFree free_chunk);
void SetChunkPrivate(ChunkImport import);
void SetChunkWhole(int whole);
void SetGuestLazy(int lazy);
void* BackingPointer(u64 phys);
int PoolFd();
void NotePinned(u64 phys, u64 size);
void TrapStats(u64* wait_ns, u64* apply_ns, u64* calls, u64* bytes);
u64 ProcessTimeUs();
u64 ProcessTimeCounter();
u64 TscFrequency();
u64 ReadTsc();
} // namespace Back

} // namespace BbRemote
