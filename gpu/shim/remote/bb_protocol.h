// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: messages between the game process (bb-probe) and the native GPU process (bb-gpu),
// sent over bb_channel.h. Plain structs with fixed-size fields only: the two sides are built for
// different architectures (x86-64 and arm64, both LP64, little-endian), and every struct's size is
// checked below. Guest structures (VideoOut status, buffer attributes) travel as raw bytes of the
// sizes given here; remote_front.cpp and remote_back.cpp check those sizes against the real types.
// The interface table is in docs/macos-native-gpu.md.
#pragma once

#include <cstdint>

namespace BbRemote {

inline constexpr std::uint32_t ProtocolVersion = 2;

enum Msg : std::uint32_t {
    // ---- game process -> GPU process ----
    MsgHello = 1,       ///< call: HelloArgs -> HelloReply (ranges reserved, clocks agreed)
    MsgMapMemory = 2,   ///< MapArgs: a guest mapping to mirror (before the GPU hears of it)
    MsgUnmapMemory = 3, ///< RangeArgs
    MsgWriteFault = 4,  ///< call: FaultArgs -> FaultReply
    MsgShutdown = 5,    ///< call: no payload; the GPU process exits after answering
    MsgInit = 6,        ///< call: InitArgs -> ResultReply (window, Vulkan device, GPU core)
    MsgProtectMemory = 7, ///< MirrorProtectArgs: the guest changed a mapping's protection
    MsgInvalidate = 8,  ///< RangeArgs: Rasterizer::InvalidateMemory
    MsgNoteWrite = 9,   ///< RangeArgs: data written by a path the GPU side hears of (file reads)
    MsgCpuWrite = 10,   ///< RangeArgs: libc copies over watched pages

    MsgSubmitGfx = 20,  ///< SubmitGfxArgs, then the inline command words (host data)
    MsgSubmitAsc = 21,  ///< SubmitAscArgs
    MsgSubmitDone = 22, ///< ValueArgs: the frame (0: none)
    MsgIsGpuIdle = 23,  ///< call -> ResultReply
    MsgMapComputeQueue = 24,   ///< call: ComputeQueueArgs -> ResultReply (slot index, or -1)
    MsgUnmapComputeQueue = 25, ///< ValueArgs: slot index

    MsgVoOpen = 40,             ///< call -> VoReply (result: the handle)
    MsgVoClose = 41,            ///< call: VoArgs
    MsgVoRegisterBuffers = 42,  ///< call: VoRegisterArgs -> VoReply
    MsgVoUnregisterBuffers = 43, ///< call: VoArgs (value: attribute index) -> VoReply
    MsgVoChangeAttribute = 44,  ///< call: VoAttributeArgs -> VoReply
    MsgVoSubmitFlip = 45,       ///< call: VoFlipArgs -> VoReply (result: 1 queued, 0 queue full)
    MsgVoSubmitEopFlip = 46,    ///< call: VoFlipArgs -> VoReply (registered for the next flip IRQ)
    MsgVoSetFlipRate = 47,      ///< VoArgs
    MsgVoSetHdr = 48,           ///< VoArgs
    MsgVoIsHdrSupported = 49,   ///< call -> ResultReply
    MsgVoSetGamma = 50,         ///< GammaArgs

    MsgTextInputBegin = 60, ///< call: TextInputArgs -> ResultReply
    MsgTextInputPoll = 61,  ///< call -> TextInputReply

    // ---- GPU process -> game process ----
    MsgProtect = 100,       ///< call: ProtectArgs -> ResultReply (the game process applies it)
    MsgIrq = 101,           ///< ValueArgs: Platform::InterruptId
    MsgFramesRetired = 102, ///< ValueArgs (Libraries::GnmDriver::NoteFramesRetired)
    MsgReleaseSubmissionLock = 103,
    MsgVoFlipStatus = 104,  ///< VoStatusArgs: flip status changed (and maybe a flip event)
    MsgVoVblank = 105,      ///< VoVblankArgs
    MsgRestart = 106,       ///< the settings menu asked for a restart
    MsgGpuFailed = 107,     ///< TextArgs: the GPU process is about to exit
};

/// Sizes of the guest structures carried as raw bytes (checked against the real types).
inline constexpr std::uint32_t FlipStatusBytes = 64;      ///< Libraries::VideoOut::FlipStatus
inline constexpr std::uint32_t VblankStatusBytes = 40;    ///< SceVideoOutVblankStatus
inline constexpr std::uint32_t ResolutionBytes = 48;      ///< SceVideoOutResolutionStatus
inline constexpr std::uint32_t BufferAttributeBytes = 40; ///< BufferAttribute
inline constexpr std::uint32_t VideoOutBufferBytes = 24;  ///< VideoOutBuffer
inline constexpr std::uint32_t AttributeGroupBytes = 48;  ///< BufferAttributeGroup
inline constexpr std::uint32_t MaxDisplayBuffers = 16, MaxDisplayBufferGroups = 4;

struct HelloArgs {
    std::uint32_t protocol;
    std::uint32_t page_size; ///< the game process's view (4096 under Rosetta)
    std::uint64_t pool_size; ///< bytes of the shared pool (direct + flexible memory + host span)
    std::uint64_t guest_begin, guest_end; ///< the guest range to reserve
    std::uint64_t low_begin, low_end;     ///< the low region (game image, shared data segments)
    /// Clocks: the game process's process-time origin and a TSC sample, on CLOCK_MONOTONIC.
    std::uint64_t start_monotonic_ns;
    std::uint64_t tsc_sample, tsc_sample_monotonic_ns, tsc_frequency;
};
struct HelloReply {
    std::uint32_t protocol;
    std::uint32_t page_size; ///< the GPU process's (16384 on Apple Silicon)
    std::int32_t error;      ///< 0, or why the GPU process cannot run (reservation failed...)
    std::uint32_t pad;
};

enum MapKind : std::uint32_t { MapDirect = 1, MapFlexible = 2, MapShared = 3 };
enum MapFlags : std::uint32_t {
    MapUnshared = 1, ///< private memory in the game process (executable): zeros here
};
struct MapArgs {
    std::uint64_t address, size;
    std::uint64_t offset; ///< in the shared pool
    std::uint32_t kind;   ///< MapKind (MapShared: not part of the guest's mapping table)
    std::int32_t prot;    ///< the guest's protection bits
    std::int32_t type;    ///< direct memory type (-1: not direct memory)
    std::uint32_t flags;  ///< MapFlags
};
struct RangeArgs {
    std::uint64_t address, size;
};
struct MirrorProtectArgs {
    std::uint64_t address, size;
    std::int32_t prot, type; ///< type -1: unchanged
};
struct ProtectArgs {
    std::uint64_t address, size;
    std::uint32_t read, write;
};
struct ResultReply {
    std::int32_t result;
    std::uint32_t pad;
};
struct ValueArgs {
    std::uint64_t value;
};

struct FaultArgs {
    std::uint64_t address, rip;
    std::uint32_t is_write, pad;
};
/// The protections the GPU side decided while handling the fault: the faulting thread applies them
/// itself (it may hold the runtime's lock, which the game process's reader would wait for).
inline constexpr std::uint32_t MaxFaultProtects = 160;
struct FaultReply {
    std::int32_t handled;
    std::uint32_t count;
    ProtectArgs protects[MaxFaultProtects];
};

struct InitArgs {
    std::uint32_t sdk_version, psf_attributes;
    std::int32_t width, height;
    char title[128];
    char serial[32];
    char user_dir[1024];
};

struct SubmitGfxArgs {
    std::uint64_t dcb, ccb;             ///< guest addresses (0 when inline or empty)
    std::uint32_t dcb_dwords, ccb_dwords;
    std::uint32_t inline_dcb, inline_ccb; ///< 1: the words follow this struct (dcb first)
};
struct SubmitAscArgs {
    std::uint64_t address;
    std::uint32_t vqid, dwords;
};
struct ComputeQueueArgs {
    std::uint64_t ring_base, read_ptr;
    std::uint32_t ring_size_dw, pipe_id;
};

struct VoArgs {
    std::int32_t handle;
    std::int32_t value;
};
struct VoRegisterArgs {
    std::int32_t handle, start, count, pad;
    std::uint8_t attribute[BufferAttributeBytes];
    std::uint64_t addresses[MaxDisplayBuffers];
};
struct VoAttributeArgs {
    std::int32_t handle, index;
    std::uint8_t attribute[BufferAttributeBytes];
};
struct VoFlipArgs {
    std::int32_t handle, index;
    std::uint32_t mode, pad;
    std::int64_t flip_arg;
};
/// The GPU process's port after a call: the game process's copy follows it.
struct VoState {
    std::uint64_t version; ///< of flip_status (newer ones replace older ones)
    std::uint8_t flip_status[FlipStatusBytes];
    std::uint8_t slots[VideoOutBufferBytes * MaxDisplayBuffers];
    std::uint8_t groups[AttributeGroupBytes * MaxDisplayBufferGroups];
    std::uint8_t resolution[ResolutionBytes];
    std::uint32_t is_open, pad;
};
struct VoReply {
    std::int32_t result;
    std::uint32_t pad;
    VoState state;
};
struct VoStatusArgs {
    std::uint64_t version;
    std::uint8_t flip_status[FlipStatusBytes];
    std::int64_t flip_arg;
    std::uint32_t flip_event; ///< 1: a flip completed (trigger the flip events)
    std::uint32_t pad;
};
struct VoVblankArgs {
    std::uint8_t status[VblankStatusBytes]; ///< after the vblank
    std::uint64_t event_count;              ///< the count the events carry (before it)
};
struct GammaArgs {
    float gamma;
    std::uint32_t pad;
};
struct TextInputArgs {
    char initial[1024];
    char prompt[256];
};
struct TextInputReply {
    std::int32_t state;
    std::uint32_t pad;
    char text[2048];
};
struct TextArgs {
    char text[512];
};

static_assert(sizeof(HelloArgs) == 80 && sizeof(HelloReply) == 16 && sizeof(MapArgs) == 40 &&
              sizeof(RangeArgs) == 16 && sizeof(MirrorProtectArgs) == 24 &&
              sizeof(ProtectArgs) == 24 && sizeof(ResultReply) == 8 && sizeof(ValueArgs) == 8);
static_assert(sizeof(FaultArgs) == 24 && sizeof(FaultReply) == 8 + 24 * MaxFaultProtects &&
              sizeof(FaultReply) <= 4096);
static_assert(sizeof(InitArgs) == 16 + 128 + 32 + 1024 && sizeof(SubmitGfxArgs) == 32 &&
              sizeof(SubmitAscArgs) == 16 && sizeof(ComputeQueueArgs) == 24);
static_assert(sizeof(VoArgs) == 8 && sizeof(VoRegisterArgs) == 16 + 40 + 128 &&
              sizeof(VoAttributeArgs) == 48 && sizeof(VoFlipArgs) == 24 &&
              sizeof(VoState) == 8 + 64 + 384 + 192 + 48 + 8 && sizeof(VoReply) == 8 + 704 &&
              sizeof(VoStatusArgs) == 8 + 64 + 16 && sizeof(VoVblankArgs) == 48 &&
              sizeof(GammaArgs) == 8);
static_assert(sizeof(TextInputArgs) <= 4096 && sizeof(TextInputReply) <= 4096 &&
              sizeof(TextArgs) == 512);

} // namespace BbRemote
