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

inline constexpr std::uint32_t ProtocolVersion = 11;

enum Msg : std::uint32_t {
    // ---- game process -> GPU process ----
    MsgHello = 1,       ///< call: HelloArgs -> HelloReply (ranges reserved, clocks agreed)
    MsgMapMemory = 2,   ///< MapArgs: a guest mapping to mirror (before the GPU hears of it)
    MsgUnmapMemory = 3, ///< RangeArgs
    /// call: FaultArgs -> FaultReply; meanwhile MsgProtect asked of the faulting thread
    /// (Channel::Reply::Ask): it applies the fault's protections itself, as they are decided
    MsgWriteFault = 4,
    MsgShutdown = 5,    ///< call: no payload; the GPU process exits after answering
    MsgInit = 6,        ///< call: InitArgs -> ResultReply (window, Vulkan device, GPU core)
    MsgProtectMemory = 7, ///< MirrorProtectArgs: the guest changed a mapping's protection
    MsgInvalidate = 8,  ///< RangeArgs: Rasterizer::InvalidateMemory
    MsgNoteWrite = 9,   ///< RangeArgs: data written by a path the GPU side hears of (file reads)
    MsgCpuWrite = 10,   ///< RangeArgs: libc copies over watched pages
    MsgSettingsChanged = 11, ///< the game's System menu changed bbport.ini (BbSettings::ReloadLive)
    // bbport 0.7's memory model (the GPU uses the game's memory in place).
    MsgGuestMemory = 12, ///< call -> GuestMemoryReply: how bb-gpu's GPU takes the game's memory
    /// RangeArgs: the game's hooks (CoreServices::fresh_range); BB_REMOTE_FRESH_SYNC=1: a call on
    /// ControlBlock::fresh -> ResultReply, answered once bb-gpu's buffer cache has the range
    MsgFreshRange = 13,
    MsgDeadRange = 14,   ///< RangeArgs: CoreServices::dead_range
    MsgLiveRange = 15,   ///< RangeArgs: CoreServices::live_range
    MsgGpuRemap = 16,    ///< RangeArgs: the range's GPU memory changed (a heap piece's chunk)
    MsgShowTrophy = 17,  ///< TrophyArgs: a trophy unlocked (BbOverlay::ShowTrophy, bb-gpu's)

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

    // ---- game process -> GPU process, on ControlBlock::protect ----
    // The runtime's direct memory chunk allocator (runtime_memory_set_guest_chunk_allocator),
    // whose chunks bb-gpu's GPU imports from its own view of the pool. May come with the
    // runtime's lock held: served by a bb-gpu thread that never waits for the game process.
    MsgChunkAlloc = 70, ///< call: ChunkArgs -> ResultReply (-3: imported, -1: none)
    MsgChunkFree = 71,  ///< call: ChunkArgs -> ResultReply

    // ---- GPU process -> game process ----
    /// call: ProtectArgs -> ResultReply (the game process applies it), or asked during a
    /// MsgWriteFault (the faulting thread applies it)
    MsgProtect = 100,
    MsgIrq = 101,           ///< ValueArgs: BbIrq::Line
    MsgFramesRetired = 102, ///< ValueArgs (BbGnm::NoteFramesRetired)
    MsgReleaseSubmissionLock = 103,
    MsgVoFlipStatus = 104,  ///< VoStatusArgs: flip status changed (and maybe a flip event)
    MsgVoVblank = 105,      ///< VoVblankArgs
    MsgRestart = 106,       ///< the settings menu asked for a restart
    MsgGpuFailed = 107,     ///< TextArgs: the GPU process is about to exit
    MsgGameStats = 108,     ///< StatsArgs: frame stats were printed; print the game process's
    // bbport 0.6's button prompts (rare: when the prompts change).
    MsgPadPrompt = 110,     ///< call: PadPromptArgs -> PadPromptReply (runtime_pad_prompt)
    MsgFileHostPath = 111,  ///< call: PathArgs -> PathReply (runtime_file_host_path, /app0/ only)
    // bbport 0.6's live settings: the settings menu and F3 are bb-gpu's, the game's code is here.
    MsgGameEffects = 112,   ///< ValueArgs: bit e = BbSettings effects[e] (the hot ones switch now)
    MsgFrameLimit = 113,    ///< ValueArgs: BbFrameRate::SetLimit (0: none)
    MsgDisplayAspect = 114, ///< ValueArgs: float bits, BbGuestHooks::SetDisplayAspect (the output's)
    MsgMonitor = 115,       ///< ValueArgs: BbSettings::PackSize of the window's monitor (NoteMonitor)
    /// call on ControlBlock::protect: ChunkArgs -> ResultReply (runtime_memory_note_pinned)
    MsgNotePinned = 116,
    MsgGameSettings = 117,  ///< GameSettingsArgs: settings the game's code reads (the menu's)
};

/// Sizes of the guest structures carried as raw bytes (checked against the real types).
inline constexpr std::uint32_t FlipStatusBytes = 64;      ///< BbVideoOut::FlipStatus
inline constexpr std::uint32_t VblankStatusBytes = 40;    ///< BbVideoOut::VblankStatus
inline constexpr std::uint32_t ResolutionBytes = 48;      ///< BbVideoOut::ResolutionStatus
inline constexpr std::uint32_t BufferAttributeBytes = 40; ///< BbVideoOut::BufferAttribute
inline constexpr std::uint32_t VideoOutBufferBytes = 24;  ///< BbVideoOut::DisplayBuffer
inline constexpr std::uint32_t AttributeGroupBytes = 48;  ///< BbVideoOut::BufferAttributeGroup
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
inline constexpr std::uint32_t MaxTakenRanges = 32;
struct HelloReply {
    std::uint32_t protocol;
    std::uint32_t page_size; ///< the GPU process's (16384 on Apple Silicon)
    std::int32_t error;      ///< 0, or why the GPU process cannot run (reservation failed...)
    std::uint32_t taken_count;
    /// Parts of the guest range the GPU process's own memory occupies (its allocator sets them
    /// aside before it can reserve anything): the game's mappings must stay out of them.
    std::uint64_t taken[MaxTakenRanges][2];
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
/// ProtectArgs.read with this bit: a write trap (runtime_memory_trap) rather than a protection;
/// the rest of `read` is the trap's reason and `write` is 1 (on) or 0 (off).
inline constexpr std::uint32_t ProtectTrap = 0x80000000u;
struct ResultReply {
    std::int32_t result;
    std::uint32_t pad;
};
struct ValueArgs {
    std::uint64_t value;
};

struct FaultArgs {
    std::uint64_t address, rip;
    std::uint32_t is_write;
    std::uint32_t lock_held; ///< the faulting thread holds the runtime's lock (it faulted in it)
    std::uint32_t reasons;   ///< the game's write trap reasons for the page (runtime_memory.c)
    std::uint32_t pad;
};
/// bbport (protocol 11): the fault's protections are no longer in the reply. The faulting thread
/// applied each when bb-gpu's handler decided it (MsgProtect asked of it), with the runtime's
/// lock held or not: in the handlers' order, before they went on. Applied after the reply they
/// could land after newer ones the protection thread applied meanwhile, and the ones past the
/// reply's room were calls the protection thread could not serve while this thread held the lock.
struct FaultReply {
    std::int32_t handled;
    std::uint32_t pad;
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
    /// 1: the command processor copies the words at submit (bbport 0.6's internal FX
    /// initialization clears), instead of bb-gpu keeping inline words for good.
    std::uint32_t force_copy, reserved;
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
struct StatsArgs {
    double window_s; ///< seconds the frame stats covered
    double frames;   ///< frames in them
};
struct PadPromptArgs {
    char name[32]; ///< runtime_pad.c's input name
    std::int32_t prefer_mouse;
    std::uint32_t pad;
};
struct PadPromptReply {
    std::int32_t kind; ///< runtime_pad_prompt's result
    std::uint32_t pad;
    char label[64];
};
struct PathArgs {
    char path[1024];
};
struct PathReply {
    std::int32_t result; ///< 0: found
    std::uint32_t pad;
    char path[1024];
};
struct ChunkArgs {
    std::uint64_t phys, size; ///< direct memory: offsets in the shared pool
};
/// What bb-gpu's BbGuestMemory::Install set (its runtime_memory_set_guest_chunk_* calls).
struct GuestMemoryReply {
    std::uint32_t allocator;       ///< 1: a chunk allocator is set
    std::uint32_t host_import;     ///< 1: its chunks are host imports of the pool (the only kind
                                   ///< the game process can map: dma-buf chunks are bb-gpu's)
    std::uint32_t whole, lazy;     ///< runtime_memory_set_guest_chunk_whole, _lazy
    std::uint32_t free_chunks;     ///< 1: runtime_memory_set_guest_chunk_free was called
    std::uint32_t private_regions; ///< 1: runtime_memory_set_guest_chunk_private was called
};
struct TrophyArgs {
    char name[256];
    char icon[1024]; ///< the trophy's TROPnnn.PNG on this computer, or empty
    float delay;     ///< seconds before it shows
    char grade;      ///< B, S, G, P
    std::uint8_t pad[3];
};
struct GameSettingsArgs {
    float camera_speed[4];       ///< BbSettings::camera_speed
    std::uint32_t light_grid_opt;
    std::uint32_t pad;
};

static_assert(sizeof(HelloArgs) == 80 && sizeof(HelloReply) == 16 + 16 * MaxTakenRanges &&
              sizeof(MapArgs) == 40 &&
              sizeof(RangeArgs) == 16 && sizeof(MirrorProtectArgs) == 24 &&
              sizeof(ProtectArgs) == 24 && sizeof(ResultReply) == 8 && sizeof(ValueArgs) == 8);
static_assert(sizeof(FaultArgs) == 32 && sizeof(FaultReply) == 8);
static_assert(sizeof(InitArgs) == 16 + 128 + 32 + 1024 && sizeof(SubmitGfxArgs) == 40 &&
              sizeof(SubmitAscArgs) == 16 && sizeof(ComputeQueueArgs) == 24);
static_assert(sizeof(VoArgs) == 8 && sizeof(VoRegisterArgs) == 16 + 40 + 128 &&
              sizeof(VoAttributeArgs) == 48 && sizeof(VoFlipArgs) == 24 &&
              sizeof(VoState) == 8 + 64 + 384 + 192 + 48 + 8 && sizeof(VoReply) == 8 + 704 &&
              sizeof(VoStatusArgs) == 8 + 64 + 16 && sizeof(VoVblankArgs) == 48 &&
              sizeof(GammaArgs) == 8);
static_assert(sizeof(TextInputArgs) <= 4096 && sizeof(TextInputReply) <= 4096 &&
              sizeof(TextArgs) == 512 && sizeof(StatsArgs) == 16);
static_assert(sizeof(PadPromptArgs) == 40 && sizeof(PadPromptReply) == 72 &&
              sizeof(PathArgs) == 1024 && sizeof(PathReply) == 1032);
static_assert(sizeof(TrophyArgs) == 1288 && sizeof(ChunkArgs) == 16 && sizeof(GuestMemoryReply) == 24 &&
              sizeof(GameSettingsArgs) == 24);

} // namespace BbRemote
