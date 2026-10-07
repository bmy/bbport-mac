// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: messages between the game process and the native GPU process (bb_channel.h). Plain
// structs with fixed-size fields only: the two sides are built for different architectures
// (x86-64 and arm64, both LP64, little-endian), and every struct's size is checked below.
// The interface table is in docs/macos-native-gpu.md; types are added as phases land.
#pragma once

#include <cstdint>

namespace BbRemote {

inline constexpr std::uint32_t ProtocolVersion = 1;

enum Msg : std::uint32_t {
    // Game process -> GPU process.
    MsgHello = 1,       ///< call: HelloArgs -> HelloReply
    MsgMapMemory = 2,   ///< MapArgs: mirror a guest mapping of the shared pool
    MsgUnmapMemory = 3, ///< RangeArgs
    MsgWriteFault = 4,  ///< call: RangeArgs (the faulting page) -> ResultReply
    MsgShutdown = 5,    ///< call: no payload; the GPU process exits after answering

    // GPU process -> game process.
    MsgProtect = 100, ///< call: ProtectArgs -> ResultReply (the game process applies it)
};

struct HelloArgs {
    std::uint32_t protocol;
    std::uint32_t page_size; ///< the game process's view (4096 under Rosetta)
    std::uint64_t pool_size; ///< bytes of the shared pool (direct + flexible memory)
    std::uint64_t guest_begin, guest_end; ///< the range to reserve
};
struct HelloReply {
    std::uint32_t protocol;
    std::uint32_t page_size; ///< the GPU process's (16384 on Apple Silicon)
    std::int32_t error;      ///< 0, or why the GPU process cannot run (reservation failed...)
    std::uint32_t pad;
};
struct MapArgs {
    std::uint64_t address, size;
    std::uint64_t offset; ///< in the shared pool
};
struct RangeArgs {
    std::uint64_t address, size;
};
struct ProtectArgs {
    std::uint64_t address, size;
    std::uint32_t read, write;
};
struct ResultReply {
    std::int32_t result;
    std::uint32_t pad;
};

static_assert(sizeof(HelloArgs) == 32 && sizeof(HelloReply) == 16 && sizeof(MapArgs) == 24 &&
              sizeof(RangeArgs) == 16 && sizeof(ProtectArgs) == 24 && sizeof(ResultReply) == 8);

} // namespace BbRemote
