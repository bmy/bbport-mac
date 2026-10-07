// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the GPU process channel (gpu/shim/remote/bb_channel.h) between two real processes:
// ordering, ring wrap-around, synchronous calls both ways (a call made from inside a handler,
// as bb-gpu does for page protection while answering a write fault), and call latency.
//   c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_channel.cpp -o /tmp/channel-test
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "remote/bb_channel.h"

using namespace BbRemote;

namespace {
enum : std::uint32_t { Data = 1, Echo = 2, FaultLike = 3, ProtectLike = 4, Quit = 5 };

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #x);          \
            std::_Exit(1);                                                                         \
        }                                                                                          \
    } while (0)

int Backend(void* memory) {
    Channel channel{memory, Side::Backend};
    CHECK(channel.Valid());
    std::uint64_t expected = 0;
    bool quit = false;
    while (!quit && channel.ReceiveOne(
                        [&](const Message& m, auto&& reply) {
                            switch (m.type) {
                            case Data: {
                                std::uint64_t value;
                                CHECK(m.size >= sizeof(value));
                                std::memcpy(&value, m.data, sizeof(value));
                                CHECK(value == expected); // in order
                                // The rest of the payload is the value's low byte repeated.
                                const auto* bytes = static_cast<const std::uint8_t*>(m.data);
                                for (std::uint32_t i = sizeof(value); i < m.size; ++i) {
                                    CHECK(bytes[i] == std::uint8_t(value));
                                }
                                ++expected;
                                break;
                            }
                            case Echo:
                                reply(m.data, m.size);
                                break;
                            case FaultLike: {
                                // A call back to the frontend while answering one of its calls.
                                std::uint64_t page;
                                std::memcpy(&page, m.data, sizeof(page));
                                std::uint64_t answer = 0;
                                CHECK(channel.Call(ProtectLike, &page, sizeof(page), &answer,
                                                   sizeof(answer)) == sizeof(answer));
                                CHECK(answer == page * 2);
                                const std::uint64_t result = answer + 1;
                                reply(&result, sizeof(result));
                                break;
                            }
                            case Quit:
                                reply(&expected, sizeof(expected));
                                quit = true;
                                break;
                            }
                        },
                        [] { return false; })) {
    }
    return 0;
}
} // namespace

int main() {
    void* memory = mmap(nullptr, ChannelBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                        -1, 0);
    CHECK(memory != MAP_FAILED);
    Channel::Init(memory);
    const pid_t child = fork();
    if (child == 0) {
        std::_Exit(Backend(memory));
    }
    Channel channel{memory, Side::Frontend};
    // The frontend's reader: answers the backend's calls.
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        while (channel.ReceiveOne(
            [](const Message& m, auto&& reply) {
                if (m.type == ProtectLike) {
                    std::uint64_t page;
                    std::memcpy(&page, m.data, sizeof(page));
                    const std::uint64_t answer = page * 2;
                    reply(&answer, sizeof(answer));
                }
            },
            [&] { return stop.load(); })) {
        }
    });

    // 1. Ordered messages of varying sizes: several passes around the 1 MiB ring.
    const std::uint64_t messages = 200000;
    std::vector<std::uint8_t> buffer(MaxPayload);
    for (std::uint64_t i = 0; i < messages; ++i) {
        const std::uint32_t size = 8 + std::uint32_t((i * 2654435761u) % (MaxPayload - 8));
        std::memcpy(buffer.data(), &i, 8);
        std::memset(buffer.data() + 8, std::uint8_t(i), size - 8);
        channel.Send(Data, buffer.data(), size);
    }

    // 2. Calls from several threads at once, interleaved with messages.
    std::vector<std::thread> callers;
    std::atomic<std::uint64_t> calls{0};
    for (int t = 0; t < 4; ++t) {
        callers.emplace_back([&, t] {
            for (std::uint64_t i = 0; i < 2000; ++i) {
                std::uint64_t value = (std::uint64_t(t) << 32) | i, back = 0;
                CHECK(channel.Call(Echo, &value, sizeof(value), &back, sizeof(back)) == 8);
                CHECK(back == value);
                std::uint64_t page = value, result = 0;
                CHECK(channel.Call(FaultLike, &page, sizeof(page), &result, sizeof(result)) == 8);
                CHECK(result == page * 2 + 1);
                calls.fetch_add(2);
            }
        });
    }
    for (auto& t : callers) {
        t.join();
    }

    // 3. Round-trip latency of a small call (spin, no sleep: the peer reader is busy-polling).
    const int rounds = 20000;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        std::uint32_t value = i, back = 0;
        channel.Call(Echo, &value, sizeof(value), &back, sizeof(back));
        CHECK(back == std::uint32_t(i));
    }
    const double us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
            .count() /
        rounds;

    std::uint64_t received = 0;
    CHECK(channel.Call(Quit, nullptr, 0, &received, sizeof(received)) == 8);
    CHECK(received == messages);
    int status = 0;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    stop = true;
    channel.WakeReader();
    reader.join();
    std::printf("channel: %llu messages in order, %llu calls (both ways), %.1f us per call\n",
                (unsigned long long)received, (unsigned long long)calls.load(), us);
    return 0;
}
