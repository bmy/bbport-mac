// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#include <upnperrors.h>
#include "common/logging/log.h"
#include "net_upnp.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <net/if.h>
#endif

namespace Libraries::Net {

UPnPClient& UPnPClient::Instance() {
    static UPnPClient instance;
    return instance;
}

UPnPClient::~UPnPClient() {
    m_map_cancel = true;
    if (m_thread.joinable())
        m_thread.join();
    if (m_map_thread.joinable())
        m_map_thread.join();
    // Leave nothing behind on the router (the lease ends the mapping too when the game ends
    // without this destructor: _Exit, a crash).
    if (const u16 port = m_mapped_port.exchange(0); port != 0)
        RemoveMapping(port);
    if (m_available.load())
        FreeUPNPUrls(&m_urls);
}

void UPnPClient::Start() {
    if (!m_p2p_features_enabled.load()) {
        LOG_INFO(Lib_Net, "UPNP: skipped because Matching2 is disabled");
        return;
    }

    bool expected = false;
    if (!m_started.compare_exchange_strong(expected, true))
        return;
    LOG_INFO(Lib_Net, "UPNP: starting discovery thread");
    m_thread = std::thread(&UPnPClient::DiscoverThread, this);
}

void UPnPClient::SetP2PFeaturesEnabled(bool enabled) {
    m_p2p_features_enabled.store(enabled);
}

bool UPnPClient::WaitReady(int timeout_ms) {
    std::unique_lock lock(m_mutex);
    m_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return m_done.load(); });
    return m_available.load();
}

bool UPnPClient::IsAvailable() const {
    return m_available.load();
}

u32 UPnPClient::GetExternalIp() const {
    return m_external_ip_nbo.load();
}

std::string UPnPClient::GetExternalIpString() const {
    return m_available.load() ? std::string(m_external_ip_str) : std::string{};
}

u16 UPnPClient::GetExternalPort() const {
    return m_external_port.load();
}

void UPnPClient::DiscoverThread() {
    int error = 0;
    UPNPDev* devlist = upnpDiscover(2000, nullptr, nullptr, UPNP_LOCAL_PORT_ANY, 0, 2, &error);

    if (!devlist) {
        LOG_INFO(Lib_Net, "UPNP: no IGD found (error={})", error);
        m_done = true;
        m_cv.notify_all();
        return;
    }

    char wan_addr[40]{};
    // [extra] miniupnpc API version compatibility (< 18 uses 5 args, >= 18 uses 7 args)
#if MINIUPNPC_API_VERSION >= 18 // [extra]
    const int ret = UPNP_GetValidIGD(devlist, &m_urls, &m_data, m_lan_addr, sizeof(m_lan_addr),
                                     wan_addr, sizeof(wan_addr));
#else // [extra]
    const int ret = UPNP_GetValidIGD(devlist, &m_urls, &m_data, m_lan_addr, sizeof(m_lan_addr)); // [extra]
#endif // [extra]
    freeUPNPDevlist(devlist);

    if (ret != 1) {
        LOG_INFO(Lib_Net, "UPNP: no valid IGD (ret={})", ret);
        m_done = true;
        m_cv.notify_all();
        return;
    }

    char external_ip[40]{};
    if (UPNP_GetExternalIPAddress(m_urls.controlURL, m_data.first.servicetype, external_ip) !=
        UPNPCOMMAND_SUCCESS) {
        LOG_WARNING(Lib_Net, "UPNP: IGD found but failed to get external IP");
        FreeUPNPUrls(&m_urls);
        m_done = true;
        m_cv.notify_all();
        return;
    }

    u32 ip_nbo = 0;
    inet_pton(AF_INET, external_ip, &ip_nbo);
    m_external_ip_nbo.store(ip_nbo);
    std::strncpy(m_external_ip_str, external_ip, sizeof(m_external_ip_str) - 1);
    m_available = true;

    LOG_INFO(Lib_Net, "UPNP: IGD found - external IP={} lan={} (port mapping available)",
             external_ip, m_lan_addr);

    m_done = true;
    m_cv.notify_all();
}

// bbport: the router's port mapping lasts MappingLease seconds and is renewed while the game
// runs (the game usually ends without removing it: _Exit, a crash). Mappings this PC's earlier
// sessions left (same description and address, another port) are removed first, so they do not
// pile up.
namespace {
constexpr const char* MappingDescription = "bbport";
constexpr const char* MappingLease = "7200";
constexpr auto MappingRenewal = std::chrono::hours(1);
constexpr int OnlyPermanentLeasesSupported = 725;
} // namespace

void UPnPClient::RemoveStaleMappings(u16 port) {
    std::vector<std::string> stale;
    for (int index = 0; index < 256; ++index) {
        char ext_port[6]{}, int_client[16]{}, int_port[6]{}, protocol[4]{}, desc[80]{}, enabled[4]{},
            remote_host[64]{}, duration[16]{};
        const std::string index_str = std::to_string(index);
        if (UPNP_GetGenericPortMappingEntry(m_urls.controlURL, m_data.first.servicetype,
                                            index_str.c_str(), ext_port, int_client, int_port,
                                            protocol, desc, enabled, remote_host,
                                            duration) != UPNPCOMMAND_SUCCESS) {
            break; // the end of the list (or a router that does not list its mappings)
        }
        if (std::strcmp(desc, MappingDescription) == 0 && std::strcmp(protocol, "UDP") == 0 &&
            std::strcmp(int_client, m_lan_addr) == 0 && std::to_string(port) != ext_port) {
            stale.emplace_back(ext_port);
        }
    }
    for (const std::string& old_port : stale) {
        const int ret = UPNP_DeletePortMapping(m_urls.controlURL, m_data.first.servicetype,
                                               old_port.c_str(), "UDP", nullptr);
        LOG_INFO(Lib_Net, "UPNP: removed an earlier session's UDP port mapping {} ({})", old_port,
                 ret == UPNPCOMMAND_SUCCESS ? "done" : strupnperror(ret));
    }
}

void UPnPClient::AddMapping(u16 port) {
    if (!m_p2p_features_enabled.load())
        return;

    if (!m_available.load())
        return;

    RemoveStaleMappings(port);
    const std::string port_str = std::to_string(port);
    int ret = UPNP_AddPortMapping(m_urls.controlURL, m_data.first.servicetype, port_str.c_str(),
                                  port_str.c_str(), m_lan_addr, MappingDescription, "UDP", nullptr,
                                  MappingLease);
    const bool permanent = ret == OnlyPermanentLeasesSupported;
    if (permanent) {
        ret = UPNP_AddPortMapping(m_urls.controlURL, m_data.first.servicetype, port_str.c_str(),
                                  port_str.c_str(), m_lan_addr, MappingDescription, "UDP", nullptr,
                                  "0");
    }

    if (ret == UPNPCOMMAND_SUCCESS) {
        m_external_port.store(port);
        LOG_INFO(Lib_Net, "UPNP: mapped UDP port {} ({})", port,
                 permanent ? "this router keeps it until the next session removes it"
                           : "for 2 hours, renewed while the game runs");
    } else {
        // Store the requested port as best-guess even on failure so callers still get a value.
        m_external_port.store(port);
        LOG_WARNING(Lib_Net, "UPNP: failed to map port {} ({})", port, strupnperror(ret));
    }
}

void UPnPClient::MapPortAsync(u16 port) {
    std::scoped_lock lock{m_map_mutex};
    m_map_cancel = true;
    if (m_map_thread.joinable())
        m_map_thread.join();
    m_map_cancel = false;
    m_map_thread = std::thread([this, port] {
        if (!WaitReady(10'000)) {
            LOG_INFO(Lib_Net, "UPNP: no IGD, UDP port {} is not forwarded", port);
            return;
        }
        const u16 previous = m_mapped_port.exchange(port);
        if (previous != 0 && previous != port)
            RemoveMapping(previous);
        AddMapping(port);
        // bbport: the lease renewed every hour until another request or UnmapPort (a second's
        // wait for them at most).
        auto renew = std::chrono::steady_clock::now() + MappingRenewal;
        while (!m_map_cancel.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (!m_map_cancel.load() && std::chrono::steady_clock::now() >= renew) {
                AddMapping(port);
                renew = std::chrono::steady_clock::now() + MappingRenewal;
            }
        }
    });
}

void UPnPClient::UnmapPort(u16 port) {
    std::scoped_lock lock{m_map_mutex};
    m_map_cancel = true;
    if (m_map_thread.joinable())
        m_map_thread.join();
    u16 expected = port;
    if (m_mapped_port.compare_exchange_strong(expected, 0))
        RemoveMapping(port);
}

void UPnPClient::RemoveMapping(u16 port) {
    if (!m_available.load())
        return;

    const std::string port_str = std::to_string(port);
    UPNP_DeletePortMapping(m_urls.controlURL, m_data.first.servicetype, port_str.c_str(), "UDP",
                           nullptr);

    LOG_INFO(Lib_Net, "UPNP: removed UDP port mapping {}", port);
}

} // namespace Libraries::Net
