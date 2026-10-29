# bbport online module, second generation: libbbnet20.so. shadPS4 v0.20.0's network, NP and
# shadNet sources (vendor/shadps4, 529ceb3, with vendor-changes.diff) and The Hunter's Dream
# backend (thd/, from droogie/bbhost), as Supermedo/Awaken put them together
# (SHADNET-PORT.md). The port's only online module (src/runtime_net.c loads it, BB_NET_MODULE
# overrides the path; the former shadp2p module gpu/bbnet was removed 2026-10-26). Hidden
# symbols and -Bsymbolic: its C++ never meets libbbgpu's.

set(BBPORT_VERSION "0.6-pre1")
set(BBNET_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/shadps4")
set(SHADNET_PROTO_OUT20 "${CMAKE_CURRENT_BINARY_DIR}/shadnet_proto_gen")
file(MAKE_DIRECTORY "${SHADNET_PROTO_OUT20}")
# Cross builds (Windows from Linux) run the build machine's protoc, of the sysroot's version.
set(BBNET_PROTOC protobuf::protoc)
if (CMAKE_CROSSCOMPILING)
    find_program(BBNET_HOST_PROTOC protoc NO_CMAKE_FIND_ROOT_PATH REQUIRED)
    set(BBNET_PROTOC "${BBNET_HOST_PROTOC}")
endif()
add_custom_command(
    OUTPUT "${SHADNET_PROTO_OUT20}/shadnet.pb.cc" "${SHADNET_PROTO_OUT20}/shadnet.pb.h"
    COMMAND ${BBNET_PROTOC} "--proto_path=${BBNET_VENDOR}/shadnet"
            "--cpp_out=${SHADNET_PROTO_OUT20}" "${BBNET_VENDOR}/shadnet/shadnet.proto"
    DEPENDS "${BBNET_VENDOR}/shadnet/shadnet.proto")
file(GLOB_RECURSE BBNET20_SOURCES CONFIGURE_DEPENDS
    vendor/shadps4/core/libraries/net/*.cpp vendor/shadps4/core/libraries/network/*.cpp
    vendor/shadps4/core/libraries/np/*.cpp vendor/shadps4/core/net/*.cpp
    vendor/shadps4/shadnet/*.cpp)
# Trophies, commerce and the profile and Facebook dialogs stay in the C runtime.
list(FILTER BBNET20_SOURCES EXCLUDE REGEX
    "np_trophy|trophy_ui|np_sns_facebook_dialog|np_profile_dialog/|np_commerce/")
# BBSC, Bloodborne seamless co-op (GPL-2.0+): its game code and the port's stand-ins for
# shadPS4's UI (bbsc/bbsc_glue.cpp). Installed only for Bloodborne with BB_BBSC=1.
file(GLOB BBNET20_BBSC_SOURCES CONFIGURE_DEPENDS bbsc/core/*.cpp bbsc/bbsc_glue.cpp)
file(GLOB BBNET20_THD_SOURCES CONFIGURE_DEPENDS thd/*.cpp compat/common/*.cpp)

add_library(bbnet20 SHARED ${BBNET20_SOURCES} ${BBNET20_THD_SOURCES} ${BBNET20_BBSC_SOURCES}
    "${SHADNET_PROTO_OUT20}/shadnet.pb.cc" bbnet_glue.cpp)
target_include_directories(bbnet20 PRIVATE include bbsc compat . .. ../shim "${BBNET_VENDOR}"
    "${BBNET_VENDOR}/shadnet" "${SHADNET_PROTO_OUT20}" ../third_party/imgui ../third_party/httplib)
# The version the shadNet client reports to the server (BB_CLIENT_VERSION overrides it).
target_compile_definitions(bbnet20 PRIVATE BOOST_ASIO_STANDALONE NDEBUG
    CPPHTTPLIB_OPENSSL_SUPPORT CPPHTTPLIB_ZLIB_SUPPORT BBNET_PORT_VERSION="${BBPORT_VERSION}")
target_compile_options(bbnet20 PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden
    -Wno-deprecated-declarations -Wno-missing-field-initializers)
target_link_libraries(bbnet20 PRIVATE nlohmann_json::nlohmann_json protobuf::libprotobuf
    OpenSSL::SSL OpenSSL::Crypto ZLIB::ZLIB "${MINIUPNPC_LIBRARY}" Vulkan::Headers fmt::fmt
    Boost::headers magic_enum::magic_enum)
if (WIN32)
    # Windows: bbgpu is inside bb-probe.exe (static), so bbnet20.dll takes the few functions it
    # needs (log, paths, guest code hooks) from the executable's import library, not a second copy
    # of the GPU library. Next to bb-probe.exe as bin/bbnet20.dll (no "lib" prefix).
    target_include_directories(bbnet20 PRIVATE $<TARGET_PROPERTY:bbgpu,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(bbnet20 PRIVATE $<TARGET_PROPERTY:bbgpu,INTERFACE_COMPILE_DEFINITIONS>)
    # wepoll (piscisaureus/wepoll v1.5.8, BSD-2-Clause): epoll over AFD for shadPS4's host_net.
    target_sources(bbnet20 PRIVATE ../third_party/wepoll/wepoll.c)
    target_include_directories(bbnet20 PRIVATE ../third_party/wepoll)
    set_target_properties(bb-probe PROPERTIES ENABLE_EXPORTS ON)
    target_link_options(bb-probe PRIVATE -Wl,--export-all-symbols)
    target_link_libraries(bbnet20 PRIVATE bb-probe ws2_32 iphlpapi crypt32 PkgConfig::SDL3)
    set_target_properties(bbnet20 PROPERTIES PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
else()
    target_link_options(bbnet20 PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
    set_target_properties(bbnet20 PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries(bbnet20 PRIVATE bbgpu)
endif()
target_include_directories(bbnet20 PRIVATE "${MINIUPNPC_INCLUDE}")
