# bbport online module: libbbnet.so, the PS4 network and PSN libraries with the shadNet client
# (shadp2p, a shadPS4 fork, GPL-2.0), loaded by the runtime only for online play
# (src/runtime_net.c). Its headers (include/) extend the port's shim for these libraries alone;
# everything in it is hidden and bound to itself (-Bsymbolic) except bbnet.h's two calls, so its
# copies of shadPS4's kernel shims and symbol table never meet libbbgpu's.
# Needs protobuf, nlohmann_json, OpenSSL, zlib and miniupnpc (shell.nix); without them the port
# builds without it and plays offline (gpu/CMakeLists.txt: BB_NET).

set(SHADNET_PROTO_OUT "${CMAKE_CURRENT_BINARY_DIR}/shadnet_proto_gen")
file(MAKE_DIRECTORY "${SHADNET_PROTO_OUT}")
add_custom_command(
    OUTPUT "${SHADNET_PROTO_OUT}/shadnet.pb.cc" "${SHADNET_PROTO_OUT}/shadnet.pb.h"
    COMMAND protobuf::protoc "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shadnet"
            "--cpp_out=${SHADNET_PROTO_OUT}" "${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shadnet/shadnet.proto"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shadps4/shadnet/shadnet.proto")
file(GLOB_RECURSE BBNET_SOURCES CONFIGURE_DEPENDS shadps4/core/libraries/network/*.cpp
    shadps4/core/libraries/np/*.cpp shadps4/shadnet/*.cpp)
# Trophies and the system dialogs stay in the C runtime; their shadPS4 versions need the
# emulator's UI layers.
list(FILTER BBNET_SOURCES EXCLUDE REGEX
    "np_trophy|trophy_ui|np_sns_facebook_dialog|np_profile_dialog/|np_commerce/")

add_library(bbnet SHARED ${BBNET_SOURCES} "${SHADNET_PROTO_OUT}/shadnet.pb.cc" bbnet_glue.cpp)
# The module's own headers first, then the port's shim and shadPS4 (common/, logging).
target_include_directories(bbnet PRIVATE include shadps4 shadps4/shadnet "${SHADNET_PROTO_OUT}"
    ${CMAKE_CURRENT_SOURCE_DIR} ../shim ../shadps4 ../third_party/imgui ../third_party/httplib)
target_compile_definitions(bbnet PRIVATE BOOST_ASIO_STANDALONE NDEBUG
    CPPHTTPLIB_OPENSSL_SUPPORT CPPHTTPLIB_ZLIB_SUPPORT)
target_compile_options(bbnet PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden
    -Wno-deprecated-declarations -Wno-missing-field-initializers)
target_link_options(bbnet PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
# libbbgpu (already in the process) provides shadPS4's common code: logging, threads' names.
set_target_properties(bbnet PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}") # beside libbbgpu.so, where the runtime looks
target_link_libraries(bbnet PRIVATE bbgpu nlohmann_json::nlohmann_json protobuf::libprotobuf
    OpenSSL::SSL OpenSSL::Crypto ZLIB::ZLIB "${MINIUPNPC_LIBRARY}" Vulkan::Headers fmt::fmt
    Boost::headers magic_enum::magic_enum)
target_include_directories(bbnet PRIVATE "${MINIUPNPC_INCLUDE}")
