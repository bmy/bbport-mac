# bbport GPU library: shadPS4 video core + shader recompiler + Gnm/VideoOut
# drivers, adapted to the native_probe runtime through shims in shim/.
cmake_minimum_required(VERSION 3.24)
project(bbgpu LANGUAGES C CXX)
if (APPLE)
    enable_language(OBJCXX) # bbport: the MetalFX upscaler's Metal side (shim/bbport_metalfx.mm)
endif()
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
# Frame pointers keep the loader's thread dumps (watchdog, SIGUSR2 sampling) walkable.
add_compile_options(-fno-omit-frame-pointer)

# bbport: link-time optimization (BB_LTO, default on) and profile-guided optimization
# (BB_PGO: off, generate = instrumented build writing into BB_PGO_DIR, use = build with that
# profile). Covers bbgpu and the static libraries linked into it. No -march: the same build
# runs on the Steam Deck.
option(BB_LTO "Link-time optimization of bbgpu" ON)
set(BB_PGO "off" CACHE STRING "Profile-guided optimization: off, generate, use")
set(BB_PGO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pgo" CACHE PATH "Profile directory")
if (BB_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT bb_ipo_ok OUTPUT bb_ipo_error)
    if (bb_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        if (APPLE)
            # bbport (macOS): ThinLTO, with a cache so that the app's Update relinks quickly.
            set(CMAKE_C_COMPILE_OPTIONS_IPO -flto=thin)
            set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
            add_link_options(-flto=thin "-Wl,-cache_path_lto,${CMAKE_BINARY_DIR}/lto-cache")
        endif()
    else()
        message(WARNING "LTO unavailable: ${bb_ipo_error}")
    endif()
endif()
if (BB_PGO STREQUAL "generate")
    # Atomic counters: the GPU, recording, preparation and copy threads run the same code.
    add_compile_options(-fprofile-generate=${BB_PGO_DIR} -fprofile-update=atomic)
    add_link_options(-fprofile-generate=${BB_PGO_DIR})
    add_compile_definitions(BB_PGO_GENERATE)
elseif (BB_PGO STREQUAL "use")
    # Partial training keeps unprofiled code optimized normally; counters from concurrent
    # threads can be slightly inconsistent (correction). Functions changed since the profile
    # (or built differently for it, bbgpu_init) are compiled without it.
    add_compile_options(-fprofile-use=${BB_PGO_DIR} -fprofile-partial-training
                        -fprofile-correction -Wno-missing-profile -Wno-coverage-mismatch)
endif()

find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(magic_enum QUIET)
if(NOT magic_enum_FOUND)
    # Header-only; packaged on few systems (the Windows cross build's MSYS2 has none).
    include(FetchContent)
    FetchContent_Declare(magic_enum GIT_REPOSITORY https://github.com/Neargye/magic_enum.git GIT_TAG v0.9.7)
    FetchContent_MakeAvailable(magic_enum)
endif()
find_package(tsl-robin-map REQUIRED)
find_package(VulkanMemoryAllocator CONFIG QUIET)
if (NOT TARGET GPUOpen::VulkanMemoryAllocator)
    # The header alone, without its CMake package (MSYS2: include/vma/vk_mem_alloc.h).
    find_path(VMA_INCLUDE_DIR vk_mem_alloc.h PATH_SUFFIXES vma REQUIRED)
    add_library(GPUOpen::VulkanMemoryAllocator INTERFACE IMPORTED)
    target_include_directories(GPUOpen::VulkanMemoryAllocator INTERFACE ${VMA_INCLUDE_DIR})
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(XXHASH REQUIRED IMPORTED_TARGET libxxhash)
pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
if (NOT WIN32 AND NOT APPLE) # bbport (macOS): no X11 there
    pkg_check_modules(X11 REQUIRED IMPORTED_TARGET x11)
endif()
find_package(miniz QUIET)
if(NOT miniz_FOUND)
    # Distributions without a miniz package (Debian, Ubuntu: #136): built from its release.
    include(FetchContent)
    FetchContent_Declare(miniz GIT_REPOSITORY https://github.com/richgel999/miniz.git GIT_TAG 3.1.2)
    FetchContent_MakeAvailable(miniz)
    if(NOT TARGET miniz::miniz)
        add_library(miniz::miniz ALIAS miniz)
    endif()
endif()
find_package(Zydis REQUIRED)
find_program(GLSLANG_COMPILER NAMES glslang glslangValidator REQUIRED)
# xbyak (header-only; shader recompiler): from the system on Linux, the macOS prefix otherwise.
# bbport: x86-64 only; elsewhere the SRT walker is bytecode (docs/macos-native-gpu.md).
if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(XBYAK_INCLUDE_DIR "")
else()
    find_path(XBYAK_INCLUDE_DIR xbyak/xbyak.h REQUIRED)
endif()

set(SIRIT_USE_SYSTEM_SPIRV_HEADERS OFF CACHE BOOL "" FORCE)
if (WIN32)
    # Every target, the third-party ones too: what glibc and MSVC have and MinGW does not.
    include_directories(BEFORE platform/win32/include)
    enable_language(RC)
    # Bit-fields laid out as on Linux (GCC's way), not MSVC's: the GPU registers and the PS4's
    # structures are declared with them.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-mno-ms-bitfields>)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/platform/win32/include/bbport_win32.h>")
endif()
if (APPLE)
    # bbport (macOS): the few Linux functions the library calls that macOS lacks (gettid,
    # process_vm_readv, per-thread getrusage), as platform/win32 does for Windows.
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/platform/macos/bbport_macos.h>")
endif()
add_subdirectory(third_party/sirit EXCLUDE_FROM_ALL)

# bbport: FSR 3.1 temporal upscaling / frame generation, native Vulkan (MIT, FireBurn/FSR-Vulkan).
set(FFX_VK_PORTABLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FFX_VK_PORTABLE_BUILD_PROBE OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/fsr-vulkan EXCLUDE_FROM_ALL)

# Host shaders -> headers: SPIR-V arrays from glslang, in bbgpu/shaders/.
set(SHADER_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SHADER_HEADERS "")
function(add_shader_header SOURCE OUT_DIR SHADER_NAME)
    set(OUT ${OUT_DIR}/${SHADER_NAME}.h)
    string(TOUPPER ${SHADER_NAME} SYMBOL)
    set(DEFINES ${ARGN})
    list(TRANSFORM DEFINES PREPEND "-D")
    add_custom_command(OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}
        COMMAND ${GLSLANG_COMPILER} --target-env vulkan1.3 -o ${OUT} --vn ${SYMBOL} ${DEFINES}
                -I${CMAKE_CURRENT_SOURCE_DIR}/third_party/amd-fsr1 ${SOURCE}
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${OUT}" -P ${CMAKE_CURRENT_SOURCE_DIR}/render/shaders/header.cmake
        DEPENDS ${SOURCE} VERBATIM)
    set(SHADER_HEADERS ${SHADER_HEADERS} ${OUT} PARENT_SCOPE)
endfunction()
set(OWN_SHADER_DIR ${SHADER_INCLUDE}/bbgpu/shaders)
function(add_own_shader FILENAME SHADER_NAME)
    add_shader_header(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} ${OWN_SHADER_DIR} ${SHADER_NAME} ${ARGN})
    set(SHADER_HEADERS ${SHADER_HEADERS} PARENT_SCOPE)
endfunction()
add_own_shader(render/shaders/depth_from_color.frag depth_from_color_frag)
add_own_shader(render/shaders/texel_copy.frag texel_copy_frag)
add_own_shader(render/shaders/texel_copy.frag texel_copy_msaa_frag MSAA_SOURCE=1)
add_own_shader(buffers/shaders/fault_scan.comp fault_scan_comp)
add_own_shader(render/shaders/fullscreen.vert fullscreen_vert)
add_own_shader(render/shaders/present_encode.frag present_encode_frag)
add_own_shader(upscale/shaders/camera_motion.comp camera_motion_comp)
add_own_shader(upscale/shaders/upscale_merge.comp upscale_merge_comp)
add_own_shader(upscale/shaders/upscale_reactive.comp upscale_reactive_comp)
add_own_shader(upscale/shaders/taa.comp taa_comp)
add_own_shader(upscale/shaders/taa_sharpen.comp taa_sharpen_comp)
add_own_shader(upscale/shaders/taa_sharpen.comp taa_sharpen_ldr_comp LDR_OUTPUT=1)
add_own_shader(upscale/shaders/taa_sharpen.comp taa_sharpen_half_comp HALF_INPUT=1)
add_own_shader(upscale/shaders/fsr4_color.comp fsr4_decode_comp)
add_own_shader(upscale/shaders/fsr4_color.comp fsr4_encode_comp ENCODE=1)
add_own_shader(upscale/shaders/fsr4_reactive.comp fsr4_reactive_comp)
add_own_shader(render/shaders/buffer_multi_copy.comp buffer_multi_copy_comp)
add_own_shader(render/shaders/occlusion_resolve.comp occlusion_resolve_comp)
add_own_shader(render/shaders/gpu_timestamp.comp gpu_timestamp_comp)
add_own_shader(render/shaders/indirect_guard.comp indirect_guard_comp)
add_own_shader(render/shaders/indirect_draw_guard.comp indirect_draw_guard_comp)
add_own_shader(upscale/shaders/depth_resample.frag depth_resample_frag)
add_own_shader(upscale/shaders/depth_resample.frag depth_stencil_resample_frag HAS_STENCIL=1)
add_own_shader(upscale/shaders/depth_resample.frag depth_stencil_bits_frag STENCIL_BITS=1)
add_own_shader(upscale/shaders/camera_motion.comp camera_motion_debug_comp DEBUG_OVERLAY=1)
foreach(BPP IN ITEMS 8 16 32 64 96 128)
    add_own_shader(textures/shaders/detile.comp detile_micro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=0)
    add_own_shader(textures/shaders/detile.comp detile_macro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=1)
endforeach()
add_custom_target(host_shaders DEPENDS ${SHADER_HEADERS})

file(GLOB_RECURSE RECOMPILER CONFIGURE_DEPENDS shader/*.cpp)
# Own utilities (types, logging, paths, locks, containers): common/.
file(GLOB COMMON CONFIGURE_DEPENDS common/*.cpp common/logging/*.cpp)
# Own system libraries (Gnm, VideoOut, AvPlayer, event queues, interrupts): platform/.
file(GLOB_RECURSE PLATFORM_SOURCES CONFIGURE_DEPENDS platform/*.cpp)
# platform/win32: the POSIX subset over Win32 (with its headers in platform/win32/include).
if (NOT WIN32)
    list(FILTER PLATFORM_SOURCES EXCLUDE REGEX "/platform/win32/")
endif()
if (NOT APPLE)
    list(FILTER PLATFORM_SOURCES EXCLUDE REGEX "/platform/macos/")
endif()
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale libswresample)
file(GLOB SHIM_SOURCES CONFIGURE_DEPENDS shim/*.cpp shim/*/*.cpp)
# The layer's own modules (independent of the shadPS4 code): layer/.
file(GLOB LAYER_SOURCES CONFIGURE_DEPENDS layer/*.cpp)
# Game profiles (what the translator knows about one game, out of its core): games/.
file(GLOB GAME_SOURCES CONFIGURE_DEPENDS games/*.cpp)
# The translator's own command processor (docs/TRANSLATOR_ARCHITECTURE.ru.md, step 2): cp/.
file(GLOB CP_SOURCES CONFIGURE_DEPENDS cp/*.cpp memory/*.cpp gcn/*.cpp textures/*.cpp buffers/*.cpp vulkan/*.cpp)
# The translator's rendering pieces (labels, timestamps, occlusion, draw preparation, profiling,
# frame capture, prompts): render/. Temporal upscaling, motion vectors, scene resolution: upscale/.
file(GLOB RENDER_SOURCES CONFIGURE_DEPENDS render/*.cpp)
file(GLOB UPSCALE_SOURCES CONFIGURE_DEPENDS upscale/*.cpp upscale/fsr411/*.cpp)
# Windows: linked into bb-probe.exe (a DLL cannot leave the runtime's symbols to the executable
# that loads it, as -Wl,--allow-shlib-undefined does on Linux).
if (WIN32)
    set(BBGPU_LIBRARY_TYPE STATIC)
else()
    set(BBGPU_LIBRARY_TYPE SHARED)
endif()
add_library(bbgpu ${BBGPU_LIBRARY_TYPE} ${RECOMPILER} ${COMMON} ${PLATFORM_SOURCES} ${SHIM_SOURCES} ${LAYER_SOURCES} ${GAME_SOURCES} ${CP_SOURCES}
    ${RENDER_SOURCES} ${UPSCALE_SOURCES})
add_dependencies(bbgpu host_shaders)
# bbport: MetalFX temporal upscaler, Metal side (Objective-C++ with ARC). MetalFX.framework is
# opened at run time (dlopen), so a missing x86-64 slice cannot break the link or the start.
# bbport: the Metal command path (metal/, docs/MACOS_METAL_RENDERER.md); its interop test is a no-op
# elsewhere.
target_sources(bbgpu PRIVATE metal/metal_interop_test.cpp metal/mtl_layer.cpp)
if (APPLE)
    target_sources(bbgpu PRIVATE shim/bbport_metalfx.mm metal/metal_interop.mm)
    set_source_files_properties(shim/bbport_metalfx.mm metal/metal_interop.mm PROPERTIES
                                COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(bbgpu PRIVATE "-framework Metal" "-framework Foundation"
                                        "-framework CoreFoundation")
endif()
# bbport: the Metal command path's shader translation and pipeline states (metal/mtl_*, stage 2:
# the shadow build, BB_METAL_SHADOW=1), in the native GPU process only, with SPIRV-Cross from
# setup_deps.sh (BB_ARCH=arm64). Without SPIRV-Cross, or in any other build, BB_METAL_BACKEND
# stays undefined and the hooks are no-ops (metal/vk_metal_shadow.h).
if (APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR
               CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"))
    # Only the pinned build (its stamp, setup_deps.sh), never another copy (Homebrew's).
    unset(bb_spvc_prefix)
    foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
        if (EXISTS "${prefix}/share/spirv_cross_msl/.bbport-rev")
            set(bb_spvc_prefix "${prefix}")
            break()
        endif()
    endforeach()
    # Searched again at every configure: a path an older configure cached may be another copy.
    unset(spirv_cross_core_DIR CACHE)
    unset(spirv_cross_glsl_DIR CACHE)
    unset(spirv_cross_msl_DIR CACHE)
    unset(BB_SPIRV_CROSS_INCLUDE_DIR CACHE)
    if (bb_spvc_prefix)
        file(STRINGS "${bb_spvc_prefix}/share/spirv_cross_msl/.bbport-rev" bb_spvc_rev LIMIT_COUNT 1)
        find_package(spirv_cross_core CONFIG QUIET PATHS "${bb_spvc_prefix}" NO_DEFAULT_PATH)
        find_package(spirv_cross_glsl CONFIG QUIET PATHS "${bb_spvc_prefix}" NO_DEFAULT_PATH)
        find_package(spirv_cross_msl CONFIG QUIET PATHS "${bb_spvc_prefix}" NO_DEFAULT_PATH)
        find_path(BB_SPIRV_CROSS_INCLUDE_DIR spirv_cross/spirv_msl.hpp
                  PATHS "${bb_spvc_prefix}/include" NO_DEFAULT_PATH)
    endif()
    if (bb_spvc_prefix AND spirv_cross_core_FOUND AND spirv_cross_glsl_FOUND AND
        spirv_cross_msl_FOUND AND BB_SPIRV_CROSS_INCLUDE_DIR)
        set(BB_METAL_OBJCXX_SOURCES metal/mtl_device.mm metal/mtl_encoder.mm
            metal/mtl_geometry.mm metal/mtl_frame.mm metal/mtl_pso_cache.mm metal/mtl_shader.mm
            metal/mtl_shadow.mm metal/mtl_tessellation.mm metal/vk_to_mtl.mm)
        target_sources(bbgpu PRIVATE ${BB_METAL_OBJCXX_SOURCES} metal/mtl_spirv_geometry.cpp
            metal/mtl_spirv_pervertex.cpp metal/mtl_spirv_rectlist.cpp metal/mtl_spirv_robust.cpp
            metal/mtl_buffer_barriers.cpp metal/mtl_object.cpp metal/mtl_order.cpp)
        set_source_files_properties(${BB_METAL_OBJCXX_SOURCES} PROPERTIES
                                    COMPILE_OPTIONS "-fobjc-arc")
        target_include_directories(bbgpu PRIVATE ${BB_SPIRV_CROSS_INCLUDE_DIR})
        target_link_libraries(bbgpu PRIVATE spirv-cross-msl spirv-cross-glsl spirv-cross-core)
        # The MSL cache's header names it: a new pin does not reuse shaders of the old one.
        target_compile_definitions(bbgpu PRIVATE BB_METAL_BACKEND=1
                                   BB_SPIRV_CROSS_VERSION="${bb_spvc_rev}")
        message(STATUS "Metal command path: SPIRV-Cross ${bb_spvc_rev}, the shadow build is "
                       "available")
    else()
        message(STATUS "Metal command path: no SPIRV-Cross (BB_ARCH=arm64 setup_deps.sh), off")
    endif()
endif()
# In-game settings menu: Dear ImGui (third_party/imgui, MIT) with its Vulkan backend, and an
# embedded Cyrillic font (third_party/fonts).
target_sources(bbgpu PRIVATE third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp)
target_include_directories(bbgpu PRIVATE third_party/imgui third_party/imgui/backends)
# Dear ImGui stays private to libbbgpu.so. The library is linked into bb-probe, so its exports are
# in the global symbol scope; a Vulkan layer carrying its own Dear ImGui without -Bsymbolic
# (vkShade 0.1) had its ImGui_ImplVulkan_Init bound to this copy, which called the Vulkan loader
# with the layer's down-chain physical device handle ("Invalid physicalDevice", abort).
set_source_files_properties(third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")
target_compile_definitions(bbgpu PRIVATE IMGUI_IMPL_VULKAN_NO_PROTOTYPES
    BB_FONT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/fonts/DejaVuSans.ttf")
set_source_files_properties(shim/bbport_overlay.cpp PROPERTIES
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fonts/DejaVuSans.ttf")
target_include_directories(bbgpu PRIVATE . shim layer games cp ${SHADER_INCLUDE} third_party/gcn/include third_party/half/include ${XBYAK_INCLUDE_DIR})
if (WIN32)
    target_include_directories(bbgpu PUBLIC platform/win32/include)
endif()
target_compile_definitions(bbgpu PRIVATE BOOST_ASIO_STANDALONE NDEBUG)
target_compile_options(bbgpu PRIVATE -Wno-deprecated-declarations -Wno-missing-field-initializers)
# Loaded with bb-probe at startup (DT_NEEDED, never dlopen): thread_local access without a
# TLS descriptor call (_dl_tlsdesc_return was ~2% of the draw recording thread).
if (APPLE)
    # bbport (macOS): Mach-O has one TLS model (TLV); the runtime symbols (runtime_memory_*,
    # runtime_process_*) resolve against bb-probe when it loads the library.
    target_link_options(bbgpu PRIVATE -undefined dynamic_lookup)
elseif (NOT WIN32)
    target_compile_options(bbgpu PRIVATE -ftls-model=initial-exec)
    # Runtime symbols (runtime_memory_*, runtime_process_*) resolve against bb-probe.
    target_link_options(bbgpu PRIVATE -Wl,--allow-shlib-undefined)
endif()
target_link_libraries(bbgpu PRIVATE ffx-vulkan::portable ffx-vulkan::fsr4-v07-vulkan)
target_link_libraries(bbgpu PUBLIC Vulkan::Headers fmt::fmt Boost::headers magic_enum::magic_enum
    tsl::robin_map GPUOpen::VulkanMemoryAllocator PkgConfig::XXHASH PkgConfig::SDL3 PkgConfig::FFMPEG miniz::miniz Zydis::Zydis sirit)
if (NOT WIN32 AND NOT APPLE) # bbport (macOS): no X11 there
    target_link_libraries(bbgpu PUBLIC PkgConfig::X11)
endif()

# Windows: the runtime (src/, C) and this library in one bb-probe.exe (on Linux build.sh links the
# runtime against libbbgpu.so). Below 1 TiB stay the host heap and thread stacks the game packs
# into 40-bit fields: no high-entropy address randomization.
if (WIN32)
    file(GLOB BB_RUNTIME_SOURCES CONFIGURE_DEPENDS ../src/runtime*.c)
    file(GLOB BB_ATRAC9_SOURCES CONFIGURE_DEPENDS ../third_party/LibAtrac9/C/src/*.c)
    add_library(bbatrac9 STATIC ${BB_ATRAC9_SOURCES})
    target_compile_options(bbatrac9 PRIVATE -w)
    add_executable(bb-probe ../src/probe.c ../src/vulkan_smoke.c ${BB_RUNTIME_SOURCES}
        platform/win32/bb-probe.rc)
    set_target_properties(bb-probe PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
    target_include_directories(bb-probe PRIVATE .. ../src)
    target_compile_options(bb-probe PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra>)
    target_link_libraries(bb-probe PRIVATE bbgpu bbatrac9 pthread ws2_32 bcrypt dbghelp winmm onecore
        ${Vulkan_LIBRARIES})
    # Plain -l names in the packages' CMake files (Zydis: Zycore) are found in the sysroot.
    foreach(root ${CMAKE_FIND_ROOT_PATH})
        target_link_directories(bb-probe PRIVATE ${root}/lib)
    endforeach()
    target_link_options(bb-probe PRIVATE -Wl,--disable-high-entropy-va
        -Wl,--stack,16777216 -Wl,--error-limit=0)
endif()

# bbport: bb-gpu, the native GPU process (docs/macos-native-gpu.md): the GPU core for a game
# process, which starts it by default once built. It provides the library's runtime imports from its own
# view of guest memory (shim/remote/remote_back.cpp). Built on demand
# (tools/macos/build_gpu_native.sh).
add_executable(bb-gpu EXCLUDE_FROM_ALL bb_gpu/bb_gpu_main.cpp)
target_include_directories(bb-gpu PRIVATE . shim third_party/gcn/include)
target_link_libraries(bb-gpu PRIVATE bbgpu)
if (APPLE)
    target_link_options(bb-gpu PRIVATE -Wl,-export_dynamic)
else()
    target_link_options(bb-gpu PRIVATE -rdynamic)
endif()

# Explicit test targets; neither launches the game nor needs a Vulkan device.
add_executable(motion-history-test EXCLUDE_FROM_ALL ../tests/test_motion_history.cpp)
target_compile_options(motion-history-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(motion-history-test PRIVATE ..)
add_executable(motion-shader-test EXCLUDE_FROM_ALL ../tests/test_motion_shaders.cpp ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(motion-shader-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include)
target_link_libraries(motion-shader-test PRIVATE bbgpu)
add_executable(ui-composition-test EXCLUDE_FROM_ALL ../tests/test_ui_composition.cpp)
target_compile_options(ui-composition-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(ui-composition-test PRIVATE ..)
add_executable(cp-decoder-test EXCLUDE_FROM_ALL ../tests/test_cp_decoder.cpp cp/cp_decoder.cpp)
target_compile_options(cp-decoder-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(cp-decoder-test PRIVATE cp)
add_executable(fx-init-clear-test EXCLUDE_FROM_ALL ../tests/test_fx_init_clear.cpp cp/cp_decoder.cpp)
target_compile_options(fx-init-clear-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(fx-init-clear-test PRIVATE . shim)
add_executable(effect-flags-test EXCLUDE_FROM_ALL ../tests/test_effect_flags.cpp)
target_compile_options(effect-flags-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(effect-flags-test PRIVATE shim)
add_executable(settings-test EXCLUDE_FROM_ALL ../tests/test_settings.cpp shim/bbport_settings.cpp)
target_compile_options(settings-test PRIVATE -UNDEBUG -Wall -Wextra -Werror)
target_include_directories(settings-test PRIVATE .. shim)

add_executable(scene-resolution-test EXCLUDE_FROM_ALL ../tests/test_scene_resolution.cpp ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(scene-resolution-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include)
# The test's Vulkan-Hpp inline methods have a different set of platform extensions
# from vk_platform.cpp. Keep them local and use an explicit dispatcher in the test.
target_compile_options(scene-resolution-test PRIVATE -UNDEBUG -fPIC)
set_source_files_properties(../tests/test_scene_resolution.cpp PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")
target_link_libraries(scene-resolution-test PRIVATE bbgpu)
add_executable(taa-shader-test EXCLUDE_FROM_ALL ../tests/test_taa_shader.cpp ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(taa-shader-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include)
target_compile_options(taa-shader-test PRIVATE -UNDEBUG -fPIC)
set_source_files_properties(../tests/test_taa_shader.cpp PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")
target_link_libraries(taa-shader-test PRIVATE bbgpu)
add_dependencies(taa-shader-test host_shaders)

add_executable(camera-motion-test EXCLUDE_FROM_ALL ../tests/test_camera_motion.cpp ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(camera-motion-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include)
target_compile_options(camera-motion-test PRIVATE -UNDEBUG -fPIC)
target_link_libraries(camera-motion-test PRIVATE bbgpu)
add_dependencies(camera-motion-test host_shaders)

add_executable(translation-memory-test EXCLUDE_FROM_ALL ../tests/test_translation_memory.cpp
    ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(translation-memory-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include)
target_compile_options(translation-memory-test PRIVATE -UNDEBUG -fPIC)
set_source_files_properties(../tests/test_translation_memory.cpp PROPERTIES
    COMPILE_OPTIONS "-fvisibility=hidden;-fvisibility-inlines-hidden")
target_link_libraries(translation-memory-test PRIVATE bbgpu)

# bbport: FSR 4 benchmark outside the game (tools/fsr4_bench.cpp): ninja -C out/gpu fsr4-bench.
add_executable(upscaler-support-test EXCLUDE_FROM_ALL ../tests/test_upscaler_support.cpp
    shim/bbport_settings.cpp)
target_include_directories(upscaler-support-test PRIVATE shim)
target_compile_options(upscaler-support-test PRIVATE -UNDEBUG)

add_executable(fsr4-bench EXCLUDE_FROM_ALL tools/fsr4_bench.cpp
    upscale/fsr411/fsr411.cpp)
target_include_directories(fsr4-bench PRIVATE upscale/fsr411)
target_link_libraries(fsr4-bench PRIVATE ffx-vulkan::fsr4-v07-vulkan ffx-vulkan::fsr4-v07-assets Vulkan::Vulkan)

# bbport online module (gpu/bbnet, libbbnet.so): built when its libraries are there (shell.nix);
# BB_NET=OFF leaves it out. Without it the port plays offline.
option(BB_NET "Build the online module (shadNet)" ON)
if (BB_NET)
    find_package(Protobuf CONFIG QUIET)
    find_package(nlohmann_json CONFIG QUIET)
    find_package(OpenSSL QUIET)
    find_package(ZLIB QUIET)
    find_path(MINIUPNPC_INCLUDE miniupnpc.h PATH_SUFFIXES miniupnpc)
    find_library(MINIUPNPC_LIBRARY NAMES miniupnpc)
    if (Protobuf_FOUND AND nlohmann_json_FOUND AND OpenSSL_FOUND AND ZLIB_FOUND AND
        MINIUPNPC_INCLUDE AND MINIUPNPC_LIBRARY)
        add_subdirectory(bbnet)
        add_subdirectory(bbnet20)
        message(STATUS "Online module: on (libbbnet.so)")
    else()
        # Named one by one (#145: "the build has no online module" with no hint why).
        set(missing "")
        foreach(dep Protobuf nlohmann_json OpenSSL ZLIB)
            if (NOT ${dep}_FOUND)
                list(APPEND missing ${dep})
            endif()
        endforeach()
        if (NOT MINIUPNPC_INCLUDE OR NOT MINIUPNPC_LIBRARY)
            list(APPEND missing miniupnpc)
        endif()
        list(JOIN missing ", " missing)
        message(STATUS "Online module: off (missing: ${missing}; shell.nix lists the packages)")
    endif()
endif()

# bbport: the shader corpus replay (render/shader_capture.h): ninja -C out/gpu shader-corpus-test.
add_executable(shader-corpus-test EXCLUDE_FROM_ALL ../tests/test_shader_corpus.cpp
    ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(shader-corpus-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include third_party/half/include)
target_compile_options(shader-corpus-test PRIVATE -UNDEBUG -fPIC)
target_link_libraries(shader-corpus-test PRIVATE bbgpu)

add_executable(srt-walker-fault-test EXCLUDE_FROM_ALL ../tests/test_srt_walker_fault.cpp
    ../tests/test_gpu_runtime_stubs.cpp)
target_include_directories(srt-walker-fault-test PRIVATE . shim ${SHADER_INCLUDE} third_party/gcn/include third_party/half/include)
target_compile_options(srt-walker-fault-test PRIVATE -UNDEBUG -Wall -Wextra)
target_link_libraries(srt-walker-fault-test PRIVATE bbgpu)
