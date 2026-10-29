// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: MIT
//
// bbport_dlss.dll built with llvm-mingw (packaging/windows/package.sh): NVIDIA's nvsdk_ngx_s.lib
// is MSVC code (static CRT) and needs these few pieces of Microsoft's C++ runtime, which MinGW
// does not have. The rest of the C runtime it uses (malloc, memcpy, ...) comes from the UCRT,
// as MSVC's does. Only for the MinGW build; an MSVC build links the real ones.
#if defined(_WIN32) && defined(__MINGW32__)

#include <windows.h>

#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>

extern "C" {

// Code using floating point references it (MSVC's marker that the FP runtime is linked).
int _fltused = 0x9875;

// /GS stack cookies: the value is only compared with itself.
uintptr_t __security_cookie = 0x00002B992DDFA232ull;
void __security_check_cookie(uintptr_t cookie) {
    if (cookie != __security_cookie) {
        std::fputs("bbport_dlss: stack cookie overwritten in the DLSS SDK code\n", stderr);
        std::abort();
    }
}

// The language-specific handlers of /GS frames. The SDK's code is not unwound by C++ exceptions
// here (its errors are NVSDK_NGX_Result codes; allocation failures end in std::_Xbad_alloc,
// which aborts): nothing to handle, the search goes on.
EXCEPTION_DISPOSITION __GSHandlerCheck(PEXCEPTION_RECORD, void*, PCONTEXT, void*) {
    return ExceptionContinueSearch;
}
EXCEPTION_DISPOSITION __GSHandlerCheck_EH(PEXCEPTION_RECORD, void*, PCONTEXT, void*) {
    return ExceptionContinueSearch;
}

// Thread-safe function-local statics (MSVC's thread_safe_statics.cpp): the guard is 0 before,
// -1 while one thread runs the initializer, then the epoch it finished in.
static std::mutex init_mutex;
static std::condition_variable init_done;
int _Init_global_epoch = INT_MIN;
__thread int _Init_thread_epoch = INT_MIN;

void _Init_thread_header(int* guard) {
    std::unique_lock lock{init_mutex};
    if (*guard == 0) {
        *guard = -1; // this thread initializes
        return;
    }
    init_done.wait(lock, [guard] { return *guard != -1; });
    // Done (by another thread): the caller sees it through its thread epoch and skips.
    _Init_thread_epoch = _Init_global_epoch;
}

void _Init_thread_footer(int* guard) {
    {
        std::scoped_lock lock{init_mutex};
        ++_Init_global_epoch;
        *guard = _Init_global_epoch;
        _Init_thread_epoch = _Init_global_epoch;
    }
    init_done.notify_all();
}

void _Init_thread_abort(int* guard) {
    {
        std::scoped_lock lock{init_mutex};
        *guard = 0;
    }
    init_done.notify_all();
}

} // extern "C"

// MSVC-mangled C++ runtime entry points.
void* MsvcOperatorNew(unsigned long long size) __asm__("??2@YAPEAX_K@Z");
void* MsvcOperatorNew(unsigned long long size) {
    return ::operator new(size);
}
void MsvcOperatorDelete(void* pointer) __asm__("??3@YAXPEAX@Z");
void MsvcOperatorDelete(void* pointer) {
    ::operator delete(pointer);
}
[[noreturn]] void MsvcXbadAlloc() __asm__("?_Xbad_alloc@std@@YAXXZ");
[[noreturn]] void MsvcXbadAlloc() {
    std::fputs("bbport_dlss: out of memory in the DLSS SDK code\n", stderr);
    std::abort();
}
[[noreturn]] void MsvcXlengthError(const char* message) __asm__("?_Xlength_error@std@@YAXPEBD@Z");
[[noreturn]] void MsvcXlengthError(const char* message) {
    std::fprintf(stderr, "bbport_dlss: length error in the DLSS SDK code: %s\n", message);
    std::abort();
}

#endif
