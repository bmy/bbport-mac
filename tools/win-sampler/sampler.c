// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Windows thread sampler (diagnostics): where a running process's threads spend their time.
//   sampler.exe PID SECONDS [NAME_SUBSTRING]
// Suspends each thread whose description contains NAME_SUBSTRING (all threads without one)
// about every millisecond, reads its instruction pointer, and prints per thread
// "module+offset count" lines of the hottest addresses (and the module bases), ready for
// llvm-symbolizer --obj=<module> (tools/win-sampler/symbolize.py groups them by function).
// Build (MSYS2 CLANG64): clang -O2 sampler.c -o sampler.exe -lpsapi
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define MAX_THREADS 64
#define TABLE (1 << 20)

typedef struct {
    DWORD id;
    HANDLE handle;
    char name[128];
    unsigned long long samples, idle;
} Thread;

typedef struct {
    unsigned long long rip;
    unsigned long long caller; ///< outside the main module: the first return address into it
    unsigned long long chain[6]; ///< the first return addresses into it (BB chain mode)
    unsigned thread;
    unsigned count;
} Slot;

static Slot table[TABLE];

static void note(unsigned thread, unsigned long long rip, unsigned long long caller,
                 const unsigned long long* chain) {
    unsigned long long h = (rip * 0x9E3779B97F4A7C15ull) ^ (caller * 31) ^ thread;
    for (int k = 0; k < 6; ++k) h ^= chain[k] * (k + 7);
    for (unsigned i = (unsigned)(h >> 44) & (TABLE - 1);; i = (i + 1) & (TABLE - 1)) {
        if (table[i].count == 0) {
            table[i].rip = rip;
            table[i].caller = caller;
            memcpy(table[i].chain, chain, sizeof table[i].chain);
            table[i].thread = thread;
            table[i].count = 1;
            return;
        }
        if (table[i].rip == rip && table[i].caller == caller && table[i].thread == thread &&
            !memcmp(table[i].chain, chain, sizeof table[i].chain)) {
            ++table[i].count;
            return;
        }
    }
}

static int by_count(const void* a, const void* b) {
    const Slot* x = a;
    const Slot* y = b;
    if (x->thread != y->thread) return x->thread < y->thread ? -1 : 1;
    return x->count > y->count ? -1 : x->count < y->count;
}

typedef HRESULT(WINAPI* GetThreadDescriptionFn)(HANDLE, PWSTR*);

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: sampler PID SECONDS [THREAD_NAME_SUBSTRING]\n");
        return 2;
    }
    const DWORD pid = strtoul(argv[1], NULL, 0);
    const double seconds = atof(argv[2]);
    const char* filter = argc > 3 ? argv[3] : NULL;
    GetThreadDescriptionFn describe = (GetThreadDescriptionFn)(void*)GetProcAddress(
        GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
    timeBeginPeriod(1);

    Thread threads[MAX_THREADS];
    int count = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te = {sizeof te};
    for (BOOL ok = Thread32First(snap, &te); ok && count < MAX_THREADS; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid) continue;
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION,
                              FALSE, te.th32ThreadID);
        if (!h) continue;
        char name[128] = "";
        PWSTR wname = NULL;
        if (describe && SUCCEEDED(describe(h, &wname)) && wname) {
            WideCharToMultiByte(CP_UTF8, 0, wname, -1, name, sizeof name, NULL, NULL);
            LocalFree(wname);
        }
        if (filter && !strstr(name, filter)) {
            CloseHandle(h);
            continue;
        }
        threads[count].id = te.th32ThreadID;
        threads[count].handle = h;
        snprintf(threads[count].name, sizeof threads[count].name, "%s", name[0] ? name : "?");
        threads[count].samples = threads[count].idle = 0;
        ++count;
    }
    CloseHandle(snap);
    if (count == 0) {
        fprintf(stderr, "no matching threads\n");
        return 1;
    }

    // The main module (the first one listed): samples outside it look up the stack for a return
    // address into it (the first word pointing into it, a heuristic).
    HANDLE target = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    unsigned long long main_lo = 0, main_hi = 0;
    {
        HMODULE first;
        DWORD got = 0;
        MODULEINFO mi;
        if (target && EnumProcessModules(target, &first, sizeof first, &got) &&
            GetModuleInformation(target, first, &mi, sizeof mi)) {
            main_lo = (unsigned long long)mi.lpBaseOfDll;
            main_hi = main_lo + mi.SizeOfImage;
        }
    }
    LARGE_INTEGER freq, start, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    ULONG64 last_cycles[MAX_THREADS] = {0};
    for (;;) {
        QueryPerformanceCounter(&now);
        if ((now.QuadPart - start.QuadPart) / (double)freq.QuadPart >= seconds) break;
        for (int i = 0; i < count; ++i) {
            // A thread that ran no cycles since the last sample is waiting: counted as idle.
            ULONG64 cycles = 0;
            QueryThreadCycleTime(threads[i].handle, &cycles);
            const int ran = cycles != last_cycles[i];
            last_cycles[i] = cycles;
            if (SuspendThread(threads[i].handle) == (DWORD)-1) continue;
            CONTEXT c;
            c.ContextFlags = CONTEXT_CONTROL;
            static unsigned long long stack[8192];
            SIZE_T read = 0;
            int got = 0;
            if (GetThreadContext(threads[i].handle, &c)) {
                got = 1;
                if (c.Rip < main_lo || c.Rip >= main_hi) {
                    ReadProcessMemory(target, (void*)c.Rsp, stack, sizeof stack, &read);
                }
            }
            ResumeThread(threads[i].handle);
            if (got) {
                ++threads[i].samples;
                // A return address: a call instruction ends right before it (E8 rel32, or
                // FF /2 of 2, 3, 6 or 7 bytes).
                unsigned long long caller = 0, chain[6] = {0};
                int depth = 0;
                for (SIZE_T w = 0; w < read / 8 && depth < 6; ++w) {
                    const unsigned long long ret = stack[w];
                    if (ret < main_lo + 8 || ret >= main_hi) continue;
                    unsigned char code[8];
                    SIZE_T n = 0;
                    if (!ReadProcessMemory(target, (void*)(ret - 8), code, 8, &n) || n != 8) continue;
                    int is_call = code[3] == 0xE8;
                    static const int lengths[] = {2, 3, 6, 7};
                    for (int k = 0; k < 4 && !is_call; ++k) {
                        const int at = 8 - lengths[k];
                        is_call = code[at] == 0xFF && ((code[at + 1] >> 3) & 7) == 2;
                        if (!is_call && at > 0 && (code[at - 1] & 0xF0) == 0x40) {
                            is_call = 0; // REX-prefixed forms are covered by the longer lengths
                        }
                    }
                    if (is_call) {
                        if (!caller) caller = ret;
                        chain[depth++] = ret;
                    }
                }
                if (ran) note(i, c.Rip, caller, chain);
                else ++threads[i].idle;
            }
        }
        Sleep(1);
    }

    HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    HMODULE modules[512];
    DWORD needed = 0;
    int nmod = 0;
    MODULEINFO info[512];
    char names[512][MAX_PATH];
    if (process && EnumProcessModules(process, modules, sizeof modules, &needed)) {
        nmod = (int)(needed / sizeof(HMODULE));
        if (nmod > 512) nmod = 512;
        for (int m = 0; m < nmod; ++m) {
            GetModuleInformation(process, modules[m], &info[m], sizeof info[m]);
            GetModuleFileNameExA(process, modules[m], names[m], MAX_PATH);
        }
    }
    for (int m = 0; m < nmod; ++m) {
        printf("module %p %s\n", info[m].lpBaseOfDll, names[m]);
    }
    int used = 0;
    for (int i = 0; i < TABLE; ++i) {
        if (table[i].count) table[used++] = table[i];
    }
    qsort(table, used, sizeof(Slot), by_count);
    for (int t = 0; t < count; ++t) {
        printf("thread %lu %s: %llu samples, %llu idle\n", threads[t].id, threads[t].name,
               threads[t].samples, threads[t].idle);
    }
    for (int i = 0; i < used; ++i) {
        const unsigned long long rip = table[i].rip;
        const char* mod = "?";
        unsigned long long off = rip;
        for (int m = 0; m < nmod; ++m) {
            const unsigned long long base = (unsigned long long)info[m].lpBaseOfDll;
            if (rip >= base && rip < base + info[m].SizeOfImage) {
                mod = strrchr(names[m], '\\') ? strrchr(names[m], '\\') + 1 : names[m];
                off = rip - base;
                break;
            }
        }
        if (table[i].caller) {
            printf("sample %s %s+0x%llx caller+0x%llx %u chain", threads[table[i].thread].name, mod,
                   off, table[i].caller - main_lo, table[i].count);
            for (int k = 0; k < 6 && table[i].chain[k]; ++k) printf(" 0x%llx", table[i].chain[k] - main_lo);
            printf("\n");
        } else {
            printf("sample %s %s+0x%llx %u\n", threads[table[i].thread].name, mod, off, table[i].count);
        }
    }
    return 0;
}
