// Cheat-Engine-style search in the running game (bb-probe) for a value that changes only while the
// player does something (the character creator's model turned with the stick): read from outside
// with ReadProcessMemory, candidates kept in a file between passes.
//   memscan changing <candidates>   - every 4-byte float in the guest's writable memory that changes
//                                      over ~40 ms now (the player is turning the model)
//   memscan changing <candidates> keep  - of the candidates, those that change now
//   memscan still <candidates>      - of the candidates, those that do not change now (stick let go)
//   memscan show <candidates> [n]   - the first n with their values now
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t Lo = 0x800000000ull, Hi = 0x1300000000ull;

static DWORD find_game(void) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 e = {.dwSize = sizeof e};
    DWORD pid = 0;
    for (BOOL ok = Process32First(snap, &e); ok; ok = Process32Next(snap, &e))
        if (_stricmp(e.szExeFile, "bb-probe.exe") == 0) pid = e.th32ProcessID;
    CloseHandle(snap);
    return pid;
}

static int plausible(float v) { return isfinite(v) && fabsf(v) < 100000.0f; }

typedef struct { uint64_t address; float value; } Candidate;

static Candidate *load(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) { *count = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    Candidate *c = malloc(size ? size : 1);
    *count = fread(c, sizeof *c, size / sizeof *c, f);
    fclose(f);
    return c;
}

static void save(const char *path, const Candidate *c, size_t count) {
    FILE *f = fopen(path, "wb");
    fwrite(c, sizeof *c, count, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: memscan changing|still|show <candidates> [keep|n]\n"); return 2; }
    const DWORD pid = find_game();
    if (!pid) { fprintf(stderr, "no bb-probe\n"); return 1; }
    HANDLE p = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, pid);
    if (!p) { fprintf(stderr, "cannot open the game (%lu)\n", GetLastError()); return 1; }
    const char *mode = argv[1], *path = argv[2];

    if (strcmp(mode, "peek") == 0) {
        // memscan peek <hex address>...: the floats there now.
        for (int i = 2; i < argc; ++i) {
            const uint64_t address = strtoull(argv[i], NULL, 16);
            float v = 0;
            ReadProcessMemory(p, (void *)address, &v, 4, NULL);
            printf("%#llx: %.7g\n", (unsigned long long)address, v);
        }
        return 0;
    }
    if (strcmp(mode, "find") == 0) {
        // memscan find <value> <tolerance>: every float that close to the value, now.
        const float want = (float)atof(argv[2]), tolerance = argc > 3 ? (float)atof(argv[3]) : 1e-4f;
        const size_t chunk = 16 << 20;
        unsigned char *a = malloc(chunk);
        int shown = 0;
        for (uint64_t at = Lo; at < Hi && shown < 200;) {
            MEMORY_BASIC_INFORMATION m;
            if (!VirtualQueryEx(p, (void *)at, &m, sizeof m)) break;
            const uint64_t base = (uint64_t)m.BaseAddress, end = base + m.RegionSize;
            if (m.State == MEM_COMMIT && (m.Protect & (PAGE_READWRITE | PAGE_WRITECOPY)) && !(m.Protect & PAGE_GUARD)) {
                for (uint64_t c = base; c < end && c < Hi; c += chunk) {
                    const size_t n = (size_t)((end - c) < chunk ? (end - c) : chunk);
                    SIZE_T got = 0;
                    if (!ReadProcessMemory(p, (void *)c, a, n, &got)) continue;
                    for (size_t i = 0; i + 4 <= n && shown < 200; i += 4) {
                        float v;
                        memcpy(&v, a + i, 4);
                        if (fabsf(v - want) <= tolerance) { printf("%#llx: %.7g\n", (unsigned long long)(c + i), v); ++shown; }
                    }
                }
            }
            at = end > at ? end : at + 0x1000;
        }
        return 0;
    }
    if (strcmp(mode, "changing") == 0 && !(argc > 3 && strcmp(argv[3], "keep") == 0)) {
        // First pass: the whole writable guest memory read twice, 40 ms apart.
        size_t cap = 1 << 20, count = 0;
        Candidate *out = malloc(cap * sizeof *out);
        uint64_t scanned = 0;
        const size_t chunk = 16 << 20;
        unsigned char *a = malloc(chunk), *b = malloc(chunk);
        for (uint64_t at = Lo; at < Hi;) {
            MEMORY_BASIC_INFORMATION m;
            if (!VirtualQueryEx(p, (void *)at, &m, sizeof m)) break;
            const uint64_t base = (uint64_t)m.BaseAddress, end = base + m.RegionSize;
            if (m.State == MEM_COMMIT && (m.Protect & (PAGE_READWRITE | PAGE_WRITECOPY)) && !(m.Protect & PAGE_GUARD)) {
                for (uint64_t c = base; c < end && c < Hi; c += chunk) {
                    const size_t n = (size_t)((end - c) < chunk ? (end - c) : chunk);
                    SIZE_T got1 = 0, got2 = 0;
                    if (!ReadProcessMemory(p, (void *)c, a, n, &got1)) continue;
                    Sleep(40);
                    if (!ReadProcessMemory(p, (void *)c, b, n, &got2)) continue;
                    scanned += n;
                    for (size_t i = 0; i + 4 <= n; i += 4) {
                        float va, vb;
                        memcpy(&va, a + i, 4);
                        memcpy(&vb, b + i, 4);
                        if (va != vb && plausible(va) && plausible(vb)) {
                            if (count == cap) { cap *= 2; out = realloc(out, cap * sizeof *out); }
                            out[count++] = (Candidate){c + i, vb};
                        }
                    }
                }
            }
            at = end > at ? end : at + 0x1000;
        }
        save(path, out, count);
        printf("scanned %.1f GiB: %zu changing\n", scanned / 1073741824.0, count);
        return 0;
    }

    size_t count = 0;
    Candidate *c = load(path, &count);
    if (strcmp(mode, "show") == 0) {
        const size_t n = argc > 3 ? (size_t)atoi(argv[3]) : 40;
        for (size_t i = 0; i < count && i < n; ++i) {
            float v = 0;
            ReadProcessMemory(p, (void *)c[i].address, &v, 4, NULL);
            printf("%#llx: %g (was %g)\n", (unsigned long long)c[i].address, v, c[i].value);
        }
        printf("%zu candidates\n", count);
        return 0;
    }
    // Filter passes over the candidates: two reads 40 ms apart (batched by page).
    const int want_change = strcmp(mode, "changing") == 0;
    float *first = malloc(count * sizeof *first + 1);
    for (size_t i = 0; i < count; ++i) ReadProcessMemory(p, (void *)c[i].address, &first[i], 4, NULL);
    Sleep(40);
    size_t kept = 0;
    for (size_t i = 0; i < count; ++i) {
        float v = 0;
        if (!ReadProcessMemory(p, (void *)c[i].address, &v, 4, NULL)) continue;
        const int changed = v != first[i];
        if (changed == want_change && plausible(v)) c[kept++] = (Candidate){c[i].address, v};
    }
    save(path, c, kept);
    printf("%zu of %zu kept (%s)\n", kept, count, want_change ? "changing" : "still");
    return 0;
}
