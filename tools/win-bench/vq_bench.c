// VirtualQuery cost inside a large uniform section view: RegionSize is the run of pages with the
// same attributes from the address on, which the kernel finds by walking them.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

int main(void) {
    const size_t size = 3ull << 30;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT,
                                        (DWORD)(size >> 32), (DWORD)size, NULL);
    unsigned char* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
    for (int touched = 0; touched < 2; ++touched) {
        if (touched) memset(view, 1, size);
        const size_t offsets[] = {0, 256ull << 20, 1ull << 30, 2ull << 30, size - (1 << 20)};
        for (int i = 0; i < 5; ++i) {
            MEMORY_BASIC_INFORMATION info;
            const double t0 = now_ms();
            VirtualQuery(view + offsets[i], &info, sizeof(info));
            const double t1 = now_ms();
            printf("%s VirtualQuery at +%4zu MiB: %.3f ms, region %zu MiB\n", touched ? "touched  " : "untouched",
                   offsets[i] >> 20, t1 - t0, info.RegionSize >> 20);
        }
    }
    // A protected 1 MiB in the middle: the queries before it stop there.
    DWORD old;
    VirtualProtect(view + (1ull << 30), 1 << 20, PAGE_READONLY, &old);
    MEMORY_BASIC_INFORMATION info;
    const double t0 = now_ms();
    VirtualQuery(view, &info, sizeof(info));
    printf("with a read-only MiB at +1024: VirtualQuery at 0: %.3f ms, region %zu MiB\n", now_ms() - t0,
           info.RegionSize >> 20);
    return 0;
}
