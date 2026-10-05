/* tls_gs_probe: checks the macOS guest-TLS scheme (src/runtime_thread.c, __APPLE__) before
 * anything else is ported. On x86-64 macOS (native or Rosetta 2) GS points at the thread's
 * pthread TSD array, so `mov rax, gs:[key*8]` must equal pthread_getspecific(key):
 *   1. compiled code, on the main thread and on a second thread with a different value;
 *   2. the exact 9-byte instruction the port writes into the game image, placed in freshly
 *      mmap'd memory and executed there (the game's code is mapped the same way at run time).
 * Build and run on the Mac:
 *   clang -arch x86_64 -O1 -o tls_gs_probe tools/macos/tls_gs_probe.c && ./tls_gs_probe
 * Expected last line: "tls_gs_probe: OK". Under Rosetta, `sysctl sysctl.proc_translated`
 * from an x86-64 shell prints 1. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static pthread_key_t key;
static void *(*generated)(void);
static int failures;

static void *read_gs(uint64_t offset) {
    void *value;
    __asm__ volatile("movq %%gs:(%1), %0" : "=r"(value) : "r"(offset));
    return value;
}

static void check(const char *where, void *expected) {
    if (pthread_setspecific(key, expected)) { printf("%s: pthread_setspecific failed\n", where); ++failures; return; }
    void *compiled = read_gs((uint64_t)key * 8);
    void *mapped = generated();
    printf("%s: key=%lu expected=%p compiled=%p mapped-code=%p\n", where, (unsigned long)key, expected, compiled, mapped);
    if (compiled != expected || mapped != expected) ++failures;
}

static void *second_thread(void *unused) {
    (void)unused;
    static uint64_t other_tcb[4];
    check("second thread", other_tcb);
    return NULL;
}

int main(void) {
#if !defined(__x86_64__)
    puts("tls_gs_probe: build with -arch x86_64 (the port runs as x86-64 under Rosetta)");
    return 2;
#endif
    if (pthread_key_create(&key, NULL)) { puts("pthread_key_create failed"); return 1; }
    /* mov rax, gs:[disp32]; ret  - the port's rewritten guest instruction plus a return. */
    unsigned char code[10] = {0x65, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0, 0xc3};
    uint32_t displacement = (uint32_t)((uint64_t)key * 8);
    memcpy(code + 5, &displacement, 4);
    long page = sysconf(_SC_PAGESIZE);
    unsigned char *mem = mmap(NULL, (size_t)page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mem == MAP_FAILED) { puts("mmap failed"); return 1; }
    memcpy(mem, code, sizeof(code));
    if (mprotect(mem, (size_t)page, PROT_READ | PROT_EXEC)) { puts("mprotect RX failed"); return 1; }
    generated = (void *(*)(void))mem;

    static uint64_t main_tcb[4];
    check("main thread", main_tcb);
    pthread_t thread;
    pthread_create(&thread, NULL, second_thread, NULL);
    pthread_join(thread, NULL);
    check("main thread again", main_tcb);
    puts(failures ? "tls_gs_probe: FAILED" : "tls_gs_probe: OK");
    return failures != 0;
}
