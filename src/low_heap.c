/* Low heap for macOS: host objects the guest receives (thread handles, TLS blocks, mutexes,
 * semaphores...) must live below 1 TiB, because PS4 code packs pointers into 40 bits. On Linux
 * the non-PIE brk heap is already there (probe.c mallopt); macOS puts malloc at ~96 TiB.
 *
 * The runtime's C files call malloc/calloc/realloc/aligned_alloc/free through macros in
 * platform.h that land here. Design: an 8 GiB region from runtime_low_map, cut into 1 MiB
 * chunks; each chunk serves one power-of-two size class (16 B .. 512 KiB), so blocks are
 * naturally aligned to their size and need no header; a side table maps chunk -> class.
 * Larger requests take runs of whole chunks. Freed blocks go to per-class free lists.
 * Pointers outside the region (from strdup, realpath, opendir, ...) go to the system free. */
#ifdef __APPLE__
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define BB_LOW_HEAP_IMPL
#include "runtime.h"

#define CHUNK_SHIFT 20
#define CHUNK ((size_t)1 << CHUNK_SHIFT)
#define CHUNKS 8192 /* 8 GiB, committed only when touched */
#define MIN_SHIFT 4
#define MAX_SMALL_SHIFT 19 /* 512 KiB; larger requests use whole chunks */
#define CLASSES (MAX_SMALL_SHIFT - MIN_SHIFT + 1)
#define LARGE 0xFF
#define FREE_CHUNK 0xFE

static unsigned char *base;
static unsigned char chunk_class[CHUNKS]; /* class index, LARGE (run head/body) or FREE_CHUNK */
static uint32_t run_length[CHUNKS];        /* for LARGE run heads: chunks in the run */
static void *free_list[CLASSES];
static size_t next_chunk;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int init_locked(void) {
    if (base) return 1;
    base = runtime_low_map(CHUNKS * CHUNK, PROT_READ | PROT_WRITE);
    if (!base) { fputs("STOP: low heap: cannot reserve 8 GiB below 1 TiB\n", stderr); exit(21); }
    if ((uintptr_t)base + CHUNKS * CHUNK > (UINT64_C(1) << 40)) {
        fprintf(stderr, "STOP: low heap at %p is above 1 TiB\n", (void *)base); exit(21);
    }
    memset(chunk_class, FREE_CHUNK, sizeof(chunk_class));
    return 1;
}
/* Reserve before main: the first allocation may happen while runtime_memory holds its lock. */
__attribute__((constructor)) static void low_heap_init(void) {
    pthread_mutex_lock(&lock);
    init_locked();
    pthread_mutex_unlock(&lock);
}
static int in_heap(const void *p) {
    return base && (const unsigned char *)p >= base && (const unsigned char *)p < base + CHUNKS * CHUNK;
}
static int class_of(size_t size) {
    int shift = MIN_SHIFT;
    while (((size_t)1 << shift) < size) ++shift;
    return shift - MIN_SHIFT;
}
/* First run of `count` free chunks (chunks are reused after large frees). */
static long take_chunks(size_t count) {
    for (size_t i = 0, run = 0; i < next_chunk; ++i) {
        run = chunk_class[i] == FREE_CHUNK ? run + 1 : 0;
        if (run == count) return (long)(i + 1 - count);
    }
    if (next_chunk + count > CHUNKS) return -1;
    long first = (long)next_chunk;
    next_chunk += count;
    return first;
}
static void *alloc_locked(size_t size) {
    init_locked();
    if (!size) size = 1;
    if (size > ((size_t)1 << MAX_SMALL_SHIFT)) {
        size_t count = (size + CHUNK - 1) >> CHUNK_SHIFT;
        long first = take_chunks(count);
        if (first < 0) return NULL;
        for (size_t i = 0; i < count; ++i) chunk_class[first + i] = LARGE;
        run_length[first] = (uint32_t)count;
        return base + ((size_t)first << CHUNK_SHIFT);
    }
    int c = class_of(size);
    if (!free_list[c]) {
        long chunk = take_chunks(1);
        if (chunk < 0) return NULL;
        chunk_class[chunk] = (unsigned char)c;
        size_t block = (size_t)1 << (c + MIN_SHIFT);
        unsigned char *start = base + ((size_t)chunk << CHUNK_SHIFT);
        for (size_t at = CHUNK; at >= block; at -= block) { /* push in reverse: ascending pops */
            void **b = (void **)(start + at - block);
            *b = free_list[c];
            free_list[c] = b;
        }
    }
    void **b = free_list[c];
    free_list[c] = *b;
    return b;
}
static size_t block_size(const void *p) {
    size_t chunk = (size_t)((const unsigned char *)p - base) >> CHUNK_SHIFT;
    unsigned char c = chunk_class[chunk];
    if (c == LARGE) return (size_t)run_length[chunk] << CHUNK_SHIFT;
    return (size_t)1 << (c + MIN_SHIFT);
}
static void free_locked(void *p) {
    size_t chunk = (size_t)((unsigned char *)p - base) >> CHUNK_SHIFT;
    unsigned char c = chunk_class[chunk];
    if (c == LARGE) {
        size_t count = run_length[chunk];
        madvise(p, count << CHUNK_SHIFT, MADV_FREE); /* give the pages back, keep the range */
        for (size_t i = 0; i < count; ++i) chunk_class[chunk + i] = FREE_CHUNK;
        return;
    }
    void **b = p;
    *b = free_list[c];
    free_list[c] = b;
}

void *bb_low_malloc(size_t size) {
    pthread_mutex_lock(&lock);
    void *p = alloc_locked(size);
    pthread_mutex_unlock(&lock);
    return p;
}
void *bb_low_calloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) return NULL;
    void *p = bb_low_malloc(count * size);
    if (p) memset(p, 0, count * size);
    return p;
}
void *bb_low_aligned_alloc(size_t alignment, size_t size) {
    /* Blocks are aligned to their power-of-two size, and large runs to 1 MiB. */
    if (alignment > CHUNK) return NULL;
    return bb_low_malloc(size < alignment ? alignment : size);
}
void bb_low_free(void *p) {
    if (!p) return;
    if (!in_heap(p)) { free(p); return; }
    pthread_mutex_lock(&lock);
    free_locked(p);
    pthread_mutex_unlock(&lock);
}
void *bb_low_realloc(void *p, size_t size) {
    if (!p) return bb_low_malloc(size);
    if (!in_heap(p)) return realloc(p, size); /* system allocation: stays a system allocation */
    if (!size) { bb_low_free(p); return NULL; }
    size_t old = block_size(p);
    if (size <= old && size > old / 2) return p;
    void *q = bb_low_malloc(size);
    if (q) { memcpy(q, p, old < size ? old : size); bb_low_free(p); }
    return q;
}
char *bb_low_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = bb_low_malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}
int bb_low_posix_memalign(void **out, size_t alignment, size_t size) {
    void *p = bb_low_aligned_alloc(alignment, size);
    if (!p) return 12; /* ENOMEM */
    *out = p;
    return 0;
}
#endif
