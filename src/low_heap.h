/* macOS: the runtime's C heap calls go to the low heap (src/low_heap.c), so pointers the
 * guest receives stay below 1 TiB. Included by runtime.h for C files only. */
#ifndef BB_LOW_HEAP_H
#define BB_LOW_HEAP_H
#if defined(__APPLE__) && !defined(__cplusplus) && !defined(BB_LOW_HEAP_IMPL)
#include <stddef.h>
void *bb_low_malloc(size_t size);
void *bb_low_calloc(size_t count, size_t size);
void *bb_low_realloc(void *p, size_t size);
void *bb_low_aligned_alloc(size_t alignment, size_t size);
void bb_low_free(void *p);
char *bb_low_strdup(const char *s);
int bb_low_posix_memalign(void **out, size_t alignment, size_t size);
#define malloc bb_low_malloc
#define calloc bb_low_calloc
#define realloc bb_low_realloc
#define aligned_alloc bb_low_aligned_alloc
#define free bb_low_free
#define strdup bb_low_strdup
#define posix_memalign bb_low_posix_memalign
#endif
#endif
