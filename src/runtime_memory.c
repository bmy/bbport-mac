// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* PS4 virtual memory: direct (physical) pool, flexible memory, reservations,
 * protection changes and queries. One sparse memfd backs the whole direct
 * pool, so any mapping of any physical range aliases the same storage.
 * Guest addresses are placed below 1 TiB (PS4 user range): GPU descriptors
 * encode 40-bit addresses, so host-default 0x7f... addresses would not fit. */
#define _GNU_SOURCE
#include "runtime.h"
#include "platform.h"
#include "../gpu/shim/bb_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
/* sceKernelGetDirectMemorySize on retail PS4: 5056 MiB. BB_DMEM_MB raises it (the resolution
 * patches above 1080p need about 4 GiB more; run.sh sets it). */
static uint64_t game_pool_bytes(void) {
    static uint64_t size;
    if (!size) {
        const char *env = bb_env("BB_DMEM_MB");
        uint64_t mb = env ? strtoull(env, NULL, 10) : 0;
        if (mb < 5056 || mb > 16384) mb = 5056;
        size = mb * 1024 * 1024;
    }
    return size;
}
/* bbport: direct memory past the game's size, for the port's own guest allocations
 * (runtime_extra_direct_map: the render resource cache's larger backing). The game neither sees
 * nor allocates it. BB_DMEM_EXTRA_MB (0..2048, default 512). */
static uint64_t extra_pool_bytes(void) {
    static int64_t size=-1;
    if (size<0) {
        const char *env = bb_env("BB_DMEM_EXTRA_MB");
        uint64_t mb = env ? strtoull(env, NULL, 10) : 512;
        if (mb > 2048) mb = 2048;
        size = (int64_t)(mb * 1024 * 1024);
    }
    return (uint64_t)size;
}
static uint64_t pool_size_bytes(void) { return game_pool_bytes() + extra_pool_bytes(); }
#define POOL_SIZE pool_size_bytes()
#define GAME_POOL_SIZE game_pool_bytes()
#define FLEXIBLE_SIZE (UINT64_C(448) * 1024 * 1024)
#define PAGE UINT64_C(16384)
#define USER_MIN BB_USER_MIN
#define USER_MAX UINT64_C(0xfc00000000)
#define LIMIT 4096
#define INVALID ((int32_t)UINT32_C(0x80020016))
#define NO_MEMORY ((int32_t)UINT32_C(0x8002000c))
#define ACCESS ((int32_t)UINT32_C(0x8002000d))
#define NO_SPACE ((int32_t)UINT32_C(0x80020023))
#define MAP_FIXED_FLAG 0x10
#define MAP_NO_OVERWRITE 0x80
enum { KIND_RESERVED=1, KIND_DIRECT, KIND_FLEXIBLE };
typedef struct { uint64_t start, size; int type, used; } Block;
/* private_mem (bbport): not in the pool (macOS executable memory, private regions): not shared
 * with the GPU process. */
typedef struct { uintptr_t start, end; int kind, prot, type; uint64_t phys; int private_mem; } Vma;
typedef struct {
    uintptr_t start, end; uint64_t offset; int32_t protection, memory_type;
    uint32_t flags; char name[32];
} VirtualQueryInfo; /* flags: flexible, direct, stack, pooled, committed */
typedef struct { void *start; uint64_t offset, length; int8_t prot, type; int16_t reserved; int32_t operation; } BatchEntry;
_Static_assert(sizeof(VirtualQueryInfo)==72,"PS4 virtual query info layout");
_Static_assert(sizeof(BatchEntry)==32,"PS4 batch map entry layout");

/* Region table lock. Lookups (GPU per-draw queries, VirtualQuery, GPU page
 * protection) share it; map/unmap/protect take it exclusively. Both nest: code
 * under the lock may write guest memory, and a GPU-tracked page then faults into
 * runtime_memory_gpu_protect on the same thread. Readers nest because glibc's
 * default rwlock prefers readers; a thread holding it exclusively skips relocking. */
static pthread_rwlock_t lock=PTHREAD_RWLOCK_INITIALIZER;
static __thread unsigned exclusive_depth;
/* bbport (native GPU process, docs/macos-native-gpu.md): changes to the mapping table, for the
 * GPU process's copy of guest memory. Queued under the lock and passed on in order once the
 * outermost writer has left it (the hook sends a message that may wait for the GPU process,
 * whose own calls into this file need the lock), before the operation returns to the guest. */
typedef void (*MirrorHook)(int op, uintptr_t address, uint64_t size, uint64_t phys, int kind,
                           int prot, int type, int shared);
enum { MIRROR_MAP, MIRROR_UNMAP, MIRROR_PROTECT };
typedef struct { int op; uintptr_t address; uint64_t size, phys; int kind, prot, type, shared; } MirrorEvent;
static MirrorHook mirror_hook;
static MirrorEvent *mirror_events; static size_t mirror_count, mirror_capacity;
static __thread int mirror_queued; /* this thread queued events: it sends them before returning */
static __thread unsigned read_depth; /* runtime_memory_lock_held */
static pthread_mutex_t mirror_send=PTHREAD_MUTEX_INITIALIZER;
static void flush_mirror(void);
/* Odd while a writer holds the lock; readers' region caches are valid for one even value. */
static uint64_t table_generation;
/* Held or waited for longer than BB_MEMORY_LOCK_REPORT_MS (20; 0: off): printed with the code that
 * took it (bb-probe offsets), at most 200 lines. The GPU side's per-draw range queries wait for a
 * writer: 165-220 ms stalls of stage A while areas load (GTX 1050 laptop, Windows). */
static uint64_t lock_report_ns=~(uint64_t)0, lock_held_since;
static const void *lock_writer;
static unsigned lock_reports;
static uint64_t lock_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec; }
static void lock_site(const void *site, char *out, size_t size) {
    Dl_info info; const char *module="?"; uintptr_t off=(uintptr_t)site;
    if (site && dladdr(site,&info) && info.dli_fbase) {
        off=(uintptr_t)site-(uintptr_t)info.dli_fbase; if (info.dli_fname) module=info.dli_fname;
    }
    const char *base=module;
    for (const char *c=module;*c;++c) if (*c=='/' || *c=='\\') base=c+1;
    snprintf(out,size,"%s+%#lx",base,(unsigned long)off);
}
/* `site`: the code that took the lock; `writer`: who held it meanwhile (read waits). */
static void lock_report(const char *what, uint64_t ns, const void *site, const void *writer) {
    if (__atomic_add_fetch(&lock_reports,1,__ATOMIC_RELAXED)>200) return;
    char a[160], b[160];
    lock_site(site,a,sizeof a);
    if (writer) { lock_site(writer,b,sizeof b); printf("Runtime: memory map %s %.1f ms at %s (held by %s)\n",what,ns/1e6,a,b); }
    else printf("Runtime: memory map %s %.1f ms at %s\n",what,ns/1e6,a);
}
static uint64_t lock_threshold(void) {
    if (lock_report_ns==~(uint64_t)0) {
        const char *e=bb_env("BB_MEMORY_LOCK_REPORT_MS");
        lock_report_ns=(uint64_t)(e && *e ? atof(e) : 20.0)*1000000u;
    }
    return lock_report_ns;
}
static void write_lock(void) {
    if (!exclusive_depth++) {
        const uint64_t t0=lock_threshold() ? lock_now_ns() : 0;
        pthread_rwlock_wrlock(&lock); __atomic_add_fetch(&table_generation,1,__ATOMIC_ACQ_REL);
        if (t0) {
            lock_held_since=lock_now_ns(); lock_writer=__builtin_return_address(0);
            if (lock_held_since-t0>=lock_threshold()) lock_report("write wait",lock_held_since-t0,lock_writer,NULL);
        }
    }
}
static void write_unlock(void) {
    if (!--exclusive_depth) {
        const uint64_t since=lock_held_since; const void *writer=lock_writer;
        __atomic_add_fetch(&table_generation,1,__ATOMIC_RELEASE); pthread_rwlock_unlock(&lock);
        if (since && lock_threshold()) { const uint64_t held=lock_now_ns()-since; if (held>=lock_threshold()) lock_report("held",held,writer,NULL); }
        /* bbport: another thread may be sending this thread's events right now: flush_mirror
         * waits for it (mirror_send), so they are out before the guest goes on. */
        if (mirror_queued) { mirror_queued=0; flush_mirror(); }
    }
}
static void read_lock(void) {
    ++read_depth;
    if (exclusive_depth) return;
    if (!pthread_rwlock_tryrdlock(&lock)) return;
    const uint64_t t0=lock_threshold() ? lock_now_ns() : 0;
    pthread_rwlock_rdlock(&lock);
    if (t0) { const uint64_t w=lock_now_ns()-t0; if (w>=lock_threshold()) lock_report("read wait",w,__builtin_return_address(0),lock_writer); }
}
static void read_unlock(void) { --read_depth; if (!exclusive_depth) pthread_rwlock_unlock(&lock); }
/* bbport (native GPU process): this thread holds the lock (it faulted inside the runtime). */
int runtime_memory_lock_held(void) { return exclusive_depth || read_depth; }
static Block blocks[LIMIT];
static Vma *vmas; static size_t vma_count, vma_capacity;
static int pool_fd=-1;
/* bbport: direct memory in GPU-visible system memory the GPU library allocates (dma-buf chunks,
 * the PC memory model): the GPU reads the game's data in place instead of the CPU copying it
 * into staging buffers first. Chunks are allocated when the game allocates direct memory in them;
 * a chunk the GPU library cannot provide stays in the memfd. */
#define CHUNK (UINT64_C(256) << 20)
typedef int (*GuestChunkAlloc)(uint64_t phys, uint64_t size);
static GuestChunkAlloc chunk_alloc;
static int *chunk_fds; /* per CHUNK of direct memory: -1 not decided, -2 memfd, -3 memfd the GPU
                        * imported (host memory: VK_EXT_external_memory_host), else a dma-buf fd */
#define CHUNK_IMPORTED (-3)
/* Released memory in a chunk the GPU uses is cleared, not punched out: the GPU holds those pages. */
#define GPU_CHUNK(fd) ((fd)>=0 || (fd)==CHUNK_IMPORTED || (fd)==-4)
/* bbport: memfd ranges the GPU imported (lazy guest memory: udmabuf or host import pins their
 * pages): released memory there is cleared, not punched out (the GPU would keep the old pages). */
#define PIN_GRANULE (UINT64_C(2) << 20)
static unsigned char *pinned; /* per PIN_GRANULE of the whole memfd (flexible memory too) */
/* A release's check and punch, and an import's note, one at a time: an import noted after its
 * pages were pinned, or between a release's check and its punch, kept the old pages while the
 * memfd got new ones (the GPU's writes there never seen by the CPU again: a label the guest
 * waited for forever, 2026-10-23). */
static pthread_mutex_t pin_mutex=PTHREAD_MUTEX_INITIALIZER;
/* bbport: lazy guest memory: the game's direct memory stays in the memfd (the GPU imports what it
 * uses); only the port's own direct memory past it gets chunks (runtime_extra_direct_map). */
static int lazy_game;
/* Drivers whose dma-buf maps at offset 0 only (NVIDIA): a chunk per direct allocation instead of
 * the CHUNK grid. The game maps each of its allocations once, whole, so its mappings and the
 * backing view start at offset 0 of their chunk. */
static unsigned char *backing_base; /* second view of the pool: host writes bypass guest/GPU page protection */
static int whole_chunks;
typedef struct { uint64_t phys, size; int fd; uintptr_t va; } Region;
/* bbport: private regions (BbGuestMemory's private mode, Windows drivers): a direct allocation's
 * memory is private memory placed where the game maps it, its only view, imported by the GPU
 * (drivers import VirtualAlloc'd memory, not views of sections; on Linux anonymous memory). The
 * runtime's own accesses (backing view) go through that address. The game maps each allocation
 * once, whole, and keeps it: a second mapping elsewhere stops the game. */
#define CHUNK_PRIVATE (-4)
typedef int (*GuestChunkImport)(uint64_t phys, uint64_t size, void *pointer);
static GuestChunkImport private_import;
#define MAX_REGIONS 256
static Region regions[MAX_REGIONS];
static size_t region_count;
/* The region holding phys (sorted by phys), or NULL; *next: start of the next region above. */
static const Region *region_at(uint64_t phys, uint64_t *next) {
    *next=POOL_SIZE;
    for (size_t i=0;i<region_count;++i) {
        if (phys<regions[i].phys) { *next=regions[i].phys; return NULL; }
        if (phys<regions[i].phys+regions[i].size) return &regions[i];
    }
    return NULL;
}
/* The runtime's address of direct memory phys: its private region's guest address, else the
 * backing view. */
static unsigned char *backing_at(uint64_t phys) {
    if (private_import && phys<POOL_SIZE) {
        uint64_t next;
        const Region *r=region_at(phys,&next);
        if (r && r->fd==CHUNK_PRIVATE) return (unsigned char *)(r->va+(phys-r->phys));
    }
    return backing_base+phys;
}
/* A chunk the GPU library already made for [phys,+size) (runtime_memory_commit: its import takes
 * ~170 ms on NVIDIA, which the write lock must not span: every GPU query of the memory map waited
 * for it, stage A stood still while a heap grew). add_region takes it instead of asking again. */
typedef struct { int valid, fd; uint64_t phys, size; } PreChunk;
static __thread PreChunk pre_chunks[8];
static void add_region(uint64_t phys, uint64_t size) {
    if (region_count==MAX_REGIONS) return;
    int fd=0, found=0;
    for (size_t i=0;i<sizeof(pre_chunks)/sizeof(pre_chunks[0]);++i)
        if (pre_chunks[i].valid && pre_chunks[i].phys==phys && pre_chunks[i].size==size) { fd=pre_chunks[i].fd; pre_chunks[i].valid=0; found=1; break; }
    if (!found) fd=chunk_alloc(phys,size);
    if (fd>=0 && mmap(backing_base+phys,size,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0)==MAP_FAILED) {
        close(fd);
        fd=-1;
    }
    size_t i=region_count;
    while (i>0 && regions[i-1].phys>phys) { regions[i]=regions[i-1]; --i; }
    regions[i]=(Region){phys,size,fd>=0 || fd==CHUNK_IMPORTED ? fd : -2,0};
    ++region_count;
}
#define FLEX_SPAN (UINT64_C(1024) * 1024 * 1024)
static uint64_t host_span; /* bbport: pool bytes after flexible memory (runtime_memory_host_pool) */
static uint64_t flex_bitmap[FLEX_SPAN/PAGE/64];
/* GPU hooks (bbgpu): notified outside the lock, in order, after each operation. */
typedef void (*GpuRange)(uintptr_t address, uint64_t size);
static GpuRange hook_map, hook_unmap, hook_invalidate;
static GpuRange hook_note_write; /* bbport: data written into GPU memory (runtime_memory_note_write) */
enum { HOOK_MAP, HOOK_UNMAP, HOOK_INVALIDATE };
typedef struct { int kind; uintptr_t address; uint64_t size; } PendingHook;
static PendingHook pending[64]; static size_t pending_count;
static size_t allocations, maps, flexible_maps, protects, queries;
static uint64_t live_bytes, flexible_bytes;

static int valid_alignment(uint64_t a) { return a >= PAGE && a <= POOL_SIZE && !(a & (a-1)); }
static uint64_t align_up(uint64_t n, uint64_t a) { return (n+a-1) & ~(a-1); }
/* bbport (native GPU process): a change to the mapping table for the GPU process (lock held).
 * A failed allocation stops the process: the GPU process's view would be wrong. */
static void queue_mirror(int op, uintptr_t address, uint64_t size, uint64_t phys, int kind, int prot,
                         int type, int shared) {
    if (!mirror_hook || !size) return;
    if (mirror_count==mirror_capacity) {
        size_t capacity=mirror_capacity ? mirror_capacity*2 : 64;
        MirrorEvent *next=realloc(mirror_events,capacity*sizeof(*next));
        if (!next) { fputs("STOP: GPU process memory mirror queue\n",stderr); exit(21); }
        mirror_events=next; mirror_capacity=capacity;
    }
    mirror_events[mirror_count]=(MirrorEvent){op,address,size,phys,kind,prot,type,shared};
    __atomic_store_n(&mirror_count,mirror_count+1,__ATOMIC_RELEASE);
    mirror_queued=1;
}
/* Outside the lock, by the thread that queued (after another one's send of them, if any, has
 * finished): a guest thread's next GPU submission cannot overtake its own mapping. */
static void flush_mirror(void) {
    pthread_mutex_lock(&mirror_send);
    for (;;) {
        pthread_rwlock_wrlock(&lock);
        size_t n=mirror_count;
        MirrorEvent *events=mirror_events;
        if (!n) { pthread_rwlock_unlock(&lock); break; }
        mirror_events=NULL; mirror_count=0; mirror_capacity=0;
        pthread_rwlock_unlock(&lock);
        for (size_t i=0;i<n;++i)
            mirror_hook(events[i].op,events[i].address,events[i].size,events[i].phys,events[i].kind,
                        events[i].prot,events[i].type,events[i].shared);
        free(events);
    }
    pthread_mutex_unlock(&mirror_send);
}
#ifdef __APPLE__
/* bbport (macOS): no hole punching in shared memory objects: released pool pages are zeroed
 * through the backing view (they stay resident). */
static void pool_discard(uint64_t phys, uint64_t size) { memset(backing_base+phys,0,size); }
#endif
/* The rest by topic, in src/runtime_memory/ (one translation unit with this file). */
#include "runtime_memory/backing.c"
#include "runtime_memory/vma.c"
#include "runtime_memory/kernel_api.c"
#include "runtime_memory/low.c"
#include "runtime_memory/gpu_interface.c"
#include "runtime_memory/heap_commit.c"
#include "runtime_memory/gpu_traps.c"
/* bbport (macOS): like runtime_low_map, from the top of the low region down. The low heap
 * (low_heap.c, reserved before main) takes its 8 GiB there, so the game image is still the
 * first mapping at LOW_MIN, as on Linux: upstream code uses image base 0x800000000. */
void *runtime_low_map_high(size_t size, int prot) {
    static uintptr_t high_next=BB_LOW_MAX;
    size=align_up(size,PAGE);
    write_lock();
    void *p=MAP_FAILED;
    while (high_next>=low_next+size+PAGE) {
        high_next-=size+PAGE; /* unmapped gap above catches overruns */
        p=bb_map_noreplace((void *)high_next,size,prot);
        if (p!=MAP_FAILED) break;
    }
    if (p!=MAP_FAILED) low_record((uintptr_t)p,(uintptr_t)p+size,prot);
    write_unlock();
    return p==MAP_FAILED ? NULL : p;
}
/* bbport (native GPU process): the pool gets `host_bytes` more after flexible memory for the GPU
 * process split (its control block, shared image data); must be called before the pool exists
 * (before the game allocates memory). The pool's descriptor, where the host span starts and the
 * pool's size; -1 when the pool exists already without it. */
int runtime_memory_host_pool(uint64_t host_bytes, int *fd, uint64_t *host_offset, uint64_t *total) {
    write_lock();
    if (pool_fd<0) host_span=align_up(host_bytes,PAGE);
    const int ok=host_span && !pool();
    if (ok) { *fd=pool_fd; *host_offset=POOL_SIZE+FLEX_SPAN; *total=POOL_SIZE+FLEX_SPAN+host_span; }
    write_unlock();
    return ok ? 0 : -1;
}
/* bbport (native GPU process): [start, end) of the user range is the GPU process's own memory
 * (its allocator's): reserved here, so the guest's mappings go elsewhere. Before the game maps
 * anything; -1 when something is mapped there already. */
int runtime_memory_exclude(uintptr_t start, uintptr_t end) {
    start&=~(uintptr_t)(PAGE-1);
    end=align_up(end,PAGE);
    if (start<USER_MIN) start=USER_MIN;
    if (end>USER_MAX) end=USER_MAX;
    if (start>=end) return 0;
    write_lock();
    const int ok=!overlaps(start,end,0) && !vma_insert(vma_index(start),(Vma){start,end,KIND_RESERVED,0,0,0,0});
    write_unlock();
    return ok ? 0 : -1;
}
/* bbport (native GPU process): every change to the mapping table from now on (and the mappings
 * that exist, as maps) goes to `hook`, in order, outside the lock. */
void runtime_memory_set_mirror_hook(MirrorHook hook) {
    write_lock();
    mirror_hook=hook;
    for (size_t i=0;i<vma_count;++i)
        if (vmas[i].kind!=KIND_RESERVED)
            queue_mirror(MIRROR_MAP,vmas[i].start,vmas[i].end-vmas[i].start,vmas[i].phys,vmas[i].kind,
                         vmas[i].prot,vmas[i].type,!vmas[i].private_mem);
    write_unlock();
}
