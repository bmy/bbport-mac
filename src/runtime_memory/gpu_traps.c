// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: GPU hooks, CPU write notes and write traps, writes and reads through the backing view, region queries for the GPU, the report.
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
void runtime_memory_set_gpu_hooks(GpuRange map, GpuRange unmap, GpuRange invalidate) {
    const char *toggles=bb_env("BB_TOGGLE_FILE");
    /* bbport: BB_TOGGLES=<mask>: the same bits from the start, without a file (launcher). */
    if (bb_env("BB_TOGGLES")) {
        runtime_disabled_optimizations=strtoull(bb_env("BB_TOGGLES"),NULL,0);
        printf("Runtime: disabled optimizations mask=%llu (BB_TOGGLES)\n",
               (unsigned long long)runtime_disabled_optimizations);
    }
    static pthread_t watcher;
    if (!watcher && (parse_experiment_cycle() || toggles)) pthread_create(&watcher,NULL,toggle_watcher,(void *)toggles);
    write_lock();
    hook_map=map; hook_unmap=unmap; hook_invalidate=invalidate;
    for (size_t i=0;i<vma_count;++i) if (vmas[i].kind!=KIND_RESERVED) queue_hook(HOOK_MAP,vmas[i].start,vmas[i].end-vmas[i].start);
    write_unlock();
    flush_hooks();
}
/* bbport: the CPU is about to write [address, address+size) without the guest's code doing it
 * (a file read into it): the GPU side is told first (Rasterizer::InvalidateMemory), for the parts
 * the GPU can access, as a driver is told of a resource update. Its write tracking then has
 * nothing to catch there. */
void runtime_memory_set_note_write_hook(GpuRange note) { hook_note_write=note; }
void runtime_memory_note_write(uintptr_t address, uint64_t size) {
    if (!hook_invalidate || !size) return;
    uintptr_t starts[8], ends[8];
    int n=0;
    read_lock();
    for (size_t i=vma_index(address); i<vma_count && vmas[i].start<address+size && n<8; ++i) {
        if (vmas[i].kind==KIND_RESERVED || !(vmas[i].prot & 0x30)) continue;
        starts[n]=vmas[i].start>address ? vmas[i].start : address;
        ends[n]=vmas[i].end<address+size ? vmas[i].end : address+size;
        ++n;
    }
    read_unlock();
    GpuRange hook=hook_note_write ? hook_note_write : hook_invalidate;
    for (int i=0;i<n;++i) hook(starts[i],ends[i]-starts[i]);
}
/* bbport: pages whose CPU writes the GPU side must hear
 * of (a VRAM copy of them, a texture made from them), one bit per 4 KiB page below 2^40. The GPU
 * side sets and clears them where it used to protect; the writers it cannot see coming (libc
 * imports, file reads, hooked game code) check them after writing and tell it, as a driver is
 * told of a resource update. */
#define WATCH_LIMIT (1ull<<40)
#include "../../gpu/shim/bb_env.h"
static _Atomic(uint64_t) *write_watch;
static GpuRange hook_cpu_write;
void runtime_memory_set_cpu_write_hook(GpuRange hook) {
    if (!write_watch) {
        void *bits=mmap(NULL,WATCH_LIMIT>>15,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if (bits==MAP_FAILED) { perror("bbport: write watch bitmap"); return; }
#ifdef _WIN32
        fprintf(stderr,"Runtime: write watch bitmap at %p\n",bits);
#endif
        write_watch=bits;
    }
    hook_cpu_write=hook;
}
void runtime_memory_set_write_watch(uintptr_t address, uint64_t size, int watch) {
    if (!write_watch || !size || address>=WATCH_LIMIT) return;
    uint64_t first=address>>12, last=(address+size-1)>>12;
    if (last>=(WATCH_LIMIT>>12)) last=(WATCH_LIMIT>>12)-1;
    for (uint64_t word=first>>6; word<=last>>6; ++word) {
        const uint64_t lo=word==first>>6 ? first&63 : 0, hi=word==last>>6 ? last&63 : 63;
        const uint64_t mask=(hi==63 ? ~0ull : (2ull<<hi)-1) & ~((1ull<<lo)-1);
        if (watch) atomic_fetch_or_explicit(&write_watch[word],mask,memory_order_release);
        else atomic_fetch_and_explicit(&write_watch[word],~mask,memory_order_release);
    }
}
void runtime_memory_note_cpu_write(uintptr_t address, uint64_t size) {
    if (!write_watch || !size || address>=WATCH_LIMIT || address+size>WATCH_LIMIT) return;
    const uint64_t first=address>>12, last=(address+size-1)>>12;
    for (uint64_t word=first>>6; word<=last>>6; ++word) {
        const uint64_t lo=word==first>>6 ? first&63 : 0, hi=word==last>>6 ? last&63 : 63;
        const uint64_t mask=(hi==63 ? ~0ull : (2ull<<hi)-1) & ~((1ull<<lo)-1);
        if (atomic_load_explicit(&write_watch[word],memory_order_acquire) & mask) {
            hook_cpu_write(address,size);
            return;
        }
    }
}
/* bbport: write traps the GPU side keeps on guest pages, by reason (bits: a VRAM copy of the page,
 * an image made from it, ...). A page with any reason is read-only for the game even where its
 * own protection allows writes; the fault goes to the GPU library's handlers, and each clears
 * only its own reason. One owner for these protections: the layer's VRAM copies and the texture
 * cache trap the same pages, and one's unprotect must not drop the other's trap. Reasons from 16 up
 * trap reads too (no access: the page's data is newer elsewhere, a VRAM copy). Lock order: the
 * runtime's lock, then trap_mutex (as kernel_mprotect, which keeps the traps). */
#define TRAP_LIMIT (1ull<<40)
#define TRAP_NO_ACCESS 0xf0u
static uint8_t *trap_reasons; /* one byte per 4 KiB page below TRAP_LIMIT (reserved, not committed) */
static pthread_mutex_t trap_mutex=PTHREAD_MUTEX_INITIALIZER;
/* A page's trap level: 0 none, 1 read-only, 2 no access. */
static int trap_level(uint8_t reasons) { return !reasons ? 0 : (reasons & TRAP_NO_ACCESS) ? 2 : 1; }
/* The pages [a, b) to the game's protection, without write at level 1, without access at 2
 * (runtime lock held). */
static void trap_apply(uintptr_t a, uintptr_t b, int level) {
    for (size_t i=vma_index(a); i<vma_count && vmas[i].start<b; ++i) {
        if (vmas[i].kind==KIND_RESERVED) continue;
        const uintptr_t s=vmas[i].start>a ? vmas[i].start : a, e=vmas[i].end<b ? vmas[i].end : b;
        const int prot=host_prot(vmas[i].prot);
        mprotect((void *)s,e-s,level==2 ? PROT_NONE : level ? (prot & ~PROT_WRITE) : prot);
    }
}
/* No traps in [start, end) any more: a mapping replaced or gone (runtime lock held). */
static void trap_forget(uintptr_t start, uintptr_t end) {
    if (!trap_reasons || start>=TRAP_LIMIT) return;
    if (end>TRAP_LIMIT) end=TRAP_LIMIT;
    pthread_mutex_lock(&trap_mutex);
    memset(trap_reasons+(start>>12),0,((end+4095)>>12)-(start>>12));
    pthread_mutex_unlock(&trap_mutex);
}
/* Pages with traps in [start, end) read-only (or no access) again after the game changed their
 * protection (runtime lock held). */
static void trap_reapply(uintptr_t start, uintptr_t end) {
    if (!trap_reasons || start>=TRAP_LIMIT) return;
    if (end>TRAP_LIMIT) end=TRAP_LIMIT;
    pthread_mutex_lock(&trap_mutex);
    uintptr_t run=0; int run_level=0;
    for (uintptr_t p=start>>12; p<(end+4095)>>12; ++p) {
        const int level=trap_level(trap_reasons[p]);
        if (run && level!=run_level) { trap_apply(run,p<<12,run_level); run=0; }
        if (level && !run) { run=p<<12; run_level=level; }
    }
    if (run) trap_apply(run,end,run_level);
    pthread_mutex_unlock(&trap_mutex);
}
/* Where runtime_memory_trap's time goes (frame statistics' stall lines): waiting for the runtime's
 * lock and trap_mutex, and in the protection changes; calls and bytes changed. */
static _Atomic(uint64_t) trap_stat_wait_ns, trap_stat_apply_ns, trap_stat_calls, trap_stat_bytes;
static uint64_t trap_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec;
}
void runtime_memory_trap_stats(uint64_t *wait_ns, uint64_t *apply_ns, uint64_t *calls, uint64_t *bytes) {
    *wait_ns=atomic_load_explicit(&trap_stat_wait_ns,memory_order_relaxed);
    *apply_ns=atomic_load_explicit(&trap_stat_apply_ns,memory_order_relaxed);
    *calls=atomic_load_explicit(&trap_stat_calls,memory_order_relaxed);
    *bytes=atomic_load_explicit(&trap_stat_bytes,memory_order_relaxed);
}
/* bbport (native GPU process, protocol 12): per page, the newest of bb-gpu's trap changes applied
 * there (its order, bb_trap_copy.h); allocated on the first runtime_memory_trap_seq. */
static uint64_t *trap_seqs;
static void trap_change(uintptr_t address, uint64_t size, unsigned reason, int on, uint64_t seq, int lift);
void runtime_memory_trap(uintptr_t address, uint64_t size, unsigned reason, int on) {
    trap_change(address,size,reason,on,0,0);
}
/* bbport (native GPU process): runtime_memory_trap from bb-gpu, `seq` its order of the change
 * (recorded per page). A lift (`lift`: off, reasons bb-gpu found nobody there holds) is skipped on
 * pages a newer change already reached: an owner's trap set after bb-gpu checked stays, in
 * whichever order the two arrive (one from the faulting thread, one from the protection thread). */
void runtime_memory_trap_seq(uintptr_t address, uint64_t size, unsigned reason, int on, uint64_t seq, int lift) {
    if (!__atomic_load_n(&trap_seqs,__ATOMIC_ACQUIRE)) {
        void *bytes=mmap(NULL,(TRAP_LIMIT>>12)*sizeof(uint64_t),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if (bytes==MAP_FAILED) { trap_change(address,size,reason,on && !lift,0,0); return; } /* unordered, as before */
        uint64_t *expected=NULL;
        if (!__atomic_compare_exchange_n(&trap_seqs,&expected,(uint64_t *)bytes,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))
            munmap(bytes,(TRAP_LIMIT>>12)*sizeof(uint64_t));
    }
    trap_change(address,size,reason,on && !lift,seq,lift);
}
static void trap_change(uintptr_t address, uint64_t size, unsigned reason, int on, uint64_t seq, int lift) {
    if (!size || address>=TRAP_LIMIT) return;
    if (!trap_reasons) {
        void *bytes=mmap(NULL,TRAP_LIMIT>>12,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if (bytes==MAP_FAILED) { perror("bbport: write trap table"); return; }
#ifdef _WIN32
        fprintf(stderr,"Runtime: write trap table at %p\n",bytes);
#endif
        uint8_t *expected=NULL;
        if (!__atomic_compare_exchange_n(&trap_reasons,&expected,(uint8_t *)bytes,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))
            munmap(bytes,TRAP_LIMIT>>12);
    }
    uintptr_t end=address+size>TRAP_LIMIT ? TRAP_LIMIT : address+size;
    const uint64_t t0=trap_now_ns();
    read_lock();
    pthread_mutex_lock(&trap_mutex);
    const uint64_t t1=trap_now_ns();
    /* Runs of pages whose trap level changes, one mprotect each. */
    uintptr_t run=0; int run_level=0;
    uint64_t changed=0;
    const uintptr_t first=address>>12, last=(end-1)>>12;
    uint64_t *const seqs=seq ? __atomic_load_n(&trap_seqs,__ATOMIC_ACQUIRE) : NULL;
    for (uintptr_t p=first; p<=last; ++p) {
        const int skip=seqs && lift && seqs[p]>seq; /* bbport: a newer change reached it */
        if (seqs && !lift && seqs[p]<seq) seqs[p]=seq;
        const uint8_t old=trap_reasons[p], now=skip ? old : on ? (uint8_t)(old|reason) : (uint8_t)(old & ~reason);
        if (old!=now) trap_reasons[p]=now;
        const int level=trap_level(now), change=trap_level(old)!=level;
        if (run && (!change || level!=run_level)) { changed+=(p<<12)-run; trap_apply(run,p<<12,run_level); run=0; }
        if (change && !run) { run=p<<12; run_level=level; }
    }
    if (run) { changed+=((last+1)<<12)-run; trap_apply(run,(last+1)<<12,run_level); }
    pthread_mutex_unlock(&trap_mutex);
    read_unlock();
    const uint64_t t2=trap_now_ns();
    atomic_fetch_add_explicit(&trap_stat_wait_ns,t1-t0,memory_order_relaxed);
    atomic_fetch_add_explicit(&trap_stat_apply_ns,t2-t1,memory_order_relaxed);
    atomic_fetch_add_explicit(&trap_stat_calls,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&trap_stat_bytes,changed,memory_order_relaxed);
}
/* bbport: a libc write about to land on pages with write traps: the GPU side is told first and
 * lifts its traps, instead of the write faulting (a fault handler waits for the draws the GPU
 * thread has queued; libc copies into small textures every frame made the game's frames late).
 * Its announcement after the write (runtime_memory_note_cpu_write) stays as it was. */
void runtime_memory_prepare_cpu_write(uintptr_t address, uint64_t size) {
    const uint8_t *table=__atomic_load_n(&trap_reasons,__ATOMIC_ACQUIRE);
    if (!table || !size || address>=TRAP_LIMIT || !hook_cpu_write) return;
    uint64_t last=(address+size-1)>>12;
    if (last>=(TRAP_LIMIT>>12)) last=(TRAP_LIMIT>>12)-1;
    for (uint64_t p=address>>12; p<=last; ++p)
        if (__atomic_load_n(&table[p],__ATOMIC_RELAXED)) { hook_cpu_write(address,size); return; }
}
unsigned runtime_memory_trap_reasons(uintptr_t address) {
    const uint8_t *table=__atomic_load_n(&trap_reasons,__ATOMIC_ACQUIRE);
    return table && address<TRAP_LIMIT ? __atomic_load_n(&table[address>>12],__ATOMIC_ACQUIRE) : 0;
}
/* bbport: where direct memory at a guest address lives (the GPU uses the game's
 * memory where it is): its offset in direct memory and the end of the mapping, which is contiguous
 * there. 0 for anything else (flexible memory, reserved or unmapped addresses). */
int runtime_memory_direct_phys(uintptr_t address, uint64_t *phys, uintptr_t *end) {
    read_lock();
    const size_t i=vma_index(address);
    const int ok = i<vma_count && vmas[i].start<=address && address<vmas[i].end && vmas[i].kind==KIND_DIRECT;
    if (ok) {
        *phys=vmas[i].phys+(address-vmas[i].start);
        *end=vmas[i].end;
    }
    read_unlock();
    return ok;
}
/* bbport: protection and direct memory type of the mapping at address (type -1: not direct
 * memory); 0 when nothing is mapped there. */
int runtime_memory_vma_info(uintptr_t address, int *prot, int *type, uintptr_t *end) {
    read_lock();
    const size_t i=vma_index(address);
    const int ok = i<vma_count && vmas[i].start<=address && vmas[i].kind!=KIND_RESERVED;
    if (ok) {
        *prot=vmas[i].prot;
        *type=vmas[i].kind==KIND_DIRECT ? vmas[i].type : -1;
        *end=vmas[i].end;
    }
    read_unlock();
    return ok;
}
/* Writes through the backing view; 0 when part of the range has no backing. */
/* Guest-visible writes from the GPU side (fence labels, WriteData, downloads): every byte is
 * stored once. glibc's memcpy copies 4-7 bytes as two overlapping 4-byte stores (the same
 * 4 bytes twice for a 4-byte label): a thread preempted between them stored the label again
 * after the guest had seen it and freed the block that held it (a free-list link ending in
 * 0x00000004: guest fault 0x263b8e7). Aligned 2, 4 and 8-byte pieces are single stores. */
static void store_once(unsigned char *dst, const unsigned char *src, uint64_t n) {
    if (n > 64) { memcpy(dst,src,n); return; }
    while (n) {
        const uintptr_t at=(uintptr_t)dst;
        if (n>=8 && !(at&7)) { uint64_t v; memcpy(&v,src,8); __atomic_store_n((uint64_t *)dst,v,__ATOMIC_RELEASE); dst+=8; src+=8; n-=8; }
        else if (n>=4 && !(at&3)) { uint32_t v; memcpy(&v,src,4); __atomic_store_n((uint32_t *)dst,v,__ATOMIC_RELEASE); dst+=4; src+=4; n-=4; }
        else if (n>=2 && !(at&1)) { uint16_t v; memcpy(&v,src,2); __atomic_store_n((uint16_t *)dst,v,__ATOMIC_RELEASE); dst+=2; src+=2; n-=2; }
        else { __atomic_store_n(dst,*src,__ATOMIC_RELEASE); ++dst; ++src; --n; }
    }
}
/* A host write to private memory is a write to the guest's own pages: pages with write traps are
 * opened for it and closed again (read lock held). A guest thread writing those pages meanwhile is
 * not caught: the price of one view. */
static void private_store(unsigned char *dst, const unsigned char *src, uint64_t n) {
    const uintptr_t a=(uintptr_t)dst&~(uintptr_t)(PAGE-1), b=((uintptr_t)dst+n+PAGE-1)&~(uintptr_t)(PAGE-1);
    int trapped=0;
    if (trap_reasons && b<=TRAP_LIMIT) {
        pthread_mutex_lock(&trap_mutex);
        for (uintptr_t p=a>>12; p<(b>>12) && !trapped; ++p) trapped=trap_reasons[p]!=0;
        if (!trapped) pthread_mutex_unlock(&trap_mutex);
    }
    if (trapped) {
        trap_apply(a,b,0);
        store_once(dst,src,n);
        for (uintptr_t p=a>>12; p<(b>>12); ++p) trap_apply(p<<12,(p+1)<<12,trap_level(trap_reasons[p]));
        pthread_mutex_unlock(&trap_mutex);
    } else {
        store_once(dst,src,n);
    }
}
int runtime_memory_write_backing(uintptr_t address, const void *data, uint64_t size) {
    read_lock();
    int ok=1;
    for (uintptr_t at=address, end=address+size; at<end && ok;) {
        size_t i=vma_index(at);
        if (i==vma_count || vmas[i].start>at || vmas[i].kind==KIND_RESERVED) { ok=0; break; }
        uint64_t n=(vmas[i].end<end ? vmas[i].end : end)-at;
        unsigned char *dst=backing_at(vmas[i].phys+(at-vmas[i].start));
        if (private_import && dst==(unsigned char *)at) private_store(dst,(const unsigned char *)data+(at-address),n);
        else store_once(dst,(const unsigned char *)data+(at-address),n);
        at+=n;
    }
    read_unlock();
    return ok;
}
/* Reads through the backing view: the host's copies of guest memory (uploads), which the GPU's
 * read protection (BB_READBACKS=2: GPU-written pages) must not stop. A fault there would ask the
 * GPU thread for a readback while it waits for these very copies (a hang at start). Gaps read
 * as zeros. */
void runtime_memory_read_backing(uintptr_t address, void *data, uint64_t size) {
    unsigned char *out=data;
    read_lock();
    for (uintptr_t at=address, end=address+size; at<end;) {
        const size_t i=vma_index(at);
        uint64_t n;
        if (i<vma_count && vmas[i].start<=at && vmas[i].kind!=KIND_RESERVED) {
            n=(vmas[i].end<end ? vmas[i].end : end)-at;
            memcpy(out+(at-address),backing_at(vmas[i].phys+(at-vmas[i].start)),n);
        } else {
            const uintptr_t next=i==vma_count ? end : vmas[i].start>at ? vmas[i].start : vmas[i].end;
            n=(next<end ? next : end)-at;
            memset(out+(at-address),0,n);
        }
        at+=n;
    }
    read_unlock();
}
/* Per-thread cache of recently found regions for the GPU's per-draw queries:
 * entries hold for as long as the table generation they were read at. */
typedef struct { uint64_t generation; uintptr_t start, end; int kind; } CachedRegion;
static __thread CachedRegion region_cache[8];
static __thread unsigned region_cache_next;
static const CachedRegion *cached_region(uintptr_t address) {
    uint64_t generation=__atomic_load_n(&table_generation,__ATOMIC_ACQUIRE);
    if (generation & 1) return NULL;
    for (unsigned i=0;i<8;++i) {
        const CachedRegion *c=&region_cache[i];
        if (c->generation==generation && c->end && c->start<=address && address<c->end) return c;
    }
    return NULL;
}
/* Called under the read lock after a lookup of address. */
static void cache_region(size_t i) {
    uint64_t generation=__atomic_load_n(&table_generation,__ATOMIC_ACQUIRE);
    if (i==vma_count || (generation & 1)) return;
    region_cache[region_cache_next++ % 8]=(CachedRegion){generation,vmas[i].start,vmas[i].end,vmas[i].kind};
}
/* The mapping table's generation (odd while it changes): callers cache regions against it. */
const uint64_t *runtime_memory_generation(void) { return &table_generation; }
/* Mapped length from address, up to size, across adjacent mappings. */
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size) {
    const CachedRegion *c=(runtime_disabled_optimizations & 1) ? NULL : cached_region(address);
    if (c && c->kind!=KIND_RESERVED && size<=c->end-address) return size;
    read_lock();
    { size_t i=vma_index(address); if (i<vma_count && vmas[i].start<=address) cache_region(i); }
    uint64_t length=0;
    for (size_t i=vma_index(address); i<vma_count && length<size; ++i) {
        if (vmas[i].start>address+length || vmas[i].kind==KIND_RESERVED) break;
        length=vmas[i].end-address;
    }
    read_unlock();
    return length<size ? length : size;
}
/* Region around address: [start,end) mapped or a gap; 0 past the last mapping. */
int runtime_memory_region(uintptr_t address, uintptr_t *start, uintptr_t *end, int *mapped) {
    const CachedRegion *c=(runtime_disabled_optimizations & 1) ? NULL : cached_region(address);
    if (c) { *start=c->start; *end=c->end; *mapped=c->kind!=KIND_RESERVED; return 1; }
    read_lock();
    size_t i=vma_index(address);
    int found=i<vma_count;
    if (found && vmas[i].start<=address) {
        cache_region(i);
        *start=vmas[i].start; *end=vmas[i].end; *mapped=vmas[i].kind!=KIND_RESERVED;
    } else if (found) {
        *start=i ? vmas[i-1].end : 0; *end=vmas[i].start; *mapped=0;
    }
    read_unlock();
    return found;
}
/* GPU page tracking: restrict host access, never beyond the guest's own protection. */
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write) {
    read_lock();
    for (size_t i=vma_index(address); i<vma_count && vmas[i].start<address+size; ++i) {
        if (vmas[i].kind==KIND_RESERVED) continue;
        uintptr_t a=vmas[i].start>address ? vmas[i].start : address;
        uintptr_t b=vmas[i].end<address+size ? vmas[i].end : address+size;
        int guest=host_prot(vmas[i].prot);
        int want=(read ? PROT_READ : 0)|(write ? PROT_WRITE|PROT_READ : 0)|(guest & PROT_EXEC);
        mprotect((void *)a,b-a,guest & want);
    }
    read_unlock();
}
/* Host view of a guest range, for the GPU backend: the range must be mapped. */
int runtime_memory_is_mapped(uintptr_t address, uint64_t size) {
    const CachedRegion *c=(runtime_disabled_optimizations & 1) ? NULL : cached_region(address);
    if (c && c->kind!=KIND_RESERVED && size<=c->end-address) return 1;
    read_lock();
    int result=covered(address,address+size,0);
    read_unlock();
    return result;
}
void runtime_memory_report(void) {
    printf("Runtime: direct memory allocations=%zu, maps=%zu, live=%" PRIu64 ", budget=%" PRIu64 "\n",
           allocations, maps, live_bytes, POOL_SIZE);
    printf("Runtime: flexible maps=%zu, in use=%" PRIu64 ", protects=%zu, queries=%zu, regions=%zu\n",
           flexible_maps, flexible_bytes, protects, queries, vma_count);
}
