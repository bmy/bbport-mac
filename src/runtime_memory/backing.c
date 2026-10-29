// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: the memfd pool and its backing view: chunks the GPU library makes, deferred (growing) heaps, zeroing, host mappings of physical ranges.
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
#include "../../gpu/shim/bb_env.h"
static int host_prot(int prot) {
    /* GPU read/write bits (0x10/0x20) need host access for the future GPU backend. */
    return ((prot & 0x11) ? PROT_READ : 0) | ((prot & 0x22) ? PROT_WRITE|PROT_READ : 0) | ((prot & 4) ? PROT_EXEC|PROT_READ : 0);
}
/* bbport: the memfd's pages as 2 MiB folios where the kernel gives shmem huge pages on advice
 * (transparent_hugepage/shmem_enabled=advise): the GPU reads the game's memory in place through
 * imports of these pages, and 4 KiB ones cost it 0.8 ms a frame on a Steam Deck (TLB). */
static void advise_huge(void *p, uint64_t n) {
#if !defined(_WIN32) && defined(MADV_HUGEPAGE)
    static int off=-1;
    if (off<0) { const char *e=bb_env("BB_GAME_HUGE"); off=e && e[0]=='0'; }
    if (off && p!=backing_base) return; /* BB_GAME_HUGE=0 (tests): the backing view alone */
    (void)madvise(p,n,MADV_HUGEPAGE);
#else
    (void)p; (void)n;
#endif
}
/* Direct memory occupies [0,POOL_SIZE) of the memfd, flexible memory [POOL_SIZE,+FLEX_SPAN). */
static int pool(void) {
    if (pool_fd>=0) return 0;
#ifdef _WIN32
    /* A pagefile-backed section (gpu/platform/win32/posix_memory.cpp). */
    pool_fd=bb_section_create(POOL_SIZE+FLEX_SPAN);
    if (pool_fd<0) return -1;
#elif defined(__APPLE__)
    /* bbport (macOS): no memfd: an unlinked POSIX shared memory object (as upstream shadPS4); the
     * native GPU process maps it too. host_span: runtime_memory_host_pool. */
    char name[64];
    snprintf(name,sizeof(name),"/bbport-guest-%d",(int)getpid());
    pool_fd=shm_open(name,O_RDWR|O_CREAT|O_EXCL,0600);
    if (pool_fd>=0) shm_unlink(name);
    if (pool_fd<0 || ftruncate(pool_fd,(off_t)(POOL_SIZE+FLEX_SPAN+host_span))) return -1;
#else
    pool_fd=memfd_create("bb-guest-memory", MFD_CLOEXEC|MFD_ALLOW_SEALING);
    if (pool_fd<0 || ftruncate(pool_fd,(off_t)(POOL_SIZE+FLEX_SPAN+host_span))) return -1;
    /* udmabuf (the GPU's lazy imports of guest memory) needs the size sealed. */
    (void)fcntl(pool_fd,F_ADD_SEALS,F_SEAL_SHRINK|F_SEAL_GROW);
#endif
#ifdef __APPLE__
    /* bbport (macOS): the view 2 MiB-aligned. Under Rosetta mmap aligns to 4 KiB only, and the GPU's
     * host memory imports of it (KosmicKrisp) need 16 KiB-aligned pointers. */
    const uint64_t view_span=POOL_SIZE+FLEX_SPAN+host_span, view_align=UINT64_C(2)<<20;
    void *reserved=mmap(NULL,view_span+view_align,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
    if (reserved==MAP_FAILED) return -1;
    const uintptr_t lo=(uintptr_t)reserved, at=(lo+view_align-1)&~(uintptr_t)(view_align-1);
    void *view=mmap((void *)at,view_span,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,pool_fd,0);
    if (view==MAP_FAILED) { munmap(reserved,view_span+view_align); return -1; }
    if (at>lo) munmap(reserved,at-lo);
    if (lo+view_span+view_align>at+view_span) munmap((void *)(at+view_span),lo+view_span+view_align-(at+view_span));
#else
    void *view=mmap(NULL,POOL_SIZE+FLEX_SPAN+host_span,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_NORESERVE,pool_fd,0);
    if (view==MAP_FAILED) return -1;
#endif
    backing_base=view;
    advise_huge(view,POOL_SIZE);
    chunk_fds=malloc((POOL_SIZE/CHUNK+1)*sizeof(int));
    if (!chunk_fds) return -1;
    pinned=calloc((POOL_SIZE+FLEX_SPAN)/PIN_GRANULE+1,1);
    if (!pinned) return -1;
    for (uint64_t c=0;c<=POOL_SIZE/CHUNK;++c) chunk_fds[c]=-1;
    return 0;
}
/* Size of direct memory chunk c (the last one may be shorter). */
static uint64_t chunk_size(uint64_t c) {
    const uint64_t start=c*CHUNK;
    return POOL_SIZE-start<CHUNK ? POOL_SIZE-start : CHUNK;
}
/* Decides where the chunks of [phys, phys+size) of direct memory live (lock held). */
/* bbport: a heap growing on demand (the game's GraphicsPrivate heap, gpu/games/bloodborne/hooks_heap.cpp,
 * BB_HEAP_GROW): its direct allocation (grow_size bytes) gets GPU-visible chunks for its first
 * grow_initial bytes only. The rest stays in the memfd, whose pages are allocated only when touched
 * (the heap's allocator holds it reserved, so nothing touches it), until runtime_memory_commit gives
 * a piece of it its own chunk. */
static uint64_t grow_initial, deferred_phys, deferred_end;
static const uint64_t *grow_size_at; /* the heap's size where the game keeps it (patched by then) */
void runtime_memory_set_grow(const uint64_t *size_at, uint64_t initial) {
    write_lock(); grow_size_at=size_at; grow_initial=initial; write_unlock();
}
int runtime_memory_deferred(uint64_t *phys, uint64_t *end) {
    read_lock(); *phys=deferred_phys; *end=deferred_end; read_unlock();
    return deferred_end>deferred_phys;
}
/* bbport BB_HOST_ONLY=phys,phys... (hex, the starts of direct allocations; experiments): those get
 * no GPU chunk and stay in the memfd (pages only as the game touches them). The GPU side reads
 * such memory through the arena (copies) and its paged bindings see zero there (counted). */
static uint64_t host_only_ranges[16][2];
static size_t host_only_count;
static uint64_t host_only_ordinals; /* bit i: the game's i-th direct allocation (the game profile) */
static int host_only_next; /* the allocation being placed is one of them */
void runtime_memory_set_host_only_allocations(uint64_t mask) { host_only_ordinals=mask; }
static int in_host_only(uint64_t phys) {
    for (size_t i=0;i<host_only_count;++i)
        if (phys>=host_only_ranges[i][0] && phys<host_only_ranges[i][1]) return 1;
    return 0;
}
static int host_only(uint64_t phys) {
    const char *list=bb_env("BB_HOST_ONLY");
    if (!list) return host_only_next;
    for (const char *at=list; at && *at;) {
        char *end=NULL;
        const uint64_t start=strtoull(at,&end,16);
        if (end==at) break;
        if (start==phys) return 1;
        at=*end ? end+1 : end;
    }
    return 0;
}
static void ensure_chunks(uint64_t phys, uint64_t size) {
    if (!chunk_fds || phys>=POOL_SIZE || private_import) return; /* private: map_private */
    if (lazy_game && phys<GAME_POOL_SIZE) return; /* the game's memory stays in the memfd */
    if (whole_chunks) {
        if (!chunk_alloc) return;
        if (in_host_only(phys)) return;
        const uint64_t end=phys+size<POOL_SIZE ? phys+size : POOL_SIZE;
        for (uint64_t at=phys;at<end;) {
            uint64_t next;
            const Region *r=region_at(at,&next);
            if (r) { at=r->phys+r->size; continue; }
            const uint64_t gap_end=next<end ? next : end;
            if (deferred_end>deferred_phys && at<deferred_end && deferred_phys<gap_end) {
                if (at<deferred_phys) add_region(at,deferred_phys-at);
                if (deferred_end<gap_end) add_region(deferred_end,gap_end-deferred_end);
            } else add_region(at,gap_end-at);
            at=gap_end;
        }
        return;
    }
    const uint64_t last=(phys+size-1<POOL_SIZE ? phys+size-1 : POOL_SIZE-1)/CHUNK;
    for (uint64_t c=phys/CHUNK;c<=last;++c) {
        if (chunk_fds[c]!=-1) continue;
        int fd=chunk_alloc ? chunk_alloc(c*CHUNK,chunk_size(c)) : -1;
        if (fd>=0 && mmap(backing_base+c*CHUNK,chunk_size(c),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0)==MAP_FAILED) {
            close(fd);
            fd=-1;
        }
        chunk_fds[c]=fd>=0 || fd==CHUNK_IMPORTED ? fd : -2;
    }
}
/* bbport: the GPU library's chunks for the gaps ensure_chunks will fill in [phys,+size), made with
 * the write lock let go (held by the caller, once): a new allocation's import took 200-360 ms on
 * the GTX 1050 laptop and every GPU query of the memory map waited for it (stage A stood still,
 * the hitches felt on entering a new part of an area, 2026-10-25). The block is reserved already,
 * so nobody else places anything there meanwhile. BB_DIRECT_IMPORT_LOCKED=1: under the lock. */
static void import_chunks_unlocked(uint64_t phys, uint64_t size) {
    static int locked=-1;
    if (locked<0) { const char *e=bb_env("BB_DIRECT_IMPORT_LOCKED"); locked=e && e[0]=='1'; }
    memset(pre_chunks,0,sizeof(pre_chunks));
    if (locked || exclusive_depth!=1 || !whole_chunks || !chunk_alloc || !chunk_fds || private_import ||
        phys>=POOL_SIZE || (lazy_game && phys<GAME_POOL_SIZE) || in_host_only(phys)) return;
    uint64_t gaps[8][2]; size_t count=0;
    const uint64_t end=phys+size<POOL_SIZE ? phys+size : POOL_SIZE;
    for (uint64_t at=phys;at<end;) {
        uint64_t next;
        const Region *r=region_at(at,&next);
        if (r) { at=r->phys+r->size; continue; }
        const uint64_t gap_end=next<end ? next : end;
        if (deferred_end>deferred_phys && at<deferred_end && deferred_phys<gap_end) {
            if (at<deferred_phys && count<8) { gaps[count][0]=at; gaps[count][1]=deferred_phys-at; ++count; }
            if (deferred_end<gap_end && count<8) { gaps[count][0]=deferred_end; gaps[count][1]=gap_end-deferred_end; ++count; }
        } else if (count<8) { gaps[count][0]=at; gaps[count][1]=gap_end-at; ++count; }
        at=gap_end;
    }
    if (!count) return;
    write_unlock();
    for (size_t i=0;i<count;++i) {
        const int fd=chunk_alloc(gaps[i][0],gaps[i][1]);
        if (fd>=0 || fd==CHUNK_IMPORTED) pre_chunks[i]=(PreChunk){1,fd,gaps[i][0],gaps[i][1]};
    }
    write_lock();
}
static void (*chunk_free)(uint64_t phys, uint64_t size);
/* Chunks import_chunks_unlocked made that ensure_chunks did not take (the gaps changed): given back. */
static void release_pre_chunks(void) {
    for (size_t i=0;i<sizeof(pre_chunks)/sizeof(pre_chunks[0]);++i) {
        if (!pre_chunks[i].valid) continue;
        pre_chunks[i].valid=0;
        if (pre_chunks[i].fd>=0) close(pre_chunks[i].fd);
        if (chunk_free) chunk_free(pre_chunks[i].phys,pre_chunks[i].size);
        printf("Runtime: a chunk made ahead for direct memory %#" PRIx64 "+%" PRIu64 " MiB was not needed\n",
               pre_chunks[i].phys,pre_chunks[i].size>>20);
    }
}
#ifdef _WIN32
/* bbport Windows: released memfd memory reads as zero again, but a pagefile-backed section has no
 * holes: fallocate cleared it with memset, bringing every released page into RAM (flexible memory
 * the game frees by the hundreds of MiB). Its pages are reset instead (no RAM, content undefined)
 * and the range noted; it is cleared when mapped again (map_phys). The write lock held. */
typedef struct { uint64_t start, end; } ZeroRange;
static ZeroRange *to_zero;
static size_t to_zero_count, to_zero_cap;
static void zero_later(uint64_t at, uint64_t n) {
    uint64_t start=at, end=at+n;
    size_t w=0;
    for (size_t i=0;i<to_zero_count;++i) { /* merged with the ranges it touches */
        const ZeroRange r=to_zero[i];
        if (r.end<start || r.start>end) { to_zero[w++]=r; continue; }
        if (r.start<start) start=r.start;
        if (r.end>end) end=r.end;
    }
    to_zero_count=w;
    if (to_zero_count==to_zero_cap) {
        const size_t cap=to_zero_cap ? to_zero_cap*2 : 64;
        ZeroRange *grown=realloc(to_zero,cap*sizeof(*grown));
        if (!grown) { memset(backing_base+at,0,n); return; }
        to_zero=grown; to_zero_cap=cap;
    }
    to_zero[to_zero_count++]=(ZeroRange){start,end};
    bb_section_discard(pool_fd,at,n);
    bb_discard(backing_base+at,n);
}
/* [phys, +size) is mapped again: its pending clears are done now. */
static void zero_now(uint64_t phys, uint64_t size) {
    const uint64_t end=phys+size;
    int any=0;
    for (size_t i=0;i<to_zero_count && !any;++i) any=to_zero[i].start<end && phys<to_zero[i].end;
    if (!any) return;
    ZeroRange *out=malloc((to_zero_count*2+1)*sizeof(*out));
    size_t w=0;
    for (size_t i=0;i<to_zero_count;++i) {
        const ZeroRange r=to_zero[i];
        if (r.end<=phys || r.start>=end) { if (out) out[w++]=r; continue; }
        const uint64_t a=r.start>phys ? r.start : phys, b=r.end<end ? r.end : end;
        if (!out) { memset(backing_base+r.start,0,r.end-r.start); continue; } /* no memory: all now */
        memset(backing_base+a,0,b-a);
        if (r.start<a) out[w++]=(ZeroRange){r.start,a};
        if (b<r.end) out[w++]=(ZeroRange){b,r.end};
    }
    free(to_zero);
    to_zero=out; to_zero_count=out ? w : 0; to_zero_cap=out ? to_zero_count*2+1 : 0;
}
#endif
/* Private regions: the allocation's memory placed at address (its first mapping) and imported by
 * the GPU, or the mapping it already has. */
static void *map_private(uintptr_t address, uint64_t size, int prot, uint64_t phys) {
    uint64_t next;
    const Region *r=region_at(phys,&next);
    if (r) {
        if (r->fd!=CHUNK_PRIVATE || phys+size>r->phys+r->size || r->va+(phys-r->phys)!=address) {
            fprintf(stderr,"STOP: direct memory %#" PRIx64 "+%#" PRIx64 " mapped at %#" PRIxPTR ", but its "
                    "memory lives at %#" PRIxPTR " (private regions: one mapping per allocation)\n",
                    phys,size,address,(uintptr_t)(r->va+(phys-r->phys))); /* bbport: uintptr_t (macOS) */
            return MAP_FAILED;
        }
    } else {
        if (region_count==MAX_REGIONS || phys+size>next) return MAP_FAILED;
#ifdef _WIN32
        void *p=bb_private_at((void *)address,size);
#else
        void *p=mmap((void *)address,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED,-1,0);
#endif
        if (p==MAP_FAILED) return MAP_FAILED;
        /* A heap growing on demand: only its first part for the GPU now (runtime_memory_commit). */
        /* Memory the GPU never uses (the game profile): none. */
        const uint64_t imported = deferred_end>deferred_phys && phys<deferred_phys &&
                                  deferred_phys<phys+size ? deferred_phys-phys : size;
        if (!in_host_only(phys) && private_import(phys,imported,p)) {
            fprintf(stderr,"STOP: the GPU could not import direct memory %#" PRIx64 "+%" PRIu64 " MiB at %p\n",
                    phys,size>>20,p);
            fflush(stderr);
            exit(23);
        }
        size_t i=region_count;
        while (i>0 && regions[i-1].phys>phys) { regions[i]=regions[i-1]; --i; }
        regions[i]=(Region){phys,size,CHUNK_PRIVATE,address};
        ++region_count;
    }
    if (mprotect((void *)address,size,prot)) return MAP_FAILED;
    return (void *)address;
}
/* mmap of direct or flexible memory at phys, split where it crosses chunks. */
static void *map_phys(uintptr_t address, uint64_t size, int prot, uint64_t phys) {
#ifdef _WIN32
    zero_now(phys,size);
#endif
    if (private_import && chunk_fds && phys<POOL_SIZE) return map_private(address,size,prot,phys);
    if (!chunk_fds || phys>=POOL_SIZE)
        return mmap((void *)address,size,prot,MAP_SHARED|MAP_FIXED,pool_fd,(off_t)phys);
    ensure_chunks(phys,size);
    for (uint64_t done=0;whole_chunks && done<size;) {
        const uint64_t at=phys+done;
        uint64_t next;
        const Region *r=region_at(at,&next);
        const uint64_t room=r ? r->phys+r->size-at : next-at, n=size-done<room ? size-done : room;
        void *p = r && r->fd>=0
            ? mmap((void *)(address+done),n,prot,MAP_SHARED|MAP_FIXED,r->fd,(off_t)(at-r->phys))
            : mmap((void *)(address+done),n,prot,MAP_SHARED|MAP_FIXED,pool_fd,(off_t)at);
        if (p==MAP_FAILED) {
            static int reported;
            if (reported++ < 4)
                fprintf(stderr,"Runtime: mapping direct memory %#" PRIx64 "+%#" PRIx64 " from %s failed: %s\n",
                        at,n,r && r->fd>=0 ? "a GPU-visible dma-buf (whole-allocation chunk)" : "the memfd",strerror(errno));
            return MAP_FAILED;
        }
        if (!(r && r->fd>=0)) advise_huge(p,n);
        done+=n;
        if (done==size) return (void *)address;
    }
    for (uint64_t done=0;done<size;) {
        const uint64_t at=phys+done, c=at/CHUNK;
        const uint64_t room=c*CHUNK+chunk_size(c)-at, n=size-done<room ? size-done : room;
        void *p = chunk_fds[c]>=0
            ? mmap((void *)(address+done),n,prot,MAP_SHARED|MAP_FIXED,chunk_fds[c],(off_t)(at-c*CHUNK))
            : mmap((void *)(address+done),n,prot,MAP_SHARED|MAP_FIXED,pool_fd,(off_t)at);
        if (p==MAP_FAILED) {
            static int reported;
            if (reported++ < 4)
                fprintf(stderr,"Runtime: mapping direct memory %#" PRIx64 "+%#" PRIx64 " from %s failed: %s\n",
                        at,n,chunk_fds[c]>=0 ? "a GPU-visible dma-buf chunk" : "the memfd",strerror(errno));
            return MAP_FAILED;
        }
        if (chunk_fds[c]<0) advise_huge(p,n);
        done+=n;
    }
    return (void *)address;
}
/* Released memfd memory: punched out, or cleared where the GPU pinned its pages. */
static void punch_or_clear(uint64_t phys, uint64_t size) {
    for (uint64_t done=0;done<size;) {
        const uint64_t at=phys+done, g=at/PIN_GRANULE;
        uint64_t n=(g+1)*PIN_GRANULE-at;
        if (n>size-done) n=size-done;
        pthread_mutex_lock(&pin_mutex);
        if (pinned && at<POOL_SIZE+FLEX_SPAN && __atomic_load_n(&pinned[g],__ATOMIC_ACQUIRE)) {
            memset(backing_base+at,0,n);
            static uint64_t cleared; cleared+=n;
            if (bb_env("BB_PIN_CLEAR_LOG") && n>=(1u<<20))
                printf("Runtime: released %#" PRIx64 "+%" PRIu64 " KiB pinned by the GPU: cleared (%" PRIu64 " MiB so far)\n",at,n>>10,cleared>>20);
        }
#ifdef _WIN32
        else zero_later(at,n);
#elif defined(__APPLE__)
        else pool_discard(at,n);
#else
        else fallocate(pool_fd,FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE,(off_t)at,(off_t)n);
#endif
        pthread_mutex_unlock(&pin_mutex);
        done+=n;
    }
}
/* Returns released physical memory to zero: hole punching in the memfd, a clear in a chunk. */
static void zero_phys(uint64_t phys, uint64_t size) {
    for (uint64_t done=0;done<size;) {
        const uint64_t at=phys+done;
        if (!chunk_fds || at>=POOL_SIZE) {
            punch_or_clear(at,size-done);
            return;
        }
        if (whole_chunks) {
            uint64_t next;
            const Region *r=region_at(at,&next);
            const uint64_t room=r ? r->phys+r->size-at : next-at, n=size-done<room ? size-done : room;
            if (r && GPU_CHUNK(r->fd)) memset(backing_at(at),0,n);
            else punch_or_clear(at,n);
            done+=n;
            continue;
        }
        const uint64_t c=at/CHUNK, room=c*CHUNK+chunk_size(c)-at, n=size-done<room ? size-done : room;
        if (GPU_CHUNK(chunk_fds[c])) memset(backing_base+at,0,n);
        else punch_or_clear(at,n);
        done+=n;
    }
}
