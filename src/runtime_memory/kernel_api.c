// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: the sceKernel* memory functions the game calls (direct/flexible memory, protection, queries, batch maps) and their resolver.
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
#include "../../gpu/shim/bb_env.h"
static ABI uint64_t direct_size(void) { return GAME_POOL_SIZE; }
static ABI int32_t direct_allocate(int64_t low, int64_t high, uint64_t size,
                                  uint64_t alignment, int type, int64_t *out) {
    if (!alignment) alignment = PAGE;
    if (!out || low < 0 || high < 0 || !size || size % PAGE || !valid_alignment(alignment) || type < 0 || type > 10)
        return INVALID;
    uint64_t end = (uint64_t)high < GAME_POOL_SIZE ? (uint64_t)high : GAME_POOL_SIZE;
    if ((uint64_t)low >= end || size > end - (uint64_t)low) return NO_SPACE;
    write_lock();
    if (pool()) { write_unlock(); return NO_MEMORY; }
    uint64_t start = align_up((uint64_t)low, alignment);
    int slot = -1;
    for (int i=0; i<LIMIT; ++i) if (!blocks[i].used) { slot=i; break; }
    for (;;) {
        if (slot<0 || start > end || size > end-start) { write_unlock(); return NO_SPACE; }
        int overlap = 0;
        for (int i=0; i<LIMIT; ++i) {
            Block b = blocks[i];
            if (b.used && start < b.start+b.size && b.start < start+size) {
                start = align_up(b.start+b.size, alignment); overlap = 1; break;
            }
        }
        if (!overlap) break;
    }
    blocks[slot] = (Block){start, size, type, 1}; *out = (int64_t)start;
    static int ordinal;
    host_only_next = ordinal<64 && (host_only_ordinals>>ordinal & 1);
    ++ordinal;
    if (host_only(start) && host_only_count<16) {
        host_only_ranges[host_only_count][0]=start; host_only_ranges[host_only_count][1]=start+size;
        ++host_only_count;
        printf("Runtime: direct memory %#" PRIx64 "+%" PRIu64 " MiB without GPU memory (the game profile, BB_HOST_ONLY)\n",start,size>>20);
    }
    if (grow_size_at && size==*grow_size_at && grow_initial<size && !deferred_end &&
        ((whole_chunks && chunk_alloc) || private_import) && !lazy_game &&
        !mirror_hook) { /* bbport: the GPU process mirrors the pool, not a heap's chunks */
        deferred_phys=start+grow_initial; deferred_end=start+size;
        printf("Runtime: heap growing on demand: direct memory %#" PRIx64 "+%" PRIu64 " MiB, its GPU memory for the first %" PRIu64 " MiB now\n",
               start,size>>20,grow_initial>>20);
    }
    import_chunks_unlocked(start,size);
    ensure_chunks(start,size);
    release_pre_chunks();
    host_only_next=0;
    ++allocations; live_bytes += size;
    write_unlock();
    printf("Runtime: direct allocation offset=0x%" PRIx64 ", size=%" PRIu64 " bytes, type=%d (from guest %p)\n",
           start, size, type, __builtin_return_address(0));
    return 0;
}
/* bbport: `size` bytes of direct memory past the game's size (extra_pool_bytes), mapped for the
 * CPU and the GPU (read and write) at an address of the runtime's choosing; the GPU side is told
 * as of a game mapping. 0 or the error. */
static int32_t map_direct_locked(void **out, uint64_t size, int prot, int flags, int64_t physical, uint64_t alignment);
int runtime_extra_direct_map(uint64_t size, uint64_t alignment, void **out) {
    if (!out || !size || size % PAGE || !valid_alignment(alignment)) return INVALID;
    write_lock();
    if (pool()) { write_unlock(); return NO_MEMORY; }
    const uint64_t end=POOL_SIZE;
    uint64_t start=align_up(GAME_POOL_SIZE,alignment);
    int slot=-1;
    for (int i=0;i<LIMIT;++i) if (!blocks[i].used) { slot=i; break; }
    for (;;) {
        if (slot<0 || start>end || size>end-start) { write_unlock(); return NO_SPACE; }
        int overlap=0;
        for (int i=0;i<LIMIT;++i) {
            const Block b=blocks[i];
            if (b.used && start<b.start+b.size && b.start<start+size) { start=align_up(b.start+b.size,alignment); overlap=1; break; }
        }
        if (!overlap) break;
    }
    blocks[slot]=(Block){start,size,3,1};
    import_chunks_unlocked(start,size);
    ensure_chunks(start,size);
    release_pre_chunks();
    *out=NULL;
    const int32_t result=map_direct_locked(out,size,0x33,0,(int64_t)start,alignment);
    if (result) blocks[slot].used=0;
    write_unlock();
    flush_hooks();
    if (!result) printf("Runtime: port direct memory %p, size=%" PRIu64 " (past the game's %" PRIu64 " MiB)\n", *out, size, GAME_POOL_SIZE>>20);
    return result;
}
static int32_t map_direct_locked(void **out, uint64_t size, int prot, int flags, int64_t physical, uint64_t alignment) {
    if (!alignment) alignment=PAGE;
    if (!out || physical < 0 || (uint64_t)physical % PAGE || !size || size % PAGE || !valid_alignment(alignment) ||
        (prot & ~0x37)) return INVALID;
    if (!direct_range_allocated((uint64_t)physical,size)) return INVALID;
    int32_t result=place(out,size,prot,flags,alignment,KIND_DIRECT,block_type((uint64_t)physical),(uint64_t)physical);
    if (!result) ++maps;
    return result;
}
int bbgpu_gpu_touched(uint64_t va);
void bbgpu_straddle_report(void);
/* BB_ZERO_SCAN=seconds (diagnostics): that long after the first mapping, how much of each direct
 * memory mapping is still all-zero 4 KiB pages (memory the game reserved but never filled). */
static struct { uintptr_t address; uint64_t size; } zero_scan_maps[64];
static atomic_int zero_scan_count;
static void *zero_scan(void *arg) {
    sleep((unsigned)(uintptr_t)arg);
    static const uint8_t zero[4096];
    uint64_t total=0, zeros=0;
    const int count=atomic_load(&zero_scan_count);
    for (int i=0;i<count;++i) {
        uint64_t z=0;
        for (uint64_t at=0; at<zero_scan_maps[i].size; at+=4096)
            z += !memcmp((const void *)(zero_scan_maps[i].address+at), zero, 4096);
        printf("Zero scan: %#" PRIxPTR " %" PRIu64 " MiB, %" PRIu64 " MiB all-zero pages\n",
               zero_scan_maps[i].address, zero_scan_maps[i].size>>20, (z*4096)>>20);
        total+=zero_scan_maps[i].size; zeros+=z*4096;
    }
    printf("Zero scan: direct memory %" PRIu64 " MiB mapped, %" PRIu64 " MiB all-zero pages\n", total>>20, zeros>>20);
    /* Whole granules of zero pages: what lazily allocated chunks of that size would leave out. */
    static const unsigned shifts[]={21,24,26,28};
    for (int s=0;s<4;++s) {
        const uint64_t g=UINT64_C(1)<<shifts[s];
        uint64_t whole=0;
        for (int i=0;i<count;++i)
            for (uint64_t at=0; at<zero_scan_maps[i].size; at+=g) {
                const uint64_t n=zero_scan_maps[i].size-at<g ? zero_scan_maps[i].size-at : g;
                int all=1;
                for (uint64_t p=0;p<n && all;p+=4096) all=!memcmp((const void *)(zero_scan_maps[i].address+at+p),zero,4096);
                if (all) whole+=n;
            }
        printf("Zero scan: %" PRIu64 " MiB granules: %" PRIu64 " MiB in all-zero ones\n", g>>20, whole>>20);
    }
    /* 16 MiB granules (by address) the GPU never used: zero ones would cost nothing if allocated
     * lazily; nonzero ones only the CPU touched. */
    uint64_t zero_untouched=0, zero_touched=0, data_untouched=0, data_touched=0;
    for (int i=0;i<count;++i) {
        const uintptr_t a=zero_scan_maps[i].address, e=a+zero_scan_maps[i].size;
        for (uintptr_t g=a & ~(uintptr_t)0xffffff; g<e; g+=0x1000000) {
            const uintptr_t lo=g>a ? g : a, hi=g+0x1000000<e ? g+0x1000000 : e;
            int all=1;
            for (uintptr_t p=lo;p<hi && all;p+=4096) all=!memcmp((const void *)p,zero,4096);
            const int gpu=bbgpu_gpu_touched(g);
            *(all ? (gpu ? &zero_touched : &zero_untouched) : (gpu ? &data_touched : &data_untouched)) += hi-lo;
        }
    }
    printf("Zero scan: 16 MiB granules: all-zero %" PRIu64 " MiB never used by the GPU, %" PRIu64 " MiB used; "
           "with data %" PRIu64 " MiB never used by the GPU, %" PRIu64 " MiB used\n",
           zero_untouched>>20, zero_touched>>20, data_untouched>>20, data_touched>>20);
    bbgpu_straddle_report();
    return NULL;
}

static void zero_scan_note(uintptr_t address, uint64_t size) {
    const char *env=bb_env("BB_ZERO_SCAN");
    if (!env || !*env) return;
    const int index=atomic_fetch_add(&zero_scan_count,1);
    if (index>=64) return;
    zero_scan_maps[index].address=address; zero_scan_maps[index].size=size;
    if (index==0) {
        pthread_t thread;
        if (!pthread_create(&thread,NULL,zero_scan,(void *)(uintptr_t)strtoul(env,NULL,10))) pthread_detach(thread);
    }
}
static ABI int32_t direct_map(void **out, uint64_t size, int prot, int flags, int64_t physical, uint64_t alignment) {

    write_lock();
    int32_t result=map_direct_locked(out,size,prot,flags,physical,alignment);
    write_unlock();
    flush_hooks();
    if (!result) printf("Runtime: mapped direct memory %p, size=%" PRIu64 ", prot=0x%x\n", *out, size, prot);
    if (!result) zero_scan_note((uintptr_t)*out,size);
    else printf("Runtime: mapping direct memory %#" PRIx64 ", size=%" PRIu64 " failed (error %#x)\n", (uint64_t)physical, size, (unsigned)result);
    return result;
}
static ABI int32_t direct_map_named(void **out, uint64_t size, int prot, int flags, int64_t physical,
                                    uint64_t alignment, const char *name) {
    (void)name; return direct_map(out,size,prot,flags,physical,alignment);
}
static int32_t unmap_locked(uintptr_t start, uint64_t size) {
    if (start%PAGE || !size) return INVALID;
    uint64_t end=start+align_up(size,PAGE);
    for (size_t i=vma_index(start); i<vma_count && vmas[i].start<end; ++i)
        if (vmas[i].kind!=KIND_RESERVED) {
            uintptr_t a=vmas[i].start>start ? vmas[i].start : start, b=vmas[i].end<end ? vmas[i].end : end;
            queue_hook(HOOK_UNMAP,a,b-a);
        }
    for (size_t i=vma_index(start); i<vma_count && vmas[i].start<end; ++i) {
        uintptr_t a=vmas[i].start>start ? vmas[i].start : start, b=vmas[i].end<end ? vmas[i].end : end;
        if (private_import && vmas[i].kind==KIND_DIRECT && vmas[i].phys<POOL_SIZE) {
            /* The GPU keeps its import of a private region: the memory stays, out of reach. */
            static int told;
            if (!told++) printf("Runtime: direct memory unmapped at %#" PRIxPTR " stays reserved (private regions)\n",a);
            mprotect((void *)a,b-a,PROT_NONE);
            continue;
        }
        if (munmap((void *)a,b-a)) return INVALID;
    }
    drop_range(start,end);
    trap_forget(start,end);
    queue_mirror(MIRROR_UNMAP,start,end-start,0,0,0,0,0);
    return 0;
}
static ABI int32_t direct_unmap(void *address, uint64_t size) {
    /* The GPU forgets the range before it disappears (as shadPS4's early GPU unmap). */
    if (hook_unmap && !((uintptr_t)address%PAGE) && size) hook_unmap((uintptr_t)address,align_up(size,PAGE));
    write_lock();
    size_t before=pending_count;
    int32_t result=unmap_locked((uintptr_t)address,size);
    pending_count=before; /* already notified above */
    write_unlock();
    return result;
}
static ABI int32_t direct_release(uint64_t start, uint64_t size) {
    if (start % PAGE || size % PAGE) return INVALID;
    if (!size) return 0;
    write_lock();
    if (!direct_range_allocated(start,size)) { write_unlock(); return INVALID; }
    /* Mappings of released physical pages disappear with them. */
    for (size_t i=0; i<vma_count;) {
        Vma v=vmas[i];
        uint64_t length=v.end-v.start;
        if (v.kind==KIND_DIRECT && v.phys<start+size && start<v.phys+length) {
            uint64_t a=v.phys>start ? v.phys : start, b=v.phys+length<start+size ? v.phys+length : start+size;
            unmap_locked(v.start+(a-v.phys),b-a);
            i=vma_index(v.start);
            continue;
        }
        ++i;
    }
    for (int i=0; i<LIMIT; ++i) {
        Block *b=&blocks[i];
        if (!b->used || b->start>=start+size || start>=b->start+b->size) continue;
        uint64_t a=b->start>start ? b->start : start, e=b->start+b->size<start+size ? b->start+b->size : start+size;
        Block left=*b, right=*b;
        left.size=a-b->start; right.start=e; right.size=b->start+b->size-e;
        b->used=0;
        if (left.size) *b=left;
        if (right.size) for (int j=0;j<LIMIT;++j) if (!blocks[j].used) { blocks[j]=right; break; }
        live_bytes-=e-a;
    }
    zero_phys(start,size); /* zero on reuse */
    write_unlock();
    flush_hooks();
    return 0;
}
static int32_t map_flexible_locked(void **inout, uint64_t size, int prot, int flags) {
    if (!inout || !size || size%PAGE || (prot & ~0x37)) return INVALID;
    if (flexible_bytes+size>FLEXIBLE_SIZE || pool()) return NO_MEMORY;
    uint64_t phys=flex_alloc(size);
    if (phys==UINT64_MAX) return NO_MEMORY;
    int32_t result=place(inout,size,prot,flags,PAGE,KIND_FLEXIBLE,0,phys);
    if (result) flex_free(phys,size); else ++flexible_maps;
    return result;
}
static ABI int32_t map_flexible(void **inout, uint64_t size, int prot, int flags) {
    write_lock();
    int32_t result=map_flexible_locked(inout,size,prot,flags);
    write_unlock();
    flush_hooks();
    if (!result) printf("Runtime: mapped flexible memory %p, size=%" PRIu64 ", prot=0x%x\n", *inout, size, prot);
    else fprintf(stderr,"Runtime: flexible map of %" PRIu64 " bytes failed (0x%x), in use=%" PRIu64 "\n",
                 size, (unsigned)result, flexible_bytes);
    return result;
}
static ABI int32_t map_flexible_named(void **inout, uint64_t size, int prot, int flags, const char *name) {
    (void)name; return map_flexible(inout,size,prot,flags);
}
static ABI int32_t release_flexible(void *address, uint64_t size) { return direct_unmap(address,size); }
/* POSIX mmap: the game's libc grows its malloc heap with anonymous private mappings once the
 * heap it started with is full (seen after minutes of area streaming). They come from flexible
 * memory as sceKernelMapFlexibleMemory does; files are not mapped. MAP_FAILED (-1) and errno on
 * failure, as POSIX. */
static _Atomic uint64_t heap_growths;
uint64_t runtime_heap_growths(void) { return atomic_load(&heap_growths); }
static ABI void *posix_mmap(void *addr, uint64_t size, int prot, int flags, int fd, int64_t offset) {
    atomic_fetch_add(&heap_growths,1);
    enum { MAP_ANON_FLAG=0x1000, POSIX_ENOMEM=12, POSIX_ENODEV=19, POSIX_EINVAL=22 };
    if (!(flags & MAP_ANON_FLAG) || fd!=-1 || offset) { *runtime_errno()=POSIX_ENODEV; return (void *)-1; }
    if (!size) { *runtime_errno()=POSIX_EINVAL; return (void *)-1; }
    void *out=(flags & MAP_FIXED_FLAG) ? addr : NULL;
    const int32_t result=map_flexible(&out,align_up(size,PAGE),prot & 0x37,flags & MAP_FIXED_FLAG);
    uint64_t sites[3];
    runtime_guest_call_sites(sites);
    printf("Runtime: mmap %#" PRIx64 " bytes for the game (its heap grows) from +%#" PRIx64 " < +%#" PRIx64 " < +%#" PRIx64 "\n",
           size,sites[0],sites[1],sites[2]);
    if (result) { *runtime_errno()=result==NO_MEMORY ? POSIX_ENOMEM : POSIX_EINVAL; return (void *)-1; }
    return out;
}
static ABI int32_t reserve_range(void **inout, uint64_t size, int flags, uint64_t alignment) {
    if (!alignment) alignment=PAGE;
    if (!inout || !size || size%PAGE || !valid_alignment(alignment)) return INVALID;
    write_lock();
    if ((flags & MAP_FIXED_FLAG) && overlaps((uintptr_t)*inout,(uintptr_t)*inout+size,0)) {
        write_unlock(); return NO_MEMORY;
    }
    int32_t result=place(inout,size,0,flags,alignment,KIND_RESERVED,0,0);
    write_unlock();
    if (!result) printf("Runtime: reserved virtual range %p, size=%" PRIu64 "\n", *inout, size);
    return result;
}
static int32_t protect_locked(uintptr_t start, uint64_t size, int prot, int type) {
    uintptr_t end=start+align_up(size,PAGE);
    start&=~(uintptr_t)(PAGE-1);
    if ((prot & ~0x37) || !covered(start,end,0)) return INVALID;
    if (mprotect((void *)start,end-start,host_prot(prot))) return INVALID;
    trap_reapply(start,end); /* pages with write traps stay read-only */
    int error; size_t i=carve(start,end,&error);
    if (error) return NO_MEMORY;
    for (; i<vma_count && vmas[i].start<end; ++i) { vmas[i].prot=prot; if (type>=0) vmas[i].type=type; }
    queue_mirror(MIRROR_PROTECT,start,end-start,0,0,prot,type,0);
    queue_hook(HOOK_INVALIDATE,start,end-start);
    ++protects;
    return 0;
}
static ABI int32_t kernel_mprotect(const void *address, uint64_t size, int prot) {
    write_lock();
    int32_t result=protect_locked((uintptr_t)address,size,prot,-1);
    write_unlock();
    flush_hooks();
    return result;
}
static ABI int32_t query_protection(void *address, void **start, void **end, uint32_t *prot) {
    read_lock();
    size_t i=vma_index((uintptr_t)address);
    if (i==vma_count || vmas[i].start>(uintptr_t)address || vmas[i].kind==KIND_RESERVED) {
        read_unlock(); return ACCESS;
    }
    /* Report the whole contiguous run with identical protection. */
    size_t first=i, last=i;
    while (first && vmas[first-1].end==vmas[first].start && vmas[first-1].kind!=KIND_RESERVED && vmas[first-1].prot==vmas[i].prot) --first;
    while (last+1<vma_count && vmas[last+1].start==vmas[last].end && vmas[last+1].kind!=KIND_RESERVED && vmas[last+1].prot==vmas[i].prot) ++last;
    if (start) *start=(void *)vmas[first].start;
    if (end) *end=(void *)vmas[last].end;
    if (prot) *prot=(uint32_t)vmas[i].prot;
    __atomic_add_fetch(&queries,1,__ATOMIC_RELAXED);
    read_unlock();
    return 0;
}
static ABI int32_t virtual_query(const void *address, int flags, VirtualQueryInfo *info, uint64_t info_size) {
    if (!info || info_size!=sizeof(*info)) return INVALID;
    read_lock();
    size_t i=vma_index((uintptr_t)address);
    if (i==vma_count || (vmas[i].start>(uintptr_t)address && !(flags & 1))) { read_unlock(); return ACCESS; }
    Vma v=vmas[i];
    memset(info,0,sizeof(*info));
    info->start=v.start; info->end=v.end;
    info->offset=v.kind==KIND_DIRECT ? v.phys : 0;
    info->protection=v.prot; info->memory_type=v.type;
    info->flags=(v.kind==KIND_FLEXIBLE ? 1u : 0) | (v.kind==KIND_DIRECT ? 2u : 0) | (v.kind!=KIND_RESERVED ? 16u : 0);
    strcpy(info->name, v.kind==KIND_RESERVED ? "reserved" : v.kind==KIND_DIRECT ? "direct" : "flexible");
    __atomic_add_fetch(&queries,1,__ATOMIC_RELAXED);
    read_unlock();
    return 0;
}
static ABI int32_t direct_memory_type(uint64_t phys, int *type, void **start, void **end) {
    read_lock();
    for (int i=0;i<LIMIT;++i) if (blocks[i].used && phys>=blocks[i].start && phys-blocks[i].start<blocks[i].size) {
        if (type) *type=blocks[i].type;
        if (start) *start=(void *)(uintptr_t)blocks[i].start;
        if (end) *end=(void *)(uintptr_t)(blocks[i].start+blocks[i].size);
        read_unlock(); return 0;
    }
    read_unlock();
    return ACCESS;
}
/* The direct memory allocation containing `offset` (or, flags & 1, the next one after it). */
typedef struct { int64_t start, end; int32_t memory_type; } DirectQueryInfo;
static ABI int32_t direct_query(int64_t offset, int flags, DirectQueryInfo *info, uint64_t size) {
    if (!info || size < sizeof(DirectQueryInfo) || offset < 0) return INVALID;
    read_lock();
    int best=-1;
    for (int i=0;i<LIMIT;++i) {
        if (!blocks[i].used) continue;
        if (blocks[i].start>=GAME_POOL_SIZE) continue; /* the port's own (runtime_extra_direct_map) */
        const uint64_t at=(uint64_t)offset;
        if (at>=blocks[i].start && at-blocks[i].start<blocks[i].size) { best=i; break; }
        if ((flags & 1) && blocks[i].start>at && (best<0 || blocks[i].start<blocks[best].start)) best=i;
    }
    if (best>=0) {
        info->start=(int64_t)blocks[best].start;
        info->end=(int64_t)(blocks[best].start+blocks[best].size);
        info->memory_type=blocks[best].type;
    }
    read_unlock();
    return best>=0 ? 0 : (int32_t)UINT32_C(0x80020002); /* ENOENT */
}
static ABI int32_t batch_map2(BatchEntry *entries, int count, int *processed, int flags) {
    if (!entries || count<0) return INVALID;
    int32_t result=0; int done=0;
    write_lock();
    for (; done<count; ++done) {
        BatchEntry *e=&entries[done];
        switch (e->operation) {
        case 0: result=map_direct_locked(&e->start,e->length,(uint8_t)e->prot,flags,(int64_t)e->offset,0); break;
        case 1: result=unmap_locked((uintptr_t)e->start,e->length); break;
        case 2: result=protect_locked((uintptr_t)e->start,e->length,(uint8_t)e->prot,-1); break;
        case 3: result=map_flexible_locked(&e->start,e->length,(uint8_t)e->prot,flags); break;
        case 4: result=protect_locked((uintptr_t)e->start,e->length,(uint8_t)e->prot,e->type); break;
        default: result=INVALID;
        }
        if (result) break;
    }
    write_unlock();
    flush_hooks();
    if (processed) *processed=done;
    return result;
}
static ABI int32_t batch_map(BatchEntry *entries, int count, int *processed) {
    return batch_map2(entries,count,processed,MAP_FIXED_FLAG);
}
static const RuntimeExport exports[]={
    {"sceKernelGetDirectMemorySize",direct_size}, {"sceKernelAllocateDirectMemory",direct_allocate},
    {"sceKernelMapDirectMemory",direct_map}, {"sceKernelMapNamedDirectMemory",direct_map_named},
    {"sceKernelReleaseDirectMemory",direct_release}, {"sceKernelMunmap",direct_unmap}, {"munmap",direct_unmap},
    {"sceKernelMapFlexibleMemory",map_flexible}, {"sceKernelMapNamedFlexibleMemory",map_flexible_named},
    {"sceKernelReleaseFlexibleMemory",release_flexible}, {"sceKernelReserveVirtualRange",reserve_range},
    {"sceKernelMprotect",kernel_mprotect}, {"sceKernelQueryMemoryProtection",query_protection},
    {"sceKernelVirtualQuery",virtual_query}, {"sceKernelGetDirectMemoryType",direct_memory_type},
    {"sceKernelDirectMemoryQuery",direct_query},
    {"sceKernelBatchMap",batch_map}, {"sceKernelBatchMap2",batch_map2},
    {"mmap",posix_mmap},
};
uintptr_t runtime_memory_resolve(const char *name) {
    /* Direct-memory NIDs are exercised by test_runtime.c before names are loaded. */
    if (runtime_nid_is(name, "pO96TwzOm5E")) return (uintptr_t)direct_size;
    if (runtime_nid_is(name, "rTXw65xmLIA")) return (uintptr_t)direct_allocate;
    if (runtime_nid_is(name, "L-Q3LEjIbgA")) return (uintptr_t)direct_map;
    if (runtime_nid_is(name, "MBuItvba6z8")) return (uintptr_t)direct_release;
    if (runtime_nid_is(name, "cQke9UuBQOk")) return (uintptr_t)direct_unmap;
    return RUNTIME_LOOKUP(exports,name);
}
