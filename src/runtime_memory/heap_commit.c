// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: heaps growing and shrinking on demand (BB_HEAP_GROW): chunks made ahead, commit, decommit.
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
/* bbport: [phys, phys+size) of the deferred range gets its chunk (BB_HEAP_GROW): the game's
 * mappings of it are moved onto it (what the memfd held there, nothing normally, copied first) and
 * the GPU side is told (unmap, map). 0, or -1 when it cannot. */
static uint64_t trap_now_ns(void);
/* bbport: the next heap piece's chunk made ahead, on a thread of its own (runtime_memory_commit_ahead):
 * runtime_memory_commit takes it instead of making it in the game's allocating thread, its heap
 * locked. A host memory import of 320 MiB took 120-170 ms there while areas streamed in (RX 7800 XT,
 * Windows, 2026-10-27; 12 ms in a quiet process), every one a frame of that length. Up to
 * AHEAD_SLOTS ranges at once (pieces were asked for 0.1-0.2 s apart while loading, one ahead was
 * still being made); each taken by the commit of exactly its range. */
#define AHEAD_SLOTS 4
#include "../../gpu/shim/bb_env.h"
static pthread_mutex_t ahead_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ahead_cond=PTHREAD_COND_INITIALIZER;
static struct { int state; uint64_t phys, size; int fd; } ahead[AHEAD_SLOTS]; /* state: 0 free, 1 being made, 2 made */
static void *ahead_main(void *arg) {
    const size_t slot=(size_t)(uintptr_t)arg;
    pthread_mutex_lock(&ahead_mutex);
    const uint64_t phys=ahead[slot].phys, size=ahead[slot].size;
    pthread_mutex_unlock(&ahead_mutex);
    const uint64_t t0=trap_now_ns();
    const int fd=chunk_alloc(phys,size);
    printf("Runtime: direct memory %#" PRIx64 "+%" PRIu64 " MiB: its chunk made ahead in %.0f ms%s\n",phys,size>>20,
           (trap_now_ns()-t0)/1e6,fd>=0 || fd==CHUNK_IMPORTED ? "" : " (failed)");
    pthread_mutex_lock(&ahead_mutex);
    ahead[slot].fd=fd;
    ahead[slot].state=2;
    pthread_cond_broadcast(&ahead_cond);
    pthread_mutex_unlock(&ahead_mutex);
    return NULL;
}
void runtime_memory_commit_ahead(uint64_t phys, uint64_t size) {
    if (!chunk_alloc || private_import || !chunk_fds || !size || size%PAGE || phys%PAGE || phys+size>POOL_SIZE ||
        in_host_only(phys)) return;
    pthread_mutex_lock(&ahead_mutex);
    size_t slot=AHEAD_SLOTS;
    for (size_t i=0;i<AHEAD_SLOTS;++i) {
        if (ahead[i].state && ahead[i].phys==phys) { pthread_mutex_unlock(&ahead_mutex); return; }
        if (!ahead[i].state && slot==AHEAD_SLOTS) slot=i;
    }
    if (slot==AHEAD_SLOTS) { pthread_mutex_unlock(&ahead_mutex); return; }
    ahead[slot].state=1; ahead[slot].phys=phys; ahead[slot].size=size; ahead[slot].fd=-1;
    pthread_mutex_unlock(&ahead_mutex);
    pthread_t thread;
    if (pthread_create(&thread,NULL,ahead_main,(void *)(uintptr_t)slot)==0) { pthread_detach(thread); return; }
    pthread_mutex_lock(&ahead_mutex);
    ahead[slot].state=0;
    pthread_mutex_unlock(&ahead_mutex);
}
/* The chunk made ahead for exactly [phys,+size) (waited for while it is being made): 1 and *fd,
 * else 0. */
static int take_ahead(uint64_t phys, uint64_t size, int *fd) {
    int took=0;
    pthread_mutex_lock(&ahead_mutex);
    for (size_t i=0;i<AHEAD_SLOTS;++i) {
        if (!ahead[i].state || ahead[i].phys!=phys || ahead[i].size!=size) continue;
        while (ahead[i].state==1) pthread_cond_wait(&ahead_cond,&ahead_mutex);
        if (ahead[i].fd>=0 || ahead[i].fd==CHUNK_IMPORTED) { *fd=ahead[i].fd; took=1; }
        ahead[i].state=0;
        break;
    }
    pthread_mutex_unlock(&ahead_mutex);
    return took;
}
int runtime_memory_commit(uint64_t phys, uint64_t size) {
    /* The GPU library's chunk for the piece, made before the lock (BB_COMMIT_IMPORT_LOCKED=1: under
     * it, as before). The checks below still decide; a chunk made for nothing is given back. */
    static int locked_import=-1;
    if (locked_import<0) { const char *e=bb_env("BB_COMMIT_IMPORT_LOCKED"); locked_import=e && e[0]=='1'; }
    memset(pre_chunks,0,sizeof(pre_chunks));
    const uint64_t t0=trap_now_ns();
    int ahead_fd;
    if (take_ahead(phys,size,&ahead_fd)) {
        pre_chunks[0]=(PreChunk){1,ahead_fd,phys,size};
    } else if (!locked_import && !private_import && chunk_alloc && chunk_fds && phys<POOL_SIZE && size && size%PAGE==0 &&
        phys%PAGE==0 && phys+size<=POOL_SIZE && !in_host_only(phys)) {
        const int fd=chunk_alloc(phys,size);
        if (fd>=0 || fd==CHUNK_IMPORTED) pre_chunks[0]=(PreChunk){1,fd,phys,size};
    }
    const uint64_t t1=trap_now_ns();
    write_lock();
    const uint64_t t2=trap_now_ns();
    uint64_t next;
    if (private_import) {
        /* Private regions: the allocation's region holds it all; the piece's pages get imported. */
        const Region *pr=region_at(phys,&next);
        if (!(deferred_end>deferred_phys) || phys<deferred_phys || phys+size>deferred_end || !pr ||
            pr->fd!=CHUNK_PRIVATE || phys+size>pr->phys+pr->size) { write_unlock(); return -1; }
        const uintptr_t va=pr->va+(phys-pr->phys);
        if (private_import(phys,size,(void *)va)) { write_unlock(); return -1; }
        queue_hook(HOOK_UNMAP,va,size);
        queue_hook(HOOK_MAP,va,size);
        write_unlock();
        flush_hooks();
        printf("Runtime: heap grown: direct memory %#" PRIx64 "+%" PRIu64 " MiB imported by the GPU (private)\n",phys,size>>20);
        return 0;
    }
    if (!(deferred_end>deferred_phys) || phys<deferred_phys || phys+size>deferred_end || !size ||
        size%PAGE || phys%PAGE || region_at(phys,&next) || next<phys+size || region_count==MAX_REGIONS) {
        write_unlock(); return -1;
    }
    /* Pages the memfd allocated there (the reservation was touched after all): kept. */
    int touched=0;
/* Windows: the section has no holes to tell; the reservation is never touched. bbport (macOS):
 * neither has a shared memory object (no lseek, no pread), and its chunks are host imports of the
 * pool's own pages, which need no copy. */
#if !defined(_WIN32) && !defined(__APPLE__)
    off_t data=lseek(pool_fd,(off_t)phys,SEEK_DATA);
    if (data>=0 && (uint64_t)data<phys+size) touched=1;
#endif
    unsigned char *saved=NULL;
    if (touched) {
        saved=malloc(size);
        if (!saved || pread(pool_fd,saved,size,(off_t)phys)!=(ssize_t)size) { free(saved); write_unlock(); return -1; }
    }
    add_region(phys,size);
    const Region *r=region_at(phys,&next);
    if (r && r->fd==CHUNK_IMPORTED) {
        /* Host memory import (NVIDIA): the GPU imported the memfd's own pages there; the game's
         * mappings stay as they are, the GPU side only hears of it. */
        free(saved);
        for (size_t i=0;i<vma_count;++i) {
            const Vma v=vmas[i];
            if (v.kind!=KIND_DIRECT) continue;
            const uint64_t vp=v.phys, ve=v.phys+(v.end-v.start);
            const uint64_t a=vp>phys ? vp : phys, b=ve<phys+size ? ve : phys+size;
            if (a>=b) continue;
            queue_hook(HOOK_UNMAP,v.start+(a-vp),b-a);
            queue_hook(HOOK_MAP,v.start+(a-vp),b-a);
        }
        const uint64_t t3=trap_now_ns();
        write_unlock();
        flush_hooks();
        const uint64_t t4=trap_now_ns();
        printf("Runtime: heap grown: direct memory %#" PRIx64 "+%" PRIu64 " MiB imported by the GPU "
               "(import %.1f ms, lock wait %.1f ms, locked %.1f ms, GPU side told %.1f ms)\n",phys,size>>20,
               (t1-t0)/1e6,(t2-t1)/1e6,(t3-t2)/1e6,(t4-t3)/1e6);
        return 0;
    }
    if (!r || r->fd<0) {
        /* No chunk (add_region left the range to the memfd): as it was. */
        free(saved); write_unlock(); return -1;
    }
    if (saved) { memcpy(backing_base+phys,saved,size); free(saved); }
#ifndef __APPLE__ /* bbport (macOS): no hole punching; the backing view is the chunk now */
    fallocate(pool_fd,FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE,(off_t)phys,(off_t)size);
#endif
    for (size_t i=0;i<vma_count;++i) {
        const Vma v=vmas[i];
        if (v.kind!=KIND_DIRECT) continue;
        const uint64_t vp=v.phys, ve=v.phys+(v.end-v.start);
        const uint64_t a=vp>phys ? vp : phys, b=ve<phys+size ? ve : phys+size;
        if (a>=b) continue;
        const uintptr_t va=v.start+(a-vp);
        if (mmap((void *)va,b-a,host_prot(v.prot),MAP_SHARED|MAP_FIXED,r->fd,(off_t)(a-r->phys))==MAP_FAILED) {
            fprintf(stderr,"STOP: moving %#" PRIxPTR "+%#" PRIx64 " onto its new GPU memory failed: %s\n",va,b-a,strerror(errno));
            exit(21);
        }
        queue_hook(HOOK_UNMAP,va,b-a);
        queue_hook(HOOK_MAP,va,b-a);
    }
    write_unlock();
    flush_hooks();
    printf("Runtime: heap grown: direct memory %#" PRIx64 "+%" PRIu64 " MiB now has its GPU memory%s\n",
           phys,size>>20,touched ? " (the memfd's pages there copied)" : "");
    return 0;
}
/* bbport: the GPU library frees a chunk the runtime gave back (runtime_memory_decommit). */
void runtime_memory_set_guest_chunk_free(void (*free_chunk)(uint64_t phys, uint64_t size)) {
    write_lock(); chunk_free=free_chunk; write_unlock();
}
/* bbport: [phys, phys+size) of direct memory holds nothing of interest any more (a heap's piece
 * given back): its pages go back to the system. Linux: holes in the memfd (zero again). Windows: the
 * section's pages reset and trimmed through every view of them (the backing view, the game's
 * mappings), not zeroed; again once the GPU's import let them go (runtime_memory_discard_direct).
 * The write lock held. */
static void discard_locked(uint64_t phys, uint64_t size) {
#ifdef _WIN32
    bb_discard(backing_at(phys),size);
    bb_section_discard(pool_fd,phys,size);
    for (size_t i=0;i<vma_count;++i) {
        const Vma v=vmas[i];
        if (v.kind!=KIND_DIRECT) continue;
        const uint64_t vp=v.phys, ve=v.phys+(v.end-v.start);
        const uint64_t a=vp>phys ? vp : phys, b=ve<phys+size ? ve : phys+size;
        if (a<b) bb_discard((void *)(v.start+(a-vp)),b-a);
    }
#elif defined(__APPLE__)
    pool_discard(phys,size); /* bbport (macOS): no hole punching */
#else
    fallocate(pool_fd,FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE,(off_t)phys,(off_t)size);
#endif
}
void runtime_memory_discard_direct(uint64_t phys, uint64_t size) {
#ifdef _WIN32
    write_lock();
    discard_locked(phys,size);
    write_unlock();
#else
    (void)phys; (void)size; /* the memfd's hole is there already */
#endif
}
/* bbport: the reverse of runtime_memory_commit (BB_HEAP_GROW): a piece the heap holds reserved
 * again goes back to the memfd (a hole: no RAM), the GPU side is told, its chunk is freed. 0, or
 * -1 when it cannot. */
int runtime_memory_decommit(uint64_t phys, uint64_t size) {
    write_lock();
    if (!chunk_free || !(deferred_end>deferred_phys) || phys<deferred_phys || phys+size>deferred_end) {
        write_unlock(); return -1;
    }
    if (private_import) {
        /* Private regions: the piece's import goes; its pages go back to the system (zero again). */
        uint64_t next;
        const Region *pr=region_at(phys,&next);
        if (!pr || pr->fd!=CHUNK_PRIVATE || phys+size>pr->phys+pr->size) { write_unlock(); return -1; }
        const uintptr_t va=pr->va+(phys-pr->phys);
        madvise((void *)va,size,MADV_DONTNEED);
        queue_hook(HOOK_UNMAP,va,size);
        queue_hook(HOOK_MAP,va,size);
        void (*release)(uint64_t, uint64_t)=chunk_free;
        write_unlock();
        flush_hooks();
        release(phys,size);
        printf("Runtime: heap shrunk: direct memory %#" PRIx64 "+%" PRIu64 " MiB gave its GPU memory back (private)\n",phys,size>>20);
        return 0;
    }
    size_t index=region_count;
    for (size_t i=0;i<region_count;++i)
        if (regions[i].phys==phys && regions[i].size==size) { index=i; break; }
    if (index<region_count && regions[index].fd==CHUNK_IMPORTED) {
        /* Host memory import (NVIDIA): the import goes, the memfd's pages there are punched out. */
        for (size_t i=0;i<vma_count;++i) {
            const Vma v=vmas[i];
            if (v.kind!=KIND_DIRECT) continue;
            const uint64_t vp=v.phys, ve=v.phys+(v.end-v.start);
            const uint64_t a=vp>phys ? vp : phys, b=ve<phys+size ? ve : phys+size;
            if (a>=b) continue;
            queue_hook(HOOK_UNMAP,v.start+(a-vp),b-a);
            queue_hook(HOOK_MAP,v.start+(a-vp),b-a);
        }
        for (size_t i=index;i+1<region_count;++i) regions[i]=regions[i+1];
        --region_count;
        discard_locked(phys,size);
        void (*release)(uint64_t, uint64_t)=chunk_free;
        write_unlock();
        flush_hooks();
        release(phys,size);
        printf("Runtime: heap shrunk: direct memory %#" PRIx64 "+%" PRIu64 " MiB gave its GPU memory back (host import)\n",phys,size>>20);
        return 0;
    }
    /* Only where the GPU library frees the chunk (dma-buf chunks; not host imports). */
    if (index==region_count || regions[index].fd<0 || !chunk_free || !(deferred_end>deferred_phys) ||
        phys<deferred_phys || phys+size>deferred_end) { write_unlock(); return -1; }
    const int fd=regions[index].fd;
    for (size_t i=0;i<vma_count;++i) {
        const Vma v=vmas[i];
        if (v.kind!=KIND_DIRECT) continue;
        const uint64_t vp=v.phys, ve=v.phys+(v.end-v.start);
        const uint64_t a=vp>phys ? vp : phys, b=ve<phys+size ? ve : phys+size;
        if (a>=b) continue;
        const uintptr_t va=v.start+(a-vp);
        if (mmap((void *)va,b-a,host_prot(v.prot),MAP_SHARED|MAP_FIXED,pool_fd,(off_t)a)==MAP_FAILED) {
            fprintf(stderr,"STOP: moving %#" PRIxPTR "+%#" PRIx64 " back to the memfd failed: %s\n",va,b-a,strerror(errno));
            exit(21);
        }
        queue_hook(HOOK_UNMAP,va,b-a);
        queue_hook(HOOK_MAP,va,b-a);
    }
    if (mmap(backing_base+phys,size,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,pool_fd,(off_t)phys)==MAP_FAILED) {
        fprintf(stderr,"STOP: the backing view of %#" PRIx64 " back to the memfd failed: %s\n",phys,strerror(errno));
        exit(21);
    }
#ifdef __APPLE__
    pool_discard(phys,size); /* bbport (macOS): no hole punching */
#else
    fallocate(pool_fd,FALLOC_FL_PUNCH_HOLE|FALLOC_FL_KEEP_SIZE,(off_t)phys,(off_t)size);
#endif
    for (size_t i=index;i+1<region_count;++i) regions[i]=regions[i+1];
    --region_count;
    close(fd);
    void (*release)(uint64_t, uint64_t)=chunk_free;
    write_unlock();
    flush_hooks();
    if (release) release(phys,size);
    printf("Runtime: heap shrunk: direct memory %#" PRIx64 "+%" PRIu64 " MiB gave its GPU memory back\n",phys,size>>20);
    return 0;
}
