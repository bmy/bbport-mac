// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: the GPU library's setters (gpu/shim): diagnostics toggles, chunk allocators, pinned ranges.
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
/* ---- GPU library interface (gpu/shim/bbgpu.cpp) ---- */
/* Optimizations switched off at run time (diagnostics): the number in the file named by
 * BB_TOGGLE_FILE, re-read every 250 ms. Bits: 1 region cache, 2 fetch shader cache,
 * 4 page tracking early exit, 8 pending-op poll limit, 16 threaded Vulkan recording,
 * 32 texture descriptor cache, 64 lock-free upload check, 128 image lookup cache,
 * 256 buffer uploads on the recording thread, 512 barrier tracker insert memo,
 * 1024 texture binding memo, 2048 page-granular read tracking for barriers,
 * 4096 resource sharps read by the draw-preparation workers,
 * 8192 prepared draws from the draw-preparation workers,
 * 16384 small read-only buffer copies on the recording thread,
 * 32768 hot pages (opt-in with BB_HOT_PAGES=1),
 * 65536 unprotect the 256 KiB window around a guest write fault (BB_FAULT_WINDOW KiB),
 * 131072 large guest memory copies split across copy threads (BB_COPY_THREADS),
 * 262144 wait for guest copies at fences again,
 * 524288 small guest copies batched for the copy threads instead of the recording thread,
 * 1073741824 the lock-free UpdateImage path for clean, tracked images. */
/* Bits 32 and up: the draw pipeline and related GPU thread work (gpu/shim/bbport_toggles.h). */
uint64_t runtime_disabled_optimizations;
/* bbport: a second number in BB_TOGGLE_FILE: temporary experiment bits (BbToggle::Experiment). */
uint64_t runtime_experiment_bits;
/* Speculative readers of guest memory (GPU draw-preparation workers) register a recovery
 * point: a fault on that thread jumps back to it instead of terminating (probe.c). */
__thread sigjmp_buf *runtime_fault_recover;
/* BB_EXPERIMENT_CYCLE=bits,bits,...:seconds (test builds sent to players): the experiment bits
 * step through the list, each for that many seconds, over and over, without a toggle file; a
 * player standing still in one place gives an A/B of every step in one log. */
static unsigned long long experiment_cycle[16];
static int experiment_cycle_count;
static long experiment_cycle_seconds=20;
static void *toggle_watcher(void *path) {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC,&start);
    for (unsigned long long last=ULLONG_MAX, last_experiment=ULLONG_MAX;;) {
        FILE *f=path ? fopen(path,"r") : NULL;
        unsigned long long value=0, experiment=0;
        if (f) { if (fscanf(f,"%llu %llu",&value,&experiment)<1) value=0; fclose(f); }
        if (experiment_cycle_count) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC,&now);
            const long step=(long)(now.tv_sec-start.tv_sec)/experiment_cycle_seconds;
            experiment=experiment_cycle[step%experiment_cycle_count];
        }
        __atomic_store_n(&runtime_experiment_bits,(uint64_t)experiment,__ATOMIC_RELEASE);
        if (experiment!=last_experiment) {
            printf("Runtime: experiment bits=%llu\n",experiment);
            last_experiment=experiment;
        }
        if (value!=last) {
            /* bbport: BB_TOGGLES (set at start) stays disabled. */
            const char *fixed=bb_env("BB_TOGGLES");
            __atomic_store_n(&runtime_disabled_optimizations,
                             (uint64_t)value|(fixed ? strtoull(fixed,NULL,0) : 0),__ATOMIC_RELEASE);
            printf("Runtime: disabled optimizations mask=%llu\n",value);
            last=value;
        }
        struct timespec t={0,250000000}; nanosleep(&t,NULL);
    }
    return NULL;
}
static int parse_experiment_cycle(void) {
    const char *cycle=bb_env("BB_EXPERIMENT_CYCLE");
    if (!cycle || !*cycle) return 0;
    for (const char *p=cycle; *p && *p!=':' && experiment_cycle_count<16;) {
        char *end;
        experiment_cycle[experiment_cycle_count++]=strtoull(p,&end,10);
        p=*end==',' ? end+1 : end;
    }
    const char *colon=strchr(cycle,':');
    if (colon && atol(colon+1)>0) experiment_cycle_seconds=atol(colon+1);
    printf("Runtime: experiment cycle of %d steps, %ld s each\n",experiment_cycle_count,experiment_cycle_seconds);
    return experiment_cycle_count;
}
/* bbport: the GPU library provides direct memory chunks (see CHUNK). */
void runtime_memory_set_guest_chunk_whole(int whole) { whole_chunks=whole; }
/* bbport: the host address of direct memory phys in the backing view (host memory import). */
void *runtime_memory_backing_pointer(uint64_t phys) { return backing_base ? backing_at(phys) : NULL; }
/* bbport: lazy guest memory (see lazy_game). */
void runtime_memory_set_guest_lazy(int lazy) { lazy_game=lazy; }
/* bbport: the memfd of guest memory (lazy guest memory: udmabuf over it), -1 before it exists. */
int runtime_memory_pool_fd(void) { return pool_fd; }
/* bbport: [phys, phys+size) of the memfd is about to be pinned by the GPU (an import; noted
 * before its pages are taken): kept on release from now on. */
void runtime_memory_note_pinned(uint64_t phys, uint64_t size) {
    if (!pinned || !size) return;
    pthread_mutex_lock(&pin_mutex);
    for (uint64_t g=phys/PIN_GRANULE; g<=(phys+size-1)/PIN_GRANULE && g<=(POOL_SIZE+FLEX_SPAN)/PIN_GRANULE; ++g)
        __atomic_store_n(&pinned[g],1,__ATOMIC_RELEASE);
    pthread_mutex_unlock(&pin_mutex);
}
/* bbport: private regions (see CHUNK_PRIVATE); import(phys, size, pointer) gives the GPU an
 * allocation's memory at its first mapping, 0 on success. */
void runtime_memory_set_guest_chunk_private(int (*import)(uint64_t phys, uint64_t size, void *pointer)) {
    write_lock();
    private_import=import;
    whole_chunks=1;
    write_unlock();
}
void runtime_memory_set_guest_chunk_allocator(int (*alloc)(uint64_t phys, uint64_t size)) {
    write_lock();
    chunk_alloc=alloc;
    write_unlock();
}
