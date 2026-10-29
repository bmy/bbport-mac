// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* runtime_memory.c part: the low region [LOW_MIN, USER_MIN) for host-owned memory the guest sees (image, stacks, trampolines).
 * #included by runtime_memory.c in order (one translation unit: it shares that file's
 * statics); not compiled on its own. */
/* Host-owned memory the guest can see (image, stacks, trampolines) lives in
 * [LOW_MIN, USER_MIN): PS4 code packs pointers into 40-bit fields. */
#define LOW_MIN UINT64_C(0x0800000000)
static uintptr_t low_next=LOW_MIN;
/* The guest's address range, reserved before anything else can land there (Windows: one
 * placeholder the game's mappings replace in; Linux needs nothing). */
void runtime_memory_reserve(void) {
#ifdef _WIN32
    if (bb_reserve_arena(USER_MIN,USER_MAX)) exit(21);
#endif
}
/* bbport: what is mapped in the low region and with which protection (the image's segments, stacks,
 * trampolines), for runtime_memory_low_info: the online module reads the game's static pointers
 * there (#282: BBSC took every one for unmapped, no hunter's kit, "?GoodsName?" items). */
typedef struct { uintptr_t start, end; int prot; } LowRange;
#define MAX_LOW_RANGES 2048
static LowRange low_ranges[MAX_LOW_RANGES];
static size_t low_range_count;
/* [start, end) now has `prot` (write lock held); ranges it splits keep theirs around it. */
static void low_record(uintptr_t start, uintptr_t end, int prot) {
    static LowRange kept[MAX_LOW_RANGES]; /* under the write lock; guest threads have small stacks */
    size_t n=0;
    for (size_t i=0;i<low_range_count;++i) {
        const LowRange r=low_ranges[i];
        if (r.end<=start || r.start>=end) { if (n<MAX_LOW_RANGES) kept[n++]=r; continue; }
        if (r.start<start && n<MAX_LOW_RANGES) kept[n++]=(LowRange){r.start,start,r.prot};
        if (r.end>end && n<MAX_LOW_RANGES) kept[n++]=(LowRange){end,r.end,r.prot};
    }
    if (n<MAX_LOW_RANGES) kept[n++]=(LowRange){start,end,prot};
    /* Sorted by start (few entries: insertion sort). */
    for (size_t i=1;i<n;++i) {
        LowRange r=kept[i]; size_t j=i;
        while (j>0 && kept[j-1].start>r.start) { kept[j]=kept[j-1]; --j; }
        kept[j]=r;
    }
    /* Neighbours with the same protection join: every loader mprotect of one page left a range
     * of its own, and past MAX_LOW_RANGES the copy below ran over the variables after
     * low_ranges (Windows: trap_reasons, the write trap table, pointed into the low region). */
    size_t joined=0;
    for (size_t i=0;i<n;++i) {
        if (joined && kept[joined-1].end==kept[i].start && kept[joined-1].prot==kept[i].prot) kept[joined-1].end=kept[i].end;
        else kept[joined++]=kept[i];
    }
    n=joined;
    memcpy(low_ranges,kept,n*sizeof(*kept));
    low_range_count=n;
}
/* bbport: the loader changed the protection of part of the low region (mmap PROT_ bits). */
void runtime_low_protect(uintptr_t start, size_t size, int prot) {
    if (start<LOW_MIN || start>=USER_MIN || !size) return;
    write_lock();
    low_record(start,start+align_up(size,PAGE),prot);
    write_unlock();
}
/* bbport: the low region's mapping holding `address`: 1 with its PS4 protection bits (1 read,
 * 2 write, 4 execute) and its end, 0 when it is not mapped (gaps between segments included). */
int runtime_memory_low_info(uintptr_t address, int *prot, uintptr_t *end) {
    if (address<LOW_MIN || address>=USER_MIN) return 0;
    int found=0;
    read_lock();
    for (size_t i=0;i<low_range_count;++i) {
        const LowRange r=low_ranges[i];
        if (address>=r.start && address<r.end) {
            if (r.prot & (PROT_READ|PROT_WRITE|PROT_EXEC)) {
                *prot=((r.prot & PROT_READ) ? 1 : 0) | ((r.prot & PROT_WRITE) ? 2 : 0) | ((r.prot & PROT_EXEC) ? 4 : 0);
                *end=r.end;
                found=1;
            }
            break;
        }
    }
    read_unlock();
    return found;
}
void *runtime_low_map(size_t size, int prot) {
#ifdef _WIN32
    const uint64_t granule=UINT64_C(65536); /* VirtualAlloc places allocations at 64 KiB */
#else
    const uint64_t granule=PAGE;
#endif
    size=align_up(size,PAGE);
    write_lock();
    void *p=MAP_FAILED;
    low_next=align_up(low_next,granule);
    /* bbport (macOS): host low memory ends below Rosetta's commpage (BB_LOW_MAX, platform.h). */
    while (low_next+size<=BB_LOW_MAX) {
        p=bb_map_noreplace((void *)low_next,size,prot);
        low_next=align_up(low_next+size+PAGE,granule); /* unmapped gap catches overruns */
        if (p!=MAP_FAILED) break;
    }
    if (p!=MAP_FAILED) low_record((uintptr_t)p,(uintptr_t)p+size,prot);
#ifdef _WIN32
    fprintf(stderr,"Runtime: low memory %p+%#zx (prot %d)\n",p==MAP_FAILED ? NULL : p,size,prot);
#endif
    write_unlock();
    return p==MAP_FAILED ? NULL : p;
}
