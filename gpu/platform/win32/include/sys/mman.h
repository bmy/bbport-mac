// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the mmap/mprotect subset the GPU library uses, over VirtualAlloc/VirtualProtect
// (posix_win32.cpp). Anonymous mappings only; protections of guest memory (the runtime's views)
// change with VirtualProtect as on Linux with mprotect.
#pragma once

#include <stddef.h>
#include <sys/types.h>

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4

#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_FIXED_NOREPLACE 0x100000
#define MAP_FAILED ((void*)-1)

#define MADV_DONTNEED 4

#ifdef __cplusplus
extern "C" {
#endif

void* mmap(void* address, size_t size, int prot, int flags, int fd, off_t offset);
int munmap(void* address, size_t size);
int mprotect(void* address, size_t size, int prot);
int madvise(void* address, size_t size, int advice);

/// Guest memory (posix_memory.cpp): a pagefile-backed section standing for a memfd (mmap maps
/// views of it, fallocate zeroes), and the placeholder reservation MAP_FIXED maps replace in.
int bb_section_create(unsigned long long size);
int bb_reserve_arena(unsigned long long begin, unsigned long long end);
/* Committed private memory replacing [address, address + size) of the arena (bbport: the GPU
 * imports it; see runtime_memory.c private regions). MAP_FAILED on failure. */
void* bb_private_at(void* address, size_t size);
/// bbport: [address, +size) of a view or private memory holds nothing of interest any more: its pages
/// go back to the system (MEM_RESET, out of the working set). Not zeroed: a heap's piece given back.
void bb_discard(void* address, size_t size);
/// The same for [offset, +length) of a section (bb_section_create), through its own view.
int bb_section_discard(int fd, unsigned long long offset, unsigned long long length);
#define FALLOC_FL_KEEP_SIZE 1
#define FALLOC_FL_PUNCH_HOLE 2
int fallocate(int fd, int mode, off_t offset, off_t length);

#ifdef __cplusplus
}
#endif
