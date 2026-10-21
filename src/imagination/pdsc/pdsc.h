/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef PDSC_H
#define PDSC_H

/**
 * \file pdsc.h
 *
 * \brief Main PDS compiler interface header.
 */

#include "util/bitset.h"
#include "util/blob.h"
#include "util/macros.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* TODO: alloc of temps, consts, etc. is by block size, not one by one. */

/* TODO: probably don't have all the meta stuff during release builds. */

typedef struct _pdsc_program pdsc_program;

enum ENUM_PACKED pdsc_feature {
   PDSC_FEATURE_DOUT_EXT = BITFIELD_BIT(0),
   PDSC_FEATURE_IDIV = BITFIELD_BIT(1),
   PDSC_FEATURE_STMP = BITFIELD_BIT(2),
   PDSC_FEATURE_DDMADT = BITFIELD_BIT(3),
   PDSC_FEATURE_CACHE_HIERARCHY = BITFIELD_BIT(4),
   PDSC_FEATURE_MCU_CACHE_CONTROLS = BITFIELD_BIT(5),
};
static_assert(sizeof(enum pdsc_feature) == sizeof(uint8_t),
              "sizeof(enum pdsc_feature) != sizeof(uint8_t)");

void pdsc_init(void);

pdsc_program *pdsc_program_create(void *mem_ctx,
                                  enum pdsc_feature features,
                                  const char *name);

void pdsc_validate_program(const pdsc_program *p);
void pdsc_print_program(FILE *fp,
                        const pdsc_program *p,
                        const uint32_t *const_buffer);

void pdsc_encode_program(pdsc_program *p);
void pdsc_decode_program(pdsc_program *p);
void pdsc_finalize_program(pdsc_program *p);

typedef struct _pdsc_patch {
   uint8_t patch_id;
   uint64_t value;
} pdsc_patch;
void pdsc_patch_program(pdsc_program *p,
                        unsigned num_patches,
                        const pdsc_patch patches[num_patches],
                        uint32_t *buffer);

void pdsc_print_binary(FILE *fp, const pdsc_program *p);

uint8_t pdsc_program_temps_used(const pdsc_program *p);
unsigned pdsc_program_code_size(const pdsc_program *p);
unsigned pdsc_program_data_size(const pdsc_program *p);

const void *pdsc_program_code(const pdsc_program *p);
const void *pdsc_program_data(const pdsc_program *p);

/* Sometimes the control stream mandates a specific layout for the PDS program
 * usually requiring a linked program in a single buffer with a fixed code
 * offset after the data section. We return the mapped address of the data
 * section to allow the caller to patch the program after writing.
 */
void *pdsc_program_write_linked(const pdsc_program *p,
                                void *target,
                                unsigned code_offset);
unsigned pdsc_program_linked_size(const pdsc_program *p, unsigned code_offset);

void pdsc_serialize(struct blob *blob, const pdsc_program *p);
pdsc_program *pdsc_deserialize(void *mem_ctx, struct blob_reader *blob);

#endif /* PDSC_H */
