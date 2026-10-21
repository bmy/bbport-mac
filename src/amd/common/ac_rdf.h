/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * Writer for the RDF (Radeon Data File) format. Byte-compatible with AMD's amdrdf.
 *
 */
#ifndef AC_RDF_H
#define AC_RDF_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AC_RDF_IDENTIFIER_SIZE 16

/* Used only as input to the writer functions. Not part of RDF format. */
struct ac_rdf_chunk_info {
   char identifier[AC_RDF_IDENTIFIER_SIZE]; /* Chunk name (null-padded) */
   uint32_t version;                        /* Chunk version */
   const void *header;                      /* Optional header data, may be NULL */
   uint64_t header_size;                    /* Header size in bytes */
   const void *data;                        /* Chunk data; NULL when using begin/end */
   uint64_t data_size;                      /* Data size in bytes; 0 when using begin/end */
   bool use_compression;                    /* Enable compression */
};

struct ac_rdf_writer;

/* Writer can write output data to file or memory buffer. */
struct ac_rdf_writer *ac_rdf_writer_init_buffer(void);
struct ac_rdf_writer *ac_rdf_writer_init_file(const char *filename);
void ac_rdf_writer_destroy(struct ac_rdf_writer *rdf_writer);

/* One-shot: write a chunk whose data is the single contiguous buffer in chunk->data. */
bool ac_rdf_chunk_write(struct ac_rdf_writer *rdf_writer, const struct ac_rdf_chunk_info *chunk);

/* Streaming: for chunks whose data is produced in pieces, or whose size is unknown up front. */
bool ac_rdf_chunk_begin(struct ac_rdf_writer *rdf_writer, const struct ac_rdf_chunk_info *chunk);
bool ac_rdf_chunk_append(struct ac_rdf_writer *rdf_writer, const void *data, uint64_t data_size);
bool ac_rdf_chunk_end(struct ac_rdf_writer *rdf_writer);

/* Write the chunk index and back-patch the file header. Call once, after all chunks. */
bool ac_rdf_finalize(struct ac_rdf_writer *rdf_writer);

/* Buffer only: after finalize, returns complete data in RDF format (owned by the writer). */
void ac_rdf_get_buffer(struct ac_rdf_writer *rdf_writer, const void **data, uint64_t *size);

#ifdef __cplusplus
}
#endif
#endif /* AC_RDF_H */
