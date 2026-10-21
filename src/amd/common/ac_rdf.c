/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * Writer for the RDF (Radeon Data File) format.
 *
 * On-disk layout (byte-compatible with AMD's RDF specification):
 *
 *    [32-byte file header][chunk 0 header][chunk 0 data]...[chunk N][chunk index]
 *
 * Chunks are packed tightly, each has an optional header followed by its data.
 * The chunk index is a flat array of 64-byte entries. The file header's
 * index offset/size are back-patched to point at the index on finalize.
 *
 */
#include "ac_rdf.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/u_dynarray.h"

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

#define AC_RDF_VERSION 3

struct ac_rdf_file_header {
   char identifier[8];
   uint32_t version;
   uint32_t reserved;
   uint64_t index_offset;
   uint64_t index_size;
};
static_assert(sizeof(struct ac_rdf_file_header) == 32, "ac_rdf_file_header must be 32 bytes");

struct ac_rdf_index_entry {
   char chunk_identifier[AC_RDF_IDENTIFIER_SIZE];
   uint8_t is_zstd_compressed;
   uint8_t reserved[3];
   uint32_t version;
   uint64_t chunk_header_offset;
   uint64_t chunk_header_size;
   uint64_t chunk_data_offset;
   uint64_t chunk_data_size; /* Chunk data size as stored (compressed size when compressed). */
   uint64_t uncompressed_data_size;  /* Data size before compression; 0 when not compressed. */
};
static_assert(sizeof(struct ac_rdf_index_entry) == 64, "ac_rdf_index_entry must be 64 bytes");

struct ac_rdf_writer {
   /* Output to file or memory buffer. */
   FILE *file;            /* File output */
   uint8_t *buffer;       /* Buffer output; Grown on demand; NULL when using file. */
   size_t buffer_size;    /* Allocated buffer output size; 0 when using file. */
   uint64_t write_offset; /* Next byte position in the output, for either one. */

   struct util_dynarray chunk_index; /* One ac_rdf_index_entry per finished chunk. */

   /* State for the chunk currently being constructed between chunk_begin() and chunk_end(). */
   bool chunk_begun;                             /* True between chunk_begin() and chunk_end() */
   struct ac_rdf_index_entry current_chunk;      /* Currently constructed chunk */
   struct util_dynarray uncompressed_chunk_data; /* Gathered until chunk_end() compresses it */
};

/* Resize buffer so it can hold at least the requested number of bytes. */
static bool
rdf_resize_buffer(struct ac_rdf_writer *rdf_writer, uint64_t needed_size)
{
   bool ret = true;

   if (rdf_writer->buffer_size < needed_size) {
      /* Calculate new size */
      size_t new_size = MAX3(needed_size, rdf_writer->buffer_size * 2, 4096);

      /* Allocate new buffer */
      uint8_t *new_buffer = realloc(rdf_writer->buffer, new_size);
      if (new_buffer) {
         rdf_writer->buffer = new_buffer;
         rdf_writer->buffer_size = new_size;
      } else {
         ret = false;
      }
   }

   return ret;
}

/* Write data to file or buffer output at the current offset and advance the offset. */
static bool
rdf_write(struct ac_rdf_writer *rdf_writer, const void *data, uint64_t data_size)
{
   if (!data_size)
      return true;

   bool ret = false;

   if (rdf_writer->file) {
      ret = fwrite(data, 1, data_size, rdf_writer->file) == data_size;
   } else {
      ret = rdf_resize_buffer(rdf_writer, rdf_writer->write_offset + data_size);
      if (ret)
         memcpy(rdf_writer->buffer + rdf_writer->write_offset, data, data_size);
   }

   if (ret)
      rdf_writer->write_offset += data_size;

   return ret;
}

/* Patch 32-byte rdf file header at offset 0. Needed because data size is not known until all data
 * is available. */
static bool
rdf_patch_file_header(struct ac_rdf_writer *rdf_writer, const struct ac_rdf_file_header *header)
{
   bool ret = true;

   if (rdf_writer->file)
      ret = !fseek(rdf_writer->file, 0, SEEK_SET) &&
            fwrite(header, sizeof(*header), 1, rdf_writer->file) == 1;
   else
      memcpy(rdf_writer->buffer, header, sizeof(*header)); /* The first 32 bytes always exist */

   return ret;
}

static struct ac_rdf_writer *
rdf_writer_init(FILE *file)
{
   struct ac_rdf_writer *rdf_writer = calloc(1, sizeof(*rdf_writer));

   if (rdf_writer) {
      rdf_writer->file = file;

      util_dynarray_init(&rdf_writer->chunk_index, NULL);
      util_dynarray_init(&rdf_writer->uncompressed_chunk_data, NULL);

      /* Reserve the file header up front; it is back-patched once the index offset and size
       * are known in finalize(). The write offset lands at 32 afterwards. */
      struct ac_rdf_file_header file_header = {
         .identifier = "AMD_RDF ",
         .version = AC_RDF_VERSION,
      };

      if (!rdf_write(rdf_writer, &file_header, sizeof(file_header))) {
         ac_rdf_writer_destroy(rdf_writer);
         rdf_writer = NULL;
      }
   } else if (file) {
      fclose(file);
   }

   return rdf_writer;
}

struct ac_rdf_writer *
ac_rdf_writer_init_buffer(void)
{
   return rdf_writer_init(NULL);
}

struct ac_rdf_writer *
ac_rdf_writer_init_file(const char *filename)
{
   assert(filename);

   FILE *file = fopen(filename, "wb");
   return file ? rdf_writer_init(file) : NULL;
}

void
ac_rdf_writer_destroy(struct ac_rdf_writer *rdf_writer)
{
   assert(rdf_writer);

   util_dynarray_fini(&rdf_writer->chunk_index);
   util_dynarray_fini(&rdf_writer->uncompressed_chunk_data);

   if (rdf_writer->file)
      fclose(rdf_writer->file);
   free(rdf_writer->buffer);

   free(rdf_writer);
}

bool
ac_rdf_chunk_begin(struct ac_rdf_writer *rdf_writer, const struct ac_rdf_chunk_info *chunk)
{
   assert(rdf_writer);
   assert(!rdf_writer->chunk_begun);
   assert(chunk);

   uint64_t header_offset = rdf_writer->write_offset; /* Store header offset before write. */

   /* Optional header; may advance write_offset */
   bool ret = rdf_write(rdf_writer, chunk->header, chunk->header_size);

   if (ret) {
      rdf_writer->current_chunk = (struct ac_rdf_index_entry){
         .version = chunk->version ? chunk->version : 1,
         .chunk_header_offset = header_offset,
         .chunk_header_size = chunk->header_size,
         .chunk_data_offset = rdf_writer->write_offset,
#ifdef HAVE_ZSTD
         .is_zstd_compressed = chunk->use_compression ? 1 : 0,
#endif
      };
      memcpy(rdf_writer->current_chunk.chunk_identifier, chunk->identifier, AC_RDF_IDENTIFIER_SIZE);

      util_dynarray_clear(&rdf_writer->uncompressed_chunk_data);
      rdf_writer->chunk_begun = true;
   }

   return ret;
}

bool
ac_rdf_chunk_append(struct ac_rdf_writer *rdf_writer, const void *data, uint64_t data_size)
{
   assert(rdf_writer);
   assert(rdf_writer->chunk_begun);

   if (!data_size)
      return true;

   bool ret = false;

   if (rdf_writer->current_chunk.is_zstd_compressed) {
      /* Compression needs the whole chunk at once, so gather it until chunk_end().
       * util_dynarray indexes with unsigned, capping the uncompressed data at ~4 GB. */
      if (data_size <= UINT_MAX) {
         void *dest = util_dynarray_grow_bytes(&rdf_writer->uncompressed_chunk_data, data_size, 1);
         if (dest) {
            memcpy(dest, data, data_size);
            ret = true;
         }
      }
   } else {
      /* If not compressed, write immediately. */
      ret = rdf_write(rdf_writer, data, data_size);
   }

   return ret;
}

bool
ac_rdf_chunk_end(struct ac_rdf_writer *rdf_writer)
{
   assert(rdf_writer);
   assert(rdf_writer->chunk_begun);

   bool ret = false;

   if (rdf_writer->current_chunk.is_zstd_compressed) {
#ifdef HAVE_ZSTD
      /* 1. Get max possible buffer size for compressed data. */
      size_t compressed_buffer_size = ZSTD_compressBound(rdf_writer->uncompressed_chunk_data.size);
      uint8_t *compressed_buffer = malloc(compressed_buffer_size);

      if (compressed_buffer) {
         /* 2. Compress data (ZSTD_CLEVEL_DEFAULT matches amdrdf). */
         size_t compressed_data_size = ZSTD_compress(compressed_buffer, compressed_buffer_size,
                                                     rdf_writer->uncompressed_chunk_data.data,
                                                     rdf_writer->uncompressed_chunk_data.size,
                                                     ZSTD_CLEVEL_DEFAULT);

         /* 3. Write compressed data to the output. */
         if (!ZSTD_isError(compressed_data_size) &&
             rdf_write(rdf_writer, compressed_buffer, compressed_data_size)) {
            rdf_writer->current_chunk.chunk_data_size = compressed_data_size;
            rdf_writer->current_chunk.uncompressed_data_size = rdf_writer->uncompressed_chunk_data.size;
            ret = true;
         }

         free(compressed_buffer);
      }
#endif
   } else {
      /* Uncompressed case */
      rdf_writer->current_chunk.chunk_data_size = rdf_writer->write_offset -
                                                  rdf_writer->current_chunk.chunk_data_offset;
      assert(rdf_writer->current_chunk.uncompressed_data_size == 0); /* Should be 0 for uncompressed */
      ret = true;
   }

   if (ret) {
      /* Add chunk to the chunk index */
      util_dynarray_append_typed(&rdf_writer->chunk_index, struct ac_rdf_index_entry,
                                 rdf_writer->current_chunk);
      rdf_writer->chunk_begun = false;
   }

   return ret;
}

bool
ac_rdf_chunk_write(struct ac_rdf_writer *rdf_writer, const struct ac_rdf_chunk_info *chunk)
{
   return ac_rdf_chunk_begin(rdf_writer, chunk) &&
          ac_rdf_chunk_append(rdf_writer, chunk->data, chunk->data_size) &&
          ac_rdf_chunk_end(rdf_writer);
}

bool
ac_rdf_finalize(struct ac_rdf_writer *rdf_writer)
{
   assert(rdf_writer);
   assert(!rdf_writer->chunk_begun);

   /* Append the chunk index, then back-patch the header to point at it. */
   uint64_t index_offset = rdf_writer->write_offset;
   uint64_t entry_count = util_dynarray_num_elements(&rdf_writer->chunk_index,
                                                     struct ac_rdf_index_entry);
   uint64_t index_size = entry_count * sizeof(struct ac_rdf_index_entry);

   if (!rdf_write(rdf_writer, util_dynarray_begin(&rdf_writer->chunk_index), index_size))
      return false;

   struct ac_rdf_file_header file_header = {
      .identifier = "AMD_RDF ",
      .version = AC_RDF_VERSION,
      .index_offset = index_offset,
      .index_size = index_size,
   };

   return rdf_patch_file_header(rdf_writer, &file_header);
}

void
ac_rdf_get_buffer(struct ac_rdf_writer *rdf_writer, const void **data, uint64_t *size)
{
   assert(rdf_writer);
   assert(!rdf_writer->file);
   assert(data);
   assert(size);

   /* Write offset is exactly the current buffer length. */
   *data = rdf_writer->buffer;
   *size = rdf_writer->write_offset;
}
