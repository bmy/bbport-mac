/*
 * Copyright © 2019 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_CS_H
#define TU_CS_H

#include "tu_common.h"

#include "freedreno_pm4.h"

#include "tu_knl.h"

/* For breadcrumbs we may open a network socket based on the envvar,
 * it's not something that should be enabled by default.
 */
#define TU_BREADCRUMBS_ENABLED 0

enum tu_cs_mode
{

   /*
    * A command stream in TU_CS_MODE_GROW mode grows automatically whenever it
    * is full.  tu_cs_begin must be called before command packet emission and
    * tu_cs_end must be called after.
    *
    * This mode may create multiple entries internally.  The entries must be
    * submitted together.
    */
   TU_CS_MODE_GROW,

   /*
    * A command stream in TU_CS_MODE_EXTERNAL mode wraps an external,
    * fixed-size buffer.  tu_cs_begin and tu_cs_end are optional and have no
    * effect on it.
    *
    * This mode does not create any entry or any BO.
    */
   TU_CS_MODE_EXTERNAL,

   /*
    * A command stream in TU_CS_MODE_SUB_STREAM mode does not support direct
    * command packet emission.  tu_cs_begin_sub_stream must be called to get a
    * sub-stream to emit comamnd packets to.  When done with the sub-stream,
    * tu_cs_end_sub_stream must be called.
    *
    * This mode does not create any entry internally.
    */
   TU_CS_MODE_SUB_STREAM,
};

struct tu_cs_entry
{
   /* No ownership */
   const struct tu_bo *bo;

   uint32_t size;
   uint32_t offset;
};

struct tu_cs_memory {
   uint32_t *map;
   uint64_t iova;
   bool writeable;
};

struct tu_draw_state {
   uint64_t iova;
   uint16_t size;
   bool writeable;
};

struct tu_bo_array {
   struct tu_bo **bos;
   uint32_t bo_count;
   uint32_t bo_capacity;
   uint32_t *start;
};

#define TU_COND_EXEC_STACK_SIZE 4

struct tu_pkt;
struct tu_crb;

/* Temporary struct for tracking a register state to be written, used by
 * a6xx-pack.h and tu_cs_emit_regs()
 */
struct tu_reg_value {
   uint32_t reg;
   uint64_t value;
   bool is_address;
};

#define fd_reg_pair tu_reg_value

struct tu_scratch_slot {
   unsigned slot;
};

/** Boxed type for GPU virtual addresses, to disambiguate vs. uint32_t immed. */
struct tu_gpuva {
   uint64_t gpuva;
   tu_gpuva(uint64_t arg) : gpuva(arg) {}
};

struct tu_cs
{
   uint32_t *start;
   uint32_t *cur;
   uint32_t *reserved_end;
   uint32_t *end;
   const char *name;

   struct tu_device *device;
   enum tu_cs_mode mode;
   bool writeable;
   uint32_t next_bo_size;
   VkResult status;

   struct tu_cs_entry *entries;
   uint32_t entry_count;
   uint32_t entry_capacity;

   struct tu_bo_array read_only, read_write;

   /* Optional BO that this CS is sub-allocated from for TU_CS_MODE_SUB_STREAM */
   struct tu_bo *refcount_bo;

   /* iova that this CS starts with in TU_CS_MODE_EXTERNAL */
   uint64_t external_iova;

   /* state for cond_exec_start/cond_exec_end */
   uint32_t cond_stack_depth;
   uint32_t cond_flags[TU_COND_EXEC_STACK_SIZE];
   uint32_t *cond_dwords[TU_COND_EXEC_STACK_SIZE];

   uint32_t breadcrumb_emit_after;

   tu_pkt *pkt;

   /*
    * Helpers for various pm4 pkt building:
    */

   tu_crb crb(uint32_t nregs);

   void scratch_to_reg(struct fd_reg_pair reg, struct tu_scratch_slot scratch, unsigned cnt);
   void reg_to_scratch(struct tu_scratch_slot scratch, struct fd_reg_pair reg, unsigned cnt);
   void scratch_write(struct tu_scratch_slot scratch, uint32_t *val, unsigned cnt);
   void scratch_write(struct tu_scratch_slot scratch, uint32_t val) {
      scratch_write(scratch, &val, 1);
   }

   /**
    * A flexible src/dst type for pm4 pkt helpers, which can be register/scratch/immediate
    * (the latter for src args only).  The actual types supported depend on the pm4 pkt.  If
    * there is only one possible type, use the underlying type directly (ie fd_reg_pair, etc)
    */
   struct tu_pm4_arg {
      enum {
         TU_PM4_IMMED,
         TU_PM4_GPUVA,
         TU_PM4_REG,
         TU_PM4_SCRATCH,
      } type;

      union {
         uint32_t imm;
         struct tu_gpuva gpuva;
         struct fd_reg_pair reg;
         struct tu_scratch_slot scratch;
      };

      tu_pm4_arg(uint32_t arg) : type(TU_PM4_IMMED), imm(arg) {}
      tu_pm4_arg(struct tu_gpuva arg) : type(TU_PM4_GPUVA), gpuva(arg) {}
      tu_pm4_arg(struct fd_reg_pair arg) : type(TU_PM4_REG), reg(arg) {}
      tu_pm4_arg(struct tu_scratch_slot arg) : type(TU_PM4_SCRATCH), scratch(arg) {}

      operator uint32_t() const {
         if (type == TU_PM4_IMMED)
            return imm;
         else if (type == TU_PM4_REG)
            return reg.reg;
         else if (type == TU_PM4_SCRATCH)
            return scratch.slot;
         else
            UNREACHABLE("bad type");
      }
   };

   struct tu_rmw_args {
      bool skip_wfm;
      uint32_t rotate;
      bool src1_add;
      struct tu_pm4_arg src0, src1;
   };

   void rmw(struct tu_pm4_arg dst, struct tu_rmw_args args);

   struct tu_reg_to_mem_args {
      unsigned cnt;     /* value of 0 treated as 1 by sqe */
      bool is_64b;
      bool accumulate;
   };

   void reg_to_mem(uint64_t va, struct fd_reg_pair reg, struct tu_reg_to_mem_args args = {});

   /* For reg64: */
   void reg_to_mem(uint64_t va, struct fd_reg_pair reg_lo, struct fd_reg_pair reg_hi) {
      reg_to_mem(va, reg_lo, { .cnt = 2, .is_64b = true });
   }

   struct tu_cond_write_args {
      enum cp_cond_function function;
      bool signed_compare;
      uint32_t ref;
      uint32_t mask = ~0;
      uint32_t write_data;
   };

   void cond_write(struct tu_pm4_arg dst, struct tu_pm4_arg src, struct tu_cond_write_args args);

   struct tu_mem_to_reg_args {
      bool one_reg_wr;
      unsigned cnt;     /* value of 0 treated as 1 by sqe */
      bool shift_by_2;
      bool wait_cache_flush;
   };

   void mem_to_reg(struct fd_reg_pair reg, uint64_t va, struct tu_mem_to_reg_args args = {});
};

void
tu_breadcrumbs_init(struct tu_device *device);

void
tu_breadcrumbs_finish(struct tu_device *device);

void
tu_cs_init(struct tu_cs *cs,
           struct tu_device *device,
           enum tu_cs_mode mode,
           uint32_t initial_size, const char *name);

void
tu_cs_init_external(struct tu_cs *cs, struct tu_device *device,
                    uint32_t *start, uint32_t *end, uint64_t iova,
                    bool writeable);

void
tu_cs_init_suballoc(struct tu_cs *cs, struct tu_device *device,
                    struct tu_suballoc_bo *bo);

void
tu_cs_finish(struct tu_cs *cs);

void
tu_cs_begin(struct tu_cs *cs);

void
tu_cs_end(struct tu_cs *cs);

void
tu_cs_set_writeable(struct tu_cs *cs, bool writeable);

VkResult
tu_cs_begin_sub_stream_aligned(struct tu_cs *cs, uint32_t count,
                               uint32_t size, struct tu_cs *sub_cs);

static inline VkResult
tu_cs_begin_sub_stream(struct tu_cs *cs, uint32_t size, struct tu_cs *sub_cs)
{
   return tu_cs_begin_sub_stream_aligned(cs, size, 1, sub_cs);
}


VkResult
tu_cs_alloc(struct tu_cs *cs,
            uint32_t count,
            uint32_t size,
            struct tu_cs_memory *memory);

struct tu_cs_entry
tu_cs_end_sub_stream(struct tu_cs *cs, struct tu_cs *sub_cs);

static inline struct tu_draw_state
tu_cs_end_draw_state(struct tu_cs *cs, struct tu_cs *sub_cs)
{
   struct tu_cs_entry entry = tu_cs_end_sub_stream(cs, sub_cs);
   if (!entry.bo)
      return (struct tu_draw_state) {};
   return (struct tu_draw_state) {
      .iova = entry.bo->iova + entry.offset,
      .size = entry.size / sizeof(uint32_t),
      .writeable = sub_cs->writeable,
   };
}

VkResult
tu_cs_reserve_space(struct tu_cs *cs, uint32_t reserved_size);

/* Write sink for OOM state: tu_cs_emit writes here harmlessly instead of
 * needing a per-dword status branch that would prevent vectorization.
 * Sized to the PM4 maximum (14-bit count field + header = 16384 dwords) so
 * no single tu_cs_reserve call can overflow it.
 */
#define TU_CS_FAIL_SINK_SIZE 16384
extern uint32_t tu_cs_fail_sink[TU_CS_FAIL_SINK_SIZE];

static inline void
tu_cs_fail(struct tu_cs *cs, VkResult result)
{
   cs->status = result;
   cs->cur = tu_cs_fail_sink;
   cs->reserved_end = tu_cs_fail_sink + ARRAY_SIZE(tu_cs_fail_sink);
}

uint64_t
tu_cs_get_cur_iova(const struct tu_cs *cs);

static inline struct tu_draw_state
tu_cs_draw_state(struct tu_cs *sub_cs, struct tu_cs *cs, uint32_t size)
{
   struct tu_cs_memory memory;

   VkResult result = tu_cs_alloc(sub_cs, size, 1, &memory);
   if (result != VK_SUCCESS) {
      /* Freshly declared, nothing owned yet so safe to zero. */
      memset(cs, 0, sizeof(*cs));
      tu_cs_fail(cs, result);
      return (struct tu_draw_state) {};
   }
   tu_cs_init_external(cs, sub_cs->device, memory.map, memory.map + size,
                       memory.iova, memory.writeable);
   tu_cs_begin(cs);
   tu_cs_reserve_space(cs, size);

   return (struct tu_draw_state) {
      .iova = memory.iova,
      .size = size,
      .writeable = sub_cs->writeable,
   };
}

void
tu_cs_reset(struct tu_cs *cs);

VkResult
tu_cs_add_entries(struct tu_cs *cs, struct tu_cs *target);

/**
 * Get the size of the command packets emitted since the last call to
 * tu_cs_add_entry.
 */
static inline uint32_t
tu_cs_get_size(const struct tu_cs *cs)
{
   return cs->cur - cs->start;
}

/**
 * Return true if there is no command packet emitted since the last call to
 * tu_cs_add_entry.
 */
static inline uint32_t
tu_cs_is_empty(const struct tu_cs *cs)
{
   return tu_cs_get_size(cs) == 0;
}

/**
 * Discard all entries.  This allows \a cs to be reused while keeping the
 * existing BOs and command packets intact.
 */
static inline void
tu_cs_discard_entries(struct tu_cs *cs)
{
   assert(cs->mode == TU_CS_MODE_GROW);
   cs->entry_count = 0;
}

/**
 * Get the size needed for tu_cs_emit_call.
 */
static inline uint32_t
tu_cs_get_call_size(const struct tu_cs *cs)
{
   assert(cs->mode == TU_CS_MODE_GROW);
   /* each CP_INDIRECT_BUFFER needs 4 dwords */
   return cs->entry_count * 4;
}

/**
 * Assert that we did not exceed the reserved space.
 */
static inline void
tu_cs_sanity_check(const struct tu_cs *cs)
{
   assert(cs->start <= cs->cur);
   assert(cs->cur <= cs->reserved_end);
   assert(cs->reserved_end <= cs->end);
}

void
tu_cs_emit_sync_breadcrumb(struct tu_cs *cs, uint8_t opcode, uint16_t cnt);

/**
 * Emit a uint32_t value into a command stream, without boundary checking.
 */
static inline void
tu_cs_emit(struct tu_cs *cs, uint32_t value)
{
   assert(cs->cur < cs->reserved_end);
   *cs->cur = value;
   ++cs->cur;

#if TU_BREADCRUMBS_ENABLED
   cs->breadcrumb_emit_after--;
   if (cs->breadcrumb_emit_after == 0)
      tu_cs_emit_sync_breadcrumb(cs, -1, 0);
#endif
}

/**
 * Emit an array of uint32_t into a command stream, without boundary checking.
 */
static inline void
tu_cs_emit_array(struct tu_cs *cs, const uint32_t *values, uint32_t length)
{
   assert(cs->cur + length <= cs->reserved_end);
   memcpy(cs->cur, values, sizeof(uint32_t) * length);
   cs->cur += length;
}

/**
 * Get the size of the remaining space in the current BO.
 */
static inline uint32_t
tu_cs_get_space(const struct tu_cs *cs)
{
   return cs->end - cs->cur;
}

static inline void
tu_cs_reserve(struct tu_cs *cs, uint32_t reserved_size)
{
   if (unlikely(cs->status != VK_SUCCESS)) {
      tu_cs_fail(cs, cs->status);
      return;
   }

   if (cs->mode != TU_CS_MODE_GROW) {
      assert(tu_cs_get_space(cs) >= reserved_size);
      assert(cs->reserved_end == cs->end);
      return;
   }

   if (tu_cs_get_space(cs) >= reserved_size &&
       cs->entry_count < cs->entry_capacity) {
      cs->reserved_end = cs->cur + reserved_size;
      return;
   }

   tu_cs_reserve_space(cs, reserved_size);
}

static inline VkResult
tu_cs_get_status(const struct tu_cs *cs)
{
   return cs->status;
}

/**
 * Emit a type-4 command packet header into a command stream.
 */
static inline void
tu_cs_emit_pkt4(struct tu_cs *cs, uint16_t regindx, uint16_t cnt)
{
   assert(!cs->pkt); /* No mixing with the packet builders.*/
   tu_cs_reserve(cs, cnt + 1);
   tu_cs_emit(cs, pm4_pkt4_hdr(regindx, cnt));
}

/**
 * Emit a type-7 command packet header into a command stream.
 */
static inline void
tu_cs_emit_pkt7(struct tu_cs *cs, uint8_t opcode, uint16_t cnt)
{
   assert(!cs->pkt); /* No mixing with the packet builders.*/
#if TU_BREADCRUMBS_ENABLED
   tu_cs_emit_sync_breadcrumb(cs, opcode, cnt + 1);
#endif

   tu_cs_reserve(cs, cnt + 1);
   tu_cs_emit(cs, pm4_pkt7_hdr(opcode, cnt));
}

static inline void
tu_cs_emit_wfi(struct tu_cs *cs)
{
   tu_cs_emit_pkt7(cs, CP_WAIT_FOR_IDLE, 0);
}

static inline void
tu_cs_emit_qw(struct tu_cs *cs, uint64_t value)
{
   tu_cs_emit(cs, (uint32_t) value);
   tu_cs_emit(cs, (uint32_t) (value >> 32));
}

static inline void
tu_cs_emit_write_reg(struct tu_cs *cs, uint16_t reg, uint32_t value)
{
   tu_cs_emit_pkt4(cs, reg, 1);
   tu_cs_emit(cs, value);
}

/**
 * Emit a CP_INDIRECT_BUFFER command packet.
 */
static inline void
tu_cs_emit_ib(struct tu_cs *cs, const struct tu_cs_entry *entry)
{
   assert(entry->bo);
   assert(entry->size && entry->offset + entry->size <= entry->bo->size);
   assert(entry->size % sizeof(uint32_t) == 0);
   assert(entry->offset % sizeof(uint32_t) == 0);

   tu_cs_emit_pkt7(cs, CP_INDIRECT_BUFFER, 3);
   tu_cs_emit_qw(cs, entry->bo->iova + entry->offset);
   tu_cs_emit(cs, entry->size / sizeof(uint32_t));
}

/* for compute which isn't using SET_DRAW_STATE */
static inline void
tu_cs_emit_state_ib(struct tu_cs *cs, struct tu_draw_state state)
{
   if (state.size) {
      tu_cs_emit_pkt7(cs, CP_INDIRECT_BUFFER, 3);
      tu_cs_emit_qw(cs, state.iova);
      tu_cs_emit(cs, state.size);
   }
}

/**
 * Emit a CP_INDIRECT_BUFFER command packet for each entry in the target
 * command stream.
 */
static inline void
tu_cs_emit_call(struct tu_cs *cs, const struct tu_cs *target)
{
   assert(target->mode == TU_CS_MODE_GROW);
   for (uint32_t i = 0; i < target->entry_count; i++)
      tu_cs_emit_ib(cs, target->entries + i);
}

/**
 * Emit a CP_NOP with a string tail into the command stream.
 */
void
tu_cs_emit_debug_string(struct tu_cs *cs, const char *string, int len);

void
tu_cs_emit_debug_magic_strv(struct tu_cs *cs,
                            uint32_t magic,
                            const char *fmt,
                            va_list args);

__attribute__((format(printf, 2, 3))) void
tu_cs_emit_debug_msg(struct tu_cs *cs, const char *fmt, ...);

/**
 * Emit a single message into the CS that denote the calling function and any
 * optional printf-style parameters when utrace markers are enabled.
 */
#define TU_CS_DEBUG_MSG(CS, FORMAT_STRING, ...)                              \
   do {                                                                      \
      if (unlikely(u_trace_markers_enabled(&(CS)->device->trace_context)))   \
         tu_cs_emit_debug_msg(CS, "%s(" FORMAT_STRING ")", __func__,         \
                              ## __VA_ARGS__);                               \
   } while (0)

typedef struct tu_cs *tu_debug_scope;

__attribute__((format(printf, 3, 4))) void
tu_cs_trace_start(struct u_trace_context *utctx,
                  void *cs,
                  const char *fmt,
                  ...);

__attribute__((format(printf, 3, 4))) void
tu_cs_trace_end(struct u_trace_context *utctx, void *cs, const char *fmt, ...);

__attribute__((format(printf, 3, 4))) void
tu_cs_trace_singular(struct u_trace_context *utctx, void *cs, const char *fmt, ...);

struct tu_cs_patchable_state {
   uint32_t *nop_header;
   uint32_t dwords;
   uint32_t enable_patch;
   uint32_t disable_patch;
   uint32_t *_end;
};

/* Create a region that can be enabled/disabled by updating CP_NOP header. */
static struct tu_cs_patchable_state
tu_cs_patchable_start(struct tu_cs *cs, uint32_t reserved_size)
{
   tu_cs_reserve(cs, reserved_size);
   uint32_t *nop_header = cs->cur;
   tu_cs_emit_pkt7(cs, CP_NOP, 0);
   return (struct tu_cs_patchable_state) {
      .nop_header = nop_header,
      .dwords = 0,
      .enable_patch = pm4_pkt7_hdr(CP_NOP, 0),
      .disable_patch = 0,
      ._end = cs->end,
   };
}

static void
tu_cs_patchable_end(struct tu_cs *cs,
                    bool enabled_by_default,
                    struct tu_cs_patchable_state *state)
{
   assert(cs->end == state->_end);
   state->dwords = cs->cur - state->nop_header - 1;
   state->disable_patch = pm4_pkt7_hdr(CP_NOP, state->dwords);
   if (!enabled_by_default)
      *state->nop_header = state->disable_patch;
}

/* Helpers for bracketing a large sequence of commands of unknown size inside
 * a CP_COND_REG_EXEC packet.
 */
static inline void
tu_cond_exec_start(struct tu_cs *cs, uint32_t cond_flags)
{
   assert(cs->cond_stack_depth < TU_COND_EXEC_STACK_SIZE);

   ASSERTED enum compare_mode mode =
      (enum compare_mode) pkt_field_get(CP_COND_REG_EXEC_0_MODE, cond_flags);
   assert(mode == PRED_TEST || mode == RENDER_MODE || mode == THREAD_MODE);

   tu_cs_emit_pkt7(cs, CP_COND_REG_EXEC, 2);
   tu_cs_emit(cs, cond_flags);

   cs->cond_flags[cs->cond_stack_depth] = cond_flags;
   cs->cond_dwords[cs->cond_stack_depth] = cs->cur;

   /* Emit dummy DWORD field here */
   tu_cs_emit(cs, RENDER_MODE_CP_COND_REG_EXEC_1_DWORDS(0));

   cs->cond_stack_depth++;
}
#define CP_COND_EXEC_0_RENDER_MODE_GMEM \
   (CP_COND_REG_EXEC_0_MODE(RENDER_MODE) | CP_COND_REG_EXEC_0_GMEM)
#define CP_COND_EXEC_0_RENDER_MODE_SYSMEM \
   (CP_COND_REG_EXEC_0_MODE(RENDER_MODE) | CP_COND_REG_EXEC_0_SYSMEM)

static inline void
tu_cond_exec_end(struct tu_cs *cs)
{
   assert(cs->cond_stack_depth > 0);
   cs->cond_stack_depth--;

   cs->cond_flags[cs->cond_stack_depth] = 0;
   /* Subtract one here to account for the DWORD field itself. */
   uint32_t cond_len = cs->cur - cs->cond_dwords[cs->cond_stack_depth] - 1;
   if (cond_len) {
      *cs->cond_dwords[cs->cond_stack_depth] = cond_len;
   } else {
      /* rewind the CS to drop the empty cond reg packet. */
      cs->cur = cs->cur - 3;
   }
}

static inline void
tu7_thread_control(struct tu_cs *cs, enum cp_thread thread)
{
   tu_cs_emit_pkt7(cs, CP_THREAD_CONTROL, 1);
   tu_cs_emit(cs, CP_THREAD_CONTROL_0_THREAD(thread));
}

static inline void
tu7_set_pred_mask(struct tu_cs *cs, uint32_t mask, uint32_t val)
{
   tu_cs_emit_pkt7(cs, CP_REG_TEST, 3);
   tu_cs_emit(cs, A6XX_CP_REG_TEST_0_PRED_UPDATE |
                  A6XX_CP_REG_TEST_0_SKIP_WAIT_FOR_ME);
   tu_cs_emit(cs, mask);
   tu_cs_emit(cs, val);
}

static inline void
tu7_set_pred_bit(struct tu_cs *cs, enum tu_predicate_bit bit, bool val)
{
   tu7_set_pred_mask(cs, 1u << bit, val ? (1u << bit) : 0);
}

static inline void
tu7_write_onchip_timestamp(struct tu_cs *cs, enum tu_onchip_addr onchip_addr)
{
   tu_cs_emit_pkt7(cs, CP_EVENT_WRITE7, 2);
   tu_cs_emit(cs, CP_EVENT_WRITE7_0_WRITE_DST(EV_DST_ONCHIP) |
                  CP_EVENT_WRITE7_0_WRITE_SRC(EV_WRITE_TIMESTAMP_SUM) |
                  CP_EVENT_WRITE7_0_EVENT(DUMMY_EVENT) |
                  CP_EVENT_WRITE7_0_WRITE_ENABLED);
   tu_cs_emit(cs, onchip_addr);
}

static inline void
tu7_wait_onchip_timestamp(struct tu_cs *cs, enum tu_onchip_addr onchip_addr)
{
   tu_cs_emit_pkt7(cs, CP_WAIT_TIMESTAMP, 3);
   tu_cs_emit(cs, CP_WAIT_TIMESTAMP_0_WAIT_DST(TS_WAIT_ONCHIP) |
                  CP_WAIT_TIMESTAMP_0_WAIT_VALUE_SRC(TS_WAIT_GE_TIMESTAMP_SUM));
   tu_cs_emit_qw(cs, onchip_addr);
}

static inline void
tu7_wait_onchip_val(struct tu_cs *cs, enum tu_onchip_addr onchip_addr,
                    uint32_t val)
{
   tu_cs_emit_pkt7(cs, CP_WAIT_REG_MEM, 6);
   tu_cs_emit(cs, CP_WAIT_REG_MEM_0_FUNCTION(WRITE_EQ) |
                  CP_WAIT_REG_MEM_0_POLL(POLL_ON_CHIP));
   tu_cs_emit_qw(cs, onchip_addr);
   tu_cs_emit(cs, CP_WAIT_REG_MEM_3_REF(val));
   tu_cs_emit(cs, CP_WAIT_REG_MEM_4_MASK(~0));
   tu_cs_emit(cs, CP_WAIT_REG_MEM_5_DELAY_LOOP_CYCLES(0));
}

uint64_t
tu_cs_emit_data_nop(struct tu_cs *cs,
                    const uint32_t *data,
                    uint32_t size,
                    uint32_t align);

#include "a6xx-pack.xml.h"
#include "adreno-pm4-pack.xml.h"

#define __assert_eq(a, b)                                               \
   do {                                                                 \
      if ((a) != (b)) {                                                 \
         fprintf(stderr, "assert failed: " #a " (0x%x) != " #b " (0x%x)\n", a, b); \
         assert((a) == (b));                                            \
      }                                                                 \
   } while (0)

#define __ONE_REG(i, regs)                                      \
   do {                                                         \
      if (i < ARRAY_SIZE(regs) && regs[i].reg > 0) {            \
         __assert_eq(regs[0].reg + i, regs[i].reg);             \
         *p++ = regs[i].value;                                  \
         if (regs[i].is_address)                                \
            *p++ = regs[i].value >> 32;                         \
      }                                                         \
   } while (0)

/* Emits a sequence of register writes in order using a pkt4.  This will check
 * (at runtime on a !NDEBUG build) that the registers were actually set up in
 * order in the code.
 *
 * Note that references to buffers aren't automatically added to the CS,
 * unlike in freedreno.  We are clever in various places to avoid duplicating
 * the reference add work.
 *
 * Also, 64-bit address registers don't have a way (currently) to set a 64-bit
 * address without having a reference to a BO, since the .dword field in the
 * register's struct is only 32-bit wide.  We should fix this in the pack
 * codegen later.
 */
#define tu_cs_emit_regs(cs, ...) do {                   \
   const struct fd_reg_pair regs[] = { __VA_ARGS__ };   \
   unsigned count = ARRAY_SIZE(regs);                   \
                                                        \
   STATIC_ASSERT(ARRAY_SIZE(regs) > 0);                 \
   STATIC_ASSERT(ARRAY_SIZE(regs) <= 16);               \
                                                        \
   tu_cs_emit_pkt4((cs), regs[0].reg, count);             \
   uint32_t *p = (cs)->cur;                               \
   __ONE_REG( 0, regs);                                 \
   __ONE_REG( 1, regs);                                 \
   __ONE_REG( 2, regs);                                 \
   __ONE_REG( 3, regs);                                 \
   __ONE_REG( 4, regs);                                 \
   __ONE_REG( 5, regs);                                 \
   __ONE_REG( 6, regs);                                 \
   __ONE_REG( 7, regs);                                 \
   __ONE_REG( 8, regs);                                 \
   __ONE_REG( 9, regs);                                 \
   __ONE_REG(10, regs);                                 \
   __ONE_REG(11, regs);                                 \
   __ONE_REG(12, regs);                                 \
   __ONE_REG(13, regs);                                 \
   __ONE_REG(14, regs);                                 \
   __ONE_REG(15, regs);                                 \
   (cs)->cur = p;                                         \
   } while (0)

/**
 * Helper base class for any pkt building.
 */
struct tu_pkt {
 protected:
   tu_pkt(tu_cs *cs, enum adreno_pm4_type3_packets pkt, unsigned ndwords)
       : cs(cs), pkt(pkt)
   {
      tu_cs_reserve(cs, ndwords + 1);
      start = cs->cur;
      cs->cur++; /* Leave a slot for the packet header when we're done. */
      assert(!cs->pkt);
#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wdangling-pointer="
#endif
      /* Safe because destructor calls flush() which clears the potentially
       * dangling pointer:
       */
      cs->pkt = this;
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif
   }

   ~tu_pkt()
   {
      flush();
   }

   void append(uint32_t value)
   {
      tu_cs_emit(cs, value);
   }

   void append(const uint32_t *dwords, uint32_t sizedwords)
   {
      tu_cs_emit_array(cs, dwords, sizedwords);
   }

 public:
   void flush()
   {
      bool skip_empty = (pkt == CP_CONTEXT_REG_BUNCH);

      if (!cs) {
         /* Skip double flushing, such as an early flush and then going out of
          * scope later.
          */
         return;
      }

      assert(start);
      assert(cs->pkt == this);

      unsigned cnt = cs->cur - start - 1;

      if (skip_empty && !cnt) {
         /* rewind the start pointer to the start of the packet. */
         cs->cur = start;
      } else {
         *start = pm4_pkt7_hdr(pkt, cnt);
      }
      cs->pkt = NULL;
      cs = NULL;
   }

 private:
   tu_cs *cs;
   enum adreno_pm4_type3_packets pkt;
   uint32_t *start;
};

/**
 * A builder for CP_CONTEXT_REG_BUNCH.  This packet can write an arbitrary
 * sequence of registers (payload consists of pairs of offset,value).  It should
 * be as fast as a pkt4 packet writing a consecutive sequence of registers,
 * without the constraint of the registers being sequential, making it easier to
 * use when cmdstream emit involves if/else/loops.  And should be less brittle
 * if registers shift around between generations.  This builder intentionally
 * encourages use of tu_reg_value.
 *
 * For TU_CS_MODE_GROW command streams (cmd->cs, cmd->draw_cs), you have to pass
 * the maximum number of regs you might emit in the CONTEXT_REG_BUNCH.  For
 * external/sub-stream CSes, you don't need to bother, and overflow of the
 * preallocated space will be caught by tu_cs_emit().
 */
struct tu_crb : public tu_pkt {
 public:
   /* Constructor to use with tu_cs: */
   tu_crb(tu_cs *cs, unsigned nregs)
       : tu_pkt(cs, CP_CONTEXT_REG_BUNCH, nregs * 2)
   {
      assert(nregs != 0 || cs->mode != TU_CS_MODE_GROW);
   }

   tu_crb(tu_cs *cs): tu_pkt(cs, CP_CONTEXT_REG_BUNCH, 0)
   {
      assert(cs->mode != TU_CS_MODE_GROW);
   }

   /* Append a <reg32> to CRB: */
   tu_crb &add(struct tu_reg_value reg)
   {
      append(reg.reg);
      append(reg.value);
      return *this;
   }

   /* Append a <reg64> to CRB: */
   tu_crb &add(struct tu_reg_value reg_lo, struct tu_reg_value reg_hi)
   {
      __assert_eq(reg_hi.reg, 0);
      uint64_t val = reg_lo.value;
      append(reg_lo.reg);
      append(val);
      append(reg_lo.reg + 1);
      append(val >> 32);
      return *this;
   }

   /* for with_crb() */
   bool first = true;

private:
   /* Disallow copy constructor to prevent mistakes with using tu_crb instead
    * of tu_crb& as function param:
    */
   tu_crb(const tu_crb &);
};


#define with_crb(...) \
   for (tu_crb crb(__VA_ARGS__); crb.first; crb.first = false)


/**
 * A builder for an arbitrary PKT7 (for CRB, use fd_crb instead)
 */
class tu_pkt7 final : public tu_pkt {
public:
   tu_pkt7(tu_cs *cs, enum adreno_pm4_type3_packets pkt, unsigned ndwords) :
      tu_pkt(cs, pkt, ndwords)
   {
   }

   /* Allow appending a "naked" dwords: */
   tu_pkt7& add(uint32_t val) {
      append(val);
      off_++;
      return *this;
   }

   tu_pkt7& add(struct fd_reg_pair reg) {
      __assert_eq(off_, reg.reg);
      off_ = reg.reg + 1;
      append(reg.value);
      return *this;
   }

   tu_pkt7& add(struct fd_reg_pair reg_lo, struct fd_reg_pair reg_hi) {
      __assert_eq(reg_hi.reg, 0);
      __assert_eq(off_, reg_lo.reg);
      off_ = reg_lo.reg + 2;
      uint64_t val = reg_lo.value;
      append(val);
      append(val >> 32);
      return *this;
   }

private:
   /* Disallow copy constructor to prevent mistakes with using fd_pkt7 instead
    * of fd_pkt7& as function param:
    */
   tu_pkt7(const tu_pkt7 &);

   /* for debugging: */
   unsigned off_ = 0;
};

inline void
tu_cs::rmw(struct tu_pm4_arg dst, struct tu_rmw_args args)
{
   assert((dst.type == tu_pm4_arg::TU_PM4_REG) || (dst.type == tu_pm4_arg::TU_PM4_SCRATCH));
   assert((args.src0.type == tu_pm4_arg::TU_PM4_REG) || (args.src0.type == tu_pm4_arg::TU_PM4_IMMED));
   assert((args.src1.type == tu_pm4_arg::TU_PM4_REG) || (args.src1.type == tu_pm4_arg::TU_PM4_IMMED));

   tu_pkt7(this, CP_REG_RMW, 3)
      .add(CP_REG_RMW_0(
         .dst_reg = dst,
         .dst_scratch = dst.type == tu_pm4_arg::TU_PM4_SCRATCH,
         .skip_wait_for_me = args.skip_wfm,
         .rotate = args.rotate,
         .src1_add = args.src1_add,
         .src1_is_reg = args.src1.type == tu_pm4_arg::TU_PM4_REG,
         .src0_is_reg = args.src0.type == tu_pm4_arg::TU_PM4_REG,
      ))
      .add(args.src0)
      .add(args.src1);
}

inline void
tu_cs::cond_write(struct tu_pm4_arg dst, struct tu_pm4_arg src, struct tu_cond_write_args args)
{
   assert((dst.type == tu_pm4_arg::TU_PM4_REG) || (dst.type == tu_pm4_arg::TU_PM4_GPUVA));

   enum poll_memory_type poll;

   if (src.type == tu_pm4_arg::TU_PM4_REG)
      poll = POLL_REGISTER;
   else if (src.type == tu_pm4_arg::TU_PM4_GPUVA)
      poll = POLL_MEMORY;
   else if (src.type == tu_pm4_arg::TU_PM4_SCRATCH)
      poll = POLL_SCRATCH;
   else
      UNREACHABLE("invalid src type");

   assert((src.type == tu_pm4_arg::TU_PM4_REG) ||
          (src.type == tu_pm4_arg::TU_PM4_GPUVA) ||
          (src.type == tu_pm4_arg::TU_PM4_SCRATCH));

   tu_pkt7 pkt(this, CP_COND_WRITE5, 8);

   pkt.add(CP_COND_WRITE5_0(
      .function = args.function,
      .signed_compare = args.signed_compare,
      .poll = poll,
      .write_memory = dst.type == tu_pm4_arg::TU_PM4_GPUVA,
   ));

   if (src.type == tu_pm4_arg::TU_PM4_GPUVA) {
      pkt.add(src.gpuva.gpuva);
      pkt.add(src.gpuva.gpuva >> 32);
   } else {
      pkt.add(src);
      pkt.add(0);
   }

   pkt.add(CP_COND_WRITE5_3(.ref = args.ref));
   pkt.add(CP_COND_WRITE5_4(.mask = args.mask));

   if (dst.type == tu_pm4_arg::TU_PM4_GPUVA) {
      pkt.add(dst.gpuva.gpuva);
      pkt.add(dst.gpuva.gpuva >> 32);
   } else {
      pkt.add(dst);
      pkt.add(0);
   }

   pkt.add(CP_COND_WRITE5_7(.write_data = args.write_data));
}

inline void
tu_cs::reg_to_mem(uint64_t va, struct fd_reg_pair reg, struct tu_reg_to_mem_args args)
{
   tu_pkt7(this, CP_REG_TO_MEM, 3)
      .add(CP_REG_TO_MEM_0(
         .reg = reg.reg,
         .cnt = args.cnt,
         .is_64b = args.is_64b,
         .accumulate = args.accumulate,
      ))
      .add(va)
      .add(va >> 32);
}

inline void
tu_cs::mem_to_reg(struct fd_reg_pair reg, uint64_t va, struct tu_mem_to_reg_args args)
{
   tu_pkt7(this, CP_MEM_TO_REG, 3)
      .add(CP_MEM_TO_REG_0(
         .reg = reg.reg,
         .one_reg_wr = args.one_reg_wr,
         .cnt = args.cnt,
         .shift_by_2 = args.shift_by_2,
         .wait_cache_flush = args.wait_cache_flush,
      ))
      .add(va)
      .add(va >> 32);
}

template <chip CHIP>
static inline fd_reg_pair
tu_scratch_reg(int idx, uint32_t val = 0)
{
   if (CHIP >= A8XX) {
      return CP_SCRATCH_GLOBAL_REG(CHIP, idx, val);
   } else {
      return CP_SCRATCH_REG(CHIP, idx, val);
   }
}

#endif /* TU_CS_H */
