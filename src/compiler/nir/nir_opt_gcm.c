/*
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_instr_set.h"
#include "util/sparse_bitset.h"


/*
 * Implements Global Code Motion.  A description of GCM can be found in
 * "Global Code Motion; Global Value Numbering" by Cliff Click.
 * Unfortunately, the algorithm presented in the paper is broken in a
 * number of ways.  The algorithm used here differs substantially from the
 * one in the paper but it is, in my opinion, much easier to read and
 * verify correcness.
 */

/* This is used to stop GCM moving instruction out of a loop if the loop
 * contains too many instructions and moving them would create excess spilling.
 *
 * This is the fallback for drivers that don't set max_gcm_loop_pressure, which
 * measures what a loop actually holds instead of guessing it from how much code
 * is in there.
 */
#define MAX_LOOP_INSTRUCTIONS 100

/* Register pressure bookkeeping for a single loop.  Kept in the gcm_block_info
 * of the loop's header block, which is the one block every loop has to itself.
 */
struct gcm_loop_pressure {
   /* Highest pressure seen at any instruction inside the loop, including its
    * nested loops. Computed once, before anything is moved.
    */
   unsigned peak;

   /* Pressure added to the loop by the hoists we have already committed to.
    * The results of those hoists are live across the whole loop, so they push
    * against the same limit the loop's own peak does.
    */
   unsigned hoisted;
};

struct gcm_block_info {
   /* Number of loops this block is inside */
   unsigned loop_depth;

   /* Number of ifs this block is inside */
   unsigned if_depth;

   unsigned loop_instr_count;

   /* Register pressure at the end of this block, which is where instructions
    * moved here land. Only filled in when the driver asked for the pressure
    * heuristic.
    */
   unsigned tail_pressure;

   /* Only valid if the block is the header block of a loop, i.e. the block
    * gcm_loop_pressure_info() looks it up by, and only filled in when the
    * driver asked for the pressure heuristic.
    */
   struct gcm_loop_pressure loop_pressure;

   /* The loop the block is nested inside or NULL */
   nir_loop *loop;

   /* The last instruction inserted into this block.  This is used as we
    * traverse the instructions and insert them back into the program to
    * put them in the right order.
    */
   nir_instr *last_instr;
};

struct gcm_instr_info {
   nir_block *early_block;
};

/* Flags used in the instr->pass_flags field for various instruction states */
enum {
   GCM_INSTR_PINNED = (1 << 0),
   GCM_INSTR_SCHEDULE_EARLIER_ONLY = (1 << 1),
   GCM_INSTR_SCHEDULED_EARLY = (1 << 2),
   GCM_INSTR_SCHEDULED_LATE = (1 << 3),
   GCM_INSTR_PLACED = (1 << 4),
   GCM_INSTR_CHARGED = (1 << 5),
};

struct gcm_state {
   nir_function_impl *impl;
   nir_instr *instr;

   bool progress;

   bool second_early_run;

   /* Whether the function contains any loop at all. The second early run only
    * ever changes a block for instructions inside a loop, so when there are
    * none it would recompute exactly what the first run produced and we can
    * skip straight to a single run.
    */
   bool has_loop;

   /* Pressure a loop can be under and still absorb a hoist for free, or zero to
    * decide that from the loop's instruction count instead.
    */
   unsigned max_loop_pressure;

   /* What a divergent component costs against that budget.  One when the target
    * compiles at a single SIMD width, more when it compiles the same NIR at
    * several and the widest one has to fit too.
    */
   unsigned divergent_scale;

   /* The list of non-pinned instructions.  As we do the late scheduling,
    * we pull non-pinned instructions out of their blocks and place them in
    * this list.  This saves us from having linked-list problems when we go
    * to put instructions back in their blocks.
    */
   struct exec_list instrs;

   struct gcm_block_info *blocks;

   unsigned num_instrs;
   struct gcm_instr_info *instr_infos;
};

static unsigned
get_loop_instr_count(struct exec_list *cf_list)
{
   unsigned loop_instr_count = 0;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, block) {
            loop_instr_count++;
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *if_stmt = nir_cf_node_as_if(node);
         loop_instr_count += get_loop_instr_count(&if_stmt->then_list);
         loop_instr_count += get_loop_instr_count(&if_stmt->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         assert(!nir_loop_has_continue_construct(loop));
         loop_instr_count += get_loop_instr_count(&loop->body);
         break;
      }
      default:
         UNREACHABLE("Invalid CF node type");
      }
   }

   return loop_instr_count;
}

/* Recursively walks the CFG and builds the block_info structure */
static void
gcm_build_block_info(struct exec_list *cf_list, struct gcm_state *state,
                     nir_loop *loop, unsigned loop_depth, unsigned if_depth,
                     unsigned loop_instr_count)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);
         state->blocks[block->index].if_depth = if_depth;
         state->blocks[block->index].loop_depth = loop_depth;
         state->blocks[block->index].loop_instr_count = loop_instr_count;
         state->blocks[block->index].loop = loop;
         break;
      }
      case nir_cf_node_if: {
         nir_if *if_stmt = nir_cf_node_as_if(node);
         gcm_build_block_info(&if_stmt->then_list, state, loop, loop_depth,
                              if_depth + 1, loop_instr_count);
         gcm_build_block_info(&if_stmt->else_list, state, loop, loop_depth,
                              if_depth + 1, loop_instr_count);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         assert(!nir_loop_has_continue_construct(loop));
         state->has_loop = true;
         gcm_build_block_info(&loop->body, state, loop, loop_depth + 1, if_depth,
                              state->max_loop_pressure ?
                                 0 : get_loop_instr_count(&loop->body));
         break;
      }
      default:
         UNREACHABLE("Invalid CF node type");
      }
   }
}

static bool
is_src_scalarizable(nir_src *src)
{

   nir_instr *src_instr = nir_def_instr(src->ssa);
   switch (src_instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *src_alu = nir_instr_as_alu(src_instr);

      /* ALU operations with output_size == 0 should be scalarized.  We
       * will also see a bunch of vecN operations from scalarizing ALU
       * operations and, since they can easily be copy-propagated, they
       * are ok too.
       */
      return nir_op_infos[src_alu->op].output_size == 0 ||
             src_alu->op == nir_op_vec2 ||
             src_alu->op == nir_op_vec3 ||
             src_alu->op == nir_op_vec4;
   }

   case nir_instr_type_load_const:
      /* These are trivially scalarizable */
      return true;

   case nir_instr_type_undef:
      return true;

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *src_intrin = nir_instr_as_intrinsic(src_instr);

      switch (src_intrin->intrinsic) {
      case nir_intrinsic_load_deref: {
         /* Don't scalarize if we see a load of a local variable because it
          * might turn into one of the things we can't scalarize.
          */
         nir_deref_instr *deref = nir_src_as_deref(src_intrin->src[0]);
         return !nir_deref_mode_may_be(deref, (nir_var_function_temp |
                                               nir_var_shader_temp));
      }

      case nir_intrinsic_interp_deref_at_centroid:
      case nir_intrinsic_interp_deref_at_sample:
      case nir_intrinsic_interp_deref_at_offset:
      case nir_intrinsic_load_uniform:
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_ssbo:
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_input:
      case nir_intrinsic_load_per_primitive_input:
      case nir_intrinsic_load_ssbo_intel:
         return true;
      default:
         break;
      }

      return false;
   }

   default:
      /* We can't scalarize this type of instruction */
      return false;
   }
}

static bool
is_binding_uniform(nir_src src)
{
   nir_binding binding = nir_chase_binding(src);
   if (!binding.success)
      return false;

   for (unsigned i = 0; i < binding.num_indices; i++) {
      if (!nir_src_is_always_uniform(binding.indices[i]))
         return false;
   }

   return true;
}

static void
pin_intrinsic(nir_intrinsic_instr *intrin)
{
   nir_instr *instr = &intrin->instr;

   if (!nir_intrinsic_can_reorder(intrin)) {
      instr->pass_flags = GCM_INSTR_PINNED;
      return;
   }

   instr->pass_flags = 0;

   /* If the intrinsic requires a uniform source, we can't safely move it across non-uniform
    * control flow if it's not uniform at the point it's defined.
    * Stores and atomics can never be re-ordered, so we don't have to consider them here.
    */
   bool non_uniform = nir_intrinsic_has_access(intrin) &&
                      (nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM);
   if (!non_uniform &&
       (intrin->intrinsic == nir_intrinsic_load_ubo ||
        intrin->intrinsic == nir_intrinsic_load_ssbo ||
        intrin->intrinsic == nir_intrinsic_load_ssbo_intel ||
        intrin->intrinsic == nir_intrinsic_get_ubo_size ||
        intrin->intrinsic == nir_intrinsic_get_ssbo_size ||
        nir_intrinsic_has_image_dim(intrin) ||
        ((intrin->intrinsic == nir_intrinsic_load_deref ||
          intrin->intrinsic == nir_intrinsic_deref_buffer_array_length) &&
         nir_deref_mode_may_be(nir_src_as_deref(intrin->src[0]),
                               nir_var_mem_ubo | nir_var_mem_ssbo)))) {
      if (!is_binding_uniform(intrin->src[0]))
         instr->pass_flags = GCM_INSTR_PINNED;
   } else if (intrin->intrinsic == nir_intrinsic_load_push_constant ||
              intrin->intrinsic == nir_intrinsic_load_push_data_intel) {
      if (!nir_src_is_always_uniform(intrin->src[0]))
         instr->pass_flags = GCM_INSTR_PINNED;
   } else if (intrin->intrinsic == nir_intrinsic_load_deref &&
              nir_deref_mode_is(nir_src_as_deref(intrin->src[0]),
                                nir_var_mem_push_const)) {
      nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
      while (deref->deref_type != nir_deref_type_var) {
         if (nir_deref_instr_is_arr(deref) &&
             !nir_src_is_always_uniform(deref->arr.index)) {
            instr->pass_flags = GCM_INSTR_PINNED;
            return;
         }
         deref = nir_deref_instr_parent(deref);
         if (!deref) {
            instr->pass_flags = GCM_INSTR_PINNED;
            return;
         }
      }
   }
}

/* Walks the instruction list and marks immovable instructions as pinned or
 * placed.
 *
 * This function also serves to initialize the instr->pass_flags field.
 * After this is completed, all instructions' pass_flags fields will be set
 * to either GCM_INSTR_PINNED, GCM_INSTR_PLACED or 0.
 */
static void
gcm_pin_instructions(nir_function_impl *impl, struct gcm_state *state)
{
   state->num_instrs = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         /* Index the instructions for use in gcm_state::instrs */
         instr->index = state->num_instrs++;

         switch (instr->type) {
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instr);

            if (alu->op == nir_op_mov && !is_src_scalarizable(&alu->src[0].src)) {
               instr->pass_flags = GCM_INSTR_PINNED;
            } else {
               instr->pass_flags = 0;
            }
            break;
         }

         case nir_instr_type_tex: {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (nir_tex_instr_has_implicit_derivative(tex))
               instr->pass_flags = GCM_INSTR_SCHEDULE_EARLIER_ONLY;

            for (unsigned i = 0; i < tex->num_srcs; i++) {
               nir_tex_src *src = &tex->src[i];
               switch (src->src_type) {
               case nir_tex_src_texture_deref:
               case nir_tex_src_texture_2_deref:
                  if (!tex->texture_non_uniform && !is_binding_uniform(src->src))
                     instr->pass_flags = GCM_INSTR_PINNED;
                  break;
               case nir_tex_src_sampler_deref:
               case nir_tex_src_sampler_2_deref:
                  if (!tex->sampler_non_uniform && !is_binding_uniform(src->src))
                     instr->pass_flags = GCM_INSTR_PINNED;
                  break;
               case nir_tex_src_texture_offset:
               case nir_tex_src_texture_handle:
                  if (!tex->texture_non_uniform && !nir_src_is_always_uniform(src->src))
                     instr->pass_flags = GCM_INSTR_PINNED;
                  break;
               case nir_tex_src_sampler_offset:
               case nir_tex_src_sampler_handle:
                  if (!tex->sampler_non_uniform && !nir_src_is_always_uniform(src->src))
                     instr->pass_flags = GCM_INSTR_PINNED;
                  break;
               default:
                  break;
               }
            }
            break;
         }

         case nir_instr_type_deref:
         case nir_instr_type_load_const:
            instr->pass_flags = 0;
            break;

         case nir_instr_type_intrinsic:
            pin_intrinsic(nir_instr_as_intrinsic(instr));
            break;

         case nir_instr_type_call:
         case nir_instr_type_cmat_call:
            instr->pass_flags = GCM_INSTR_PINNED;
            break;

         case nir_instr_type_jump:
         case nir_instr_type_undef:
         case nir_instr_type_phi:
            instr->pass_flags = GCM_INSTR_PLACED;
            break;

         default:
            UNREACHABLE("Invalid instruction type in GCM");
         }

         if (!(instr->pass_flags & GCM_INSTR_PLACED)) {
            /* If this is an unplaced instruction, go ahead and pull it out of
             * the program and put it on the instrs list.  This has a couple
             * of benifits.  First, it makes the scheduling algorithm more
             * efficient because we can avoid walking over basic blocks.
             * Second, it keeps us from causing linked list confusion when
             * we're trying to put everything in its proper place at the end
             * of the pass.
             *
             * Note that we don't use nir_instr_remove here because that also
             * cleans up uses and defs and we want to keep that information.
             */
            exec_node_remove(&instr->node);
            exec_list_push_tail(&state->instrs, &instr->node);
         }
      }
   }
}

struct gcm_freed_srcs_state {
   /* Block the instruction would move to, which decides which of its srcs
    * stop being live.
    */
   unsigned block_idx;

   /* Register pressure the srcs of the instruction would stop taking up after
    * the block if we moved the instruction. Srcs that don't contribute to
    * register pressure, constants for example, are not counted.
    */
   unsigned reg_size;

   /* Set if anything else reads a src the move would free up. */
   bool shared;

   struct gcm_instr_info *instr_infos;

   /* Copy of gcm_state::divergent_scale. */
   unsigned divergent_scale;

   /* The instr we are checking if we should move */
   nir_instr *instr;

   /* The loop the instr is being moved out of */
   nir_loop *loop;
};

static bool
gcm_loop_contains_block(nir_loop *loop, nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop && nir_cf_node_as_loop(node) == loop)
         return true;
   }

   return false;
}

/* Rough estimate of how many registers a def occupies, in registers at the
 * narrowest SIMD width the target compiles at.
 *
 * A value that differs between lanes takes another register every time the
 * width doubles, so on a target that compiles the same NIR more than once it is
 * charged for the widest width worth protecting.  One that is the same in every
 * lane costs what it costs however wide the dispatch is, so it is charged once.
 * The components are packed the way the backend packs them, so a pair of 16 bit
 * ones share a register rather than taking one each.
 */
static unsigned
gcm_def_reg_size(const nir_def *def, unsigned divergent_scale)
{
   /* A target that compiles at a single width has nothing to scale for, and
    * gets the size it has always been given.
    */
   if (divergent_scale == 1)
      return def->num_components * DIV_ROUND_UP(def->bit_size, 32);

   unsigned size = DIV_ROUND_UP(def->num_components * def->bit_size, 32);

   return def->divergent ? size * divergent_scale : size;
}

/* The loop immediately containing the given loop, or NULL if it is outermost. */
static nir_loop *
gcm_parent_loop(nir_loop *loop)
{
   for (nir_cf_node *node = loop->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(node);
   }

   return NULL;
}

static struct gcm_loop_pressure *
gcm_loop_pressure_info(struct gcm_state *state, nir_loop *loop)
{
   return &state->blocks[nir_loop_first_block(loop)->index].loop_pressure;
}

/* Scratch the pressure walk reuses for every block, so that walking a function
 * full of loops doesn't allocate once per block.
 */
struct gcm_pressure_scratch {
   /* The defs live at the point the walk has reached, one bit per def.  A plain
    * bitset rather than the sparse one liveness itself uses: the walk tests and
    * flips a bit for every source and def it passes, and here that is an index
    * into an array rather than a tree lookup.
    */
   BITSET_WORD *live;

   /* The bits set in `live' since the last reset, so that clearing it again
    * costs what the block touched rather than the size of the function.  Bits
    * the walk has since cleared are in here too, which does no harm.
    */
   unsigned *touched;
   unsigned num_touched;

   /* Every def in the function, indexed by def->index, so that a bit in a live
    * set can be turned back into the size of what it stands for.
    */
   nir_def **defs;

   /* Copy of gcm_state::divergent_scale, so the walk can size a def without
    * carrying the whole state around.
    */
   unsigned divergent_scale;
};

struct gcm_pressure_state {
   struct gcm_pressure_scratch *scratch;

   unsigned pressure;
};

static void
gcm_pressure_set_live(struct gcm_pressure_scratch *scratch, unsigned index)
{
   /* A def is only ever made live once per block: within a block every use of
    * it comes after it is defined, so walking backwards we are done with it by
    * the time the walk kills it.  That is what bounds `touched' by the number
    * of defs in the function.
    */
   assert(!BITSET_TEST(scratch->live, index));

   BITSET_SET(scratch->live, index);
   scratch->touched[scratch->num_touched++] = index;
}

static void
gcm_pressure_reset_live(struct gcm_pressure_scratch *scratch)
{
   for (unsigned i = 0; i < scratch->num_touched; i++)
      BITSET_CLEAR(scratch->live, scratch->touched[i]);

   scratch->num_touched = 0;
}

static bool
gcm_record_def(nir_def *def, void *void_defs)
{
   nir_def **defs = void_defs;
   defs[def->index] = def;

   return true;
}

/* Walking backwards, a def stops being live at the instruction that produces
 * it.  A def nothing goes on to read is not in the live set at all and never
 * counted against pressure.
 */
static bool
gcm_pressure_kill_def(nir_def *def, void *void_state)
{
   struct gcm_pressure_state *state = void_state;

   if (BITSET_TEST(state->scratch->live, def->index)) {
      BITSET_CLEAR(state->scratch->live, def->index);
      state->pressure -= gcm_def_reg_size(def, state->scratch->divergent_scale);
   }

   return true;
}

/* Walking backwards, a def becomes live at the last instruction that reads it.
 * Instructions that read the same def twice only make it live once.
 */
static bool
gcm_pressure_gen_src(nir_src *src, void *void_state)
{
   struct gcm_pressure_state *state = void_state;

   if (!BITSET_TEST(state->scratch->live, src->ssa->index)) {
      gcm_pressure_set_live(state->scratch, src->ssa->index);
      state->pressure += gcm_def_reg_size(src->ssa,
                                         state->scratch->divergent_scale);
   }

   return true;
}

/* What is still live when the block ends, which is what an instruction placed
 * here has to fit alongside.  Seeds the scratch live set with it, since the
 * walk below starts from exactly the same place.
 */
static unsigned
gcm_block_tail_pressure(nir_block *block, struct gcm_pressure_scratch *scratch)
{
   unsigned pressure = 0;

   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, index) {
      assert(scratch->defs[index] != NULL);
      pressure += gcm_def_reg_size(scratch->defs[index],
                                  scratch->divergent_scale);
      gcm_pressure_set_live(scratch, index);
   }

   return pressure;
}

/* Highest register pressure reached at any point in the block.  This walks the
 * block backwards from its live out set, the same way liveness itself is
 * computed, so that pressure peaks in the middle of a long block are seen and
 * not just the pressure at its edges.
 */
static unsigned
gcm_block_peak_pressure(nir_block *block, struct gcm_pressure_scratch *scratch)
{
   struct gcm_pressure_state state = {
      .scratch = scratch,
      .pressure = gcm_block_tail_pressure(block, scratch),
   };

   unsigned peak = state.pressure;

   nir_foreach_instr_reverse(instr, block) {
      /* Phis don't read their sources in this block, they read them in the
       * predecessors, so liveness stops at them.  As they come first we can
       * just stop walking.
       */
      if (instr->type == nir_instr_type_phi)
         break;

      nir_foreach_def(instr, gcm_pressure_kill_def, &state);
      nir_foreach_src(instr, gcm_pressure_gen_src, &state);

      peak = MAX2(peak, state.pressure);
   }

   gcm_pressure_reset_live(scratch);

   return peak;
}

/* Records the peak pressure of every loop in the function, before we have moved
 * anything.  A loop's peak has to account for its nested loops too, because a
 * value hoisted into it is held across those as well.
 */
static void
gcm_compute_loop_pressure(struct gcm_state *state)
{
   nir_function_impl *impl = state->impl;

   /* Liveness sizes a bitset per block from impl->ssa_alloc, which is a high
    * water mark: it counts every index ever handed out, including those of
    * instructions long since deleted.  By the time we get here that can be
    * several times the number of defs the function still has, which spreads the
    * live sets over far more sparse bitset nodes than they need.  Renumbering
    * first costs one walk and is paid back many times over.
    */
   nir_index_ssa_defs(impl);

   nir_metadata_require(impl, nir_metadata_live_defs);

   void *mem_ctx = ralloc_context(NULL);
   struct gcm_pressure_scratch scratch = {
      .live = rzalloc_array(mem_ctx, BITSET_WORD, BITSET_WORDS(impl->ssa_alloc)),
      .touched = ralloc_array(mem_ctx, unsigned, impl->ssa_alloc),
      .defs = rzalloc_array(mem_ctx, nir_def *, impl->ssa_alloc),
      .divergent_scale = state->divergent_scale,
   };

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         nir_foreach_def(instr, gcm_record_def, scratch.defs);
   }

   nir_foreach_block(block, impl) {
      nir_loop *loop = state->blocks[block->index].loop;

      if (loop == NULL) {
         state->blocks[block->index].tail_pressure =
            gcm_block_tail_pressure(block, &scratch);
         gcm_pressure_reset_live(&scratch);
         continue;
      }

      unsigned peak = gcm_block_peak_pressure(block, &scratch);

      for (; loop != NULL; loop = gcm_parent_loop(loop)) {
         struct gcm_loop_pressure *pressure =
            gcm_loop_pressure_info(state, loop);
         pressure->peak = MAX2(pressure->peak, peak);
      }
   }

   ralloc_free(mem_ctx);
}

/* The loops a move from the instruction's block to the given block would lift
 * the result out of, from the innermost outwards.  The result becomes live
 * across every one of them.
 */
#define gcm_foreach_loop_hoisted_out_of(loop, state, instr, block)     \
   for (nir_loop *loop = (state)->blocks[(instr)->block->index].loop;  \
        loop != NULL && !gcm_loop_contains_block(loop, block);         \
        loop = gcm_parent_loop(loop))

/* Whether the loops the move would lift the instruction out of can hold its
 * result without being pushed past what their registers can take.  A loop with
 * room to spare absorbs the result without issue, so there is no need to make
 * the move pay for itself out of the sources it frees.
 *
 * Drivers that haven't set a pressure limit get the old heuristics instead:
 * A loop with few enough instructions in it is assumed to have room.  What
 * makes such a loop safe to empty out is that the instruction ends up outside
 * every loop, where holding its result costs nothing, so only a loop that
 * isn't nested inside another one qualifies.
 */
static bool
gcm_loops_have_room_for_instr(struct gcm_state *state, nir_instr *instr,
                              nir_block *block)
{
   struct gcm_block_info *info = &state->blocks[instr->block->index];

   if (!state->max_loop_pressure) {
      return info->loop_depth == 1 &&
             info->loop_instr_count < MAX_LOOP_INSTRUCTIONS;
   }

   nir_def *def = nir_instr_def(instr);
   if (!def)
      return false;

   unsigned cost = gcm_def_reg_size(def, state->divergent_scale);

   /* Where it lands has to have room for it too. Landing inside another loop
    * means being held across that one from now on, so what matters there is its
    * peak; landing outside every loop, it only has to fit alongside what is
    * still live where it is put, which is the end of the block.
    */
   nir_loop *landing_loop = state->blocks[block->index].loop;
   unsigned landing = landing_loop != NULL ?
                      gcm_loop_pressure_info(state, landing_loop)->peak :
                      state->blocks[block->index].tail_pressure;

   gcm_foreach_loop_hoisted_out_of(loop, state, instr, block) {
      struct gcm_loop_pressure *pressure = gcm_loop_pressure_info(state, loop);
      unsigned held = MAX2(pressure->peak, landing);

      if (held + pressure->hoisted + cost > state->max_loop_pressure)
         return false;
   }

   return true;
}

/* Moves a hoist on or off the loops it takes the result out of, so that later
 * moves see the pressure it added. Instructions that are let out on their type
 * rather than on what they cost count the same as any other: a texture result
 * held across a loop takes a register whether or not anything weighed it.
 */
static void
gcm_move_loop_pressure(struct gcm_state *state, nir_instr *instr,
                       nir_block *block, bool charge)
{
   if (!state->max_loop_pressure)
      return;

   nir_def *def = nir_instr_def(instr);
   if (def == NULL)
      return;

   /* Landing after a loop rather than above it leaves nothing held across the
    * loop, so there is nothing for it to pay. Only a result computed before the
    * loop and read from inside it occupies a register for the whole of it.
    *
    * The loops the move leaves are nested, so asking this of the innermost one
    * answers it for all of them: a block that doesn't dominate its header can't
    * dominate the header of a loop that contains it either.
    */
   nir_loop *innermost = state->blocks[instr->block->index].loop;
   if (innermost == NULL ||
       !nir_block_dominates(block, nir_loop_first_block(innermost)))
      return;

   unsigned def_reg_size = gcm_def_reg_size(def, state->divergent_scale);

   gcm_foreach_loop_hoisted_out_of(loop, state, instr, block) {
      struct gcm_loop_pressure *pressure = gcm_loop_pressure_info(state, loop);

      if (charge) {
         pressure->hoisted += def_reg_size;
      } else {
         assert(pressure->hoisted >= def_reg_size);
         pressure->hoisted -= def_reg_size;
      }
   }
}

static void
gcm_charge_loop_hoist(struct gcm_state *state, nir_instr *instr,
                      nir_block *block)
{
   gcm_move_loop_pressure(state, instr, block, true);
}

/* Hands back what a move was charged, for when it turns out not to be the move
 * that gets made. Asking the same question of the same blocks as the charge did
 * gives back exactly what it took.
 */
static void
gcm_refund_loop_hoist(struct gcm_state *state, nir_instr *instr,
                      nir_block *block)
{
   gcm_move_loop_pressure(state, instr, block, false);
}

struct gcm_first_read_state {
   const nir_def *def;
   nir_src *first;
};

static bool
gcm_find_first_read(nir_src *src, void *void_state)
{
   struct gcm_first_read_state *state = void_state;

   if (src->ssa != state->def)
      return true;

   state->first = src;

   return false;
}

/* An instruction can read the same def from more than one src e.g.
 * fmul ssa_7.x, ssa_7.y. Comparing the src against the first one the
 * instruction reads the def from lets us account for it exactly once.
 */
static bool
gcm_is_first_read_of_def(nir_instr *instr, nir_src *src)
{
   struct gcm_first_read_state state = { .def = src->ssa };

   nir_foreach_src(instr, gcm_find_first_read, &state);

   return state.first == src;
}

/* Adds up the register pressure that moving the instruction to the state's
 * block would free, by tallying the srcs that stop being live once they are no
 * longer read from where the instruction sits now.  The answer accumulates
 * into the state; the return value only tells nir_foreach_src to keep going.
 */
static bool
gcm_add_freed_src_pressure(nir_src *src, void *void_state)
{
   struct gcm_freed_srcs_state *state = void_state;

   /* Don't count sources that don't contribute to register pressure */
   if (nir_def_instr(src->ssa)->type == nir_instr_type_load_const) {
      return true;
   }

   if (!gcm_is_first_read_of_def(state->instr, src))
      return true;

   /* Register pressure of the defs of the users we would move along with the
    * instruction. Their results have to live across the loop in place of the
    * src, so this has to be paid for out of what the src frees up.
    */
   unsigned moveable_reg_size = 0;
   bool shared = false;

   nir_foreach_use_including_if(use_src, src->ssa) {
      /* An if condition can't be moved anywhere, so the src stays live if the
       * branch is at or after the block we are moving from.
       */
      if (nir_src_is_if(use_src)) {
         nir_cf_node *prev =
            nir_cf_node_prev(&nir_src_use_if(use_src)->cf_node);
         shared = true;

         if (gcm_loop_contains_block(state->loop, nir_cf_node_as_block(prev)) ||
             nir_cf_node_as_block(prev)->index >= state->instr->block->index)
            return true;
         continue;
      }

      nir_instr *use_instr = nir_src_use_instr(use_src);

      /* Skip dead instructions. These get cleaned up elsewhere */
      if (use_instr->block == NULL) {
         continue;
      }

      /* Skip if its used by the instruction were are trying to move */
      if (use_instr == state->instr)
         continue;

      shared = true;

      bool use_in_loop = gcm_loop_contains_block(state->loop, use_instr->block);

      /* A read that happens before the loop is done with by the time we get
       * here, so it says nothing about whether the src has to survive the
       * block.
       */
      if (!use_in_loop && use_instr->block->index < state->instr->block->index)
         continue;

      /* Any user we leave behind keeps the whole src live. Registers are
       * handed out a def at a time, so there is nothing to be gained from
       * freeing up some of its components while a user of the rest stays put.
       */
      if (!use_in_loop || use_instr->block->index > state->instr->block->index)
         return true;

      /* What is left is a user inside the loop that isn't below us. Sitting
       * above us in program order doesn't make it harmless, because the loop
       * comes back around to it:
       *
       *    loop {
       *       a = fadd ssa_7, x     <- reads ssa_7 again every iteration
       *       ...
       *       b = fmul ssa_7, y     <- the one we are thinking of moving out
       *    }
       *
       * Taking b out of the loop doesn't stop ssa_7 being live across it,
       * because a still wants it. Whether moving b frees anything comes down to
       * whether a can come too, which is the same question we ask of a user
       * sharing our block, so ask it the same way instead of letting program
       * order answer it. These are ordinary instructions rather than phis, and
       * there are a lot of them: on a Cyberpunk capture the users reaching this
       * point are 97% ALU and under 1% phi.
       *
       * If they can be hoisted to the same place they stop keeping the src
       * live, but their own results then live across the loop instead. For
       * example all three of these users of ssa_7 can move together, so ssa_7
       * stops being live and three scalars take its place:
       *
       *    vec1 32 ssa_430 = ffma ssa_7.x, ssa_23.x, ssa_4.x
       *    vec1 32 ssa_433 = ffma ssa_7.y, ssa_23.x, ssa_4.y
       *    vec1 32 ssa_436 = ffma ssa_7.z, ssa_23.x, ssa_4.z
       *
       * Placed instructions are not in gcm_state::instrs so they only get
       * scheduled early if they are reached via the sources of an unplaced
       * instruction. A phi that is only used by other phis or by an if
       * condition never is, leaving it without an early block. It can't be
       * moved anywhere anyway, so just use the block it's already in.
       */
      struct gcm_instr_info *info = &state->instr_infos[use_instr->index];
      nir_block *use_early_block =
         info->early_block ? info->early_block : use_instr->block;

      if (use_early_block->index > state->block_idx)
         return true;

      nir_def *use_def = nir_instr_def(use_instr);
      if (use_def && gcm_is_first_read_of_def(use_instr, use_src))
         moveable_reg_size += gcm_def_reg_size(use_def, state->divergent_scale);
   }

   /* If moving the other users costs more registers than the src frees up we
    * are better off leaving everything where it is.
    */
   unsigned freed_reg_size = gcm_def_reg_size(src->ssa, state->divergent_scale);
   if (moveable_reg_size > freed_reg_size)
      return true;

   state->reg_size += freed_reg_size;

   if (shared)
      state->shared = true;

   return true;
}


/* How much moving this instruction would free up over what it costs, weighed
 * the same way the caller does so we can look at what moving one instruction
 * would do for the next one along.
 */
static unsigned
gcm_move_reg_surplus(struct gcm_state *state, nir_instr *instr,
                     nir_block *block)
{
   nir_def *def = nir_instr_def(instr);
   if (!def)
      return 0;

   struct gcm_freed_srcs_state freed_srcs = {0};
   freed_srcs.block_idx = block->index;
   freed_srcs.instr = instr;
   freed_srcs.instr_infos = state->instr_infos;
   freed_srcs.divergent_scale = state->divergent_scale;
   freed_srcs.loop = state->blocks[instr->block->index].loop;
   nir_foreach_src(instr, gcm_add_freed_src_pressure, &freed_srcs);

   unsigned def_reg_size = gcm_def_reg_size(def, state->divergent_scale);
   if (freed_srcs.reg_size <= def_reg_size)
      return 0;

   return freed_srcs.reg_size - def_reg_size;
}

struct gcm_use_srcs_state {
   struct gcm_state *state;

   /* The value being moved out, which the reader gets to follow */
   const nir_def *def;

   nir_loop *loop;

   bool can_leave;
};

static bool
gcm_src_stuck_in_loop(nir_src *src, void *void_state)
{
   struct gcm_use_srcs_state *st = void_state;

   /* This one is coming with us */
   if (src->ssa == st->def)
      return true;

   nir_instr *src_instr = nir_def_instr(src->ssa);

   /* Dead, cleaned up elsewhere */
   if (!src_instr->block)
      return true;

   /* Judge the source by its early block, not by where it happens to be
    * sitting right now. What really pins a use is a source that can never be
    * outside: a phi, something pinned, or an instruction the early scheduling
    * already decided to keep in the loop.
    *
    * Placed instructions that are only reachable through phis never get an
    * early block; they cannot move, so where they sit is where they stay.
    */
   struct gcm_instr_info *info = &st->state->instr_infos[src_instr->index];
   nir_block *highest = info->early_block ? info->early_block
                                          : src_instr->block;

   if (st->state->blocks[highest->index].loop == st->loop)
      st->can_leave = false;

   return st->can_leave;
}

/* Could this use actually follow us out of the loop?
 *
 * A use with another source that can never be outside the loop can never
 * be outside it either.
 */
static bool
gcm_use_can_leave_loop(struct gcm_state *state, nir_instr *use_instr,
                       const nir_def *def, nir_loop *loop)
{
   struct gcm_use_srcs_state st = {
      .state = state,
      .def = def,
      .loop = loop,
      .can_leave = true,
   };

   nir_foreach_src(use_instr, gcm_src_stuck_in_loop, &st);

   return st.can_leave;
}

/* Would moving this instruction out let something using it move also?
 *
 * An instruction that only breaks even is still worth moving when a use can
 * follow it out of the loop and free up more than it costs, because the pair
 * taken together comes out ahead even though the first half of it does not.
 * The ones to leave behind are those that feed nothing, since all they do is
 * swap one value the loop carries for another while giving the scheduler less
 * to work with inside the loop.
 */
static bool
gcm_move_frees_registers_for_use(struct gcm_state *state, nir_instr *instr,
                                 nir_block *block)
{
   nir_def *def = nir_instr_def(instr);
   if (!def)
      return false;

   nir_loop *loop = state->blocks[instr->block->index].loop;

   unsigned best_surplus = 0;

   nir_foreach_use_including_if(use_src, def) {
      /* A branch reads it where it is, so it can't be the reason we move */
      if (nir_src_is_if(use_src))
         continue;

      nir_instr *use_instr = nir_src_use_instr(use_src);

      /* Skip dead instructions. These get cleaned up elsewhere */
      if (!use_instr->block)
         continue;

      /* Read from outside the loop, so the value is carried across it either
       * way and this reader is not going anywhere on our account.
       */
      if (state->blocks[use_instr->block->index].loop != loop)
         continue;

      if (use_instr->pass_flags & (GCM_INSTR_PINNED | GCM_INSTR_PLACED))
         continue;

      /* A use that has already been late scheduled and is still in the loop
       * is not following anybody anywhere.
       */
      if (use_instr->pass_flags & GCM_INSTR_SCHEDULED_LATE)
         continue;

      /* It still keeps our result live wherever it ends up, it just cannot be
       * the reason we move.
       */
      if (!gcm_use_can_leave_loop(state, use_instr, def, loop))
         continue;

      unsigned surplus = gcm_move_reg_surplus(state, use_instr, block);
      best_surplus = MAX2(best_surplus, surplus);
   }

   /* What the reader frees has to be worth what we are taking out of the loop.
    * A single reader really does carry the value out with it, so the loop is
    * not left holding it -- but the ledger saying so counts values live across
    * the loop and nothing else, and we are asking it about a difference of a
    * register or two. Treating that as free is how a vec4 leaves a loop on the
    * strength of one register's worth of gain, which is well inside what this
    * ledger can actually tell apart. Ask for a gain worth the size of the thing
    * being moved instead, whether or not the loop ends up holding it.
    */
   unsigned cost = gcm_def_reg_size(def, state->divergent_scale);

   return best_surplus && best_surplus >= cost;
}

/* Decides whether the instruction can be moved to the given block. Sets
 * *pays_for_itself when the move is one the cost model let out because it frees
 * up at least as much as it takes, which is what tells the loops it leaves that
 * they are no worse off holding the result.
 */
static bool
set_block_for_loop_instr(struct gcm_state *state, nir_instr *instr,
                         nir_block *block, bool *pays_for_itself)
{
   *pays_for_itself = false;

   /* If the instruction wasn't in a loop to begin with we don't want to push
    * it down into one.
    */
   nir_loop *loop = state->blocks[instr->block->index].loop;
   if (loop == NULL)
      return true;

   assert(!nir_loop_has_continue_construct(loop));
   if (nir_block_dominates(instr->block, block))
      return true;

   /* If the instruction is in a block ending in a break don't bother trying to
    * move it outside the loop.
    */
   if (nir_block_ends_in_break(instr->block))
      return false;

   /* If the loop only executes a single time i.e its wrapped in a:
    *    do{ ... break; } while(true)
    * Don't move the instruction as it will not help anything.
    */
   if (loop->info->limiting_terminator == NULL && !loop->info->complex_loop &&
       nir_block_ends_in_break(nir_loop_last_block(loop)))
      return false;

   /* A loop that runs at most twice saves at most one repeat of the
    * instruction, which isn't worth holding its result across the loop for. A
    * trip count of zero means we don't know how many times it runs, which is
    * no reason to keep anything in.
    */
   if (loop->info->max_trip_count != 0 && loop->info->max_trip_count <= 2)
      return false;

   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_tex ||
       (instr->type == nir_instr_type_intrinsic &&
        nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_resource_intel))
      return true;

   /* Being too aggressive with how we pull instructions out of loops can
    * result in extra register pressure and spilling. For example its fairly
    * common for loops in compute shaders to calculate SSBO offsets using
    * the workgroup id, subgroup id and subgroup invocation, pulling all
    * these calculations outside the loop causes register pressure.
    *
    * What we spend to move an instruction out is a register held across every
    * loop it leaves, so a loop that isn't using the registers it has can hand
    * one over for nothing.
    */
   if (gcm_loops_have_room_for_instr(state, instr, block))
      return true;

   /* The loops are full, so from here the move has to pay its own way: it is
    * only worth making if it would free up at least as much register pressure
    * as it costs. Hoisting the instruction extends the live range of its
    * result across the loop, so the sources that stop being live after the
    * block it would be moved to have to be worth at least as much as the
    * result we are hoisting.
    *
    * This is what keeps wide results in the loop. A vec4 load_ubo built from a
    * couple of scalar sources, for example, would trade two registers for four
    * held across the whole loop, so it stays where it is unless it really does
    * free up four components worth of sources.
    */
   nir_def *def = nir_instr_def(instr);
   if (!def)
      return false;

   struct gcm_freed_srcs_state freed_srcs = {0};
   freed_srcs.block_idx = block->index;
   freed_srcs.instr = instr;
   freed_srcs.instr_infos = state->instr_infos;
   freed_srcs.divergent_scale = state->divergent_scale;
   freed_srcs.loop = loop;
   nir_foreach_src(instr, gcm_add_freed_src_pressure, &freed_srcs);

   unsigned def_reg_size = gcm_def_reg_size(def, state->divergent_scale);

   if (freed_srcs.reg_size < def_reg_size)
      return false;

   /* Moving out something that frees up more than it costs always leaves the
    * loop better off.
    */
   if (freed_srcs.reg_size > def_reg_size) {
      *pays_for_itself = true;
      return true;
   }

   /* Nothing else uses what this frees up, so unlike a break even move in
    * general it really does hand the loop back as much as it takes.
    */
   if (!freed_srcs.shared) {
      *pays_for_itself = true;
      return true;
   }

   /* The rest only break even on their own, so they are worth moving out only
    * when they let one of their uses out too.
    */
   *pays_for_itself = gcm_move_frees_registers_for_use(state, instr, block);

   return *pays_for_itself;
}

static void
gcm_schedule_early_instr(nir_instr *instr, struct gcm_state *state);

/** Update an instructions schedule for the given source
 *
 * This function is called iteratively as we walk the sources of an
 * instruction.  It ensures that the given source instruction has been
 * scheduled and then update this instruction's block if the source
 * instruction is lower down the tree.
 */
static bool
gcm_schedule_early_src(nir_src *src, void *void_state)
{
   struct gcm_state *state = void_state;
   nir_instr *instr = state->instr;

   gcm_schedule_early_instr(nir_def_instr(src->ssa), void_state);

   /* While the index isn't a proper dominance depth, it does have the
    * property that if A dominates B then A->index <= B->index.  Since we
    * know that this instruction must have been dominated by all of its
    * sources at some point (even if it's gone through value-numbering),
    * all of the sources must lie on the same branch of the dominance tree.
    * Therefore, we can just go ahead and just compare indices.
    */
   struct gcm_instr_info *src_info =
      &state->instr_infos[nir_def_instr(src->ssa)->index];
   struct gcm_instr_info *info = &state->instr_infos[instr->index];
   if (info->early_block->index < src_info->early_block->index)
      info->early_block = src_info->early_block;

   /* We need to restore the state instruction because it may have been
    * changed through the gcm_schedule_early_instr call above.  Since we
    * may still be iterating through sources and future calls to
    * gcm_schedule_early_src for the same instruction will still need it.
    */
   state->instr = instr;

   return true;
}

/** Schedules an instruction early
 *
 * This function performs a recursive depth-first search starting at the
 * given instruction and proceeding through the sources to schedule
 * instructions as early as they can possibly go in the dominance tree.
 * The instructions are "scheduled" by updating the early_block field of
 * the corresponding gcm_instr_state entry.
 */
static void
gcm_schedule_early_instr(nir_instr *instr, struct gcm_state *state)
{
   if (instr->pass_flags & GCM_INSTR_SCHEDULED_EARLY)
      return;

   instr->pass_flags |= GCM_INSTR_SCHEDULED_EARLY;

   /* Pinned/placed instructions always get scheduled in their original block so
    * we don't need to do anything.  Also, bailing here keeps us from ever
    * following the sources of phi nodes which can be back-edges.
    */
   if (instr->pass_flags & GCM_INSTR_PINNED ||
       instr->pass_flags & GCM_INSTR_PLACED) {
      state->instr_infos[instr->index].early_block = instr->block;
      return;
   }

   /* Start with the instruction at the top.  As we iterate over the
    * sources, it will get moved down as needed.
    */
   state->instr_infos[instr->index].early_block = nir_start_block(state->impl);
   state->instr = instr;

   nir_foreach_src(instr, gcm_schedule_early_src, state);

   /* If we are not going to move an instruction out of a loop due to our
    * heuristics we need to set the earlier block so that we don't end up trying
    * to schedule its users outside of the loop.
    */
   if (state->second_early_run) {
      struct gcm_instr_info *info = &state->instr_infos[instr->index];
      bool pays_for_itself;
      if (state->blocks[instr->block->index].loop &&
          info->early_block != instr->block) {
         if (set_block_for_loop_instr(state, instr, info->early_block,
                                      &pays_for_itself)) {
            /* Spend what letting this one out costs. Marking it charged keeps
             * the second early run from charging the loop a second time for
             * the same register.
             */
            if (!pays_for_itself) {
               gcm_charge_loop_hoist(state, instr, info->early_block);
               instr->pass_flags |= GCM_INSTR_CHARGED;
            }
         } else {
            info->early_block = instr->block;
         }
      }
   }
}

static bool
set_block_to_if_block(struct gcm_state *state, nir_instr *instr,
                      nir_block *block)
{
   if (instr->type == nir_instr_type_load_const)
      return true;

   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_resource_intel)
      return true;

   /* TODO: Figure out some more heuristics to allow more to be moved into
    * if-statements.
    */

   return false;
}

static nir_block *
gcm_choose_block_for_instr(nir_instr *instr, nir_block *early_block,
                           nir_block *late_block, struct gcm_state *state,
                           bool *pays_for_itself)
{
   *pays_for_itself = false;

   /* We might have tried to force the the instruction to stay inside a loop in
    * gcm_schedule_early_instr() but if GVN has already moved the LCA higher
    * then we need to fix it up here.
    */
   if (!nir_block_dominates(early_block, late_block))
      early_block = nir_dominance_lca(early_block, late_block);

   bool block_set = false;

   /* First see if we can push the instruction down into an if-statements block */
   nir_block *best = late_block;
   for (nir_block *block = late_block; block != NULL; block = block->imm_dom) {
      if (state->blocks[block->index].loop_depth >
          state->blocks[instr->block->index].loop_depth) {
         /* Still stop at the early block, going any higher would put the
          * instruction above its own sources.
          */
         if (block == early_block)
            break;

         continue;
      }

      if (state->blocks[block->index].if_depth >=
             state->blocks[best->index].if_depth &&
          set_block_to_if_block(state, instr, block)) {
         /* If we are pushing the instruction into an if we want it to be
          * in the earliest block not the latest to avoid creating register
          * pressure issues. So we don't break unless we come across the
          * block the instruction was originally in.
          */
         best = block;
         block_set = true;
         if (block == instr->block)
            break;
      } else if (block == instr->block) {
         /* If we couldn't push the instruction later just put is back where it
          * was previously.
          */
         if (!block_set)
            best = block;
         break;
      }

      if (block == early_block)
         break;
   }

   /* Now see if we can evict the instruction from a loop */
   for (nir_block *block = late_block; block != NULL; block = block->imm_dom) {
      if (state->blocks[block->index].loop_depth <
          state->blocks[best->index].loop_depth) {
         bool block_pays_for_itself;
         if (set_block_for_loop_instr(state, instr, block,
                                      &block_pays_for_itself)) {
            best = block;
            *pays_for_itself = block_pays_for_itself;
         } else if (block == instr->block) {
            if (!block_set)
               best = block;
            break;
         }
      }

      if (block == early_block)
         break;
   }

   return best;
}

static void
gcm_schedule_late_instr(nir_instr *instr, struct gcm_state *state);

/** Schedules the instruction associated with the given SSA def late
 *
 * This function works by first walking all of the uses of the given SSA
 * definition, ensuring that they are scheduled, and then computing the LCA
 * (least common ancestor) of its uses.  It then schedules this instruction
 * as close to the LCA as possible while trying to stay out of loops.
 */
static bool
gcm_schedule_late_def(nir_def *def, void *void_state)
{
   struct gcm_state *state = void_state;

   nir_block *lca = NULL;

   nir_foreach_use(use_src, def) {
      nir_instr *use_instr = nir_src_use_instr(use_src);

      gcm_schedule_late_instr(use_instr, state);

      /* Phi instructions are a bit special.  SSA definitions don't have to
       * dominate the sources of the phi nodes that use them; instead, they
       * have to dominate the predecessor block corresponding to the phi
       * source.  We handle this by looking through the sources, finding
       * any that are usingg this SSA def, and using those blocks instead
       * of the one the phi lives in.
       */
      if (use_instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(use_instr);

         nir_foreach_phi_src(phi_src, phi) {
            if (phi_src->src.ssa == def)
               lca = nir_dominance_lca(lca, phi_src->pred);
         }
      } else {
         lca = nir_dominance_lca(lca, use_instr->block);
      }
   }

   nir_foreach_if_use(use_src, def) {
      nir_if *if_stmt = nir_src_use_if(use_src);

      /* For if statements, we consider the block to be the one immediately
       * preceding the if CF node.
       */
      nir_block *pred_block =
         nir_cf_node_as_block(nir_cf_node_prev(&if_stmt->cf_node));

      lca = nir_dominance_lca(lca, pred_block);
   }

   nir_instr *instr = nir_def_instr(def);
   nir_block *early_block = state->instr_infos[instr->index].early_block;

   /* Refund the early run's provisional charge before deciding anything.  A
    * dead instruction never pays it back, and choosing a block while it still
    * stands counts this instruction's cost twice.  The move we settle on is
    * charged below.
    */
   if (instr->pass_flags & GCM_INSTR_CHARGED) {
      gcm_refund_loop_hoist(state, instr, early_block);
      instr->pass_flags &= ~GCM_INSTR_CHARGED;
   }

   /* Some instructions may never be used.  Flag them and the instruction
    * placement code will get rid of them for us.
    */
   if (lca == NULL) {
      instr->block = NULL;
      return true;
   }

   if (instr->pass_flags & GCM_INSTR_SCHEDULE_EARLIER_ONLY &&
       lca != instr->block && nir_block_dominates(instr->block, lca)) {
      lca = instr->block;
   }

   /* We now have the LCA of all of the uses.  If our invariants hold,
    * this is dominated by the block that we chose when scheduling early.
    * We now walk up the dominance tree and pick the lowest block that is
    * as far outside loops as we can get.
    */
   bool pays_for_itself;
   nir_block *best_block =
      gcm_choose_block_for_instr(instr, early_block, lca, state,
                                 &pays_for_itself);

   if (nir_def_block(def) != best_block) {
      /* Make the loops it leaves pay for holding the result, so the moves we
       * look at after this one see the registers this one used up. A move that
       * frees up as much as it takes leaves them no worse off, so there is
       * nothing to charge it for.
       */
      if (!pays_for_itself)
         gcm_charge_loop_hoist(state, instr, best_block);

      state->progress = true;
   }

   instr->block = best_block;

   return true;
}

/** Schedules an instruction late
 *
 * This function performs a depth-first search starting at the given
 * instruction and proceeding through its uses to schedule instructions as
 * late as they can reasonably go in the dominance tree.  The instructions
 * are "scheduled" by updating their instr->block field.
 *
 * The name of this function is actually a bit of a misnomer as it doesn't
 * schedule them "as late as possible" as the paper implies.  Instead, it
 * first finds the lates possible place it can schedule the instruction and
 * then possibly schedules it earlier than that.  The actual location is as
 * far down the tree as we can go while trying to stay out of loops.
 */
static void
gcm_schedule_late_instr(nir_instr *instr, struct gcm_state *state)
{
   if (instr->pass_flags & GCM_INSTR_SCHEDULED_LATE)
      return;

   instr->pass_flags |= GCM_INSTR_SCHEDULED_LATE;

   /* Pinned/placed instructions are already scheduled so we don't need to do
    * anything.  Also, bailing here keeps us from ever following phi nodes
    * which can be back-edges.
    */
   if (instr->pass_flags & GCM_INSTR_PLACED ||
       instr->pass_flags & GCM_INSTR_PINNED)
      return;

   nir_foreach_def(instr, gcm_schedule_late_def, state);
}

static bool
gcm_replace_def_with_undef(nir_def *def, void *void_state)
{
   struct gcm_state *state = void_state;

   if (nir_def_is_unused(def))
      return true;

   nir_undef_instr *undef =
      nir_undef_instr_create(state->impl->function->shader,
                             def->num_components, def->bit_size);
   nir_instr_insert(nir_before_impl(state->impl), &undef->instr);
   nir_def_rewrite_uses(def, &undef->def);

   return true;
}

/** Places an instrution back into the program
 *
 * The earlier passes of GCM simply choose blocks for each instruction and
 * otherwise leave them alone.  This pass actually places the instructions
 * into their chosen blocks.
 *
 * To do so, we simply insert instructions in the reverse order they were
 * extracted. This will simply place instructions that were scheduled earlier
 * onto the end of their new block and instructions that were scheduled later to
 * the start of their new block.
 */
static void
gcm_place_instr(nir_instr *instr, struct gcm_state *state)
{
   if (instr->pass_flags & GCM_INSTR_PLACED)
      return;

   instr->pass_flags |= GCM_INSTR_PLACED;

   if (instr->block == NULL) {
      nir_foreach_def(instr, gcm_replace_def_with_undef, state);
      nir_instr_remove(instr);
      state->progress = true;
      return;
   }

   struct gcm_block_info *block_info = &state->blocks[instr->block->index];
   exec_node_remove(&instr->node);

   if (block_info->last_instr) {
      exec_node_insert_node_before(&block_info->last_instr->node,
                                   &instr->node);
   } else {
      /* Schedule it at the end of the block */
      nir_instr *jump_instr = nir_block_last_instr(instr->block);
      if (jump_instr && jump_instr->type == nir_instr_type_jump) {
         exec_node_insert_node_before(&jump_instr->node, &instr->node);
      } else {
         exec_list_push_tail(&instr->block->instr_list, &instr->node);
      }
   }

   block_info->last_instr = instr;
}

/**
 * Are instructions a and b both contained in the same if/else block?
 */
static bool
weak_gvn(const nir_instr *a, const nir_instr *b)
{
   const struct nir_cf_node *ap = a->block->cf_node.parent;
   const struct nir_cf_node *bp = b->block->cf_node.parent;
   return ap && ap == bp && ap->type == nir_cf_node_if;
}

static bool
opt_gcm_impl(nir_shader *shader, nir_function_impl *impl, bool value_number)
{
   nir_metadata_require(impl, nir_metadata_block_index |
                              nir_metadata_dominance |
                              nir_metadata_dominance_lca);
   nir_metadata_require(impl, nir_metadata_loop_analysis,
                        shader->options->force_indirect_unrolling,
                        shader->options->force_indirect_unrolling_sampler);

   /* A previous pass may have left pass_flags dirty, so clear it all out. */
   nir_foreach_block(block, impl)
      nir_foreach_instr(instr, block)
         instr->pass_flags = 0;

   struct gcm_state state;

   state.impl = impl;
   state.instr = NULL;
   state.progress = false;
   state.second_early_run = false;
   state.has_loop = false;
   state.max_loop_pressure = shader->options->max_gcm_loop_pressure;
   state.divergent_scale = MAX2(shader->options->gcm_divergent_pressure_scale, 1);
   exec_list_make_empty(&state.instrs);
   state.blocks = rzalloc_array(NULL, struct gcm_block_info, impl->num_blocks);

   gcm_build_block_info(&impl->body, &state, NULL, 0, 0, ~0u);

   /* Only loops give us anything to weigh a move against, and computing
    * liveness for a function without any of them is pure overhead.
    */
   if (state.has_loop && state.max_loop_pressure) {
      /* Nothing outside a loop asks what a def costs, so a function without
       * one needs neither of these.  Divergence is only read by the scaled
       * cost model, so a target that compiles at a single width skips it too.
       */
      if (state.divergent_scale > 1)
         nir_metadata_require(impl, nir_metadata_divergence);

      gcm_compute_loop_pressure(&state);
   }

   gcm_pin_instructions(impl, &state);

   state.instr_infos =
      rzalloc_array(NULL, struct gcm_instr_info, state.num_instrs);

   /* Perform (at least some) Global Value Numbering (GVN).
    *
    * We perform full GVN when `value_number' is true.  This can be too
    * aggressive, moving values far away and extending their live ranges,
    * so we don't always want to do it.
    *
    * Otherwise, we perform 'weaker' GVN: if identical ALU instructions appear
    * on both sides of the same if/else block, we allow them to be moved.
    * This cleans up a lot of mess without being -too- aggressive.
    */
   struct set gvn_set;
   nir_instr_set_init(&gvn_set, NULL);

   foreach_list_typed_safe(nir_instr, instr, node, &state.instrs) {
      if (instr->pass_flags & GCM_INSTR_PINNED)
         continue;

      if (nir_instr_set_add_or_rewrite(&gvn_set, instr, NULL, NULL,
                                       value_number ? NULL : weak_gvn)) {
         state.progress = true;
         nir_instr_remove(instr);
      }
   }
   nir_instr_set_fini(&gvn_set);

   /* The second early run needs every instruction to already have an early
    * block so that it can look ahead at the users of an instruction's sources.
    * That first run is pure overhead when the function has no loops, because
    * then the second run can never move anything and just reproduces it.
    */
   if (state.has_loop) {
      foreach_list_typed(nir_instr, instr, node, &state.instrs)
         gcm_schedule_early_instr(instr, &state);

      foreach_list_typed(nir_instr, instr, node, &state.instrs)
         instr->pass_flags &= ~GCM_INSTR_SCHEDULED_EARLY;

      state.second_early_run = true;
   }

   foreach_list_typed(nir_instr, instr, node, &state.instrs)
      gcm_schedule_early_instr(instr, &state);

   foreach_list_typed(nir_instr, instr, node, &state.instrs)
      gcm_schedule_late_instr(instr, &state);

   while (!exec_list_is_empty(&state.instrs)) {
      nir_instr *instr = exec_node_data(nir_instr,
                                        state.instrs.tail_sentinel.prev, node);
      gcm_place_instr(instr, &state);
   }

   ralloc_free(state.blocks);
   ralloc_free(state.instr_infos);

   if (state.progress) {
      nir_progress(true, impl, nir_metadata_control_flow);
   } else {
      nir_progress(true, impl,
                   nir_metadata_control_flow | nir_metadata_loop_analysis |
                   nir_metadata_live_defs);
   }

   return state.progress;
}

bool
nir_opt_gcm(nir_shader *shader, bool value_number)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      progress |= opt_gcm_impl(shader, impl, value_number);
   }

   return progress;
}
