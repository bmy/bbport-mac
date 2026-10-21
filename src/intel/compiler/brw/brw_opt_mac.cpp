/*
 * Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "brw_shader.h"
#include "brw_builder.h"
#include "brw_cfg.h"

/** @file
 * Replace some VGRF with accumulator and some MAD with MAC.
 *
 * The MAC instruction performs a multiply-and-accumulate operation. It is
 * similar to MAD in operation, but there are some key differences.
 *
 * 1. MAC is a two-source instruction, so an immediate value can be a full
 *    32-bit value.
 *
 * 2. The value added to the multiply result always come from accumulator
 *    acc0, acc1, or both. In cases where only a single accumulator is read,
 *    it is specified using QtrCtrl.
 *
 * 3. As a general restriction on accumulator usage, an instruction cannot
 *    read one accumulator and write a different one.
 *
 * 4. As a general restriction on accumulator usage, there is no regioning on
 *    accumulator access, so cases where source and destination regionings
 *    must match can be problematic.
 *
 * On platforms supported by this compiler, there are at least two
 * accumulators. For 32-bit float values, each accumulator holds 8 *
 * reg_unit(devinfo) elements. This means Xe2 platforms can support up to
 * SIMD32 MAC, and previous platforms can support up to SIMD16 MAC.
 *
 * Gfx9/11/12: 2 accumulators
 * Xe/Xe2/Xe3: 4 accumulators
 *
 * Some Xe3 platforms have 8 accumulators, but the 4 additional accumulators
 * are only available in Large GRF mode. Bspec 73583 (r71009) says:
 *
 *    Accumulators registers acc4-acc7 are allowed only in Large GRF mode.
 *
 * This is important: On Gfx12.5 platforms, a SIMD32 MAC cannot be used. Such
 * an instruction would read and write too many registers (more than 2 SIMD8
 * registers). As is common in these situations, the instruction would need to
 * be split into two SIMD16 instructions. The second SIMD16 MAC would have to
 * read from acc2/acc3, but there is no way to specify this. This means care
 * must be taken to not generate instructions that would require such
 * splitting.
 *
 * As a result of the many restrictions on accumulator usage, this
 * optimization pass has to be more aware of platform specifics and access
 * restrictions than most passes. One misstep here can result in cases that
 * SIMD lowering or region lowering cannot fix without adding many
 * instructions.
 *
 * Before the creation of this pass, the accumulator is only used for
 * SHADER_OPCODE_MULH and USUB_SAT / ISUB_SAT. As a result, existing uses of
 * the accumulator are only live within a single block. This assumption
 * greatly simplifies this pass, and this pass will continue this
 * tradition. There is no global, inter-block data flow / liveness analysis
 * for accumulators. Having accumulators only live within a block means that
 * ad hoc, local liveness tracking is sufficient.
 *
 * FINISHME: There is a possible future enhancement. Xe and earlier platforms
 * can write a GRF and an accumulator by using AccWrEn. This could enable the
 * optimization to proceed in additional cases.
 *
 * Accumulator usage in non-MAC instructions has performance benefits that we
 * do not fully understand. The amount of benefit seems to vary by platform.
 *
 * Currently, many passes, such as the scheduler and scoreboarding, do not
 * know about more than one accumulator. Accessing acc3 will interfere with
 * acc0. For this reason, only acc0/acc1 (as a pair) are used.
 *
 * Being too greedy in use of the accumulator leads to problems for the
 * scheduler. Since it is a single shared resource, having many instructions
 * use it can create artificial dependencies between instructions. As a
 * result, this pass tries to find a single sequence of instructions that are
 * already dependent to use the accumulator. The combination of accumulator
 * destinations and MAD immediate values replaced by MAC is used to determine
 * the optimal path.
 *
 * Four passes are made over the instructions in each block:
 *
 * 1. Examine each source of each instruction to determine if that source
 *    could read from an accumulator. If it cannot, mark the instruction
 *    generating that value as not a candidate for writing to an accumulator.
 *
 *    That pass runs back-to-front so that liveness of accumulators (due to
 *    instructions that already read or write accumulators) can be tracked an
 *    considered.
 *
 * 2. \c find_optimal_path: This pass performs two tasks. Each instruction
 *    that was not removed from consideration by the first pass is further
 *    examined to determine whether it violates any other restrictions on
 *    having an accumulator as a destination.
 *
 *    In addition, the sources of the instruction are examined to determine
 *    which is on the optimal path.
 *
 * 3. \c munge_types: MOV and SEL instructions may read and write integer
 *    values. Integer accumulators are a very different beast than floating
 *    point accumulators, so these instruction must have their types converted
 *    to float types.
 *
 * 4. \c apply_replacement: Convert instructions to read and / or write the
 *    appropriate accumulator.
 *
 * FINISHME: 16-bit float types (HF and BF) are not yet supported.
 *
 * FINISHME: If a MAD has immediates for src0 and src2, if the heuristic for
 * src1 is 1, it may be better to add a MOV to load the src0 immediate into
 * the accumulator.
 */

/**
 * Which accumulators are read by the instruction?
 */
static unsigned
accumulators_read(const brw_inst *inst, const intel_device_info *devinfo)
{
   unsigned acc = 0;

   if (inst->reads_accumulator_implicitly()) {
      acc = brw_implicit_accumulator_bits(inst->exec_size, inst->group,
                                          get_exec_type(inst));
   }

   for (unsigned i = 0; i < inst->sources; i++) {
      if (brw_reg_is_arf(inst->src[i], BRW_ARF_ACCUMULATOR)) {
         acc |= brw_explicit_accumulator_bits(inst->exec_size, inst->src[i]);
      }
   }

   return acc;
}

/**
 * Which accumulators are written by the instruction?
 */
static unsigned
accumulators_written(brw_shader &s, const brw_inst *inst)
{
   unsigned acc = 0;

   if (inst->writes_accumulator_implicitly(s.devinfo)) {
      acc = brw_implicit_accumulator_bits(inst->exec_size, inst->group,
                                          get_exec_type(inst));
   }

   switch (inst->opcode) {
   /* DPAS of f16 that is lowered will dirty the accumulator. */
   case BRW_OPCODE_DPAS:
      if (brw_type_is_float(inst->dst.type) && s.compiler->lower_dpas) {
         acc |= brw_explicit_accumulator_bits(inst->exec_size,
                                              retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                                                     BRW_TYPE_HF));
      }
      break;

   /* Dword integer multiply that needs to be lowered will dirty the
    * accumulator.
    */
   case BRW_OPCODE_MUL:
      if ((inst->src[0].type == BRW_TYPE_UD ||
          inst->src[0].type == BRW_TYPE_D) &&
          !s.devinfo->has_integer_dword_mul) {
         acc |= brw_explicit_accumulator_bits(inst->exec_size,
                                              retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                                                     BRW_TYPE_D));
      }
      break;

   /* MULH will be lowered and will dirty the accumulator. */
   case SHADER_OPCODE_MULH:
      acc |= brw_explicit_accumulator_bits(inst->exec_size,
                                           retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                                                  BRW_TYPE_D));
      break;

   /* USUB_SAT and ISUB_SAT that are small enough SIMD get lowered
    * to a version that uses the accumulator.
    */
   case SHADER_OPCODE_USUB_SAT:
   case SHADER_OPCODE_ISUB_SAT:
      if (inst->exec_size == 8 * reg_unit(s.devinfo)) {
         acc |= brw_explicit_accumulator_bits(inst->exec_size,
                                              retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                                                     BRW_TYPE_D));
      }
      break;

   default:
      break;
   }

   if (brw_reg_is_arf(inst->dst, BRW_ARF_ACCUMULATOR))
      acc |= brw_explicit_accumulator_bits(inst->exec_size, inst->dst);

   return acc;
}

static bool
accumulator_access_between(brw_shader &s,
                           const brw_inst *front,
                           const brw_inst *back,
                           unsigned mask)
{
   foreach_inst_in_block_starting_from(brw_inst, inst, front) {
      if (inst == back)
         break;

      if (((accumulators_read(inst, s.devinfo) |
            accumulators_written(s, inst)) & mask) != 0) {
         return true;
      }
   }

   return false;
}

static brw_reg
apply_negate(const brw_reg r)
{
   if (r.file != IMM)
      return negate(r);
   else {
      assert(r.type == BRW_TYPE_F);
      return brw_imm_f(-r.f);
   }
}

static brw_reg
apply_abs(const brw_reg r)
{
   if (r.file != IMM)
      return brw_abs(r);
   else {
      assert(r.type == BRW_TYPE_F);
      return brw_imm_f(fabsf(r.f));
   }
}

/**
 * Can the source modifiers on a register be transfered to the instruction
 * generating the value?
 */
static bool
source_mods_can_reassociate(const brw_reg &reg, const brw_inst *inst,
                            unsigned *abs_count, unsigned *negate_count)
{
   *abs_count = 0;
   *negate_count = 0;

   if (reg.negate) {
      switch (inst->opcode) {
      case BRW_OPCODE_MOV:
      case BRW_OPCODE_MUL:
      case BRW_OPCODE_RNDZ:
         *negate_count = 1;
         break;

      case BRW_OPCODE_ADD:
      case BRW_OPCODE_MAD:
      case BRW_OPCODE_CSEL:
         *negate_count = 2;
         break;

      case BRW_OPCODE_SEL:
         /* FINISHME: This is possible, but MIN has to be converted to
          * MAX and vice versa.
          */
         if (inst->conditional_mod != BRW_CONDITIONAL_NONE)
            return false;

         *negate_count = 2;
         break;

      default:
         /* Cannot push source modifier down into the source instruction. */
         return false;
      }
   }

   if (reg.abs) {
      switch (inst->opcode) {
      case BRW_OPCODE_MOV:
      case BRW_OPCODE_RNDZ:
         *abs_count = 1;
         break;

      case BRW_OPCODE_MUL:
      case BRW_OPCODE_CSEL:
         *abs_count = 2;
         break;

      case BRW_OPCODE_SEL:
         if (inst->conditional_mod != BRW_CONDITIONAL_NONE)
            return false;

         *abs_count = 2;
         break;

      default:
         /* Cannot push source modifier down into the source instruction. */
         return false;
      }
   }

   return true;
}

static bool
can_transform_to_mac(brw_shader &s, const brw_inst *inst, unsigned src,
                     const brw_inst *src_inst, const brw_reg &acc)
{
   unsigned abs_count;
   unsigned negate_count;

   return s.devinfo->verx10 < 350 &&
          inst->opcode == BRW_OPCODE_MAD && src == 0 &&
          (inst->src[1].file != IMM || inst->src[2].file != IMM) &&
          inst->exec_size <= 16 * reg_unit(s.devinfo) &&
          acc.nr == BRW_ARF_ACCUMULATOR &&
          source_mods_can_reassociate(inst->src[0], src_inst,
                                      &abs_count, &negate_count);
}

static brw_inst *
transform_to_mac(brw_shader &s, brw_inst *inst)
{
   if (inst->src[1].file == IMM) {
      inst->src[0] = inst->src[2];
   } else {
      inst->src[0] = inst->src[1];
      inst->src[1] = inst->src[2];
   }

   inst = brw_transform_inst(s, inst, BRW_OPCODE_MAC, 2);

   /* If the instruction was previously
    *
    *    mad(16) null:F, %4:F, %2:F, %3:F    AccWrEn
    *
    * convert it to
    *
    *    mac(16) acc0:F, %2:F, %3:F
    */
   if (inst->writes_accumulator && inst->dst.is_null()) {
      inst->writes_accumulator = false;
      inst->dst = retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                         brw_float_type_for_reg_type(inst->dst.type));
   }

   assert(!brw_reg_is_arf(inst->src[0], BRW_ARF_ACCUMULATOR));
   assert(!brw_reg_is_arf(inst->src[1], BRW_ARF_ACCUMULATOR));

   return inst;
}

static unsigned
isolate_lsb(unsigned x)
{
   return x & -x;
}

static void
preprocess_block(brw_shader &s, const brw_def_analysis &defs, bblock_t *b,
                 unsigned acc_available, bool *flags)
{
   assert(acc_available != 0);

   brw_reg acc = brw_acc_reg(8 * reg_unit(s.devinfo));
   unsigned live_acc = 0;

   acc.nr += ffs(acc_available) - 1;

   foreach_inst_in_block_reverse(brw_inst, inst, b) {
      const unsigned acc_written = accumulators_written(s, inst);
      const unsigned acc_read = accumulators_read(inst, s.devinfo);

      assert(acc_written == 0 || acc_read == 0 || acc_written == acc_read);

      live_acc = (live_acc & ~acc_written) | acc_read;

      /* If the instruction already reads an accumulator, it is not possible
       * to make it read more. This would either require the instruction read
       * two different accumulators or having the instruction that generates
       * one of the sources overwrite the existing accumulator value. Neither
       * is valid.
       */
      if (acc_read)
         continue;

      if (s.devinfo->verx10 >= 125 && inst->dst.type == BRW_TYPE_DF)
         continue;

      for (unsigned i = 0; i < inst->sources; i++) {
         unsigned nr = inst->src[i].nr;
         brw_inst *src_inst = defs.get(inst->src[i]);

         if (src_inst == NULL)
            continue;

         if (src_inst->block != inst->block)
            continue;

         /* FINISHME: Allow HF and BF too. */
         if (brw_type_size_bits(inst->src[i].type) != 32)
            continue;

         if (defs.get_use_count(inst->src[i]) > 1)
            continue;

         if (inst->exec_size != src_inst->exec_size)
            continue;

         /* If the instruction can be modified, its result type either already
          * is or will be modified to be a float type.
          */
         const brw_reg_type t =
            brw_type_with_size(BRW_TYPE_F,
                               brw_type_size_bits(src_inst->dst.type));

         const unsigned acc_to_write =
            brw_explicit_accumulator_bits(src_inst->exec_size, retype(acc, t));

         /* This can fail for several reasons. If acc_to_write requires more
          * accumulators than exist on a platform (e.g., SIMD32 needs 4
          * accumulators, but Gfx12 has only 2). If acc_to_write is acc1, but
          * it would need to also write acc2. Etc.
          */
         if ((acc_to_write & acc_available) != acc_to_write)
            continue;

         if (acc_written != 0) {
            /* The lowest set bit of acc_written and acc_to_write must be the
             * same, or the accumulator access are not compatible. acc_written
             * must also be a proper subset of acc_to_write.
             */
            if (isolate_lsb(acc_written) != isolate_lsb(acc_to_write))
               continue;

            if ((acc_written & acc_to_write) != acc_written)
               continue;
         }

         if ((live_acc & acc_to_write) != 0) {
            continue;
         }

         /* Opcodes in this switch statement are grouped by number of
          * sources. The majority of platform-dependent restrictions are based
          * on the number of sources of the instruction.
          */
         switch (inst->opcode) {
         case BRW_OPCODE_FRC:
         case BRW_OPCODE_RNDD:
         case BRW_OPCODE_RNDE:
         case BRW_OPCODE_RNDU:
         case BRW_OPCODE_RNDZ:
            break;

         case BRW_OPCODE_MOV:
            if (brw_type_is_int(inst->src[i].type)) {
               if (inst->src[i].abs || inst->src[i].negate)
                  continue;

               if (inst->conditional_mod != BRW_CONDITIONAL_NONE)
                  continue;

               /* Page 493 (page 497 of the PDF), section "Accumulator
                * Restrictions," of the Tiger Lake PRM Volume 9: Render Engine
                * says:
                *
                *    6. INT2FLT format conversion cannot use accumulator as
                *       source.
                *
                * This is a new restriction with Gfx12. Integer accumulator
                * access is very different than floating point accumulator
                * access, and this pass does not deal with that.
                *
                * The requirement here is that when the source is an integer,
                * there is no type conversions at all. This enables the
                * optimization pass to modify the source and the destination
                * to be float types.
                */
               if (inst->dst.type != inst->src[i].type || inst->saturate)
                  continue;

            } else if (s.devinfo->verx10 == 90 && inst->dst.type == BRW_TYPE_DF) {
               /* No supporting information could be found in the PRMs or the
                * Bspec, but it has been experimentally determined that
                *
                *    mov(8)          g2<1>DF         acc0<8,8,1>F
                *
                * does not work correctly on Gfx9.
                */
               continue;
            } else if (brw_type_is_int(inst->dst.type)) {
               continue;
            }
            break;

         case BRW_OPCODE_ADD:
         case BRW_OPCODE_MUL:
            /* Bspec 2995 (r151198) says:
             *
             *    Accumulator registers may be accessed explicitly as src0
             *    operands only.
             *
             * Strangely, page 743 (page 777 of the PDF), section "Accumulator
             * Registers Summary," of the Skylake PRM Volume 7: 3D-Media-GPGPU
             * lists this restirction under the heading, "Limits on SIMD16
             * Float Operations."
             *
             * We'll take the conservative approach of assuming the
             * restriction applies in SIMD8 as well.
             *
             * It's something of a moot point as floating point ADD and MUL
             * are commutative.
             */
            if (!brw_type_is_float(inst->src[i].type))
               continue;

            break;

         case BRW_OPCODE_CMP:
         case BRW_OPCODE_CMPN:
            /* CMPN is not commutative. CMP is commutative, but this it not
             * currently handled.
             *
             * Bspec 47251 (r48459) says for [ACM, ACMPLUS, ATS, PVC, RLT,
             * MAR, MTL, ARL]:
             *
             *    Accumulator registers may be accessed explicitly on src0 and
             *    src1 operand.
             */
            if (s.devinfo->verx10 < 125 && i > 0)
               continue;

            if (!brw_type_is_float(inst->src[i].type))
               continue;

            break;

         case BRW_OPCODE_SEL:
            if (brw_type_is_float(inst->src[i].type)) {
               if (s.devinfo->verx10 < 125 &&
                   i != 0 &&
                   inst->predicate == BRW_PREDICATE_NONE &&
                   !inst->is_commutative()) {
                  continue;
               }
            } else {
               if (inst->predicate != BRW_PREDICATE_NONE) {
                  /* A predicated SEL is a glorified MOV, and it will inherit the
                   * same restirctions as MOV.
                   *
                   * Restrictions on which sources can be accumulators are
                   * ignored because the predicated SEL is commutative.
                   *
                   * Both sources have to be checked here. If src0 is okay but
                   * src1 is not, the flag be incorrectly set. See unit test
                   * sel_neg_d_src1.
                   */
                  if (inst->src[0].abs || inst->src[0].negate ||
                      inst->src[1].abs || inst->src[1].negate)
                     continue;

                  if (inst->dst.type != inst->src[0].type ||
                      inst->dst.type != inst->src[1].type ||
                      inst->saturate)
                     continue;

               } else {
                  /* Integer MIN / MAX operations cannot be modified to have
                   * float sources.
                   *
                   * Restrictions on which sources can be accumulators are
                   * ignored because MIN and MAX are commutative.
                   */
                  continue;
               }
            }
            break;

         case BRW_OPCODE_MAD: {
            /* Bspec 3122 (r155755) says for [SNB, IVB, VLV, VLVT, HSW, BDW,
             * CHV, SKL, BXT, KBL, KBLH, GLK, GLV, CFL, WHL, AML, CML]:
             *
             *    5. Instructions with three source operands cannot use
             *       explicit accumulator operands. AccWrEn may be allowed for
             *       implicitly updating the accumulator.
             *
             * It also says for [BDW, CHV, SKL, BXT, KBL, KBLH, GLK, GLV, CFL,
             * WHL, AML, CML, CNL, ICLLP, ICLHP, JSL, LKF1, LKFR]:
             *
             *    Integer source operands [for MAD] cannot be accumulators.
             *
             * Page 296 (page 314 of the PDF), section
             * "EU_INSTRUCTION_ALIGN1_THREE_SRC," of the Ice Lake PRM Volume
             * 2d: Command Reference: Structures list the source 0 and source
             * 2 register file values as being either GRF or IMM. Source 1 is
             * listed as either GRF or ARF with the note
             *
             *    Selects Architectural Register File as source 1. Only
             *    Accumulator is allowed.
             *
             * The documentation for the MAD instruction on Ice Lake shows
             * examples using acc0 as src0 to replace instructions that were
             * removed on that platform.
             *
             * Bspec 47367 (r63640) for Xe says:
             *
             *    3. Src2 cannot use accumulator for 3 source instructions.
             *
             * src1 and src2 are commutative for MAD, so both will be allowed
             * on the assumption that other code will fix them up.
             */
            unsigned a, b;

            bool can_reassociate = source_mods_can_reassociate(inst->src[i],
                                                               src_inst,
                                                               &a, &b);
            unsigned implicit_bits =
               brw_implicit_accumulator_bits(inst->exec_size,
                                             inst->group,
                                             inst->dst.type);

            /* The implicit accumulator source of of MAC can't have any source
             * modifiers applied. Therefore they need to be reassociated to
             * the instruction that generates the accumulator.
             */
            if (s.devinfo->verx10 < 350 &&
                (acc_to_write & implicit_bits) == implicit_bits &&
                i == 0 && inst->exec_size <= 16 * reg_unit(s.devinfo) &&
                can_reassociate) {
               /* This can be a MAC, break out of the switch. */
               break;
            }

            /* Gfx9 cannot have accumulator as any source of MAD. If this
             * cannot be converted to MAC, no progress can be made.
             */
            if (s.devinfo->verx10 == 90)
               continue;

            /* For Gfx11, the Bspec says src0 and src1 can be an
             * accumulator. src0 requires the src0 type to be NF.  NF is not
             * supported by this compiler, so this encoding is not
             * possible.
             */
            if (s.devinfo->verx10 == 110 && i == 0)
               continue;

            if (s.devinfo->verx10 <= 120) {
               /* No supporting documentation has been found in the Bspec or
                * the PRMs. However, the TGL simulator will produce the warning:
                *
                *    Source modifier is not allowed if source is an accumulator
                *    for 3 src instructions.
                *
                * Source modifiers have to either be moved to the instruction
                * generating the accumulator value or to the other
                * multiplicand.
                *
                * It has been experimentally determined (via
                * dEQP-VK.spirv_assembly.instruction.graphics.float_controls2.fp32.input_args.refract_testedWithout_NSZ_arg1_minusZero_arg2_one_res_minusZero_deco_frag)
                * that Gfx11 has the same restriction.
                */
               if (!can_reassociate && (i == 0 || inst->src[i].abs))
                  continue;
            }
            break;
         }

         case BRW_OPCODE_CSEL:
            /* CSEL behaves very similarly to MAD by virtue of it being a
             * 3-source instruction. It does not have the luxury of
             * transformation to a 2-source instruction (e.g., MAD to MAC
             * conversion) or commutative arguments. As a result, various
             * restrictions must be more strictly enforced.
             *
             * For Gfx11, the Bspec says src0 and src1 can be an
             * accumulator. src0 requires the src0 type to be NF.  NF is not
             * supported by this compiler, so this encoding is not
             * possible.
             */
            if (s.devinfo->verx10 == 90) {
               continue;
            } else if (s.devinfo->verx10 == 110) {
               if (i != 1)
                  continue;
            } else if (s.devinfo->verx10 < 200) {
               if (i > 1)
                  continue;
            }

            /* No supporting documentation has been found in the Bspec or the
             * PRMs. However, the TGL simulator will produce the warning:
             *
             *    Source modifier is not allowed if source is an accumulator
             *    for 3 src instructions.
             *
             * It has been experimentally determined that Gfx11 has the same
             * restriction.
             *
             * FINISHME: Use source_mods_can_reassociate here.
             */
            if (s.devinfo->verx10 <= 120 &&
                (inst->src[i].negate || inst->src[i].abs)) {
                  continue;
            }
            break;

         default:
            continue;
         }

         if (!accumulator_access_between(s, src_inst, inst, acc_to_write)) {
            flags[nr] = true;
         }
      }
   }
}

struct path {
   brw_inst *inst;

   /**
    * Link to path node for instruction generating source of this instruction.
    */
   path *next;

   /**
    * Number of instructions whose destination is replaced with accumulator.
    */
   unsigned num_replacements;

   /**
    * Number of MAD immediate sources eliminated by MAC conversion
    *
    * For the most part, MAD instructions cannot have float immediate sources.
    * Replacing src0 with the accumulator and converting the instruction to
    * MAC means an immediate doesn't need to be explicitly loaded into a
    * register.
    */
   unsigned num_mad_immediates;

   /** Number of instructions in the path */
   unsigned length;

   /**
    * Which source is on the optimal path?
    *
    * This source may not be replaced by an accumulator. It will be on the
    * dependency path. A value of -1 means it is the first instruction of the
    * path. This also implies ::next will be \c NULL.
    */
   int8_t src;

   /**
    * Should the destination of this instruction be replaced with accumulator?
    */
   bool replace_destination;

   /**
    * Number of other paths that point to this path (via ::next).
    *
    * While only defs that have a single use can be converted to write an
    * accumulator, other instructions can be "inert" nodes in long dependency
    * chains. The full set of dependent instructions is a DAG.
    *
    * This count is used to trim unused portions from the end of a path. See
    * the end of \c find_optimal_path.
    */
   unsigned num_paths;

   void init(brw_inst *_inst, path *_next, int _src)
   {
      inst = _inst;
      next = _next;
      src = _src;
      replace_destination = false;
      num_paths = 0;

      if (next != nullptr) {
         num_replacements = next->num_replacements;
         num_mad_immediates = next->num_mad_immediates;
         length = next->length + 1;
         next->num_paths++;
      } else {
         num_replacements = 0;
         num_mad_immediates = 0;
         length = 1;
      }
   }

   path *remove()
   {
      path *n = next;

      assert(num_paths == 0);

      if (next != nullptr) {
         assert(next->num_paths > 0);
         next->num_paths--;
      }

      inst = nullptr;
      next = nullptr;
      src = -1;
      replace_destination = false;
      num_replacements = 0;
      num_mad_immediates = 0;
      length = 0;

      return n;
   }
};

static path *
get_path_for_inst(path *all_paths, unsigned num_paths, const brw_inst *inst)
{
   if (num_paths == 0)
      return nullptr;

   unsigned i = num_paths;
   do {
      i--;
      if (all_paths[i].inst == inst)
         return &all_paths[i];
   } while (i != 0);

   return nullptr;
}

static int
heuristic(path *a, path *b, unsigned mode = 0)
{
   if (a->num_replacements == 0)
      return -1;

   if (b == nullptr)
      return 1;

   /* Based on experiments in the Intel perf CI, IRL and LIR seem to have the
    * best perfomance over all with no statistical difference between them. LRI
    * was bettern in some benchmarks (e.g., Witcher3 and Shadow of the Tomb
    * Raider), but it was worse in others (e.g., Cyberpunk and Total War:
    * Warhammer 3).
    *
    * IRL is the default.
    */
   uint8_t order[6][3] = {
      { 0, 1, 2 }, /* IRL */
      { 0, 2, 1 }, /* ILR */
      { 1, 2, 0 }, /* RLI */
      { 1, 0, 2 }, /* RIL */
      { 2, 0, 1 }, /* LIR */
      { 2, 1, 0 }, /* LRI */
   };

   for (unsigned i = 0; i < 3; i++) {
      switch (order[mode % 6][i]) {
      case 0:
         if (a->num_mad_immediates != b->num_mad_immediates)
            return a->num_mad_immediates - b->num_mad_immediates;
         break;

      case 1:
         if (a->num_replacements != b->num_replacements)
            return a->num_replacements - b->num_replacements;
         break;

      case 2:
         if (a->length != b->length)
            return a->length - b->length;
         break;
      }
   }

   /* If they are equal but a extends b, return a. */
   return a->next == b ? 1 : -1;
}

static path *
find_optimal_path(brw_shader &s, const brw_def_analysis &defs, bblock_t *b,
                  unsigned acc_to_write, bool *flags, path *all_paths)
{
   path *best = nullptr;
   unsigned idx = 0;

   foreach_inst_in_block(brw_inst, inst, b) {
      path *best_inst = nullptr;
      int best_src = -1;

      for (unsigned i = 0; i < inst->sources; i++) {
         brw_inst *src_inst = defs.get(inst->src[i]);

         if (src_inst == NULL)
            continue;

         if (src_inst->block != inst->block)
            continue;

         path *src_path = get_path_for_inst(all_paths, idx, src_inst);

         if (src_path != nullptr) {
            int cmp = heuristic(src_path, best_inst);

            if (cmp > 0 || (cmp == 0 && src_path == best)) {
               best_inst = src_path;
               best_src = i;
            }
         }
      }

      all_paths[idx].init(inst, best_inst, best_src);

      /* If this instruction can write the accumulator, update
       * num_replacements. If it is a MAD that would have an immediate source
       * eliminated, update num_mad_immediates.
       */
      if (defs.get(inst->dst) && flags[inst->dst.nr] &&
          /* 3-source instructions can only write accumulator via AccWrEn. It
           * is not possible to split a SIMD32 instruction that uses AccWrEn.
           */
          !(s.devinfo->ver <= 11 && inst->exec_size == 32 &&
            inst->is_3src(s.compiler)) &&
          /* FINISHME: Allow HF and BF too. */
          brw_type_size_bits(inst->dst.type) == 32) {
          bool replace_destination = false;

         switch (inst->opcode) {
         case BRW_OPCODE_FRC:
         case BRW_OPCODE_RNDD:
         case BRW_OPCODE_RNDE:
         case BRW_OPCODE_RNDU:
         case BRW_OPCODE_RNDZ:
            replace_destination = true;
            break;

         case BRW_OPCODE_MOV:
            if (brw_type_size_bits(inst->src[0].type) > 32)
               break;

            if (brw_type_is_float(inst->dst.type) ||
                (inst->dst.type == inst->src[0].type &&
                 !inst->src[0].abs &&
                 !inst->src[0].negate &&
                 !inst->saturate)) {
               replace_destination = true;
            }

            break;

         case BRW_OPCODE_ADD:
         case BRW_OPCODE_MUL:
            if (brw_type_is_float(inst->dst.type))
               replace_destination = true;

            break;

         case BRW_OPCODE_CMP:
         case BRW_OPCODE_CMPN:
            /* While CMP and CMPN can be allowed to have accumulator sources,
             * an accumulator destination seems perilous. Every user of the
             * result is likely to be an integer instruction, and float and
             * integer accumulators are different.
             */
            break;

         case BRW_OPCODE_SEL:
            if (brw_type_is_float(inst->dst.type)) {
               replace_destination = true;
            } else if (!inst->saturate &&
                       inst->conditional_mod == BRW_CONDITIONAL_NONE) {
               bool all_match = true;

               for (unsigned i = 0; i < 2; i++) {
                  if (inst->dst.type != inst->src[i].type ||
                      inst->src[i].abs ||
                      inst->src[i].negate) {
                     all_match = false;
                     break;
                  }
               }

               if (all_match)
                  replace_destination = true;
            }
            break;

         case BRW_OPCODE_MAD:
            if (brw_type_is_float(inst->dst.type)) {
               unsigned implicit_bits =
                  brw_implicit_accumulator_bits(inst->exec_size,
                                                inst->group,
                                                inst->dst.type);
               bool can_use_accwren =
                  (acc_to_write & implicit_bits) == implicit_bits;

               /* Gfx9 can only write acc0 by using AccWrEn or converting the
                * MAD to MAC.
                */
               if (s.devinfo->verx10 > 90 || can_use_accwren)
                  replace_destination = true;

               /* The instruction is a MAD that will be converted to MAC. If
                * one of the multiplicands is an immediate, the conversion to
                * MAC means it won't need to be loaded into a register.
                */
               if (best_src == 0 && can_use_accwren &&
                   (inst->src[1].file == IMM || inst->src[2].file == IMM)) {
                  all_paths[idx].num_mad_immediates++;
               }
            }
            break;

         case BRW_OPCODE_CSEL: {
            if (!brw_type_is_float(inst->dst.type))
               break;

            unsigned implicit_bits =
               brw_implicit_accumulator_bits(inst->exec_size,
                                             inst->group,
                                             inst->dst.type);

            /* Gfx9 can only write acc0 by using AccWrEn. This means the
             * accumulator written must be the one that would be selected by
             * the instruction's group.
             */
            if (s.devinfo->verx10 > 90 ||
                (acc_to_write & implicit_bits) == implicit_bits) {
               replace_destination = true;
            }

            break;
         }

         default:
            break;
         }

         all_paths[idx].replace_destination = replace_destination;

         if (replace_destination)
            all_paths[idx].num_replacements++;
      }

      if (heuristic(&all_paths[idx], best) > 0)
         best = &all_paths[idx];

      idx++;
   }

   /* Trim off tails that do not have any reads or writes of the accumulator.
    */
   for (int i = idx - 1; i >= 0; i--) {
      path *p = &all_paths[i];

      while (p && p->num_paths == 0) {
         if (p->next != nullptr && !p->next->replace_destination) {
            p = p->remove();
         } else if (p->next == nullptr && p->replace_destination) {
            /* This can occur if the user of this result selected a different
             * source instruction as the best path.
             */
            p = p->remove();
         } else {
            break;
         }
      }
   }

   best = nullptr;
   for (int i = idx - 1; i >= 0; i--) {
      if (all_paths[i].num_paths == 0 && heuristic(&all_paths[i], best) > 0)
         best = &all_paths[i];
   }

   return best;
}

/**
 * Adjust types of MOV and SEL instructions to match float types used by acc.
 *
 * Type adjustment has to be performed after the optimal sequence is selected
 * but before accumulator substitution begins. This is due to sequences like
 *
 *    (+f0.0) sel    g37:UD    ...:UD    ...:UD
 *            mad    ...       -|g37:F|
 *
 * In this case, the -|...| needs to be applied to the sources of the SEL, but
 * those sources have to be :F before that can happen.
 */
static void
munge_types(path *curr)
{
   brw_inst *inst = curr->inst;

   if (inst->opcode == BRW_OPCODE_MOV || inst->opcode == BRW_OPCODE_SEL) {
      brw_reg_type t;

      if (curr->replace_destination && brw_type_is_int(inst->dst.type)) {
         ASSERTED const brw_reg_type dst_old_type = inst->dst.type;

         t = brw_type_with_size(BRW_TYPE_F,
                                brw_type_size_bits(inst->dst.type));

         inst->dst = retype(inst->dst, t);

         for (unsigned i = 0; i < inst->sources; i++) {
            assert(dst_old_type == inst->src[i].type);
            inst->src[i] = retype(inst->src[i], t);
         }
      }

      if (curr->next && curr->next->replace_destination) {
         t = curr->next->inst->dst.type;

         if (brw_type_is_int(t))
            t = brw_type_with_size(BRW_TYPE_F, brw_type_size_bits(t));

         if (inst->dst.type == inst->src[0].type)
            inst->dst = retype(inst->dst, t);

         for (unsigned i = 0; i < inst->sources; i++) {
            assert(!(brw_type_is_int(inst->src[i].type) && (inst->src[i].abs ||
                                                            inst->src[i].negate)));
            assert(brw_type_size_bits(inst->src[i].type) ==
                   brw_type_size_bits(t));
            inst->src[i] = retype(inst->src[i], t);
         }
      }
   }

   if (curr->next != nullptr)
      munge_types(curr->next);
}

static void
apply_replacement(brw_shader &s, const brw_def_analysis &defs, bblock_t *b,
                  unsigned acc_to_write, path *curr)
{
   brw_inst *inst = curr->inst;
   brw_reg acc;

   /* This avoids assertion failures in brw_type_with_size when dst.type is,
    * for example, int8.
    */
   if (curr->replace_destination ||
       (curr->next != nullptr && curr->next->replace_destination)) {
      const brw_reg_type ref = curr->replace_destination ?
                               inst->dst.type : inst->src[curr->src].type;

      acc = retype(brw_acc_reg(8 * reg_unit(s.devinfo)),
                   brw_float_type_for_reg_type(ref));

      acc.nr += ffs(acc_to_write) - 1;
   }

   if (curr->replace_destination) {
      /* For pre-CNL, the Bspec says:
       *
       *    No explicit accumulator access because this is a three-source
       *    instruction. AccWrEn is allowed for implicitly updating the
       *    accumulator.
       *
       * Page 475 (page 481 of the PDF) of the Ice Lake PRM Volume
       * 9: Render Engine says:
       *
       *    The 3-source instructions have the following
       *    restrictions:
       *
       *    - Only GRF registers can be sources and only GRF
       *      registers can be the destination.
       */
      if (inst->is_3src(s.compiler) && s.devinfo->ver <= 11) {
         inst->writes_accumulator = true;
         inst->dst = retype(brw_null_reg(), acc.type);
      } else {
         inst->dst = acc;
      }
   }

   if (curr->next != nullptr && curr->next->replace_destination) {
      brw_inst *src_inst = defs.get(inst->src[curr->src]);

      /* MAC cannot have source modifiers on the (implicit)
       * accumulator. 3-source instructions on Gfx12 and Gfx11 cannot have
       * source modifiers on the accumulator. Fix that up first.
       */
      if (curr->src >= 0 &&
          (s.devinfo->verx10 <= 120 || curr->src == 0) &&
          inst->opcode == BRW_OPCODE_MAD) {
         unsigned abs_count;
         unsigned negate_count;

         if (curr->src > 0 && inst->src[curr->src].negate &&
             !inst->src[curr->src].abs) {
            /* A negation of src1 or src2 can just be moved to the other
             * source.
             */
            const unsigned other = curr->src == 1 ? 2 : 1;

            inst->src[curr->src].negate = false;
            inst->src[other] = apply_negate(inst->src[other]);
         } else if (source_mods_can_reassociate(inst->src[curr->src], src_inst,
                                                &abs_count, &negate_count)) {
            assert(abs_count == 0 || abs_count == src_inst->sources);

            for (unsigned i = 0; i < abs_count; i++)
               src_inst->src[i] = apply_abs(src_inst->src[i]);

            for (unsigned i = 0; i < negate_count; i++)
               src_inst->src[i] = apply_negate(src_inst->src[i]);

            inst->src[curr->src].negate = false;
            inst->src[curr->src].abs = false;
         } else {
            assert(s.devinfo->verx10 > 120);
         }
      }

      if (can_transform_to_mac(s, inst, curr->src, src_inst, acc)) {
         transform_to_mac(s, inst);
      } else if (curr->src >= 0) {
         acc.abs = inst->src[curr->src].abs;
         acc.negate = inst->src[curr->src].negate;

         inst->src[curr->src] = acc;

         if (inst->opcode == BRW_OPCODE_SEL) {
            if (!brw_type_is_float(inst->dst.type))
               inst->dst = retype(inst->dst, acc.type);

            if (!brw_type_is_float(inst->src[1 - curr->src].type)) {
               inst->src[1 - curr->src] =
                  retype(inst->src[1 - curr->src], acc.type);
            }
         }

         if (curr->src == 1) {
            if (inst->is_commutative()) {
               SWAP(inst->src[0], inst->src[1]);
            } else if (inst->opcode == BRW_OPCODE_SEL &&
                       inst->predicate != BRW_PREDICATE_NONE) {
               SWAP(inst->src[0], inst->src[1]);
               inst->predicate_inverse = !inst->predicate_inverse;
            } else {
               assert(s.devinfo->verx10 >= 110);
            }
         } else if (curr->src == 2 && inst->opcode == BRW_OPCODE_MAD) {
            SWAP(inst->src[1], inst->src[2]);
         }
      }
   }

   if (curr->next != nullptr)
      apply_replacement(s, defs, b, acc_to_write, curr->next);
}

bool
brw_opt_mac(brw_shader &s)
{
   /* Number of SIMD8 accumulator slots. */
   unsigned num_accumulators;

   if (s.devinfo->verx10 < 125) {
      num_accumulators = 2;
   } else if (s.devinfo->verx10 < 200) {
      num_accumulators = 4;
   } else {
      num_accumulators = 8;
   }

   unsigned acc_nr = 0;

   unsigned acc;
   if (acc_nr % 4 == 0) {
      acc = 0xf << acc_nr;
   } else if (acc_nr % 2 == 0) {
      acc = 0x3 << acc_nr;
   } else {
      acc = 0x1 << acc_nr;
   }

   acc &= (1U << num_accumulators) - 1;

   const brw_def_analysis &defs = s.def_analysis.require();
   bool progress = false;
   bool *flags = NULL;
   path *all_paths = NULL;

   flags = new bool[defs.count()];
   if (flags == NULL)
      return false;

   all_paths = new path[s.cfg->total_instructions];
   if (all_paths == NULL) {
      delete [] flags;
      return false;
   }

   foreach_block(block, s.cfg) {
      memset(flags, 0, sizeof(*flags) * defs.count());
      memset(all_paths, 0, sizeof(*all_paths) * s.cfg->total_instructions);

      /* The preprocess pass determines which instructions could be allowed to
       * write the accumulator.
       */
      preprocess_block(s, defs, block, acc, flags);

      /* A second pass to find all the possible
       */
      path *best = find_optimal_path(s, defs, block, acc, flags, all_paths);
      if (best != nullptr) {
         munge_types(best);
         apply_replacement(s, defs, block, acc, best);
         progress = true;
      }
   }

   delete [] flags;
   delete [] all_paths;

   if (progress)
      s.invalidate_analysis(BRW_DEPENDENCY_INSTRUCTIONS |
                            BRW_DEPENDENCY_VARIABLES);
   return progress;
}
