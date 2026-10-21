/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "brw_builder.h"

class mac_test : public brw_shader_pass_test {};

TEST_F(mac_test, interleave)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.MOV(src3, brw_imm_f(4.0));
   bld.MUL(dst0, src0, src1);
   bld.MUL(dst1, src2, src3);
   bld.MAD(dst2, dst0, src2, src3);
   bld.MAD(dst3, dst1, src0, src1);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(src2, brw_imm_f(3.0));
   exp.MOV(src3, brw_imm_f(4.0));
   exp.MUL(dst0, src0, src1);
   exp.MUL(acc0, src2, src3);
   exp.MAD(dst2, dst0, src2, src3);
   exp.MAC(dst3, src0, src1);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, chain)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.MOV(src3, brw_imm_f(4.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src2, src3);
   bld.MAD(dst2, dst1, src0, src1);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(src2, brw_imm_f(3.0));
   exp.MOV(src3, brw_imm_f(4.0));
   exp.MUL(acc0, src0, src1);
   exp.MAC(acc0, src2, src3);
   exp.MAC(dst2, src0, src1);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, chain_gfx9_simd32)
{
   set_gfx_verx10(90);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 32);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, BRW_TYPE_F);
   brw_reg src3 = vgrf(bld, BRW_TYPE_F);

   /* Gfx9/11/12 only have acc0 and acc1. A split SIMD32 instruction would
    * need acc2 and acc3 as well.
    */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.MOV(src3, brw_imm_f(4.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src2, src3);
   bld.MAD(dst2, dst1, src0, src1);

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, chain_gfx125_simd32)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 32);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 32);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, src1, dst0, brw_imm_f(4.0));
   bld.MAD(dst2, src0, dst1, src1);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   /* brw_builder uses fix_3src_operand to prevent accumulators from being
    * used as sources to MAD. Work around this by setting the source after the
    * builder is done.
    */
   brw_inst *inst;

   /* MAC can only be SIMD-split to use acc0 and acc1. Splitting SIMD32 would
    * require acc2 and acc3. Therefore, MAD with acc0 as a source is
    * expected. This would eventually be split into instructions that use acc0
    * and acc2.
    */
   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(src2, brw_imm_f(3.0));
   exp.MUL(acc0, src0, src1);

   inst = exp.MAD(acc0, src1, dst0, brw_imm_f(4.0));
   inst->src[1] = acc0;

   inst = exp.MAD(dst2, src0, dst1, src1);
   inst->src[1] = acc0;

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, chain2)
{
   /* This test exercises a specific bug that existed during development of
    * the pass. When trying to propagate accumulator usage up the expression
    * tree, only src0 was tested.
    */
   set_gfx_verx10(200);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(1.0));
   bld.MOV(src2, brw_imm_f(1.0));
   bld.MOV(src3, brw_imm_f(1.0));

   bld.MUL(dst0, src0, src1);
   bld.MUL(dst1, src0, dst0);
   bld.MAD(dst2, dst1, src2, src3);
   bld.MAD(dst3, dst2, src0, src1);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(1.0));
   exp.MOV(src2, brw_imm_f(1.0));
   exp.MOV(src3, brw_imm_f(1.0));
   exp.MUL(acc0, src0, src1);

   /* The original instruction was constructed so that commutivity of MUL
    * would need to be applied to put acc0 as source 0.
    */
   exp.MUL(acc0, acc0, src0);
   exp.MAC(acc0, src2, src3);
   exp.MAC(dst3, src0, src1);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, imm_src2)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src2, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(src2, brw_imm_f(3.0));
   exp.MUL(acc0, acc0, src1);
   exp.MAC(dst1, src2, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, imm_src1_and_src2)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, brw_imm_f(3.0), brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, acc0, src1);

   brw_inst *inst;

   /* Conversion to MAC would get in the way of constant folding. */
   inst = exp.MAD(dst1, dst0, brw_imm_f(3.0), brw_imm_f(4.0));
   inst->src[0] = acc0;

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, two_uses_null_dst)
{
   /* This tests is for a bug that existed during development of the
    * brw_opt_mac pass. Some incorrect modifications had been made to
    * brw_analysis_def.cpp that resulted in get_use_count(addend->dst) being
    * incorrect.
    */
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   bld.ADD(dst0, src0, src1);
   bld.MUL(dst1, dst0, brw_imm_f(0.5));
   bld.MAD(retype(brw_null_reg(), BRW_TYPE_F), dst0, src2, brw_imm_f(-2.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_mov)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(acc0, acc0);
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_mov_df)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   bld.MOV(src0, brw_imm_df(1.0));
   bld.MOV(src1, brw_imm_df(2.0));
   bld.MAD(dst0, src0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, same_source_reused)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.MAD(dst0, src0, src1, src0);

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_mov_integer_neg)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.MOV(retype(dst0, BRW_TYPE_D),
           negate(retype(src0, BRW_TYPE_D)));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_sel)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD),
           retype(src1, BRW_TYPE_UD))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.SEL(acc0,
           acc0,
           retype(src1, BRW_TYPE_F))
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_i2f)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(retype(dst0, BRW_TYPE_F),
           retype(src0, BRW_TYPE_UD));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(acc0,
           retype(src0, BRW_TYPE_UD));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_f2i)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.MOV(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_F));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, math)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.COS(dst0, src0);
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, BFN)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));
   bld.BFN(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD),
           retype(src1, BRW_TYPE_UD),
           retype(src2, BRW_TYPE_UD),
           0xfe);
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

#define INTEGER_3SRC(opcode, verx10)                            \
   TEST_F(mac_test, opcode)                                     \
   {                                                            \
      if (verx10 != 0)                                          \
         set_gfx_verx10(verx10);                                \
                                                                \
      brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);  \
                                                                \
      brw_reg dst0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst1 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst2 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src1 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src2 = vgrf(bld, BRW_TYPE_F);                     \
                                                                \
      bld.MOV(src0, brw_imm_f(1.0));                            \
      bld.MOV(src1, brw_imm_f(2.0));                            \
      bld. opcode (retype(dst0, BRW_TYPE_UD),                   \
                   retype(src0, BRW_TYPE_UD),                   \
                   retype(src1, BRW_TYPE_UD),                   \
                   retype(src2, BRW_TYPE_UD));                  \
      bld. opcode (retype(dst1, BRW_TYPE_UD),                   \
                   retype(src2, BRW_TYPE_UD),                   \
                   retype(src1, BRW_TYPE_UD),                   \
                   retype(src0, BRW_TYPE_UD));                  \
      bld.MAD(dst2, dst0, dst1, brw_imm_f(4.0));                \
                                                                \
      EXPECT_NO_PROGRESS(brw_opt_mac, bld);                     \
   }

INTEGER_3SRC(ADD3, 125);
INTEGER_3SRC(BFI2, 0);
INTEGER_3SRC(DP4A, 120);
INTEGER_3SRC(MAD,  110);

#define INTEGER_2SRC(opcode, verx10)                            \
   TEST_F(mac_test, opcode)                                     \
   {                                                            \
      if (verx10 != 0)                                          \
         set_gfx_verx10(verx10);                                \
                                                                \
      brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);  \
                                                                \
      brw_reg dst0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst1 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst2 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src1 = vgrf(bld, BRW_TYPE_F);                     \
                                                                \
      bld.MOV(src0, brw_imm_f(1.0));                            \
      bld.MOV(src1, brw_imm_f(2.0));                            \
      bld. opcode (retype(dst0, BRW_TYPE_UD),                   \
                   retype(src0, BRW_TYPE_UD),                   \
                   retype(src1, BRW_TYPE_UD));                  \
      bld. opcode (retype(dst1, BRW_TYPE_UD),                   \
                   retype(src0, BRW_TYPE_UD),                   \
                   retype(src1, BRW_TYPE_UD));                  \
      bld.MAD(dst2, dst0, dst1, brw_imm_f(4.0));                \
                                                                \
      EXPECT_NO_PROGRESS(brw_opt_mac, bld);                     \
   }

INTEGER_2SRC(ADD, 0);
INTEGER_2SRC(AND, 0);
INTEGER_2SRC(ASR, 0);
INTEGER_2SRC(AVG, 0);
INTEGER_2SRC(MUL, 0);
INTEGER_2SRC(OR, 0);
INTEGER_2SRC(ROL, 110);
INTEGER_2SRC(ROR, 110);
INTEGER_2SRC(SHL, 0);
INTEGER_2SRC(SHR, 0);
INTEGER_2SRC(XOR, 0);

#define INTEGER_1SRC(opcode)                                    \
   TEST_F(mac_test, opcode)                                     \
   {                                                            \
      brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);  \
                                                                \
      brw_reg dst0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst1 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg dst2 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src0 = vgrf(bld, BRW_TYPE_F);                     \
      brw_reg src1 = vgrf(bld, BRW_TYPE_F);                     \
                                                                \
      bld.MOV(src0, brw_imm_f(1.0));                            \
      bld.MOV(src1, brw_imm_f(2.0));                            \
      bld. opcode (retype(dst0, BRW_TYPE_UD),                   \
                   retype(src0, BRW_TYPE_UD));                  \
      bld. opcode (retype(dst1, BRW_TYPE_UD),                   \
                   retype(src1, BRW_TYPE_UD));                  \
      bld.MAD(dst2, dst0, dst1, brw_imm_f(4.0));                \
                                                                \
      EXPECT_NO_PROGRESS(brw_opt_mac, bld);                     \
   }

INTEGER_1SRC(BFREV);
INTEGER_1SRC(CBIT);
INTEGER_1SRC(FBH);
INTEGER_1SRC(FBL);
INTEGER_1SRC(LZD);
INTEGER_1SRC(NOT);

TEST_F(mac_test, addend_is_imin)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD),
           retype(src1, BRW_TYPE_UD))
      ->conditional_mod = BRW_CONDITIONAL_L;
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_fmin)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(dst0, src0, src1)
      ->conditional_mod = BRW_CONDITIONAL_L;
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.SEL(acc0, acc0, src1)
      ->conditional_mod = BRW_CONDITIONAL_L;
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_sel_integer_neg)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.CMP(retype(brw_null_reg(), BRW_TYPE_F),
           src0, src1, BRW_CONDITIONAL_Z);
   bld.SEL(retype(dst0, BRW_TYPE_D),
           negate(retype(src0, BRW_TYPE_D)),
           retype(src1, BRW_TYPE_D))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_sel_integer_abs)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.CMP(retype(brw_null_reg(), BRW_TYPE_F),
           src0, src1, BRW_CONDITIONAL_Z);
   bld.SEL(retype(dst0, BRW_TYPE_D),
           brw_abs(retype(src0, BRW_TYPE_D)),
           retype(src1, BRW_TYPE_D))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, acc_write_between)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.MUL(dst0, src0, src1);
   bld.MOV(acc0, brw_imm_f(3.0));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, mulh_between)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.emit(SHADER_OPCODE_MULH,
            dst2,
            retype(src0, BRW_TYPE_UD),
            retype(src1, BRW_TYPE_UD));
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, dword_mul_between)
{
   set_gfx_verx10(110);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);

   /* On Gfx11, this will be lowered to something that uses MACH and the
    * accumulator.
    */
   bld.MUL(dst2, retype(src0, BRW_TYPE_UD), retype(src1, BRW_TYPE_UD));

   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, usub_sat_between)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 8);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);

   /* This will be lowered to something that uses the accumulator. */
   bld.emit(SHADER_OPCODE_USUB_SAT,
            dst2,
            retype(src0, BRW_TYPE_UD),
            retype(src1, BRW_TYPE_UD));

   bld.MAD(dst1, dst0, src0, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, acc_read_after)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));
   bld.MOV(dst2, acc0);

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, acc_read_and_write_after)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));
   bld.MUL(acc0, src0, src1);
   bld.MOV(dst2, acc0);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, src0, src1);
   exp.MAC(dst1, src1, brw_imm_f(4.0));
   exp.MUL(acc0, src0, src1);
   exp.MOV(dst2, acc0);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_of_mul)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, negate(acc0), src1);
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_abs_of_mul)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, negate(brw_abs(src0)), negate(src1));
   bld.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, brw_abs(acc0), brw_abs(src1));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_abs_of_mul)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, negate(brw_abs(src0)), negate(src1));
   bld.MAD(dst1, negate(brw_abs(dst0)), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, negate(brw_abs(acc0)), brw_abs(src1));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_of_add)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.ADD(dst0, src0, src1);
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.ADD(acc0, negate(acc0), negate(src1));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_abs_of_add)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.ADD(dst0, negate(brw_abs(src0)), negate(src1));
   bld.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_neg_abs_of_add)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.ADD(dst0, negate(brw_abs(src0)), negate(src1));
   bld.MAD(dst1, negate(brw_abs(dst0)), src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, addend_is_neg_of_mad)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   bld.MAD(dst0, src0, src1, src2);
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(-1.0));
   exp.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   exp.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   exp.MAC(acc0, negate(src1), src2);
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_abs_of_mad)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   bld.MAD(dst0, brw_abs(src0), negate(brw_abs(src1)), src2);
   bld.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   exp.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   exp.MAC(dst0, negate(brw_abs(src1)), src2);
   exp.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_abs_of_mad)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   bld.MAD(dst0, brw_abs(src0), negate(brw_abs(src1)), src2);
   bld.MAD(dst1, negate(brw_abs(dst0)), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   exp.BFREV(retype(src2, BRW_TYPE_UD), brw_imm_ud(3));
   exp.MAC(dst0, negate(brw_abs(src1)), src2);
   exp.MAD(dst1, negate(brw_abs(dst0)), src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_of_sel)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(dst0, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.SEL(acc0, negate(acc0), negate(src1))
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_abs_of_sel)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(dst0, negate(brw_abs(src0)), negate(src1))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.SEL(acc0, brw_abs(acc0), brw_abs(src1))
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_abs_of_sel)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* This is carefully constructed to (possibly) trigger an assertion
    * failure. When the immediate source of the SEL is negated while
    * converting the MAD to MAC, the source must have already been changed
    * from UD to F.
    */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.SEL(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD),
           retype(brw_imm_f(2.0), BRW_TYPE_UD))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MAD(dst1, negate(brw_abs(dst0)), src0, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   exp.SEL(acc0, negate(brw_abs(src0)), brw_imm_f(-2.0))
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.MAC(dst1, src0, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_of_rndz)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.RNDZ(dst0, src0);
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.RNDZ(acc0, negate(acc0));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_abs_of_rndz)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.RNDZ(dst0, negate(src0));
   bld.MAD(dst1, brw_abs(dst0), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.RNDZ(acc0, brw_abs(acc0));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_abs_of_rndz)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.RNDZ(dst0, negate(src0));
   bld.MAD(dst1, negate(brw_abs(dst0)), src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.RNDZ(acc0, negate(brw_abs(acc0)));
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_frc)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.FRC(dst0, src0);
   bld.MAD(dst1, dst0, src1, brw_imm_f(4.0));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.FRC(acc0, acc0);
   exp.MAC(dst1, src1, brw_imm_f(4.0));

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, addend_is_neg_of_frc)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.FRC(dst0, src0);
   bld.MAD(dst1, negate(dst0), src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, destination_is_acc2)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);
   brw_reg acc2 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   acc2.nr = BRW_ARF_ACCUMULATOR + 2;

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.MUL(dst0, src0, src1);
   bld.MAD(acc2, dst0, src1, brw_imm_f(4.0));

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, skia_1681)
{
   /* Test case mimics some code from shaders/skia/1681.shader_test that was
    * not handled correctly.
    */
   set_gfx_verx10(200);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst3 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst4 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst5 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src2 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MOV(src2, brw_imm_f(3.0));

   bld.MUL(dst0, src0, src1);

   bld.MAD(dst1, src2, src0, src1);
   bld.MUL(dst2, dst1, brw_imm_f(255.0));
   bld.MAD(dst3, brw_imm_f(0.5), dst0, dst2);
   bld.RNDD(dst4, dst3);
   bld.MAD(dst5, brw_imm_f(1.0), dst4, brw_imm_f(0.00392157));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   /* brw_builder uses fix_3src_operand to prevent accumulators from being
    * used as sources to MAD. Work around this by setting the source after the
    * builder is done.
    */
   brw_inst *inst;

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MOV(acc0, brw_imm_f(3.0));

   exp.MUL(dst0, src0, src1);

   exp.MAC(acc0, src0, src1);
   exp.MUL(acc0, acc0, brw_imm_f(255.0));

   inst = exp.MAD(acc0, brw_imm_f(0.5), dst0, dst2);
   inst->src[2] = inst->src[1];
   inst->src[1] = acc0;

   exp.RNDD(acc0, acc0);

   inst = exp.MAD(dst5, brw_imm_f(1.0), dst4, brw_imm_f(0.00392157));
   inst->src[1] = acc0;

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, cmp_to_null)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.MUL(dst0, src0, src1);
   bld.MAD(dst1, dst0, src1, brw_imm_f(3.0));
   bld.CMP(retype(brw_null_reg(), BRW_TYPE_F),
           dst1, brw_imm_f(0.0), BRW_CONDITIONAL_Z);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(acc0, brw_imm_f(1.0));
   exp.MOV(src1, brw_imm_f(2.0));
   exp.MUL(acc0, acc0, src1);
   exp.MAC(acc0, src1, brw_imm_f(3.0));
   exp.CMP(retype(brw_null_reg(), BRW_TYPE_F),
           acc0, brw_imm_f(0.0), BRW_CONDITIONAL_Z);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, sel_to_sel)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg dst2 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg dst3 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(src0, brw_imm_ud(1));
   bld.BFREV(src1, brw_imm_ud(2));
   bld.SEL(dst0, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.SEL(dst1, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.ADD(dst2, dst1, src1);
   bld.SEL(dst3, dst1, dst0)
      ->predicate = BRW_PREDICATE_NORMAL;

   EXPECT_PROGRESS(brw_opt_mac, bld);

   brw_inst *inst;

   exp.BFREV(src0, brw_imm_ud(1));
   exp.BFREV(src1, brw_imm_ud(2));
   exp.SEL(acc0, retype(src0, BRW_TYPE_F), retype(src1, BRW_TYPE_F))
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.SEL(dst1, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.ADD(dst2, dst1, src1);

   inst = exp.SEL(retype(dst3, BRW_TYPE_F), acc0, retype(dst1, BRW_TYPE_F));
   inst->predicate = BRW_PREDICATE_NORMAL;
   inst->predicate_inverse = true;

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, weird_sel_gfx120)
{
   set_gfx_verx10(120);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* On platforms older than Gfx12.5, this should not make progress. The
    * conditional modifier such that the SEL is not commutative, and src1
    * cannot be an accumulator.
    */
   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(dst0, src0, src1)
      ->conditional_mod = BRW_CONDITIONAL_G;
   bld.SEL(dst1, src0, dst0)
      ->conditional_mod = BRW_CONDITIONAL_LE;

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, weird_sel_gfx125)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   bld.MOV(src0, brw_imm_f(1.0));
   bld.MOV(src1, brw_imm_f(2.0));
   bld.SEL(dst0, src0, src1)
      ->conditional_mod = BRW_CONDITIONAL_G;
   bld.SEL(dst1, src0, dst0)
      ->conditional_mod = BRW_CONDITIONAL_LE;

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.MOV(src0, brw_imm_f(1.0));
   exp.MOV(acc0, brw_imm_f(2.0));
   exp.SEL(acc0, src0, acc0)
      ->conditional_mod = BRW_CONDITIONAL_G;
   exp.SEL(dst1, src0, acc0)
      ->conditional_mod = BRW_CONDITIONAL_LE;
}

TEST_F(mac_test, mov_nz_float)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* Sources must be initialized for defs analysis to work. */
   bld.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   bld.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   bld.SEL(retype(dst0, BRW_TYPE_UD),
           retype(src0, BRW_TYPE_UD),
           retype(src1, BRW_TYPE_UD))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MOV(retype(brw_null_reg(), BRW_TYPE_F), dst0)
      ->conditional_mod = BRW_CONDITIONAL_NZ;

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.BFREV(retype(src0, BRW_TYPE_UD), brw_imm_ud(1));
   exp.BFREV(retype(src1, BRW_TYPE_UD), brw_imm_ud(2));
   exp.SEL(acc0, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   exp.MOV(retype(brw_null_reg(), BRW_TYPE_F), acc0)
      ->conditional_mod = BRW_CONDITIONAL_NZ;

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, mov_nz_int)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_UD);
   brw_reg src0 = vgrf(bld, BRW_TYPE_UD);
   brw_reg src1 = vgrf(bld, BRW_TYPE_UD);

   /* Changing the MOV from integer to float would change the semantics of
    * conditional modifier.
    */
   bld.BFREV(src0, brw_imm_ud(1));
   bld.BFREV(src1, brw_imm_ud(2));
   bld.SEL(dst0, src0, src1)
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.MOV(retype(brw_null_reg(), BRW_TYPE_UD), dst0)
      ->conditional_mod = BRW_CONDITIONAL_NZ;

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, mad_dest_gfx9_simd32)
{
   set_gfx_verx10(90);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 32);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* MAD can only write acc0 and acc1 on Gfx9. Splitting SIMD32 is not
    * possible.
    */
   bld.BFREV(src0, brw_imm_ud(1));
   bld.BFREV(src1, brw_imm_ud(2));
   bld.MAD(dst0, src0, src1, brw_imm_f(3.0));
   bld.ADD(dst1, dst0, src1);

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, add_dest_gfx9_simd32)
{
   set_gfx_verx10(90);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 32);

   brw_reg dst0 = vgrf(bld, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, BRW_TYPE_F);
   brw_reg src1 = vgrf(bld, BRW_TYPE_F);

   /* Gfx9/11/12 only have acc0 and acc1. A split SIMD32 instruction would
    * need acc2 and acc3 as well.
    */
   bld.BFREV(src0, brw_imm_ud(1));
   bld.BFREV(src1, brw_imm_ud(2));
   bld.ADD(dst0, src0, src1);
   bld.MUL(dst1, dst0, src1);

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}

TEST_F(mac_test, mov_of_acc)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   bld.BFREV(src0, brw_imm_ud(1));
   bld.ADD(dst0, src0, brw_imm_f(1.0));
   bld.MOV(retype(dst1, BRW_TYPE_UD), retype(dst0, BRW_TYPE_UD));

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.BFREV(src0, brw_imm_ud(1));
   exp.ADD(acc0, src0, brw_imm_f(1.0));
   exp.MOV(dst1, acc0);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, hf_to_bf)
{
   set_gfx_verx10(125);

   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_F);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_BF);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_UD);
   brw_reg acc0 = retype(brw_acc_reg(8 * reg_unit(devinfo)),
                         BRW_TYPE_F);

   /* This sequence mimics the HF to BF converion generated by NIR.
    */
   bld.BFREV(src0, brw_imm_ud(1));
   bld.MOV(dst0, retype(src0, BRW_TYPE_HF));
   bld.MOV(dst1, dst0);

   EXPECT_PROGRESS(brw_opt_mac, bld);

   exp.BFREV(src0, brw_imm_ud(1));
   exp.MOV(acc0, retype(src0, BRW_TYPE_HF));
   exp.MOV(dst1, acc0);

   EXPECT_SHADERS_MATCH(bld, exp);
}

TEST_F(mac_test, sel_neg_d_src1)
{
   brw_builder bld = make_shader(MESA_SHADER_FRAGMENT, 16);
   brw_builder exp = make_shader(MESA_SHADER_FRAGMENT, 16);

   brw_reg dst0 = vgrf(bld, exp, BRW_TYPE_D);
   brw_reg dst1 = vgrf(bld, exp, BRW_TYPE_D);
   brw_reg src0 = vgrf(bld, exp, BRW_TYPE_D);

   /* Sources must be initialized for defs analysis to work. */
   bld.MOV(src0, brw_imm_d(37));
   bld.SEL(retype(dst0, BRW_TYPE_UD),
           brw_imm_ud(2),
           brw_imm_ud(1))
      ->predicate = BRW_PREDICATE_NORMAL;
   bld.SEL(retype(dst1, BRW_TYPE_D),
           retype(dst0, BRW_TYPE_D),
           negate(retype(src0, BRW_TYPE_D)))
      ->predicate = BRW_PREDICATE_NORMAL;

   EXPECT_NO_PROGRESS(brw_opt_mac, bld);
}
