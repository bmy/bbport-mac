/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "brw_shader.h"
#include "brw_builder.h"

static brw_reg
build_ex_desc(const brw_builder &bld, unsigned reg_size, bool unspill)
{
   /* Use a different area of the address register than what is used in
    * brw_lower_logical_sends.c (brw_address_reg(2)) so we don't have
    * interactions between the spill/fill instructions and the other send
    * messages.
    */
   brw_reg ex_desc = bld.vaddr(BRW_TYPE_UD,
                               BRW_ADDRESS_SUBREG_INDIRECT_SPILL_DESC);

   brw_builder ubld = bld.uniform();

   ubld.AND(ex_desc,
            retype(brw_vec1_grf(0, 5), BRW_TYPE_UD),
            brw_imm_ud(INTEL_MASK(31, 10)));

   const intel_device_info *devinfo = bld.shader->devinfo;
   if (devinfo->ver >= 20 || intel_has_extended_bindless(devinfo)) {
      ubld.SHR(ex_desc, ex_desc, brw_imm_ud(4));
   } else {
      if (unspill) {
         ubld.OR(ex_desc, ex_desc, brw_imm_ud(GEN_SFID_UGM));
      } else {
         ubld.OR(ex_desc,
                 ex_desc,
                 brw_imm_ud(brw_message_ex_desc(devinfo, reg_size) | GEN_SFID_UGM));
      }
   }

   return ex_desc;
}

static void
encode_const_offset(brw_send_inst *inst,
                    const intel_device_info *devinfo,
                    enum lsc_opcode op,
                    unsigned const_offset)
{
   if (const_offset == 0)
      return;

   gen_lsc_ex_desc ex_desc = {};
   ex_desc.addr_type = LSC_ADDR_SURFTYPE_SS;
   ex_desc.surface_state.base_offset = const_offset;
   gen_lsc_ex_desc_encode(devinfo, op, &ex_desc, &inst->offset);
   inst->ex_desc_imm = true;
}

static void
brw_lower_lsc_fill(const intel_device_info *devinfo, brw_shader &s,
                   brw_scratch_inst *inst)
{
   assert(devinfo->verx10 >= 125);

   const brw_builder bld(inst);
   brw_reg dst = inst->dst;
   brw_reg offset = inst->src[FILL_SRC_PAYLOAD1];

   const unsigned reg_size = inst->dst.component_size(inst->exec_size) /
                             REG_SIZE;
   brw_reg ex_desc = build_ex_desc(bld, reg_size, true);

   /* LSC is limited to SIMD16 (SIMD32 on Xe2) load/store but we can
    * load more using transpose messages.
    */
   const bool use_transpose = inst->use_transpose;
   const brw_builder ubld = use_transpose ? bld.uniform() : bld;

   uint32_t desc = lsc_msg_desc(devinfo, LSC_OP_LOAD,
                                LSC_ADDR_SURFTYPE_SS,
                                LSC_ADDR_SIZE_A32,
                                LSC_DATA_SIZE_D32,
                                use_transpose ? reg_size * 8 : 1 /* num_channels */,
                                use_transpose,
                                LSC_CACHE(devinfo, LOAD, L1STATE_L3MOCS));


   brw_send_inst *unspill_inst = ubld.SEND();
   unspill_inst->dst = dst;

   unspill_inst->src[SEND_SRC_DESC] = brw_imm_ud(0);
   unspill_inst->src[SEND_SRC_EX_DESC] = ex_desc;
   unspill_inst->src[SEND_SRC_PAYLOAD1] = offset;
   unspill_inst->src[SEND_SRC_PAYLOAD2] = brw_reg();

   unspill_inst->sfid = GEN_SFID_UGM;
   unspill_inst->header_size = 0;
   unspill_inst->mlen = brw_lsc_msg_addr_len(devinfo, LSC_ADDR_SIZE_A32,
                                             unspill_inst->exec_size);
   unspill_inst->ex_mlen = 0;
   unspill_inst->size_written =
      brw_lsc_msg_dest_len(devinfo, LSC_DATA_SIZE_D32, bld.dispatch_width()) * REG_SIZE;
   unspill_inst->has_side_effects = false;
   unspill_inst->is_volatile = true;
   unspill_inst->bindless_surface = true;

   unspill_inst->src[0] = brw_imm_ud(
      desc |
      brw_message_desc(devinfo,
                       unspill_inst->mlen,
                       unspill_inst->size_written / REG_SIZE,
                       unspill_inst->header_size));

   if (inst->use_base_offset)
      encode_const_offset(unspill_inst, devinfo, LSC_OP_LOAD, inst->offset);

   assert(unspill_inst->size_written == inst->size_written);
   assert(unspill_inst->size_read(devinfo, SEND_SRC_PAYLOAD1) == inst->size_read(devinfo, FILL_SRC_PAYLOAD1));

   inst->remove();
}

static void
brw_lower_lsc64_fill(const intel_device_info *devinfo, brw_shader &s,
                     brw_scratch_inst *inst)
{
   assert(devinfo->verx10 >= 350);

   const brw_builder bld(inst);
   brw_reg dst = inst->dst;
   brw_reg offset = inst->src[FILL_SRC_PAYLOAD1];

   const unsigned reg_size = inst->dst.component_size(inst->exec_size) /
                             REG_SIZE;

   /* LSC is limited to SIMD16 (SIMD32 on Xe2) load/store but we can
    * load more using transpose messages.
    */
   const bool use_transpose = inst->use_transpose;
   const brw_builder ubld = use_transpose ? bld.uniform() : bld;

   brw_send_inst *unspill_inst = ubld.SEND();
   unspill_inst->combined_desc = lsc_64bit_msg_desc(devinfo,
                                                    GEN_SFID_UGM,
                                                    LSC_OP_LOAD,
                                                    LSC_ADDR_SIZE_A32,
                                                    LSC_DATA_SIZE_D32,
                                                    use_transpose ? reg_size * 8 : 1 /* num_channels */,
                                                    use_transpose,
                                                    LSC_CACHE(devinfo, LOAD, L1STATE_L3MOCS),
                                                    0 /* scale_offset */,
                                                    (inst->use_base_offset ? inst->offset : 0) / 4,
                                                    0 /* surface_state_index */);;

   unspill_inst->dst = dst;
   unspill_inst->src[SENDG_SRC_IND_0_DESC] = brw_s0(BRW_TYPE_UQ, 7);
   unspill_inst->src[SENDG_SRC_IND_1_DESC] = brw_reg();
   unspill_inst->src[SEND_SRC_PAYLOAD1] = offset;
   unspill_inst->src[SEND_SRC_PAYLOAD2] = brw_reg();

   unspill_inst->sfid = GEN_SFID_UGM;
   unspill_inst->header_size = 0;
   unspill_inst->mlen = brw_lsc_msg_addr_len(devinfo, LSC_ADDR_SIZE_A32,
                                             unspill_inst->exec_size);
   unspill_inst->ex_mlen = 0;
   unspill_inst->size_written =
      brw_lsc_msg_dest_len(devinfo, LSC_DATA_SIZE_D32, bld.dispatch_width()) * REG_SIZE;
   unspill_inst->has_side_effects = false;
   unspill_inst->is_volatile = true;
   unspill_inst->bindless_surface = true;
   unspill_inst->efficient_64bit = true;

   assert(unspill_inst->size_written == inst->size_written);
   assert(unspill_inst->size_read(devinfo, SEND_SRC_PAYLOAD1) == inst->size_read(devinfo, FILL_SRC_PAYLOAD1));

   inst->remove();
}

static void
brw_lower_lsc_spill(const intel_device_info *devinfo, brw_scratch_inst *inst)
{
   assert(devinfo->verx10 >= 125);

   const brw_builder bld(inst);
   brw_reg offset = inst->src[SPILL_SRC_PAYLOAD1];
   brw_reg src = inst->src[SPILL_SRC_PAYLOAD2];

   const unsigned reg_size = src.component_size(bld.dispatch_width()) /
                             REG_SIZE;

   assert(!inst->use_transpose);

   const brw_reg ex_desc = build_ex_desc(bld, reg_size, false);

   brw_send_inst *spill_inst = bld.SEND();

   spill_inst->src[SEND_SRC_DESC]     = brw_imm_ud(0);
   spill_inst->src[SEND_SRC_EX_DESC]  = ex_desc;
   spill_inst->src[SEND_SRC_PAYLOAD1] = offset;
   spill_inst->src[SEND_SRC_PAYLOAD2] = src;

   spill_inst->sfid = GEN_SFID_UGM;
   uint32_t desc = lsc_msg_desc(devinfo, LSC_OP_STORE,
                                LSC_ADDR_SURFTYPE_SS,
                                LSC_ADDR_SIZE_A32,
                                LSC_DATA_SIZE_D32,
                                1 /* num_channels */,
                                false /* transpose */,
                                LSC_CACHE(devinfo, LOAD, L1STATE_L3MOCS));
   spill_inst->header_size = 0;
   spill_inst->mlen = brw_lsc_msg_addr_len(devinfo, LSC_ADDR_SIZE_A32,
                                       bld.dispatch_width());
   spill_inst->ex_mlen = reg_size;
   spill_inst->size_written = 0;
   spill_inst->has_side_effects = true;
   spill_inst->is_volatile = false;
   spill_inst->bindless_surface = true;

   spill_inst->src[0] = brw_imm_ud(
      desc |
      brw_message_desc(devinfo,
                       spill_inst->mlen,
                       spill_inst->size_written / REG_SIZE,
                       spill_inst->header_size));

   if (inst->use_base_offset)
      encode_const_offset(spill_inst, devinfo, LSC_OP_STORE, inst->offset);

   assert(spill_inst->size_written == inst->size_written);
   assert(spill_inst->size_read(devinfo, SEND_SRC_PAYLOAD1) == inst->size_read(devinfo, SPILL_SRC_PAYLOAD1));
   assert(spill_inst->size_read(devinfo, SEND_SRC_PAYLOAD2) == inst->size_read(devinfo, SPILL_SRC_PAYLOAD2));

   inst->remove();
}

static void
brw_lower_lsc64_spill(const intel_device_info *devinfo, brw_scratch_inst *inst)
{
   assert(devinfo->verx10 >= 350);

   const brw_builder bld(inst);
   brw_reg offset = inst->src[SPILL_SRC_PAYLOAD1];
   brw_reg src = inst->src[SPILL_SRC_PAYLOAD2];

   const unsigned reg_size = src.component_size(bld.dispatch_width()) /
                             REG_SIZE;

   assert(!inst->use_transpose);

   brw_send_inst *spill_inst = bld.SEND();
   spill_inst->combined_desc = lsc_64bit_msg_desc(devinfo,
                                                  GEN_SFID_UGM,
                                                  LSC_OP_STORE,
                                                  LSC_ADDR_SIZE_A32,
                                                  LSC_DATA_SIZE_D32,
                                                  1 /* num_channels */,
                                                  false /* use_transpose */,
                                                  LSC_CACHE(devinfo, STORE, L1STATE_L3MOCS),
                                                  0 /* scale_offset */,
                                                  (inst->use_base_offset ? inst->offset : 0) / 4,
                                                  0 /* surface_state_index */);

   spill_inst->src[SENDG_SRC_IND_0_DESC] = brw_s0(BRW_TYPE_UQ, 7);
   spill_inst->src[SENDG_SRC_IND_1_DESC] = brw_reg();
   spill_inst->src[SEND_SRC_PAYLOAD1] = offset;
   spill_inst->src[SEND_SRC_PAYLOAD2] = src;

   spill_inst->sfid = GEN_SFID_UGM;
   spill_inst->header_size = 0;
   spill_inst->mlen = brw_lsc_msg_addr_len(devinfo, LSC_ADDR_SIZE_A32,
                                       bld.dispatch_width());
   spill_inst->ex_mlen = reg_size;
   spill_inst->size_written = 0;
   spill_inst->has_side_effects = true;
   spill_inst->is_volatile = false;
   spill_inst->bindless_surface = true;
   spill_inst->efficient_64bit = true;

   assert(spill_inst->size_written == inst->size_written);
   assert(spill_inst->size_read(devinfo, SEND_SRC_PAYLOAD1) == inst->size_read(devinfo, SPILL_SRC_PAYLOAD1));
   assert(spill_inst->size_read(devinfo, SEND_SRC_PAYLOAD2) == inst->size_read(devinfo, SPILL_SRC_PAYLOAD2));

   inst->remove();
}

bool
brw_lower_fill_and_spill(brw_shader &s)
{
   bool progress = false;

   if (s.key->use_efficient_64bit) {
      brw_inst *first_inst =
         s.cfg->first_block()->start();
      const brw_builder ubld = brw_builder(first_inst).exec_all().group(1, 0);
      ubld.emit(SHADER_OPCODE_MOV_RELOC_IMM, brw_s0(BRW_TYPE_UD, 14),
                brw_imm_ud(BRW_SHADER_RELOC_SCRATCH64_SURFACE_LOW), brw_imm_ud(0));
      ubld.emit(SHADER_OPCODE_MOV_RELOC_IMM, brw_s0(BRW_TYPE_UD, 15),
                brw_imm_ud(BRW_SHADER_RELOC_SCRATCH64_SURFACE_HIGH), brw_imm_ud(0));
   }


   foreach_block_and_inst_safe(block, brw_inst, inst, s.cfg) {
      switch (inst->opcode) {
      case SHADER_OPCODE_LSC_FILL:
         if (s.key->use_efficient_64bit)
            brw_lower_lsc64_fill(s.devinfo, s, inst->as_scratch());
         else
            brw_lower_lsc_fill(s.devinfo, s, inst->as_scratch());
         progress = true;
         break;

      case SHADER_OPCODE_LSC_SPILL:
         if (s.key->use_efficient_64bit)
            brw_lower_lsc64_spill(s.devinfo, inst->as_scratch());
         else
            brw_lower_lsc_spill(s.devinfo, inst->as_scratch());
         progress = true;
         break;

      default:
         break;
      }
   }

   if (progress)
      s.invalidate_analysis(BRW_DEPENDENCY_INSTRUCTIONS |
                            BRW_DEPENDENCY_VARIABLES);

   return progress;
}
