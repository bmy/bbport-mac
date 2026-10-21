/*
 * Copyright (c) 2026 Igalia S.L.
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

#ifndef GENX_HUC_H
#define GENX_HUC_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "anv_private.h"

#define ANV_HUC_S2L_KERNEL_DESCRIPTOR    1
#define ANV_HUC_PRODUCT_FAMILY_SKL       2
#define ANV_HUC_PRODUCT_FAMILY_BXT       3
#define ANV_HUC_PRODUCT_FAMILY_KBL       4
#define ANV_HUC_PRODUCT_FAMILY_ICL       6
/* Same value since TGL */
#define ANV_HUC_PRODUCT_FAMILY_TGL       8
#define ANV_HUC_DMEM_DEST_OFFSET         0x2000
#if GFX_VER >= 11
#define ANV_HUC_STATUS_MMIO_OFFSET       0x2000
#define ANV_HUC_STATUS2_MMIO_OFFSET      0x23b0
#else
#define ANV_HUC_STATUS_MMIO_OFFSET       0xd000
#define ANV_HUC_STATUS2_MMIO_OFFSET      0xd3b0
#endif
#define ANV_HUC_STATUS2_IMEM_LOADED_BIT  0x40
#define ANV_HUC_STATUS_S2L_OK_BIT        0x8000
#define ANV_HUC_S2L_SLICE_CMD_SIZE       ((13 + 18 + 18 + 42 + 42 + 3) * 4)

struct anv_huc_hevc_s2l_pic_bss {
   uint32_t pic_width_in_min_cbs_y;
   uint32_t pic_height_in_min_cbs_y;
   uint8_t  log2_min_luma_coding_block_size_minus3;
   uint8_t  log2_diff_max_min_luma_coding_block_size;
   uint16_t chroma_format_idc : 2;
   uint16_t separate_colour_plane_flag : 1;
   uint16_t bit_depth_luma_minus8 : 4;
   uint16_t bit_depth_chroma_minus8 : 4;
   uint16_t log2_max_pic_order_cnt_lsb_minus4 : 4;
   uint16_t sample_adaptive_offset_enabled_flag : 1;
   uint8_t  num_short_term_ref_pic_sets;
   uint8_t  long_term_ref_pics_present_flag : 1;
   uint8_t  num_long_term_ref_pics_sps : 6;
   uint8_t  sps_temporal_mvp_enable_flag : 1;
   uint8_t  num_ref_idx_l0_default_active_minus1 : 4;
   uint8_t  num_ref_idx_l1_default_active_minus1 : 4;
   int8_t   pic_init_qp_minus26;
   uint8_t  dependent_slice_segments_enabled_flag : 1;
   uint8_t  cabac_init_present_flag : 1;
   uint8_t  pps_slice_chroma_qp_offsets_present_flag : 1;
   uint8_t  weighted_pred_flag : 1;
   uint8_t  weighted_bipred_flag : 1;
   uint8_t  output_flag_present_flag : 1;
   uint8_t  tiles_enabled_flag : 1;
   uint8_t  entropy_coding_sync_enabled_flag : 1;
   uint8_t  loop_filter_across_slices_enabled_flag : 1;
   uint8_t  deblocking_filter_override_enabled_flag : 1;
   uint8_t  pic_disable_deblocking_filter_flag : 1;
   uint8_t  lists_modification_present_flag : 1;
   uint8_t  slice_segment_header_extension_present_flag : 1;
   uint8_t  high_precision_offsets_enabled_flag : 1;
   uint8_t  chroma_qp_offset_list_enabled_flag : 1;
   uint8_t  reserved0 : 1;
   int32_t  CurrPicOrderCntVal;
   int32_t  PicOrderCntValList[15];
   uint8_t  RefPicSetStCurrBefore[8];
   uint8_t  RefPicSetStCurrAfter[8];
   uint8_t  RefPicSetLtCurr[8];
   uint16_t RefFieldPicFlag;
   uint16_t RefBottomFieldFlag;
   int8_t   pps_beta_offset_div2;
   int8_t   pps_tc_offset_div2;
   uint16_t StRPSBits;
   uint8_t  num_tile_columns_minus1;
   uint8_t  num_tile_rows_minus1;
   uint16_t column_width[20];
   uint16_t row_height[22];
   uint16_t NumSlices;
   uint8_t  num_extra_slice_header_bits;
   int8_t   RefIdxMapping[15];
   uint8_t  reserved1;
   uint16_t reserved2;
   uint32_t reserved3;
   uint32_t reserved4;
#if GFX_VER >= 12
   uint8_t  IsRealTileEnable;
   uint8_t  NumScalablePipes;
   uint8_t  IsSCCIBCMode;
   uint8_t  IsSCCPLTMode;
   uint8_t  MVRControlIdc;
   uint8_t  UseSliceACTOffset;
   int8_t   pps_act_y_qp_offset;
   int8_t   pps_act_cb_qp_offset;
   int8_t   pps_act_cr_qp_offset;
   uint8_t  PredictorPaletteSize;
   uint16_t PredictorPaletteEntries[3][128];
   uint32_t BatchBufferSize;
#endif
};

struct anv_huc_hevc_s2l_slice_bss {
   uint32_t BSNALunitDataLocation;
   uint32_t SliceBytesInBuffer;
   uint32_t reserved[4];
};

struct anv_huc_hevc_s2l_bss {
   uint32_t ProductFamily;
   uint16_t RevId;
   /* WaDummyReference: tell the S2L kernel to emit a dummy
    * HCP_REF_IDX_STATE for missing reference slots.
    * Sets 1 on gen11 and gen12
    */
   uint32_t DummyRefIdxState;
#if GFX_VER >= 12
   /* Wa_14010222001: tell the S2L kernel to emit a dummy
    * HCP_VD_CONTROL_STATE (SFC/scalability corner cases).
    * 1 on Gen12 and DG.
    */
   uint32_t DummyVDControlState;
   /* Wa_2209620131: tell the S2L kernel to emit MFX_WAIT and
    * VD_PIPELINE_FLUSH for scalability (hang WA).
    * Same platforms as Wa_14010222001.
    */
   uint32_t WaTileFlushScalability;
#endif
   struct anv_huc_hevc_s2l_pic_bss PictureBss;
   struct anv_huc_hevc_s2l_slice_bss SliceBss[];
};

static_assert(sizeof(struct anv_huc_hevc_s2l_slice_bss) == 24, "huc s2l dmem layout");
#if GFX_VER >= 12
static_assert(sizeof(struct anv_huc_hevc_s2l_pic_bss) == 1016, "huc s2l dmem layout");
static_assert(offsetof(struct anv_huc_hevc_s2l_bss, PictureBss) == 0x14, "huc s2l dmem layout");
static_assert(offsetof(struct anv_huc_hevc_s2l_bss, SliceBss) == 0x40c, "huc s2l dmem layout");
#else
static_assert(sizeof(struct anv_huc_hevc_s2l_pic_bss) == 232, "huc s2l dmem layout");
static_assert(offsetof(struct anv_huc_hevc_s2l_bss, PictureBss) == 0xc, "huc s2l dmem layout");
static_assert(offsetof(struct anv_huc_hevc_s2l_bss, SliceBss) == 0xf4, "huc s2l dmem layout");
#endif

static uint32_t
anv_huc_product_family(const struct intel_device_info *info)
{
   switch (info->platform) {
   case INTEL_PLATFORM_SKL:
      return ANV_HUC_PRODUCT_FAMILY_SKL;
   case INTEL_PLATFORM_BXT:
      return ANV_HUC_PRODUCT_FAMILY_BXT;
   case INTEL_PLATFORM_KBL:
   case INTEL_PLATFORM_GLK:
   case INTEL_PLATFORM_CFL:
      return ANV_HUC_PRODUCT_FAMILY_KBL;
   case INTEL_PLATFORM_ICL:
   case INTEL_PLATFORM_EHL:
      return ANV_HUC_PRODUCT_FAMILY_ICL;
   default:
      return ANV_HUC_PRODUCT_FAMILY_TGL;
   }
}

/* Fill the DMEM data consumed by the HuC S2L kernel: SPS/PPS derived picture
 * parameters, reference picture sets and per-slice bitstream locations. This
 * is everything the kernel needs to parse the slice headers and generate the
 * HCP slice commands, except the slice headers themselves, which HuC reads
 * from the bitstream buffer at execution time. The layout must match the
 * HUC_HEVC_S2L_BSS structure the firmware expects: the gen12 variant carries
 * two extra header dwords and a tail of gen12-only picture fields, so the
 * pre-gen12 layout is not just a prefix of it.
 */
static void
anv_h265_huc_s2l_fill_dmem(struct anv_cmd_buffer *cmd_buffer,
                           const VkVideoDecodeInfoKHR *frame_info,
                           const VkVideoDecodeH265PictureInfoKHR *h265_pic_info,
                           const StdVideoH265SequenceParameterSet *sps,
                           const StdVideoH265PictureParameterSet *pps,
                           const uint8_t *dpb_idx,
                           const uint8_t *hcp_ref_idx,
                           struct anv_huc_hevc_s2l_bss *bss)
{
   const struct intel_device_info *info = cmd_buffer->device->info;
   const StdVideoDecodeH265PictureInfo *std_pic = h265_pic_info->pStdPictureInfo;
   struct anv_huc_hevc_s2l_pic_bss *pic = &bss->PictureBss;
   struct anv_huc_hevc_s2l_slice_bss *slices = bss->SliceBss;

   bss->ProductFamily = anv_huc_product_family(info);
   bss->RevId = info->revision;
#if GFX_VER >= 12
   bss->DummyRefIdxState = GFX_VERx10 == 120;
   bss->DummyVDControlState =
      GFX_VERx10 == 120 || intel_device_info_is_dg2(info);
   bss->WaTileFlushScalability =
      GFX_VERx10 == 120 || intel_device_info_is_dg2(info);
#else
   bss->DummyRefIdxState = GFX_VER == 11;
#endif

   uint32_t min_cb_size = 1 << (sps->log2_min_luma_coding_block_size_minus3 + 3);
   pic->pic_width_in_min_cbs_y = sps->pic_width_in_luma_samples / min_cb_size;
   pic->pic_height_in_min_cbs_y = sps->pic_height_in_luma_samples / min_cb_size;
   pic->log2_min_luma_coding_block_size_minus3 =
      sps->log2_min_luma_coding_block_size_minus3;
   pic->log2_diff_max_min_luma_coding_block_size =
      sps->log2_diff_max_min_luma_coding_block_size;
   pic->chroma_format_idc = sps->chroma_format_idc;
   pic->separate_colour_plane_flag = sps->flags.separate_colour_plane_flag;
   pic->bit_depth_luma_minus8 = sps->bit_depth_luma_minus8;
   pic->bit_depth_chroma_minus8 = sps->bit_depth_chroma_minus8;
   pic->log2_max_pic_order_cnt_lsb_minus4 = sps->log2_max_pic_order_cnt_lsb_minus4;
   pic->sample_adaptive_offset_enabled_flag =
      sps->flags.sample_adaptive_offset_enabled_flag;
   pic->num_short_term_ref_pic_sets = sps->num_short_term_ref_pic_sets;
   pic->long_term_ref_pics_present_flag = sps->flags.long_term_ref_pics_present_flag;
   pic->num_long_term_ref_pics_sps = sps->num_long_term_ref_pics_sps;
   pic->sps_temporal_mvp_enable_flag = sps->flags.sps_temporal_mvp_enabled_flag;
   pic->num_ref_idx_l0_default_active_minus1 =
      pps->num_ref_idx_l0_default_active_minus1;
   pic->num_ref_idx_l1_default_active_minus1 =
      pps->num_ref_idx_l1_default_active_minus1;
   pic->pic_init_qp_minus26 = pps->init_qp_minus26;
   pic->dependent_slice_segments_enabled_flag =
      pps->flags.dependent_slice_segments_enabled_flag;
   pic->cabac_init_present_flag = pps->flags.cabac_init_present_flag;
   pic->pps_slice_chroma_qp_offsets_present_flag =
      pps->flags.pps_slice_chroma_qp_offsets_present_flag;
   pic->weighted_pred_flag = pps->flags.weighted_pred_flag;
   pic->weighted_bipred_flag = pps->flags.weighted_bipred_flag;
   pic->output_flag_present_flag = pps->flags.output_flag_present_flag;
   pic->tiles_enabled_flag = pps->flags.tiles_enabled_flag;
   pic->entropy_coding_sync_enabled_flag =
      pps->flags.entropy_coding_sync_enabled_flag;
   pic->loop_filter_across_slices_enabled_flag =
      pps->flags.pps_loop_filter_across_slices_enabled_flag;
   pic->deblocking_filter_override_enabled_flag =
      pps->flags.deblocking_filter_override_enabled_flag;
   pic->pic_disable_deblocking_filter_flag =
      pps->flags.pps_deblocking_filter_disabled_flag;
   pic->lists_modification_present_flag =
      pps->flags.lists_modification_present_flag;
   pic->slice_segment_header_extension_present_flag =
      pps->flags.slice_segment_header_extension_present_flag;
   pic->high_precision_offsets_enabled_flag =
      sps->flags.high_precision_offsets_enabled_flag;
   pic->chroma_qp_offset_list_enabled_flag =
      pps->flags.chroma_qp_offset_list_enabled_flag;

   pic->CurrPicOrderCntVal = std_pic->PicOrderCntVal;

   for (unsigned i = 0; i < frame_info->referenceSlotCount; i++) {
      int slot_idx = frame_info->pReferenceSlots[i].slotIndex;

      if (slot_idx < 0)
         continue;

      pic->PicOrderCntValList[i] = vk_video_h265_poc_by_slot(frame_info, slot_idx);
   }

   memset(pic->RefPicSetStCurrBefore, 0xff, sizeof(pic->RefPicSetStCurrBefore));
   memset(pic->RefPicSetStCurrAfter, 0xff, sizeof(pic->RefPicSetStCurrAfter));
   memset(pic->RefPicSetLtCurr, 0xff, sizeof(pic->RefPicSetLtCurr));

   for (unsigned i = 0; i < STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE; i++) {
      if (std_pic->RefPicSetStCurrBefore[i] != 0xff)
         pic->RefPicSetStCurrBefore[i] = dpb_idx[std_pic->RefPicSetStCurrBefore[i]];
      if (std_pic->RefPicSetStCurrAfter[i] != 0xff)
         pic->RefPicSetStCurrAfter[i] = dpb_idx[std_pic->RefPicSetStCurrAfter[i]];
      if (std_pic->RefPicSetLtCurr[i] != 0xff)
         pic->RefPicSetLtCurr[i] = dpb_idx[std_pic->RefPicSetLtCurr[i]];
   }

   pic->pps_beta_offset_div2 = pps->pps_beta_offset_div2;
   pic->pps_tc_offset_div2 = pps->pps_tc_offset_div2;
   pic->StRPSBits = std_pic->NumBitsForSTRefPicSetInSlice;

   if (pps->flags.tiles_enabled_flag) {
      uint32_t ctb_size = 1 << (sps->log2_diff_max_min_luma_coding_block_size +
                                sps->log2_min_luma_coding_block_size_minus3 + 3);
      uint32_t width_in_ctb =
         DIV_ROUND_UP(sps->pic_width_in_luma_samples, ctb_size);
      uint32_t height_in_ctb =
         DIV_ROUND_UP(sps->pic_height_in_luma_samples, ctb_size);
      unsigned cols = pps->num_tile_columns_minus1;
      unsigned rows = pps->num_tile_rows_minus1;

      pic->num_tile_columns_minus1 = cols;
      pic->num_tile_rows_minus1 = rows;

      if (pps->flags.uniform_spacing_flag) {
         for (unsigned i = 0; i <= cols; i++) {
            pic->column_width[i] = ((i + 1) * width_in_ctb) / (cols + 1) -
                                   (i * width_in_ctb) / (cols + 1);
         }
         for (unsigned i = 0; i <= rows; i++) {
            pic->row_height[i] = ((i + 1) * height_in_ctb) / (rows + 1) -
                                 (i * height_in_ctb) / (rows + 1);
         }
      } else {
         pic->column_width[cols] = width_in_ctb;
         for (unsigned i = 0; i < cols; i++) {
            pic->column_width[i] = pps->column_width_minus1[i] + 1;
            pic->column_width[cols] -= pic->column_width[i];
         }
         pic->row_height[rows] = height_in_ctb;
         for (unsigned i = 0; i < rows; i++) {
            pic->row_height[i] = pps->row_height_minus1[i] + 1;
            pic->row_height[rows] -= pic->row_height[i];
         }
      }
   }

   pic->NumSlices = h265_pic_info->sliceSegmentCount;
   pic->num_extra_slice_header_bits = pps->num_extra_slice_header_bits;

   memset(pic->RefIdxMapping, 0xff, sizeof(pic->RefIdxMapping));
   for (unsigned i = 0; i < frame_info->referenceSlotCount; i++)
      pic->RefIdxMapping[i] = hcp_ref_idx[i];

   uint32_t buffer_offset = frame_info->srcBufferOffset & 4095;

   for (unsigned s = 0; s < h265_pic_info->sliceSegmentCount; s++) {
      slices[s].BSNALunitDataLocation =
         buffer_offset + h265_pic_info->pSliceSegmentOffsets[s];
      slices[s].SliceBytesInBuffer =
         anv_h265_slice_size(frame_info, h265_pic_info, s);
   }
}

/* Emit the HuC S2L (short-to-long) sequence. The S2L firmware kernel parses
 * the HEVC slice headers from the bitstream buffer on the GPU at execution
 * time and writes the corresponding HCP slice commands into second_bb_addr,
 * which the caller then executes as a second-level batch buffer. This avoids
 * parsing the bitstream on the CPU at record time, which would violate the
 * Vulkan spec as the application is allowed to fill the buffer after
 * recording.
 */
static void
genX(h265_huc_s2l)(struct anv_cmd_buffer *cmd_buffer,
                   const VkVideoDecodeInfoKHR *frame_info,
                   const VkVideoDecodeH265PictureInfoKHR *h265_pic_info,
                   const StdVideoH265SequenceParameterSet *sps,
                   const StdVideoH265PictureParameterSet *pps,
                   const uint8_t *dpb_idx,
                   const uint8_t *hcp_ref_idx,
                   struct anv_address second_bb_addr)
{
   ANV_FROM_HANDLE(anv_buffer, src_buffer, frame_info->srcBuffer);
   struct anv_device *device = cmd_buffer->device;

   const uint32_t slice_bss_offset =
      offsetof(struct anv_huc_hevc_s2l_bss, SliceBss);
   uint32_t dmem_size =
      align(slice_bss_offset + h265_pic_info->sliceSegmentCount *
            sizeof(struct anv_huc_hevc_s2l_slice_bss), 64);
   struct anv_state dmem_state =
      anv_cmd_buffer_alloc_temporary_state(cmd_buffer, dmem_size, 4096);

   if (dmem_state.map == NULL)
      return;

   memset(dmem_state.map, 0, dmem_size);
   anv_h265_huc_s2l_fill_dmem(cmd_buffer, frame_info, h265_pic_info, sps, pps,
                              dpb_idx, hcp_ref_idx, dmem_state.map);

   struct anv_address dmem_addr =
      anv_cmd_buffer_temporary_state_address(cmd_buffer, dmem_state);

   struct anv_state status_state =
      anv_cmd_buffer_alloc_temporary_state(cmd_buffer, 16, 8);

   if (status_state.map == NULL)
      return;

   struct anv_address status_addr =
      anv_cmd_buffer_temporary_state_address(cmd_buffer, status_state);

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_IMEM_STATE), imem) {
      imem.HUCFirmwareDescriptor = ANV_HUC_S2L_KERNEL_DESCRIPTOR;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_PIPE_MODE_SELECT), sel);

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_IND_OBJ_BASE_ADDR_STATE), ind) {
      ind.HUCIndirectStreamInObjectAddress =
         anv_address_add(src_buffer->address, frame_info->srcBufferOffset & ~4095);

      ind.HUCIndirectStreamInObjectAddressAttributes = ANV_VID_ATTR(device, src_buffer->address.bo);
      ind.HUCIndirectStreamInObjectAccessUpperBound =
         anv_address_add(src_buffer->address,
                         align64(frame_info->srcBufferRange + frame_info->srcBufferOffset, 4096));
      ind.HUCIndirectStreamOutObjectAddressAttributes = ANV_VID_ATTR(device, NULL);
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_VIRTUAL_ADDR_STATE), va) {
      va.HUCVirtualAddressRegion[0] = (struct GENX(HUC_VIRTUAL_ADDR_REGION)) {
         .Address = second_bb_addr,
         .AddressAttributes = ANV_VID_ATTR(device, second_bb_addr.bo),
      };

      for (unsigned i = 1; i < 16; i++)
         va.HUCVirtualAddressRegion[i].AddressAttributes = ANV_VID_ATTR(device, NULL);
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_DMEM_STATE), dmem) {
      dmem.HUCDataSourceAddress = dmem_addr;
      dmem.HUCDataSourceAddressAttributes = ANV_VID_ATTR(device, dmem_addr.bo);
      dmem.HUCDataDestinationAddress = (struct anv_address) {
         NULL, ANV_HUC_DMEM_DEST_OFFSET,
      };
      dmem.HUCDataLength = dmem_size / 64;
   }

   uint32_t buffer_offset = frame_info->srcBufferOffset & 4095;
   uint32_t last_slice = h265_pic_info->sliceSegmentCount - 1;

   for (unsigned s = 0; s <= last_slice; s++) {
      anv_batch_emit(&cmd_buffer->batch, GENX(HUC_STREAM_OBJECT), so) {
         so.IndirectStreamInDataLength =
            anv_h265_slice_size(frame_info, h265_pic_info, s);
         so.IndirectStreamInAddress =
            buffer_offset + h265_pic_info->pSliceSegmentOffsets[s];
         so.HUCProcessing = true;
         so.StartCodeByte0 = 0;
         so.StartCodeByte1 = 0;
         so.StartCodeByte2 = 1;
         so.StartCodeSearchEngine = true;
         so.EmulationPreventionByteRemoval = true;
         so.DRMLengthMode = StartCodeMode;
         so.HUCBitstreamEnable = true;
      }

      if (s == last_slice) {
         anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_DATA_IMM), sdi) {
            sdi.Address = status_addr;
            sdi.ImmediateData = ANV_HUC_STATUS2_IMEM_LOADED_BIT;
         }

         anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
            srm.RegisterAddress = ANV_HUC_STATUS2_MMIO_OFFSET;
#if GFX_VER >= 11
            srm.AddCSMMIOStartOffset = 1;
#endif
            srm.MemoryAddress = anv_address_add(status_addr, 4);
         }
      }

      anv_batch_emit(&cmd_buffer->batch, GENX(HUC_START), start) {
         start.LastStreamObject = (s == last_slice);
      }
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(VD_PIPELINE_FLUSH), flush) {
      flush.HEVCPipelineDone = true;
      flush.HEVCPipelineCommandFlush = true;
      flush.VDCommandMessageParserDone = true;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), flush);

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_CONDITIONAL_BATCH_BUFFER_END), cbbe) {
      cbbe.CompareSemaphore = 1;
      cbbe.CompareMaskMode = CompareMaskModeEnabled;
      cbbe.CompareDataDword = 0;
      cbbe.CompareAddress = status_addr;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_DATA_IMM), sdi) {
      sdi.Address = anv_address_add(status_addr, 8);
      sdi.ImmediateData = ANV_HUC_STATUS_S2L_OK_BIT;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_STORE_REGISTER_MEM), srm) {
      srm.RegisterAddress = ANV_HUC_STATUS_MMIO_OFFSET;
#if GFX_VER >= 11
      srm.AddCSMMIOStartOffset = 1;
#endif
      srm.MemoryAddress = anv_address_add(status_addr, 12);
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_CONDITIONAL_BATCH_BUFFER_END), cbbe) {
      cbbe.CompareSemaphore = 1;
      cbbe.CompareMaskMode = CompareMaskModeEnabled;
      cbbe.CompareDataDword = 0;
      cbbe.CompareAddress = anv_address_add(status_addr, 8);
   }
}

static void
anv_huc_emit_copy(struct anv_cmd_buffer *cmd_buffer,
                  struct anv_address dst_addr,
                  struct anv_address src_addr,
                  uint32_t size)
{
   struct anv_device *device = cmd_buffer->device;
   uint32_t src_offset = src_addr.offset & 4095;
   uint32_t dst_offset = dst_addr.offset & 4095;
   struct anv_address src_base = { src_addr.bo, src_addr.offset - src_offset };
   struct anv_address dst_base = { dst_addr.bo, dst_addr.offset - dst_offset };

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_PIPE_MODE_SELECT), sel) {
      sel.IndirectStreamOutEnable = true;
      sel.MediaSoftResetCounter = 2400;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MFX_WAIT), mfx) {
      mfx.MFXSyncControlFlag = 1;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_IND_OBJ_BASE_ADDR_STATE), ind) {
      ind.HUCIndirectStreamInObjectAddress = src_base;
      ind.HUCIndirectStreamInObjectAddressAttributes = ANV_VID_ATTR(device, src_base.bo);

      ind.HUCIndirectStreamInObjectAccessUpperBound =
         anv_address_add(src_base, align64(src_offset + size, 4096));

      ind.HUCIndirectStreamOutObjectAddress = dst_base;
      ind.HUCIndirectStreamOutObjectAddressAttributes = ANV_VID_ATTR(device, dst_base.bo);
      ind.HUCIndirectStreamOutObjectAccessUpperBound =
         anv_address_add(dst_base, align64(dst_offset + size, 4096));
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(HUC_STREAM_OBJECT), so) {
      so.IndirectStreamInDataLength = size;
      so.IndirectStreamInAddress = src_offset;
      so.HUCProcessing = true;
      so.IndirectStreamOutAddress = dst_offset;
      so.StreamOut = true;
      so.HUCBitstreamEnable = true;
   }

   anv_batch_emit(&cmd_buffer->batch, GENX(MI_FLUSH_DW), flush);
}

#endif
