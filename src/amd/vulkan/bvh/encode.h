/*
 * Copyright © 2022 Friedrich Vock
 * Copyright © 2025 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

/* Helpers for encoding BVH nodes on different HW generations. */

#ifndef RADV_BVH_ENCODE_H
#define RADV_BVH_ENCODE_H

#include "bvh_helpers.h"
#include "invocation_cluster.h"

void
radv_encode_aabb_gfx10_3(VOID_REF dst_addr, vk_ir_aabb_node src)
{
   REF(radv_bvh_aabb_node) dst = REF(radv_bvh_aabb_node)(dst_addr);

   DEREF(dst).primitive_id = src.primitive_id;
   DEREF(dst).geometry_id_and_flags = src.geometry_id_and_flags;
}

#if ((VK_USED_BUILD_FLAGS & VK_BUILD_FLAG_PROPAGATE_CULL_FLAGS) != 0)
void
radv_encode_instance_gfx10_3(VOID_REF dst_addr, vk_ir_instance_node src)
{
   REF(radv_bvh_instance_node) dst = REF(radv_bvh_instance_node)(dst_addr);

   radv_accel_struct_header blas_header = DEREF(REF(radv_accel_struct_header)(src.base_ptr));

   uint64_t ptr = addr_to_node(src.base_ptr + blas_header.bvh_offset);
   if (VK_TEST_BUILD_FLAG_PROPAGATE_CULL_FLAGS)
      ptr |= radv_encode_blas_pointer_flags(src.sbt_offset_and_flags >> 24, blas_header.geometry_type);

   DEREF(dst).bvh_ptr = ptr;
   DEREF(dst).bvh_offset = blas_header.bvh_offset;

   mat4 transform = mat4(src.otw_matrix);
   mat4 inv_transform = transpose(inverse(transpose(transform)));
   DEREF(dst).wto_matrix = mat3x4(inv_transform);
   DEREF(dst).otw_matrix = mat3x4(transform);

   DEREF(dst).custom_instance_and_mask = src.custom_instance_and_mask;
   DEREF(dst).sbt_offset_and_flags = radv_encode_sbt_offset_and_flags(src.sbt_offset_and_flags);
   DEREF(dst).instance_id = src.instance_id;
}

void
radv_encode_triangle_gfx10_3(VOID_REF dst_addr, vk_ir_triangle_node src)
{
   REF(radv_bvh_triangle_node) dst = REF(radv_bvh_triangle_node)(dst_addr);

   uint32_t barycentrics_control = 9;
   if (VK_TEST_BUILD_FLAG_PROPAGATE_CULL_FLAGS) {
      bool opaque = (src.geometry_id_and_flags & VK_GEOMETRY_OPAQUE) != 0;
      barycentrics_control |= (opaque ? 128 : 0);
   }

   DEREF(dst).coords = src.coords;
   DEREF(dst).triangle_id = src.triangle_id;
   DEREF(dst).geometry_id_and_flags = src.geometry_id_and_flags;
   DEREF(dst).id = barycentrics_control;
}
#endif

struct bit_writer {
   uint64_t addr;
   uint32_t offset;
   uint32_t temp;
   uint32_t count;
   uint32_t total_count;
};

void
bit_writer_init(out bit_writer writer, uint64_t addr)
{
   writer.addr = addr;
   writer.offset = 0;
   writer.temp = 0;
   writer.count = 0;
   writer.total_count = 0;
}

void
bit_writer_write(inout bit_writer writer, uint32_t data, uint32_t bit_size)
{
   writer.total_count += bit_size;

   if (writer.count + bit_size >= 32) {
      writer.temp = writer.temp | (data << writer.count);

      REF(uint32_t) dst = REF(uint32_t)(writer.addr + writer.offset);
      DEREF(dst) = writer.temp;
      writer.offset += 4;

      bit_size = bit_size - (32 - writer.count);
      if (writer.count == 0)
         data = 0;
      else
         data = data >> (32 - writer.count);

      writer.temp = 0;
      writer.count = 0;
   }

   writer.temp = writer.temp | (data << writer.count);
   writer.count += bit_size;
}

void
bit_writer_skip_to(inout bit_writer writer, uint32_t target)
{
   /* Flush the remaining data. */
   if (writer.count > 0) {
      REF(uint32_t) dst = REF(uint32_t)(writer.addr + writer.offset);
      DEREF(dst) = writer.temp;
      writer.temp = 0;
   }

   writer.count = target % 32;
   writer.total_count = target;
   writer.offset = (target / 32) * 4;
}

void
bit_writer_finish(inout bit_writer writer)
{
   /* Flush the remaining data. */
   if (writer.count > 0) {
      REF(uint32_t) dst = REF(uint32_t)(writer.addr + writer.offset);
      DEREF(dst) = writer.temp;
   }

   writer.temp = 0;
   writer.count = 0;
   writer.total_count = 0;
}

struct radv_gfx12_box_node_encoder {
   vk_aabb total_bounds;
   vec3 aligned_extent;
};

void
radv_gfx12_box_node_encoder_init(inout radv_gfx12_box_node_encoder encoder, vk_aabb total_bounds)
{
   encoder.total_bounds = total_bounds;

   vec3 extent = total_bounds.max - total_bounds.min;
   encoder.aligned_extent = uintBitsToFloat((floatBitsToUint(extent) + uvec3(0x7fffff)) & 0x7f800000);
}

uint32_t
radv_gfx12_box_node_encoder_get_exponents_count(radv_gfx12_box_node_encoder encoder,
                                                uint32_t child_node_count_minus_one)
{
   uvec3 extent_exponents = floatBitsToUint(encoder.aligned_extent) >> 23;
   uint32_t result = child_node_count_minus_one << 28;
   result |= extent_exponents.x << 0;
   result |= extent_exponents.y << 8;
   result |= extent_exponents.z << 16;
   return result;
}

void
radv_gfx12_box_node_encoder_set_child_bounds(radv_gfx12_box_node_encoder encoder, inout radv_gfx12_box_child child,
                                             vk_aabb aabb)
{
   vec3 origin = encoder.total_bounds.min;
   vec3 aligned_extent = encoder.aligned_extent;

   child.dword0 = (child.dword0 & 0xFF000000) |
                  min(uint32_t(floor((aabb.min.x - origin.x) / aligned_extent.x * float(0x1000))), 0xfff) |
                  (min(uint32_t(floor((aabb.min.y - origin.y) / aligned_extent.y * float(0x1000))), 0xfff) << 12);
   child.dword1 = (child.dword1 & 0xFF000000) |
                  min(uint32_t(floor((aabb.min.z - origin.z) / aligned_extent.z * float(0x1000))), 0xfff) |
                  (min(uint32_t(ceil((aabb.max.x - origin.x) / aligned_extent.x * float(0x1000))) - 1, 0xfff) << 12);
   child.dword2 = (child.dword2 & 0xFF000000) |
                  min(uint32_t(ceil((aabb.max.y - origin.y) / aligned_extent.y * float(0x1000))) - 1, 0xfff) |
                  (min(uint32_t(ceil((aabb.max.z - origin.z) / aligned_extent.z * float(0x1000))) - 1, 0xfff) << 12);
}

radv_gfx12_box_child
radv_gfx12_box_node_encoder_get_child(radv_gfx12_box_node_encoder encoder, vk_aabb child_aabb, uint32_t type,
                                      uint32_t size, uint32_t flags, uint32_t mask)
{
   radv_gfx12_box_child box_child;
   box_child.dword0 = flags << 24;
   box_child.dword1 = mask << 24;
   box_child.dword2 = (type << 24) | (size << 28);

   radv_gfx12_box_node_encoder_set_child_bounds(encoder, box_child, child_aabb);

   return box_child;
}

#define RADV_GFX12_UPDATABLE_PRIMITIVE_NODE_INDICES_OFFSET                                                             \
   (align(RADV_GFX12_PRIMITIVE_NODE_HEADER_SIZE, 32) / 8 + 9 * 4)

void
radv_encode_triangle_gfx12(VOID_REF dst, vk_ir_triangle_node src)
{
   bit_writer child_writer;
   bit_writer_init(child_writer, dst);

   bit_writer_write(child_writer, 31, 5); /* x_vertex_bits_minus_one */
   bit_writer_write(child_writer, 31, 5); /* y_vertex_bits_minus_one */
   bit_writer_write(child_writer, 31, 5); /* z_vertex_bits_minus_one */
   bit_writer_write(child_writer, 0, 5);  /* trailing_zero_bits */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_base_bits_div_2 */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_bits_div_2 */
   bit_writer_write(child_writer, 0, 3);  /* triangle_pair_count_minus_one */
   bit_writer_write(child_writer, 0, 1);  /* vertex_type */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_base_bits */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_bits */
   bit_writer_write(child_writer, RADV_GFX12_UPDATABLE_PRIMITIVE_NODE_INDICES_OFFSET * 8 + 32, 10);

   bit_writer_write(child_writer, floatBitsToUint(src.coords[0][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[0][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[0][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[1][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[1][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[1][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[2][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[2][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.coords[2][2]), 32);

   bit_writer_write(child_writer, 0, 64 - RADV_GFX12_PRIMITIVE_NODE_HEADER_SIZE + 4);

   bit_writer_write(child_writer, src.geometry_id_and_flags & 0xfffffff, 28);
   bit_writer_write(child_writer, src.triangle_id, 28);

   bit_writer_skip_to(child_writer, 32 * 32 - RADV_GFX12_PRIMITIVE_NODE_PAIR_DESC_SIZE);

   uint32_t opaque = (src.geometry_id_and_flags & VK_GEOMETRY_OPAQUE) != 0 ? 1 : 0;

   bit_writer_write(child_writer, 1, 1);      /* prim_range_stop */
   bit_writer_write(child_writer, 0, 1);      /* tri1_double_sided */
   bit_writer_write(child_writer, 0, 1);      /* tri1_opaque */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v0_index */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v1_index */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v2_index */
   bit_writer_write(child_writer, 0, 1);      /* tri0_double_sided */
   bit_writer_write(child_writer, opaque, 1); /* tri0_opaque */
   bit_writer_write(child_writer, 0, 4);      /* tri0_v0_index */
   bit_writer_write(child_writer, 1, 4);      /* tri0_v1_index */
   bit_writer_write(child_writer, 2, 4);      /* tri0_v2_index */

   bit_writer_finish(child_writer);
}

void
radv_encode_triangle_gfx12(VOID_REF dst, vk_ir_triangle_node src0, vk_ir_triangle_node src1)
{
   bit_writer child_writer;
   bit_writer_init(child_writer, dst);

   bit_writer_write(child_writer, 31, 5); /* x_vertex_bits_minus_one */
   bit_writer_write(child_writer, 31, 5); /* y_vertex_bits_minus_one */
   bit_writer_write(child_writer, 31, 5); /* z_vertex_bits_minus_one */
   bit_writer_write(child_writer, 0, 5);  /* trailing_zero_bits */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_base_bits_div_2 */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_bits_div_2 */
   bit_writer_write(child_writer, 0, 3);  /* triangle_pair_count_minus_one */
   bit_writer_write(child_writer, 0, 1);  /* vertex_type */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_base_bits */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_bits */
   /* header + (9 floats + geometry_id) * 2 triangles */
   bit_writer_write(child_writer, RADV_GFX12_PRIMITIVE_NODE_HEADER_SIZE + 2 * 9 * 32 + 2 * 28, 10);

   bit_writer_write(child_writer, floatBitsToUint(src0.coords[0][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[0][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[0][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[1][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[1][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[1][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[2][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[2][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src0.coords[2][2]), 32);

   bit_writer_write(child_writer, floatBitsToUint(src1.coords[0][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[0][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[0][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[1][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[1][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[1][2]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[2][0]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[2][1]), 32);
   bit_writer_write(child_writer, floatBitsToUint(src1.coords[2][2]), 32);

   bit_writer_write(child_writer, src1.geometry_id_and_flags & 0xfffffff, 28);
   bit_writer_write(child_writer, src0.geometry_id_and_flags & 0xfffffff, 28);
   bit_writer_write(child_writer, src0.triangle_id, 28);
   bit_writer_write(child_writer, src1.triangle_id, 28);

   bit_writer_skip_to(child_writer, 32 * 32 - RADV_GFX12_PRIMITIVE_NODE_PAIR_DESC_SIZE);

   uint32_t opaque0 = (src0.geometry_id_and_flags & VK_GEOMETRY_OPAQUE) != 0 ? 1 : 0;
   uint32_t opaque1 = (src1.geometry_id_and_flags & VK_GEOMETRY_OPAQUE) != 0 ? 1 : 0;

   bit_writer_write(child_writer, 1, 1);       /* prim_range_stop */
   bit_writer_write(child_writer, 0, 1);       /* tri1_double_sided */
   bit_writer_write(child_writer, opaque1, 1); /* tri1_opaque */
   bit_writer_write(child_writer, 3, 4);       /* tri1_v0_index */
   bit_writer_write(child_writer, 4, 4);       /* tri1_v1_index */
   bit_writer_write(child_writer, 5, 4);       /* tri1_v2_index */
   bit_writer_write(child_writer, 0, 1);       /* tri0_double_sided */
   bit_writer_write(child_writer, opaque0, 1); /* tri0_opaque */
   bit_writer_write(child_writer, 0, 4);       /* tri0_v0_index */
   bit_writer_write(child_writer, 1, 4);       /* tri0_v1_index */
   bit_writer_write(child_writer, 2, 4);       /* tri0_v2_index */

   bit_writer_finish(child_writer);
}

void
radv_encode_aabb_gfx12(VOID_REF dst, vk_ir_aabb_node src)
{
   bit_writer child_writer;
   bit_writer_init(child_writer, dst);

   bit_writer_write(child_writer, 0, 5);  /* x_vertex_bits_minus_one */
   bit_writer_write(child_writer, 0, 5);  /* y_vertex_bits_minus_one */
   bit_writer_write(child_writer, 0, 5);  /* z_vertex_bits_minus_one */
   bit_writer_write(child_writer, 0, 5);  /* trailing_zero_bits */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_base_bits_div_2 */
   bit_writer_write(child_writer, 14, 4); /* geometry_index_bits_div_2 */
   bit_writer_write(child_writer, 0, 3);  /* triangle_pair_count_minus_one */
   bit_writer_write(child_writer, 0, 1);  /* vertex_type */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_base_bits */
   bit_writer_write(child_writer, 28, 5); /* primitive_index_bits */
   bit_writer_write(child_writer, RADV_GFX12_UPDATABLE_PRIMITIVE_NODE_INDICES_OFFSET * 8 + 32, 10);

   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.min.x), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.min.y), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.min.z), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.max.x), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.max.y), 32);
   bit_writer_write(child_writer, floatBitsToUint(src.base.aabb.max.z), 32);

   bit_writer_skip_to(child_writer, RADV_GFX12_UPDATABLE_PRIMITIVE_NODE_INDICES_OFFSET * 8 + 4);

   bit_writer_write(child_writer, src.geometry_id_and_flags & 0xfffffff, 28);
   bit_writer_write(child_writer, src.primitive_id, 28);

   bit_writer_skip_to(child_writer, 32 * 32 - RADV_GFX12_PRIMITIVE_NODE_PAIR_DESC_SIZE);

   uint32_t opaque = (src.geometry_id_and_flags & VK_GEOMETRY_OPAQUE) != 0 ? 1 : 0;

   bit_writer_write(child_writer, 1, 1);      /* prim_range_stop */
   bit_writer_write(child_writer, 0, 1);      /* tri1_double_sided */
   bit_writer_write(child_writer, 0, 1);      /* tri1_opaque */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v0_index */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v1_index */
   bit_writer_write(child_writer, 0, 4);      /* tri1_v2_index */
   bit_writer_write(child_writer, 0, 1);      /* tri0_double_sided */
   bit_writer_write(child_writer, opaque, 1); /* tri0_opaque */
   bit_writer_write(child_writer, 0xf, 4);    /* tri0_v0_index */
   bit_writer_write(child_writer, 0xf, 4);    /* tri0_v1_index */
   bit_writer_write(child_writer, 0, 4);      /* tri0_v2_index */

   bit_writer_finish(child_writer);
}

void
radv_write_instance_filter_gfx12(REF(radv_accel_struct_header_gfx12) dst, radv_invocation_cluster cluster,
                                 vk_aabb total_bounds, vk_aabb bounds, uint32_t valid_child_count_minus_one)
{
   radv_gfx12_box_node_encoder encoder;
   radv_gfx12_box_node_encoder_init(encoder, total_bounds);

   radv_gfx12_box_child child;
   child.dword0 = 0xffffffff;
   child.dword1 = 0xfff;
   child.dword2 = 0;

   if (cluster.invocation_index == 0) {
      DEREF(dst).instance_child_count_exponents = radv_gfx12_box_node_encoder_get_exponents_count(encoder, 0);
      child = radv_gfx12_box_node_encoder_get_child(encoder, total_bounds, 5, 0, 0, 0xff);
   }

   if (cluster.invocation_index < 4) {
      DEREF(dst).instance_children[cluster.invocation_index] = child;
   }
}

/* Writes both the HW node and user data. */
void
radv_encode_instance_gfx12(VOID_REF dst_addr, vk_ir_instance_node src, uint32_t parent_id)
{
   radv_accel_struct_header_gfx12 blas_header = DEREF(REF(radv_accel_struct_header_gfx12)(src.base_ptr));

   mat4 transform = mat4(src.otw_matrix);
   mat4 wto_matrix = transpose(inverse(transpose(transform)));

   REF(radv_gfx12_instance_node) dst = REF(radv_gfx12_instance_node)(dst_addr);
   DEREF(dst).wto_matrix = mat3x4(wto_matrix);

   uint32_t flags = src.sbt_offset_and_flags >> 24;
   uint32_t instance_pointer_flags = 0;

   uint64_t bvh_addr = addr_to_node(src.base_ptr + blas_header.base.bvh_offset);
   bvh_addr |= radv_encode_blas_pointer_flags(flags, blas_header.base.geometry_type);
   DEREF(dst).pointer_flags_bvh_addr = bvh_addr;

   DEREF(dst).parent_id = parent_id;
   DEREF(dst).cull_mask_user_data = (src.sbt_offset_and_flags & 0xffffff) | (src.custom_instance_and_mask & 0xff000000);

   DEREF(dst).origin = blas_header.base.aabb.min;
   DEREF(dst).child_count_exponents = blas_header.instance_child_count_exponents;
   DEREF(dst).children = blas_header.instance_children;

   REF(radv_gfx12_instance_node_user_data) user_data =
      REF(radv_gfx12_instance_node_user_data)(dst_addr + RADV_GFX12_BVH_NODE_SIZE);
   DEREF(user_data).otw_matrix = src.otw_matrix;
   DEREF(user_data).custom_instance = src.custom_instance_and_mask & 0xffffff;
   DEREF(user_data).instance_index = src.instance_id;
   DEREF(user_data).bvh_offset = blas_header.base.bvh_offset;
   DEREF(user_data).leaf_node_offsets_offset = blas_header.base.leaf_node_offsets_offset;
}

#endif
