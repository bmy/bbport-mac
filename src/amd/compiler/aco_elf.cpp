/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "aco_interface.h"

#if AMD_LLVM_AVAILABLE
#include "util/u_dynarray.h"
#include "util/u_endian.h"

#include <llvm/BinaryFormat/ELF.h>

#include "ac_gpu_info.h"
#include "ac_shader_util.h"
#include <sid.h>
#include <vector>

using namespace llvm::ELF;

struct aco_elf_section {
   Elf64_Shdr header;
   const void* data;
   unsigned align;
   unsigned index;
};

struct aco_elf_writer {
   util_dynarray strtab;
   unsigned strtab_index;

   unsigned num_sections;
   struct aco_elf_section sections[64];
};

static unsigned
aco_elf_add_string(struct aco_elf_writer* writer, const char* str)
{
   unsigned res = util_dynarray_num_elements(&writer->strtab, char);
   util_dynarray_append_array(&writer->strtab, char, str, strlen(str) + 1);

   struct aco_elf_section* strtab = &writer->sections[writer->strtab_index];
   strtab->header.sh_size = util_dynarray_num_elements(&writer->strtab, char);
   strtab->data = writer->strtab.data;

   return res;
}

static struct aco_elf_section*
aco_elf_add_section(struct aco_elf_writer* writer, Elf64_Shdr header, unsigned align,
                    const void* data)
{
   assert(writer->num_sections < ARRAY_SIZE(writer->sections));
   struct aco_elf_section* section = &writer->sections[writer->num_sections];
   section->header = header;
   section->data = data;
   section->align = align;
   section->index = writer->num_sections++;
   return section;
}

static void
aco_elf_init(struct aco_elf_writer* writer)
{
   memset(writer, 0, sizeof(*writer));

   util_dynarray_init(&writer->strtab, NULL);

   Elf64_Shdr shdr = {0};
   shdr.sh_type = SHT_NULL;
   struct aco_elf_section* null = aco_elf_add_section(writer, shdr, 1, NULL);

   shdr.sh_type = SHT_STRTAB;
   struct aco_elf_section* strtab = aco_elf_add_section(writer, shdr, 1, NULL);
   writer->strtab_index = strtab->index;

   null->header.sh_name = aco_elf_add_string(writer, "");
   strtab->header.sh_name = aco_elf_add_string(writer, ".strtab");
}

static void
aco_elf_finish(struct aco_elf_writer* writer)
{
   util_dynarray_fini(&writer->strtab);
}

static size_t
aco_elf_get_size(struct aco_elf_writer* writer)
{
   size_t size = sizeof(Elf64_Ehdr) + writer->num_sections * sizeof(Elf64_Shdr);
   for (unsigned i = 0; i < writer->num_sections; i++) {
      size = align(size, writer->sections[i].align);
      size += writer->sections[i].header.sh_size;
   }
   return size;
}

static void
aco_elf_write(struct aco_elf_writer* writer, uint64_t e_flags, uint8_t* dst)
{
   /* See https://man.archlinux.org/man/elf.5.en. */
   Elf64_Ehdr* ehdr = (Elf64_Ehdr*)dst;
   ehdr->e_ident[EI_MAG0] = ElfMagic[0];
   ehdr->e_ident[EI_MAG1] = ElfMagic[1];
   ehdr->e_ident[EI_MAG2] = ElfMagic[2];
   ehdr->e_ident[EI_MAG3] = ElfMagic[3];
   ehdr->e_ident[EI_CLASS] = ELFCLASS64;
   ehdr->e_ident[EI_DATA] = UTIL_ARCH_BIG_ENDIAN ? ELFDATA2MSB : ELFDATA2LSB;
   ehdr->e_ident[EI_VERSION] = EV_CURRENT;
   ehdr->e_ident[EI_OSABI] = ELFOSABI_AMDGPU_MESA3D;
   ehdr->e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V2;
   memset(ehdr->e_ident + EI_PAD, 0, EI_NIDENT - EI_PAD);
   ehdr->e_type = ET_REL;
   ehdr->e_machine = EM_AMDGPU;
   ehdr->e_version = EV_CURRENT;
   ehdr->e_entry = 0;
   ehdr->e_flags = e_flags;
   ehdr->e_ehsize = sizeof(Elf64_Ehdr);
   ehdr->e_phentsize = sizeof(Elf64_Phdr);
   ehdr->e_phnum = 0;
   ehdr->e_shentsize = sizeof(Elf64_Shdr);
   ehdr->e_shnum = writer->num_sections;
   ehdr->e_shstrndx = writer->strtab_index;
   ehdr->e_phoff = 0;
   ehdr->e_shoff = sizeof(Elf64_Ehdr);

   Elf64_Shdr* shdr = (Elf64_Shdr*)(dst + ehdr->e_shoff);
   size_t data_offset = ehdr->e_shoff + writer->num_sections * sizeof(Elf64_Shdr);

   for (unsigned i = 0; i < writer->num_sections; i++) {
      data_offset = align(data_offset, writer->sections[i].align);

      shdr[i] = writer->sections[i].header;
      shdr[i].sh_offset = data_offset;
      memcpy(dst + data_offset, writer->sections[i].data, shdr[i].sh_size);

      data_offset += writer->sections[i].header.sh_size;
   }
}

static unsigned
get_elf_eflags(const ac_compiler_info* compiler_info)
{
   switch (compiler_info->gfx_level) {
   case GFX6: return EF_AMDGPU_MACH_AMDGCN_GFX601 /* Pitcairn/Verde */;
   case GFX7: return EF_AMDGPU_MACH_AMDGCN_GFX704 /* Bonaire */;
   case GFX8: return EF_AMDGPU_MACH_AMDGCN_GFX803 /* Jiji/Polaris/VegaM */;
   case GFX9: {
      if (compiler_info && compiler_info->has_accelerated_dot_product) {
         assert(compiler_info->has_fma_mix);
         return EF_AMDGPU_MACH_AMDGCN_GFX906 /* Vega20 */;
      } else if (compiler_info && compiler_info->has_fma_mix) {
         return EF_AMDGPU_MACH_AMDGCN_GFX904 /* Vega12 */;
      } else {
         return EF_AMDGPU_MACH_AMDGCN_GFX900 /* Vega10 */;
      }
   }
   case GFX10: {
      if (compiler_info && compiler_info->has_image_bvh_intersect_ray) {
         assert(!compiler_info->has_accelerated_dot_product);
         return EF_AMDGPU_MACH_AMDGCN_GFX1013;
      } else {
         return EF_AMDGPU_MACH_AMDGCN_GFX1012 /* Navi14 */;
      }
   }
   case GFX10_3: return EF_AMDGPU_MACH_AMDGCN_GFX1030 /* Navi21 */;
   case GFX11: return EF_AMDGPU_MACH_AMDGCN_GFX1100 /* Navi31 */;
   case GFX11_5: return EF_AMDGPU_MACH_AMDGCN_GFX1150 /* Strix1 */;
   case GFX11_7: return 0x5d /* EF_AMDGPU_MACH_AMDGCN_GFX1170 */;
   case GFX12: return EF_AMDGPU_MACH_AMDGCN_GFX1201;
   case GFX12_1: // TODO
   default: UNREACHABLE("");
   }
}

static void
add_config(struct aco_elf_writer* elf, const ac_shader_config* config, unsigned wave_size,
           const ac_compiler_info* compiler_info, uint32_t data[16])
{
   unsigned vgpr_alloc = compiler_info->wave64_vgpr_encode_granularity;
   if (wave_size == 32)
      vgpr_alloc *= 2;

   unsigned scratch_alloc = compiler_info->gfx_level >= GFX11 ? 256 : 1024;

   Elf64_Shdr shdr = {0};
   shdr.sh_name = aco_elf_add_string(elf, ".AMDGPU.config");
   shdr.sh_type = SHT_PROGBITS;
   shdr.sh_addralign = 4;
   shdr.sh_size = 16 * 4;

   /* ac_parse_shader_binary_config() doesn't care whether the stage is correct */

   data[0] = R_00B848_COMPUTE_PGM_RSRC1;
   data[1] = S_00B028_VGPRS(DIV_ROUND_UP(config->num_vgprs, vgpr_alloc) - 1) |
             S_00B028_SGPRS(DIV_ROUND_UP(config->num_sgprs, 8) - 1) |
             S_00B028_FLOAT_MODE(config->float_mode) |
             S_00B848_MEM_ORDERED(config->mem_ordered);

   data[2] = R_00B84C_COMPUTE_PGM_RSRC2;
   data[3] = S_00B84C_LDS_SIZE(
      ac_shader_encode_lds_size(config->lds_size, compiler_info->gfx_level, MESA_SHADER_COMPUTE));

   data[4] = R_00B8A0_COMPUTE_PGM_RSRC3;
   data[5] = S_00B8A0_SHARED_VGPR_CNT(config->num_shared_vgprs);

   data[6] = R_0286CC_SPI_PS_INPUT_ENA;
   data[7] = config->spi_ps_input_ena;

   data[8] = R_0286D0_SPI_PS_INPUT_ADDR;
   data[9] = config->spi_ps_input_addr;

   data[10] = R_00B860_COMPUTE_TMPRING_SIZE;
   data[11] = S_00B860_WAVESIZE(DIV_ROUND_UP(config->scratch_bytes_per_wave, scratch_alloc));

   data[12] = 0x4; /* SPILLED_SGPRS */
   data[13] = config->spilled_sgprs;

   data[14] = 0x8; /* SPILLED_VGPRS */
   data[15] = config->spilled_vgprs;

   aco_elf_add_section(elf, shdr, 4, data);
}

size_t
aco_create_elf(const ac_compiler_info* compiler_info, const aco_callback_params* params,
               size_t size_before, size_t size_after, void** data)
{
   struct aco_elf_writer elf;
   aco_elf_init(&elf);

   uint32_t config_data[16];
   add_config(&elf, &params->config, params->wave_size, compiler_info, config_data);

   /* Create a copy if we need to adjust the code for constant relocations. */
   std::vector<uint32_t> code;
   if (params->constants_size)
      code.insert(code.end(), params->code, params->code + params->exec_size);

   Elf64_Shdr code_shdr = {0};
   code_shdr.sh_name = aco_elf_add_string(&elf, ".text");
   code_shdr.sh_type = SHT_PROGBITS;
   code_shdr.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
   code_shdr.sh_addralign = 256;
   code_shdr.sh_size = params->exec_size;
   aco_elf_section* text =
      aco_elf_add_section(&elf, code_shdr, 4, code.size() ? code.data() : params->code);

   aco_elf_section* rodata = NULL;
   if (params->constants_size) {
      Elf64_Shdr const_shdr = {0};
      const_shdr.sh_name = aco_elf_add_string(&elf, ".rodata");
      const_shdr.sh_type = SHT_PROGBITS;
      const_shdr.sh_flags = SHF_ALLOC;
      const_shdr.sh_addralign = 4;
      const_shdr.sh_size = params->constants_size;
      rodata = aco_elf_add_section(&elf, const_shdr, 4, params->constants);
   }

   std::vector<Elf64_Sym> symbols;
   unsigned rodata_sym = 0;

   Elf64_Sym undef_sym = {0};
   undef_sym.st_shndx = SHN_UNDEF;
   symbols.push_back(undef_sym);

   if (params->config.lds_size) {
      Elf64_Sym lds_sym = {0};
      lds_sym.st_name = aco_elf_add_string(&elf, "lds");
      lds_sym.st_value = 65536;
      lds_sym.st_size = params->config.lds_size;
      lds_sym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
      lds_sym.st_other = STV_DEFAULT;
      lds_sym.st_shndx = 0xff00; /* SHN_AMDGPU_LDS */
      symbols.push_back(lds_sym);
   }

   if (rodata) {
      Elf64_Sym ro_sym = {0};
      ro_sym.st_name = aco_elf_add_string(&elf, "rodata");
      ro_sym.st_value = 0;
      ro_sym.st_size = params->constants_size;
      ro_sym.setBindingAndType(STB_LOCAL, STT_OBJECT);
      ro_sym.st_other = STV_DEFAULT;
      ro_sym.st_shndx = rodata->index;
      rodata_sym = symbols.size();
      symbols.push_back(ro_sym);
   }

   aco_elf_section* symtab = NULL;
   if (symbols.size() > 1) {
      Elf64_Shdr shdr = {0};
      shdr.sh_name = aco_elf_add_string(&elf, ".symtab");
      shdr.sh_type = SHT_SYMTAB;
      shdr.sh_link = elf.strtab_index;
      shdr.sh_entsize = sizeof(Elf64_Sym);
      shdr.sh_size = sizeof(Elf64_Sym) * symbols.size();
      symtab = aco_elf_add_section(&elf, shdr, 8, symbols.data());
   }

   std::vector<Elf64_Rel> relocs;
   for (size_t i = 0; i < params->num_symbols; i++) {
      const aco_symbol* sym = &params->symbols[i];
      if (sym->id != aco_symbol_const_data_addr || !rodata)
         continue;

      code[sym->offset] -= params->constants - (uint8_t*)params->code;
      code[sym->offset] += sym->offset * 4;

      Elf64_Rel rel;
      rel.r_offset = sym->offset * 4;
      rel.setSymbolAndType(rodata_sym, R_AMDGPU_REL32);
      relocs.push_back(rel);
   }

   if (!relocs.empty()) {
      Elf64_Shdr rel_shdr = {0};
      rel_shdr.sh_name = aco_elf_add_string(&elf, ".rel.text");
      rel_shdr.sh_type = SHT_REL;
      rel_shdr.sh_entsize = sizeof(Elf64_Rel);
      rel_shdr.sh_size = sizeof(Elf64_Rel) * relocs.size();
      rel_shdr.sh_info = text->index;
      rel_shdr.sh_link = symtab->index;
      aco_elf_add_section(&elf, rel_shdr, 8, relocs.data());
   }

   if (params->disasm_size) {
      Elf64_Shdr disasm_shdr = {0};
      disasm_shdr.sh_name = aco_elf_add_string(&elf, ".AMDGPU.disasm");
      disasm_shdr.sh_type = SHT_PROGBITS;
      disasm_shdr.sh_size = params->disasm_size;
      aco_elf_add_section(&elf, disasm_shdr, 1, params->disasm_str);
   }

   uint8_t stats_data[sizeof(struct amd_stats)];
   if (params->stats) {
      amd_stats_serialize(stats_data, params->stats);

      Elf64_Shdr stats_shdr = {0};
      stats_shdr.sh_name = aco_elf_add_string(&elf, ".ACO.stats");
      stats_shdr.sh_type = SHT_PROGBITS;
      stats_shdr.sh_size = sizeof(struct amd_stats);
      aco_elf_add_section(&elf, stats_shdr, 8, stats_data);
   }

   uint64_t e_flags = get_elf_eflags(compiler_info);

   size_t elf_size = aco_elf_get_size(&elf);
   uint8_t* binary = (uint8_t*)calloc(size_before + elf_size + size_after, 1);
   aco_elf_write(&elf, e_flags, binary + size_before);
   aco_elf_finish(&elf);

   *data = binary;
   return elf_size;
}
#else
size_t
aco_create_elf(const struct ac_compiler_info* compiler_info, const aco_callback_params* params,
               size_t size_before, size_t size_after, void** data)
{
   *data = calloc(size_before + size_after, 1);
   return 0;
}
#endif
