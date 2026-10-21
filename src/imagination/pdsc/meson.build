# Copyright © 2025 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT

inc_powervr_pdsc = include_directories(['.'])

libpowervr_pdsc_files = files(
  'pdsc.c',
  'pdsc_binary.c',
  'pdsc_debug.c',
  'pdsc_ops.c',
  'pdsc_print.c',
  'pdsc_validate.c',
)

libpowervr_pdsc = static_library(
  'powervr_pdsc',
  [libpowervr_pdsc_files],
  include_directories : [
    inc_imagination,
    inc_powervr_pdsc,
    inc_include,
    inc_src,
  ],
  # Suppress 'parameter passing for argument of type ... changed in GCC ...' warnings.
  c_args : [imagination_c_args, no_override_init_args, '-Wno-psabi'],
  dependencies : [idep_mesautil],
  gnu_symbol_visibility : 'hidden',
  install : false,
)

if get_option('precomp-compiler') == 'system'
  prog_pdsc_precomp = find_program('pdsc_precomp', native : true)
else
  prog_pdsc_precomp = executable(
    'pdsc_precomp',
    ['pdsc_precomp.c'],
    link_with : [libpowervr_pdsc],
    include_directories : [
      inc_imagination,
      inc_powervr_pdsc,
      inc_include,
      inc_src,
    ],
    # Suppress 'parameter passing for argument of type ... changed in GCC ...' warnings.
    c_args : [imagination_c_args, no_override_init_args, '-Wno-psabi'],
    dependencies : [idep_mesautil],
    # If we can run host binaries directly, just build pdsc_precomp for the host.
    # Most commonly this happens when doing a cross compile from an x86_64 build
    # machine to an x86 host
    native : not meson.can_run_host_binaries(),
    install : get_option('install-precomp-compiler'),
  )
endif

pdsc_precomp_programs = custom_target(
  'pdsc_precomp_programs',
  output : ['pdsc_precomp_programs.h', 'pdsc_precomp_programs.c'],
  command : [prog_pdsc_precomp, '@OUTPUT@'],
)

idep_pdsc_precomp_programs_headers = declare_dependency(
  sources : [pdsc_precomp_programs[0]],
  include_directories : include_directories('.'),
)

idep_pdsc_precomp_programs = declare_dependency(
  sources : [pdsc_precomp_programs],
  include_directories : include_directories('.'),
)

libpowervr_pdsc_precomp = static_library(
  'pdsc_precomp',
  [pdsc_precomp_programs],
  include_directories : [inc_imagination],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  dependencies: [idep_mesautil],
  install : false,
)
