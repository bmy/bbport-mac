# Copyright © 2026 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT

pvr_tracepoint_files = custom_target(
  'pvr_tracepoints.[ch]',
  input : 'pvr_tracepoints.py',
  output : ['pvr_tracepoints.h',
            'pvr_tracepoints.c'],
  command : [
    prog_python, '@INPUT@',
    '-p', join_paths(dir_source_root, 'src/util/perf/'),
    '--utrace-hdr', '@OUTPUT0@',
    '--utrace-src', '@OUTPUT1@',
  ],
  depend_files : u_trace_py,
)

libpvr_driver_ds_deps = [
  idep_mesautil,
]

idep_pvr_driver_ds_headers = declare_dependency(
  sources : pvr_tracepoint_files[0],
  include_directories : [inc_include, inc_imagination],
)

libpvr_driver_ds = static_library(
  'pvr-driver-ds',
  sources : ['pvr_driver_ds.cc', pvr_tracepoint_files],
  include_directories : [inc_src, inc_include, inc_imagination],
  dependencies : [libpvr_driver_ds_deps],
  gnu_symbol_visibility : 'hidden',
)

idep_pvr_driver_ds = declare_dependency(
  link_with : libpvr_driver_ds,
  include_directories : [inc_include, inc_imagination],
  dependencies : idep_pvr_driver_ds_headers,
)
