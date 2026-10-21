torx_py_sources = files(
  '__init__.py',
  'backend.py',
  'debug.py',
  'decompose_quant_nodes_pass.py',
  'exir_to_gallium.py',
  'fold_qdq_with_annotated_qparams_pass.py',
  'gallium.py',
  'partitioner.py',
  'quantizer.py',
)

py.install_sources(
  torx_py_sources,
  subdir: 'torx',
)

libtorx_files = files(
    'TorxBackend.cpp',
    'torx_backend.c',
)

libtorx_st = static_library(
  'torx_st',
  [libtorx_files, sha1_h],
  gnu_symbol_visibility : 'hidden',
  include_directories : [ inc_include, inc_src, inc_util, inc_gallium, inc_gallium_aux ],
  dependencies : [ dep_executorch.partial_dependency(compile_args: true, includes: true) ],
)
