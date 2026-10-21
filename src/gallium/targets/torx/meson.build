libtorx_files = files(
   'torx_device.c',
   'torx_target.c',
)

libtorx = shared_library(
  'torx',
  [libtorx_files],
  include_directories : [ inc_src, inc_util, inc_include, inc_gallium, inc_gallium_aux, inc_gallium_winsys, inc_gallium_drivers ],
  link_whole : [ libtorx_st ],
  link_with : [ libpipe_loader_static, libws_null, libwsw, libswdri, libswkmsdri, libgallium ],
  gnu_symbol_visibility : 'hidden',
  link_args : ld_args_build_id,
  override_options : ['b_lundef=false'],
  dependencies : [
    dep_executorch,
    driver_etnaviv,
    driver_rocket,
    driver_ethosu,
    idep_nir,
    idep_mesautil,
  ],
  install : true,
)

if with_tests
  torx_targets_dir = meson.current_source_dir()
  torx_ml_devices = ['ethosu-65-256-98304', 'ethosu-85-256-2097152']
  torx_tests_dir = meson.current_build_dir() / 'tests'

  # This is just so modifications to the wrap file will trigger a reconfigure.
  fs.read(meson.project_source_root() / 'subprojects' / 'npu-model-zoo.wrap')

  zoo = subproject('npu-model-zoo', required : get_option('wrap_mode') != 'nodownload')
  torx_available_models = []
  torx_model_pt2 = {}
  torx_model_dir = {}
  if zoo.found()
    zoo_test_dir = zoo.get_variable('zoo_test_dir')
    foreach model, formats : zoo.get_variable('zoo_test_models')
      if 'pt2' in formats
        torx_available_models += model
        torx_model_pt2 += {model : formats['pt2']}
        torx_model_dir += {model : zoo_test_dir / model}
      endif
    endforeach
  else
    message('torx: the model zoo is not available, skipping the model tests')
  endif

  # Step 1: Split whole-model .pt2 into per-op .pt2 files + CPU references.
  # This is fast and sequential.
  torx_split_targets = {}
  foreach model : torx_available_models
    torx_split_targets += {model : custom_target(
      'split_torx_@0@'.format(model),
      command : [
        prog_python, files('tests/split_torx_tests.py'),
        torx_model_dir[model],
        torx_tests_dir,
        '@OUTPUT@',
      ],
      output : 'split_torx_@0@.stamp'.format(model),
      build_by_default : true,
      depend_files : [files('tests/split_torx_tests.py'), torx_model_pt2[model]],
    )}
  endforeach

  # Step 2: Compile the test artifacts for each model and device: every
  # per-op .pt2 (in parallel inside the script) and the whole model.
  compile_torx_tests_targets = []
  foreach device : torx_ml_devices
    foreach model : torx_available_models
      compile_torx_tests_targets += custom_target(
        'compile_torx_@0@_@1@'.format(device, model),
        command : [
          prog_python, files('tests/compile_torx_tests.py'),
          '--all',
          torx_tests_dir / model,
          '@OUTPUT@',
        ],
        output : 'compile_@0@_@1@.stamp'.format(device, model),
        build_by_default : true,
        depends : [libtorx, torx_split_targets[model]],
        depend_files : [torx_py_sources, files('tests/compile_torx_common.py', 'tests/compile_torx_tests.py')],
        env : {
          'LD_LIBRARY_PATH' : meson.current_build_dir(),
          'MESA_ML_DEVICE' : device,
        },
      )

      compile_torx_tests_targets += custom_target(
        'compile_model_@0@_@1@'.format(device, model),
        command : [
          prog_python, files('tests/compile_torx_tests.py'),
          '--whole-model',
          torx_model_pt2[model],
          torx_tests_dir / model,
          '@OUTPUT@',
        ],
        output : 'compile_model_@0@_@1@.stamp'.format(device, model),
        build_by_default : true,
        depends : [libtorx, torx_split_targets[model]],
        depend_files : [torx_py_sources, files('tests/compile_torx_common.py', 'tests/compile_torx_tests.py')],
        env : {
          'LD_LIBRARY_PATH' : meson.current_build_dir(),
          'MESA_ML_DEVICE' : device,
        },
      )
    endforeach
  endforeach

  if torx_available_models.length() > 0
    executable('test_torx',
      files('tests/test_torx.cpp', 'tests/test_torx_helpers.cpp'),
      include_directories : [ inc_src, inc_include, inc_gallium, inc_gallium_aux,
                              include_directories('tests'), include_directories('../../frontends/torx') ],
      link_with : libtorx,
      override_options : ['b_asneeded=false'],
      dependencies : [
        dep_executorch,
        dep_executorch_portable_ops,
        dep_executorch_quantized_ops,
        idep_gtest,
        idep_mesautil,
        idep_parson,
      ],
      install : false,
    )
  endif
endif