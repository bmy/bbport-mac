# Copyright © 2022 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT

imagination_c_args = cc.get_supported_arguments(
  '-Werror=pointer-arith',
)

inc_imagination = include_directories([
  '.',
  'common',
  'include',
])

subdir('csbgen')

if with_imagination_vk or with_tools.contains('imagination')
  subdir('common')
  subdir('pco')
  subdir('ds')
  subdir('pdsc')
endif

if with_imagination_vk
  subdir('vulkan')

  if with_tools.contains('drm-shim')
    subdir('drm-shim')
  endif
endif
