# Copyright © 2026 Collabora, Ltd.
# SPDX-License-Identifier: MIT

_util_rust_args = [
  rust_global_args,
  rust_2024_lint_args,
  '-Aclippy::identity_op',
  '-Aclippy::len_zero',
  '-Aclippy::manual_range_contains',
  # normally this is a good one, but we use it where the "better" code is worse
  '-Aclippy::needless_range_loop',
  '-Aclippy::redundant_field_names',
  '-Aclippy::upper_case_acronyms',
  '-Aclippy::vec_box',
  '-Aclippy::write_with_newline',
  # warns about public function might dereference a raw pointer, but nothing is
  # actually public here
  '-Aclippy::not_unsafe_ptr_arg_deref',
  '-Anon_snake_case',
]

_libmesa_util_rs = static_library(
  'mesa_util',
  files('lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : _util_rust_args,
)

idep_mesa_util_rs = declare_dependency(
  link_with : _libmesa_util_rs,
)
