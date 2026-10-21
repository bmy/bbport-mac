# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

"""Hand-written ctypes bindings for the Gallium ML interface.

Only the types used by the Torx ExecuTorch backend are defined here.
Covers pipe_tensor, pipe_ml_operation, pipe_ml_subgraph, pipe_ml_device,
and the torx_device_probe() entry point.  Derived from
src/gallium/include/pipe/p_state.h and src/gallium/targets/torx/torx_device.h.
"""

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint,
    c_ubyte,
    c_void_p,
)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_libtorx = ctypes.CDLL("libtorx.so")

# ---------------------------------------------------------------------------
# pipe_tensor
# ---------------------------------------------------------------------------

class pipe_tensor(Structure):
    pass

pipe_tensor._fields_ = [
    ("data",        POINTER(c_ubyte)),
    ("index",       c_uint),
    ("dims",        c_uint * 4),
    ("rank",        c_uint),
    ("scale",       c_float),
    ("scales",      POINTER(c_float)),
    ("zero_point",  c_int),
    ("zero_points", POINTER(c_int)),
    ("is_signed",   c_bool),
    ("is_constant", c_bool),
    ("is_external_output", c_bool),
    ("type_size",   c_ubyte),
]

# ---------------------------------------------------------------------------
# pipe_ml_operation_type constants  (enum pipe_ml_operation_type)
# ---------------------------------------------------------------------------

PIPE_ML_OPERATION_TYPE_ADD             = 0
PIPE_ML_OPERATION_TYPE_CONVOLUTION     = 1
PIPE_ML_OPERATION_TYPE_POOLING         = 2
PIPE_ML_OPERATION_TYPE_CONCATENATION   = 3
PIPE_ML_OPERATION_TYPE_PACK            = 4
PIPE_ML_OPERATION_TYPE_SPLIT           = 5
PIPE_ML_OPERATION_TYPE_UNPACK          = 6
PIPE_ML_OPERATION_TYPE_PAD             = 7
PIPE_ML_OPERATION_TYPE_SCATTER_ND      = 8
PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED = 9
PIPE_ML_OPERATION_TYPE_BATCH_MATMUL    = 10
PIPE_ML_OPERATION_TYPE_RESHAPE         = 11
PIPE_ML_OPERATION_TYPE_RELU            = 12
PIPE_ML_OPERATION_TYPE_ABSOLUTE        = 13
PIPE_ML_OPERATION_TYPE_LOGISTIC        = 14
PIPE_ML_OPERATION_TYPE_TANH            = 15
PIPE_ML_OPERATION_TYPE_RSQRT           = 16
PIPE_ML_OPERATION_TYPE_HSWISH          = 17
PIPE_ML_OPERATION_TYPE_SUBTRACT        = 18
PIPE_ML_OPERATION_TYPE_TRANSPOSE       = 19
PIPE_ML_OPERATION_TYPE_STRIDED_SLICE   = 20
PIPE_ML_OPERATION_TYPE_RESIZE          = 21
PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR = 22
PIPE_ML_OPERATION_TYPE_ARGMAX          = 23
PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH  = 24
PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE  = 25
PIPE_ML_OPERATION_TYPE_MUL             = 26
PIPE_ML_OPERATION_TYPE_LEAKY_RELU      = 27
PIPE_ML_OPERATION_TYPE_QUANTIZE        = 28
PIPE_ML_OPERATION_TYPE_MAXIMUM         = 29
PIPE_ML_OPERATION_TYPE_MINIMUM         = 30
PIPE_ML_OPERATION_TYPE_SOFTMAX         = 31
PIPE_ML_OPERATION_TYPE_MEAN            = 32

# ---------------------------------------------------------------------------
# Anonymous sub-structs for the pipe_ml_operation union
# ---------------------------------------------------------------------------

class _conv(Structure):
    _fields_ = [
        ("weight_tensor",          POINTER(pipe_tensor)),
        ("bias_tensor",            POINTER(pipe_tensor)),
        ("stride_x",               c_uint),
        ("stride_y",               c_uint),
        ("padding_top",            c_uint),
        ("padding_bottom",         c_uint),
        ("padding_left",           c_uint),
        ("padding_right",          c_uint),
        ("pointwise",              c_bool),
        ("depthwise",              c_bool),
        ("activation_min",         c_int),
        ("activation_max",         c_int),
        ("relu",                   c_bool),
        ("dilation_width_factor",  c_uint),
        ("dilation_height_factor", c_uint),
    ]

class _add(Structure):
    _fields_ = [("relu", c_bool)]

# Only the union members Torx sets are mirrored. conv is the largest
# member and fixes the union's size, which _check_abi() verifies against
# the C definition.
class _op_union(Union):
    _fields_ = [
        ("conv", _conv),
        ("add",  _add),
    ]

# ---------------------------------------------------------------------------
# pipe_ml_operation
# ---------------------------------------------------------------------------

class pipe_ml_operation(Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type",           c_int),
        ("input_tensors",  POINTER(POINTER(pipe_tensor))),
        ("input_count",    c_uint),
        ("output_tensors", POINTER(POINTER(pipe_tensor))),
        ("output_count",   c_uint),
        ("_u",             _op_union),
    ]

# ---------------------------------------------------------------------------
# pipe_ml_subgraph / pipe_ml_device  (forward-declared, fields set below)
# ---------------------------------------------------------------------------

class pipe_ml_device(Structure):
    pass

class pipe_ml_subgraph(Structure):
    pass

pipe_ml_subgraph._fields_ = [
    ("device", POINTER(pipe_ml_device)),
]

# Function pointer return values that are pointers use c_void_p so that
# callers can cast them (matches ctypesgen's UNCHECKED() wrapper behaviour).
pipe_ml_device._fields_ = [
    ("id",                    c_char_p),
    ("ml_operation_supported", CFUNCTYPE(c_bool,
        POINTER(pipe_ml_device), POINTER(pipe_ml_operation))),
    ("ml_subgraph_create",    CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_operation), c_uint)),
    ("ml_subgraph_serialize", CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_subgraph), POINTER(c_size_t))),
    ("ml_subgraph_destroy",   CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_subgraph))),
    ("ml_device_destroy",     CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device))),
]

# ---------------------------------------------------------------------------
# torx_device_probe
# ---------------------------------------------------------------------------

torx_device_probe = _libtorx.torx_device_probe
torx_device_probe.argtypes = []
torx_device_probe.restype = POINTER(pipe_ml_device)


# ---------------------------------------------------------------------------
# ABI self-check against torx_abi_layout() in torx_device.c
# ---------------------------------------------------------------------------

_torx_abi_layout = _libtorx.torx_abi_layout
_torx_abi_layout.argtypes = [POINTER(c_uint)]
_torx_abi_layout.restype = POINTER(ctypes.c_uint32)


def _check_abi():
    """Compare this mirror's layout with the one libtorx was built with.

    The entries here and in torx_abi_layout() must stay in the same
    order; a count or value mismatch means the mirror is out of sync
    with p_state.h.
    """
    expected = [
        ("sizeof(pipe_tensor)", ctypes.sizeof(pipe_tensor), True),
        ("pipe_tensor.data", pipe_tensor.data.offset, True),
        ("pipe_tensor.index", pipe_tensor.index.offset, True),
        ("pipe_tensor.dims", pipe_tensor.dims.offset, True),
        ("pipe_tensor.rank", pipe_tensor.rank.offset, True),
        ("pipe_tensor.scale", pipe_tensor.scale.offset, True),
        ("pipe_tensor.scales", pipe_tensor.scales.offset, True),
        ("pipe_tensor.zero_point", pipe_tensor.zero_point.offset, True),
        ("pipe_tensor.zero_points", pipe_tensor.zero_points.offset, True),
        ("pipe_tensor.is_signed", pipe_tensor.is_signed.offset, True),
        ("pipe_tensor.is_constant", pipe_tensor.is_constant.offset, True),
        ("pipe_tensor.is_external_output",
         pipe_tensor.is_external_output.offset, True),
        ("pipe_tensor.type_size", pipe_tensor.type_size.offset, True),
        ("sizeof(pipe_ml_operation)", ctypes.sizeof(pipe_ml_operation), True),
        ("pipe_ml_operation.type", pipe_ml_operation.type.offset, True),
        ("pipe_ml_operation.input_tensors",
         pipe_ml_operation.input_tensors.offset, True),
        ("pipe_ml_operation.input_count",
         pipe_ml_operation.input_count.offset, True),
        ("pipe_ml_operation.output_tensors",
         pipe_ml_operation.output_tensors.offset, True),
        ("pipe_ml_operation.output_count",
         pipe_ml_operation.output_count.offset, True),
        ("pipe_ml_operation.conv.weight_tensor",
         pipe_ml_operation.conv.offset + _conv.weight_tensor.offset, True),
        ("pipe_ml_operation.conv.bias_tensor",
         pipe_ml_operation.conv.offset + _conv.bias_tensor.offset, True),
        ("pipe_ml_operation.conv.stride_x",
         pipe_ml_operation.conv.offset + _conv.stride_x.offset, True),
        ("pipe_ml_operation.conv.padding_top",
         pipe_ml_operation.conv.offset + _conv.padding_top.offset, True),
        ("pipe_ml_operation.conv.pointwise",
         pipe_ml_operation.conv.offset + _conv.pointwise.offset, True),
        ("pipe_ml_operation.conv.activation_min",
         pipe_ml_operation.conv.offset + _conv.activation_min.offset, True),
        ("pipe_ml_operation.conv.activation_max",
         pipe_ml_operation.conv.offset + _conv.activation_max.offset, True),
        ("pipe_ml_operation.conv.relu",
         pipe_ml_operation.conv.offset + _conv.relu.offset, True),
        ("pipe_ml_operation.conv.dilation_width_factor",
         pipe_ml_operation.conv.offset + _conv.dilation_width_factor.offset,
         True),
        ("pipe_ml_operation.conv.dilation_height_factor",
         pipe_ml_operation.conv.offset + _conv.dilation_height_factor.offset,
         True),
        ("pipe_ml_operation.add.relu",
         pipe_ml_operation.add.offset + _add.relu.offset, True),
        # The mirror only covers a prefix of pipe_ml_device.
        ("sizeof(pipe_ml_device)", ctypes.sizeof(pipe_ml_device), False),
        ("pipe_ml_device.id", pipe_ml_device.id.offset, True),
        ("pipe_ml_device.ml_operation_supported",
         pipe_ml_device.ml_operation_supported.offset, True),
        ("pipe_ml_device.ml_subgraph_create",
         pipe_ml_device.ml_subgraph_create.offset, True),
        ("pipe_ml_device.ml_subgraph_serialize",
         pipe_ml_device.ml_subgraph_serialize.offset, True),
        ("pipe_ml_device.ml_subgraph_destroy",
         pipe_ml_device.ml_subgraph_destroy.offset, True),
        ("pipe_ml_device.ml_device_destroy",
         pipe_ml_device.ml_device_destroy.offset, True),
        ("PIPE_ML_OPERATION_TYPE_CONVOLUTION",
         PIPE_ML_OPERATION_TYPE_CONVOLUTION, True),
        ("PIPE_ML_OPERATION_TYPE_ADD", PIPE_ML_OPERATION_TYPE_ADD, True),
        ("PIPE_ML_OPERATION_TYPE_MINIMUM",
         PIPE_ML_OPERATION_TYPE_MINIMUM, True),
        ("PIPE_ML_OPERATION_TYPE_MEAN",
         PIPE_ML_OPERATION_TYPE_MEAN, True),
    ]

    count = c_uint()
    values = _torx_abi_layout(ctypes.byref(count))
    if count.value != len(expected):
        raise RuntimeError(
            f"Torx: gallium.py expects {len(expected)} ABI entries, libtorx "
            f"reports {count.value}; the ctypes mirror is out of sync with "
            f"p_state.h")
    for i, (name, want, exact) in enumerate(expected):
        got = values[i]
        if (got != want) if exact else (got < want):
            raise RuntimeError(
                f"Torx: ABI mismatch for {name}: gallium.py has {want}, "
                f"libtorx reports {got}; the ctypes mirror is out of sync "
                f"with p_state.h")


_check_abi()
