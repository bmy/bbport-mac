# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import ctypes
import logging
from dataclasses import dataclass, field

import torch
from executorch.exir.backend.utils import exir_ops

from . import gallium
from .fold_qdq_with_annotated_qparams_pass import QuantArgs

supported_ops = {
    exir_ops.edge.aten.convolution.default: gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION,
    exir_ops.edge.aten.add.Tensor: gallium.PIPE_ML_OPERATION_TYPE_ADD,
}
# Activation ops that are fused into the preceding convolution or add.
# Two variants: FUSABLE_ACTIVATION_OPS uses ExIR edge-dialect op references
# (used during lowering and partitioning); FUSABLE_ACTIVATION_OPS_ATEN uses
# base-aten op references (used during PT2E quantizer annotation).
FUSABLE_ACTIVATION_OPS = {
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.relu_.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.hardtanh_.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.clamp_.default,
}
FUSABLE_ACTIVATION_OPS_ATEN = {
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_.default,
}
RELU_OPS = {
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.relu_.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
}


def activation_is_absorbable(activation: torch.fx.Node,
                             producer_is_conv: bool) -> bool:
    """Return whether producer's operation can absorb this activation.

    Convolutions carry arbitrary clamp bounds in activation_min/max; add
    only has a relu flag, so its clamp must start at zero and end at 6 or
    be unbounded above.
    """
    if activation.target in RELU_OPS:
        return True
    if producer_is_conv:
        return True
    low = activation.args[1] if len(activation.args) > 1 else None
    high = activation.args[2] if len(activation.args) > 2 else None
    return low == 0.0 and (high is None or high == 6.0)


def fused_activation(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the activation user absorbed by node's operation, if any."""
    if node.target not in supported_ops:
        return None
    users = list(node.users.keys())
    if len(users) != 1 or users[0].target not in FUSABLE_ACTIVATION_OPS:
        return None
    is_conv = (supported_ops[node.target] ==
               gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION)
    return users[0] if activation_is_absorbable(users[0], is_conv) else None


# Identity quantization parameters — a mathematical no-op (scale=1.0, zp=0).
# Used as placeholder when FoldAndAnnotateQParamsPass hasn't run yet
# (e.g. during partitioning), so operation structs carry valid scale/zp
# fields for ml_operation_supported().
IDENTITY_QPARAMS = QuantArgs(
    scale=1.0, zp=0, qmin=-128, qmax=127, dtype=torch.int8
)

CLONE_OP = exir_ops.edge.dim_order_ops._clone_dim_order.default


def is_identity_clone(node: torch.fx.Node) -> bool:
    """Return whether node is a _clone_dim_order that keeps the dim order."""
    if node.target != CLONE_OP:
        return False
    val = node.meta.get("val")
    input_val = node.args[0].meta.get("val")
    return (val is not None and input_val is not None and
            val.dim_order() == input_val.dim_order())

@dataclass
class GalliumGraph:
    """Converted Gallium graph plus owned backing storage for ctypes pointers."""

    operations: list[gallium.pipe_ml_operation] = field(default_factory=list)
    node_to_tensor_idx: dict[torch.fx.Node, int] = field(default_factory=dict)
    next_tensor_id: int = -1
    _buffers: list[object] = field(default_factory=list, repr=False, compare=False)

    def own(self, *buffers: object):
        """Keep driver-visible storage alive until release_buffers()."""
        self._buffers.extend(buffers)

    def release_buffers(self):
        """Free owned storage once the driver holds its own copies."""
        self._buffers.clear()

    def allocate_tensor_id(self) -> int:
        self.next_tensor_id += 1
        return self.next_tensor_id

    def resolve_idx(self, node: torch.fx.Node, default: int | None = None) -> int | None:
        """Return Gallium tensor index for a node, or default if unknown."""
        gidx = self.node_to_tensor_idx.get(node)
        return gidx if gidx is not None else default

def _require_quant_args_dtype_in(
    qparams: QuantArgs, role: str, allowed_dtypes: tuple[torch.dtype, ...]
) -> None:
    if qparams.dtype not in allowed_dtypes:
        allowed = ", ".join([str(dtype) for dtype in allowed_dtypes])
        raise ValueError(
            f"Torx currently supports quantized int8 models only. "
            f"Expected {role} quantization dtype(s) {allowed}, got {qparams.dtype}"
        )

def _derive_bias_qparams(input_qparams: QuantArgs, weight_qparams: QuantArgs) -> QuantArgs:
    if input_qparams.per_channel:
        raise ValueError("Per-channel activation quantization is not supported for bias derivation")

    qmin = torch.iinfo(torch.int32).min
    qmax = torch.iinfo(torch.int32).max

    if weight_qparams.per_channel:
        input_scale = input_qparams.get_scale_per_tensor()
        weight_scales = weight_qparams.get_scale_per_channel()
        scales = [input_scale * weight_scale for weight_scale in weight_scales]
        return QuantArgs(
            scale=scales,
            zp=[0 for _ in scales],
            qmin=qmin,
            qmax=qmax,
            dtype=torch.int32,
            axis=weight_qparams.axis,
            per_channel=True,
        )

    scale = input_qparams.get_scale_per_tensor() * weight_qparams.get_scale_per_tensor()
    return QuantArgs(
        scale=scale,
        zp=0,
        qmin=qmin,
        qmax=qmax,
        dtype=torch.int32,
        axis=0,
        per_channel=False,
    )

def fill_tensor(tensor: torch._subclasses.fake_tensor.FakeTensor, ptensor: gallium.pipe_tensor, graph: GalliumGraph, qparams: QuantArgs | None = None, *, index: int | None = None, is_constant: bool = False):
    translation = [0, 2, 3, 1]  # NCHW to NHWC

    if index is not None:
        ptensor.index = index
    else:
        ptensor.index = graph.allocate_tensor_id()

    ptensor.rank = tensor.ndim
    ptensor.is_constant = is_constant

    if qparams is not None and qparams.per_channel:
        scales = (ctypes.c_float * len(qparams.scale))(*qparams.scale)
        zero_points = (ctypes.c_int * len(qparams.zp))(*qparams.zp)
        ptensor.scales = ctypes.cast(scales, ctypes.POINTER(ctypes.c_float))
        ptensor.zero_points = ctypes.cast(zero_points, ctypes.POINTER(ctypes.c_int))
        ptensor.scale = 0.0
        ptensor.zero_point = 0
        graph.own(scales, zero_points)
    elif qparams is not None:
        ptensor.scale = qparams.scale
        ptensor.zero_point = qparams.zp

    if qparams is not None and qparams.dtype in (torch.qint8, torch.qint32, torch.int8, torch.int32):
        ptensor.is_signed = True
    elif qparams is not None and qparams.dtype == torch.quint8:
        ptensor.is_signed = False
    else:
        ptensor.is_signed = tensor.dtype == torch.qint8

    _dtype_to_size = {
        torch.int8: 1, torch.qint8: 1, torch.quint8: 1,
        torch.int32: 4, torch.qint32: 4,
        torch.float32: 4,
    }
    if qparams is not None and qparams.dtype in _dtype_to_size:
        ptensor.type_size = _dtype_to_size[qparams.dtype]
    elif tensor.dtype in _dtype_to_size:
        ptensor.type_size = _dtype_to_size[tensor.dtype]
    else:
        ptensor.type_size = tensor.element_size()

    for i in range(4):
        if tensor.ndim == 4:
            ptensor.dims[i] = tensor.shape[translation[i]]
        elif i < tensor.ndim:
            ptensor.dims[i] = tensor.shape[i]
        else:
            ptensor.dims[i] = 1

def _set_tensor_data(graph: GalliumGraph, ptensor: gallium.pipe_tensor, data: torch.Tensor):
    """Point ptensor.data at the tensor's storage, which the graph owns."""
    graph.own(data)
    ptensor.data = ctypes.cast(ctypes.c_void_p(data.data_ptr()),
                               ctypes.POINTER(ctypes.c_uint8))


def fill_conv(node: torch.fx.Node, poperation: gallium.pipe_ml_operation, graph: GalliumGraph, input_qparams: QuantArgs, weight_qparams: QuantArgs, bias_qparams: QuantArgs):
    weights = node.args[1]

    # Detect depthwise early — needed for weight layout decisions below.
    groups = node.args[8]
    input_channels = node.args[0].meta['val'].shape[1]  # NCHW in ExIR
    is_depthwise = (groups == input_channels and input_channels > 1)

    poperation.conv.weight_tensor = ctypes.pointer(gallium.pipe_tensor())
    fill_tensor(weights.meta['val'], poperation.conv.weight_tensor.contents, graph, weight_qparams, is_constant=True)

    if is_depthwise:
        # The Ethos-U NPU (and Teflon/TFLite) expects depthwise weights in
        # [1, kH, kW, C_out] layout.  PyTorch shape is [C_out, 1, kH, kW];
        # fill_tensor's NCHW→NHWC translation produces dims [C_out, kH, kW, 1],
        # which is wrong.  Override dims to match the expected layout.
        weight_shape = weights.meta['val'].shape  # [C_out, 1, kH, kW]
        wt = poperation.conv.weight_tensor.contents
        wt.dims[0] = 1
        wt.dims[1] = weight_shape[2]  # kH
        wt.dims[2] = weight_shape[3]  # kW
        wt.dims[3] = weight_shape[0]  # C_out

    if 'torx_data' in weights.meta:
        if is_depthwise:
            # PyTorch [C_out, 1, kH, kW] → [1, kH, kW, C_out] to match TFLite/Ethos-U
            contents = weights.meta['torx_data'].permute(1, 2, 3, 0).contiguous()
        else:
            # PyTorch stores Conv2d weights in OIHW format; the Gallium driver
            # expects OHWI.  The graph owns the transposed copy until the
            # driver has read it.
            contents = weights.meta['torx_data'].permute(0, 2, 3, 1).contiguous()
        _set_tensor_data(graph, poperation.conv.weight_tensor.contents, contents)

    biases = node.args[2] if len(node.args) > 2 else None
    if biases is not None:
        poperation.conv.bias_tensor = ctypes.pointer(gallium.pipe_tensor())
        # The Ethos-U NPU expects biases quantized with scale = input_scale * weight_scale.
        # Use derived qparams (not the independently observed ones from PyTorch).
        driver_bias_qp = _derive_bias_qparams(input_qparams, weight_qparams)
        fill_tensor(biases.meta['val'], poperation.conv.bias_tensor.contents, graph, driver_bias_qp, is_constant=True)
        if 'torx_data' in biases.meta:
            # Re-quantize bias from PyTorch's observed scale to the scale the NPU
            # expects (input_scale * weight_scale).
            bias_data = biases.meta['torx_data']
            src_zp = torch.tensor(bias_qparams.zp, dtype=torch.float64)
            src_scale = torch.tensor(bias_qparams.scale, dtype=torch.float64)
            bias_float = (bias_data.to(torch.float64) - src_zp) * src_scale
            if driver_bias_qp.per_channel:
                scales = torch.tensor(driver_bias_qp.scale, dtype=torch.float64)
                bias_requantized = torch.round(bias_float / scales).clamp(
                    torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
                ).to(torch.int32).contiguous()
            else:
                bias_requantized = torch.round(bias_float / driver_bias_qp.scale).clamp(
                    torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
                ).to(torch.int32).contiguous()
            _set_tensor_data(graph, poperation.conv.bias_tensor.contents,
                             bias_requantized)
    else:
        poperation.conv.bias_tensor = None

    # aten.convolution.default args layout:
    #   0: input, 1: weight, 2: bias, 3: stride, 4: padding,
    #   5: dilation, 6: transposed, 7: output_padding, 8: groups

    # Stride — args[3] is [stride_h, stride_w]
    stride = node.args[3]
    poperation.conv.stride_y = stride[0]
    poperation.conv.stride_x = stride[1]

    # Dilation — args[5] is [dilation_h, dilation_w]
    dilation = node.args[5]
    poperation.conv.dilation_height_factor = dilation[0]
    poperation.conv.dilation_width_factor = dilation[1]

    # Padding — args[4] is [pad_h, pad_w] (symmetric in PyTorch).
    #
    # PyTorch specifies symmetric padding: pad_h is added on BOTH top and
    # bottom, pad_w on BOTH left and right.  However the Ethos-U NPU needs
    # the actual per-side padding that each output position will use.
    #
    # With stride > 1 the last kernel window may not reach the right/bottom
    # padding at all, so we derive pad_right/pad_bottom from the geometry
    # instead of blindly mirroring the PyTorch value.  pad_left/pad_top
    # are kept as-is because they determine the kernel alignment (the first
    # output position starts at -pad_left/-pad_top).
    padding = node.args[4]
    poperation.conv.padding_top = padding[0]
    poperation.conv.padding_left = padding[1]

    input_shape = node.args[0].meta['val'].shape   # NCHW
    output_shape = node.meta['val'].shape           # NCHW
    weight_shape = weights.meta['val'].shape        # [O, I/g, kH, kW]

    eff_kernel_h = (weight_shape[2] - 1) * dilation[0] + 1
    eff_kernel_w = (weight_shape[3] - 1) * dilation[1] + 1

    poperation.conv.padding_bottom = max(
        0, (output_shape[2] - 1) * stride[0] - padding[0] + eff_kernel_h - input_shape[2]
    )
    poperation.conv.padding_right = max(
        0, (output_shape[3] - 1) * stride[1] - padding[1] + eff_kernel_w - input_shape[3]
    )

    poperation.conv.depthwise = is_depthwise

    # Pointwise — 1×1 kernel (weight shape is [out_ch, in_ch/groups, kH, kW] in ExIR)
    weight_shape = weights.meta['val'].shape
    poperation.conv.pointwise = (weight_shape[2] == 1 and weight_shape[3] == 1)

    # The relu flag only describes a true relu; bounded activations are
    # carried by activation_min/max instead.
    users = list(node.users.keys())
    poperation.conv.relu = (
        len(users) == 1 and users[0].target in (exir_ops.edge.aten.relu.default,
                                                exir_ops.edge.aten.relu_.default)
    )


def _fused_activation_bounds(node: torch.fx.Node) -> tuple[float | None, float | None]:
    activation = fused_activation(node)
    if activation is None:
        return None, None

    if activation.target in RELU_OPS:
        return 0.0, None
    low = activation.args[1] if len(activation.args) > 1 else None
    high = activation.args[2] if len(activation.args) > 2 else None
    return low, high


def set_conv_activation_range(node: torch.fx.Node,
                              poperation: gallium.pipe_ml_operation,
                              output_qp: QuantArgs):
    scale = output_qp.get_scale_per_tensor()
    zp = output_qp.zp

    poperation.conv.activation_min = output_qp.qmin
    poperation.conv.activation_max = output_qp.qmax

    low, high = _fused_activation_bounds(node)
    # int(x + 0.5) matches teflon's C rounding, not Python's round().
    if low is not None:
        poperation.conv.activation_min = max(output_qp.qmin,
                                             zp + int(low / scale + 0.5))
    if high is not None:
        poperation.conv.activation_max = min(output_qp.qmax,
                                             zp + int(high / scale + 0.5))

def exir_to_operation(node: torch.fx.Node, graph: GalliumGraph) -> gallium.pipe_ml_operation | None:
    if node.target not in supported_ops:
        return None

    # Qparams are populated by FoldAndAnnotateQParamsPass during preprocess().
    # During partitioning the pass hasn't run yet, so we fall back to identity
    # quantization (scale=1.0, zp=0) — enough for ml_operation_supported().
    input_qparams = node.meta.get("input_qparams", {})
    output_qparams = node.meta.get("output_qparams", {})

    # When a fusable activation follows this node, FoldAndAnnotateQParamsPass
    # places output_qparams on the *activation* node (since Q follows it, not
    # the conv).  Propagate those qparams here so the conv's output tensor
    # gets the correct scale/zp for the NPU.
    if not output_qparams:
        activation = fused_activation(node)
        if activation is not None:
            output_qparams = activation.meta.get("output_qparams", {})

    # The runtime backend only maps 4-D delegate IO tensors.
    output_val = node.meta.get('val')
    if not isinstance(output_val, torch.Tensor) or output_val.dim() != 4:
        return None

    operation = gallium.pipe_ml_operation()
    operation.type = supported_ops[node.target]

    is_add = (operation.type == gallium.PIPE_ML_OPERATION_TYPE_ADD)
    num_inputs = 2 if is_add else 1
    operation.input_count = num_inputs
    operation.input_tensors = (ctypes.POINTER(gallium.pipe_tensor) * operation.input_count)()

    for inp_idx in range(num_inputs):
        input_node = node.args[inp_idx]
        if not isinstance(input_node, torch.fx.Node):
            return None
        input_tensor = input_node.meta['val']
        if input_tensor.dim() != 4:
            return None
        input_qp = input_qparams.get(inp_idx, IDENTITY_QPARAMS)
        _require_quant_args_dtype_in(input_qp, f"input{inp_idx}", (torch.qint8, torch.int8))
        operation.input_tensors[inp_idx] = ctypes.pointer(gallium.pipe_tensor())
        existing_idx = graph.node_to_tensor_idx.get(input_node)
        fill_tensor(input_tensor, operation.input_tensors[inp_idx].contents, graph, input_qp, index=existing_idx)
        if input_node not in graph.node_to_tensor_idx:
            graph.node_to_tensor_idx[input_node] = operation.input_tensors[inp_idx].contents.index

    operation.output_count = 1
    operation.output_tensors = (ctypes.POINTER(gallium.pipe_tensor) * operation.output_count)()

    output_tensor = node.meta['val']
    output_qp = output_qparams.get(0, IDENTITY_QPARAMS)
    _require_quant_args_dtype_in(output_qp, "output", (torch.qint8, torch.int8))
    operation.output_tensors[0] = ctypes.pointer(gallium.pipe_tensor())
    fill_tensor(output_tensor, operation.output_tensors[0].contents, graph, output_qp)
    graph.node_to_tensor_idx[node] = operation.output_tensors[0].contents.index

    if operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION:
        weight_qp = input_qparams.get(1, IDENTITY_QPARAMS)
        _require_quant_args_dtype_in(weight_qp, "weights", (torch.qint8, torch.int8))

        # Bias is optional — node.args[2] can be None for Conv2d(bias=False)
        has_bias = len(node.args) > 2 and node.args[2] is not None
        if has_bias:
            bias_qp = input_qparams.get(2)
            if bias_qp is None:
                bias_qp = _derive_bias_qparams(input_qp, weight_qp)
            _require_quant_args_dtype_in(bias_qp, "bias", (torch.qint32, torch.int32))
        else:
            bias_qp = None

        fill_conv(node, operation, graph, input_qp, weight_qp, bias_qp)
        set_conv_activation_range(node, operation, output_qp)

    elif operation.type == gallium.PIPE_ML_OPERATION_TYPE_ADD:
        operation.add.relu = fused_activation(node) is not None

    return operation


def is_torx_convertible(node: torch.fx.Node) -> bool:
    """Return whether a node can be lowered to a Gallium ML operation."""
    return exir_to_operation(node, GalliumGraph()) is not None


def is_torx_supported(node: torch.fx.Node, ml_device) -> bool:
    """Return whether a node is both lowerable and accepted by the ML device."""
    graph = GalliumGraph()
    operation = exir_to_operation(node, graph)
    if operation is None:
        return False
    return ml_device.contents.ml_operation_supported(ml_device, operation)

def exir_to_gallium(nodes: list[torch.fx.Node]) -> GalliumGraph:
    """Translate supported ExIR nodes into a GalliumGraph.

    The returned object owns both translated operations and backing storage
    required by ctypes pointers embedded in those operations.
    """
    graph = GalliumGraph()
    for n in nodes:
        if n.target in FUSABLE_ACTIVATION_OPS:
            producer = n.args[0]
            if (isinstance(producer, torch.fx.Node) and
                    fused_activation(producer) is n):
                continue
            raise RuntimeError(
                f"Torx: activation '{n.name}' cannot be absorbed by its "
                f"producer and has no standalone lowering")
        operation = exir_to_operation(n, graph)
        if operation is not None:
            if operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION:
                if not operation.conv.weight_tensor.contents.data:
                    raise RuntimeError(f"Torx: convolution '{n.name}' has no weight data")
                if operation.conv.bias_tensor and not operation.conv.bias_tensor.contents.data:
                    raise RuntimeError(f"Torx: convolution '{n.name}' has no bias data")
            else:
                for arg in n.args:
                    if isinstance(arg, torch.fx.Node) and 'torx_data' in arg.meta:
                        raise RuntimeError(
                            f"Torx: '{n.name}' reads constant tensor '{arg.name}', "
                            f"which only convolutions support")
            graph.operations.append(operation)
            # Alias a fused activation to the operation's output index so
            # subsequent consumers reuse the same tensor.
            activation = fused_activation(n)
            if activation is not None:
                graph.node_to_tensor_idx[activation] = operation.output_tensors[0].contents.index
    return graph

ml_device = None

def get_ml_device() -> gallium.pipe_ml_device | None:
    global ml_device

    if ml_device:
        return ml_device

    ml_device = gallium.torx_device_probe()

    if not ml_device:
        raise RuntimeError("ML device not found")

    logging.debug(f"ML device found: {ml_device.contents.id.decode()}")

    return ml_device