# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

"""Debug dumps of the graphs Torx works on, gated by TORX_DEBUG=verbose:
the ExIR graph with per-node support status, and the compiled Gallium
operation table."""

import os
import sys

import torch

from . import gallium


def debug_enabled():
    return "verbose" in os.environ.get("TORX_DEBUG", "")


def _dtype_short_name(dtype):
    """Return a short name for a torch dtype, e.g. 'i8', 'i32', 'f32'."""
    names = {
        torch.int8: "i8",
        torch.qint8: "i8",
        torch.uint8: "u8",
        torch.quint8: "u8",
        torch.int32: "i32",
        torch.qint32: "i32",
        torch.float32: "f32",
        torch.float16: "f16",
        torch.float64: "f64",
        torch.int64: "i64",
        torch.bool: "bool",
    }
    return names.get(dtype, str(dtype).replace("torch.", ""))


def _op_short_name(node):
    """Return a short human-readable name for an FX node's operation."""
    if node.op in ["placeholder", "output", "get_attr"]:
        return node.op

    target = node.target
    # Get the full qualified name for matching
    if hasattr(target, "_schema"):
        full_name = str(target)
    elif hasattr(target, "__name__"):
        full_name = target.__name__
    else:
        full_name = str(target)

    # Normalize to just the final op name for matching
    # e.g. "aten.convolution.default" -> "convolution"
    # e.g. "quantized_decomposed.quantize_per_tensor.default" -> "quantize_per_tensor"
    parts = full_name.split(".")
    if len(parts) >= 2:
        name = parts[-2]  # second-to-last part (before .default)
    else:
        name = parts[-1]

    # Map common ExIR op names to short Teflon-style labels
    op_names = {
        "convolution": "CONV",
        "conv2d": "CONV",
        "add": "ADD",
        "mul": "MUL",
        "relu": "RELU",
        "max_pool2d": "MAXPOOL",
        "avg_pool2d": "AVGPOOL",
        "pad": "PAD",
        "cat": "CAT",
        "linear": "FCON",
        "mean": "MEAN",
        "hardtanh": "RELU6",
        "clamp": "CLAMP",
        "reshape": "RESHAPE",
        "permute": "PERMUTE",
        "slice_copy": "SLICE",
        "quantize_per_tensor": "Q",
        "dequantize_per_tensor": "DQ",
        "quantize_per_channel": "Q_ch",
        "dequantize_per_channel": "DQ_ch",
    }
    return op_names.get(name, name.upper())


def node_tensor_dtype(node):
    """Get the dtype of a node's output tensor from its FakeTensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.dtype
    if isinstance(val, (tuple, list)) and len(val) > 0:
        if isinstance(val[0], torch.Tensor):
            return val[0].dtype
    return None


def _get_conv_extra_info(node):
    """Return extra info string for convolution nodes (depthwise, pointwise, stride, dilation)."""
    if node.op != "call_function":
        return ""
    name = getattr(node.target, "__name__", "")
    if name not in ("convolution", "conv2d"):
        return ""

    parts = []

    # args: 0=input, 1=weight, 2=bias, 3=stride, 4=padding, 5=dilation, 6=transposed, 7=output_padding, 8=groups
    if len(node.args) > 8:
        groups = node.args[8]
        input_channels = node.args[0].meta["val"].shape[1]
        if groups == input_channels and input_channels > 1:
            parts.append("depthwise")
        elif len(node.args) > 1:
            weight_shape = node.args[1].meta["val"].shape
            if weight_shape[2] == 1 and weight_shape[3] == 1:
                parts.append("pointwise")

    if len(node.args) > 3:
        stride = node.args[3]
        if any(s > 1 for s in stride):
            parts.append(f"stride={stride[0]}x{stride[1]}")

    if len(node.args) > 5:
        dilation = node.args[5]
        if any(d > 1 for d in dilation):
            parts.append(f"dil={dilation[0]}x{dilation[1]}")

    if len(node.args) > 4:
        padding = node.args[4]
        if any(p > 0 for p in padding):
            parts.append(f"pad={padding[0]},{padding[1]}")

    return " ".join(parts)


def dump_graph_support(graph_module, op_support):
    """Print a Teflon-style debug table of all graph nodes with support status.

    Controlled by TORX_DEBUG=verbose (analogous to TEFLON_DEBUG=verbose).
    Output format matches Teflon's operation table:

        idx     type  support     inputs
        ====================================...
          0     CONV  supported   in: 0(i8) 1(i8) 2(i32) out: 3(i8) stride=2x2
          1        Q  delegated   in: 3(i8) out: 4(i8)
          ...
    """
    if not debug_enabled():
        return

    # Assign integer indices to all nodes (like TFLite tensor indices)
    node_index = {}
    idx = 0
    for node in graph_module.graph.nodes:
        node_index[node] = idx
        idx += 1

    print(f"{'idx':>3s} {'type':>8s}  {'support':<11s} {'inputs'}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    op_idx = 0
    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        op_name = _op_short_name(node)
        supported = op_support.is_node_supported({}, node)
        support_str = "supported" if supported else "unsupported"

        # Collect input references as "index(dtype)"
        input_refs = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                dtype = node_tensor_dtype(arg)
                dtype_str = _dtype_short_name(dtype) if dtype else "?"
                input_refs.append(f"{node_index[arg]}({dtype_str})")
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        dtype = node_tensor_dtype(a)
                        dtype_str = _dtype_short_name(dtype) if dtype else "?"
                        input_refs.append(f"{node_index[a]}({dtype_str})")

        # Output reference
        dtype = node_tensor_dtype(node)
        dtype_str = _dtype_short_name(dtype) if dtype else "?"
        output_ref = f"{node_index[node]}({dtype_str})"

        # Extra info for convolutions
        extra = _get_conv_extra_info(node)

        line = f"{op_idx:3d} {op_name:>8s}  {support_str:<11s} in: {' '.join(input_refs)} out: {output_ref}"
        if extra:
            line += f" {extra}"
        print(line, file=sys.stderr)

        op_idx += 1


_PIPE_ML_OP_NAMES = {
    gallium.PIPE_ML_OPERATION_TYPE_ADD: "ADD",
    gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION: "CONV",
}

def dump_compiled_graph(operations, count):
    """Print the compiled Gallium operation table (analogous to Teflon's dump_graph).

    Shows each pipe_ml_operation with its type, input/output tensor indices,
    scales, and zero points.  Controlled by TORX_DEBUG=verbose.
    """
    if not debug_enabled():
        return

    print(f"\n{'idx':>3s} {'type':<8s} {'inputs':<30s} {'outputs':<30s} {'details'}", file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    for i in range(count):
        op = operations[i]
        op_name = _PIPE_ML_OP_NAMES.get(op.type, f"type_{op.type}")

        # Format input tensors
        inputs = []
        if op.input_tensors:
            for j in range(op.input_count):
                t = op.input_tensors[j].contents
                inputs.append(f"{t.index}[{'x'.join(str(t.dims[d]) for d in range(4) if t.dims[d] > 0)}]")
        in_str = " ".join(inputs) if inputs else "-"

        # Format output tensors
        outputs = []
        if op.output_tensors:
            for j in range(op.output_count):
                t = op.output_tensors[j].contents
                outputs.append(f"{t.index}[{'x'.join(str(t.dims[d]) for d in range(4) if t.dims[d] > 0)}]")
        out_str = " ".join(outputs) if outputs else "-"

        # Extra details by operation type
        details = ""
        if op.type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION:
            parts = []
            if op.conv.depthwise:
                parts.append("depthwise")
            if op.conv.pointwise:
                parts.append("pointwise")
            parts.append(f"stride={op.conv.stride_y}x{op.conv.stride_x}")
            parts.append(f"pad={op.conv.padding_top},{op.conv.padding_bottom},{op.conv.padding_left},{op.conv.padding_right}")
            if op.conv.dilation_height_factor > 1 or op.conv.dilation_width_factor > 1:
                parts.append(f"dil={op.conv.dilation_height_factor}x{op.conv.dilation_width_factor}")
            if op.conv.relu:
                parts.append("ReLU")
            if op.conv.weight_tensor:
                wt = op.conv.weight_tensor.contents
                parts.append(f"w:{wt.index}[{'x'.join(str(wt.dims[d]) for d in range(4) if wt.dims[d] > 0)}]")
            if op.conv.bias_tensor:
                bt = op.conv.bias_tensor.contents
                parts.append(f"b:{bt.index}")
            details = " ".join(parts)
        elif op.type == gallium.PIPE_ML_OPERATION_TYPE_ADD:
            details = "ReLU" if op.add.relu else ""

        print(f"{i:3d} {op_name:<8s} {in_str:<30s} {out_str:<30s} {details}", file=sys.stderr)

    print(file=sys.stderr)
