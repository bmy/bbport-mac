# Copyright 2024-2025 Arm Limited and/or its affiliates.
# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Vendored from ExecuTorch's
# backends/arm/_passes/fold_qdq_with_annotated_qparams_pass.py, which is
# not part of ExecuTorch's public API, and adapted for Torx.

import copy
from typing import Final, NamedTuple, cast

import torch
import executorch
from executorch.backends.transforms.utils import get_param_tensor, is_buffer, is_get_attr_node, is_lifted_tensor_constant, is_param
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.backend.backend_details import ExportedProgram

qd = executorch.exir.dialects._ops.ops.edge.quantized_decomposed

DEQUANT_PER_TENSOR_OP: Final = qd.dequantize_per_tensor.default
DEQUANT_PER_TENSOR_OP_T: Final = qd.dequantize_per_tensor.tensor
DEQUANT_PER_CHANNEL_OP: Final = qd.dequantize_per_channel.default
QUANT_PER_TENSOR_OP: Final = qd.quantize_per_tensor.default
QUANT_PER_TENSOR_OP_T: Final = qd.quantize_per_tensor.tensor
QUANT_PER_CHANNEL_OP: Final = qd.quantize_per_channel.default
DQ_OPS: Final = (DEQUANT_PER_TENSOR_OP, DEQUANT_PER_TENSOR_OP_T, DEQUANT_PER_CHANNEL_OP)
Q_OPS: Final = (QUANT_PER_TENSOR_OP, QUANT_PER_TENSOR_OP_T, QUANT_PER_CHANNEL_OP)

PER_TENSOR_QDQ_OPS: Final = (
    QUANT_PER_TENSOR_OP,
    QUANT_PER_TENSOR_OP_T,
    DEQUANT_PER_TENSOR_OP,
    DEQUANT_PER_TENSOR_OP_T,
)
PER_CHANNEL_QDQ_OPS: Final = (QUANT_PER_CHANNEL_OP, DEQUANT_PER_CHANNEL_OP)

class QuantArgs(NamedTuple):
    scale: list[float] | float
    zp: list[int] | int
    qmin: int
    qmax: int
    dtype: torch.dtype
    axis: int = 0
    per_channel: bool = False

    @classmethod
    def from_operator(cls, op, args):
        if op in PER_TENSOR_QDQ_OPS:
            return cls(
                scale=cast(float, args[1]),
                zp=cast(int, args[2]),
                qmin=cast(int, args[3]),
                qmax=cast(int, args[4]),
                dtype=cast(torch.dtype, args[5]),
                axis=0,
                per_channel=False,
            )
        elif op in PER_CHANNEL_QDQ_OPS:
            return cls(
                scale=cast(list[float], args[1].tolist()),
                zp=cast(list[int], args[2].tolist()),
                axis=cast(int, args[3]),
                qmin=cast(int, args[4]),
                qmax=cast(int, args[5]),
                dtype=cast(torch.dtype, args[6]),
                per_channel=True,
            )
        else:
            # We're only handling per tensor and per channel quantization
            raise NotImplementedError(f"Unsupported quantization operation: {op}")

    def get_scale_per_tensor(self) -> float:
        if not isinstance(self.scale, float):
            raise TypeError(
                f"Expected scale {self.scale} to be a float but found scale of "
                f"type {type(self.scale)}"
            )
        return self.scale

    def get_zp_per_tensor(self) -> int:
        if not isinstance(self.zp, int):
            raise TypeError(
                f"Expected zero point {self.zp} to be an int but found zp of "
                f"type {type(self.zp)}"
            )
        return self.zp

    def get_scale_per_channel(self) -> list[float]:
        if not isinstance(self.scale, list):
            raise TypeError(
                f"Expected scale {self.scale} to be a list but found scale of "
                f"type {type(self.scale)}"
            )
        return self.scale

    def get_zp_per_channel(self) -> list[int]:
        if not isinstance(self.zp, list):
            raise TypeError(
                f"Expected zero point {self.zp} to be a list but found zp of "
                f"type {type(self.zp)}"
            )
        return self.zp

def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_get_attr_node(node)
        or is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )

class FoldAndAnnotateQParamsPass(ExportPass):
    """
    A pass that walks the graph and removes any DQ and Q nodes before and after the target
     node.
     The quantization parameters from the DQ/Q nodes are stored as meta values to be
     accessible for later lowering and serialization passes.
     The assumption is that the quantization annotation adds DQ nodes for all tensor
     inputs to the target one Q node to the output.

     Example ('executorch_exir_dialects_edge__ops_' prefix removed from operators for readability):

        x_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(x, 0.05487706884741783, -128, -128, 127, torch.int8)

        x_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(x_q, 0.05487706884741783, -128, -128, 127, torch.int8)
        aten_add_tensor: "f32[5]" = ops_aten_add_Tensor(x_dq, x_dq)
        aten_add_tensor_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(aten_add_tensor, 0.05487706884741783, -128, -128, 127, torch.int8)

        output_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(aten_add_tensor_q, 0.05487706884741783, -128, -128, 127, torch.int8)

     Becomes:
        x_q: "i8[5]" = quantized_decomposed_quantize_per_tensor_default(x, 0.05487706884741783, -128, -128, 127, torch.int8)

        aten_add_tensor: "i8[5]" = aten_add_Tensor(x_q, x_q)

        output_dq: "f32[5]" = quantized_decomposed_dequantize_per_tensor_default(aten_add_tensor_q, 0.05487706884741783, -128, -128, 127, torch.int8)

    The quantization parameters for x_dq and aten_add_tensor_q are stored in meta for the aten_add_tensor node.

    """

    def __init__(self, exported_program: torch.export.ExportedProgram):
        super(FoldAndAnnotateQParamsPass, self).__init__()
        self.exported_program = exported_program

    def collect_arg_qparams(self, arg_list: list[torch.fx.Node]):
        """Return (ok, qparams, dq_nodes) for one argument group.

        Reads the graph without modifying it. ok is False when the group's
        inputs carry mismatched qparams.
        """
        input_qparams = None
        dq_nodes = set()
        for arg in arg_list:
            if not isinstance(arg, torch.fx.Node):
                return True, None, set()

            arg_quant_params = None
            if arg.target in DQ_OPS:
                args = arg.args
                scales = args[1]
                if (
                    isinstance(args[1], torch.fx.Node)
                    and self.exported_program is not None
                    and is_param_node(self.exported_program, args[1])
                ):
                    scales = get_param_tensor(self.exported_program, args[1])
                zps = args[2]
                if (
                    isinstance(args[2], torch.fx.Node)
                    and self.exported_program is not None
                    and is_param_node(self.exported_program, args[2])
                ):
                    zps = get_param_tensor(self.exported_program, args[2])
                arg_quant_params = QuantArgs.from_operator(
                    arg.target, (args[0], scales, zps, *args[3:])
                )
                dq_nodes.add(arg)
            if input_qparams is not None and input_qparams != arg_quant_params:
                return False, None, set()
            input_qparams = arg_quant_params
        return True, input_qparams, dq_nodes

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:

        # Loop over the graph nodes and find any node in the 'targeted_ops' list.
        for n in graph_module.graph.nodes:
            n = cast(torch.fx.Node, n)
            if n.op != "call_function":
                continue
            # Don't fold chains of quant-ops into each other.
            if n.target in (*Q_OPS, *DQ_OPS):
                continue

            # Make sure we haven't already set qparams meta information on the node
            if "input_qparams" in n.meta:
                raise RuntimeError(
                    f'Unexpected key "input_qparams" found in meta for node {n}. '
                    "input_qparams should not have been set at this point"
                )
            if "output_qparams" in n.meta:
                raise RuntimeError(
                    f'Unexpected key "output_qparams" found in meta for node {n}. '
                    "output_qparams should not have been set at this point"
                )

            # for the inputs and outputs search the graph for quantization info and
            # store the information in a dict with order of the _tensor_ inputs as key,
            # ignoring any other arguments to the target node.
            n.meta["input_qparams"] = {}
            n.meta["output_qparams"] = {}
            arg_groups = {}
            mismatched = False
            for i, arg in enumerate(n.args):
                if isinstance(arg, list):
                    arg_list = arg
                elif isinstance(arg, torch.fx.Node):
                    arg_list = [arg]
                else:
                    continue
                ok, qparams, dq_nodes = self.collect_arg_qparams(arg_list)
                if not ok:
                    mismatched = True
                    break
                if qparams is not None:
                    arg_groups[i] = (qparams, dq_nodes)

            if mismatched:
                del n.meta["input_qparams"]
                del n.meta["output_qparams"]
                continue

            for i, (qparams, dq_nodes) in arg_groups.items():
                n.meta["input_qparams"][i] = qparams
                for dq in dq_nodes:
                    n.replace_input_with(dq, cast(torch.fx.Node, dq.args[0]))
                    if len(dq.users) == 0:
                        graph_module.graph.erase_node(dq)

            # Copy the users, since we are modifying it.
            users_copy = copy.copy(n.users)
            q_idx = 0
            for user in users_copy:
                if user.target not in Q_OPS:
                    continue

                n.meta["output_qparams"][q_idx] = QuantArgs.from_operator(
                    user.target, user.args
                )
                q_idx += 1

                user.replace_all_uses_with(n)
                graph_module.graph.erase_node(user)

        # retrace the graph to update the fake tensor types
        graph_module = super().call(graph_module).graph_module

        graph_module.recompile()
        return PassResult(graph_module, True)
