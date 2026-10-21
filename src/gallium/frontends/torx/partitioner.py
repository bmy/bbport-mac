# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import torch

from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import generate_pattern_op_partitions
from executorch.exir.backend.partitioner import (
   DelegationSpec,
   Partitioner,
   PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase

from .backend import TorxBackend
from .debug import dump_graph_support, node_tensor_dtype
from .exir_to_gallium import (
    FUSABLE_ACTIVATION_OPS,
    fused_activation,
    get_ml_device,
    is_identity_clone,
    is_torx_convertible,
    is_torx_supported,
)


def _is_partitioned(node: torch.fx.Node, tag: str) -> bool:
    return node.meta.get("delegation_tag") == tag


QUANTIZE_OPS = (
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
)

DEQUANTIZE_OPS = (
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
)

class TorxSupportedOperators(OperatorSupportBase):

    @staticmethod
    def _is_graph_input(node: torch.fx.Node) -> bool:
        return node.op == "placeholder" and "from_node" not in node.meta

    @staticmethod
    def _is_input_quantization(node: torch.fx.Node) -> bool:
        """A quantize_per_tensor that converts a graph input from float32 to int8."""
        if node.target not in QUANTIZE_OPS:
            return False
        return (
            len(node.args) > 0
            and isinstance(node.args[0], torch.fx.Node)
            and TorxSupportedOperators._is_graph_input(node.args[0])
        )

    @staticmethod
    def _is_output_dequantization(node: torch.fx.Node) -> bool:
        """A dequantize_per_tensor that converts int8 back to float32 for the graph output."""
        if node.target not in DEQUANTIZE_OPS:
            return False
        users = list(node.users.keys())
        return len(users) == 1 and users[0].op == "output"

    @staticmethod
    def _is_frozen_param(node: torch.fx.Node) -> bool:
        """A placeholder carrying constant weights or biases (not a user input)."""
        return node.op == "placeholder" and "from_node" in node.meta

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:

        if self._is_graph_input(node):
            return False

        if node.op == "output":
            return False

        if self._is_input_quantization(node):
            return False

        if self._is_output_dequantization(node):
            return False

        # Frozen parameter placeholders (weights, biases) belong in the delegate.
        if self._is_frozen_param(node):
            return True

        # Internal Q/DQ nodes (for weights, biases, inter-op) belong in the
        # delegate so FoldAndAnnotateQParamsPass can fold them in preprocess().
        if node.target in QUANTIZE_OPS or node.target in DEQUANTIZE_OPS:
            return True

        # An activation is only delegated when its producer's operation
        # absorbs it; a standalone activation has no lowering.
        if node.target in FUSABLE_ACTIVATION_OPS:
            producer = node.args[0]
            return (isinstance(producer, torch.fx.Node) and
                    fused_activation(producer) is node)

        # A clone fed directly by a graph input could end up passing that
        # input straight to the partition output once removed, leaving the
        # delegate's IO entries unresolvable.
        if is_identity_clone(node) and node.args[0].op != "placeholder":
            return True

        ml_device = get_ml_device()
        return is_torx_supported(node, ml_device)

class TorxPartitioner(Partitioner):
    def __init__(self) -> None:
        super().__init__({})
        self.op_support = TorxSupportedOperators()
        self.delegation_spec = DelegationSpec(
            TorxBackend.__name__,
            [],
        )

    def _detag_boundary_nodes(self, graph_module: torch.fx.GraphModule, tag: str) -> None:
        # Detag boundary Q/DQ just like Arm integer-only partitioning does:
        # keep Q/DQ on the host side and delegate only int kernels.
        for node in graph_module.graph.nodes:
            if not _is_partitioned(node, tag):
                continue

            is_q_node = node.target in QUANTIZE_OPS
            is_dq_node = node.target in DEQUANTIZE_OPS

            is_boundary_q_node = (
                is_q_node
                and len(node.all_input_nodes) > 0
                and not _is_partitioned(node.all_input_nodes[0], tag)
            )
            is_boundary_dq_node = is_dq_node and any(
                not _is_partitioned(user, tag) for user in node.users
            )

            if is_boundary_q_node or is_boundary_dq_node:
                node.meta.pop("delegation_tag", None)
                continue

            # For non Q/DQ nodes, if any input comes from outside the partition
            # as float, this cannot be a pure int delegate boundary.
            if not is_q_node and not is_dq_node:
                for input_node in node.all_input_nodes:
                    if _is_partitioned(input_node, tag):
                        continue
                    dtype = node_tensor_dtype(input_node)
                    if dtype is not None and dtype.is_floating_point:
                        node.meta.pop("delegation_tag", None)
                        break

    def partition(self, edge_exported_program: ExportedProgram) -> PartitionResult:
        dump_graph_support(edge_exported_program.graph_module, self.op_support)

        partition_tags = {}
        partition_list = generate_pattern_op_partitions(
            edge_exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            delegation_tag = f"tag{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag

            self._detag_boundary_nodes(edge_exported_program.graph_module, delegation_tag)

            tagged_nodes = [
                n for n in partition.nodes
                if n.meta.get("delegation_tag") == delegation_tag
            ]
            has_ml_op = any(
                n.op == "call_function" and is_torx_convertible(n)
                for n in tagged_nodes
            )

            if not tagged_nodes or not has_ml_op:
                for n in tagged_nodes:
                    n.meta.pop("delegation_tag", None)
                continue

            partition_tags[delegation_tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=edge_exported_program,
            partition_tags=partition_tags,
        )
