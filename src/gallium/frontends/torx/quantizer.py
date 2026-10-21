# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import logging
from typing import List, cast

from torch import fx, per_tensor_affine, per_tensor_symmetric, qint8, qint32
import torch
from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
)

from torchao.quantization.pt2e.quantizer import (
    OperatorConfig,
    Quantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    annotate_input_qspec_map,
    annotate_output_qspec,
)

from .exir_to_gallium import FUSABLE_ACTIVATION_OPS_ATEN, activation_is_absorbable

Q_ANNOTATION_KEY = "quantization_annotation"

def is_annotated(node: fx.Node) -> bool:
    """Given a node return whether the node is annotated."""
    if Q_ANNOTATION_KEY not in node.meta:
        return False

    annotation = cast(QuantizationAnnotation, node.meta[Q_ANNOTATION_KEY])
    if not annotation._annotated:
        return False

    return True

def mark_node_as_annotated(node: fx.Node) -> None:
    """Marks node as annotated. If needed, an empty  QuantizationAnnotation is added
    to the quantization_annotation node meta entry.
    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    node.meta[Q_ANNOTATION_KEY]._annotated = True

class TorxQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

        self.io_spec = QuantizationSpec(dtype=qint8,
                                        quant_min=-128,
                                        quant_max=127,
                                        observer_or_fake_quant_ctr=HistogramObserver,
                                        qscheme=per_tensor_affine)

        self.weight_spec = QuantizationSpec(dtype=qint8,
                                            observer_or_fake_quant_ctr=MinMaxObserver,
                                            qscheme=per_tensor_symmetric)

        self.bias_spec = QuantizationSpec(dtype=qint32,
                                          observer_or_fake_quant_ctr=MinMaxObserver,
                                          qscheme=per_tensor_symmetric)

    def _annotate_io(
        self,
        model: fx.GraphModule,
    ):
        for node in model.graph.nodes:
            if is_annotated(node):
                continue
            if node.op == "placeholder" and len(node.users) > 0:
                annotate_output_qspec(
                    node,
                    self.io_spec,
                )
                mark_node_as_annotated(node)
            if node.op == "output":
                parent = node.all_input_nodes[0]
                annotate_input_qspec_map(
                    node, parent, self.io_spec
                )
                mark_node_as_annotated(node)

    def _annotate_with_optional_fused_act(self, node: fx.Node, input_qspec_map: dict) -> None:
        """Annotate node with input specs; if a single fusable activation follows,
        move the output Q/DQ annotation onto the activation node instead."""
        users = list(node.users.keys())
        act_node = None
        if (len(users) == 1 and users[0].target in FUSABLE_ACTIVATION_OPS_ATEN and
                activation_is_absorbable(
                    users[0], node.target == torch.ops.aten.conv2d.default)):
            act_node = users[0]

        if act_node is not None:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            act_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=self.io_spec,
                _annotated=True,
            )
        else:
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=self.io_spec,
                _annotated=True,
            )

    def _annotate_operations(self, model: fx.GraphModule):
        for node in model.graph.nodes:
            if is_annotated(node):
                continue
            if node.op == "call_function":
                if node.target == torch.ops.aten.conv2d.default:
                    input_qspec_map = {}
                    input_act = node.args[0]
                    assert isinstance(input_act, fx.Node)
                    input_qspec_map[input_act] = self.io_spec

                    weight = node.args[1]
                    assert isinstance(weight, fx.Node)
                    input_qspec_map[weight] = self.weight_spec

                    if len(node.args) > 2:
                        bias = node.args[2]
                        if isinstance(bias, fx.Node):
                            input_qspec_map[bias] = self.bias_spec

                    # Fused conv+activation: output Q/DQ goes after the
                    # activation so the observer sees clamped values.
                    self._annotate_with_optional_fused_act(node, input_qspec_map)

                elif node.target == torch.ops.aten.add.Tensor:
                    input_qspec_map = {}
                    input_qspec_map[node.args[0]] = self.io_spec
                    input_qspec_map[node.args[1]] = self.io_spec

                    self._annotate_with_optional_fused_act(node, input_qspec_map)

    def annotate(self, model):
        # Annotate operations first, then only add I/O quantisation if at
        # least one operation was annotated.  This avoids triggering the
        # expensive convert_pt2e() path for graphs that contain only
        # unsupported ops (e.g. standalone BatchNorm layers), saving ~5 s
        # per such test.
        self._annotate_operations(model)

        has_annotated_op = any(
            is_annotated(node)
            for node in model.graph.nodes
            if node.op == "call_function"
        )
        if has_annotated_op:
            self._annotate_io(model)

        return model

    def validate(self, model: fx.GraphModule) -> None:
        for node in model.graph.nodes:
            if Q_ANNOTATION_KEY not in node.meta:
                logging.debug(f"Node {node.name} is not annotated for quantization.")

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []
