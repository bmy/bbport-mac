# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import ctypes
import struct
from typing import final, List

import torch

from executorch.exir.backend.backend_api import BackendDetails
from executorch.exir.backend.backend_details import (
   CompileSpec,
   ExportedProgram,
   PreprocessResult,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.pass_manager import PassManager
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform

_libc = ctypes.CDLL(None)

from . import gallium
from .debug import dump_compiled_graph
from .exir_to_gallium import CLONE_OP, exir_to_gallium, get_ml_device
from .fold_qdq_with_annotated_qparams_pass import FoldAndAnnotateQParamsPass


def _build_inputs_name_map(edge_program: ExportedProgram) -> dict[str, str]:
    return {spec.arg.name: spec.target for spec in edge_program.graph_signature.input_specs}


def _attach_torx_data(graph_module: torch.fx.GraphModule, inputs_name_map: dict[str, str], state_dict) -> None:
    for node in graph_module.graph.nodes:
        op_name = node.target
        state_name = inputs_name_map.get(op_name)
        if state_name in state_dict:
            node.meta['torx_data'] = state_dict[state_name]


def _resolve_io_idx(gallium_graph, node, role):
    gallium_idx = gallium_graph.resolve_idx(node)
    if gallium_idx is None:
        raise RuntimeError(
            f"Torx: no Gallium tensor recorded for {role} '{node.name}'")
    return gallium_idx


def _build_io_entries(graph_module: torch.fx.GraphModule, gallium_graph):
    input_entries = []
    for node in graph_module.graph.nodes:
        if node.op != "placeholder" or 'torx_data' in node.meta:
            continue
        input_entries.append(_resolve_io_idx(gallium_graph, node, "input"))

    output_entries = []
    for node in graph_module.graph.nodes:
        if node.op != "output":
            continue
        out_args = node.args[0]
        if not isinstance(out_args, (list, tuple)):
            out_args = [out_args]
        for out_node in out_args:
            output_entries.append(_resolve_io_idx(gallium_graph, out_node,
                                                  "output"))

    return input_entries, output_entries


def _mark_external_outputs(gallium_graph, output_entries):
    outputs = set(output_entries)
    for op in gallium_graph.operations:
        for j in range(op.output_count):
            tensor = op.output_tensors[j].contents
            if tensor.index in outputs:
                tensor.is_external_output = True


def _compile_gallium_subgraph(ml_device, gallium_graph) -> bytes:
    if not gallium_graph.operations:
        return b''

    pipe_operations = (gallium.pipe_ml_operation * len(gallium_graph.operations))()
    for i, op in enumerate(gallium_graph.operations):
        pipe_operations[i] = op

    dump_compiled_graph(pipe_operations, len(gallium_graph.operations))

    subgraph = ml_device.contents.ml_subgraph_create(ml_device, pipe_operations, len(gallium_graph.operations))
    if not subgraph:
        raise RuntimeError("Torx: ml_subgraph_create() failed — driver rejected the subgraph")
    subgraph = ctypes.cast(subgraph, ctypes.POINTER(gallium.pipe_ml_subgraph))

    gallium_graph.release_buffers()

    buffer_length = ctypes.c_size_t()
    buffer_address = ml_device.contents.ml_subgraph_serialize(ml_device, subgraph, ctypes.byref(buffer_length))
    if not buffer_address or buffer_length.value == 0:
        raise RuntimeError("Torx: ml_subgraph_serialize() failed — compilation produced no output")
    blob = ctypes.string_at(buffer_address, buffer_length.value)
    ml_device.contents.ml_subgraph_destroy(ml_device, subgraph)
    _libc.free(ctypes.c_void_p(buffer_address))
    return blob


def _pack_io_data(input_entries, output_entries) -> bytes:
    # struct torx_io_map in torx_backend.h
    io_data = struct.pack('<II', len(input_entries), len(output_entries))
    for gidx in input_entries + output_entries:
        io_data += struct.pack('<I', gidx)
    return io_data


@final
class TorxBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        del compile_specs  # Reserved by ExecuTorch API; unused by current Torx backend.

        inputs_name_map = _build_inputs_name_map(edge_program)
        data_sources = {**edge_program.state_dict, **edge_program.constants}
        _attach_torx_data(edge_program.graph_module, inputs_name_map, data_sources)

        result = PassManager(passes=[
            RemoveCloneOpsTransform(preserve_input_output_copies=True,
                                    eliminate_quant_dequant_pairs=False),
            FoldAndAnnotateQParamsPass(edge_program),
        ])(edge_program.graph_module)
        graph_module = result.graph_module

        # A clone surviving to translation would be dropped there silently.
        assert not any(n.op == "call_function" and n.target == CLONE_OP
                       for n in graph_module.graph.nodes)

        # Re-attach torx_data to the new graph_module's placeholders, because
        # ExportPass creates a fresh graph module that only migrates meta["val"].
        _attach_torx_data(graph_module, inputs_name_map, data_sources)

        ml_device = get_ml_device()
        exir_operations = [node for node in graph_module.graph.nodes if node.op == "call_function" or node.op == "call_method"]
        gallium_graph = exir_to_gallium(exir_operations)
        input_entries, output_entries = _build_io_entries(graph_module, gallium_graph)
        _mark_external_outputs(gallium_graph, output_entries)
        buffer = _compile_gallium_subgraph(ml_device, gallium_graph)

        named_data_store = NamedDataStore()
        named_data_store.add_named_data("torx_device_id", ml_device.contents.id, None, None)

        # Build io_map with Gallium tensor indices for delegate I/O.
        io_data = _pack_io_data(input_entries, output_entries)

        # Append io_map to processed bytes so each partition carries
        # its own I/O mapping (NamedDataStore keys must be unique
        # across all partitions in a .pte).
        # struct torx_blob_header in torx_backend.h
        io_map_header = struct.pack('<II', len(buffer), len(io_data))
        processed_bytes = io_map_header + buffer + io_data

        return PreprocessResult(
            processed_bytes=processed_bytes,
            debug_handle_map={},
            data_store_output=named_data_store.get_named_data_store_output(),
        )

