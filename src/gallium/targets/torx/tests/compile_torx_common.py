# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
#
# SPDX-License-Identifier: MIT

"""Shared helpers for compile_torx_model.py and compile_torx_tests.py."""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# Make the torx frontend package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "frontends"))

import torch
from torch.export import export
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def configure_known_warning_filters() -> None:
    """Suppress known third-party warning noise with narrow filters only."""
    warnings.filterwarnings(
        "ignore",
        message=(
            r"The given buffer is not writable, and PyTorch does not support "
            r"non-writable tensors\..*"
        ),
        category=UserWarning,
        module=r"torch\.export\.pt2_archive\._package",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=FutureWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"guard_size_oblivious will be removed\..*",
        category=FutureWarning,
        module=r"executorch\.exir\.tensor",
    )


def get_device():
    """Return MESA_ML_DEVICE or exit with an error."""
    device = os.environ.get('MESA_ML_DEVICE', '')
    if not device:
        print("MESA_ML_DEVICE not set", file=sys.stderr)
        sys.exit(1)
    return device


def extract_quant_output_params(module):
    """Return a list of output (scale, zero_point) tuples from output-bound
    dequantize_per_tensor nodes.

    Returns an empty list when no dequantized outputs are found.
    """
    output_qparams = []

    for node in module.graph.nodes:
        if node.op != "output":
            continue

        out_args = node.args[0]
        if not isinstance(out_args, (list, tuple)):
            out_args = [out_args]

        for out_node in out_args:
            if not isinstance(out_node, torch.fx.Node):
                continue
            if out_node.op != "call_function":
                continue
            if "dequantize_per_tensor" not in str(out_node.target):
                continue
            if len(out_node.args) < 3:
                continue
            output_qparams.append((out_node.args[1], out_node.args[2]))

    return output_qparams


def compile_to_pte(module, example_inputs, pte_path):
    """Export, lower with TorxPartitioner, and write a .pte file."""
    from torx import DecomposeQuantNodesPass, TorxPartitioner

    exported = export(module, example_inputs, strict=True)

    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[TorxPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    # Decompose remaining host-side quantized_decomposed q/dq nodes before
    # to_executorch so strict ToOutVar lowering can succeed.
    edge_program = edge_program.transform([DecomposeQuantNodesPass()])

    exec_prog = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=False,
            external_constants=False,
        )
    )

    with open(pte_path, "wb") as f:
        exec_prog.write_to_file(f)


def quantize_module(module, example_inputs, *, is_prepared=False):
    """Prepare, calibrate, and convert a module to PT2E quantized form."""
    from torx import TorxQuantizer

    if not is_prepared:
        module = prepare_pt2e(module, TorxQuantizer())
    module(*example_inputs)  # calibration
    return convert_pt2e(module)


def compute_max_quant_error(float_ref_path, quant_output):
    """Return the max absolute error between the float reference and quantised output.

    quant_output is expected to be a single tensor (callers iterate over multi-output
    results themselves).
    """
    float_ref = np.fromfile(str(float_ref_path), dtype=np.float32)
    return float(
        np.abs(float_ref - quant_output.detach().numpy().flatten()).max()
    )


def write_stamp(stamp_file):
    """Write the meson stamp file to signal completion."""
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text("done\n")


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)
