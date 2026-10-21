# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
#
# SPDX-License-Identifier: MIT

"""Compile per-op and whole-model .pt2 files into device test artifacts.

This is the second (slow, parallelisable) step of test artifact
generation.  Per-op mode loads a .pt2 produced by split_torx_tests.py,
quantises it, records float test inputs, compiles it with the Torx
delegate, and writes device-specific files.  Whole-model mode compiles
the full model with mixed NPU/CPU delegation via TorxPartitioner, with
ExecuTorch orchestrating data movement between delegates.

Output per op or model per device:

  <dir>/<device>.pte            -- compiled program
  <dir>/<device>-input-{i}.data -- float32 input(s)
  <dir>/<device>.json           -- quantisation metadata
  <dir>/output-{i}.data         -- float32 CPU reference (model mode)

Usage:
  python compile_torx_tests.py <op_dir> <stamp_file>
  python compile_torx_tests.py --all <model_dir> <stamp_file>
  python compile_torx_tests.py --whole-model <pt2_file> <out_dir> <stamp_file>

--all compiles every numbered per-op directory under <model_dir>, in
parallel across processes.

Requires MESA_ML_DEVICE to be set.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from compile_torx_common import (
    compile_to_pte,
    configure_known_warning_filters,
    compute_max_quant_error,
    extract_quant_output_params,
    get_device,
    prepare_pt2e,
    quantize_module,
    torch,
    write_json,
    write_stamp,
)


def _collect_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        result = []
        for item in obj:
            result.extend(_collect_tensors(item))
        return result
    return []


def _write_artifacts(out_dir, module, example_inputs, output_qparams, device):
    """Persist inputs, quantisation metadata, and the compiled .pte."""
    for i, inp in enumerate(example_inputs):
        inp.numpy().tofile(str(out_dir / (device + f"-input-{i}.data")))

    with torch.no_grad():
        quant_output = module(*example_inputs)
    quant_outputs = _collect_tensors(quant_output)

    num_outputs = len(output_qparams)
    max_quant_errors = []
    for i in range(num_outputs):
        max_quant_errors.append(
            compute_max_quant_error(out_dir / f"output-{i}.data", quant_outputs[i])
        )

    compile_to_pte(module, example_inputs, out_dir / (device + ".pte"))

    write_json(out_dir / (device + ".json"), {
        "num_outputs": num_outputs,
        "output_scales": [float(s) for s, _ in output_qparams],
        "output_zero_points": [int(zp) for _, zp in output_qparams],
        "max_quant_errors": max_quant_errors,
    })


def compile_op(op_dir, device):
    pt2_path = op_dir / "op.pt2"
    if not pt2_path.exists():
        print(f"op.pt2 not found in {op_dir}", file=sys.stderr)
        sys.exit(1)

    ep = torch.export.load(str(pt2_path))
    module = ep.module()

    example_inputs_raw = torch.load(
        str(op_dir / "example_input.pt"), weights_only=True,
    )
    # Normalise to a tuple — old format saved a single tensor, new format
    # saves a tuple.
    if isinstance(example_inputs_raw, torch.Tensor):
        example_inputs = (example_inputs_raw,)
    else:
        example_inputs = tuple(example_inputs_raw)

    from torx import TorxQuantizer

    module = prepare_pt2e(module, TorxQuantizer())

    from torx.quantizer import Q_ANNOTATION_KEY
    has_quantized_ops = any(
        Q_ANNOTATION_KEY in node.meta
        and node.meta[Q_ANNOTATION_KEY]._annotated
        for node in module.graph.nodes
        if node.op == "call_function"
    )

    if not has_quantized_ops:
        print(f"    SKIP {op_dir.name}: no quantisable operations",
              file=sys.stderr)
        return

    module = quantize_module(module, example_inputs, is_prepared=True)

    output_qparams = extract_quant_output_params(module)
    if not output_qparams:
        print(f"    SKIP {op_dir.name}: could not extract quant params",
              file=sys.stderr)
        return

    _write_artifacts(op_dir, module, example_inputs, output_qparams, device)


def compile_model(pt2_path, out_dir, device):
    if not pt2_path.exists():
        print(f"{pt2_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    ep = torch.export.load(str(pt2_path))
    module = ep.module()

    from torch.export.graph_signature import InputKind
    user_input_names = [spec.arg.name
                        for spec in ep.graph_signature.input_specs
                        if spec.kind == InputKind.USER_INPUT]

    name_to_shape = {}
    for node in ep.graph.nodes:
        if node.op == "placeholder" and node.name in user_input_names:
            name_to_shape[node.name] = [int(s) for s in node.meta['val'].shape]
    input_shapes = [name_to_shape[name] for name in user_input_names
                    if name in name_to_shape]

    if not input_shapes:
        print("Could not determine input shape", file=sys.stderr)
        sys.exit(1)

    # Generate deterministic example inputs and re-export with concrete
    # shapes to remove shape guards from the loaded .pt2.
    torch.manual_seed(42)
    example_inputs = tuple(torch.randn(s) for s in input_shapes)
    ep = torch.export.export(module, example_inputs, strict=False)
    module = ep.module()

    with torch.no_grad():
        float_output = module(*example_inputs)
    for i, out in enumerate(_collect_tensors(float_output)):
        out.numpy().tofile(str(out_dir / f"output-{i}.data"))

    torch.save(example_inputs, str(out_dir / "example_input.pt"))

    module = quantize_module(module, example_inputs)

    output_qparams = extract_quant_output_params(module)
    if not output_qparams:
        print("    ERROR: could not extract quant params", file=sys.stderr)
        sys.exit(1)

    _write_artifacts(out_dir, module, example_inputs, output_qparams, device)


def _compile_op_worker(op_dir_str):
    configure_known_warning_filters()
    try:
        compile_op(Path(op_dir_str), get_device())
        return op_dir_str, None
    except Exception as e:
        return op_dir_str, f"{type(e).__name__}: {e}"


def compile_all_ops(model_dir, device):
    op_dirs = sorted(d for d in model_dir.iterdir()
                     if d.is_dir() and d.name.isdigit())
    if not op_dirs:
        print(f"No per-op directories under {model_dir}", file=sys.stderr)
        sys.exit(1)

    failures = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for op_dir, error in pool.map(_compile_op_worker,
                                      [str(d) for d in op_dirs]):
            if error is not None:
                failures += 1
                print(f"    FAIL {Path(op_dir).name}: {error}",
                      file=sys.stderr)

    if failures:
        print(f"{failures} of {len(op_dirs)} ops failed to compile",
              file=sys.stderr)
        sys.exit(1)


def main():
    configure_known_warning_filters()

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true",
                      help="compile every numbered op directory under the "
                           "given model directory")
    mode.add_argument("--whole-model", action="store_true",
                      help="compile a whole-model .pt2 with mixed NPU/CPU "
                           "delegation")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    if args.whole_model:
        if len(args.paths) != 3:
            parser.error("--whole-model needs <pt2_file> <out_dir> <stamp_file>")
        pt2_path, out_dir, stamp_file = (Path(p) for p in args.paths)
        compile_model(pt2_path, out_dir, get_device())
    elif args.all:
        if len(args.paths) != 2:
            parser.error("--all needs <model_dir> <stamp_file>")
        model_dir, stamp_file = (Path(p) for p in args.paths)
        compile_all_ops(model_dir, get_device())
    else:
        if len(args.paths) != 2:
            parser.error("per-op mode needs <op_dir> <stamp_file>")
        op_dir, stamp_file = (Path(p) for p in args.paths)
        compile_op(op_dir, get_device())

    write_stamp(stamp_file)


if __name__ == "__main__":
    main()
