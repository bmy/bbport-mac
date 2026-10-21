# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
#
# SPDX-License-Identifier: MIT

"""Split whole-model .pt2 files into per-op .pt2 files and float references.

This is the first (fast, sequential) step of test artifact generation.
For each conv2d / add chain in the model it:

  1. Builds a standalone nn.Module with copied parameters.
  2. Exports it as a per-op .pt2 file.
  3. Runs float CPU inference to produce a reference output.

Output per op:

  <output_dir>/<model>/<op>/op.pt2            -- per-op ExportedProgram
  <output_dir>/<model>/<op>/example_input.pt  -- float input tensor(s)
  <output_dir>/<model>/<op>/output-{i}.data   -- float32 CPU reference(s)

The .pt2 and example_input.pt are consumed by compile_torx_tests.py
(the second, parallelisable step) and not needed at test runtime.

Usage:
  python split_torx_tests.py <models_dir> <output_dir> <stamp_file>
"""

import shutil
import sys
import warnings
from pathlib import Path

# python3 -P leaves the script's directory off sys.path.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from extract_subgraph import _build_subgraph, _get_attr_val as _get_attr  # noqa: E402

# Ops with no standalone hardware equivalent, skipped when numbering range
# extractions: training/bookkeeping ops, ops fused into conv at lowering,
# and shape-only ops.
_RANGES_EXCLUDED_OPS = {
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.new_zeros.default",
    "aten.full.default",
    "aten.dropout.default",
    "aten.slice.Tensor",
    "aten.view_as.default",
    "aten.detach.default",
    "aten.batch_norm.default",
    "aten.hardtanh_.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.flatten.using_ints",
}


def _configure_known_warning_filters() -> None:
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

# aten ops recognised as activations that can follow batch_norm.
_ACTIVATION_OPS = {
    "aten.relu.default",
    "aten.relu_.default",
    "aten.hardtanh.default",
    "aten.hardtanh_.default",
    "aten.clamp.default",
    "aten.clamp_.default",
    "aten.silu.default",
    "aten.silu_.default",
}


def _build_module(torch, gm, conv_node, chain_nodes):
    """Build a standalone nn.Module from a conv2d chain.

    chain_nodes is [conv] or [conv, bn] or [conv, bn, activation].
    Parameters are copied from the ExportedProgram's GraphModule.
    """
    # --- Conv2d ---------------------------------------------------------
    weight = _get_attr(gm, str(conv_node.args[1].target))
    out_ch, in_ch_over_groups, *kernel = weight.shape
    bias_arg = conv_node.args[2] if len(conv_node.args) > 2 else None
    has_bias = isinstance(bias_arg, torch.fx.Node)
    stride = conv_node.args[3] if len(conv_node.args) > 3 else [1, 1]
    padding = conv_node.args[4] if len(conv_node.args) > 4 else [0, 0]
    dilation = conv_node.args[5] if len(conv_node.args) > 5 else [1, 1]
    groups = conv_node.args[6] if len(conv_node.args) > 6 else 1
    in_ch = in_ch_over_groups * groups

    conv = torch.nn.Conv2d(
        in_ch, out_ch, kernel, stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=has_bias,
    )
    conv.weight.data.copy_(weight)
    if has_bias:
        conv.bias.data.copy_(_get_attr(gm, str(bias_arg.target)))

    layers = [conv]

    # --- BatchNorm2d (optional) -----------------------------------------
    if len(chain_nodes) >= 2:
        bn_node = chain_nodes[1]
        bn_weight = _get_attr(gm, str(bn_node.args[1].target))
        bn_bias = _get_attr(gm, str(bn_node.args[2].target))
        bn_mean = _get_attr(gm, str(bn_node.args[3].target))
        bn_var = _get_attr(gm, str(bn_node.args[4].target))
        eps = bn_node.args[7]
        bn = torch.nn.BatchNorm2d(out_ch, eps=eps)
        bn.weight.data.copy_(bn_weight)
        bn.bias.data.copy_(bn_bias)
        bn.running_mean.data.copy_(bn_mean)
        bn.running_var.data.copy_(bn_var)
        bn.eval()
        layers.append(bn)

    # --- Activation (optional) ------------------------------------------
    if len(chain_nodes) >= 3:
        act_node = chain_nodes[2]
        target_name = str(act_node.target)
        if "hardtanh" in target_name:
            min_val = act_node.args[1] if len(act_node.args) > 1 else -1.0
            max_val = act_node.args[2] if len(act_node.args) > 2 else 1.0
            if min_val == 0.0 and max_val == 6.0:
                layers.append(torch.nn.ReLU6())
            else:
                layers.append(torch.nn.Hardtanh(min_val, max_val))
        elif "relu" in target_name:
            layers.append(torch.nn.ReLU())
        elif "silu" in target_name:
            layers.append(torch.nn.SiLU())

    if len(layers) == 1:
        return layers[0]
    return torch.nn.Sequential(*layers)


def _build_add_module(torch, act_node=None):
    """Build a standalone nn.Module for an elementwise add, optionally fused
    with an activation the driver's add can absorb: relu, or a clamp at
    exactly [0, 6]."""

    act_target = str(act_node.target) if act_node is not None else ""

    if act_node is None or "silu" in act_target:
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y
        return AddModule()

    if "relu" in act_target:
        class AddReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.act = torch.nn.ReLU()
            def forward(self, x, y):
                return self.act(x + y)
        return AddReLUModule()

    if "hardtanh" in act_target or "clamp" in act_target:
        low = act_node.args[1] if len(act_node.args) > 1 else None
        high = act_node.args[2] if len(act_node.args) > 2 else None
        if low == 0.0 and high == 6.0:
            class AddReLU6Module(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.act = torch.nn.ReLU6()
                def forward(self, x, y):
                    return self.act(x + y)
            return AddReLU6Module()
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y
        return AddModule()

    # Unknown activation — fall back to plain add (no fusion).
    class AddModule(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return AddModule()


def split_model(pt2_path, torch):
    """Load a whole-model .pt2 and split into per-op submodules.

    Returns a list of ``(index_name, module, example_inputs)`` tuples.
    ``example_inputs`` is a tuple of tensors (length 1 for conv, 2 for add).
    """
    ep = torch.export.load(str(pt2_path))
    gm = ep.module()

    ops = []
    idx = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if str(node.target) != "aten.conv2d.default":
            continue

        chain = [node]

        bn = None
        for u in node.users:
            if str(u.target) == "aten.batch_norm.default":
                bn = u
                break
        if bn is not None:
            chain.append(bn)

            for u in bn.users:
                target = str(u.target)
                if target in _ACTIVATION_OPS:
                    chain.append(u)
                    break
                if "detach" in target or "clone" in target:
                    for u2 in u.users:
                        if str(u2.target) in _ACTIVATION_OPS:
                            chain.append(u2)
                            break

        mod = _build_module(torch, gm, node, chain)
        mod.eval()

        input_meta = node.args[0].meta.get("val")
        if input_meta is None:
            continue
        torch.manual_seed(42 + idx)
        example_input = torch.randn(input_meta.shape)

        ops.append((f"{idx:03d}", mod, (example_input,)))
        idx += 1

    # --- ADD ops --------------------------------------------------------
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if str(node.target) != "aten.add.Tensor":
            continue

        # Check for fused activation
        act_node = None
        for u in node.users:
            if str(u.target) in _ACTIVATION_OPS:
                act_node = u
                break

        mod = _build_add_module(torch, act_node)
        mod.eval()

        input0_meta = node.args[0].meta.get("val") if isinstance(node.args[0], torch.fx.Node) else None
        input1_meta = node.args[1].meta.get("val") if isinstance(node.args[1], torch.fx.Node) else None
        if input0_meta is None or input1_meta is None:
            continue
        torch.manual_seed(42 + idx)
        example_input0 = torch.randn(input0_meta.shape)
        example_input1 = torch.randn(input1_meta.shape)

        ops.append((f"{idx:03d}", mod, (example_input0, example_input1)))
        idx += 1

    return ops


def split_model_ranges(pt2_path, torch):
    """Split a whole-model .pt2 into prefix ranges [0..0], [0..1], ..., [0..N-1].

    Returns a list of ``(range_name, module, example_inputs)`` tuples, where
    *range_name* is the zero-padded end index. Ops in _RANGES_EXCLUDED_OPS are
    skipped when numbering.
    """
    ep = torch.export.load(str(pt2_path))
    gm = ep.module()

    # Endpoints are numbered by the ops with hardware compute, but every
    # call_function up to an endpoint is selected, so the prefix reproduces
    # the model's real dataflow through bn/activation/layout ops too.
    prefixes = []
    prefix = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        prefix.append(node)
        target = str(node.target)
        if target.startswith("aten.") and target not in _RANGES_EXCLUDED_OPS:
            prefixes.append(list(prefix))

    if not prefixes:
        print("  No target ops found - nothing to extract ranges from.",
              file=sys.stderr)
        return []

    total = len(prefixes)
    print(f"  Extracting {total} prefix ranges...", file=sys.stderr)

    ranges = []
    for end, nodes in enumerate(prefixes):
        torch.manual_seed(42)
        try:
            new_gm, example_inputs, _ = _build_subgraph(gm, set(nodes))
        except Exception as e:
            print(f"  SKIP range [0..{end}]: {e}", file=sys.stderr)
            continue
        ranges.append((f"{end:03d}", new_gm, tuple(example_inputs)))

    return ranges


def main():
    _configure_known_warning_filters()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranges", action="store_true",
                        help="extract prefix ranges instead of per-op tests")
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("stamp_file", type=Path)
    cli = parser.parse_args()
    ranges_mode = cli.ranges
    models_dir = cli.models_dir
    output_root = cli.output_dir
    stamp_file = cli.stamp_file

    import torch
    from torch.export import export

    pt2_files = sorted(models_dir.glob("*.pt2"))
    if not pt2_files:
        print(f"No .pt2 files found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    for pt2_file in pt2_files:
        model_name = pt2_file.stem

        if ranges_mode:
            ops = split_model_ranges(pt2_file, torch)
            model_out_dir = output_root / model_name / "ranges"
        else:
            ops = split_model(pt2_file, torch)
            model_out_dir = output_root / model_name

        # Remove stale numbered directories left over from a previous run
        # that produced a different op or range count; whole-model artifacts
        # are files and are left untouched.
        op_names = {op_name for op_name, _, _ in ops}
        if model_out_dir.exists():
            for entry in model_out_dir.iterdir():
                if entry.is_dir() and entry.name.isdigit() and entry.name not in op_names:
                    shutil.rmtree(entry)

        for op_name, mod, example_inputs in ops:
            output_dir = model_out_dir / op_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Range modules are fx-stitched GraphModules, which only export
            # non-strict; don't pay a doomed strict trace for each one.
            strictness = (False,) if ranges_mode else (True, False)
            ep = None
            for strict in strictness:
                try:
                    ep = export(mod, example_inputs, strict=strict)
                    break
                except Exception as e:
                    export_error = e
            if ep is None:
                print(f"    SKIP {model_name}/{op_name}: export failed: {export_error}",
                      file=sys.stderr)
                continue

            # Save per-op .pt2 and example input(s) (consumed by compile step).
            torch.export.save(ep, str(output_dir / "op.pt2"))
            torch.save(example_inputs, str(output_dir / "example_input.pt"))

            # Compute float CPU reference output(s).
            def _flatten_outputs(x):
                if isinstance(x, torch.Tensor):
                    return [x]
                if isinstance(x, (list, tuple)):
                    result = []
                    for item in x:
                        result.extend(_flatten_outputs(item))
                    return result
                return [torch.tensor(x)]

            module = ep.module()
            with torch.no_grad():
                float_output = module(*example_inputs)
            flat_outs = _flatten_outputs(float_output)
            for i, out in enumerate(flat_outs):
                out.numpy().tofile(str(output_dir / f"output-{i}.data"))

    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text("done\n")


if __name__ == "__main__":
    main()
