#!/usr/bin/env python3
"""Extract a node-range subgraph from a quantized .pt2 file, stripping q/dq nodes.

Usage:
    python3 extract_subgraph.py <src.pt2> <dst.pt2> <range> [<range> ...]

<range> is N or N-M (inclusive), using absolute graph.nodes indices.

All quantize_* / dequantize_* nodes are stripped:
  - dequantize_per_channel  -> precomputed float weight buffer
  - dequantize_per_tensor   -> float placeholder (external) or q->dq passthrough
  - quantize_per_tensor     -> dropped; float input used in place of int8 output
"""

import sys
import torch
import torch.fx
import torch.ao.quantization.fx._decomposed  # noqa: F401 – registers quantized_decomposed ops


def _parse_ranges(args):
    indices = set()
    for arg in args:
        if '-' in arg:
            a, b = arg.split('-', 1)
            indices.update(range(int(a), int(b) + 1))
        else:
            indices.add(int(arg))
    return indices


def _is_q(n):
    t = str(n.target)
    return n.op == 'call_function' and 'quantize' in t and 'dequantize' not in t


def _is_dq_tensor(n):
    return n.op == 'call_function' and 'dequantize_per_tensor' in str(n.target)


def _is_dq_channel(n):
    return n.op == 'call_function' and 'dequantize_per_channel' in str(n.target)


def _is_dq(n):
    return _is_dq_tensor(n) or _is_dq_channel(n)


def _get_attr_val(gm, target):
    obj = gm
    for part in target.split('.'):
        obj = getattr(obj, part)
    return obj


def _set_attr(module, target, value):
    """Set a (possibly dotted) attribute on module, creating sub-modules as needed."""
    parts = target.split('.')
    for part in parts[:-1]:
        if not hasattr(module, part):
            setattr(module, part, torch.nn.Module())
        module = getattr(module, part)
    if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
        module.register_buffer(parts[-1], value)
    else:
        setattr(module, parts[-1], value)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _build_subgraph(gm, selected):
    """Extract *selected* call_function nodes from *gm*.

    Returns ``(new_gm, example_inputs)`` ready for ``torch.export.export``.
    """
    nodes = list(gm.graph.nodes)

    new_root = torch.nn.Module()
    new_graph = torch.fx.Graph()
    _used = set()
    env = {}  # old_node -> new_node

    def _fresh(base):
        name = base.replace('.', '_')
        if name not in _used:
            _used.add(name)
            return name
        i = 0
        while f"{name}_{i}" in _used:
            i += 1
        _used.add(f"{name}_{i}")
        return f"{name}_{i}"

    def _resolve(x):
        """Map an old-graph node/value to a new-graph node/value."""
        if not isinstance(x, torch.fx.Node):
            return x
        if x in env:
            return env[x]

        if x.op == 'get_attr':
            val = _get_attr_val(gm, x.target)
            if isinstance(val, torch.Tensor):
                t = val.detach().clone()
                # Biases are sometimes stored as int32 in quantized models; promote
                # to float so the extracted float subgraph is type-consistent.
                if not t.is_floating_point() and t.dtype != torch.bool:
                    t = t.to(torch.float32)
                # Conv-bn folding during quantization re-registers weights as
                # parameters, which fails if they round-tripped as buffers.
                if isinstance(val, torch.nn.Parameter) and t.is_floating_point():
                    t = torch.nn.Parameter(t, requires_grad=False)
                _set_attr(new_root, x.target, t)
            else:
                _set_attr(new_root, x.target, val)
            ga = new_graph.get_attr(x.target)
            ga.meta = dict(x.meta)
            env[x] = ga
            return ga

        # External activation node – promote to a float placeholder.
        ph = new_graph.placeholder(_fresh(f"x_{x.name}"))
        if 'val' in x.meta:
            ph.meta['val'] = x.meta['val']
        env[x] = ph
        return ph

    # --- Main node loop ---------------------------------------------------
    for node in nodes:
        if node not in selected:
            continue
        if node.op in ('placeholder', 'get_attr', 'output'):
            continue

        if _is_q(node):
            env[node] = _resolve(node.args[0])

        elif _is_dq_channel(node):
            w_n, sc_n, zp_n = node.args[0], node.args[1], node.args[2]
            axis, qmin, qmax, dtype = node.args[3], node.args[4], node.args[5], node.args[6]
            with torch.no_grad():
                fw = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                    _get_attr_val(gm, w_n.target),
                    _get_attr_val(gm, sc_n.target),
                    _get_attr_val(gm, zp_n.target),
                    axis, qmin, qmax, dtype,
                )
            buf = _fresh(f"w_{node.name}")
            new_root.register_buffer(buf, fw)
            ga = new_graph.get_attr(buf)
            env[node] = ga

        elif _is_dq_tensor(node):
            inp = node.args[0]
            while isinstance(inp, torch.fx.Node) and _is_q(inp):
                inp = inp.args[0]
            env[node] = _resolve(inp)

        else:
            new_node = new_graph.node_copy(node, _resolve)
            new_node.meta = dict(node.meta)
            env[node] = new_node

    # --- Determine subgraph outputs ----------------------------------------
    # A non-q/dq node is an output if its value (possibly through a q->dq chain)
    # is consumed by a node outside the selected set.
    def _consumed_outside(node, _vis=None):
        if _vis is None:
            _vis = set()
        if node in _vis:
            return False
        _vis.add(node)
        for u in node.users:
            if u not in selected:
                return True
            if (_is_q(u) or _is_dq_tensor(u)) and _consumed_outside(u, _vis):
                return True
        return False

    seen = set()
    out_vals = []
    for node in nodes:
        if node not in selected or node.op != 'call_function':
            continue
        if _is_q(node) or _is_dq(node):
            continue
        if _consumed_outside(node):
            mapped = env.get(node)
            if isinstance(mapped, torch.fx.Node) and mapped not in seen:
                seen.add(mapped)
                out_vals.append(mapped)

    if not out_vals:
        raise RuntimeError(
            "No subgraph outputs found – widen the range or check range boundaries."
        )

    new_graph.output(out_vals[0] if len(out_vals) == 1 else tuple(out_vals))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(new_root, new_graph)

    # --- Build concrete example inputs from placeholder FakeTensor metadata ---
    example_inputs = []
    for n in new_graph.nodes:
        if n.op != 'placeholder':
            continue
        fake = n.meta.get('val')
        if fake is None or not hasattr(fake, 'shape'):
            raise RuntimeError(
                f"No shape metadata on placeholder '{n.name}'. "
                "Cannot build example inputs – check that the source .pt2 was "
                "produced by torch.export.export."
            )
        dt = fake.dtype if hasattr(fake, 'dtype') else torch.float32
        if dt.is_floating_point:
            example_inputs.append(torch.randn(tuple(fake.shape), dtype=dt))
        else:
            example_inputs.append(torch.zeros(tuple(fake.shape), dtype=dt))

    return new_gm, example_inputs, out_vals


def extract(src_pt2, dst_pt2, range_args):
    print(f"Loading {src_pt2} ...")
    ep = torch.export.load(src_pt2)
    gm = ep.module()
    nodes = list(gm.graph.nodes)

    selected = {nodes[i] for i in _parse_ranges(range_args) if i < len(nodes)}
    print(f"Selected {len(selected)} nodes from range(s): {' '.join(range_args)}")

    new_gm, example_inputs, out_vals = _build_subgraph(gm, selected)

    # --- Print summary -------------------------------------------------------
    call_nodes = [n for n in new_gm.graph.nodes if n.op == 'call_function']
    print(f"\n{len(example_inputs)} input(s)  |  "
          f"{len(call_nodes)} call_function ops  |  "
          f"{len(out_vals)} output(s)\n")
    for n in call_nodes:
        fv = n.meta.get('val')
        shape = str(tuple(fv.shape)) if fv is not None and hasattr(fv, 'shape') else '?'
        print(f"  {n.name:50s}  {str(n.target)[:45]:45s}  {shape}")

    # --- Export and save -----------------------------------------------------
    print(f"\nExporting ...")
    ep_new = torch.export.export(new_gm, tuple(example_inputs), strict=False)
    torch.export.save(ep_new, dst_pt2)
    print(f"Saved -> {dst_pt2}")


# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <src.pt2> <dst.pt2> <range> [<range> ...]",
            file=sys.stderr,
        )
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2], sys.argv[3:])


if __name__ == '__main__':
    main()
