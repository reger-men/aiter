# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FX pre-grad pass: rewrite ``cat([leaky_relu(x[..., :W]), x[..., W:] * y])``
to ``torch.ops.aiter.block_cat_fused(x, y, slope, W)``.

This pass operates on the Dynamo-traced FX graph (pre-AOTAutograd). When
a user model builds the cat-construction pattern below, the matcher
collapses it to one op call, which AOTAutograd then uses (along with the
registered backward) to emit a single fused backward Triton kernel
instead of decomposing into the per-half pointwise pair.

Pattern in the Dynamo IR::

    l12 = linear(x, w12, b12)
    l4  = linear(x, w4,  b4)
    s_lo  = operator.getitem(l12, (Ellipsis, slice(None, W1)))   # or torch.narrow
    leaky = torch.nn.functional.leaky_relu(s_lo, slope)
    s_hi  = operator.getitem(l12, (Ellipsis, slice(W1, None)))
    mul   = operator.mul(s_hi, l4)
    cat   = torch.cat([leaky, mul], dim=-1)

The pass is opt-in. Install with::

    from aiter.inductor import install_block_cat_fused_pass
    install_block_cat_fused_pass()

It sets ``torch._inductor.config.pre_grad_custom_pass`` to the matcher.
If the user has their own ``pre_grad_custom_pass`` they should chain
manually.
"""
from __future__ import annotations

import logging
import operator
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters

# Importing this module registers the custom op via aiter.ops.block_cat_fused.
from ..ops import block_cat_fused as _bcf  # noqa: F401


log = logging.getLogger("aiter")


def _val_of(node):
    if not hasattr(node, "meta"):
        return None
    # Use explicit `is None` instead of `or` here: when meta["val"] is a
    # Tensor (which can happen if the matcher is invoked on a post-grad
    # graph or a test graph that populates meta["val"] directly), the
    # `or` short-circuit calls `Tensor.__bool__`, which raises for any
    # multi-element tensor.
    v = node.meta.get("val")
    if v is None:
        v = node.meta.get("example_value")
    return v


def _shape_of(node):
    val = _val_of(node)
    if val is None:
        return None
    try:
        return tuple(int(s) for s in val.shape)
    except Exception:
        return None


def _is_call(node, *targets):
    if node.op != "call_function":
        return False
    return node.target in targets


def _is_cat(node):
    return _is_call(
        node,
        torch.cat,
        getattr(torch, "concatenate", None),
        getattr(torch, "concat", None),
    )


def _is_leaky(node):
    return _is_call(
        node,
        torch.nn.functional.leaky_relu,
        getattr(torch, "leaky_relu", None),
    )


def _is_mul(node):
    return _is_call(node, torch.mul, operator.mul) or (
        node.op == "call_method" and node.target == "mul"
    )


def _is_getitem(node):
    return _is_call(node, operator.getitem)


def _is_narrow(node):
    return _is_call(node, torch.narrow) or (
        node.op == "call_method" and node.target == "narrow"
    )


def _parse_lastdim_slice(node):
    """Return (x_node, start, stop) if the node represents a last-dim slice
    of x, else None.

    Supports:
      * operator.getitem(x, (..., slice(start, stop)))    Dynamo Python form
      * operator.getitem(x, slice(start, stop))            1-D form
      * torch.narrow(x, dim, start, length)                pre-grad emit form
      * x.narrow(dim, start, length)                       method form
    """
    if _is_getitem(node):
        if len(node.args) < 2:
            return None
        src = node.args[0]
        idx = node.args[1]
        if isinstance(idx, slice):
            return src, idx.start or 0, idx.stop
        if not isinstance(idx, tuple):
            return None
        last = idx[-1]
        if not isinstance(last, slice):
            return None
        if last.step is not None and last.step != 1:
            return None
        for prior in idx[:-1]:
            if prior is Ellipsis:
                continue
            if isinstance(prior, slice) and prior.start is None and prior.stop is None:
                continue
            return None
        start = last.start if last.start is not None else 0
        stop = last.stop
        return src, start, stop

    if _is_narrow(node):
        if node.op == "call_function":
            if len(node.args) < 4:
                return None
            src, dim, start, length = node.args[0], node.args[1], node.args[2], node.args[3]
        else:
            if len(node.args) < 4:
                return None
            src, dim, start, length = node.args[0], node.args[1], node.args[2], node.args[3]
        src_shape = _shape_of(src)
        if src_shape is None:
            return None
        rank = len(src_shape)
        if not isinstance(dim, int):
            return None
        dim_norm = dim + rank if dim < 0 else dim
        if dim_norm != rank - 1:
            return None
        if not isinstance(start, int) or not isinstance(length, int):
            return None
        return src, start, start + length

    return None


def _leaky_slope(node):
    if len(node.args) >= 2:
        return float(node.args[1])
    return float(node.kwargs.get("negative_slope", 0.01))


def _try_match_block_cat(cat_node):
    """Match ``cat([leaky_relu(x[..., :W]), x[..., W:] * y])`` and return
    (x_node, y_node, slope, W1) or None.
    """
    if len(cat_node.args) < 1:
        return None
    inputs = cat_node.args[0]
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        return None
    cat_dim = cat_node.kwargs.get("dim", 0)
    if len(cat_node.args) >= 2:
        cat_dim = cat_node.args[1]

    cat_shape = _shape_of(cat_node)
    if cat_shape is None or len(cat_shape) < 2:
        return None
    rank = len(cat_shape)
    if isinstance(cat_dim, int) and cat_dim < 0:
        cat_dim_norm = cat_dim + rank
    else:
        cat_dim_norm = int(cat_dim) if isinstance(cat_dim, int) else None
    if cat_dim_norm != rank - 1:
        return None

    leaky_node, mul_node = inputs[0], inputs[1]
    if not _is_leaky(leaky_node):
        return None
    if not _is_mul(mul_node):
        return None

    leaky_in = leaky_node.args[0] if leaky_node.args else None
    if leaky_in is None:
        return None
    sl_lo = _parse_lastdim_slice(leaky_in)
    if sl_lo is None:
        return None
    x_lo, start_lo, stop_lo = sl_lo
    if start_lo != 0 or stop_lo is None:
        return None
    W1 = int(stop_lo)

    if len(mul_node.args) < 2:
        return None
    a, b = mul_node.args[0], mul_node.args[1]
    sl_hi = _parse_lastdim_slice(a)
    y_node = b
    if sl_hi is None:
        sl_hi = _parse_lastdim_slice(b)
        y_node = a
    if sl_hi is None:
        return None
    x_hi, start_hi, stop_hi = sl_hi
    if x_hi is not x_lo:
        return None
    if start_hi != W1:
        return None

    x_shape = _shape_of(x_lo)
    if x_shape is None:
        return None
    W12 = int(x_shape[-1])
    if stop_hi is not None and int(stop_hi) != W12:
        return None
    if not (0 < W1 < W12):
        return None

    if not hasattr(y_node, "meta"):
        return None
    y_shape = _shape_of(y_node)
    if y_shape is None or y_shape[-1] != W12 - W1:
        return None
    if y_shape[0] != x_shape[0]:
        return None

    slope = _leaky_slope(leaky_node)
    return x_lo, y_node, slope, W1


def fuse_block_cat_fused_in_graph(graph: fx.Graph) -> int:
    """Run the matcher on the given Graph. Returns # replacements."""
    n = 0
    target = torch.ops.aiter.block_cat_fused.default
    cat_nodes = [
        node
        for node in list(graph.nodes)
        if node.op == "call_function" and _is_cat(node)
    ]
    for cat_node in cat_nodes:
        match = _try_match_block_cat(cat_node)
        if match is None:
            continue
        x_node, y_node, slope, W1 = match
        with graph.inserting_before(cat_node):
            new_node = graph.call_function(
                target, args=(x_node, y_node, float(slope), int(W1)),
            )
        cat_val = _val_of(cat_node)
        if cat_val is not None:
            new_node.meta["val"] = cat_val
            new_node.meta["example_value"] = cat_val
        cat_node.replace_all_uses_with(new_node)
        graph.erase_node(cat_node)
        counters["inductor"]["block_cat_fused"] += 1
        log.debug(
            "block_cat_fused: replaced cat(x_shape=%s, y_shape=%s, slope=%.3g, W1=%d)",
            _shape_of(x_node), _shape_of(y_node), slope, W1,
        )
        n += 1
    return n


def block_cat_fused_pre_grad_pass(graph: fx.Graph):
    """Top-level pickleable pre-grad pass entry point.

    Inductor passes the pre-grad ``torch.fx.Graph`` here; we run the
    block_cat_fused matcher and return the graph.
    """
    n = fuse_block_cat_fused_in_graph(graph)
    if n > 0:
        log.info("block_cat_fused: pre-grad pass replaced %d cat nodes", n)
    return graph


def install_block_cat_fused_pass():
    """Install the matcher as Inductor's ``pre_grad_custom_pass``.

    If the user already has their own ``pre_grad_custom_pass`` this
    function will overwrite it; chain manually if you need to keep both.
    """
    import torch._inductor.config as ic
    ic.pre_grad_custom_pass = block_cat_fused_pre_grad_pass
    return block_cat_fused_pre_grad_pass


__all__ = [
    "block_cat_fused_pre_grad_pass",
    "fuse_block_cat_fused_in_graph",
    "install_block_cat_fused_pass",
]
