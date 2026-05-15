# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inductor-side hooks for AITER ops.

This subpackage holds optional helpers that wire AITER custom ops into
``torch.compile`` (Inductor) via FX pre-grad matchers. Nothing here runs
automatically; the user calls an ``install_*`` function once at startup
to opt in.
"""
from .block_cat_fused_pass import (
    block_cat_fused_pre_grad_pass,
    fuse_block_cat_fused_in_graph,
    install_block_cat_fused_pass,
)

__all__ = [
    "block_cat_fused_pre_grad_pass",
    "fuse_block_cat_fused_in_graph",
    "install_block_cat_fused_pass",
]
