# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the block_cat_fused custom op and its pre-grad matcher.

The op rewrites ``cat([leaky_relu(x[..., :W1], slope), x[..., W1:] * y])``
into one ``torch.ops.aiter.block_cat_fused`` call so AOTAutograd sees one
op (and dispatches the registered fused backward) instead of decomposing
into per-half pointwise backwards.

These tests cover:
  * forward / backward correctness vs an eager torch reference (CPU fp32
    fallback path and GPU bf16 Triton path),
  * the FX pre-grad matcher fires on the expected pattern,
  * the matcher does NOT fire on a few negative cases (wrong cat axis,
    mismatched leading shape).

Run with::

    pytest op_tests/test_block_cat_fused.py
"""
import unittest

import torch
import torch.nn.functional as F

from torch._dynamo.utils import counters

import aiter.ops.block_cat_fused  # noqa: F401  - registers the op
from aiter.inductor.block_cat_fused_pass import block_cat_fused_pre_grad_pass


HAS_GPU = torch.cuda.is_available()


def _eager_block_cat(l12, l4, slope, W1):
    lo = F.leaky_relu(l12[..., :W1], slope)
    hi = l12[..., W1:] * l4
    return torch.cat([lo, hi], dim=-1)


class TestBlockCatFusedOp(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_matches_eager_cpu_fp32(self):
        l12 = torch.randn(8, 16, dtype=torch.float32)
        l4 = torch.randn(8, 10, dtype=torch.float32)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 6)
        ref = _eager_block_cat(l12, l4, 0.01, 6)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-6, atol=1e-6))

    def test_backward_matches_eager_cpu_fp32(self):
        l12 = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
        l4 = torch.randn(8, 10, dtype=torch.float32, requires_grad=True)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 6)
        out.sum().backward()
        g12_op, g4_op = l12.grad.clone(), l4.grad.clone()
        l12.grad = None
        l4.grad = None
        ref = _eager_block_cat(l12, l4, 0.01, 6)
        ref.sum().backward()
        self.assertTrue(torch.allclose(l12.grad, g12_op, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(l4.grad, g4_op, rtol=1e-6, atol=1e-6))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_forward_matches_eager_gpu_bf16(self):
        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        ref = _eager_block_cat(l12, l4, 0.01, 32)
        # bf16 + Triton: small ULP drift is expected
        self.assertTrue(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))

    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_backward_matches_eager_gpu_bf16(self):
        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = torch.ops.aiter.block_cat_fused(l12, l4, 0.01, 32)
        out.sum().backward()
        g12_op, g4_op = l12.grad.clone(), l4.grad.clone()
        l12.grad = None
        l4.grad = None
        ref = _eager_block_cat(l12, l4, 0.01, 32)
        ref.sum().backward()
        self.assertTrue(torch.allclose(l12.grad, g12_op, rtol=2e-2, atol=2e-2))
        self.assertTrue(torch.allclose(l4.grad, g4_op, rtol=2e-2, atol=2e-2))


class TestBlockCatFusedMatcher(unittest.TestCase):
    """Pre-grad matcher tests.

    Compiles a small module that constructs the target cat pattern and
    asserts the matcher fires (or does not, for negative cases) by
    reading the inductor counter.
    """

    def setUp(self):
        torch.manual_seed(0)
        counters.clear()

    def _compile_with_pass(self, fn):
        import torch._inductor.config as ic
        ic.pre_grad_custom_pass = block_cat_fused_pre_grad_pass
        return torch.compile(fn, mode="default", dynamic=False)

    @unittest.skipIf(not HAS_GPU, "GPU + Inductor required")
    def test_matcher_fires_on_block_cat_pattern(self):
        W1 = 32

        def f(l12, l4):
            lo = F.leaky_relu(l12[..., :W1], 0.01)
            hi = l12[..., W1:] * l4
            return torch.cat([lo, hi], dim=-1)

        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        ref = f(l12, l4)

        compiled = self._compile_with_pass(f)
        out = compiled(l12, l4)
        self.assertTrue(torch.allclose(out, ref, rtol=2e-2, atol=2e-2))
        self.assertEqual(counters["inductor"]["block_cat_fused"], 1)

    @unittest.skipIf(not HAS_GPU, "GPU + Inductor required")
    def test_matcher_does_not_fire_on_wrong_cat_axis(self):
        W1 = 32

        def f(l12, l4):
            lo = F.leaky_relu(l12[..., :W1], 0.01)
            hi = l12[..., W1:] * l4
            return torch.cat([lo, hi], dim=0)  # wrong axis

        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
        l4 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        compiled = self._compile_with_pass(f)
        try:
            _ = compiled(l12, l4)
        except Exception:
            pass
        self.assertEqual(counters["inductor"]["block_cat_fused"], 0)

    @unittest.skipIf(not HAS_GPU, "GPU + Inductor required")
    def test_matcher_does_not_fire_on_mismatched_leading_shape(self):
        W1 = 32

        def f(l12, l4):
            lo = F.leaky_relu(l12[..., :W1], 0.01)
            hi = l12[..., W1:] * l4
            return torch.cat([lo, hi], dim=-1)

        l12 = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
        l4 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)  # 32 != 64
        compiled = self._compile_with_pass(f)
        try:
            compiled(l12, l4)
        except Exception:
            pass
        self.assertEqual(counters["inductor"]["block_cat_fused"], 0)


if __name__ == "__main__":
    unittest.main()
