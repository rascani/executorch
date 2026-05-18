# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Phase 1 of KWT transformer support: rewrite

    cortex_m.dequantize_per_tensor → aten.native_layer_norm → getitem(0)
                                   → cortex_m.quantize_per_tensor

into a single `cortex_m::quantized_layer_norm` op, folding the surrounding
quant/dequant scales and zero-points into the new op's args.  γ (weight)
and β (bias) stay float — they're read by the runtime kernel directly
from .rodata.

Unlike the int8 conv / linear lowerings, LayerNorm's inner math is float
internally; the cortex_m op dequantizes, runs the float LN, and
requantizes.  See backends/cortex_m/ops/operators.py for the python
reference impl that this pass targets.
"""

from __future__ import annotations

import executorch.backends.cortex_m.ops.operators  # noqa: F401

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult


def _is_quantize(node) -> bool:
    return (
        getattr(node, "op", None) == "call_function"
        and node.target == exir_ops.edge.cortex_m.quantize_per_tensor.default
    )


def _is_dequantize(node) -> bool:
    return (
        getattr(node, "op", None) == "call_function"
        and node.target == exir_ops.edge.cortex_m.dequantize_per_tensor.default
    )


def _is_getitem(node, index) -> bool:
    import operator

    return (
        getattr(node, "op", None) == "call_function"
        and node.target == operator.getitem
        and len(node.args) >= 2
        and node.args[1] == index
    )


class ConvertLayerNormPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for ln_node in list(graph_module.graph.nodes):
            if ln_node.op != "call_function":
                continue
            if ln_node.target != exir_ops.edge.aten.native_layer_norm.default:
                continue

            # Input chain: ln.args[0] should be a cortex_m.dequantize_per_tensor.
            dequant = ln_node.args[0]
            if not _is_dequantize(dequant):
                continue

            # Output chain: ln has 3 outputs (out, mean, rstd); we need the
            # getitem(0) and verify its only user is a cortex_m.quantize.
            getitem_node = None
            for user in ln_node.users:
                if _is_getitem(user, 0):
                    getitem_node = user
                    break
            if getitem_node is None:
                continue
            if len(getitem_node.users) != 1:
                continue
            quant = next(iter(getitem_node.users))
            if not _is_quantize(quant):
                continue

            # Extract quant params from the surrounding nodes.
            # dequantize_per_tensor.args = (input, scale, zp, qmin, qmax, dtype)
            # quantize_per_tensor.args   = (input, scale, zp, qmin, qmax, dtype)
            input_q_tensor = dequant.args[0]
            input_scale = float(dequant.args[1])
            input_zp = int(dequant.args[2])
            output_scale = float(quant.args[1])
            output_zp = int(quant.args[2])

            # LN op args: (input, normalized_shape, weight, bias, eps).
            # normalized_shape is implicit in weight.shape for our op.
            weight = ln_node.args[2]
            bias = ln_node.args[3] if len(ln_node.args) > 3 else None
            eps = float(ln_node.args[4]) if len(ln_node.args) > 4 else 1e-5

            new_args = (
                input_q_tensor,
                input_zp,
                input_scale,
                weight,
                bias,
                eps,
                output_zp,
                output_scale,
            )

            with graph_module.graph.inserting_before(quant):
                new_node = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.cortex_m.quantized_layer_norm.default,
                    args=new_args,
                    kwargs={},
                )
                new_node.meta = dict(quant.meta)

            quant.replace_all_uses_with(new_node)
            nodes_to_erase.extend([quant, getitem_node, ln_node, dequant])
            modified = True

        for n in nodes_to_erase:
            try:
                graph_module.graph.erase_node(n)
            except RuntimeError:
                # Already removed via a downstream replace_all_uses_with cleanup.
                pass

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
