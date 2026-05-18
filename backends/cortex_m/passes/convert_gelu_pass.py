# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Phase 2 of KWT transformer support: rewrite

    cortex_m.dequantize_per_tensor → aten.gelu → cortex_m.quantize_per_tensor

into a single `cortex_m::quantized_gelu` op, folding the surrounding
quant/dequant scales/zps into the new op's args and propagating the
`approximate` kwarg ("none" vs "tanh") as the boolean `approximate_tanh`
flag.

GELU is pointwise and stateless, so unlike LayerNorm there's no
gamma/beta to extract — the AOT runtime side just needs the four
(in_zp, in_scale, out_zp, out_scale) numbers + the approximate flag to
precompute the 256-byte LUT the standalone kernel gathers from.
"""

from __future__ import annotations

import executorch.backends.cortex_m.ops.operators  # noqa: F401

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


class ConvertGELUPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for gelu_node in list(graph_module.graph.nodes):
            if gelu_node.op != "call_function":
                continue
            if gelu_node.target != exir_ops.edge.aten.gelu.default:
                continue

            dequant = gelu_node.args[0]
            if not _is_dequantize(dequant):
                continue
            if len(gelu_node.users) != 1:
                continue
            quant = next(iter(gelu_node.users))
            if not _is_quantize(quant):
                continue

            input_q_tensor = dequant.args[0]
            input_scale = float(dequant.args[1])
            input_zp = int(dequant.args[2])
            output_scale = float(quant.args[1])
            output_zp = int(quant.args[2])

            # `approximate` is keyword-only on aten.gelu, but upstream
            # arg-normalization passes can shift it to positional
            # gelu_node.args[1].  Read whichever is present (default
            # 'none' if neither).
            approximate = "none"
            if "approximate" in gelu_node.kwargs:
                approximate = str(gelu_node.kwargs["approximate"])
            elif len(gelu_node.args) >= 2 and isinstance(gelu_node.args[1], str):
                approximate = gelu_node.args[1]
            approximate_tanh = approximate == "tanh"

            new_args = (
                input_q_tensor,
                input_zp,
                input_scale,
                output_zp,
                output_scale,
                approximate_tanh,
            )

            with graph_module.graph.inserting_before(quant):
                new_node = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.cortex_m.quantized_gelu.default,
                    args=new_args,
                    kwargs={},
                )
                new_node.meta = dict(quant.meta)

            quant.replace_all_uses_with(new_node)
            nodes_to_erase.extend([quant, gelu_node, dequant])
            modified = True

        for n in nodes_to_erase:
            try:
                graph_module.graph.erase_node(n)
            except RuntimeError:
                pass

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
