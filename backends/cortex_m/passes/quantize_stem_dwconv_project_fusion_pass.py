# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Fuse cortex_m.quantize_per_tensor into the existing stem chain.  The
quantized input tensor (3*H*W bytes for the typical (1,3,H,W) input —
150 KB at H=W=224) currently materializes in the activation arena
before the stem reads it.  Streaming the quantize row-by-row into the
stem-fused kernel eliminates that allocation.

Runs after StemDwconvProjectFusionPass.
"""

from __future__ import annotations

import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class QuantizeStemDwconvProjectFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for stem_node in list(graph_module.graph.nodes):
            if stem_node.op != "call_function":
                continue
            if (
                stem_node.target
                != exir_ops.edge.cortex_m.quantized_stem_dwconv2d_conv2d_fused.default
            ):
                continue

            quant_node = stem_node.args[0]
            if (
                getattr(quant_node, "op", None) != "call_function"
                or quant_node.target
                != exir_ops.edge.cortex_m.quantize_per_tensor.default
            ):
                continue
            if len(quant_node.users) != 1:
                continue

            # cortex_m.quantize_per_tensor args: (input, scale, zp, qmin, qmax, dtype)
            (
                q_in,
                q_scale,
                q_zp,
                q_qmin,
                q_qmax,
                _q_dtype,
            ) = quant_node.args

            (
                _s_in,
                s_w, s_b,
                s_in_off, s_out_off, s_mults, s_shifts, s_amin, s_amax,
                d_w, d_b, d_stride, d_pad,
                d_in_off, d_out_off, d_mults, d_shifts, d_amin, d_amax,
                p_w, p_b,
                p_in_off, p_out_off, p_mults, p_shifts, p_amin, p_amax,
            ) = stem_node.args

            fused_args = (
                q_in,
                float(q_scale), int(q_zp), int(q_qmin), int(q_qmax),
                s_w, s_b,
                int(s_in_off), int(s_out_off), s_mults, s_shifts,
                int(s_amin), int(s_amax),
                d_w, d_b,
                list(d_stride), list(d_pad),
                int(d_in_off), int(d_out_off), d_mults, d_shifts,
                int(d_amin), int(d_amax),
                p_w, p_b,
                int(p_in_off), int(p_out_off), p_mults, p_shifts,
                int(p_amin), int(p_amax),
            )

            with graph_module.graph.inserting_before(stem_node):
                new_node = graph_module.graph.call_function(
                    exir_ops.edge.cortex_m.quantize_stem_dwconv2d_conv2d_fused.default,
                    fused_args,
                )
                new_node.meta = dict(stem_node.meta)

            stem_node.replace_all_uses_with(new_node)
            nodes_to_erase.append(stem_node)
            nodes_to_erase.append(quant_node)
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
