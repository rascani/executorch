# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Phase G: extend the Phase F quantize+stem+B0 fused op to also absorb
the following 3/4-op inverted-residual block (B1 in MV2).  The B0
project output (16 channels x 112^2 = 200 KB at 1.0/r=224) was the
new arena peak after Phase F.  Streaming it row-by-row into B1's
expand eliminates that allocation.
"""

from __future__ import annotations

import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class QuantizeStemInvertedResidualFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for ir_node in list(graph_module.graph.nodes):
            if ir_node.op != "call_function":
                continue
            if (
                ir_node.target
                != exir_ops.edge.cortex_m.quantized_conv2d_dwconv2d_conv2d_fused.default
            ):
                continue

            f_node = ir_node.args[0]
            if (
                getattr(f_node, "op", None) != "call_function"
                or f_node.target
                != exir_ops.edge.cortex_m.quantize_stem_dwconv2d_conv2d_fused.default
            ):
                continue
            if len(f_node.users) != 1:
                continue

            # Bail if the IR block has a residual — B1 in MV2 doesn't,
            # but we check defensively.  The residual_input position in
            # quantized_conv2d_dwconv2d_conv2d_fused args is 27.
            if len(ir_node.args) < 28 or ir_node.args[27] is not None:
                continue

            (
                q_in,
                q_scale, q_zp, q_qmin, q_qmax,
                s_w, s_b,
                s_in_off, s_out_off, s_mults, s_shifts, s_amin, s_amax,
                b0_dw_w, b0_dw_b,
                b0_dw_stride, b0_dw_pad,
                b0_dw_in_off, b0_dw_out_off, b0_dw_mults, b0_dw_shifts,
                b0_dw_amin, b0_dw_amax,
                b0_p_w, b0_p_b,
                b0_p_in_off, b0_p_out_off, b0_p_mults, b0_p_shifts,
                b0_p_amin, b0_p_amax,
            ) = f_node.args

            (
                _ir_in,
                b1_e_w, b1_e_b,
                b1_e_in_off, b1_e_out_off, b1_e_mults, b1_e_shifts,
                b1_e_amin, b1_e_amax,
                b1_dw_w, b1_dw_b,
                b1_dw_stride, b1_dw_pad,
                b1_dw_in_off, b1_dw_out_off, b1_dw_mults, b1_dw_shifts,
                b1_dw_amin, b1_dw_amax,
                b1_p_w, b1_p_b,
                b1_p_in_off, b1_p_out_off, b1_p_mults, b1_p_shifts,
                b1_p_amin, b1_p_amax,
                _residual_input,
                *_residual_fields,
            ) = ir_node.args

            fused_args = (
                q_in,
                float(q_scale), int(q_zp), int(q_qmin), int(q_qmax),
                s_w, s_b,
                int(s_in_off), int(s_out_off), s_mults, s_shifts,
                int(s_amin), int(s_amax),
                b0_dw_w, b0_dw_b,
                list(b0_dw_stride), list(b0_dw_pad),
                int(b0_dw_in_off), int(b0_dw_out_off), b0_dw_mults, b0_dw_shifts,
                int(b0_dw_amin), int(b0_dw_amax),
                b0_p_w, b0_p_b,
                int(b0_p_in_off), int(b0_p_out_off), b0_p_mults, b0_p_shifts,
                int(b0_p_amin), int(b0_p_amax),
                b1_e_w, b1_e_b,
                int(b1_e_in_off), int(b1_e_out_off), b1_e_mults, b1_e_shifts,
                int(b1_e_amin), int(b1_e_amax),
                b1_dw_w, b1_dw_b,
                list(b1_dw_stride), list(b1_dw_pad),
                int(b1_dw_in_off), int(b1_dw_out_off), b1_dw_mults, b1_dw_shifts,
                int(b1_dw_amin), int(b1_dw_amax),
                b1_p_w, b1_p_b,
                int(b1_p_in_off), int(b1_p_out_off), b1_p_mults, b1_p_shifts,
                int(b1_p_amin), int(b1_p_amax),
            )

            with graph_module.graph.inserting_before(ir_node):
                new_node = graph_module.graph.call_function(
                    exir_ops.edge.cortex_m.quantize_stem_inverted_residual_fused.default,
                    fused_args,
                )
                new_node.meta = dict(ir_node.meta)

            ir_node.replace_all_uses_with(new_node)
            nodes_to_erase.append(ir_node)
            nodes_to_erase.append(f_node)
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
