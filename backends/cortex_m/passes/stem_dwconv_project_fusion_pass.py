# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Fuse the MV2 first-block chain: stem conv (3x3 stride-2 pad-1, Cin=3)
+ the existing quantized_dwconv2d_conv2d_fused (B0's dwconv+project)
into one op.

Runs AFTER DwconvProjectFusionPass, which is responsible for producing
the dwconv+project fused op (the downstream half of the pattern).
"""

from __future__ import annotations

import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class StemDwconvProjectFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for dwp_node in list(graph_module.graph.nodes):
            if dwp_node.op != "call_function":
                continue
            if (
                dwp_node.target
                != exir_ops.edge.cortex_m.quantized_dwconv2d_conv2d_fused.default
            ):
                continue

            # The dwp's input is the candidate stem conv.
            stem_node = dwp_node.args[0]
            if (
                getattr(stem_node, "op", None) != "call_function"
                or stem_node.target
                != exir_ops.edge.cortex_m.quantized_conv2d.default
            ):
                continue

            # Stem must be 3x3 stride-2 pad-1 with Cin=3.
            (
                s_in,
                s_w, s_b,
                s_stride, s_pad, _s_dil,
                s_in_off, s_out_off,
                s_mults, s_shifts,
                s_amin, s_amax,
            ) = stem_node.args

            weight_shape = get_first_fake_tensor(s_w).shape
            if len(weight_shape) != 4:
                continue
            # OHWI: (out_c, kH, kW, in_c)
            if weight_shape[1] != 3 or weight_shape[2] != 3 or weight_shape[3] != 3:
                continue
            if tuple(s_stride) != (2, 2) or tuple(s_pad) != (1, 1):
                continue
            if len(stem_node.users) != 1:
                continue

            (
                _d_in,
                d_w, d_b,
                d_stride, d_pad,
                d_in_off, d_out_off,
                d_mults, d_shifts,
                d_amin, d_amax,
                p_w, p_b,
                p_in_off, p_out_off,
                p_mults, p_shifts,
                p_amin, p_amax,
            ) = dwp_node.args

            fused_args = (
                s_in,
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

            with graph_module.graph.inserting_before(dwp_node):
                new_node = graph_module.graph.call_function(
                    exir_ops.edge.cortex_m.quantized_stem_dwconv2d_conv2d_fused.default,
                    fused_args,
                )
                new_node.meta = dict(dwp_node.meta)

            dwp_node.replace_all_uses_with(new_node)
            nodes_to_erase.append(dwp_node)
            nodes_to_erase.append(stem_node)
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
