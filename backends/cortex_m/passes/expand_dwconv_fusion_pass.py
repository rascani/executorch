# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Pattern-match MV2 inverted-residual expand+dwconv chains and fuse them
into a single cortex_m.quantized_conv2d_dwconv2d_fused op.

The unfused chain (1x1 conv -> 3x3 dwconv) materializes a full HxWxC_expand
int8 tensor between the two ops; this is the dominant memory peak in early
MV2 blocks (e.g. 1.2 MB at width=1.0/r=224 B2).  The fused op signals to
the standalone codegen that the runtime kernel should stream expand rows
through a 3-row rolling buffer that the dwconv consumes immediately,
eliminating the intermediate.

Constraints required for the pattern to fuse:
    * Expand is cortex_m.quantized_conv2d with weight spatial = (1, 1),
      stride = (1, 1), padding = (0, 0), dilation = (1, 1).
    * Dwconv is cortex_m.quantized_depthwise_conv2d with depth_multiplier=1.
    * Expand output has exactly one consumer (the dwconv).  This rules out
      blocks where the expand output also feeds a residual add or other
      branch.
"""

from __future__ import annotations

import logging

import executorch.backends.cortex_m.ops.operators  # noqa: F401  (op registration)
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult

logger = logging.getLogger(__name__)


class ExpandDwconvFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for dw_node in list(graph_module.graph.nodes):
            if dw_node.op != "call_function":
                continue
            if (
                dw_node.target
                != exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default
            ):
                continue

            expand_node = dw_node.args[0]
            if (
                getattr(expand_node, "op", None) != "call_function"
                or expand_node.target
                != exir_ops.edge.cortex_m.quantized_conv2d.default
            ):
                continue

            # Expand weight: OHWI = (C_out, H, W, C_in).  Need 1x1.
            weight_node = expand_node.args[1]
            weight_shape = get_first_fake_tensor(weight_node).shape
            if len(weight_shape) != 4 or weight_shape[1] != 1 or weight_shape[2] != 1:
                continue

            (
                e_in,
                e_w,
                e_b,
                e_stride,
                e_pad,
                _e_dil,
                e_in_off,
                e_out_off,
                e_mults,
                e_shifts,
                e_amin,
                e_amax,
            ) = expand_node.args

            if tuple(e_stride) != (1, 1) or tuple(e_pad) != (0, 0):
                continue

            # Single-user invariant: expand output may only feed the dwconv.
            if len(expand_node.users) != 1:
                continue

            (
                _d_in,
                d_w,
                d_b,
                d_stride,
                d_pad,
                _d_dil,
                d_dm,
                d_in_off,
                d_out_off,
                d_mults,
                d_shifts,
                d_amin,
                d_amax,
            ) = dw_node.args

            if int(d_dm) != 1:
                continue

            fused_args = (
                e_in,
                e_w,
                e_b,
                int(e_in_off),
                int(e_out_off),
                e_mults,
                e_shifts,
                int(e_amin),
                int(e_amax),
                d_w,
                d_b,
                list(d_stride),
                list(d_pad),
                int(d_in_off),
                int(d_out_off),
                d_mults,
                d_shifts,
                int(d_amin),
                int(d_amax),
            )

            with graph_module.graph.inserting_before(dw_node):
                fused_node = graph_module.graph.call_function(
                    exir_ops.edge.cortex_m.quantized_conv2d_dwconv2d_fused.default,
                    fused_args,
                )
                # Copy meta from the dwconv (output shape/qparams match).
                fused_node.meta = dict(dw_node.meta)

            dw_node.replace_all_uses_with(fused_node)
            nodes_to_erase.append(dw_node)
            nodes_to_erase.append(expand_node)
            modified = True

        for n in nodes_to_erase:
            graph_module.graph.erase_node(n)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
