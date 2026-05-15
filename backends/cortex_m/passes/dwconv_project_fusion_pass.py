# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Pattern-match MV2 B0-style inverted-residual blocks where there is no
expand conv (expand_ratio == 1), just a 3x3 depthwise followed by a 1x1
project.

In standard torchvision MV2, only features[1] (the first inverted-residual)
has expand_ratio=1.  Its 32-channel dwconv output at 112^2 (= 401 KB at
width=1.0/r=224) is one of the dominant arena tensors, but it isn't
matched by ExpandDwconvFusionPass because no expand precedes the dwconv.

This pass fills that gap.  It runs AFTER both
ExpandDwconvFusionPass and ExpandDwconvProjectFusionPass so it only
matches dwconv->conv2d pairs that weren't already absorbed into a 3/4-op
fusion (which would have included an expand).
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


class DwconvProjectFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        for project_node in list(graph_module.graph.nodes):
            if project_node.op != "call_function":
                continue
            if (
                project_node.target
                != exir_ops.edge.cortex_m.quantized_conv2d.default
            ):
                continue

            dw_node = project_node.args[0]
            if (
                getattr(dw_node, "op", None) != "call_function"
                or dw_node.target
                != exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default
            ):
                continue

            # Project must be 1x1, stride 1, padding 0.
            (
                _p_in,
                p_w, p_b,
                p_stride, p_pad, _p_dil,
                p_in_off, p_out_off,
                p_mults, p_shifts,
                p_amin, p_amax,
            ) = project_node.args

            weight_shape = get_first_fake_tensor(p_w).shape
            if len(weight_shape) != 4 or weight_shape[1] != 1 or weight_shape[2] != 1:
                continue
            if tuple(p_stride) != (1, 1) or tuple(p_pad) != (0, 0):
                continue

            if len(dw_node.users) != 1:
                continue

            (
                d_in,
                d_w, d_b,
                d_stride, d_pad, _d_dil,
                d_dm,
                d_in_off, d_out_off,
                d_mults, d_shifts,
                d_amin, d_amax,
            ) = dw_node.args

            if int(d_dm) != 1:
                continue

            fused_args = (
                d_in,
                d_w, d_b,
                list(d_stride), list(d_pad),
                int(d_in_off), int(d_out_off), d_mults, d_shifts,
                int(d_amin), int(d_amax),
                p_w, p_b,
                int(p_in_off), int(p_out_off), p_mults, p_shifts,
                int(p_amin), int(p_amax),
            )

            with graph_module.graph.inserting_before(project_node):
                fused_node = graph_module.graph.call_function(
                    exir_ops.edge.cortex_m.quantized_dwconv2d_conv2d_fused.default,
                    fused_args,
                )
                fused_node.meta = dict(project_node.meta)

            project_node.replace_all_uses_with(fused_node)
            nodes_to_erase.append(project_node)
            nodes_to_erase.append(dw_node)
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
