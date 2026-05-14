# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Extend the 2-op expand+dwconv fusion to also absorb the project conv1x1
that follows in every MV2 inverted-residual block, plus the optional
quantized_add that follows when the block has a residual skip path.

Pattern this pass matches:

    quantized_conv2d_dwconv2d_fused (F)
        -> quantized_conv2d (P, 1x1)
        [-> quantized_add (A, with the other input == F's input)]

Constraints:
    * P is a 1x1 conv: weight shape (out_c, 1, 1, in_c), stride (1, 1),
      padding (0, 0), dilation (1, 1).
    * F has exactly one user (P), and P has either zero users or exactly
      one (A).
    * When A is present, one of A's two inputs must be identical to F's
      input node (the inverted-residual skip path is the block input).

When the residual is absent the pass still fuses F + P (producing a node
with residual_input = None).  This single fused op replaces all 2-3 ops
with one runtime call, and eliminates the dwconv-output and project-
output intermediates from the activation arena.

This pass MUST run after ExpandDwconvFusionPass, since it operates on
the 2-op fused nodes it produces.
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


class ExpandDwconvProjectFusionPass(ExportPass):
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

            # Project conv must follow a 2-op fused node.
            fused_node = project_node.args[0]
            if (
                getattr(fused_node, "op", None) != "call_function"
                or fused_node.target
                != exir_ops.edge.cortex_m.quantized_conv2d_dwconv2d_fused.default
            ):
                continue

            # Project must be 1x1, stride 1, padding 0.
            (
                _p_in,
                p_w,
                p_b,
                p_stride,
                p_pad,
                _p_dil,
                p_in_off,
                p_out_off,
                p_mults,
                p_shifts,
                p_amin,
                p_amax,
            ) = project_node.args

            weight_shape = get_first_fake_tensor(p_w).shape
            if len(weight_shape) != 4 or weight_shape[1] != 1 or weight_shape[2] != 1:
                continue
            if tuple(p_stride) != (1, 1) or tuple(p_pad) != (0, 0):
                continue

            # The 2-op fused node must have exactly one user (this project).
            if len(fused_node.users) != 1:
                continue

            # Check for an optional residual add downstream of the project.
            add_node = None
            if len(project_node.users) == 1:
                cand = next(iter(project_node.users))
                if (
                    cand.op == "call_function"
                    and cand.target == exir_ops.edge.cortex_m.quantized_add.default
                ):
                    # Residual skip must be the same node as the fused op's
                    # input (the block input feeds both expand and add).
                    fused_input = fused_node.args[0]
                    self_arg = cand.args[0]
                    other_arg = cand.args[4]
                    if self_arg is project_node and other_arg is fused_input:
                        add_node = cand
                        residual_input = other_arg
                        residual_self_args_offset = 0
                    elif other_arg is project_node and self_arg is fused_input:
                        add_node = cand
                        residual_input = self_arg
                        # quantized_add args are symmetric in semantics but
                        # carry self_* and other_* params at fixed positions.
                        # When the project output is `other`, swap so the
                        # fused op always treats project as `self`.
                        residual_self_args_offset = 4
                    else:
                        add_node = None

            # Build the fused args.  Layout (mirrors operators.py schema):
            (
                f_in,
                e_w, e_b,
                e_in_off, e_out_off, e_mults, e_shifts, e_amin, e_amax,
                d_w, d_b,
                d_stride, d_pad,
                d_in_off, d_out_off, d_mults, d_shifts, d_amin, d_amax,
            ) = fused_node.args

            if add_node is None:
                # Default residual params (won't be consumed).
                fused_args = (
                    f_in,
                    e_w, e_b,
                    int(e_in_off), int(e_out_off), e_mults, e_shifts,
                    int(e_amin), int(e_amax),
                    d_w, d_b,
                    list(d_stride), list(d_pad),
                    int(d_in_off), int(d_out_off), d_mults, d_shifts,
                    int(d_amin), int(d_amax),
                    p_w, p_b,
                    int(p_in_off), int(p_out_off), p_mults, p_shifts,
                    int(p_amin), int(p_amax),
                    None,  # residual_input
                    0, 0, 0,  # self zp/mult/shift
                    0, 0, 0,  # other zp/mult/shift
                    0, 0, 0,  # output zp/mult/shift
                    -128, 127,  # residual amin/amax
                )
            else:
                # quantized_add args:
                # (self, self_zp, self_mult, self_shift, other, other_zp,
                #  other_mult, other_shift, output_zp, output_mult,
                #  output_shift, amin, amax)
                a_args = add_node.args
                if residual_self_args_offset == 0:
                    a_self_zp, a_self_m, a_self_s = a_args[1], a_args[2], a_args[3]
                    a_other_zp, a_other_m, a_other_s = a_args[5], a_args[6], a_args[7]
                else:
                    # Swap: project_out is the 'other' input to the add — read
                    # the other_* params for it and the self_* params for the
                    # skip side.
                    a_self_zp, a_self_m, a_self_s = a_args[5], a_args[6], a_args[7]
                    a_other_zp, a_other_m, a_other_s = a_args[1], a_args[2], a_args[3]
                a_out_zp, a_out_m, a_out_s = a_args[8], a_args[9], a_args[10]
                a_amin, a_amax = a_args[11], a_args[12]

                fused_args = (
                    f_in,
                    e_w, e_b,
                    int(e_in_off), int(e_out_off), e_mults, e_shifts,
                    int(e_amin), int(e_amax),
                    d_w, d_b,
                    list(d_stride), list(d_pad),
                    int(d_in_off), int(d_out_off), d_mults, d_shifts,
                    int(d_amin), int(d_amax),
                    p_w, p_b,
                    int(p_in_off), int(p_out_off), p_mults, p_shifts,
                    int(p_amin), int(p_amax),
                    residual_input,
                    int(a_self_zp), int(a_self_m), int(a_self_s),
                    int(a_other_zp), int(a_other_m), int(a_other_s),
                    int(a_out_zp), int(a_out_m), int(a_out_s),
                    int(a_amin), int(a_amax),
                )

            replacement_target = exir_ops.edge.cortex_m.quantized_conv2d_dwconv2d_conv2d_fused.default
            insert_before = add_node if add_node is not None else project_node
            terminal_node = add_node if add_node is not None else project_node

            with graph_module.graph.inserting_before(insert_before):
                new_node = graph_module.graph.call_function(
                    replacement_target,
                    fused_args,
                )
                new_node.meta = dict(terminal_node.meta)

            terminal_node.replace_all_uses_with(new_node)
            nodes_to_erase.append(terminal_node)
            if add_node is not None:
                nodes_to_erase.append(project_node)
            nodes_to_erase.append(fused_node)
            modified = True

        for n in nodes_to_erase:
            try:
                graph_module.graph.erase_node(n)
            except RuntimeError:
                # Already erased via cascading dead-code elimination
                pass

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
