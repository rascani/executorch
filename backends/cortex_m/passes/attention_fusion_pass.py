# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).

"""Phase 5 of KWT transformer support: fold the
    cortex_m.quantized_batch_matmul (QK^T)
      → cortex_m.softmax
        [→ cortex_m.dequantize_per_tensor → cortex_m.quantize_per_tensor]   # optional rezp
          → cortex_m.quantized_batch_matmul (AV)
chain into a single cortex_m.quantized_fused_attention op so the
standalone runtime can stream the (S × S) score matrix row-by-row.

The optional dequant→quant in the middle appears because PT2E sees
softmax's fixed output spec (zp=-128, scale=1/256) and the AV BMM's
input observer separately; if their zps match (which they do for
softmax → BMM since both want zp=-128), the rezp is a near no-op but
its scale change still needs to be folded into the AV BMM's
requantize multiplier.
"""

from __future__ import annotations

import executorch.backends.cortex_m.ops.operators  # noqa: F401

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult


def _is(node, target) -> bool:
    return getattr(node, "op", None) == "call_function" and node.target == target


def _single_user(node):
    if len(node.users) != 1:
        return None
    return next(iter(node.users))


class AttentionFusionPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        nodes_to_erase: list = []

        bmm_op = exir_ops.edge.cortex_m.quantized_batch_matmul.default
        softmax_op = exir_ops.edge.cortex_m.softmax.default
        dequant_op = exir_ops.edge.cortex_m.dequantize_per_tensor.default
        quant_op = exir_ops.edge.cortex_m.quantize_per_tensor.default
        fused_op = exir_ops.edge.cortex_m.quantized_fused_attention.default

        for sm_node in list(graph_module.graph.nodes):
            if not _is(sm_node, softmax_op):
                continue

            # Backward: softmax input must be a QK^T BMM whose only user
            # is this softmax.
            bmm_qk = sm_node.args[0]
            if not _is(bmm_qk, bmm_op):
                continue
            if _single_user(bmm_qk) is not sm_node:
                continue

            # Forward: softmax → [dequant → quant →] BMM (AV).
            next_node = _single_user(sm_node)
            if next_node is None:
                continue

            rezp_dequant = None
            rezp_quant = None
            if _is(next_node, dequant_op):
                rezp_dequant = next_node
                rezp_quant = _single_user(rezp_dequant)
                if not _is(rezp_quant, quant_op):
                    continue
                bmm_av = _single_user(rezp_quant)
                if bmm_av is None:
                    continue
            else:
                bmm_av = next_node

            if not _is(bmm_av, bmm_op):
                continue
            # bmm_av.args[0] should be the softmax / quant output we walked through.
            expected_lhs = rezp_quant if rezp_quant is not None else sm_node
            if bmm_av.args[0] is not expected_lhs:
                continue

            # If there's a rezp, require zps to match (no-op rezp on zp);
            # we don't yet support an effective zp shift inside the fused op.
            if rezp_dequant is not None and rezp_quant is not None:
                if int(rezp_dequant.args[2]) != int(rezp_quant.args[2]):
                    continue

            # bmm_qk args: (lhs=q, lhs_offset=q_off, rhs_transposed=k,
            #               rhs_offset=k_off, output_zp, output_mult, output_shift)
            (q_node, q_off, k_node, k_off,
             qk_output_zp, qk_output_mult, qk_output_shift) = bmm_qk.args

            # softmax args: (input, dim, input_zp, output_zp, input_mult,
            #                input_shift, diff_min)
            (_sm_in, sm_dim, sm_input_zp, sm_output_zp,
             sm_input_mult, sm_input_shift, sm_diff_min) = sm_node.args
            # Sanity: softmax input zp matches QK^T output zp.
            if int(sm_input_zp) != int(qk_output_zp):
                continue
            # Only softmax over the last dim of (B, S, S) — the standalone
            # streaming kernel assumes per-row.
            if int(sm_dim) not in (-1, 2):
                continue

            # bmm_av args: (lhs=probs/quant, lhs_offset=probs_off,
            #               rhs_transposed=v, rhs_offset=v_off,
            #               output_zp, output_mult, output_shift)
            (_av_lhs, probs_off, v_node, v_off,
             av_output_zp, av_output_mult, av_output_shift) = bmm_av.args

            new_args = (
                q_node, k_node, v_node,
                int(q_off), int(k_off),
                int(qk_output_zp), int(qk_output_mult), int(qk_output_shift),
                int(sm_input_mult), int(sm_input_shift), int(sm_diff_min),
                int(v_off),
                int(av_output_zp), int(av_output_mult), int(av_output_shift),
            )

            with graph_module.graph.inserting_before(bmm_av):
                new_node = graph_module.graph.create_node(
                    "call_function",
                    target=fused_op,
                    args=new_args,
                    kwargs={},
                )
                new_node.meta = dict(bmm_av.meta)

            bmm_av.replace_all_uses_with(new_node)
            nodes_to_erase.append(bmm_av)
            if rezp_quant is not None:
                nodes_to_erase.append(rezp_quant)
            if rezp_dequant is not None:
                nodes_to_erase.append(rezp_dequant)
            nodes_to_erase.append(sm_node)
            nodes_to_erase.append(bmm_qk)
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
