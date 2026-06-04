# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposePowPass(ExportPass):
    """Rewrite `aten.pow(x, 2)` into `aten.mul(x, x)`, pre-annotation.

    The quantizer has no pattern for `pow`, so a squared term (e.g. the
    `real**2 + imag**2` magnitude block) is left in fp32. Squaring is exactly
    `x * x`, which the quantizer annotates and the existing passes lower to
    `cortex_m.quantized_mul`. Run this before annotation so the quantizer sees
    the multiply.
    """

    _TARGET = torch.ops.aten.pow.Tensor_Scalar

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != self._TARGET:
                continue
            if node.args[1] != 2:
                continue

            x = node.args[0]
            with graph.inserting_before(node):
                mul = create_node(
                    graph, torch.ops.aten.mul.Tensor, args=(x, x), from_node=node
                )
            node.replace_all_uses_with(mul)
            graph.erase_node(node)
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
