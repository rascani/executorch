# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

_PRESERVED_META_KEYS = ("input_qparams", "output_qparams", "custom")


class ConvertConv1dToConv2dPass(ExportPass):
    """Express Conv1d as Conv2d with a unit-height spatial dimension."""

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def _unsqueeze_weight(self, weight_node: torch.fx.Node) -> None:
        weight = get_param_tensor(self.exported_program, weight_node)
        if weight is None:
            raise RuntimeError("Conv1d weight must be a lifted constant tensor")
        if weight.dim() == 4:
            return
        if weight.dim() != 3:
            raise RuntimeError(f"Expected a rank-3 Conv1d weight, got {weight.dim()}")

        weight_2d = weight.unsqueeze(2).contiguous()
        signature = self.exported_program.graph_signature
        if target := signature.inputs_to_parameters.get(weight_node.name):
            parameter = self.exported_program.state_dict[target]
            self.exported_program.state_dict[target] = torch.nn.Parameter(
                weight_2d, requires_grad=parameter.requires_grad
            )
        elif target := signature.inputs_to_buffers.get(weight_node.name):
            if target in signature.non_persistent_buffers:
                self.exported_program.constants[target] = weight_2d
            else:
                self.exported_program.state_dict[target] = weight_2d
        elif target := signature.inputs_to_lifted_tensor_constants.get(
            weight_node.name
        ):
            self.exported_program.constants[target] = weight_2d
        elif weight_node.op == "get_attr":
            setattr(weight_node.graph.owning_module, weight_node.target, weight_2d)
        else:
            raise RuntimeError("Conv1d weight must be a parameter, buffer, or constant")

        weight_node.meta["val"] = weight_node.meta["val"].unsqueeze(2)

    @staticmethod
    def _conv1d_weight_node(node: torch.fx.Node) -> torch.fx.Node | None:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.convolution.default
            or len(node.args) < 2
        ):
            return None
        weight_node = node.args[1]
        if not isinstance(weight_node, torch.fx.Node):
            return None
        weight = weight_node.meta.get("val")
        if not isinstance(weight, torch.Tensor) or weight.dim() != 3:
            return None
        return weight_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        preserved_meta: dict[str, dict] = {}
        modified = False
        candidates = {
            node: weight_node
            for node in graph.nodes
            if (weight_node := self._conv1d_weight_node(node)) is not None
        }

        for node, weight_node in candidates.items():
            if any(
                candidates.get(user) is not weight_node
                or any(
                    arg is weight_node
                    for index, arg in enumerate(user.args)
                    if index != 1
                )
                for user in weight_node.users
            ):
                continue
            self._unsqueeze_weight(weight_node)

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                raise RuntimeError("Conv1d input must be an FX node")
            with graph.inserting_before(node):
                input_2d = graph.call_function(
                    exir_ops.edge.aten.unsqueeze_copy.default,
                    args=(input_node, 2),
                )

            args = list(node.args)
            args[0] = input_2d
            args[3] = [1, *list(args[3])]
            args[4] = [0, *list(args[4])]
            args[5] = [1, *list(args[5])]
            args[7] = [0, *list(args[7])]
            node.args = tuple(args)

            with graph.inserting_after(node):
                output_1d = graph.call_function(
                    exir_ops.edge.aten.squeeze_copy.dim,
                    args=(node, 2),
                )
            for user in list(node.users):
                if user is not output_1d:
                    user.replace_input_with(node, output_1d)

            preserved_meta[node.name] = {
                key: node.meta[key] for key in _PRESERVED_META_KEYS if key in node.meta
            }
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        result = super().call(graph_module)
        for node in result.graph_module.graph.nodes:
            node.meta.update(preserved_meta.get(node.name, {}))
        return PassResult(result.graph_module, True)
