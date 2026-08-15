# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.transforms.convert_conv1d_to_conv2d_pass import (
    ConvertConv1dToConv2dPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._program import _transform


class Conv1d(torch.nn.Module):
    def __init__(self, bias: bool = True, groups: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 3, padding=1, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(x)


class BufferConv1d(torch.nn.Module):
    def __init__(self, persistent: bool):
        super().__init__()
        self.register_buffer("weight", torch.randn(4, 2, 3), persistent=persistent)

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1)


class ConstantConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 2, 3)

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1)


class SharedConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 2, 3))

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1) + F.conv1d(x, self.weight, padding=1)


class MixedUseWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 2, 3))

    def forward(self, x):
        return F.conv1d(x, self.weight, padding=1), self.weight


class WeightUsedAsConvInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.randn(4, 2, 3))
        self.other_weight = torch.nn.Parameter(torch.randn(3, 2, 1))

    def forward(self, x):
        return F.conv1d(x, self.shared, padding=1), F.conv1d(
            self.shared, self.other_weight
        )


def _edge(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    return to_edge(
        torch.export.export(model.eval(), inputs),
        compile_config=EdgeCompileConfig(_skip_dim_order=True),
    ).exported_program()


def _convert_and_compare(model: torch.nn.Module, x: torch.Tensor):
    reference = model(x)
    edge = _edge(model, (x,))
    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))
    torch.testing.assert_close(converted.module()(x), reference)
    return converted


@pytest.mark.parametrize("bias, groups", [(True, 1), (False, 1), (False, 2)])
def test_convert_conv1d_to_conv2d(bias: bool, groups: int):
    converted = _convert_and_compare(
        Conv1d(bias=bias, groups=groups), torch.randn(1, 2, 8)
    )
    [conv] = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert conv.args[1].meta["val"].shape == torch.Size([4, 2 // groups, 1, 3])
    assert conv.args[3:6] == ([1, 1], [0, 1], [1, 1])
    assert conv.args[0].target == exir_ops.edge.aten.unsqueeze_copy.default
    assert next(iter(conv.users)).target == exir_ops.edge.aten.squeeze_copy.dim


@pytest.mark.parametrize(
    "model", [BufferConv1d(True), BufferConv1d(False), ConstantConv1d()]
)
def test_convert_lifted_weight_storage(model: torch.nn.Module):
    _convert_and_compare(model, torch.randn(1, 2, 8))


def test_convert_shared_conv1d_weight_once():
    converted = _convert_and_compare(SharedConv1d(), torch.randn(1, 2, 8))
    convs = [
        node
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]

    assert len(convs) == 2
    assert convs[0].args[1] is convs[1].args[1]
    assert convs[0].args[1].meta["val"].dim() == 4


def test_skip_weight_with_non_conv1d_user():
    model = MixedUseWeight().eval()
    x = torch.randn(1, 2, 8)
    reference = model(x)
    edge = _edge(model, (x,))
    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified
    actual = edge.module()(x)
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])


def test_skip_weight_used_as_another_conv1d_input():
    model = WeightUsedAsConvInput().eval()
    x = torch.randn(1, 2, 8)
    reference = model(x)
    edge = _edge(model, (x,))
    converted = _transform(edge, ConvertConv1dToConv2dPass(edge))

    actual = converted.module()(x)
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])
    weight_ranks = sorted(
        node.args[1].meta["val"].dim()
        for node in converted.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    )
    assert weight_ranks == [3, 4]


def test_rank4_weight_with_singleton_args_is_not_converted():
    edge = _edge(torch.nn.Conv2d(2, 4, 3, padding=1), (torch.randn(1, 2, 8, 8),))
    [conv] = [
        node
        for node in edge.graph.nodes
        if node.target == exir_ops.edge.aten.convolution.default
    ]
    args = list(conv.args)
    args[3:6] = ([1], [1], [1])
    conv.args = tuple(args)

    result = ConvertConv1dToConv2dPass(edge).call(edge.graph_module)

    assert not result.modified
    assert conv.args[3:6] == ([1], [1], [1])


def test_conversion_is_idempotent():
    converted = _convert_and_compare(Conv1d(), torch.randn(1, 2, 8))

    result = ConvertConv1dToConv2dPass(converted).call(converted.graph_module)

    assert not result.modified
