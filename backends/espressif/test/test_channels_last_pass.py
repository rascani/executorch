# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.kernels.quantized  # noqa: F401
import pytest
import torch
from executorch.backends.espressif.test.tester import EspTester
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops

_PERMUTE = exir_ops.edge.aten.permute_copy.default
_LAYOUT_PERMUTE = exir_ops.edge.channels_last.permute_copy.default
_CLONE = exir_ops.edge.dim_order_ops._clone_dim_order.default
_MUL = exir_ops.edge.aten.mul.Tensor
_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


class PermutedMul(torch.nn.Module):
    def __init__(self, shared_float=False, output_order=(0, 2, 3, 1)):
        super().__init__()
        self.shared_float = shared_float
        self.output_order = output_order

    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()
        product = (x * y).permute(self.output_order).contiguous()
        if self.shared_float:
            return product, torch.relu(x)
        return product


class Mul(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
@pytest.mark.parametrize("shared_float", [False, True])
def test_multiply_layout_boundaries(entry_point, shared_float):
    inputs = (
        torch.linspace(-1, 2, 120).reshape(2, 3, 5, 4),
        torch.linspace(-3, 1, 120).reshape(2, 3, 5, 4),
    )
    tester = EspTester(PermutedMul(shared_float=shared_float), inputs)
    tester.quantize().export()
    if entry_point == "combined":
        tester.to_edge_transform_and_lower()
    else:
        tester.to_edge().check_node_count({_PERMUTE: 3, _CLONE: 3}).run_passes()
    tester.check_node_count(
        {
            _PERMUTE: 1 if shared_float else 0,
            _CLONE: 0,
            _MUL: 1,
            _QUANTIZE: 3,
            _DEQUANTIZE: 3,
        }
    ).run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    graph = tester.get_artifact().exported_program().graph
    multiply = next(node for node in graph.nodes if node.target == _MUL)
    assert tuple(multiply.meta["val"].shape) == (2, 3, 5, 4)
    assert multiply.meta["val"].is_contiguous()
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert not program.execution_plan[0].delegates
    assert "aten::mul" in {op.name for op in program.execution_plan[0].operators}


def test_non_inverse_output_permutation_is_preserved():
    inputs = (torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4))
    tester = (
        EspTester(PermutedMul(output_order=(0, 2, 1, 3)), inputs)
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .check_node_count({_PERMUTE: 1, _MUL: 1, _QUANTIZE: 3, _DEQUANTIZE: 3})
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )
    graph = tester.get_artifact().exported_program().graph
    permute = next(node for node in graph.nodes if node.target == _PERMUTE)
    assert list(permute.args[1]) == [0, 1, 3, 2]
    assert tuple(permute.meta["val"].shape) == (2, 3, 4, 5)


@pytest.mark.parametrize("shape", [(), (7,), (2, 4, 3, 5)])
def test_standalone_multiply_keeps_its_shape(shape):
    inputs = (torch.randn(shape), torch.randn(shape))
    tester = (
        EspTester(Mul(), inputs)
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .check_node_count(
            {_PERMUTE: 0, _LAYOUT_PERMUTE: 0, _MUL: 1, _QUANTIZE: 3, _DEQUANTIZE: 3}
        )
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )
    graph = tester.get_artifact().exported_program().graph
    multiply = next(node for node in graph.nodes if node.target == _MUL)
    assert tuple(multiply.meta["val"].shape) == shape


@pytest.mark.parametrize("multiply", [False, True])
def test_shared_layout_permutations(multiply):
    class LayoutCopies(torch.nn.Module):
        def forward(self, x):
            x = torch.ops.channels_last.permute_copy.default(x, [0, 3, 1, 2])
            if multiply:
                x = x * x
            return torch.permute_copy(x, (0, 2, 3, 1))

    inputs = (torch.randn(2, 3, 5, 4),)
    (
        EspTester(LayoutCopies(), inputs)
        .quantize()
        .export()
        .to_edge()
        .check_node_count({_PERMUTE: 1, _LAYOUT_PERMUTE: 1})
        .run_passes()
        .check_node_count(
            {
                _PERMUTE: 0,
                _LAYOUT_PERMUTE: 0,
                _MUL: int(multiply),
                _QUANTIZE: 2 if multiply else 0,
                _DEQUANTIZE: 2 if multiply else 0,
            }
        )
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )


def test_channels_last_memory_format_stays_unquantized():
    inputs = tuple(
        torch.randn(2, 4, 3, 5).to(memory_format=torch.channels_last) for _ in range(2)
    )
    tester = (
        EspTester(Mul(), inputs)
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .check_node_count({_PERMUTE: 0, _MUL: 1, _QUANTIZE: 0, _DEQUANTIZE: 0})
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )
    program = tester.get_artifact().exported_program()
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    assert all(node.meta["val"].dim_order() == (0, 2, 3, 1) for node in placeholders)
    output = program.module()(*inputs)
    assert output.is_contiguous(memory_format=torch.channels_last)


def test_noncontiguous_inputs_keep_layout_copies():
    inputs = tuple(
        torch.randn(2, 3, 5, 4).to(memory_format=torch.channels_last) for _ in range(2)
    )
    tester = (
        EspTester(PermutedMul(), inputs)
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .check_node_count(
            {_PERMUTE: 3, _CLONE: 3, _MUL: 1, _QUANTIZE: 3, _DEQUANTIZE: 3}
        )
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )
    output = tester.get_artifact().exported_program().module()(*inputs)
    assert output.shape == inputs[0].shape
    assert output.is_contiguous()
