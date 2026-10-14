# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from executorch.backends.espressif.quantizer import EspQuantizer
from executorch.backends.espressif.test.tester import EspTester
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

_QUANTIZE = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = torch.ops.quantized_decomposed.dequantize_per_tensor.default


class Mul(torch.nn.Module):
    def forward(self, x, y):
        return x * y


@pytest.mark.parametrize("shape", [(), (17,), (2, 3), (1, 2, 3, 4)])
def test_multiply_quantization(shape):
    inputs = (
        torch.linspace(-1.7, 2.3, math.prod(shape)).reshape(shape),
        torch.linspace(-3.0, 1.0, math.prod(shape)).reshape(shape),
    )
    tester = EspTester(Mul(), inputs).quantize()
    quantized = tester.get_artifact()
    quantize_nodes = [
        node for node in quantized.graph.nodes if node.target == _QUANTIZE
    ]
    assert len(quantize_nodes) == 3
    for node in quantize_nodes:
        assert node.args[1] > 0
        assert -128 <= node.args[2] <= 127
        assert node.args[3:6] == (-128, 127, torch.int8)
    mul = next(
        node
        for node in quantized.graph.nodes
        if node.target == torch.ops.aten.mul.Tensor
    )
    assert all(arg.target == _DEQUANTIZE for arg in mul.args)
    x_scale, y_scale = (arg.args[1] for arg in mul.args)
    output_scale = next(node.args[1] for node in quantize_nodes if node.args[0] is mul)
    error_bound = (
        inputs[1].abs().max().item() * x_scale / 2
        + inputs[0].abs().max().item() * y_scale / 2
        + x_scale * y_scale / 4
        + output_scale / 2
    )
    torch.testing.assert_close(
        quantized(*inputs), Mul()(*inputs), atol=error_bound + 1e-6, rtol=0
    )
    tester.export()


def test_calibration_tracks_inputs_and_output_independently():
    inputs = (torch.tensor([1.0, 2.0, 4.0]), torch.tensor([-1.0, -0.5, -0.25]))
    samples = [inputs, tuple(value * 2 for value in inputs)]
    quantized = (
        EspTester(Mul(), inputs, calibration_samples=samples).quantize().get_artifact()
    )
    quantize_nodes = [
        node for node in quantized.graph.nodes if node.target == _QUANTIZE
    ]
    assert [node.args[1] for node in quantize_nodes] == pytest.approx(
        [8 / 255, 2 / 255, 4 / 255]
    )
    assert [node.args[2] for node in quantize_nodes] == [-128, 127, 127]


def test_repeated_input_preserves_float_consumers():
    class SquareAndAdd(torch.nn.Module):
        def forward(self, x):
            return x * x, x + 0.25

    inputs = (torch.linspace(-1, 1, 17),)
    model = SquareAndAdd()
    quantized = EspTester(model, inputs).quantize().get_artifact()
    quantize_nodes = [
        node for node in quantized.graph.nodes if node.target == _QUANTIZE
    ]
    assert len(quantize_nodes) == 2
    mul = next(
        node
        for node in quantized.graph.nodes
        if node.target == torch.ops.aten.mul.Tensor
    )
    assert mul.args[0].args[0] is mul.args[1].args[0]
    torch.testing.assert_close(quantized(*inputs)[1], model(*inputs)[1], atol=0, rtol=0)


def test_chained_multiplications():
    class MulChain(torch.nn.Module):
        def forward(self, x, y):
            return (x * y) * y

    inputs = (torch.linspace(-1, 1, 17), torch.linspace(0, 0.5, 17))
    model = MulChain()
    quantized = EspTester(model, inputs).quantize().get_artifact()
    muls = [
        node
        for node in quantized.graph.nodes
        if node.target == torch.ops.aten.mul.Tensor
    ]
    assert len(muls) == 2
    assert all(arg.target == _DEQUANTIZE for mul in muls for arg in mul.args)
    torch.testing.assert_close(quantized(*inputs), model(*inputs), atol=0.01, rtol=0)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param((torch.ones(2, 3), torch.ones(3)), id="broadcast"),
        pytest.param((torch.ones(2, 3), 2.0), id="scalar-argument"),
        pytest.param(
            (
                torch.ones(2, 3, dtype=torch.float16),
                torch.ones(2, 3, dtype=torch.float16),
            ),
            id="float16",
        ),
        pytest.param(
            (torch.ones(2, 3, dtype=torch.int32), torch.ones(2, 3, dtype=torch.int32)),
            id="integer",
        ),
        pytest.param(
            (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float16)), id="mixed-dtypes"
        ),
        pytest.param((torch.ones(2, 3).t(), torch.ones(2, 3).t()), id="noncontiguous"),
        pytest.param((torch.empty(0, 3), torch.empty(0, 3)), id="empty"),
    ],
)
def test_unsupported_inputs_stay_unquantized(inputs):
    quantized = EspTester(Mul(), inputs).quantize().get_artifact()
    assert not any(node.target == _QUANTIZE for node in quantized.graph.nodes)
    torch.testing.assert_close(quantized(*inputs), Mul()(*inputs), atol=0, rtol=0)


def test_unsupported_operator_stays_unquantized():
    inputs = (torch.linspace(-1, 1, 17),)
    model = torch.nn.ReLU()
    quantized = EspTester(model, inputs).quantize().get_artifact()
    assert not any(node.target == _QUANTIZE for node in quantized.graph.nodes)
    torch.testing.assert_close(quantized(*inputs), model(*inputs), atol=0, rtol=0)


def test_validate_rejects_unsupported_annotated_multiply():
    model = torch.export.export(Mul(), (torch.ones(2, 3), torch.ones(3))).module()
    quantizer = EspQuantizer()
    quantizer.validate(model)
    mul = next(
        node for node in model.graph.nodes if node.target == torch.ops.aten.mul.Tensor
    )
    mul.meta["quantization_annotation"] = QuantizationAnnotation(_annotated=True)
    with pytest.raises(ValueError, match="EspQuantizer requires.*equal-shaped.*mul"):
        quantizer.validate(model)


def test_dynamic_shapes_stay_unquantized():
    inputs = (torch.ones(2, 3), torch.ones(2, 3))
    batch = torch.export.Dim("batch", min=1, max=8)
    graph_module = torch.export.export(
        Mul(),
        inputs,
        dynamic_shapes=({0: batch}, {0: batch}),
    ).module()
    prepared = prepare_pt2e(graph_module, EspQuantizer())
    prepared(*inputs)
    quantized = convert_pt2e(prepared)
    assert not any(node.target == _QUANTIZE for node in quantized.graph.nodes)
    new_inputs = (torch.ones(5, 3), torch.full((5, 3), 2.0))
    torch.testing.assert_close(
        quantized(*new_inputs), Mul()(*new_inputs), atol=0, rtol=0
    )
