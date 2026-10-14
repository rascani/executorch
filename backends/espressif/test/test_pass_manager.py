# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.kernels.quantized  # noqa: F401
import pytest
import torch
from executorch.backends.espressif.test.tester import EspRunPasses, EspTester
from executorch.exir import to_edge
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops


class MultiplyAndRelu(torch.nn.Module):
    def forward(self, x, y):
        return x * y, torch.relu(x)


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
@pytest.mark.parametrize("broadcast", [False, True])
def test_quantized_and_float_operations_survive_export(entry_point, broadcast):
    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 3 if broadcast else 6).reshape(
            (3,) if broadcast else (2, 3)
        ),
    )
    tester = EspTester(MultiplyAndRelu(), inputs).quantize().export()
    if entry_point == "combined":
        tester.to_edge_transform_and_lower()
    else:
        tester.to_edge().run_passes()
    quantize_count = 0 if broadcast else 3
    tester.check_node_count(
        {
            exir_ops.edge.aten.mul.Tensor: 1,
            exir_ops.edge.aten.relu.default: 1,
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: quantize_count,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: quantize_count,
        }
    ).run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert program.execution_plan[0].name == "forward"
    assert not program.execution_plan[0].delegates
    assert {op.name for op in program.execution_plan[0].operators} >= {
        "aten::mul",
        "aten::relu",
    }


def test_passes_preserve_method_constants():
    class Scale(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.register_buffer("scale", scale)

        def forward(self, x):
            return x * self.scale

    inputs = (torch.tensor([-1.0, 2.0, 3.0]),)
    models = {
        "forward": Scale(torch.tensor([2.0, 3.0, 4.0])),
        "other": Scale(torch.tensor([5.0, 6.0, 7.0])),
    }
    edge = to_edge(
        {name: torch.export.export(model, inputs) for name, model in models.items()},
        constant_methods={"version": 1},
    )
    stage = EspRunPasses()
    stage.run(edge)
    transformed = stage.artifact
    assert transformed.methods == {"forward", "other"}
    assert transformed.config_methods == {"version"}
    for name, model in models.items():
        program = transformed.exported_program(name)
        program.validate()
        torch.testing.assert_close(
            program.module()(*inputs), model(*inputs), atol=0, rtol=0
        )
    serialized = deserialize_pte_binary(transformed.to_executorch().buffer).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
