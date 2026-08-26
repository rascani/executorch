# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.library import cmsis_nn  # noqa: F401
from executorch.backends.cortex_m.op_backend import CortexMOpBackend
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir import to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

_LOGGER = "executorch.backends.cortex_m.op_backend"


class ConvRelu(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _quantized_edge_program(model: torch.nn.Module, inputs: tuple):
    exported = torch.export.export(model.eval(), inputs)
    prepared = prepare_pt2e(exported.module(), CortexMQuantizer())
    prepared(*inputs)
    converted = convert_pt2e(prepared)
    return to_edge(
        torch.export.export(converted, inputs),
        compile_config=cortex_m_edge_compile_config(),
    )


def _backend(target: str = "cortex-m55") -> CortexMOpBackend:
    return CortexMOpBackend(
        target_config=CortexMTargetConfig.from_target_string(target)
    )


def _op_names(program) -> set:
    return {
        getattr(node.target, "_name", str(node.target))
        for node in program.graph.nodes
        if node.op == "call_function"
    }


class TestCortexMOpBackend(unittest.TestCase):
    def test_channels_last_input_lowers_to_cmsis_kernels(self) -> None:
        manager = _quantized_edge_program(
            ConvRelu(), (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),)
        )
        with self.assertNoLogs(_LOGGER, level="INFO"):
            lowered = manager.to_op_backend(_backend())
        names = _op_names(lowered.exported_program())
        self.assertIn("cortex_m::quantized_conv2d", names)

    def test_contiguous_convolution_warns_about_layout(self) -> None:
        # The backend cannot convert the layout, so it has to say so.
        manager = _quantized_edge_program(ConvRelu(), (torch.randn(1, 3, 8, 8),))
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            manager.to_op_backend(_backend())
        self.assertIn("channels_last", "".join(logs.output))

    def test_the_portable_kernels_it_leaves_behind_are_listed(self) -> None:
        # The list feeds the runner's operator list, so it has to name
        # everything the layout failure drags down, not just the conv.
        manager = _quantized_edge_program(ConvRelu(), (torch.randn(1, 3, 8, 8),))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            manager.to_op_backend(_backend())
        listed = "".join(line for line in logs.output if "portable kernels" in line)
        for name in ("aten::convolution", "aten::clamp"):
            self.assertIn(name, listed)

    def test_an_op_with_no_cmsis_equivalent_is_not_blamed_on_layout(self) -> None:
        # int64 has no CMSIS-NN kernel at any layout, so layout advice is a lie.
        class IntAdd(torch.nn.Module):
            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                return idx + idx

        manager = _quantized_edge_program(IntAdd(), (torch.arange(4),))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            manager.to_op_backend(_backend())
        joined = "".join(logs.output)
        self.assertIn("aten::add", joined)
        self.assertNotIn("channels_last", joined)

    def test_a_single_channel_convolution_is_not_missed(self) -> None:
        # is_channels_last is True for any C==1 tensor, so checking the conv's
        # INPUT stays silent here while the quantizer, which checks the conv's
        # output, rejects the pattern. Mono/grayscale first layers hit this.
        class Mono(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.conv(x))

        manager = _quantized_edge_program(Mono(), (torch.randn(1, 1, 8, 8),))
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            manager.to_op_backend(_backend())
        self.assertIn("channels_last", "".join(logs.output))

    def test_the_advice_the_warning_gives_actually_works(self) -> None:
        # A warning that prescribes a fix which does not lower the operator is
        # worse than no warning.
        inputs = (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),)
        manager = _quantized_edge_program(ConvRelu(), inputs)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            lowered = manager.to_op_backend(_backend())
        self.assertIn(
            "cortex_m::quantized_conv2d", _op_names(lowered.exported_program())
        )

    def test_a_delegate_is_not_reported_as_a_portable_kernel(self) -> None:
        # An operator backend runs after partitioning, so a call_delegate node
        # is expected; it names no kernel the runner registers.
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower

        class Add(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        manager = to_edge_transform_and_lower(
            torch.export.export(Add(), (torch.randn(2, 2),)),
            partitioner=[XnnpackPartitioner()],
        )
        with self.assertNoLogs(_LOGGER, level="INFO"):
            manager.to_op_backend(_backend())

    def test_kernels_inside_control_flow_are_reported(self) -> None:
        # A branch's operators need registering too, and they live in a
        # submodule rather than the top-level graph.
        class Branch(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.cond(
                    x.sum() > 0, lambda y: y.sin(), lambda y: y.cos(), (x,)
                )

        manager = to_edge(torch.export.export(Branch(), (torch.randn(2, 2),)))
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            manager.to_op_backend(_backend())
        listed = "".join(logs.output)
        self.assertIn("aten::sin", listed)
        self.assertIn("aten::cos", listed)

    def test_it_does_not_mutate_the_program_it_was_given(self) -> None:
        inputs = (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),)
        program = _quantized_edge_program(ConvRelu(), inputs).exported_program()
        before = sorted(program.state_dict)
        _backend().lower(program, "forward")
        self.assertEqual(sorted(program.state_dict), before)

    def test_the_target_config_reaches_the_pass_manager(self) -> None:
        # quantized_linear encodes its weights differently for MVE, so a
        # target that is recorded and then dropped ships an artifact whose
        # kernel reads an argument that is not there.
        class Linear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        def weights(target: str) -> list:
            manager = _quantized_edge_program(Linear(), (torch.randn(1, 8),))
            program = manager.to_op_backend(_backend(target)).exported_program()
            self.assertIn("cortex_m::quantized_linear", _op_names(program))
            return sorted(program.state_dict)

        self.assertNotEqual(weights("cortex-m55"), weights("cortex-m33"))


if __name__ == "__main__":
    unittest.main()
