# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.cortex_m.passes.aten_to_cortex_m_pass  # noqa: F401
import torch

from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    cmsis_nn_conv_buffer_size,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def _run_conv2d(op, x, weight, bias):
    out_channels = weight.shape[0]
    return op(
        x,
        weight,
        bias,
        [1, 1],
        [1, 1],
        [1, 1],
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_depthwise_conv2d(op, x, weight, bias):
    out_channels = weight.shape[3]
    return op(
        x,
        weight,
        bias,
        [1, 1],
        [1, 1],
        [1, 1],
        1,
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_transpose_conv2d(op, x, weight, bias):
    out_channels = weight.shape[0]
    return op(
        x,
        weight,
        bias,
        [2, 2],
        [1, 1],
        [0, 0],
        [1, 1],
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_avg_pool2d(op, x):
    return op(
        x,
        [2, 2],
        [2, 2],
        [0, 0],
        False,
        0,
        1 << 30,
        1,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_max_pool2d(op, x):
    return op(
        x,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        False,
        0,
        0,
        -128,
        127,
    )


def test_nhwc_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 3, 8, 8), dtype=torch.int8)
    weight = torch.randint(-4, 4, (4, 3, 3, 3), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_conv2d(
        torch.ops.cortex_m.quantized_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_conv2d(
        torch.ops.cortex_m.quantized_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_depthwise_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)
    weight = torch.randint(-4, 4, (1, 3, 3, 4), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_depthwise_conv2d(
        torch.ops.cortex_m.quantized_depthwise_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_depthwise_conv2d(
        torch.ops.cortex_m.quantized_depthwise_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_transpose_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 3, 6, 6), dtype=torch.int8)
    weight = torch.randint(-4, 4, (4, 3, 3, 3), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_transpose_conv2d(
        torch.ops.cortex_m.quantized_transpose_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_transpose_conv2d(
        torch.ops.cortex_m.quantized_transpose_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_avg_pool2d_matches_legacy_layout():
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)

    legacy = _run_avg_pool2d(
        torch.ops.cortex_m.quantized_avg_pool2d,
        x.to(memory_format=torch.channels_last),
    )
    explicit = _run_avg_pool2d(
        torch.ops.cortex_m.quantized_avg_pool2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_max_pool2d_matches_legacy_layout():
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)

    legacy = _run_max_pool2d(
        torch.ops.cortex_m.quantized_max_pool2d,
        x.to(memory_format=torch.channels_last),
    )
    explicit = _run_max_pool2d(
        torch.ops.cortex_m.quantized_max_pool2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_conv2d_fake_shape_is_logical_nhwc():
    with FakeTensorMode():
        output = _run_conv2d(
            torch.ops.cortex_m.quantized_conv2d_nhwc,
            torch.empty(2, 10, 6, 3, dtype=torch.int8),
            torch.empty(5, 3, 3, 3, dtype=torch.int8),
            torch.empty(5, dtype=torch.int32),
        )

    assert output.shape == torch.Size([2, 10, 6, 5])
    assert output.dim_order() == (0, 1, 2, 3)


def test_nhwc_and_legacy_scratch_sizes_match():
    backend = CortexMTargetConfig(cpu=CortexM.M55).backend

    def make_node(target, input_shape, output_shape):
        graph = torch.fx.Graph()
        with FakeTensorMode() as mode:
            input_node = graph.placeholder("input")
            input_node.meta["val"] = mode.from_tensor(
                torch.empty(input_shape, dtype=torch.int8)
            )
            weight_node = graph.placeholder("weight")
            weight_node.meta["val"] = mode.from_tensor(
                torch.empty(4, 3, 3, 3, dtype=torch.int8)
            )
            conv = graph.call_function(
                target,
                args=(
                    input_node,
                    weight_node,
                    None,
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    5,
                    -3,
                    None,
                    None,
                    -128,
                    127,
                    None,
                ),
            )
            conv.meta["val"] = mode.from_tensor(
                torch.empty(output_shape, dtype=torch.int8)
            )
        return conv

    legacy = make_node(
        exir_ops.edge.cortex_m.quantized_conv2d.default,
        (1, 3, 10, 6),
        (1, 4, 10, 6),
    )
    explicit = make_node(
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        (1, 10, 6, 3),
        (1, 10, 6, 4),
    )

    assert cmsis_nn_conv_buffer_size(backend, legacy) == cmsis_nn_conv_buffer_size(
        backend, explicit, nhwc_logical=True
    )
