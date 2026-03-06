# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.cortex_m.test.tester import CortexMQuantize, CortexMTester
from executorch.backends.test.suite.flow import TestFlow

CORTEX_M_TEST_FLOW = TestFlow(
    name="cortex_m",
    backend="cortex_m",
    tester_factory=CortexMTester,
    quantize=True,
    quantize_stage_factory=CortexMQuantize,
    is_delegated=False,
    supports_serialize=False,
    skip_patterns=[
        "test_clamp",
        "test_embedding",
        "test_floor_divide",
        "test_hardswish",
        "test_hardtanh",
        "test_index_put",
        "test_index_select",
        "test_masked_fill",
        "test_maxpool1d",
        "test_add_f32_alpha",
        "test_subtract_f32_alpha",
        "test_conv1d_padding_modes",
        "test_conv2d_padding_modes",
        "test_conv3d_padding_modes",
        "test_divide_f32_floor",
        "test_divide_f32_trunc",
        "test_relu_dtype",
        "test_relu_f32_single_dim",
        "test_round_decimals",
        "test_mean_output_dtype",
        "test_conformer",
        "test_alexnet",
        "test_densenet161",
        "test_inception_v3",
        "test_mobilenet_v3_small",
        "test_regnet_y_1_6gf",
        "test_resnet50",
        "test_resnext50_32x4d",
        "test_squeezenet1_1",
        "test_swin_v2_t",
        "test_vgg11",
        "test_wide_resnet50_2",
        "test_convnext_small",
    ],
)
