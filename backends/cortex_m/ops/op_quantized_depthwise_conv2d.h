/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

Tensor& quantized_depthwise_conv2d_out_impl(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    Int64ArrayRef stride,
    Int64ArrayRef padding,
    Int64ArrayRef dilation,
    int64_t depth_multiplier,
    int64_t input_offset,
    int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    int64_t activation_min,
    int64_t activation_max,
    const Tensor& scratch,
    ActivationLayout layout,
    Tensor& out);

} // namespace native
} // namespace cortex_m
