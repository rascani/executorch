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

Tensor& quantized_avg_pool2d_out_impl(
    KernelRuntimeContext& context,
    const Tensor& input,
    Int64ArrayRef kernel_size,
    Int64ArrayRef stride,
    Int64ArrayRef padding,
    bool ceil_mode,
    int64_t zero_point,
    int64_t multiplier,
    int64_t shift,
    const Tensor& scratch,
    ActivationLayout layout,
    Tensor& out);

} // namespace native
} // namespace cortex_m
