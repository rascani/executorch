/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "op_quantized_max_pool2d.h"

namespace cortex_m {
namespace native {

// cppcheck-suppress unusedFunction
Tensor& quantized_max_pool2d_nhwc_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const bool ceil_mode,
    const int64_t input_zero_point,
    const int64_t output_zero_point,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  return quantized_max_pool2d_out_impl(
      context,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      input_zero_point,
      output_zero_point,
      activation_min,
      activation_max,
      ActivationLayout::NHWCLogical,
      out);
}

} // namespace native
} // namespace cortex_m
