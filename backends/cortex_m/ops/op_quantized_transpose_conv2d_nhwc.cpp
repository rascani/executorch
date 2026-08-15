/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "op_quantized_transpose_conv2d.h"

namespace cortex_m {
namespace native {

// cppcheck-suppress unusedFunction
Tensor& quantized_transpose_conv2d_nhwc_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef output_padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& scratch,
    const Tensor& output_scratch,
    Tensor& out) {
  return quantized_transpose_conv2d_out_impl(
      context,
      input,
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      input_offset,
      output_offset,
      requantize_multipliers,
      requantize_shifts,
      activation_min,
      activation_max,
      scratch,
      output_scratch,
      ActivationLayout::NHWCLogical,
      out);
}

} // namespace native
} // namespace cortex_m
