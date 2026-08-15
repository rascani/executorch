/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "op_quantized_avg_pool2d.h"

namespace cortex_m {
namespace native {

// cppcheck-suppress unusedFunction
Tensor& quantized_avg_pool2d_nhwc_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const bool ceil_mode,
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    const Tensor& scratch,
    Tensor& out) {
  return quantized_avg_pool2d_out_impl(
      context,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      zero_point,
      multiplier,
      shift,
      scratch,
      ActivationLayout::NHWCLogical,
      out);
}

} // namespace native
} // namespace cortex_m
