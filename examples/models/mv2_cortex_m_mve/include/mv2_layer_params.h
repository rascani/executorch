/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * LayerParams structs shared between mv2_inference.c and the generated
 * mv2_params.h.  Each struct holds the parameters needed by one kernel
 * family; per-layer instances are emitted as static const by the dumper.
 *
 * Per-channel arrays (multipliers, shifts) are stored as separate const arrays
 * referenced via pointer; this keeps the struct small and lets the linker
 * place them in flash.
 */

#ifndef MV2_LAYER_PARAMS_H_
#define MV2_LAYER_PARAMS_H_

#include <stdint.h>

typedef struct {
  uint32_t num_elements;
  float scale;
  int8_t zero_point;
  int8_t qmin;
  int8_t qmax;
} QuantInputParams;

typedef struct {
  uint32_t in_features;
  uint32_t out_features;
  const int8_t* weight;       // [out_features, in_features] row-major
  const int32_t* bias;        // [out_features]; nullable
  const int32_t* kernel_sum;  // [out_features] = sum_in(weight) * input_offset + bias; nullable
  int32_t input_offset;
  int32_t filter_offset;
  int32_t output_offset;
  int32_t multiplier;
  int32_t shift;
  int8_t activation_min;
  int8_t activation_max;
} LinearParams;

typedef struct {
  // NHWC int8 input, NHWC int8 output, OHWI int8 weights (groups=1).
  uint32_t in_h;
  uint32_t in_w;
  uint32_t in_c;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t out_c;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t pad_h;
  uint32_t pad_w;
  const int8_t* weight;            // [out_c, kernel_h, kernel_w, in_c] (OHWI)
  const int32_t* bias;             // [out_c]; nullable
  const int32_t* requant_mults;    // [out_c]
  const int32_t* requant_shifts;   // [out_c]
  int32_t input_offset;
  int32_t output_offset;
  int8_t activation_min;
  int8_t activation_max;
} Conv2dParams;

typedef struct {
  // NHWC int8 input, IHWO int8 weights (1, kH, kW, C), depth_multiplier always 1.
  uint32_t in_h;
  uint32_t in_w;
  uint32_t in_c;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t out_c;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t pad_h;
  uint32_t pad_w;
  const int8_t* weight;            // [1, kernel_h, kernel_w, out_c] (IHWO)
  const int32_t* bias;             // [out_c]; nullable
  const int32_t* requant_mults;    // [out_c]
  const int32_t* requant_shifts;   // [out_c]
  int32_t input_offset;
  int32_t output_offset;
  int8_t activation_min;
  int8_t activation_max;
} DepthwiseConv2dParams;

typedef struct {
  // Elementwise add of two NHWC int8 tensors of identical shape.
  uint32_t num_elements;
  int32_t self_zero_point;
  int32_t self_multiplier;
  int32_t self_shift;
  int32_t other_zero_point;
  int32_t other_multiplier;
  int32_t other_shift;
  int32_t output_zero_point;
  int32_t output_multiplier;
  int32_t output_shift;
  int8_t activation_min;
  int8_t activation_max;
} QuantizedAddParams;

typedef struct {
  // NHWC int8 input/output.
  uint32_t in_h;
  uint32_t in_w;
  uint32_t channels;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t pad_h;
  uint32_t pad_w;
  int32_t zero_point;
  int32_t multiplier;
  int32_t shift;
} AvgPool2dParams;

#endif
