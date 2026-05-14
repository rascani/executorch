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
  const int8_t*  requant_shifts;   // [out_c] — fits in int8 (frexp exponent, ~[-30, +5])
  int32_t input_offset;
  int32_t output_offset;
  int8_t activation_min;
  int8_t activation_max;
  /* Optional packed weight blob for the first 3x3 stride-2 in_c=3 conv.
   * Layout: [out_c, 32] int8.  Each row is the 27 OHWI weights for one OC
   * (in kh, kw, ic order) followed by 5 zero padding bytes.  When non-null,
   * `bias` is required to hold bias_with_offset (bias + input_offset * sum_w),
   * `input_offset` is 0, and the runtime uses a vmladavaq_s8 im2col path. */
  const int8_t* weight_packed_32;
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
  const int8_t*  requant_shifts;   // [out_c] — int8 packed
  int32_t input_offset;
  int32_t output_offset;
  int8_t activation_min;
  int8_t activation_max;
  /* Optional bias + input_offset * sum(weight[c]) over ALL kernel positions.
   * Used by fast-path tiles when every (kh, kw) is in-bounds (i.e., away
   * from the pad ring) so the runtime can skip the per-tap input_offset
   * add.  NULL if not pre-computed for this layer. */
  const int32_t* bias_with_offset_full;
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

typedef struct {
  /* Fused MV2 expand+dwconv: 1x1 conv -> 3x3 depthwise conv, no
   * materialized expand intermediate.  Spatial dims describe the input
   * to the 1x1 expand and the output of the 3x3 dwconv; the runtime
   * computes expand rows on-demand into a 3-row rolling buffer that the
   * dwconv consumes immediately. */
  uint32_t in_h;
  uint32_t in_w;
  uint32_t in_c;
  uint32_t expand_out_c;           /* shared channel count for expand output and dwconv */
  uint32_t out_h;
  uint32_t out_w;
  uint32_t kernel_h;               /* dwconv kernel height (always 3) */
  uint32_t kernel_w;               /* dwconv kernel width (always 3) */
  uint32_t stride_h;               /* dwconv stride */
  uint32_t stride_w;
  uint32_t pad_h;                  /* dwconv padding (always 1) */
  uint32_t pad_w;
  /* Expand 1x1 weight + bias + requantize (offset folded into bias). */
  const int8_t*  expand_weight;            /* [expand_out_c, 1, 1, in_c] OHWI */
  const int32_t* expand_bias;              /* [expand_out_c]; nullable */
  const int32_t* expand_requant_mults;
  const int8_t*  expand_requant_shifts;
  int32_t expand_input_offset;             /* typically 0 (folded into bias) */
  int32_t expand_output_offset;
  int8_t  expand_activation_min;
  int8_t  expand_activation_max;
  /* Dwconv 3x3 weight + bias + requantize. */
  const int8_t*  dw_weight;                /* [1, kH, kW, expand_out_c] IHWO */
  const int32_t* dw_bias;                  /* [expand_out_c]; nullable */
  const int32_t* dw_bias_with_offset_full; /* optional, like DepthwiseConv2dParams */
  const int32_t* dw_requant_mults;
  const int8_t*  dw_requant_shifts;
  int32_t dw_input_offset;
  int32_t dw_output_offset;
  int8_t  dw_activation_min;
  int8_t  dw_activation_max;
} FusedConv2dDwconv2dParams;

#endif
