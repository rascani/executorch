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

typedef struct {
  /* MV2 first-stage chain: stem 3x3 stride-2 pad-1 Cin=3 -> B0 dwconv 3x3
   * stride-1 pad-1 -> B0 project 1x1.  Streams stem output rows into a
   * 3-row rolling buffer the dwconv consumes immediately, then a 1-row
   * dwconv-output scratch into the project. */
  uint32_t in_h, in_w, in_c;        /* in_c == 3 */
  uint32_t stem_out_h, stem_out_w;  /* = in_{h,w}/2 */
  uint32_t stem_out_c;
  uint32_t out_h, out_w;            /* project output spatial */
  uint32_t project_out_c;
  uint32_t stem_kernel_h, stem_kernel_w;  /* always 3 */
  uint32_t stem_stride_h, stem_stride_w;  /* always 2 */
  uint32_t stem_pad_h, stem_pad_w;        /* always 1 */
  uint32_t dw_kernel_h, dw_kernel_w;      /* always 3 */
  uint32_t dw_stride_h, dw_stride_w;
  uint32_t dw_pad_h, dw_pad_w;
  /* Stem 3x3 with optional packed-32 weight layout. */
  const int8_t*  stem_weight;
  const int8_t*  stem_weight_packed_32;
  const int32_t* stem_bias;
  const int32_t* stem_requant_mults;
  const int8_t*  stem_requant_shifts;
  int32_t stem_input_offset;        /* original, NOT folded — used for pad fill */
  int32_t stem_output_offset;
  int8_t  stem_activation_min;
  int8_t  stem_activation_max;
  /* Dwconv 3x3 */
  const int8_t*  dw_weight;
  const int32_t* dw_bias;
  const int32_t* dw_bias_with_offset_full;
  const int32_t* dw_requant_mults;
  const int8_t*  dw_requant_shifts;
  int32_t dw_input_offset;
  int32_t dw_output_offset;
  int8_t  dw_activation_min;
  int8_t  dw_activation_max;
  /* Project 1x1 */
  const int8_t*  project_weight;
  const int32_t* project_bias;
  const int32_t* project_requant_mults;
  const int8_t*  project_requant_shifts;
  int32_t project_input_offset;
  int32_t project_output_offset;
  int8_t  project_activation_min;
  int8_t  project_activation_max;
} FusedStemDwconv2dConv2dParams;

typedef struct {
  /* MV2 B0-style block (expand_ratio=1): 3x3 dwconv -> 1x1 project, no
   * preceding expand conv.  Single rolling-row scratch for the dwconv
   * output that the project conv consumes immediately. */
  uint32_t in_h, in_w, in_c;
  uint32_t out_h, out_w;            /* project output spatial (= dwconv output spatial) */
  uint32_t project_out_c;
  uint32_t kernel_h, kernel_w;      /* dwconv kernel (always 3) */
  uint32_t stride_h, stride_w;
  uint32_t pad_h, pad_w;            /* dwconv padding (always 1) */
  /* Dwconv 3x3 */
  const int8_t*  dw_weight;
  const int32_t* dw_bias;
  const int32_t* dw_bias_with_offset_full;
  const int32_t* dw_requant_mults;
  const int8_t*  dw_requant_shifts;
  int32_t dw_input_offset;
  int32_t dw_output_offset;
  int8_t  dw_activation_min;
  int8_t  dw_activation_max;
  /* Project 1x1 */
  const int8_t*  project_weight;
  const int32_t* project_bias;
  const int32_t* project_requant_mults;
  const int8_t*  project_requant_shifts;
  int32_t project_input_offset;
  int32_t project_output_offset;
  int8_t  project_activation_min;
  int8_t  project_activation_max;
} FusedDwconv2dConv2dParams;

typedef struct {
  /* Full MV2 inverted-residual block: 1x1 expand -> 3x3 dwconv -> 1x1
   * project [-> int8 residual add].  Two intermediates (expand output
   * and dwconv output) are streamed through small rolling buffers in
   * mv2_fused_scratch; the project output and optional add go directly
   * to the block output slot. */
  uint32_t in_h, in_w, in_c;
  uint32_t expand_out_c;            /* width of expand + dwconv */
  uint32_t dw_out_h, dw_out_w;      /* dwconv output spatial (= project input spatial) */
  uint32_t out_h, out_w;            /* project (and final) output spatial */
  uint32_t project_out_c;           /* width of project + final output */
  uint32_t kernel_h, kernel_w;      /* dwconv kernel (always 3) */
  uint32_t stride_h, stride_w;      /* dwconv stride */
  uint32_t pad_h, pad_w;            /* dwconv padding (always 1) */
  /* Expand 1x1 */
  const int8_t*  expand_weight;
  const int32_t* expand_bias;
  const int32_t* expand_requant_mults;
  const int8_t*  expand_requant_shifts;
  int32_t expand_input_offset;
  int32_t expand_output_offset;
  int8_t  expand_activation_min;
  int8_t  expand_activation_max;
  /* Dwconv 3x3 */
  const int8_t*  dw_weight;
  const int32_t* dw_bias;
  const int32_t* dw_bias_with_offset_full;
  const int32_t* dw_requant_mults;
  const int8_t*  dw_requant_shifts;
  int32_t dw_input_offset;
  int32_t dw_output_offset;
  int8_t  dw_activation_min;
  int8_t  dw_activation_max;
  /* Project 1x1 */
  const int8_t*  project_weight;
  const int32_t* project_bias;
  const int32_t* project_requant_mults;
  const int8_t*  project_requant_shifts;
  int32_t project_input_offset;
  int32_t project_output_offset;
  int8_t  project_activation_min;
  int8_t  project_activation_max;
  /* Optional residual add: when residual_present != 0, the runtime adds
   * the residual input (passed as a separate arg to the kernel) to the
   * project output via the standard quantized_add math. */
  uint8_t residual_present;
  int32_t residual_self_zero_point;
  int32_t residual_self_multiplier;
  int32_t residual_self_shift;
  int32_t residual_other_zero_point;
  int32_t residual_other_multiplier;
  int32_t residual_other_shift;
  int32_t residual_output_zero_point;
  int32_t residual_output_multiplier;
  int32_t residual_output_shift;
  int8_t  residual_activation_min;
  int8_t  residual_activation_max;
} FusedInvertedResidualParams;

#endif
