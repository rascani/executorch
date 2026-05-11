/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Standalone MobileNetV2 inference for Cortex-M55 + Helium (MVE).
 *
 * All kernels are static functions in this translation unit so the compiler
 * inlines kernel bodies into the entry-point call sequence and constant-folds
 * each layer's parameters (passed as a pointer to a static const LayerParams).
 *
 * Helium MVE intrinsics are used on Cortex-M55 (__ARM_FEATURE_MVE_INT defined);
 * a scalar fallback is provided so the same TU compiles on host x86 for
 * correctness testing.  The scalar path mirrors backends/cortex_m/passes/
 * passes_utils.py:103 `requantize_cmsis` bit-exactly.
 *
 * Phase A kernels: quantize_input, gemv_s8.  Phase B/C/D append more kernels
 * above the entry point.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define MV2_USE_MVE 1
#else
#define MV2_USE_MVE 0
#endif

#include "mv2_inference.h"
#include "mv2_layer_params.h"
#include "mve_helpers.h"
#include "mv2_params.h"

extern uint8_t mv2_arena[];

static void quantize_input(const float* in, int8_t* out, const QuantInputParams* p) {
  const float inv_scale = 1.0f / p->scale;
  const int32_t zp = p->zero_point;
  const int32_t qmin = p->qmin;
  const int32_t qmax = p->qmax;
  uint32_t n = p->num_elements;
  uint32_t i = 0;

#if MV2_USE_MVE
  while (i + 4 <= n) {
    float32x4_t f = vld1q_f32(in + i);
    float32x4_t scaled = vmulq_n_f32(f, inv_scale);
    int32x4_t q = vcvtaq_s32_f32(scaled);
    q = vaddq_n_s32(q, zp);
    q = vmaxq_s32(q, vdupq_n_s32(qmin));
    q = vminq_s32(q, vdupq_n_s32(qmax));
    out[i + 0] = (int8_t)vgetq_lane_s32(q, 0);
    out[i + 1] = (int8_t)vgetq_lane_s32(q, 1);
    out[i + 2] = (int8_t)vgetq_lane_s32(q, 2);
    out[i + 3] = (int8_t)vgetq_lane_s32(q, 3);
    i += 4;
  }
#endif

  for (; i < n; ++i) {
    float scaled = in[i] * inv_scale;
    /* round-half-away-from-zero, matching torch.round / vcvtaq_s32_f32 */
    int32_t q = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    q += zp;
    if (q < qmin) q = qmin;
    if (q > qmax) q = qmax;
    out[i] = (int8_t)q;
  }
}

static void conv2d_s8(const int8_t* input, int8_t* output, const Conv2dParams* p) {
  /* NHWC int8 input, NHWC int8 output, OHWI int8 weights, per-channel requant.
   * Matches the math in backends/cortex_m/ops/operators.py:751
   * quantized_conv2d_impl bit-exactly when using scalar_requantize. */
  const uint32_t in_h = p->in_h, in_w = p->in_w, in_c = p->in_c;
  const uint32_t out_h = p->out_h, out_w = p->out_w, out_c = p->out_c;
  const uint32_t k_h = p->kernel_h, k_w = p->kernel_w;
  const uint32_t stride_h = p->stride_h, stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h, pad_w = (int32_t)p->pad_w;
  const int32_t input_offset = p->input_offset;
  const int32_t output_offset = p->output_offset;
  const int32_t act_min = p->activation_min;
  const int32_t act_max = p->activation_max;
  const int8_t* weight = p->weight;
  const int32_t* bias = p->bias;
  const int32_t* mults = p->requant_mults;
  const int32_t* shifts = p->requant_shifts;

  const size_t in_row_stride = (size_t)in_w * in_c;
  const size_t out_row_stride = (size_t)out_w * out_c;
  const size_t w_oc_stride = (size_t)k_h * k_w * in_c;

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      for (uint32_t oc = 0; oc < out_c; ++oc) {
        int32_t acc = (bias != (const int32_t*)0) ? bias[oc] : 0;
        const int8_t* w_oc = weight + (size_t)oc * w_oc_stride;
        for (uint32_t kh = 0; kh < k_h; ++kh) {
          const int32_t ih = (int32_t)(oh * stride_h) - pad_h + (int32_t)kh;
          if (ih < 0 || (uint32_t)ih >= in_h) continue;
          for (uint32_t kw = 0; kw < k_w; ++kw) {
            const int32_t iw = (int32_t)(ow * stride_w) - pad_w + (int32_t)kw;
            if (iw < 0 || (uint32_t)iw >= in_w) continue;
            const int8_t* x = input + (size_t)ih * in_row_stride
                              + (size_t)iw * in_c;
            const int8_t* w = w_oc + ((size_t)kh * k_w + kw) * in_c;
            uint32_t ic = 0;
#if MV2_USE_MVE
            while (ic + 16 <= in_c) {
              int8x16_t v_x = vld1q_s8(x + ic);
              int8x16_t v_w = vld1q_s8(w + ic);
              acc = vmladavaq_s8(acc, v_x, v_w);
              /* input_offset contribution from the 16 lanes */
              acc += vaddvq_s8(v_w) * input_offset;
              ic += 16;
            }
#endif
            for (; ic < in_c; ++ic) {
              acc += ((int32_t)x[ic] + input_offset) * (int32_t)w[ic];
            }
          }
        }
        acc = scalar_requantize(acc, mults[oc], shifts[oc]);
        acc += output_offset;
        if (acc < act_min) acc = act_min;
        if (acc > act_max) acc = act_max;
        output[(size_t)oh * out_row_stride + (size_t)ow * out_c + oc] = (int8_t)acc;
      }
    }
  }
}

static void dwconv2d_s8(const int8_t* input, int8_t* output,
                        const DepthwiseConv2dParams* p) {
  /* NHWC int8 input, IHWO int8 weights [1, kH, kW, C], depth_multiplier=1
   * (out_c == in_c).  Per-channel requant, fused ReLU bounds.
   * Mirrors quantized_depthwise_conv2d_impl in operators.py:858 bit-exactly
   * when using scalar_requantize. */
  const uint32_t in_h = p->in_h, in_w = p->in_w, in_c = p->in_c;
  const uint32_t out_h = p->out_h, out_w = p->out_w, out_c = p->out_c;
  const uint32_t k_h = p->kernel_h, k_w = p->kernel_w;
  const uint32_t stride_h = p->stride_h, stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h, pad_w = (int32_t)p->pad_w;
  const int32_t input_offset = p->input_offset;
  const int32_t output_offset = p->output_offset;
  const int32_t act_min = p->activation_min;
  const int32_t act_max = p->activation_max;
  const int8_t* weight = p->weight;
  const int32_t* bias = p->bias;
  const int32_t* mults = p->requant_mults;
  const int32_t* shifts = p->requant_shifts;
  const size_t in_row_stride = (size_t)in_w * in_c;
  const size_t out_row_stride = (size_t)out_w * out_c;
  const size_t w_row_stride = (size_t)k_w * out_c;

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      for (uint32_t oc = 0; oc < out_c; ++oc) {
        int32_t acc = (bias != (const int32_t*)0) ? bias[oc] : 0;
        for (uint32_t kh = 0; kh < k_h; ++kh) {
          const int32_t ih = (int32_t)(oh * stride_h) - pad_h + (int32_t)kh;
          if (ih < 0 || (uint32_t)ih >= in_h) continue;
          const int8_t* w_row = weight + (size_t)kh * w_row_stride;
          const int8_t* x_row = input + (size_t)ih * in_row_stride;
          for (uint32_t kw = 0; kw < k_w; ++kw) {
            const int32_t iw = (int32_t)(ow * stride_w) - pad_w + (int32_t)kw;
            if (iw < 0 || (uint32_t)iw >= in_w) continue;
            int32_t x = (int32_t)x_row[(size_t)iw * in_c + oc] + input_offset;
            int32_t w = (int32_t)w_row[(size_t)kw * out_c + oc];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, mults[oc], shifts[oc]);
        acc += output_offset;
        if (acc < act_min) acc = act_min;
        if (acc > act_max) acc = act_max;
        output[(size_t)oh * out_row_stride + (size_t)ow * out_c + oc] = (int8_t)acc;
      }
    }
  }
}

static void add_s8(const int8_t* a, const int8_t* b, int8_t* out,
                   const QuantizedAddParams* p) {
  /* Two-stage requantize: lift each input by SHIFT_INT8=20 and requantize
   * to a shared fixed-point representation, sum, requantize to output scale.
   * Matches operators.py:168 quantized_add_impl bit-exactly. */
  const uint32_t n = p->num_elements;
  const int32_t SHIFT_INT8 = 20;
  for (uint32_t i = 0; i < n; ++i) {
    int32_t self_lifted = ((int32_t)a[i] - p->self_zero_point) << SHIFT_INT8;
    int32_t other_lifted = ((int32_t)b[i] - p->other_zero_point) << SHIFT_INT8;
    int32_t self_fp = scalar_requantize(self_lifted, p->self_multiplier, p->self_shift);
    int32_t other_fp = scalar_requantize(other_lifted, p->other_multiplier, p->other_shift);
    int32_t sum_fp = self_fp + other_fp;
    int32_t result = scalar_requantize(sum_fp, p->output_multiplier, p->output_shift);
    result += p->output_zero_point;
    if (result < p->activation_min) result = p->activation_min;
    if (result > p->activation_max) result = p->activation_max;
    out[i] = (int8_t)result;
  }
}

static void avgpool_s8(const int8_t* input, int8_t* output, const AvgPool2dParams* p) {
  /* NHWC int8 in/out, count_include_pad=False (matches operators.py:1215). */
  const uint32_t in_h = p->in_h, in_w = p->in_w, channels = p->channels;
  const uint32_t out_h = p->out_h, out_w = p->out_w;
  const uint32_t k_h = p->kernel_h, k_w = p->kernel_w;
  const uint32_t stride_h = p->stride_h, stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h, pad_w = (int32_t)p->pad_w;
  const int32_t zp = p->zero_point;
  const size_t in_row_stride = (size_t)in_w * channels;
  const size_t out_row_stride = (size_t)out_w * channels;

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int32_t h_start = (int32_t)(oh * stride_h) - pad_h;
      const int32_t w_start = (int32_t)(ow * stride_w) - pad_w;
      const int32_t h_end = h_start + (int32_t)k_h;
      const int32_t w_end = w_start + (int32_t)k_w;
      const int32_t ih_lo = h_start < 0 ? 0 : h_start;
      const int32_t iw_lo = w_start < 0 ? 0 : w_start;
      const int32_t ih_hi = h_end > (int32_t)in_h ? (int32_t)in_h : h_end;
      const int32_t iw_hi = w_end > (int32_t)in_w ? (int32_t)in_w : w_end;
      const int32_t count = (ih_hi - ih_lo) * (iw_hi - iw_lo);
      if (count <= 0) continue;

      for (uint32_t c = 0; c < channels; ++c) {
        int32_t sum = 0;
        for (int32_t ih = ih_lo; ih < ih_hi; ++ih) {
          for (int32_t iw = iw_lo; iw < iw_hi; ++iw) {
            sum += (int32_t)input[(size_t)ih * in_row_stride
                                  + (size_t)iw * channels + c];
          }
        }
        /* Mirror operators.py:1215 quantized_avg_pool2d_impl: float avg of
         * raw int8 values, then torch.round (banker's / round-to-even). */
        float avg_f = (float)sum / (float)count;
        int32_t out = (int32_t)nearbyintf(avg_f);
        if (out < -128) out = -128;
        if (out > 127) out = 127;
        output[(size_t)oh * out_row_stride + (size_t)ow * channels + c] = (int8_t)out;
      }
    }
  }
}

static void gemv_s8(const int8_t* input, int8_t* output, const LinearParams* p) {
  const uint32_t in_features = p->in_features;
  const uint32_t out_features = p->out_features;
  const int32_t filter_offset = p->filter_offset;
  const int32_t output_offset = p->output_offset;
  const int32_t multiplier = p->multiplier;
  const int32_t shift = p->shift;
  const int32_t act_min = p->activation_min;
  const int32_t act_max = p->activation_max;
  const int32_t* kernel_sum = p->kernel_sum;
  const int32_t* bias = p->bias;

  /* sum_j input[j] — shared across all output channels. */
  int32_t input_sum = 0;
  {
    uint32_t k = 0;
#if MV2_USE_MVE
    int8x16_t ones = vdupq_n_s8(1);
    while (k + 16 <= in_features) {
      int8x16_t v = vld1q_s8(input + k);
      input_sum = vmladavaq_s8(input_sum, v, ones);
      k += 16;
    }
#endif
    for (; k < in_features; ++k) {
      input_sum += (int32_t)input[k];
    }
  }

  for (uint32_t j = 0; j < out_features; ++j) {
    const int8_t* w_row = p->weight + (size_t)j * in_features;
    int32_t mm_acc = 0;
    uint32_t i = 0;
#if MV2_USE_MVE
    while (i + 16 <= in_features) {
      int8x16_t a = vld1q_s8(input + i);
      int8x16_t b = vld1q_s8(w_row + i);
      mm_acc = vmladavaq_s8(mm_acc, a, b);
      i += 16;
    }
#endif
    for (; i < in_features; ++i) {
      mm_acc += (int32_t)input[i] * (int32_t)w_row[i];
    }
    int32_t acc = mm_acc + input_sum * filter_offset;
    if (kernel_sum != (const int32_t*)0) acc += kernel_sum[j];
    if (bias != (const int32_t*)0) acc += bias[j];
    acc = scalar_requantize(acc, multiplier, shift);
    acc += output_offset;
    if (acc < act_min) acc = act_min;
    if (acc > act_max) acc = act_max;
    output[j] = (int8_t)acc;
  }
}

void mobilenet_v2_inference(const float* input_float, int8_t* output_q) {
  /* Body generated by tools/dump_mv2_artifacts.py — emits one call per
   * lowered cortex_m op.  Output is int8 logits; the Python test side
   * dequantizes with MV2_OUTPUT_SCALE / MV2_OUTPUT_ZERO_POINT.  */
  (void)input_float;
  (void)output_q;
#include "mv2_inference_body.h"
}
