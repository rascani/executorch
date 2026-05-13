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

#ifdef MV2_PROFILE_KERNELS
/* Per-kernel cycle accounting.  Off by default — set MV2_PROFILE_KERNELS
 * at compile time to enable.  Counters are uint64_t to handle whole-model
 * accumulation; the runner prints them after inference. */
#include <stdio.h>

uint64_t mv2_prof_quantize_input = 0;
uint64_t mv2_prof_conv2d_3x3_first = 0;
uint64_t mv2_prof_conv2d_1x1 = 0;
uint64_t mv2_prof_dwconv2d = 0;
uint64_t mv2_prof_add_s8 = 0;
uint64_t mv2_prof_avgpool = 0;
uint64_t mv2_prof_gemv = 0;
uint32_t mv2_prof_n_quantize_input = 0;
uint32_t mv2_prof_n_conv2d_3x3_first = 0;
uint32_t mv2_prof_n_conv2d_1x1 = 0;
uint32_t mv2_prof_n_dwconv2d = 0;
uint32_t mv2_prof_n_add_s8 = 0;
uint32_t mv2_prof_n_avgpool = 0;
uint32_t mv2_prof_n_gemv = 0;

static inline uint32_t prof_read_cycles(void) {
  return *(volatile uint32_t*)0xE0001004u;  /* DWT_CYCCNT */
}
#define PROF_START(name) uint32_t __prof_start_##name = prof_read_cycles()
#define PROF_END(name)   do { \
    mv2_prof_##name += (uint64_t)(prof_read_cycles() - __prof_start_##name); \
    mv2_prof_n_##name++; \
  } while (0)
#else
#define PROF_START(name) ((void)0)
#define PROF_END(name)   ((void)0)
#endif

static void quantize_input(const float* in, int8_t* out, const QuantInputParams* p) {
  PROF_START(quantize_input);
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
  PROF_END(quantize_input);
}

static __attribute__((always_inline)) inline void conv2d_s8(const int8_t* input, int8_t* output, const Conv2dParams* p) {
  /* NHWC int8 input, NHWC int8 output, OHWI int8 weights, per-channel requant.
   * Matches the math in backends/cortex_m/ops/operators.py:751
   * quantized_conv2d_impl bit-exactly when using scalar_requantize.
   *
   * always_inline so the LayerParams (a static const struct in mv2_params.h)
   * gets const-folded into the kernel per-call: gate-checks become
   * compile-time constants and dead paths drop out.  Trades code size
   * for inner-loop quality. */
#ifdef MV2_PROFILE_KERNELS
  uint32_t __prof_t0 = prof_read_cycles();
  const int __prof_is_first = (p->kernel_h > 1u);
#endif
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
  const int8_t*  shifts = p->requant_shifts;

  const size_t in_row_stride = (size_t)in_w * in_c;
  const size_t out_row_stride = (size_t)out_w * out_c;
  const size_t w_oc_stride = (size_t)k_h * k_w * in_c;

#if MV2_USE_MVE
  /* First-conv specialization: 3x3 stride-2 pad-1 in_c=3.  Uses pre-packed
   * weights (27 OHWI bytes + 5 zero bytes per OC) and an im2col patch built
   * inline per output pixel.  Math: AOT folds `input_offset * sum(w[oc])`
   * into bias; runtime fills cropped patch positions with -input_offset so
   * the algebra still produces the correct per-pixel partial sum.
   *
   * Per output pixel: 1 patch build (~10 cycles incl. bounds checks) +
   * 8 OC blocks × (2 vmladavaq_s8 + bias load + requant + scatter store +
   * setup) ≈ 90 cycles.  vs ~560 cycles in the generic 4-OC path → ~6×. */
  if (p->weight_packed_32 != (const int8_t*)0
      && k_h == 3u && k_w == 3u && in_c == 3u
      && stride_h == 2u && stride_w == 2u
      && pad_h == 1 && pad_w == 1
      && (out_c & 3u) == 0u) {
    const int8_t* w_packed = p->weight_packed_32;
    const int8_t  pad_byte = (int8_t)(-input_offset);
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      const int32_t ih_base = (int32_t)(oh * 2u) - 1;  // pad=1
      const int kh_valid[3] = {
        (ih_base + 0 >= 0 && ih_base + 0 < (int32_t)in_h) ? 1 : 0,
        (ih_base + 1 >= 0 && ih_base + 1 < (int32_t)in_h) ? 1 : 0,
        (ih_base + 2 >= 0 && ih_base + 2 < (int32_t)in_h) ? 1 : 0,
      };
      for (uint32_t ow = 0; ow < out_w; ++ow) {
        const int32_t iw_base = (int32_t)(ow * 2u) - 1;  // pad=1
        /* Build the 32-byte patch: 27 input bytes in (kh, kw, ic) order,
         * with -input_offset for cropped positions, followed by 5 zero pads
         * (which match the zero-padded weight tail and contribute nothing
         * to the MAC). */
        int8_t patch[32] = {0};
        for (uint32_t kh = 0; kh < 3; ++kh) {
          int32_t ih = ih_base + (int32_t)kh;
          int8_t* p_row = patch + kh * 9;
          if (kh_valid[kh]) {
            const int8_t* x_row = input + (size_t)ih * in_row_stride;
            for (uint32_t kw = 0; kw < 3; ++kw) {
              int32_t iw = iw_base + (int32_t)kw;
              int8_t* p_pos = p_row + kw * 3;
              if (iw >= 0 && iw < (int32_t)in_w) {
                const int8_t* x_pos = x_row + (size_t)iw * 3;
                p_pos[0] = x_pos[0];
                p_pos[1] = x_pos[1];
                p_pos[2] = x_pos[2];
              } else {
                p_pos[0] = pad_byte;
                p_pos[1] = pad_byte;
                p_pos[2] = pad_byte;
              }
            }
          } else {
            /* All 9 ints in this row are pad */
            p_row[0] = pad_byte; p_row[1] = pad_byte; p_row[2] = pad_byte;
            p_row[3] = pad_byte; p_row[4] = pad_byte; p_row[5] = pad_byte;
            p_row[6] = pad_byte; p_row[7] = pad_byte; p_row[8] = pad_byte;
          }
        }

        int8x16_t vx_lo = vld1q_s8(patch + 0);
        int8x16_t vx_hi = vld1q_s8(patch + 16);

        for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
          int32_t a0 = bias[ocb + 0];
          int32_t a1 = bias[ocb + 1];
          int32_t a2 = bias[ocb + 2];
          int32_t a3 = bias[ocb + 3];

          int8x16_t w0_lo = vld1q_s8(w_packed + (size_t)(ocb + 0) * 32 + 0);
          int8x16_t w0_hi = vld1q_s8(w_packed + (size_t)(ocb + 0) * 32 + 16);
          int8x16_t w1_lo = vld1q_s8(w_packed + (size_t)(ocb + 1) * 32 + 0);
          int8x16_t w1_hi = vld1q_s8(w_packed + (size_t)(ocb + 1) * 32 + 16);
          int8x16_t w2_lo = vld1q_s8(w_packed + (size_t)(ocb + 2) * 32 + 0);
          int8x16_t w2_hi = vld1q_s8(w_packed + (size_t)(ocb + 2) * 32 + 16);
          int8x16_t w3_lo = vld1q_s8(w_packed + (size_t)(ocb + 3) * 32 + 0);
          int8x16_t w3_hi = vld1q_s8(w_packed + (size_t)(ocb + 3) * 32 + 16);

          a0 = vmladavaq_s8(a0, vx_lo, w0_lo);
          a0 = vmladavaq_s8(a0, vx_hi, w0_hi);
          a1 = vmladavaq_s8(a1, vx_lo, w1_lo);
          a1 = vmladavaq_s8(a1, vx_hi, w1_hi);
          a2 = vmladavaq_s8(a2, vx_lo, w2_lo);
          a2 = vmladavaq_s8(a2, vx_hi, w2_hi);
          a3 = vmladavaq_s8(a3, vx_lo, w3_lo);
          a3 = vmladavaq_s8(a3, vx_hi, w3_hi);

          int32x4_t mult = vld1q_s32(mults + ocb);
          int32x4_t shft = vldrbq_s32(shifts + ocb);
          int32x4_t accv = vsetq_lane_s32(a0, vdupq_n_s32(0), 0);
          accv = vsetq_lane_s32(a1, accv, 1);
          accv = vsetq_lane_s32(a2, accv, 2);
          accv = vsetq_lane_s32(a3, accv, 3);
          accv = mve_requantize_per_channel(accv, mult, shft);
          accv = vaddq_n_s32(accv, output_offset);
          accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
          vstrbq_s32(output + (size_t)oh * out_row_stride
                       + (size_t)ow * out_c + ocb, accv);
        }
      }
    }
#ifdef MV2_PROFILE_KERNELS
    {
      uint32_t __prof_dt = prof_read_cycles() - __prof_t0;
      mv2_prof_conv2d_3x3_first += __prof_dt;
      mv2_prof_n_conv2d_3x3_first++;
    }
#endif
    return;
  }

  /* Fastest path: 1x1 stride-1 conv, no padding, even out_w.  Tiles 2 output
   * pixels per inner iteration sharing the 4 weight-row loads, on top of the
   * 4-OC blocking.  Each IC chunk runs 8 MVE MACs against 6 loads — ~22%
   * fewer instructions per output pixel than the single-pixel path.  All MV2
   * 1x1 convs except the head (out_w=7) hit this path. */
  if (k_h == 1u && k_w == 1u && pad_h == 0 && pad_w == 0
      && stride_h == 1u && stride_w == 1u
      && (out_c & 3u) == 0u && (out_w & 1u) == 0u
      && input_offset == 0) {
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      const int8_t* row_in = input + (size_t)oh * in_row_stride;
      int8_t* row_out = output + (size_t)oh * out_row_stride;
      for (uint32_t ow = 0; ow < out_w; ow += 2) {
        const int8_t* x0 = row_in + (size_t)(ow + 0) * in_c;
        const int8_t* x1 = row_in + (size_t)(ow + 1) * in_c;
        for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
          int32_t a0_0 = bias[ocb + 0], a0_1 = bias[ocb + 0];
          int32_t a1_0 = bias[ocb + 1], a1_1 = bias[ocb + 1];
          int32_t a2_0 = bias[ocb + 2], a2_1 = bias[ocb + 2];
          int32_t a3_0 = bias[ocb + 3], a3_1 = bias[ocb + 3];
          const int8_t* w0 = weight + (size_t)(ocb + 0) * w_oc_stride;
          const int8_t* w1 = weight + (size_t)(ocb + 1) * w_oc_stride;
          const int8_t* w2 = weight + (size_t)(ocb + 2) * w_oc_stride;
          const int8_t* w3 = weight + (size_t)(ocb + 3) * w_oc_stride;
          uint32_t ic = 0;
          while (ic + 16 <= in_c) {
            int8x16_t v_x0 = vld1q_s8(x0 + ic);
            int8x16_t v_x1 = vld1q_s8(x1 + ic);
            int8x16_t v_w0 = vld1q_s8(w0 + ic);
            int8x16_t v_w1 = vld1q_s8(w1 + ic);
            int8x16_t v_w2 = vld1q_s8(w2 + ic);
            int8x16_t v_w3 = vld1q_s8(w3 + ic);
            a0_0 = vmladavaq_s8(a0_0, v_x0, v_w0);
            a1_0 = vmladavaq_s8(a1_0, v_x0, v_w1);
            a2_0 = vmladavaq_s8(a2_0, v_x0, v_w2);
            a3_0 = vmladavaq_s8(a3_0, v_x0, v_w3);
            a0_1 = vmladavaq_s8(a0_1, v_x1, v_w0);
            a1_1 = vmladavaq_s8(a1_1, v_x1, v_w1);
            a2_1 = vmladavaq_s8(a2_1, v_x1, v_w2);
            a3_1 = vmladavaq_s8(a3_1, v_x1, v_w3);
            ic += 16;
          }
          for (; ic < in_c; ++ic) {
            int32_t x0_v = (int32_t)x0[ic];
            int32_t x1_v = (int32_t)x1[ic];
            int32_t w0_v = (int32_t)w0[ic];
            int32_t w1_v = (int32_t)w1[ic];
            int32_t w2_v = (int32_t)w2[ic];
            int32_t w3_v = (int32_t)w3[ic];
            a0_0 += x0_v * w0_v;  a0_1 += x1_v * w0_v;
            a1_0 += x0_v * w1_v;  a1_1 += x1_v * w1_v;
            a2_0 += x0_v * w2_v;  a2_1 += x1_v * w2_v;
            a3_0 += x0_v * w3_v;  a3_1 += x1_v * w3_v;
          }
          int32x4_t mult = vld1q_s32(mults + ocb);
          int32x4_t shft = vldrbq_s32(shifts + ocb);
          int32x4_t accv0 = vsetq_lane_s32(a0_0, vdupq_n_s32(0), 0);
          accv0 = vsetq_lane_s32(a1_0, accv0, 1);
          accv0 = vsetq_lane_s32(a2_0, accv0, 2);
          accv0 = vsetq_lane_s32(a3_0, accv0, 3);
          accv0 = mve_requantize_per_channel(accv0, mult, shft);
          accv0 = vaddq_n_s32(accv0, output_offset);
          accv0 = vmaxq_s32(accv0, v_act_min);
          accv0 = vminq_s32(accv0, v_act_max);
          vstrbq_s32(row_out + (size_t)(ow + 0) * out_c + ocb, accv0);
          int32x4_t accv1 = vsetq_lane_s32(a0_1, vdupq_n_s32(0), 0);
          accv1 = vsetq_lane_s32(a1_1, accv1, 1);
          accv1 = vsetq_lane_s32(a2_1, accv1, 2);
          accv1 = vsetq_lane_s32(a3_1, accv1, 3);
          accv1 = mve_requantize_per_channel(accv1, mult, shft);
          accv1 = vaddq_n_s32(accv1, output_offset);
          accv1 = vmaxq_s32(accv1, v_act_min);
          accv1 = vminq_s32(accv1, v_act_max);
          vstrbq_s32(row_out + (size_t)(ow + 1) * out_c + ocb, accv1);
        }
      }
    }
#ifdef MV2_PROFILE_KERNELS
    {
      uint32_t __prof_dt = prof_read_cycles() - __prof_t0;
      mv2_prof_conv2d_1x1 += __prof_dt;
      mv2_prof_n_conv2d_1x1++;
    }
#endif
    return;
  }
  /* Generic 4-OC-blocked path: handles all out_w cases (incl. odd) plus
   * convs with kernel > 1, padding, stride > 1. */
  if ((out_c & 3u) == 0u) {
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    /* The extractor folds input_offset * sum(weight[oc]) into the bias for
     * safe layers (1x1 with no padding), and sets input_offset == 0 to
     * signal "no runtime offset needed".  Skip the sum_w accumulation in
     * that case — it saves a `vaddvq_s8` per 16-channel chunk per OC. */
    const int offset_baked = (input_offset == 0);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      for (uint32_t ow = 0; ow < out_w; ++ow) {
        for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
          int32_t a0 = (bias != (const int32_t*)0) ? bias[ocb + 0] : 0;
          int32_t a1 = (bias != (const int32_t*)0) ? bias[ocb + 1] : 0;
          int32_t a2 = (bias != (const int32_t*)0) ? bias[ocb + 2] : 0;
          int32_t a3 = (bias != (const int32_t*)0) ? bias[ocb + 3] : 0;
          int32_t sum_w0 = 0, sum_w1 = 0, sum_w2 = 0, sum_w3 = 0;
          const int8_t* w0_oc = weight + (size_t)(ocb + 0) * w_oc_stride;
          const int8_t* w1_oc = weight + (size_t)(ocb + 1) * w_oc_stride;
          const int8_t* w2_oc = weight + (size_t)(ocb + 2) * w_oc_stride;
          const int8_t* w3_oc = weight + (size_t)(ocb + 3) * w_oc_stride;
          for (uint32_t kh = 0; kh < k_h; ++kh) {
            const int32_t ih = (int32_t)(oh * stride_h) - pad_h + (int32_t)kh;
            if (ih < 0 || (uint32_t)ih >= in_h) continue;
            for (uint32_t kw = 0; kw < k_w; ++kw) {
              const int32_t iw = (int32_t)(ow * stride_w) - pad_w + (int32_t)kw;
              if (iw < 0 || (uint32_t)iw >= in_w) continue;
              const int8_t* x = input + (size_t)ih * in_row_stride
                                + (size_t)iw * in_c;
              const size_t w_off = ((size_t)kh * k_w + kw) * in_c;
              uint32_t ic = 0;
              if (offset_baked) {
                while (ic + 16 <= in_c) {
                  int8x16_t v_x  = vld1q_s8(x + ic);
                  int8x16_t v_w0 = vld1q_s8(w0_oc + w_off + ic);
                  int8x16_t v_w1 = vld1q_s8(w1_oc + w_off + ic);
                  int8x16_t v_w2 = vld1q_s8(w2_oc + w_off + ic);
                  int8x16_t v_w3 = vld1q_s8(w3_oc + w_off + ic);
                  a0 = vmladavaq_s8(a0, v_x, v_w0);
                  a1 = vmladavaq_s8(a1, v_x, v_w1);
                  a2 = vmladavaq_s8(a2, v_x, v_w2);
                  a3 = vmladavaq_s8(a3, v_x, v_w3);
                  ic += 16;
                }
              } else {
                while (ic + 16 <= in_c) {
                  int8x16_t v_x  = vld1q_s8(x + ic);
                  int8x16_t v_w0 = vld1q_s8(w0_oc + w_off + ic);
                  int8x16_t v_w1 = vld1q_s8(w1_oc + w_off + ic);
                  int8x16_t v_w2 = vld1q_s8(w2_oc + w_off + ic);
                  int8x16_t v_w3 = vld1q_s8(w3_oc + w_off + ic);
                  a0 = vmladavaq_s8(a0, v_x, v_w0);
                  a1 = vmladavaq_s8(a1, v_x, v_w1);
                  a2 = vmladavaq_s8(a2, v_x, v_w2);
                  a3 = vmladavaq_s8(a3, v_x, v_w3);
                  sum_w0 += vaddvq_s8(v_w0);
                  sum_w1 += vaddvq_s8(v_w1);
                  sum_w2 += vaddvq_s8(v_w2);
                  sum_w3 += vaddvq_s8(v_w3);
                  ic += 16;
                }
              }
              for (; ic < in_c; ++ic) {
                int32_t x_v = (int32_t)x[ic];
                a0 += x_v * (int32_t)w0_oc[w_off + ic];
                a1 += x_v * (int32_t)w1_oc[w_off + ic];
                a2 += x_v * (int32_t)w2_oc[w_off + ic];
                a3 += x_v * (int32_t)w3_oc[w_off + ic];
                if (!offset_baked) {
                  sum_w0 += (int32_t)w0_oc[w_off + ic];
                  sum_w1 += (int32_t)w1_oc[w_off + ic];
                  sum_w2 += (int32_t)w2_oc[w_off + ic];
                  sum_w3 += (int32_t)w3_oc[w_off + ic];
                }
              }
            }
          }
          if (!offset_baked) {
            a0 += sum_w0 * input_offset;
            a1 += sum_w1 * input_offset;
            a2 += sum_w2 * input_offset;
            a3 += sum_w3 * input_offset;
          }
          int32x4_t accv = vsetq_lane_s32(a0, vdupq_n_s32(0), 0);
          accv = vsetq_lane_s32(a1, accv, 1);
          accv = vsetq_lane_s32(a2, accv, 2);
          accv = vsetq_lane_s32(a3, accv, 3);
          int32x4_t mult = vld1q_s32(mults + ocb);
          int32x4_t shft = vldrbq_s32(shifts + ocb);
          accv = mve_requantize_per_channel(accv, mult, shft);
          accv = vaddq_n_s32(accv, output_offset);
          accv = vmaxq_s32(accv, v_act_min);
          accv = vminq_s32(accv, v_act_max);
          vstrbq_s32(output + (size_t)oh * out_row_stride
                       + (size_t)ow * out_c + ocb, accv);
        }
      }
    }
#ifdef MV2_PROFILE_KERNELS
    {
      uint32_t __prof_dt = prof_read_cycles() - __prof_t0;
      if (__prof_is_first) {
        mv2_prof_conv2d_3x3_first += __prof_dt;
        mv2_prof_n_conv2d_3x3_first++;
      } else {
        mv2_prof_conv2d_1x1 += __prof_dt;
        mv2_prof_n_conv2d_1x1++;
      }
    }
#endif
    return;
  }
#endif

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
#ifdef MV2_PROFILE_KERNELS
  uint32_t __prof_dt = prof_read_cycles() - __prof_t0;
  if (__prof_is_first) {
    mv2_prof_conv2d_3x3_first += __prof_dt;
    mv2_prof_n_conv2d_3x3_first++;
  } else {
    mv2_prof_conv2d_1x1 += __prof_dt;
    mv2_prof_n_conv2d_1x1++;
  }
#endif
}

static __attribute__((always_inline)) inline void dwconv2d_s8(
    const int8_t* input, int8_t* output, const DepthwiseConv2dParams* p) {
  PROF_START(dwconv2d);
  /* NHWC int8 input, IHWO int8 weights [1, kH, kW, C], depth_multiplier=1
   * (out_c == in_c).  Per-channel requant, fused ReLU bounds.
   * Mirrors quantized_depthwise_conv2d_impl in operators.py:858 bit-exactly.
   *
   * Fast path (MV2_USE_MVE && out_c % 4 == 0): processes 4 output channels per
   * inner iteration via vldrbq_s32 / vmulq_s32 / vaddq_s32 and the vector
   * mve_requantize_per_channel.  This is ~4x faster than the scalar path
   * since depthwise has no input-channel reduction — each output channel
   * needs only its matching input channel, so the natural vectorization axis
   * is the channel dimension.  All MV2 depthwise layers (channels in
   * {32, 96, 144, 192, 384, 576, 960}) hit this path.
   *
   * Scalar fallback runs when MVE is unavailable or out_c is not a multiple
   * of 4 (first conv's in_c=3 input never reaches a depthwise stage so this
   * branch is unused in MV2 today; it's here for correctness on synthetic
   * tests with arbitrary channel counts). */
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
  const int8_t*  shifts = p->requant_shifts;
  const size_t in_row_stride = (size_t)in_w * in_c;
  const size_t out_row_stride = (size_t)out_w * out_c;
  const size_t w_row_stride = (size_t)k_w * out_c;

#if MV2_USE_MVE
  /* 3x3 stride-1 pad-1 fast path: 4-pixel spatial tiling.
   *
   * Per (oh, cb=4, 4-pixel tile): each kh-row shares 3 weight loads across 4
   * adjacent output pixels and shares 6 input column loads via the sliding
   * window (pixel ow uses iw cols [ow-1, ow, ow+1]; 4 pixels span iw cols
   * [ow_base-1 .. ow_base+4]).  Per kh: 6 vldrbq_s32 + 3 vldrbq_s32 +
   * 6 vaddq_s32 (input_offset add) + 12 (vmulq_s32 + vaddq_s32) MAC pairs
   * → ~33 vector ops for 16 outputs (4 pixels × 4 channels) = ~2.06 ops/output
   * vs 4-OC × 1-pixel ~9 ops/output (4.3x reduction in inner-loop ops).
   *
   * Triggers when 13 of 17 MV2 dw layers run (out_w in {7, 14, 28, 56, 112},
   * stride 1, pad 1).  Boundary pixels (ow=0 and the right-edge tail) fall
   * through to the per-pixel slow path below since their kernel-window cols
   * include the pad position — handled by the generic 4-wide path. */
  if (k_h == 3u && k_w == 3u && stride_h == 1u && stride_w == 1u &&
      pad_h == 1 && pad_w == 1 && (out_c & 3u) == 0u && out_w >= 6u) {
    const int32x4_t v_in_off  = vdupq_n_s32(input_offset);
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      int kh_valid[3];
      int32_t ih_for_kh[3];
      for (uint32_t kh = 0; kh < 3; ++kh) {
        int32_t ih = (int32_t)oh - 1 + (int32_t)kh;
        kh_valid[kh] = (ih >= 0 && (uint32_t)ih < in_h) ? 1 : 0;
        ih_for_kh[kh] = ih;
      }

      /* Left boundary: ow = 0 (uses scalar pad position iw=-1). */
      for (uint32_t cb = 0; cb < out_c; cb += 4) {
        int32x4_t acc = (bias != (const int32_t*)0)
            ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
        for (uint32_t kh = 0; kh < 3; ++kh) {
          if (!kh_valid[kh]) continue;
          const int8_t* w_row = weight + (size_t)kh * w_row_stride;
          const int8_t* x_row = input + (size_t)ih_for_kh[kh] * in_row_stride;
          /* iw = -1, 0, 1; skip iw=-1 (out of bounds). */
          for (uint32_t kw = 1; kw < 3; ++kw) {
            int32_t iw = (int32_t)kw - 1;
            int32x4_t x = vldrbq_s32(x_row + (size_t)iw * in_c + cb);
            int32x4_t w = vldrbq_s32(w_row + (size_t)kw * out_c + cb);
            x = vaddq_s32(x, v_in_off);
            acc = vaddq_s32(acc, vmulq_s32(x, w));
          }
        }
        int32x4_t mult = vld1q_s32(mults + cb);
        int32x4_t shft = vldrbq_s32(shifts + cb);
        acc = mve_requantize_per_channel(acc, mult, shft);
        acc = vaddq_n_s32(acc, output_offset);
        acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
        vstrbq_s32(output + (size_t)oh * out_row_stride + cb, acc);
      }

      /* Interior 4-pixel tiles: ow_base in [1, out_w-5].  Each tile reads
       * 6 contiguous input columns iw in [ow_base-1 .. ow_base+4] which are
       * all in-bounds. */
      uint32_t ow_base = 1;
      for (; ow_base + 5 <= out_w; ow_base += 4) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t bias_v = (bias != (const int32_t*)0)
              ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
          int32x4_t acc0 = bias_v;
          int32x4_t acc1 = bias_v;
          int32x4_t acc2 = bias_v;
          int32x4_t acc3 = bias_v;

          for (uint32_t kh = 0; kh < 3; ++kh) {
            if (!kh_valid[kh]) continue;
            const int8_t* x_base =
                input + (size_t)ih_for_kh[kh] * in_row_stride
                      + (size_t)(ow_base - 1) * in_c + cb;
            const int8_t* w_base = weight + (size_t)kh * w_row_stride + cb;
            int32x4_t w0 = vldrbq_s32(w_base + 0 * out_c);
            int32x4_t w1 = vldrbq_s32(w_base + 1 * out_c);
            int32x4_t w2 = vldrbq_s32(w_base + 2 * out_c);
            int32x4_t x0 = vaddq_s32(vldrbq_s32(x_base + 0 * in_c), v_in_off);
            int32x4_t x1 = vaddq_s32(vldrbq_s32(x_base + 1 * in_c), v_in_off);
            int32x4_t x2 = vaddq_s32(vldrbq_s32(x_base + 2 * in_c), v_in_off);
            acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
            acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
            acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
            int32x4_t x3 = vaddq_s32(vldrbq_s32(x_base + 3 * in_c), v_in_off);
            acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
            acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
            acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
            int32x4_t x4 = vaddq_s32(vldrbq_s32(x_base + 4 * in_c), v_in_off);
            acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
            acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
            acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
            int32x4_t x5 = vaddq_s32(vldrbq_s32(x_base + 5 * in_c), v_in_off);
            acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
            acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
            acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
          }

          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc0 = mve_requantize_per_channel(acc0, mult, shft);
          acc1 = mve_requantize_per_channel(acc1, mult, shft);
          acc2 = mve_requantize_per_channel(acc2, mult, shft);
          acc3 = mve_requantize_per_channel(acc3, mult, shft);
          acc0 = vaddq_n_s32(acc0, output_offset);
          acc1 = vaddq_n_s32(acc1, output_offset);
          acc2 = vaddq_n_s32(acc2, output_offset);
          acc3 = vaddq_n_s32(acc3, output_offset);
          acc0 = vminq_s32(vmaxq_s32(acc0, v_act_min), v_act_max);
          acc1 = vminq_s32(vmaxq_s32(acc1, v_act_min), v_act_max);
          acc2 = vminq_s32(vmaxq_s32(acc2, v_act_min), v_act_max);
          acc3 = vminq_s32(vmaxq_s32(acc3, v_act_min), v_act_max);

          int8_t* out_p = output + (size_t)oh * out_row_stride
                                 + (size_t)ow_base * out_c + cb;
          vstrbq_s32(out_p + 0 * out_c, acc0);
          vstrbq_s32(out_p + 1 * out_c, acc1);
          vstrbq_s32(out_p + 2 * out_c, acc2);
          vstrbq_s32(out_p + 3 * out_c, acc3);
        }
      }

      /* Right tail: ow in [ow_base .. out_w-1].  All cols valid except
       * iw=out_w for ow=out_w-1 (which is the pad position). */
      for (uint32_t ow = ow_base; ow < out_w; ++ow) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t acc = (bias != (const int32_t*)0)
              ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
          for (uint32_t kh = 0; kh < 3; ++kh) {
            if (!kh_valid[kh]) continue;
            const int8_t* w_row = weight + (size_t)kh * w_row_stride;
            const int8_t* x_row = input + (size_t)ih_for_kh[kh] * in_row_stride;
            for (uint32_t kw = 0; kw < 3; ++kw) {
              int32_t iw = (int32_t)ow - 1 + (int32_t)kw;
              if (iw < 0 || (uint32_t)iw >= in_w) continue;
              int32x4_t x = vldrbq_s32(x_row + (size_t)iw * in_c + cb);
              int32x4_t w = vldrbq_s32(w_row + (size_t)kw * out_c + cb);
              x = vaddq_s32(x, v_in_off);
              acc = vaddq_s32(acc, vmulq_s32(x, w));
            }
          }
          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc = mve_requantize_per_channel(acc, mult, shft);
          acc = vaddq_n_s32(acc, output_offset);
          acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
          vstrbq_s32(output + (size_t)oh * out_row_stride
                       + (size_t)ow * out_c + cb, acc);
        }
      }
    }
    PROF_END(dwconv2d);
    return;
  }

  /* 3x3 stride-2 pad-1 fast path: 2-pixel spatial tiling.
   *
   * Per (oh, cb=4, 2-pixel tile): adjacent output pixels at stride 2 share
   * 1 input column (col 2*ow_base+1 = 2*(ow_base+1)-1).  Per kh row:
   * 5 input loads + 3 weight loads + 6 (vmulq + vaddq) MAC pairs = 20 vector
   * ops for 8 outputs (2 pixels × 4 channels) = 2.5 ops/output (vs ~3.4 for
   * the 4-OC × 1-pixel path).  Handles 4 of 17 MV2 dw layers (stride-2
   * downsamples). */
  if (k_h == 3u && k_w == 3u && stride_h == 2u && stride_w == 2u &&
      pad_h == 1 && pad_w == 1 && (out_c & 3u) == 0u && out_w >= 2u) {
    const int32x4_t v_in_off  = vdupq_n_s32(input_offset);
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      int kh_valid[3];
      int32_t ih_for_kh[3];
      for (uint32_t kh = 0; kh < 3; ++kh) {
        int32_t ih = (int32_t)(oh * 2u) - 1 + (int32_t)kh;
        kh_valid[kh] = (ih >= 0 && (uint32_t)ih < in_h) ? 1 : 0;
        ih_for_kh[kh] = ih;
      }

      /* Left boundary: ow = 0 uses pad col iw=-1.  Single-pixel slow path. */
      for (uint32_t cb = 0; cb < out_c; cb += 4) {
        int32x4_t acc = (bias != (const int32_t*)0)
            ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
        for (uint32_t kh = 0; kh < 3; ++kh) {
          if (!kh_valid[kh]) continue;
          const int8_t* w_row = weight + (size_t)kh * w_row_stride;
          const int8_t* x_row = input + (size_t)ih_for_kh[kh] * in_row_stride;
          for (uint32_t kw = 1; kw < 3; ++kw) {
            int32_t iw = (int32_t)kw - 1;
            int32x4_t x = vldrbq_s32(x_row + (size_t)iw * in_c + cb);
            int32x4_t w = vldrbq_s32(w_row + (size_t)kw * out_c + cb);
            x = vaddq_s32(x, v_in_off);
            acc = vaddq_s32(acc, vmulq_s32(x, w));
          }
        }
        int32x4_t mult = vld1q_s32(mults + cb);
        int32x4_t shft = vldrbq_s32(shifts + cb);
        acc = mve_requantize_per_channel(acc, mult, shft);
        acc = vaddq_n_s32(acc, output_offset);
        acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
        vstrbq_s32(output + (size_t)oh * out_row_stride + cb, acc);
      }

      /* Interior 2-pixel tiles.  Pixel ow uses iw cols [2*ow-1, 2*ow, 2*ow+1].
       * Pair (ow_base, ow_base+1) uses cols [2*ow_base-1 .. 2*ow_base+3] (5
       * cols).  All in-bounds when 2*ow_base-1 >= 0 AND 2*ow_base+3 < in_w. */
      uint32_t ow_base = 1;
      for (; ow_base + 2 <= out_w && (int32_t)(ow_base * 2u + 3) < (int32_t)in_w;
           ow_base += 2) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t bias_v = (bias != (const int32_t*)0)
              ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
          int32x4_t acc0 = bias_v;
          int32x4_t acc1 = bias_v;

          for (uint32_t kh = 0; kh < 3; ++kh) {
            if (!kh_valid[kh]) continue;
            const int8_t* x_base =
                input + (size_t)ih_for_kh[kh] * in_row_stride
                      + (size_t)(ow_base * 2u - 1u) * in_c + cb;
            const int8_t* w_base = weight + (size_t)kh * w_row_stride + cb;
            int32x4_t w0 = vldrbq_s32(w_base + 0 * out_c);
            int32x4_t w1 = vldrbq_s32(w_base + 1 * out_c);
            int32x4_t w2 = vldrbq_s32(w_base + 2 * out_c);
            int32x4_t x0 = vaddq_s32(vldrbq_s32(x_base + 0 * in_c), v_in_off);
            int32x4_t x1 = vaddq_s32(vldrbq_s32(x_base + 1 * in_c), v_in_off);
            int32x4_t x2 = vaddq_s32(vldrbq_s32(x_base + 2 * in_c), v_in_off);
            acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
            acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
            acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
            int32x4_t x3 = vaddq_s32(vldrbq_s32(x_base + 3 * in_c), v_in_off);
            int32x4_t x4 = vaddq_s32(vldrbq_s32(x_base + 4 * in_c), v_in_off);
            acc1 = vaddq_s32(acc1, vmulq_s32(x2, w0));
            acc1 = vaddq_s32(acc1, vmulq_s32(x3, w1));
            acc1 = vaddq_s32(acc1, vmulq_s32(x4, w2));
          }

          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc0 = mve_requantize_per_channel(acc0, mult, shft);
          acc1 = mve_requantize_per_channel(acc1, mult, shft);
          acc0 = vaddq_n_s32(acc0, output_offset);
          acc1 = vaddq_n_s32(acc1, output_offset);
          acc0 = vminq_s32(vmaxq_s32(acc0, v_act_min), v_act_max);
          acc1 = vminq_s32(vmaxq_s32(acc1, v_act_min), v_act_max);
          int8_t* out_p = output + (size_t)oh * out_row_stride
                                 + (size_t)ow_base * out_c + cb;
          vstrbq_s32(out_p + 0 * out_c, acc0);
          vstrbq_s32(out_p + 1 * out_c, acc1);
        }
      }

      /* Right tail: ow_base..out_w-1.  Single-pixel slow path with per-kw bounds. */
      for (uint32_t ow = ow_base; ow < out_w; ++ow) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t acc = (bias != (const int32_t*)0)
              ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
          for (uint32_t kh = 0; kh < 3; ++kh) {
            if (!kh_valid[kh]) continue;
            const int8_t* w_row = weight + (size_t)kh * w_row_stride;
            const int8_t* x_row = input + (size_t)ih_for_kh[kh] * in_row_stride;
            for (uint32_t kw = 0; kw < 3; ++kw) {
              int32_t iw = (int32_t)(ow * 2u) - 1 + (int32_t)kw;
              if (iw < 0 || (uint32_t)iw >= in_w) continue;
              int32x4_t x = vldrbq_s32(x_row + (size_t)iw * in_c + cb);
              int32x4_t w = vldrbq_s32(w_row + (size_t)kw * out_c + cb);
              x = vaddq_s32(x, v_in_off);
              acc = vaddq_s32(acc, vmulq_s32(x, w));
            }
          }
          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc = mve_requantize_per_channel(acc, mult, shft);
          acc = vaddq_n_s32(acc, output_offset);
          acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
          vstrbq_s32(output + (size_t)oh * out_row_stride
                       + (size_t)ow * out_c + cb, acc);
        }
      }
    }
    PROF_END(dwconv2d);
    return;
  }

  /* Generic 4-wide path: stride-2 layers, edge cases without 4-pixel tiles. */
  if ((out_c & 3u) == 0u) {
    const int32x4_t v_in_off  = vdupq_n_s32(input_offset);
    const int32x4_t v_act_min = vdupq_n_s32(act_min);
    const int32x4_t v_act_max = vdupq_n_s32(act_max);
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      for (uint32_t ow = 0; ow < out_w; ++ow) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t acc = (bias != (const int32_t*)0)
              ? vld1q_s32(bias + cb) : vdupq_n_s32(0);
          for (uint32_t kh = 0; kh < k_h; ++kh) {
            const int32_t ih = (int32_t)(oh * stride_h) - pad_h + (int32_t)kh;
            if (ih < 0 || (uint32_t)ih >= in_h) continue;
            const int8_t* w_row = weight + (size_t)kh * w_row_stride;
            const int8_t* x_row = input + (size_t)ih * in_row_stride;
            for (uint32_t kw = 0; kw < k_w; ++kw) {
              const int32_t iw = (int32_t)(ow * stride_w) - pad_w + (int32_t)kw;
              if (iw < 0 || (uint32_t)iw >= in_w) continue;
              int32x4_t x = vldrbq_s32(x_row + (size_t)iw * in_c + cb);
              int32x4_t w = vldrbq_s32(w_row + (size_t)kw * out_c + cb);
              x = vaddq_s32(x, v_in_off);
              acc = vaddq_s32(acc, vmulq_s32(x, w));
            }
          }
          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc = mve_requantize_per_channel(acc, mult, shft);
          acc = vaddq_n_s32(acc, output_offset);
          acc = vmaxq_s32(acc, v_act_min);
          acc = vminq_s32(acc, v_act_max);
          vstrbq_s32(output + (size_t)oh * out_row_stride
                       + (size_t)ow * out_c + cb, acc);
        }
      }
    }
    PROF_END(dwconv2d);
    return;
  }
#endif

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
  PROF_END(dwconv2d);
}

static void add_s8(const int8_t* a, const int8_t* b, int8_t* out,
                   const QuantizedAddParams* p) {
  PROF_START(add_s8);
  /* Two-stage requantize: lift each input by SHIFT_INT8=20 and requantize
   * to a shared fixed-point representation, sum, requantize to output scale.
   * Matches operators.py:168 quantized_add_impl bit-exactly. */
  const uint32_t n = p->num_elements;
  const int32_t SHIFT_INT8 = 20;
  uint32_t i = 0;

#if MV2_USE_MVE
  /* Vector path: 4 int8 elements per iter. ~4x throughput vs scalar
   * because three of the per-element steps (two input requantizes + the
   * output requantize) collapse into vector ops. */
  const int32x4_t v_self_zp = vdupq_n_s32(p->self_zero_point);
  const int32x4_t v_other_zp = vdupq_n_s32(p->other_zero_point);
  const int32x4_t v_self_mult = vdupq_n_s32(p->self_multiplier);
  const int32x4_t v_self_shift = vdupq_n_s32(p->self_shift);
  const int32x4_t v_other_mult = vdupq_n_s32(p->other_multiplier);
  const int32x4_t v_other_shift = vdupq_n_s32(p->other_shift);
  const int32x4_t v_out_mult = vdupq_n_s32(p->output_multiplier);
  const int32x4_t v_out_shift = vdupq_n_s32(p->output_shift);
  const int32x4_t v_act_min = vdupq_n_s32(p->activation_min);
  const int32x4_t v_act_max = vdupq_n_s32(p->activation_max);
  while (i + 4 <= n) {
    int32x4_t av = vldrbq_s32(a + i);
    int32x4_t bv = vldrbq_s32(b + i);
    av = vsubq_s32(av, v_self_zp);
    bv = vsubq_s32(bv, v_other_zp);
    av = vshlq_n_s32(av, SHIFT_INT8);
    bv = vshlq_n_s32(bv, SHIFT_INT8);
    int32x4_t a_fp = mve_requantize_per_channel(av, v_self_mult, v_self_shift);
    int32x4_t b_fp = mve_requantize_per_channel(bv, v_other_mult, v_other_shift);
    int32x4_t sum_fp = vaddq_s32(a_fp, b_fp);
    int32x4_t result = mve_requantize_per_channel(sum_fp, v_out_mult, v_out_shift);
    result = vaddq_n_s32(result, p->output_zero_point);
    result = vmaxq_s32(result, v_act_min);
    result = vminq_s32(result, v_act_max);
    vstrbq_s32(out + i, result);
    i += 4;
  }
#endif

  for (; i < n; ++i) {
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
  PROF_END(add_s8);
}

static void avgpool_s8(const int8_t* input, int8_t* output, const AvgPool2dParams* p) {
  PROF_START(avgpool);
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
  PROF_END(avgpool);
}

static void gemv_s8(const int8_t* input, int8_t* output, const LinearParams* p) {
  PROF_START(gemv);
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
  PROF_END(gemv);
}

void mobilenet_v2_inference(const float* input_float, int8_t* output_q) {
  /* Body generated by tools/dump_mv2_artifacts.py — emits one call per
   * lowered cortex_m op.  Output is int8 logits; the Python test side
   * dequantizes with MV2_OUTPUT_SCALE / MV2_OUTPUT_ZERO_POINT.  */
  (void)input_float;
  (void)output_q;
#include "mv2_inference_body.h"
}
