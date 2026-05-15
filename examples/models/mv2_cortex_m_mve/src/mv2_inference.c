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

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
/* Forward declaration: defined below alongside the Phase B fused kernel.
 * Called from conv2d_s8's stem fast path AND from the stem+B0 fused
 * runtime so both paths share the same 2-pixel batched inner. */
static __attribute__((noinline))
void _stem_packed_row_mve(
    const int8_t* input,
    int8_t* out_row,
    uint32_t oh,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_w,
    uint32_t out_c,
    const int8_t* w_packed,
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t input_offset,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max);

/* Noinline 1x1 conv fast path.  Out-of-lining gives the inline-asm inner
 * kernel a clean register context (no surrounding always_inline live
 * values), which the constraint solver couldn't satisfy when this body
 * was inlined into conv2d_s8.  All 34 MV2 1x1 layers hit this; the
 * per-layer function-call frame is paid 34 times per inference
 * (negligible vs. ~38M PMU saved by eliminating the per-tile helper
 * call frames that the previous out-of-line asm helper paid). */
static __attribute__((noinline))
void mv2_conv2d_1x1_fast(const int8_t* input, int8_t* output,
                         const Conv2dParams* p) {
  const uint32_t in_c   = p->in_c;
  const uint32_t out_h  = p->out_h;
  const uint32_t out_w  = p->out_w;
  const uint32_t out_c  = p->out_c;
  const int32_t  output_offset = p->output_offset;
  const int32_t  act_min = p->activation_min;
  const int32_t  act_max = p->activation_max;
  const int8_t*  weight  = p->weight;
  const int32_t* bias    = p->bias;
  const int32_t* mults   = p->requant_mults;
  const int8_t*  shifts  = p->requant_shifts;
  const size_t in_row_stride  = (size_t)in_c;                 /* in_w = out_w for 1x1 stride-1 */
  const size_t out_row_stride = (size_t)out_w * out_c;
  const size_t w_oc_stride    = (size_t)in_c;                 /* k_h = k_w = 1 */
  const int32x4_t v_act_min = vdupq_n_s32(act_min);
  const int32x4_t v_act_max = vdupq_n_s32(act_max);

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    const int8_t* row_in = input + (size_t)oh * (size_t)out_w * (size_t)in_c;
    int8_t* row_out = output + (size_t)oh * out_row_stride;
    uint32_t ow = 0;
    for (; ow + 1 < out_w; ow += 2) {
      const int8_t* x0 = row_in + (size_t)(ow + 0) * in_c;
      const int8_t* x1 = row_in + (size_t)(ow + 1) * in_c;
      for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
        int32_t a0_0 = bias[ocb + 0];
        int32_t a1_0 = bias[ocb + 1];
        int32_t a2_0 = bias[ocb + 2];
        int32_t a3_0 = bias[ocb + 3];
        int32_t a0_1 = a0_0, a1_1 = a1_0, a2_1 = a2_0, a3_1 = a3_0;
        const int8_t* x0_p = x0;
        const int8_t* x1_p = x1;
        const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
        const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
        /* OC pair {0,1} × 2 pixels: 1 weight load is shared across
         * 2 pixel accumulators per OC, halving the per-output load
         * count vs. the per-pixel 4-OC asm.  Inner: 8 ops/iter for
         * 4 outputs = 2.0 ops/output, vs. the old 9 ops/iter for
         * 4 outputs = 2.25 ops/output. */
        __asm__ volatile (
          "wlstp.8     lr, %[n], 1f                 \n"
          "2:                                       \n"
          "  vldrb.8     q0, [%[x0]], #16           \n"
          "  vldrb.8     q2, [%[x1]], #16           \n"
          "  vldrb.8     q1, [%[w0]], #16           \n"
          "  vmladava.s8 %[a00], q0, q1             \n"
          "  vmladava.s8 %[a01], q2, q1             \n"
          "  vldrb.8     q1, [%[w1]], #16           \n"
          "  vmladava.s8 %[a10], q0, q1             \n"
          "  vmladava.s8 %[a11], q2, q1             \n"
          "  letp        lr, 2b                     \n"
          "1:                                       \n"
          : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
            [w0] "+r"(w0_p), [w1] "+r"(w1_p),
            [a00] "+Te"(a0_0), [a01] "+Te"(a0_1),
            [a10] "+Te"(a1_0), [a11] "+Te"(a1_1)
          : [n] "r"(in_c)
          : "q0", "q1", "q2", "lr", "memory"
        );
        /* Reset input pointers for block B; weight pointers for OCs
         * {2,3} start fresh from base. */
        x0_p = x0;
        x1_p = x1;
        const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
        const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
        __asm__ volatile (
          "wlstp.8     lr, %[n], 1f                 \n"
          "2:                                       \n"
          "  vldrb.8     q0, [%[x0]], #16           \n"
          "  vldrb.8     q2, [%[x1]], #16           \n"
          "  vldrb.8     q1, [%[w0]], #16           \n"
          "  vmladava.s8 %[a00], q0, q1             \n"
          "  vmladava.s8 %[a01], q2, q1             \n"
          "  vldrb.8     q1, [%[w1]], #16           \n"
          "  vmladava.s8 %[a10], q0, q1             \n"
          "  vmladava.s8 %[a11], q2, q1             \n"
          "  letp        lr, 2b                     \n"
          "1:                                       \n"
          : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
            [w0] "+r"(w2_p), [w1] "+r"(w3_p),
            [a00] "+Te"(a2_0), [a01] "+Te"(a2_1),
            [a10] "+Te"(a3_0), [a11] "+Te"(a3_1)
          : [n] "r"(in_c)
          : "q0", "q1", "q2", "lr", "memory"
        );
        int32x4_t mult = vld1q_s32(mults + ocb);
        int32x4_t shft = vldrbq_s32(shifts + ocb);
        int32x4_t accv0 = {a0_0, a1_0, a2_0, a3_0};
        int32x4_t accv1 = {a0_1, a1_1, a2_1, a3_1};
        accv0 = mve_requantize_per_channel_neg_shift(accv0, mult, shft);
        accv0 = vaddq_n_s32(accv0, output_offset);
        accv0 = vminq_s32(vmaxq_s32(accv0, v_act_min), v_act_max);
        vstrbq_s32(row_out + (size_t)(ow + 0) * out_c + ocb, accv0);
        accv1 = mve_requantize_per_channel_neg_shift(accv1, mult, shft);
        accv1 = vaddq_n_s32(accv1, output_offset);
        accv1 = vminq_s32(vmaxq_s32(accv1, v_act_min), v_act_max);
        vstrbq_s32(row_out + (size_t)(ow + 1) * out_c + ocb, accv1);
      }
    }
    /* Odd-out_w tail: process the final pixel with a 4-OC × 1-pixel asm
     * block.  Required for out_w in {1, 7}; without this, layers like the
     * head conv (out_w=7) fall to the slow generic path. */
    if (ow < out_w) {
      const int8_t* x0 = row_in + (size_t)ow * in_c;
      for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
        int32_t a0 = bias[ocb + 0];
        int32_t a1 = bias[ocb + 1];
        int32_t a2 = bias[ocb + 2];
        int32_t a3 = bias[ocb + 3];
        const int8_t* x_p  = x0;
        const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
        const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
        const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
        const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
        __asm__ volatile (
          "wlstp.8     lr, %[n], 1f                 \n"
          "2:                                       \n"
          "  vldrb.8     q0, [%[x]], #16            \n"
          "  vldrb.8     q1, [%[w0]], #16           \n"
          "  vmladava.s8 %[a0], q0, q1              \n"
          "  vldrb.8     q1, [%[w1]], #16           \n"
          "  vmladava.s8 %[a1], q0, q1              \n"
          "  vldrb.8     q1, [%[w2]], #16           \n"
          "  vmladava.s8 %[a2], q0, q1              \n"
          "  vldrb.8     q1, [%[w3]], #16           \n"
          "  vmladava.s8 %[a3], q0, q1              \n"
          "  letp        lr, 2b                     \n"
          "1:                                       \n"
          : [x] "+r"(x_p),
            [w0] "+r"(w0_p), [w1] "+r"(w1_p), [w2] "+r"(w2_p), [w3] "+r"(w3_p),
            [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
          : [n] "r"(in_c)
          : "q0", "q1", "lr", "memory"
        );
        int32x4_t mult = vld1q_s32(mults + ocb);
        int32x4_t shft = vldrbq_s32(shifts + ocb);
        int32x4_t accv = {a0, a1, a2, a3};
        accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
        accv = vaddq_n_s32(accv, output_offset);
        accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
        vstrbq_s32(row_out + (size_t)ow * out_c + ocb, accv);
      }
    }
  }
}
#endif

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
    /* The Phase C-tuned _stem_packed_row_mve helper does 2-pixel batched
     * inner loops with shared weight loads.  Used here for the standalone
     * stem path AND inside the stem+B0 fused kernel (Phase B), so any
     * improvement here helps both code paths automatically. */
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      _stem_packed_row_mve(
          input, output + (size_t)oh * out_row_stride, oh,
          in_h, in_w, out_w, out_c,
          p->weight_packed_32, bias, mults, shifts,
          input_offset, output_offset, act_min, act_max);
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

  /* Fastest path: 1x1 stride-1 conv, no padding.  Dispatches to a noinline
   * helper so the inner-loop inline asm gets a clean register context (no
   * surrounding always_inline live values to fight over r-reg allocation).
   * Handles odd out_w via a 4-OC × 1-pixel tail after the 2-pixel-paired
   * main loop, so out_w=7 layers (head + several block layers in the 7×7
   * stage) hit this path instead of the slower generic 4-OC path. */
  if (k_h == 1u && k_w == 1u && pad_h == 0 && pad_w == 0
      && stride_h == 1u && stride_w == 1u
      && (out_c & 3u) == 0u
      && input_offset == 0) {
    mv2_conv2d_1x1_fast(input, output, p);
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
          int32x4_t accv = {a0, a1, a2, a3};
          int32x4_t mult = vld1q_s32(mults + ocb);
          int32x4_t shft = vldrbq_s32(shifts + ocb);
          accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
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

/* ===========================================================================
 * conv2d_dwconv2d_fused_s8 — fused MV2 expand+dwconv kernel.
 *
 * Computes a 1x1 expand conv followed by a 3x3 depthwise conv without
 * materializing the full HxWxC_expand intermediate tensor.  The expand
 * output is produced one row at a time into a 3-row rolling buffer
 * (mv2_fused_scratch, sized by AOT) that the dwconv consumes immediately.
 * After dwconv row r is produced, expand row r*stride-1 can be evicted
 * and the buffer reused for row r*stride+1.
 *
 * Bit-exact correspondence: every int8 expand pixel is computed with the
 * same scalar_requantize/clamp/cast sequence as the unfused conv1x1, so
 * the rolling buffer contents are byte-identical to the corresponding
 * rows of what an unfused dwconv input would see.  See the Python derisk
 * prototype at examples/models/mv2_cortex_m_mve/docs (and the proof
 * across B1/B2 shapes) for the validation that this schedule preserves
 * outputs at max_diff=0.
 *
 * This is a scalar reference implementation — MVE optimization can layer
 * on top once correctness is established.  The primary win is the arena
 * reduction (e.g. B1 at 0.5/128 drops 196 KB peak to a 9 KB rolling
 * buffer), not raw throughput.
 * ===========================================================================
 */
extern int8_t mv2_fused_scratch[];

/* Scalar reference: bit-exact mirror of the unfused 1x1 conv+requantize chain.
 * Used as the fallback for host builds (where MVE intrinsics are unavailable)
 * and as the correctness oracle for the MVE fast path below. */
static void _fused_expand_row_scalar(
    const int8_t* in_row,
    int8_t* dst,
    const FusedConv2dDwconv2dParams* p) {
  const uint32_t in_w = p->in_w;
  const uint32_t in_c = p->in_c;
  const uint32_t expand_out_c = p->expand_out_c;
  const int32_t e_in_off = p->expand_input_offset;
  const int32_t e_out_off = p->expand_output_offset;
  const int32_t e_amin = p->expand_activation_min;
  const int32_t e_amax = p->expand_activation_max;
  for (uint32_t ow = 0; ow < in_w; ++ow) {
    const int8_t* x_pos = in_row + (size_t)ow * in_c;
    int8_t* dst_pos = dst + (size_t)ow * expand_out_c;
    for (uint32_t oc = 0; oc < expand_out_c; ++oc) {
      int32_t acc = (p->expand_bias != (const int32_t*)0) ? p->expand_bias[oc] : 0;
      const int8_t* w_oc = p->expand_weight + (size_t)oc * in_c;
      for (uint32_t ic = 0; ic < in_c; ++ic) {
        acc += ((int32_t)x_pos[ic] + e_in_off) * (int32_t)w_oc[ic];
      }
      acc = scalar_requantize(acc, p->expand_requant_mults[oc],
                              p->expand_requant_shifts[oc]);
      acc += e_out_off;
      if (acc < e_amin) acc = e_amin;
      if (acc > e_amax) acc = e_amax;
      dst_pos[oc] = (int8_t)acc;
    }
  }
}

#if MV2_USE_MVE
/* MVE fast path: 1x1 conv applied to a single input row.  Same 2-pixel x 4-OC
 * inline-asm tile as mv2_conv2d_1x1_fast above, with the outer oh loop removed
 * and Conv2dParams replaced by FusedConv2dDwconv2dParams's expand_* fields.
 * Noinline for the same reason as mv2_conv2d_1x1_fast: the asm constraint
 * solver fails when there's significant live state in the surrounding scope
 * (and the fused kernel's outer loop has plenty). */
static __attribute__((noinline))
void _fused_expand_row_mve(
    const int8_t* in_row,
    int8_t* dst,
    const FusedConv2dDwconv2dParams* p) {
  const uint32_t in_w = p->in_w;
  const uint32_t in_c = p->in_c;
  const uint32_t out_c = p->expand_out_c;
  const int32_t  output_offset = p->expand_output_offset;
  const int32_t  act_min = p->expand_activation_min;
  const int32_t  act_max = p->expand_activation_max;
  const int8_t*  weight  = p->expand_weight;
  const int32_t* bias    = p->expand_bias;
  const int32_t* mults   = p->expand_requant_mults;
  const int8_t*  shifts  = p->expand_requant_shifts;
  const size_t w_oc_stride = (size_t)in_c;
  const int32x4_t v_act_min = vdupq_n_s32(act_min);
  const int32x4_t v_act_max = vdupq_n_s32(act_max);

  uint32_t ow = 0;
  for (; ow + 1 < in_w; ow += 2) {
    const int8_t* x0 = in_row + (size_t)(ow + 0) * in_c;
    const int8_t* x1 = in_row + (size_t)(ow + 1) * in_c;
    for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
      int32_t a0_0 = bias[ocb + 0];
      int32_t a1_0 = bias[ocb + 1];
      int32_t a2_0 = bias[ocb + 2];
      int32_t a3_0 = bias[ocb + 3];
      int32_t a0_1 = a0_0, a1_1 = a1_0, a2_1 = a2_0, a3_1 = a3_0;
      const int8_t* x0_p = x0;
      const int8_t* x1_p = x1;
      const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
      const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x0]], #16           \n"
        "  vldrb.8     q2, [%[x1]], #16           \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a00], q0, q1             \n"
        "  vmladava.s8 %[a01], q2, q1             \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a10], q0, q1             \n"
        "  vmladava.s8 %[a11], q2, q1             \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
          [w0] "+r"(w0_p), [w1] "+r"(w1_p),
          [a00] "+Te"(a0_0), [a01] "+Te"(a0_1),
          [a10] "+Te"(a1_0), [a11] "+Te"(a1_1)
        : [n] "r"(in_c)
        : "q0", "q1", "q2", "lr", "memory"
      );
      x0_p = x0;
      x1_p = x1;
      const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
      const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x0]], #16           \n"
        "  vldrb.8     q2, [%[x1]], #16           \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a00], q0, q1             \n"
        "  vmladava.s8 %[a01], q2, q1             \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a10], q0, q1             \n"
        "  vmladava.s8 %[a11], q2, q1             \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
          [w0] "+r"(w2_p), [w1] "+r"(w3_p),
          [a00] "+Te"(a2_0), [a01] "+Te"(a2_1),
          [a10] "+Te"(a3_0), [a11] "+Te"(a3_1)
        : [n] "r"(in_c)
        : "q0", "q1", "q2", "lr", "memory"
      );
      int32x4_t mult = vld1q_s32(mults + ocb);
      int32x4_t shft = vldrbq_s32(shifts + ocb);
      int32x4_t accv0 = {a0_0, a1_0, a2_0, a3_0};
      int32x4_t accv1 = {a0_1, a1_1, a2_1, a3_1};
      accv0 = mve_requantize_per_channel_neg_shift(accv0, mult, shft);
      accv0 = vaddq_n_s32(accv0, output_offset);
      accv0 = vminq_s32(vmaxq_s32(accv0, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)(ow + 0) * out_c + ocb, accv0);
      accv1 = mve_requantize_per_channel_neg_shift(accv1, mult, shft);
      accv1 = vaddq_n_s32(accv1, output_offset);
      accv1 = vminq_s32(vmaxq_s32(accv1, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)(ow + 1) * out_c + ocb, accv1);
    }
  }
  /* Odd-in_w tail: 1-pixel 4-OC asm block. */
  if (ow < in_w) {
    const int8_t* x0 = in_row + (size_t)ow * in_c;
    for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
      int32_t a0 = bias[ocb + 0];
      int32_t a1 = bias[ocb + 1];
      int32_t a2 = bias[ocb + 2];
      int32_t a3 = bias[ocb + 3];
      const int8_t* x_p  = x0;
      const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
      const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
      const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
      const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x]], #16            \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a0], q0, q1              \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a1], q0, q1              \n"
        "  vldrb.8     q1, [%[w2]], #16           \n"
        "  vmladava.s8 %[a2], q0, q1              \n"
        "  vldrb.8     q1, [%[w3]], #16           \n"
        "  vmladava.s8 %[a3], q0, q1              \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x] "+r"(x_p),
          [w0] "+r"(w0_p), [w1] "+r"(w1_p), [w2] "+r"(w2_p), [w3] "+r"(w3_p),
          [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
        : [n] "r"(in_c)
        : "q0", "q1", "lr", "memory"
      );
      int32x4_t mult = vld1q_s32(mults + ocb);
      int32x4_t shft = vldrbq_s32(shifts + ocb);
      int32x4_t accv = {a0, a1, a2, a3};
      accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
      accv = vaddq_n_s32(accv, output_offset);
      accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)ow * out_c + ocb, accv);
    }
  }
}
#endif

#if MV2_USE_MVE
/* Generic single-row 1x1 conv MVE kernel.  Same 2-pixel x 4-OC inline asm
 * tile as _fused_expand_row_mve, but takes raw args so it can serve both
 * the expand-row and project-row inner loops in the inverted-residual
 * kernel.  Caller must guarantee: out_c % 4 == 0, in_c is byte-addressable
 * (any value), and input_offset has already been folded into bias. */
static __attribute__((noinline))
void _conv1x1_row_mve_args(
    const int8_t* in_row,
    int8_t* dst,
    uint32_t in_w,
    uint32_t in_c,
    uint32_t out_c,
    const int8_t* weight,
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max) {
  const size_t w_oc_stride = (size_t)in_c;
  const int32x4_t v_act_min = vdupq_n_s32(act_min);
  const int32x4_t v_act_max = vdupq_n_s32(act_max);

  uint32_t ow = 0;
  for (; ow + 1 < in_w; ow += 2) {
    const int8_t* x0 = in_row + (size_t)(ow + 0) * in_c;
    const int8_t* x1 = in_row + (size_t)(ow + 1) * in_c;
    for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
      int32_t a0_0 = bias[ocb + 0];
      int32_t a1_0 = bias[ocb + 1];
      int32_t a2_0 = bias[ocb + 2];
      int32_t a3_0 = bias[ocb + 3];
      int32_t a0_1 = a0_0, a1_1 = a1_0, a2_1 = a2_0, a3_1 = a3_0;
      const int8_t* x0_p = x0;
      const int8_t* x1_p = x1;
      const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
      const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x0]], #16           \n"
        "  vldrb.8     q2, [%[x1]], #16           \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a00], q0, q1             \n"
        "  vmladava.s8 %[a01], q2, q1             \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a10], q0, q1             \n"
        "  vmladava.s8 %[a11], q2, q1             \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
          [w0] "+r"(w0_p), [w1] "+r"(w1_p),
          [a00] "+Te"(a0_0), [a01] "+Te"(a0_1),
          [a10] "+Te"(a1_0), [a11] "+Te"(a1_1)
        : [n] "r"(in_c)
        : "q0", "q1", "q2", "lr", "memory"
      );
      x0_p = x0;
      x1_p = x1;
      const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
      const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x0]], #16           \n"
        "  vldrb.8     q2, [%[x1]], #16           \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a00], q0, q1             \n"
        "  vmladava.s8 %[a01], q2, q1             \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a10], q0, q1             \n"
        "  vmladava.s8 %[a11], q2, q1             \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x0] "+r"(x0_p), [x1] "+r"(x1_p),
          [w0] "+r"(w2_p), [w1] "+r"(w3_p),
          [a00] "+Te"(a2_0), [a01] "+Te"(a2_1),
          [a10] "+Te"(a3_0), [a11] "+Te"(a3_1)
        : [n] "r"(in_c)
        : "q0", "q1", "q2", "lr", "memory"
      );
      int32x4_t mult = vld1q_s32(mults + ocb);
      int32x4_t shft = vldrbq_s32(shifts + ocb);
      int32x4_t accv0 = {a0_0, a1_0, a2_0, a3_0};
      int32x4_t accv1 = {a0_1, a1_1, a2_1, a3_1};
      accv0 = mve_requantize_per_channel_neg_shift(accv0, mult, shft);
      accv0 = vaddq_n_s32(accv0, output_offset);
      accv0 = vminq_s32(vmaxq_s32(accv0, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)(ow + 0) * out_c + ocb, accv0);
      accv1 = mve_requantize_per_channel_neg_shift(accv1, mult, shft);
      accv1 = vaddq_n_s32(accv1, output_offset);
      accv1 = vminq_s32(vmaxq_s32(accv1, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)(ow + 1) * out_c + ocb, accv1);
    }
  }
  if (ow < in_w) {
    const int8_t* x0 = in_row + (size_t)ow * in_c;
    for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
      int32_t a0 = bias[ocb + 0];
      int32_t a1 = bias[ocb + 1];
      int32_t a2 = bias[ocb + 2];
      int32_t a3 = bias[ocb + 3];
      const int8_t* x_p  = x0;
      const int8_t* w0_p = weight + (size_t)(ocb + 0) * w_oc_stride;
      const int8_t* w1_p = weight + (size_t)(ocb + 1) * w_oc_stride;
      const int8_t* w2_p = weight + (size_t)(ocb + 2) * w_oc_stride;
      const int8_t* w3_p = weight + (size_t)(ocb + 3) * w_oc_stride;
      __asm__ volatile (
        "wlstp.8     lr, %[n], 1f                 \n"
        "2:                                       \n"
        "  vldrb.8     q0, [%[x]], #16            \n"
        "  vldrb.8     q1, [%[w0]], #16           \n"
        "  vmladava.s8 %[a0], q0, q1              \n"
        "  vldrb.8     q1, [%[w1]], #16           \n"
        "  vmladava.s8 %[a1], q0, q1              \n"
        "  vldrb.8     q1, [%[w2]], #16           \n"
        "  vmladava.s8 %[a2], q0, q1              \n"
        "  vldrb.8     q1, [%[w3]], #16           \n"
        "  vmladava.s8 %[a3], q0, q1              \n"
        "  letp        lr, 2b                     \n"
        "1:                                       \n"
        : [x] "+r"(x_p),
          [w0] "+r"(w0_p), [w1] "+r"(w1_p), [w2] "+r"(w2_p), [w3] "+r"(w3_p),
          [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
        : [n] "r"(in_c)
        : "q0", "q1", "lr", "memory"
      );
      int32x4_t mult = vld1q_s32(mults + ocb);
      int32x4_t shft = vldrbq_s32(shifts + ocb);
      int32x4_t accv = {a0, a1, a2, a3};
      accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
      accv = vaddq_n_s32(accv, output_offset);
      accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
      vstrbq_s32(dst + (size_t)ow * out_c + ocb, accv);
    }
  }
}
#endif

static int8_t* _fused_expand_row(
    const int8_t* input,
    int32_t row_idx,
    int8_t* dst,
    const FusedConv2dDwconv2dParams* p) {
  /* Compute one row of the expand 1x1 conv output at row_idx of the input.
   * If row_idx is out of bounds (top/bottom pad), fill with -dw_input_offset
   * so the dwconv's (pixel + dw_input_offset) yields 0 for the pad pixels. */
  if (row_idx < 0 || (uint32_t)row_idx >= p->in_h) {
    int8_t fill = (int8_t)(-p->dw_input_offset);
    const uint32_t row_bytes = p->in_w * p->expand_out_c;
    for (uint32_t i = 0; i < row_bytes; ++i) dst[i] = fill;
    return dst;
  }
  const int8_t* in_row = input + (size_t)row_idx * (size_t)p->in_w * p->in_c;
#if MV2_USE_MVE
  /* Fast path: expand_out_c is a multiple of 4 and expand_input_offset
   * is 0 (folded into expand_bias by the extractor).  Both invariants hold
   * for every MV2 fusion candidate by construction. */
  if ((p->expand_out_c & 3u) == 0u && p->expand_input_offset == 0
      && p->expand_bias != (const int32_t*)0) {
    _fused_expand_row_mve(in_row, dst, p);
    return dst;
  }
#endif
  _fused_expand_row_scalar(in_row, dst, p);
  return dst;
}

static __attribute__((always_inline)) inline void conv2d_dwconv2d_fused_s8(
    const int8_t* input, int8_t* output,
    const FusedConv2dDwconv2dParams* p) {
  const uint32_t in_h = p->in_h;
  const uint32_t in_w = p->in_w;
  const uint32_t expand_out_c = p->expand_out_c;
  const uint32_t out_h = p->out_h;
  const uint32_t out_w = p->out_w;
  const uint32_t stride_h = p->stride_h;
  const uint32_t stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h;
  const int32_t pad_w = (int32_t)p->pad_w;
  const int32_t d_in_off = p->dw_input_offset;
  const int32_t d_out_off = p->dw_output_offset;
  const int32_t d_amin = p->dw_activation_min;
  const int32_t d_amax = p->dw_activation_max;

  const uint32_t row_bytes = in_w * expand_out_c;
  int8_t* rolling[3] = {
    mv2_fused_scratch + 0 * row_bytes,
    mv2_fused_scratch + 1 * row_bytes,
    mv2_fused_scratch + 2 * row_bytes,
  };
  /* Sentinel = -2 means "slot is uninitialized".  -1 is a real expand
   * row index (the synthesized top-pad row), so we can't reuse it. */
  int32_t cached_row[3] = {-2, -2, -2};
  int next_slot = 0;

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    /* dwconv at row oh reads expand rows ih_base-1, ih_base, ih_base+1
     * where ih_base = oh * stride_h.  pad_h=1 makes ih_base-1 == -1 a
     * synthesized zero-row at the first output row when stride_h=1, or
     * 2*oh-1 in general; same logic for the bottom edge. */
    const int32_t ih_base = (int32_t)(oh * stride_h);
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t want = ih_base + dr - pad_h + 1;  /* effective expand row index = ih_base + dr */
      /* Map dr to the actual expand row: oh*stride_h + dr - (pad_h - 1).
       * For pad_h=1 this is just ih_base + dr - 0 = ih_base + dr;
       * we treat indices outside [0, in_h) as padding. */
      (void)want;
      int32_t row_idx = ih_base + dr;
      int found = 0;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) {
          found = 1;
          break;
        }
      }
      if (!found) {
        cached_row[next_slot] = row_idx;
        _fused_expand_row(input, row_idx, rolling[next_slot], p);
        next_slot = (next_slot + 1) % 3;
      }
    }
    /* Look up the three rows in cache order. */
    int8_t* row_at[3] = {0, 0, 0};
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) {
          row_at[dr + 1] = rolling[i];
          break;
        }
      }
    }

    int8_t* out_row = output + (size_t)oh * (size_t)out_w * expand_out_c;

#if MV2_USE_MVE
    /* MVE fast path mirrors the dwconv2d_s8 3x3 inner tiles, with the
     * critical simplification that the rolling buffer's synthesized pad
     * rows (-dw_input_offset bytes) let us *always* use
     * dw_bias_with_offset_full and skip the per-tap offset add — even at
     * top/bottom output rows where the unfused kernel has to fall back
     * to per-tap accumulation.  Math: pad pixel contribution =
     * (-dw_input_offset) * weight, which cancels the "extra" tap in
     * dw_bias_with_offset_full's pre-fold = bias + dw_input_offset *
     * sum_all_9_taps.  See derivation in docs/DESIGN.md if added.
     *
     * Horizontal boundary (ow=0 and the right tail) still uses scalar
     * per-pixel because the rolling buffer doesn't have horizontal pad
     * columns synthesized. */
    const int can_mve = (expand_out_c & 3u) == 0u
        && p->dw_bias_with_offset_full != (const int32_t*)0
        && p->kernel_h == 3u && p->kernel_w == 3u
        && (uint32_t)pad_h == 1u && (uint32_t)pad_w == 1u
        && (stride_w == 1u || stride_w == 2u);
    if (can_mve) {
      const int32x4_t v_act_min = vdupq_n_s32(d_amin);
      const int32x4_t v_act_max = vdupq_n_s32(d_amax);
      const size_t dw_w_row_stride = (size_t)3 * expand_out_c;  /* IHWO kw=3 */
      const int8_t* dw_w = p->dw_weight;
      const int32_t* d_bias_off = p->dw_bias_with_offset_full;

      uint32_t ow_mve_start, ow_mve_end;
      if (stride_w == 1u) {
        ow_mve_start = 1u;
        /* 4-pixel tile needs cols [ow_base-1 .. ow_base+4]; require ow_base+5 <= out_w */
        ow_mve_end = (out_w >= 5u) ? (out_w - 4u) : 1u;
      } else {
        ow_mve_start = 1u;
        /* 2-pixel tile at s2 needs cols [2*ow_base-1 .. 2*ow_base+3]; require 2*ow_base+3 < in_w */
        ow_mve_end = (in_w >= 4u) ? ((in_w - 3u) / 2u) : 1u;
        if (ow_mve_end > out_w) ow_mve_end = out_w;
        /* round to even step since we tile pairs of pixels */
        ow_mve_end = ow_mve_start + ((ow_mve_end - ow_mve_start) & ~1u);
      }
      if (ow_mve_end <= ow_mve_start) {
        ow_mve_end = ow_mve_start;
      }

      /* Boundary helper: MVE per-pixel, channel-vectorized.  Per cb=4,
       * per-tap pixel + dw_input_offset gets multiplied by weight, accumulated
       * across the (up to 9) in-bounds taps of the 3x3 window.  Mirrors the
       * boundary path in dwconv2d_s8 — the rolling buffer's synthesized pad
       * rows mean we don't need a kh_valid branch here either. */
      const int32x4_t v_in_off = vdupq_n_s32(d_in_off);
      #define _FUSED_BOUNDARY_PIXEL_MVE(ow_val) \
      do { \
        const uint32_t _ow = (ow_val); \
        const int32_t _iw_base = (int32_t)(_ow * stride_w); \
        for (uint32_t cb = 0; cb < expand_out_c; cb += 4) { \
          int32x4_t acc = (p->dw_bias != (const int32_t*)0) \
              ? vld1q_s32(p->dw_bias + cb) : vdupq_n_s32(0); \
          for (int dr = 0; dr < 3; ++dr) { \
            const int32_t sr = ih_base + dr - pad_h; \
            if (sr < 0 || (uint32_t)sr >= in_h) continue; \
            const int8_t* row = row_at[dr]; \
            const int8_t* w_row = dw_w + (size_t)dr * dw_w_row_stride + cb; \
            for (int dc = 0; dc < 3; ++dc) { \
              const int32_t sc = _iw_base + dc - pad_w; \
              if (sc < 0 || (uint32_t)sc >= in_w) continue; \
              int32x4_t x = vldrbq_s32(row + (size_t)sc * expand_out_c + cb); \
              int32x4_t w = vldrbq_s32(w_row + (size_t)dc * expand_out_c); \
              x = vaddq_s32(x, v_in_off); \
              acc = vaddq_s32(acc, vmulq_s32(x, w)); \
            } \
          } \
          int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb); \
          int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb); \
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft); \
          acc = vaddq_n_s32(acc, d_out_off); \
          acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max); \
          vstrbq_s32(out_row + (size_t)_ow * expand_out_c + cb, acc); \
        } \
      } while (0)

      /* Left boundary: ow=0 */
      for (uint32_t ow = 0; ow < ow_mve_start && ow < out_w; ++ow) {
        _FUSED_BOUNDARY_PIXEL_MVE(ow);
      }

      /* Interior MVE tiles */
      if (stride_w == 1u) {
        /* 4-pixel tile: pixel ow uses iw cols [ow-1, ow, ow+1].  Pair of 4
         * pixels ow_base..ow_base+3 uses cols [ow_base-1 .. ow_base+4] = 6
         * contiguous in-bounds input columns. */
        for (uint32_t ow_base = ow_mve_start; ow_base + 3 < ow_mve_end + 3 && ow_base + 4 <= out_w; ow_base += 4) {
          if (ow_base + 4 > ow_mve_end + 3) break;
          for (uint32_t cb = 0; cb < expand_out_c; cb += 4) {
            int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
            int32x4_t acc0 = bias_v;
            int32x4_t acc1 = bias_v;
            int32x4_t acc2 = bias_v;
            int32x4_t acc3 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base - 1) * expand_out_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * expand_out_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * expand_out_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * expand_out_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * expand_out_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * expand_out_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * expand_out_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * expand_out_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * expand_out_c);
              int32x4_t x5 = vldrbq_s32(x_base + 5 * expand_out_c);
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
              acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
              acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
              acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
              acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
              acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
              acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
            acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc2 = vaddq_n_s32(acc2, d_out_off);
            acc3 = vaddq_n_s32(acc3, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_act_min), v_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_act_min), v_act_max);
            acc2 = vminq_s32(vmaxq_s32(acc2, v_act_min), v_act_max);
            acc3 = vminq_s32(vmaxq_s32(acc3, v_act_min), v_act_max);
            int8_t* out_p = out_row + (size_t)ow_base * expand_out_c + cb;
            vstrbq_s32(out_p + 0 * expand_out_c, acc0);
            vstrbq_s32(out_p + 1 * expand_out_c, acc1);
            vstrbq_s32(out_p + 2 * expand_out_c, acc2);
            vstrbq_s32(out_p + 3 * expand_out_c, acc3);
          }
        }
      } else { /* stride_w == 2 */
        /* 2-pixel tile: pixel ow uses iw cols [2*ow-1, 2*ow, 2*ow+1].
         * Pair (ow_base, ow_base+1) uses 5 cols [2*ow_base-1 .. 2*ow_base+3]. */
        for (uint32_t ow_base = ow_mve_start; ow_base + 1 < ow_mve_end + 1 && ow_base + 2 <= out_w; ow_base += 2) {
          if (ow_base + 2 > ow_mve_end + 1) break;
          if ((int32_t)(ow_base * 2u + 3u) >= (int32_t)in_w) break;
          for (uint32_t cb = 0; cb < expand_out_c; cb += 4) {
            int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
            int32x4_t acc0 = bias_v;
            int32x4_t acc1 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base * 2u - 1u) * expand_out_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * expand_out_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * expand_out_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * expand_out_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * expand_out_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * expand_out_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * expand_out_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * expand_out_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * expand_out_c);
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x4, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_act_min), v_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_act_min), v_act_max);
            int8_t* out_p = out_row + (size_t)ow_base * expand_out_c + cb;
            vstrbq_s32(out_p + 0 * expand_out_c, acc0);
            vstrbq_s32(out_p + 1 * expand_out_c, acc1);
          }
        }
      }

      /* Right tail: ow values past the MVE-tile region.  Scalar. */
      uint32_t ow_tail_start;
      if (stride_w == 1u) {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 4 + 1) * 4 : 0);
        if (ow_tail_start > out_w) ow_tail_start = out_w;
      } else {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 2 + 1) * 2 : 0);
        if (ow_tail_start > out_w) ow_tail_start = out_w;
      }
      for (uint32_t ow = ow_tail_start; ow < out_w; ++ow) {
        _FUSED_BOUNDARY_PIXEL_MVE(ow);
      }
      #undef _FUSED_BOUNDARY_PIXEL_MVE
      continue;  /* skip the scalar fallback below — we handled this oh fully */
    }
#endif

    /* Scalar fallback: every output pixel via the original 9-tap inner. */
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int32_t iw_base = (int32_t)(ow * stride_w);
      for (uint32_t c = 0; c < expand_out_c; ++c) {
        int32_t acc;
        if (p->dw_bias != (const int32_t*)0) {
          acc = p->dw_bias[c];
        } else {
          acc = 0;
        }
        for (int dr = 0; dr < 3; ++dr) {
          const int32_t sr = ih_base + dr - pad_h;
          if (sr < 0 || (uint32_t)sr >= in_h) continue;
          const int8_t* row = row_at[dr];
          for (int dc = 0; dc < 3; ++dc) {
            const int32_t sc = iw_base + dc - pad_w;
            if (sc < 0 || (uint32_t)sc >= in_w) continue;
            const int32_t x = (int32_t)row[(size_t)sc * expand_out_c + c] + d_in_off;
            const int32_t w =
                (int32_t)p->dw_weight[((size_t)dr * 3 + dc) * expand_out_c + c];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, p->dw_requant_mults[c], p->dw_requant_shifts[c]);
        acc += d_out_off;
        if (acc < d_amin) acc = d_amin;
        if (acc > d_amax) acc = d_amax;
        out_row[(size_t)ow * expand_out_c + c] = (int8_t)acc;
      }
    }
  }
}


#if MV2_USE_MVE
/* Helper: compute one row of the stem 3x3 stride-2 pad-1 Cin=3 packed-32
 * convolution into `out_row`.  Mirrors the inner of the conv2d_s8
 * packed-32 fast path with the outer oh loop removed.  Caller supplies
 * the global output-row index `oh`; the helper handles its own kh
 * validity check.  Weights are pre-packed as 32 bytes per OC (27 OHWI +
 * 5 zero tail). */
static __attribute__((noinline))
void _stem_packed_row_mve(
    const int8_t* input,
    int8_t* out_row,
    uint32_t oh,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_w,
    uint32_t out_c,
    const int8_t* w_packed,
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t input_offset,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max) {
  const size_t in_row_stride = (size_t)in_w * 3;
  const int8_t pad_byte = (int8_t)(-input_offset);
  const int32x4_t v_act_min = vdupq_n_s32(act_min);
  const int32x4_t v_act_max = vdupq_n_s32(act_max);
  const int32_t ih_base = (int32_t)(oh * 2u) - 1;
  const int kh_valid[3] = {
    (ih_base + 0 >= 0 && ih_base + 0 < (int32_t)in_h) ? 1 : 0,
    (ih_base + 1 >= 0 && ih_base + 1 < (int32_t)in_h) ? 1 : 0,
    (ih_base + 2 >= 0 && ih_base + 2 < (int32_t)in_h) ? 1 : 0,
  };
  /* Helper to build one 32-byte patch (27 OHWI bytes + 5-byte zero tail). */
  #define _BUILD_PATCH(_patch, _ow)                                          \
    do {                                                                     \
      const int32_t _iw_base = (int32_t)((_ow) * 2u) - 1;                    \
      for (uint32_t kh = 0; kh < 3; ++kh) {                                  \
        int32_t ih = ih_base + (int32_t)kh;                                  \
        int8_t* p_row = (_patch) + kh * 9;                                   \
        if (kh_valid[kh]) {                                                  \
          const int8_t* x_row = input + (size_t)ih * in_row_stride;          \
          for (uint32_t kw = 0; kw < 3; ++kw) {                              \
            int32_t iw = _iw_base + (int32_t)kw;                             \
            int8_t* p_pos = p_row + kw * 3;                                  \
            if (iw >= 0 && iw < (int32_t)in_w) {                             \
              const int8_t* x_pos = x_row + (size_t)iw * 3;                  \
              p_pos[0] = x_pos[0];                                           \
              p_pos[1] = x_pos[1];                                           \
              p_pos[2] = x_pos[2];                                           \
            } else {                                                         \
              p_pos[0] = pad_byte; p_pos[1] = pad_byte; p_pos[2] = pad_byte; \
            }                                                                \
          }                                                                  \
        } else {                                                             \
          for (int i = 0; i < 9; ++i) p_row[i] = pad_byte;                   \
        }                                                                    \
      }                                                                      \
    } while (0)

  /* 2-pixel batched inner: shares weight loads across two adjacent output
   * pixels.  Each OC tile (4 OCs) does 16 vmladava (8 per pixel) on 8
   * weight loads, vs the per-pixel kernel which loads weights twice. */
  uint32_t ow = 0;
  for (; ow + 1 < out_w; ow += 2) {
    int8_t patch0[32] = {0};
    int8_t patch1[32] = {0};
    _BUILD_PATCH(patch0, ow);
    _BUILD_PATCH(patch1, ow + 1);
    int8x16_t vx0_lo = vld1q_s8(patch0 + 0);
    int8x16_t vx0_hi = vld1q_s8(patch0 + 16);
    int8x16_t vx1_lo = vld1q_s8(patch1 + 0);
    int8x16_t vx1_hi = vld1q_s8(patch1 + 16);
    for (uint32_t ocb = 0; ocb < out_c; ocb += 4) {
      int32_t a0_0 = bias[ocb + 0], a0_1 = a0_0;
      int32_t a1_0 = bias[ocb + 1], a1_1 = a1_0;
      int32_t a2_0 = bias[ocb + 2], a2_1 = a2_0;
      int32_t a3_0 = bias[ocb + 3], a3_1 = a3_0;
      int8x16_t w0_lo = vld1q_s8(w_packed + (size_t)(ocb + 0) * 32 + 0);
      int8x16_t w0_hi = vld1q_s8(w_packed + (size_t)(ocb + 0) * 32 + 16);
      int8x16_t w1_lo = vld1q_s8(w_packed + (size_t)(ocb + 1) * 32 + 0);
      int8x16_t w1_hi = vld1q_s8(w_packed + (size_t)(ocb + 1) * 32 + 16);
      int8x16_t w2_lo = vld1q_s8(w_packed + (size_t)(ocb + 2) * 32 + 0);
      int8x16_t w2_hi = vld1q_s8(w_packed + (size_t)(ocb + 2) * 32 + 16);
      int8x16_t w3_lo = vld1q_s8(w_packed + (size_t)(ocb + 3) * 32 + 0);
      int8x16_t w3_hi = vld1q_s8(w_packed + (size_t)(ocb + 3) * 32 + 16);
      a0_0 = vmladavaq_s8(a0_0, vx0_lo, w0_lo);
      a0_0 = vmladavaq_s8(a0_0, vx0_hi, w0_hi);
      a0_1 = vmladavaq_s8(a0_1, vx1_lo, w0_lo);
      a0_1 = vmladavaq_s8(a0_1, vx1_hi, w0_hi);
      a1_0 = vmladavaq_s8(a1_0, vx0_lo, w1_lo);
      a1_0 = vmladavaq_s8(a1_0, vx0_hi, w1_hi);
      a1_1 = vmladavaq_s8(a1_1, vx1_lo, w1_lo);
      a1_1 = vmladavaq_s8(a1_1, vx1_hi, w1_hi);
      a2_0 = vmladavaq_s8(a2_0, vx0_lo, w2_lo);
      a2_0 = vmladavaq_s8(a2_0, vx0_hi, w2_hi);
      a2_1 = vmladavaq_s8(a2_1, vx1_lo, w2_lo);
      a2_1 = vmladavaq_s8(a2_1, vx1_hi, w2_hi);
      a3_0 = vmladavaq_s8(a3_0, vx0_lo, w3_lo);
      a3_0 = vmladavaq_s8(a3_0, vx0_hi, w3_hi);
      a3_1 = vmladavaq_s8(a3_1, vx1_lo, w3_lo);
      a3_1 = vmladavaq_s8(a3_1, vx1_hi, w3_hi);
      int32x4_t mult = vld1q_s32(mults + ocb);
      int32x4_t shft = vldrbq_s32(shifts + ocb);
      int32x4_t accv0 = {a0_0, a1_0, a2_0, a3_0};
      int32x4_t accv1 = {a0_1, a1_1, a2_1, a3_1};
      accv0 = mve_requantize_per_channel_neg_shift(accv0, mult, shft);
      accv0 = vaddq_n_s32(accv0, output_offset);
      accv0 = vminq_s32(vmaxq_s32(accv0, v_act_min), v_act_max);
      vstrbq_s32(out_row + (size_t)(ow + 0) * out_c + ocb, accv0);
      accv1 = mve_requantize_per_channel_neg_shift(accv1, mult, shft);
      accv1 = vaddq_n_s32(accv1, output_offset);
      accv1 = vminq_s32(vmaxq_s32(accv1, v_act_min), v_act_max);
      vstrbq_s32(out_row + (size_t)(ow + 1) * out_c + ocb, accv1);
    }
  }
  /* Odd-out_w tail: single pixel via the original per-pixel pattern. */
  if (ow < out_w) {
    int8_t patch[32] = {0};
    _BUILD_PATCH(patch, ow);
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
      int32x4_t accv = {a0, a1, a2, a3};
      accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
      accv = vaddq_n_s32(accv, output_offset);
      accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
      vstrbq_s32(out_row + (size_t)ow * out_c + ocb, accv);
    }
  }
  #undef _BUILD_PATCH
}
#endif

/* Scalar reference for stem-row computation (mirrors the unfused scalar
 * conv2d_s8 path).  Used by host build + as a fallback when packed-32
 * weights aren't available. */
static void _stem_row_scalar(
    const int8_t* input,
    int8_t* out_row,
    uint32_t oh,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_w,
    uint32_t out_c,
    const int8_t* weight,  /* OHWI [out_c, 3, 3, 3] */
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t input_offset,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max) {
  const size_t in_row_stride = (size_t)in_w * 3;
  const size_t w_oc_stride = (size_t)3 * 3 * 3;
  for (uint32_t ow = 0; ow < out_w; ++ow) {
    for (uint32_t oc = 0; oc < out_c; ++oc) {
      int32_t acc = (bias != (const int32_t*)0) ? bias[oc] : 0;
      const int8_t* w_oc = weight + (size_t)oc * w_oc_stride;
      for (uint32_t kh = 0; kh < 3; ++kh) {
        const int32_t ih = (int32_t)(oh * 2u) - 1 + (int32_t)kh;
        if (ih < 0 || (uint32_t)ih >= in_h) continue;
        for (uint32_t kw = 0; kw < 3; ++kw) {
          const int32_t iw = (int32_t)(ow * 2u) - 1 + (int32_t)kw;
          if (iw < 0 || (uint32_t)iw >= in_w) continue;
          const int8_t* x = input + (size_t)ih * in_row_stride
                            + (size_t)iw * 3;
          const int8_t* w = w_oc + ((size_t)kh * 3 + kw) * 3;
          for (uint32_t ic = 0; ic < 3; ++ic) {
            acc += ((int32_t)x[ic] + input_offset) * (int32_t)w[ic];
          }
        }
      }
      acc = scalar_requantize(acc, mults[oc], shifts[oc]);
      acc += output_offset;
      if (acc < act_min) acc = act_min;
      if (acc > act_max) acc = act_max;
      out_row[(size_t)ow * out_c + oc] = (int8_t)acc;
    }
  }
}

/* ===========================================================================
 * stem_dwconv2d_conv2d_fused_s8 — MV2 first-stage chain in one call.
 *
 * Layout in mv2_fused_scratch:
 *   [-- 3 stem-output rows (rolling) --][-- 1 dwconv row --][-- 1 project row --]
 *
 * Per project output row r (== B0 dwconv output row):
 *   1. Ensure stem-output rows ih_dw_base-1, ih_dw_base, ih_dw_base+1
 *      exist in the rolling buffer; compute any missing.
 *   2. B0 dwconv reads 3 stem rows, writes 1 dwconv row to scratch.
 *   3. Project 1x1 reads dwconv row, writes 1 project row.
 *   4. Copy project row into output arena slot.
 *
 * Eliminates the stem output tensor entirely (was 32 x 112^2 = 401 KB at
 * 1.0/224) along with the dwconv output that Phase A already eliminated.
 * ===========================================================================
 */
static __attribute__((always_inline)) inline void stem_dwconv2d_conv2d_fused_s8(
    const int8_t* input, int8_t* output,
    const FusedStemDwconv2dConv2dParams* p) {
  const uint32_t in_h = p->in_h;
  const uint32_t in_w = p->in_w;
  const uint32_t stem_out_h = p->stem_out_h;
  const uint32_t stem_out_w = p->stem_out_w;
  const uint32_t stem_out_c = p->stem_out_c;
  const uint32_t out_w = p->out_w;
  const uint32_t out_h = p->out_h;
  const uint32_t project_out_c = p->project_out_c;
  const uint32_t dw_stride_h = p->dw_stride_h;
  const uint32_t dw_stride_w = p->dw_stride_w;
  const int32_t dw_pad_h = (int32_t)p->dw_pad_h;
  const int32_t dw_pad_w = (int32_t)p->dw_pad_w;
  const int32_t d_in_off = p->dw_input_offset;
  const int32_t d_out_off = p->dw_output_offset;
  const int32_t d_amin = p->dw_activation_min;
  const int32_t d_amax = p->dw_activation_max;
  const int32_t p_out_off = p->project_output_offset;
  const int32_t p_amin = p->project_activation_min;
  const int32_t p_amax = p->project_activation_max;

  const uint32_t stem_row_bytes = stem_out_w * stem_out_c;
  const uint32_t dwconv_out_w = stem_out_w / dw_stride_w;
  int8_t* rolling[3] = {
    mv2_fused_scratch + 0u * stem_row_bytes,
    mv2_fused_scratch + 1u * stem_row_bytes,
    mv2_fused_scratch + 2u * stem_row_bytes,
  };
  int8_t* dwconv_row_buf = mv2_fused_scratch + 3u * stem_row_bytes;
  int8_t* project_row_buf = dwconv_row_buf + dwconv_out_w * stem_out_c;

  int32_t cached_row[3] = {-2, -2, -2};
  int next_slot = 0;

  /* Compute stem row helper (dispatches to MVE packed-32 when available). */
  #define _COMPUTE_STEM_ROW(s, dst) do {                                     \
    if ((int32_t)(s) < 0 || (uint32_t)(s) >= stem_out_h) {                   \
      /* synthesize pad row as -dw_input_offset bytes */                     \
      int8_t fill = (int8_t)(-d_in_off);                                     \
      for (uint32_t i = 0; i < stem_row_bytes; ++i) (dst)[i] = fill;         \
    } else {                                                                 \
      MV2_STEM_ROW_DISPATCH((dst), (uint32_t)(s));                           \
    }                                                                        \
  } while (0)

  /* MVE path requires packed-32 weights + (stem_out_c % 4 == 0); falls
   * back to scalar otherwise. */
  #if MV2_USE_MVE
    #define MV2_STEM_ROW_DISPATCH(_dst, _s)                                  \
      do {                                                                   \
        if (p->stem_weight_packed_32 != (const int8_t*)0                     \
            && (stem_out_c & 3u) == 0u) {                                    \
          _stem_packed_row_mve(input, (_dst), (_s),                          \
              in_h, in_w, stem_out_w, stem_out_c,                            \
              p->stem_weight_packed_32, p->stem_bias,                        \
              p->stem_requant_mults, p->stem_requant_shifts,                 \
              p->stem_input_offset, p->stem_output_offset,                   \
              p->stem_activation_min, p->stem_activation_max);               \
        } else {                                                             \
          _stem_row_scalar(input, (_dst), (_s),                              \
              in_h, in_w, stem_out_w, stem_out_c,                            \
              p->stem_weight, p->stem_bias,                                  \
              p->stem_requant_mults, p->stem_requant_shifts,                 \
              p->stem_input_offset, p->stem_output_offset,                   \
              p->stem_activation_min, p->stem_activation_max);               \
        }                                                                    \
      } while (0)
  #else
    #define MV2_STEM_ROW_DISPATCH(_dst, _s)                                  \
      _stem_row_scalar(input, (_dst), (_s),                                  \
          in_h, in_w, stem_out_w, stem_out_c,                                \
          p->stem_weight, p->stem_bias,                                      \
          p->stem_requant_mults, p->stem_requant_shifts,                     \
          p->stem_input_offset, p->stem_output_offset,                       \
          p->stem_activation_min, p->stem_activation_max)
  #endif

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    const int32_t ih_base = (int32_t)(oh * dw_stride_h);
    /* Ensure 3 stem rows in rolling buffer (rows ih_base-1, ih_base, ih_base+1). */
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      int found = 0;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) { found = 1; break; }
      }
      if (!found) {
        cached_row[next_slot] = row_idx;
        _COMPUTE_STEM_ROW(row_idx, rolling[next_slot]);
        next_slot = (next_slot + 1) % 3;
      }
    }
    /* Look up the 3 rows in cache order. */
    int8_t* row_at[3] = {0, 0, 0};
    int kh_valid[3] = {0, 0, 0};
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) {
          row_at[dr + 1] = rolling[i];
          kh_valid[dr + 1] = (row_idx >= 0 && (uint32_t)row_idx < stem_out_h) ? 1 : 0;
          break;
        }
      }
    }

    int8_t* out_row = output + (size_t)oh * (size_t)out_w * project_out_c;

    /* Dwconv inner: 3x3 over the 3 stem rows.  Synthesized pad rows have
     * value -d_in_off so (pixel + d_in_off) = 0 → bias_with_offset_full
     * works for both real and pad rows (same trick as the existing
     * inverted-residual kernel). */
#if MV2_USE_MVE
    const int can_mve_dw = (stem_out_c & 3u) == 0u
        && p->dw_bias_with_offset_full != (const int32_t*)0
        && p->dw_kernel_h == 3u && p->dw_kernel_w == 3u
        && (uint32_t)dw_pad_h == 1u && (uint32_t)dw_pad_w == 1u
        && (dw_stride_w == 1u || dw_stride_w == 2u);
    (void)kh_valid;  /* not used in MVE path — pad rows synthesized */
    if (can_mve_dw && (project_out_c & 3u) == 0u && p->project_bias != (const int32_t*)0) {
      const int32x4_t v_dw_act_min = vdupq_n_s32(d_amin);
      const int32x4_t v_dw_act_max = vdupq_n_s32(d_amax);
      const size_t dw_w_row_stride = (size_t)3 * stem_out_c;
      const int8_t* dw_w = p->dw_weight;
      const int32_t* d_bias_off = p->dw_bias_with_offset_full;

      uint32_t ow_mve_start, ow_mve_end;
      if (dw_stride_w == 1u) {
        ow_mve_start = 1u;
        ow_mve_end = (dwconv_out_w >= 5u) ? (dwconv_out_w - 4u) : 1u;
      } else {
        ow_mve_start = 1u;
        ow_mve_end = (stem_out_w >= 4u) ? ((stem_out_w - 3u) / 2u) : 1u;
        if (ow_mve_end > dwconv_out_w) ow_mve_end = dwconv_out_w;
        ow_mve_end = ow_mve_start + ((ow_mve_end - ow_mve_start) & ~1u);
      }
      if (ow_mve_end <= ow_mve_start) ow_mve_end = ow_mve_start;

      const int32x4_t v_dw_in_off = vdupq_n_s32(d_in_off);
      #define _SDP_BOUNDARY_PIXEL(ow_val) \
      do { \
        const uint32_t _ow = (ow_val); \
        const int32_t _iw_base = (int32_t)(_ow * dw_stride_w); \
        for (uint32_t cb = 0; cb < stem_out_c; cb += 4) { \
          int32x4_t acc = (p->dw_bias != (const int32_t*)0) \
              ? vld1q_s32(p->dw_bias + cb) : vdupq_n_s32(0); \
          for (int dr = 0; dr < 3; ++dr) { \
            const int32_t sr = ih_base + dr - dw_pad_h; \
            if (sr < 0 || (uint32_t)sr >= stem_out_h) continue; \
            const int8_t* row = row_at[dr]; \
            const int8_t* w_row = dw_w + (size_t)dr * dw_w_row_stride + cb; \
            for (int dc = 0; dc < 3; ++dc) { \
              const int32_t sc = _iw_base + dc - dw_pad_w; \
              if (sc < 0 || (uint32_t)sc >= stem_out_w) continue; \
              int32x4_t x = vldrbq_s32(row + (size_t)sc * stem_out_c + cb); \
              int32x4_t w = vldrbq_s32(w_row + (size_t)dc * stem_out_c); \
              x = vaddq_s32(x, v_dw_in_off); \
              acc = vaddq_s32(acc, vmulq_s32(x, w)); \
            } \
          } \
          int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb); \
          int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb); \
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft); \
          acc = vaddq_n_s32(acc, d_out_off); \
          acc = vminq_s32(vmaxq_s32(acc, v_dw_act_min), v_dw_act_max); \
          vstrbq_s32(dwconv_row_buf + (size_t)_ow * stem_out_c + cb, acc); \
        } \
      } while (0)

      for (uint32_t ow = 0; ow < ow_mve_start && ow < dwconv_out_w; ++ow) {
        _SDP_BOUNDARY_PIXEL(ow);
      }
      if (dw_stride_w == 1u) {
        for (uint32_t ow_base = ow_mve_start;
             ow_base + 4 <= dwconv_out_w && ow_base + 4 <= ow_mve_end + 3;
             ow_base += 4) {
          for (uint32_t cb = 0; cb < stem_out_c; cb += 4) {
            int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
            int32x4_t acc0 = bias_v, acc1 = bias_v, acc2 = bias_v, acc3 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base - 1) * stem_out_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * stem_out_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * stem_out_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * stem_out_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * stem_out_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * stem_out_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * stem_out_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * stem_out_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * stem_out_c);
              int32x4_t x5 = vldrbq_s32(x_base + 5 * stem_out_c);
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
              acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
              acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
              acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
              acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
              acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
              acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
            acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc2 = vaddq_n_s32(acc2, d_out_off);
            acc3 = vaddq_n_s32(acc3, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
            acc2 = vminq_s32(vmaxq_s32(acc2, v_dw_act_min), v_dw_act_max);
            acc3 = vminq_s32(vmaxq_s32(acc3, v_dw_act_min), v_dw_act_max);
            int8_t* o = dwconv_row_buf + (size_t)ow_base * stem_out_c + cb;
            vstrbq_s32(o + 0 * stem_out_c, acc0);
            vstrbq_s32(o + 1 * stem_out_c, acc1);
            vstrbq_s32(o + 2 * stem_out_c, acc2);
            vstrbq_s32(o + 3 * stem_out_c, acc3);
          }
        }
      }
      uint32_t ow_tail_start;
      if (dw_stride_w == 1u) {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 4 + 1) * 4 : 0);
      } else {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 2 + 1) * 2 : 0);
      }
      if (ow_tail_start > dwconv_out_w) ow_tail_start = dwconv_out_w;
      for (uint32_t ow = ow_tail_start; ow < dwconv_out_w; ++ow) {
        _SDP_BOUNDARY_PIXEL(ow);
      }
      #undef _SDP_BOUNDARY_PIXEL

      _conv1x1_row_mve_args(
          dwconv_row_buf, project_row_buf,
          out_w, stem_out_c, project_out_c,
          p->project_weight, p->project_bias,
          p->project_requant_mults, p->project_requant_shifts,
          p_out_off, p_amin, p_amax);
      __builtin_memcpy(out_row, project_row_buf,
                       (size_t)out_w * project_out_c);
      continue;
    }
#endif

    /* Scalar fallback: dwconv inner + project. */
    for (uint32_t ow = 0; ow < dwconv_out_w; ++ow) {
      const int32_t iw_base = (int32_t)(ow * dw_stride_w);
      for (uint32_t c = 0; c < stem_out_c; ++c) {
        int32_t acc = (p->dw_bias != (const int32_t*)0) ? p->dw_bias[c] : 0;
        for (int dr = 0; dr < 3; ++dr) {
          const int32_t sr = ih_base + dr - dw_pad_h;
          if (sr < 0 || (uint32_t)sr >= stem_out_h) continue;
          const int8_t* row = row_at[dr];
          for (int dc = 0; dc < 3; ++dc) {
            const int32_t sc = iw_base + dc - dw_pad_w;
            if (sc < 0 || (uint32_t)sc >= stem_out_w) continue;
            const int32_t x = (int32_t)row[(size_t)sc * stem_out_c + c] + d_in_off;
            const int32_t w =
                (int32_t)p->dw_weight[((size_t)dr * 3 + dc) * stem_out_c + c];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, p->dw_requant_mults[c], p->dw_requant_shifts[c]);
        acc += d_out_off;
        if (acc < d_amin) acc = d_amin;
        if (acc > d_amax) acc = d_amax;
        dwconv_row_buf[(size_t)ow * stem_out_c + c] = (int8_t)acc;
      }
    }
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int8_t* x_pos = dwconv_row_buf + (size_t)ow * stem_out_c;
      int8_t* out_pos = out_row + (size_t)ow * project_out_c;
      for (uint32_t oc = 0; oc < project_out_c; ++oc) {
        int32_t acc = (p->project_bias != (const int32_t*)0) ? p->project_bias[oc] : 0;
        const int8_t* w_oc = p->project_weight + (size_t)oc * stem_out_c;
        for (uint32_t ic = 0; ic < stem_out_c; ++ic) {
          acc += ((int32_t)x_pos[ic] + p->project_input_offset) * (int32_t)w_oc[ic];
        }
        acc = scalar_requantize(acc, p->project_requant_mults[oc],
                                p->project_requant_shifts[oc]);
        acc += p_out_off;
        if (acc < p_amin) acc = p_amin;
        if (acc > p_amax) acc = p_amax;
        out_pos[oc] = (int8_t)acc;
      }
    }
  }
  #undef MV2_STEM_ROW_DISPATCH
  #undef _COMPUTE_STEM_ROW
}


/* Scalar quantize-row helper — used by host build and as the fallback
 * branch when MVE is unavailable. */
static void _quantize_input_row_scalar(
    const float* in_row,
    int8_t* dst,
    uint32_t in_w,
    uint32_t in_c,
    float scale,
    int32_t zero_point,
    int32_t qmin,
    int32_t qmax) {
  const float inv_scale = 1.0f / scale;
  const uint32_t n = in_w * in_c;
  for (uint32_t i = 0; i < n; ++i) {
    float scaled = in_row[i] * inv_scale;
    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    r += zero_point;
    if (r < qmin) r = qmin;
    if (r > qmax) r = qmax;
    dst[i] = (int8_t)r;
  }
}

/* Scalar stem-row from explicit row pointers (NULL = pad). */
static void _stem_row_scalar_from_rows(
    const int8_t* row_at[3],
    int8_t* out_row,
    uint32_t in_w,
    uint32_t out_w,
    uint32_t out_c,
    const int8_t* weight,  /* OHWI [out_c, 3, 3, 3] */
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t input_offset,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max) {
  const size_t w_oc_stride = (size_t)3 * 3 * 3;
  for (uint32_t ow = 0; ow < out_w; ++ow) {
    for (uint32_t oc = 0; oc < out_c; ++oc) {
      int32_t acc = (bias != (const int32_t*)0) ? bias[oc] : 0;
      const int8_t* w_oc = weight + (size_t)oc * w_oc_stride;
      for (uint32_t kh = 0; kh < 3; ++kh) {
        if (!row_at[kh]) continue;
        for (uint32_t kw = 0; kw < 3; ++kw) {
          const int32_t iw = (int32_t)(ow * 2u) - 1 + (int32_t)kw;
          if (iw < 0 || (uint32_t)iw >= in_w) continue;
          const int8_t* x = row_at[kh] + (size_t)iw * 3;
          const int8_t* w = w_oc + ((size_t)kh * 3 + kw) * 3;
          for (uint32_t ic = 0; ic < 3; ++ic) {
            acc += ((int32_t)x[ic] + input_offset) * (int32_t)w[ic];
          }
        }
      }
      acc = scalar_requantize(acc, mults[oc], shifts[oc]);
      acc += output_offset;
      if (acc < act_min) acc = act_min;
      if (acc > act_max) acc = act_max;
      out_row[(size_t)ow * out_c + oc] = (int8_t)acc;
    }
  }
}

#if MV2_USE_MVE
/* Quantize one row of float input into int8 dst using the per-tensor
 * (scale, zero_point) params.  Mirrors quantize_input's pixel loop with
 * the outer row loop removed. */
static void _quantize_input_row_mve(
    const float* in_row,
    int8_t* dst,
    uint32_t in_w,
    uint32_t in_c,
    float scale,
    int32_t zero_point,
    int32_t qmin,
    int32_t qmax) {
  const float inv_scale = 1.0f / scale;
  const uint32_t n = in_w * in_c;
  uint32_t i = 0;
  /* 4-wide vector quantize: vcvtq + vrnd + vaddq + vminq + vmaxq + vstrb */
  for (; i + 4 <= n; i += 4) {
    float32x4_t vf = vld1q_f32(in_row + i);
    vf = vmulq_n_f32(vf, inv_scale);
    int32x4_t vi = vcvtnq_s32_f32(vf);          /* round-to-nearest-even */
    vi = vaddq_n_s32(vi, zero_point);
    vi = vmaxq_s32(vi, vdupq_n_s32(qmin));
    vi = vminq_s32(vi, vdupq_n_s32(qmax));
    vstrbq_s32(dst + i, vi);
  }
  for (; i < n; ++i) {
    float scaled = in_row[i] * inv_scale;
    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    r += zero_point;
    if (r < qmin) r = qmin;
    if (r > qmax) r = qmax;
    dst[i] = (int8_t)r;
  }
}

/* Variant of _stem_packed_row_mve that takes the 3 input row pointers
 * directly (NULL = pad row).  Lets the caller supply a 3-row rolling
 * buffer of quantized input. */
static __attribute__((noinline))
void _stem_packed_row_mve_from_rows(
    const int8_t* row_at[3],
    int8_t* out_row,
    uint32_t in_w,
    uint32_t out_w,
    uint32_t out_c,
    const int8_t* w_packed,
    const int32_t* bias,
    const int32_t* mults,
    const int8_t* shifts,
    int32_t input_offset,
    int32_t output_offset,
    int32_t act_min,
    int32_t act_max) {
  (void)in_w;  /* kept for API symmetry */
  const int8_t pad_byte = (int8_t)(-input_offset);
  const int32x4_t v_act_min = vdupq_n_s32(act_min);
  const int32x4_t v_act_max = vdupq_n_s32(act_max);
  for (uint32_t ow = 0; ow < out_w; ++ow) {
    const int32_t iw_base = (int32_t)(ow * 2u) - 1;
    int8_t patch[32] = {0};
    for (uint32_t kh = 0; kh < 3; ++kh) {
      int8_t* p_row = patch + kh * 9;
      if (row_at[kh]) {
        for (uint32_t kw = 0; kw < 3; ++kw) {
          int32_t iw = iw_base + (int32_t)kw;
          int8_t* p_pos = p_row + kw * 3;
          if (iw >= 0 && iw < (int32_t)in_w) {
            const int8_t* x_pos = row_at[kh] + (size_t)iw * 3;
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
        for (int i = 0; i < 9; ++i) p_row[i] = pad_byte;
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
      int32x4_t accv = {a0, a1, a2, a3};
      accv = mve_requantize_per_channel_neg_shift(accv, mult, shft);
      accv = vaddq_n_s32(accv, output_offset);
      accv = vminq_s32(vmaxq_s32(accv, v_act_min), v_act_max);
      vstrbq_s32(out_row + (size_t)ow * out_c + ocb, accv);
    }
  }
}
#endif

/* ===========================================================================
 * quantize_stem_dwconv2d_conv2d_fused_s8 — Phase F:
 *   per-tensor quantize + stem 3x3 stride-2 + B0 dwconv + B0 project.
 *
 * Float input comes from outside arena (entry-point argument).  A 3-row
 * int8 rolling buffer holds the recently-quantized input rows; only
 * newly-needed rows are quantized on each stem output iteration.
 * Eliminates the H*W*C int8 quantized-input tensor from the arena
 * (150 KB at 1.0/224).
 *
 * Layout in mv2_fused_scratch:
 *   [-- 3 int8 input rows --][-- 3 stem-output rows --][-- 1 dwconv row --][-- 1 project row --]
 *
 * The remainder mirrors stem_dwconv2d_conv2d_fused_s8.
 * ===========================================================================
 */
static __attribute__((always_inline)) inline void quantize_stem_dwconv2d_conv2d_fused_s8(
    const float* input_float, int8_t* output,
    const FusedQuantizeStemDwconv2dConv2dParams* p) {
  const uint32_t in_h = p->in_h;
  const uint32_t in_w = p->in_w;
  const uint32_t in_c = p->in_c;
  const uint32_t stem_out_h = p->stem_out_h;
  const uint32_t stem_out_w = p->stem_out_w;
  const uint32_t stem_out_c = p->stem_out_c;
  const uint32_t out_w = p->out_w;
  const uint32_t out_h = p->out_h;
  const uint32_t project_out_c = p->project_out_c;
  const uint32_t dw_stride_h = p->dw_stride_h;
  const uint32_t dw_stride_w = p->dw_stride_w;
  const int32_t dw_pad_h = (int32_t)p->dw_pad_h;
  const int32_t dw_pad_w = (int32_t)p->dw_pad_w;
  const int32_t d_in_off = p->dw_input_offset;
  const int32_t d_out_off = p->dw_output_offset;
  const int32_t d_amin = p->dw_activation_min;
  const int32_t d_amax = p->dw_activation_max;
  const int32_t p_out_off = p->project_output_offset;
  const int32_t p_amin = p->project_activation_min;
  const int32_t p_amax = p->project_activation_max;

  const uint32_t input_row_bytes = in_w * in_c;
  const uint32_t stem_row_bytes = stem_out_w * stem_out_c;
  const uint32_t dwconv_out_w = stem_out_w / dw_stride_w;

  /* Layout: 3 int8 input rows, then 3 stem-output rows, then 1 dwconv row,
   * then 1 project row. */
  int8_t* qin_rolling[3] = {
    mv2_fused_scratch + 0u * input_row_bytes,
    mv2_fused_scratch + 1u * input_row_bytes,
    mv2_fused_scratch + 2u * input_row_bytes,
  };
  int8_t* stem_rolling[3] = {
    mv2_fused_scratch + 3u * input_row_bytes + 0u * stem_row_bytes,
    mv2_fused_scratch + 3u * input_row_bytes + 1u * stem_row_bytes,
    mv2_fused_scratch + 3u * input_row_bytes + 2u * stem_row_bytes,
  };
  int8_t* dwconv_row_buf = mv2_fused_scratch + 3u * input_row_bytes + 3u * stem_row_bytes;
  int8_t* project_row_buf = dwconv_row_buf + dwconv_out_w * stem_out_c;

  /* Quantized-input rolling buffer: track which input row each slot holds. */
  int32_t qin_cached_row[3] = {-2, -2, -2};
  int qin_next_slot = 0;

  /* Stem-output rolling buffer state (same as stem_dwconv2d kernel). */
  int32_t stem_cached_row[3] = {-2, -2, -2};
  int stem_next_slot = 0;

  /* Returns a pointer to the int8 quantized row for the given input row
   * index, quantizing if not already cached.  Returns NULL for
   * out-of-bounds rows (pad). */
  #if MV2_USE_MVE
    #define _QUANT_ROW_DISPATCH(_src, _dst) \
      _quantize_input_row_mve((_src), (_dst), in_w, in_c, \
          p->quant_scale, p->quant_zero_point, p->quant_qmin, p->quant_qmax)
  #else
    #define _QUANT_ROW_DISPATCH(_src, _dst) \
      _quantize_input_row_scalar((_src), (_dst), in_w, in_c, \
          p->quant_scale, p->quant_zero_point, p->quant_qmin, p->quant_qmax)
  #endif

  #define _ENSURE_QIN_ROW(row_idx) \
    do { \
      if ((int32_t)(row_idx) >= 0 && (uint32_t)(row_idx) < in_h) { \
        int _found = 0; \
        for (int _i = 0; _i < 3; ++_i) { \
          if (qin_cached_row[_i] == (int32_t)(row_idx)) { _found = 1; break; } \
        } \
        if (!_found) { \
          qin_cached_row[qin_next_slot] = (int32_t)(row_idx); \
          _QUANT_ROW_DISPATCH( \
              input_float + (size_t)(row_idx) * in_w * in_c, \
              qin_rolling[qin_next_slot]); \
          qin_next_slot = (qin_next_slot + 1) % 3; \
        } \
      } \
    } while (0)

  #define _GET_QIN_ROW(row_idx) ({ \
    const int8_t* _r = (const int8_t*)0; \
    if ((int32_t)(row_idx) >= 0 && (uint32_t)(row_idx) < in_h) { \
      for (int _i = 0; _i < 3; ++_i) { \
        if (qin_cached_row[_i] == (int32_t)(row_idx)) { _r = qin_rolling[_i]; break; } \
      } \
    } \
    _r; \
  })

  /* Compute one stem-output row at `stem_oh`, writing to `dst`.
   * Quantizes input rows on demand. */
  #define _COMPUTE_STEM_ROW(stem_oh, dst) \
    do { \
      if ((int32_t)(stem_oh) < 0 || (uint32_t)(stem_oh) >= stem_out_h) { \
        /* Synthesize pad row */ \
        int8_t fill = (int8_t)(-d_in_off); \
        for (uint32_t _i = 0; _i < stem_row_bytes; ++_i) (dst)[_i] = fill; \
      } else { \
        const int32_t _ih_base = (int32_t)((stem_oh) * 2u) - 1; \
        _ENSURE_QIN_ROW(_ih_base + 0); \
        _ENSURE_QIN_ROW(_ih_base + 1); \
        _ENSURE_QIN_ROW(_ih_base + 2); \
        const int8_t* _rows[3] = { \
          _GET_QIN_ROW(_ih_base + 0), \
          _GET_QIN_ROW(_ih_base + 1), \
          _GET_QIN_ROW(_ih_base + 2), \
        }; \
        _STEM_ROW_FROM_ROWS_DISPATCH(_rows, (dst)); \
      } \
    } while (0)

  #if MV2_USE_MVE
    #define _STEM_ROW_FROM_ROWS_DISPATCH(_rows, _dst) \
      do { \
        if (p->stem_weight_packed_32 != (const int8_t*)0 \
            && (stem_out_c & 3u) == 0u) { \
          _stem_packed_row_mve_from_rows( \
              (_rows), (_dst), in_w, stem_out_w, stem_out_c, \
              p->stem_weight_packed_32, p->stem_bias, \
              p->stem_requant_mults, p->stem_requant_shifts, \
              p->stem_input_offset, p->stem_output_offset, \
              p->stem_activation_min, p->stem_activation_max); \
        } else { \
          _stem_row_scalar_from_rows( \
              (_rows), (_dst), in_w, stem_out_w, stem_out_c, \
              p->stem_weight, p->stem_bias, \
              p->stem_requant_mults, p->stem_requant_shifts, \
              p->stem_input_offset, p->stem_output_offset, \
              p->stem_activation_min, p->stem_activation_max); \
        } \
      } while (0)
  #else
    #define _STEM_ROW_FROM_ROWS_DISPATCH(_rows, _dst) \
      _stem_row_scalar_from_rows( \
          (_rows), (_dst), in_w, stem_out_w, stem_out_c, \
          p->stem_weight, p->stem_bias, \
          p->stem_requant_mults, p->stem_requant_shifts, \
          p->stem_input_offset, p->stem_output_offset, \
          p->stem_activation_min, p->stem_activation_max)
  #endif

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    const int32_t ih_base = (int32_t)(oh * dw_stride_h);
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      int found = 0;
      for (int i = 0; i < 3; ++i) {
        if (stem_cached_row[i] == row_idx) { found = 1; break; }
      }
      if (!found) {
        stem_cached_row[stem_next_slot] = row_idx;
        _COMPUTE_STEM_ROW(row_idx, stem_rolling[stem_next_slot]);
        stem_next_slot = (stem_next_slot + 1) % 3;
      }
    }
    int8_t* row_at[3] = {0, 0, 0};
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      for (int i = 0; i < 3; ++i) {
        if (stem_cached_row[i] == row_idx) {
          row_at[dr + 1] = stem_rolling[i];
          break;
        }
      }
    }

    int8_t* out_row = output + (size_t)oh * (size_t)out_w * project_out_c;

#if MV2_USE_MVE
    /* Same dwconv MVE inner + project + memcpy as stem_dwconv2d_conv2d_fused_s8.
     * Duplicated here to avoid plumbing a callback through; consider
     * factoring into a helper if a fourth variant appears. */
    const int32x4_t v_dw_act_min = vdupq_n_s32(d_amin);
    const int32x4_t v_dw_act_max = vdupq_n_s32(d_amax);
    const size_t dw_w_row_stride = (size_t)3 * stem_out_c;
    const int8_t* dw_w = p->dw_weight;
    const int32_t* d_bias_off = p->dw_bias_with_offset_full;
    const int32x4_t v_dw_in_off = vdupq_n_s32(d_in_off);

    uint32_t ow_mve_start = 1u;
    uint32_t ow_mve_end;
    if (dw_stride_w == 1u) {
      ow_mve_end = (dwconv_out_w >= 5u) ? (dwconv_out_w - 4u) : 1u;
    } else {
      ow_mve_end = (stem_out_w >= 4u) ? ((stem_out_w - 3u) / 2u) : 1u;
      if (ow_mve_end > dwconv_out_w) ow_mve_end = dwconv_out_w;
      ow_mve_end = ow_mve_start + ((ow_mve_end - ow_mve_start) & ~1u);
    }
    if (ow_mve_end <= ow_mve_start) ow_mve_end = ow_mve_start;

    #define _QSDP_BOUNDARY_PIXEL(ow_val) \
    do { \
      const uint32_t _ow = (ow_val); \
      const int32_t _iw_base = (int32_t)(_ow * dw_stride_w); \
      for (uint32_t cb = 0; cb < stem_out_c; cb += 4) { \
        int32x4_t acc = (p->dw_bias != (const int32_t*)0) \
            ? vld1q_s32(p->dw_bias + cb) : vdupq_n_s32(0); \
        for (int dr = 0; dr < 3; ++dr) { \
          const int32_t sr = ih_base + dr - dw_pad_h; \
          if (sr < 0 || (uint32_t)sr >= stem_out_h) continue; \
          const int8_t* row = row_at[dr]; \
          const int8_t* w_row = dw_w + (size_t)dr * dw_w_row_stride + cb; \
          for (int dc = 0; dc < 3; ++dc) { \
            const int32_t sc = _iw_base + dc - dw_pad_w; \
            if (sc < 0 || (uint32_t)sc >= stem_out_w) continue; \
            int32x4_t x = vldrbq_s32(row + (size_t)sc * stem_out_c + cb); \
            int32x4_t w = vldrbq_s32(w_row + (size_t)dc * stem_out_c); \
            x = vaddq_s32(x, v_dw_in_off); \
            acc = vaddq_s32(acc, vmulq_s32(x, w)); \
          } \
        } \
        int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb); \
        int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb); \
        acc = mve_requantize_per_channel_neg_shift(acc, mult, shft); \
        acc = vaddq_n_s32(acc, d_out_off); \
        acc = vminq_s32(vmaxq_s32(acc, v_dw_act_min), v_dw_act_max); \
        vstrbq_s32(dwconv_row_buf + (size_t)_ow * stem_out_c + cb, acc); \
      } \
    } while (0)

    for (uint32_t ow = 0; ow < ow_mve_start && ow < dwconv_out_w; ++ow) {
      _QSDP_BOUNDARY_PIXEL(ow);
    }
    if (dw_stride_w == 1u) {
      for (uint32_t ow_base = ow_mve_start;
           ow_base + 4 <= dwconv_out_w && ow_base + 4 <= ow_mve_end + 3;
           ow_base += 4) {
        for (uint32_t cb = 0; cb < stem_out_c; cb += 4) {
          int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
          int32x4_t acc0 = bias_v, acc1 = bias_v, acc2 = bias_v, acc3 = bias_v;
          for (int dr = 0; dr < 3; ++dr) {
            const int8_t* x_base = row_at[dr]
                + (size_t)(ow_base - 1) * stem_out_c + cb;
            const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
            int32x4_t w0 = vldrbq_s32(w_base + 0 * stem_out_c);
            int32x4_t w1 = vldrbq_s32(w_base + 1 * stem_out_c);
            int32x4_t w2 = vldrbq_s32(w_base + 2 * stem_out_c);
            int32x4_t x0 = vldrbq_s32(x_base + 0 * stem_out_c);
            int32x4_t x1 = vldrbq_s32(x_base + 1 * stem_out_c);
            int32x4_t x2 = vldrbq_s32(x_base + 2 * stem_out_c);
            int32x4_t x3 = vldrbq_s32(x_base + 3 * stem_out_c);
            int32x4_t x4 = vldrbq_s32(x_base + 4 * stem_out_c);
            int32x4_t x5 = vldrbq_s32(x_base + 5 * stem_out_c);
            acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
            acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
            acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
            acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
            acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
            acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
            acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
            acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
            acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
            acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
            acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
            acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
          }
          int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
          int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
          acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
          acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
          acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
          acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
          acc0 = vaddq_n_s32(acc0, d_out_off);
          acc1 = vaddq_n_s32(acc1, d_out_off);
          acc2 = vaddq_n_s32(acc2, d_out_off);
          acc3 = vaddq_n_s32(acc3, d_out_off);
          acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
          acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
          acc2 = vminq_s32(vmaxq_s32(acc2, v_dw_act_min), v_dw_act_max);
          acc3 = vminq_s32(vmaxq_s32(acc3, v_dw_act_min), v_dw_act_max);
          int8_t* o = dwconv_row_buf + (size_t)ow_base * stem_out_c + cb;
          vstrbq_s32(o + 0 * stem_out_c, acc0);
          vstrbq_s32(o + 1 * stem_out_c, acc1);
          vstrbq_s32(o + 2 * stem_out_c, acc2);
          vstrbq_s32(o + 3 * stem_out_c, acc3);
        }
      }
    }
    uint32_t ow_tail_start;
    if (dw_stride_w == 1u) {
      ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
          ? ((ow_mve_end - ow_mve_start - 1) / 4 + 1) * 4 : 0);
    } else {
      ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
          ? ((ow_mve_end - ow_mve_start - 1) / 2 + 1) * 2 : 0);
    }
    if (ow_tail_start > dwconv_out_w) ow_tail_start = dwconv_out_w;
    for (uint32_t ow = ow_tail_start; ow < dwconv_out_w; ++ow) {
      _QSDP_BOUNDARY_PIXEL(ow);
    }
    #undef _QSDP_BOUNDARY_PIXEL

    _conv1x1_row_mve_args(
        dwconv_row_buf, project_row_buf,
        out_w, stem_out_c, project_out_c,
        p->project_weight, p->project_bias,
        p->project_requant_mults, p->project_requant_shifts,
        p_out_off, p_amin, p_amax);
    __builtin_memcpy(out_row, project_row_buf,
                     (size_t)out_w * project_out_c);
#else
    /* Scalar fallback: identical body to stem_dwconv2d's scalar path,
     * but reading from row_at[] (rolling buffer of stem rows). */
    for (uint32_t ow = 0; ow < dwconv_out_w; ++ow) {
      const int32_t iw_base = (int32_t)(ow * dw_stride_w);
      for (uint32_t c = 0; c < stem_out_c; ++c) {
        int32_t acc = (p->dw_bias != (const int32_t*)0) ? p->dw_bias[c] : 0;
        for (int dr = 0; dr < 3; ++dr) {
          const int32_t sr = ih_base + dr - dw_pad_h;
          if (sr < 0 || (uint32_t)sr >= stem_out_h) continue;
          const int8_t* row = row_at[dr];
          for (int dc = 0; dc < 3; ++dc) {
            const int32_t sc = iw_base + dc - dw_pad_w;
            if (sc < 0 || (uint32_t)sc >= stem_out_w) continue;
            const int32_t x = (int32_t)row[(size_t)sc * stem_out_c + c] + d_in_off;
            const int32_t w =
                (int32_t)p->dw_weight[((size_t)dr * 3 + dc) * stem_out_c + c];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, p->dw_requant_mults[c], p->dw_requant_shifts[c]);
        acc += d_out_off;
        if (acc < d_amin) acc = d_amin;
        if (acc > d_amax) acc = d_amax;
        dwconv_row_buf[(size_t)ow * stem_out_c + c] = (int8_t)acc;
      }
    }
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int8_t* x_pos = dwconv_row_buf + (size_t)ow * stem_out_c;
      int8_t* out_pos = out_row + (size_t)ow * project_out_c;
      for (uint32_t oc = 0; oc < project_out_c; ++oc) {
        int32_t acc = (p->project_bias != (const int32_t*)0) ? p->project_bias[oc] : 0;
        const int8_t* w_oc = p->project_weight + (size_t)oc * stem_out_c;
        for (uint32_t ic = 0; ic < stem_out_c; ++ic) {
          acc += ((int32_t)x_pos[ic] + p->project_input_offset) * (int32_t)w_oc[ic];
        }
        acc = scalar_requantize(acc, p->project_requant_mults[oc],
                                p->project_requant_shifts[oc]);
        acc += p_out_off;
        if (acc < p_amin) acc = p_amin;
        if (acc > p_amax) acc = p_amax;
        out_pos[oc] = (int8_t)acc;
      }
    }
#endif
  }
  #undef _ENSURE_QIN_ROW
  #undef _GET_QIN_ROW
  #undef _COMPUTE_STEM_ROW
  #undef _QUANT_ROW_DISPATCH
  #undef _STEM_ROW_FROM_ROWS_DISPATCH
  (void)project_row_buf;  /* unused in scalar fallback */
}


/* ===========================================================================
 * dwconv2d_conv2d_fused_s8 — MV2 B0-style block (expand_ratio=1):
 *   3x3 dwconv -> 1x1 project, no preceding expand.
 *
 * The dwconv reads directly from the (already-materialized) input tensor
 * in the arena.  Its output is produced one row at a time into a small
 * scratch buffer in mv2_fused_scratch (no rolling needed since the
 * project conv is 1x1 and consumes a single row).  The project conv
 * writes its output directly to the output arena slot.
 *
 * Memory layout in mv2_fused_scratch:
 *   [-- 1 dwconv-output row --][-- 1 project-output row --]
 *
 * The project_row_buf is allocated because _conv1x1_row_mve_args
 * re-reads its input across OC iterations and cannot safely alias
 * src/dst.  Total scratch added: out_w * (in_c + project_out_c) bytes.
 *
 * Eliminates the full HxWxC dwconv intermediate (which at MV2-1.0/r=224
 * is the 32-channel 112^2 = 401 KB tensor driving the B0 peak).  After
 * fusion, the new arena peak comes from input + project output (rather
 * than input + dwconv output), reducing 1.0/224 from 803 KB to ~600 KB.
 * ===========================================================================
 */
static __attribute__((always_inline)) inline void dwconv2d_conv2d_fused_s8(
    const int8_t* input, int8_t* output,
    const FusedDwconv2dConv2dParams* p) {
  const uint32_t in_h = p->in_h;
  const uint32_t in_w = p->in_w;
  const uint32_t in_c = p->in_c;
  const uint32_t out_h = p->out_h;
  const uint32_t out_w = p->out_w;
  const uint32_t project_out_c = p->project_out_c;
  const uint32_t stride_h = p->stride_h;
  const uint32_t stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h;
  const int32_t pad_w = (int32_t)p->pad_w;
  const int32_t d_in_off = p->dw_input_offset;
  const int32_t d_out_off = p->dw_output_offset;
  const int32_t d_amin = p->dw_activation_min;
  const int32_t d_amax = p->dw_activation_max;
  const int32_t p_out_off = p->project_output_offset;
  const int32_t p_amin = p->project_activation_min;
  const int32_t p_amax = p->project_activation_max;

  int8_t* dwconv_row_buf = mv2_fused_scratch;
  int8_t* project_row_buf = dwconv_row_buf + out_w * in_c;

  const size_t in_row_stride = (size_t)in_w * in_c;

  for (uint32_t oh = 0; oh < out_h; ++oh) {
    const int32_t ih_base = (int32_t)(oh * stride_h);
    int8_t* out_row = output + (size_t)oh * (size_t)out_w * project_out_c;

#if MV2_USE_MVE
    const int can_mve = (in_c & 3u) == 0u
        && (project_out_c & 3u) == 0u
        && p->dw_bias_with_offset_full != (const int32_t*)0
        && p->project_bias != (const int32_t*)0
        && p->kernel_h == 3u && p->kernel_w == 3u
        && (uint32_t)pad_h == 1u && (uint32_t)pad_w == 1u
        && (stride_w == 1u || stride_w == 2u);
    if (can_mve) {
      const int32x4_t v_dw_act_min = vdupq_n_s32(d_amin);
      const int32x4_t v_dw_act_max = vdupq_n_s32(d_amax);
      const int32x4_t v_dw_in_off = vdupq_n_s32(d_in_off);
      const size_t dw_w_row_stride = (size_t)3 * in_c;
      const int8_t* dw_w = p->dw_weight;
      const int32_t* d_bias_off = p->dw_bias_with_offset_full;

      /* Pointers to 3 input rows (with optional bound checks via NULL).
       * Unlike the rolling-buffer kernels, the input rows are
       * materialized contiguously in the arena and out-of-bound rows
       * are simply skipped (no -dw_input_offset synthesis). */
      const int8_t* row_at[3];
      int kh_valid[3];
      for (int dr = 0; dr < 3; ++dr) {
        int32_t sr = ih_base + dr - pad_h;
        kh_valid[dr] = (sr >= 0 && (uint32_t)sr < in_h) ? 1 : 0;
        row_at[dr] = kh_valid[dr]
            ? input + (size_t)sr * in_row_stride
            : (const int8_t*)0;
      }
      /* bias_with_offset_full assumes every tap contributes
       * dw_input_offset * weight.  When any kh row is out of bounds
       * (top/bottom edge), that assumption breaks for the skipped row's
       * 3 taps — fall back to plain bias and add the offset per tap.
       * Mirrors dwconv2d_s8's interior/boundary handling. */
      const int all_kh_valid = kh_valid[0] && kh_valid[1] && kh_valid[2];
      const int32_t* eff_bias = all_kh_valid ? d_bias_off : p->dw_bias;
      const int skip_offset_add = all_kh_valid;

      uint32_t ow_mve_start, ow_mve_end;
      if (stride_w == 1u) {
        ow_mve_start = 1u;
        ow_mve_end = (out_w >= 5u) ? (out_w - 4u) : 1u;
      } else {
        ow_mve_start = 1u;
        ow_mve_end = (in_w >= 4u) ? ((in_w - 3u) / 2u) : 1u;
        if (ow_mve_end > out_w) ow_mve_end = out_w;
        ow_mve_end = ow_mve_start + ((ow_mve_end - ow_mve_start) & ~1u);
      }
      if (ow_mve_end <= ow_mve_start) ow_mve_end = ow_mve_start;

      #define _DWP_BOUNDARY_PIXEL(ow_val) \
      do { \
        const uint32_t _ow = (ow_val); \
        const int32_t _iw_base = (int32_t)(_ow * stride_w); \
        for (uint32_t cb = 0; cb < in_c; cb += 4) { \
          int32x4_t acc = (p->dw_bias != (const int32_t*)0) \
              ? vld1q_s32(p->dw_bias + cb) : vdupq_n_s32(0); \
          for (int dr = 0; dr < 3; ++dr) { \
            if (row_at[dr] == (const int8_t*)0) continue; \
            const int8_t* row = row_at[dr]; \
            const int8_t* w_row = dw_w + (size_t)dr * dw_w_row_stride + cb; \
            for (int dc = 0; dc < 3; ++dc) { \
              const int32_t sc = _iw_base + dc - pad_w; \
              if (sc < 0 || (uint32_t)sc >= in_w) continue; \
              int32x4_t x = vldrbq_s32(row + (size_t)sc * in_c + cb); \
              int32x4_t w = vldrbq_s32(w_row + (size_t)dc * in_c); \
              x = vaddq_s32(x, v_dw_in_off); \
              acc = vaddq_s32(acc, vmulq_s32(x, w)); \
            } \
          } \
          int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb); \
          int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb); \
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft); \
          acc = vaddq_n_s32(acc, d_out_off); \
          acc = vminq_s32(vmaxq_s32(acc, v_dw_act_min), v_dw_act_max); \
          vstrbq_s32(dwconv_row_buf + (size_t)_ow * in_c + cb, acc); \
        } \
      } while (0)

      for (uint32_t ow = 0; ow < ow_mve_start && ow < out_w; ++ow) {
        _DWP_BOUNDARY_PIXEL(ow);
      }

      if (stride_w == 1u) {
        for (uint32_t ow_base = ow_mve_start;
             ow_base + 4 <= out_w && ow_base + 4 <= ow_mve_end + 3;
             ow_base += 4) {
          for (uint32_t cb = 0; cb < in_c; cb += 4) {
            int32x4_t bias_v = (eff_bias != (const int32_t*)0)
                ? vld1q_s32(eff_bias + cb) : vdupq_n_s32(0);
            int32x4_t acc0 = bias_v, acc1 = bias_v, acc2 = bias_v, acc3 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              if (row_at[dr] == (const int8_t*)0) continue;
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base - 1) * in_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * in_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * in_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * in_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * in_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * in_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * in_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * in_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * in_c);
              int32x4_t x5 = vldrbq_s32(x_base + 5 * in_c);
              if (!skip_offset_add) {
                x0 = vaddq_s32(x0, v_dw_in_off);
                x1 = vaddq_s32(x1, v_dw_in_off);
                x2 = vaddq_s32(x2, v_dw_in_off);
                x3 = vaddq_s32(x3, v_dw_in_off);
                x4 = vaddq_s32(x4, v_dw_in_off);
                x5 = vaddq_s32(x5, v_dw_in_off);
              }
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
              acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
              acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
              acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
              acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
              acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
              acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
            acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc2 = vaddq_n_s32(acc2, d_out_off);
            acc3 = vaddq_n_s32(acc3, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
            acc2 = vminq_s32(vmaxq_s32(acc2, v_dw_act_min), v_dw_act_max);
            acc3 = vminq_s32(vmaxq_s32(acc3, v_dw_act_min), v_dw_act_max);
            int8_t* o = dwconv_row_buf + (size_t)ow_base * in_c + cb;
            vstrbq_s32(o + 0 * in_c, acc0);
            vstrbq_s32(o + 1 * in_c, acc1);
            vstrbq_s32(o + 2 * in_c, acc2);
            vstrbq_s32(o + 3 * in_c, acc3);
          }
        }
      } else {
        for (uint32_t ow_base = ow_mve_start;
             ow_base + 2 <= out_w && ow_base + 2 <= ow_mve_end + 1;
             ow_base += 2) {
          if ((int32_t)(ow_base * 2u + 3u) >= (int32_t)in_w) break;
          for (uint32_t cb = 0; cb < in_c; cb += 4) {
            int32x4_t bias_v = (eff_bias != (const int32_t*)0)
                ? vld1q_s32(eff_bias + cb) : vdupq_n_s32(0);
            int32x4_t acc0 = bias_v, acc1 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              if (row_at[dr] == (const int8_t*)0) continue;
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base * 2u - 1u) * in_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * in_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * in_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * in_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * in_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * in_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * in_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * in_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * in_c);
              if (!skip_offset_add) {
                x0 = vaddq_s32(x0, v_dw_in_off);
                x1 = vaddq_s32(x1, v_dw_in_off);
                x2 = vaddq_s32(x2, v_dw_in_off);
                x3 = vaddq_s32(x3, v_dw_in_off);
                x4 = vaddq_s32(x4, v_dw_in_off);
              }
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x4, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
            int8_t* o = dwconv_row_buf + (size_t)ow_base * in_c + cb;
            vstrbq_s32(o + 0 * in_c, acc0);
            vstrbq_s32(o + 1 * in_c, acc1);
          }
        }
      }

      uint32_t ow_tail_start;
      if (stride_w == 1u) {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 4 + 1) * 4 : 0);
      } else {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 2 + 1) * 2 : 0);
      }
      if (ow_tail_start > out_w) ow_tail_start = out_w;
      for (uint32_t ow = ow_tail_start; ow < out_w; ++ow) {
        _DWP_BOUNDARY_PIXEL(ow);
      }
      #undef _DWP_BOUNDARY_PIXEL

      /* Project 1x1 on the dwconv row -> project_row_buf */
      _conv1x1_row_mve_args(
          dwconv_row_buf, project_row_buf,
          out_w, in_c, project_out_c,
          p->project_weight, p->project_bias,
          p->project_requant_mults, p->project_requant_shifts,
          p_out_off, p_amin, p_amax);

      /* Memcpy project_row_buf -> out_row */
      __builtin_memcpy(out_row, project_row_buf,
                       (size_t)out_w * project_out_c);
      continue;
    }
#endif

    /* Scalar fallback */
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int32_t iw_base = (int32_t)(ow * stride_w);
      for (uint32_t c = 0; c < in_c; ++c) {
        int32_t acc = (p->dw_bias != (const int32_t*)0) ? p->dw_bias[c] : 0;
        for (int dr = 0; dr < 3; ++dr) {
          const int32_t sr = ih_base + dr - pad_h;
          if (sr < 0 || (uint32_t)sr >= in_h) continue;
          const int8_t* row = input + (size_t)sr * in_row_stride;
          for (int dc = 0; dc < 3; ++dc) {
            const int32_t sc = iw_base + dc - pad_w;
            if (sc < 0 || (uint32_t)sc >= in_w) continue;
            const int32_t x = (int32_t)row[(size_t)sc * in_c + c] + d_in_off;
            const int32_t w =
                (int32_t)p->dw_weight[((size_t)dr * 3 + dc) * in_c + c];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, p->dw_requant_mults[c], p->dw_requant_shifts[c]);
        acc += d_out_off;
        if (acc < d_amin) acc = d_amin;
        if (acc > d_amax) acc = d_amax;
        dwconv_row_buf[(size_t)ow * in_c + c] = (int8_t)acc;
      }
    }
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int8_t* x_pos = dwconv_row_buf + (size_t)ow * in_c;
      int8_t* out_pos = out_row + (size_t)ow * project_out_c;
      for (uint32_t oc = 0; oc < project_out_c; ++oc) {
        int32_t acc = (p->project_bias != (const int32_t*)0) ? p->project_bias[oc] : 0;
        const int8_t* w_oc = p->project_weight + (size_t)oc * in_c;
        for (uint32_t ic = 0; ic < in_c; ++ic) {
          acc += ((int32_t)x_pos[ic] + p->project_input_offset) * (int32_t)w_oc[ic];
        }
        acc = scalar_requantize(acc, p->project_requant_mults[oc],
                                p->project_requant_shifts[oc]);
        acc += p_out_off;
        if (acc < p_amin) acc = p_amin;
        if (acc > p_amax) acc = p_amax;
        out_pos[oc] = (int8_t)acc;
      }
    }
  }
}


/* ===========================================================================
 * inverted_residual_fused_s8 — full MV2 inverted-residual block in one call.
 *
 * Fuses 1x1 expand + 3x3 dwconv + 1x1 project [+ optional int8 residual
 * add].  All intermediates stream row-by-row through the same
 * mv2_fused_scratch arena:
 *
 *     [---- 3 expand-output rows (rolling) ----][- 1 dwconv-output row -]
 *      0                          3*in_w*expand_out_c
 *
 * Per output row r:
 *   1. Ensure expand rows ih_base-1, ih_base, ih_base+1 are computed
 *      (synthesized as -dw_input_offset for top/bottom pad rows).
 *   2. 3x3 dwconv reads from the 3-row rolling buffer, writes one row of
 *      dwconv output into the dwconv_row scratch.
 *   3. 1x1 project reads dwconv_row, writes one row of project output
 *      directly into the output arena slot.
 *   4. If residual_present, quantized_add of project_row + skip_row in
 *      place (block input read from `residual_input` at offset r).
 *
 * Scalar reference for bit-exact validation; MVE fast paths can be added
 * later.  See conv2d_dwconv2d_fused_s8 above for the same pattern at the
 * 2-op level — most of the rolling-buffer logic is identical.
 * ===========================================================================
 */
static __attribute__((always_inline)) inline void inverted_residual_fused_s8(
    const int8_t* input,
    const int8_t* residual_input,
    int8_t* output,
    const FusedInvertedResidualParams* p) {
  const uint32_t in_h = p->in_h;
  const uint32_t in_w = p->in_w;
  const uint32_t in_c = p->in_c;
  const uint32_t expand_out_c = p->expand_out_c;
  const uint32_t dw_out_w = p->dw_out_w;
  const uint32_t dw_out_h = p->dw_out_h;
  const uint32_t out_w = p->out_w;
  const uint32_t project_out_c = p->project_out_c;
  const uint32_t stride_h = p->stride_h;
  const uint32_t stride_w = p->stride_w;
  const int32_t pad_h = (int32_t)p->pad_h;
  const int32_t pad_w = (int32_t)p->pad_w;
  const int32_t d_in_off = p->dw_input_offset;
  const int32_t d_out_off = p->dw_output_offset;
  const int32_t d_amin = p->dw_activation_min;
  const int32_t d_amax = p->dw_activation_max;
  const int32_t p_in_off = p->project_input_offset;
  const int32_t p_out_off = p->project_output_offset;
  const int32_t p_amin = p->project_activation_min;
  const int32_t p_amax = p->project_activation_max;
  const int32_t SHIFT_INT8 = 20;

  const uint32_t expand_row_bytes = in_w * expand_out_c;
  const uint32_t dwconv_row_bytes = dw_out_w * expand_out_c;
  int8_t* rolling[3] = {
    mv2_fused_scratch + 0u * expand_row_bytes,
    mv2_fused_scratch + 1u * expand_row_bytes,
    mv2_fused_scratch + 2u * expand_row_bytes,
  };
  int8_t* dwconv_row_buf = mv2_fused_scratch + 3u * expand_row_bytes;
  /* Project row scratch — separate buffer because _conv1x1_row_mve_args
   * re-reads its input across OC iterations and can't safely alias src/dst. */
  int8_t* project_row_buf = dwconv_row_buf + dwconv_row_bytes;

  /* Build a FusedConv2dDwconv2dParams-style view for the existing
   * row-helpers.  Only the expand-relevant fields are read by
   * _fused_expand_row*. */
  FusedConv2dDwconv2dParams ep = {
    .in_h = in_h, .in_w = in_w, .in_c = in_c,
    .expand_out_c = expand_out_c,
    .out_h = dw_out_h, .out_w = dw_out_w,
    .kernel_h = p->kernel_h, .kernel_w = p->kernel_w,
    .stride_h = stride_h, .stride_w = stride_w,
    .pad_h = p->pad_h, .pad_w = p->pad_w,
    .expand_weight = p->expand_weight,
    .expand_bias = p->expand_bias,
    .expand_requant_mults = p->expand_requant_mults,
    .expand_requant_shifts = p->expand_requant_shifts,
    .expand_input_offset = p->expand_input_offset,
    .expand_output_offset = p->expand_output_offset,
    .expand_activation_min = p->expand_activation_min,
    .expand_activation_max = p->expand_activation_max,
    .dw_input_offset = d_in_off,
  };

  int32_t cached_row[3] = {-2, -2, -2};
  int next_slot = 0;

  for (uint32_t oh = 0; oh < dw_out_h; ++oh) {
    /* 1. Ensure 3 expand rows in rolling buffer. */
    const int32_t ih_base = (int32_t)(oh * stride_h);
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      int found = 0;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) { found = 1; break; }
      }
      if (!found) {
        cached_row[next_slot] = row_idx;
        _fused_expand_row(input, row_idx, rolling[next_slot], &ep);
        next_slot = (next_slot + 1) % 3;
      }
    }
    int8_t* row_at[3] = {0, 0, 0};
    for (int dr = -1; dr <= 1; ++dr) {
      int32_t row_idx = ih_base + dr;
      for (int i = 0; i < 3; ++i) {
        if (cached_row[i] == row_idx) {
          row_at[dr + 1] = rolling[i];
          break;
        }
      }
    }

    /* 2. Dwconv 3x3 inner — write to dwconv_row_buf. */
    int8_t* out_row = output + (size_t)oh * (size_t)out_w * project_out_c;
    const int8_t* skip_row = (p->residual_present && residual_input != (const int8_t*)0)
        ? residual_input + (size_t)oh * (size_t)out_w * project_out_c
        : (const int8_t*)0;

#if MV2_USE_MVE
    const int can_mve = (expand_out_c & 3u) == 0u
        && (project_out_c & 3u) == 0u
        && p->dw_bias_with_offset_full != (const int32_t*)0
        && p->project_bias != (const int32_t*)0
        && p->kernel_h == 3u && p->kernel_w == 3u
        && (uint32_t)pad_h == 1u && (uint32_t)pad_w == 1u
        && (stride_w == 1u || stride_w == 2u);
    if (can_mve) {
      const int32x4_t v_dw_act_min = vdupq_n_s32(d_amin);
      const int32x4_t v_dw_act_max = vdupq_n_s32(d_amax);
      const int32x4_t v_dw_in_off = vdupq_n_s32(d_in_off);
      const size_t dw_w_row_stride = (size_t)3 * expand_out_c;
      const int8_t* dw_w = p->dw_weight;
      const int32_t* d_bias_off = p->dw_bias_with_offset_full;

      uint32_t ow_mve_start, ow_mve_end;
      if (stride_w == 1u) {
        ow_mve_start = 1u;
        ow_mve_end = (dw_out_w >= 5u) ? (dw_out_w - 4u) : 1u;
      } else {
        ow_mve_start = 1u;
        ow_mve_end = (in_w >= 4u) ? ((in_w - 3u) / 2u) : 1u;
        if (ow_mve_end > dw_out_w) ow_mve_end = dw_out_w;
        ow_mve_end = ow_mve_start + ((ow_mve_end - ow_mve_start) & ~1u);
      }
      if (ow_mve_end <= ow_mve_start) ow_mve_end = ow_mve_start;

      /* Same MVE boundary macro as in conv2d_dwconv2d_fused_s8 — produces
       * one dwconv output pixel into dwconv_row_buf, channel-vectorized. */
      #define _IRES_BOUNDARY_PIXEL(ow_val) \
      do { \
        const uint32_t _ow = (ow_val); \
        const int32_t _iw_base = (int32_t)(_ow * stride_w); \
        for (uint32_t cb = 0; cb < expand_out_c; cb += 4) { \
          int32x4_t acc = (p->dw_bias != (const int32_t*)0) \
              ? vld1q_s32(p->dw_bias + cb) : vdupq_n_s32(0); \
          for (int dr = 0; dr < 3; ++dr) { \
            const int32_t sr = ih_base + dr - pad_h; \
            if (sr < 0 || (uint32_t)sr >= in_h) continue; \
            const int8_t* row = row_at[dr]; \
            const int8_t* w_row = dw_w + (size_t)dr * dw_w_row_stride + cb; \
            for (int dc = 0; dc < 3; ++dc) { \
              const int32_t sc = _iw_base + dc - pad_w; \
              if (sc < 0 || (uint32_t)sc >= in_w) continue; \
              int32x4_t x = vldrbq_s32(row + (size_t)sc * expand_out_c + cb); \
              int32x4_t w = vldrbq_s32(w_row + (size_t)dc * expand_out_c); \
              x = vaddq_s32(x, v_dw_in_off); \
              acc = vaddq_s32(acc, vmulq_s32(x, w)); \
            } \
          } \
          int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb); \
          int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb); \
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft); \
          acc = vaddq_n_s32(acc, d_out_off); \
          acc = vminq_s32(vmaxq_s32(acc, v_dw_act_min), v_dw_act_max); \
          vstrbq_s32(dwconv_row_buf + (size_t)_ow * expand_out_c + cb, acc); \
        } \
      } while (0)

      /* Left boundary */
      for (uint32_t ow = 0; ow < ow_mve_start && ow < dw_out_w; ++ow) {
        _IRES_BOUNDARY_PIXEL(ow);
      }

      /* Interior MVE tiles */
      if (stride_w == 1u) {
        for (uint32_t ow_base = ow_mve_start;
             ow_base + 4 <= dw_out_w && ow_base + 4 <= ow_mve_end + 3;
             ow_base += 4) {
          for (uint32_t cb = 0; cb < expand_out_c; cb += 4) {
            int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
            int32x4_t acc0 = bias_v, acc1 = bias_v, acc2 = bias_v, acc3 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base - 1) * expand_out_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * expand_out_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * expand_out_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * expand_out_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * expand_out_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * expand_out_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * expand_out_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * expand_out_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * expand_out_c);
              int32x4_t x5 = vldrbq_s32(x_base + 5 * expand_out_c);
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
              acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
              acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
              acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
              acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
              acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
              acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
            acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc2 = vaddq_n_s32(acc2, d_out_off);
            acc3 = vaddq_n_s32(acc3, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
            acc2 = vminq_s32(vmaxq_s32(acc2, v_dw_act_min), v_dw_act_max);
            acc3 = vminq_s32(vmaxq_s32(acc3, v_dw_act_min), v_dw_act_max);
            int8_t* o = dwconv_row_buf + (size_t)ow_base * expand_out_c + cb;
            vstrbq_s32(o + 0 * expand_out_c, acc0);
            vstrbq_s32(o + 1 * expand_out_c, acc1);
            vstrbq_s32(o + 2 * expand_out_c, acc2);
            vstrbq_s32(o + 3 * expand_out_c, acc3);
          }
        }
      } else {
        for (uint32_t ow_base = ow_mve_start;
             ow_base + 2 <= dw_out_w && ow_base + 2 <= ow_mve_end + 1;
             ow_base += 2) {
          if ((int32_t)(ow_base * 2u + 3u) >= (int32_t)in_w) break;
          for (uint32_t cb = 0; cb < expand_out_c; cb += 4) {
            int32x4_t bias_v = vld1q_s32(d_bias_off + cb);
            int32x4_t acc0 = bias_v, acc1 = bias_v;
            for (int dr = 0; dr < 3; ++dr) {
              const int8_t* x_base = row_at[dr]
                  + (size_t)(ow_base * 2u - 1u) * expand_out_c + cb;
              const int8_t* w_base = dw_w + (size_t)dr * dw_w_row_stride + cb;
              int32x4_t w0 = vldrbq_s32(w_base + 0 * expand_out_c);
              int32x4_t w1 = vldrbq_s32(w_base + 1 * expand_out_c);
              int32x4_t w2 = vldrbq_s32(w_base + 2 * expand_out_c);
              int32x4_t x0 = vldrbq_s32(x_base + 0 * expand_out_c);
              int32x4_t x1 = vldrbq_s32(x_base + 1 * expand_out_c);
              int32x4_t x2 = vldrbq_s32(x_base + 2 * expand_out_c);
              int32x4_t x3 = vldrbq_s32(x_base + 3 * expand_out_c);
              int32x4_t x4 = vldrbq_s32(x_base + 4 * expand_out_c);
              acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
              acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
              acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
              acc1 = vaddq_s32(acc1, vmulq_s32(x2, w0));
              acc1 = vaddq_s32(acc1, vmulq_s32(x3, w1));
              acc1 = vaddq_s32(acc1, vmulq_s32(x4, w2));
            }
            int32x4_t mult = vld1q_s32(p->dw_requant_mults + cb);
            int32x4_t shft = vldrbq_s32(p->dw_requant_shifts + cb);
            acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
            acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
            acc0 = vaddq_n_s32(acc0, d_out_off);
            acc1 = vaddq_n_s32(acc1, d_out_off);
            acc0 = vminq_s32(vmaxq_s32(acc0, v_dw_act_min), v_dw_act_max);
            acc1 = vminq_s32(vmaxq_s32(acc1, v_dw_act_min), v_dw_act_max);
            int8_t* o = dwconv_row_buf + (size_t)ow_base * expand_out_c + cb;
            vstrbq_s32(o + 0 * expand_out_c, acc0);
            vstrbq_s32(o + 1 * expand_out_c, acc1);
          }
        }
      }

      /* Right tail */
      uint32_t ow_tail_start;
      if (stride_w == 1u) {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 4 + 1) * 4 : 0);
      } else {
        ow_tail_start = ow_mve_start + ((ow_mve_end > ow_mve_start)
            ? ((ow_mve_end - ow_mve_start - 1) / 2 + 1) * 2 : 0);
      }
      if (ow_tail_start > dw_out_w) ow_tail_start = dw_out_w;
      for (uint32_t ow = ow_tail_start; ow < dw_out_w; ++ow) {
        _IRES_BOUNDARY_PIXEL(ow);
      }
      #undef _IRES_BOUNDARY_PIXEL

      /* 3a. Project 1x1 MVE — dwconv_row_buf -> project_dst.
       * Writes to project_row_buf when residual is present (so the residual
       * add below has a clean source), or directly to out_row otherwise. */
      int8_t* project_dst = (skip_row != (const int8_t*)0) ? project_row_buf : out_row;
      _conv1x1_row_mve_args(
          dwconv_row_buf, project_dst,
          dw_out_w, expand_out_c, project_out_c,
          p->project_weight, p->project_bias,
          p->project_requant_mults, p->project_requant_shifts,
          p_out_off, p_amin, p_amax);

      /* 3b. MVE residual add — combine project_dst + skip_row -> out_row. */
      if (skip_row != (const int8_t*)0) {
        const uint32_t row_n = out_w * project_out_c;
        const int32x4_t v_s_zp = vdupq_n_s32(p->residual_self_zero_point);
        const int32x4_t v_o_zp = vdupq_n_s32(p->residual_other_zero_point);
        const int32x4_t v_s_m = vdupq_n_s32(p->residual_self_multiplier);
        const int32x4_t v_s_s = vdupq_n_s32(p->residual_self_shift);
        const int32x4_t v_o_m = vdupq_n_s32(p->residual_other_multiplier);
        const int32x4_t v_o_s = vdupq_n_s32(p->residual_other_shift);
        const int32x4_t v_r_m = vdupq_n_s32(p->residual_output_multiplier);
        const int32x4_t v_r_s = vdupq_n_s32(p->residual_output_shift);
        const int32x4_t v_r_min = vdupq_n_s32(p->residual_activation_min);
        const int32x4_t v_r_max = vdupq_n_s32(p->residual_activation_max);
        uint32_t i = 0;
        for (; i + 4 <= row_n; i += 4) {
          int32x4_t av = vldrbq_s32(project_dst + i);
          int32x4_t bv = vldrbq_s32(skip_row + i);
          av = vsubq_s32(av, v_s_zp);
          bv = vsubq_s32(bv, v_o_zp);
          av = vshlq_n_s32(av, SHIFT_INT8);
          bv = vshlq_n_s32(bv, SHIFT_INT8);
          int32x4_t a_fp = mve_requantize_per_channel_neg_shift(av, v_s_m, v_s_s);
          int32x4_t b_fp = mve_requantize_per_channel_neg_shift(bv, v_o_m, v_o_s);
          int32x4_t sum_fp = vaddq_s32(a_fp, b_fp);
          int32x4_t r = mve_requantize_per_channel_neg_shift(sum_fp, v_r_m, v_r_s);
          r = vaddq_n_s32(r, p->residual_output_zero_point);
          r = vmaxq_s32(r, v_r_min);
          r = vminq_s32(r, v_r_max);
          vstrbq_s32(out_row + i, r);
        }
        for (; i < row_n; ++i) {
          int32_t sl = ((int32_t)project_dst[i] - p->residual_self_zero_point) << SHIFT_INT8;
          int32_t ol = ((int32_t)skip_row[i] - p->residual_other_zero_point) << SHIFT_INT8;
          int32_t sf = scalar_requantize(sl, p->residual_self_multiplier, p->residual_self_shift);
          int32_t of = scalar_requantize(ol, p->residual_other_multiplier, p->residual_other_shift);
          int32_t r = scalar_requantize(sf + of, p->residual_output_multiplier, p->residual_output_shift);
          r += p->residual_output_zero_point;
          if (r < p->residual_activation_min) r = p->residual_activation_min;
          if (r > p->residual_activation_max) r = p->residual_activation_max;
          out_row[i] = (int8_t)r;
        }
      }
      continue;  /* MVE path handled this oh fully */
    }
#endif

    /* Scalar fallback */
    for (uint32_t ow = 0; ow < dw_out_w; ++ow) {
      const int32_t iw_base = (int32_t)(ow * stride_w);
      for (uint32_t c = 0; c < expand_out_c; ++c) {
        int32_t acc = (p->dw_bias != (const int32_t*)0) ? p->dw_bias[c] : 0;
        for (int dr = 0; dr < 3; ++dr) {
          const int32_t sr = ih_base + dr - pad_h;
          if (sr < 0 || (uint32_t)sr >= in_h) continue;
          const int8_t* row = row_at[dr];
          for (int dc = 0; dc < 3; ++dc) {
            const int32_t sc = iw_base + dc - pad_w;
            if (sc < 0 || (uint32_t)sc >= in_w) continue;
            const int32_t x = (int32_t)row[(size_t)sc * expand_out_c + c] + d_in_off;
            const int32_t w =
                (int32_t)p->dw_weight[((size_t)dr * 3 + dc) * expand_out_c + c];
            acc += x * w;
          }
        }
        acc = scalar_requantize(acc, p->dw_requant_mults[c], p->dw_requant_shifts[c]);
        acc += d_out_off;
        if (acc < d_amin) acc = d_amin;
        if (acc > d_amax) acc = d_amax;
        dwconv_row_buf[(size_t)ow * expand_out_c + c] = (int8_t)acc;
      }
    }
    for (uint32_t ow = 0; ow < out_w; ++ow) {
      const int8_t* x_pos = dwconv_row_buf + (size_t)ow * expand_out_c;
      int8_t* out_pos = out_row + (size_t)ow * project_out_c;
      const int8_t* skip_pos = (skip_row != (const int8_t*)0)
          ? skip_row + (size_t)ow * project_out_c : (const int8_t*)0;
      for (uint32_t oc = 0; oc < project_out_c; ++oc) {
        int32_t acc = (p->project_bias != (const int32_t*)0) ? p->project_bias[oc] : 0;
        const int8_t* w_oc = p->project_weight + (size_t)oc * expand_out_c;
        for (uint32_t ic = 0; ic < expand_out_c; ++ic) {
          acc += ((int32_t)x_pos[ic] + p_in_off) * (int32_t)w_oc[ic];
        }
        acc = scalar_requantize(acc, p->project_requant_mults[oc],
                                p->project_requant_shifts[oc]);
        acc += p_out_off;
        if (acc < p_amin) acc = p_amin;
        if (acc > p_amax) acc = p_amax;
        int8_t project_val = (int8_t)acc;
        if (skip_pos != (const int8_t*)0) {
          int32_t self_lifted = ((int32_t)project_val - p->residual_self_zero_point)
              << SHIFT_INT8;
          int32_t other_lifted = ((int32_t)skip_pos[oc] - p->residual_other_zero_point)
              << SHIFT_INT8;
          int32_t self_fp = scalar_requantize(
              self_lifted, p->residual_self_multiplier, p->residual_self_shift);
          int32_t other_fp = scalar_requantize(
              other_lifted, p->residual_other_multiplier, p->residual_other_shift);
          int32_t sum_fp = self_fp + other_fp;
          int32_t r = scalar_requantize(
              sum_fp, p->residual_output_multiplier, p->residual_output_shift);
          r += p->residual_output_zero_point;
          if (r < p->residual_activation_min) r = p->residual_activation_min;
          if (r > p->residual_activation_max) r = p->residual_activation_max;
          project_val = (int8_t)r;
        }
        out_pos[oc] = project_val;
      }
    }
  }
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
        acc = mve_requantize_per_channel_neg_shift(acc, mult, shft);
        acc = vaddq_n_s32(acc, output_offset);
        acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
        vstrbq_s32(output + (size_t)oh * out_row_stride + cb, acc);
      }

      /* Interior 4-pixel tiles: ow_base in [1, out_w-5].  Each tile reads
       * 6 contiguous input columns iw in [ow_base-1 .. ow_base+4] which are
       * all in-bounds.  When the kh dimension is also fully in-bounds
       * (middle rows: 1 <= oh <= out_h-2), all 9 kernel positions
       * contribute, so we can use the AOT-folded bias_with_offset_full
       * and skip the per-tap vaddq_s32(x, v_in_off) entirely. */
      const int32_t* bias_ptr_interior =
          (kh_valid[0] && kh_valid[1] && kh_valid[2]
           && p->bias_with_offset_full != (const int32_t*)0)
          ? p->bias_with_offset_full : bias;
      const int skip_offset_add =
          (bias_ptr_interior == p->bias_with_offset_full);
      uint32_t ow_base = 1;
      for (; ow_base + 5 <= out_w; ow_base += 4) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t bias_v = (bias_ptr_interior != (const int32_t*)0)
              ? vld1q_s32(bias_ptr_interior + cb) : vdupq_n_s32(0);
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
            int32x4_t x0 = vldrbq_s32(x_base + 0 * in_c);
            int32x4_t x1 = vldrbq_s32(x_base + 1 * in_c);
            int32x4_t x2 = vldrbq_s32(x_base + 2 * in_c);
            if (!skip_offset_add) {
              x0 = vaddq_s32(x0, v_in_off);
              x1 = vaddq_s32(x1, v_in_off);
              x2 = vaddq_s32(x2, v_in_off);
            }
            acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
            acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
            acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
            int32x4_t x3 = vldrbq_s32(x_base + 3 * in_c);
            if (!skip_offset_add) x3 = vaddq_s32(x3, v_in_off);
            acc1 = vaddq_s32(acc1, vmulq_s32(x1, w0));
            acc1 = vaddq_s32(acc1, vmulq_s32(x2, w1));
            acc1 = vaddq_s32(acc1, vmulq_s32(x3, w2));
            int32x4_t x4 = vldrbq_s32(x_base + 4 * in_c);
            if (!skip_offset_add) x4 = vaddq_s32(x4, v_in_off);
            acc2 = vaddq_s32(acc2, vmulq_s32(x2, w0));
            acc2 = vaddq_s32(acc2, vmulq_s32(x3, w1));
            acc2 = vaddq_s32(acc2, vmulq_s32(x4, w2));
            int32x4_t x5 = vldrbq_s32(x_base + 5 * in_c);
            if (!skip_offset_add) x5 = vaddq_s32(x5, v_in_off);
            acc3 = vaddq_s32(acc3, vmulq_s32(x3, w0));
            acc3 = vaddq_s32(acc3, vmulq_s32(x4, w1));
            acc3 = vaddq_s32(acc3, vmulq_s32(x5, w2));
          }

          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
          acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
          acc2 = mve_requantize_per_channel_neg_shift(acc2, mult, shft);
          acc3 = mve_requantize_per_channel_neg_shift(acc3, mult, shft);
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
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft);
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
        acc = mve_requantize_per_channel_neg_shift(acc, mult, shft);
        acc = vaddq_n_s32(acc, output_offset);
        acc = vminq_s32(vmaxq_s32(acc, v_act_min), v_act_max);
        vstrbq_s32(output + (size_t)oh * out_row_stride + cb, acc);
      }

      /* Interior 2-pixel tiles.  Pixel ow uses iw cols [2*ow-1, 2*ow, 2*ow+1].
       * Pair (ow_base, ow_base+1) uses cols [2*ow_base-1 .. 2*ow_base+3] (5
       * cols).  All in-bounds when 2*ow_base-1 >= 0 AND 2*ow_base+3 < in_w.
       * When all 3 kh rows are also valid, swap in bias_with_offset_full to
       * skip per-tap input_offset adds. */
      const int32_t* s2_bias_ptr =
          (kh_valid[0] && kh_valid[1] && kh_valid[2]
           && p->bias_with_offset_full != (const int32_t*)0)
          ? p->bias_with_offset_full : bias;
      const int s2_skip_off = (s2_bias_ptr == p->bias_with_offset_full);
      uint32_t ow_base = 1;
      for (; ow_base + 2 <= out_w && (int32_t)(ow_base * 2u + 3) < (int32_t)in_w;
           ow_base += 2) {
        for (uint32_t cb = 0; cb < out_c; cb += 4) {
          int32x4_t bias_v = (s2_bias_ptr != (const int32_t*)0)
              ? vld1q_s32(s2_bias_ptr + cb) : vdupq_n_s32(0);
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
            int32x4_t x0 = vldrbq_s32(x_base + 0 * in_c);
            int32x4_t x1 = vldrbq_s32(x_base + 1 * in_c);
            int32x4_t x2 = vldrbq_s32(x_base + 2 * in_c);
            if (!s2_skip_off) {
              x0 = vaddq_s32(x0, v_in_off);
              x1 = vaddq_s32(x1, v_in_off);
              x2 = vaddq_s32(x2, v_in_off);
            }
            acc0 = vaddq_s32(acc0, vmulq_s32(x0, w0));
            acc0 = vaddq_s32(acc0, vmulq_s32(x1, w1));
            acc0 = vaddq_s32(acc0, vmulq_s32(x2, w2));
            int32x4_t x3 = vldrbq_s32(x_base + 3 * in_c);
            int32x4_t x4 = vldrbq_s32(x_base + 4 * in_c);
            if (!s2_skip_off) {
              x3 = vaddq_s32(x3, v_in_off);
              x4 = vaddq_s32(x4, v_in_off);
            }
            acc1 = vaddq_s32(acc1, vmulq_s32(x2, w0));
            acc1 = vaddq_s32(acc1, vmulq_s32(x3, w1));
            acc1 = vaddq_s32(acc1, vmulq_s32(x4, w2));
          }

          int32x4_t mult = vld1q_s32(mults + cb);
          int32x4_t shft = vldrbq_s32(shifts + cb);
          acc0 = mve_requantize_per_channel_neg_shift(acc0, mult, shft);
          acc1 = mve_requantize_per_channel_neg_shift(acc1, mult, shft);
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
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft);
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
          acc = mve_requantize_per_channel_neg_shift(acc, mult, shft);
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
    int32x4_t a_fp = mve_requantize_per_channel_neg_shift(av, v_self_mult, v_self_shift);
    int32x4_t b_fp = mve_requantize_per_channel_neg_shift(bv, v_other_mult, v_other_shift);
    int32x4_t sum_fp = vaddq_s32(a_fp, b_fp);
    int32x4_t result = mve_requantize_per_channel_neg_shift(sum_fp, v_out_mult, v_out_shift);
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
