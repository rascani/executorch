/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * MVE helpers for the standalone MobileNetV2 inference path.
 *
 * mve_requantize_per_channel matches backends/cortex_m/passes/passes_utils.py:103
 * requantize_cmsis bit-exactly. That function uses round-half-away-from-zero, which
 * differs from vrshlq_s32's round-half-up, so the right-shift step is implemented
 * manually using a sign-mask threshold bump.
 */

#ifndef MV2_MVE_HELPERS_H_
#define MV2_MVE_HELPERS_H_

#include <stdint.h>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define MV2_HAVE_MVE 1
#else
#define MV2_HAVE_MVE 0
#endif

#if MV2_HAVE_MVE
static inline int32x4_t mve_requantize_per_channel(
    int32x4_t acc, int32x4_t multiplier, int32x4_t shift) {
  int32x4_t zero = vdupq_n_s32(0);
  int32x4_t one = vdupq_n_s32(1);
  int32x4_t left = vmaxq_s32(shift, zero);
  int32x4_t right = vminq_s32(shift, zero);

  int32x4_t shifted = vshlq_s32(acc, left);
  int32x4_t product = vqrdmulhq_s32(shifted, multiplier);

  int32x4_t neg_right = vnegq_s32(right);
  int32x4_t mask = vsubq_s32(vshlq_s32(one, neg_right), one);
  int32x4_t remainder = vandq_s32(product, mask);
  int32x4_t shifted_down = vshlq_s32(product, right);
  int32x4_t threshold = vshrq_n_s32(mask, 1);
  int32x4_t neg_sign_mask = vshrq_n_s32(product, 31);
  threshold = vsubq_s32(threshold, neg_sign_mask);
  mve_pred16_t p_bump = vcmpgtq_s32(remainder, threshold);
  int32x4_t bump = vpselq_s32(vdupq_n_s32(-1), zero, p_bump);
  return vsubq_s32(shifted_down, bump);
}
#endif /* MV2_HAVE_MVE */

static inline int32_t scalar_requantize(int32_t acc, int32_t multiplier, int32_t shift) {
  int64_t value = (int64_t)acc << (shift > 0 ? shift : 0);
  int64_t product = value * (int64_t)multiplier + (1LL << 30);
  int32_t result = (int32_t)(product >> 31);
  if (shift < 0) {
    int32_t right = -shift;
    int32_t mask = (1 << right) - 1;
    int32_t remainder = result & mask;
    int32_t shifted_down = result >> right;
    int32_t threshold = mask >> 1;
    if (result < 0) {
      threshold += 1;
    }
    if (remainder > threshold) {
      shifted_down += 1;
    }
    result = shifted_down;
  }
  return result;
}

#endif
