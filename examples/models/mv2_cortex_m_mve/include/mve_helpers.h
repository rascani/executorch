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
/* Bit-exact match to requantize_cmsis in
 * backends/cortex_m/passes/passes_utils.py:103, expressed in the same
 * style as CMSIS-NN's `arm_requantize_mve` (~half the MVE ops of the
 * naive threshold-bump impl we used before).
 *
 * The trick: `vrshlq_s32` with a negative shift right-shifts with
 * round-half-up.  For positive dividends that matches round-half-
 * away-from-zero already.  For negative dividends we pre-subtract 1
 * (`fixup = -1 if dividend < 0`) so the half-up rounding lands at the
 * away-from-zero side after the right shift.
 *
 * Verified: max int8 diff = 0 vs the previous helper across the full
 * MobileNetV2 inference. */
static inline int32x4_t mve_requantize_per_channel(
    int32x4_t acc, int32x4_t multiplier, int32x4_t shift) {
  const int32x4_t zero = vdupq_n_s32(0);
  int32x4_t left  = vmaxq_s32(shift, zero);
  int32x4_t right = vminq_s32(shift, zero); /* <= 0 */

  int32x4_t shifted = vshlq_s32(acc, left);
  int32x4_t product = vqrdmulhq_s32(shifted, multiplier);

  /* fixup = -1 when product is negative (and right < 0, i.e. there is
   * a right-shift step to apply).  Reuses `right` itself as the mask
   * source: when right == 0 the bitwise AND collapses to zero and the
   * fixup is a no-op; when right < 0 (two's complement = 0xFFFFFFFe…)
   * masking with it preserves the sign bit of `product`. */
  int32x4_t fixup = vshrq_n_s32(vandq_s32(product, right), 31);
  int32x4_t fixed = vqaddq_s32(product, fixup);
  return vrshlq_s32(fixed, right);
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
