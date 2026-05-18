/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Hand-written standalone kernels for KWT-1 (Keyword Spotting
 * Transformer) on Cortex-M55 + Helium MVE.  Phased per docs/PLAN.md.
 *
 * Phase 1: layer_norm_s8 — int8 in / int8 out / float32 internal LN.
 *          Scalar reference + MVE fast path (when MV2_USE_MVE is set
 *          and embed_dim % 4 == 0).
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#define KWT_1_USE_MVE 1
#include <arm_mve.h>
#else
#define KWT_1_USE_MVE 0
#endif

#include "kwt_1_layer_params.h"
#include "kwt_1_kernels.h"


/* Bit-exact match to backends/cortex_m/passes/passes_utils.py
 * `requantize_cmsis`: tie-away-from-zero on the final right shift, as
 * used by CMSIS-NN's `arm_nn_requantize`.  Shared by BMM, fused
 * attention, and any other kernel that requantizes an int32
 * accumulator. */
static inline int32_t kwt_1_requantize(
    int32_t acc, int32_t multiplier, int32_t shift) {
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


/* Per-row int8 → float dequant → torch-style two-pass LayerNorm → γβ
 * affine → int8 requant.  Scalar reference; the MVE fast path lives
 * below this in the same file.
 *
 * Math (matches torch.nn.functional.layer_norm on float inputs):
 *   x = (q - in_zp) * in_scale                          (dequant)
 *   mean = (sum_c x) / D
 *   var  = (sum_c (x - mean)^2) / D                     (biased, /N not /(N-1))
 *   rstd = 1 / sqrt(var + eps)
 *   y    = (x - mean) * rstd * gamma[c] + beta[c]
 *   out  = clamp(round(y / out_scale) + out_zp, -128, 127)
 */
static void layer_norm_s8_scalar(
    const int8_t* input, int8_t* output, const LayerNormParams* p) {
  const uint32_t D = p->embed_dim;
  const float in_scale = p->input_scale;
  const float in_zp = (float)p->input_zp;
  const float inv_out_scale = 1.0f / p->output_scale;
  const float out_zp_f = (float)p->output_zp;
  const float eps = p->eps;

  for (uint32_t r = 0; r < p->num_rows; ++r) {
    const int8_t* row_in = input + (size_t)r * D;
    int8_t* row_out = output + (size_t)r * D;

    /* Pass 1: mean. */
    float sum = 0.0f;
    for (uint32_t c = 0; c < D; ++c) {
      sum += ((float)row_in[c] - in_zp) * in_scale;
    }
    const float mean = sum / (float)D;

    /* Pass 2: variance.  Two-pass keeps the rounding behaviour
     * aligned with torch's CPU LN impl, which computes the sum of
     * squared deviations from the *computed* mean. */
    float sumsq = 0.0f;
    for (uint32_t c = 0; c < D; ++c) {
      float x = ((float)row_in[c] - in_zp) * in_scale;
      float d = x - mean;
      sumsq += d * d;
    }
    const float var = sumsq / (float)D;
    const float rstd = 1.0f / sqrtf(var + eps);

    /* Pass 3: write out. */
    for (uint32_t c = 0; c < D; ++c) {
      float x = ((float)row_in[c] - in_zp) * in_scale;
      float n = (x - mean) * rstd;
      float g = (p->gamma != (const float*)0) ? p->gamma[c] : 1.0f;
      float b = (p->beta  != (const float*)0) ? p->beta[c]  : 0.0f;
      float y = n * g + b;
      float qf = y * inv_out_scale + out_zp_f;
      int32_t q = (int32_t)(qf + (qf >= 0.0f ? 0.5f : -0.5f));
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      row_out[c] = (int8_t)q;
    }
  }
}


/* Public entry point — dispatches scalar vs MVE based on shape and
 * build flags.  Phase 1 ships scalar only; MVE variant follows once
 * the scalar path is bit-exact end-to-end. */
void layer_norm_s8(
    const int8_t* input, int8_t* output, const LayerNormParams* p) {
  layer_norm_s8_scalar(input, output, p);
}


/* Phase 2: int8 → int8 GELU via 256-byte LUT.  The LUT bakes in the
 * input/output quant params and the approximate form; runtime cost
 * per element is one int8 table lookup.  Indexed as
 *   output[i] = lut[(uint8_t)(input[i] + 128)]
 * — adding 128 turns the signed int8 input into an unsigned index in
 * [0, 256).
 *
 * Scalar implementation is bit-exact and fast enough for KWT-1
 * (~75K GELU evaluations per inference; under 1% of cycles).  An MVE
 * `vldrbq_gather_offset_s8` 16-wide gather would speed it up further
 * but adds modest complexity for marginal gain — deferred to the
 * tuning phase.
 */
void gelu_lut_s8(
    const int8_t* input, int8_t* output, const GELUParams* p) {
  const int8_t* lut = p->lut;
  const uint32_t n = p->num_elements;
  for (uint32_t i = 0; i < n; ++i) {
    output[i] = lut[(uint8_t)(input[i] + (int8_t)0x80)];
  }
}


/* Phase 3: int8 batched matmul.  Computes
 *     output[b, m, n] = requantize(
 *         sum_k (lhs[b,m,k] + lhs_offset) * (rhs_transposed[b,n,k] + rhs_offset),
 *         mult, shift) + output_zp
 *     clamp to [-128, 127]
 *
 * Layout matches cortex_m::quantized_batch_matmul: rhs is stored with
 * the contraction dim last, so both lhs and rhs inner loops are
 * stride-1 along K — MVE-friendly.
 *
 * For KWT-1 attention with nhead=1:
 *   - QK^T: lhs = Q(1,99,64), rhs_transposed = K(1,99,64), out (1,99,99)
 *           (B=1, M=99, K=64, N=99)
 *   - AV:   lhs = scores(1,99,99), rhs_transposed = V^T(1,64,99), out (1,99,64)
 *           (B=1, M=99, K=99, N=64) — V is transposed at AOT (constant)
 *
 * Scalar reference; MVE fast path follows once bit-exactness is gated. */
void batch_matmul_s8(
    const int8_t* lhs, const int8_t* rhs_transposed,
    int8_t* output, const BMMParams* p) {
  const uint32_t B = p->batch;
  const uint32_t M = p->M;
  const uint32_t K = p->K;
  const uint32_t N = p->N;
  const int32_t lhs_off = p->lhs_offset;
  const int32_t rhs_off = p->rhs_offset;
  const int32_t out_zp = p->output_zp;
  const int32_t mult = p->output_multiplier;
  const int32_t shift = p->output_shift;

  for (uint32_t b = 0; b < B; ++b) {
    const int8_t* lhs_b = lhs + (size_t)b * M * K;
    const int8_t* rhs_b = rhs_transposed + (size_t)b * N * K;
    int8_t* out_b = output + (size_t)b * M * N;
    for (uint32_t m = 0; m < M; ++m) {
      const int8_t* lhs_row = lhs_b + (size_t)m * K;
      for (uint32_t n = 0; n < N; ++n) {
        const int8_t* rhs_row = rhs_b + (size_t)n * K;
        int32_t acc = 0;
        for (uint32_t k = 0; k < K; ++k) {
          int32_t a = (int32_t)lhs_row[k] + lhs_off;
          int32_t c = (int32_t)rhs_row[k] + rhs_off;
          acc += a * c;
        }
        int32_t r = kwt_1_requantize(acc, mult, shift) + out_zp;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        out_b[(size_t)m * N + n] = (int8_t)r;
      }
    }
  }
}
