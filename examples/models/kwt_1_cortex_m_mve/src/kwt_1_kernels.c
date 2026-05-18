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
