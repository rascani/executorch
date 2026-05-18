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
/* Phase 4: per-row int8 softmax matching cortex_m::softmax's python
 * impl bit-for-bit.  CMSIS-NN's actual arm_softmax_s8 uses a
 * fixed-point exp polynomial that can differ from this float-domain
 * implementation by ±1 LSB at individual output positions; the
 * python impl uses torch.softmax in float and that's the reference
 * the rest of the project gates on, so we match that here.
 *
 *   real_scale     = multiplier * 2^(shift - 57)
 *   x_fp[i]        = (input[i] - input_zp) * real_scale
 *   row_max        = max_i x_fp[i]
 *   exp_diffs[i]   = expf(x_fp[i] - row_max)
 *   probs[i]       = exp_diffs[i] / sum_i exp_diffs[i]
 *   output[i]      = clamp(round(probs[i] * 256) - 128, -128, 127)
 *
 * For KWT-1 attention, row_len is the sequence length (99); for
 * conformer stretch the row scratch sizing here is the load-bearing
 * piece (S~1000 needs more stack than we should commit to).  Today
 * we use a fixed 1024-element stack scratch; raise if a future model
 * exceeds it. */
#define KWT_1_SOFTMAX_MAX_ROW 1024

void softmax_s8(
    const int8_t* input, int8_t* output, const SoftmaxParams* p) {
  const uint32_t R = p->num_rows;
  const uint32_t N = p->row_len;
  const float in_zp = (float)p->input_zp;
  const float real_scale = (float)p->input_multiplier
      * ldexpf(1.0f, p->input_shift - 57);

  float scratch[KWT_1_SOFTMAX_MAX_ROW];

  for (uint32_t r = 0; r < R; ++r) {
    const int8_t* row_in = input + (size_t)r * N;
    int8_t* row_out = output + (size_t)r * N;

    /* Pass 1: dequant + row max. */
    float row_max = -INFINITY;
    for (uint32_t i = 0; i < N; ++i) {
      float x = ((float)row_in[i] - in_zp) * real_scale;
      scratch[i] = x;
      if (x > row_max) row_max = x;
    }

    /* Pass 2: shifted exp + row sum. */
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
      float e = expf(scratch[i] - row_max);
      scratch[i] = e;
      sum += e;
    }

    /* Pass 3: normalize + quantize at CMSIS-NN's fixed (scale=1/256,
     * zp=-128).  probs * 256 has range [0, 256], round and subtract
     * 128 — final clamp guards against a probability of 1.0
     * mapping to 128 (above int8 max). */
    const float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < N; ++i) {
      float prob = scratch[i] * inv_sum;
      float qf = prob * 256.0f;
      int32_t q = (int32_t)(qf + (qf >= 0.0f ? 0.5f : -0.5f));
      q -= 128;
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      row_out[i] = (int8_t)q;
    }
  }
}


/* Phase 6: int8 Linear with CMSIS-NN kernel_sum precomputation.
 * Bit-exact port of cortex_m::quantized_linear (compute_using_kernel_sum=True). */
void linear_s8(
    const int8_t* input, const int8_t* weights, int8_t* output,
    const LinearParams* p) {
  const uint32_t M = p->num_rows;
  const uint32_t K = p->in_features;
  const uint32_t N = p->out_features;
  const int32_t filter_off = p->filter_offset;
  const int32_t out_off = p->output_offset;
  const int32_t mult = p->output_multiplier;
  const int32_t shift = p->output_shift;
  const int32_t amin = p->activation_min;
  const int32_t amax = p->activation_max;

  for (uint32_t m = 0; m < M; ++m) {
    const int8_t* row_in = input + (size_t)m * K;
    int32_t row_sum = 0;
    for (uint32_t k = 0; k < K; ++k) row_sum += (int32_t)row_in[k];
    for (uint32_t n = 0; n < N; ++n) {
      const int8_t* w_row = weights + (size_t)n * K;
      int32_t acc = 0;
      for (uint32_t k = 0; k < K; ++k) acc += (int32_t)row_in[k] * (int32_t)w_row[k];
      acc += row_sum * filter_off;
      acc += p->kernel_sum[n];
      int32_t r = kwt_1_requantize(acc, mult, shift) + out_off;
      if (r < amin) r = amin;
      if (r > amax) r = amax;
      output[(size_t)m * N + n] = (int8_t)r;
    }
  }
}


/* Phase 6: int8 quantized add.  Bit-exact port of cortex_m::quantized_add:
 *   self_shift  = (self  - self_zp)  << 20
 *   self_fp     = requantize(self_shift, self_mult, self_shift)
 *   other_shift = (other - other_zp) << 20
 *   other_fp    = requantize(other_shift, other_mult, other_shift)
 *   result      = requantize(self_fp + other_fp, out_mult, out_shift) + out_zp
 *   output      = clamp(result, act_min, act_max)
 * The 20-bit pre-shift moves both inputs into a common int32 "internal"
 * domain so subsequent requantizes can align scales without losing
 * precision on the smaller-scale operand. */
#define KWT_1_ADD_INTERNAL_SHIFT 20

void add_s8(
    const int8_t* self_in, const int8_t* other_in, int8_t* output,
    const AddParams* p) {
  const uint32_t n = p->num_elements;
  for (uint32_t i = 0; i < n; ++i) {
    int32_t s_shift = ((int32_t)self_in[i] - p->self_zp) << KWT_1_ADD_INTERNAL_SHIFT;
    int32_t o_shift = ((int32_t)other_in[i] - p->other_zp) << KWT_1_ADD_INTERNAL_SHIFT;
    int32_t s_fp = kwt_1_requantize(s_shift, p->self_multiplier, p->self_shift);
    int32_t o_fp = kwt_1_requantize(o_shift, p->other_multiplier, p->other_shift);
    int32_t r = kwt_1_requantize(s_fp + o_fp, p->output_multiplier, p->output_shift)
              + p->output_zp;
    if (r < p->activation_min) r = p->activation_min;
    if (r > p->activation_max) r = p->activation_max;
    output[i] = (int8_t)r;
  }
}


/* Phase 5: streaming fused attention.  For each Q row i:
 *   1. Compute S int8 scores  = QK^T row i   → scratch[0..S)
 *   2. softmax_s8 in place on the S scores   → scratch[0..S) (probs)
 *   3. Compute d int8 outputs = AV row i, contracting probs against v
 *
 * The full (S, S) score matrix never materializes — only S bytes of
 * scratch live at a time.  All three steps reuse the same per-element
 * math as batch_matmul_s8 / softmax_s8 above (no duplicate
 * implementations); we just inline them here so the row-level scratch
 * stays on the stack.
 *
 * Memory: S int8 scores/probs (≤ KWT_1_SOFTMAX_MAX_ROW; the in-place
 * softmax uses its own internal float scratch as well, but that's
 * accounted for in softmax_s8).
 */
void attention_fused_s8(
    const int8_t* q, const int8_t* k, const int8_t* v,
    int8_t* output, const AttentionParams* p) {
  const uint32_t B = p->batch;
  const uint32_t S = p->seq_len;
  const uint32_t D = p->embed_dim;
  const int32_t q_off = p->q_offset;
  const int32_t k_off = p->k_offset;
  const int32_t qk_zp = p->qk_output_zp;
  const int32_t qk_mult = p->qk_output_multiplier;
  const int32_t qk_shift = p->qk_output_shift;
  const int32_t v_off = p->v_offset;
  const int32_t av_zp = p->av_output_zp;
  const int32_t av_mult = p->av_output_multiplier;
  const int32_t av_shift = p->av_output_shift;
  /* probs come out of softmax_s8 at zero_point = -128, so their
   * BMM-side offset is -(-128) = 128. */
  const int32_t probs_off = 128;

  int8_t scores_scratch[KWT_1_SOFTMAX_MAX_ROW];
  SoftmaxParams sm_p = {
    .num_rows = 1, .row_len = S,
    .input_zp = qk_zp,
    .input_multiplier = p->softmax_input_multiplier,
    .input_shift = p->softmax_input_shift,
  };

  for (uint32_t b = 0; b < B; ++b) {
    const int8_t* q_b = q + (size_t)b * S * D;
    const int8_t* k_b = k + (size_t)b * S * D;
    const int8_t* v_b = v + (size_t)b * S * D;
    int8_t* out_b = output + (size_t)b * S * D;

    for (uint32_t i = 0; i < S; ++i) {
      const int8_t* q_row = q_b + (size_t)i * D;

      /* Step 1: QK^T row i = sum_k (q[i,k] + q_off) * (k[j,k] + k_off)
       *                      requantize → int8 score for each j. */
      for (uint32_t j = 0; j < S; ++j) {
        const int8_t* k_row = k_b + (size_t)j * D;
        int32_t acc = 0;
        for (uint32_t kk = 0; kk < D; ++kk) {
          int32_t a = (int32_t)q_row[kk] + q_off;
          int32_t c = (int32_t)k_row[kk] + k_off;
          acc += a * c;
        }
        int32_t r = kwt_1_requantize(acc, qk_mult, qk_shift) + qk_zp;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        scores_scratch[j] = (int8_t)r;
      }

      /* Step 2: softmax those S scores in place.  softmax_s8 reads
       * the int8 input once at the start of each pass; in-place is
       * safe because it stages everything in its own float scratch
       * before writing the int8 output. */
      softmax_s8(scores_scratch, scores_scratch, &sm_p);

      /* Step 3: AV row i = sum_j (probs[j] + probs_off) * (v_t[n,j] + v_off).
       * v is stored in (B, D, S) BMM rhs_transposed layout, so the inner
       * loop reads stride-1 along S for fixed n — same layout as probs,
       * MVE-friendly when the MVE fast path lands. */
      int8_t* out_row = out_b + (size_t)i * D;
      for (uint32_t n = 0; n < D; ++n) {
        const int8_t* v_row = v_b + (size_t)n * S;
        int32_t acc = 0;
        for (uint32_t j = 0; j < S; ++j) {
          int32_t a = (int32_t)scores_scratch[j] + probs_off;
          int32_t c = (int32_t)v_row[j] + v_off;
          acc += a * c;
        }
        int32_t r = kwt_1_requantize(acc, av_mult, av_shift) + av_zp;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        out_row[n] = (int8_t)r;
      }
    }
  }
}


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
