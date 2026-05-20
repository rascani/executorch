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


#if KWT_1_USE_MVE
/* Bit-exact MVE version of kwt_1_requantize for the common shift<=0 case,
 * which is what add_s8 and the linears use.  Implements CMSIS-NN's
 * round-away-from-zero tie-break by hand because the architectural
 * vrshlq_s32 rounds ties to positive infinity.
 *
 * For shift==0, this is just vqrdmulhq_n_s32.  For shift<0, we do
 *   rdmh = vqrdmulhq_n_s32(acc, mult)        (a*b round /2^31)
 *   shifted = rdmh >> right (arithmetic, NO rounding)
 *   remainder = rdmh & ((1<<right)-1)
 *   threshold = ((1<<right)-1) >> 1 + (rdmh < 0 ? 1 : 0)
 *   increment shifted by 1 wherever remainder > threshold
 */
static inline int32x4_t kwt_1_mve_requantize_nonpos(
    int32x4_t acc, int32_t multiplier, int32_t shift) {
  int32x4_t rdmh = vqrdmulhq_n_s32(acc, multiplier);
  if (shift == 0) return rdmh;
  const int32_t right = -shift;
  const int32_t mask_s = (1 << right) - 1;
  const int32_t half = mask_s >> 1;
  int32x4_t shifted = vshlq_s32(rdmh, vdupq_n_s32(-right));
  int32x4_t remainder = vandq_s32(rdmh, vdupq_n_s32(mask_s));
  /* is_neg lane mask: -1 where rdmh<0, 0 elsewhere (arithmetic shift by 31). */
  int32x4_t is_neg = vshrq_n_s32(rdmh, 31);
  /* threshold = half - is_neg  ⇒ half+1 when neg, half when non-neg. */
  int32x4_t threshold = vsubq_s32(vdupq_n_s32(half), is_neg);
  /* increment lane by 1 where remainder > threshold. */
  mve_pred16_t pred = vcmpgtq_s32(remainder, threshold);
  return vaddq_m_n_s32(shifted, shifted, 1, pred);
}
#endif


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


#if KWT_1_USE_MVE
/* MVE LayerNorm: three-pass like the scalar version, but with the
 * inner loops over D vectorized 4 float lanes at a time.  D must be a
 * multiple of 4; for KWT-1 D=64 is comfortably aligned.  Bit-exactness
 * vs the scalar reference depends on the summation order, since
 * torch's LN doesn't guarantee left-to-right reduction either.
 * Empirically with D=64 the saved reference produced by torch's LN
 * matches both the pairwise-vector and strict-scalar accumulations.
 */
static void layer_norm_s8_mve(
    const int8_t* input, int8_t* output, const LayerNormParams* p) {
  const uint32_t D = p->embed_dim;
  const float in_scale = p->input_scale;
  const float inv_out_scale = 1.0f / p->output_scale;
  const float out_zp_f = (float)p->output_zp;
  const float eps = p->eps;
  const float inv_D = 1.0f / (float)D;
  const int32_t in_zp_i = p->input_zp;

  for (uint32_t r = 0; r < p->num_rows; ++r) {
    const int8_t* row_in = input + (size_t)r * D;
    int8_t* row_out = output + (size_t)r * D;

    /* Pass 1: dequant + mean. */
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    for (uint32_t c = 0; c < D; c += 4) {
      int32x4_t i_v = vldrbq_s32(row_in + c);
      i_v = vsubq_n_s32(i_v, in_zp_i);
      float32x4_t fv = vmulq_n_f32(vcvtq_f32_s32(i_v), in_scale);
      sum_v = vaddq_f32(sum_v, fv);
    }
    /* MVE doesn't expose a float horizontal-sum intrinsic in this
     * toolchain; reduce the 4 lanes via vgetq_lane. */
    const float mean = (vgetq_lane_f32(sum_v, 0) + vgetq_lane_f32(sum_v, 1)
                      + vgetq_lane_f32(sum_v, 2) + vgetq_lane_f32(sum_v, 3))
                     * inv_D;

    /* Pass 2: sum of squared deviations. */
    float32x4_t sumsq_v = vdupq_n_f32(0.0f);
    const float32x4_t mean_v = vdupq_n_f32(mean);
    for (uint32_t c = 0; c < D; c += 4) {
      int32x4_t i_v = vldrbq_s32(row_in + c);
      i_v = vsubq_n_s32(i_v, in_zp_i);
      float32x4_t fv = vmulq_n_f32(vcvtq_f32_s32(i_v), in_scale);
      float32x4_t dv = vsubq_f32(fv, mean_v);
      sumsq_v = vfmaq_f32(sumsq_v, dv, dv);
    }
    const float var = (vgetq_lane_f32(sumsq_v, 0) + vgetq_lane_f32(sumsq_v, 1)
                     + vgetq_lane_f32(sumsq_v, 2) + vgetq_lane_f32(sumsq_v, 3))
                    * inv_D;
    const float rstd = 1.0f / sqrtf(var + eps);

    /* Pass 3: normalize + γβ + requant.  γ and β are float32. */
    const float32x4_t rstd_v = vdupq_n_f32(rstd);
    const float32x4_t inv_os_v = vdupq_n_f32(inv_out_scale);
    const float32x4_t out_zp_v = vdupq_n_f32(out_zp_f);
    const int32x4_t v_neg128 = vdupq_n_s32(-128);
    const int32x4_t v_pos127 = vdupq_n_s32(127);
    const int has_beta = (p->beta != (const float*)0);
    for (uint32_t c = 0; c < D; c += 4) {
      int32x4_t i_v = vldrbq_s32(row_in + c);
      i_v = vsubq_n_s32(i_v, in_zp_i);
      float32x4_t x = vmulq_n_f32(vcvtq_f32_s32(i_v), in_scale);
      float32x4_t norm = vmulq_f32(vsubq_f32(x, mean_v), rstd_v);
      float32x4_t g = vldrwq_f32(p->gamma + c);
      float32x4_t y = vmulq_f32(norm, g);
      if (has_beta) y = vaddq_f32(y, vldrwq_f32(p->beta + c));
      float32x4_t qf = vaddq_f32(vmulq_f32(y, inv_os_v), out_zp_v);
      /* round-half-away-from-zero: rint with ties-to-nearest is the
       * MVE default and matches torch.round for the magnitudes we hit.
       * For exact-tie handling matching the scalar `(qf + (qf>=0?0.5:-0.5))`
       * cast we use vcvtaq_s32_f32 (away-from-zero round). */
      int32x4_t qi = vcvtaq_s32_f32(qf);
      qi = vminq_s32(vmaxq_s32(qi, v_neg128), v_pos127);
      vstrbq_s32(row_out + c, qi);
    }
  }
}
#endif

/* Public entry point — dispatches scalar vs MVE based on shape and
 * build flags.  Phase 1 shipped scalar only; Phase 8 adds the MVE
 * fast path when D is a multiple of 4 (always true for KWT-1's D=64). */
void layer_norm_s8(
    const int8_t* input, int8_t* output, const LayerNormParams* p) {
#if KWT_1_USE_MVE
  if ((p->embed_dim & 3u) == 0u) {
    layer_norm_s8_mve(input, output, p);
    return;
  }
#endif
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

/* When KWT_1_FAST_EXPF=1, softmax_s8 uses a 5th-order polynomial in 2^f
 * (with i,f from y = x*log2(e)) instead of newlib's `expf`.  Cost
 * drops from ~300 cycles to ~30 cycles per call.  Accuracy is ~4 ULP
 * on the [-16, 0] range we actually feed softmax (anything below -16
 * is clamped to 0 since exp(-16) ≈ 1e-7 quantizes to 0 at int8/256
 * resolution anyway).  Not bit-exact against newlib `expf` — the
 * default path stays bit-exact for verification; opt-in for shipping. */
#ifndef KWT_1_FAST_EXPF
#define KWT_1_FAST_EXPF 0
#endif

#if KWT_1_FAST_EXPF
static inline float kwt_1_fast_expf(float x) {
  if (x < -16.0f) return 0.0f;
  if (x >  16.0f) x = 16.0f;
  const float log2e = 1.44269504088896341f;
  float y = x * log2e;
  /* floor(y) for arbitrary sign — (int) truncates toward zero, so
   * subtract 1 for negative non-integer y. */
  int32_t i = (int32_t)y;
  float fi = (float)i;
  if (y < fi) { i -= 1; fi -= 1.0f; }
  float f = y - fi;
  /* Minimax polynomial for 2^f on [0, 1]: ~4 ULP. */
  float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (
            0.0555041f + f * (0.0096180f + f * 0.0013357f))));
  /* 2^i by direct IEEE-754 exponent injection. */
  union { float f; int32_t i; } u;
  u.i = (i + 127) << 23;
  return p * u.f;
}
#define KWT_1_EXPF(x) kwt_1_fast_expf(x)
#else
#define KWT_1_EXPF(x) expf(x)
#endif

__attribute__((always_inline))
static inline void softmax_s8(
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
      float e = KWT_1_EXPF(scratch[i] - row_max);
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


/* Phase 7: int8 N-D transpose with up to rank-4 perm.  Output is
 * contiguous; input is read at strides encoded in p->in_stride.
 *
 * Phase 8: rank-3 fast path that drops the per-element idx-array walk
 * (which dominated runtime — the rank-N general version was ~43
 * cycles/byte).  All three transposes KWT-1 emits are rank-3 so this
 * is the hot path. */
void transpose_s8(
    const int8_t* input, int8_t* output, const TransposeParams* p) {
  const uint32_t rank = p->rank;
  if (rank == 3) {
    const uint32_t S0 = p->shape[0];
    const uint32_t S1 = p->shape[1];
    const uint32_t S2 = p->shape[2];
    const int32_t st0 = (int32_t)p->in_stride[0];
    const int32_t st1 = (int32_t)p->in_stride[1];
    const int32_t st2 = (int32_t)p->in_stride[2];
    int8_t* out = output;
    for (uint32_t i0 = 0; i0 < S0; ++i0) {
      const int8_t* row0 = input + (size_t)i0 * st0;
      for (uint32_t i1 = 0; i1 < S1; ++i1) {
        const int8_t* row1 = row0 + (size_t)i1 * st1;
        for (uint32_t i2 = 0; i2 < S2; ++i2) {
          *out++ = row1[(size_t)i2 * st2];
        }
      }
    }
    return;
  }
  uint32_t total = 1;
  for (uint32_t r = 0; r < rank; ++r) total *= p->shape[r];
  uint32_t idx[4] = {0};
  for (uint32_t out_i = 0; out_i < total; ++out_i) {
    size_t in_off = 0;
    for (uint32_t r = 0; r < rank; ++r) in_off += (size_t)idx[r] * p->in_stride[r];
    output[out_i] = input[in_off];
    for (int r = (int)rank - 1; r >= 0; --r) {
      idx[r]++;
      if (idx[r] < p->shape[r]) break;
      idx[r] = 0;
    }
  }
}


/* Phase 10: int8 mean across one axis with fused dequant + requant.
 * Matches the dequant → aten.mean.dim → quant trio the lowering pass
 * inserts around a float mean op.  For each (o, n) in outer × inner,
 * sums `reduce` int8 lanes into an int32, then rescales via
 *   ratio = input_scale / (reduce * output_scale)
 * to land back in int8.  Bit-exact with the python reference (round
 * half away from zero, clamp to int8). */
void mean_dim_s8(
    const int8_t* input, int8_t* output, const MeanDimParams* p) {
  const uint32_t O = p->outer;
  const uint32_t R = p->reduce;
  const uint32_t I = p->inner;
  const int32_t in_zp = p->input_zp;
  const float ratio = p->input_scale / ((float)R * p->output_scale);
  const float out_zp_f = (float)p->output_zp;
  for (uint32_t o = 0; o < O; ++o) {
    const int8_t* in_outer = input + (size_t)o * R * I;
    int8_t* out_outer = output + (size_t)o * I;
    for (uint32_t n = 0; n < I; ++n) {
      int32_t sum = 0;
      for (uint32_t r = 0; r < R; ++r) {
        sum += (int32_t)in_outer[(size_t)r * I + n] - in_zp;
      }
      float qf = (float)sum * ratio + out_zp_f;
      int32_t q = (int32_t)(qf + (qf >= 0.0f ? 0.5f : -0.5f));
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      out_outer[n] = (int8_t)q;
    }
  }
}


/* Phase 6: int8 Linear with CMSIS-NN kernel_sum precomputation.
 * Bit-exact port of cortex_m::quantized_linear (compute_using_kernel_sum=True).
 *
 * Phase 8: MVE fast path when K is a multiple of 16.  KWT-1's six
 * linears all have K ∈ {64, 256}, so the fast path is always taken.
 * Strategy: tile N by 4 so the input row is loaded once for four
 * output channels, then vmladavaq_s8 the int8 dot products into
 * scalar int32 accumulators.  The row-sum is vectorized too via
 * vaddvaq_s8. */
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

#if KWT_1_USE_MVE
  /* M-tile-2 fast path for large M (canonical KWT-1's S=98 rows).
   * Halves weight-load traffic by sharing the 4 N-tile weight vectors
   * across two input rows.  Earlier attempts regressed on M=8 (the
   * OneBlock test) because the 8-acc inner blew through the M55
   * issue queue; at M=98 the 49 m-pair iterations amortize that
   * cost.  Guarded on M >= 32; handles odd M with a single-row tail. */
  /* Skip when K=64: that case has its own better fast path (input
   * hoist + N-tile-4) further down.  For K=40 (canonical KWT-1's
   * embed Linear) and K=128 (FFN project) at M=98 this commit
   * tightens both significantly. */
  if (K >= 16 && K != 64 && M >= 32 && shift <= 0) {
    const uint32_t K_full = K & ~15u;
    const uint32_t M_pairs = M & ~1u;
    const int32x4_t v_amin = vdupq_n_s32(amin);
    const int32x4_t v_amax = vdupq_n_s32(amax);
    const int32x4_t v_out_off = vdupq_n_s32(out_off);
    for (uint32_t m = 0; m < M_pairs; m += 2) {
      const int8_t* row0 = input + (size_t)(m + 0) * K;
      const int8_t* row1 = input + (size_t)(m + 1) * K;
      int32_t rs0 = 0, rs1 = 0;
      for (uint32_t k = 0; k < K_full; k += 16) {
        rs0 = vaddvaq_s8(rs0, vldrbq_s8(row0 + k));
        rs1 = vaddvaq_s8(rs1, vldrbq_s8(row1 + k));
      }
      for (uint32_t k = K_full; k < K; ++k) {
        rs0 += (int32_t)row0[k];
        rs1 += (int32_t)row1[k];
      }
      const int32_t off0 = rs0 * filter_off;
      const int32_t off1 = rs1 * filter_off;
      uint32_t n = 0;
      for (; n + 4 <= N; n += 4) {
        const int8_t* w0 = weights + (size_t)(n + 0) * K;
        const int8_t* w1 = weights + (size_t)(n + 1) * K;
        const int8_t* w2 = weights + (size_t)(n + 2) * K;
        const int8_t* w3 = weights + (size_t)(n + 3) * K;
        int32_t a00=0, a01=0, a02=0, a03=0;
        int32_t a10=0, a11=0, a12=0, a13=0;
        for (uint32_t k = 0; k < K_full; k += 16) {
          int8x16_t iv0 = vldrbq_s8(row0 + k);
          int8x16_t iv1 = vldrbq_s8(row1 + k);
          int8x16_t wv0 = vldrbq_s8(w0 + k);
          int8x16_t wv1 = vldrbq_s8(w1 + k);
          int8x16_t wv2 = vldrbq_s8(w2 + k);
          int8x16_t wv3 = vldrbq_s8(w3 + k);
          a00 = vmladavaq_s8(a00, iv0, wv0);
          a01 = vmladavaq_s8(a01, iv0, wv1);
          a02 = vmladavaq_s8(a02, iv0, wv2);
          a03 = vmladavaq_s8(a03, iv0, wv3);
          a10 = vmladavaq_s8(a10, iv1, wv0);
          a11 = vmladavaq_s8(a11, iv1, wv1);
          a12 = vmladavaq_s8(a12, iv1, wv2);
          a13 = vmladavaq_s8(a13, iv1, wv3);
        }
        for (uint32_t k = K_full; k < K; ++k) {
          int32_t r0 = (int32_t)row0[k];
          int32_t r1 = (int32_t)row1[k];
          int32_t s0 = (int32_t)w0[k], s1 = (int32_t)w1[k];
          int32_t s2 = (int32_t)w2[k], s3 = (int32_t)w3[k];
          a00 += r0 * s0; a01 += r0 * s1; a02 += r0 * s2; a03 += r0 * s3;
          a10 += r1 * s0; a11 += r1 * s1; a12 += r1 * s2; a13 += r1 * s3;
        }
        int32x4_t ks = vldrwq_s32(p->kernel_sum + n);
        int32x4_t acc0 = {a00, a01, a02, a03};
        int32x4_t acc1 = {a10, a11, a12, a13};
        acc0 = vaddq_s32(vaddq_n_s32(acc0, off0), ks);
        acc1 = vaddq_s32(vaddq_n_s32(acc1, off1), ks);
        acc0 = kwt_1_mve_requantize_nonpos(acc0, mult, shift);
        acc1 = kwt_1_mve_requantize_nonpos(acc1, mult, shift);
        acc0 = vminq_s32(vmaxq_s32(vaddq_s32(acc0, v_out_off), v_amin), v_amax);
        acc1 = vminq_s32(vmaxq_s32(vaddq_s32(acc1, v_out_off), v_amin), v_amax);
        vstrbq_s32(output + (size_t)(m + 0) * N + n, acc0);
        vstrbq_s32(output + (size_t)(m + 1) * N + n, acc1);
      }
      for (; n < N; ++n) {
        const int8_t* w_row = weights + (size_t)n * K;
        int32_t a0 = 0, a1 = 0;
        for (uint32_t k = 0; k < K_full; k += 16) {
          int8x16_t wv = vldrbq_s8(w_row + k);
          a0 = vmladavaq_s8(a0, vldrbq_s8(row0 + k), wv);
          a1 = vmladavaq_s8(a1, vldrbq_s8(row1 + k), wv);
        }
        for (uint32_t k = K_full; k < K; ++k) {
          int32_t w = (int32_t)w_row[k];
          a0 += (int32_t)row0[k] * w;
          a1 += (int32_t)row1[k] * w;
        }
        int32_t ks_n = p->kernel_sum[n];
        int32_t r0 = kwt_1_requantize(a0 + off0 + ks_n, mult, shift) + out_off;
        int32_t r1 = kwt_1_requantize(a1 + off1 + ks_n, mult, shift) + out_off;
        if (r0 < amin) r0 = amin; else if (r0 > amax) r0 = amax;
        if (r1 < amin) r1 = amin; else if (r1 > amax) r1 = amax;
        output[(size_t)(m + 0) * N + n] = (int8_t)r0;
        output[(size_t)(m + 1) * N + n] = (int8_t)r1;
      }
    }
    /* Odd-M tail: one trailing row processed via tile-of-4 N (same
     * structure as the K >= 16 single-row path below, minus the
     * K=64-specific hoist). */
    if (M_pairs != M) {
      const uint32_t m = M_pairs;
      const int8_t* row_in = input + (size_t)m * K;
      int32_t rs = 0;
      for (uint32_t k = 0; k < K_full; k += 16) {
        rs = vaddvaq_s8(rs, vldrbq_s8(row_in + k));
      }
      for (uint32_t k = K_full; k < K; ++k) rs += (int32_t)row_in[k];
      const int32_t off = rs * filter_off;
      uint32_t n = 0;
      for (; n + 4 <= N; n += 4) {
        const int8_t* w0 = weights + (size_t)(n + 0) * K;
        const int8_t* w1 = weights + (size_t)(n + 1) * K;
        const int8_t* w2 = weights + (size_t)(n + 2) * K;
        const int8_t* w3 = weights + (size_t)(n + 3) * K;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (uint32_t k = 0; k < K_full; k += 16) {
          int8x16_t iv = vldrbq_s8(row_in + k);
          a0 = vmladavaq_s8(a0, iv, vldrbq_s8(w0 + k));
          a1 = vmladavaq_s8(a1, iv, vldrbq_s8(w1 + k));
          a2 = vmladavaq_s8(a2, iv, vldrbq_s8(w2 + k));
          a3 = vmladavaq_s8(a3, iv, vldrbq_s8(w3 + k));
        }
        for (uint32_t k = K_full; k < K; ++k) {
          int32_t r = (int32_t)row_in[k];
          a0 += r * (int32_t)w0[k];
          a1 += r * (int32_t)w1[k];
          a2 += r * (int32_t)w2[k];
          a3 += r * (int32_t)w3[k];
        }
        int32x4_t acc = {a0, a1, a2, a3};
        int32x4_t ks = vldrwq_s32(p->kernel_sum + n);
        acc = vaddq_n_s32(acc, off);
        acc = vaddq_s32(acc, ks);
        acc = kwt_1_mve_requantize_nonpos(acc, mult, shift);
        acc = vaddq_s32(acc, v_out_off);
        acc = vminq_s32(vmaxq_s32(acc, v_amin), v_amax);
        vstrbq_s32(output + (size_t)m * N + n, acc);
      }
      for (; n < N; ++n) {
        const int8_t* w_row = weights + (size_t)n * K;
        int32_t acc = 0;
        for (uint32_t k = 0; k < K_full; k += 16) {
          acc = vmladavaq_s8(acc, vldrbq_s8(row_in + k), vldrbq_s8(w_row + k));
        }
        for (uint32_t k = K_full; k < K; ++k) {
          acc += (int32_t)row_in[k] * (int32_t)w_row[k];
        }
        acc += off + p->kernel_sum[n];
        int32_t r = kwt_1_requantize(acc, mult, shift) + out_off;
        if (r < amin) r = amin; else if (r > amax) r = amax;
        output[(size_t)m * N + n] = (int8_t)r;
      }
    }
    return;
  }

  /* Accept any K ≥ 16; the MVE bulk loop handles K_full = K & ~15u
   * lanes via vmladavaq_s8 and a scalar tail handles the last K % 16.
   * The K=64 hoist specialization (further down) still applies only
   * to that exact width. */
  if (K >= 16) {
    const uint32_t K_full = K & ~15u;
    for (uint32_t m = 0; m < M; ++m) {
      const int8_t* row_in = input + (size_t)m * K;
      /* row_sum = sum_k row_in[k] via vaddvaq_s8 + scalar tail. */
      int32_t row_sum = 0;
      for (uint32_t k = 0; k < K_full; k += 16) {
        row_sum = vaddvaq_s8(row_sum, vldrbq_s8(row_in + k));
      }
      for (uint32_t k = K_full; k < K; ++k) row_sum += (int32_t)row_in[k];
      const int32_t row_off_term = row_sum * filter_off;

      uint32_t n = 0;
      const int32x4_t v_amin = vdupq_n_s32(amin);
      const int32x4_t v_amax = vdupq_n_s32(amax);
      const int32x4_t v_out_off = vdupq_n_s32(out_off);
      /* Tile-of-4 N with vectorized requantize+clamp+narrow.  shift is
       * a runtime value but is constant across the entire linear, so
       * the compiler hoists the branch in kwt_1_mve_requantize_nonpos.
       * Restriction: this fast path assumes shift <= 0 (true for every
       * KWT-1 linear and for CMSIS-NN's typical PT2E output).  Falls
       * back to the per-lane scalar requantize otherwise. */
      if (shift <= 0) {
        /* K=64 specialization: hoist the 4 input vectors out of the N
         * loop so the same 64 bytes of input stay resident in MVE Q
         * registers across all 16 (=N/4) tile iterations.  Cuts input
         * loads by 16× vs reloading them per tile.  The compiler picks
         * registers for iv0..iv3 and the 4 weight vectors (vldrbq the
         * w pointers); 8 Q regs available, all 8 in use, no spill. */
        if (K == 64) {
          int8x16_t iv0 = vldrbq_s8(row_in +  0);
          int8x16_t iv1 = vldrbq_s8(row_in + 16);
          int8x16_t iv2 = vldrbq_s8(row_in + 32);
          int8x16_t iv3 = vldrbq_s8(row_in + 48);
          for (; n + 4 <= N; n += 4) {
            const int8_t* w0 = weights + (size_t)(n + 0) * K;
            const int8_t* w1 = weights + (size_t)(n + 1) * K;
            const int8_t* w2 = weights + (size_t)(n + 2) * K;
            const int8_t* w3 = weights + (size_t)(n + 3) * K;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            a0 = vmladavaq_s8(a0, iv0, vldrbq_s8(w0 +  0));
            a1 = vmladavaq_s8(a1, iv0, vldrbq_s8(w1 +  0));
            a2 = vmladavaq_s8(a2, iv0, vldrbq_s8(w2 +  0));
            a3 = vmladavaq_s8(a3, iv0, vldrbq_s8(w3 +  0));
            a0 = vmladavaq_s8(a0, iv1, vldrbq_s8(w0 + 16));
            a1 = vmladavaq_s8(a1, iv1, vldrbq_s8(w1 + 16));
            a2 = vmladavaq_s8(a2, iv1, vldrbq_s8(w2 + 16));
            a3 = vmladavaq_s8(a3, iv1, vldrbq_s8(w3 + 16));
            a0 = vmladavaq_s8(a0, iv2, vldrbq_s8(w0 + 32));
            a1 = vmladavaq_s8(a1, iv2, vldrbq_s8(w1 + 32));
            a2 = vmladavaq_s8(a2, iv2, vldrbq_s8(w2 + 32));
            a3 = vmladavaq_s8(a3, iv2, vldrbq_s8(w3 + 32));
            a0 = vmladavaq_s8(a0, iv3, vldrbq_s8(w0 + 48));
            a1 = vmladavaq_s8(a1, iv3, vldrbq_s8(w1 + 48));
            a2 = vmladavaq_s8(a2, iv3, vldrbq_s8(w2 + 48));
            a3 = vmladavaq_s8(a3, iv3, vldrbq_s8(w3 + 48));
            int32x4_t acc = {a0, a1, a2, a3};
            int32x4_t ks = vldrwq_s32(p->kernel_sum + n);
            acc = vaddq_n_s32(acc, row_off_term);
            acc = vaddq_s32(acc, ks);
            acc = kwt_1_mve_requantize_nonpos(acc, mult, shift);
            acc = vaddq_s32(acc, v_out_off);
            acc = vminq_s32(vmaxq_s32(acc, v_amin), v_amax);
            vstrbq_s32(output + (size_t)m * N + n, acc);
          }
        } else {
        for (; n + 4 <= N; n += 4) {
          const int8_t* w0 = weights + (size_t)(n + 0) * K;
          const int8_t* w1 = weights + (size_t)(n + 1) * K;
          const int8_t* w2 = weights + (size_t)(n + 2) * K;
          const int8_t* w3 = weights + (size_t)(n + 3) * K;
          int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          for (uint32_t k = 0; k < K_full; k += 16) {
            int8x16_t iv = vldrbq_s8(row_in + k);
            a0 = vmladavaq_s8(a0, iv, vldrbq_s8(w0 + k));
            a1 = vmladavaq_s8(a1, iv, vldrbq_s8(w1 + k));
            a2 = vmladavaq_s8(a2, iv, vldrbq_s8(w2 + k));
            a3 = vmladavaq_s8(a3, iv, vldrbq_s8(w3 + k));
          }
          for (uint32_t k = K_full; k < K; ++k) {
            int32_t r = (int32_t)row_in[k];
            a0 += r * (int32_t)w0[k];
            a1 += r * (int32_t)w1[k];
            a2 += r * (int32_t)w2[k];
            a3 += r * (int32_t)w3[k];
          }
          /* Pack (a0,a1,a2,a3) into int32x4, add row_off_term and the
           * kernel_sum vector, requantize, add out_off, clamp. */
          int32x4_t acc = {a0, a1, a2, a3};
          int32x4_t ks = vldrwq_s32(p->kernel_sum + n);
          acc = vaddq_n_s32(acc, row_off_term);
          acc = vaddq_s32(acc, ks);
          acc = kwt_1_mve_requantize_nonpos(acc, mult, shift);
          acc = vaddq_s32(acc, v_out_off);
          acc = vminq_s32(vmaxq_s32(acc, v_amin), v_amax);
          /* Narrow int32x4 → int8x4 and store via vstrbq_s32. */
          vstrbq_s32(output + (size_t)m * N + n, acc);
        }
        }
      } else {
        for (; n + 4 <= N; n += 4) {
          const int8_t* w0 = weights + (size_t)(n + 0) * K;
          const int8_t* w1 = weights + (size_t)(n + 1) * K;
          const int8_t* w2 = weights + (size_t)(n + 2) * K;
          const int8_t* w3 = weights + (size_t)(n + 3) * K;
          int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          for (uint32_t k = 0; k < K_full; k += 16) {
            int8x16_t iv = vldrbq_s8(row_in + k);
            a0 = vmladavaq_s8(a0, iv, vldrbq_s8(w0 + k));
            a1 = vmladavaq_s8(a1, iv, vldrbq_s8(w1 + k));
            a2 = vmladavaq_s8(a2, iv, vldrbq_s8(w2 + k));
            a3 = vmladavaq_s8(a3, iv, vldrbq_s8(w3 + k));
          }
          for (uint32_t k = K_full; k < K; ++k) {
            int32_t r = (int32_t)row_in[k];
            a0 += r * (int32_t)w0[k];
            a1 += r * (int32_t)w1[k];
            a2 += r * (int32_t)w2[k];
            a3 += r * (int32_t)w3[k];
          }
          a0 += row_off_term + p->kernel_sum[n + 0];
          a1 += row_off_term + p->kernel_sum[n + 1];
          a2 += row_off_term + p->kernel_sum[n + 2];
          a3 += row_off_term + p->kernel_sum[n + 3];
          int32_t r0 = kwt_1_requantize(a0, mult, shift) + out_off;
          int32_t r1 = kwt_1_requantize(a1, mult, shift) + out_off;
          int32_t r2 = kwt_1_requantize(a2, mult, shift) + out_off;
          int32_t r3 = kwt_1_requantize(a3, mult, shift) + out_off;
          if (r0 < amin) r0 = amin; else if (r0 > amax) r0 = amax;
          if (r1 < amin) r1 = amin; else if (r1 > amax) r1 = amax;
          if (r2 < amin) r2 = amin; else if (r2 > amax) r2 = amax;
          if (r3 < amin) r3 = amin; else if (r3 > amax) r3 = amax;
          int8_t* out_row = output + (size_t)m * N + n;
          out_row[0] = (int8_t)r0; out_row[1] = (int8_t)r1;
          out_row[2] = (int8_t)r2; out_row[3] = (int8_t)r3;
        }
      }
      /* Tail in N (K-vectorized still). */
      for (; n < N; ++n) {
        const int8_t* w_row = weights + (size_t)n * K;
        int32_t acc = 0;
        for (uint32_t k = 0; k < K_full; k += 16) {
          acc = vmladavaq_s8(acc, vldrbq_s8(row_in + k), vldrbq_s8(w_row + k));
        }
        for (uint32_t k = K_full; k < K; ++k) {
          acc += (int32_t)row_in[k] * (int32_t)w_row[k];
        }
        acc += row_off_term + p->kernel_sum[n];
        int32_t r = kwt_1_requantize(acc, mult, shift) + out_off;
        if (r < amin) r = amin; else if (r > amax) r = amax;
        output[(size_t)m * N + n] = (int8_t)r;
      }
    }
    return;
  }
#endif

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
#if KWT_1_USE_MVE
  /* Both adds in KWT-1's encoder block have negative output_shift and
   * shift in {0, -2} for the inputs; kwt_1_mve_requantize_nonpos
   * handles shift<=0 bit-exactly.  Activation bounds for KWT-1 are
   * always [-128, 127] (no fused ReLU here), so the clamp degenerates
   * to a saturating cast.  Process 4 int32 lanes at a time after
   * widening from int8. */
  if (p->self_shift <= 0 && p->other_shift <= 0 && p->output_shift <= 0) {
    const int32_t s_zp = p->self_zp;
    const int32_t o_zp = p->other_zp;
    const int32_t s_mult = p->self_multiplier;
    const int32_t o_mult = p->other_multiplier;
    const int32_t out_mult = p->output_multiplier;
    const int32_t s_sh = p->self_shift;
    const int32_t o_sh = p->other_shift;
    const int32_t out_sh = p->output_shift;
    const int32_t out_zp = p->output_zp;
    const int32_t amin = p->activation_min;
    const int32_t amax = p->activation_max;
    /* Widen 16 int8 → four int32x4 quads via vldrbq_s32 (widening
     * gather load).  This avoids the explicit vmovlb/vmovlt chain and
     * keeps lane order natural so the narrowing stores work the same
     * way back. */
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
      int32x4_t s_q0 = vldrbq_s32(self_in + i + 0);
      int32x4_t s_q1 = vldrbq_s32(self_in + i + 4);
      int32x4_t s_q2 = vldrbq_s32(self_in + i + 8);
      int32x4_t s_q3 = vldrbq_s32(self_in + i + 12);
      int32x4_t o_q0 = vldrbq_s32(other_in + i + 0);
      int32x4_t o_q1 = vldrbq_s32(other_in + i + 4);
      int32x4_t o_q2 = vldrbq_s32(other_in + i + 8);
      int32x4_t o_q3 = vldrbq_s32(other_in + i + 12);
      #define ADD_ONE(sq, oq) ({                                                       \
        int32x4_t _sv = vshlq_n_s32(vsubq_n_s32((sq), s_zp), KWT_1_ADD_INTERNAL_SHIFT); \
        int32x4_t _ov = vshlq_n_s32(vsubq_n_s32((oq), o_zp), KWT_1_ADD_INTERNAL_SHIFT); \
        int32x4_t _sf = kwt_1_mve_requantize_nonpos(_sv, s_mult, s_sh);                \
        int32x4_t _of = kwt_1_mve_requantize_nonpos(_ov, o_mult, o_sh);                \
        int32x4_t _r  = kwt_1_mve_requantize_nonpos(vaddq_s32(_sf, _of), out_mult, out_sh); \
        _r = vaddq_n_s32(_r, out_zp);                                                  \
        vminq_s32(vmaxq_s32(_r, vdupq_n_s32(amin)), vdupq_n_s32(amax));                \
      })
      int32x4_t r0 = ADD_ONE(s_q0, o_q0);
      int32x4_t r1 = ADD_ONE(s_q1, o_q1);
      int32x4_t r2 = ADD_ONE(s_q2, o_q2);
      int32x4_t r3 = ADD_ONE(s_q3, o_q3);
      #undef ADD_ONE
      vstrbq_s32(output + i + 0, r0);
      vstrbq_s32(output + i + 4, r1);
      vstrbq_s32(output + i + 8, r2);
      vstrbq_s32(output + i + 12, r3);
    }
    /* Scalar tail. */
    for (; i < n; ++i) {
      int32_t s_v = ((int32_t)self_in[i] - p->self_zp) << KWT_1_ADD_INTERNAL_SHIFT;
      int32_t o_v = ((int32_t)other_in[i] - p->other_zp) << KWT_1_ADD_INTERNAL_SHIFT;
      int32_t s_fp = kwt_1_requantize(s_v, p->self_multiplier, p->self_shift);
      int32_t o_fp = kwt_1_requantize(o_v, p->other_multiplier, p->other_shift);
      int32_t r = kwt_1_requantize(s_fp + o_fp, p->output_multiplier, p->output_shift)
                + p->output_zp;
      if (r < p->activation_min) r = p->activation_min;
      if (r > p->activation_max) r = p->activation_max;
      output[i] = (int8_t)r;
    }
    return;
  }
#endif
  for (uint32_t i = 0; i < n; ++i) {
    int32_t s_shift_val = ((int32_t)self_in[i] - p->self_zp) << KWT_1_ADD_INTERNAL_SHIFT;
    int32_t o_shift_val = ((int32_t)other_in[i] - p->other_zp) << KWT_1_ADD_INTERNAL_SHIFT;
    int32_t s_fp = kwt_1_requantize(s_shift_val, p->self_multiplier, p->self_shift);
    int32_t o_fp = kwt_1_requantize(o_shift_val, p->other_multiplier, p->other_shift);
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

#if KWT_1_USE_MVE
  /* MVE fast path.  D must be a multiple of 16 (KWT-1's d=64 fits) so
   * the QK^T D-loop vectorizes via vmladavaq_s8.  The AV step rounds
   * the contraction dim S up to a multiple of 16 (S_pad) and uses a
   * zero-padded v scratch so the same vmladavaq_s8 covers both short
   * (S=8) and long (canonical S=98) sequences.
   *
   * Kernel-sum reformulation lets the inner dot use raw int8 (no
   * per-lane zp subtraction):
   *   sum (q+q_off)(k+k_off) = qk_dot + q_off*k_sum + k_off*q_sum
   *                          + D*q_off*k_off
   *   sum (p+probs_off)(v+v_off) = pv_dot + probs_off*v_sum
   *                              + v_off*p_sum + S*probs_off*v_off
   */
  #define KWT_1_AV_MAX_S_PAD 128
  if ((D & 15u) == 0u && D <= 64 && ((S + 15u) & ~15u) <= KWT_1_AV_MAX_S_PAD) {
    /* MVE fast path scope: D multiple of 16 and D ≤ 64 (v_col_sums
     * static); S_pad ≤ KWT_1_AV_MAX_S_PAD (v_padded static).  Both
     * limits comfortably cover canonical KWT-1 (D=64, S=98 → 112). */
    int32_t k_row_sums[KWT_1_SOFTMAX_MAX_ROW];
    int32_t v_col_sums[64];
    int8_t v_padded[64 * KWT_1_AV_MAX_S_PAD];
    const uint32_t S_pad = (S + 15u) & ~15u;
    for (uint32_t b = 0; b < B; ++b) {
      const int8_t* q_b = q + (size_t)b * S * D;
      const int8_t* k_b = k + (size_t)b * S * D;
      const int8_t* v_b = v + (size_t)b * S * D;
      int8_t* out_b = output + (size_t)b * S * D;

      for (uint32_t j = 0; j < S; ++j) {
        const int8_t* k_row = k_b + (size_t)j * D;
        int32_t s = 0;
        for (uint32_t kk = 0; kk < D; kk += 16) {
          s = vaddvaq_s8(s, vldrbq_s8(k_row + kk));
        }
        k_row_sums[j] = s;
      }
      /* v_col_sums[n] = sum_j v_b[n, j] (v is stored (D, S) in BMM rhs_T
       * layout).  S=8, each row D values, but it's a row sum over S. */
      for (uint32_t n = 0; n < D; ++n) {
        const int8_t* v_row = v_b + (size_t)n * S;
        int32_t s = 0;
        for (uint32_t jj = 0; jj < S; ++jj) s += (int32_t)v_row[jj];
        v_col_sums[n] = s;
      }
      /* Build a (D, S_pad) zero-padded copy of v for the AV step's
       * vmladavaq_s8 inner.  S_pad = ceil(S/16)*16; padding lets a
       * single 16-lane vmladavaq_s8 sweep the contraction without a
       * predicated tail (the predicated-load path leaked nonzero
       * junk through the sum on this gcc+M55 combo).  Stack cost
       * = D * S_pad bytes; e.g. 1 KB at S=8 or ~7 KB at S=98. */
      for (uint32_t n = 0; n < D; ++n) {
        const int8_t* v_row = v_b + (size_t)n * S;
        int8_t* dst = v_padded + (size_t)n * S_pad;
        for (uint32_t jj = 0; jj < S; ++jj) dst[jj] = v_row[jj];
        for (uint32_t jj = S; jj < S_pad; ++jj) dst[jj] = 0;
      }
      /* Constants for the kernel-sum reformulation:
       *   sum (q+q_off)(k+k_off) = qk_dot + q_off*k_sum + k_off*q_sum + D*q_off*k_off
       *   sum (p+probs_off)(v+v_off) = pv_dot + probs_off*v_sum + v_off*p_sum + S*probs_off*v_off
       */
      const int32_t qk_const_term = (int32_t)D * q_off * k_off;
      const int32_t av_const_term = (int32_t)S * probs_off * v_off;

      for (uint32_t i = 0; i < S; ++i) {
        const int8_t* q_row = q_b + (size_t)i * D;
        /* q_row_sum */
        int32_t q_row_sum = 0;
        for (uint32_t kk = 0; kk < D; kk += 16) {
          q_row_sum = vaddvaq_s8(q_row_sum, vldrbq_s8(q_row + kk));
        }
        const int32_t q_off_term = k_off * q_row_sum;

        /* Step 1: QK^T row i.  Tile by 4 j's so the per-score
         * requantize-clamp-store runs once per 4 scores via the
         * vector requantize path. */
        const int32_t qk_row_const = q_off_term + qk_const_term;
        uint32_t j = 0;
        if (qk_shift <= 0) {
          const int32x4_t v_neg128 = vdupq_n_s32(-128);
          const int32x4_t v_pos127 = vdupq_n_s32(127);
          const int32x4_t v_qk_zp = vdupq_n_s32(qk_zp);
          const int32x4_t v_qk_row_const = vdupq_n_s32(qk_row_const);
          /* D=64 hoist: q_row's four 16-byte vectors stay resident
           * across all S/4 j-tile iterations.  Saves S/4 × 4 - 4
           * = S - 4 q_row loads per query row (~94 loads/query at
           * S=98, ~9200 cycles per attention call). */
          if (D == 64) {
            int8x16_t qv0 = vldrbq_s8(q_row +  0);
            int8x16_t qv1 = vldrbq_s8(q_row + 16);
            int8x16_t qv2 = vldrbq_s8(q_row + 32);
            int8x16_t qv3 = vldrbq_s8(q_row + 48);
            for (; j + 4 <= S; j += 4) {
              const int8_t* k0 = k_b + (size_t)(j + 0) * D;
              const int8_t* k1 = k_b + (size_t)(j + 1) * D;
              const int8_t* k2 = k_b + (size_t)(j + 2) * D;
              const int8_t* k3 = k_b + (size_t)(j + 3) * D;
              int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
              d0 = vmladavaq_s8(d0, qv0, vldrbq_s8(k0 +  0));
              d1 = vmladavaq_s8(d1, qv0, vldrbq_s8(k1 +  0));
              d2 = vmladavaq_s8(d2, qv0, vldrbq_s8(k2 +  0));
              d3 = vmladavaq_s8(d3, qv0, vldrbq_s8(k3 +  0));
              d0 = vmladavaq_s8(d0, qv1, vldrbq_s8(k0 + 16));
              d1 = vmladavaq_s8(d1, qv1, vldrbq_s8(k1 + 16));
              d2 = vmladavaq_s8(d2, qv1, vldrbq_s8(k2 + 16));
              d3 = vmladavaq_s8(d3, qv1, vldrbq_s8(k3 + 16));
              d0 = vmladavaq_s8(d0, qv2, vldrbq_s8(k0 + 32));
              d1 = vmladavaq_s8(d1, qv2, vldrbq_s8(k1 + 32));
              d2 = vmladavaq_s8(d2, qv2, vldrbq_s8(k2 + 32));
              d3 = vmladavaq_s8(d3, qv2, vldrbq_s8(k3 + 32));
              d0 = vmladavaq_s8(d0, qv3, vldrbq_s8(k0 + 48));
              d1 = vmladavaq_s8(d1, qv3, vldrbq_s8(k1 + 48));
              d2 = vmladavaq_s8(d2, qv3, vldrbq_s8(k2 + 48));
              d3 = vmladavaq_s8(d3, qv3, vldrbq_s8(k3 + 48));
              int32x4_t dots = {d0, d1, d2, d3};
              int32x4_t ks_v = vldrwq_s32(k_row_sums + j);
              int32x4_t acc = vaddq_s32(dots, vmulq_n_s32(ks_v, q_off));
              acc = vaddq_s32(acc, v_qk_row_const);
              acc = kwt_1_mve_requantize_nonpos(acc, qk_mult, qk_shift);
              acc = vaddq_s32(acc, v_qk_zp);
              acc = vminq_s32(vmaxq_s32(acc, v_neg128), v_pos127);
              vstrbq_s32(scores_scratch + j, acc);
            }
          } else {
            for (; j + 4 <= S; j += 4) {
              const int8_t* k0 = k_b + (size_t)(j + 0) * D;
              const int8_t* k1 = k_b + (size_t)(j + 1) * D;
              const int8_t* k2 = k_b + (size_t)(j + 2) * D;
              const int8_t* k3 = k_b + (size_t)(j + 3) * D;
              int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
              for (uint32_t kk = 0; kk < D; kk += 16) {
                int8x16_t qv = vldrbq_s8(q_row + kk);
                d0 = vmladavaq_s8(d0, qv, vldrbq_s8(k0 + kk));
                d1 = vmladavaq_s8(d1, qv, vldrbq_s8(k1 + kk));
                d2 = vmladavaq_s8(d2, qv, vldrbq_s8(k2 + kk));
                d3 = vmladavaq_s8(d3, qv, vldrbq_s8(k3 + kk));
              }
              int32x4_t dots = {d0, d1, d2, d3};
              int32x4_t ks_v = vldrwq_s32(k_row_sums + j);
              int32x4_t acc = vaddq_s32(dots, vmulq_n_s32(ks_v, q_off));
              acc = vaddq_s32(acc, v_qk_row_const);
              acc = kwt_1_mve_requantize_nonpos(acc, qk_mult, qk_shift);
              acc = vaddq_s32(acc, v_qk_zp);
              acc = vminq_s32(vmaxq_s32(acc, v_neg128), v_pos127);
              vstrbq_s32(scores_scratch + j, acc);
            }
          }
        }
        for (; j < S; ++j) {
          const int8_t* k_row = k_b + (size_t)j * D;
          int32_t dot = 0;
          for (uint32_t kk = 0; kk < D; kk += 16) {
            dot = vmladavaq_s8(dot, vldrbq_s8(q_row + kk), vldrbq_s8(k_row + kk));
          }
          int32_t acc = dot + q_off * k_row_sums[j] + qk_row_const;
          int32_t r = kwt_1_requantize(acc, qk_mult, qk_shift) + qk_zp;
          if (r < -128) r = -128;
          if (r >  127) r =  127;
          scores_scratch[j] = (int8_t)r;
        }

        /* Step 2: softmax in place. */
        softmax_s8(scores_scratch, scores_scratch, &sm_p);

        /* Step 3: AV row i = sum_j (probs[j]+128)(v[n,j]+v_off).
         * Kernel-sum reformulation; the dot sweeps the padded
         * contraction in 16-byte chunks (one vmladavaq_s8 per chunk),
         * with zero-padded tail bytes contributing 0.  Probs sum is
         * computed across the same padded buffer for consistency. */
        for (uint32_t jj = S; jj < S_pad; ++jj) scores_scratch[jj] = 0;
        int32_t probs_sum = 0;
        for (uint32_t kk = 0; kk < S_pad; kk += 16) {
          probs_sum = vaddvaq_s8(probs_sum, vldrbq_s8(scores_scratch + kk));
        }
        const int32_t probs_row_term = v_off * probs_sum + av_const_term;

        int8_t* out_row = out_b + (size_t)i * D;
        const int32x4_t v_neg128 = vdupq_n_s32(-128);
        const int32x4_t v_pos127 = vdupq_n_s32(127);
        const int32x4_t v_av_zp = vdupq_n_s32(av_zp);
        const int32x4_t v_probs_row_term = vdupq_n_s32(probs_row_term);
        const int32_t probs_off_local = probs_off;
        uint32_t n = 0;
        /* AV step in tile-of-4 outputs: each of the 4 dots sweeps
         * S_pad in 16-byte chunks; for S=8 (S_pad=16) the inner is
         * a single vmladavaq_s8 per dot, for S=98 (S_pad=112) it's
         * 7 vmladavaq_s8 per dot.  Probs are loaded fresh per chunk
         * so 8 Q-regs hold {probs_v, v0..v3, scratch}. */
        if (av_shift <= 0) {
          for (; n + 4 <= D; n += 4) {
            const int8_t* vr0 = v_padded + (size_t)(n + 0) * S_pad;
            const int8_t* vr1 = v_padded + (size_t)(n + 1) * S_pad;
            const int8_t* vr2 = v_padded + (size_t)(n + 2) * S_pad;
            const int8_t* vr3 = v_padded + (size_t)(n + 3) * S_pad;
            int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (uint32_t kk = 0; kk < S_pad; kk += 16) {
              int8x16_t probs_v = vldrbq_s8(scores_scratch + kk);
              d0 = vmladavaq_s8(d0, probs_v, vldrbq_s8(vr0 + kk));
              d1 = vmladavaq_s8(d1, probs_v, vldrbq_s8(vr1 + kk));
              d2 = vmladavaq_s8(d2, probs_v, vldrbq_s8(vr2 + kk));
              d3 = vmladavaq_s8(d3, probs_v, vldrbq_s8(vr3 + kk));
            }
            int32x4_t dots = {d0, d1, d2, d3};
            int32x4_t vsum = vldrwq_s32(v_col_sums + n);
            int32x4_t acc = vaddq_s32(dots, vmulq_n_s32(vsum, probs_off_local));
            acc = vaddq_s32(acc, v_probs_row_term);
            acc = kwt_1_mve_requantize_nonpos(acc, av_mult, av_shift);
            acc = vaddq_s32(acc, v_av_zp);
            acc = vminq_s32(vmaxq_s32(acc, v_neg128), v_pos127);
            vstrbq_s32(out_row + n, acc);
          }
        }
        for (; n < D; ++n) {
          const int8_t* v_row = v_padded + (size_t)n * S_pad;
          int32_t dot = 0;
          for (uint32_t kk = 0; kk < S_pad; kk += 16) {
            dot = vmladavaq_s8(dot, vldrbq_s8(scores_scratch + kk),
                                    vldrbq_s8(v_row + kk));
          }
          int32_t acc = dot + probs_off * v_col_sums[n] + probs_row_term;
          int32_t r = kwt_1_requantize(acc, av_mult, av_shift) + av_zp;
          if (r < -128) r = -128;
          if (r >  127) r =  127;
          out_row[n] = (int8_t)r;
        }
      }
    }
    return;
  }
scalar_path:;
#endif

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

      /* Step 2: softmax those S scores in place. */
      softmax_s8(scores_scratch, scores_scratch, &sm_p);

      /* Step 3: AV row i = sum_j (probs[j] + probs_off) * (v_t[n,j] + v_off). */
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
