/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Parameter structs for the standalone KWT-1 (Keyword Spotting
 * Transformer) inference kernels.  Each struct describes one fused op's
 * compile-time constants — the AOT artifact dumper emits a static
 * const instance per op in generated/kwt_1_params.h, and the inference
 * function passes pointers to those instances into the kernels.
 *
 * Phased per docs/PLAN.md.  Phase 1: LayerNormParams.  Future phases
 * (GELU, BMM, softmax, fused attention, fused FFN) extend this header.
 */

#ifndef KWT_1_LAYER_PARAMS_H_
#define KWT_1_LAYER_PARAMS_H_

#include <stdint.h>

typedef struct {
  /* Phase 1: per-tensor int8 input/output around a float32 LayerNorm.
   * num_rows is the total number of D-wide rows to normalize
   * (e.g. batch × seq_len).  γ and β are float32; β may be NULL.
   * Normalize over the trailing dim of size embed_dim. */
  uint32_t num_rows;
  uint32_t embed_dim;
  int32_t input_zp;
  float input_scale;
  int32_t output_zp;
  float output_scale;
  float eps;
  const float* gamma;
  const float* beta;
} LayerNormParams;

typedef struct {
  /* Phase 2: int8 -> int8 GELU via a 256-byte LUT.  The LUT is
   * precomputed at AOT time as
   *   lut[x + 128] = clamp(round(gelu((x - in_zp) * in_scale) / out_scale)
   *                        + out_zp, -128, 127)
   * for x in [-128, 127], baking the input/output quant params and the
   * approximate form (erf or tanh) into the table.  Per-element
   * runtime cost: one int8 gather.  All other quant params are
   * implicit in the LUT; the kernel needs only num_elements and the
   * LUT pointer. */
  uint32_t num_elements;
  const int8_t* lut;
} GELUParams;

typedef struct {
  /* Phase 6: int8 Linear, CMSIS-NN kernel_sum form (matches the
   * cortex_m::quantized_linear op exactly).  Math:
   *   acc[n] = sum_k (input[m,k] * weights[n,k])
   *          + row_sum[m] * filter_offset
   *          + kernel_sum[n]                       (precomputed at AOT)
   *   out[m,n] = clamp(requantize(acc, mult, shift) + output_offset,
   *                    act_min, act_max)
   * kernel_sum bakes in
   *   kernel_sum[n] = sum_k (w[n,k] + filter_offset) * input_offset
   *                 + (bias[n] if bias is present else 0).
   * Requantize is per-tensor (one mult/shift). */
  uint32_t num_rows;        /* M = prod(input.shape[:-1]) */
  uint32_t in_features;     /* K */
  uint32_t out_features;    /* N */
  int32_t input_offset;     /* -input_zp */
  int32_t filter_offset;    /* -weight_zp */
  int32_t output_offset;
  int32_t output_multiplier;
  int32_t output_shift;
  int32_t activation_min;
  int32_t activation_max;
  const int32_t* kernel_sum;  /* (N,) */
} LinearParams;

typedef struct {
  /* Phase 10: int8 mean across one axis with fused dequant + requant.
   * Used for KWT-1's sequence-mean pooling: input (outer, reduce, inner)
   * → output (outer, inner) via mean over the middle axis.  The kernel
   * works in int32 accumulation then rescales by
   *   ratio = input_scale / (reduce * output_scale)
   * The +output_zp / -input_zp shift folds into the sum. */
  uint32_t outer;
  uint32_t reduce;
  uint32_t inner;
  int32_t input_zp;
  float    input_scale;
  int32_t output_zp;
  float    output_scale;
} MeanDimParams;

typedef struct {
  /* Phase 7: int8 N-D transpose with an arbitrary permutation of dims.
   * Sized for KWT-1's needs: 3-D transpose of (B, S, D) shapes; the
   * permutation is encoded as the strides of the input read in
   * output-major order.  shape[r] is the size of axis r in the
   * OUTPUT tensor; in_stride[r] is the stride in the INPUT tensor
   * (in elements) when output axis r increments by 1.  Then
   *   out[i0,i1,i2] = in[sum_r i_r * in_stride[r]]
   * Holds for any rank ≤ 4; KWT uses rank-3. */
  uint32_t rank;
  uint32_t shape[4];
  uint32_t in_stride[4];
} TransposeParams;

typedef struct {
  /* Phase 6: per-element quantized add, matching cortex_m::quantized_add
   * (CMSIS-NN convention).  Each input is rescaled to a shared
   * "internal" int32 domain via `(x - zp) << 20`, requantized to align
   * scales, summed, then requantized to the output spec.  Used for
   * the two residual adds in each transformer encoder block.  Both
   * inputs are required to be the same shape (no broadcast for now). */
  uint32_t num_elements;
  int32_t self_zp;
  int32_t self_multiplier;
  int32_t self_shift;
  int32_t other_zp;
  int32_t other_multiplier;
  int32_t other_shift;
  int32_t output_zp;
  int32_t output_multiplier;
  int32_t output_shift;
  int32_t activation_min;
  int32_t activation_max;
} AddParams;

typedef struct {
  /* Phase 5: streaming fused attention.  Inputs q, k, v all in
   * (B, S, d) layout; per-Q-row we compute S scores via QK^T,
   * softmax them, then produce d outputs via AV — never
   * materializing the full (S, S) score matrix.
   *
   * AV uses v directly (no v^T materialization): inside the kernel
   * the AV inner loop indexes v[b, j, n] for fixed n / varying j.
   *
   * q_offset, k_offset, v_offset are negated quant zero-points
   * (CMSIS-NN convention, matching BMMParams above).  Three sets of
   * requantize params for the QK^T BMM, softmax, and AV BMM. */
  uint32_t batch;
  uint32_t seq_len;
  uint32_t embed_dim;
  int32_t q_offset;
  int32_t k_offset;
  int32_t qk_output_zp;
  int32_t qk_output_multiplier;
  int32_t qk_output_shift;
  int32_t softmax_input_multiplier;
  int32_t softmax_input_shift;
  int32_t v_offset;
  int32_t av_output_zp;
  int32_t av_output_multiplier;
  int32_t av_output_shift;
} AttentionParams;

typedef struct {
  /* Phase 4: per-row int8 softmax matching CMSIS-NN convention.
   * Output zero-point is fixed at -128 and output scale at 1/256 by
   * cortex_m::softmax's contract — those are baked into the kernel.
   * input_multiplier and input_shift encode the input scale per
   *   real_scale = (multiplier / 2^31) * 2^shift / 2^(31 - 5)
   *             = multiplier * 2^(shift - 57)
   * (the 5 is SOFTMAX_INPUT_INTEGER_BITS from operators.py — number
   * of bits reserved for the integer part of the fixed-point exp
   * argument; matches CMSIS-NN's arm_softmax_s8 convention). */
  uint32_t num_rows;
  uint32_t row_len;
  int32_t input_zp;
  int32_t input_multiplier;
  int32_t input_shift;
} SoftmaxParams;

typedef struct {
  /* Phase 3: int8 (B, M, K) × int8 (B, N, K) → int8 (B, M, N) batched
   * matmul.  rhs is stored *transposed* (i.e. the second matrix's
   * contraction dim is last, same as cortex_m::quantized_batch_matmul),
   * so the kernel's inner loop reads along the last dim of both lhs
   * and rhs — friendly to MVE row-major loads.
   *
   * lhs_offset / rhs_offset are the negated quant zero-points (CMSIS-NN
   * convention): the inner loop adds them to the raw int8 values
   * before multiplying.  output_zp / output_multiplier / output_shift
   * implement arm_nn_requantize on the int32 accumulator. */
  uint32_t batch;
  uint32_t M;
  uint32_t K;
  uint32_t N;
  int32_t lhs_offset;
  int32_t rhs_offset;
  int32_t output_zp;
  int32_t output_multiplier;
  int32_t output_shift;
} BMMParams;

#endif
