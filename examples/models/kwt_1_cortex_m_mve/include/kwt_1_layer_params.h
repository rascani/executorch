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
