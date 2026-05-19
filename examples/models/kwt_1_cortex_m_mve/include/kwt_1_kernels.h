/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Public entry points for the standalone KWT-1 kernels.  Phased per
 * docs/PLAN.md; expand as each phase lands.
 */

#ifndef KWT_1_KERNELS_H_
#define KWT_1_KERNELS_H_

#include <stdint.h>

#include "kwt_1_layer_params.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 1: int8 in / int8 out LayerNorm with float32 γ/β. */
void layer_norm_s8(const int8_t* input, int8_t* output,
                   const LayerNormParams* p);

/* Phase 2: int8 in / int8 out GELU via a 256-byte LUT. */
void gelu_lut_s8(const int8_t* input, int8_t* output,
                 const GELUParams* p);

/* Phase 3: int8 batched matmul (B, M, K) × (B, N, K)^T → (B, M, N). */
void batch_matmul_s8(const int8_t* lhs, const int8_t* rhs_transposed,
                     int8_t* output, const BMMParams* p);

/* Phase 4: per-row int8 softmax.  Output uses CMSIS-NN's fixed
 * (scale=1/256, zp=-128) — those values are not in SoftmaxParams. */
void softmax_s8(const int8_t* input, int8_t* output,
                const SoftmaxParams* p);

/* Phase 5: streaming fused attention.  Stream-processes one Q row
 * at a time → never materializes the (S, S) score matrix. */
void attention_fused_s8(const int8_t* q, const int8_t* k, const int8_t* v,
                        int8_t* output, const AttentionParams* p);

/* Phase 6: int8 quantized add (residual path). */
void add_s8(const int8_t* self, const int8_t* other, int8_t* output,
            const AddParams* p);

/* Phase 6: int8 Linear (CMSIS-NN kernel_sum form).  weights is
 * (out_features, in_features). */
void linear_s8(const int8_t* input, const int8_t* weights,
               int8_t* output, const LinearParams* p);

/* Phase 7: int8 N-D transpose (rank ≤ 4). */
void transpose_s8(const int8_t* input, int8_t* output,
                  const TransposeParams* p);

/* Phase 10: int8 mean across one axis with fused dequant + requant. */
void mean_dim_s8(const int8_t* input, int8_t* output,
                 const MeanDimParams* p);

#ifdef __cplusplus
}
#endif

#endif
