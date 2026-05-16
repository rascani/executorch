/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Standalone KWT-1 inference for Cortex-M55 + Helium MVE.  Phased per
 * docs/PLAN.md:
 *   Phase 0 — deterministic round-trip stub (no transformer math).
 *             Quantizes the float input deterministically into the
 *             arena and reduces it to KWT_1_OUTPUT_NUM_ELEMENTS int8
 *             values via a position-folded sum.  Purpose is to
 *             exercise the host + FVP build, the harness round-trip,
 *             and the fixture/output plumbing end-to-end before any
 *             real kernels exist.
 *   Phase 1+ — LayerNorm, GELU, BMM, softmax, fused attention, fused
 *              FFN; the body will be replaced row-by-row as the
 *              kernels land.
 */

#include <stdint.h>
#include <math.h>

#include "kwt_1_inference.h"
#include "kwt_1_arena.h"

extern uint8_t kwt_1_arena[];

void kwt_1_inference(const float* input, int8_t* output) {
  /* Phase 0: quantize input with a fixed scale/zp so the round-trip is
   * deterministic, lay it down into the arena, then reduce to
   * KWT_1_OUTPUT_NUM_ELEMENTS via a position-folded modulo sum.
   * Nothing about this resembles the real KWT-1 forward pass — the
   * point is that input → output is a known function the test
   * harness can verify bit-exactly. */
  const float quant_scale = 0.0625f;     /* 1/16, picks reasonable int8 range */
  const int32_t quant_zp = 0;
  const int32_t qmin = -128;
  const int32_t qmax = 127;

  int8_t* arena_in = (int8_t*)kwt_1_arena;
  for (unsigned i = 0; i < KWT_1_INPUT_NUM_ELEMENTS; ++i) {
    float scaled = input[i] / quant_scale;
    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    r += quant_zp;
    if (r < qmin) r = qmin;
    if (r > qmax) r = qmax;
    arena_in[i] = (int8_t)r;
  }

  for (unsigned j = 0; j < KWT_1_OUTPUT_NUM_ELEMENTS; ++j) {
    int32_t acc = 0;
    for (unsigned i = j; i < KWT_1_INPUT_NUM_ELEMENTS;
         i += KWT_1_OUTPUT_NUM_ELEMENTS) {
      acc += arena_in[i];
    }
    /* Reduce to int8 by averaging, then clamp. */
    int32_t n = (KWT_1_INPUT_NUM_ELEMENTS - j + KWT_1_OUTPUT_NUM_ELEMENTS - 1)
                / KWT_1_OUTPUT_NUM_ELEMENTS;
    if (n <= 0) n = 1;
    int32_t avg = acc / n;
    if (avg < qmin) avg = qmin;
    if (avg > qmax) avg = qmax;
    output[j] = (int8_t)avg;
  }
}
