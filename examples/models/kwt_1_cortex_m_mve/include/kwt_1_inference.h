/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Public entry point for the standalone KWT-1 (Keyword Spotting
 * Transformer) inference function.  Input is the post-MFCC feature
 * tensor in float32 (KWT_1_INPUT_NUM_ELEMENTS == 40*98 = 3920 at the
 * canonical KWT-1 spec); output is the int8-quantized logit vector
 * (KWT_1_OUTPUT_NUM_ELEMENTS == 35 for Speech Commands v2).
 *
 * Phased build per docs/PLAN.md.  At Phase 0 this function is a
 * deterministic round-trip stub used to exercise the harness end-to-end
 * before any transformer math lands.
 */

#ifndef KWT_1_INFERENCE_H_
#define KWT_1_INFERENCE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kwt_1_inference(const float* input, int8_t* output);

#ifdef __cplusplus
}
#endif

#endif
