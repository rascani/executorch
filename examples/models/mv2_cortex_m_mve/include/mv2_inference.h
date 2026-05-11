/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 */

#ifndef MV2_INFERENCE_H_
#define MV2_INFERENCE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mobilenet_v2_inference(const float* input_nchw, int8_t* logits_q);

#ifdef __cplusplus
}
#endif

#endif
