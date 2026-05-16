/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Host-side harness for fast correctness iteration of kwt_1_inference.
 * Reads KWT_1_INPUT_NUM_ELEMENTS float32 values from stdin (the post-MFCC
 * feature tensor, NCHW or row-major depending on Phase 1+ convention),
 * runs the inference, and writes KWT_1_OUTPUT_NUM_ELEMENTS int8 values
 * to stdout — no MVE assumed.
 *
 * The FVP-target counterpart (runner_fvp.cpp) embeds the input fixture
 * and prints output between RESULT_BEGIN / RESULT_END markers via
 * semihosting.
 */

#include <stdint.h>
#include <stdio.h>

#include "kwt_1_inference.h"
#include "kwt_1_arena.h"

int main(void) {
  /* Static storage to avoid blowing the stack on KWT-1-sized inputs. */
  static float input[KWT_1_INPUT_NUM_ELEMENTS];
  static int8_t output[KWT_1_OUTPUT_NUM_ELEMENTS];
  if (fread(input, sizeof(float), KWT_1_INPUT_NUM_ELEMENTS, stdin)
      != KWT_1_INPUT_NUM_ELEMENTS) {
    fprintf(stderr, "runner_host: short read on stdin\n");
    return 1;
  }
  kwt_1_inference(input, output);
  if (fwrite(output, sizeof(int8_t), KWT_1_OUTPUT_NUM_ELEMENTS, stdout)
      != KWT_1_OUTPUT_NUM_ELEMENTS) {
    fprintf(stderr, "runner_host: short write on stdout\n");
    return 1;
  }
  return 0;
}
