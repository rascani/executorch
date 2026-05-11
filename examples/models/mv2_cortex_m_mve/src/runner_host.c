/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Host-side harness for fast correctness iteration of mobilenet_v2_inference.
 * Reads MV2_INPUT_NUM_ELEMENTS float32 values from stdin, runs the inference,
 * and writes MV2_OUTPUT_NUM_ELEMENTS int8 values to stdout — no MVE assumed.
 *
 * The FVP-target counterpart (runner_fvp.cpp) embeds the input fixture and
 * prints output between RESULT_BEGIN / RESULT_END markers via semihosting.
 */

#include <stdint.h>
#include <stdio.h>

#include "mv2_inference.h"
#include "mv2_arena.h"

int main(void) {
  /* Static storage to avoid blowing the stack on MV2-sized inputs. */
  static float input[MV2_INPUT_NUM_ELEMENTS];
  static int8_t output[MV2_OUTPUT_NUM_ELEMENTS];
  if (fread(input, sizeof(float), MV2_INPUT_NUM_ELEMENTS, stdin)
      != MV2_INPUT_NUM_ELEMENTS) {
    fprintf(stderr, "runner_host: short read on stdin\n");
    return 1;
  }
  mobilenet_v2_inference(input, output);
  if (fwrite(output, sizeof(int8_t), MV2_OUTPUT_NUM_ELEMENTS, stdout)
      != MV2_OUTPUT_NUM_ELEMENTS) {
    fprintf(stderr, "runner_host: short write on stdout\n");
    return 1;
  }
  return 0;
}
