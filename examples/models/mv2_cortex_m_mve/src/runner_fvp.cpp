/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Cortex-M55 Corstone-300 FVP runner.  Embeds the input fixture, calls
 * mobilenet_v2_inference, prints int8 logits between RESULT_BEGIN/RESULT_END
 * markers via semihosting (newlib --specs=rdimon.specs handles the syscalls).
 *
 * Build via CMakeLists.txt with -DMV2_BUILD_FVP=ON.  Note that the linker
 * script (Corstone-300.ld) places mv2_arena in the DDR section; the runner
 * itself lives in flash.
 */

#include <stdint.h>
#include <stdio.h>

#include "mv2_inference.h"
#include "mv2_arena.h"
#include "input_fixture.h"

int main(void) {
  /* Disable stdout buffering so the host sees each line as it's produced. */
  setvbuf(stdout, NULL, _IONBF, 0);

  int8_t output[MV2_OUTPUT_NUM_ELEMENTS];
  mobilenet_v2_inference(mv2_fixture_input, output);

  printf("RESULT_BEGIN\n");
  for (unsigned i = 0; i < MV2_OUTPUT_NUM_ELEMENTS; ++i) {
    printf("%d\n", (int)output[i]);
  }
  printf("RESULT_END\n");

  /* Semihosting exit — Corstone-300 picks this up via the FVP. */
  return 0;
}
