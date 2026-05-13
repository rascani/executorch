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

#include "ARMCM55.h"

#include "mv2_inference.h"
#include "mv2_arena.h"
#include "input_fixture.h"

/* DWT cycle counter registers — Cortex-M55 implements both DWT and the
 * cycle counter (CYCCNT).  Read alongside the PMU CCNTR for cross-check;
 * see docs/BENCHMARK.md for why we report PMU as the canonical CYCLES. */
#define DWT_CTRL    (*(volatile uint32_t*)0xE0001000u)
#define DWT_CYCCNT  (*(volatile uint32_t*)0xE0001004u)
#define DEMCR       (*(volatile uint32_t*)0xE000EDFCu)
#define DEMCR_TRCENA   (1u << 24)
#define DWT_CTRL_CYCCNTENA (1u << 0)

static void enable_cycle_counter(void) {
  DEMCR    |= DEMCR_TRCENA;
  DWT_CYCCNT = 0;
  DWT_CTRL |= DWT_CTRL_CYCCNTENA;

  /* Enable PMU cycle counter (same source the cortex_m backend's runner
   * uses via ARM_PMU_Get_CCNTR), so cross-runner comparisons read the
   * same counter. */
  ARM_PMU_Enable();
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);

  static int8_t output[MV2_OUTPUT_NUM_ELEMENTS];

  enable_cycle_counter();
  uint32_t dwt_start = DWT_CYCCNT;
  uint32_t pmu_start = ARM_PMU_Get_CCNTR();
  mobilenet_v2_inference(mv2_fixture_input, output);
  uint32_t dwt_elapsed = DWT_CYCCNT - dwt_start;
  uint32_t pmu_elapsed = ARM_PMU_Get_CCNTR() - pmu_start;

  /* CYCLES reports PMU (matches cortex_m runner's source). */
  printf("CYCLES %u\n", (unsigned)pmu_elapsed);
  printf("CYCLES_DWT %u\n", (unsigned)dwt_elapsed);
  printf("CYCLES_PMU %u\n", (unsigned)pmu_elapsed);

#ifdef MV2_PROFILE_KERNELS
  extern uint64_t mv2_prof_quantize_input, mv2_prof_conv2d_3x3_first,
                  mv2_prof_conv2d_1x1, mv2_prof_dwconv2d, mv2_prof_add_s8,
                  mv2_prof_avgpool, mv2_prof_gemv;
  extern uint32_t mv2_prof_n_quantize_input, mv2_prof_n_conv2d_3x3_first,
                  mv2_prof_n_conv2d_1x1, mv2_prof_n_dwconv2d, mv2_prof_n_add_s8,
                  mv2_prof_n_avgpool, mv2_prof_n_gemv;
  printf("PROF quantize_input n=%u dwt=%llu\n",
         mv2_prof_n_quantize_input, (unsigned long long)mv2_prof_quantize_input);
  printf("PROF conv2d_3x3_first n=%u dwt=%llu\n",
         mv2_prof_n_conv2d_3x3_first, (unsigned long long)mv2_prof_conv2d_3x3_first);
  printf("PROF conv2d_1x1 n=%u dwt=%llu\n",
         mv2_prof_n_conv2d_1x1, (unsigned long long)mv2_prof_conv2d_1x1);
  printf("PROF dwconv2d n=%u dwt=%llu\n",
         mv2_prof_n_dwconv2d, (unsigned long long)mv2_prof_dwconv2d);
  printf("PROF add_s8 n=%u dwt=%llu\n",
         mv2_prof_n_add_s8, (unsigned long long)mv2_prof_add_s8);
  printf("PROF avgpool n=%u dwt=%llu\n",
         mv2_prof_n_avgpool, (unsigned long long)mv2_prof_avgpool);
  printf("PROF gemv n=%u dwt=%llu\n",
         mv2_prof_n_gemv, (unsigned long long)mv2_prof_gemv);
#endif

  printf("RESULT_BEGIN\n");
  for (unsigned i = 0; i < MV2_OUTPUT_NUM_ELEMENTS; ++i) {
    printf("%d\n", (int)output[i]);
  }
  printf("RESULT_END\n");

  /* Semihosting exit — Corstone-300 picks this up via the FVP. */
  return 0;
}
