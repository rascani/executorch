/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Cortex-M55 Corstone-300 FVP runner for KWT-1.  Embeds the input fixture,
 * calls kwt_1_inference, prints int8 logits between RESULT_BEGIN/RESULT_END
 * markers via semihosting (newlib --specs=rdimon.specs handles the
 * syscalls).
 *
 * Build via CMakeLists.txt with -DKWT_1_BUILD_FVP=ON.  Note that the
 * linker script (Corstone-300.ld) places kwt_1_arena in the DDR section;
 * the runner itself lives in flash.
 */

#include <stdint.h>
#include <stdio.h>

#include "ARMCM55.h"

#include "kwt_1_inference.h"
#include "kwt_1_arena.h"
#include "kwt_1_profile.h"
#include "input_fixture.h"

#if KWT_1_PROFILE
extern "C" uint32_t kwt_1_profile_now(void) {
  return ARM_PMU_Get_CCNTR();
}
extern "C" void kwt_1_profile_dump(void) {
  extern uint32_t kwt_1_profile_cycles[];
  extern const char* kwt_1_profile_names[];
  extern uint32_t kwt_1_profile_count;
  uint32_t total = 0;
  for (uint32_t i = 0; i < kwt_1_profile_count; ++i) total += kwt_1_profile_cycles[i];
  printf("PROFILE_BEGIN\n");
  for (uint32_t i = 0; i < kwt_1_profile_count; ++i) {
    uint32_t c = kwt_1_profile_cycles[i];
    uint32_t pct = total ? (uint32_t)((uint64_t)c * 1000u / total) : 0u;
    printf("%2u %-24s %10u %u.%u%%\n",
           (unsigned)i,
           kwt_1_profile_names[i] ? kwt_1_profile_names[i] : "(null)",
           (unsigned)c, (unsigned)(pct / 10), (unsigned)(pct % 10));
  }
  printf("PROFILE_TOTAL %u\n", (unsigned)total);
  printf("PROFILE_END\n");
}
#endif

/* DWT cycle counter registers — Cortex-M55 implements both DWT and the
 * cycle counter (CYCCNT).  Read alongside the PMU CCNTR for cross-check;
 * see mv2_cortex_m_mve/docs/BENCHMARK.md for why we report PMU as the
 * canonical CYCLES (DWT is internally aliased 8x on this FVP). */
#define DWT_CTRL    (*(volatile uint32_t*)0xE0001000u)
#define DWT_CYCCNT  (*(volatile uint32_t*)0xE0001004u)
#define DEMCR       (*(volatile uint32_t*)0xE000EDFCu)
#define DEMCR_TRCENA   (1u << 24)
#define DWT_CTRL_CYCCNTENA (1u << 0)

static void enable_cycle_counter(void) {
  DEMCR    |= DEMCR_TRCENA;
  DWT_CYCCNT = 0;
  DWT_CTRL |= DWT_CTRL_CYCCNTENA;

  ARM_PMU_Enable();
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);

  static int8_t output[KWT_1_OUTPUT_NUM_ELEMENTS];

  enable_cycle_counter();
  uint32_t dwt_start = DWT_CYCCNT;
  uint32_t pmu_start = ARM_PMU_Get_CCNTR();
  kwt_1_inference(kwt_1_fixture_input, output);
  uint32_t dwt_elapsed = DWT_CYCCNT - dwt_start;
  uint32_t pmu_elapsed = ARM_PMU_Get_CCNTR() - pmu_start;

  printf("CYCLES %u\n", (unsigned)pmu_elapsed);
  printf("CYCLES_DWT %u\n", (unsigned)dwt_elapsed);
  printf("CYCLES_PMU %u\n", (unsigned)pmu_elapsed);

  kwt_1_profile_dump();

  printf("RESULT_BEGIN\n");
  for (unsigned i = 0; i < KWT_1_OUTPUT_NUM_ELEMENTS; ++i) {
    printf("%d\n", (int)output[i]);
  }
  printf("RESULT_END\n");

  return 0;
}
