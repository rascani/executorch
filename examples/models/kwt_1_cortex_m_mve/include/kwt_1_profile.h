/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Per-kernel cycle profiling for kwt_1_inference.  Enabled by building
 * with -DKWT_1_PROFILE=1; otherwise the macros expand to no-ops and the
 * generated inference body collapses back to a plain sequence of kernel
 * calls.
 *
 * When enabled, each call site in kwt_1_inference_body.h is wrapped in
 * KWT_1_PROFILE_BEGIN/END that snapshot the Cortex-M55 PMU cycle counter
 * (ARM_PMU_Get_CCNTR) and accumulate the per-kernel delta into
 * kwt_1_profile_cycles[idx].  The runner calls kwt_1_profile_dump() after
 * inference completes to print the table.
 */

#ifndef KWT_1_PROFILE_H_
#define KWT_1_PROFILE_H_

#include <stdint.h>

#ifndef KWT_1_PROFILE
#define KWT_1_PROFILE 0
#endif

#define KWT_1_PROFILE_MAX_KERNELS 32

#ifdef __cplusplus
extern "C" {
#endif

#if KWT_1_PROFILE

extern uint32_t kwt_1_profile_cycles[KWT_1_PROFILE_MAX_KERNELS];
extern const char* kwt_1_profile_names[KWT_1_PROFILE_MAX_KERNELS];
extern uint32_t kwt_1_profile_count;

uint32_t kwt_1_profile_now(void);
void kwt_1_profile_dump(void);

#define KWT_1_PROFILE_BEGIN()  uint32_t _kwt_t0 = kwt_1_profile_now()
#define KWT_1_PROFILE_END(idx, name) do { \
    kwt_1_profile_cycles[(idx)] += kwt_1_profile_now() - _kwt_t0; \
    kwt_1_profile_names[(idx)] = (name); \
    if ((idx) + 1 > kwt_1_profile_count) kwt_1_profile_count = (idx) + 1; \
  } while (0)

#else

#define KWT_1_PROFILE_BEGIN()       ((void)0)
#define KWT_1_PROFILE_END(idx, name) ((void)0)
static inline void kwt_1_profile_dump(void) {}

#endif

#ifdef __cplusplus
}
#endif

#endif
