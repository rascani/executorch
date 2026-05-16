/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Activation arena for the standalone KWT-1 (Keyword Spotting
 * Transformer) inference path.
 *
 * Sized by the AOT memory planner (Phase 1+); the size macro lives in the
 * generated kwt_1_arena.h.  On Corstone-300 the array is placed in the
 * DDR region via the linker script (see CMakeLists.txt /
 * Corstone-300.ld) — on host builds it sits in plain .bss.
 *
 * Phase 0: KWT_1_ARENA_BYTES is set to MAX(input_bytes, output_bytes)
 * so the round-trip stub has somewhere to land its input.  Real
 * memory planning kicks in at Phase 1 when the first AOT pass runs.
 */

#include <stdint.h>

#include "kwt_1_arena.h"

#if defined(__ARM_ARCH) && defined(USE_CORSTONE_LINKER)
__attribute__((section(".bss.tensor_arena"), aligned(16)))
#else
__attribute__((aligned(16)))
#endif
uint8_t kwt_1_arena[KWT_1_ARENA_BYTES];

/* Per-kernel scratch buffer for the transformer fused ops (e.g. fused
 * attention's per-row score buffer).  Phase 0 sets KWT_1_SCRATCH_BYTES
 * to 0; Phase 5+ will size it from the largest layer's row scratch. */
#if KWT_1_SCRATCH_BYTES > 0
__attribute__((aligned(16)))
int8_t kwt_1_scratch[KWT_1_SCRATCH_BYTES];
#endif
