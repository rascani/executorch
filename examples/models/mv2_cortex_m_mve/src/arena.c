/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Activation arena for the standalone MobileNetV2 inference path.
 *
 * Sized by the AOT memory planner; the size macro lives in the generated
 * mv2_arena.h.  On Corstone-300 the array is placed in the DDR region via
 * the linker script (see CMakeLists.txt / Corstone-300.ld) — on host builds
 * it sits in plain .bss.
 */

#include <stdint.h>

#include "mv2_arena.h"

#if defined(__ARM_ARCH) && defined(USE_CORSTONE_LINKER)
__attribute__((section(".bss.tensor_arena"), aligned(16)))
#else
__attribute__((aligned(16)))
#endif
uint8_t mv2_arena[MV2_ARENA_BYTES];
