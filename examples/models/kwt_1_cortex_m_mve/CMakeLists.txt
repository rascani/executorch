# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
#
# Build configuration for the standalone KWT-1 (Keyword Spotting
# Transformer) Cortex-M55 + MVE inference example.  Phased rollout per
# docs/PLAN.md; this file is set up at Phase 0 with the harness only,
# the kernel + weights references will be added as later phases land.
#
# Two targets:
#   - kwt_1_runner_host : built with the system compiler (gcc/clang) for
#                         correctness testing on dev machines.  No MVE
#                         intrinsics emitted; scalar fallback path is used.
#                         Reads float32 input fixture from stdin, writes
#                         int8 output to stdout.
#   - kwt_1_runner_fvp  : cross-compiled with arm-none-eabi-gcc for
#                         Cortex-M55 + Helium MVE; runs on the Corstone-300
#                         FVP via semihosting.  Enabled with
#                         -DKWT_1_BUILD_FVP=ON and the toolchain file
#                         fvp/toolchain-arm-none-eabi.cmake.
#
# Both targets require KWT_1_GENERATED_DIR populated by
# tools/dump_kwt_1_artifacts.py before configure time.  At Phase 0 the
# only generated content is kwt_1_arena.h (arena/input/output sizes) and
# input_fixture.h (deterministic MFCC-shaped input).  Phase 1+ will
# also emit kwt_1_weights.c and kwt_1_params.h.

cmake_minimum_required(VERSION 3.20)
project(kwt_1_cortex_m_mve C CXX ASM)

option(KWT_1_BUILD_FVP "Cross-compile for Corstone-300 FVP" OFF)

set(KWT_1_GENERATED_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
    CACHE PATH "Directory holding generated kwt_1_arena.h, input_fixture.h, and (Phase 1+) kwt_1_weights.{c,h}, kwt_1_params.h"
)
set(KWT_1_CMSIS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../arm/arm-scratch/ethos-u/core_software"
    CACHE PATH "Path to the ethos_u core_software root (provides ARMCM55 startup + CMSIS Core headers)"
)

if(NOT EXISTS "${KWT_1_GENERATED_DIR}/kwt_1_arena.h")
  message(FATAL_ERROR
    "KWT_1_GENERATED_DIR=${KWT_1_GENERATED_DIR} does not contain kwt_1_arena.h. "
    "Run tools/dump_kwt_1_artifacts.py first."
  )
endif()

set(KWT_1_INFERENCE_SOURCES
    src/kwt_1_inference.c
    src/kwt_1_kernels.c
    src/arena.c
)
if(EXISTS "${KWT_1_GENERATED_DIR}/kwt_1_weights.c")
  list(APPEND KWT_1_INFERENCE_SOURCES "${KWT_1_GENERATED_DIR}/kwt_1_weights.c")
endif()

set(KWT_1_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${KWT_1_GENERATED_DIR}
)

# -----------------------------------------------------------------------------
# Host build (correctness loop)
# -----------------------------------------------------------------------------
if(NOT KWT_1_BUILD_FVP)
  add_executable(kwt_1_runner_host
      ${KWT_1_INFERENCE_SOURCES}
      src/runner_host.c
  )
  target_include_directories(kwt_1_runner_host PRIVATE ${KWT_1_INCLUDE_DIRS})
  target_compile_options(kwt_1_runner_host PRIVATE -O2 -Wall -Wextra)
  target_link_libraries(kwt_1_runner_host PRIVATE m)
endif()

# -----------------------------------------------------------------------------
# FVP build (Cortex-M55 + Helium MVE on Corstone-300)
# -----------------------------------------------------------------------------
if(KWT_1_BUILD_FVP)
  if(NOT EXISTS "${KWT_1_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/startup_ARMCM55.c")
    message(FATAL_ERROR
      "KWT_1_CMSIS_DIR=${KWT_1_CMSIS_DIR} does not contain the ARMCM55 device "
      "files.  Run examples/arm/setup.sh and then point KWT_1_CMSIS_DIR at "
      "examples/arm/arm-scratch/ethos-u/core_software."
    )
  endif()

  set(KWT_1_CMSIS_INCLUDE
      ${KWT_1_CMSIS_DIR}/cmsis_6/CMSIS/Core/Include
      ${KWT_1_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Include
      ${KWT_1_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Config
  )
  set(KWT_1_CMSIS_SOURCES
      ${KWT_1_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/startup_ARMCM55.c
      ${KWT_1_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/system_ARMCM55.c
  )

  add_executable(kwt_1_runner_fvp
      ${KWT_1_INFERENCE_SOURCES}
      src/runner_fvp.cpp
      ${KWT_1_CMSIS_SOURCES}
  )
  target_include_directories(kwt_1_runner_fvp PRIVATE
      ${KWT_1_INCLUDE_DIRS}
      ${KWT_1_CMSIS_INCLUDE}
  )
  target_compile_definitions(kwt_1_runner_fvp PRIVATE
      USE_CORSTONE_LINKER
      ARMCM55
  )
  target_compile_options(kwt_1_runner_fvp PRIVATE
      -O3
      -ffunction-sections
      -fdata-sections
      -Wall
      -Wextra
  )
  set_target_properties(kwt_1_runner_fvp PROPERTIES
      SUFFIX ".elf"
      LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fvp/corstone-300.ld"
  )
  target_link_options(kwt_1_runner_fvp PRIVATE
      "-T${CMAKE_CURRENT_SOURCE_DIR}/fvp/corstone-300.ld"
      -Wl,-Map=$<TARGET_FILE:kwt_1_runner_fvp>.map
  )
endif()
