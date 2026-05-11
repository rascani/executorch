# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
#
# Build configuration for the standalone MobileNetV2 Cortex-M55 + MVE
# inference example.
#
# Two targets:
#   - mv2_runner_host  : built with the system compiler (gcc/clang) for
#                        correctness testing on dev machines.  No MVE
#                        intrinsics emitted; scalar fallback path is used.
#   - mv2_runner_fvp   : cross-compiled with arm-none-eabi-gcc for
#                        Cortex-M55 + Helium MVE; runs on the Corstone-300
#                        FVP via semihosting.  Enabled with -DMV2_BUILD_FVP=ON
#                        and the toolchain file fvp/toolchain-arm-none-eabi.cmake.
#
# Both targets require the generated/ directory populated by
# tools/dump_mv2_artifacts.py before configure time.

cmake_minimum_required(VERSION 3.20)
project(mv2_cortex_m_mve C CXX ASM)

option(MV2_BUILD_FVP "Cross-compile for Corstone-300 FVP" OFF)

set(MV2_GENERATED_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
    CACHE PATH "Directory holding generated mv2_weights.{c,h}, mv2_params.h, mv2_arena.h, input_fixture.h"
)
set(MV2_CMSIS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../arm/arm-scratch/ethos-u/core_software"
    CACHE PATH "Path to the ethos_u core_software root (provides ARMCM55 startup + CMSIS Core headers)"
)

if(NOT EXISTS "${MV2_GENERATED_DIR}/mv2_weights.h")
  message(FATAL_ERROR
    "MV2_GENERATED_DIR=${MV2_GENERATED_DIR} does not contain generated headers. "
    "Run tools/dump_mv2_artifacts.py first."
  )
endif()

set(MV2_INFERENCE_SOURCES
    src/mv2_inference.c
    src/arena.c
    ${MV2_GENERATED_DIR}/mv2_weights.c
)

set(MV2_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MV2_GENERATED_DIR}
)

# -----------------------------------------------------------------------------
# Host build (correctness loop)
# -----------------------------------------------------------------------------
if(NOT MV2_BUILD_FVP)
  add_executable(mv2_runner_host
      ${MV2_INFERENCE_SOURCES}
      src/runner_host.c
  )
  target_include_directories(mv2_runner_host PRIVATE ${MV2_INCLUDE_DIRS})
  target_compile_options(mv2_runner_host PRIVATE -O2 -Wall -Wextra)
endif()

# -----------------------------------------------------------------------------
# FVP build (Cortex-M55 + Helium MVE on Corstone-300)
# -----------------------------------------------------------------------------
if(MV2_BUILD_FVP)
  if(NOT EXISTS "${MV2_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/startup_ARMCM55.c")
    message(FATAL_ERROR
      "MV2_CMSIS_DIR=${MV2_CMSIS_DIR} does not contain the ARMCM55 device "
      "files.  Run examples/arm/setup.sh and then point MV2_CMSIS_DIR at "
      "examples/arm/arm-scratch/ethos-u/core_software."
    )
  endif()

  set(MV2_CMSIS_INCLUDE
      ${MV2_CMSIS_DIR}/cmsis_6/CMSIS/Core/Include
      ${MV2_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Include
      ${MV2_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Config
  )
  set(MV2_CMSIS_SOURCES
      ${MV2_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/startup_ARMCM55.c
      ${MV2_CMSIS_DIR}/Cortex_DFP/Device/ARMCM55/Source/system_ARMCM55.c
  )

  add_executable(mv2_runner_fvp
      ${MV2_INFERENCE_SOURCES}
      src/runner_fvp.cpp
      ${MV2_CMSIS_SOURCES}
  )
  target_include_directories(mv2_runner_fvp PRIVATE
      ${MV2_INCLUDE_DIRS}
      ${MV2_CMSIS_INCLUDE}
  )
  target_compile_definitions(mv2_runner_fvp PRIVATE
      USE_CORSTONE_LINKER
      ARMCM55
  )
  target_compile_options(mv2_runner_fvp PRIVATE
      -O3
      -ffunction-sections
      -fdata-sections
      -Wall
      -Wextra
  )
  set_target_properties(mv2_runner_fvp PROPERTIES
      SUFFIX ".elf"
      LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fvp/corstone-300.ld"
  )
  target_link_options(mv2_runner_fvp PRIVATE
      "-T${CMAKE_CURRENT_SOURCE_DIR}/fvp/corstone-300.ld"
      -Wl,-Map=$<TARGET_FILE:mv2_runner_fvp>.map
  )
endif()
